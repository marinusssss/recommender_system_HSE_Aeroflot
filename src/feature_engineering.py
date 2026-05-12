import pandas as pd
import numpy as np
import logging
from src.logger import setup_logger
from src import config

feature_logger = setup_logger('feature_engineering', config.LOG_FILE_PATH, level=config.LOG_LEVEL)

def create_user_features(interactions_df, user_flight_history):
    """
    Создает фичи пользователей на основе истории взаимодействий.
    
    Parameters:
    -----------
    interactions_df : pandas.DataFrame
        DataFrame с взаимодействиями пользователей
    user_flight_history : pandas.DataFrame
        DataFrame с историей полетов пользователей
        
    Returns:
    --------
    dict
        Словарь фичей пользователей {user_idx: {feature: value}}
    """
    feature_logger.info("Начинается создание фичей пользователей...")
    
    # Агрегация базовых статистик ИЗ interactions_df
    user_agg_features = interactions_df.groupby('user_idx').agg(
        total_flights=('item_idx', 'count'),
        avg_interaction=('interaction_score', 'mean'),
        cities_visited_count=('item_idx', 'nunique'),
        # Добавляем сезонные предпочтения
        winter_flights=('season', lambda x: (x == 'winter').sum()),
        summer_flights=('season', lambda x: (x == 'summer').sum())
    ).reset_index()
    
    # Расчет доли сезонных перелетов
    user_agg_features['winter_ratio'] = user_agg_features['winter_flights'] / user_agg_features['total_flights']
    user_agg_features['summer_ratio'] = user_agg_features['summer_flights'] / user_agg_features['total_flights']
    
    # Создание категориальной фичи активности
    user_agg_features['activity_group'] = pd.cut(
        user_agg_features['total_flights'], 
        bins=[0, 5, 20, 50, np.inf],
        labels=['low', 'medium', 'high', 'very_high']
    ).astype(str)
    
    # Преобразование в словарь для LightFM
    user_features_dict = {}
    for _, row in user_agg_features.iterrows():
        user_features_dict[row['user_idx']] = {
            'total_flights': row['total_flights'],
            'avg_interaction': row['avg_interaction'],
            'cities_visited': row['cities_visited_count'],
            'winter_ratio': row['winter_ratio'],
            'summer_ratio': row['summer_ratio'],
            'activity_group': row['activity_group']
        }
    
    feature_logger.info(f"Создано фичей для {len(user_features_dict)} пользователей.")
    return user_features_dict

# -----------------------------------------------------------------------------------------------------------------------------------
def create_item_features(interactions_df, city_to_zone, seasons=None):
    """
    Создает фичи городов (items) на основе взаимодействий.
    
    Parameters:
    -----------
    interactions_df : pandas.DataFrame
        DataFrame с взаимодействиями
    city_to_zone : dict
        Словарь соответствия городов зонам
    seasons : list, optional
        Список сезонов для создания сезонных фич
        
    Returns:
    --------
    dict
        Словарь фичей элементов {item_idx: {feature: value}}
    """
    feature_logger.info("Начинается создание фичей городов...")
    if seasons is None:
        seasons = ['winter', 'summer']
    item_agg_features = interactions_df.groupby('item_idx').agg(
        popularity=('user_idx', 'count'),
        avg_interaction_city=('interaction_score', 'mean'),
        unique_users_count=('user_idx', 'nunique')
    ).reset_index()
    
    # Сезонная популярность
    if 'season' in interactions_df.columns:
        for season in seasons:
            season_mask = interactions_df['season'] == season
            season_popularity = interactions_df[season_mask].groupby('item_idx').size()
            item_agg_features[f'pop_{season}'] = item_agg_features['item_idx'].map(season_popularity).fillna(0)
    else:
        feature_logger.warning("Предупреждение: столбец 'season' не найден. Сезонные фичи не созданы.")
        for season in seasons:
            item_agg_features[f'pop_{season}'] = 0
            
    if city_to_zone: 
        item_agg_features['zone'] = item_agg_features['item_idx'].map(city_to_zone).fillna('unknown')
    else:
        feature_logger.warning("city_to_zone пуст или отсутствует. Присваиваем 'unknown' для зоны.")
        item_agg_features['zone'] = 'unknown'
    
    # Создание категориальной фичи популярности
    bins_popularity = [0, 100, 1000, 10000, np.inf]
    labels_popularity = ['very_low', 'low', 'medium', 'high']
    try:
        item_agg_features['popularity_group'] = pd.cut(item_agg_features['popularity'], 
                                                        bins=bins_popularity, labels=labels_popularity, 
                                                        right=False, include_lowest=True).astype(str)
    except Exception as e:
        feature_logger.error(f"Ошибка при создании popularity_group: {e}. Присваиваем 'unknown'.")
        item_agg_features['popularity_group'] = 'unknown'
    item_features_dict = item_agg_features.set_index('item_idx').apply(lambda row: row.to_dict(), axis=1).to_dict()
    feature_logger.info("Создание фичей городов завершено.")
    return item_features_dict

# -----------------------------------------------------------------------------------------------------------------------------------
def get_lightfm_embeddings(model, dataset, user_id_map, item_id_map):
    user_embeddings, user_biases = model.get_user_representations()
    item_embeddings, item_biases = model.get_item_representations()
    return user_embeddings, user_biases, item_embeddings, item_biases

# -----------------------------------------------------------------------------------------------------------------------------------
def create_user_and_item_features(interactions_df, user_flight_history, city_to_zone, full_flights_df):
    """
    Создает фичи для пользователей и городов на основе различных источников данных.
    
    Parameters:
    -----------
    interactions_df : pandas.DataFrame
        DataFrame с взаимодействиями
    user_flight_history : pandas.DataFrame
        История полетов пользователей
    city_to_zone : dict
        Словарь соответствия городов зонам
    full_flights_df : pandas.DataFrame
        Полные данные о полетах
        
    Returns:
    --------
    tuple
        (user_features_dict, item_features_dict) - словари фичей пользователей и элементов
    """
    feature_logger.info("Создание фичей пользователей и городов...")
    
    # Создание фичей пользователей
    user_agg_features = interactions_df.groupby('user_idx').agg(
        total_flights=('item_idx', 'count'), 
        avg_interaction=('interaction_score', 'mean'), 
        cities_visited_count=('item_idx', 'nunique')
    ).reset_index()

    if user_flight_history is not None and not user_flight_history.empty:
        # Проверяем, существуют ли колонки, прежде чем объединять
        cols_to_merge = ['user_idx']
        if 'total_flights_history' in user_flight_history.columns:
            cols_to_merge.append('total_flights_history')
        if 'avg_price_history' in user_flight_history.columns:
            cols_to_merge.append('avg_price_history')
        
        user_agg_features = pd.merge(user_agg_features, user_flight_history[cols_to_merge], 
                                     on='user_idx', how='left')
        
        if 'total_flights_history' in user_agg_features.columns:
            user_agg_features['total_flights_history'] = user_agg_features['total_flights_history'].fillna(0)
        else:
            user_agg_features['total_flights_history'] = 0
            
        if 'avg_price_history' in user_agg_features.columns:
            user_agg_features['avg_price_history'] = user_agg_features['avg_price_history'].fillna(0)
        else:
            user_agg_features['avg_price_history'] = 0
            
    else:
        feature_logger.warning("user_flight_history пуст или отсутствует. Добавляем фичи со значениями 0.")
        user_agg_features['total_flights_history'] = 0
        user_agg_features['avg_price_history'] = 0
    
    bins_flights = [0, 5, 20, 50, np.inf]
    labels_flights = ['low', 'medium', 'high', 'very_high']
    try:
        user_agg_features['activity_group'] = pd.cut(user_agg_features['total_flights'], 
                                                      bins=bins_flights, labels=labels_flights, 
                                                      right=False, include_lowest=True).astype(str)
    except Exception as e:
        feature_logger.error(f"Ошибка при создании activity_group: {e}. Присваиваем 'unknown'.")
        user_agg_features['activity_group'] = 'unknown'

    user_features_dict = user_agg_features.set_index('user_idx').apply(lambda row: row.to_dict(), axis=1).to_dict()

    # Создание фичей городов
    item_agg_features = interactions_df.groupby('item_idx').agg(
        popularity=('user_idx', 'count'),
        avg_interaction_city=('interaction_score', 'mean'),
        unique_users_count=('user_idx', 'nunique')
    ).reset_index()
    
    if city_to_zone:
        item_agg_features['zone'] = item_agg_features['item_idx'].map(city_to_zone).fillna('unknown')
    else:
        feature_logger.warning("city_to_zone пуст или отсутствует. Присваиваем 'unknown' для зоны.")
        item_agg_features['zone'] = 'unknown'

    bins_popularity = [0, 100, 1000, 10000, np.inf]
    labels_popularity = ['very_low', 'low', 'medium', 'high']
    try:
        item_agg_features['popularity_group'] = pd.cut(item_agg_features['popularity'], 
                                                        bins=bins_popularity, labels=labels_popularity, 
                                                        right=False, include_lowest=True).astype(str)
    except Exception as e:
        feature_logger.error(f"Ошибка при создании popularity_group: {e}. Присваиваем 'unknown'.")
        item_agg_features['popularity_group'] = 'unknown'

    full_flights_mapped = full_flights_df.copy()
    full_flights_mapped.dropna(subset=['item_idx'], inplace=True)
    full_flights_mapped['item_idx'] = full_flights_mapped['item_idx'].astype(int)
    
    flt_type_counts = full_flights_mapped.groupby('item_idx')['FLT_TYPE'].nunique().to_dict()
    svc_cls_counts = full_flights_mapped.groupby('item_idx')['SVC_CLS_DESC'].nunique().to_dict()

    item_agg_features['unique_flt_types'] = item_agg_features['item_idx'].map(flt_type_counts).fillna(0)
    item_agg_features['unique_svc_cls'] = item_agg_features['item_idx'].map(svc_cls_counts).fillna(0)
    
    item_features_dict = item_agg_features.set_index('item_idx').apply(lambda row: row.to_dict(), axis=1).to_dict()
    
    feature_logger.info("Создание фичей завершено.")
    return user_features_dict, item_features_dict