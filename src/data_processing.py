import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from src.logger import setup_logger
from src.config import DEFAULT_INTERACTION_COEF

data_logger = setup_logger('data_processing', 'logs/training.log')

# -----------------------------------------------------------------------------------------------------------------------------------
def load_and_combine_flights(path_23, path_24):
    """
    Загружает, объединяет и выполняет базовую чистку данных о полетах.
    
    Parameters:
    -----------
    path_23 : str
        Путь к данным за 2023 год
    path_24 : str
        Путь к данным за 2024 год
        
    Returns:
    --------
    pandas.DataFrame or None
        Объединенные данные о полетах или None при ошибке
    """
    try:
        # Загрузка данных из parquet файлов
        flights_23 = pd.read_parquet(path_23)
        flights_24 = pd.read_parquet(path_24)
    except FileNotFoundError as e:
        data_logger.error(f"ОШИБКА: Файл данных не найден: {e.filename}")
        data_logger.error("Пожалуйста, убедитесь, что файлы .parquet находятся в папке data/raw/")
        return None
    
    def filter_active_users(flights_data, min_flights=2, max_flights=200):
        """
        Фильтрация пользователей по активности.
        """
        user_activity = flights_data['FRQTFLR_CARD_ID'].value_counts()
        valid_users = user_activity[
            (user_activity >= min_flights) & 
            (user_activity <= max_flights)
        ].index
        
        filtered_data = flights_data[flights_data['FRQTFLR_CARD_ID'].isin(valid_users)]
        
        print(f"Фильтрация пользователей: {len(valid_users)} из {len(user_activity)} осталось")
        
        return filtered_data

    # Объединение данных и базовая очистка
    combined_flights = pd.concat([flights_23, flights_24], ignore_index=True)
    
    # Удаление записей, где аэропорт вылета и прилета совпадают
    combined_flights = combined_flights[combined_flights['AIP_ARVL'] != combined_flights['MAIN_AIRPORT']]
    
    # Удаление записей с пропущенными идентификаторами пользователей или городов
    combined_flights = combined_flights.dropna(subset=['FRQTFLR_CARD_ID', 'CITY_ARVL'])
    
    # Преобразование даты вылета в datetime
    combined_flights['SCHD_DEPTR_DT'] = pd.to_datetime(combined_flights['SCHD_DEPTR_DT'])
    
    # Сортируем по ID пользователя, затем по дате вылета от самой ранней до самой поздней
    combined_flights = combined_flights.sort_values(by=['FRQTFLR_CARD_ID', 'SCHD_DEPTR_DT'], ascending=[True, True]).reset_index(drop=True)
    
    combined_flights = filter_active_users(combined_flights, min_flights=2, max_flights=200)
    
    return combined_flights

# -----------------------------------------------------------------------------------------------------------------------------------
def prepare_seasonal_data(flights_data, interaction_coef=DEFAULT_INTERACTION_COEF, min_flights_per_user=2, max_flights_per_user=200):
    """
    Готовит данные для модели, возвращая все необходимые маппинги.
    
    Parameters:
    -----------
    flights_data : pandas.DataFrame
        Исходные данные о полетах
    interaction_coef : float, optional
        Коэффициент для преобразования количества полетов в оценку взаимодействия
    min_flights_per_user : int, optional
        Минимальное количество перелетов для включения пользователя
    max_flights_per_user : int, optional
        Максимальное количество перелетов для включения пользователя
        
    Returns:
    --------
    dict
        Словарь с подготовленными данными и метаинформацией
    """
    data_logger.info("Начинается подготовка сезонных данных...")

    # Выбор необходимых колонок и удаление пропущенных значений
    df = flights_data[['FRQTFLR_CARD_ID', 'CITY_ARVL', 'SEASON', 'SCHD_DEPTR_DT']].dropna().copy()
    df.columns = ['user_id', 'city_name', 'season', 'departure_date']
    
    user_flight_counts = df['user_id'].value_counts()
    valid_users = user_flight_counts[
        (user_flight_counts >= min_flights_per_user) & 
        (user_flight_counts <= max_flights_per_user)
    ].index
    
    df = df[df['user_id'].isin(valid_users)]
    data_logger.info(f"После фильтрации: {len(valid_users)} пользователей, {len(df)} перелетов")
    
    # Преобразование текстовых идентификаторов в числовые индексы начиная с 0
    df['user_idx'], user_map = pd.factorize(df['user_id'])
    df['item_idx'], item_map = pd.factorize(df['city_name'])
    
    # Создание обратных маппингов (индекс -> исходное значение)
    user_idx_to_id = dict(enumerate(user_map))
    item_idx_to_name = dict(enumerate(item_map))
    
    # Подсчет количества уникальных пользователей и элементов
    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()
    
    # Группировка по пользователям, элементам и сезонам
    interactions = df.groupby(['user_idx', 'item_idx', 'season']).size().reset_index(name='count')
    interactions['interaction_score'] = np.log1p(interactions['count']) * interaction_coef
    
    user_flight_history = df.groupby(['user_idx', 'item_idx']).size().reset_index(name='frequency')

    total_interactions = len(interactions)
    # Расчет популярности элементов (доля взаимодействий)
    item_popularity = (df['item_idx'].value_counts() / total_interactions).to_dict()

    data_logger.info("Подготовка данных завершена.")

    return {
        "interactions_df": interactions,
        "user_flight_history": user_flight_history,
        "user_idx_to_id": user_idx_to_id,
        'user_id_to_idx': {v: k for k, v in user_idx_to_id.items()},
        'item_idx_to_name': item_idx_to_name,
        "num_users": num_users,
        "num_items": num_items,
        "item_popularity": item_popularity,
        "full_temporal_data": df
    }

# -----------------------------------------------------------------------------------------------------------------------------------
def leave_n_out_split(full_temporal_data, n_val=2, n_test=2, min_interactions=6):
    """
    Функция реализует разделение данных Leave-N-Out с ГАРАНТИЕЙ временного порядка.
    Использует ГЛОБАЛЬНЫЕ индексы для избежания пересечений между наборами.
    
    Parameters:
    -----------
    full_temporal_data : pandas.DataFrame
        Полные временные данные с user_idx, item_idx, season, departure_date
    n_val : int, optional
        Количество взаимодействий для валидации
    n_test : int, optional
        Количество взаимодействий для теста
    min_interactions : int, optional
        Минимальное количество взаимодействий для включения пользователя
        
    Returns:
    --------
    tuple
        train_agg, val_agg, test_agg
    """
    data_logger.info("Начинается корректное разделение данных...")
    
    # Сортируем по пользователю и дате
    full_sorted = full_temporal_data.sort_values(['user_idx', 'departure_date']).reset_index(drop=True)
    
    # Создаем списки для ГЛОБАЛЬНЫХ индексов
    train_indices = []
    val_indices = []
    test_indices = []
    
    user_groups = full_sorted.groupby('user_idx')
    
    for user_idx, user_group in tqdm(user_groups, desc="Корректное разделение данных"):
        # user_group УЖЕ имеет правильные глобальные индексы из full_sorted
        num_flights = len(user_group)
        
        if num_flights >= min_interactions:
            test_start = num_flights - n_test
            val_start = test_start - n_val
            
            if val_start > 0:
                # ВАЖНО: используем .index из user_group напрямую (глобальные индексы)
                train_indices.extend(user_group.index[:val_start].tolist())
                val_indices.extend(user_group.index[val_start:test_start].tolist())
                test_indices.extend(user_group.index[test_start:].tolist())
    
    # Создаем разделенные DataFrame по ГЛОБАЛЬНЫМ индексам
    train_df = full_sorted.loc[train_indices].reset_index(drop=True)
    val_df = full_sorted.loc[val_indices].reset_index(drop=True)
    test_df = full_sorted.loc[test_indices].reset_index(drop=True)
    
    # Агрегируем взаимодействия для обучения моделей
    train_agg = train_df.groupby(['user_idx', 'item_idx', 'season']).size().reset_index(name='count')
    train_agg['interaction_score'] = np.log1p(train_agg['count']) * DEFAULT_INTERACTION_COEF
    
    val_agg = val_df.groupby(['user_idx', 'item_idx', 'season']).size().reset_index(name='count')
    val_agg['interaction_score'] = np.log1p(val_agg['count']) * DEFAULT_INTERACTION_COEF
    
    test_agg = test_df.groupby(['user_idx', 'item_idx', 'season']).size().reset_index(name='count')
    test_agg['interaction_score'] = np.log1p(test_agg['count']) * DEFAULT_INTERACTION_COEF
    
    data_logger.info(f"Обучающая выборка: {len(train_agg)} агрегированных взаимодействий")
    data_logger.info(f"Валидационная выборка: {len(val_agg)} агрегированных взаимодействий")
    data_logger.info(f"Тестовая выборка: {len(test_agg)} агрегированных взаимодействий")
    
    # Проверка пересечений по ГЛОБАЛЬНЫМ индексам
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    data_logger.info(f"Пересечения train-val: {len(train_set & val_set)}")
    data_logger.info(f"Пересечения train-test: {len(train_set & test_set)}")
    data_logger.info(f"Пересечения val-test: {len(val_set & test_set)}")
    
    # Проверка уникальности индексов
    total_unique = len(train_set) + len(val_set) + len(test_set)
    total_expected = len(train_indices) + len(val_indices) + len(test_indices)
    data_logger.info(f"Уникальность индексов: {total_unique == total_expected}")
    
    return train_agg, val_agg, test_agg

# -----------------------------------------------------------------------------------------------------------------------------------
def create_seasonal_matrices(df, num_users, num_items, include_visited=False):
    """
    Создает сезонные разреженные матрицы взаимодействий.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с взаимодействиями
    num_users : int
        Количество пользователей
    num_items : int
        Количество элементов
    include_visited : bool, optional
        Включать ли матрицы посещенных элементов
        
    Returns:
    --------
    tuple
        seasonal_matrices, seasonal_visited_matrices
    """
    seasonal_matrices = {}
    seasonal_visited_matrices = {}

    # Создание матриц для каждого сезона
    for season in df['season'].unique():
        season_df = df[df['season'] == season]
        
        # Извлечение данных для создания разреженных матриц
        rows_inter, cols_inter = season_df['user_idx'].values, season_df['item_idx'].values
        scores_inter = season_df['interaction_score'].values
        
        # Создание разреженной матрицы взаимодействий
        matrix_interactions = csr_matrix((scores_inter, (rows_inter, cols_inter)), shape=(num_users, num_items))
        seasonal_matrices[season] = matrix_interactions

        # Создание бинарной матрицы посещенных элементов
        if include_visited:
            matrix_visited = csr_matrix((np.ones_like(rows_inter), (rows_inter, cols_inter)), shape=(num_users, num_items))
            seasonal_visited_matrices[season] = matrix_visited

    return seasonal_matrices, seasonal_visited_matrices

# -----------------------------------------------------------------------------------------------------------------------------------
def create_zone_mapping(flights_data):
    """
    Создает маппинг город -> зона.
    
    Parameters:
    -----------
    flights_data : pandas.DataFrame
        Данные о полетах
        
    Returns:
    --------
    dict
        Словарь с маппингом города на зону
    """
    # Очистка зон от лишних пробелов перед созданием словаря
    flights_data['DEPTR_ZONE'] = flights_data['DEPTR_ZONE'].str.strip()
    flights_data['ARVL_ZONE'] = flights_data['ARVL_ZONE'].str.strip()
    
    # Создание маппингов для городов вылета и прилета
    deptr_mapping = flights_data[['CITY_DEPTR', 'DEPTR_ZONE']].dropna().drop_duplicates()
    arvl_mapping = flights_data[['CITY_ARVL', 'ARVL_ZONE']].dropna().drop_duplicates()
    
    # Преобразование в словари
    deptr_dict = deptr_mapping.set_index('CITY_DEPTR')['DEPTR_ZONE'].to_dict()
    arvl_dict = arvl_mapping.set_index('CITY_ARVL')['ARVL_ZONE'].to_dict()
    
    # Объединение словарей
    return {**deptr_dict, **arvl_dict}