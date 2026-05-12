from lightfm import LightFM # type: ignore
from lightfm.data import Dataset # type: ignore
from lightfm.evaluation import precision_at_k, recall_at_k # type: ignore
import numpy as np # type: ignore
from scipy.sparse import coo_matrix # type: ignore
from src import config

from scipy.sparse import csr_matrix

def prepare_lightfm_dataset(interactions_df, user_features=None, item_features=None):
    """
    Подготавливает dataset для LightFM с проверкой соответствия ID.
    
    Создает dataset, добавляет пользователей, элементы и их признаки,
    строит матрицы взаимодействий и признаков.
    
    Parameters:
    -----------
    interactions_df : pandas.DataFrame
        DataFrame с взаимодействиями пользователь-элемент
    user_features : dict, optional
        Словарь с признаками пользователей
    item_features : dict, optional
        Словарь с признаками элементов
        
    Returns:
    --------
    tuple
        interactions, weights, dataset, user_features_matrix, item_features_matrix,
        user_id_map, user_idx_map, item_id_map, item_idx_map
    """
    # Инициализация dataset LightFM
    dataset = Dataset()
    
    # Получаем уникальные ID пользователей и городов из данных
    user_ids = interactions_df['user_idx'].unique()
    item_ids = interactions_df['item_idx'].unique()
    
    print(f"Подготовка dataset: {len(user_ids)} пользователей, {len(item_ids)} городов")
    
    # Добавляем пользователей и города в dataset
    dataset.fit(users=user_ids, items=item_ids)
    
    # Получаем маппинги LightFM
    user_id_map, user_idx_map, item_id_map, item_idx_map = dataset.mapping()
    
    # Собираем все возможные фичи
    all_user_features = set()
    all_item_features = set()
    
    # Фильтруем фичи пользователей по существующим ID
    filtered_user_features = {}
    if user_features:
        for user_id, features in user_features.items():
            if user_id in user_ids: # Только существующие пользователи
                filtered_user_features[user_id] = features
                all_user_features.update(features)
    
    # Фильтруем фичи городов по существующим ID
    filtered_item_features = {}
    if item_features:
        for item_id, features in item_features.items():
            if item_id in item_ids:
                filtered_item_features[item_id] = features
                all_item_features.update(features)
    
    # Добавляем сезонные фичи
    dataset.fit_partial(item_features=[f"season_{s}" for s in ['winter', 'summer']])
    
    # Добавляем пользовательские фичи и фичи городов в dataset
    if all_user_features:
        dataset.fit_partial(user_features=list(all_user_features))
    if all_item_features:
        dataset.fit_partial(item_features=list(all_item_features))
    
    # Строим матрицы взаимодействий с весами
    interactions, weights = dataset.build_interactions(
        [(row['user_idx'], row['item_idx'], row['interaction_score']) 
         for _, row in interactions_df.iterrows()]
    )
    
    # Строим матрицы признаков только для существующих ID
    user_features_matrix = None
    if filtered_user_features:
        user_features_list = [
            (user_id, list(features)) 
            for user_id, features in filtered_user_features.items()
        ]
        user_features_matrix = dataset.build_user_features(user_features_list, normalize=False)
    
    item_features_matrix = None
    if filtered_item_features:
        item_features_list = [
            (item_id, list(features)) 
            for item_id, features in filtered_item_features.items()
        ]
        item_features_matrix = dataset.build_item_features(item_features_list, normalize=False)
    
    return (interactions, weights, dataset, user_features_matrix, item_features_matrix,
            user_id_map, user_idx_map, item_id_map, item_idx_map)

def train_lightfm(interactions, weights, dataset, 
                 user_features_matrix=None, 
                 item_features_matrix=None,
                 num_components=30, 
                 loss='warp', 
                 learning_rate=0.05, 
                 epochs=20):
    """
    Обучает модель LightFM с поддержкой фичей.
    
    Parameters:
    -----------
    interactions : scipy.sparse matrix
        Матрица взаимодействий пользователь-элемент
    weights : scipy.sparse matrix
        Матрица весов взаимодействий
    dataset : lightfm.data.Dataset
        Dataset LightFM
    user_features_matrix : scipy.sparse matrix, optional
        Матрица признаков пользователей
    item_features_matrix : scipy.sparse matrix, optional
        Матрица признаков элементов
    num_components : int, optional
        Размерность латентного пространства
    loss : str, optional
        Функция потерь ('warp', 'bpr', 'logistic')
    learning_rate : float, optional
        Скорость обучения
    epochs : int, optional
        Количество эпох обучения
        
    Returns:
    --------
    LightFM
        Обученная модель LightFM
    """
    # Инициализация модели LightFM
    model = LightFM(
        no_components=num_components,
        loss=loss,
        learning_rate=learning_rate,
        random_state=config.RANDOM_STATE
    )
    
    # Обучение модели
    model.fit(
        interactions, 
        sample_weight=weights,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        epochs=epochs, 
        num_threads = 1
    )
    return model

def recommend_lightfm(model, dataset, user_ids, 
                               item_ids=None,  # Фильтрация по сезону
                               user_features_matrix=None,
                               item_features_matrix=None,
                               num_items=10):
    """
    Генерирует рекомендации с помощью обученной модели LightFM.
    
    Parameters:
    -----------
    model : LightFM
        Обученная модель
    dataset : lightfm.data.Dataset
        Dataset LightFM
    user_ids : list
        Список ID пользователей для рекомендаций
    item_ids : list, optional
        Список ID элементов для фильтрации
    user_features_matrix : scipy.sparse matrix, optional
        Матрица признаков пользователей
    item_features_matrix : scipy.sparse matrix, optional
        Матрица признаков элементов
    num_items : int, optional
        Количество рекомендаций на пользователя
        
    Returns:
    --------
    dict
        Словарь с рекомендациями для каждого пользователя
    """
    # Получение маппингов из dataset
    user_id_map, _, item_id_map, _ = dataset.mapping()
    recommendations = {}
    
    # Фильтр по релевантным городам (если указаны)
    if item_ids:
        item_ids_internal = [item_id_map[i] for i in item_ids if i in item_id_map]
    else:
        item_ids_internal = np.arange(len(item_id_map))
    
    # Генерация рекомендаций для каждого пользователя
    for user_id in user_ids:
        if user_id not in user_id_map:
            recommendations[user_id] = []
            continue
            
        internal_user_id = user_id_map[user_id]
        
        try:
            # Предсказание рейтингов только для релевантных городов
            scores = model.predict(
                user_ids=internal_user_id,
                item_ids=item_ids_internal,
                user_features=user_features_matrix,
                item_features=item_features_matrix
            )
            # Ранжирование элементов по убыванию score
            ranked_items = np.argsort(-scores)
            top_items = [item_ids_internal[i] for i in ranked_items[:num_items]]
            # Обратный маппинг к внешним ID
            inv_item_map = {v: k for k, v in item_id_map.items()}
            recommendations[user_id] = [inv_item_map[i] for i in top_items]
        except Exception:
            recommendations[user_id] = []
    
    return recommendations

# -----------------------------------------------------------------------------------------------------------------------------------
def get_lightfm_embeddings(model, dataset, user_id_map, item_id_map):
    """
    Извлекает эмбеддинги LightFM в виде словарей с проверкой типов.
    
    Parameters:
    -----------
    model : LightFM
        Обученная модель
    dataset : lightfm.data.Dataset
        Dataset LightFM
    user_id_map : dict
        Маппинг внешних ID пользователей к внутренним
    item_id_map : dict
        Маппинг внешних ID элементов к внутренним
        
    Returns:
    --------
    tuple
        user_embeddings, user_bias, item_embeddings, item_bias
    """
    # Получение внутренних представлений из модели
    user_emb, user_bias = model.get_user_representations()
    item_emb, item_bias = model.get_item_representations()
    
    # Преобразование в словари с проверкой типов
    user_embeddings = {}
    for external_id, internal_id in user_id_map.items():
        try:
            # Преобразуем internal_id в int
            internal_id_int = int(internal_id)
            if internal_id_int < user_emb.shape[0]:
                user_embeddings[external_id] = user_emb[internal_id_int]
        except (ValueError, TypeError):
            # Пропускаем нечисловые ID
            continue
    
    item_embeddings = {}
    for external_id, internal_id in item_id_map.items():
        try:
            # Преобразуем internal_id в int
            internal_id_int = int(internal_id)
            if internal_id_int < item_emb.shape[0]:
                item_embeddings[external_id] = item_emb[internal_id_int]
        except (ValueError, TypeError):
            # Пропускаем нечисловые ID
            continue
    
    return user_embeddings, user_bias, item_embeddings, item_bias