import pandas as pd
import numpy as np
from catboost import CatBoostRanker, Pool
from src import config
from tqdm import tqdm
from src.logger import setup_logger


ranker_logger = setup_logger('catboost_ranker', config.LOG_FILE_PATH, level=config.LOG_LEVEL)


# -----------------------------------------------------------------------------------------------------------------------------------
def prepare_catboost_data(lightfm_candidates, true_interactions_df, 
                          user_features_dict, item_features_dict, 
                          lightfm_model, user_id_map, item_id_map, inv_item_map,
                          lightfm_user_features_matrix=None, lightfm_item_features_matrix=None,
                          exclude_popular_cities=None):
    ranker_logger.info("Подготовка данных для CatBoostRanker")
    relevance_lookup = {}
    user_true_items_dict = {}
    
    for _, row in true_interactions_df.iterrows():
        uid = row['user_idx']
        iid = row['item_idx']
        cnt = row.get('count', 1)  # count из агрегации в leave_n_out_split
        
        key = (uid, iid)
        relevance_lookup[key] = relevance_lookup.get(key, 0) + cnt
        
        if uid not in user_true_items_dict:
            user_true_items_dict[uid] = set()
        user_true_items_dict[uid].add(iid)
    
    ranker_logger.info(f"Построен словарь релевантности: {len(relevance_lookup)} пар (user, item)")

    ranker_data = []
    excluded_negative_count = 0
    total_negative_count = 0
    
    for user_idx, candidates in tqdm(lightfm_candidates.items(), desc="Генерация обучающих пар для CatBoost", miniters=100):
        # Получение истинных items для пользователя
        user_true_items = user_true_items_dict.get(user_idx, set())
        
        # Объединение кандидатов и истинных items
        all_items_for_user = set(candidates)
        for true_item in user_true_items:
            all_items_for_user.add(true_item) 
        
        # Получение внутреннего ID пользователя для LightFM
        internal_user_id = user_id_map.get(user_idx)
        lightfm_rank_map = {item: rank for rank, item in enumerate(candidates)}
        
        for item_idx in all_items_for_user:
            is_positive = item_idx in user_true_items
            
            if not is_positive and exclude_popular_cities and item_idx in exclude_popular_cities:
                excluded_negative_count += 1
                continue
                
            total_negative_count += not is_positive
            
            row = {'user_idx': user_idx, 'item_idx': item_idx}
            if is_positive:
                count = relevance_lookup.get((user_idx, item_idx), 1)
                row['relevance_score'] = float(np.log1p(count))
            else:
                row['relevance_score'] = 0.0
            
            # Получение скора LightFM для пары (user, item)
            internal_item_id = item_id_map.get(item_idx)
            if internal_user_id is not None and internal_item_id is not None:
                lightfm_score = lightfm_model.predict(
                    user_ids=np.array([internal_user_id]), 
                    item_ids=np.array([internal_item_id]), 
                    user_features=lightfm_user_features_matrix, 
                    item_features=lightfm_item_features_matrix
                )[0]
                row['lightfm_score'] = lightfm_score
            else:
                row['lightfm_score'] = 0.0
            row['lightfm_rank'] = lightfm_rank_map.get(item_idx, -1)
            
            # Добавление фичей пользователя
            user_feats = user_features_dict.get(user_idx, {})
            for k, v in user_feats.items():
                row[f'user_{k}'] = v
            
            # Добавление фичей города
            item_feats = item_features_dict.get(item_idx, {})
            for k, v in item_feats.items():
                row[f'item_{k}'] = v
            
            ranker_data.append(row)

    ranker_df = pd.DataFrame(ranker_data)
    
    ranker_df['query_id'] = ranker_df['user_idx']
    # Статистика по взвешенной релевантности
    pos_mask = ranker_df['relevance_score'] > 0
    ranker_logger.info(
        f"Релевантность: позитивных={pos_mask.sum()}, "
        f"среднее={ranker_df.loc[pos_mask, 'relevance_score'].mean():.3f}, "
        f"макс={ranker_df.loc[pos_mask, 'relevance_score'].max():.3f}"
    )
    
    categorical_features = []
    if user_features_dict:
        sample_user_feats = next(iter(user_features_dict.values()))
        for k, v in sample_user_feats.items():
            if isinstance(v, (str, bool)) or pd.isna(v):
                categorical_features.append(f'user_{k}')
    if item_features_dict:
        sample_item_feats = next(iter(item_features_dict.values()))
        for k, v in sample_item_feats.items():
            if isinstance(v, (str, bool)) or pd.isna(v):
                categorical_features.append(f'item_{k}')

    final_categorical_features = []
    for col in categorical_features:
        if col in ranker_df.columns:
            ranker_df[col] = ranker_df[col].astype(str).replace('nan', 'Unknown_Category')
            final_categorical_features.append(col)
        else:
            ranker_logger.warning(f"Предупреждение: Категориальная фича '{col}' не найдена в DataFrame. Пропускаем.")
    
    ranker_logger.info(f"Подготовлено {len(ranker_df)} пар для CatBoostRanker.")
    ranker_logger.info(f"Категориальные фичи: {final_categorical_features}")

    return ranker_df, final_categorical_features



def train_catboost_ranker(train_df, val_df, categorical_features,
                          user_features_dict, 
                          item_features_dict,  
                          iterations=config.CATBOOST_EPOCHS,
                          learning_rate=config.CATBOOST_LEARNING_RATE,
                          early_stopping_rounds=config.CATBOOST_EARLY_STOPPING_ROUNDS):
    ranker_logger.info("Обучение CatBoostRanker")

    features = [col for col in train_df.columns if col not in ['user_idx', 'item_idx', 'relevance_score', 'query_id']]
    X_train = train_df[features]
    y_train = train_df['relevance_score']
    groups_train = train_df['query_id']
    X_val = val_df[features]
    y_val = val_df['relevance_score']
    groups_val = val_df['query_id']

    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=groups_train,
        cat_features=[X_train.columns.get_loc(col) for col in categorical_features if col in X_train.columns] 
    )
    val_pool = Pool(
        data=X_val,
        label=y_val,
        group_id=groups_val,
        cat_features=[X_val.columns.get_loc(col) for col in categorical_features if col in X_val.columns] 
    )

    model = CatBoostRanker(
        iterations=iterations,
        learning_rate=learning_rate,
        random_seed=config.RANDOM_STATE,
        loss_function='YetiRankPairwise', 
        verbose=100,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric='NDCG:top=10' 
    )
    model.fit(train_pool, eval_set=val_pool)
    
    artifacts = {
        'model': model,
        'features': features,
        'categorical_features': categorical_features,
        'user_features_dict': user_features_dict,
        'item_features_dict': item_features_dict
    }
    ranker_logger.info("Обучение CatBoostRanker завершено.")
    return artifacts

def predict_catboost_ranker(model, data_df, features, categorical_features,
                             user_col='user_idx', item_col='item_idx', num_recommendations=config.K_VALUES_EVALUATION[-1]):
    ranker_logger.info("Получение предсказаний от CatBoostRanker")
    
    # Подготовка данных для предсказания
    predictions_df = data_df[[user_col, item_col] + features].copy()

    for col in categorical_features:
        if col in predictions_df.columns:
            predictions_df[col] = predictions_df[col].astype(str).replace('nan', 'Unknown_Category')
    
    predictions_df['predicted_relevance'] = model.predict(predictions_df[features])

    # Генерация рекомендаций
    recommendations = {}
    for user_idx, group in tqdm(predictions_df.groupby(user_col), desc="Генерация финальных рекомендаций CatBoost"):
        top_items = group.sort_values(by='predicted_relevance', ascending=False)[item_col].tolist()
        recommendations[user_idx] = top_items[:num_recommendations]
        
    return recommendations
