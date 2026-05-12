import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from src import config
from src.utils import load_pickle, save_pickle
from src.catboost_ranker import prepare_catboost_data, predict_catboost_ranker, train_catboost_ranker
from src.feature_engineering import create_user_and_item_features
from src.evaluation import evaluate_catboost_ranker
from src.data_processing import load_and_combine_flights
from src.lightfm_model import recommend_lightfm
from src.logger import setup_logger
import traceback

ranker_logger = setup_logger('catboost_reranking', config.LOG_FILE_PATH, level=config.LOG_LEVEL)


def main():

    config.setup_directories()
    try:

        ranker_logger.info("1. Загрузка данных и артефактов")
        combined_flights_df = load_and_combine_flights(config.FLIGHTS_23_PATH, config.FLIGHTS_24_PATH)
        if combined_flights_df is None:
            ranker_logger.error("Не удалось загрузить исходные данные. Завершение.")
            return

        train_data = pd.read_parquet(config.TRAIN_DATA_PATH)
        val_data   = pd.read_parquet(config.VAL_DATA_PATH)
        test_data  = pd.read_parquet(config.TEST_DATA_PATH)
        ranker_logger.info(
            f"Данные загружены: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
        )

        lightfm_artifacts = load_pickle(config.LIGHTFM_ARTIFACTS_PATH)
        lightfm_model               = lightfm_artifacts['model']
        lightfm_dataset             = lightfm_artifacts['dataset']
        lightfm_user_features_matrix = lightfm_artifacts['user_features_matrix']
        lightfm_item_features_matrix  = lightfm_artifacts['item_features_matrix']
        user_id_map, _, item_id_map, _ = lightfm_dataset.mapping()
        inv_item_map = {v: k for k, v in item_id_map.items()}
        city_to_zone = lightfm_artifacts.get('city_to_zone', {})

        data_artifacts = load_pickle(config.DATA_ARTIFACTS_PATH)
        user_flight_history = data_artifacts.get('user_flight_history', pd.DataFrame())

        combined_flights_df = combined_flights_df.rename(columns={'AIP_ARVL': 'city_name'})
        item_name_to_idx = {name: idx for idx, name in lightfm_artifacts['item_idx_to_name'].items()}
        combined_flights_df['item_idx'] = combined_flights_df['city_name'].map(item_name_to_idx)

        ranker_logger.info("2. Создание фичей пользователей и городов (на train+val)")
        full_train = pd.concat([train_data, val_data])

        user_features_dict, item_features_dict = create_user_and_item_features(
            interactions_df=full_train,
            user_flight_history=user_flight_history,
            city_to_zone=city_to_zone,
            full_flights_df=combined_flights_df
        )
        ranker_logger.info(
            f"3. Генерация кандидатов от LightFM для train пользователей "
            f"(K={config.CATBOOST_CANDIDATE_COUNT}):"
        )
        train_users = train_data['user_idx'].unique()
        lightfm_candidates_train = recommend_lightfm(
            lightfm_model, lightfm_dataset, train_users,
            user_features_matrix=lightfm_user_features_matrix,
            item_features_matrix=lightfm_item_features_matrix,
            num_items=config.CATBOOST_CANDIDATE_COUNT
        )
        ranker_logger.info(f"Кандидатов для train: {len(lightfm_candidates_train)} пользователей.")

        ranker_logger.info("4. Подготовка обучающих пар для CatBoost:")
        exclude_popular = None
        if config.EXCLUDE_POPULAR_CITIES:
            popularity = train_data['item_idx'].value_counts()
            threshold = popularity.quantile(config.POPULARITY_PERCENTILE / 100)
            exclude_popular = set(popularity[popularity >= threshold].index.tolist())
            ranker_logger.info(f"Исключаем {len(exclude_popular)} популярных городов из негативов.")

        train_df_catboost, final_cat_features = prepare_catboost_data(
            lightfm_candidates_train,
            train_data,                      # ← relevance labels из train, не из test!
            user_features_dict,
            item_features_dict,
            lightfm_model,
            user_id_map, item_id_map, inv_item_map,
            lightfm_user_features_matrix,
            lightfm_item_features_matrix,
            exclude_popular_cities=exclude_popular
        )
        ranker_logger.info(f"Обучающих пар для CatBoost: {len(train_df_catboost)}")

        ranker_logger.info(
            f"5. Генерация кандидатов от LightFM для val пользователей "
            f"(K={config.CATBOOST_CANDIDATE_COUNT})"
        )
        val_users = val_data['user_idx'].unique()
        lightfm_candidates_val = recommend_lightfm(
            lightfm_model, lightfm_dataset, val_users,
            user_features_matrix=lightfm_user_features_matrix,
            item_features_matrix=lightfm_item_features_matrix,
            num_items=config.CATBOOST_CANDIDATE_COUNT
        )

        val_df_catboost, _ = prepare_catboost_data(
            lightfm_candidates_val,
            val_data,                    
            user_features_dict,
            item_features_dict,
            lightfm_model,
            user_id_map, item_id_map, inv_item_map,
            lightfm_user_features_matrix,
            lightfm_item_features_matrix,
            exclude_popular_cities=None   
        )
        ranker_logger.info(f"Валидационных пар для CatBoost: {len(val_df_catboost)}")

        ranker_logger.info("6. Обучение CatBoostRanker")
        catboost_artifacts = train_catboost_ranker(
            train_df_catboost,
            val_df_catboost,
            final_cat_features,
            user_features_dict,
            item_features_dict,
            iterations=config.CATBOOST_EPOCHS,
            learning_rate=config.CATBOOST_LEARNING_RATE,
            early_stopping_rounds=config.CATBOOST_EARLY_STOPPING_ROUNDS
        )

        save_pickle(catboost_artifacts, config.CATBOOST_RANKER_ARTIFACTS_PATH)
        ranker_logger.info(f"Артефакты CatBoostRanker сохранены: {config.CATBOOST_RANKER_ARTIFACTS_PATH}")

        model_ranker  = catboost_artifacts['model']
        ranker_features = catboost_artifacts['features']
        cat_features    = catboost_artifacts['categorical_features']


        ranker_logger.info("7. Генерация кандидатов для test пользователей")
        test_users = test_data['user_idx'].unique()
        lightfm_candidates_test = recommend_lightfm(
            lightfm_model, lightfm_dataset, test_users,
            user_features_matrix=lightfm_user_features_matrix,
            item_features_matrix=lightfm_item_features_matrix,
            num_items=config.CATBOOST_CANDIDATE_COUNT
        )

        test_df_catboost, _ = prepare_catboost_data(
            lightfm_candidates_test,
            test_data,                    
            user_features_dict,
            item_features_dict,
            lightfm_model,
            user_id_map, item_id_map, inv_item_map,
            lightfm_user_features_matrix,
            lightfm_item_features_matrix,
            exclude_popular_cities=None
        )

        ranker_logger.info("8. Получение финальных рекомендаций от CatBoostRanker")
        catboost_recommendations = predict_catboost_ranker(
            model_ranker, test_df_catboost, ranker_features, cat_features,
            num_recommendations=config.K_VALUES_EVALUATION[-1]
        )
        ranker_logger.info(
            f"Рекомендации сгенерированы для {len(catboost_recommendations)} пользователей."
        )

        ranker_logger.info("9. Оценка CatBoostRanker на test_data")
        results_df = evaluate_catboost_ranker(
            test_data,
            catboost_recommendations,
            k_values=config.K_VALUES_EVALUATION,
            save_path=config.CATBOOST_RANKER_VISUALIZATIONS_DIR
        )

        ranker_logger.info(f"Результаты:\n{results_df}")

    except FileNotFoundError as e:
        ranker_logger.error(f"Файл не найден: {e.filename}. Проверьте пути в config.py.")
    except Exception as e:
        ranker_logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        ranker_logger.critical(f"Трассировка:\n{traceback.format_exc()}")


if __name__ == '__main__':
    main()
