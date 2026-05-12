# scripts/run_inference.py

import pandas as pd
import numpy as np
import os
import traceback

from src import config
from src.utils import load_pickle
from src.lightfm_model import recommend_lightfm
from src.catboost_ranker import prepare_catboost_data, predict_catboost_ranker
from src.data_processing import load_and_combine_flights
from src.feature_engineering import create_user_and_item_features
from src.logger import setup_logger

inference_logger = setup_logger('inference', config.LOG_FILE_PATH)

def main():
    """
    Основной пайплайн для генерации рекомендаций без учета сезонности.
    Создает CSV-отчет с рекомендациями для 100 случайных пользователей.
    
    Workflow:
    1. Загрузка артефактов моделей и данных
    2. Выбор пользователей для тестирования
    3. Создание фичей пользователей и городов
    4. Генерация кандидатов LightFM
    5. Подготовка данных для CatBoostRanker
    6. Получение финальных рекомендаций
    7. Формирование отчета в CSV
    """
    config.setup_directories()

    try:
        inference_logger.info("1. Загрузка артефактов моделей и исходных данных...")
        
        lightfm_artifacts = load_pickle(config.LIGHTFM_ARTIFACTS_PATH)
        lightfm_model = lightfm_artifacts['model']
        lightfm_dataset = lightfm_artifacts['dataset']
        lightfm_user_features_matrix = lightfm_artifacts.get('user_features_matrix')
        lightfm_item_features_matrix = lightfm_artifacts.get('item_features_matrix')
        user_id_map, _, item_id_map, _ = lightfm_dataset.mapping()
        
        catboost_artifacts = load_pickle(config.CATBOOST_RANKER_ARTIFACTS_PATH)
        catboost_model = catboost_artifacts['model']
        catboost_features = catboost_artifacts['features']
        catboost_categorical_features = catboost_artifacts['categorical_features']

        combined_flights_df = load_and_combine_flights(config.FLIGHTS_23_PATH, config.FLIGHTS_24_PATH)
        
        data_artifacts = load_pickle(config.DATA_ARTIFACTS_PATH)
        # Получаем DataFrame с историей полетов
        user_flight_history = data_artifacts.get('interactions_df', pd.DataFrame())
        
        if combined_flights_df is None or user_flight_history.empty:
            inference_logger.error("Не удалось загрузить необходимые данные. Завершение.")
            return

        # Дополнительно получаем маппинги и фичи
        user_idx_to_id = data_artifacts['user_idx_to_id']
        item_idx_to_name = data_artifacts['item_idx_to_name']
        
        inference_logger.info("Все артефакты и данные успешно загружены.")
        
        inference_logger.info("2. Выбор 5000 случайных пользователей...")
        all_user_idx = user_flight_history['user_idx'].unique()
        # Выбираем 5000 пользователей, у которых есть хотя бы 5 взаимодействий, 
        # чтобы история была более содержательной
        user_counts = user_flight_history['user_idx'].value_counts()
        eligible_users = user_counts[user_counts >= 5].index.tolist()
        
        if len(eligible_users) < 5000:
            inference_logger.warning("Недостаточно пользователей с 5+ взаимодействиями. Выбираем из всех доступных.")
            selected_user_idx = np.random.choice(all_user_idx, size=5000, replace=False)
        else:
            selected_user_idx = np.random.choice(eligible_users, size=5000, replace=False)

        inference_logger.info(f"Выбрано {len(selected_user_idx)} пользователей для генерации отчета.")
        
        inference_logger.info("3. Создание фичей для выбранных пользователей и городов...")
        
        # Фильтруем interactions_df только для выбранных пользователей
        selected_user_interactions = user_flight_history[
            user_flight_history['user_idx'].isin(selected_user_idx)
        ]
        
        # Переименовываем столбцы для правильной работы create_user_and_item_features
        combined_flights_df.rename(columns={'AIP_ARVL': 'city_name'}, inplace=True)
        item_name_to_idx = {name: idx for idx, name in item_idx_to_name.items()}
        combined_flights_df['item_idx'] = combined_flights_df['city_name'].map(item_name_to_idx)
        city_to_zone = lightfm_artifacts.get('city_to_zone', {})
        
        user_features_dict, item_features_dict = create_user_and_item_features(
            interactions_df=selected_user_interactions, 
            user_flight_history=data_artifacts.get('user_flight_history', pd.DataFrame()),
            city_to_zone=city_to_zone,
            full_flights_df=combined_flights_df
        )

        lightfm_candidates = recommend_lightfm(
            lightfm_model, 
            lightfm_dataset, 
            selected_user_idx,
            user_features_matrix=lightfm_user_features_matrix,
            item_features_matrix=lightfm_item_features_matrix,
            num_items=30  # Генерируем 30 кандидатов для реранкинга
        )
        
        empty_true_interactions = pd.DataFrame(columns=['user_idx', 'item_idx', 'relevance_score'])

        # Создаем датафрейм для предсказания CatBoost
        catboost_df, _ = prepare_catboost_data(
            lightfm_candidates, 
            empty_true_interactions, 
            user_features_dict, 
            item_features_dict, 
            lightfm_model, 
            user_id_map, 
            item_id_map, 
            {v: k for k, v in item_id_map.items()}, 
            lightfm_user_features_matrix, 
            lightfm_item_features_matrix
        )

        inference_logger.info(f"Подготовлено {len(catboost_df)} пар для CatBoost Ranker.")

        final_recommendations = predict_catboost_ranker(
            catboost_model, 
            catboost_df, 
            catboost_features, 
            catboost_categorical_features,
            num_recommendations=6
        )

        inference_logger.info(f"Получено финальных рекомендаций для {len(final_recommendations)} пользователей.")
        
        results_list = []
        for user_idx in selected_user_idx:
            # Получаем user_id из user_idx
            user_id = user_idx_to_id.get(user_idx, f'unknown_user_{user_idx}')
            
            # Получаем историю полетов из data_artifacts['interactions_df']
            user_history_df = user_flight_history[user_flight_history['user_idx'] == user_idx]
            
            # Группируем историю и формируем строку 'Город:количество'
            history_str = ", ".join([
                f"{item_idx_to_name.get(item_idx)}:{count}"
                for item_idx, count in user_history_df.groupby('item_idx').size().items()
            ])
            
            # Получаем рекомендации и преобразуем их из idx в имена городов
            recommended_items_idx = final_recommendations.get(user_idx, [])
            recommended_items_names = [item_idx_to_name.get(idx, f'unknown_city_{idx}') for idx in recommended_items_idx]

            row_data = {
                'user_id': user_id,
                'user_history': history_str
            }
            # Добавляем рекомендации в словарь
            for i, city in enumerate(recommended_items_names):
                row_data[f'recommendation_{i+1}'] = city
            
            results_list.append(row_data)

        results_df = pd.DataFrame(results_list)

        output_path = config.PROCESSED_DATA_DIR / 'lightfm_catboost_recommendations.csv'
        results_df.to_csv(output_path, index=False)

        inference_logger.info(f"Отчет с рекомендациями сохранен в: {output_path}")

    except Exception as e:
        inference_logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        traceback_str = traceback.format_exc()
        inference_logger.critical(f"Трассировка ошибки:\n{traceback_str}")

if __name__ == '__main__':
    main()