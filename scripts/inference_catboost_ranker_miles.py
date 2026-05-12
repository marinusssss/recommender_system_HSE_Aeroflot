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
from src.miles_logic import load_miles_data, create_zone_mapping, get_flight_miles_cost

# Настраиваем логгер для нового скрипта
inference_logger = setup_logger('inference', config.LOG_FILE_PATH)

def main():
    """
    Основной пайплайн для генерации рекомендаций с учетом баланса миль.
    Создает CSV-отчет с рекомендациями, фильтрованными по доступности миль.
    
    Workflow:
    1. Загрузка артефактов и данных о милях
    2. Выбор пользователей для тестирования
    3. Создание фичей и генерация кандидатов
    4. Фильтрация рекомендаций по балансу миль
    5. Формирование отчета с информацией о стоимости перелетов
    """
    config.setup_directories()

    try:
        inference_logger.info("1. Загрузка артефактов моделей и исходных данных")
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
        user_flight_history = data_artifacts.get('interactions_df', pd.DataFrame())
        
        miles_id_df, miles_dict_df = load_miles_data()  # баланс миль пользователей и стоимость перелетов между зонами
        if combined_flights_df is None or user_flight_history.empty:
            inference_logger.error("Не удалось загрузить необходимые данные. Завершение.")
            return

        user_idx_to_id = data_artifacts['user_idx_to_id']
        item_idx_to_name = data_artifacts['item_idx_to_name']
        inference_logger.info("Все артефакты и данные успешно загружены.")
        
        inference_logger.info("2. Выбор 5000 случайных пользователей...")
        all_user_idx = user_flight_history['user_idx'].unique()
        user_counts = user_flight_history['user_idx'].value_counts()
        eligible_users = user_counts[user_counts >= 5].index.tolist()  # выбираем пользователей, у которых >=5 перелетов в истории
        if len(eligible_users) < 5000:
            inference_logger.warning("Недостаточно пользователей с 5+ взаимодействиями. Выбираем из всех доступных.")
            selected_user_idx = np.random.choice(all_user_idx, size=5000, replace=False)
        else:
            selected_user_idx = np.random.choice(eligible_users, size=5000, replace=False)  # выбираем случайных 5000

        inference_logger.info(f"Выбрано {len(selected_user_idx)} пользователей для генерации отчета.")
        
        inference_logger.info("3. Создание фичей для выбранных пользователей и городов...")
        selected_user_interactions = user_flight_history[
            user_flight_history['user_idx'].isin(selected_user_idx)
        ]
        combined_flights_df.rename(columns={'AIP_ARVL': 'city_name'}, inplace=True)
        item_name_to_idx = {name: idx for idx, name in item_idx_to_name.items()}
        combined_flights_df['item_idx'] = combined_flights_df['city_name'].map(item_name_to_idx)
        
        # Очищаем данные о зонах перед созданием city_to_zone
        combined_flights_df['DEPTR_ZONE'] = combined_flights_df['DEPTR_ZONE'].str.strip()
        combined_flights_df['ARVL_ZONE'] = combined_flights_df['ARVL_ZONE'].str.strip()
        city_to_zone = create_zone_mapping(combined_flights_df)
        user_features_dict, item_features_dict = create_user_and_item_features(
            interactions_df=selected_user_interactions,
            user_flight_history=data_artifacts.get('user_flight_history', pd.DataFrame()),
            city_to_zone=city_to_zone,
            full_flights_df=combined_flights_df
        )

        inference_logger.info("4. Генерация кандидатов от LightFM (100 кандидатов)...")
        lightfm_candidates = recommend_lightfm(  # генерируется 100 кандидатов от LightFM, которые будут переранжированы CatBoostRanker
            lightfm_model,
            lightfm_dataset,
            selected_user_idx,
            user_features_matrix=lightfm_user_features_matrix,
            item_features_matrix=lightfm_item_features_matrix,
            num_items=100
        )
        inference_logger.info("5. Подготовка данных для CatBoost Ranker")
        empty_true_interactions = pd.DataFrame(columns=['user_idx', 'item_idx', 'relevance_score'])

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

        inference_logger.info("6. Получение финальных рекомендаций от CatBoost Ranker...")
        final_recommendations = predict_catboost_ranker(  
            catboost_model,
            catboost_df,
            catboost_features,
            catboost_categorical_features,
            num_recommendations=30
        )

        inference_logger.info(f"Получено финальных рекомендаций от ранжировщика для {len(final_recommendations)} пользователей.")

        inference_logger.info("7. Фильтрация рекомендаций по балансу миль")
        miles_balance_map = miles_id_df.set_index('FRQTFLR_CARD_ID')['END_BALANCE'].to_dict()
        most_popular_deptr_zone = combined_flights_df['DEPTR_ZONE'].mode()[0]
        
        filtered_recommendations_details = {}
        user_deptr_zones = {}

        for user_idx, recommended_items_idx in final_recommendations.items():
            user_id = user_idx_to_id.get(user_idx)
            user_balance = miles_balance_map.get(user_id, 0)
            user_history_entry = combined_flights_df[combined_flights_df['FRQTFLR_CARD_ID'] == user_id]
            
            deptr_zone = most_popular_deptr_zone
            if not user_history_entry.empty:
                deptr_zone = user_history_entry['DEPTR_ZONE'].mode()[0]
            
            deptr_zone = deptr_zone.strip()
            user_deptr_zones[user_idx] = deptr_zone
            
            high_demand_list = []
            low_demand_list = []

            for item_idx in recommended_items_idx:
                if len(high_demand_list) >= 6 and len(low_demand_list) >= 6:
                    break
                
                arvl_city_name = item_idx_to_name.get(item_idx)
                if not arvl_city_name:
                    continue
                
                arvl_zone = city_to_zone.get(arvl_city_name)
                if not arvl_zone:
                    continue

                arvl_zone = arvl_zone.strip()
                
                # Порядок классов для проверки
                flight_classes = ['Бизнес', 'Комфорт', 'Эконом']

                # Поиск для высокого спроса
                if len(high_demand_list) < 6:
                    for flight_class in flight_classes:
                        demand_costs = get_flight_miles_cost(deptr_zone, arvl_zone, miles_dict_df, flight_class)
                        if demand_costs['Высокий'] is not None and user_balance >= demand_costs['Высокий']:
                            high_demand_list.append({
                                'item_idx': item_idx,
                                'class': flight_class,
                                'cost': demand_costs['Высокий'],
                                'demand_type': 'Высокий'
                            })
                            break # Если нашли подходящий класс, переходим к следующему городу

                # Поиск для низкого спроса
                if len(low_demand_list) < 6:
                    for flight_class in flight_classes:
                        demand_costs = get_flight_miles_cost(deptr_zone, arvl_zone, miles_dict_df, flight_class)
                        if demand_costs['Низкий'] is not None and user_balance >= demand_costs['Низкий']:
                            low_demand_list.append({
                                'item_idx': item_idx,
                                'class': flight_class,
                                'cost': demand_costs['Низкий'],
                                'demand_type': 'Низкий'
                            })
                            break # Если нашли подходящий класс, переходим к следующему городу

            filtered_recommendations_details[user_idx] = {
                'high_demand': high_demand_list,
                'low_demand': low_demand_list
            }

        inference_logger.info("Фильтрация по милям завершена.")

        inference_logger.info("8. Сбор данных в итоговый DataFrame")
        results_list = []
        for user_idx in selected_user_idx:
            user_id = user_idx_to_id.get(user_idx, f'unknown_user_{user_idx}')
            user_history_df = user_flight_history[user_flight_history['user_idx'] == user_idx]
            history_str = ", ".join([
                f"{item_idx_to_name.get(item_idx)}:{count}"
                for item_idx, count in user_history_df.groupby('item_idx').size().items()
            ])
            
            recommended_items = filtered_recommendations_details.get(user_idx, {})
            high_demand_list = recommended_items.get('high_demand', [])
            low_demand_list = recommended_items.get('low_demand', [])
            
            # Добавляем данные для высокого спроса
            row_data_high = {
                'user_id': user_id,
                'user_history': history_str,
                'miles_balance': miles_balance_map.get(user_id, 0),
                'home_departure_zone': user_deptr_zones.get(user_idx, 'unknown'),
                'demand_type': 'Высокий',
                'recommendation_sent': 1 if len(high_demand_list) >= 6 else 0
            }
            for i in range(6):
                if i < len(high_demand_list):
                    item_details = high_demand_list[i]
                    item_idx = item_details['item_idx']
                    row_data_high[f'recommendation_{i+1}'] = item_idx_to_name.get(item_idx)
                    row_data_high[f'class_{i+1}'] = item_details['class']
                    row_data_high[f'cost_{i+1}'] = item_details['cost']
                else:
                    row_data_high[f'recommendation_{i+1}'] = np.nan
                    row_data_high[f'class_{i+1}'] = np.nan
                    row_data_high[f'cost_{i+1}'] = np.nan
            results_list.append(row_data_high)
            
            # Добавляем данные для низкого спроса
            row_data_low = {
                'user_id': user_id,
                'user_history': history_str,
                'miles_balance': miles_balance_map.get(user_id, 0),
                'home_departure_zone': user_deptr_zones.get(user_idx, 'unknown'),
                'demand_type': 'Низкий',
                'recommendation_sent': 1 if len(low_demand_list) >= 6 else 0
            }
            for i in range(6):
                if i < len(low_demand_list):
                    item_details = low_demand_list[i]
                    item_idx = item_details['item_idx']
                    row_data_low[f'recommendation_{i+1}'] = item_idx_to_name.get(item_idx)
                    row_data_low[f'class_{i+1}'] = item_details['class']
                    row_data_low[f'cost_{i+1}'] = item_details['cost']
                else:
                    row_data_low[f'recommendation_{i+1}'] = np.nan
                    row_data_low[f'class_{i+1}'] = np.nan
                    row_data_low[f'cost_{i+1}'] = np.nan
            results_list.append(row_data_low)

        results_df = pd.DataFrame(results_list)
        
        # Упорядочиваем колонки в итоговом DataFrame
        cols = ['user_id', 'user_history', 'miles_balance', 'recommendation_sent', 'demand_type', 'home_departure_zone']
        for i in range(6):
            cols.extend([f'recommendation_{i+1}', f'class_{i+1}', f'cost_{i+1}'])
        results_df = results_df[cols]

        output_path = config.PROCESSED_DATA_DIR / 'lightfm_catboost_miles_recommendations.csv'
        results_df.to_csv(output_path, index=False)

        inference_logger.info(f"Отчет с рекомендациями по милям сохранен в: {output_path}")

    except Exception as e:
        inference_logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        traceback_str = traceback.format_exc()
        inference_logger.critical(f"Трассировка ошибки:\n{traceback_str}")

if __name__ == '__main__':
    main()