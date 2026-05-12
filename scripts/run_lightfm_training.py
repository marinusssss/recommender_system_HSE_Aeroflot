import datetime
import pandas as pd
from src import config
from src.utils import save_pickle, load_pickle
from src.data_processing import (
    load_and_combine_flights, 
    create_zone_mapping,
    prepare_seasonal_data,
    leave_n_out_split
)
from src.lightfm_model import prepare_lightfm_dataset, train_lightfm
from src.feature_engineering import create_user_features, create_item_features
from src.evaluation import evaluate_and_compare_lightfm, plot_results
from src.logger import setup_logger
import os
import traceback

lightfm_logger = setup_logger('lightfm_training', config.LOG_FILE_PATH, level=config.LOG_LEVEL)

# -----------------------------------------------------------------------------------------------------------------------------------
def main():
    """
    Основной пайплайн обучения модели LightFM.
    
    Выполняет полный цикл подготовки данных, обучения модели и оценки:
    1. Загрузка и подготовка данных
    2. Создание пользовательских и товарных фич
    3. Подготовка dataset для LightFM
    4. Обучение модели
    5. Сохранение артефактов
    6. Оценка качества
    """
    
    config.setup_directories()
    
    try:
        lightfm_logger.info("1. Загрузка или создание подготовленных данных и разделение выборок")

        # Проверяем существование предварительно обработанных данных
        if (config.TRAIN_DATA_PATH.exists() and
                config.VAL_DATA_PATH.exists() and
                config.TEST_DATA_PATH.exists() and
                config.DATA_ARTIFACTS_PATH.exists()):
            
            lightfm_logger.info("Найдены ранее сохраненные разделенные данные. Загрузка")
            # Загрузка предобработанных данных
            train_data = pd.read_parquet(config.TRAIN_DATA_PATH)
            val_data = pd.read_parquet(config.VAL_DATA_PATH)
            test_data = pd.read_parquet(config.TEST_DATA_PATH)
            data_artifacts = load_pickle(config.DATA_ARTIFACTS_PATH)
            combined_flights = load_and_combine_flights(config.FLIGHTS_23_PATH, config.FLIGHTS_24_PATH)
            
            lightfm_logger.info(f"Загружено {len(train_data)} строк train, {len(val_data)} строк val, {len(test_data)} строк test.")
            
        else:
            # Если предобработанных данных нет - создаем их с нуля
            lightfm_logger.info("Разделенные данные не найдены. Выполнение полной подготовки и разделения...")
            combined_flights = load_and_combine_flights(config.FLIGHTS_23_PATH, config.FLIGHTS_24_PATH)

            if combined_flights is None:
                lightfm_logger.error("Не удалось загрузить исходные данные. Завершение.")
                return

            data_artifacts = prepare_seasonal_data(combined_flights, interaction_coef=config.DEFAULT_INTERACTION_COEF)

            lightfm_logger.info("2. Разделение данных")
            interactions_df_for_split = data_artifacts['interactions_df']
            
            # Разделение данных методом Leave-N-Out
            full_temporal_data = data_artifacts['full_temporal_data'] 

            train_data, val_data, test_data = leave_n_out_split(full_temporal_data)

            # Сохранение разделенных данных
            train_data.to_parquet(config.TRAIN_DATA_PATH, index=False)
            val_data.to_parquet(config.VAL_DATA_PATH, index=False)
            test_data.to_parquet(config.TEST_DATA_PATH, index=False)
            save_pickle(data_artifacts, config.DATA_ARTIFACTS_PATH)
            lightfm_logger.info(f"Данные разделены и сохранены в {config.PROCESSED_DATA_DIR}.")
        
        
        # Объединение train и validation данных для финального обучения
        full_train = pd.concat([train_data, val_data])
        
        # -----------------------------------------------------------------------------------------------------------------------------------
        lightfm_logger.info("2. Создание фичей для пользователей...")
        user_features = create_user_features(full_train, data_artifacts['user_flight_history'])
        lightfm_logger.info(f"Создано фичей для {len(user_features)} пользователей.")
        
        # -----------------------------------------------------------------------------------------------------------------------------------
        lightfm_logger.info("3. Создание фичей для городов")
        
        # Создание маппинга городов к зонам
        city_to_zone = create_zone_mapping(combined_flights) if combined_flights is not None else {}
        
        # Создание признаков городов (элементов)
        item_features = create_item_features(
            full_train, 
            city_to_zone,
            seasons=['winter', 'summer']
        )
        lightfm_logger.info(f"Создано фичей для {len(item_features)} городов.")
        
        lightfm_logger.info(f"Общий размер обучающих данных: {len(full_train)} записей.")
        
        # -----------------------------------------------------------------------------------------------------------------------------------
        lightfm_logger.info("4. Подготовка dataset для LightFM")
        
        # Подготовка dataset LightFM с взаимодействиями и признаками
        interactions, weights, dataset, user_features_matrix, item_features_matrix, \
            user_id_map, user_idx_map, item_id_map, item_idx_map = prepare_lightfm_dataset(
            full_train,
            user_features=user_features,
            item_features=item_features
        )
        
        # -----------------------------------------------------------------------------------------------------------------------------------
        lightfm_logger.info("5. Обучение модели LightFM")
        
        # Обучение модели LightFM с использованием WARP loss
        model = train_lightfm(
            interactions, 
            weights, 
            dataset,
            user_features_matrix=user_features_matrix,
            item_features_matrix=item_features_matrix,
            num_components=64,
            loss='warp',
            epochs=30
        )
        lightfm_logger.info("Модель LightFM успешно обучена.")
        
        # -----------------------------------------------------------------------------------------------------------------------------------
        lightfm_logger.info("6. Сохранение артефактов для инференса")
        
        lightfm_artifacts = {
            'model': model,                                        
            'dataset': dataset,                                     
            'user_features_matrix': user_features_matrix,           
            'item_features_matrix': item_features_matrix,          
            'user_id_to_idx': data_artifacts['user_id_to_idx'],    
            'item_idx_to_name': data_artifacts['item_idx_to_name'], 
            'city_to_zone': city_to_zone,                      
            'user_id_map': user_id_map,                           
            'user_idx_map': user_idx_map,                           
            'item_id_map': item_id_map,                             
            'item_idx_map': item_idx_map,                           
        }

        if 'user_flight_history' in data_artifacts:
            lightfm_artifacts['user_flight_history'] = data_artifacts['user_flight_history']

        # Сохранение артефактов для последующего использования
        save_pickle(lightfm_artifacts, config.LIGHTFM_ARTIFACTS_PATH)
        lightfm_logger.info(f"Артефакты LightFM сохранены в: {config.LIGHTFM_ARTIFACTS_PATH}")
        
        # -----------------------------------------------------------------------------------------------------------------------------------
        lightfm_logger.info("7. Оценка финальной модели")
        
        # Оценка качества модели на тестовых данных
        results_df = evaluate_and_compare_lightfm(
            test_data, 
            lightfm_artifacts,
            k_values=config.K_VALUES_EVALUATION
        )
        lightfm_logger.info(f"Метрики качества финальной модели:\n{results_df}")
        
        # Визуализация результатов
        plot_path = config.LIGHTFM_VISUALIZATIONS_DIR / "final_lightfm_performance.png"
        os.makedirs(config.LIGHTFM_VISUALIZATIONS_DIR, exist_ok=True)
        plot_results(results_df, k_values=config.K_VALUES_EVALUATION, save_path=str(plot_path))
        lightfm_logger.info(f"График метрик сохранен в: {plot_path}")
        
        # Создание текстового отчета
        evaluation_summary_path = config.EXPERIMENTS_DIR / 'lightfm_evaluation_summary.txt'
        with open(evaluation_summary_path, 'w') as f:
            f.write(f"Время выполнения: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Результаты метрик (LightFM):\n")
            if 'LightFM' in results_df.index.get_level_values('Model'):
                lightfm_results = results_df.loc['LightFM']
                f.write(lightfm_results.to_string())
            else:
                f.write("Результаты LightFM не найдены в DataFrame.")
            f.write("\n\n")
            
            f.write("Сравнение с базовыми моделями:\n")
            f.write(results_df.to_string())
            
        lightfm_logger.info(f"Отчет по оценке сохранен в: {evaluation_summary_path}")

        # Сохранение результатов в JSON файл
        os.makedirs(os.path.dirname(config.LIGHTFM_EVALUATION_RESULTS_PATH), exist_ok=True)
        results_df.to_json(config.LIGHTFM_EVALUATION_RESULTS_PATH, indent=4)
        lightfm_logger.info(f"Результаты оценки сохранены в: {config.LIGHTFM_EVALUATION_RESULTS_PATH}")
        
    
    except Exception as e:
        lightfm_logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}.")
        traceback_str = traceback.format_exc()
        lightfm_logger.critical(f"Трассировка ошибки:\n{traceback_str}")


if __name__ == '__main__':
    main()