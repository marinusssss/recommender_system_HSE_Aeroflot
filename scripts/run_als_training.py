import pandas as pd
import datetime
import os
from pathlib import Path
from src import config
from src.utils import save_pickle, load_pickle
from src.data_processing import (
    load_and_combine_flights,
    prepare_seasonal_data,
    leave_n_out_split,
    create_seasonal_matrices,
    create_zone_mapping
)
from src.als_model import optimize_als_hyperparams, train_als_seasonal_models
from src.evaluation import evaluate_and_compare, plot_results
from src.logger import setup_logger 


als_logger = setup_logger('als_training', config.LOG_FILE_PATH, level=config.LOG_LEVEL)

def main():
    """
    Основной пайплайн обучения ALS модели.
    
    Выполняет полный цикл подготовки данных, обучения модели и оценки:
    1. Загрузка и подготовка данных
    2. Разделение на train/val/test
    3. Оптимизация гиперпараметров
    4. Обучение финальной модели
    5. Оценка качества
    6. Сохранение артефактов
    """

    als_logger.info("--- Начинается пайплайн обучения модели ALS ---")
    config.setup_directories()
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    als_logger.info("1. Загрузка или создание подготовленных данных и разделение выборок")

    # Проверяем существование предварительно обработанных данных
    if (config.TRAIN_DATA_PATH.exists() and
            config.VAL_DATA_PATH.exists() and
            config.TEST_DATA_PATH.exists() and
            config.DATA_ARTIFACTS_PATH.exists()):

        als_logger.info("Найдены ранее сохраненные разделенные данные. Загрузка...")
        # Загрузка предобработанных данных
        train_data = pd.read_parquet(config.TRAIN_DATA_PATH)
        val_data = pd.read_parquet(config.VAL_DATA_PATH)
        test_data = pd.read_parquet(config.TEST_DATA_PATH)
        data_artifacts = load_pickle(config.DATA_ARTIFACTS_PATH)

        # Загрузка исходных данных для создания маппинга городов к зонам
        combined_flights = load_and_combine_flights(config.FLIGHTS_23_PATH, config.FLIGHTS_24_PATH)
        if combined_flights is None:
            als_logger.error("Не удалось загрузить исходные данные для создания city_to_zone. Завершение.")
            return

        als_logger.info(f"Загружено {len(train_data)} строк train, {len(val_data)} строк val, {len(test_data)} строк test.")

        # Объединяем все данные для пересчета метрик при необходимости
        all_data = pd.concat([train_data, val_data, test_data])
        
        # Проверяем и пересчитываем количество уникальных пользователей, если их нет в сохраненных данных
        if 'num_users' not in data_artifacts:
            data_artifacts['num_users'] = all_data['user_idx'].nunique()
        num_users = data_artifacts['num_users']

        # Проверяем и пересчитываем количество уникальных городов, если их нет в сохраненных данных
        if 'num_items' not in data_artifacts:
            data_artifacts['num_items'] = all_data['item_idx'].nunique()
        num_items = data_artifacts['num_items']
        
        # Аналогично проверяем и пересчитываем популярность городов
        if 'item_popularity' not in data_artifacts:
            als_logger.warning("Ключ 'item_popularity' не найден в артефактах. Пересчитываем...")
            item_popularity = (train_data['item_idx'].value_counts() / len(train_data)).to_dict()
            data_artifacts['item_popularity'] = item_popularity
        item_popularity = data_artifacts['item_popularity']
        
        user_flight_history = data_artifacts['user_flight_history']

    # Если предобработанных данных нет - создаем их с нуля
    else:
        als_logger.info("Разделенные данные не найдены. Выполнение полной подготовки и разделения")
        combined_flights = load_and_combine_flights(config.FLIGHTS_23_PATH, config.FLIGHTS_24_PATH)

        if combined_flights is None:
            als_logger.error("Не удалось загрузить исходные данные. Завершение.")
            return

        data_artifacts = prepare_seasonal_data(combined_flights, interaction_coef=config.DEFAULT_INTERACTION_COEF)
        
        # --- (2.) Разделение данных ---
        als_logger.info("2. Разделение данных...")
        interactions_df_for_split = data_artifacts['interactions_df']
        full_temporal_data = data_artifacts['full_temporal_data'] 

        train_data, val_data, test_data = leave_n_out_split(full_temporal_data)

        # Сохранение разделенных данных
        train_data.to_parquet(config.TRAIN_DATA_PATH, index=False)
        val_data.to_parquet(config.VAL_DATA_PATH, index=False)
        test_data.to_parquet(config.TEST_DATA_PATH, index=False)
        save_pickle(data_artifacts, config.DATA_ARTIFACTS_PATH)
        als_logger.info(f"Данные разделены и сохранены в {config.PROCESSED_DATA_DIR}.")

        # Извлечение метаинформации из артефактов
        num_users = data_artifacts['num_users']
        num_items = data_artifacts['num_items']
        user_flight_history = data_artifacts['user_flight_history']
        item_popularity = data_artifacts['item_popularity']

    # Объединение train и validation данных для финального обучения
    final_train_data = pd.concat([train_data, val_data])

    als_logger.info(f"Полное количество взаимодействий (из artifacts): {len(data_artifacts['interactions_df'])}")
    als_logger.info(f"Обучающая выборка: {len(train_data)}")
    als_logger.info(f"Валидационная выборка: {len(val_data)}")
    als_logger.info(f"Тестовая выборка: {len(test_data)}")

    # -----------------------------------------------------------------------------------------------------------------------------------
    als_logger.info("3. Оптимизация гиперпараметров")
    best_params, study_df = optimize_als_hyperparams(
        train_data, val_data, num_users, num_items,
        n_trials=config.OPTUNA_TRIALS
    )
    als_logger.info(f"Найдены лучшие параметры: {best_params}")

    # Сохранение результатов оптимизации гиперпараметров
    hyperparams_output_path = config.HYPERPARAMS_RESULTS_PATH
    study_df.to_csv(hyperparams_output_path, index=False)
    als_logger.info(f"Результаты подбора гиперпараметров сохранены в: {hyperparams_output_path}")

    # -----------------------------------------------------------------------------------------------------------------------------------
    als_logger.info("4. Обучение финальной модели")
    # Создание сезонных матриц для финального обучения
    final_seasonal_matrices, final_seasonal_visited_matrices_for_model = create_seasonal_matrices(
        final_train_data, num_users, num_items, include_visited=True
    )
    
    # Сохранение матриц
    save_pickle(final_seasonal_matrices, config.FINAL_SEASONAL_MATRICES_PATH)
    save_pickle(final_seasonal_visited_matrices_for_model, config.FINAL_SEASONAL_VISITED_MATRICES_PATH)
    als_logger.info("Сезонные матрицы сохранены.")

    # Обучение финальных моделей с лучшими параметрами
    seasonal_models = train_als_seasonal_models(final_seasonal_matrices, **best_params)
    als_logger.info("Финальные сезонные модели обучены.")

    # -----------------------------------------------------------------------------------------------------------------------------------
    als_logger.info("5. Оценка финальной модели")
    
    # Создание тестовых матриц для оценки
    _, test_visited_matrices = create_seasonal_matrices(
        test_data, num_users, num_items,
        include_visited=True
    )

    # Оценка качества модели
    results_df = evaluate_and_compare(
        test_data,
        seasonal_models,
        final_seasonal_matrices,
        test_visited_matrices,
        num_items,
        item_popularity,
        k_values=config.K_VALUES_EVALUATION,
        allow_visited=config.ALLOW_VISITED_IN_RECOMMENDATIONS
    )
    als_logger.info(f"Метрики качества финальной модели:\n{results_df}")

    # Визуализация результатов
    plot_path = config.VISUALIZATIONS_DIR / "final_model_performance.png"
    plot_results(results_df, k_values=config.K_VALUES_EVALUATION, save_path=str(plot_path))
    als_logger.info(f"График метрик сохранен в: {plot_path}")

    # Создание отчета об оценке
    with open(config.EVALUATION_SUMMARY_PATH, 'w') as f:
        f.write("--- Отчет по оценке модели ALS ---\n")
        f.write(f"Время выполнения: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Лучшие параметры ALS: {best_params}\n\n")
        f.write("Результаты метрик (ALS):\n")
        if 'ALS' in results_df.index.get_level_values('Model'):
            als_results = results_df.loc['ALS']
            f.write(als_results.to_string())
        else:
            f.write("ALS результаты не найдены в DataFrame.")
        f.write("\n\n")
    als_logger.info(f"Отчет по оценке сохранен в: {config.EVALUATION_SUMMARY_PATH}")

    # -----------------------------------------------------------------------------------------------------------------------------------
    als_logger.info("6. Сохранение всех артефактов для инференса")
    final_artifacts = {
        'seasonal_models': seasonal_models,
        'seasonal_matrices': final_seasonal_matrices,
        'user_id_to_idx': data_artifacts['user_id_to_idx'],
        'item_idx_to_name': data_artifacts['item_idx_to_name'],
        'user_flight_history': data_artifacts['user_flight_history'],
        'item_popularity': data_artifacts['item_popularity'],
        'city_to_zone': create_zone_mapping(combined_flights),
        'num_users': num_users,
        'num_items': num_items
    }
    save_pickle(final_artifacts, config.RECOMMENDER_ARTIFACTS_PATH)
    als_logger.info(f"Артефакты инференса сохранены в: {config.RECOMMENDER_ARTIFACTS_PATH}")



if __name__ == '__main__':
    main()