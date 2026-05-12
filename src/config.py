import os
from pathlib import Path
from src.logger import setup_logger

# Базовые пути 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
LOGS_DIR = BASE_DIR / "logs" 

# Настройка логирования 
LOG_FILE_PATH = LOGS_DIR / "training.log"
LOG_LEVEL = "INFO"
LOG_MAX_BYTES = 5 * 1024 * 1024 # 5 MB

# Настраиваем логгер для config
config_logger = setup_logger('config_setup', LOG_FILE_PATH, level=LOG_LEVEL)

# Названия файлов
FLIGHTS_23_PATH = RAW_DATA_DIR / "flights_with_zones_23.parquet"
FLIGHTS_24_PATH = RAW_DATA_DIR / "flights_with_zones_24.parquet"
MILES_ID_PATH = RAW_DATA_DIR / "id_miles.parquet"
MILES_DICT_PATH = RAW_DATA_DIR / "miles_dict.parquet"

# Названия моделей 
RECOMMENDER_ARTIFACTS_PATH = MODELS_DIR / "als_model.pkl"
LIGHTFM_ARTIFACTS_PATH = MODELS_DIR / "lightfm_model.pkl"
CATBOOST_RANKER_ARTIFACTS_PATH = MODELS_DIR / "catboost_ranker.pkl"

# Processed Data Paths
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / 'train_data.parquet'
VAL_DATA_PATH = PROCESSED_DATA_DIR / 'val_data.parquet'
TEST_DATA_PATH = PROCESSED_DATA_DIR / 'test_data.parquet'
DATA_ARTIFACTS_PATH = PROCESSED_DATA_DIR / 'data_artifacts.pickle'
FINAL_SEASONAL_MATRICES_PATH = PROCESSED_DATA_DIR / 'final_seasonal_matrices.pickle'
FINAL_SEASONAL_VISITED_MATRICES_PATH = PROCESSED_DATA_DIR / 'final_seasonal_visited_matrices.pickle'

# Параметры разделения данных
N_VAL_OUT = 2
N_TEST_OUT = 2
MIN_INTERACTIONS_FOR_SPLIT = 4


# Hyperparameter Optimization & Evaluation Paths
HYPERPARAMS_RESULTS_PATH = EXPERIMENTS_DIR / 'als_hyperparameter_results.csv' 
EVALUATION_RESULTS_PATH = EXPERIMENTS_DIR / 'als_evaluation_results.json' 
EVALUATION_SUMMARY_PATH = EXPERIMENTS_DIR / 'als_evaluation_summary.txt' 

# Детальный анализ ALS по квантилям
DETAILED_EVALUATION_PATH = EXPERIMENTS_DIR / 'detailed_evaluation_results.json'
USER_ACTIVITY_QUANTILES = [0.3, 0.7]
CITY_POPULARITY_QUANTILES = [0.3, 0.7]

# Параметры обучения и оценки
K_VALUES_EVALUATION = [3, 5, 7, 10]
DEFAULT_INTERACTION_COEF = 1
RANDOM_STATE = 42
NUM_THREADS = 3 
OPTUNA_TRIALS = 10 

# Общие параметры рекомендаций
ALLOW_VISITED_IN_RECOMMENDATIONS = True

# LightFM пути 
LIGHTFM_EVALUATION_RESULTS_PATH = EXPERIMENTS_DIR / 'lightfm_evaluation_results.json' 
LIGHTFM_VISUALIZATIONS_DIR = VISUALIZATIONS_DIR / 'lightfm'
LIGHTFM_NUM_THREADS = 1 

# CatBoostRanker пути 
CATBOOST_RANKER_VISUALIZATIONS_DIR = VISUALIZATIONS_DIR / 'catboost_ranker'
CATBOOST_CANDIDATE_COUNT = 30
CATBOOST_EPOCHS = 500
CATBOOST_LEARNING_RATE = 0.1
CATBOOST_EARLY_STOPPING_ROUNDS = 200

# CatBoostRanker параметры исключения популярных городов 
EXCLUDE_POPULAR_CITIES = False
POPULARITY_PERCENTILE = 95  # Процентиль популярности для исключения

# Создание директорий 
def setup_directories():
    """Создает все необходимые директории для проекта."""
    try:
        for path in [
            DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
            OUTPUTS_DIR, MODELS_DIR, VISUALIZATIONS_DIR, EXPERIMENTS_DIR, 
            LIGHTFM_VISUALIZATIONS_DIR, CATBOOST_RANKER_VISUALIZATIONS_DIR,
            LOGS_DIR
        ]:
            path.mkdir(parents=True, exist_ok=True)
        config_logger.info("Структура папок проверена и готова к работе.")
    except Exception as e:
        config_logger.critical(f"Критическая ошибка при создании структуры папок: {e}")