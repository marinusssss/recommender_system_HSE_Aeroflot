import optuna
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from src import config
from src.logger import setup_logger
from src.evaluation import _get_recommendations_for_user

# Настройка логгирования
model_training_logger = setup_logger('model_training', config.LOG_FILE_PATH, level=config.LOG_LEVEL)

def _create_matrices_for_optuna(df, num_users, num_items, include_visited=True): 
    seasonal_matrices = {}
    seasonal_visited_matrices = {}

    # Создание матриц для каждого сезона
    for season in df['season'].unique():
        season_df = df[df['season'] == season]

        rows_inter, cols_inter = season_df['user_idx'].values, season_df['item_idx'].values
        scores_inter = season_df['interaction_score'].values

        matrix_interactions = csr_matrix((scores_inter, (rows_inter, cols_inter)), shape=(num_users, num_items))
        seasonal_matrices[season] = matrix_interactions
        if include_visited:
            matrix_visited = csr_matrix((np.ones_like(rows_inter), (rows_inter, cols_inter)), shape=(num_users, num_items))
            seasonal_visited_matrices[season] = matrix_visited

    return seasonal_matrices, seasonal_visited_matrices

def optimize_als_hyperparams(train_data, val_data, num_users, num_items, n_trials=10, timeout=3600, random_state=42):
    model_training_logger.info("Запуск оптимизации гиперпараметров ALS с помощью Optuna")
    
    # Создание матриц для валидационной выборки
    model_training_logger.info("Создание матриц для валидационной выборки...")
    val_matrices_for_eval, val_visited_for_eval = _create_matrices_for_optuna(
        val_data, num_users, num_items, include_visited=True
    )
    # Группировка данных для оценки
    actual_relevant_items_per_user = val_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
    users_to_evaluate = sorted(list(actual_relevant_items_per_user.keys()))
    model_training_logger.info(f"Количество пользователей для оценки: {len(users_to_evaluate)}")
    
    def objective(trial):

        # Задание диапазона параметров для подбора лучших
        factors = trial.suggest_int('factors', 64, 256)
        regularization = trial.suggest_float('regularization', 1e-3, 0.1, log=True)
        iterations = trial.suggest_int('iterations', 10, 50)
        alpha = trial.suggest_float('alpha', 1.0, 40.0)

        model_training_logger.info(f"Начало пробного запуска Optuna #{trial.number} с параметрами: {trial.params}")

        seasonal_models = {}
        # Создание матриц для обучающих данных
        train_matrices, _ = _create_matrices_for_optuna(train_data, num_users, num_items, include_visited=True)
        
        for season, train_matrix in train_matrices.items():
            model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                alpha=alpha,
                random_state=random_state,
                num_threads=config.NUM_THREADS
            )
            model.fit(train_matrix)
            seasonal_models[season] = model

        # Оценка на валидационной выборке
        recalls = []
        sample_size_val = min(5000, len(users_to_evaluate))
        sampled_users = np.random.choice(users_to_evaluate, size=sample_size_val, replace=False)
        
        # Оценка recall для каждого пользователя
        for user_idx in tqdm(sampled_users, desc=f"Пробная версия №{trial.number}"):
            true_items = set(actual_relevant_items_per_user.get(user_idx, []))
            
            season = val_data[val_data['user_idx'] == user_idx]['season'].iloc[0]
            
            recommended = _get_recommendations_for_user( 
                user_idx,
                season,
                seasonal_models,
                val_matrices_for_eval,
                val_visited_for_eval,
                n=config.K_VALUES_EVALUATION[-1],
                allow_visited=config.ALLOW_VISITED_IN_RECOMMENDATIONS
            )
            
            if recommended is not None:
                hits = true_items.intersection(set(recommended))
                num_true_items = len(true_items)
                if num_true_items > 0:
                    recalls.append(len(hits) / num_true_items)
                else:
                    recalls.append(0.0)
        
        recall_at_k = np.mean(recalls) if recalls else 0.0
        model_training_logger.info(f"Пробная версия №{trial.number} завершена. Recall@K: {recall_at_k:.4f}")
        return recall_at_k

    study = optuna.create_study( 
        direction='maximize', # Максимизируем recall
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout) # Запуск оптимизации гиперпараметров

    model_training_logger.info(f"Оптимизация завершена. Лучшая пробная версия: {study.best_trial.value:.4f}")
    model_training_logger.info(f"Лучшие параметры: {study.best_params}")

    return study.best_params, study.trials_dataframe()

def train_als_seasonal_models(seasonal_train_matrices, factors, regularization, iterations, alpha):
    model_training_logger.info(
        f"Начало обучения финальных ALS моделей с параметрами: "
        f"factors={factors}, regularization={regularization}, iterations={iterations}, alpha={alpha}"
    )
    seasonal_models = {}
    
    # Обучение модели для каждого сезона
    for season, train_matrix in tqdm(seasonal_train_matrices.items(), desc="Обучение финальных ALS моделей"):
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            random_state=config.RANDOM_STATE,
            num_threads=config.NUM_THREADS
        )
        model.fit(train_matrix)
        seasonal_models[season] = model
        model_training_logger.info(f"Модель для сезона '{season}' успешно обучена.")
        
    return seasonal_models