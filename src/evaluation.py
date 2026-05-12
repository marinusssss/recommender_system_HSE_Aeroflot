import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json 

from .config import K_VALUES_EVALUATION, ALLOW_VISITED_IN_RECOMMENDATIONS, VISUALIZATIONS_DIR, EVALUATION_RESULTS_PATH

from src import config
import seaborn as sns
from .evaluation_utils import assign_user_activity_groups, assign_city_popularity_groups, calculate_group_metrics
from .lightfm_model import recommend_lightfm
from catboost import CatBoostRanker # Для загрузки модели
from src.logger import setup_logger
try:
    ALLOW_VISITED_IN_RECOMMENDATIONS = config.ALLOW_VISITED_IN_RECOMMENDATIONS
except AttributeError:
    ALLOW_VISITED_IN_RECOMMENDATIONS = True # Значение по умолчанию

evaluation_logger = setup_logger('evaluation', config.LOG_FILE_PATH, level=config.LOG_LEVEL)

def run_complex_evaluation(test_data, full_interactions, seasonal_models, seasonal_matrices, seasonal_visited, 
                          num_items, item_popularity, k_values=K_VALUES_EVALUATION):
    """
    Проводит комплексную оценку модели на различных подгруппах пользователей и городов. (для оценки ALS модели)
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Тестовые данные для оценки
    full_interactions : pandas.DataFrame
        Полные данные взаимодействий для определения групп
    seasonal_models : dict
        Словарь сезонных моделей {season: model}
    seasonal_matrices : dict
        Словарь сезонных матриц взаимодействий
    seasonal_visited : dict
        Словарь посещенных элементов по сезонам
    num_items : int
        Общее количество элементов
    item_popularity : dict
        Словарь популярности элементов {item_idx: popularity_score}
    k_values : list, optional
        Список значений K для оценки
        
    Returns:
    --------
    dict
        Словарь с результатами оценки по всем группам
    """
    results = {}
    
    evaluation_logger.info("Общая оценка на тестовом наборе")
    overall_results_df = evaluate_and_compare(
        test_data, seasonal_models, seasonal_matrices, seasonal_visited,
        num_items, item_popularity, k_values
    )
    
    # Извлекаем результаты ALS
    als_results = overall_results_df.loc['ALS']
    
    # Преобразуем результаты в словарь с плоской структурой
    overall_metrics = {}
    for k in k_values:
        for metric in ['Recall', 'Precision', 'F1-Score', 'NDCG', 'MAP']:
            key = f"{metric}@{k}"
            overall_metrics[key] = als_results.loc[k, metric]
    
    results['overall'] = overall_metrics
    
    evaluation_logger.info("Определение групп пользователей и городов")
    user_activity = assign_user_activity_groups(full_interactions)
    city_popularity = assign_city_popularity_groups(full_interactions)
    
    # 3. Подготовка данных с группами
    evaluation_logger.info("Объединение данных с группами...")
    test_data_with_groups = test_data.merge(
        user_activity, on='user_idx', how='left'
    ).merge(
        city_popularity, on='item_idx', how='left'
    )
    
    # Заполнение пропусков
    test_data_with_groups['activity_group'] = test_data_with_groups['activity_group'].fillna('medium')
    test_data_with_groups['popularity_group'] = test_data_with_groups['popularity_group'].fillna('medium')
    
    # 4. Оценка по группам пользователей
    evaluation_logger.info("Оценка по группам пользователей")
    for group in ['low', 'medium', 'high']:
        group_mask = test_data_with_groups['activity_group'] == group
        group_size = group_mask.sum()
        evaluation_logger.info(f"Оценка для пользователей: {group} (n={group_size})")
        
        if group_size == 0:
            results[f"user_{group}"] = {f"{metric}@{k}": 0 for k in k_values for metric in ['Recall', 'Precision', 'NDCG']}
            evaluation_logger.warning("  Пропуск: нет данных")
            continue
        
        group_metrics = calculate_group_metrics(
            test_data_with_groups, group_mask, seasonal_models,
            seasonal_matrices, seasonal_visited, k_values,
            allow_visited=config.ALLOW_VISITED_IN_RECOMMENDATIONS
        )
        results[f"user_{group}"] = group_metrics
        evaluation_logger.info(f"  Recall@10: {group_metrics.get('Recall@10', 0):.4f}")
    
    evaluation_logger.info("Оценка по группам городов")
    for group in ['low', 'medium', 'high']:
        group_mask = test_data_with_groups['popularity_group'] == group
        group_size = group_mask.sum()
        evaluation_logger.info(f"Оценка для городов: {group} (n={group_size})")
        
        if group_size == 0:
            results[f"city_{group}"] = {f"{metric}@{k}": 0 for k in k_values for metric in ['Recall', 'Precision', 'NDCG']}
            evaluation_logger.warning("  Пропуск: нет данных")
            continue
        
        group_metrics = calculate_group_metrics(
            test_data_with_groups, group_mask, seasonal_models,
            seasonal_matrices, seasonal_visited, k_values,
            allow_visited=config.ALLOW_VISITED_IN_RECOMMENDATIONS
        )
        results[f"city_{group}"] = group_metrics
        evaluation_logger.info(f"  Recall@10: {group_metrics.get('Recall@10', 0):.4f}")
    
    evaluation_logger.info("Оценка по комбинированным группам (Recall@10)")
    heatmap_data = {}
    for user_group in ['low', 'medium', 'high']:
        for city_group in ['low', 'medium', 'high']:
            group_mask = (
                (test_data_with_groups['activity_group'] == user_group) & 
                (test_data_with_groups['popularity_group'] == city_group))
            group_size = group_mask.sum()
            group_name = f"user_{user_group}_city_{city_group}"
            evaluation_logger.info(f"Оценка для: {group_name} (n={group_size})")
            
            if group_size == 0:
                recall = 0
                results[group_name] = {'Recall@10': recall}
                heatmap_data[(user_group, city_group)] = recall
                evaluation_logger.info(f"  Recall@10: {recall:.4f} (пропуск)")
                continue
            
            group_metrics = calculate_group_metrics(
                test_data_with_groups, group_mask, seasonal_models,
                seasonal_matrices, seasonal_visited, [10],
                allow_visited=config.ALLOW_VISITED_IN_RECOMMENDATIONS
            )
            recall = group_metrics.get('Recall@10', 0)
            results[group_name] = {'Recall@10': recall}
            heatmap_data[(user_group, city_group)] = recall
            evaluation_logger.info(f"  Recall@10: {recall:.4f}")
    
    visualize_group_results(results, heatmap_data)
    
    return results

# -----------------------------------------------------------------------------------------------------------------------------------    
def visualize_group_results(results, heatmap_data):
    heatmap_matrix = pd.DataFrame(
        index=['low', 'medium', 'high'],
        columns=['low', 'medium', 'high']
    )
    
    for (user_group, city_group), recall in heatmap_data.items():
        heatmap_matrix.loc[user_group, city_group] = recall
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_matrix.astype(float), annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Recall@10 по группам пользователей и городов")
    plt.xlabel("Популярность города")
    plt.ylabel("Активность пользователя")
    plt.savefig(config.VISUALIZATIONS_DIR / "recall_heatmap.png", dpi=300)
    
    user_groups = {k: v for k, v in results.items() if k.startswith('user_')}
    plot_comparison(user_groups, "Сравнение по группам пользователей")
    
    city_groups = {k: v for k, v in results.items() if k.startswith('city_')}
    plot_comparison(city_groups, "Сравнение по группам городов")

# -----------------------------------------------------------------------------------------------------------------------------------
def plot_comparison(group_data, title):
    """
    Строит графики сравнения метрик для групп.
    
    Parameters:
    -----------
    group_data : dict
        Данные групп для сравнения
    title : str
        Заголовок графика
    """
    metrics = ['Recall@5', 'Recall@10', 'Precision@5', 'Precision@10', 'NDCG@10']
    fig, axes = plt.subplots(1, len(metrics), figsize=(25, 5))
    
    for i, metric in enumerate(metrics):
        values = [data.get(metric, 0) for data in group_data.values()]
        groups = list(group_data.keys())
        
        axes[i].bar(groups, values, color=['#2C7BB6', '#ABD9E9', '#FDAE61'])
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1)
        
        # Добавление значений на столбцы
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(config.VISUALIZATIONS_DIR / f"{title.replace(' ', '_')}.png", dpi=300)

#-------------------------------------------------------------------------------------
def _get_recommendations_for_user(user_idx, season, seasonal_models_dict, seasonal_matrices_dict, seasonal_visited_dict, n, allow_visited):
    """
    Вспомогательная функция для получения рекомендаций и их фильтрации (ALS).
    
    Parameters:
    -----------
    user_idx : int
        ID пользователя
    season : str
        Сезон для выбора модели
    seasonal_models_dict : dict
        Словарь сезонных моделей
    seasonal_matrices_dict : dict
        Словарь сезонных матриц
    seasonal_visited_dict : dict
        Словарь посещенных элементов
    n : int
        Количество рекомендаций
    allow_visited : bool
        Разрешить рекомендовать посещенные элементы
        
    Returns:
    --------
    list or None
        Список рекомендованных элементов или None при ошибке
    """
    if season not in seasonal_models_dict or season not in seasonal_matrices_dict:
        return None

    model_for_season = seasonal_models_dict[season]
    matrix_for_season = seasonal_matrices_dict[season]

    if user_idx >= matrix_for_season.shape[0]:
        return None

    user_items = matrix_for_season[user_idx]

    filter_items = None
    if not allow_visited and season in seasonal_visited_dict:
        if user_idx < seasonal_visited_dict[season].shape[0]:
            filter_items = seasonal_visited_dict[season][user_idx].indices

    try:
        recommended, _ = model_for_season.recommend(
            user_idx,
            user_items,
            N=n,
            filter_already_liked_items=not allow_visited,
            filter_items=filter_items
        )
        return recommended
    except Exception:
        return None

# -----------------------------------------------------------------------------------------------------------------------------------
def evaluate_and_compare(test_data, seasonal_models, seasonal_matrices, seasonal_visited,
                         num_items, item_popularity, k_values=K_VALUES_EVALUATION, allow_visited=ALLOW_VISITED_IN_RECOMMENDATIONS):
    """
    Проводит общую оценку рекомендательной системы и сравнение с базовыми методами.
    Метрики рассчитываются по пользователям, а затем усредняются. (ALS)
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Тестовые данные
    seasonal_models : dict
        Словарь сезонных моделей
    seasonal_matrices : dict
        Словарь сезонных матриц
    seasonal_visited : dict
        Словарь посещенных элементов
    num_items : int
        Количество элементов
    item_popularity : dict
        Словарь популярности элементов
    k_values : list
        Список значений K для оценки
    allow_visited : bool
        Разрешить рекомендовать посещенные элементы
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame с результатами оценки
    """
    # Группировка тестовых данных по пользователям
    actual_relevant_items_per_user = test_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
    users_to_evaluate = sorted(list(actual_relevant_items_per_user.keys()))
    
    # Создание списка для хранения всех результатов
    all_results = []
    
    # Предрассчет top-K для TopPopular
    top_items = sorted(range(num_items), key=lambda idx: item_popularity.get(idx, 0), reverse=True)

    for k in k_values:    
        # --- Оценка ALS ---
        als_precisions, als_recalls, als_f1_scores, als_ndcg_scores, als_ap_scores, als_hit_rates = [], [], [], [], [], []
        for user_idx in tqdm(users_to_evaluate, desc=f"Evaluating ALS @{k}"):
            true_items = set(actual_relevant_items_per_user.get(user_idx, []))
            
            # Находим сезон для текущего пользователя
            season = test_data[test_data['user_idx'] == user_idx]['season'].iloc[0]
            
            recommended = _get_recommendations_for_user(
                user_idx, season, seasonal_models, seasonal_matrices, seasonal_visited, k, allow_visited
            )

            if recommended is not None:
                hits = true_items.intersection(set(recommended))
                num_hits = len(hits)
                num_true_items = len(true_items)
                
                if num_true_items > 0:
                    als_precisions.append(num_hits / k)
                    als_recalls.append(num_hits / num_true_items)
                    
                    if num_hits > 0:
                        als_f1_scores.append(2 * (als_precisions[-1] * als_recalls[-1]) / (als_precisions[-1] + als_recalls[-1]))
                    else:
                        als_f1_scores.append(0.0)

                    als_ndcg_scores.append(calculate_ndcg_at_k_per_user(recommended, true_items, k))
                    als_ap_scores.append(calculate_ap_at_k_per_user(recommended, true_items, k))
                    als_hit_rates.append(1 if num_hits > 0 else 0)
        
        # Добавляем результаты ALS
        all_results.append({
            'Model': 'ALS', 'K': k,
            'Recall': np.mean(als_recalls) if als_recalls else 0,
            'Precision': np.mean(als_precisions) if als_precisions else 0,
            'F1-Score': np.mean(als_f1_scores) if als_f1_scores else 0,
            'NDCG': np.mean(als_ndcg_scores) if als_ndcg_scores else 0,
            'MAP': np.mean(als_ap_scores) if als_ap_scores else 0,
            'HitRate': np.mean(als_hit_rates) if als_hit_rates else 0,
        })


        rand_precisions, rand_recalls, rand_f1_scores, rand_ndcg_scores, rand_ap_scores, rand_hit_rates = [], [], [], [], [], []
        for user_idx in tqdm(users_to_evaluate, desc=f"Evaluating Random @{k}"):
            true_items = set(actual_relevant_items_per_user.get(user_idx, []))
            
            # Генерируем случайные рекомендации
            random_items = np.random.choice(num_items, size=k, replace=False)
            
            hits = true_items.intersection(set(random_items))
            num_hits = len(hits)
            num_true_items = len(true_items)
            
            if num_true_items > 0:
                rand_precisions.append(num_hits / k)
                rand_recalls.append(num_hits / num_true_items)
                
                if num_hits > 0:
                    rand_f1_scores.append(2 * (rand_precisions[-1] * rand_recalls[-1]) / (rand_precisions[-1] + rand_recalls[-1]))
                else:
                    rand_f1_scores.append(0.0)

                rand_ndcg_scores.append(calculate_ndcg_at_k_per_user(random_items, true_items, k))
                rand_ap_scores.append(calculate_ap_at_k_per_user(random_items, true_items, k))
                rand_hit_rates.append(1 if num_hits > 0 else 0)

        # Добавляем результаты Random
        all_results.append({
            'Model': 'Random', 'K': k,
            'Recall': np.mean(rand_recalls) if rand_recalls else 0,
            'Precision': np.mean(rand_precisions) if rand_precisions else 0,
            'F1-Score': np.mean(rand_f1_scores) if rand_f1_scores else 0,
            'NDCG': np.mean(rand_ndcg_scores) if rand_ndcg_scores else 0,
            'MAP': np.mean(rand_ap_scores) if rand_ap_scores else 0,
            'HitRate': np.mean(rand_hit_rates) if rand_hit_rates else 0,
        })

        top_precisions, top_recalls, top_f1_scores, top_ndcg_scores, top_ap_scores, top_hit_rates = [], [], [], [], [], []
        current_top_k_items = top_items[:k]
        
        for user_idx in tqdm(users_to_evaluate, desc=f"Evaluating TopPopular @{k}"):
            true_items = set(actual_relevant_items_per_user.get(user_idx, []))
            
            hits = true_items.intersection(set(current_top_k_items))
            num_hits = len(hits)
            num_true_items = len(true_items)
            
            if num_true_items > 0:
                top_precisions.append(num_hits / k)
                top_recalls.append(num_hits / num_true_items)
                
                if num_hits > 0:
                    top_f1_scores.append(2 * (top_precisions[-1] * top_recalls[-1]) / (top_precisions[-1] + top_recalls[-1]))
                else:
                    top_f1_scores.append(0.0)
                    
                top_ndcg_scores.append(calculate_ndcg_at_k_per_user(current_top_k_items, true_items, k))
                top_ap_scores.append(calculate_ap_at_k_per_user(current_top_k_items, true_items, k))
                top_hit_rates.append(1 if num_hits > 0 else 0)

        # Добавляем результаты TopPopular
        all_results.append({
            'Model': 'TopPopular', 'K': k,
            'Recall': np.mean(top_recalls) if top_recalls else 0,
            'Precision': np.mean(top_precisions) if top_precisions else 0,
            'F1-Score': np.mean(top_f1_scores) if top_f1_scores else 0,
            'NDCG': np.mean(top_ndcg_scores) if top_ndcg_scores else 0,
            'MAP': np.mean(top_ap_scores) if top_ap_scores else 0,
            'HitRate': np.mean(top_hit_rates) if top_hit_rates else 0,
        })

    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index(['Model', 'K'])

    # Сохранение результатов в JSON файл
    os.makedirs(os.path.dirname(config.EVALUATION_RESULTS_PATH), exist_ok=True)
    results_df.to_json(config.EVALUATION_RESULTS_PATH, indent=4)
    evaluation_logger.info(f"Результаты оценки сохранены в: {config.EVALUATION_RESULTS_PATH}")

    return results_df

# -----------------------------------------------------------------------------------------------------------------------------------
def plot_results(results_df, k_values=K_VALUES_EVALUATION, save_path="results_plot.png"):
    """
    Строит и сохраняет график метрик для моделей.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame с результатами оценки
    k_values : list
        Список значений K
    save_path : str
        Путь для сохранения графика
    """
    # Получаем список уникальных моделей из DataFrame
    models = results_df.index.get_level_values('Model').unique()
    
    metrics = ['Recall', 'Precision', 'F1-Score', 'NDCG', 'MAP', 'HitRate']
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 18), sharex=True)
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for model in models:
            data = results_df.loc[model, metric].reset_index()
            ax.plot(data['K'], data[metric], marker='o', label=model)
            
        ax.set_title(f'{metric}@K')
        ax.set_xlabel('K')
        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    evaluation_logger.info(f"График результатов сохранен в: {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------------------------------------------------------------
def calculate_ndcg_at_k_per_user(recommended, true_items, k):
    """
    Вычисляет NDCG@K для одного пользователя.
    
    Parameters:
    -----------
    recommended : list
        Список рекомендованных элементов
    true_items : set
        Множество истинных элементов
    k : int
        Значение K для расчета
        
    Returns:
    --------
    float
        Значение NDCG@K
    """
    dcg = 0.0
    for rank, item in enumerate(recommended[:k]):
        if item in true_items:
            dcg += 1.0 / np.log2(rank + 2)
            
    num_true_items = min(len(true_items), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(num_true_items))

    return dcg / idcg if idcg > 0 else 0

# -----------------------------------------------------------------------------------------------------------------------------------
def calculate_ap_at_k_per_user(recommended, true_items, k):
    """
    Вычисляет Average Precision@K (AP@K) для одного пользователя.
    
    Parameters:
    -----------
    recommended : list
        Список рекомендованных элементов
    true_items : set
        Множество истинных элементов
    k : int
        Значение K для расчета
        
    Returns:
    --------
    float
        Значение AP@K
    """
    ap = 0.0
    num_hits = 0
    recommended_at_k = recommended[:k]

    for rank, item in enumerate(recommended_at_k):
        if item in true_items:
            num_hits += 1
            ap += num_hits / (rank + 1.0)
            
    num_true_items = len(true_items)
    return ap / num_true_items if num_true_items > 0 else 0

# -----------------------------------------------------------------------------------------------------------------------------------
def evaluate_and_compare_lightfm(test_data, lightfm_artifacts, k_values=config.K_VALUES_EVALUATION):
    """
    Проводит комплексную оценку модели LightFM и сравнивает с базовыми методами.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Тестовые данные
    lightfm_artifacts : dict
        Артефакты LightFM модели
    k_values : list
        Список значений K для оценки
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame с результатами оценки
    """
    model = lightfm_artifacts['model']
    dataset = lightfm_artifacts['dataset']
    user_features_matrix = lightfm_artifacts['user_features_matrix']
    item_features_matrix = lightfm_artifacts['item_features_matrix']
    item_idx_to_name = lightfm_artifacts['item_idx_to_name']
    
    actual_relevant_items_per_user = test_data.groupby('user_idx')['item_idx'].apply(list).to_dict()
    users_to_evaluate = sorted(list(actual_relevant_items_per_user.keys()))
    
    if not users_to_evaluate:
        evaluation_logger.info("Нет пользователей для оценки. Проверьте тестовые данные.")
        return pd.DataFrame()
    
    num_items = len(item_idx_to_name)
    all_results = []
    
    # Генерация рекомендаций для LightFM
    lightfm_recommendations = recommend_lightfm(
        model, dataset, users_to_evaluate,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        num_items=max(k_values)
    )

    # Оценка базовых моделей
    item_popularity = lightfm_artifacts.get('item_popularity', test_data['item_idx'].value_counts(normalize=True).to_dict())
    top_items = sorted(range(num_items), key=lambda idx: item_popularity.get(idx, 0), reverse=True)
    
    # Оценка для всех моделей и K
    for k in k_values:
        for model_name in ['LightFM', 'Random', 'TopPopular']:
            precisions, recalls, f1_scores, ndcg_scores, ap_scores, hit_rates = [], [], [], [], [], []

            for user_idx in tqdm(users_to_evaluate, desc=f"Evaluating {model_name} @{k}"):
                true_items = set(actual_relevant_items_per_user.get(user_idx, []))
                
                # Получение рекомендаций в зависимости от модели
                if model_name == 'LightFM':
                    recommended = lightfm_recommendations.get(user_idx, [])
                elif model_name == 'Random':
                    recommended = np.random.choice(num_items, size=k, replace=False)
                elif model_name == 'TopPopular':
                    recommended = top_items[:k]
                
                recommended_at_k = recommended[:k]
                hits = true_items.intersection(set(recommended_at_k))
                num_hits = len(hits)
                num_true_items = len(true_items)
                
                if num_true_items > 0:
                    precision = num_hits / k
                    recall = num_hits / num_true_items
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    ndcg = calculate_ndcg_at_k_per_user(recommended_at_k, true_items, k)
                    ap = calculate_ap_at_k_per_user(recommended_at_k, true_items, k)
                    hit_rate = 1 if num_hits > 0 else 0

                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1)
                    ndcg_scores.append(ndcg)
                    ap_scores.append(ap)
                    hit_rates.append(hit_rate)

            # Сохранение результатов для текущей модели и K
            all_results.append({
                'Model': model_name, 'K': k,
                'Recall': np.mean(recalls) if recalls else 0,
                'Precision': np.mean(precisions) if precisions else 0,
                'F1-Score': np.mean(f1_scores) if f1_scores else 0,
                'NDCG': np.mean(ndcg_scores) if ndcg_scores else 0,
                'MAP': np.mean(ap_scores) if ap_scores else 0,
                'HitRate': np.mean(hit_rates) if hit_rates else 0,
            })
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index(['Model', 'K'])
    
    # Сохранение результатов в JSON файл
    os.makedirs(os.path.dirname(config.LIGHTFM_EVALUATION_RESULTS_PATH), exist_ok=True)
    results_df.to_json(config.LIGHTFM_EVALUATION_RESULTS_PATH, indent=4)
    evaluation_logger.info(f"Результаты оценки LightFM сохранены в: {config.LIGHTFM_EVALUATION_RESULTS_PATH}")
    
    return results_df
     
# -----------------------------------------------------------------------------------------------------------------------------------
def evaluate_catboost_ranker(test_data, catboost_recommendations, k_values, save_path=None):
    """
    Оценивает CatBoostRanker по метрикам Recall@K, Precision@K, F1@K, NDCG@K, MAP@K и HitRate@K.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Тестовые данные с истинными взаимодействиями
    catboost_recommendations : dict
        Словарь рекомендаций {user_idx: [item_idx_list]}
    k_values : list
        Список значений K для оценки
    save_path : str, optional
        Путь для сохранения результатов
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame с результатами оценки
    """
    evaluation_logger.info("Оценка CatBoostRanker...")
    
    # Группируем истинные элементы по пользователям из test_data
    actual_relevant_items_per_user = test_data.groupby('user_idx')['item_idx'].apply(list).to_dict()

    all_results = []
    
    # Получаем список уникальных пользователей, для которых есть рекомендации и истинные данные
    users_to_evaluate = sorted(list(set(catboost_recommendations.keys()) & set(actual_relevant_items_per_user.keys())))
    
    if not users_to_evaluate:
        evaluation_logger.error("Нет общих пользователей для оценки. Проверьте данные.")
        return pd.DataFrame()

    for k in k_values:
        evaluation_logger.info(f"--- Вычисление метрик для CatBoostRanker K={k} ---")
        
        precisions_at_k = []
        recalls_at_k = []
        f1_scores_at_k = []
        ndcg_scores_at_k = []
        ap_scores_at_k = [] # Для MAP
        hitrate_at_k = []   # <-- Добавляем новый список для HitRate

        for user_idx in tqdm(users_to_evaluate, desc=f"Evaluating CatBoostRanker @{k}"):
            true_items = set(actual_relevant_items_per_user.get(user_idx, []))
            recommended = catboost_recommendations.get(user_idx, [])
            
            # Обрезаем рекомендации до K
            recommended_at_k = recommended[:k]

            # Находим пересечение истинных и рекомендованных
            hits = true_items.intersection(set(recommended_at_k))
            num_hits = len(hits)
            
            # Precision@K: доля релевантных среди рекомендованных
            precision = num_hits / k
            
            # Recall@K: доля найденных релевантных среди всех релевантных
            recall = num_hits / len(true_items) if true_items else 0
            
            # F1-Score
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            # NDCG@K
            ndcg = 0.0
            if true_items:
                idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(true_items), k)))
                dcg = sum(1.0 / np.log2(rank + 2) for rank, rec_item in enumerate(recommended_at_k) if rec_item in true_items)
                ndcg = dcg / idcg if idcg > 0 else 0
                
            # MAP@K
            ap = 0.0
            if true_items:
                num_hits_cumulative = 0
                for rank, rec_item in enumerate(recommended_at_k):
                    if rec_item in true_items:
                        num_hits_cumulative += 1
                        ap += num_hits_cumulative / (rank + 1.0)
                ap = ap / len(true_items) if true_items else 0
                
            # HitRate@K: попал ли хотя бы один релевантный элемент в топ-K
            hitrate = 1.0 if num_hits > 0 else 0.0
            
            # Добавляем результаты пользователя в списки
            precisions_at_k.append(precision)
            recalls_at_k.append(recall)
            f1_scores_at_k.append(f1)
            ndcg_scores_at_k.append(ndcg)
            ap_scores_at_k.append(ap)
            hitrate_at_k.append(hitrate) # <-- Добавляем HitRate

        # Усреднение (надо?)
        avg_precision = np.mean(precisions_at_k) if precisions_at_k else 0
        avg_recall = np.mean(recalls_at_k) if recalls_at_k else 0
        avg_f1_score = np.mean(f1_scores_at_k) if f1_scores_at_k else 0
        avg_ndcg_score = np.mean(ndcg_scores_at_k) if ndcg_scores_at_k else 0
        avg_map = np.mean(ap_scores_at_k) if ap_scores_at_k else 0
        avg_hitrate = np.mean(hitrate_at_k) if hitrate_at_k else 0 
        
        all_results.append({
            'Model': 'CatBoostRanker', 'K': k,
            'Recall': avg_recall,
            'Precision': avg_precision,
            'F1-Score': avg_f1_score,
            'NDCG': avg_ndcg_score,
            'MAP': avg_map,
            'HitRate': avg_hitrate 
        })
        evaluation_logger.info(f"  CatBoostRanker: Recall@{k}={avg_recall:.4f}, Precision@{k}={avg_precision:.4f}, "
              f"F1@{k}={avg_f1_score:.4f}, NDCG@{k}={avg_ndcg_score:.4f}, MAP@{k}={avg_map:.4f}, "
              f"HitRate@{k}={avg_hitrate:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index(['Model', 'K'])
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        results_df.to_json(os.path.join(save_path, 'catboost_evaluation_results.json'), indent=4)
        evaluation_logger.info(f"Результаты оценки CatBoostRanker сохранены в: {save_path}")

    return results_df

# -----------------------------------------------------------------------------------------------------------------------------------
def plot_catboost_results(results_df, k_values, save_path):
    """
    Визуализирует результаты CatBoostRanker.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame с результатами оценки
    k_values : list
        Список значений K
    save_path : str
        Путь для сохранения графиков
    """
    metrics_to_plot = ['Recall', 'Precision', 'F1-Score', 'NDCG', 'MAP']
    model_name = 'CatBoostRanker'
    color = '#A6CEE3' 
    marker = 'X' 

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        metric_values = [results_df.loc[(model_name, k), metric] for k in k_values]
        
        plt.plot(k_values, metric_values, marker=marker, label=model_name,
                 color=color, linewidth=2, markersize=10)
        
        for i, value in enumerate(metric_values):
            plt.annotate(f'{value:.3f}', 
                         xy=(k_values[i], value),
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=9)

        plt.title(f'Оценка CatBoostRanker: {metric}@K', fontsize=16)
        plt.xlabel('K', fontsize=14)
        plt.ylabel(f'{metric}@K', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xticks(k_values)
        plt.ylim(bottom=0)
        
        plt.annotate(f'Создано: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                     xy=(0.01, 0.01),
                     xycoords='figure fraction',
                     fontsize=8,
                     alpha=0.7)

        plt.tight_layout()
        metric_save_path = save_path / f"catboost_ranker_{metric.lower()}_evaluation.png"
        plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
        evaluation_logger.info(f"График {metric} сохранен в: {metric_save_path}")
        plt.show()