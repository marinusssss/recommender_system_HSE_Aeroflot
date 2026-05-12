
import pandas as pd
import numpy as np
import traceback
from math import log2
from collections import Counter

from src import config
from src.utils import load_pickle
from src.lightfm_model import recommend_lightfm
from src.catboost_ranker import prepare_catboost_data, predict_catboost_ranker
from src.data_processing import load_and_combine_flights
from src.feature_engineering import create_user_and_item_features
from src.miles_logic import load_miles_data, get_flight_miles_cost
from src.logger import setup_logger

eval_logger = setup_logger('miles_eval', config.LOG_FILE_PATH)
K_VALUES = config.K_VALUES_EVALUATION


def compute_metrics(true_items_dict, recommendations_dict, K):
    recalls, precisions, ndcgs, aps, hits = [], [], [], [], []
    for user, true_items in true_items_dict.items():
        rec = recommendations_dict.get(user, [])[:K]
        if not true_items or not rec:
            continue
        true_set = set(true_items)
        hits_at_k = [1 if item in true_set else 0 for item in rec]
        n_hits = sum(hits_at_k)
        n_true = len(true_set)
        recalls.append(n_hits / n_true if n_true > 0 else 0.0)
        precisions.append(n_hits / K)
        hits.append(1.0 if n_hits > 0 else 0.0)
        dcg = sum(hits_at_k[j] / log2(j + 2) for j in range(len(rec)))
        idcg = sum(1.0 / log2(j + 2) for j in range(min(n_true, K)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        ap, cum = 0.0, 0
        for j in range(len(rec)):
            if hits_at_k[j]:
                cum += 1
                ap += cum / (j + 1)
        aps.append(ap / n_true if n_true > 0 else 0.0)
    return {
        'Recall': np.mean(recalls) if recalls else 0.0,
        'Precision': np.mean(precisions) if precisions else 0.0,
        'NDCG': np.mean(ndcgs) if ndcgs else 0.0,
        'MAP': np.mean(aps) if aps else 0.0,
        'HitRate': np.mean(hits) if hits else 0.0,
        'n_users': len(recalls)
    }


def compute_affordability(recommendations_dict, user_balances, cost_lookup, K):
    ratios = []
    for user, rec in recommendations_dict.items():
        top_k = rec[:K]
        if not top_k:
            continue
        balance = user_balances.get(user, 0)
        affordable = sum(
            1 for item in top_k
            if cost_lookup.get((user, item), float('inf')) <= balance
        )
        ratios.append(affordable / len(top_k))
    return np.mean(ratios) if ratios else 0.0


def apply_miles_filter(recommendations_dict, user_balances, cost_lookup, max_K):
    filtered = {}
    for user, rec in recommendations_dict.items():
        balance = user_balances.get(user, 0)
        user_filtered = []
        for item in rec:
            cost = cost_lookup.get((user, item), float('inf'))
            if cost <= balance:
                user_filtered.append(item)
                if len(user_filtered) >= max_K:
                    break
        filtered[user] = user_filtered
    return filtered


def build_cost_lookup(recommendations_dict, user_deptr_zones,
                      item_idx_to_name, city_to_zone, miles_dict_df):
    eval_logger.info("Построение справочника стоимости в милях...")
    cost_lookup = {}
    flight_classes = ['Эконом', 'Комфорт', 'Бизнес']
    n_found = 0

    for i, (user_idx, rec_items) in enumerate(recommendations_dict.items()):
        if i % 50000 == 0 and i > 0:
            eval_logger.info(f"  ...обработано {i}/{len(recommendations_dict)}, найдено: {n_found}")
        deptr_zone = user_deptr_zones.get(user_idx, 'R1')
        for item_idx in rec_items:
            city_name = item_idx_to_name.get(item_idx)
            if not city_name:
                continue
            arvl_zone = city_to_zone.get(city_name)
            if not arvl_zone:
                continue
            arvl_zone = str(arvl_zone).strip()
            for fc in flight_classes:
                result = get_flight_miles_cost(deptr_zone, arvl_zone, miles_dict_df, fc)
                cost = result.get('cost')
                if cost is not None:
                    cost_lookup[(user_idx, item_idx)] = cost
                    n_found += 1
                    break

    eval_logger.info(f"Справочник стоимости: {len(cost_lookup)} пар, найдено={n_found}")
    return cost_lookup



def build_personal_popular(train_data, all_items, max_k=30):
    eval_logger.info("Построение PersonalPopular бейзлайна...")

    # Глобальная популярность
    global_pop = train_data['item_idx'].value_counts().index.tolist()
    user_item_counts = train_data.groupby(['user_idx', 'item_idx']).size().reset_index(name='count')

    recommendations = {}
    for user_idx, grp in user_item_counts.groupby('user_idx'):
        # Сортируем по убыванию частоты
        personal = grp.sort_values('count', ascending=False)['item_idx'].tolist()

        # Добираем глобально популярными
        if len(personal) < max_k:
            seen = set(personal)
            for item in global_pop:
                if item not in seen:
                    personal.append(item)
                    if len(personal) >= max_k:
                        break

        recommendations[user_idx] = personal[:max_k]
    eval_logger.info(f"PersonalPopular: рекомендации для {len(recommendations)} пользователей")
    return recommendations


def main():
    config.setup_directories()
    eval_logger.info("=" * 70)
    eval_logger.info("  ОЦЕНКА: каскад + PersonalPopular + мильный фильтр")
    eval_logger.info("=" * 70)

    try:
        eval_logger.info("1. Загрузка данных...")
        test_data = pd.read_parquet(config.TEST_DATA_PATH)
        train_data = pd.read_parquet(config.TRAIN_DATA_PATH)
        val_data = pd.read_parquet(config.VAL_DATA_PATH)
        full_train = pd.concat([train_data, val_data])

        lightfm_artifacts = load_pickle(config.LIGHTFM_ARTIFACTS_PATH)
        lightfm_model = lightfm_artifacts['model']
        lightfm_dataset = lightfm_artifacts['dataset']
        lightfm_uf = lightfm_artifacts.get('user_features_matrix')
        lightfm_if = lightfm_artifacts.get('item_features_matrix')
        user_id_map, _, item_id_map, _ = lightfm_dataset.mapping()
        inv_item_map = {v: k for k, v in item_id_map.items()}

        catboost_artifacts = load_pickle(config.CATBOOST_RANKER_ARTIFACTS_PATH)
        cb_model = catboost_artifacts['model']
        cb_features = catboost_artifacts['features']
        cb_cat = catboost_artifacts['categorical_features']

        data_artifacts = load_pickle(config.DATA_ARTIFACTS_PATH)
        item_idx_to_name = data_artifacts['item_idx_to_name']
        user_idx_to_id = data_artifacts['user_idx_to_id']
        user_flight_history = data_artifacts.get('user_flight_history', pd.DataFrame())

        combined_flights = load_and_combine_flights(config.FLIGHTS_23_PATH, config.FLIGHTS_24_PATH)
        combined_flights = combined_flights.rename(columns={'AIP_ARVL': 'city_name'})
        item_name_to_idx = {name: idx for idx, name in item_idx_to_name.items()}
        combined_flights['item_idx'] = combined_flights['city_name'].map(item_name_to_idx)
        combined_flights['DEPTR_ZONE'] = combined_flights['DEPTR_ZONE'].str.strip()
        combined_flights['ARVL_ZONE'] = combined_flights['ARVL_ZONE'].str.strip()

        city_to_zone = {}
        for _, row in combined_flights[['CITY_ARVL', 'ARVL_ZONE']].dropna().drop_duplicates().iterrows():
            city_to_zone[row['CITY_ARVL']] = row['ARVL_ZONE']

        miles_id_df, miles_dict_df = load_miles_data()
        miles_balance_by_id = miles_id_df.set_index('FRQTFLR_CARD_ID')['END_BALANCE'].to_dict()

        user_balances = {}
        for uidx, uid in user_idx_to_id.items():
            user_balances[uidx] = miles_balance_by_id.get(uid, 0)

        fallback_zone = str(combined_flights['DEPTR_ZONE'].mode()[0]).strip()
        user_deptr_zones = {}
        for uidx, uid in user_idx_to_id.items():
            uf = combined_flights[combined_flights['FRQTFLR_CARD_ID'] == uid]
            if not uf.empty:
                user_deptr_zones[uidx] = str(uf['DEPTR_ZONE'].mode()[0]).strip()
            else:
                user_deptr_zones[uidx] = fallback_zone

        true_items_dict = test_data.groupby('user_idx')['item_idx'].apply(set).to_dict()
        test_users = test_data['user_idx'].unique()

        user_activity = full_train.groupby('user_idx').size()
        terciles = user_activity.quantile([1/3, 2/3]).values
        t1, t2 = terciles[0], terciles[1]
        eval_logger.info(f"Терцили активности: T1={t1:.0f}, T2={t2:.0f}")

        cohort_map = {}
        for uidx, cnt in user_activity.items():
            if cnt <= t1:
                cohort_map[uidx] = 'low'
            elif cnt <= t2:
                cohort_map[uidx] = 'medium'
            else:
                cohort_map[uidx] = 'high'

        # Размеры когорт
        for c in ['low', 'medium', 'high']:
            n = sum(1 for u in true_items_dict if cohort_map.get(u) == c)
            eval_logger.info(f"  Когорта {c}: {n} пользователей")

        all_items = list(item_idx_to_name.keys())
        rec_personal_pop = build_personal_popular(full_train, all_items, max_k=30)

        # Каскад LightFM - CatBoostRanker
        eval_logger.info("2. Генерация рекомендаций каскада")
        user_features_dict, item_features_dict = create_user_and_item_features(
            interactions_df=full_train,
            user_flight_history=user_flight_history,
            city_to_zone=city_to_zone,
            full_flights_df=combined_flights
        )

        lightfm_candidates = recommend_lightfm(
            lightfm_model, lightfm_dataset, test_users,
            user_features_matrix=lightfm_uf,
            item_features_matrix=lightfm_if,
            num_items=config.CATBOOST_CANDIDATE_COUNT
        )

        test_df_cb, _ = prepare_catboost_data(
            lightfm_candidates, test_data,
            user_features_dict, item_features_dict,
            lightfm_model, user_id_map, item_id_map, inv_item_map,
            lightfm_uf, lightfm_if,
            exclude_popular_cities=None
        )

        rec_cascade = predict_catboost_ranker(
            cb_model, test_df_cb, cb_features, cb_cat,
            num_recommendations=30
        )
        eval_logger.info(f"Каскад: рекомендации для {len(rec_cascade)} пользователей.")

        all_recs = {}
        for u in test_users:
            items = set(rec_cascade.get(u, []))
            items.update(rec_personal_pop.get(u, []))
            all_recs[u] = list(items)

        cost_lookup = build_cost_lookup(
            all_recs, user_deptr_zones,
            item_idx_to_name, city_to_zone, miles_dict_df
        )

        print(f"cost_lookup: {len(cost_lookup)} пар")
        max_k = max(K_VALUES)
        rec_cascade_filt = apply_miles_filter(rec_cascade, user_balances, cost_lookup, max_k)
        rec_pp_filt = apply_miles_filter(rec_personal_pop, user_balances, cost_lookup, max_k)


        header = f"{'K':>3} | {'Модель':<25} | {'NDCG':>8} | {'Recall':>8} | {'MAP':>8} | {'HR':>8} | {'Afford':>8} | {'N':>8}"
        print(header)

        all_results = []
        models_recs = [
            ('Cascade R^b', rec_cascade, False),
            ('Cascade R^w (miles)', rec_cascade_filt, True),
            ('PersonalPop R^b', rec_personal_pop, False),
            ('PersonalPop R^w (miles)', rec_pp_filt, True),
        ]

        for K in K_VALUES:
            for name, recs, is_filtered in models_recs:
                recs_k = {u: items[:K] for u, items in recs.items()}
                m = compute_metrics(true_items_dict, recs_k, K)
                aff = compute_affordability(recs_k, user_balances, cost_lookup, K)
                print(f"{K:3d} | {name:<25} | {m['NDCG']:8.4f} | {m['Recall']:8.4f} | "
                      f"{m['MAP']:8.4f} | {m['HitRate']:8.4f} | {aff:8.4f} | {m['n_users']:8d}")
                all_results.append({
                    'K': K, 'model': name,
                    **{k: v for k, v in m.items() if k != 'n_users'},
                    'Affordability': aff, 'n_users': m['n_users']
                })
            print() 

        K_main = 10

        header2 = f"{'Когорта':<10} | {'Модель':<25} | {'NDCG':>8} | {'Recall':>8} | {'HR':>8} | {'Afford':>8} | {'N':>8}"
        print(header2)
        print("-" * len(header2))

        for cohort in ['low', 'medium', 'high']:
            cohort_users = {u: items for u, items in true_items_dict.items()
                           if cohort_map.get(u) == cohort}
            if not cohort_users:
                continue

            for name, recs, is_filt in [
                ('Cascade R^b', rec_cascade, False),
                ('Cascade R^w', rec_cascade_filt, True),
                ('PersonalPop R^b', rec_personal_pop, False),
                ('PersonalPop R^w', rec_pp_filt, True),
            ]:
                recs_k = {u: recs.get(u, [])[:K_main] for u in cohort_users}
                m = compute_metrics(cohort_users, recs_k, K_main)
                aff = compute_affordability(recs_k, user_balances, cost_lookup, K_main)
                print(f"{cohort:<10} | {name:<25} | {m['NDCG']:8.4f} | {m['Recall']:8.4f} | "
                      f"{m['HitRate']:8.4f} | {aff:8.4f} | {m['n_users']:8d}")
            print()

        test_balances = [user_balances.get(u, 0) for u in test_users]
        print(f"Мильные балансы — Медиана: {np.median(test_balances):,.0f}, "
              f"Среднее: {np.mean(test_balances):,.0f}, "
              f"25p: {np.percentile(test_balances, 25):,.0f}, "
              f"75p: {np.percentile(test_balances, 75):,.0f}")

        n_empty_cascade = sum(1 for v in rec_cascade_filt.values() if len(v) == 0)
        n_empty_pp = sum(1 for v in rec_pp_filt.values() if len(v) == 0)
        print(f"Пустых списков после фильтра: каскад={n_empty_cascade}, PersonalPop={n_empty_pp}")

        pd.DataFrame(all_results).to_csv(
            config.EXPERIMENTS_DIR / 'miles_filter_evaluation_v3.csv', index=False
        )

    except Exception as e:
        eval_logger.critical(f"ОШИБКА: {e}")
        eval_logger.critical(traceback.format_exc())

if __name__ == '__main__':
    main()
