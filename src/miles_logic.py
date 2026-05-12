import pandas as pd
import numpy as np
from datetime import datetime


def load_miles_data():
    """
    Загружает данные о милях пользователей и стоимости перелетов.
    """
    miles_id_df = pd.read_parquet('data/raw/id_miles.parquet')
    miles_dict_df = pd.read_parquet('data/raw/miles_dict.parquet')

    miles_dict_df['START_DATE'] = pd.to_datetime(miles_dict_df['START_DATE'])
    miles_dict_df['END_DATE'] = miles_dict_df['END_DATE'].astype(str)
    miles_dict_df['END_DATE'] = miles_dict_df['END_DATE'].str.replace('9999-12-31', '2262-04-11')
    miles_dict_df['END_DATE'] = pd.to_datetime(miles_dict_df['END_DATE'], errors='coerce')
    miles_dict_df['MILES_COST'] = pd.to_numeric(miles_dict_df['MILES_COST'], errors='coerce')
    miles_dict_df.dropna(subset=['MILES_COST'], inplace=True)

    return miles_id_df, miles_dict_df


def create_zone_mapping(flights_data):
    """
    Создает маппинг между городами и их зонами на основе истории полетов.
    """
    city_to_zone = {}
    deptr_mapping = flights_data[['CITY_DEPTR', 'DEPTR_ZONE']].dropna().drop_duplicates()
    for _, row in deptr_mapping.iterrows():
        city_to_zone[row['CITY_DEPTR']] = row['DEPTR_ZONE']

    arvl_mapping = flights_data[['CITY_ARVL', 'ARVL_ZONE']].dropna().drop_duplicates()
    for _, row in arvl_mapping.iterrows():
        city_to_zone[row['CITY_ARVL']] = row['ARVL_ZONE']

    return city_to_zone


def get_flight_miles_cost(deptr_zone, arvl_zone, miles_dict_df, flight_class='Эконом'):
    """
    Определяет стоимость перелета в милях между указанными зонами.
    Обрабатывает три варианта FLAG_SEASON: 'Высокий', 'Низкий', 'n/a'.
    Тарифы с 'n/a' считаются единым тарифом без сезонности.
    """
    route_costs = miles_dict_df[
        (miles_dict_df['DEPTR_ZONE'].str.strip() == deptr_zone.strip()) &
        (miles_dict_df['ARVL_ZONE'].str.strip() == arvl_zone.strip()) &
        (miles_dict_df['CLASS'] == flight_class) &
        (miles_dict_df['FLAG_RT/OW'] == 'OW')
    ]

    if route_costs.empty:
        return {'cost': None}

    # Берём самые свежие тарифы
    route_costs = route_costs.sort_values('START_DATE', ascending=False)

    # Проверяем n/a (единый тариф без сезонности)
    na_costs = route_costs[route_costs['FLAG_SEASON'] == 'n/a']
    if not na_costs.empty:
        return {'cost': na_costs['MILES_COST'].iloc[0]}

    # Если есть сезонные тарифы — берём минимальный
    costs = []
    for season in ['Низкий', 'Высокий']:
        s = route_costs[route_costs['FLAG_SEASON'] == season]
        if not s.empty:
            costs.append(s['MILES_COST'].iloc[0])

    if costs:
        return {'cost': min(costs)}

    return {'cost': None}


def filter_recommendations_by_miles(user_id, recommendations, miles_id_df, miles_dict_df,
                                     city_to_zone, combined_flights, limit=12):
    """
    Фильтрует список рекомендаций по балансу миль пользователя.

    Для каждого кандидата перебирает классы обслуживания (Бизнес → Комфорт → Эконом)
    и оставляет город, если пользователь может оплатить перелёт хотя бы в одном классе
    по минимальной стоимости (из высокого и низкого сезонов спроса).

    Parameters
    ----------
    user_id : any
        ID пользователя (FRQTFLR_CARD_ID).
    recommendations : list of (item_idx, city_name)
        Отсортированный список кандидатов от ранжировщика.
    miles_id_df : pd.DataFrame
        Баланс миль пользователей: колонки FRQTFLR_CARD_ID, END_BALANCE.
    miles_dict_df : pd.DataFrame
        Справочник стоимости перелётов в милях.
    city_to_zone : dict
        Маппинг {city_name: zone}.
    combined_flights : pd.DataFrame
        Полные данные о полётах (нужны для определения зоны вылета пользователя).
    limit : int
        Максимальное число рекомендаций в результате.

    Returns
    -------
    list of (item_idx, city_name, min_cost)
        Доступные рекомендации с минимальной стоимостью в милях.
    """
    # Баланс миль пользователя
    user_row = miles_id_df[miles_id_df['FRQTFLR_CARD_ID'] == user_id]
    if user_row.empty:
        # Пользователь не найден — возвращаем кандидатов без фильтрации
        return [(idx, name, None) for idx, name in recommendations[:limit]]

    user_balance = float(user_row['END_BALANCE'].iloc[0])

    # Зона вылета пользователя (берём наиболее частую)
    fallback_zone = str(combined_flights['DEPTR_ZONE'].mode()[0]).strip()
    user_flights = combined_flights[combined_flights['FRQTFLR_CARD_ID'] == user_id]
    deptr_zone = str(user_flights['DEPTR_ZONE'].mode()[0]).strip() if not user_flights.empty else fallback_zone

    flight_classes = ['Бизнес', 'Комфорт', 'Эконом']
    filtered = []

    for item_idx, city_name in recommendations:
        if len(filtered) >= limit:
            break

        arvl_zone = city_to_zone.get(city_name)
        if not arvl_zone:
            continue
        arvl_zone = str(arvl_zone).strip()

        for flight_class in flight_classes:
            demand_costs = get_flight_miles_cost(deptr_zone, arvl_zone, miles_dict_df, flight_class)
            # Берём минимальную стоимость из доступных сезонов
            available_costs = [v for v in demand_costs.values() if v is not None]
            if not available_costs:
                continue
            min_cost = min(available_costs)
            if user_balance >= min_cost:
                filtered.append((item_idx, city_name, min_cost))
                break  # Нашли подходящий класс — переходим к следующему городу

    return filtered
