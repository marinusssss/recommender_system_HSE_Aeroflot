# Рекомендательная система авиаперелетов

## Описание проекта

Данный проект посвящен разработке и сравнению моделей машинного обучения для построения рекомендательной системы авиаперелетов. Система предназначена для персоназированного предложения направлений полетов клиентам авиакомпании на основе истории их перелетов.

## Структура проекта

```bash
RECOMMEND_MOD_REP/

├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── flights_with_zones_23.parquet # Данные о полетах за 2023 год с информацией о зонах перелета
│   │   └── flights_with_zones_24.parquet # Данные о полетах за 2023 год с информацией о зонах перелета
│   │   ├── id_miles.parquet              # Данные о балансе миль пользователей
│   │   └── miles_dict.parquet            # Данные о стоимости перелетов (в милях) в разные зоны
│   └── processed/
│       ├── data_artifacts.pickle   
│       ├── final_seasonal_matrices.pickle     
│       ├── final_seasonal_visited_matrices.pickle     
│       └── test_data.parquet
│       └── train_data.parquet
│       └── val_data.parquet
│       └── lightfm_catboost_recommendations.csv
│       └── lightfm_catboost_miles_recommendations.csv  
├── experiments/
│   └── ...                                 # Метрики качества моделей
├── logs/
│   └── training.log                  
├── notebooks/
│   └── data_eda.ipynb
├── outputs/
│   ├── models/
│   │   └── als_model.pkl                   # Сохраненная ALS модель
│   │   └── lightfm_model.pkl               # Сохраненная LightFM модель
│   │   └── catboost_ranker.pkl             # Сохраненная (LightFM +) CatBoost Ranker модель
│   ├── visualizations/
│   │   └── ...                             # Сохраненные графики
├── scripts/
│   └── run_als_training.py
│   └── run_lightfm_training.py
│   └── run_reranking_training.py
│   └── inference_catboost_ranker_no_season.py
│   └── inference_catboost_ranker_seasons.py
│   └── inference_catboost_ranker_miles.py
├── src/
│   ├── __init__.py
│   ├── config.py                           # Конфигурации: пути к данным, гиперпараметры по умолчанию
│   ├── data_processing.py                  # файл с кодом для первичной обработки входных данных, разделения и преобразования для работы моделей
│   ├── als_model.py                        # функции для обучения ALS модели
│   ├── catboost_ranker.py                  # функции для обучения CatBoost Ranker
│   ├── lightfm_model.py                    # функции для обучения LightFM модели
│   ├── evaluation.py                       # Функции оценки моделей
│   ├── feature_engineering.py              # Функции фичей
│   ├── logger.py                           # Функции настройки логгирования
│   ├── miles_logic.py                      # Функции с логикой фильтрации рекомнедаций с учетом миль
│   └── recommenders.py             
│   └── utils.py      
```



## Данные

Проект использует три основных источника данных:

1.  **`flights_with_zones_*.parquet`**: Исторические данные о перелетах пользователей за 2023 и 2024 годы.
    *   **Содержит:** ID пользователя (`FRQTFLR_CARD_ID`), аэропорты вылета/прилета (`AIP_DEPTR`, `AIP_ARVL`), города, регионы, зоны (`DEPTR_ZONE`, `ARVL_ZONE`), дату, класс обслуживания, стоимость и др.
    *   **Пример:** `786547840, SVO, ALA, Moscow, Almaty, R1, MA, ...`

2.  **`id_miles.parquet`**: Данные о балансе бонусных миль на картах лояльности пользователей.
    *   **Содержит:** `FRQTFLR_CARD_ID`, `END_BALANCE`.

3.  **`miles_dict.parquet`**: Справочник стоимости перелетов между зонами в милях.
    *   **Содержит:** Зоны вылета и прилета, класс обслуживания, сезонность, тип перелета (туда-обратно/в одну сторону), стоимость в милях.

## Инструкция по запуску

1.  **Обучение моделей:**
    Модели можно обучить последовательно с помощью скриптов:
    ```bash
    python -m scripts.run_als_training
    python -m scripts.run_lightfm_training
    python -m scripts.run_reranking_training
    ```
    Логи обучения будут сохранены в `logs/training.log`.

2.  **Генерация рекомендаций (Инференс):**
    После обучения моделей используйте скрипты инференса для генерации предсказаний.
    ```bash
    # Без учета сезонности
    python -m scripts.inference_catboost_ranker_no_season
    # С учетом баланса миль
    python -m scripts.inference_catboost_ranker_miles
    ```
    Результаты (списки рекомендаций) будут сохранены в `data/processed/`.






