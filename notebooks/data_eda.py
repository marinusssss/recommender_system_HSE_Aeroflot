
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FLIGHTS_23 = Path('data/raw/flights_with_zones_23.parquet')
FLIGHTS_24 = Path('data/raw/flights_with_zones_24.parquet')
MILES_ID   = Path('data/raw/id_miles.parquet')
MILES_DICT = Path('data/raw/miles_dict.parquet')
OUT_DIR    = Path('diploma_eda_output')
OUT_DIR.mkdir(exist_ok=True)

STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.family':      'DejaVu Sans',
}
plt.rcParams.update(STYLE)
sns.set_palette('muted')

log_lines = []

def log(text=''):
    print(text)
    log_lines.append(text)

def save_log():
    with open(OUT_DIR / 'diploma_eda_results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

# ----------------------------------------------------------------------------
# Загрузка


df23 = pd.read_parquet(FLIGHTS_23)
df24 = pd.read_parquet(FLIGHTS_24)
raw = pd.concat([df23, df24], ignore_index=True)

log(f'Записей 2023:          {len(df23):>12,}')
log(f'Записей 2024:          {len(df24):>12,}')
log(f'Итого (сырые):         {len(raw):>12,}')
log(f'Колонки: {list(raw.columns)}')

# Очистка данных (базовая)
log('Предобработка данных:')

raw['SCHD_DEPTR_DT'] = pd.to_datetime(raw.get('SCHD_DEPTR_DT', raw.index), errors='coerce')
n_before = len(raw)

df = raw[raw['AIP_ARVL'] != raw['MAIN_AIRPORT']].copy() # Удаляем self-loops
log(f'Удалено self-loops: {n_before - len(df):>10,}')

df = df.dropna(subset=['FRQTFLR_CARD_ID', 'CITY_ARVL']) # Удаляем NA
log(f'После dropna: {len(df):>10,}')

activity = df['FRQTFLR_CARD_ID'].value_counts() # Фильтрация пользователей
valid_users = activity[(activity >= 2) & (activity <= 200)].index
n_users_before = df['FRQTFLR_CARD_ID'].nunique()
df = df[df['FRQTFLR_CARD_ID'].isin(valid_users)]
log(f'Пользователей до фильтрации:{n_users_before:>10,}')
log(f'Пользователей после:{df["FRQTFLR_CARD_ID"].nunique():>10,}')
log(f'Записей после фильтрации:{len(df):>10,}')
log(f'Уникальных направлений:{df["CITY_ARVL"].nunique():>10,}')

# Матрица взаимодействий и разреженность
log('Матрица взаимодействий, разреженность:')

n_users = df['FRQTFLR_CARD_ID'].nunique()
n_items = df['CITY_ARVL'].nunique()
n_possible = n_users * n_items

# Уникальные пары пользователь-направление
unique_pairs = df.groupby(['FRQTFLR_CARD_ID', 'CITY_ARVL']).size().reset_index(name='count')
n_interactions = len(unique_pairs)
sparsity = 1 - n_interactions / n_possible

log(f'Пользователей (|U|):{n_users:>10,}')
log(f'Направлений (|I|):{n_items:>10,}')
log(f'Возможных пар |U|×|I|:{n_possible:>10,}')
log(f'Наблюдаемых пар:{n_interactions:>10,}')
log(f'Разреженность матрицы:{sparsity:>10.4%}')

# Активность
log('Активность УПЛ:')

user_unique_dest = unique_pairs.groupby('FRQTFLR_CARD_ID')['CITY_ARVL'].count() # Число уникальных направлений на пользователя
log('Уникальных направлений на пользователя:')
log(f'  Минимум:{user_unique_dest.min()}')
log(f'  Медиана:{user_unique_dest.median():.1f}')
log(f'  Среднее:{user_unique_dest.mean():.2f}')
log(f'  75-й перц.:{user_unique_dest.quantile(0.75):.1f}')
log(f'  90-й перц.:{user_unique_dest.quantile(0.90):.1f}')
log(f'  95-й перц.:{user_unique_dest.quantile(0.95):.1f}')
log(f'  Максимум:{user_unique_dest.max()}')

user_flights = df.groupby('FRQTFLR_CARD_ID').size() # Число полётов на пользователя
log('\nПолётов на пользователя (после фильтрации):')
log(f'  Минимум:{user_flights.min()}')
log(f'  Медиана:{user_flights.median():.1f}')
log(f'  Среднее:{user_flights.mean():.2f}')
log(f'  75-й перц.:{user_flights.quantile(0.75):.1f}')
log(f'  95-й перц.:{user_flights.quantile(0.95):.1f}')
log(f'  Максимум: {user_flights.max()}')


n_cold = (user_unique_dest == 1).sum() # Cold-start: пользователи с 1 уникальным направлением
log(f'\nПользователей с 1 уникальным направлением (cold-start): {n_cold:,} ({n_cold/n_users:.1%})')
log(f'Пользователей с 2–3 уникальными направлениями:          '
    f'{((user_unique_dest >= 2) & (user_unique_dest <= 3)).sum():,}')
log(f'Пользователей с 4+ уникальными направлениями:           '
    f'{(user_unique_dest >= 4).sum():,}')

# Квантили активности
q30 = user_flights.quantile(0.30)
q70 = user_flights.quantile(0.70)
low  = (user_flights <= q30).sum()
mid  = ((user_flights > q30) & (user_flights <= q70)).sum()
high = (user_flights > q70).sum()
log(f'\nСегментация по активности (квантили 30/70):')
log(f'  Низкая (≤{q30:.0f} полётов):{low:,} пользователей ({low/n_users:.1%})')
log(f'  Средняя ({q30:.0f}–{q70:.0f} полётов):{mid:,} пользователей ({mid/n_users:.1%})')
log(f'  Высокая (>{q70:.0f} полётов):{high:,} пользователей ({high/n_users:.1%})')

# Анализ направлений
log('Анализ популярности направлений')

item_popularity = unique_pairs.groupby('CITY_ARVL')['count'].sum().sort_values(ascending=False)
total_inter = item_popularity.sum()

top10_share  = item_popularity.head(10).sum() / total_inter
top20_share  = item_popularity.head(20).sum() / total_inter
top50_share  = item_popularity.head(50).sum() / total_inter

log(f'Топ-10 направлений: {top10_share:.1%} всех взаимодействий')
log(f'Топ-20 направлений: {top20_share:.1%} всех взаимодействий')
log(f'Топ-50 направлений: {top50_share:.1%} всех взаимодействий')
log(f'Остальные {n_items - 50} направлений: {1 - top50_share:.1%} взаимодействий')

log('\nТоп-15 направлений по числу уникальных пользователей:')
top_items = unique_pairs.groupby('CITY_ARVL')['FRQTFLR_CARD_ID'].count().sort_values(ascending=False).head(15)
for city, cnt in top_items.items():
    log(f'  {city:<30} {cnt:>8,} пользователей  ({cnt/n_users:.1%})')

# Повторные визиты
log('Анализ повторных визитов:')

repeat_pairs = unique_pairs[unique_pairs['count'] > 1]
repeat_users = repeat_pairs['FRQTFLR_CARD_ID'].nunique()

log(f'Уникальных пар с повторными визитами: {len(repeat_pairs):,} ({len(repeat_pairs)/n_interactions:.1%})')
log(f'Пользователей с повторными визитами:  {repeat_users:,} ({repeat_users/n_users:.1%})')

avg_repeats = unique_pairs[unique_pairs['count'] > 1]['count'].mean()
log(f'Среднее число полётов в повторных парах: {avg_repeats:.1f}')

log('Сезоны:')

season_counts = df['SEASON'].value_counts()
for s, cnt in season_counts.items():
    log(f'  {s}: {cnt:,} записей ({cnt/len(df):.1%})')

seasonal_dest = df.groupby('SEASON')['CITY_ARVL'].nunique()
log(f'\nУникальных направлений по сезонам:')
for s, cnt in seasonal_dest.items():
    log(f'  {s}: {cnt} направлений')

dest_season = df.groupby(['CITY_ARVL', 'SEASON']).size().unstack(fill_value=0) # Направления с выраженной сезонностью
if 'summer' in dest_season.columns and 'winter' in dest_season.columns:
    dest_season['ratio'] = dest_season['summer'] / (dest_season['winter'] + 1)
    top_summer = dest_season.nlargest(5, 'ratio').index.tolist()
    top_winter = dest_season.nsmallest(5, 'ratio').index.tolist()
    log(f'\nНаправления с максимальным летним спросом: {top_summer}')
    log(f'Направления с максимальным зимним спросом: {top_winter}')

log('Анализ мильных балансов:')

if MILES_ID.exists():
    miles_df = pd.read_parquet(MILES_ID)
    miles_df['END_BALANCE'] = pd.to_numeric(miles_df['END_BALANCE'], errors='coerce')
    miles_df = miles_df.dropna(subset=['END_BALANCE'])

    log(f'Участников в базе миль: {len(miles_df):,}')
    log(f'Участников с ненулевым балансом: {(miles_df["END_BALANCE"] > 0).sum():,}')
    log(f'\nРаспределение баланса (END_BALANCE):')
    log(f'  Минимум:{miles_df["END_BALANCE"].min():>15,.0f}')
    log(f'  25-й перцентиль:{miles_df["END_BALANCE"].quantile(0.25):>15,.0f}')
    log(f'  Медиана:{miles_df["END_BALANCE"].median():>15,.0f}')
    log(f'  Среднее:{miles_df["END_BALANCE"].mean():>15,.0f}')
    log(f'  75-й перцентиль:{miles_df["END_BALANCE"].quantile(0.75):>15,.0f}')
    log(f'  90-й перцентиль:{miles_df["END_BALANCE"].quantile(0.90):>15,.0f}')
    log(f'  Максимум:{miles_df["END_BALANCE"].max():>15,.0f}')

    # Доля, у которых хватит миль хотя бы на один эконом-перелёт
    pct_5k  = (miles_df['END_BALANCE'] >= 5000).mean()
    pct_15k = (miles_df['END_BALANCE'] >= 15000).mean()
    pct_30k = (miles_df['END_BALANCE'] >= 30000).mean()
    log(f'\nДоля участников с балансом ≥5 000 миль:  {pct_5k:.1%}')
    log(f'Доля участников с балансом ≥15 000 миль: {pct_15k:.1%}')
    log(f'Доля участников с балансом ≥30 000 миль: {pct_30k:.1%}')
else:
    log('Файл id_miles.parquet не найден — пропуск.')

log('Разделение данных:')

if 'SCHD_DEPTR_DT' in df.columns:
    df_sorted = df.sort_values(['FRQTFLR_CARD_ID', 'SCHD_DEPTR_DT'])
else:
    df_sorted = df.copy()

MIN_INT = 4
eligible = df_sorted.groupby('FRQTFLR_CARD_ID').filter(
    lambda x: len(x) >= MIN_INT
)['FRQTFLR_CARD_ID'].nunique()

log(f'Пользователей с ≥{MIN_INT} записями (пригодных для split): {eligible:,}')
log(f'Пользователей исключено из split: {n_users - eligible:,}')

# Оценка размеров выборок
user_groups = df_sorted[df_sorted['FRQTFLR_CARD_ID'].isin(
    df_sorted.groupby('FRQTFLR_CARD_ID').filter(lambda x: len(x) >= MIN_INT)['FRQTFLR_CARD_ID']
)].groupby('FRQTFLR_CARD_ID')

train_counts, val_counts, test_counts = [], [], []
for uid, grp in user_groups:
    n = len(grp)
    if n >= MIN_INT:
        test_counts.append(min(2, n))
        val_counts.append(min(2, n - 2))
        train_counts.append(max(0, n - 4))

log(f'\nОценочные размеры выборок:')
log(f'Train:{sum(train_counts):,} записей')
log(f'Val:{sum(val_counts):,} записей')
log(f'Test:{sum(test_counts):,} записей')

train_frac = sum(train_counts) / (sum(train_counts) + sum(val_counts) + sum(test_counts))
log(f'  Доля train: {train_frac:.1%}')

log('Визуализация:')

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Разведочный анализ данных — Аэрофлот Бонус 2023–2024',
             fontsize=15, fontweight='bold', y=1.01)

# 1. Распределение уникальных направлений
ax = axes[0, 0]
bins = range(1, min(user_unique_dest.max() + 2, 30))
ax.hist(user_unique_dest.clip(upper=30), bins=bins, color='steelblue',
        edgecolor='white', linewidth=0.5)
ax.axvline(user_unique_dest.median(), color='crimson', ls='--',
           label=f'Медиана = {user_unique_dest.median():.0f}')
ax.set_xlabel('Число уникальных направлений')
ax.set_ylabel('Число пользователей')
ax.set_title('Уникальных направлений на пользователя')
ax.legend()

# 2. Распределение активности (полёты)
ax = axes[0, 1]
ax.hist(user_flights.clip(upper=100), bins=50,
        color='seagreen', edgecolor='white', linewidth=0.5)
ax.axvline(user_flights.median(), color='crimson', ls='--',
           label=f'Медиана = {user_flights.median():.0f}')
ax.set_xlabel('Число полётов (ограничено до 100)')
ax.set_ylabel('Число пользователей')
ax.set_title('Распределение активности пользователей')
ax.legend()

# 3. Long tail направлений
ax = axes[0, 2]
sorted_pop = item_popularity.values / item_popularity.sum() * 100
ax.plot(range(1, len(sorted_pop) + 1), sorted_pop.cumsum(),
        color='darkorange', linewidth=2)
ax.axhline(80, color='crimson', ls='--', alpha=0.7, label='80%')
ax.set_xlabel('Топ-N направлений (отсортировано по популярности)')
ax.set_ylabel('Кумулятивная доля взаимодействий, %')
ax.set_title('Кривая Лоренца (long tail каталога)')
ax.legend()

# 4. Сезонность
ax = axes[1, 0]
season_counts_plot = df.groupby('SEASON').size()
ax.bar(season_counts_plot.index, season_counts_plot.values,
       color=['#4C72B0', '#DD8452'], edgecolor='white')
for i, (s, v) in enumerate(season_counts_plot.items()):
    ax.text(i, v + 50000, f'{v/len(df):.1%}', ha='center', fontsize=11)
ax.set_ylabel('Число полётов')
ax.set_title('Сезонное распределение перелётов')

# 5. Топ-15 направлений
ax = axes[1, 1]
top15 = top_items.head(15)[::-1]
bars = ax.barh(range(len(top15)), top15.values, color='mediumpurple', edgecolor='white')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=9)
ax.set_xlabel('Число уникальных пользователей')
ax.set_title('Топ-15 направлений по охвату аудитории')

# 6. Тип рейса
ax = axes[1, 2]
flt_counts = df['FLT_TYPE'].value_counts()
ax.pie(flt_counts.values, labels=flt_counts.index,
       autopct='%1.1f%%', colors=['#4C72B0', '#DD8452'],
       startangle=90, wedgeprops={'edgecolor': 'white'})
ax.set_title('Тип перелёта (ВВЛ / МВЛ)')

plt.tight_layout()
fig.savefig(OUT_DIR / 'eda_summary.png', dpi=180, bbox_inches='tight')
plt.close()
log(f'График сохранён: {OUT_DIR}/eda_summary.png')


log(f'Общее число записей (сырые):{len(raw):>12,}')
log(f'После фильтрации:{len(df):>12,}')
log(f'Уникальных пользователей:{n_users:>12,}')
log(f'Уникальных направлений:{n_items:>12,}')
log(f'Уникальных пар (пользователь-город):{n_interactions:>12,}')
log(f'Разреженность матрицы R:{sparsity:>11.2%}')
log(f'Медиана уникальных направлений/user:{user_unique_dest.median():>12.1f}')
log(f'Медиана полётов/user:{user_flights.median():>12.1f}')
log(f'Доля летних перелётов:{season_counts.get("summer",0)/len(df):>11.1%}')
log(f'Доля внутренних (ВВЛ):{df["FLT_TYPE"].value_counts().get("ВВЛ",0)/len(df):>11.1%}')
log(f'Топ-10 направлений / все взаим.:{top10_share:>11.1%}')

save_log()
log(f'\Результаты сохранены в {OUT_DIR}/diploma_eda_results.txt')
