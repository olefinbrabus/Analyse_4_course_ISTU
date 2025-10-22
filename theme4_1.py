import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from IPython.display import display

np.random.seed(42)

# 1) генерація якісного датасету на 300 об'єктів про послуги з аналізу даних для ІТ-стартапів
n = 300
startup_ids = [f"S{str(i).zfill(3)}" for i in range(1, 51)]
domains = [
    "fintech",
    "healthtech",
    "edtech",
    "ecommerce",
    "marketplace",
    "mobility",
    "gaming",
    "media",
    "saas",
    "iot",
]
services = [
    "etl_pipeline",
    "eda_audit",
    "ab_test_design",
    "churn_prediction",
    "recommender_mvp",
    "price_optimization",
    "time_series_forecast",
    "anomaly_detection",
    "nlp_sentiment",
    "cv_classification",
]
pricing = ["fixed_price", "time_and_materials", "subscription"]
regions = ["UA", "EU", "US", "APAC"]
toolstacks = [
    "python+pandas+sklearn",
    "python+spark+mlflow",
    "python+dbt+bigquery",
    "python+airflow+postgres",
    "python+docker+ray",
]


def random_date(start=datetime(2023, 1, 1), end=datetime(2025, 10, 1)):
    delta = end - start
    return start + timedelta(days=int(np.random.randint(0, delta.days + 1)))


rows = []
for i in range(n):
    sid = np.random.choice(startup_ids)
    domain = np.random.choice(domains)
    service = np.random.choice(services)
    price_model = np.random.choice(pricing, p=[0.35, 0.5, 0.15])
    complexity = np.random.randint(1, 6)  # 1−5
    team_size = np.random.randint(1, 7)  # 1−6
    duration_w = np.random.randint(2, 17)  # 2−16 тижнів
    region = np.random.choice(regions, p=[0.35, 0.35, 0.2, 0.1])
    start_dt = random_date()
    data_gb = np.round(
        np.random.lognormal(mean=2.2, sigma=0.6), 2
    )  # позитивно-скошені обсяги
    kpi_baseline = np.round(
        np.random.uniform(0.2, 0.8), 3
    )  # наприклад, базова retention чи ctr
    expected_gain = np.round(
        np.random.normal(loc=0.12, scale=0.06), 3
    )  # очікуваний приріст
    expected_gain = float(np.clip(expected_gain, -0.05, 0.35))
    tools = np.random.choice(toolstacks)

    # бюджет як функція складності, розміру команди і тривалості
    rate = {"UA": 35, "EU": 65, "US": 85, "APAC": 45}[region]
    budget = duration_w * team_size * 40 * rate * (0.8 + 0.1 * complexity)
    budget *= np.random.uniform(0.9, 1.15)
    budget = int(np.round(budget, 0))

    rows.append(
        {
            "engagement_id": f"E{str(i+1).zfill(3)}",
            "startup_id": sid,
            "domain": domain,
            "service": service,
            "pricing_model": price_model,
            "complexity_1_5": complexity,
            "team_size": team_size,
            "duration_weeks": duration_w,
            "region": region,
            "start_date": start_dt.date(),
            "data_volume_gb": data_gb,
            "kpi_baseline": kpi_baseline,
            "expected_gain_pct": expected_gain,
            "toolstack": tools,
            "budget_usd": budget,
        }
    )

df = pd.DataFrame(rows).sort_values(["start_date", "startup_id"]).reset_index(drop=True)
df.to_csv("analytics_services_dataset.csv", index=False)

print("Розмір датасету:", df.shape)
print("Збережено у файл: analytics_services_dataset.csv")
display(df.head(10))

# 2) приклад динамічного часового ряду для демонстрацій у проєкті
# моделюємо тижневий kpi, напр. weekly_active_users для фічі з прогнозуванням попиту
weeks = pd.date_range("2024-01-07", "2025-12-28", freq="W")
t = np.arange(len(weeks))

# компоненти ряду
trend = 2000 + 8.0 * t  # повільне зростання
season = 180 * np.sin(2 * np.pi * t / 52) + 90 * np.cos(
    2 * np.pi * t / 26
)  # річна + піврічна сезонність
noise = np.random.normal(0, 120, size=len(t))  # білий шум
events = np.zeros_like(t, dtype=float)
events[[20, 48, 70]] = [450, -300, 380]  # реліз фічі, збій, маркетинг

y = trend + season + events + noise
ts = pd.DataFrame({"date": weeks, "wau": y}).set_index("date")


# допоміжні функції
def moving_average(x, window=5):
    return pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy()


def exp_smoothing(x, alpha=0.3):
    s = np.zeros_like(x, dtype=float)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
    return s


def detrend(x):
    # лінійне вилучення тренду
    coeffs = np.polyfit(np.arange(len(x)), x, deg=1)
    return x - np.polyval(coeffs, np.arange(len(x))), coeffs


# оцінка тренду через kendall tau
tau, pval = kendalltau(t, y)

print(
    f"kendall_tau={tau:.3f}  pvalue={pval:.4f}  інтерпретація: p<0.05 → наявний монотонний тренд"
)

# згладжування
ma7 = moving_average(y, window=7)
ema = exp_smoothing(y, alpha=0.25)
window = 9 if len(y) >= 9 else (len(y) // 2) * 2 + 1
sg = savgol_filter(y, window_length=window, polyorder=2)  # згладжування Савітцкі–Ґолея

# проста декомпозиція
y_dt, coeffs = detrend(y)
season_est = moving_average(y_dt, window=52)  # груба оцінка сезонності
residual = y - (np.polyval(coeffs, t) + np.nan_to_num(season_est))

# графіки
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
axes[0].plot(ts.index, y, color="#1f77b4", label="ряд")
axes[0].set_title("Тижневий ряд wau")
axes[0].legend(loc="upper left")

axes[1].plot(ts.index, ma7, label="cov moving average 7", color="#ff7f0e")
axes[1].plot(ts.index, ema, label="exp smoothing α=0.25", color="#2ca02c")
axes[1].plot(ts.index, sg, label="savgol window=9", color="#d62728")
axes[1].set_title("Згладжування")
axes[1].legend(loc="upper left")

axes[2].plot(ts.index, np.polyval(coeffs, t), color="#9467bd", label="тренд")
axes[2].plot(ts.index, season_est, color="#8c564b", label="сезонність ~52")
axes[2].set_title("Оцінені компоненти")
axes[2].legend(loc="upper left")

axes[3].plot(ts.index, residual, color="#7f7f7f", label="залишок")
axes[3].axhline(0, color="black", linewidth=0.8)
axes[3].set_title("Залишок")
axes[3].legend(loc="upper left")

plt.tight_layout()
plt.show()

# 3) приклади прикріплені до предмету
# − перевірка впливу релізу фічі на зростання − простий before vs after
release_week = weeks[20]
before = ts.loc[:release_week, "wau"].tail(8).mean()
after = ts.loc[release_week + pd.Timedelta(weeks=1) :, "wau"].head(8).mean()
uplift = (after - before) / max(before, 1e-9)
print(f"Середній uplift після релізу ≈ {uplift*100:.1f}%")

# − корисні зрізи по датасету послуг
pivot_cost = df.pivot_table(
    index="service", values="budget_usd", aggfunc=["mean", "median", "count"]
)
print(pivot_cost.sort_values(("mean", "budget_usd"), ascending=False).head(10))

# − збереження підмножин для навчальних прикладів
df_forecast = df[df["service"] == "time_series_forecast"].copy()
df_forecast.to_csv("subset_time_series_forecast.csv", index=False)
print("Збережено subset_time_series_forecast.csv для прикладів по часових рядах")
