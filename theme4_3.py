from __future__ import annotations

import math
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    # корисно в ноутбуку, безпечно ігнорується у звичайному python
    from IPython.display import display  # type: ignore
except Exception:
    display = print

# -----------------------------
# секція 0. генерація датасету
# -----------------------------

RNG_SEED = 20251022
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

industries = [
    "fintech",
    "healthtech",
    "ecommerce",
    "edtech",
    "mobility",
    "gaming",
    "agritech",
    "martech",
    "proptech",
    "cybersec",
    "media",
    "iot",
]

service_categories = [
    "exploratory-analysis",
    "dashboards-bi",
    "etl-pipelines",
    "ml-classification",
    "ml-regression",
    "nlp",
    "cv-vision",
    "time-series-forecast",
    "recommendation",
    "ab-testing",
    "anomaly-detection",
    "optimization",
]

problems = {
    "fintech": ["fraud-score", "churn", "credit-scoring", "lifetime-value"],
    "healthtech": ["readmission", "triage", "claims-coding"],
    "ecommerce": ["conversion", "return-rate", "demand-forecast"],
    "edtech": ["dropout", "engagement", "content-tagging"],
    "mobility": ["eta", "surge-pricing", "dispatch"],
    "gaming": ["matchmaking", "cheat-detect", "arpdaus"],
    "agritech": ["yield-prediction", "ndvi-alerts", "price-forecast"],
    "martech": ["lead-scoring", "attribution", "segmentation"],
    "proptech": ["price-estimate", "tenant-risk", "maintenance"],
    "cybersec": ["intrusion-detect", "phishing", "log-anomalies"],
    "media": ["ctr", "topic-model", "toxicity"],
    "iot": ["predictive-maintenance", "sensor-fusion", "battery-life"],
}

deliverables = [
    "notebook-report",
    "ml-api",
    "docker-service",
    "airflow-dag",
    "powerbi-dash",
    "superset-dash",
    "fastapi-endpoint",
    "dbt-models",
]

stacks = [
    "python-pandas",
    "python-pySpark",
    "python-fastapi",
    "python-sklearn",
    "python-pytorch",
    "python-tensorflow",
    "dbt-bigquery",
    "airflow-gcp",
    "airflow-aws",
    "kafka",
    "snowflake",
    "postgres",
    "duckdb",
    "ray",
    "mlflow",
]

regions = ["eu", "us", "uk", "ua", "asia", "latam"]

seniority = ["junior", "middle", "senior"]

base_price = {
    "exploratory-analysis": 2500,
    "dashboards-bi": 3000,
    "etl-pipelines": 4000,
    "ml-classification": 6000,
    "ml-regression": 5500,
    "nlp": 6500,
    "cv-vision": 7000,
    "time-series-forecast": 5800,
    "recommendation": 7200,
    "ab-testing": 3500,
    "anomaly-detection": 5200,
    "optimization": 6400,
}

base_duration = {
    "exploratory-analysis": (2, 4),
    "dashboards-bi": (3, 6),
    "etl-pipelines": (4, 8),
    "ml-classification": (5, 10),
    "ml-regression": (5, 9),
    "nlp": (6, 12),
    "cv-vision": (6, 12),
    "time-series-forecast": (5, 10),
    "recommendation": (6, 12),
    "ab-testing": (3, 6),
    "anomaly-detection": (4, 8),
    "optimization": (5, 10),
}

metrics_pool = [
    "auc",
    "precision",
    "recall",
    "f1",
    "mae",
    "rmse",
    "mape",
    "lift@10",
    "ndcg@10",
    "latency-ms",
    "throughput-rps",
    "availability",
]

contact_channels = ["slack", "email", "jira", "notion", "github", "zoom"]
upsell_options = [
    "monitoring",
    "mlops",
    "support-24x7",
    "experiment-platform",
    "no-upsell",
]


def sample_duration(cat: str) -> int:
    lo, hi = base_duration[cat]
    return int(np.random.randint(lo, hi + 1))


def price_with_noise(cat: str, senior: str, region: str) -> int:
    p = base_price[cat]
    senior_k = {"junior": 0.9, "middle": 1.0, "senior": 1.25}[senior]
    region_k = {
        "us": 1.25,
        "uk": 1.2,
        "eu": 1.1,
        "ua": 0.8,
        "asia": 0.9,
        "latam": 0.85,
    }[region]
    noise = np.random.normal(0, p * 0.07)
    return int(max(1200, p * senior_k * region_k + noise))


def choose_metrics(cat: str) -> List[str]:
    if cat in ["ml-regression", "time-series-forecast"]:
        base = ["rmse", "mae", "mape"]
    elif cat in ["ml-classification", "anomaly-detection", "nlp", "cv-vision"]:
        base = ["auc", "precision", "recall", "f1"]
    elif cat in ["recommendation"]:
        base = ["ndcg@10", "lift@10", "precision"]
    elif cat in ["etl-pipelines", "dashboards-bi"]:
        base = ["latency-ms", "throughput-rps", "availability"]
    else:
        base = random.sample(metrics_pool, k=3)
    extra = [m for m in metrics_pool if m not in base]
    add = random.sample(extra, k=min(1, len(extra))) if np.random.rand() < 0.35 else []
    return base + add


def choose_stack(cat: str) -> List[str]:
    mapping = {
        "ml-classification": ["python-sklearn", "mlflow", "postgres"],
        "ml-regression": ["python-sklearn", "mlflow", "duckdb"],
        "nlp": ["python-pytorch", "python-tensorflow", "mlflow"],
        "cv-vision": ["python-pytorch", "mlflow"],
        "etl-pipelines": ["airflow-gcp", "dbt-bigquery", "kafka"],
        "dashboards-bi": ["powerbi-dash", "snowflake", "postgres"],
        "time-series-forecast": ["python-sklearn", "python-pytorch", "duckdb"],
        "recommendation": ["python-sklearn", "ray", "postgres"],
        "ab-testing": ["python-pandas", "duckdb"],
        "anomaly-detection": ["python-sklearn", "kafka", "postgres"],
        "optimization": ["python-pandas", "python-pySpark"],
        "exploratory-analysis": ["python-pandas", "duckdb"],
    }
    base = mapping.get(cat, ["python-pandas"])
    # невелика варіативність
    extra = random.sample([s for s in stacks if s not in base], k=1)
    return list(dict.fromkeys(base + extra))


rows = []
for i in range(1, 301):
    ind = random.choice(industries)
    cat = random.choice(service_categories)
    prob = random.choice(problems[ind])
    seni = random.choices(seniority, weights=[0.25, 0.5, 0.25])[0]
    reg = random.choice(regions)
    dur = sample_duration(cat)
    price = price_with_noise(cat, seni, reg)
    metrics = choose_metrics(cat)
    stk = choose_stack(cat)
    deliv = random.sample(deliverables, k=random.choice([1, 2, 2, 3]))
    sla_days = random.choice([7, 14, 21, 30])
    contact = random.choice(contact_channels)
    ups = random.choice(upsell_options)
    rows.append(
        {
            "id": i,
            "industry": ind,
            "service_category": cat,
            "problem": prob,
            "deliverables": ",".join(deliv),
            "key_metrics": ",".join(metrics),
            "price_usd": price,
            "duration_weeks": dur,
            "stack": ",".join(stk),
            "seniority": seni,
            "region": reg,
            "sla_days": sla_days,
            "contact_channel": contact,
            "upsell": ups,
            "notes": f"scope-{random.randint(1,4)}",
        }
    )

df = pd.DataFrame(rows)
df.to_csv("data_services_300.csv", index=False)

print("dataset saved -> data_services_300.csv")
try:
    display(df.head(8))
except Exception:
    pass

# ----------------------------------
# секція 1. задача 1 — стаціонарний ряд
# ----------------------------------

y = np.array(
    [1.6, 0.8, 1.2, 0.5, 0.9, 1.1, 1.1, 0.6, 1.5, 0.8, 0.9, 1.2, 0.5, 1.3, 0.8, 1.2],
    dtype=float,
)
t = np.arange(1, len(y) + 1)

# а) графік ряду
plt.figure(figsize=(9, 3.2))
plt.plot(t, y, marker="o", lw=1.5, color="#1f77b4")
plt.title("Задача 1а — графік часового ряду")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(alpha=0.35)
plt.tight_layout()

# б) наближена автокореляція першого порядку за графіком y(t+1) від y(t)
yt = y[:-1]
yt1 = y[1:]
slope, intercept, r_value, p_value, std_err = stats.linregress(yt, yt1)
rho1_approx = slope  # для стаціонарного процесу схил ≈ rho1

# в) точний коефіцієнт автокореляції першого порядку
rho1_exact = float(np.corrcoef(yt, yt1)[0, 1])

# графік розсіювання з лінією регресії
x_grid = np.linspace(yt.min() - 0.05, yt.max() + 0.05, 100)
y_pred = intercept + slope * x_grid

plt.figure(figsize=(4.8, 4.3))
plt.scatter(yt, yt1, s=36, color="#2ca02c", alpha=0.85, label="пари (y_t, y_{t+1})")
plt.plot(
    x_grid,
    y_pred,
    color="#d62728",
    lw=2,
    label=f"регресія: ŷ = {intercept:.3f} + {slope:.3f}·x",
)
plt.title("Задача 1б–в — y(t+1) від y(t)")
plt.xlabel("y(t)")
plt.ylabel("y(t+1)")
plt.legend()
plt.grid(alpha=0.35)
plt.tight_layout()

print(f"rho1 наближено за графіком ≈ {rho1_approx:.3f}")
print(f"rho1 точно за кореляцією   = {rho1_exact:.3f}")

# ----------------------------------
# секція 2. задача 2 — випадкове блукання з напрямом
# ----------------------------------
# модель: y_t = y_{t-1} + μ + ε_t, де ε_t ~ iid(0, σ_ε^2)
# прогноз на горизонт h: y_{t+h|t} = y_t + h·μ
# помилка прогнозу: e_{t+h} = sum_{j=1..h} ε_{t+j}
# дисперсія помилки: Var(e_{t+h}) = h·σ_ε^2
# середня квадратична похибка: RMSE(h) = sqrt(h)·σ_ε


def drift_estimate(y_series: np.ndarray) -> Tuple[float, float]:
    """оцінки μ та σ_ε за різницями Δy_t = y_t − y_{t-1}"""
    diffs = np.diff(y_series)
    mu_hat = float(np.mean(diffs))
    sigma_eps_hat = float(np.std(diffs - mu_hat, ddof=1))
    return mu_hat, sigma_eps_hat


def rw_drift_forecast(y_t: float, mu_hat: float, h: int) -> float:
    return y_t + h * mu_hat


def rw_drift_rmse(sigma_eps_hat: float, h: int) -> float:
    return math.sqrt(h) * sigma_eps_hat


# демонстрація на наданому ряді як приклад використання формул
mu_hat, sigma_eps_hat = drift_estimate(y)
for h in [1, 2, 4]:
    y_fore = rw_drift_forecast(y[-1], mu_hat, h)
    rmse = rw_drift_rmse(sigma_eps_hat, h)
    print(f"h={h}: прогноз y(t+{h}) = {y_fore:.3f}  — RMSE(h) ≈ {rmse:.3f}")

plt.show()

# кінець файлу
