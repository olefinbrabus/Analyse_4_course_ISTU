from __future__ import annotations

import math
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------
# налаштування
# ---------------------------
RNG_SEED = 20251022
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
N = 300

# ---------------------------
# словники доменів і послуг
# ---------------------------
DOMAINS = [
    "fintech","healthtech","edtech","ecommerce","gaming","travel","proptech",
    "adtech","security","media","mlops","devtools"
]
SERVICES = [
    "etl","real-time streaming","batch analytics","dashboards","ab-testing",
    "forecasting","recommendations","anomaly detection","nlp","cv","rl",
    "feature store","data catalog","privacy-anonymization","fraud detection"
]
USE_CASES = [
    "churn prediction","conversion uplift","dynamic pricing","demand forecast",
    "fraud scoring","content ranking","advert bid optimization","personalization",
    "quality alerts","support triage","inventory optimization","sentiment"
]
CLOUDS = ["aws","gcp","azure","on-prem"]
REGIONS = ["eu-central","eu-west","us-east","us-west","ap-southeast"]
STACKS = [
    "python-pandas-sklearn","python-pytorch-mlflow","python-spark-delta",
    "scala-spark","node-kafka-flink","go-clickhouse","rust-arrow","python-duckdb"
]
PRICING = ["fixed","tiered","usage","freemium+usage"]
SECURITY = ["low","medium","high","very-high"]

# ---------------------------
# допоміжні генератори
# ---------------------------
def _id(prefix: str, k: int) -> str:
    return f"{prefix}-{k:04d}"

def _choose_w(weights: List[float], items: List[str]) -> str:
    return random.choices(items, weights=weights, k=1)[0]

def _lognorm(mean: float, sigma: float, size: int = 1) -> np.ndarray:
    # параметри для lognormal через середнє і sigma приблизно
    mu = math.log(mean) - 0.5 * sigma**2
    return np.random.lognormal(mean=mu, sigma=sigma, size=size)

def _clip(x, a, b):
    return float(np.clip(x, a, b))

def _priority(score: float) -> str:
    if score >= 0.75: return "p1"
    if score >= 0.5: return "p2"
    if score >= 0.25: return "p3"
    return "p4"

def _status(prob: float) -> str:
    return np.random.choice(["idea","pilot","mvp","scale"], p=[0.1,0.25,0.35,0.3])

def _bool(p: float) -> bool:
    return random.random() < p

def _lift_from_ab(baseline: float, variant: float) -> float:
    if baseline <= 0: return 0.0
    return (variant - baseline) / baseline * 100.0

# ---------------------------
# синтетичні залежності
# ---------------------------
def synthesize_row(k: int) -> dict:
    domain = random.choice(DOMAINS)
    service = random.choice(SERVICES)
    use_case = random.choice(USE_CASES)

    cloud = _choose_w([0.45,0.3,0.2,0.05], CLOUDS)
    region = _choose_w([0.35,0.25,0.25,0.1,0.05], REGIONS)
    stack = random.choice(STACKS)

    # складність − впливає на затримку, команду, ціну, ризики
    complexity = np.clip(int(np.round(stats.beta(2,2).rvs()*3))+1,1,4)  # 1..4
    team_size = int(np.clip(np.round(np.random.normal(4+complexity*1.2, 1.5)), 2, 14))

    # обсяг і швидкість даних
    data_volume_gb = _clip(_lognorm(mean=80+complexity*40, sigma=1.0)[0], 1, 5000)
    velocity_rows_per_s = _clip(_lognorm(mean=500*(1+0.6*complexity), sigma=1.0)[0], 10, 2e6)

    # латентність і SLA
    base_latency = max(5, np.random.gamma(shape=2+0.5*complexity, scale=15))
    latency_ms = float(np.round(_clip(base_latency, 5, 3000), 2))
    sla_target = float(np.round(0.95 + 0.01*random.randint(0,4), 3))  # 0.95..0.99

    # конфіденційність і нормативи
    security_level = _choose_w([0.15,0.35,0.35,0.15], SECURITY)
    gdpr = region.startswith("eu")
    hipaa = domain == "healthtech"
    anonymization = _bool(0.7 if gdpr or hipaa else 0.35)

    # режими обробки
    realtime = service in ["real-time streaming","fraud detection","anomaly detection","recommendations","nlp","cv"]
    batch = not realtime or _bool(0.4)

    # потреба в ML
    ml_required = service in ["forecasting","recommendations","anomaly detection","nlp","cv","fraud detection","rl"] or _bool(0.25)

    # тарифікація
    pricing_model = _choose_w([0.25,0.35,0.35,0.05], PRICING)

    # ефекти продукту
    base_conv = _clip(np.random.beta(2,18), 0.005, 0.15) * (1.1 if use_case in ["personalization","content ranking"] else 1.0)
    ab_testing = _bool(0.6)
    variant_conv = base_conv * (1 + np.random.normal(0.06, 0.05) if ab_testing else 1 + np.random.normal(0.01,0.02))
    lift_pct = _clip(_lift_from_ab(base_conv, variant_conv), -10, 40)

    nps_delta = float(np.round(np.random.normal(3.0 if dashboards_or_support(service:=service) else 1.0, 2.5),2))
    conversion_pct = float(np.round(variant_conv*100,3))
    churn_pct = float(np.round(np.clip(np.random.normal(8 - lift_pct*0.2, 3.5), 1.0, 35.0),2))

    # економіка
    users_k = _clip(_lognorm(mean=50+complexity*15, sigma=0.9)[0], 1, 5000)  # ~активні користувачі, тис
    arpu_usd = float(np.round(_clip(np.random.normal(12+complexity*5, 6), 2, 120),2))
    cac_usd = float(np.round(_clip(np.random.normal(25+complexity*8, 12), 5, 250),2))
    mrr_usd = float(np.round(users_k*1000 * (conversion_pct/100.0) * arpu_usd,2))
    cltv_usd = float(np.round(arpu_usd * 12 * (1 - churn_pct/100.0) / max(0.05, churn_pct/100.0),2))
    price_usd_per_month = float(np.round(
        (50 + 0.02*data_volume_gb + 0.00002*velocity_rows_per_s + 70*complexity) *
        (1.4 if security_level in ["high","very-high"] else 1.0) *
        (1.2 if realtime else 1.0), 2))

    profit_margin_pct = float(np.round(np.clip(np.random.normal(35 - complexity*4, 8), 5, 70),2))

    # процесні метрики
    deadline_days = int(np.clip(np.random.normal(30 + complexity*20, 10), 10, 180))
    risk_score = float(np.round(np.clip(
        0.15*complexity + 0.1*(1 if realtime else 0) + 0.15*(security_level in ["high","very-high"]) +
        0.1*(gdpr or hipaa) + np.random.beta(2,5), 0, 1), 3))
    priority = _priority(risk_score)
    status = _status(risk_score)

    return {
        "startup_id": _id("st", k),
        "domain": domain,
        "service": service,
        "use_case": use_case,
        "cloud": cloud,
        "region": region,
        "stack": stack,
        "complexity_lvl_1_4": complexity,
        "team_size": team_size,
        "data_volume_gb": round(data_volume_gb,2),
        "velocity_rows_per_s": int(velocity_rows_per_s),
        "latency_ms": latency_ms,
        "sla_target": sla_target,
        "security_level": security_level,
        "gdpr": bool(gdpr),
        "hipaa": bool(hipaa),
        "anonymization": anonymization,
        "realtime": bool(realtime),
        "batch": bool(batch),
        "ml_required": bool(ml_required),
        "pricing_model": pricing_model,
        "price_usd_per_month": price_usd_per_month,
        "ab_testing": bool(ab_testing),
        "lift_pct": float(round(lift_pct,2)),
        "nps_delta": nps_delta,
        "conversion_pct": conversion_pct,
        "churn_pct": churn_pct,
        "cac_usd": cac_usd,
        "arpu_usd": arpu_usd,
        "mrr_usd": mrr_usd,
        "cltv_usd": cltv_usd,
        "profit_margin_pct": profit_margin_pct,
        "deadline_days": deadline_days,
        "risk_score_0_1": risk_score,
        "priority": priority,
        "status": status,
    }

def dashboards_or_support(service: str) -> bool:
    return service in ["dashboards","support triage"] if "support triage" in SERVICES else service == "dashboards"

# ---------------------------
# генерація датафрейму
# ---------------------------
def build_dataset(n: int = N) -> pd.DataFrame:
    rows = [synthesize_row(i+1) for i in range(n)]
    df = pd.DataFrame(rows)
    # узгодження полів
    df["roi_payback_months"] = np.round(np.clip((df["cac_usd"] / np.maximum(1.0, df["arpu_usd"])), 1, 48), 1)
    df["unit_economics_ok"] = (df["profit_margin_pct"] >= 25) & (df["roi_payback_months"] <= 12)
    # корисні категорії
    df["gdpr_or_hipaa"] = df["gdpr"] | df["hipaa"]
    return df

# ---------------------------
# аналітика − приклади питань
# ---------------------------

def analytics(df: pd.DataFrame) -> dict:
    out = {}

    # 1 − кореляції для числових ознак
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr().round(3)
    out["corr_price"] = corr["price_usd_per_month"].sort_values(ascending=False).head(10)

    # 2 − різниця латентності між realtime та batch
    lat_rt = df.loc[df["realtime"], "latency_ms"]
    lat_bt = df.loc[df["batch"], "latency_ms"]
    # нормальність тут не гарантується − беремо mann-whitney
    mw = stats.mannwhitneyu(lat_rt, lat_bt, alternative="two-sided")
    out["mw_latency_pvalue"] = float(mw.pvalue)

    # 3 − вплив ab-testing на lift
    lift_ab = df.loc[df["ab_testing"], "lift_pct"]
    lift_no = df.loc[~df["ab_testing"], "lift_pct"]
    t2 = stats.ttest_ind(lift_ab, lift_no, equal_var=False)
    out["t_ab_lift_pvalue"] = float(t2.pvalue)

    # 4 − 95% ДІ для середнього churn_pct
    mean_churn = df["churn_pct"].mean()
    sem = stats.sem(df["churn_pct"])
    ci = stats.t.interval(0.95, len(df)-1, loc=mean_churn, scale=sem)
    out["churn_mean_ci95"] = (round(ci[0],2), round(ci[1],2))

    # 5 − проста регресія price ~ latency + data_volume + realtime + security
    # псевдо-оцінка через кореляції та стандартизацію − швидко і наочно
    Z = df[["price_usd_per_month","latency_ms","data_volume_gb","realtime"]].copy()
    Z["security_high"] = df["security_level"].isin(["high","very-high"]).astype(int)
    Z = (Z - Z.mean())/Z.std(ddof=0)
    y = Z["price_usd_per_month"].values
    X = Z.drop(columns=["price_usd_per_month"]).values
    X = np.c_[np.ones(len(Z)), X]  # константа
    beta = np.linalg.pinv(X) @ y
    names = ["const","latency_ms","data_volume_gb","realtime","security_high"]
    out["ols_coeffs_std"] = {n: float(round(b,3)) for n,b in zip(names, beta)}

    return out

# ---------------------------
# візуалізації
# ---------------------------
def plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    df["price_usd_per_month"].hist(bins=30, color="#4e79a7")
    plt.title("розподіл ціни − usd/місяць")
    plt.xlabel("price_usd_per_month")
    plt.ylabel("частота")

    plt.subplot(1,3,2)
    plt.scatter(df["data_volume_gb"], df["price_usd_per_month"], s=12, alpha=0.6, c="#f28e2b")
    plt.xscale("log")
    plt.title("ціна vs обсяг даних")
    plt.xlabel("data_volume_gb (log)")
    plt.ylabel("price_usd_per_month")

    plt.subplot(1,3,3)
    means = df.groupby("security_level")["price_usd_per_month"].mean().reindex(SECURITY)
    means.plot(kind="bar", color="#59a14f")
    plt.title("середня ціна за рівнем безпеки")
    plt.xlabel("security_level")
    plt.ylabel("avg price")

    plt.tight_layout()
    plt.show()

# ---------------------------
# головний блок
# ---------------------------
def main() -> None:
    df = build_dataset(N)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    # збереження
    df.to_csv("startup_data_services_300.csv", index=False)
    print("saved: startup_data_services_300.csv")

    # приклади запитань
    print(EXAMPLE_QUESTIONS)

    # аналітика
    out = analytics(df)

    # графіки
    plots(df)

if __name__ == "__main__":
    main()