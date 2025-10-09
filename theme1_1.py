# Перероблений приклад: реалістичні дані, кореляційний аналіз, графіки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, linregress

sns.set(style="whitegrid", rc={"figure.dpi": 110})
np.random.seed(42)

# Генерація синтетичних, але реалістичних даних (місяць, витрати на рекламу в тис. $, продажі в одиницях)
months = pd.date_range(start="2023-01-01", periods=24, freq="MS")
t = np.arange(len(months))

# Витрати на рекламу: базовий рівень + невеликий тренд + сезонність + шум
ad_kusd = 20 + 0.8 * t + 3.5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3.0, size=len(t))
ad_kusd = np.round(ad_kusd, 2)

# Продажі: залежність від реклами + власна сезонність (наприклад, піки в листопаді-грудні) + шум
sales_units = 300 + 9.5 * ad_kusd + 40 * np.sin(2 * np.pi * (t + 2) / 12) + np.random.normal(0, 30, size=len(t))
sales_units = np.round(sales_units).astype(int)

df = pd.DataFrame({"Month": months, "Ad_kUSD": ad_kusd, "Sales_units": sales_units})

# Статистики кореляції
pearson_r, pearson_p = pearsonr(df["Ad_kUSD"], df["Sales_units"])
spearman_r, spearman_p = spearmanr(df["Ad_kUSD"], df["Sales_units"])

# Лінійна регресія (OLS)
slope, intercept, r_value, p_value, std_err = linregress(df["Ad_kUSD"], df["Sales_units"])
r_squared = r_value**2

# Вивід у консоль
print("=== Кореляційний аналіз ===")
print(f"Діапазон спостережень: {df['Month'].iloc[0].date()} — {df['Month'].iloc[-1].date()}")
print(f"Кількість спостережень: {len(df)}")
print()
print(f"Коефіцієнт Пірсона: r = {pearson_r:.3f}, p = {pearson_p:.4f}")
print(f"Коефіцієнт Спірмена: rho = {spearman_r:.3f}, p = {spearman_p:.4f}")
print()
print("Лінійна регресія (Sales = intercept + slope * Ad_kUSD):")
print(f"  slope = {slope:.3f}")
print(f"  intercept = {intercept:.3f}")
print(f"  R² = {r_squared:.3f}")
print(f"  p (slope) = {p_value:.4f}")
print()

# Побудова графіків

# 1) Діаграма розсіяння + регресійна лінія
plt.figure(figsize=(8,5))
sns.regplot(x="Ad_kUSD", y="Sales_units", data=df,
            scatter_kws={"s":70, "color":"#1f77b4", "edgecolor":"k"},
            line_kws={"color":"#2ca02c", "linewidth":2}, ci=95)
plt.title("Scatter + OLS: Продажі від витрат на рекламу")
plt.xlabel("Витрати на рекламу (тис. $)")
plt.ylabel("Продажі (одиниць)")
plt.tight_layout()
plt.show()

# 2) Часовий ряд: витрати vs продажі (дві осі)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df["Month"], df["Ad_kUSD"], marker="o", color="#1f77b4", label="Ad (k$)")
ax.set_ylabel("Ad (k$)", color="#1f77b4")
ax2 = ax.twinx()
ax2.plot(df["Month"], df["Sales_units"], marker="s", color="#ff7f0e", label="Sales (units)")
ax2.set_ylabel("Sales (units)", color="#ff7f0e")
ax.set_xlabel("Місяць")
fig.autofmt_xdate(rotation=45)
plt.title("Часовий ряд: витрати на рекламу та продажі")
plt.tight_layout()
plt.show()

# 3) Залишки: histogram і residuals vs fitted
fitted = intercept + slope * df["Ad_kUSD"]
residuals = df["Sales_units"] - fitted

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.hist(residuals, bins=8, color="#7f7f7f", edgecolor="k")
plt.title("Гістограма залишків")
plt.xlabel("Залишок")
plt.ylabel("Частота")

plt.subplot(1,2,2)
plt.scatter(fitted, residuals, color="#1f77b4", edgecolor="k", s=60)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.title("Залишки vs передбачені значення")
plt.xlabel("Передбачені значення")
plt.ylabel("Залишки")
plt.tight_layout()
plt.show()

ua_df = pd.DataFrame({
    "Місяць" : df["Month"],
    "Кількість Витрат\nтис. долларів" : df["Ad_kUSD"],
    "Проданих\nреклам" : df["Sales_units"],
})

# 4) Кореляційна матриця
plt.figure(figsize=(4,3))
sns.heatmap(ua_df[["Кількість Витрат\nтис. долларів","Проданих\nреклам"]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Кореляційна матриця")
plt.tight_layout()
plt.show()