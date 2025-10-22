import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# 1) Емуляція реальних статичних даних
# часовий ряд: місяці за 10 років
periods = 10 * 12
idx = pd.date_range(start="2015-01-01", periods=periods, freq="M")

# Супутниковий індекс (наприклад, NDVI-подібний): сезонність + слабкий позитивний тренд + шум
t = np.arange(periods)
seasonal = 0.15 * np.sin(2 * np.pi * (t % 12) / 12)  # місячна сезонність
trend_ndvi = 0.0008 * t  # повільне збільшення індексу з роками
ndvi = 0.45 + seasonal + trend_ndvi + np.random.normal(scale=0.02, size=periods)

# Другий індекс (наприклад, EVI-подібний) — корельований з NDVI, але з іншим шумом
evi = 0.30 + 0.9 * (ndvi - ndvi.mean()) + np.random.normal(scale=0.03, size=periods)

# Врожайність (тонн/га) — має сезонну компоненту (рік як агрономічний цикл), слабкий тренд, залежність від NDVI з лагом 2-3 місяці
lag = 3
yield_base = 3.0 + 0.002 * t  # базовий повільний тренд урожайності
yield_season = 0.25 * np.cos(2 * np.pi * (t % 12) / 12)  # сезонність врожаю
yield_from_ndvi = 1.2 * np.roll(ndvi, lag)  # вплив NDVI з лагом
noise_yield = np.random.normal(scale=0.12, size=periods)
yield_series = (
    yield_base
    + yield_season
    + 0.5 * (yield_from_ndvi - yield_from_ndvi.mean())
    + noise_yield
)

# Зберемо в DataFrame
df = pd.DataFrame({"NDVI": ndvi, "EVI": evi, "Yield": yield_series}, index=idx)

# 2) Графічне відображення та висновок щодо тренду
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["NDVI"], label="NDVI", color="green")
plt.plot(df.index, df["Yield"], label="Yield (t/ha)", color="brown", alpha=0.8)
plt.title("NDVI і Врожайність (імітація)")
plt.legend()
plt.xlabel("Час")
plt.tight_layout()
plt.show()

# Лінійна регресія для швидкої перевірки тренду (NDVI і Yield)
slope_ndvi, intercept, rvalue, pvalue_ndvi, stderr = stats.linregress(
    np.arange(periods), df["NDVI"]
)
slope_yield, _, _, pvalue_yield, _ = stats.linregress(np.arange(periods), df["Yield"])

print(f"NDVI: slope = {slope_ndvi:.6f}, p-value = {pvalue_ndvi:.4f}")
print(f"Yield: slope = {slope_yield:.6f}, p-value = {pvalue_yield:.4f}")

# 3) Критерій «висхідних/низхідних серій» (runs test) на наявність тренду у Yield
# Метод: беремо знаки перших різниць, підрахунок серій, нормальне наближення
diff = np.diff(df["Yield"].values)
# позбутися нульових різниць
signs = np.sign(diff)
mask = signs != 0
signs = signs[mask]
n_pos = np.sum(signs > 0)
n_neg = np.sum(signs < 0)
runs = 1 + np.sum(signs[:-1] != signs[1:])

n = n_pos + n_neg
# очікувана кількість серій та дисперсія
expected_runs = (2 * n_pos * n_neg) / n + 1
var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))
z = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0.0
p_two = 2 * (1 - stats.norm.cdf(abs(z)))

print("\nRuns test для перших різниць врожайності:")
print(f"Параметри: n_pos={n_pos}, n_neg={n_neg}, runs={runs}")
print(f"expected_runs={expected_runs:.3f}, z={z:.3f}, p-value={p_two:.4f}")

alpha = 0.05
if p_two < alpha:
    runs_conclusion = "Тенденція присутня (відхилена гіпотеза випадковості серій)."
else:
    runs_conclusion = "Тенденція відсутня (неможливо відхилити випадковість серій)."
print("Висновок за критерієм серій:", runs_conclusion)

# 4) Розклад динамічного ряду (тренд + сезон + шум) вручну через рухоме середнє (період 12)
period = 12
# центроване рухоме середнє (центрований фільтр довжини period)
trend_ma = df["Yield"].rolling(window=period, center=True, min_periods=1).mean()

# сезонність: для кожного місяця обчислити середнє по роках з урахуванням тренду
detrended = df["Yield"] - trend_ma
# Для отримання сезонної складової: усереднити по місяцях (groupby month)
seasonal_est = detrended.groupby(detrended.index.month).transform("mean")
# остаточна випадкова компонента
residual = df["Yield"] - trend_ma - seasonal_est

# Графіки розкладу
plt.figure(figsize=(12, 9))
plt.subplot(4, 1, 1)
plt.plot(df.index, df["Yield"], label="Оригінал", color="black")
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(df.index, trend_ma, label="Тренд (MA12)", color="red")
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(df.index, seasonal_est, label="Сезонність (місяць середній)", color="blue")
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(df.index, residual, label="Випадкова компонента", color="gray")
plt.legend()
plt.tight_layout()
plt.show()

# Коментарі по складовим: оцінка амплітуд і значущості
print("\nОцінка складових:")
print(
    f"Тренд: останнє значення тренду = {trend_ma.iloc[-1]:.3f}, початкове = {trend_ma.iloc[0]:.3f}"
)
season_amp = (
    seasonal_est.groupby(seasonal_est.index.month).mean().max()
    - seasonal_est.groupby(seasonal_est.index.month).mean().min()
)
print(f"Амплітуда сезонності (max-min по місяцях) = {season_amp:.3f}")
resid_std = residual.std(skipna=True)
print(f"Стандартне відхилення випадкової компоненти = {resid_std:.3f}")


# 5) Побудова автокореляційної функції випадкової компоненти та висновок щодо коректності розкладу
def acf(x, nlags):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    n = len(x)
    x_mean = x.mean()
    c0 = np.sum((x - x_mean) ** 2) / n
    acf_vals = []
    for lag in range(nlags + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            acov = np.sum((x[: n - lag] - x_mean) * (x[lag:] - x_mean)) / n
            acf_vals.append(acov / c0 if c0 != 0 else 0.0)
    return np.array(acf_vals)


nlags = 24
acf_vals = acf(residual.values, nlags=nlags)
lags = np.arange(nlags + 1)

plt.figure(figsize=(10, 4))
plt.stem(lags, acf_vals, basefmt=" ")
conf = 1.96 / np.sqrt(
    np.sum(~np.isnan(residual))
)  # межа значущості для незалежного білого шуму
plt.hlines([conf, -conf], xmin=0, xmax=nlags, colors="red", linestyles="dashed")
plt.title("Автокореляційна функція випадкової компоненти")
plt.xlabel("Лаг")
plt.ylabel("ACF")
plt.tight_layout()
plt.show()

# Висновок по АКФ: перевірка білої шумовості
n_res = np.sum(~np.isnan(residual))
significant_lags = np.where(np.abs(acf_vals) > conf)[0]
print(
    f"\nКількість значущих лагів (|ACF| > ±{conf:.3f}): {len(significant_lags)}; лаги: {significant_lags.tolist()}"
)
if len(significant_lags) <= 1:  # лаг 0 завжди значущий
    acf_conclusion = "Розклад коректний: випадкова компонента близька до білого шуму."
else:
    acf_conclusion = "Розклад сумнівний: залишилися автокореляції у випадковій компоненті — можлива неповна очистка тренду/сезонності або AR-структура."
print("Висновок по АКФ:", acf_conclusion)
