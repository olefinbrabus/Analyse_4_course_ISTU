# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
import warnings
warnings.filterwarnings("ignore")

# 1. Генерація реалістичного часовго ряду: MAU (щомісячно), length=120
np.random.seed(42)
n = 120
index = pd.date_range(start='2016-01-01', periods=n, freq='M')

# Компоненти: повільний експонентальний/лінійний тренд + сезонність (рік) + автокореляція (AR(1)) + шум + випадкові шоки
t = np.arange(n)
trend = 50 + 0.8 * t  # лінійний тренд (зростання користувачів)
seasonal = 10 * np.sin(2 * np.pi * (t % 12) / 12)  # сезонність по місяцях
# AR(1) компонент
phi = 0.5
ar = np.zeros(n)
eps = np.random.normal(scale=6.0, size=n)
for i in range(1, n):
    ar[i] = phi * ar[i-1] + eps[i]
# Випадкові шоки (реальні події, флaктивація)
shocks = np.zeros(n)
shock_dates = [24, 60, 90]  # умовні події
shocks[shock_dates] = np.array([-30, 40, -20])
series = trend + seasonal + ar + shocks

# DataFrame
df = pd.Series(series, index=index, name='MAU').to_frame()

# Візуалізація початкового ряду
plt.figure(figsize=(10,4))
plt.plot(df.index, df['MAU'], label='MAU')
plt.title('Щомісячна кількість активних користувачів (згенерований ряд)')
plt.xlabel('Дата')
plt.ylabel('MAU')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Перевірка стаціонарності — ADF тест
def adf_report(x, title='ADF Test'):
    res = adfuller(x, autolag='AIC')
    print(title)
    print(f'ADF statistic: {res[0]:.4f}')
    print(f'p-value: {res[1]:.4f}')
    print('Used lag:', res[2])
    print('nobs:', res[3])
    for key, val in res[4].items():
        print(f'Critial Value ({key}): {val:.4f}')
    print()

adf_report(df['MAU'], 'ADF on original series')

# 3. Якщо нестаціонарний — застосуємо диференціювання.
# Зазвичай спочатку d=1, але перевіримо сезонне диференціювання теж (12).
df['diff1'] = df['MAU'].diff(1)
df['diff12'] = df['MAU'].diff(12)
df['diff1_12'] = df['MAU'].diff(12).diff(1)

plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(df['diff1'], label='1st diff')
plt.title('1-ша різниця')
plt.subplot(3,1,2)
plt.plot(df['diff12'], label='12th diff')
plt.title('Сезонна (12) різниця')
plt.subplot(3,1,3)
plt.plot(df['diff1_12'].dropna(), label='1st diff of seasonal diff')
plt.title('1-ша різниця від сезонної різниці')
plt.tight_layout()
plt.show()

adf_report(df['diff1'].dropna(), 'ADF on 1st difference')
adf_report(df['diff12'].dropna(), 'ADF on 12th difference')
adf_report(df['diff1_12'].dropna(), 'ADF on 1st of 12th difference')

# Висновок: за результатами ADF (див. вивід) виберемо d=1 та сезонне D=1 якщо потрібно.
# Для простоти підберемо не-сезонну ARIMA(p,1,q) — якщо сезону недостатньо сильного.
# 4. Підбір моделі ARIMA: грід-пошук за p,q (d=1)
y = df['MAU'].astype(float)
y_diff = y.diff(1).dropna()

best_aic = np.inf
best_order = None
best_model = None

p_max = 4
q_max = 4
for p in range(0, p_max+1):
    for q in range(0, q_max+1):
        try:
            model = ARIMA(y, order=(p,1,q))
            res = model.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = (p,1,q)
                best_model = res
        except Exception:
            continue

print('Best ARIMA order by AIC:', best_order)
print('AIC:', best_aic)
print()

# 5. Параметри моделі та діагностика
print(best_model.summary())

# Діагностика залишків
resid = best_model.resid
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(resid)
plt.title('Резидуали моделі')
plt.subplot(2,1,2)
plot_acf(resid.dropna(), lags=24, ax=plt.gca())
plt.tight_layout()
plt.show()

# Тест Льюнга-Бокса (нульова гіпотеза: немає автокорреляції залишків)
lb = acorr_ljungbox(resid.dropna(), lags=[12], return_df=True)
print('Ljung-Box test (lag=12):')
print(lb)
print()

# Тест нормальності (Jarque-Bera)
jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(resid.dropna())
print('Jarque-Bera:', jb_stat, 'p-value:', jb_pvalue, 'skew:', skew, 'kurtosis:', kurtosis)
print()

# 6. Якщо модель адекватна — прогноз на 10 періодів уперед
steps = 10
forecast_res = best_model.get_forecast(steps=steps)
forecast_mean = forecast_res.predicted_mean
conf_int = forecast_res.conf_int(alpha=0.05)

# Побудова прогнозу
plt.figure(figsize=(10,4))
plt.plot(y.index, y, label='Історія')
plt.plot(pd.date_range(y.index[-1]+pd.offsets.MonthEnd(1), periods=steps, freq='M'),
         forecast_mean, label='Прогноз', color='red')
plt.fill_between(pd.date_range(y.index[-1]+pd.offsets.MonthEnd(1), periods=steps, freq='M'),
                 conf_int.iloc[:,0], conf_int.iloc[:,1], color='pink', alpha=0.4)
plt.title(f'Прогноз ARIMA{best_order} на {steps} місяців')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Вивід прогнозних значень
fc_index = pd.date_range(y.index[-1]+pd.offsets.MonthEnd(1), periods=steps, freq='M')
fc_table = pd.DataFrame({
    'forecast': forecast_mean.values,
    'ci_lower': conf_int.iloc[:,0].values,
    'ci_upper': conf_int.iloc[:,1].values
}, index=fc_index)
print('Прогнозні значення (10 періодів):')
print(fc_table.round(2))

# Оцінимо MAPE на останніх 12-х точках для перевірки прогностичної адекватності (backtest)
train_end = -12
model_bt = ARIMA(y[:train_end], order=best_order).fit()
fc_bt = model_bt.get_forecast(steps=12).predicted_mean
actual = y[train_end:]
mape = np.mean(np.abs((actual - fc_bt) / actual)) * 100
print()
print(f'Backtest MAPE (12 months): {mape:.2f}%')