import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

# 1. Генерація штучних даних
np.random.seed(42)
n = 200

# незалежні змінні
age = np.random.normal(50, 10, n)        # вік
treatment = np.random.binomial(1, 0.4, n) # група лікування (0/1)

# час до події
baseline_hazard = 0.02
time = np.random.exponential(1/(baseline_hazard*np.exp(0.03*age - 0.5*treatment)), n)

# подія (1 - сталася, 0 - цензурована)
event = np.random.binomial(1, 0.7, n)

df = pd.DataFrame({
    "time": time,
    "event": event,
    "age": age,
    "treatment": treatment
})

# 2. Модель Кокса
cph = CoxPHFitter()
cph.fit(df, duration_col="time", event_col="event")
cph.print_summary()

# 3. Побудова кривих виживання
kmf = KaplanMeierFitter()
plt.figure(figsize=(8,5))

for group in df["treatment"].unique():
    mask = df["treatment"] == group
    kmf.fit(df["time"][mask], df["event"][mask], label=f"Treatment={group}")
    kmf.plot_survival_function(ci_show=True)

plt.title("Криві виживання Kaplan-Meier для різних груп лікування")
plt.xlabel("Час")
plt.ylabel("Ймовірність виживання")
plt.grid()
plt.show()