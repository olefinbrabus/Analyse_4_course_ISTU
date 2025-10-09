import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Дані
data = {
    "Area": [55, 68, 75, 80, 95, 110, 125, 140, 160, 180],
    "Rooms": [2, 2, 3, 3, 3, 4, 4, 5, 5, 6],
    "Floor": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    "Price": [60, 70, 80, 85, 100, 120, 135, 150, 170, 200]
}
df = pd.DataFrame(data)

# Перевірка кореляції між змінними
print("Коваріаційна матриця:\n", df.corr())

# Залежна та незалежні змінні
X = df[["Area", "Rooms", "Floor"]]
y = df["Price"]

# Побудова моделі лінійної регресії
model = LinearRegression()
model.fit(X, y)

# Прогноз
y_pred = model.predict(X)

# Вивід параметрів моделі
print("\nКоефіцієнти (вплив змінних):", model.coef_)
print("Вільний член (intercept):", model.intercept_)
print("R^2 (коефіцієнт детермінації):", r2_score(y, y_pred))
print("RMSE (середньоквадратична помилка):", np.sqrt(mean_squared_error(y, y_pred)))

# Графічна візуалізація залежності площі та ціни
plt.figure(figsize=(10,6))
sns.scatterplot(x="Area", y="Price", size="Rooms", hue="Floor", palette="viridis", data=df, s=150)
plt.plot(df["Area"], y_pred, color="red", linewidth=2, label="Лінія регресії")
plt.title("Лінійна регресія: ціна залежно від площі, кількості кімнат та поверху")
plt.xlabel("Площа (м²)")
plt.ylabel("Ціна (тис. дол.)")
plt.legend()
plt.show()


# graph = {
#     "A": ["B", "C"],
#     "B": ["D", "E"],
#     "C": ["F"],
#     "D": [],
#     "E": ["F"],
#     "F": []
# }
#
# visited = set()
#
# def dfs(node):
#     if node not in visited:
#         print(node, end=" ")
#         visited.add(node)
#         for neighbor in graph[node]:
#             dfs(neighbor)
#
# print("Пошук у глибину (DFS):")
# dfs("A")
