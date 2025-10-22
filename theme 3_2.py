import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --- Навчальна вибірка ---
X = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
y = np.sin(X)

# --- Кількість RBF-нейронів ---
n_hidden = 8

# --- Вибір центрів методом k-means ---
kmeans = KMeans(n_clusters=n_hidden, random_state=0).fit(X)
centers = kmeans.cluster_centers_

# --- Оцінка ширини (σ) ---
d_max = np.max([np.linalg.norm(c1 - c2) for c1 in centers for c2 in centers])
sigma = d_max / np.sqrt(2 * n_hidden)


# --- Функція RBF ---
def rbf(x, c, sigma):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * sigma**2))


# --- Матриця Φ ---
Phi = np.zeros((X.shape[0], n_hidden))
for i in range(X.shape[0]):
    for j in range(n_hidden):
        Phi[i, j] = rbf(X[i], centers[j], sigma)

# --- Навчання вихідних ваг (лінійна регресія) ---
W = np.linalg.pinv(Phi) @ y
y_pred = Phi @ W

# --- Візуалізація ---
plt.figure(figsize=(12, 6))

# 1. Виходи окремих RBF-нейронів
plt.subplot(1, 2, 1)
for j in range(n_hidden):
    plt.plot(X, Phi[:, j], label=f"RBF {j+1}")
plt.title("Виходи окремих RBF-нейронів")
plt.legend()

# 2. Порівняння апроксимації
plt.subplot(1, 2, 2)
plt.plot(X, y, label="sin(x)", color="black")
plt.plot(X, y_pred, label="RBFN вихід", color="red")
plt.scatter(centers, np.zeros_like(centers), color="blue", marker="x", label="Центри")
plt.title("Апроксимація sin(x) RBF-мережею")
plt.legend()

plt.show()
