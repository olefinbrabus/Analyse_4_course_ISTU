import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # логістична сигмоїда

np.random.seed(42)

# --- 1. Генерація вибірки ---
X_all = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)  # вхід
y_all = np.sin(X_all)  # ціль у [-1,1]
# масштабуємо ціль у [0,1], бо вихід сигмоїди в [0,1]
y_all_s = (y_all + 1) / 2

# train / val / test
n = X_all.shape[0]
idx = np.arange(n)
np.random.shuffle(idx)
train_i = idx[: int(0.7 * n)]
val_i = idx[int(0.7 * n) : int(0.85 * n)]
test_i = idx[int(0.85 * n) :]

X_train, y_train = X_all[train_i], y_all_s[train_i]
X_val, y_val = X_all[val_i], y_all_s[val_i]
X_test, y_test = X_all[test_i], y_all_s[test_i]


# --- Допоміжні функції ---
def sigmoid(z):
    return expit(z)


def sigmoid_prime(a):
    return a * (1 - a)  # де a = sigmoid(z)


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# --- 2. Одношарова мережа ---
def train_single_layer(
    X_train, y_train, X_val, y_val, lr=0.5, epochs=2000, early_stop=300
):
    n = X_train.shape[0]
    W = np.random.randn(1, 1) * 0.5
    b = np.zeros((1,))
    history = {"train_loss": [], "val_loss": []}
    best_val = np.inf
    wait = 0
    best_params = (W.copy(), b.copy())
    for ep in range(epochs):
        # forward
        z = X_train @ W + b
        a = sigmoid(z)
        loss = 0.5 * mse(a, y_train)
        # backward
        dL_da = a - y_train
        dL_dz = dL_da * sigmoid_prime(a)
        grad_W = (X_train.T @ dL_dz) / n
        grad_b = np.mean(dL_dz, axis=0)
        # update
        W -= lr * grad_W
        b -= lr * grad_b
        # record
        history["train_loss"].append(loss)
        val_pred = sigmoid(X_val @ W + b)
        history["val_loss"].append(0.5 * mse(val_pred, y_val))
        # early stopping
        if history["val_loss"][-1] < best_val - 1e-9:
            best_val = history["val_loss"][-1]
            best_params = (W.copy(), b.copy())
            wait = 0
        else:
            wait += 1
            if wait > early_stop:
                break
    W, b = best_params
    return W, b, history


# --- 3. Багатошаровий персептрон (8–1) ---
def train_mlp_1hidden(
    X_train, y_train, X_val, y_val, hidden_units=8, lr=0.5, epochs=5000, early_stop=500
):
    n = X_train.shape[0]
    W1 = np.random.randn(1, hidden_units) * 0.5
    b1 = np.zeros((hidden_units,))
    W2 = np.random.randn(hidden_units, 1) * 0.5
    b2 = np.zeros((1,))
    history = {"train_loss": [], "val_loss": []}
    best_val = np.inf
    wait = 0
    best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
    for ep in range(epochs):
        # forward
        z1 = X_train @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        loss = 0.5 * mse(a2, y_train)
        # backward
        dL_da2 = a2 - y_train
        dL_dz2 = dL_da2 * sigmoid_prime(a2)
        grad_W2 = (a1.T @ dL_dz2) / n
        grad_b2 = np.mean(dL_dz2, axis=0)
        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * sigmoid_prime(a1)
        grad_W1 = (X_train.T @ dL_dz1) / n
        grad_b1 = np.mean(dL_dz1, axis=0)
        # update
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        # record
        history["train_loss"].append(loss)
        val_pred = sigmoid(sigmoid(X_val @ W1 + b1) @ W2 + b2)
        history["val_loss"].append(0.5 * mse(val_pred, y_val))
        # early stopping
        if history["val_loss"][-1] < best_val - 1e-9:
            best_val = history["val_loss"][-1]
            best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            wait = 0
        else:
            wait += 1
            if wait > early_stop:
                break
    W1, b1, W2, b2 = best_params
    return W1, b1, W2, b2, history


# Тренування
W_s, b_s, hist_s = train_single_layer(
    X_train, y_train, X_val, y_val, lr=0.5, epochs=3000, early_stop=200
)
W1, b1, W2, b2, hist_m = train_mlp_1hidden(
    X_train, y_train, X_val, y_val, hidden_units=8, lr=0.5, epochs=5000, early_stop=400
)


# Передбачення
def predict_single(X):
    return sigmoid(X @ W_s + b_s) * 2 - 1


def predict_mlp(X):
    a1 = sigmoid(X @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    return (a2 * 2 - 1), a1


y_pred_s = predict_single(X_test)
y_pred_m, a1_test = predict_mlp(X_test)

mse_single = np.mean((y_pred_s - np.sin(X_test)) ** 2)
mse_mlp = np.mean((y_pred_m - np.sin(X_test)) ** 2)

# --- Діаграми ---
# 1) Цільова функція і передбачення
plt.figure(figsize=(10, 5))
order = np.argsort(X_all.ravel())
plt.plot(
    X_all.ravel()[order],
    np.sin(X_all).ravel()[order],
    "k-",
    lw=2,
    label="sin(x) (ціль)",
)
plt.plot(
    X_test.ravel(),
    y_pred_s.ravel(),
    "o",
    label=f"Одношарова (MSE={mse_single:.4f})",
    alpha=0.6,
)
plt.plot(
    X_test.ravel(),
    y_pred_m.ravel(),
    "x",
    label=f"MLP 8–1 (MSE={mse_mlp:.4f})",
    alpha=0.6,
)
plt.legend()
plt.title("Ціль vs Передбачення")
plt.grid(True)

# 2) Криві втрат
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist_s["train_loss"], label="train loss single")
plt.plot(hist_s["val_loss"], label="val loss single")
plt.yscale("log")
plt.legend()
plt.title("Втрати – одношарова")
plt.subplot(1, 2, 2)
plt.plot(hist_m["train_loss"], label="train loss mlp")
plt.plot(hist_m["val_loss"], label="val loss mlp")
plt.yscale("log")
plt.legend()
plt.title("Втрати – MLP")

# 3) Активації прихованого шару
X_grid = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
a1_grid = sigmoid(X_grid @ W1 + b1)
plt.figure(figsize=(10, 6))
for i in range(a1_grid.shape[1]):
    plt.plot(X_grid.ravel(), a1_grid[:, i], label=f"нейрон {i+1}")
plt.title("Активації прихованого шару (8 нейронів)")
plt.xlabel("x")
plt.legend(loc="upper right", ncol=2)

# 4) Гістограми ваг та розподіл помилок
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(W1.ravel(), bins=20, alpha=0.7, label="W1")
plt.hist(W2.ravel(), bins=20, alpha=0.7, label="W2")
plt.legend()
plt.title("Гістограми ваг")
plt.subplot(1, 2, 2)
errors = y_pred_m.ravel() - np.sin(X_test).ravel()
plt.hist(errors, bins=30)
plt.title("Розподіл помилок (MLP)")
plt.xlabel("помилка")

plt.show()

print("MSE одношарова (тест):", mse_single)
print("MSE MLP-8-1 (тест):", mse_mlp)
