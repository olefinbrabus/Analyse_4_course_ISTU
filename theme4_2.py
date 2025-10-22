from dataclasses import dataclass
from typing import Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize

rng = np.random.default_rng(42)


# ----------------------------
# 1) ДАТАСЕТ НА 300 ОБ'ЄКТІВ
# ----------------------------
def make_dataset(n: int = 300, noise_reg: float = 0.5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Інформативні ознаки
    x1 = rng.normal(0, 1.0, n)
    x2 = rng.normal(0, 1.0, n)
    x3 = rng.uniform(-2 * np.pi, 2 * np.pi, n)
    x4 = rng.normal(0, 1.0, n)
    x5 = rng.uniform(-1.0, 1.0, n)
    x6 = rng.lognormal(mean=0.0, sigma=0.6, size=n)  # слабо-інформативна

    # Категоріальні/кластерні патерни
    group = rng.integers(0, 3, n)  # 3 кластери
    c1 = (group == 1).astype(int)
    c2 = (group == 2).astype(int)

    # Регресійна ціль: нелінійність + взаємодія
    y_reg = (
        3.0 * x1
        - 2.0 * (x2**2)
        + 0.5 * np.sin(3.0 * x3)
        + 1.5 * x4 * x5
        + 0.8 * c1
        - 0.6 * c2
        + rng.normal(0, noise_reg, n)
    )

    # Класифікаційна ціль: логістична лінкомб
    z = 1.2 * x1 - 1.6 * x2 + 0.9 * np.sin(x3) + 0.8 * x4 * x5 + 0.7 * c1 - 0.7 * c2
    p = 1.0 / (1.0 + np.exp(-z))
    y_cls = (p > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x5": x5,
            "x6": x6,
            "group": group,
            "c1": c1,
            "c2": c2,
            "y_reg": y_reg,
            "y_cls": y_cls,
            "p_hat": p,
        }
    )
    return df


df = make_dataset()
# Попередній огляд
print(df.head())
print(df.describe())

# Візуалізація кількох розподілів та зв’язків
fig = plt.figure(figsize=(12, 6))
gs = GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["x1"], bins=30, color="#4e79a7")
ax1.set_title("x1")
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(df["x2"], df["y_reg"], s=10, alpha=0.6)
ax2.set_title("x2 vs y_reg")
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df["x4"] * df["x5"], df["y_reg"], s=10, alpha=0.6)
ax3.set_title("x4*x5 vs y_reg")
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df["p_hat"], bins=30, color="#f28e2b")
ax4.set_title("p_hat")
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df["x1"], df["x2"], c=df["group"], s=10, cmap="viridis")
ax5.set_title("x1 vs x2 (group)")
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(df["x3"], df["y_cls"] + np.random.normal(0, 0.02, len(df)), s=8, alpha=0.6)
ax6.set_title("x3 vs y_cls (jitter)")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 2) ПРИКЛАДИ ГРАДІЄНТНИХ МЕТОДІВ НА ЦЬОМУ ДАТАСЕТІ
#    Порівняємо CG та квазіньютонівські методи (BFGS, L-BFGS-B)
# -------------------------------------------------


# Логістична регресія: негативний log-likelihood
def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def logloss_grad_hess(w: np.ndarray, X: np.ndarray, y: np.ndarray):
    z = X @ w
    p = sigmoid(z)
    # Логлосс
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    # Градієнт
    grad = X.T @ (p - y) / X.shape[0]
    # Гессіан
    S = p * (1 - p)
    H = (X.T * S) @ X / X.shape[0]
    return loss, grad, H


# Формуємо матрицю ознак з взаємодією
X = np.c_[np.ones(len(df)), df[["x1", "x2", "x3", "x4", "x5", "c1", "c2"]].values]
y = df["y_cls"].values.astype(float)


def fun(w):
    return logloss_grad_hess(w, X, y)[0]


def jac(w):
    return logloss_grad_hess(w, X, y)[1]


w0 = np.zeros(X.shape[1])

# Conjugate Gradient (CG) через scipy.optimize.minimize
res_cg = minimize(fun, w0, jac=jac, method="CG", options={"maxiter": 500, "gtol": 1e-6})
print("CG status:", res_cg.success, "iters:", res_cg.nit, "loss:", res_cg.fun)

# BFGS
res_bfgs = minimize(
    fun, w0, jac=jac, method="BFGS", options={"maxiter": 500, "gtol": 1e-6}
)
print("BFGS status:", res_bfgs.success, "iters:", res_bfgs.nit, "loss:", res_bfgs.fun)

# L-BFGS-B
res_lbfgs = minimize(
    fun, w0, jac=jac, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-6}
)
print(
    "L-BFGS-B status:",
    res_lbfgs.success,
    "iters:",
    res_lbfgs.nit,
    "loss:",
    res_lbfgs.fun,
)

# Порівняльний барчарт кількості ітерацій і досягнутого лосу
labels = ["CG", "BFGS", "L-BFGS-B"]
nit = [res_cg.nit, res_bfgs.nit, res_lbfgs.nit]
losses = [res_cg.fun, res_bfgs.fun, res_lbfgs.fun]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(labels, nit, color=["#4e79a7", "#f28e2b", "#e15759"])
ax[0].set_title("К-сть ітерацій")
ax[1].bar(labels, losses, color=["#4e79a7", "#f28e2b", "#e15759"])
ax[1].set_title("Фінальний логлосс")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 3) СПРОЩЕНА ЕВОЛЮЦІЙНА ОПТИМІЗАЦІЯ (Genetic Algorithm)
#    Мінімізація функції Растрiгіна у 2D
# -------------------------------------------------


def rastrigin(v: np.ndarray) -> float:
    A = 10
    return A * len(v) + np.sum(v**2 - A * np.cos(2 * np.pi * v))


@dataclass
class GAConfig:
    pop_size: int = 40
    n_gen: int = 80
    cx_prob: float = 0.9
    mut_prob: float = 0.1
    bounds: Tuple[float, float] = (-5.12, 5.12)
    sigma_mut: float = 0.2
    elite: int = 2


def ga_minimize(
    fun: Callable[[np.ndarray], float], dim: int, cfg: GAConfig, seed: int = 0
):
    rng = np.random.default_rng(seed)
    low, high = cfg.bounds
    pop = rng.uniform(low, high, size=(cfg.pop_size, dim))
    fitness = np.array([fun(ind) for ind in pop])

    history = [fitness.min()]
    for g in range(cfg.n_gen):
        # Відбір: турнір 2
        parents = []
        for _ in range(cfg.pop_size):
            i, j = rng.integers(0, cfg.pop_size, 2)
            parents.append(pop[i] if fitness[i] < fitness[j] else pop[j])
        parents = np.array(parents)

        # Схрещування: однорідне
        children = parents.copy()
        for i in range(0, cfg.pop_size, 2):
            if rng.random() < cfg.cx_prob and i + 1 < cfg.pop_size:
                mask = rng.random(dim) < 0.5
                a, b = children[i].copy(), children[i + 1].copy()
                children[i][mask], children[i + 1][mask] = b[mask], a[mask]

        # Мутація: нормальний шум у межах
        for i in range(cfg.pop_size):
            if rng.random() < cfg.mut_prob:
                children[i] += rng.normal(0, cfg.sigma_mut, dim)
                children[i] = np.clip(children[i], low, high)

        # Елітизм
        elite_idx = np.argsort(fitness)[: cfg.elite]
        elite = pop[elite_idx].copy()
        # Оцінка нащадків
        child_fit = np.array([fun(ind) for ind in children])
        # Заміна + еліта
        replace_idx = np.argsort(child_fit)
        pop = children.copy()
        pop[replace_idx[: cfg.elite]] = elite
        fitness = np.array([fun(ind) for ind in pop])

        history.append(fitness.min())

    best_idx = fitness.argmin()
    return pop[best_idx], fitness[best_idx], np.array(history)


cfg = GAConfig()
best, fbest, hist = ga_minimize(rastrigin, dim=2, cfg=cfg, seed=123)
print("GA best:", best, "f*:", fbest)

plt.figure(figsize=(5, 3))
plt.plot(hist, lw=2)
plt.title("GA: зниження цільової функції (Rastrigin)")
plt.xlabel("покоління")
plt.ylabel("найкраще значення")
plt.tight_layout()
plt.show()


# -------------------------------------------------
# 4) СПРОЩЕНИЙ АНТ-КОЛОНІ АЛГОРИТМ (ACO) ДЛЯ TSP
#    Невеликий приклад на 20 міст
# -------------------------------------------------
def make_tsp_coords(n=20, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(n, 2))


def dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            D[i, j] = D[j, i] = d
    return D


@dataclass
class ACOConfig:
    n_ants: int = 30
    n_iter: int = 120
    alpha: float = 1.0  # вага феромону
    beta: float = 3.0  # вага евристики (1/D)
    rho: float = 0.5  # випаровування
    Q: float = 1.0  # інтенсивність
    seed: int = 1


def aco_tsp(D: np.ndarray, cfg: ACOConfig):
    rng = np.random.default_rng(cfg.seed)
    n = D.shape[0]
    tau = np.ones((n, n)) * 1.0  # початковий феромон
    eta = 1.0 / (D + 1e-12)  # евристика

    best_len = np.inf
    best_tour = None
    best_hist = []

    for it in range(cfg.n_iter):
        tours = []
        lengths = []
        for k in range(cfg.n_ants):
            start = rng.integers(0, n)
            unvisited = list(range(n))
            tour = [start]
            unvisited.remove(start)
            while unvisited:
                i = tour[-1]
                probs = []
                for j in unvisited:
                    probs.append((tau[i, j] ** cfg.alpha) * (eta[i, j] ** cfg.beta))
                probs = np.array(probs)
                probs = probs / probs.sum()
                j = unvisited[rng.choice(len(unvisited), p=probs)]
                tour.append(j)
                unvisited.remove(j)
            # замкнути цикл
            L = sum(D[tour[t], tour[(t + 1) % n]] for t in range(n))
            tours.append(tour)
            lengths.append(L)

        # Оновлення феромону
        tau *= 1.0 - cfg.rho
        for tour, L in zip(tours, lengths):
            deposit = cfg.Q / L
            for t in range(n):
                i, j = tour[t], tour[(t + 1) % n]
                tau[i, j] += deposit
                tau[j, i] += deposit

        it_best = np.min(lengths)
        if it_best < best_len:
            best_len = it_best
            best_tour = tours[int(np.argmin(lengths))]
        best_hist.append(best_len)

    return best_tour, best_len, np.array(best_hist)


coords = make_tsp_coords(20, seed=10)
D = dist_matrix(coords)
cfg_aco = ACOConfig()
tour, L, histL = aco_tsp(D, cfg_aco)
print("ACO best length:", L)

plt.figure(figsize=(5, 3))
plt.plot(histL, lw=2)
plt.title("ACO: скорочення довжини туру")
plt.xlabel("ітерація")
plt.ylabel("найкраща довжина")
plt.tight_layout()
plt.show()

# Візуалізація найкращого туру
xs, ys = coords[:, 0], coords[:, 1]
plt.figure(figsize=(5, 5))
plt.scatter(xs, ys, s=30, color="#4e79a7")
path_x = [coords[i, 0] for i in tour] + [coords[tour[0], 0]]
path_y = [coords[i, 1] for i in tour] + [coords[tour[0], 1]]
plt.plot(path_x, path_y, "-o", alpha=0.7)
for idx, (x, y) in enumerate(coords):
    plt.text(x + 0.01, y + 0.01, str(idx), fontsize=8)
plt.title(f"ACO TSP, L={L:.3f}")
plt.tight_layout()
plt.show()
