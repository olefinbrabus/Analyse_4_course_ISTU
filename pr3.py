import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

# === Функції належності ===
def trapmf(x, a, b, c, d):
    y = np.zeros_like(x, dtype=float)
    y = np.where((x <= a) | (x >= d), 0.0, y)
    left = (x > a) & (x < b)
    y = np.where(left, (x - a) / (b - a + 1e-12), y)
    mid = (x >= b) & (x <= c)
    y = np.where(mid, 1.0, y)
    right = (x > c) & (x < d)
    y = np.where(right, (d - x) / (d - c + 1e-12), y)
    return np.clip(y, 0, 1)

def trimf(x, a, b, c):
    y = np.zeros_like(x, dtype=float)
    left = (x >= a) & (x <= b)
    y = np.where(left, (x - a) / (b - a + 1e-12), y)
    right = (x >= b) & (x <= c)
    y = np.where(right, (c - x) / (c - b + 1e-12), y)
    return np.clip(y, 0, 1)

def gaussmf(x, c, s):
    return np.exp(-0.5 * ((x - c) / (s + 1e-12)) ** 2)

# === Класи ===
class Term:
    def __init__(self, name, kind, params):
        self.name, self.kind, self.params = name, kind, list(params)
    def membership(self, x):
        if self.kind == 'trap': return trapmf(x, *self.params)
        if self.kind == 'tri':  return trimf(x, *self.params)
        if self.kind == 'gauss':return gaussmf(x, *self.params)

class FuzzyVariable:
    def __init__(self, name, universe, terms):
        self.name, self.universe, self.terms = name, universe, terms

class Rule:
    def __init__(self, antecedent, consequent, weight=1.0):
        self.antecedent, self.consequent, self.weight = antecedent, consequent, weight

class MamdaniFIS:
    def __init__(self, inputs, output, rules):
        self.inputs, self.output, self.rules = inputs, output, rules

    def infer(self, x_inputs, resolution=201):
        y0, y1 = self.output.universe
        ys = np.linspace(y0, y1, resolution)
        aggregated = np.zeros_like(ys)
        for rule in self.rules:
            mu_list = []
            for var_name, term_name, w in rule.antecedent:
                var = self.inputs[var_name]
                mu = var.terms[term_name].membership(np.array([x_inputs[var_name]]))[0]
                mu_list.append(mu * w)
            firing = min(mu_list) * rule.weight
            mu_out = self.output.terms[rule.consequent].membership(ys)
            aggregated = np.maximum(aggregated, np.minimum(mu_out, firing))
        denom = np.trapezoid(aggregated, ys)
        if denom < 1e-12: return float((y0+y1)/2)
        return float(np.trapezoid(aggregated*ys, ys)/denom)

# === Визначаємо початкову систему ===
budget = FuzzyVariable("budget",(0,100),{
    "low":  Term("low","trap",(0,0,20,40)),
    "mid":  Term("mid","tri",(25,50,75)),
    "high": Term("high","trap",(60,80,100,100)),
})
data_quality = FuzzyVariable("data_quality",(0,1),{
    "low":  Term("low","trap",(0,0,0.2,0.4)),
    "mid":  Term("mid","tri",(0.3,0.5,0.7)),
    "high": Term("high","trap",(0.6,0.8,1,1)),
})
product_stage = FuzzyVariable("product_stage",(0,1),{
    "idea":   Term("idea","trap",(0,0,0.15,0.35)),
    "mvp":    Term("mvp","gauss",(0.5,0.15)),
    "growth": Term("growth","trap",(0.65,0.85,1,1)),
})
service_priority = FuzzyVariable("service_priority",(0,1),{
    "low":  Term("low","trap",(0,0,0.2,0.4)),
    "mid":  Term("mid","tri",(0.3,0.5,0.7)),
    "high": Term("high","trap",(0.6,0.8,1,1)),
})
rules = [
    Rule([("budget","low",1.0)], "low", 1.0),
    Rule([("product_stage","idea",1.0)], "low", 1.0),
    Rule([("budget","mid",1.0),("data_quality","mid",1.0)], "mid", 1.0),
    Rule([("budget","high",1.0),("data_quality","high",1.0),("product_stage","growth",1.0)], "high", 1.0),
    Rule([("product_stage","mvp",1.0),("data_quality","high",1.0)], "high", 0.9),
    Rule([("data_quality","low",1.0)], "mid", 0.3),
]
fis_initial = MamdaniFIS({"budget":budget,"data_quality":data_quality,"product_stage":product_stage}, service_priority, rules)

# === Функція для малювання МФ ===
def plot_mfs(var, title):
    xs = np.linspace(var.universe[0], var.universe[1], 401)
    plt.figure()
    for name, term in var.terms.items():
        plt.plot(xs, term.membership(xs), label=name)
    plt.title(title); plt.xlabel(var.name); plt.ylabel("μ"); plt.legend(); plt.show()

# Початкові МФ
plot_mfs(budget, "Початкові МФ: budget")
plot_mfs(data_quality, "Початкові МФ: data_quality")
plot_mfs(product_stage, "Початкові МФ: product_stage")
plot_mfs(service_priority, "Початкові МФ: service_priority")

# === Генеруємо синтетичний датасет ===
rng = np.random.default_rng(7)
N = 200
df = pd.DataFrame({
    "budget": rng.uniform(0,100,N),
    "data_quality": rng.uniform(0,1,N),
    "product_stage": rng.choice([0.0,0.5,1.0], size=N, p=[0.3,0.4,0.3])
})
true_priority = (
    0.2 + 0.5*(1 - np.exp(-df["budget"].values/40)) +
    0.6*(df["data_quality"].values**1.2) +
    0.2*(df["product_stage"].values) -
    0.15*(df["product_stage"].values==0.0).astype(float)
)
df["target_priority"] = np.clip(true_priority/1.6 + rng.normal(0,0.04,N),0,1)
df.to_csv("fuzzy_dataset.csv", index=False)

tests = pd.DataFrame([
    {"budget":15,"data_quality":0.2,"product_stage":0.0,"label":"Idea, малий бюджет, слабкі дані"},
    {"budget":50,"data_quality":0.6,"product_stage":0.5,"label":"MVP, середній бюджет, ок дані"},
    {"budget":85,"data_quality":0.9,"product_stage":1.0,"label":"Growth, великий бюджет, високі дані"},
])
tests.to_csv("fuzzy_testcases.csv", index=False)

# === Функція RMSE ===
def rmse_current(fis, df):
    preds = [fis.infer({"budget":r.budget,"data_quality":r.data_quality,"product_stage":r.product_stage})
             for r in df.itertuples(index=False)]
    return float(np.sqrt(np.mean((np.array(preds)-df["target_priority"].values)**2))), np.array(preds)

rmse0, preds0 = rmse_current(fis_initial, df)

# === Стохастична оптимізація параметрів МФ ===
best_fis = copy.deepcopy(fis_initial)
best_rmse = rmse0
for _ in range(150):  # небагато ітерацій для прикладу
    cand = copy.deepcopy(best_fis)
    for var in cand.inputs.values():
        lo, hi = var.universe
        span = hi - lo
        for term in var.terms.values():
            term.params = [p + rng.normal(0,0.05*span) for p in term.params]
            # Обмеження у межах
            term.params = list(np.clip(term.params, lo, hi))
            term.params.sort()
    r, _ = rmse_current(cand, df)
    if r < best_rmse:
        best_rmse, best_fis = r, cand

rmse1, preds1 = rmse_current(best_fis, df)

# === Оптимізовані МФ ===
plot_mfs(best_fis.inputs["budget"], "Оптимізовані МФ: budget")
plot_mfs(best_fis.inputs["data_quality"], "Оптимізовані МФ: data_quality")
plot_mfs(best_fis.inputs["product_stage"], "Оптимізовані МФ: product_stage")

# === Порівняльний графік ===
plt.figure()
plt.scatter(df["target_priority"].values, preds0, alpha=0.6, label=f"До оптимізації (RMSE={rmse0:.3f})")
plt.scatter(df["target_priority"].values, preds1, alpha=0.6, label=f"Після оптимізації (RMSE={rmse1:.3f})")
plt.plot([0,1],[0,1],'r--')
plt.xlabel("Ціль"); plt.ylabel("Прогноз FIS"); plt.legend(); plt.title("Порівняння FIS до/після оптимізації")
plt.show()

print("RMSE до оптимізації:", rmse0)
print("RMSE після оптимізації:", rmse1)