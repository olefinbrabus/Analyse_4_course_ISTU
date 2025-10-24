import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LassoCV
from collections import Counter

rng = np.random.default_rng(42)
plt.rcParams.update({
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "font.size": 11
})

def describe(title, df):
    print(f"\n- {title}")
    display(df.head())
    print(df.describe().T)

# 1. дерево рішень
n = 900
X = np.column_stack([
    rng.normal(0, 1.0, n),
    rng.normal(2, 1.2, n),
    rng.normal(-1, 0.7, n),
    rng.normal(0, 2.0, n),
    rng.integers(0, 2, n),
    rng.normal(1, 1.5, n)
])
y = ((X[:,0] + 0.5*X[:,1] - 0.8*X[:,2] + 0.2*X[:,5]) > 1.2).astype(int)
y[X[:,4] == 1] = (y[X[:,4] == 1] + 1) % 3
y = np.clip(y, 0, 2)
df_clf = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
df_clf["target"] = y
describe("класифікаційні дані", df_clf)

X_train, X_test, y_train, y_test = train_test_split(df_clf.drop("target", axis=1), y, test_size=0.3, random_state=42, stratify=y)
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
print("accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(20,15), dpi=400)
plot_tree(tree, feature_names=X_train.columns, class_names=[str(c) for c in np.unique(y)], filled=True, rounded=True, fontsize=8)
plt.title("дерево рішень", fontsize=12)
plt.tight_layout()
plt.show()

imp = pd.Series(tree.feature_importances_, index=X_train.columns).sort_values(ascending=False)
display(imp.to_frame("importance").T)

# 2. кореляції
n2 = 250
hemoglobin = rng.normal(14.2, 1.1, n2)
ldl = 100 + 5*(hemoglobin - 14) + rng.normal(0, 10, n2)
hdl = 60 - 2*(hemoglobin - 14) + rng.normal(0, 7, n2)
df_blood = pd.DataFrame({"hemoglobin": hemoglobin, "ldl": ldl, "hdl": hdl})
describe("ліпопротеїни і гемоглобін", df_blood)

for var in ["ldl", "hdl"]:
    r_p, p_p = stats.pearsonr(df_blood[var], df_blood["hemoglobin"])
    print(f"{var}: r={r_p:.3f}, p={p_p:.3g}")
    plt.figure(figsize=(5,4), dpi=400)
    plt.scatter(df_blood[var], df_blood["hemoglobin"], s=20, alpha=0.7)
    m, b = np.polyfit(df_blood[var], df_blood["hemoglobin"], 1)
    xs = np.linspace(df_blood[var].min(), df_blood[var].max(), 100)
    plt.plot(xs, m*xs + b, color="red")
    plt.xlabel(var)
    plt.ylabel("hemoglobin")
    plt.title(f"{var} vs гемоглобін")
    plt.tight_layout()
    plt.show()

# 3. нелінійна модель
glucose = rng.normal(95, 10, n2)
def model_exp(x, a, b, c): return a + b*np.exp(x/120) + c*x
hemoglobin_nl = 12.5 + 0.2*np.exp(glucose/120) + 0.01*glucose + rng.normal(0, 0.6, n2)
popt, _ = curve_fit(model_exp, glucose, hemoglobin_nl, p0=[12, 0.1, 0.01], maxfev=10000)
pred_nl = model_exp(glucose, *popt)
r2 = r2_score(hemoglobin_nl, pred_nl)
rmse = np.sqrt(mean_squared_error(hemoglobin_nl, pred_nl))
print("нелінійна модель:", popt, "R2=", r2, "RMSE=", rmse)

plt.figure(figsize=(6,4), dpi=400)
plt.scatter(glucose, hemoglobin_nl, s=15, alpha=0.6, label="дані")
xs = np.linspace(glucose.min(), glucose.max(), 200)
plt.plot(xs, model_exp(xs, *popt), color="red", label="модель")
plt.xlabel("стабілізована глюкоза")
plt.ylabel("гемоглобін")
plt.legend()
plt.tight_layout()
plt.show()

# 4. регресія
Xr = pd.DataFrame({"ldl": ldl, "hdl": hdl, "glucose": glucose, "ldl_hdl": ldl*hdl})
yr = hemoglobin_nl
lasso = LassoCV(cv=5, random_state=42).fit(Xr, yr)
coef = pd.Series(lasso.coef_, index=Xr.columns)
print("коефіцієнти:\n", coef)

yhat = lasso.predict(Xr)
plt.figure(figsize=(5,4), dpi=400)
plt.scatter(yr, yhat, s=15, alpha=0.6)
lim = [min(yr.min(), yhat.min()), max(yr.max(), yhat.max())]
plt.plot(lim, lim, color="red")
plt.xlabel("факт")
plt.ylabel("прогноз")
plt.title("факт vs прогноз")
plt.tight_layout()
plt.show()

# 5. закон ципфа
texts = [
    "data science builds data products and services",
    "services for startups need reliable data analysis",
    "analysis and modeling enable decision making"
]
docs = [t.lower().split() for t in texts]
vocab = sorted(set(w for d in docs for w in d))
df = pd.DataFrame(0, index=range(len(docs)), columns=vocab)
for i, d in enumerate(docs):
    c = Counter(d)
    for w, cnt in c.items():
        df.loc[i, w] = cnt
N = len(docs)
dfreq = (df > 0).sum(axis=0)
idf = np.log((N + 1) / (dfreq + 1)) + 1.0
tf = df.div(df.sum(axis=1), axis=0)
tfidf = tf * idf
display(tfidf.round(3))

counts = Counter(w for d in docs for w in d)
ranked = pd.Series(sorted(counts.values(), reverse=True))
plt.figure(figsize=(5,4), dpi=400)
plt.loglog(range(1, len(ranked)+1), ranked.values, marker="o")
plt.xlabel("ранг")
plt.ylabel("частота")
plt.title("закон ципфа")
plt.tight_layout()
plt.show()