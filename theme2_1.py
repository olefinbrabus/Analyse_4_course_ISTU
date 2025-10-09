import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')

# Генерація синтетичних даних
rng = np.random.RandomState(42)
n = 10000
x1 = rng.normal(loc=0.5, scale=1.2, size=n)
x2 = rng.normal(loc=-0.2, scale=1.0, size=n)
x3 = rng.normal(loc=0.0, scale=0.8, size=n)
x4 = 0.6 * x1 + 0.4 * rng.normal(size=n)
x5 = -0.3 * x2 + 0.7 * rng.normal(size=n)
x6 = rng.normal(size=n)
x7 = rng.uniform(-1, 1, size=n)
x8 = rng.binomial(1, 0.3, size=n)

X = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8]).T
logit = (1.2 * x1 - 0.9 * x2**2 + 0.8 * np.tanh(x3) + 0.5 * x4 - 0.3 * x8)
prob = 1 / (1 + np.exp(-(logit - 0.2)))
prob = 0.15 * prob / prob.mean()
prob = np.clip(prob, 1e-6, 1-1e-6)
y = rng.binomial(1, prob)

df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1,9)])
df['y'] = y

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='y'), df['y'], test_size=0.3, stratify=df['y'], random_state=42
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Logistic': LogisticRegression(max_iter=2000),
    'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=300, random_state=42)
}

for name, m in models.items():
    if name == 'Logistic':
        m.fit(X_train_s, y_train)
    else:
        m.fit(X_train, y_train)

preds = {}
for name, m in models.items():
    if name == 'Logistic':
        preds[name] = m.predict_proba(X_test_s)[:,1]
    else:
        preds[name] = m.predict_proba(X_test)[:,1]

results = {}
for name, probas in preds.items():
    auc = roc_auc_score(y_test, probas)
    ap = average_precision_score(y_test, probas)
    brier = brier_score_loss(y_test, probas)
    y_pred = (probas >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, digits=4, zero_division=0)
    results[name] = {'auc': auc, 'ap': ap, 'brier': brier, 'cm': cm, 'report': cr}

for name, r in results.items():
    print(f'== {name} ==')
    print(f"AUC: {r['auc']:.4f}, AP: {r['ap']:.4f}, Brier: {r['brier']:.4f}")
    print('Confusion matrix (threshold 0.5):')
    print(r['cm'])
    print('Classification report:')
    print(r['report'])
    print()

# Графіки ROC, PR
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for name, probas in preds.items():
    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.plot(fpr, tpr, label=f'{name} AUC={roc_auc_score(y_test, probas):.3f}')
plt.plot([0,1],[0,1],'k--', linewidth=0.6)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend()

plt.subplot(1,2,2)
for name, probas in preds.items():
    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.plot(recall, precision, label=f'{name} AP={average_precision_score(y_test, probas):.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall'); plt.legend()
plt.tight_layout()
plt.show()