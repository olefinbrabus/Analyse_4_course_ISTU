import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Генерація синтетичних даних (імітують реальні)
X, y = make_classification(
    n_samples=800,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    random_state=42,
    weights=[0.6, 0.4],  # трохи незбалансований клас
)

# Розбиття на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Логістична регресія як проста прогностична модель
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --- ROC-крива ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
plt.xlabel("False Positive Rate (Хибно-позитивна частка)")
plt.ylabel("True Positive Rate (Чутливість)")
plt.title("ROC-крива логістичної регресії")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# --- Матриця плутанини ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Клас 0", "Клас 1"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Матриця плутанини")
plt.show()

# --- Прогнозовані ймовірності ---
plt.figure(figsize=(7, 5))
plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.6, color="red", label="Клас 0")
plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.6, color="green", label="Клас 1")
plt.axvline(0.5, color="black", linestyle="--", lw=2, label="Поріг = 0.5")
plt.xlabel("Ймовірність належності до класу 1")
plt.ylabel("Кількість спостережень")
plt.title("Розподіл прогнозованих ймовірностей")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
