import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from mord import LogisticIT

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("mobile_price_classification.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# =========================
# FEATURES & TARGET
# =========================

X = df.drop("price_range", axis=1)
y = df["price_range"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 1. LOGISTIC REGRESSION
# =========================================================

log_reg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    ))
])

log_reg_pipeline.fit(X_train, y_train)

y_pred_lr = log_reg_pipeline.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# =========================================================
# 2. RANDOM FOREST
# =========================================================

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n===== Feature Importance =====")
print(feature_importance.head(10))

# =========================================================
# 3. XGBOOST
# =========================================================

xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=4,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\n===== XGBoost =====")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# =========================================================
# 4. ORDINAL LOGISTIC REGRESSION
# =========================================================

ordinal_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticIT())
])

ordinal_model.fit(X_train, y_train)

y_pred_ord = ordinal_model.predict(X_test)

print("\n===== Ordinal Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, y_pred_ord))
print(classification_report(y_test, y_pred_ord))

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred_ord)

labels = ["Low", "Medium", "High", "Very High"]

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Price Range")
plt.ylabel("Actual Price Range")
plt.title("Confusion Matrix - Ordinal Logistic Regression")

plt.tight_layout()
plt.show()

# =========================================================
# MODEL COMPARISON VISUAL
# =========================================================

models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "Ordinal Logistic Regression"
]

accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_xgb),
    accuracy_score(y_test, y_pred_ord)
]

colors = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52"
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    accuracies,
    color=colors,
    width=0.45
)

plt.ylabel("Accuracy")
plt.title("Comparison of Model Accuracies")

plt.ylim(0.85, 1.0)

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.004,
        f"{acc:.3f}",
        ha="center"
    )

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()