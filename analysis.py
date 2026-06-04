import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
print(df.head())
print(df["Sleep Disorder"].value_counts())


df = df.drop(columns=["Person ID"])
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")  # ← add this

# Split Blood Pressure into systolic / diastolic
df[["Systolic BP", "Diastolic BP"]] = (
    df["Blood Pressure"].str.split("/", expand=True).astype(int)
)
df = df.drop(columns=["Blood Pressure"])

# One-hot encode categoricals
df = pd.get_dummies(df, columns=["Gender", "BMI Category", "Occupation"])

# Separate features and target
X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5-Fold CV setup 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#KNN — sweep k = 1, 3, 5, …, 50
k_values = range(1, 51, 2)
knn_cv_scores, knn_train_scores = [], []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_acc = cross_val_score(knn, X_scaled, y, cv=skf, scoring="accuracy").mean()
    knn.fit(X_scaled, y)
    train_acc = accuracy_score(y, knn.predict(X_scaled))
    knn_cv_scores.append(cv_acc)
    knn_train_scores.append(train_acc)

best_k = list(k_values)[np.argmax(knn_cv_scores)]
print(f"\nBest k = {best_k}  |  CV accuracy = {max(knn_cv_scores):.4f}")

#Logistic Regression
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
lr_cv_scores = []

for C in C_values:
    lr = LogisticRegression(C=C, penalty="l2", max_iter=1000, random_state=42)
    cv_acc = cross_val_score(lr, X_scaled, y, cv=skf, scoring="accuracy").mean()
    lr_cv_scores.append(cv_acc)

best_C = C_values[np.argmax(lr_cv_scores)]
print(f"Best C  = {best_C}  |  CV accuracy = {max(lr_cv_scores):.4f}")

#KNN bias-variance tradeoff 
plt.figure(figsize=(9, 5))
plt.plot(list(k_values), knn_train_scores, label="Training Accuracy", marker="o")
plt.plot(list(k_values), knn_cv_scores,    label="CV Accuracy",       marker="s")
plt.axvline(best_k, color="red", linestyle="--", label=f"Best k = {best_k}")
plt.xlabel("k"); plt.ylabel("Accuracy"); plt.title("KNN: Training vs CV Accuracy")
plt.legend(); plt.tight_layout()
plt.savefig("knn_bias_variance.png", dpi=150)
plt.show()

# Logistic Regression CV accuracy vs C 
plt.figure(figsize=(7, 4))
plt.semilogx(C_values, lr_cv_scores, marker="o", color="steelblue")
plt.axvline(best_C, color="red", linestyle="--", label=f"Best C = {best_C}")
plt.xlabel("C (regularization)"); plt.ylabel("CV Accuracy")
plt.title("Logistic Regression: CV Accuracy vs C")
plt.legend(); plt.tight_layout()
plt.savefig("lr_cv_accuracy.png", dpi=150)
plt.show()

#Final models & feature importance 
final_knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_scaled, y)
final_lr  = LogisticRegression(C=best_C, penalty="l2",
                                max_iter=1000, random_state=42).fit(X_scaled, y)

coef_df = pd.DataFrame(
    final_lr.coef_, columns=X.columns, index=final_lr.classes_
).T.abs().mean(axis=1).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
coef_df.head(15).sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 15 Feature Importances (Logistic Regression)")
plt.xlabel("|Coefficient| averaged across classes")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nTop 5 predictors:\n", coef_df.head())