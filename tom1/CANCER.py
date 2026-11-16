import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Breast Cancer Dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# MLE Logistic Regression
mle_model = LogisticRegression(penalty="l2", C=1e6, max_iter=5000)
mle_model.fit(X_train, y_train)
y_pred_mle = mle_model.predict(X_test)
# MAP Logistic Regression (L2)
map_l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=5000)
map_l2.fit(X_train, y_train)
y_pred_l2 = map_l2.predict(X_test)
# MAP Logistic Regression (L1)
map_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000)
map_l1.fit(X_train, y_train)
y_pred_l1 = map_l1.predict(X_test)


# Evaluation Function
def evaluate(name, y_true, y_pred):
    print("\n", name)
    print("Accuracy  :", accuracy_score(y_true, y_pred))
    print("Precision :", precision_score(y_true, y_pred))
    print("Recall    :", recall_score(y_true, y_pred))
    print("F1 Score  :", f1_score(y_true, y_pred))


# Evaluate Models
evaluate("MLE Logistic Regression", y_test, y_pred_mle)
evaluate("MAP Logistic Regression (L2)", y_test, y_pred_l2)
evaluate("MAP Logistic Regression (L1)", y_test, y_pred_l1)
# Coefficient Comparison
coef_df = pd.DataFrame(
    {"MLE": mle_model.coef_[0], "MAP_L2": map_l2.coef_[0], "MAP_L1": map_l1.coef_[0]}
)
print("\nCoefficient Comparison")
print(coef_df.head(10))
