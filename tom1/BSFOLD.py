import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils import resample

iris = load_iris()
X = iris.data
y = iris.target
model = DecisionTreeClassifier(random_state=42)

boot_accuracy = []
boot_f1 = []
n_iterations = 10
for i in range(n_iterations):
    X_boot, y_boot = resample(X, y, replace=True, random_state=i)
    indices = resample(range(len(X)), replace=True, random_state=i)
    oob = list(set(range(len(X))) - set(indices))
    if len(oob) == 0:
        continue
    X_test = X[oob]
    y_test = y[oob]
    model.fit(X_boot, y_boot)
    pred = model.predict(X_test)
    boot_accuracy.append(accuracy_score(y_test, pred))
    boot_f1.append(f1_score(y_test, pred, average="weighted"))
print("Bootstrapping Results")
print("Average Accuracy :", round(np.mean(boot_accuracy), 4))
print("Average F1 Score :", round(np.mean(boot_f1), 4))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracy = []
cv_f1 = []
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_accuracy.append(accuracy_score(y_test, pred))
    cv_f1.append(f1_score(y_test, pred, average="weighted"))
print("\nK-Fold Cross Validation Results")
print("Average Accuracy :", round(np.mean(cv_accuracy), 4))
print("Average F1 Score :", round(np.mean(cv_f1), 4))
