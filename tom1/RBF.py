import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0
# Flatten images
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)
# Reduce training time
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]
# Linear Kernel
linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)
# Polynomial Kernel
poly_svm = SVC(kernel="poly", degree=3)
poly_svm.fit(X_train, y_train)
poly_pred = poly_svm.predict(X_test)
# RBF Kernel
rbf_svm = SVC(kernel="rbf")
rbf_svm.fit(X_train, y_train)
rbf_pred = rbf_svm.predict(X_test)


# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("\n", model_name)
    print("-" * 40)
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


# Display results
evaluate_model(y_test, linear_pred, "Linear SVM")
evaluate_model(y_test, poly_pred, "Polynomial SVM")
evaluate_model(y_test, rbf_pred, "RBF SVM")
