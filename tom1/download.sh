#!/usr/bin/env sh
# ==============================================================================
# Ultra-Fast Instant Downloader for ML Lab Files
# ==============================================================================

set -e

# Target directory (current working directory by default)
TARGET_DIR="${TARGET_DIR:-.}"
mkdir -p "${TARGET_DIR}"

# If no specific files are requested, extract all files instantly with zero network delay
if [ $# -eq 0 ]; then

cat << '__TOM_EOF__' > "${TARGET_DIR}/Bais.py"
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

categories = ["comp.graphics", "rec.sport.baseball", "sci.space", "talk.politics.misc"]
newsgroups = fetch_20newsgroups(
    subset="all", categories=categories, remove=("headers", "footers", "quotes")
)
X_text = newsgroups.data
y = newsgroups.target

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

count_vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_train_count = count_vectorizer.fit_transform(X_train_text)
X_test_count = count_vectorizer.transform(X_test_text)

binary_vectorizer = CountVectorizer(
    stop_words="english", binary=True, max_features=5000
)
X_train_binary = binary_vectorizer.fit_transform(X_train_text)
X_test_binary = binary_vectorizer.transform(X_test_text)

mnb = MultinomialNB()
mnb.fit(X_train_count, y_train)
y_pred_mnb = mnb.predict(X_test_count)
acc_mnb = accuracy_score(y_test, y_pred_mnb)
f1_mnb = f1_score(y_test, y_pred_mnb, average="weighted")

bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train)
y_pred_bnb = bnb.predict(X_test_binary)
acc_bnb = accuracy_score(y_test, y_pred_bnb)
f1_bnb = f1_score(y_test, y_pred_bnb, average="weighted")

print("Multinomial Naive Bayes")
print("Accuracy :", acc_mnb)
print("F1 Score :", f1_mnb)
print("\nBernoulli Naive Bayes")
print("Accuracy :", acc_bnb)
print("F1 Score :", f1_bnb)

results = pd.DataFrame(
    {
        "Model": ["Multinomial NB", "Bernoulli NB"],
        "Accuracy": [acc_mnb, acc_bnb],
        "F1 Score": [f1_mnb, f1_bnb],
    }
)
print("\nComparison")
print(results)

print("\nMultinomial NB Report")
print(classification_report(y_test, y_pred_mnb, target_names=newsgroups.target_names))
print("\nBernoulli NB Report")
print(classification_report(y_test, y_pred_bnb, target_names=newsgroups.target_names))
__TOM_EOF__

cat << '__TOM_EOF__' > "${TARGET_DIR}/BSFOLD.py"
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
__TOM_EOF__

cat << '__TOM_EOF__' > "${TARGET_DIR}/CANCER.py"
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mle_model = LogisticRegression(penalty="l2", C=1e6, max_iter=5000)
mle_model.fit(X_train, y_train)
y_pred_mle = mle_model.predict(X_test)

map_l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=5000)
map_l2.fit(X_train, y_train)
y_pred_l2 = map_l2.predict(X_test)

map_l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=5000)
map_l1.fit(X_train, y_train)
y_pred_l1 = map_l1.predict(X_test)


def evaluate(name, y_true, y_pred):
    print("\n", name)
    print("Accuracy  :", accuracy_score(y_true, y_pred))
    print("Precision :", precision_score(y_true, y_pred))
    print("Recall    :", recall_score(y_true, y_pred))
    print("F1 Score  :", f1_score(y_true, y_pred))


evaluate("MLE Logistic Regression", y_test, y_pred_mle)
evaluate("MAP Logistic Regression (L2)", y_test, y_pred_l2)
evaluate("MAP Logistic Regression (L1)", y_test, y_pred_l1)

coef_df = pd.DataFrame(
    {"MLE": mle_model.coef_[0], "MAP_L2": map_l2.coef_[0], "MAP_L1": map_l1.coef_[0]}
)
print("\nCoefficient Comparison")
print(coef_df.head(10))
__TOM_EOF__

cat << '__TOM_EOF__' > "${TARGET_DIR}/KNN.py"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

k_values = [1, 3, 5, 7, 9]
results = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    results.append([k, accuracy, precision, recall, f1])
    print(f"\nK = {k}")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

results_df = pd.DataFrame(
    results, columns=["K", "Accuracy", "Precision", "Recall", "F1 Score"]
)
print("\nResults Summary")
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(results_df["K"], results_df["Accuracy"], marker="o")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs K")
plt.grid(True)
plt.show()
__TOM_EOF__

cat << '__TOM_EOF__' > "${TARGET_DIR}/RBF.py"
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:2000]
y_test = y_test[:2000]

linear_svm = SVC(kernel="linear")
linear_svm.fit(X_train, y_train)
linear_pred = linear_svm.predict(X_test)

poly_svm = SVC(kernel="poly", degree=3)
poly_svm.fit(X_train, y_train)
poly_pred = poly_svm.predict(X_test)

rbf_svm = SVC(kernel="rbf")
rbf_svm.fit(X_train, y_train)
rbf_pred = rbf_svm.predict(X_test)


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


evaluate_model(y_test, linear_pred, "Linear SVM")
evaluate_model(y_test, poly_pred, "Polynomial SVM")
evaluate_model(y_test, rbf_pred, "RBF SVM")
__TOM_EOF__

cat << '__TOM_EOF__' > "${TARGET_DIR}/svm_iris.py"
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = (iris.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

svm_model = SVC(kernel="linear", C=1)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_model.decision_function(xy)
Z = Z.reshape(XX.shape)

ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.8, linestyles=["--", "-", "--"]
)

ax.scatter(
    svm_model.support_vectors_[:, 0],
    svm_model.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="black",
)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Linear SVM Decision Boundary and Margin")
plt.show()
__TOM_EOF__

else
    BASE_URL="${BASE_URL:-https://raw.githubusercontent.com/Ashile99/ml-files/main/tom1}"
    FILES="$*"
    if command -v curl >/dev/null 2>&1; then
        FETCH_CMD="curl"
    elif command -v wget >/dev/null 2>&1; then
        FETCH_CMD="wget"
    else
        exit 1
    fi
    PID_LIST=""
    for file in ${FILES}; do
        if [ "${FETCH_CMD}" = "curl" ]; then
            curl -fsSL --connect-timeout 5 "${BASE_URL}/${file}" -o "${TARGET_DIR}/${file}" 2>/dev/null &
        else
            wget -q -O "${TARGET_DIR}/${file}" "${BASE_URL}/${file}" 2>/dev/null &
        fi
        PID_LIST="${PID_LIST} $!"
    done
    FAIL=0
    for pid in ${PID_LIST}; do
        wait "${pid}" 2>/dev/null || FAIL=1
    done
    if [ ${FAIL} -ne 0 ]; then
        exit 1
    fi
fi

echo "done"
