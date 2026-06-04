import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")
X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")

X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y
)

clf = LinearSVC(
    class_weight="balanced",
    max_iter=20000,
    random_state=0
)

clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)

print("=== RANDOM BARK SPLIT SVM ===")
print("Train clips:", len(X_tr))
print("Test clips:", len(X_te))
print("Dogs:", len(set(y)))
print("Accuracy:", accuracy_score(y_te, pred))
print("Macro F1:", f1_score(y_te, pred, average="macro"))
print(classification_report(y_te, pred, zero_division=0))
