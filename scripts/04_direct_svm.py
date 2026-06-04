import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")

X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train dogs:", len(set(y_train)))
print("Test dogs:", len(set(y_test)))

clf = LinearSVC(
    class_weight="balanced",
    max_iter=20000,
    random_state=0
)

print("\nTraining Linear SVM...")
clf.fit(X_train, y_train)

def evaluate(name, X, y):
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro")

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y, pred, zero_division=0))

evaluate("TRAIN accuracy sanity check", X_train, y_train)
evaluate("VIDEO4 cross-video test", X_test, y_test)
