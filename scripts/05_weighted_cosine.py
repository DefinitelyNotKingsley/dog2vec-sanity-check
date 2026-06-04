import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")

X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return -1
    return np.dot(a, b) / denom


print("Training Linear SVM...")

svm = LinearSVC(
    class_weight="balanced",
    max_iter=20000,
    random_state=0
)

svm.fit(X_train, y_train)

# Multi-class feature importance
weights = np.mean(np.abs(svm.coef_), axis=0)

print("Weight vector shape:", weights.shape)

# Apply weights
X_train_weighted = X_train * weights
X_test_weighted = X_test * weights

# Build weighted profiles
profiles = {}

for dog in sorted(set(y_train)):
    dog_embs = X_train_weighted[y_train == dog]
    profiles[dog] = dog_embs.mean(axis=0)

predictions = []

for emb in X_test_weighted:
    scores = {}

    for dog, profile in profiles.items():
        scores[dog] = cosine(emb, profile)

    pred = max(scores, key=scores.get)
    predictions.append(pred)

predictions = np.array(predictions)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("\n=== WEIGHTED COSINE ===")
print("Accuracy:", acc)
print("Macro F1:", f1)
print()
print(classification_report(y_test, predictions, zero_division=0))
