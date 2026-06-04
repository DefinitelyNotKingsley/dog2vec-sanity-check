import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")

X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

profiles = {}

for dog in sorted(set(y_train)):
    dog_embeddings = X_train[y_train == dog]
    profiles[dog] = dog_embeddings.mean(axis=0)

predictions = []

for emb in X_test:
    scores = {}

    for dog, profile in profiles.items():
        scores[dog] = cosine(emb, profile)

    pred = max(scores, key=scores.get)
    predictions.append(pred)

predictions = np.array(predictions)

print("=== RAW COSINE BASELINE ===")
print("Train clips:", len(X_train))
print("Test clips:", len(X_test))
print("Dogs in train:", len(set(y_train)))
print("Dogs in test:", len(set(y_test)))
print("Accuracy:", accuracy_score(y_test, predictions))
print("Macro F1:", f1_score(y_test, predictions, average="macro"))
print()
print(classification_report(y_test, predictions, zero_division=0))
