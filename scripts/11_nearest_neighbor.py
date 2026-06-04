import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

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


predictions = []

for test_emb in X_test:
    best_score = -999
    best_label = None

    for train_emb, train_label in zip(X_train, y_train):
        score = cosine(test_emb, train_emb)

        if score > best_score:
            best_score = score
            best_label = train_label

    predictions.append(best_label)

predictions = np.array(predictions)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("=== NEAREST NEIGHBOR BARK RETRIEVAL ===")
print("Train clips:", len(X_train))
print("Test clips:", len(X_test))
print("Accuracy:", acc)
print("Macro F1:", f1)
