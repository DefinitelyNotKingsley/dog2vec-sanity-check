import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")
v_train = np.load(EMB_DIR / "train_videos.npy")

X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")
v_test = np.load(EMB_DIR / "test_videos.npy")

X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])
v = np.concatenate([v_train, v_test])

videos = sorted(set(v))

print("=== LEAVE-ONE-VIDEO-OUT DIRECT SVM ===")

accs = []
f1s = []

for heldout in videos:
    train_idx = v != heldout
    test_idx = v == heldout

    clf = LinearSVC(class_weight="balanced", max_iter=20000, random_state=0)
    clf.fit(X[train_idx], y[train_idx])

    pred = clf.predict(X[test_idx])

    acc = accuracy_score(y[test_idx], pred)
    f1 = f1_score(y[test_idx], pred, average="macro")

    accs.append(acc)
    f1s.append(f1)

    print(f"{heldout}: accuracy={acc:.4f}, macro_f1={f1:.4f}, test_clips={sum(test_idx)}")

print("\nAverage accuracy:", np.mean(accs))
print("Average macro F1:", np.mean(f1s))
