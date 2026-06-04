import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")
v_train = np.load(EMB_DIR / "train_videos.npy")

X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")
v_test = np.load(EMB_DIR / "test_videos.npy")

X = np.concatenate([X_train, X_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)
v = np.concatenate([v_train, v_test], axis=0)

for video in sorted(set(v)):
    idx = np.where(v == video)[0]
    X_video = X[idx]
    y_video = y[idx]

    counts = {dog: sum(y_video == dog) for dog in set(y_video)}
    usable_dogs = [dog for dog, count in counts.items() if count >= 2]
    keep = np.array([label in usable_dogs for label in y_video])

    X_video = X_video[keep]
    y_video = y_video[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_video,
        y_video,
        test_size=0.25,
        random_state=0,
        stratify=y_video
    )

    clf = LinearSVC(
        class_weight="balanced",
        max_iter=20000,
        random_state=0
    )

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    print(f"\n=== SAME-VIDEO SPLIT: {video} ===")
    print("Dogs:", len(set(y_video)))
    print("Train clips:", len(X_tr))
    print("Test clips:", len(X_te))
    print("Accuracy:", accuracy_score(y_te, pred))
    print("Macro F1:", f1_score(y_te, pred, average="macro"))
    print(classification_report(y_te, pred, zero_division=0))
