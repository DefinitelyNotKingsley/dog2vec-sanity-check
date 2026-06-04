import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")

X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")

clf = LinearSVC(class_weight="balanced", max_iter=20000, random_state=0)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("=== DIRECT SVM CONFUSION PAIRS ===")
print("Accuracy:", accuracy_score(y_test, pred))
print("Macro F1:", f1_score(y_test, pred, average="macro"))

confusions = Counter()

for true, guess in zip(y_test, pred):
    if true != guess:
        confusions[(true, guess)] += 1

print("\nTop confused pairs:")
print("true_dog,predicted_dog,count")

for (true, guess), count in confusions.most_common(30):
    print(f"{true},{guess},{count}")
