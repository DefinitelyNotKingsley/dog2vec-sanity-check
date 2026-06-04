import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.svm import LinearSVC

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")

X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")

clf = LinearSVC(class_weight="balanced", max_iter=20000, random_state=0)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

total = defaultdict(int)
correct = defaultdict(int)

for true, guess in zip(y_test, pred):
    total[true] += 1
    if true == guess:
        correct[true] += 1

rows = []

for dog in sorted(total):
    acc = correct[dog] / total[dog]
    rows.append((acc, dog, correct[dog], total[dog]))

rows.sort()

print("=== PER-DOG CROSS-VIDEO ACCURACY ===")
print("dog,correct,total,accuracy")

for acc, dog, c, t in rows:
    print(f"{dog},{c},{t},{acc:.4f}")

print("\nWorst 10 dogs:")
for acc, dog, c, t in rows[:10]:
    print(f"{dog}: {c}/{t} = {acc:.4f}")

print("\nBest 10 dogs:")
for acc, dog, c, t in rows[-10:][::-1]:
    print(f"{dog}: {c}/{t} = {acc:.4f}")
