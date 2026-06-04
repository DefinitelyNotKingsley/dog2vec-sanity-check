import numpy as np
from pathlib import Path
from collections import defaultdict

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


# Build average profile per dog
profiles = {}
for dog in sorted(set(y_train)):
    profiles[dog] = X_train[y_train == dog].mean(axis=0)

dogs = list(profiles.keys())

top1 = 0
top3 = 0
top5 = 0

per_dog_total = defaultdict(int)
per_dog_top1 = defaultdict(int)
per_dog_top3 = defaultdict(int)
per_dog_top5 = defaultdict(int)

for emb, true_dog in zip(X_test, y_test):
    scores = []

    for dog in dogs:
        score = cosine(emb, profiles[dog])
        scores.append((dog, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_dogs = [dog for dog, score in scores]

    per_dog_total[true_dog] += 1

    if true_dog == ranked_dogs[0]:
        top1 += 1
        per_dog_top1[true_dog] += 1

    if true_dog in ranked_dogs[:3]:
        top3 += 1
        per_dog_top3[true_dog] += 1

    if true_dog in ranked_dogs[:5]:
        top5 += 1
        per_dog_top5[true_dog] += 1

n = len(y_test)

print("=== RAW COSINE TOP-K RETRIEVAL ===")
print(f"Test clips: {n}")
print(f"Dogs in profiles: {len(dogs)}")
print()
print(f"Top-1 Accuracy: {top1 / n:.4f}")
print(f"Top-3 Accuracy: {top3 / n:.4f}")
print(f"Top-5 Accuracy: {top5 / n:.4f}")

print("\n=== PER-DOG TOP-K ===")
print("dog,total,top1,top3,top5")

for dog in sorted(per_dog_total.keys()):
    total = per_dog_total[dog]
    d_top1 = per_dog_top1[dog] / total
    d_top3 = per_dog_top3[dog] / total
    d_top5 = per_dog_top5[dog] / total

    print(f"{dog},{total},{d_top1:.4f},{d_top3:.4f},{d_top5:.4f}")
