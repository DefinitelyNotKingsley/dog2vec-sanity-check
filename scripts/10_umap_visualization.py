import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import umap

EMB_DIR = Path("embeddings")

X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")
v_train = np.load(EMB_DIR / "train_videos.npy")

X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")
v_test = np.load(EMB_DIR / "test_videos.npy")

X = np.concatenate([X_train, X_test])
dogs = np.concatenate([y_train, y_test])
videos = np.concatenate([v_train, v_test])

print("Running UMAP...")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=0
)

embedding_2d = reducer.fit_transform(X)

# -------------------------
# Color by VIDEO
# -------------------------

video_names = sorted(set(videos))
video_to_idx = {v: i for i, v in enumerate(video_names)}

plt.figure(figsize=(10, 8))

for video in video_names:
    idx = videos == video
    plt.scatter(
        embedding_2d[idx, 0],
        embedding_2d[idx, 1],
        s=8,
        alpha=0.6,
        label=video
    )

plt.legend()
plt.title("Dog2Vec UMAP - Colored by Video")
plt.tight_layout()
plt.savefig("results/umap_by_video.png", dpi=300)
plt.close()

# -------------------------
# Color by DOG
# -------------------------

plt.figure(figsize=(12, 10))

unique_dogs = sorted(set(dogs))

for dog in unique_dogs:
    idx = dogs == dog
    plt.scatter(
        embedding_2d[idx, 0],
        embedding_2d[idx, 1],
        s=8,
        alpha=0.6
    )

plt.title("Dog2Vec UMAP - Colored by Dog")
plt.tight_layout()
plt.savefig("results/umap_by_dog.png", dpi=300)
plt.close()

print("Saved:")
print("results/umap_by_video.png")
print("results/umap_by_dog.png")
