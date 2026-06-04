import csv
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

DOG2VEC_DIR = Path("/Users/kingsley/external/dog2vec")
MODEL_PATH = DOG2VEC_DIR / "checkpoints/dog2vec_130k_9.pt"
MANIFEST_PATH = Path("manifests/manifest.csv")
OUTPUT_DIR = Path("embeddings")

sys.path.append(str(DOG2VEC_DIR))
from extract_feature import FeatureExtractor


def load_wav(path):
    audio, sample_rate = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio = torch.tensor(audio, dtype=torch.float32)
    return audio, sample_rate


def read_manifest():
    rows = []
    with MANIFEST_PATH.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_rows(rows, split):
    embeddings = []
    labels = []
    files = []
    videos = []

    split_rows = [r for r in rows if r["split"] == split]

    print(f"\nExtracting {split}: {len(split_rows)} clips")

    for i, row in enumerate(split_rows, start=1):
        wav_path = Path(row["file_path"])
        dog = row["dog"]
        video = row["video"]

        print(f"[{i}/{len(split_rows)}] {split} | {dog} | {video} | {wav_path.name}")

        audio, sample_rate = load_wav(wav_path)

        if sample_rate != 16000:
            print(f"WARNING: {wav_path} sample rate is {sample_rate}, expected 16000")

        with torch.no_grad():
            features = extractor.extract(audio)

        embedding = features.mean(dim=0).cpu().numpy()

        embeddings.append(embedding)
        labels.append(dog)
        files.append(str(wav_path))
        videos.append(video)

    return (
        np.array(embeddings),
        np.array(labels),
        np.array(files),
        np.array(videos),
    )


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Model:", MODEL_PATH)

    extractor = FeatureExtractor(
        model_path=str(MODEL_PATH),
        device=device,
        layer=9,
    )

    rows = read_manifest()

    train_embeddings, train_labels, train_files, train_videos = extract_rows(rows, "train")
    test_embeddings, test_labels, test_files, test_videos = extract_rows(rows, "test")

    np.save(OUTPUT_DIR / "train_embeddings.npy", train_embeddings)
    np.save(OUTPUT_DIR / "train_labels.npy", train_labels)
    np.save(OUTPUT_DIR / "train_files.npy", train_files)
    np.save(OUTPUT_DIR / "train_videos.npy", train_videos)

    np.save(OUTPUT_DIR / "test_embeddings.npy", test_embeddings)
    np.save(OUTPUT_DIR / "test_labels.npy", test_labels)
    np.save(OUTPUT_DIR / "test_files.npy", test_files)
    np.save(OUTPUT_DIR / "test_videos.npy", test_videos)

    print("\nDone.")
    print("Train embeddings shape:", train_embeddings.shape)
    print("Test embeddings shape:", test_embeddings.shape)
