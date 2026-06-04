from pathlib import Path
import csv

DATA_ROOT = Path("/Users/kingsley/Downloads/data/raw")
OUT_PATH = Path("manifests/manifest.csv")

TRAIN_VIDEOS = {"video1", "video2", "video3"}
TEST_VIDEOS = {"video4"}

rows = []

for dog_dir in sorted(DATA_ROOT.iterdir()):
    if not dog_dir.is_dir():
        continue

    dog = dog_dir.name

    for video_dir in sorted(dog_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        video = video_dir.name

        if video in TRAIN_VIDEOS:
            split = "train"
        elif video in TEST_VIDEOS:
            split = "test"
        else:
            print(f"Skipping unknown video folder: {video_dir}")
            continue

        for wav_path in sorted(video_dir.glob("*.wav")):
            rows.append({
                "file_path": str(wav_path),
                "dog": dog,
                "video": video,
                "split": split,
            })

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUT_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file_path", "dog", "video", "split"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved manifest to {OUT_PATH}")
print(f"Total clips: {len(rows)}")
print(f"Train clips: {sum(r['split'] == 'train' for r in rows)}")
print(f"Test clips: {sum(r['split'] == 'test' for r in rows)}")
print(f"Dogs: {len(set(r['dog'] for r in rows))}")
