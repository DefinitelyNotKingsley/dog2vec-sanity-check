import pandas as pd
from pathlib import Path

MANIFEST = Path("manifests/manifest.csv")

df = pd.read_csv(MANIFEST)

print("\n=== Basic Counts ===")
print(f"Total clips: {len(df)}")
print(f"Dogs: {df['dog'].nunique()}")
print(df["split"].value_counts())

print("\n=== Missing Files ===")
missing = df[~df["file_path"].apply(lambda p: Path(p).exists())]
print(f"Missing files: {len(missing)}")
if len(missing) > 0:
    print(missing.head(20))

print("\n=== Videos Per Dog ===")
videos_per_dog = df.groupby("dog")["video"].nunique().sort_values()
print(videos_per_dog)

print("\n=== Clips Per Dog/Split ===")
clips = df.groupby(["dog", "split"]).size().unstack(fill_value=0)
print(clips)

print("\n=== Dogs Missing video1-video4 ===")
expected = {"video1", "video2", "video3", "video4"}

bad = []
for dog, group in df.groupby("dog"):
    videos = set(group["video"])
    missing_videos = expected - videos
    if missing_videos:
        bad.append((dog, sorted(missing_videos)))

if bad:
    for dog, missing_videos in bad:
        print(dog, "missing", missing_videos)
else:
    print("All dogs have video1-video4")

print("\n=== Summary ===")
print("Manifest sanity check complete.")
