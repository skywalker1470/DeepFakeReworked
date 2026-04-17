import os
import argparse
import random
from pathlib import Path

REAL_DIRS = ["Celeb-real", "YouTube-real"]
FAKE_DIRS = ["Celeb-synthesis"]

LABEL_MAP = {d: 0 for d in REAL_DIRS}
LABEL_MAP.update({d: 1 for d in FAKE_DIRS})

# -----------------------------
# CONFIG (IMPORTANT)
# -----------------------------
MAX_VIDEOS_PER_CLASS = 1500   # controls dataset size


def get_test_video_names(test_file):
    stems = set()
    with open(test_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                video_path = parts[-1]
                stems.add(Path(video_path).stem)
    return stems


def collect_frames(frames_root):
    records = []
    frames_root = Path(frames_root)

    for category, label in LABEL_MAP.items():
        cat_dir = frames_root / category
        if not cat_dir.exists():
            continue

        video_dirs = [d for d in cat_dir.iterdir() if d.is_dir()]

        # limit dataset size per class
        video_dirs = random.sample(
            video_dirs,
            min(MAX_VIDEOS_PER_CLASS, len(video_dirs))
        )

        print(f"{category}: using {len(video_dirs)} videos")

        for video_dir in video_dirs:

            # 🔥 IMPORTANT: take ONLY 3 frames
            frames = sorted(video_dir.glob("*.jpg"))[:3]

            for frame_file in frames:
                records.append(
                    (str(frame_file), label, video_dir.name)
                )

    random.shuffle(records)
    return records


def split_by_video(records, test_stems, val_split=0.1, seed=42):
    test = []
    trainval = []

    for rec in records:
        if rec[2] in test_stems:
            test.append(rec)
        else:
            trainval.append(rec)

    video_dict = {}
    for rec in trainval:
        video_dict.setdefault(rec[2], []).append(rec)

    video_stems = list(video_dict.keys())
    random.seed(seed)
    random.shuffle(video_stems)

    n_val = int(len(video_stems) * val_split)
    val_stems = set(video_stems[:n_val])
    train_stems = set(video_stems[n_val:])

    train = [r for s in train_stems for r in video_dict[s]]
    val = [r for s in val_stems for r in video_dict[s]]

    return train, val, test


def write_list(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for frame_path, label, _ in records:
            f.write(f"{frame_path} {label}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./celebdf")
    parser.add_argument("--frames_root", default="./frames")
    parser.add_argument("--output_dir", default="./data_list")
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    test_file = os.path.join(
        args.dataset_root,
        "List_of_testing_videos.txt"
    )

    print("Preparing dataset lists...")

    test_stems = get_test_video_names(test_file)
    records = collect_frames(args.frames_root)

    if not records:
        print("No frames found. Run extract_frames.py first.")
        return

    print(f"Total samples: {len(records)}")

    train, val, test = split_by_video(records, test_stems, args.val_split)

    write_list(train, os.path.join(args.output_dir, "celebdf_train.txt"))
    write_list(val, os.path.join(args.output_dir, "celebdf_val.txt"))
    write_list(test, os.path.join(args.output_dir, "celebdf_test.txt"))

    print("\nDone.")
    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))


if __name__ == "__main__":
    main()