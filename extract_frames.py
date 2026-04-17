import os
import argparse
from pathlib import Path
import cv2
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN

# Dataset folders
CATEGORIES = ["Celeb-real", "YouTube-real", "Celeb-synthesis"]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize MTCNN
mtcnn = MTCNN(
    image_size=299,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    device=device
)


def extract_two_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return 0

    frame_positions = [0, total_frames // 2]
    saved = 0

    for idx, pos in enumerate(frame_positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect and crop face
        face = mtcnn(frame_rgb)

        if face is None:
            continue

        # Convert tensor → numpy
        face = face.permute(1, 2, 0).cpu().numpy()

        # Convert float [0,1] → uint8 [0,255]
        face = (face * 255).astype("uint8")

        # Save
        out_path = os.path.join(output_dir, f"frame{idx+1:04d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

        saved += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="./celebdf")
    parser.add_argument("--frames_root", default="./frames")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    frames_root = Path(args.frames_root)

    print("\n── Extracting 2 Face Crops per Video (MTCNN) ─────────────")

    total_saved = 0

    for category in CATEGORIES:
        cat_dir = dataset_root / category
        if not cat_dir.exists():
            continue

        video_files = sorted(
            [v for v in cat_dir.iterdir() if v.suffix.lower() in (".mp4", ".avi")]
        )

        print(f"\n[{category}] {len(video_files)} videos")

        for video_path in tqdm(video_files, desc=category):
            video_stem = video_path.stem
            output_dir = frames_root / category / video_stem

            # Skip if already extracted
            if output_dir.exists() and len(list(output_dir.glob("*.jpg"))) >= 2:
                continue

            saved = extract_two_frames(str(video_path), str(output_dir))
            total_saved += saved

    print("\nDone.")
    print("Total face crops saved:", total_saved)
    print("Saved in:", frames_root)


if __name__ == "__main__":
    main()