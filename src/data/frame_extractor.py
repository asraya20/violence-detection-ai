import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import uuid

RAW_ROOT = Path("datasets/raw_videos")
FRAME_ROOT = Path("datasets/frames")

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]

FRAME_SKIP = 4          # Adjust for dataset size
IMG_SIZE = 640          # YOLO optimized
MAX_FRAMES_PER_VIDEO = 150   # Prevent explosion


def get_all_videos():
    videos = []

    for dataset in RAW_ROOT.iterdir():
        if not dataset.is_dir():
            continue

        for class_folder in dataset.iterdir():
            if not class_folder.is_dir():
                continue

            if class_folder.name.lower() not in ["violence", "nonviolence"]:
                continue

            for video in class_folder.iterdir():
                if video.suffix.lower() in VIDEO_EXTENSIONS:
                    videos.append((video, class_folder.name.lower(), dataset.name))

    return videos


def process_video(args):
    video_path, label, dataset_name = args

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    saved = 0

    output_label = "violence" if label == "violence" else "non_violence"
    output_dir = FRAME_ROOT / output_label
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_prefix = f"{dataset_name}_{video_path.stem}_{uuid.uuid4().hex[:6]}"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            if saved >= MAX_FRAMES_PER_VIDEO:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            filename = f"{unique_prefix}_{saved}.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
            saved += 1

        frame_count += 1

    cap.release()
    return saved


def main():
    videos = get_all_videos()

    print(f"\nFound {len(videos)} videos")
    print(f"Using {cpu_count()} CPU cores\n")

    total_frames = 0

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_video, videos), total=len(videos)))

    total_frames = sum(results)

    print("\n========== EXTRACTION COMPLETE ==========")
    print(f"Total frames extracted: {total_frames}")
    print("Saved to: datasets/frames/")


if __name__ == "__main__":
    main()
