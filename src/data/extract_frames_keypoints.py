import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

FRAME_ROOT = Path("datasets/frames")
OUTPUT_ROOT = Path("datasets/frame_keypoints")

mp_pose = mp.solutions.pose


def init_worker():
    global pose
    pose = mp_pose.Pose(static_image_mode=True)


def process_image(image_path):
    global pose

    image = cv2.imread(str(image_path))
    if image is None:
        return None, None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return image_path.stem, np.array(keypoints)
    else:
        return image_path.stem, None


def process_class(label):

    images = sorted((FRAME_ROOT / label).glob("*.jpg"))
    print(f"\nProcessing {label} ({len(images)} images)")
    print(f"Using {cpu_count()} CPU cores")

    keypoint_dict = {}

    with Pool(cpu_count(), initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(process_image, images),
                            total=len(images)))

    for name, kp in results:
        if kp is not None:
            keypoint_dict[name] = kp

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_ROOT / f"{label}.npy", keypoint_dict)

    print(f"Saved {len(keypoint_dict)} frame keypoints")


def main():
    process_class("violence")
    process_class("non_violence")


if __name__ == "__main__":
    main()
