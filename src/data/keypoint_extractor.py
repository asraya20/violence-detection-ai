import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

FRAME_ROOT = Path("datasets/frames")
OUTPUT_ROOT = Path("datasets/keypoints")

SEQUENCE_LENGTH = 30

mp_pose = mp.solutions.pose


# Global pose object per worker
pose = None


def init_worker():
    global pose
    pose = mp_pose.Pose(static_image_mode=True)


def extract_keypoints(image_path):
    global pose

    try:
        image = cv2.imread(str(image_path))

        if image is None:
            return None

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(keypoints)
        else:
            return np.zeros(132)

    except Exception:
        return None


def process_class(label):
    images = sorted((FRAME_ROOT / label).glob("*.jpg"))

    print(f"\nProcessing {label} ({len(images)} images)")
    print(f"Using {cpu_count()} CPU cores")

    with Pool(cpu_count(), initializer=init_worker) as pool:
        keypoints_list = list(tqdm(
            pool.imap(extract_keypoints, images),
            total=len(images)
        ))

    sequences = []
    current_sequence = []
    skipped = 0

    for kp in keypoints_list:

        if kp is None:
            skipped += 1
            continue

        current_sequence.append(kp)

        if len(current_sequence) == SEQUENCE_LENGTH:
            sequences.append(current_sequence)
            current_sequence = []

    # Convert to numpy AFTER building all sequences
    sequences = np.array(sequences, dtype=np.float32)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_ROOT / f"{label}.npy", sequences)

    print(f"{label} sequences saved:", sequences.shape)
    print(f"Skipped corrupted frames: {skipped}")



def main():
    for label in ["violence", "non_violence"]:
        process_class(label)



if __name__ == "__main__":
    main()
