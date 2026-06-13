import numpy as np
from pathlib import Path
from collections import defaultdict

FRAME_ROOT = Path("datasets/frames")
KEYPOINT_ROOT = Path("datasets/frame_keypoints")
OUTPUT_ROOT = Path("datasets/sequences_clean")

SEQUENCE_LENGTH = 30


def build_sequences(label):

    print(f"\nBuilding sequences for {label}")

    frame_files = sorted((FRAME_ROOT / label).glob("*.jpg"))
    frame_names = [f.stem for f in frame_files]

    keypoint_dict = np.load(KEYPOINT_ROOT / f"{label}.npy", allow_pickle=True).item()

    video_groups = defaultdict(list)

    for name in frame_names:
        video_id = "_".join(name.split("_")[:-1])
        video_groups[video_id].append(name)

    sequences = []

    for video_id, frames in video_groups.items():

        buffer = []

        for frame_name in frames:
            if frame_name not in keypoint_dict:
                continue

            buffer.append(keypoint_dict[frame_name])

            if len(buffer) == SEQUENCE_LENGTH:
                sequences.append(buffer.copy())
                buffer.pop(0)

    sequences = np.array(sequences, dtype=np.float32)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_ROOT / f"{label}.npy", sequences)

    print("Final shape:", sequences.shape)


def main():
    build_sequences("violence")
    build_sequences("non_violence")


if __name__ == "__main__":
    main()
