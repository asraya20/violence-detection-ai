import numpy as np
from pathlib import Path
from collections import defaultdict

FRAME_ROOT = Path("datasets/frames")
KEYPOINT_ROOT = Path("datasets/keypoints")
OUTPUT_ROOT = Path("datasets/sequences_clean")

SEQUENCE_LENGTH = 30


def group_frames_by_video(label):
    files = sorted((FRAME_ROOT / label).glob("*.jpg"))

    video_groups = defaultdict(list)

    for f in files:
        name = f.stem
        video_id = "_".join(name.split("_")[:-1])
        video_groups[video_id].append(name)

    return video_groups


def rebuild_sequences(label):

    print(f"\nRebuilding {label}")

    # Load existing keypoints (flat order)
    keypoints = np.load(KEYPOINT_ROOT / f"{label}.npy")

    # We must reload frame names in same order used originally
    frame_files = sorted((FRAME_ROOT / label).glob("*.jpg"))
    frame_names = [f.stem for f in frame_files]

    # Map frame name to keypoint row
    frame_to_kp = dict(zip(frame_names, keypoints))

    video_groups = group_frames_by_video(label)

    sequences = []

    for video_id, frames in video_groups.items():

        buffer = []

        for frame_name in frames:

            if frame_name not in frame_to_kp:
                continue

            buffer.append(frame_to_kp[frame_name])

            if len(buffer) == SEQUENCE_LENGTH:
                sequences.append(buffer.copy())
                buffer.pop(0)

    sequences = np.array(sequences, dtype=np.float32)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_ROOT / f"{label}.npy", sequences)

    print("New shape:", sequences.shape)


def main():
    rebuild_sequences("violence")
    rebuild_sequences("non_violence")


if __name__ == "__main__":
    main()
