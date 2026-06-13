import numpy as np
from pathlib import Path

DATA_PATH = Path("datasets/keypoints")
OUTPUT_PATH = Path("datasets/keypoints_velocity")

def add_velocity_features(data):
    # data shape: (N, 30, 132)
    velocity = np.diff(data, axis=1)

    # pad first frame with zeros
    zero_pad = np.zeros((data.shape[0], 1, data.shape[2]))
    velocity = np.concatenate((zero_pad, velocity), axis=1)

    # concatenate original + velocity
    enhanced = np.concatenate((data, velocity), axis=2)
    return enhanced


def process_file(filename):
    print(f"Processing {filename}")
    data = np.load(DATA_PATH / filename)

    enhanced = add_velocity_features(data)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH / filename, enhanced)

    print("New shape:", enhanced.shape)


def main():
    process_file("violence.npy")
    process_file("non_violence.npy")


if __name__ == "__main__":
    main()
