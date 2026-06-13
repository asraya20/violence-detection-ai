import cv2
import time

def save_clip(frames, fps=30):

    if len(frames) == 0:
        return None

    height, width, _ = frames[0].shape

    filename = f"evidence/incident_{int(time.time())}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    return filename