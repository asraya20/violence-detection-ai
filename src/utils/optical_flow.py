import cv2
import numpy as np


class OpticalFlowAnalyzer:
    def __init__(self, resize_dim=(320, 240)):
        self.prev_gray = None
        self.resize_dim = resize_dim

    def compute_motion(self, frame):

        # Downscale frame for faster processing
        small_frame = cv2.resize(frame, self.resize_dim)

        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            2,      # reduce pyramid levels
            10,     # smaller window
            2,      # fewer iterations
            5,
            1.1,
            0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)

        self.prev_gray = gray

        return motion_score
