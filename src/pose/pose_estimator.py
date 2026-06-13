import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        return results

    def draw(self, frame, results):
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame
