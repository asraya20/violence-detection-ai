from ultralytics import YOLO
import torch


class MultiDetector:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8s.pt")
        self.model.to(self.device)
        self.model.fuse()

        if torch.cuda.is_available():
            self.model.model.half()

        # Classes we care about
        self.target_classes = [0, 34, 43]  # person, bat, knife

    def detect(self, frame):
        results = self.model(
            frame,
            classes=self.target_classes,
            conf=0.25,   # lower threshold slightly
            verbose=False
        )
        return results[0]
