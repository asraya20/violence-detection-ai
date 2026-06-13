import numpy as np
from filterpy.kalman import KalmanFilter


class Track:
    count = 0

    def __init__(self, bbox):
        self.id = Track.count
        Track.count += 1

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.eye(4, 7)

        self.kf.x[:4] = np.array(bbox).reshape((4, 1))
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))

    def update(self, bbox):
        self.kf.update(np.array(bbox))
        self.time_since_update = 0

    def get_state(self):
        return self.kf.x[:4].reshape((4,))


class SortTracker:
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        updated_tracks = []

        for det in detections:
            matched = False

            for track in self.tracks:
                pred = track.predict()
                if self.iou(pred, det) > 0.3:
                    track.update(det)
                    updated_tracks.append(track)
                    matched = True
                    break

            if not matched:
                updated_tracks.append(Track(det))

        self.tracks = updated_tracks
        return [(track.get_state(), track.id) for track in self.tracks]

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
