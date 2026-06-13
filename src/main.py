import cv2
import time
import torch
import numpy as np
from threading import Thread
from collections import deque
from datetime import datetime

from src.detection.person_detector import MultiDetector
from src.pose.pose_estimator import PoseEstimator
from src.lstm.inference import load_model, predict_sequence
from src.utils.optical_flow import OpticalFlowAnalyzer

from src.alerts.telegram_alert import send_telegram_alert
from src.utils.clip_generator import save_clip


# -----------------------------
# Webcam Thread
# -----------------------------
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


# -----------------------------
# Main Pipeline
# -----------------------------
def main():

    torch.backends.cudnn.benchmark = True

    detector = MultiDetector()
    pose = PoseEstimator()
    model = load_model()
    stream = WebcamStream()
    sequence_buffer = deque(maxlen=30)
    prediction_buffer = deque(maxlen=5)
    weapon_buffer = deque(maxlen=5)

    frame_buffer = deque(maxlen=300)

    recording_event = False
    post_event_frames = []
    pre_event_frames = []
    POST_EVENT_LENGTH = 300

    flow_analyzer = OpticalFlowAnalyzer()

    last_weapon_bbox = None
    weapon_miss_count = 0
    last_results = None
    last_pose_results = {}

    prev_time = time.time()
    frame_count = 0
    threshold = 0.7

    LOW_MOTION_THRESHOLD = 1.2
    HIGH_MOTION_THRESHOLD = 3.0

    last_alert_time = 0
    ALERT_COOLDOWN = 20

    VIOLENCE_HOLD_FRAMES = 15
    violence_counter = 0

    while True:

        frame = stream.read()
        if frame is None:
            continue

        frame_buffer.append(frame.copy())

        if recording_event:
            post_event_frames.append(frame.copy())

        original_h, original_w = frame.shape[:2]
        frame_count += 1

        motion_score = flow_analyzer.compute_motion(frame)

        small_frame = cv2.resize(frame, (640, 640))

        if frame_count % 2 == 0:
            results = detector.detect(small_frame)
            last_results = results
        else:
            results = last_results

        scale_x = original_w / 640
        scale_y = original_h / 640

        violence_prob = 0.0
        person_count = 0
        person_centers = []

        # ---------------- YOLO ----------------
        if results is not None:

            for i, box in enumerate(results.boxes):

                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_w, x2)
                y2 = min(original_h, y2)

                # ---------------- PERSON ----------------
                if cls == 0:

                    person_count += 1

                    pad_x = int((x2 - x1) * 0.3)
                    pad_y = int((y2 - y1) * 0.3)

                    x1_exp = max(0, x1 - pad_x)
                    y1_exp = max(0, y1 - pad_y)
                    x2_exp = min(original_w, x2 + pad_x)
                    y2_exp = min(original_h, y2 + pad_y)

                    person_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]

                    if person_crop.size == 0:
                        continue

                    person_crop_resized = cv2.resize(person_crop, (256, 256))

                    # store center
                    center_x = (x1_exp + x2_exp) // 2
                    center_y = (y1_exp + y2_exp) // 2
                    person_centers.append((center_x, center_y))

                    if frame_count % 3 == 0:
                        pose_results = pose.process(person_crop_resized)
                        last_pose_results[i] = pose_results
                    else:
                        pose_results = last_pose_results.get(i, None)

                    if pose_results and pose_results.pose_landmarks:

                        person_crop_drawn = pose.draw(
                            person_crop_resized.copy(), pose_results
                        )

                        person_crop_drawn = cv2.resize(
                            person_crop_drawn,
                            (x2_exp - x1_exp, y2_exp - y1_exp)
                        )

                        frame[y1_exp:y2_exp, x1_exp:x2_exp] = person_crop_drawn

                        keypoints = []
                        for lm in pose_results.pose_landmarks.landmark:
                            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

                        sequence_buffer.append(keypoints)

                        if len(sequence_buffer) == 30:

                            prob = predict_sequence(
                                model,
                                np.array(sequence_buffer)
                            )

                            prediction_buffer.append(prob)

                    cv2.rectangle(frame,
                                  (x1_exp, y1_exp),
                                  (x2_exp, y2_exp),
                                  (0, 255, 0),
                                  2)

                # ---------------- WEAPON ----------------
                elif cls in [34, 43]:

                    weapon_buffer.append(1)
                    last_weapon_bbox = (x1, y1, x2, y2)
                    weapon_miss_count = 0

        # ---------------- Interaction Detection ----------------
        interaction_detected = False

        if len(person_centers) >= 2:

            for i in range(len(person_centers)):
                for j in range(i + 1, len(person_centers)):

                    p1 = person_centers[i]
                    p2 = person_centers[j]

                    dist = np.linalg.norm(np.array(p1) - np.array(p2))

                    if dist < 120 and motion_score > HIGH_MOTION_THRESHOLD:
                        interaction_detected = True

        # ---------------- Weapon Smoothing ----------------
        weapon_detected = sum(weapon_buffer) >= 3 if len(weapon_buffer) > 0 else False

        # ---------------- Scene-Level Violence ----------------
        if len(prediction_buffer) > 0:
            violence_prob = float(np.median(prediction_buffer))

        if violence_prob > 0:

            if motion_score < LOW_MOTION_THRESHOLD:
                violence_prob *= 0.6

            elif motion_score > HIGH_MOTION_THRESHOLD:
                violence_prob *= 1.15

            violence_prob = min(violence_prob, 1.0)

        # ---------------- Context Aware Decision ----------------
# ---------------- STABLE VIOLENCE ----------------
        if violence_prob > threshold:
            violence_counter = VIOLENCE_HOLD_FRAMES
        else:
            violence_counter = max(0, violence_counter - 1)

        is_violent = violence_counter > 0

        # ---------------- Decision ----------------
        if is_violent:

            if weapon_detected:
                label = f"CRITICAL {violence_prob:.2f}"
                color = (0, 0, 255)

            elif interaction_detected:
                label = f"FIGHT {violence_prob:.2f}"
                color = (0, 140, 255)

            else:
                label = f"VIOLENCE {violence_prob:.2f}"
                color = (0, 165, 255)

            current_time = time.time()

            if not recording_event and current_time - last_alert_time > ALERT_COOLDOWN:

                print("⚠ Violence detected! Starting event recording...")

                recording_event = True
                post_event_frames = []
                pre_event_frames = list(frame_buffer)

                # 🔥 TIMESTAMP
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                send_telegram_alert(
                    f"⚠ Violence detected\n"
                    f"Time: {timestamp}\n"
                    f"Confidence: {violence_prob:.2f}"
                )

                last_alert_time = current_time

        elif weapon_detected and motion_score < LOW_MOTION_THRESHOLD:

            label = "SUSPICIOUS WEAPON"
            color = (255, 0, 0)

        else:

            label = f"NORMAL {violence_prob:.2f}"
            color = (0, 255, 0)

        # ---------------- Save Evidence ----------------
        if recording_event and len(post_event_frames) >= POST_EVENT_LENGTH:

            print("Saving full evidence clip...")

            full_clip = pre_event_frames + post_event_frames

            clip_path = save_clip(full_clip)

            print("Evidence saved:", clip_path)

            recording_event = False
            post_event_frames = []

        # ---------------- Display ----------------
        cv2.putText(frame, label, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"Motion: {motion_score:.2f}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        cv2.putText(frame, f"People: {person_count}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Interaction: {interaction_detected}",
                    (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 100, 255), 2)

        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        
        timestamp_display = datetime.now().strftime("%H:%M:%S")

        cv2.putText(frame, f"Time: {timestamp_display}",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        cv2.imshow("Violence Detection - Optical Flow Enhanced", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()