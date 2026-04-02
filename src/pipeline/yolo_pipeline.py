import os
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO
import cv2
import joblib
import pandas as pd

from src.utils import get_zone, get_time_of_day, calculate_speed


class VideoPipeline:

    def __init__(self):
        # ------------------ LOAD MODELS ------------------
        self.yolo = YOLO("yolov8n.pt")
        self.ml_model = joblib.load("model.pkl")
        self.label_encoder = joblib.load("label_encoder.pkl")
        self.preprocessor = joblib.load("preprocessor.pkl")

        print("✅ Models loaded")

    def run(self, video_path):

        # ------------------ VIDEO ------------------
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception("❌ Cannot open video")

        selected_detection = None
        z3_detected = False
        frame_count = 0
        object_tracks = {}

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = 10
        time_diff = frame_skip / fps

        # ------------------ PROCESS VIDEO ------------------
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            frame_height = frame.shape[0]

            # Zones
            line1 = int(frame_height * 0.4)
            line2 = int(frame_height * 0.75)

            # YOLO tracking
            results = self.yolo.track(frame, persist=True)

            best_in_frame = None
            max_priority = -1

            for r in results:
                for box in r.boxes:

                    if box.id is None:
                        continue

                    track_id = int(box.id[0])
                    cls = int(box.cls[0])
                    label = self.yolo.names[cls]

                    # CATEGORY
                    if label == "person":
                        mapped_label = "Human"
                    elif label in ["car", "truck", "bus"]:
                        mapped_label = "Vehicle"
                    else:
                        mapped_label = "Animal"

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = y2

                    # -------- USE FUNCTIONS --------
                    zone = get_zone(cy, line1, line2)
                    time_of_day = get_time_of_day()

                    speed = calculate_speed(
                        track_id,
                        cx,
                        cy,
                        y1,
                        y2,
                        mapped_label,
                        object_tracks,
                        time_diff
                    )

                    detection = {
                        "object": mapped_label,
                        "time": time_of_day,
                        "zone": zone,
                        "speed": speed
                    }

                    print(detection)

                    # PRIORITY
                    priority_map = {"Z1": 1, "Z2": 2, "Z3": 3}
                    priority = priority_map[zone]

                    if priority > max_priority:
                        max_priority = priority
                        best_in_frame = detection

            # -------- EVENT LOGIC --------
            if best_in_frame:

                if best_in_frame["zone"] == "Z3":
                    z3_detected = True
                    selected_detection = best_in_frame

                    print("\n🚨 Z3 DETECTED → Sending to ML")

                    df = pd.DataFrame([selected_detection])
                    x = self.preprocessor.transform(df)

                    pred = self.ml_model.predict(x)
                    pred = pred.astype(int)

                    output = self.label_encoder.inverse_transform(pred)

                    print("🎯 Prediction:", output)
                    break

                selected_detection = best_in_frame

        cap.release()

        # -------- FINAL CASE --------
        if not z3_detected and selected_detection:

            print("\n✅ No Z3 → Using last frame")

            df = pd.DataFrame([selected_detection])
            x = self.preprocessor.transform(df)

            pred = self.ml_model.predict(x)
            pred = pred.astype(int)

            output = self.label_encoder.inverse_transform(pred)

            print("🎯 Prediction:", output)