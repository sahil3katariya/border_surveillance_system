import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_CACHE_DIR"] = "/tmp"


from ultralytics import YOLO
import cv2
import joblib
import pandas as pd


from src.utils import get_zone, get_zone_lines, get_time_of_day, calculate_speed



class VideoPipeline:

    def __init__(self):
        self.yolo = YOLO("yolov8n.pt")  # auto-download safe

        

        print("✅ Models loaded")
        
    def run(self, video_path):

        # ------------------ VIDEO ------------------
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception("❌ Cannot open video")

        selected_detection = None
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
            line1, line2 = get_zone_lines(frame_height)

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
                    time_of_day = get_time_of_day(frame)

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
                    selected_detection = best_in_frame
                    return selected_detection   # RETURN ONLY DETECTION
            
                selected_detection = best_in_frame
        cap.release()

            # -------- FINAL CASE --------
        if selected_detection:
            return selected_detection
        return None
    
































    # def run(self, video_path):

    #     # ------------------ VIDEO ------------------
    #     cap = cv2.VideoCapture(video_path)

    #     if not cap.isOpened():
    #         raise Exception("❌ Cannot open video")

    #     frame_count = 0
    #     object_tracks = {}

    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     frame_skip = 10
    #     time_diff = frame_skip / fps if fps > 0 else 0.1

    #     # ------------------ PROCESS VIDEO ------------------
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         frame_count += 1

    #         # Skip frames for performance
    #         if frame_count % frame_skip != 0:
    #             continue

    #         # -------- RESIZE --------
    #         frame = cv2.resize(frame, (600, 350))
    #         h, w = frame.shape[:2]

    #         # -------- ZONES --------
    #         line1, line2 = get_zone_lines(h)

    #         cv2.line(frame, (0, line1), (w, line1), (0, 255, 255), 3)
    #         cv2.line(frame, (0, line2), (w, line2), (0, 0, 255), 3)

    #         cv2.putText(frame, "Z1", (10, line1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    #         cv2.putText(frame, "Z2", (10, line2 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #         cv2.putText(frame, "Z3", (10, line2 + 30),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    #         # -------- YOLO TRACKING --------
    #         results = self.yolo.track(frame, persist=True)

    #         best_detection = None
    #         max_priority = -1

    #         for r in results:
    #             for box in r.boxes:

    #                 if box.id is None:
    #                     continue

    #                 track_id = int(box.id[0])
    #                 cls = int(box.cls[0])
    #                 label = self.yolo.names[cls]

    #                 # -------- CATEGORY --------
    #                 if label == "person":
    #                     mapped_label = "Human"
    #                 elif label in ["car", "truck", "bus"]:
    #                     mapped_label = "Vehicle"
    #                 else:
    #                     mapped_label = "Animal"

    #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                 cx = (x1 + x2) // 2
    #                 cy = y2

    #                 # -------- LOGIC --------
    #                 zone = get_zone(cy, line1, line2)
    #                 time_of_day = get_time_of_day()

    #                 speed = calculate_speed(
    #                     track_id,
    #                     cx,
    #                     cy,
    #                     y1,
    #                     y2,
    #                     mapped_label,
    #                     object_tracks,
    #                     time_diff
    #                 )

    #                 detection = {
    #                     "object": mapped_label,
    #                     "time": time_of_day,
    #                     "zone": zone,
    #                     "speed": speed
    #                 }

    #                 # -------- PRIORITY --------
    #                 priority_map = {"Z1": 1, "Z2": 2, "Z3": 3}
    #                 priority = priority_map[zone]

    #                 if priority > max_priority:
    #                     max_priority = priority
    #                     best_detection = detection

    #                 # -------- DRAW BOX --------
    #                 color = (0, 255, 0)
    #                 if zone == "Z2":
    #                     color = (0, 255, 255)
    #                 elif zone == "Z3":
    #                     color = (0, 0, 255)

    #                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    #                 label_text = f"{mapped_label} | {zone}"
    #                 cv2.putText(frame, label_text,
    #                             (x1, y1 - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX,
    #                             0.5, color, 2)

    #         # -------- RETURN FRAME + DETECTION --------
    #         yield frame, best_detection

    #     cap.release()