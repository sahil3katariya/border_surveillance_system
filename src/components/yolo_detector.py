import os
import sys

from src.logger import logging
from src.exception import CustomException

os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load model once
        self.model = YOLO(model_path)