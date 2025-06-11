from ultralytics import YOLO
import numpy as np
import logging
import os

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/detection_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/detection_log.txt")
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.handlers = [handler]

class Detector:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.5):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_conf_thres = {0: 0.2, 1: 0.25, 2: 0.25, 3: 0.2}  # Ball, Goalkeeper, Player, Referee
        logger.info(f"Model classes: {self.model.names}")

    def detect(self, frame, frame_count):
        results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        detections = []
        all_detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for box, score, cls in zip(boxes, scores, classes):
                    cls = int(cls)
                    if score >= self.class_conf_thres.get(cls, self.conf_thres):
                        all_detections.append((cls, score))
                        x1, y1, x2, y2 = map(int, box)
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(score),
                            'class': cls,
                            'class_name': self.model.names[cls]
                        })
        logger.info(f"Frame {frame_count}: {len(detections)} detections, classes: {[d['class'] for d in detections]}")
        handler.flush()
        return detections