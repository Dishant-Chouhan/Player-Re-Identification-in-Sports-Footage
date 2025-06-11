import cv2
import numpy as np
from typing import List, Dict

class EnhancedVisualizer:
    def __init__(self):
        self.class_colors = {
            0: (0, 0, 255),   # Ball: Red
            1: (0, 255, 255), # Goalkeeper: Yellow
            2: (0, 255, 0),   # Player: Green
            3: (255, 0, 0)    # Referee: Blue
        }

    def draw(self, frame, tracks, detections, frame_count, total_frames) -> np.ndarray:
        annotated_frame = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["ltrb"])
            class_id = track["class"]
            color = self.class_colors.get(class_id, (255, 255, 255))
            label = f"{track['class_name']} {track['id']}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            if "conf" in track:
                conf_text = f"{track['conf']:.2f}"
                cv2.putText(annotated_frame, conf_text, (x1, y1 - h - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(tracks)}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return annotated_frame