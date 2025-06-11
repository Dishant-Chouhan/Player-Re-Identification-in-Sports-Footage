import os
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass, field
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/tracking_log.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/tracking_log.txt")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.handlers = [handler]
logger.debug("Logging initialized in tracking.py")

@dataclass
class ObjectFeatures:
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    size: float
    appearance_emb: Optional[np.ndarray] = None
    confidence: float = 0.0
    frame_id: int = 0
    class_id: int = 0
    class_name: str = ""

class AppearanceExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Smaller for ball
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Initialized ResNet18 for appearance embeddings")

    def extract(self, image: np.ndarray) -> Optional[np.ndarray]:
        if image.size == 0 or image.shape[0] < 5 or image.shape[1] < 5:
            return None
        try:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_t = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model(img_t).cpu().numpy().flatten()
            return emb / np.linalg.norm(emb + 1e-8)
        except Exception as e:
            logger.warning(f"Failed to extract embedding: {e}")
            return None

class EnhancedTracker:
    def __init__(self, max_disappeared_frames: int = 75, 
                 similarity_threshold: float = 0.35, 
                 max_distance_threshold: float = 100.0):
        logger.debug("Initializing EnhancedTracker")
        self.max_disappeared_frames = max_disappeared_frames
        self.similarity_threshold = similarity_threshold
        self.max_distance_threshold = max_distance_threshold
        self.objects: Dict[int, ObjectFeatures] = {}
        self.disappeared_counts: Dict[int, int] = defaultdict(int)
        self.next_id = 1
        self.object_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self.appearance_extractor = AppearanceExtractor()
        self.frame_count = 0

    def extract_features(self, frame: np.ndarray, det: Dict) -> ObjectFeatures:
        x1, y1, x2, y2 = det['bbox']
        obj_img = frame[max(0, y1):y2, max(0, x1):x2]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        size = (x2 - x1) * (y2 - y1)
        emb = self.appearance_extractor.extract(obj_img)
        return ObjectFeatures(
            bbox=det['bbox'],
            center=center,
            size=size,
            appearance_emb=emb,
            confidence=det['confidence'],
            frame_id=self.frame_count,
            class_id=det['class'],
            class_name=det['class_name']
        )

    def calculate_similarity(self, features1: ObjectFeatures, features2: ObjectFeatures) -> float:
        if features1.class_id != features2.class_id:
            return 0.0
        spatial_dist = np.sqrt(
            (features1.center[0] - features2.center[0]) ** 2 +
            (features1.center[1] - features2.center[1]) ** 2
        )
        spatial_sim = 1.0 / (1.0 + spatial_dist / self.max_distance_threshold)
        size_ratio = min(features1.size, features2.size) / max(features1.size, features2.size)
        size_sim = size_ratio if size_ratio > 0.5 else 0.0
        if features1.appearance_emb is not None and features2.appearance_emb is not None:
            appearance_sim = cosine_similarity(
                features1.appearance_emb.reshape(1, -1),
                features2.appearance_emb.reshape(1, -1)
            )[0, 0]
            weights = [0.3, 0.5, 0.2] if features1.class_id == 0 else [0.4, 0.4, 0.2]
        else:
            appearance_sim = 0.5
            weights = [0.6, 0.2, 0.2]
        total_sim = weights[0] * spatial_sim + weights[1] * appearance_sim + weights[2] * size_sim
        logger.debug(f"Similarity: {total_sim:.3f}, spatial: {spatial_sim:.3f}, appearance: {appearance_sim:.3f}, size: {size_sim:.3f}")
        return max(0.0, min(1.0, total_sim))

    def update(self, frame: np.ndarray, detections: List[Dict], frame_count: int) -> List[Dict]:
        self.frame_count = frame_count
        current_features = [self.extract_features(frame, det) for det in detections]
        matched_objects = {}
        used_detections = set()
        for obj_id, old_features in self.objects.items():
            best_match_idx = -1
            best_similarity = 0.0
            for i, new_features in enumerate(current_features):
                if i in used_detections:
                    continue
                similarity = self.calculate_similarity(old_features, new_features)
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_match_idx = i
            if best_match_idx >= 0:
                matched_objects[obj_id] = current_features[best_match_idx]
                used_detections.add(best_match_idx)
                self.disappeared_counts[obj_id] = 0
                self.object_history[obj_id].append(current_features[best_match_idx])
                logger.info(f"Frame {frame_count}: Re-identified {old_features.class_name} ID {obj_id}, similarity: {best_similarity:.3f}")
            else:
                self.disappeared_counts[obj_id] += 1
                logger.debug(f"Frame {frame_count}: {old_features.class_name} ID {obj_id} not matched")
        for i, features in enumerate(current_features):
            if i not in used_detections:
                obj_id = self.next_id
                self.next_id += 1
                matched_objects[obj_id] = features
                self.disappeared_counts[obj_id] = 0
                self.object_history[obj_id].append(features)
                logger.info(f"Frame {frame_count}: New {features.class_name} ID {obj_id}")
        objects_to_remove = [
            obj_id for obj_id, count in self.disappeared_counts.items()
            if count > self.max_disappeared_frames
        ]
        for obj_id in objects_to_remove:
            obj_name = self.objects[obj_id].class_name
            self.disappeared_counts.pop(obj_id, None)
            self.object_history.pop(obj_id, None)
            self.objects.pop(obj_id, None)
            logger.info(f"Frame {frame_count}: Removed {obj_name} ID {obj_id}")
        self.objects = matched_objects
        tracks = [
            {
                "id": obj_id,
                "ltrb": features.bbox,
                "conf": features.confidence,
                "class": features.class_id,
                "class_name": features.class_name
            }
            for obj_id, features in self.objects.items()
        ]
        logger.info(f"Frame {frame_count}: {len(tracks)} tracks, IDs: {[t['id'] for t in tracks]}, Total IDs: {self.next_id - 1}")
        handler.flush()
        return tracks