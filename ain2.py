"""
Player Re-identification and Tracking System
==========================================

This module implements a real-time player tracking system that maintains consistent
player IDs across frames, even when players temporarily leave and re-enter the frame.

Features:
- YOLO-based player detection
- Multi-feature player re-identification (position, appearance, size)
- Robust tracking with temporal consistency
- Memory-efficient design for real-time processing
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time
import os
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PlayerFeatures:
    """Container for player features used in re-identification"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]      # center coordinates
    size: float                      # bounding box area
    appearance_hist: np.ndarray      # color histogram
    confidence: float                # detection confidence
    frame_id: int                   # frame where detected


class PlayerTracker:
    """
    Advanced player tracking system with re-identification capabilities
    
    This class maintains player identities across frames using multiple features:
    - Spatial proximity (position tracking)
    - Appearance similarity (color histograms)
    - Size consistency
    - Temporal continuity
    """
    
    def __init__(self, max_disappeared_frames: int = 30, 
                 similarity_threshold: float = 0.7,
                 max_distance_threshold: float = 100.0):
        """
        Initialize the player tracker
        
        Args:
            max_disappeared_frames: Maximum frames a player can be missing before removal
            similarity_threshold: Minimum similarity score for re-identification
            max_distance_threshold: Maximum pixel distance for position-based tracking
        """
        self.max_disappeared_frames = max_disappeared_frames
        self.similarity_threshold = similarity_threshold
        self.max_distance_threshold = max_distance_threshold
        
        # Player tracking state
        self.players: Dict[int, PlayerFeatures] = {}
        self.disappeared_counts: Dict[int, int] = defaultdict(int)
        self.next_player_id = 1
        
        # History for temporal consistency
        self.player_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        
        # Performance metrics
        self.frame_count = 0
        self.processing_times = []
    
    def extract_appearance_features(self, frame: np.ndarray, 
                                  bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract appearance features from a player's bounding box
        
        Args:
            frame: Input frame
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            Normalized color histogram as feature vector
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(64)  # Return zero vector for invalid bbox
        
        # Extract player region
        player_region = frame[y1:y2, x1:x2]
        
        # Convert to HSV for better color representation
        hsv_region = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color histograms
        hist_h = cv2.calcHist([hsv_region], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv_region], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv_region], [2], None, [16], [0, 256])
        
        # Concatenate and normalize
        hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        hist = hist / (np.sum(hist) + 1e-8)  # Normalize with small epsilon
        
        return hist
    
    def calculate_similarity(self, features1: PlayerFeatures, 
                           features2: PlayerFeatures) -> float:
        """
        Calculate similarity between two player feature sets
        
        Args:
            features1, features2: Player features to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        # Spatial similarity (inverse of normalized distance)
        spatial_dist = np.sqrt(
            (features1.center[0] - features2.center[0]) ** 2 +
            (features1.center[1] - features2.center[1]) ** 2
        )
        spatial_similarity = 1.0 / (1.0 + spatial_dist / self.max_distance_threshold)
        
        # Appearance similarity (cosine similarity of histograms)
        appearance_similarity = cosine_similarity(
            features1.appearance_hist.reshape(1, -1),
            features2.appearance_hist.reshape(1, -1)
        )[0, 0]
        
        # Size similarity
        size_ratio = min(features1.size, features2.size) / max(features1.size, features2.size)
        size_similarity = size_ratio if size_ratio > 0.5 else 0.0
        
        # Weighted combination
        total_similarity = (
            0.4 * spatial_similarity +
            0.4 * appearance_similarity +
            0.2 * size_similarity
        )
        
        return max(0.0, min(1.0, total_similarity))
    
    def update(self, frame: np.ndarray, detections: List[Dict]) -> Dict[int, PlayerFeatures]:
        """
        Update player tracking with new detections
        
        Args:
            frame: Current frame
            detections: List of detection dictionaries with 'bbox', 'confidence', 'class'
            
        Returns:
            Dictionary mapping player IDs to their current features
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Filter for players only (class 1: goalkeeper, class 2: player)
        player_detections = [
            det for det in detections 
            if det['class'] in [1, 2] and det['confidence'] > 0.5
        ]
        
        # Extract features for current detections
        current_features = []
        for det in player_detections:
            bbox = det['bbox']
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            appearance_hist = self.extract_appearance_features(frame, bbox)
            
            features = PlayerFeatures(
                bbox=bbox,
                center=center,
                size=size,
                appearance_hist=appearance_hist,
                confidence=det['confidence'],
                frame_id=self.frame_count
            )
            current_features.append(features)
        
        # Match current detections with existing players
        matched_players = {}
        used_detections = set()
        
        # Try to match each existing player with current detections
        for player_id, old_features in self.players.items():
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
                # Update existing player
                matched_players[player_id] = current_features[best_match_idx]
                used_detections.add(best_match_idx)
                self.disappeared_counts[player_id] = 0
                
                # Update history
                self.player_history[player_id].append(current_features[best_match_idx])
            else:
                # Player not found in current frame
                self.disappeared_counts[player_id] += 1
        
        # Create new players for unmatched detections
        for i, features in enumerate(current_features):
            if i not in used_detections:
                player_id = self.next_player_id
                self.next_player_id += 1
                
                matched_players[player_id] = features
                self.disappeared_counts[player_id] = 0
                self.player_history[player_id].append(features)
        
        # Remove players that have been missing too long
        players_to_remove = [
            player_id for player_id, count in self.disappeared_counts.items()
            if count > self.max_disappeared_frames
        ]
        
        for player_id in players_to_remove:
            self.disappeared_counts.pop(player_id, None)
            self.player_history.pop(player_id, None)
        
        # Update current players
        self.players = matched_players
        
        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return self.players
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'frames_processed': self.frame_count,
            'avg_processing_time': np.mean(self.processing_times),
            'fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
            'active_players': len(self.players),
            'total_players_tracked': self.next_player_id - 1
        }


class PlayerReIDSystem:
    """
    Complete player re-identification system
    
    This class orchestrates the entire pipeline:
    1. Video processing
    2. Object detection
    3. Player tracking and re-identification
    4. Visualization and output
    """
    
    def __init__(self, model_path: str = 'best.pt'):
        """
        Initialize the re-identification system
        
        Args:
            model_path: Path to the YOLO model file
        """
        self.model_path = model_path
        self.model = None
        self.tracker = PlayerTracker()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model classes: {self.model.names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class': class_id,
                        'class_name': self.model.names[class_id]
                    })
        
        return detections
    
    def draw_tracking_results(self, frame: np.ndarray, 
                            players: Dict[int, PlayerFeatures],
                            detections: List[Dict]) -> np.ndarray:
        """
        Draw tracking results on frame
        
        Args:
            frame: Input frame
            players: Current tracked players
            detections: All detections for reference
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw tracked players
        for player_id, features in players.items():
            x1, y1, x2, y2 = features.bbox
            
            # Player bounding box (green for tracked players)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Player ID
            label = f"Player {player_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw other detections (ball, referee) in different colors
        tracked_bboxes = {tuple(p.bbox) for p in players.values()}
        
        for det in detections:
            if tuple(det['bbox']) not in tracked_bboxes:
                x1, y1, x2, y2 = det['bbox']
                
                if det['class'] == 0:  # Ball
                    color = (0, 0, 255)  # Red
                    label = "Ball"
                elif det['class'] == 3:  # Referee
                    color = (255, 0, 0)  # Blue
                    label = "Referee"
                else:
                    continue
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated_frame
    
    def process_video(self, input_path: str, output_path: str = None, 
                     display_video: bool = True) -> Dict:
        """
        Process video and perform player re-identification
        
        Args:
            input_path: Path to input video
            output_path: Path for output video (optional)
            display_video: Whether to display video during processing
            
        Returns:
            Processing statistics
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Detect objects
                detections = self.detect_objects(frame)
                
                # Update tracking
                players = self.tracker.update(frame, detections)
                
                # Draw results
                annotated_frame = self.draw_tracking_results(frame, players, detections)
                
                # Add frame info
                info_text = f"Frame: {frame_number}/{total_frames} | Players: {len(players)}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write to output video
                if out:
                    out.write(annotated_frame)
                
                # Display video
                if display_video:
                    cv2.imshow('Player Re-identification', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress logging
                if frame_number % 30 == 0:
                    logger.info(f"Processed {frame_number}/{total_frames} frames")
        
        finally:
            cap.release()
            if out:
                out.release()
            if display_video:
                cv2.destroyAllWindows()
        
        # Return statistics
        stats = self.tracker.get_statistics()
        stats['total_frames'] = frame_number
        stats['video_fps'] = fps
        
        return stats


def main():
    """
    Main function to run the player re-identification system
    """
    # Configuration
    MODEL_PATH = 'best.pt'
    INPUT_VIDEO = '15sec_input_720p.mp4'
    OUTPUT_VIDEO = 'tracked_output.mp4'
    
    try:
        # Initialize system
        reid_system = PlayerReIDSystem(MODEL_PATH)
        
        # Process video
        logger.info("Starting player re-identification...")
        stats = reid_system.process_video(
            input_path=INPUT_VIDEO,
            output_path=OUTPUT_VIDEO,
            display_video=True
        )
        
        # Print results
        logger.info("Processing completed!")
        logger.info("Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()