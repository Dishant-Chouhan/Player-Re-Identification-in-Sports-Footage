import cv2
import logging
import os
from utils.detection import Detector
from utils.tracking import EnhancedTracker
from utils.visualization import EnhancedVisualizer

os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)
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
logger.debug("Logging initialized in main.py")

def main():
    logger.debug("Starting main function")
    model_path = "best.pt"
    video_path = "15sec_input_720p.mp4"
    output_path = "output/output.mp4"

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.debug("Initializing detector, tracker, and visualizer")
    detector = Detector(model_path, conf_thres=0.25, iou_thres=0.5)
    tracker = EnhancedTracker(
        max_disappeared_frames=75,
        similarity_threshold=0.35,
        max_distance_threshold=100.0
    )
    visualizer = EnhancedVisualizer()

    logger.debug("Opening video")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file")
        raise ValueError("Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        logger.error("Error initializing video writer")
        raise ValueError("Failed to initialize video writer")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.debug("End of video reached")
            break
        frame_count += 1
        logger.debug(f"Processing frame {frame_count}")

        detections = detector.detect(frame, frame_count)
        logger.info(f"Frame {frame_count}: {len(detections)} detections, classes: {[d['class'] for d in detections]}")

        tracks = tracker.update(frame, detections, frame_count)
        logger.info(f"Frame {frame_count}: {len(tracks)} tracks, IDs: {[t['id'] for t in tracks]}")

        frame = visualizer.draw(frame, tracks, detections, frame_count, total_frames)
        out.write(frame)

        print(f"Processing frame {frame_count}/{total_frames}")
        handler.flush()

    logger.debug("Releasing resources")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info("Processing complete")
    print("Output saved to", output_path)
    handler.flush()

if __name__ == "__main__":
    main()