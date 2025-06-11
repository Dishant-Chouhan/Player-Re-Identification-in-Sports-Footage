import re
import os
import logging
from collections import defaultdict
from datetime import datetime
import numpy as np

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/evaluation_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
handler = logging.FileHandler("logs/evaluation_log.txt")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.handlers = [handler]

def parse_detection_log(path):
    if not os.path.exists(path):
        logger.error(f"Detection log not found: {path}")
        raise FileNotFoundError(f"Detection log not found: {path}")
    frame_detections = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*Frame (\d+): (\d+) detections, classes: \[(.*?)\]", line)
            if match:
                frame_num = int(match.group(2))
                num_detections = int(match.group(3))
                classes = [int(c) for c in match.group(4).split(", ") if match.group(4)] if match.group(4) else []
                frame_detections[frame_num] = {"num_detections": num_detections, "classes": classes}
            else:
                logger.debug(f"Failed to parse detection log line: {line.strip()}")
    if not frame_detections:
        logger.warning("No valid detection log entries found")
    return frame_detections

def parse_tracking_log(path):
    if not os.path.exists(path):
        logger.error(f"Tracking log not found: {path}")
        raise FileNotFoundError(f"Tracking log not found: {path}")
    frame_tracks = defaultdict(list)
    reid_events = defaultdict(list)
    new_id_events = defaultdict(list)
    removal_events = defaultdict(list)
    total_ids = defaultdict(int)
    timestamps = []
    with open(path, 'r') as f:
        for line in f:
            ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
            if ts_match:
                try:
                    ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                    timestamps.append(ts)
                except ValueError:
                    logger.debug(f"Invalid timestamp: {ts_match.group(1)}")
            match_tracks = re.match(r".*Frame (\d+): (\d+) tracks, IDs: \[(.*?)\], Total IDs: (\d+)", line)
            match_reid = re.match(r".*Frame (\d+): Re-identified (\w+) ID (\d+), similarity: ([\d.]+)", line)
            match_new = re.match(r".*Frame (\d+): New (\w+) ID (\d+)", line)
            match_remove = re.match(r".*Frame (\d+): Removed (\w+) ID (\d+)", line)
            if match_tracks:
                frame_num = int(match_tracks.group(1))
                num_tracks = int(match_tracks.group(2))
                ids = [int(id_) for id_ in match_tracks.group(3).split(", ") if match_tracks.group(3)] if match_tracks.group(3) else []
                frame_tracks[frame_num] = {"num_tracks": num_tracks, "ids": ids}
                total_ids['all'] = max(total_ids['all'], int(match_tracks.group(4)))
            elif match_reid:
                class_name = match_reid.group(2).lower()
                reid_events[class_name].append({"frame": int(match_reid.group(1)), "id": int(match_reid.group(3)), "similarity": float(match_reid.group(4))})
                total_ids[class_name] = max(total_ids[class_name], int(match_reid.group(3)))
            elif match_new:
                class_name = match_new.group(2).lower()
                new_id_events[class_name].append({"frame": int(match_new.group(1)), "id": int(match_new.group(3))})
                total_ids[class_name] = max(total_ids[class_name], int(match_new.group(3)))
            elif match_remove:
                class_name = match_remove.group(2).lower()
                removal_events[class_name].append({"frame": int(match_new.group(1)), "id": int(match_new.group(3))})
                total_ids[class_name] = max(total_ids[class_name], int(match_new.group(3)))
            else:
                logger.debug(f"Failed to parse tracking log line: {line.strip()}")
    if not total_ids['all']:
        logger.warning("No valid tracking log entries found")
    return frame_tracks, reid_events, new_id_events, removal_events, total_ids, timestamps

def estimate_id_switches(frame_tracks, reid_events, new_id_events, removal_events, class_name):
    active_ids = set()
    id_switches = 0
    for frame in sorted(frame_tracks.keys()):
        current_ids = set(frame_tracks[frame]["ids"])
        new_active = current_ids - active_ids
        removed_active = active_ids - current_ids
        expected_new = {e["id"] for e in new_id_events.get(class_name, []) if e["frame"] == frame}
        expected_removed = {e["id"] for e in removal_events.get(class_name, []) if e["frame"] == frame}
        reid_ids = {e["id"] for e in reid_events.get(class_name, []) if e["frame"] == frame}
        unexpected_new = new_active - expected_new - reid_ids
        unexpected_removed = removed_active - expected_removed
        switches = min(len(unexpected_new), len(unexpected_removed))
        id_switches += switches
        active_ids = current_ids
    return id_switches

def evaluate_performance(detection_log_path="logs/detection_log.txt", tracking_log_path="logs/tracking_log.txt", output_video_path="output/output.mp4"):
    for path in [detection_log_path, tracking_log_path, output_video_path]:
        if not os.path.exists(path):
            logger.error(f"Missing file: {path}")
            raise FileNotFoundError(f"File not found: {path}")

    logger.info("Parsing detection and tracking logs")
    frame_detections = parse_detection_log(detection_log_path)
    frame_tracks, reid_events, new_id_events, removal_events, total_ids, timestamps = parse_tracking_log(tracking_log_path)

    class_counts = defaultdict(list)
    for frame, det in frame_detections.items():
        class_freq = defaultdict(int)
        for cls in det["classes"]:
            class_freq[cls] += 1
        for cls in [0, 1, 2, 3]:
            class_counts[cls].append(class_freq[cls])
    avg_counts = {cls: np.mean(counts) if counts else 0.0 for cls, counts in class_counts.items()}
    std_counts = {cls: np.std(counts) if counts else 0.0 for cls, counts in class_counts.items()}

    class_names = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
    id_switches = {class_name: estimate_id_switches(frame_tracks, reid_events, new_id_events, removal_events, class_name)
                   for class_name in class_names.values()}

    reid_success = {}
    for class_name in class_names.values():
        total_attempts = len(reid_events.get(class_name, [])) + \
                         len([e for e in new_id_events.get(class_name, []) if e["frame"] > 75])
        reid_success[class_name] = (len(reid_events.get(class_name, [])) / total_attempts * 100) if total_attempts > 0 else 0.0

    time_diffs = [(t2 - t1).total_seconds() for t1, t2 in zip(timestamps[:-1], timestamps[1:]) if (t2 - t1).total_seconds() > 0]
    fps = 1 / np.mean(time_diffs) if time_diffs and np.mean(time_diffs) > 0 else 17.0
    logger.info(f"Calculated FPS: {fps:.2f} from {len(time_diffs)} valid time differences")

    report_lines = [
        "Evaluation Report",
        "================",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Video: 15sec_input_720p.mp4",
        "Model: best.pt",
        f"Output: {output_video_path}",
        "",
        "1. Detection Accuracy",
        f"- Ball: {avg_counts.get(0, 0.0):.2f} ± {std_counts.get(0, 0.0):.2f} objects/frame (Target: ~1)",
        f"- Goalkeeper: {avg_counts.get(1, 0.0):.2f} ± {std_counts.get(1, 0.0):.2f} objects/frame (Target: ~1)",
        f"- Player: {avg_counts.get(2, 0.0):.2f} ± {std_counts.get(2, 0.0):.2f} objects/frame (Target: ~16)",
        f"- Referee: {avg_counts.get(3, 0.0):.2f} ± {std_counts.get(3, 0.0):.2f} objects/frame (Target: ~2)",
        f"- Status: {'Pass' if 14 <= avg_counts.get(2, 0.0) <= 18 and avg_counts.get(0, 0.0) >= 0.8 and avg_counts.get(1, 0.0) >= 0.8 and avg_counts.get(3, 0.0) >= 1.5 else 'Fail'}",
        "",
        "2. ID Switches",
        f"- Ball: {id_switches['ball']} (Target: <2)",
        f"- Goalkeeper: {id_switches['goalkeeper']} (Target: <2)",
        f"- Player: {id_switches['player']} (Target: <5)",
        f"- Referee: {id_switches['referee']} (Target: <2)",
        f"- Status: {'Pass' if id_switches['player'] < 5 and id_switches['ball'] < 2 and id_switches['goalkeeper'] < 2 and id_switches['referee'] < 2 else 'Fail'}",
        "",
        "3. Re-Identification Success",
        f"- Ball: {reid_success['ball']:.2f}% ({len(reid_events.get('ball', []))}/{len(reid_events.get('ball', [])) + len([e for e in new_id_events.get('ball', []) if e['frame'] > 75])}) (Target: >80%)",
        f"- Goalkeeper: {reid_success['goalkeeper']:.2f}% ({len(reid_events.get('goalkeeper', []))}/{len(reid_events.get('goalkeeper', [])) + len([e for e in new_id_events.get('goalkeeper', []) if e['frame'] > 75])}) (Target: >80%)",
        f"- Player: {reid_success['player']:.2f}% ({len(reid_events.get('player', []))}/{len(reid_events.get('player', [])) + len([e for e in new_id_events.get('player', []) if e['frame'] > 75])}) (Target: >80%)",
        f"- Referee: {reid_success['referee']:.2f}% ({len(reid_events.get('referee', []))}/{len(reid_events.get('referee', [])) + len([e for e in new_id_events.get('referee', []) if e['frame'] > 75])}) (Target: >80%)",
        f"- Status: {'Pass' if all(reid_success[c] > 80 for c in class_names.values()) else 'Fail'}",
        "",
        "4. Efficiency",
        f"- FPS: {fps:.2f} (Target: ~17)",
        f"- Status: {'Pass' if fps >= 15 else 'Fail'}",
        "",
        "5. ID Assignment",
        f"- Total IDs: {total_ids['all']} (Ball: {total_ids.get('ball', 0)}, Goalkeeper: {total_ids.get('goalkeeper', 0)}, Player: {total_ids.get('player', 0)}, Referee: {total_ids.get('referee', 0)})",
        f"- Target: ~20 (Ball: ~1, Goalkeeper: ~1, Player: ~16, Referee: ~2)",
        f"- Status: {'Pass' if total_ids['all'] <= 30 and total_ids.get('player', 0) <= 20 and total_ids.get('ball', 0) <= 2 and total_ids.get('goalkeeper', 0) <= 2 and total_ids.get('referee', 0) <= 3 else 'Fail'}",
        "",
        "6. Visual Check",
        f"- Inspect {output_video_path} at frames 210–229 (~8.4–9.2s).",
        "- Verify: Red boxes for ball, yellow for goalkeeper, green for players, blue for referee.",
        "- Check: Stable IDs during re-entry (e.g., goal event)."
    ]
    report = "\n".join(report_lines)
    logger.info("Writing evaluation report")
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    print(report)
    handler.flush()

if __name__ == "__main__":
    try:
        logger.info("Starting evaluation")
        evaluate_performance()
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        print(f"Error: {str(e)}")
        raise