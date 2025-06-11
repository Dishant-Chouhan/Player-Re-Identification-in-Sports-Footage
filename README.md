# Player Re-Identification in a Single Feed

This repository contains the solution for the **Liat.ai AI/ML Intern Assignment** (Option 2: Re-Identification in a Single Feed). It processes a 15-second soccer video (`15sec_input_720p.mp4`) to detect and track objects including the **ball, goalkeeper, players**, and **referees**, using **YOLOv11** for detection and a **ResNet18-based tracker** for re-identification. The output includes an annotated video and an evaluation report.

**[Download Pre-trained Model (best.pt)](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)**

---

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 
- Input video: `15sec_input_720p.mp4`
- Pre-trained model: `best.pt`  
  (YOLOv11 | Classes: `{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}`)

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Setup

```bash
git clone <your-repo-url>
cd player_reid
```

1. Place `15sec_input_720p.mp4` and `best.pt` in the root directory.
2. (Windows only) Set environment variable to avoid duplicate library error:
```bash
set KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Running the Code

### Process the Video:
```bash
python main.py
```

**Outputs:**
- `output/output.mp4`
- `logs/detection_log.txt`
- `logs/tracking_log.txt`



The script performs:
- Object detection (YOLOv11)
- Tracking & re-identification (ResNet18-based tracker)
- Annotated video output  
  (Color-coded bounding boxes: Red = Ball, Yellow = Goalkeeper, Green = Players, Blue = Referees)

- 
## 📹 Output Video

[Watch the output video](https://drive.google.com/file/d/1fzXlMIaVKB7IqiwfyF7XfvEmVRIvyxhe/view?usp=sharing)
---

### Evaluate Performance:
```bash
python evaluate.py
```

**Outputs:**
- `evaluation_report.txt`
- `logs/evaluation_log.txt`

The evaluation script measures:
- Detection accuracy  
- ID switches  
- Re-identification success rate  
- ID assignment consistency  
- Processing efficiency  

For more detailed insights, **process `evaluate.py` and refer to the generated report**.

---

## Directory Structure

```
player_reid/
├── main.py
├── evaluate.py
├── utils/
│   ├── detection.py
│   ├── tracking.py
│   ├── visualization.py
├── logs/
│   ├── detection_log.txt
│   ├── tracking_log.txt
├── output/
│   └── output.mp4
├── requirements.txt
├── README.md
├── report.md
├── 15sec_input_720p.mp4
```

---

## Notes

- The current implementation prioritizes functionality over optimization due to time constraints.
- Refer to `report.pdf` for known limitations, challenges, and suggested improvements.
- For a re-identification example, examine frames **210–229 (~8.4–9.2s)** in `output/output.mp4`.
- For more detailed metrics and analysis, **run `evaluate.py`**.

---


