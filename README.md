# High-Precision Human Pose Estimation

State-of-the-art 2D human pose estimation pipeline using **Ultralytics YOLO11m-pose** (Medium architecture) with automatic GPU/MPS hardware acceleration. Accurately detects and tracks 17 COCO body keypoints across complex postures, yoga positions, dance movements, and video streams.

---

## Features

- **Default Model (`yolo11m-pose.pt`)**: Medium architecture used for offline processing (`main.py`) and the fine-tuning pipeline. Live mode (`live.py`) defaults to the faster `yolo11n-pose.pt`. Weights are downloaded automatically on first use.
- **Hardware Acceleration**: Automatic device detection supporting **Apple Silicon (MPS)**, **NVIDIA (CUDA)**, and fallback CPU.
- **Batch Processing**: Automatically scans and processes all images and videos from designated input folders.
- **Visual Annotations & Logging**: Outputs rendered skeleton diagrams, bounding boxes, and prints keypoint coordinates with confidence scores.

---

## 17 COCO Keypoints

The model predicts 17 anatomical keypoints across the human body:

| Index | Keypoint Name | Index | Keypoint Name |
| :--- | :--- | :--- | :--- |
| `0` | Nose | `9` | Left Wrist |
| `1` | Left Eye | `10` | Right Wrist |
| `2` | Right Eye | `11` | Left Hip |
| `3` | Left Ear | `12` | Right Hip |
| `4` | Right Ear | `13` | Left Knee |
| `5` | Left Shoulder | `14` | Right Knee |
| `6` | Right Shoulder | `15` | Left Ankle |
| `7` | Left Elbow | `16` | Right Ankle |
| `8` | Right Elbow | | |

---

## Project Structure

```
pose-estimation/
├── images/             # Input images (.png, .jpg, .jpeg, .webp)
├── videos/             # Input videos (.mp4, .avi, .mkv, .mov)
├── outputs/            # Output annotated images & videos
├── main.py             # Main inference & processing script
├── requirements.txt    # Project dependencies
├── live.py             # Live camera pose + session recording
├── yolo11m-pose.pt     # Default pose model weights (Medium, auto-downloaded)
└── README.md           # Documentation
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pose-estimation.git
   cd pose-estimation
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # On macOS / Linux
   # .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Default Run (Processes all files in `images/` and `videos/`)
Place any test images in `images/` or videos in `videos/`, then run:
```bash
python main.py
```

### 2. Custom Parameters & Options
You can customize the model, input/output folders, image resolution, and device:
```bash
# Run with high resolution for extra precision on wide shots
python main.py --imgsz 1280

# Specify custom folders
python main.py --images-dir my_images --output-dir my_outputs

# Force a specific hardware device
python main.py --device mps     # Apple Silicon
python main.py --device cuda    # NVIDIA GPU
python main.py --device cpu     # CPU
```

### 3. Live Camera Mode (`live.py`)
Runs pose estimation on a webcam or phone camera in real time and records the session:
```bash
python live.py                      # default camera, records to sessions/<timestamp>/
python live.py --list-cameras       # find the index of a phone (macOS Continuity Camera)
python live.py --source 1 --cam-fps 60
python live.py --source videos/basketball_sportsmot_test.mp4   # replay a file as if live
python live.py --no-record          # preview only
```
Press `q` or `Esc` to stop. Each session folder contains `raw.mp4` (every frame, no overlay),
`frame_times.csv` (capture timestamps), `keypoints.jsonl` (17 keypoints per person for every analysed
frame), `meta.json`, and `annotated.mp4` with `--save-annotated`. On macOS, allow camera access for
your terminal in System Settings > Privacy & Security > Camera.

---

## Model Benchmark

Measured on an Apple M4 laptop, 720p frames, pose inference only:

| Model | Device | Latency / frame | Throughput | Use |
| :--- | :--- | :--- | :--- | :--- |
| `yolo11n-pose` | CPU | 20 ms | ~51 fps | Live mode (default for `live.py`) |
| `yolo11n-pose` | MPS | 22 ms | ~45 fps | Live mode |
| `yolo11m-pose` | MPS | 52 ms | ~19 fps | Offline processing (default for `main.py`) |
| `yolo11m-pose` | CPU | 80 ms | ~12 fps | Offline processing |

---

## License

This project is licensed under the MIT License.
