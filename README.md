# High-Precision Human Pose Estimation

State-of-the-art 2D human pose estimation pipeline using **Ultralytics YOLO26m-pose** (Medium end-to-end architecture) with automatic GPU/MPS hardware acceleration. Accurately detects and tracks 17 COCO body keypoints across complex postures, yoga positions, dance movements, and video streams.

---

## Features

- **Default Model (`yolo26m-pose.pt`)**: High-performance Medium architecture (~24.2M parameters, 85.5 GFLOPs) offering fast inference and accurate joint localization on complex poses (like Warrior and Boat poses).
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
├── yolo26m-pose.pt     # Default pose model weights (Medium)
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

---

## Model Benchmark

| Model | Parameters | GFLOPs | Accuracy on Complex Poses | Recommended For |
| :--- | :--- | :--- | :--- | :--- |
| **`yolo26n-pose`** | ~3.68 M | 10.4 | Low (Struggles with wide lunges) | Edge / IoT Devices |
| **`yolo26m-pose`** | ~24.2 M | 85.5 | High (Good balance) | Real-time high-FPS video |
| **`yolo11x-pose`** | **~58.8 M** | **204.2** | **Flawless (State-of-the-Art)** | **Studio, athletic & yoga analysis** |

---

## License

This project is licensed under the MIT License.
