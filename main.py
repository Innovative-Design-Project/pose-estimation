import os
import argparse
import cv2
import torch
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def process_image(model, image_path, output_dir, device="auto", imgsz=640, conf=0.25):
    print("--------------------------------------------------")
    print(f"Loading image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read: {image_path}")
        return

    h, w, _ = image.shape
    print(f"Image dimensions: {w}x{h}")
    print(f"Running inference (imgsz={imgsz}, conf={conf})...")

    results = model(image, device=device, imgsz=imgsz, conf=conf, verbose=False)
    result = results[0]

    num_persons = len(result.boxes) if result.boxes is not None else 0
    print(f"Detected {num_persons} person(s)")

    if result.keypoints is not None and len(result.keypoints.data) > 0:
        for person_idx, keypoints in enumerate(result.keypoints.data):
            box_conf = float(result.boxes.conf[person_idx]) if result.boxes is not None else 0.0
            print(f"\nPerson {person_idx + 1} (Box confidence: {box_conf:.2f}):")
            for kp_idx, kp in enumerate(keypoints):
                x, y, kp_conf = kp.tolist()
                name = KEYPOINT_NAMES[kp_idx]
                if kp_conf >= 0.5:
                    print(f"  {name:<15}: x={x:6.1f}, y={y:6.1f} | conf={kp_conf:.2f}")

    annotated_image = result.plot()
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"annotated_{filename}")
    cv2.imwrite(output_path, annotated_image)
    print(f"Saved annotated image to: {output_path}\n")


def process_video(model, video_path, output_dir, device="auto", imgsz=640, conf=0.25):
    print("--------------------------------------------------")
    print(f"Loading video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video resolution: {width}x{height} @ {fps} fps")
    print(f"Total frames to process: {total_frames}")

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"annotated_{filename}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, device=device, imgsz=imgsz, conf=conf, verbose=False)
        result = results[0]

        annotated_frame = result.plot()
        out.write(annotated_frame)

        if frame_count % 25 == 0 or frame_count == total_frames:
            percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processing frame {frame_count}/{total_frames} ({percent:.1f}%)")

    cap.release()
    out.release()
    print(f"Finished video. Saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Human Pose Estimation using YOLO")
    parser.add_argument("--images-dir", type=str, default="images", help="Input images directory")
    parser.add_argument("--videos-dir", type=str, default="videos", help="Input videos directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model", type=str, default="yolo26m-pose.pt", help="Path to YOLO model weights")
    parser.add_argument("--device", type=str, default="", help="Device (cpu, cuda, mps)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, args.images_dir) if not os.path.isabs(args.images_dir) else args.images_dir
    videos_dir = os.path.join(base_dir, args.videos_dir) if not os.path.isabs(args.videos_dir) else args.videos_dir
    outputs_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    device = args.device if args.device else get_device()
    print(f"Hardware device selected: {device.upper()}")

    print(f"Loading weights from: {args.model}")
    try:
        model = YOLO(args.model)
        print("Model initialized successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    image_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    if os.path.exists(images_dir):
        images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]
        print(f"Found {len(images)} image(s) in {images_dir}")
        for img_name in sorted(images):
            process_image(model, os.path.join(images_dir, img_name), outputs_dir, device=device, imgsz=args.imgsz, conf=args.conf)

    video_exts = ('.mp4', '.avi', '.mkv', '.mov')
    if os.path.exists(videos_dir):
        videos = [f for f in os.listdir(videos_dir) if f.lower().endswith(video_exts)]
        print(f"Found {len(videos)} video(s) in {videos_dir}")
        for vid_name in sorted(videos):
            process_video(model, os.path.join(videos_dir, vid_name), outputs_dir, device=device, imgsz=args.imgsz, conf=args.conf)

    print("==================================================")
    print("All tasks completed.")


if __name__ == "__main__":
    main()
