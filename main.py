import cv2
import os
from ultralytics import YOLO

# coco keypoints
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def process_image(model, image_path, output_dir):
    print(f"processing {image_path}...")
    
    # read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"couldn't read {image_path}")
        return

    # run pose estimation
    results = model(image)
    
    # get first result since we only passed one image
    result = results[0]
    
    # print detected keypoints
    if result.keypoints is not None and len(result.keypoints.data) > 0:
        print("detected keypoints:")
        for person_idx, keypoints in enumerate(result.keypoints.data):
            print(f"\nperson {person_idx + 1}:")
            for kp_idx, kp in enumerate(keypoints):
                x, y, conf = kp.tolist()
                name = KEYPOINT_NAMES[kp_idx]
                
                # only show if confidence > 50%
                if conf > 0.5:
                    print(f"  {name}: ({x:.1f}, {y:.1f}) | conf: {conf:.2f}")
    else:
        print("No one detected")

    # draw skeleton on image
    annotated_image = result.plot()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # save it
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"annotated_{output_filename}")
    
    cv2.imwrite(output_path, annotated_image)
    print(f"saved to {output_path}\n")


def process_video(model, video_path, output_dir):
    print(f"processing video: {video_path}")
    
    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"couldn't open {video_path}")
        return

    # get video info for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"annotated_{output_filename}")
    
    # set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
            
        frame_count += 1
        
        # run inference
        results = model(frame)
        result = results[0]
        
        # draw predictions
        annotated_frame = result.plot()
        
        # write frame
        out.write(annotated_frame)
        
        # print progress
        if frame_count % 10 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"done {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
    cap.release()
    out.release()
    print(f"Saved video to {output_path}\n")


if __name__ == "__main__":
    # setup folders
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, "images")
    videos_dir = os.path.join(base_dir, "videos")
    outputs_dir = os.path.join(base_dir, "outputs")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # load yolo pose model
    model_name = "yolo26n-pose.pt"
    print(f"loading {model_name}...")
    try:
        model = YOLO(model_name)
        print("model loaded\n")
    except Exception as e:
        print(f"failed to load {model_name}: {e}")
        exit(1)
        
    # process all images
    image_exts = ('.jpg', '.jpeg', '.png')
    found_images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_exts)]
    
    if found_images:
        for img_name in found_images:
            process_image(model, os.path.join(images_dir, img_name), outputs_dir)
    else:
        print("No images found in images/ folder. drop some .jpg or .png files in there.")
        
    # process all videos
    video_exts = ('.mp4', '.avi', '.mkv')
    found_videos = [f for f in os.listdir(videos_dir) if f.lower().endswith(video_exts)]
    
    if found_videos:
        for vid_name in found_videos:
            process_video(model, os.path.join(videos_dir, vid_name), outputs_dir)
    else:
        print("No videos found in videos/ folder. drop some .mp4 files in there.")
