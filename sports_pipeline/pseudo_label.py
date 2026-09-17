"""Generate pseudo pose-keypoint labels for sports frames using a pretrained pose model.

Runs yolo11m-pose (COCO-pretrained, 17 keypoints) over every frame in
sports_pipeline/data/raw/<sport>/images/ and writes YOLO-pose format labels
(one txt per image: `class cx cy w h x1 y1 v1 ... x17 y17 v17`, normalized)
to sports_pipeline/data/pseudo/<sport>/labels/, mirroring the images into
sports_pipeline/data/pseudo/<sport>/images/.

These are *pseudo* labels: sports poses (occlusion, contact, sticks, motion
blur) will contain systematic errors the base model makes on this domain.
Review/correct a sample of them (Roboflow or CVAT) before fine-tuning on the
full set.
"""
import argparse
import os
import shutil

from ultralytics import YOLO

SPORTS = ["basketball", "hockey"]


def pseudo_label_sport(model, sport: str, data_root: str, conf: float, imgsz: int):
    img_dir = os.path.join(data_root, "raw", sport, "images")
    if not os.path.isdir(img_dir):
        print(f"[{sport}] no images found at {img_dir}, skipping")
        return

    out_img_dir = os.path.join(data_root, "pseudo", sport, "images")
    out_lbl_dir = os.path.join(data_root, "pseudo", sport, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"[{sport}] running pose inference on {len(images)} frames...")

    kept = 0
    for i, fname in enumerate(images):
        src_path = os.path.join(img_dir, fname)
        results = model(src_path, conf=conf, imgsz=imgsz, verbose=False)
        result = results[0]

        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            continue  # no person detected -> not useful as a training pseudo-label

        h, w = result.orig_shape
        lines = []
        for person_idx in range(len(result.boxes)):
            box = result.boxes.xywhn[person_idx].tolist()  # cx, cy, w, h (normalized)
            kps = result.keypoints.data[person_idx]  # (17, 3) in pixel coords + conf
            kp_tokens = []
            for x, y, kp_conf in kps.tolist():
                # YOLO-pose visibility flag: 0=not labeled, 1=labeled/occluded, 2=labeled/visible
                vis = 2 if kp_conf >= 0.5 else (1 if kp_conf > 0 else 0)
                kp_tokens.extend([f"{x / w:.6f}", f"{y / h:.6f}", str(vis)])
            line = "0 " + " ".join(f"{v:.6f}" for v in box) + " " + " ".join(kp_tokens)
            lines.append(line)

        shutil.copy2(src_path, os.path.join(out_img_dir, fname))
        stem = os.path.splitext(fname)[0]
        with open(os.path.join(out_lbl_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        kept += 1

        if (i + 1) % 200 == 0:
            print(f"[{sport}] {i + 1}/{len(images)} processed, {kept} with detections")

    print(f"[{sport}] done: {kept}/{len(images)} frames had person detections -> {out_lbl_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pseudo-label sports frames with a pretrained pose model")
    parser.add_argument("--model", default="yolo11m-pose.pt", help="Pretrained pose weights to use for labeling")
    parser.add_argument("--sports", nargs="+", default=SPORTS, choices=SPORTS)
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.model)
    for sport in args.sports:
        pseudo_label_sport(model, sport, args.data_root, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
