"""Convert YOLO-pose pseudo-labels into a COCO Keypoints 1.0 JSON for CVAT import.

CVAT can import "COCO Keypoints 1.0" with the skeleton already drawn per
image, which is what makes manual correction fast (drag wrong points instead
of placing 17 points from scratch). Run this after pseudo_label.py, then
import the resulting json alongside the matching images into a CVAT task.
"""
import argparse
import json
import os
import shutil

from PIL import Image

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Standard COCO person skeleton (1-indexed joint pairs).
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]


def yolo_pose_line_to_coco(line: str, img_w: int, img_h: int):
    parts = line.strip().split()
    if not parts:
        return None
    cx, cy, w, h = (float(v) for v in parts[1:5])
    kp_vals = [float(v) for v in parts[5:]]

    bbox = [
        (cx - w / 2) * img_w,
        (cy - h / 2) * img_h,
        w * img_w,
        h * img_h,
    ]

    keypoints = []
    num_labeled = 0
    for i in range(17):
        x_n, y_n, vis = kp_vals[i * 3], kp_vals[i * 3 + 1], int(kp_vals[i * 3 + 2])
        keypoints.extend([x_n * img_w, y_n * img_h, vis])
        if vis > 0:
            num_labeled += 1

    return bbox, keypoints, num_labeled


def build_coco(img_dir: str, lbl_dir: str, image_names=None) -> dict:
    """Pseudo-labelled frames by default; with `image_names`, exactly those frames
    (frames without a pseudo-label are exported with no skeletons to draw from scratch)."""
    images, annotations = [], []
    img_id, ann_id = 1, 1

    if image_names is None:
        label_stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt"))
        image_names = []
        for stem in label_stems:
            img_candidates = [
                f for f in os.listdir(img_dir)
                if os.path.splitext(f)[0] == stem and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if img_candidates:
                image_names.append(img_candidates[0])

    for img_name in image_names:
        img_path = os.path.join(img_dir, img_name)
        with Image.open(img_path) as im:
            w, h = im.size

        images.append({"id": img_id, "file_name": img_name, "width": w, "height": h})

        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(lbl_path):
            img_id += 1
            continue
        with open(lbl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                bbox, keypoints, num_labeled = yolo_pose_line_to_coco(line, w, h)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "keypoints": keypoints,
                    "num_keypoints": num_labeled,
                })
                ann_id += 1

        img_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON,
        }],
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO-pose -> COCO Keypoints 1.0 (for CVAT import)")
    parser.add_argument("--sport", required=True, choices=["basketball", "hockey"])
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument(
        "--split", choices=["val", "test"], default=None,
        help="Export only this hand-label split (from split_dataset.py) into data/gold/<sport>/<split>/",
    )
    parser.add_argument("--out", default=None, help="Output json path (default: <data-root>/pseudo/<sport>/coco_keypoints.json)")
    args = parser.parse_args()

    lbl_dir = os.path.join(args.data_root, "pseudo", args.sport, "labels")
    if args.split:
        # Images come from raw/, since frames the base model missed entirely are
        # not in pseudo/ but still need to be labelled.
        img_dir = os.path.join(args.data_root, "raw", args.sport, "images")
        with open(os.path.join(args.data_root, "splits", args.sport, f"{args.split}.txt")) as f:
            image_names = [line.strip() for line in f if line.strip()]
        gold_dir = os.path.join(args.data_root, "gold", args.sport, args.split)
        os.makedirs(os.path.join(gold_dir, "images"), exist_ok=True)
        for name in image_names:
            shutil.copy2(os.path.join(img_dir, name), os.path.join(gold_dir, "images", name))
        out_path = args.out or os.path.join(gold_dir, "coco_prefill.json")
        coco = build_coco(img_dir, lbl_dir, image_names)
        print(f"[{args.sport}/{args.split}] images to upload to CVAT: {os.path.join(gold_dir, 'images')}")
    else:
        img_dir = os.path.join(args.data_root, "pseudo", args.sport, "images")
        out_path = args.out or os.path.join(args.data_root, "pseudo", args.sport, "coco_keypoints.json")
        coco = build_coco(img_dir, lbl_dir)

    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"[{args.sport}] {len(coco['images'])} images, {len(coco['annotations'])} annotations -> {out_path}")
    print("Import into CVAT as: New Task -> Upload these images -> Import annotations -> format 'COCO Keypoints 1.0'")


if __name__ == "__main__":
    main()
