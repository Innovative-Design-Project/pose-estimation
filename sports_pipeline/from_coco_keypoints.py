"""Convert a corrected COCO Keypoints 1.0 export (from CVAT) back to YOLO-pose labels.

Run this after correcting pseudo-labels in CVAT and exporting the task as
"COCO Keypoints 1.0". Overwrites sports_pipeline/data/pseudo/<sport>/labels/
with the corrected annotations so build_dataset.py picks them up unchanged.
"""
import argparse
import json
import os


def coco_to_yolo_pose(coco: dict, lbl_dir: str):
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(lbl_dir, exist_ok=True)
    written = 0
    for img_id, img in images_by_id.items():
        w, h = img["width"], img["height"]
        stem = os.path.splitext(img["file_name"])[0]
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue

        lines = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            nw, nh = bw / w, bh / h

            kp = ann["keypoints"]
            kp_tokens = []
            for i in range(17):
                kx, ky, vis = kp[i * 3], kp[i * 3 + 1], int(kp[i * 3 + 2])
                kp_tokens.extend([f"{kx / w:.6f}", f"{ky / h:.6f}", str(vis)])

            line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kp_tokens)
            lines.append(line)

        with open(os.path.join(lbl_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Corrected COCO Keypoints 1.0 (CVAT export) -> YOLO-pose labels")
    parser.add_argument("--sport", required=True, choices=["basketball", "hockey"])
    parser.add_argument("--coco-json", required=True, help="Path to the corrected COCO Keypoints export from CVAT")
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "data"))
    args = parser.parse_args()

    with open(args.coco_json) as f:
        coco = json.load(f)

    lbl_dir = os.path.join(args.data_root, "pseudo", args.sport, "labels")
    n = coco_to_yolo_pose(coco, lbl_dir)
    print(f"[{args.sport}] wrote {n} corrected label files -> {lbl_dir}")
    print("Now re-run: python sports_pipeline/build_dataset.py")


if __name__ == "__main__":
    main()
