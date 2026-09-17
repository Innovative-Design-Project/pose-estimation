"""Download source frames for the hockey/basketball pose fine-tuning pipeline.

These Roboflow projects only have bounding-box (player/ball/puck) labels, not
keypoints -- we only want their images as a curated pool of in-domain frames.
Pose labels are generated later by pseudo_label.py.
"""
import argparse
import os
import shutil

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

# (workspace, project, version) for each sport, picked for image-count coverage.
DATASETS = {
    "basketball": ("testroom-bnm8h", "basketball-1-gl9mk", 1),  # 8595 images
    "hockey": ("sportcontract", "hockey-fwm0b", 1),  # 1583 images
}


def download(sport: str, dest_root: str) -> str:
    workspace, project, version = DATASETS[sport]
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY (see .env)")

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(
        "yolov8", location=os.path.join(dest_root, sport, "_export")
    )
    return ds.location


def collect_images(export_dir: str, out_dir: str) -> int:
    """Flatten a YOLOv8-export's train/valid/test image folders into one dir."""
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(export_dir, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(img_dir, fname)
            dst = os.path.join(out_dir, f"{split}_{fname}")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Download hockey/basketball frame pools")
    parser.add_argument("--sports", nargs="+", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    parser.add_argument("--dest", default=os.path.join(os.path.dirname(__file__), "data", "raw"))
    args = parser.parse_args()

    for sport in args.sports:
        print(f"[{sport}] downloading from Roboflow...")
        export_dir = download(sport, args.dest)
        out_dir = os.path.join(args.dest, sport, "images")
        n = collect_images(export_dir, out_dir)
        print(f"[{sport}] {n} frames ready in {out_dir}")


if __name__ == "__main__":
    main()
