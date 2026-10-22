"""Train a basketball detector (ball only) from the Roboflow export's Ball boxes.

The Roboflow pool (see download_frames.py) has Ball / Player / Referee boxes;
players are already covered by the pose model, so only class 0 (Ball) is kept.
Frames without a ball stay in as negatives so the detector learns not to fire
on heads, lights and logos.

The clip-level split from split_dataset.py is reused (val + test clips become
the detector's val set), so near-duplicate frames never leak across splits.

    python sports_pipeline/train_ball.py            # build dataset + train
    python sports_pipeline/train_ball.py --prepare-only
    python sports_pipeline/train_ball.py --resume     # continue after an interruption

Weights end up in sports_pipeline/runs/ball_detector/weights/best.pt; copy or
point live.py / analyze.py at them with --ball-model.
"""
import argparse
import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
BALL_CLASS = 0

DATA_YAML_TEMPLATE = """\
path: {root}
train: images/train
val: images/val

names:
  0: ball
"""


def read_list(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def export_label_path(export_root: str, raw_name: str) -> str:
    """raw images are `<export split>_<original name>` (see download_frames.collect_images)."""
    export_split, orig = raw_name.split("_", 1)
    return os.path.join(export_root, export_split, "labels", os.path.splitext(orig)[0] + ".txt")


def build_dataset(data_root: str, out_root: str) -> str:
    raw_img = os.path.join(data_root, "raw", "basketball", "images")
    export_root = os.path.join(data_root, "raw", "basketball", "_export")
    split_dir = os.path.join(data_root, "splits", "basketball")
    splits = {
        "train": read_list(os.path.join(split_dir, "train.txt")),
        "val": read_list(os.path.join(split_dir, "val.txt")) + read_list(os.path.join(split_dir, "test.txt")),
    }

    for split, names in splits.items():
        img_dir = os.path.join(out_root, "images", split)
        lbl_dir = os.path.join(out_root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n_balls = n_with_ball = 0
        for name in names:
            src_lbl = export_label_path(export_root, name)
            lines = []
            if os.path.exists(src_lbl):
                with open(src_lbl) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 5 and int(parts[0]) == BALL_CLASS:
                            lines.append("0 " + " ".join(parts[1:]))
            n_balls += len(lines)
            n_with_ball += bool(lines)

            dst_img = os.path.join(img_dir, name)
            if not os.path.lexists(dst_img):
                os.symlink(os.path.join(raw_img, name), dst_img)
            with open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"[{split}] {len(names)} frames, {n_with_ball} with a ball, {n_balls} ball boxes")

    yaml_path = os.path.join(out_root, "ball.yaml")
    with open(yaml_path, "w") as f:
        f.write(DATA_YAML_TEMPLATE.format(root=os.path.abspath(out_root)))
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train the basketball (ball) detector")
    parser.add_argument("--data-root", default=os.path.join(HERE, "data"))
    parser.add_argument("--out-root", default=os.path.join(HERE, "data", "ball_dataset"))
    parser.add_argument("--base-model", default="yolo11n.pt", help="COCO-pretrained detector to fine-tune")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run from runs/ball_detector/weights/last.pt")
    args = parser.parse_args()

    if args.resume:
        # Resuming restores the original run's settings (epochs, device, ...) from the checkpoint.
        YOLO(os.path.join(HERE, "runs", "ball_detector", "weights", "last.pt")).train(resume=True)
        return

    yaml_path = build_dataset(args.data_root, args.out_root)
    print(f"data.yaml: {yaml_path}")
    if args.prepare_only:
        return

    model = YOLO(args.base_model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device or None,
        project=os.path.join(HERE, "runs"),
        name="ball_detector",
        exist_ok=True,
        # The ball is small and round: keep scale/flip augmentation, skip mosaic late.
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
