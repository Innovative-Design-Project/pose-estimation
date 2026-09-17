"""Fine-tune the pretrained pose model on the sports (hockey/basketball) dataset.

Thin wrapper around `yolo pose train` so the run config lives in one place.
Starts from COCO-pretrained weights (transfer learning), not from scratch.
"""
import argparse
import os

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune yolo11m-pose on the sports dataset")
    parser.add_argument("--base-model", default="yolo11m-pose.pt")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "data", "sports_pose_dataset", "sports_pose.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default=os.path.join(os.path.dirname(__file__), "runs"))
    parser.add_argument("--name", default="sports_pose_finetune")
    args = parser.parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device or None,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
