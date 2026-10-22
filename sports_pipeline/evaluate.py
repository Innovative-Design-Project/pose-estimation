"""Evaluate pose models on the hand-labelled basketball val/test split.

Runs Ultralytics validation (COCO OKS-based keypoint mAP) for one or more
weights files against data/gold/<sport>/<split>/ and prints a comparison
table, so the baseline and each fine-tuned checkpoint are scored on exactly
the same frames:

    python sports_pipeline/evaluate.py                       # baseline yolo11m-pose on test
    python sports_pipeline/evaluate.py --models yolo11m-pose.pt runs/sports_pose_finetune/weights/best.pt

Metrics are written to runs/eval/<sport>_<split>/metrics.json.

`--labels pseudo` scores against the base model's own pseudo-labels instead of
gold labels. That only checks the plumbing (the base model will score close to
perfect against itself) -- never report those numbers.
"""
import argparse
import json
import os
import shutil

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_YAML_TEMPLATE = """\
path: {root}
train: images  # required by Ultralytics, unused for validation
val: images

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
"""


def prepare_pseudo_eval_dir(data_root: str, sport: str, split: str) -> str:
    """Mirror the split's frames with pseudo labels into a gold-like layout."""
    out = os.path.join(data_root, "eval_pseudo", sport, split)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "labels"))
    raw_img = os.path.join(data_root, "raw", sport, "images")
    pseudo_lbl = os.path.join(data_root, "pseudo", sport, "labels")
    with open(os.path.join(data_root, "splits", sport, f"{split}.txt")) as f:
        names = [line.strip() for line in f if line.strip()]
    for name in names:
        shutil.copy2(os.path.join(raw_img, name), os.path.join(out, "images", name))
        lbl = os.path.join(pseudo_lbl, os.path.splitext(name)[0] + ".txt")
        dst = os.path.join(out, "labels", os.path.splitext(name)[0] + ".txt")
        if os.path.exists(lbl):
            shutil.copy2(lbl, dst)
        else:
            open(dst, "w").close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose models on the hand-labelled basketball split")
    parser.add_argument("--models", nargs="+", default=[os.path.join(HERE, "..", "yolo11m-pose.pt")])
    parser.add_argument("--sport", default="basketball")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--labels", choices=["gold", "pseudo"], default="gold")
    parser.add_argument("--data-root", default=os.path.join(HERE, "data"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default=os.path.join(HERE, "runs", "eval"))
    args = parser.parse_args()

    if args.labels == "gold":
        eval_dir = os.path.join(args.data_root, "gold", args.sport, args.split)
        if not os.path.isdir(os.path.join(eval_dir, "labels")):
            raise SystemExit(
                f"No gold labels at {eval_dir}/labels. Hand-label the split first (see LABELING.md), "
                "or pass --labels pseudo to smoke-test the pipeline."
            )
    else:
        eval_dir = prepare_pseudo_eval_dir(args.data_root, args.sport, args.split)
        print("WARNING: scoring against pseudo labels -- plumbing check only, numbers are meaningless.")

    run_name = f"{args.sport}_{args.split}" + ("_pseudo" if args.labels == "pseudo" else "")
    out_dir = os.path.join(args.project, run_name)
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(DATA_YAML_TEMPLATE.format(root=os.path.abspath(eval_dir)))

    n_images = len(os.listdir(os.path.join(eval_dir, "images")))
    rows = []
    for weights in args.models:
        model_name = os.path.splitext(os.path.basename(weights))[0]
        model = YOLO(weights)
        metrics = model.val(
            data=yaml_path,
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device or None,
            project=out_dir,
            name=model_name,
            exist_ok=True,
            plots=True,
            verbose=False,
        )
        rows.append({
            "model": weights,
            "pose_mAP50-95": round(float(metrics.pose.map), 4),
            "pose_mAP50": round(float(metrics.pose.map50), 4),
            "box_mAP50-95": round(float(metrics.box.map), 4),
            "box_mAP50": round(float(metrics.box.map50), 4),
        })

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"split": args.split, "labels": args.labels, "images": n_images, "results": rows}, f, indent=2)

    print(f"\n{args.sport} / {args.split} ({args.labels} labels, {n_images} images)")
    print(f"{'model':<40} {'pose mAP50-95':>14} {'pose mAP50':>11} {'box mAP50-95':>13}")
    for r in rows:
        name = os.path.relpath(r["model"]) if os.path.exists(r["model"]) else r["model"]
        print(f"{name:<40} {r['pose_mAP50-95']:>14.3f} {r['pose_mAP50']:>11.3f} {r['box_mAP50-95']:>13.3f}")
    print(f"\nmetrics -> {os.path.join(out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
