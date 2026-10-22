"""Assemble the final fine-tuning dataset from (corrected) pseudo-labels.

Takes sports_pipeline/data/pseudo/<sport>/{images,labels}/ (or, after manual
review in Roboflow/CVAT, a corrected copy in the same YOLO-pose layout),
splits train/val, and writes a data.yaml usable directly with `yolo pose train`.

If split_dataset.py has been run for a sport, its clip-level split is used:
train = pseudo labels for the train clips, val = hand-labelled gold labels for
the val clips (falling back to pseudo labels, with a warning, until they exist).
Test clips are never copied, so the test set stays unseen. Without a split
file the old random frame-level split is used.
"""
import argparse
import os
import random
import shutil

DATA_YAML_TEMPLATE = """\
path: {root}
train: images/train
val: images/val

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
"""


def read_list(path: str) -> list:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def random_split_entries(img_dir: str, lbl_dir: str, val_frac: float) -> list:
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt"))
    random.shuffle(stems)
    val_stems = set(stems[:max(1, int(len(stems) * val_frac))])
    img_by_stem = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir)}
    return [
        ("val" if stem in val_stems else "train", os.path.join(img_dir, img_by_stem[stem]), os.path.join(lbl_dir, f"{stem}.txt"))
        for stem in stems if stem in img_by_stem
    ]


def clip_split_entries(sport: str, img_dir: str, lbl_dir: str, split_dir: str, gold_root: str) -> list:
    entries = []
    for name in read_list(os.path.join(split_dir, "train.txt")):
        lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(lbl):  # frames with no pseudo detections have nothing to train on
            entries.append(("train", os.path.join(img_dir, name), lbl))

    gold_img = os.path.join(gold_root, sport, "val", "images")
    gold_lbl = os.path.join(gold_root, sport, "val", "labels")
    val_names = read_list(os.path.join(split_dir, "val.txt"))
    if os.path.isdir(gold_lbl):
        for name in val_names:
            lbl = os.path.join(gold_lbl, os.path.splitext(name)[0] + ".txt")
            if os.path.exists(lbl):
                entries.append(("val", os.path.join(gold_img, name), lbl))
    else:
        print(f"[{sport}] WARNING: no gold val labels at {gold_lbl}; using pseudo labels for val. "
              "Val metrics will only measure agreement with the base model.")
        for name in val_names:
            lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            if os.path.exists(lbl):
                entries.append(("val", os.path.join(img_dir, name), lbl))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Build the merged train/val sports-pose dataset")
    parser.add_argument("--sports", nargs="+", default=["basketball"])
    parser.add_argument(
        "--source-root",
        default=os.path.join(os.path.dirname(__file__), "data", "pseudo"),
        help="Root containing <sport>/images and <sport>/labels (pseudo or corrected)",
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(os.path.dirname(__file__), "data", "sports_pose_dataset"),
    )
    parser.add_argument("--splits-root", default=os.path.join(os.path.dirname(__file__), "data", "splits"))
    parser.add_argument("--gold-root", default=os.path.join(os.path.dirname(__file__), "data", "gold"))
    parser.add_argument("--val-frac", type=float, default=0.15, help="Only used when no split file exists")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out_root, "labels", split), exist_ok=True)

    total_train, total_val = 0, 0
    for sport in args.sports:
        img_dir = os.path.join(args.source_root, sport, "images")
        lbl_dir = os.path.join(args.source_root, sport, "labels")
        if not os.path.isdir(img_dir):
            print(f"[{sport}] skipping, no images at {img_dir}")
            continue

        split_dir = os.path.join(args.splits_root, sport)
        if os.path.exists(os.path.join(split_dir, "train.txt")):
            entries = clip_split_entries(sport, img_dir, lbl_dir, split_dir, args.gold_root)
        else:
            entries = random_split_entries(img_dir, lbl_dir, args.val_frac)

        counts = {"train": 0, "val": 0}
        for split, img_path, lbl_path in entries:
            stem, ext = os.path.splitext(os.path.basename(img_path))
            shutil.copy2(img_path, os.path.join(args.out_root, "images", split, f"{sport}_{stem}{ext}"))
            shutil.copy2(lbl_path, os.path.join(args.out_root, "labels", split, f"{sport}_{stem}.txt"))
            counts[split] += 1
        total_train += counts["train"]
        total_val += counts["val"]

        print(f"[{sport}] {counts['train']} train / {counts['val']} val")

    yaml_path = os.path.join(args.out_root, "sports_pose.yaml")
    with open(yaml_path, "w") as f:
        f.write(DATA_YAML_TEMPLATE.format(root=os.path.abspath(args.out_root)))

    print(f"\nDataset ready: {total_train} train / {total_val} val frames")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
