"""Assemble the final fine-tuning dataset from (corrected) pseudo-labels.

Takes sports_pipeline/data/pseudo/<sport>/{images,labels}/ (or, after manual
review in Roboflow/CVAT, a corrected copy in the same YOLO-pose layout),
merges basketball + hockey, splits train/val, and writes a data.yaml usable
directly with `yolo pose train`.
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


def main():
    parser = argparse.ArgumentParser(description="Build the merged train/val sports-pose dataset")
    parser.add_argument("--sports", nargs="+", default=["basketball", "hockey"])
    parser.add_argument(
        "--source-root",
        default=os.path.join(os.path.dirname(__file__), "data", "pseudo"),
        help="Root containing <sport>/images and <sport>/labels (pseudo or corrected)",
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(os.path.dirname(__file__), "data", "sports_pose_dataset"),
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
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

        stems = sorted(os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt"))
        random.shuffle(stems)
        n_val = max(1, int(len(stems) * args.val_frac))
        val_stems = set(stems[:n_val])

        for stem in stems:
            split = "val" if stem in val_stems else "train"
            img_candidates = [f for f in os.listdir(img_dir) if os.path.splitext(f)[0] == stem]
            if not img_candidates:
                continue
            img_name = img_candidates[0]
            ext = os.path.splitext(img_name)[1]

            shutil.copy2(
                os.path.join(img_dir, img_name),
                os.path.join(args.out_root, "images", split, f"{sport}_{stem}{ext}"),
            )
            shutil.copy2(
                os.path.join(lbl_dir, f"{stem}.txt"),
                os.path.join(args.out_root, "labels", split, f"{sport}_{stem}.txt"),
            )
            if split == "train":
                total_train += 1
            else:
                total_val += 1

        print(f"[{sport}] {len(stems)} labeled frames -> {len(stems) - n_val} train / {n_val} val")

    yaml_path = os.path.join(args.out_root, "sports_pose.yaml")
    with open(yaml_path, "w") as f:
        f.write(DATA_YAML_TEMPLATE.format(root=os.path.abspath(args.out_root)))

    print(f"\nDataset ready: {total_train} train / {total_val} val frames")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
