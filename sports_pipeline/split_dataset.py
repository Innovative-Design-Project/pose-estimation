"""Split basketball frames into train / val / test by source clip, not by frame.

The Roboflow pool is a mix of video frames (MOV_0023_mp4-0011, 2-2_1_105, ...),
photo bursts and scraped stills, with no explicit video ids. Splitting frame by
frame would put near-identical neighbouring frames on both sides and inflate
val/test scores, so frames are first grouped into "clips":

  1. Filename grouping: strip the export-split prefix, Roboflow's `.rf.<hash>`
     suffix (the same source image can appear several times) and the trailing
     frame index, e.g. `test_MOV_0023_mp4-0011_jpg.rf.ab12.jpg` -> `MOV_0023_mp4`.
  2. Near-duplicate merging: a 64-bit dHash per image; any two frames within
     `--dup-threshold` bits are unioned into the same clip, which catches
     consecutive frames whose names carry no clip id (pure numbers, etc.).

Whole clips are then assigned to test, then val, then train. val/test are
drawn only from small clips (<= --max-group frames) so they cover many
different games/scenes instead of one long clip, and only from frames whose
Roboflow detection labels contain at least one Player/Referee (an unbiased
filter -- it does not depend on the pose model being evaluated).

val/test are the frames to hand-label in CVAT (see LABELING.md). Output:
  data/splits/basketball/{train,val,test}.txt   one image filename per line
  data/splits/basketball/groups.json            clip id -> filenames
"""
import argparse
import json
import os
import random
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

EXPORT_SPLITS = ("train", "valid", "test")
PERSON_CLASS_IDS = {1, 2}  # Roboflow export classes: 0 Ball, 1 Player, 2 Referee
IMG_EXTS = (".jpg", ".jpeg", ".png")


def source_name(fname: str) -> Tuple[str, str]:
    """`test_MOV_0023_mp4-0011_jpg.rf.ab12.jpg` -> ("test", "MOV_0023_mp4-0011")."""
    export_split, rest = fname.split("_", 1)
    rest = re.sub(r"\.rf\.[0-9a-f]+\.\w+$", "", rest)
    rest = re.sub(r"_(jpg|jpeg|png|JPEG|JPG|PNG)$", "", rest)
    return export_split, rest


def clip_key(name: str) -> Optional[str]:
    """Best-effort clip id from a source name; None if the name carries none."""
    m = re.match(r"^(.*_mp4)-\d+$", name)  # MOV_0023_mp4-0011, 00800_mp4-10
    if m:
        return m.group(1)
    m = re.match(r"^(IMG_\d{8})_\d+", name)  # IMG_20201010_103033 -> same-day session
    if m:
        return m.group(1)
    m = re.match(r"^(\d+-\d+_\d+)_\d+$", name)  # 2-2_1_105 -> 2-2_1
    if m:
        return m.group(1)
    # basketball_36 -> "basketball"; pure numbers (0282, -102) carry no clip id.
    return re.sub(r"[\d_\-]+$", "", name) or None


def dhash(path: str, size: int = 8) -> int:
    with Image.open(path) as im:
        px = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def has_person(export_root: str, export_split: str, fname: str) -> bool:
    orig = fname.split("_", 1)[1]
    lbl = os.path.join(export_root, export_split, "labels", os.path.splitext(orig)[0] + ".txt")
    if not os.path.exists(lbl):
        return False
    with open(lbl) as f:
        return any(line.split() and int(line.split()[0]) in PERSON_CLASS_IDS for line in f)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int):
        self.parent[self.find(a)] = self.find(b)


def build_groups(img_dir: str, files: List[str], dup_threshold: int) -> List[List[str]]:
    uf = UnionFind(len(files))

    by_key = {}
    for i, f in enumerate(files):
        _, name = source_name(f)
        key = clip_key(name)
        # Names without a clip id still group with their own Roboflow duplicates.
        by_key.setdefault(key if key is not None else f"__img__{name}", []).append(i)
    for idxs in by_key.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    print(f"hashing {len(files)} frames for near-duplicate merging...")
    hashes = np.array([dhash(os.path.join(img_dir, f)) for f in files], dtype=np.uint64)
    merged = 0
    for i in range(len(files)):
        xor = hashes[i + 1:] ^ hashes[i]
        dist = np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(dist <= dup_threshold)[0]:
            if uf.find(i) != uf.find(i + 1 + j):
                uf.union(i, i + 1 + j)
                merged += 1
    print(f"near-duplicate merges: {merged}")

    groups = {}
    for i, f in enumerate(files):
        groups.setdefault(uf.find(i), []).append(f)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def take_groups(groups, target: int, max_group: int, eligible: set) -> Tuple[list, list]:
    """Fill a split with whole small groups until `target` eligible frames are reached."""
    picked, rest, n = [], [], 0
    for g in groups:
        n_eligible = sum(f in eligible for f in g)
        if n < target and n_eligible and len(g) <= max_group:
            picked.append(g)
            n += n_eligible
        else:
            rest.append(g)
    return picked, rest


def main():
    parser = argparse.ArgumentParser(description="Clip-level train/val/test split for basketball frames")
    parser.add_argument("--data-root", default=os.path.join(os.path.dirname(__file__), "data"))
    parser.add_argument("--sport", default="basketball")
    parser.add_argument("--test-size", type=int, default=250, help="Frames to hand-label for the test set")
    parser.add_argument("--val-size", type=int, default=250, help="Frames to hand-label for the val set")
    parser.add_argument("--max-group", type=int, default=15, help="Largest clip allowed into val/test")
    parser.add_argument("--dup-threshold", type=int, default=6, help="dHash Hamming distance to treat as same clip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = os.path.join(args.data_root, "raw", args.sport)
    img_dir = os.path.join(raw, "images")
    export_root = os.path.join(raw, "_export")
    out_dir = os.path.join(args.data_root, "splits", args.sport)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    eligible = {f for f in files if has_person(export_root, source_name(f)[0], f)}
    print(f"{len(files)} frames, {len(eligible)} with >=1 Player/Referee box")

    groups = build_groups(img_dir, files, args.dup_threshold)
    sizes = sorted((len(g) for g in groups), reverse=True)
    print(f"{len(groups)} clips (largest: {sizes[:5]}, singletons: {sizes.count(1)})")

    random.Random(args.seed).shuffle(groups)
    test_groups, groups = take_groups(groups, args.test_size, args.max_group, eligible)
    val_groups, train_groups = take_groups(groups, args.val_size, args.max_group, eligible)

    splits = {
        # Ineligible frames in val/test clips are dropped rather than moved to
        # train, so no clip ever straddles the split boundary.
        "test": sorted(f for g in test_groups for f in g if f in eligible),
        "val": sorted(f for g in val_groups for f in g if f in eligible),
        "train": sorted(f for g in train_groups for f in g),
    }
    for name, frames in splits.items():
        with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(frames) + "\n")

    all_groups = sorted(test_groups + val_groups + train_groups, key=lambda g: g[0])
    with open(os.path.join(out_dir, "groups.json"), "w") as f:
        json.dump({f"clip_{i:05d}": g for i, g in enumerate(all_groups)}, f, indent=1)

    print(
        f"test: {len(splits['test'])} frames / {len(test_groups)} clips | "
        f"val: {len(splits['val'])} frames / {len(val_groups)} clips | "
        f"train: {len(splits['train'])} frames / {len(train_groups)} clips"
    )
    print(f"split files -> {out_dir}")


if __name__ == "__main__":
    main()
