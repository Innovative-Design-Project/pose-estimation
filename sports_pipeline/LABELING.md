# Hand-labelling the basketball val/test sets

The test set is the only trustworthy measure of whether fine-tuning helped, so
it must be labelled by people and never trained on. Val is labelled the same
way and is used to pick the best checkpoint during training.

## Workflow

```bash
# 1. Clip-level split (once; re-running with the same seed gives the same split)
python sports_pipeline/split_dataset.py

# 2. Export each split, prefilled with yolo11m-pose skeletons to correct
python sports_pipeline/to_coco_keypoints.py --sport basketball --split test
python sports_pipeline/to_coco_keypoints.py --sport basketball --split val
#    -> data/gold/basketball/<split>/images/  and  coco_prefill.json

# 3. In CVAT: new task per split, upload images/, import coco_prefill.json as
#    "COCO Keypoints 1.0", correct every frame, export as "COCO Keypoints 1.0".

# 4. Import the corrected export as gold labels
python sports_pipeline/from_coco_keypoints.py --sport basketball --split test --coco-json <export.json>
python sports_pipeline/from_coco_keypoints.py --sport basketball --split val  --coco-json <export.json>

# 5. Baseline (and later: add the fine-tuned best.pt to --models)
python sports_pipeline/evaluate.py --split test
```

`build_dataset.py` picks up the gold val labels automatically and never copies
test frames into the training set.

## Labelling rules

Consistency matters more than any single rule. Everyone labelling should
follow the same ones.

- **Who to label:** every player and referee on court, plus anyone on the bench
  or sideline whose body is at least ~40 px tall. Skip crowd and anyone smaller.
  Unlabelled people the model detects count as false positives, so don't skip
  people inconsistently.
- **Bounding box:** tight around the whole visible body, including the raised
  arm or the ball-holding hands.
- **Visibility flags:** `2` = visible, `1` = occluded but you can confidently
  place it (e.g. a hip behind another player), `0` = not labelled (out of frame
  or a pure guess). Don't guess wildly; use `0` instead.
- **Left/right** is the person's own left/right, not the image's. Check this
  on players facing the camera. It's the most common prefill error.
- **Delete** prefilled skeletons on non-people (posters, screens, reflections)
  and **add** people the model missed entirely.
- **Hard cases to get right:** wrists/elbows on shots and rebounds, overlapping
  players in the paint, motion-blurred limbs. These are where fine-tuning
  should help most, so sloppy labels here hide the gain.

Expect roughly 1–2 minutes per frame when correcting prefills, so 3–5 hours per
250-frame split.
