"""Hand landmarks (21 per hand, MediaPipe) on crops around the player's wrists.

COCO keypoints stop at the wrist, so wrist snap and finger direction need a
hand model. Running it on the whole frame is slow and finds tiny hands poorly,
so each wrist gets its own square crop sized from the forearm length, and the
detected hand closest to the pose wrist is kept. Results are in full-frame
pixel coordinates.

At typical courtside distances a hand is only ~25-35 px tall in 720p, so treat
finger-level numbers as experimental until validated on real footage.
"""
import os
import urllib.request
from typing import Dict, Optional

import cv2
import numpy as np

from .geometry import HAND_MIDDLE_MCP, HAND_MIDDLE_TIP, SIDE, angle, dist, pt

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "hand_landmarker.task")


def ensure_model(path: str = DEFAULT_MODEL) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading MediaPipe hand model to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandAnalyzer:
    def __init__(self, model_path: str = DEFAULT_MODEL, min_crop: int = 96, crop_scale: float = 1.6):
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp_image_cls = __import__("mediapipe").Image
        self.image_format = __import__("mediapipe").ImageFormat.SRGB
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.min_crop, self.crop_scale = min_crop, crop_scale

    def __call__(self, frame, kpts: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Returns {"left"/"right": (21, 3) array of x, y (pixels), z} for hands found."""
        hands = {}
        if kpts is None:
            return hands
        h, w = frame.shape[:2]
        for side, idx in SIDE.items():
            wrist, elbow = pt(kpts, idx["wrist"]), pt(kpts, idx["elbow"])
            if wrist is None:
                continue
            forearm = dist(wrist, elbow) or self.min_crop / self.crop_scale
            size = int(max(self.min_crop, forearm * self.crop_scale))
            # Centre the crop beyond the wrist, along the forearm, where the hand is.
            cx, cy = wrist
            if elbow is not None:
                cx += (wrist[0] - elbow[0]) * 0.35
                cy += (wrist[1] - elbow[1]) * 0.35
            x1, y1 = int(max(0, cx - size / 2)), int(max(0, cy - size / 2))
            x2, y2 = int(min(w, x1 + size)), int(min(h, y1 + size))
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            result = self.landmarker.detect(self.mp_image_cls(image_format=self.image_format,
                                                              data=np.ascontiguousarray(crop)))
            best, best_d = None, None
            for lms in result.hand_landmarks:
                arr = np.array([[x1 + lm.x * (x2 - x1), y1 + lm.y * (y2 - y1), lm.z] for lm in lms])
                d = dist((arr[0, 0], arr[0, 1]), wrist)
                if best_d is None or d < best_d:
                    best, best_d = arr, d
            # Reject a hand whose wrist is far from the pose wrist (probably the other hand).
            if best is not None and best_d <= 0.6 * forearm:
                hands[side] = (best, best_d)

        # Overlapping crops can find the same physical hand twice: keep the closer match.
        if len(hands) == 2:
            (l, ld), (r, rd) = hands["left"], hands["right"]
            palm = dist(tuple(l[0, :2]), tuple(l[HAND_MIDDLE_MCP, :2])) or 1.0
            if dist(tuple(l[0, :2]), tuple(r[0, :2])) < palm:  # wrists closer than one palm length
                hands.pop("left" if ld > rd else "right")
        return {side: arr for side, (arr, _) in hands.items()}

    def close(self):
        self.landmarker.close()


def wrist_flexion(kpts: Optional[np.ndarray], hand: Optional[np.ndarray], side: str) -> Optional[float]:
    """Bend between forearm and hand in degrees (0 = straight, ~60-90 = full 'gooseneck' snap)."""
    if hand is None:
        return None
    s = SIDE[side]
    elbow, wrist = pt(kpts, s["elbow"]), pt(kpts, s["wrist"])
    mcp = (float(hand[HAND_MIDDLE_MCP, 0]), float(hand[HAND_MIDDLE_MCP, 1]))
    straight = angle(elbow, wrist, mcp)
    return None if straight is None else 180.0 - straight


def fingers_down(hand: Optional[np.ndarray]) -> Optional[bool]:
    """True when the middle fingertip is below its knuckle (fingers pointing at the floor)."""
    if hand is None:
        return None
    return bool(hand[HAND_MIDDLE_TIP, 1] > hand[HAND_MIDDLE_MCP, 1])


HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]


def draw_hands(frame, hands: Dict[str, np.ndarray], color=(255, 200, 0)):
    for arr in hands.values():
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, (int(arr[a, 0]), int(arr[a, 1])), (int(arr[b, 0]), int(arr[b, 1])), color, 1, cv2.LINE_AA)
        for x, y, _ in arr:
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)
