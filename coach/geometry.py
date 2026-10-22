"""Keypoint indices and small geometry helpers (image coordinates, y grows downward)."""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SIDE = {
    "left": {"shoulder": L_SHOULDER, "elbow": L_ELBOW, "wrist": L_WRIST, "hip": L_HIP, "knee": L_KNEE, "ankle": L_ANKLE},
    "right": {"shoulder": R_SHOULDER, "elbow": R_ELBOW, "wrist": R_WRIST, "hip": R_HIP, "knee": R_KNEE, "ankle": R_ANKLE},
}

# MediaPipe hand landmark indices used for wrist analysis.
HAND_WRIST, HAND_MIDDLE_MCP, HAND_MIDDLE_TIP, HAND_INDEX_TIP = 0, 9, 12, 8

MIN_CONF = 0.4

Point = Tuple[float, float]


@dataclass
class FrameState:
    """Everything known about one analysed frame, after tracking and smoothing."""
    index: int                                   # frame index in the raw recording
    t: float                                     # seconds since session start
    kpts: Optional[np.ndarray] = None            # (17, 3) x, y, conf of the tracked player
    box: Optional[Tuple[float, float, float, float]] = None
    ball: Optional[Tuple[float, float, float]] = None   # x, y, radius
    hands: Dict[str, np.ndarray] = field(default_factory=dict)  # "left"/"right" -> (21, 3)


def pt(kpts: Optional[np.ndarray], i: int, min_conf: float = MIN_CONF) -> Optional[Point]:
    if kpts is None or kpts[i, 2] < min_conf:
        return None
    return float(kpts[i, 0]), float(kpts[i, 1])


def mid(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    if a is None:
        return b
    if b is None:
        return a
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def dist(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle(a: Optional[Point], b: Optional[Point], c: Optional[Point]) -> Optional[float]:
    """Angle ABC in degrees (at vertex b)."""
    if a is None or b is None or c is None:
        return None
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def lean_from_vertical(top: Optional[Point], bottom: Optional[Point]) -> Optional[float]:
    """Angle of the bottom->top segment from vertical, in degrees (0 = upright)."""
    if top is None or bottom is None:
        return None
    dx, dy = top[0] - bottom[0], bottom[1] - top[1]
    if dx == 0 and dy == 0:
        return None
    return abs(math.degrees(math.atan2(dx, dy)))


def joint_angle(kpts: Optional[np.ndarray], side: str, joint: str) -> Optional[float]:
    s = SIDE[side]
    if joint == "elbow":
        return angle(pt(kpts, s["shoulder"]), pt(kpts, s["elbow"]), pt(kpts, s["wrist"]))
    if joint == "knee":
        return angle(pt(kpts, s["hip"]), pt(kpts, s["knee"]), pt(kpts, s["ankle"]))
    if joint == "hip":
        return angle(pt(kpts, s["shoulder"]), pt(kpts, s["hip"]), pt(kpts, s["knee"]))
    raise ValueError(joint)


def body_height(kpts: Optional[np.ndarray]) -> Optional[float]:
    """Nose-to-ankle height in pixels (~0.9 of standing height); None if not visible."""
    nose = pt(kpts, NOSE)
    ankles = mid(pt(kpts, L_ANKLE), pt(kpts, R_ANKLE))
    if nose is None or ankles is None:
        return None
    h = ankles[1] - nose[1]
    return h if h > 0 else None


def hip_mid(kpts):
    return mid(pt(kpts, L_HIP), pt(kpts, R_HIP))


def shoulder_mid(kpts):
    return mid(pt(kpts, L_SHOULDER), pt(kpts, R_SHOULDER))


def ankle_mid(kpts):
    return mid(pt(kpts, L_ANKLE), pt(kpts, R_ANKLE))
