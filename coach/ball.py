"""Ball detection (YOLO detector trained by sports_pipeline/train_ball.py) and tracking."""
import collections
import math
import os
from typing import List, Optional, Tuple

Ball = Tuple[float, float, float]  # x, y, radius (pixels)

DEFAULT_BALL_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sports_pipeline", "runs", "ball_detector", "weights", "best.pt",
)


class BallDetector:
    """Thin wrapper; `available` is False when no trained weights exist yet."""

    def __init__(self, weights: str = DEFAULT_BALL_MODEL, device: str = "cpu", conf: float = 0.25, imgsz: int = 640):
        self.available = bool(weights) and os.path.exists(weights)
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self.model = None
        if self.available:
            from ultralytics import YOLO
            self.model = YOLO(weights)
        elif weights:
            print(f"Ball detector weights not found at {weights}; ball features are disabled. "
                  "Train them with: python sports_pipeline/train_ball.py")

    def __call__(self, frame) -> List[Tuple[float, float, float, float]]:
        """Returns [(x, y, radius, conf), ...]."""
        if not self.available:
            return []
        result = self.model(frame, device=self.device, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1 + y2 - y1) / 4, c))
        return out


class BallTracker:
    """Keeps one ball track with a constant-velocity gate, so stray detections are ignored."""

    def __init__(self, max_gap_s: float = 0.4, gate_radii: float = 6.0, history_s: float = 4.0):
        self.max_gap_s, self.gate_radii = max_gap_s, gate_radii
        self.history = collections.deque()  # (t, x, y, r)
        self.history_s = history_s

    def _predict(self, t: float) -> Optional[Ball]:
        if not self.history:
            return None
        t1, x1, y1, r1 = self.history[-1]
        if t - t1 > self.max_gap_s:
            return None
        if len(self.history) >= 2:
            t0, x0, y0, _ = self.history[-2]
            if t1 > t0:
                vx, vy = (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)
                return x1 + vx * (t - t1), y1 + vy * (t - t1), r1
        return x1, y1, r1

    def update(self, detections: List[Tuple[float, float, float, float]], t: float,
               near_box: Optional[Tuple[float, float, float, float]] = None) -> Optional[Ball]:
        """`near_box` (the player's box): a new track may only start close to the player,
        which rejects orange/red clutter (crowd, jerseys, signs) far from the action."""
        pred = self._predict(t)
        best = None
        if pred is not None:
            gate = self.gate_radii * max(pred[2], 8.0) + 400 * (t - self.history[-1][0])
            near = [d for d in detections if math.hypot(d[0] - pred[0], d[1] - pred[1]) <= gate]
            if near:
                best = min(near, key=lambda d: math.hypot(d[0] - pred[0], d[1] - pred[1]))
        else:
            if near_box is not None:
                x1, y1, x2, y2 = near_box
                w, h = x2 - x1, y2 - y1
                detections = [d for d in detections
                              if x1 - 0.75 * w <= d[0] <= x2 + 0.75 * w and y1 - 0.5 * h <= d[1] <= y2 + 0.1 * h]
            if detections:
                best = max(detections, key=lambda d: d[3])

        while self.history and t - self.history[0][0] > self.history_s:
            self.history.popleft()
        if best is None:
            return None
        self.history.append((t, best[0], best[1], best[2]))
        return best[0], best[1], best[2]
