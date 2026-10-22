"""Lock onto the practising player and smooth their keypoints over time."""
import math
from typing import List, Optional, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012): low jitter when still, low lag when moving fast."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None or t <= self.t_prev:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        a = self._alpha(self.min_cutoff + self.beta * abs(dx_hat), dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class KeypointSmoother:
    """Per-keypoint One-Euro filtering; low-confidence points are passed through and reset."""

    def __init__(self, n_kpts: int = 17, min_conf: float = 0.3, **filter_kwargs):
        self.min_conf = min_conf
        self.kwargs = filter_kwargs
        self.filters = [[OneEuroFilter(**filter_kwargs), OneEuroFilter(**filter_kwargs)] for _ in range(n_kpts)]

    def reset(self):
        self.filters = [[OneEuroFilter(**self.kwargs), OneEuroFilter(**self.kwargs)] for _ in self.filters]

    def __call__(self, kpts: np.ndarray, t: float) -> np.ndarray:
        out = kpts.copy()
        for i, (fx, fy) in enumerate(self.filters):
            if kpts[i, 2] < self.min_conf:
                # Don't let a missing joint drag the filter; restart it when the joint returns.
                fx.t_prev = fy.t_prev = None
                continue
            out[i, 0] = fx(float(kpts[i, 0]), t)
            out[i, 1] = fy(float(kpts[i, 1]), t)
        return out


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


class PlayerTracker:
    """Follows one person across frames.

    The first player is the largest confident person (the one nearest the
    tripod). After that the lock follows box overlap, so a bystander walking
    closer to the camera does not steal it. After `max_lost` frames without a
    match the lock is released and re-acquired.
    """

    def __init__(self, min_iou: float = 0.2, max_lost: int = 20, min_conf: float = 0.4):
        self.min_iou, self.max_lost, self.min_conf = min_iou, max_lost, min_conf
        self.box: Optional[Box] = None
        self.lost = 0
        self.smoother = KeypointSmoother()

    def update(self, boxes: List[Box], confs: List[float], kpts: List[np.ndarray], t: float):
        """Returns (box, smoothed kpts) of the tracked player, or (None, None)."""
        candidates = [i for i, c in enumerate(confs) if c >= self.min_conf]
        chosen = None
        if self.box is not None and candidates:
            best = max(candidates, key=lambda i: iou(self.box, boxes[i]))
            if iou(self.box, boxes[best]) >= self.min_iou:
                chosen = best
        if chosen is None:
            self.lost += 1
            if self.box is not None and self.lost <= self.max_lost:
                return None, None
            if not candidates:
                self.box = None
                return None, None
            chosen = max(candidates, key=lambda i: box_area(boxes[i]) * confs[i])
            self.smoother.reset()

        self.lost = 0
        self.box = tuple(boxes[chosen])
        return self.box, self.smoother(np.asarray(kpts[chosen], dtype=float), t)
