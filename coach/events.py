"""Shot, make/miss and dribble detection from tracked keypoints and the ball track.

Shots are triggered by pose (shooting wrist above the head with the arm
extended), which works even without a ball detector; the ball, when tracked,
refines the release frame and decides make/miss. All thresholds are in
seconds or body heights so they do not depend on fps or camera distance.
They are starting values and need tuning on real practice footage.
"""
import collections
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .geometry import (NOSE, SIDE, FrameState, ankle_mid, body_height, dist, hip_mid, joint_angle, pt,
                       shoulder_mid)
from .hoop import Hoop


@dataclass
class ShotEvent:
    kind: str                    # "jump_shot", "set_shot" or "layup"
    hand: str                    # "left" / "right": hand that released the ball
    release_index: int           # raw frame index
    release_t: float
    outcome: str                 # "make", "miss" or "unknown"
    frames: List[FrameState]     # window around the shot used for metrics
    release_pos: int             # position of the release frame inside `frames`
    ball_path: List[Tuple[float, float, float, float]] = field(default_factory=list)  # t, x, y, r after release


@dataclass
class DribbleEvent:
    index: int
    t: float
    hand: Optional[str]
    height: Optional[float]      # apex height above the ankles, in body heights


def judge_make(path: List[Tuple[float, float, float, float]], hoop: Optional[Hoop]) -> str:
    """Make if the ball crosses the rim plane going down inside the rim; miss if outside."""
    if hoop is None or len(path) < 2:
        return "unknown"
    margin = 0.15 * hoop.width
    rim_y = hoop.rim_y
    for (t0, x0, y0, _), (t1, x1, y1, _) in zip(path, path[1:]):
        if y0 < rim_y <= y1:  # descending through the rim plane
            x = x0 + (x1 - x0) * (rim_y - y0) / (y1 - y0)
            return "make" if hoop.x1 - margin <= x <= hoop.x2 + margin else "miss"
    return "unknown"


class ShotDetector:
    def __init__(self, shooting_hand: str = "right", hoop: Optional[Hoop] = None,
                 pre_s: float = 1.5, post_s: float = 1.0, flight_timeout_s: float = 2.5,
                 refractory_s: float = 1.5):
        self.shooting_hand, self.hoop = shooting_hand, hoop
        self.pre_s, self.post_s, self.flight_timeout_s, self.refractory_s = pre_s, post_s, flight_timeout_s, refractory_s
        self.buffer = collections.deque()
        self.pending = None          # (trigger_t, hand)
        self.last_trigger_t = -1e9

    @property
    def active(self) -> bool:
        """True while a shot is in progress (used to run hand landmarks only when useful)."""
        return self.pending is not None

    def _trigger(self, fs: FrameState) -> Optional[str]:
        k, bh = fs.kpts, body_height(fs.kpts)
        nose = pt(k, NOSE)
        if k is None or bh is None or nose is None:
            return None
        off_hand = "left" if self.shooting_hand == "right" else "right"
        if self._arm_shooting(k, self.shooting_hand, nose, bh):
            return self.shooting_hand
        # The guide hand rises with the ball on every jump shot, so the off hand only
        # counts as the shooter when the shooting hand is down (one-handed off-hand finish).
        shooting_wrist = pt(k, SIDE[self.shooting_hand]["wrist"])
        shoulder = pt(k, SIDE[self.shooting_hand]["shoulder"])
        shooting_down = shooting_wrist is not None and shoulder is not None and shooting_wrist[1] > shoulder[1]
        if shooting_down and self._arm_shooting(k, off_hand, nose, bh):
            return off_hand
        return None

    def _arm_shooting(self, k, hand: str, nose, bh: float) -> bool:
        wrist = pt(k, SIDE[hand]["wrist"])
        elbow_angle = joint_angle(k, hand, "elbow")
        if wrist is None or elbow_angle is None:
            return False
        return nose[1] - wrist[1] > 0.08 * bh and elbow_angle > 140 and self._rising(hand, bh)

    def _rising(self, hand: str, bh: float) -> bool:
        """Wrist moved up by >= 0.25 body heights within the last 0.5 s (a shooting motion, not a stretch)."""
        idx = SIDE[hand]["wrist"]
        now = self.buffer[-1]
        cur = pt(now.kpts, idx)
        lowest = None
        for fs in reversed(self.buffer):
            if now.t - fs.t > 0.5:
                break
            p = pt(fs.kpts, idx)
            if p is not None and (lowest is None or p[1] > lowest):
                lowest = p[1]
        return cur is not None and lowest is not None and lowest - cur[1] >= 0.25 * bh

    def update(self, fs: FrameState) -> Optional[ShotEvent]:
        self.buffer.append(fs)
        keep = self.pre_s + self.flight_timeout_s + 1.0
        while self.buffer and fs.t - self.buffer[0].t > keep:
            self.buffer.popleft()

        if self.pending is None:
            if fs.t - self.last_trigger_t >= self.refractory_s:
                hand = self._trigger(fs)
                if hand is not None:
                    self.pending = (fs.t, hand)
                    self.last_trigger_t = fs.t
            return None

        trigger_t, hand = self.pending
        elapsed = fs.t - trigger_t
        if elapsed < self.post_s:
            return None
        frames = [f for f in self.buffer if f.t >= trigger_t - self.pre_s]
        release_pos = self._release_pos(frames, trigger_t, hand)
        path = [(f.t, f.ball[0], f.ball[1], f.ball[2]) for f in frames[release_pos:] if f.ball is not None]
        outcome = judge_make(path, self.hoop)
        # With a ball and a rim we wait for the ball to resolve, up to the flight timeout.
        if outcome == "unknown" and self.hoop is not None and path and elapsed < self.flight_timeout_s:
            return None

        self.pending = None
        release = frames[release_pos]
        return ShotEvent(
            kind=classify_shot(frames, release_pos),
            hand=hand,
            release_index=release.index,
            release_t=release.t,
            outcome=outcome,
            frames=frames,
            release_pos=release_pos,
            ball_path=path,
        )

    def _release_pos(self, frames: List[FrameState], trigger_t: float, hand: str) -> int:
        widx = SIDE[hand]["wrist"]
        window = [i for i, f in enumerate(frames) if trigger_t - 0.5 <= f.t <= trigger_t + 0.4]
        if not window:
            return len(frames) - 1
        # 1) Ball leaves the hand: first frame where the ball is clearly away from both wrists, above the shoulders.
        for i in window:
            f = frames[i]
            bh = body_height(f.kpts)
            sh = shoulder_mid(f.kpts)
            if f.ball is None or bh is None or sh is None:
                continue
            wrists = [pt(f.kpts, SIDE[s]["wrist"]) for s in SIDE]
            gaps = [dist(w, f.ball[:2]) for w in wrists if w is not None]
            if gaps and min(gaps) > max(2.5 * f.ball[2], 0.12 * bh) and f.ball[1] < sh[1]:
                return max(window[0], i - 1)
        # 2) Pose only: first frame where the elbow is within 5% of its maximum extension.
        angles = [(i, joint_angle(frames[i].kpts, hand, "elbow")) for i in window]
        angles = [(i, a) for i, a in angles if a is not None]
        if angles:
            peak = max(a for _, a in angles)
            for i, a in angles:
                if a >= 0.95 * peak and pt(frames[i].kpts, widx) is not None:
                    return i
        return window[-1]


def classify_shot(frames: List[FrameState], release_pos: int) -> str:
    rel = frames[release_pos]
    bh = body_height(rel.kpts) or _median_body_height(frames)
    if not bh:
        return "jump_shot"
    approach = [f for f in frames[:release_pos + 1] if rel.t - f.t <= 1.0 and hip_mid(f.kpts) is not None]
    if len(approach) >= 2:
        dx = abs(hip_mid(approach[-1].kpts)[0] - hip_mid(approach[0].kpts)[0])
        dt = approach[-1].t - approach[0].t
        if dt > 0 and dx / dt / bh > 0.8:
            return "layup"
    return "jump_shot" if jump_height(frames, release_pos) >= 0.04 else "set_shot"


def _median_body_height(frames: List[FrameState]) -> Optional[float]:
    hs = sorted(h for h in (body_height(f.kpts) for f in frames) if h)
    return hs[len(hs) // 2] if hs else None


def jump_height(frames: List[FrameState], release_pos: int) -> float:
    """Hip rise above the standing baseline around release, in body heights."""
    bh = _median_body_height(frames)
    hips = [(f.t, hip_mid(f.kpts)) for f in frames]
    hips = [(t, h[1]) for t, h in hips if h is not None]
    if not bh or len(hips) < 3:
        return 0.0
    rel_t = frames[release_pos].t
    before = [y for t, y in hips if rel_t - 1.5 <= t <= rel_t - 0.6]
    around = [y for t, y in hips if abs(t - rel_t) <= 0.3]
    if not before or not around:
        return 0.0
    baseline = sorted(before)[len(before) // 2]
    return max(0.0, (baseline - min(around)) / bh)


class DribbleCounter:
    """Counts bounces: the ball's lowest image point below the hips, near the player."""

    def __init__(self, min_interval_s: float = 0.2):
        self.min_interval_s = min_interval_s
        self.track = collections.deque(maxlen=90)   # FrameState with a ball
        self.last_bounce_t = -1e9
        self.last_apex_y = None
        self.count = 0

    def update(self, fs: FrameState, shot_active: bool = False) -> Optional[DribbleEvent]:
        if fs.ball is None or fs.kpts is None:
            return None
        self.track.append(fs)
        if shot_active or len(self.track) < 3:
            return None
        a, b, c = self.track[-3], self.track[-2], self.track[-1]
        hip, ankles, bh = hip_mid(b.kpts), ankle_mid(b.kpts), body_height(b.kpts)
        if hip is None or bh is None:
            return None

        # Apex (highest point between bounces) gives the dribble height.
        if a.ball[1] > b.ball[1] <= c.ball[1]:
            self.last_apex_y = b.ball[1]

        is_bottom = a.ball[1] < b.ball[1] >= c.ball[1]
        below_hips = b.ball[1] > hip[1]
        near = b.box is not None and (b.box[0] - 0.5 * (b.box[2] - b.box[0]) <= b.ball[0]
                                      <= b.box[2] + 0.5 * (b.box[2] - b.box[0]))
        recent_top = min(f.ball[1] for f in self.track if b.t - f.t <= 0.35)
        amplitude = (b.ball[1] - recent_top) / bh
        if not (is_bottom and below_hips and near and amplitude > 0.12
                and b.t - self.last_bounce_t >= self.min_interval_s):
            return None

        self.last_bounce_t = b.t
        self.count += 1
        height = None
        if self.last_apex_y is not None and ankles is not None:
            height = (ankles[1] - self.last_apex_y) / bh
        return DribbleEvent(index=b.index, t=b.t, hand=self._pushing_hand(b), height=height)

    def _pushing_hand(self, bottom: FrameState) -> Optional[str]:
        """Hand closest to the ball during the 0.35 s before the bounce."""
        best, best_d = None, None
        for f in self.track:
            if not 0 <= bottom.t - f.t <= 0.35:
                continue
            for side, idx in SIDE.items():
                d = dist(pt(f.kpts, idx["wrist"]), f.ball[:2])
                if d is not None and (best_d is None or d < best_d):
                    best, best_d = side, d
        return best


def foot_plants(frames: List[FrameState], end_t: float, window_s: float = 1.5) -> List[Tuple[float, str]]:
    """Approximate foot plants (ankle at a local lowest point) in the window before `end_t`."""
    plants = []
    seq = [f for f in frames if end_t - window_s <= f.t <= end_t]
    for side in SIDE:
        idx = SIDE[side]["ankle"]
        ys = [(f.t, pt(f.kpts, idx), body_height(f.kpts)) for f in seq]
        ys = [(t, p[1], bh) for t, p, bh in ys if p is not None and bh]
        if len(ys) < 3:
            continue
        floor = max(y for _, y, _ in ys)
        for (_, y0, _), (t1, y1, bh), (_, y2, _) in zip(ys, ys[1:], ys[2:]):
            if y0 <= y1 >= y2 and floor - y1 <= 0.03 * bh:
                if not plants or plants[-1][1] != side or t1 - plants[-1][0] > 0.25:
                    plants.append((t1, side))
    plants.sort()
    # Merge consecutive plants of the same foot (one long contact).
    merged = []
    for t, side in plants:
        if merged and merged[-1][1] == side:
            continue
        merged.append((t, side))
    return merged
