"""Per-shot form metrics from the frames around a ShotEvent.

Angles are 2D (image plane), so they are only comparable between shots filmed
from the same camera position. Lengths are normalised by body height (nose to
ankles) so they do not depend on distance to the camera.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .events import ShotEvent, _median_body_height, foot_plants, jump_height
from .geometry import (NOSE, SIDE, FrameState, ankle_mid, hip_mid, joint_angle, lean_from_vertical, pt,
                       shoulder_mid)
from .hands import fingers_down, wrist_flexion


def _window(frames: List[FrameState], t0: float, t1: float) -> List[FrameState]:
    return [f for f in frames if t0 <= f.t <= t1]


def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _round(v, nd=2):
    return None if v is None else round(v, nd)


def shot_metrics(shot: ShotEvent) -> Tuple[Dict[str, Optional[float]], Dict[str, int]]:
    """Returns (metrics, key frame indices). Missing values are None."""
    frames, rp, hand = shot.frames, shot.release_pos, shot.hand
    rel = frames[rp]
    s = SIDE[hand]
    bh = _median_body_height(frames[:rp + 1]) or _median_body_height(frames)
    m: Dict[str, Optional[float]] = {}
    keys = {"release": rel.index}

    # Set point: most bent elbow while the ball/wrist is above the shoulder, just before release.
    loaded = [f for f in _window(frames, rel.t - 0.6, rel.t)
              if pt(f.kpts, s["wrist"]) and pt(f.kpts, s["shoulder"])
              and pt(f.kpts, s["wrist"])[1] < pt(f.kpts, s["shoulder"])[1]]
    loaded = loaded or _window(frames, rel.t - 0.4, rel.t)
    elbows = [(f, joint_angle(f.kpts, hand, "elbow")) for f in loaded]
    elbows = [(f, a) for f, a in elbows if a is not None]
    if elbows:
        set_frame, m["elbow_set"] = min(elbows, key=lambda fa: fa[1])
        keys["set"] = set_frame.index
    # The arm finishes extending as (or just after) the ball leaves, so take the peak shortly after release.
    ext = [joint_angle(f.kpts, hand, "elbow") for f in _window(frames, rel.t, rel.t + 0.15)]
    ext = [a for a in ext if a is not None]
    m["elbow_release"] = max(ext) if ext else None

    dip = _window(frames, rel.t - 1.0, rel.t)
    knees = [_mean([joint_angle(f.kpts, "left", "knee"), joint_angle(f.kpts, "right", "knee")]) for f in dip]
    knees = [k for k in knees if k is not None]
    m["knee_bend"] = min(knees) if knees else None

    nose, wrist = pt(rel.kpts, NOSE), pt(rel.kpts, s["wrist"])
    m["release_height"] = (nose[1] - wrist[1]) / bh if nose and wrist and bh else None

    m["jump_height"] = jump_height(frames, rp)
    if m["jump_height"] >= 0.04:
        hips = [(f.t, hip_mid(f.kpts)) for f in _window(frames, rel.t - 0.5, rel.t + 0.5)]
        hips = [(t, h[1]) for t, h in hips if h is not None]
        if hips:
            apex_t = min(hips, key=lambda th: th[1])[0]
            m["release_timing"] = rel.t - apex_t  # < 0: released on the way up
    m.setdefault("release_timing", None)

    # Follow-through: how long the shooting wrist stays above the head after release.
    hold_end = rel.t
    for f in frames[rp:]:
        w, n = pt(f.kpts, s["wrist"]), pt(f.kpts, NOSE)
        if w is None or n is None:
            continue
        if w[1] > n[1]:
            break
        hold_end = f.t
    m["follow_through"] = hold_end - rel.t
    keys["follow_through"] = min(frames[rp:], key=lambda f: abs(f.t - (rel.t + 0.3))).index

    m["body_lean"] = lean_from_vertical(shoulder_mid(rel.kpts), hip_mid(rel.kpts))

    if shot.kind != "layup" and bh:
        before = [ankle_mid(f.kpts) for f in _window(frames, rel.t - 0.6, rel.t - 0.3)]
        after = [ankle_mid(f.kpts) for f in _window(frames, rel.t + 0.5, rel.t + 0.8)]
        bx, ax = _mean([p[0] for p in before if p]), _mean([p[0] for p in after if p])
        m["landing_drift"] = abs(ax - bx) / bh if bx is not None and ax is not None else None
    else:
        m["landing_drift"] = None

    m["arc_angle"] = launch_angle(shot.ball_path)

    flex = [(f, wrist_flexion(f.kpts, f.hands.get(hand), hand)) for f in _window(frames, rel.t, rel.t + 0.5)]
    flex = [(f, a) for f, a in flex if a is not None]
    if flex:
        snap_frame, m["wrist_flexion"] = max(flex, key=lambda fa: fa[1])
        m["fingers_down"] = float(fingers_down(snap_frame.hands.get(hand)))
    else:
        m["wrist_flexion"] = m["fingers_down"] = None

    if shot.kind == "layup":
        plants = [p for p in foot_plants(frames, rel.t, 1.5) if p[0] <= rel.t]
        m["steps"] = float(len([p for p in plants if rel.t - p[0] <= 1.2])) if plants else None
        m["takeoff_left_foot"] = float(plants[-1][1] == "left") if plants else None
    return {k: _round(v) for k, v in m.items()}, keys


def launch_angle(path) -> Optional[float]:
    """Ball launch angle above horizontal (degrees) at release.

    Gravity bends the path a lot within a few frames (in pixels it is ~2000+
    px/s^2 at courtside scale), so the angle is taken from a fitted parabola's
    velocity at release, not from the chord between two points.
    """
    if len(path) < 2:
        return None
    t0 = path[0][0]
    early = [p for p in path if p[0] - t0 <= 0.4]
    if len(early) >= 4:
        ts = np.array([p[0] - t0 for p in early])
        vx = np.polyfit(ts, [p[1] for p in early], 1)[0]
        vy = np.polyfit(ts, [p[2] for p in early], 2)[1]   # dy/dt at t0
    else:
        (ta, xa, ya, _), (tb, xb, yb, _) = early[0], early[-1] if len(early) > 1 else path[1]
        if tb <= ta:
            return None
        vx, vy = (xb - xa) / (tb - ta), (yb - ya) / (tb - ta)
    if vx == 0 and vy == 0:
        return None
    return math.degrees(math.atan2(-vy, abs(vx)))
