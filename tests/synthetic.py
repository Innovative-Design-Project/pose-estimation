"""Synthetic side-view shooter for testing event detection and metrics without footage.

A right-handed player faces +x, body height (nose to ankles) 400 px, rim to
the right. Timeline: hold the ball (0-0.9 s), dip (0.9-1.15 s), lift to the
set point (1.15-1.32 s), extend and release (~1.42 s), follow-through, arms
down from 2.0 s. A small jump peaks at 1.4 s. The ball flies on a parabola
with a chosen launch angle toward a target point.
"""
import math
from typing import List, Optional, Tuple

import numpy as np

from coach.geometry import FrameState
from coach.hoop import Hoop

HOOP = Hoop(1060, 244, 1140, 256)          # rim centre (1100, 250)
UPPER, FORE, THIGH = 70.0, 65.0, 115.0
ANKLE_Y, BASE_HIP_Y = 600.0, 380.0


def _lerp(a, b, u):
    u = max(0.0, min(1.0, u))
    return a + (b - a) * u


def _arm(shoulder, theta_u, beta):
    """Elbow and wrist from upper-arm angle (deg, up = positive) and elbow interior angle."""
    tu = math.radians(theta_u)
    elbow = (shoulder[0] + UPPER * math.cos(tu), shoulder[1] - UPPER * math.sin(tu))
    tf = math.radians(theta_u + 180 - beta)
    wrist = (elbow[0] + FORE * math.cos(tf), elbow[1] - FORE * math.sin(tf))
    return elbow, wrist


def _arm_params(t):
    if t < 1.15:
        return -70.0, 80.0
    if t < 1.32:
        u = (t - 1.15) / 0.17
        return _lerp(-70, 35, u), _lerp(80, 90, u)
    if t < 1.45:
        u = (t - 1.32) / 0.13
        return _lerp(35, 70, u), _lerp(90, 175, u)
    if t < 2.0:
        return 70.0, 175.0
    u = (t - 2.0) / 0.3
    return _lerp(70, -70, u), _lerp(175, 80, u)


def _body_offset(t, jump_px):
    """(dip of the hips, rise of the whole body) at time t."""
    dip = 0.0
    if 0.9 <= t < 1.15:
        dip = _lerp(0, 30, (t - 0.9) / 0.25)
    elif 1.15 <= t < 1.3:
        dip = _lerp(30, 0, (t - 1.15) / 0.15)
    rise = jump_px * max(0.0, 1 - ((t - 1.4) / 0.2) ** 2) if 1.2 <= t <= 1.6 else 0.0
    return dip, rise


def _kpts(t, jump_px=25.0, ball_xy=None, released=False):
    dip, rise = _body_offset(t, jump_px)
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    hip = (630.0, BASE_HIP_Y + dip - rise)
    ankle = (630.0, ANKLE_Y - rise)
    d = math.dist(hip, ankle)
    knee_dx = math.sqrt(max(THIGH ** 2 - (d / 2) ** 2, 0))
    knee = (630.0 + knee_dx, (hip[1] + ankle[1]) / 2)
    torso_up = BASE_HIP_Y - 245.0
    shoulder = (630.0, hip[1] - torso_up + dip * 0.6)   # torso drops a bit less than the hips
    nose = (645.0, shoulder[1] - 45)

    k[0, :2] = nose
    k[1, :2] = (648, nose[1] - 6); k[2, :2] = (642, nose[1] - 6)
    k[3, :2] = (630, nose[1] - 2); k[4, :2] = (628, nose[1] - 2)
    k[5, :2] = (shoulder[0] - 5, shoulder[1]); k[6, :2] = shoulder
    k[11, :2] = (hip[0] - 5, hip[1]); k[12, :2] = hip
    k[13, :2] = (knee[0] - 5, knee[1]); k[14, :2] = knee
    k[15, :2] = (ankle[0] - 5, ankle[1]); k[16, :2] = ankle

    theta_u, beta = _arm_params(t)
    r_elbow, r_wrist = _arm(shoulder, theta_u, beta)
    k[8, :2], k[10, :2] = r_elbow, r_wrist
    # Guide hand on the side of the ball until release, then drops.
    if not released and ball_xy is not None:
        l_wrist = (ball_xy[0] - 14, ball_xy[1] + 6)
    else:
        l_wrist = (shoulder[0] + 10, shoulder[1] + 90)
    k[9, :2] = l_wrist
    k[7, :2] = ((shoulder[0] + l_wrist[0]) / 2 - 10, (shoulder[1] + l_wrist[1]) / 2 + 10)
    return k, r_wrist


def shot_frames(fps: float = 30.0, launch_deg: float = 50.0, target: Optional[Tuple[float, float]] = None,
                duration: float = 3.5, jump_px: float = 25.0, release_t: float = 1.42,
                with_ball: bool = True) -> List[FrameState]:
    target = target or HOOP.center
    g = 1500.0
    frames, origin = [], None
    for i in range(int(duration * fps)):
        t = i / fps
        released = t >= release_t
        k, wrist = _kpts(t, jump_px, released=released)
        ball = (wrist[0] + 6, wrist[1] - 16)
        if released:
            if origin is None:
                _, w0 = _kpts(release_t, jump_px)
                x0, y0 = w0[0] + 6, w0[1] - 16
                th = math.radians(launch_deg)
                dx = target[0] - x0
                v = math.sqrt(0.5 * g * dx ** 2 / (math.cos(th) ** 2 * (target[1] - y0 + dx * math.tan(th))))
                origin = (x0, y0, v * math.cos(th), v * math.sin(th))
            x0, y0, vx, vy = origin
            dt = t - release_t
            ball = (x0 + vx * dt, y0 - vy * dt + 0.5 * g * dt ** 2)
        k, _ = _kpts(t, jump_px, ball_xy=ball, released=released)
        on_screen = with_ball and -50 < ball[1] < 760
        frames.append(FrameState(index=i, t=t, kpts=k, box=(560.0, 140.0, 760.0, 610.0),
                                 ball=(ball[0], ball[1], 12.0) if on_screen else None))
    return frames


def dribble_frames(fps: float = 30.0, bounces: int = 10, period: float = 0.5, top_y: float = 430.0,
                   floor_y: float = 600.0) -> List[FrameState]:
    frames = []
    for i in range(int((bounces * period + 0.1) * fps)):
        t = i / fps
        phase = (t % period) / period
        y = top_y + (floor_y - top_y) * (1 - abs(1 - 2 * phase))    # top -> floor -> top
        k, _ = _kpts(0.0)
        ball_x = 700.0
        k[10, :2] = (ball_x, top_y - 10)     # right hand pushes the ball
        frames.append(FrameState(index=i, t=t, kpts=k, box=(560.0, 140.0, 760.0, 610.0), ball=(ball_x, y, 12.0)))
    return frames
