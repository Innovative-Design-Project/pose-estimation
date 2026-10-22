import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coach.coaching import Coach, issues  # noqa: E402
from coach.events import DribbleCounter, ShotDetector, judge_make  # noqa: E402
from coach.geometry import angle  # noqa: E402
from coach.hoop import Hoop  # noqa: E402
from coach.metrics import shot_metrics  # noqa: E402
from coach.store import SessionStore  # noqa: E402
from coach.tracking import OneEuroFilter, PlayerTracker  # noqa: E402
from tests.synthetic import HOOP, dribble_frames, shot_frames  # noqa: E402


def run_shots(frames, hoop=HOOP):
    det = ShotDetector(shooting_hand="right", hoop=hoop)
    return [s for s in (det.update(f) for f in frames) if s is not None]


def test_angle():
    assert angle((0, 1), (0, 0), (1, 0)) == pytest.approx(90)
    assert angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180)
    assert angle(None, (0, 0), (1, 0)) is None


def test_judge_make():
    hoop = Hoop(100, 95, 140, 105)
    through = [(0, 110, 40, 5), (0.1, 118, 90, 5), (0.2, 121, 130, 5)]
    outside = [(0, 170, 40, 5), (0.1, 180, 90, 5), (0.2, 185, 130, 5)]
    assert judge_make(through, hoop) == "make"
    assert judge_make(outside, hoop) == "miss"
    assert judge_make(through, None) == "unknown"


def test_make_detected_with_metrics():
    shots = run_shots(shot_frames())
    assert len(shots) == 1
    shot = shots[0]
    assert shot.outcome == "make"
    assert shot.kind == "jump_shot"
    assert abs(shot.release_t - 1.42) < 0.12
    m, keys = shot_metrics(shot)
    assert m["arc_angle"] == pytest.approx(50, abs=6)
    assert m["knee_bend"] == pytest.approx(111, abs=10)
    assert 75 <= m["elbow_set"] <= 115
    assert m["elbow_release"] > 155
    assert m["release_height"] > 0.15
    assert m["follow_through"] > 0.3
    assert m["jump_height"] > 0.04
    assert {"set", "release", "follow_through"} <= set(keys)


@pytest.mark.parametrize("fps", [15, 25, 60])
def test_detection_is_fps_independent(fps):
    shots = run_shots(shot_frames(fps=fps))
    assert len(shots) == 1 and shots[0].outcome == "make"
    assert shot_metrics(shots[0])[0]["arc_angle"] == pytest.approx(50, abs=7)


def test_miss_long_and_flat_shot_cue():
    shots = run_shots(shot_frames(launch_deg=33, target=(HOOP.x2 + 90, HOOP.rim_y)))
    assert len(shots) == 1 and shots[0].outcome == "miss"
    m, _ = shot_metrics(shots[0])
    assert m["arc_angle"] < 40
    cue = Coach().cue_for_shot(m, shots[0].kind, shots[0].outcome)
    assert cue.metric == "arc_angle"


def test_pose_only_shot_without_ball():
    shots = run_shots(shot_frames(with_ball=False), hoop=None)
    assert len(shots) == 1
    assert shots[0].outcome == "unknown"
    assert abs(shots[0].release_t - 1.42) < 0.15


def test_no_shot_when_standing():
    frames = shot_frames(duration=0.85)  # only the hold phase
    assert run_shots(frames) == []


def test_dribbles_counted():
    counter = DribbleCounter()
    events = [e for e in (counter.update(f) for f in dribble_frames(bounces=10)) if e is not None]
    assert len(events) == 10
    assert all(e.hand == "right" for e in events)


def test_tracker_keeps_lock_on_first_player():
    tracker = PlayerTracker()
    k = np.zeros((17, 3)); k[:, 2] = 0.9
    player, bystander_far, bystander_near = (100, 100, 200, 400), (600, 150, 650, 300), (590, 50, 900, 700)
    box, _ = tracker.update([player, bystander_far], [0.9, 0.9], [k, k], 0.0)
    assert box == player
    for i in range(1, 10):  # bystander walks closer and becomes the biggest person
        moved = (100 + i, 100, 200 + i, 400)
        box, _ = tracker.update([bystander_near, moved], [0.9, 0.9], [k, k], i / 30)
        assert box == moved


def test_one_euro_smooths_jitter():
    f = OneEuroFilter()
    rng = np.random.default_rng(0)
    raw = 100 + rng.normal(0, 3, 60)
    out = [f(x, i / 30) for i, x in enumerate(raw)]
    assert np.std(out[10:]) < np.std(raw[10:])


def test_coach_avoids_nagging():
    m = {"knee_bend": 165, "arc_angle": 35}
    coach = Coach()
    picked = [coach.cue_for_shot(m, "jump_shot", "miss").metric for _ in range(3)]
    assert picked[0] == picked[1] == "arc_angle" and picked[2] == "knee_bend"
    assert Coach().cue_for_shot({"knee_bend": 130}, "jump_shot", "make").kind == "praise"


def test_layup_wrong_foot():
    found = issues({"takeoff_left_foot": 0.0, "steps": 3}, "layup", hand="right")
    assert {c.metric for c in found} == {"takeoff_left_foot", "steps"}


def test_store_roundtrip(tmp_path):
    store = SessionStore(str(tmp_path / "coach.db"))
    store.start_session("s1", "2026-10-22T10:00:00", "cam", str(tmp_path), "right", "live")
    store.add_shot("s1", 1, 1.4, 42, "jump_shot", "right", "make", {"arc_angle": 50.0}, "Great shot", {})
    store.add_dribble("s1", 2.0, 60, "left", 0.5)
    store.finish_session("s1", 12.5)
    assert store.sessions()[0]["duration_s"] == 12.5
    assert store.shots("s1")[0]["metrics"]["arc_angle"] == 50.0
    assert len(store.dribbles("s1")) == 1
    store.start_session("s1", "2026-10-22T10:00:00", "cam", str(tmp_path), "right", "offline")
    assert store.shots("s1") == []  # re-analysis replaces old results
