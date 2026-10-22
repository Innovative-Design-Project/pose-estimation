"""CoachEngine: per-frame glue between models, detectors, coaching, voice and storage.

Used by live.py (real time, with voice) and analyze.py (offline, no voice).
Feed it every analysed frame with the pose result; it tracks the player and
ball, detects shots and dribbles, stores them, and speaks one cue per shot.
"""
import collections
import os
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np

from .ball import BallDetector, BallTracker
from .coaching import Coach, dribble_feedback
from .events import DribbleCounter, ShotDetector, ShotEvent
from .geometry import FrameState
from .hands import HandAnalyzer, draw_hands
from .hoop import Hoop
from .llm import CuePhraser
from .metrics import shot_metrics
from .store import SessionStore
from .tracking import PlayerTracker
from .voice import Voice

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 5), (0, 6)]
KEYFRAME_WIDTH = 640


def draw_player(frame, fs: FrameState, color=(0, 255, 120)):
    if fs.box is not None:
        x1, y1, x2, y2 = (int(v) for v in fs.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if fs.kpts is not None:
        for a, b in SKELETON:
            if fs.kpts[a, 2] > 0.3 and fs.kpts[b, 2] > 0.3:
                cv2.line(frame, (int(fs.kpts[a, 0]), int(fs.kpts[a, 1])), (int(fs.kpts[b, 0]), int(fs.kpts[b, 1])),
                         color, 3, cv2.LINE_AA)
        for x, y, c in fs.kpts:
            if c > 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)
    if fs.ball is not None:
        cv2.circle(frame, (int(fs.ball[0]), int(fs.ball[1])), max(4, int(fs.ball[2])), (0, 140, 255), 2)
    draw_hands(frame, fs.hands)


class CoachEngine:
    def __init__(self, session_id: str, session_dir: Optional[str], shooting_hand: str = "right",
                 hoop: Optional[Hoop] = None, ball_detector: Optional[BallDetector] = None,
                 hands_mode: str = "shot", store: Optional[SessionStore] = None,
                 voice: Optional[Voice] = None, phraser: Optional[CuePhraser] = None):
        self.session_id, self.session_dir, self.shooting_hand = session_id, session_dir, shooting_hand
        self.hoop, self.ball_detector, self.store = hoop, ball_detector, store
        self.voice, self.phraser = voice, phraser
        self.tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.shots = ShotDetector(shooting_hand=shooting_hand, hoop=hoop)
        self.dribbles = DribbleCounter()
        self.coach = Coach()
        self.hands_mode = hands_mode
        self.hands = HandAnalyzer() if hands_mode in ("shot", "always") else None

        self.frame_buffer = collections.OrderedDict()   # index -> downscaled frame, for key frames
        self.state_buffer = collections.deque(maxlen=12)  # recent FrameStates for the ball trail
        self.shot_count = self.makes = self.misses = 0
        self.last_cue = ""
        self.recent_dribbles: List = []
        self._speech_thread: Optional[threading.Thread] = None

    # ---- per frame -------------------------------------------------------------------------
    def process(self, index: int, t: float, frame, pose_result) -> Dict:
        """Returns {"state": FrameState, "shot": dict or None, "dribble": DribbleEvent or None}."""
        boxes, confs, kpts = [], [], []
        if pose_result is not None and pose_result.boxes is not None and pose_result.keypoints is not None:
            boxes = pose_result.boxes.xyxy.tolist()
            confs = pose_result.boxes.conf.tolist()
            kpts = pose_result.keypoints.data.cpu().numpy() if hasattr(pose_result.keypoints.data, "cpu") \
                else np.asarray(pose_result.keypoints.data)
        return self.process_people(index, t, frame, boxes, confs, list(kpts))

    def process_people(self, index: int, t: float, frame, boxes, confs, kpts) -> Dict:
        box, player_kpts = self.tracker.update(boxes, confs, kpts, t)
        fs = FrameState(index=index, t=t, kpts=player_kpts, box=box)

        if self.ball_detector is not None and self.ball_detector.available:
            fs.ball = self.ball_tracker.update(self.ball_detector(frame), t)

        if self.hands is not None and player_kpts is not None and (
                self.hands_mode == "always" or self.shots.active or self._wrist_above_shoulder(player_kpts)):
            fs.hands = self.hands(frame, player_kpts)

        self._remember_frame(index, frame)
        self.state_buffer.append(fs)

        out = {"state": fs, "shot": None, "dribble": None}
        shot = self.shots.update(fs)
        if shot is not None:
            out["shot"] = self._handle_shot(shot)
        dribble = self.dribbles.update(fs, shot_active=self.shots.active)
        if dribble is not None:
            out["dribble"] = dribble
            self._handle_dribble(dribble)
        return out

    def _wrist_above_shoulder(self, k) -> bool:
        side = 10 if self.shooting_hand == "right" else 9
        shoulder = 6 if self.shooting_hand == "right" else 5
        return k[side, 2] > 0.4 and k[shoulder, 2] > 0.4 and k[side, 1] < k[shoulder, 1]

    def _remember_frame(self, index: int, frame):
        h, w = frame.shape[:2]
        scale = KEYFRAME_WIDTH / w if w > KEYFRAME_WIDTH else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame.copy()
        self.frame_buffer[index] = (scale, small)
        while len(self.frame_buffer) > 200:
            self.frame_buffer.popitem(last=False)

    # ---- events ----------------------------------------------------------------------------
    def _handle_shot(self, shot: ShotEvent) -> Dict:
        self.shot_count += 1
        self.makes += shot.outcome == "make"
        self.misses += shot.outcome == "miss"
        metrics, keys = shot_metrics(shot)
        cue = self.coach.cue_for_shot(metrics, shot.kind, shot.outcome, shot.hand)
        self.last_cue = cue.text
        keyframes = self._save_keyframes(shot, keys)

        if self.store is not None:
            self.store.add_shot(self.session_id, self.shot_count, round(shot.release_t, 3), shot.release_index,
                                shot.kind, shot.hand, shot.outcome, metrics, cue.text, keyframes)
        if self.voice is not None:
            self._speak_async(cue, shot.outcome)

        return {"shot_no": self.shot_count, "kind": shot.kind, "outcome": shot.outcome, "t": shot.release_t,
                "metrics": metrics, "cue": cue.text, "keyframes": keyframes}

    def _speak_async(self, cue, outcome: str):
        # Phrasing may take a second or two; never hold up the video loop for it.
        def run():
            text = self.phraser.phrase(cue, outcome) if self.phraser is not None else cue.text
            self.last_cue = text
            self.voice.say(text)
        self._speech_thread = threading.Thread(target=run, daemon=True)
        self._speech_thread.start()

    def _handle_dribble(self, dribble):
        if self.store is not None:
            self.store.add_dribble(self.session_id, round(dribble.t, 3), dribble.index, dribble.hand,
                                   None if dribble.height is None else round(dribble.height, 3))
        self.recent_dribbles.append(dribble)
        if len(self.recent_dribbles) >= 15:
            block, self.recent_dribbles = self.recent_dribbles, []
            text = dribble_feedback([d.height for d in block], [d.t for d in block], [d.hand for d in block])
            if text and self.voice is not None:
                self.last_cue = text
                self.voice.say(text)

    def _save_keyframes(self, shot: ShotEvent, keys: Dict[str, int]) -> Dict[str, str]:
        if not self.session_dir:
            return {}
        out_dir = os.path.join(self.session_dir, "shots")
        os.makedirs(out_dir, exist_ok=True)
        states = {f.index: f for f in shot.frames}
        saved = {}
        for name, index in keys.items():
            if index not in self.frame_buffer or index not in states:
                continue
            scale, img = self.frame_buffer[index]
            img = img.copy()
            fs = states[index]
            scaled = FrameState(
                index=fs.index, t=fs.t,
                kpts=None if fs.kpts is None else fs.kpts * np.array([scale, scale, 1.0]),
                box=None if fs.box is None else tuple(v * scale for v in fs.box),
                ball=None if fs.ball is None else (fs.ball[0] * scale, fs.ball[1] * scale, fs.ball[2] * scale),
                hands={k: v * np.array([scale, scale, 1.0]) for k, v in fs.hands.items()},
            )
            draw_player(img, scaled)
            if self.hoop is not None:
                Hoop(self.hoop.x1 * scale, self.hoop.y1 * scale, self.hoop.x2 * scale, self.hoop.y2 * scale).draw(img)
            path = os.path.join(out_dir, f"shot_{self.shot_count:03d}_{name}.jpg")
            cv2.imwrite(path, img)
            saved[name] = os.path.relpath(path, self.session_dir)
        return saved

    # ---- display ---------------------------------------------------------------------------
    def draw(self, frame, fs: FrameState):
        if self.hoop is not None:
            self.hoop.draw(frame)
        trail = [s.ball for s in self.state_buffer if s.ball is not None]
        for (x0, y0, _), (x1, y1, _) in zip(trail, trail[1:]):
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 140, 255), 2, cv2.LINE_AA)
        draw_player(frame, fs)

    def hud_lines(self) -> List[str]:
        attempts = self.makes + self.misses
        pct = f"{100 * self.makes / attempts:.0f}%" if attempts else "-"
        lines = [f"Shots {self.shot_count}   Makes {self.makes}/{attempts} ({pct})   Dribbles {self.dribbles.count}"]
        if self.last_cue:
            lines.append(f"Coach: {self.last_cue}")
        return lines

    def close(self):
        if self._speech_thread is not None:
            self._speech_thread.join(timeout=5)
        if self.hands is not None:
            self.hands.close()
