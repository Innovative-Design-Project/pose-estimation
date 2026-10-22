"""Post-session analysis: run the full coaching pipeline on a recorded session or any video.

Uses the more accurate pose model by default (live mode uses the fast one), so
re-analysing a live session usually gives better metrics. Results go to the
session store (sessions/coach.db) and key frames to <session>/shots/; open them
with `streamlit run dashboard.py`.

Examples:
  python analyze.py sessions/20261022-103747                 # a live.py recording
  python analyze.py videos/practice.mp4 --hoop calibrate       # any video; mark the rim once
  python analyze.py videos/practice.mp4 --hoop 610,240,680,262 --shooting-hand left
  python analyze.py sessions/20261022-103747 --summary        # also write the LLM session summary
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from coach.ball import DEFAULT_BALL_MODEL, BallDetector
from coach.engine import CoachEngine
from coach.hoop import Hoop, calibrate
from coach.llm import session_summary
from coach.store import DEFAULT_DB, SessionStore
from main import get_device

load_dotenv()  # ANTHROPIC_API_KEY / ELEVENLABS_API_KEY from .env

HERE = os.path.dirname(os.path.abspath(__file__))


def load_frame_times(session_dir: str):
    path = os.path.join(session_dir, "frame_times.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    t0 = float(rows[0]["t"])
    return [float(r["t"]) - t0 for r in rows]


def resolve_input(path: str):
    """Returns (video_path, session_dir, session_id, is_existing_session)."""
    if os.path.isdir(path):
        video = os.path.join(path, "raw.mp4")
        if not os.path.exists(video):
            raise SystemExit(f"No raw.mp4 in {path}")
        return video, os.path.abspath(path), os.path.basename(os.path.normpath(path)), True
    stem = os.path.splitext(os.path.basename(path))[0]
    session_id = f"offline-{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session_dir = os.path.join(HERE, "sessions", session_id)
    os.makedirs(session_dir, exist_ok=True)
    return path, session_dir, session_id, False


def resolve_hoop(arg: str, session_dir: str, video: str):
    hoop_path = os.path.join(session_dir, "hoop.json")
    if arg == "none":
        return None
    if arg == "calibrate":
        cap = cv2.VideoCapture(video)
        ok, frame = cap.read()
        cap.release()
        hoop = calibrate(frame) if ok else None
    elif arg == "auto":
        hoop = Hoop.load(hoop_path) if os.path.exists(hoop_path) else None
    else:
        hoop = Hoop.parse(arg)
    if hoop is not None:
        hoop.save(hoop_path)
    return hoop


def main():
    parser = argparse.ArgumentParser(description="Analyse a recorded practice session or video")
    parser.add_argument("input", help="Session folder (from live.py) or a video file")
    parser.add_argument("--model", default=os.path.join(HERE, "yolo11m-pose.pt"))
    parser.add_argument("--ball-model", default=DEFAULT_BALL_MODEL, help="Ball detector weights ('' to disable)")
    parser.add_argument("--hoop", default="auto", help="auto (session hoop.json), calibrate, none, or x1,y1,x2,y2")
    parser.add_argument("--shooting-hand", choices=["right", "left"], default="right")
    parser.add_argument("--hands", choices=["shot", "always", "off"], default="shot",
                        help="Hand landmarks: only around shots (default), every frame, or off")
    parser.add_argument("--device", default="")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0 = all)")
    parser.add_argument("--summary", action="store_true", help="Write an LLM session summary at the end")
    args = parser.parse_args()

    video, session_dir, session_id, existing = resolve_input(args.input)
    device = args.device or get_device()
    hoop = resolve_hoop(args.hoop, session_dir, video)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = load_frame_times(session_dir) if existing else None

    meta = {}
    meta_path = os.path.join(session_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    started = meta.get("started") or datetime.now().isoformat(timespec="seconds")

    store = SessionStore(args.db)
    store.start_session(session_id, started, video, session_dir, args.shooting_hand, "offline",
                        {**meta, "analysis_model": os.path.basename(args.model), "hoop": None if hoop is None else hoop.__dict__})
    engine = CoachEngine(
        session_id, session_dir, shooting_hand=args.shooting_hand, hoop=hoop,
        ball_detector=BallDetector(args.ball_model, device=device) if args.ball_model else None,
        hands_mode=args.hands, store=store,
    )
    pose = YOLO(args.model)
    print(f"Analysing {video} ({total} frames @ {fps:.0f} fps) on {device.upper()} | session {session_id}")
    print(f"Rim: {'not set (make/miss disabled)' if hoop is None else hoop}")

    index, start = 0, time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and index >= args.max_frames):
            break
        t = times[index] if times and index < len(times) else index / fps
        result = pose(frame, device=device, imgsz=args.imgsz, verbose=False)[0]
        out = engine.process(index, t, frame, result)
        if out["shot"]:
            s = out["shot"]
            print(f"  shot {s['shot_no']:>3} @ {s['t']:6.1f}s  {s['kind']:<9} {s['outcome']:<7} -> {s['cue']}")
        index += 1
        if index % 300 == 0:
            rate = index / (time.monotonic() - start)
            print(f"  {index}/{total} frames ({rate:.1f} fps)")
    cap.release()
    engine.close()

    duration = (times[min(index, len(times)) - 1] if times else index / fps) if index else 0.0
    store.finish_session(session_id, round(duration, 2))
    attempts = engine.makes + engine.misses
    print(f"\nDone: {engine.shot_count} shots ({engine.makes}/{attempts} makes with a known result), "
          f"{engine.dribbles.count} dribbles in {duration:.1f}s of video")

    if args.summary:
        session = next(s for s in store.sessions() if s["id"] == session_id)
        shots = [{k: s[k] for k in ("shot_no", "t", "kind", "outcome", "metrics", "cue")} for s in store.shots(session_id)]
        text = session_summary({"duration_s": session["duration_s"], "shots": len(shots)}, shots,
                               {"dribbles": engine.dribbles.count})
        if text:
            store.set_summary(session_id, text)
            print("\n" + text)
        else:
            print("No summary: Claude is not configured (set ANTHROPIC_API_KEY).")
    print(f"Open the dashboard with: streamlit run dashboard.py")
    store.close()


if __name__ == "__main__":
    main()
