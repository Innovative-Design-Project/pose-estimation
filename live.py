"""Live pose estimation from a webcam / phone camera, with session recording.

A capture thread reads every camera frame, writes it to the session's raw
video and keeps only the newest frame for inference. When the model is slower
than the camera, frames are skipped for inference but never for the
recording, so the preview stays real time and the full session can be
re-analysed offline later.

Each run creates sessions/<YYYYmmdd-HHMMSS>/ with:
  raw.mp4           every captured frame (no overlay)
  frame_times.csv   capture timestamp of each raw frame (webcam fps is not constant)
  keypoints.jsonl   one line per analysed frame: raw frame index, time, people + 17 keypoints
  annotated.mp4     preview with skeleton + HUD (only with --save-annotated)
  meta.json         source, model, device, resolution, fps and frame counts

Coaching (on by default, --no-coach for pose only): the practising player is
locked and smoothed, the ball is tracked (once sports_pipeline/train_ball.py
has produced weights), shots / makes / dribbles are detected, one cue per shot
is spoken (ElevenLabs if ELEVENLABS_API_KEY is set, phrased by Claude if
ANTHROPIC_API_KEY is set), and everything is stored in sessions/coach.db plus
key frames in <session>/shots/ for the dashboard (streamlit run dashboard.py).
At start you drag a box around the rim (c skips; --hoop last reuses the previous one).

Keys in the preview window: q / Esc to stop.

Examples:
  python live.py                                  # default camera
  python live.py --list-cameras                   # find the phone (Continuity Camera) index
  python live.py --source 1 --width 1920 --height 1080 --cam-fps 60
  python live.py --source videos/basketball_sportsmot_test.mp4   # replay a file in real time
  python live.py --hoop last --shooting-hand left --voice say
"""
import argparse
import collections
import json
import os
import threading
import time
import traceback
from datetime import datetime

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from coach.ball import DEFAULT_BALL_MODEL, BallDetector
from coach.engine import CoachEngine
from coach.hoop import Hoop, calibrate
from coach.llm import CuePhraser
from coach.store import SessionStore
from coach.voice import Voice
from main import KEYPOINT_NAMES, get_device

load_dotenv()  # ANTHROPIC_API_KEY / ELEVENLABS_API_KEY from .env

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")


class FpsMeter:
    """Frames per second over a sliding time window."""

    def __init__(self, window: float = 1.0):
        self.window = window
        self.times = collections.deque()

    def tick(self, t: float = None):
        t = time.monotonic() if t is None else t
        self.times.append(t)
        while self.times and t - self.times[0] > self.window:
            self.times.popleft()

    @property
    def fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


class FrameGrabber(threading.Thread):
    """Reads every frame on its own thread, records it, and exposes the newest one."""

    def __init__(self, cap, writer=None, times_file=None, pace_fps: float = 0.0):
        super().__init__(daemon=True)
        self.cap = cap
        self.writer = writer
        self.times_file = times_file
        self.pace = 1.0 / pace_fps if pace_fps else 0.0  # file sources are replayed at their own fps
        self.lock = threading.Lock()
        self.latest = None  # (frame_index, capture_time, frame)
        self.count = 0
        self.meter = FpsMeter()
        self.stopped = False
        self.finished = False

    def run(self):
        next_t = time.monotonic()
        try:
            while not self.stopped:
                ok, frame = self.cap.read()
                if not ok:
                    break
                t = time.monotonic()
                if self.writer is not None:
                    self.writer.write(frame)
                if self.times_file is not None:
                    self.times_file.write(f"{self.count},{t:.6f}\n")
                with self.lock:
                    self.latest = (self.count, t, frame)
                self.count += 1
                self.meter.tick(t)

                if self.pace:
                    next_t += self.pace
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
        finally:
            self.finished = True

    def newest_after(self, index: int):
        with self.lock:
            if self.latest is not None and self.latest[0] > index:
                return self.latest
        return None


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def open_source(source, width: int, height: int, cam_fps: int):
    if isinstance(source, int):
        # AVFoundation is the native backend on macOS (also exposes Continuity Camera).
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") and os.uname().sysname == "Darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, cam_fps)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def list_cameras(max_index: int = 6):
    print("Probing camera indices...")
    found = 0
    for i in range(max_index):
        cap = open_source(i, 1280, 720, 30)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if not ok:
            break  # camera indices are contiguous, so the first failure ends the list
        h, w = frame.shape[:2]
        print(f"  [{i}] {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps (reported)")
        found += 1
    if not found:
        print("  No cameras could be opened. On macOS, allow camera access for your terminal "
              "in System Settings > Privacy & Security > Camera.")


def people_from_result(result):
    people = []
    if result.boxes is None or result.keypoints is None:
        return people
    boxes = result.boxes.xyxy.tolist()
    confs = result.boxes.conf.tolist()
    for box, conf, kpts in zip(boxes, confs, result.keypoints.data.tolist()):
        people.append({
            "box": [round(v, 1) for v in box],
            "conf": round(conf, 3),
            "kpts": [[round(x, 1), round(y, 1), round(c, 3)] for x, y, c in kpts],
        })
    return people


def draw_hud(frame, lines, recording: bool, elapsed: float):
    pad, line_h = 10, 24
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for line in lines) + 2 * pad
    height = line_h * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (pad, pad + 14 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    if recording:
        label = f"REC {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = frame.shape[1] - tw - 20
        cv2.circle(frame, (x - 14, 26), 8, (0, 0, 255), -1)
        cv2.putText(frame, label, (x, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


def sessions_root(args) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return args.sessions_dir if os.path.isabs(args.sessions_dir) else os.path.join(base_dir, args.sessions_dir)


def setup_hoop(args, cap, is_file: bool, root: str, session_dir):
    """Rim box from --hoop; calibration uses the first frame (file sources are rewound)."""
    last_path = os.path.join(root, "last_hoop.json")
    hoop = None
    if args.hoop == "none":
        return None
    if args.hoop == "last":
        hoop = Hoop.load(last_path) if os.path.exists(last_path) else None
        if hoop is None:
            print("No previous rim saved; make/miss is disabled. Use --hoop calibrate once.")
    elif args.hoop == "calibrate":
        if args.no_display:
            print("--no-display: skipping rim calibration (make/miss disabled)")
            return None
        ok, frame = cap.read()
        if is_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        hoop = calibrate(frame) if ok else None
    else:
        hoop = Hoop.parse(args.hoop)
    if hoop is not None:
        os.makedirs(root, exist_ok=True)
        hoop.save(last_path)
        if session_dir:
            hoop.save(os.path.join(session_dir, "hoop.json"))
    return hoop


def main():
    parser = argparse.ArgumentParser(description="Live pose estimation with session recording")
    parser.add_argument("--source", default="0", help="Camera index, stream URL or video file (default: 0)")
    parser.add_argument("--list-cameras", action="store_true", help="List camera indices that can be opened and exit")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Pose weights (n is fast enough for live use)")
    parser.add_argument("--device", default="", help="cpu, mps or cuda (default: auto)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--cam-fps", type=int, default=60, help="Requested camera fps (the camera may give less)")
    parser.add_argument("--sessions-dir", default="sessions")
    parser.add_argument("--no-record", action="store_true", help="Preview only, write nothing")
    parser.add_argument("--save-annotated", action="store_true", help="Also save the preview with overlay")
    parser.add_argument("--no-display", action="store_true", help="Run without a preview window")
    parser.add_argument("--max-seconds", type=float, default=0, help="Stop after this many seconds (0 = until q)")
    parser.add_argument("--no-coach", action="store_true", help="Pose and recording only (no shots, voice or store)")
    parser.add_argument("--hoop", default="calibrate", help="calibrate, last, none, or x1,y1,x2,y2")
    parser.add_argument("--shooting-hand", choices=["right", "left"], default="right")
    parser.add_argument("--ball-model", default=DEFAULT_BALL_MODEL, help="Ball detector weights ('' to disable)")
    parser.add_argument("--hands", choices=["shot", "always", "off"], default="shot",
                        help="Hand landmarks around shots (default), on every frame, or off")
    parser.add_argument("--voice", choices=["auto", "elevenlabs", "say", "print", "off"], default="auto")
    parser.add_argument("--no-llm", action="store_true", help="Speak the rule text instead of Claude's phrasing")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    source = parse_source(args.source)
    is_file = isinstance(source, str) and source.lower().endswith(VIDEO_EXTS)
    cap = open_source(source, args.width, args.height, args.cam_fps)
    if not cap.isOpened():
        print(f"Could not open source {args.source!r}. Try --list-cameras.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    device = args.device or get_device()
    print(f"Source {args.source}: {width}x{height} @ {reported_fps:.0f} fps (reported) | device {device.upper()}")

    model = YOLO(args.model)

    session_dir = None
    raw_writer = annotated_writer = times_file = keypoints_file = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if not args.no_record:
        session_dir = os.path.join(sessions_root(args), datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(session_dir, exist_ok=True)
        raw_writer = cv2.VideoWriter(os.path.join(session_dir, "raw.mp4"), fourcc, reported_fps, (width, height))
        times_file = open(os.path.join(session_dir, "frame_times.csv"), "w")
        times_file.write("frame,t\n")
        keypoints_file = open(os.path.join(session_dir, "keypoints.jsonl"), "w")
        if args.save_annotated:
            annotated_writer = cv2.VideoWriter(os.path.join(session_dir, "annotated.mp4"), fourcc,
                                               reported_fps, (width, height))
        print(f"Recording session to {session_dir}")

    engine = store = voice = None
    hoop = None
    if not args.no_coach:
        hoop = setup_hoop(args, cap, is_file, sessions_root(args), session_dir)
        session_id = os.path.basename(session_dir) if session_dir else "preview"
        if session_dir is not None:
            store = SessionStore(os.path.join(sessions_root(args), "coach.db"))
            store.start_session(session_id, datetime.now().isoformat(timespec="seconds"), args.source, session_dir,
                                args.shooting_hand, "live")
        voice = Voice(args.voice)
        engine = CoachEngine(
            session_id, session_dir, shooting_hand=args.shooting_hand, hoop=hoop,
            ball_detector=BallDetector(args.ball_model, device=device) if args.ball_model else None,
            hands_mode=args.hands, store=store, voice=voice, phraser=CuePhraser(enabled=not args.no_llm),
        )
        print(f"Coaching on | rim: {'not set' if hoop is None else 'set'} | voice: {voice.mode} | "
              f"ball: {'on' if engine.ball_detector and engine.ball_detector.available else 'off'}")

    grabber = FrameGrabber(cap, raw_writer, times_file, pace_fps=reported_fps if is_file else 0.0)
    pose_meter = FpsMeter()
    window = "Live pose (q to quit)"
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    start_wall = datetime.now()
    t0 = time.monotonic()
    stop = threading.Event()
    shown = {"frame": None, "lock": threading.Lock()}  # newest annotated frame for the window
    analysed = 0

    def pose_loop():
        try:
            pose_steps()
        except Exception:
            traceback.print_exc()  # end the session cleanly instead of freezing the preview
        finally:
            stop.set()

    def pose_steps():
        # Runs off the main thread: on macOS cv2.waitKey alone takes ~15 ms per call,
        # so keeping the window on the main thread stops it from throttling inference.
        nonlocal analysed
        last_index, annotated_written = -1, -1
        while not stop.is_set():
            item = grabber.newest_after(last_index)
            if item is None:
                if grabber.finished:
                    break
                time.sleep(0.002)
                continue

            index, t_capture, frame = item
            last_index = index
            infer_start = time.monotonic()
            result = model(frame, device=device, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            infer_ms = (time.monotonic() - infer_start) * 1000
            pose_meter.tick()
            analysed += 1

            people = people_from_result(result)
            if keypoints_file is not None:
                keypoints_file.write(json.dumps({
                    "frame": index, "t": round(t_capture - t0, 4), "people": people,
                }) + "\n")

            hud = []
            if engine is not None:
                out = engine.process(index, t_capture - t0, frame, result)
                annotated = frame.copy()
                engine.draw(annotated, out["state"])
                hud = engine.hud_lines()
                if out["shot"]:
                    s = out["shot"]
                    print(f"shot {s['shot_no']} {s['kind']} {s['outcome']}: {s['cue']}")
            else:
                annotated = result.plot()
            latency_ms = (time.monotonic() - t_capture) * 1000
            draw_hud(annotated, [
                f"Camera {grabber.meter.fps:5.1f} fps   Pose {pose_meter.fps:5.1f} fps",
                f"Inference {infer_ms:4.0f} ms   Latency {latency_ms:4.0f} ms",
                f"{os.path.basename(args.model)} on {device.upper()}   People: {len(people)}",
            ] + hud, recording=session_dir is not None, elapsed=time.monotonic() - t0)

            if annotated_writer is not None:
                # Repeat the annotated frame for raw frames skipped by inference, so
                # annotated.mp4 has the same frame count and timing as raw.mp4.
                for _ in range(index - annotated_written):
                    annotated_writer.write(annotated)
                annotated_written = index

            with shown["lock"]:
                shown["frame"] = annotated

    worker = threading.Thread(target=pose_loop, daemon=True)
    grabber.start()
    worker.start()
    try:
        while not stop.is_set():
            if args.max_seconds and time.monotonic() - t0 >= args.max_seconds:
                break
            if args.no_display:
                stop.wait(0.05)
                continue
            with shown["lock"]:
                frame, shown["frame"] = shown["frame"], None
            if frame is not None:
                cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=5)
        grabber.stopped = True
        grabber.join(timeout=2)
        cap.release()
        duration = time.monotonic() - t0
        for w in (raw_writer, annotated_writer):
            if w is not None:
                w.release()
        for f in (times_file, keypoints_file):
            if f is not None:
                f.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        if engine is not None:
            engine.close()
            voice.close()
            print(f"Shots {engine.shot_count} (makes {engine.makes}, misses {engine.misses}), "
                  f"dribbles {engine.dribbles.count}")

        captured = grabber.count
        measured_fps = captured / duration if duration > 0 else 0.0
        print(f"\nCaptured {captured} frames ({measured_fps:.1f} fps), analysed {analysed} "
              f"({100 * analysed / max(captured, 1):.0f}%) in {duration:.1f} s")

        if session_dir is not None:
            if abs(measured_fps - reported_fps) > 0.1 * reported_fps:
                print(f"Note: camera delivered {measured_fps:.1f} fps but reported {reported_fps:.0f}; "
                      "raw.mp4 playback speed will be off -- use frame_times.csv for timing.")
            with open(os.path.join(session_dir, "meta.json"), "w") as f:
                json.dump({
                    "source": args.source,
                    "started": start_wall.isoformat(timespec="seconds"),
                    "duration_s": round(duration, 2),
                    "resolution": [width, height],
                    "reported_fps": reported_fps,
                    "measured_fps": round(measured_fps, 2),
                    "frames_captured": captured,
                    "frames_analysed": analysed,
                    "model": args.model,
                    "device": device,
                    "imgsz": args.imgsz,
                    "conf": args.conf,
                    "keypoint_names": KEYPOINT_NAMES,
                    "coach": None if engine is None else {
                        "shooting_hand": args.shooting_hand,
                        "hoop": None if hoop is None else hoop.__dict__,
                        "hands": args.hands,
                        "ball_detector": bool(engine.ball_detector and engine.ball_detector.available),
                    },
                }, f, indent=2)
            if store is not None:
                store.finish_session(os.path.basename(session_dir), round(duration, 2),
                                     json.load(open(os.path.join(session_dir, "meta.json"))))
                store.close()
            print(f"Session saved to {session_dir}")


if __name__ == "__main__":
    main()
