"""SQLite store for sessions, shots and dribbles (sessions/coach.db by default)."""
import json
import os
import sqlite3
from typing import Dict, List, Optional

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions", "coach.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,          -- session folder name, e.g. 20261022-103747
    started TEXT,
    duration_s REAL,
    source TEXT,
    session_dir TEXT,
    shooting_hand TEXT,
    mode TEXT,                    -- "live" or "offline"
    meta TEXT,                    -- JSON
    summary TEXT                  -- LLM post-session summary (markdown)
);
CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    shot_no INTEGER,
    t REAL,                       -- release time, seconds since session start
    frame INTEGER,                -- release frame index in raw.mp4
    kind TEXT,                    -- jump_shot / set_shot / layup
    hand TEXT,
    outcome TEXT,                 -- make / miss / unknown
    metrics TEXT,                 -- JSON
    cue TEXT,
    keyframes TEXT                -- JSON: {"set": path, "release": path, "follow_through": path}
);
CREATE TABLE IF NOT EXISTS dribbles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    t REAL,
    frame INTEGER,
    hand TEXT,
    height REAL
);
"""


class SessionStore:
    def __init__(self, path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def start_session(self, session_id: str, started: str, source: str, session_dir: str,
                      shooting_hand: str, mode: str, meta: Optional[Dict] = None):
        # Re-analysing a session replaces its previous results.
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.execute(
            "INSERT INTO sessions (id, started, source, session_dir, shooting_hand, mode, meta) VALUES (?,?,?,?,?,?,?)",
            (session_id, started, source, session_dir, shooting_hand, mode, json.dumps(meta or {})),
        )
        self.conn.commit()

    def finish_session(self, session_id: str, duration_s: float, meta: Optional[Dict] = None):
        if meta is None:
            self.conn.execute("UPDATE sessions SET duration_s = ? WHERE id = ?", (duration_s, session_id))
        else:
            self.conn.execute("UPDATE sessions SET duration_s = ?, meta = ? WHERE id = ?",
                              (duration_s, json.dumps(meta), session_id))
        self.conn.commit()

    def add_shot(self, session_id: str, shot_no: int, t: float, frame: int, kind: str, hand: str,
                 outcome: str, metrics: Dict, cue: str, keyframes: Dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO shots (session_id, shot_no, t, frame, kind, hand, outcome, metrics, cue, keyframes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (session_id, shot_no, t, frame, kind, hand, outcome, json.dumps(metrics), cue, json.dumps(keyframes)),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_dribble(self, session_id: str, t: float, frame: int, hand: Optional[str], height: Optional[float]):
        self.conn.execute("INSERT INTO dribbles (session_id, t, frame, hand, height) VALUES (?,?,?,?,?)",
                          (session_id, t, frame, hand, height))
        self.conn.commit()

    def set_summary(self, session_id: str, summary: str):
        self.conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))
        self.conn.commit()

    def sessions(self) -> List[Dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM sessions ORDER BY started DESC")]

    def shots(self, session_id: str) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM shots WHERE session_id = ? ORDER BY shot_no", (session_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"] or "{}")
            d["keyframes"] = json.loads(d["keyframes"] or "{}")
            out.append(d)
        return out

    def dribbles(self, session_id: str) -> List[Dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM dribbles WHERE session_id = ? ORDER BY t", (session_id,))]

    def close(self):
        self.conn.close()
