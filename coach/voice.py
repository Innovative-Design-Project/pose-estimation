"""Spoken cues: ElevenLabs streaming TTS with a local cache and an offline fallback.

Speech runs on a background thread so it never blocks the video loop. Only the
newest pending cue is kept: if the player shoots again before a cue is spoken,
the stale cue is dropped rather than queued.

Credentials: ELEVENLABS_API_KEY. Voice: ELEVENLABS_VOICE_ID (default: a premade voice).
Without a key it uses macOS `say`, or just prints the cue on other systems.
"""
import hashlib
import os
import shutil
import subprocess
import threading
from typing import Optional

import requests

API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # ElevenLabs premade "Rachel"
MODEL_ID = "eleven_flash_v2_5"          # low-latency model
SAMPLE_RATE = 22050
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "voice_cache")


class Voice:
    def __init__(self, mode: str = "auto"):
        """mode: "elevenlabs", "say", "print", "off", or "auto" (ElevenLabs if a key is set)."""
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        self.voice_id = os.environ.get("ELEVENLABS_VOICE_ID", DEFAULT_VOICE)
        if mode == "auto":
            mode = "elevenlabs" if self.api_key else ("say" if shutil.which("say") else "print")
        if mode == "elevenlabs" and not self.api_key:
            print("ELEVENLABS_API_KEY not set; using fallback voice")
            mode = "say" if shutil.which("say") else "print"
        self.mode = mode
        self._pending: Optional[str] = None
        self._cond = threading.Condition()
        self._stopped = False
        self.speaking = False
        if mode != "off":
            threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str):
        if self.mode == "off" or not text:
            return
        with self._cond:
            self._pending = text
            self._cond.notify()

    def close(self):
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def _run(self):
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                text, self._pending = self._pending, None
            self.speaking = True
            try:
                if self.mode == "elevenlabs":
                    self._speak_elevenlabs(text)
                elif self.mode == "say":
                    subprocess.run(["say", text], check=False)
                else:
                    print(f"[coach] {text}")
            except Exception as e:  # never let audio problems kill the session
                print(f"[voice] {type(e).__name__}: {e}; cue was: {text}")
            finally:
                self.speaking = False

    def _cache_path(self, text: str) -> str:
        key = hashlib.sha1(f"{self.voice_id}|{MODEL_ID}|{SAMPLE_RATE}|{text}".encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"{key}.pcm")

    def _speak_elevenlabs(self, text: str):
        import sounddevice as sd

        path = self._cache_path(text)
        with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as out:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    out.write(f.read())
                return
            resp = requests.post(
                API_URL.format(voice_id=self.voice_id),
                params={"output_format": f"pcm_{SAMPLE_RATE}"},
                headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
                json={"text": text, "model_id": MODEL_ID},
                stream=True,
                timeout=10,
            )
            resp.raise_for_status()
            audio, carry = bytearray(), b""
            for chunk in resp.iter_content(chunk_size=4096):
                data = carry + chunk
                usable = len(data) - len(data) % 2  # int16 frames are 2 bytes
                out.write(data[:usable])           # play as it arrives
                audio += data[:usable]
                carry = data[usable:]
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(bytes(audio))
