"""Claude phrasing for live cues and post-session summaries.

The rules in coaching.py decide *what* to say; Claude only rephrases it so the
voice sounds natural and varied, and writes the after-session summary from
the shot log. Without credentials, or on any API error/timeout, the template
text is used instead, so coaching never depends on the network.

Credentials: ANTHROPIC_API_KEY (or an `ant auth login` profile).
Model: COACH_LLM_MODEL (default claude-opus-5-5).
"""
import json
import os
from typing import Dict, List, Optional

from .coaching import Cue, metric_label

MODEL = os.environ.get("COACH_LLM_MODEL", "claude-opus-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

CUE_SYSTEM = (
    "You are a basketball shooting coach speaking to a player during practice through an earpiece. "
    "You get one correction (or praise) chosen from measured shot data. Rephrase it as a single spoken "
    "cue of at most 12 words: direct, encouraging, no numbers, no units, no emojis. Never add advice "
    "beyond the correction you were given. Vary the wording from the recent cues listed."
)

SUMMARY_SYSTEM = (
    "You are a basketball shooting coach writing a short post-practice report for the player. Use only "
    "the data provided. Structure: 2-3 sentence overview; 'What went well' (2 bullets); 'Focus next "
    "session' (2 bullets, each tied to a metric and its target); 'Drills' (2 specific drills). Under 220 "
    "words. Metrics are 2D estimates from one camera; do not over-interpret small differences."
)


def _client():
    try:
        import anthropic
        return anthropic.Anthropic(timeout=8.0, max_retries=0)
    except Exception:
        return None


def _call(system: str, user: str, max_tokens: int, timeout: float) -> Optional[str]:
    client = _client()
    if client is None:
        return None
    import anthropic
    try:
        response = client.with_options(timeout=timeout).beta.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            output_config={"effort": "low"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError:
        return None  # invalid credentials: silently use templates
    except TypeError:
        return None  # the SDK raises TypeError when no credentials are configured at all
    except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        print(f"[llm] falling back to template text: {type(e).__name__}")
        return None
    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


class CuePhraser:
    def __init__(self, enabled: bool = True, timeout: float = 3.0):
        self.enabled, self.timeout = enabled, timeout
        self.recent: List[str] = []

    def phrase(self, cue: Cue, outcome: str) -> str:
        text = None
        if self.enabled:
            detail = {"type": cue.kind, "correction": cue.text, "shot_result": outcome}
            if cue.metric:
                detail["metric"] = metric_label(cue.metric)
            user = json.dumps(detail) + "\nRecent cues: " + json.dumps(self.recent[-3:])
            text = _call(CUE_SYSTEM, user, max_tokens=2000, timeout=self.timeout)
        text = (text or cue.text).strip().strip('"')
        self.recent.append(text)
        return text


def session_summary(session: Dict, shots: List[Dict], dribble_stats: Optional[Dict] = None) -> Optional[str]:
    """Markdown summary of a session, or None if Claude is unavailable."""
    payload = {"session": session, "shots": shots, "dribbling": dribble_stats or {}}
    return _call(SUMMARY_SYSTEM, json.dumps(payload, default=str), max_tokens=8000, timeout=60.0)
