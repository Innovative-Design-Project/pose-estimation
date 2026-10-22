"""Turn shot metrics into one coaching cue per shot.

TARGETS are starting ranges from common shooting guidance, measured in 2D from
a side / 45-degree camera. They should be tuned with a coach and, over time,
against each player's own best shots. A cue is chosen by how far a metric is
outside its range (relative to the range width) times its weight.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

# metric: (label, unit, low, high, cue if below low, cue if above high, weight, shot kinds)
TARGETS = {
    "knee_bend": ("Knee bend", "deg", 105, 150, "You're sinking too low - a smaller dip is enough",
                  "Bend your knees more to load the shot", 1.0, ("jump_shot", "set_shot")),
    "elbow_set": ("Elbow at set point", "deg", 70, 110, "Elbow is too tight at the set point - let the ball sit higher",
                  "Bring the elbow in under the ball at the set point", 0.9, ("jump_shot", "set_shot")),
    "elbow_release": ("Arm extension at release", "deg", 155, 181, "Extend your shooting arm fully on release",
                      None, 1.0, ("jump_shot", "set_shot")),
    "release_height": ("Release height above head", "body heights", 0.10, 0.60, "Release the ball higher, above your head",
                       None, 0.9, ("jump_shot", "set_shot")),
    "arc_angle": ("Launch angle", "deg", 42, 60, "Flat shot - shoot up through the ball for more arc",
                  "Too much arc - push the ball more toward the rim", 1.2, ("jump_shot", "set_shot")),
    "follow_through": ("Follow-through hold", "s", 0.30, 5.0, "Hold your follow-through until the ball lands",
                       None, 0.8, ("jump_shot", "set_shot")),
    "release_timing": ("Release vs jump peak", "s", -0.20, 0.08, "You're releasing too early in the jump",
                       "Release at the top of your jump, not on the way down", 0.8, ("jump_shot",)),
    "landing_drift": ("Landing drift", "body heights", 0.0, 0.12, None,
                      "Land where you took off - stay balanced", 0.7, ("jump_shot", "set_shot")),
    "body_lean": ("Body lean at release", "deg", 0, 12, None, "Stay upright - you're leaning on the release",
                  0.7, ("jump_shot", "set_shot")),
    "wrist_flexion": ("Wrist snap", "deg", 40, 180, "Snap your wrist - fingers to the rim on the follow-through",
                      None, 0.5, ("jump_shot", "set_shot")),
    "steps": ("Steps before layup", "steps", 1, 2, None, "Too many steps - two steps into the layup", 1.2, ("layup",)),
}

PRAISE_MAKE = "Great shot - that's the form"
PRAISE_MISS = "Good form on that one - same again"


@dataclass
class Cue:
    kind: str                 # "fix" or "praise"
    text: str                 # template wording (the LLM may rephrase it)
    metric: Optional[str] = None
    value: Optional[float] = None
    target: Optional[str] = None
    severity: float = 0.0


def issues(metrics: Dict[str, Optional[float]], shot_kind: str, hand: str = "right") -> List[Cue]:
    found = []
    for metric, (label, unit, low, high, cue_low, cue_high, weight, kinds) in TARGETS.items():
        value = metrics.get(metric)
        if value is None or shot_kind not in kinds:
            continue
        span = max(high - low, 1e-6)
        if value < low and cue_low:
            found.append(Cue("fix", cue_low, metric, value, f"{low}-{high} {unit}", weight * (low - value) / span))
        elif value > high and cue_high:
            found.append(Cue("fix", cue_high, metric, value, f"{low}-{high} {unit}", weight * (value - high) / span))

    if shot_kind == "layup" and metrics.get("takeoff_left_foot") is not None:
        # Right-hand layup: take off from the left foot (and vice versa).
        wrong = (metrics["takeoff_left_foot"] == 1.0) != (hand == "right")
        if wrong:
            foot = "left" if hand == "right" else "right"
            found.append(Cue("fix", f"Jump off your {foot} foot for a {hand}-hand layup", "takeoff_left_foot",
                             metrics["takeoff_left_foot"], f"{foot} foot", 1.0))
    return sorted(found, key=lambda c: c.severity, reverse=True)


class Coach:
    """Picks one cue per shot and avoids nagging about the same thing every time."""

    def __init__(self, min_severity: float = 0.15):
        self.min_severity = min_severity
        self.history: List[Optional[str]] = []

    def cue_for_shot(self, metrics: Dict[str, Optional[float]], shot_kind: str, outcome: str,
                     hand: str = "right") -> Cue:
        candidates = [c for c in issues(metrics, shot_kind, hand) if c.severity >= self.min_severity]
        cue = None
        if candidates:
            cue = candidates[0]
            # Same correction three times running: try the next issue, if it matters enough.
            if len(self.history) >= 2 and self.history[-1] == self.history[-2] == cue.metric and len(candidates) > 1:
                cue = candidates[1]
        if cue is None:
            cue = Cue("praise", PRAISE_MAKE if outcome == "make" else PRAISE_MISS)
        self.history.append(cue.metric)
        return cue


def dribble_feedback(heights: List[float], times: List[float], hands: List[Optional[str]]) -> Optional[str]:
    """Feedback over a block of dribbles (call every ~15 dribbles)."""
    if len(times) < 5:
        return None
    hs = [h for h in heights if h is not None]
    if hs and sum(hs) / len(hs) > 0.65:
        return "Dribble lower - keep the ball below your waist"
    duration = times[-1] - times[0]
    if duration > 0 and (len(times) - 1) / duration < 1.4:
        return "Pick up the pace - quicker, harder dribbles"
    known = [h for h in hands if h]
    if len(known) >= 10:
        weak = min(("left", "right"), key=known.count)
        if known.count(weak) / len(known) < 0.2:
            return f"Use your {weak} hand more"
    return "Good dribbling - stay low and keep your eyes up"


def metric_label(metric: str) -> str:
    return TARGETS[metric][0] if metric in TARGETS else metric.replace("_", " ").capitalize()
