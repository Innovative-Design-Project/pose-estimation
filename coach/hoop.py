"""Rim location. The camera is fixed during a session, so the rim is marked once."""
import json
from dataclasses import asdict, dataclass
from typing import Optional

import cv2


@dataclass
class Hoop:
    """Box around the rim opening (front to back edge of the rim, as seen in the image)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self):
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def rim_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Hoop":
        with open(path) as f:
            return cls(**json.load(f))

    @classmethod
    def parse(cls, text: str) -> "Hoop":
        x1, y1, x2, y2 = (float(v) for v in text.split(","))
        return cls(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def draw(self, frame, color=(0, 165, 255)):
        cv2.rectangle(frame, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), color, 2)
        cv2.putText(frame, "rim", (int(self.x1), int(self.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def calibrate(frame) -> Optional[Hoop]:
    """Ask the user to drag a box around the rim; Enter/Space confirms, c/Esc skips."""
    window = "Drag a box around the RIM, then press Enter (c to skip)"
    x, y, w, h = cv2.selectROI(window, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    cv2.waitKey(1)
    if w < 5 or h < 2:
        return None
    return Hoop(float(x), float(y), float(x + w), float(y + h))
