from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class FaceState:
    """
    Summarized output of the face tracker module.
    Intended as input for a face animation module.
    All values are normalized 0.0-1.0 unless stated otherwise.
    """

    timestamp: float = field(default_factory=time.time)

    # --- base expression ---
    expression: str   = "neutral"   # one of LABELS
    confidence: float = 0.0         # 0.0 - 1.0

    # --- eyes ---
    # EAR raw value — 0.0 = fully closed, ~0.35 = fully open
    ear_left:  float = 0.0
    ear_right: float = 0.0

    # normalized 0.0 (closed) to 1.0 (fully open)
    eye_open_left:  float = 0.0
    eye_open_right: float = 0.0

    # blink state
    blinking_left:  bool = False
    blinking_right: bool = False

    # --- eyebrows ---
    # raw pixel distance between brow center and eye top
    brow_raise_left:  float = 0.0
    brow_raise_right: float = 0.0

    # normalized 0.0 (neutral) to 1.0 (fully raised)
    brow_raise_left_norm:  float = 0.0
    brow_raise_right_norm: float = 0.0

    # --- mouth ---
    # MAR raw value — 0.0 = closed, ~0.6 = fully open
    mar: float = 0.0

    # normalized 0.0 (closed) to 1.0 (fully open)
    mouth_open_norm: float = 0.0

    # mouth width normalized — 0.0 (pursed) to 1.0 (wide smile)
    mouth_width_norm: float = 0.0

    # state flags
    mouth_open: bool = False

    # --- face detected ---
    face_detected: bool = False

    def to_dict(self) -> dict:
        """Serialize for sending over socket/ROS topic/queue."""
        return {
            "timestamp":           self.timestamp,
            "expression":          self.expression,
            "confidence":          round(self.confidence, 3),
            "eye_open_left":       round(self.eye_open_left, 3),
            "eye_open_right":      round(self.eye_open_right, 3),
            "blinking_left":       self.blinking_left,
            "blinking_right":      self.blinking_right,
            "brow_raise_left":     round(self.brow_raise_left_norm, 3),
            "brow_raise_right":    round(self.brow_raise_right_norm, 3),
            "mouth_open":          self.mouth_open,
            "mouth_open_norm":     round(self.mouth_open_norm, 3),
            "mouth_width_norm":    round(self.mouth_width_norm, 3),
        }