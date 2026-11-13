"""The one in-memory format every loader produces and every step consumes.

Conventions
-----------
* positions in metres; `origin` says where (0, 0) is:
    'centre' - centre spot, x in [-L/2, L/2] along the length, y in [-W/2, W/2]   (the external standardized tables)
    'corner' - a corner,    x in [0, L],          y in [0, W]
  Cleaning only uses distances and speeds, so it works in either; outputs keep the input's origin.
* NaN = not observed. Missing speed stays NaN (never 0).
* `period` per frame: 1, 2, ... for play; 0 = outside play (before kick-off, half-time, after the
  end). Nothing is ever interpolated across a period change or inside period 0.
* `team` per slot: 'H' or 'A'. A slot is a track/player channel, not necessarily a person.
* `goalkeeper` per slot: 1 yes, 0 no, -1 unknown.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class Tracking:
    frame: np.ndarray            # (n,)   frame index as used by the output (e.g. video frame)
    period: np.ndarray           # (n,)   int, 0 = outside play
    ball: np.ndarray             # (n, 2) metres
    players: np.ndarray          # (n, S, 2) metres
    speed: np.ndarray            # (n, S) m/s as provided by the source (NaN if missing/absent)
    slot_ids: np.ndarray         # (S,)   source identifiers for the slots
    team: np.ndarray             # (S,)   'H' / 'A'
    goalkeeper: np.ndarray       # (S,)   1 / 0 / -1
    fps: float
    pitch: Tuple[float, float]
    origin: str = "centre"
    ball_z: Optional[np.ndarray] = None          # (n,) metres if the source has it
    ball_speed: Optional[np.ndarray] = None      # (n,) m/s if the source provides it
    present: Optional[np.ndarray] = None         # (n, S) bool: the source had a row/value here
    frame_columns: Dict[str, np.ndarray] = field(default_factory=dict)  # passed through untouched
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.ball.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.team.shape[0])

    def validate(self):
        n, s = self.n_frames, self.n_slots
        problems = []
        if self.players.shape != (n, s, 2):
            problems.append("players shape {} != ({}, {}, 2)".format(self.players.shape, n, s))
        if self.speed.shape != (n, s):
            problems.append("speed shape {} != ({}, {})".format(self.speed.shape, n, s))
        for name in ("frame", "period"):
            if getattr(self, name).shape != (n,):
                problems.append("{} must have shape ({},)".format(name, n))
        if not set(np.unique(self.team)) <= {"H", "A"}:
            problems.append("team values must be 'H' or 'A'")
        if self.origin not in ("centre", "corner"):
            problems.append("origin must be 'centre' or 'corner'")
        if problems:
            raise ValueError("invalid Tracking: " + "; ".join(problems))
        return self


def to_origin(xy: np.ndarray, pitch, from_origin: str, to_origin_: str) -> np.ndarray:
    """Shift positions between corner and centre origins."""
    if from_origin == to_origin_:
        return xy
    shift = np.array([pitch[0] / 2.0, pitch[1] / 2.0], dtype=xy.dtype)
    return xy - shift if to_origin_ == "centre" else xy + shift
