"""Load Phase 1 cleaned data into arrays.

Two loaders, deliberately separate (merged Rulebook D6):
  * load_players(match) -> PlayerData   - players, frame index and periods ONLY. It reads the frame
    index and periods from the raw frames file with an explicit column list, so no ball column is
    ever loaded. The D feature builder only ever receives a PlayerData.
  * load_ball(match, seed) -> BallData  - the cleaned ball for one unit, for the H/A branch.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

PLAYER_COLUMNS = ["frame", "slot", "team", "is_goalkeeper", "x", "y", "player_status",
                  "speed_derived_ms"]


@dataclass(frozen=True)
class PlayerData:
    match: str
    frame: np.ndarray        # (n,)
    period: np.ndarray       # (n,)
    xy: np.ndarray           # (n, S, 2) cleaned positions, metres, centre origin; NaN = unavailable
    speed: np.ndarray        # (n, S) speed_derived_ms; NaN = not measurable
    team: np.ndarray         # (S,) 'H' / 'A'
    goalkeeper: np.ndarray   # (S,) 1 / 0 / -1
    pitch: tuple             # (length, width) metres
    fps: float

    @property
    def valid(self):
        return np.isfinite(self.xy[:, :, 0])

    @property
    def n(self):
        return self.frame.shape[0]


@dataclass(frozen=True)
class BallData:
    unit: str
    frame: np.ndarray
    period: np.ndarray
    xy: np.ndarray           # (n, 2) cleaned ball; NaN = unavailable
    status: np.ndarray       # (n,) 0 observed, 1 estimated, 2 unavailable
    conf: np.ndarray         # (n,)


def load_players(match, pitch=None, fps=25.0):
    fr = pd.read_parquet(config.path("raw_frames", match=match), columns=["frame", "period"])
    fr = fr.sort_values("frame").reset_index(drop=True)
    frame = fr["frame"].to_numpy(np.int64)
    period = fr["period"].fillna(0).to_numpy(np.int16)
    row_of = pd.Series(np.arange(len(frame)), index=frame)

    o = pd.read_parquet(config.path("cleaned_players", match=match), columns=PLAYER_COLUMNS)
    slots = np.array(sorted(o["slot"].unique()))
    col_of = pd.Series(np.arange(len(slots)), index=slots)
    n, s = len(frame), len(slots)
    xy = np.full((n, s, 2), np.nan, dtype=np.float32)
    speed = np.full((n, s), np.nan, dtype=np.float32)
    r = row_of.loc[o["frame"].to_numpy()].to_numpy()
    c = col_of.loc[o["slot"].to_numpy()].to_numpy()
    xy[r, c, 0] = o["x"].to_numpy(np.float32)
    xy[r, c, 1] = o["y"].to_numpy(np.float32)
    speed[r, c] = o["speed_derived_ms"].to_numpy(np.float32)
    first = o.drop_duplicates("slot").set_index("slot")
    team = first.loc[slots, "team"].to_numpy()
    gk = first.loc[slots, "is_goalkeeper"].fillna(-1).to_numpy(np.int8)
    if pitch is None:
        pitch = tuple(config_features()["pitch_m_default"])
    return PlayerData(match, frame, period, xy, speed, team, gk, tuple(pitch), fps)


def load_ball(match, seed):
    uid = config.unit_id(match, seed)
    f = pd.read_parquet(config.path("cleaned_frames", unit=uid),
                        columns=["frame", "period", "ball_x", "ball_y", "ball_status", "ball_conf"])
    f = f.sort_values("frame").reset_index(drop=True)
    return BallData(uid, f["frame"].to_numpy(np.int64), f["period"].fillna(0).to_numpy(np.int16),
                    f[["ball_x", "ball_y"]].to_numpy(np.float32), f["ball_status"].to_numpy(np.int8),
                    f["ball_conf"].to_numpy(np.float32))


def config_features():
    import json
    import os
    with open(os.path.join(config.CONFIG, "features.json"), encoding="utf-8") as handle:
        raw = json.load(handle)
    return {k: v for k, v in raw.items() if not k.startswith("_")}
