"""The organisers' reference baseline (more-than-a-hack-2027/starter/baseline.py), reproduced on
external matches.

Same two steps, same numbers:
  1. Where the input has a ball position, the team of the nearest player has it; where it has none,
     nobody (D).
  2. Centred majority vote over +/- `window` frames; ties go to home; D where nobody was seen.

Inputs are the UNCLEANED ones, like the original: the ball as delivered (degraded seed for IDSSE,
raw for Metrica) and the raw player positions. Positions are already in metres here.

One deliberate difference: the vote runs inside each half (the original runs straight across the
array, which on BAR-ATH crosses the half-time X gap). It changes at most `window` frames at the
period change, where the ball is almost always dead.
"""

import numpy as np
import pandas as pd

from . import config

HOME, AWAY, NOBODY = 0, 1, -1
VARIANTS = (("baseline_raw", 0), ("baseline_smooth25", 25), ("baseline_smooth50", 50),
            ("baseline_smooth125", 125))


def raw_players(match):
    """(frames, xy (n,S,2), team codes (S,)) from the raw standardized objects."""
    fr = pd.read_parquet(config.path("raw_frames", match=match), columns=["frame"])["frame"]
    frames = np.sort(fr.to_numpy(np.int64))
    o = pd.read_parquet(config.path("raw_objects", match=match), columns=["frame", "slot", "team", "x", "y"])
    slots = np.sort(o["slot"].unique())
    s_idx = pd.Series(np.arange(len(slots)), index=slots)
    f_idx = pd.Series(np.arange(len(frames)), index=frames)
    xy = np.full((len(frames), len(slots), 2), np.nan, np.float32)
    r, c = f_idx.loc[o["frame"]].to_numpy(), s_idx.loc[o["slot"]].to_numpy()
    xy[r, c, 0], xy[r, c, 1] = o["x"].to_numpy(), o["y"].to_numpy()
    team_of_slot = o.drop_duplicates("slot").set_index("slot")["team"].loc[slots].to_numpy()
    if not set(team_of_slot) <= {"H", "A"}:
        raise ValueError("{}: unexpected team codes {}".format(match, set(team_of_slot)))
    return frames, xy, np.where(team_of_slot == "H", HOME, AWAY).astype(np.int8)


def input_ball(match, seed):
    """(frames, periods, ball xy (n,2)) as delivered: degraded seed for IDSSE, raw for Metrica."""
    key, kw = ("degraded_frames", dict(seed=seed, match=match)) if seed is not None \
        else ("raw_frames", dict(match=match))
    f = pd.read_parquet(config.path(key, **kw), columns=["frame", "period", "ball_x", "ball_y"]).sort_values("frame")
    return (f["frame"].to_numpy(np.int64), f["period"].fillna(0).to_numpy(np.int16),
            f[["ball_x", "ball_y"]].to_numpy(np.float32))


def nearest_codes(xy, team, ball):
    """Per frame: team code of the nearest player to the ball, NOBODY without a ball or players."""
    d = (xy[..., 0] - ball[:, None, 0]) ** 2 + (xy[..., 1] - ball[:, None, 1]) ** 2
    d = np.where(np.isfinite(d), d, np.inf)
    best = np.argmin(d, axis=1)
    found = np.isfinite(d[np.arange(len(d)), best])
    return np.where(found, team[best], NOBODY).astype(np.int8)


def smooth(codes, window, periods):
    """Majority vote over +/- window frames inside each half; ties to home; D where none seen."""
    labels = np.full(len(codes), "D", dtype=object)
    if window <= 0:
        labels[codes == HOME], labels[codes == AWAY] = "H", "A"
        return labels
    cuts = np.r_[0, np.flatnonzero(np.diff(periods) != 0) + 1, len(codes)]
    for s, e in zip(cuts[:-1], cuts[1:]):
        c = codes[s:e]
        hc = np.concatenate(([0], np.cumsum(c == HOME)))
        ac = np.concatenate(([0], np.cumsum(c == AWAY)))
        i = np.arange(e - s)
        lo, hi = np.maximum(0, i - window), np.minimum(e - s, i + window + 1)
        h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
        seen = (h + a) > 0
        seg = np.full(e - s, "D", dtype=object)
        seg[seen] = np.where(h[seen] >= a[seen], "H", "A")
        labels[s:e] = seg
    return labels
