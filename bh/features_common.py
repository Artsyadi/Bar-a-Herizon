"""Shared helpers: attacking direction, time windows, team aggregation. Players only.

Nothing in this module touches the ball.
"""

import warnings

import numpy as np


def segments(period):
    """Half-open [start, stop) runs of constant period. Windows never cross a boundary (D3)."""
    p = np.asarray(period)
    cuts = np.flatnonzero(np.diff(p) != 0) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [len(p)]))
    return list(zip(starts.tolist(), stops.tolist()))


def window_mean(x, segs, a, b, min_frac=0.5):
    """Mean of x over frames [t+a, t+b] (inclusive) inside the same segment, ignoring NaN.
    NaN where fewer than min_frac of the window's frames are finite."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    need = max(1.0, min_frac * (b - a + 1))
    for s, e in segs:
        v = x[s:e]
        fin = np.isfinite(v)
        c = np.concatenate(([0], np.cumsum(fin)))
        t = np.concatenate(([0.0], np.cumsum(np.where(fin, v, 0.0))))
        i = np.arange(len(v))
        lo = np.clip(i + a, 0, len(v))
        hi = np.clip(i + b + 1, 0, len(v))
        cnt = c[hi] - c[lo]
        tot = t[hi] - t[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:e] = np.where((cnt >= need) & (hi > lo), tot / np.maximum(cnt, 1), np.nan)
    return out.astype(np.float32)


def lag_diff(x, segs, k):
    """x[t] - x[t-k] inside the same segment; NaN where t-k is outside it."""
    x = np.asarray(x, dtype=np.float32)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for s, e in segs:
        if e - s > k:
            out[s + k:e] = x[s + k:e] - x[s:e - k]
    return out


def lag_values(x, segs, k):
    """x[t-k] (k > 0) or x[t+|k|] (k < 0) inside the same segment, NaN otherwise. Works on (n, ...)."""
    x = np.asarray(x)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for s, e in segs:
        if e - s > abs(k):
            if k > 0:
                out[s + k:e] = x[s:e - k]
            elif k < 0:
                out[s:e + k] = x[s - k:e]
            else:
                out[s:e] = x[s:e]
    return out


def windowed(name, x, segs, fps, cfg):
    """Base value + centred rolling means + (next - previous) change. Returns dict name -> array."""
    out = {name: np.asarray(x, dtype=np.float32)}
    for w in cfg["rolling_mean_s"]:
        h = int(round(w * fps / 2.0))
        out["{}_m{}s".format(name, w)] = window_mean(x, segs, -h, h)
    k = int(round(cfg["change_half_window_s"] * fps))
    out["{}_chg{}s".format(name, cfg["change_half_window_s"])] = (
        window_mean(x, segs, 1, k) - window_mean(x, segs, -k, -1))
    return out


def attack_direction(xy, team, period):
    """+1 where the home team attacks towards +x, -1 towards -x, NaN in period 0.

    Per half, the team whose deepest player (minimum x, median over the half) sits nearer the -x
    goal line defends that goal - in practice the goalkeeper. Checked against the standardized
    home_attacks_right on all 20 external halves (all matched).
    """
    n = xy.shape[0]
    d = np.full(n, np.nan, dtype=np.float32)
    home, away = team == "H", team == "A"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s, e in segments(period):
            if period[s] <= 0:
                continue
            mh = np.nanmedian(np.nanmin(xy[s:e][:, home, 0], axis=1))
            ma = np.nanmedian(np.nanmin(xy[s:e][:, away, 0], axis=1))
            if np.isfinite(mh) and np.isfinite(ma):
                d[s:e] = 1.0 if mh < ma else -1.0
    return d


def team_masks(team):
    return {"h": team == "H", "a": team == "A"}


def nanstat(fn, arr, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(arr, axis=axis).astype(np.float32)


def velocity(xy, segs, fps, k=2):
    """Centred velocity (n, S, 2) over +/- k frames; NaN where either end is missing or across a half."""
    ahead = lag_values(xy, segs, -k)
    behind = lag_values(xy, segs, k)
    return ((ahead - behind) / (2.0 * k / fps)).astype(np.float32)
