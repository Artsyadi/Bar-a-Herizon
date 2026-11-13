"""H/A features: ball + players, for live frames (merged Rulebook 1.5, C7.3a, C7.4-C7.6, 1.7).
Spec: docs/feature_spec_v1.md §3. Written from the home team's side (+ = home).

ball_z is never used (BAR-ATH has no ball height).
"""

import numpy as np
import pandas as pd

from .data import BallData, PlayerData
from .features_common import (attack_direction, lag_values, nanstat, segments, velocity,
                              window_mean, windowed)

GROUP_RULEBOOK = {
    "ball": ("A1 ball state", "1.7, C7.6"),
    "bmot": ("A2 ball motion", "C7.3a"),
    "near": ("A3 nearest players", "1.5, C7.4.8"),
    "carr": ("A4 carrier relationship", "1.5, C7.3a, L1"),
    "pers": ("A5 persistence", "1.5, L1"),
    "last": ("A6 last reliable relationship", "L3, C7.6"),
    "ctx": ("A7 team context", "1.6, C7.4.5, C7.6"),
}


def _frames_since(flag, segs, cap):
    """Frames since flag was last True (0 where True), within segments, capped at `cap`.
    Before the first True in a segment: cap."""
    out = np.full(flag.shape, cap, dtype=np.float32)
    for s, e in segs:
        f = flag[s:e]
        idx = np.where(f, np.arange(e - s), -1)
        last = np.maximum.accumulate(idx)
        since = np.where(last >= 0, np.arange(e - s) - last, cap)
        out[s:e] = np.minimum(since, cap)
    return out


def _frames_until(flag, segs, cap):
    rev = [(s, e) for s, e in segs]
    out = np.full(flag.shape, cap, dtype=np.float32)
    for s, e in rev:
        f = flag[s:e][::-1]
        idx = np.where(f, np.arange(e - s), -1)
        last = np.maximum.accumulate(idx)
        until = np.where(last >= 0, np.arange(e - s) - last, cap)
        out[s:e] = np.minimum(until, cap)[::-1]
    return out


def _ffill(x, segs):
    """Forward-fill NaN inside segments (no carry across a half)."""
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for s, e in segs:
        v = x[s:e]
        idx = np.where(np.isfinite(v), np.arange(e - s), -1)
        last = np.maximum.accumulate(idx)
        out[s:e] = np.where(last >= 0, v[np.maximum(last, 0)], np.nan)
    return out


def _signed_run_seconds(code, segs, fps):
    """code: +1 home / -1 away / 0 none / NaN unknown. Seconds the current non-zero code has lasted
    (signed: + home, - away); 0 where code is 0, NaN where unknown."""
    out = np.full(code.shape, np.nan, dtype=np.float32)
    for s, e in segs:
        c = code[s:e]
        key = np.where(np.isfinite(c), c, 9.0)
        change = np.concatenate(([True], key[1:] != key[:-1]))
        start = np.maximum.accumulate(np.where(change, np.arange(e - s), 0))
        length = (np.arange(e - s) - start + 1) / fps
        out[s:e] = np.where(np.isfinite(c), np.where(c != 0, np.sign(c) * length, 0.0), np.nan)
    return out


def build_ha_features(pdata, bdata, cfg):
    if not isinstance(pdata, PlayerData) or not isinstance(bdata, BallData):
        raise TypeError("H/A features take PlayerData and BallData")
    if not np.array_equal(pdata.frame, bdata.frame):
        raise ValueError("player and ball frames differ")
    hc, gc, wc = cfg["ha_features"], cfg["gating"], cfg["windows"]
    fps = pdata.fps
    L = pdata.pitch[0]
    xy, team, period = pdata.xy, pdata.team, pdata.period
    n = pdata.n
    segs = segments(period)
    valid = pdata.valid
    cols = {"h": team == "H", "a": team == "A"}
    dir_h = attack_direction(xy, team, period)
    dirs = {"h": dir_h, "a": -dir_h}

    b = bdata.xy.astype(np.float32)
    status = bdata.status.astype(np.float32)
    avail = bdata.status != 2
    observed = bdata.status == 0
    base, meta, to_window = {}, {}, set()

    def add(name, values, group, unit, desc, window=True):
        base[name] = np.asarray(values, dtype=np.float32)
        meta[name] = {"group": group, "unit": unit, "description": desc}
        if window:
            to_window.add(name)

    # ---- A1 ball state ---------------------------------------------------------------------
    cap = int(hc["ball_gap_cap_s"] * fps)
    add("ha_ball_status", status, "ball", "code", "0 observed, 1 estimated, 2 unavailable", False)
    add("ha_ball_conf", bdata.conf, "ball", "0-1", "cleaner confidence of the ball position", False)
    add("ha_ball_since_obs_s", _frames_since(observed, segs, cap) / fps, "ball", "s",
        "seconds since the ball was last observed (capped)", False)
    add("ha_ball_until_obs_s", _frames_until(observed, segs, cap) / fps, "ball", "s",
        "seconds until the ball is next observed (capped)", False)

    # ---- A2 ball motion --------------------------------------------------------------------
    vb = velocity(b[:, None, :], segs, fps)[:, 0, :]
    sb = np.hypot(vb[:, 0], vb[:, 1])
    add("ha_bmot_speed", sb, "bmot", "m/s", "ball speed (centred difference, available frames)")
    k = int(round(hc["ball_speed_change_lag_s"] * fps))
    b_back, b_ahead = lag_values(b, segs, k), lag_values(b, segs, -k)
    vin = (b - b_back) / (k / fps)
    vout = (b_ahead - b) / (k / fps)
    add("ha_bmot_dv", np.hypot(*(vout - vin).T), "bmot", "m/s",
        "change of ball velocity, outgoing vs incoming over 0.4 s")
    add("ha_bmot_dspeed", np.hypot(*vout.T) - np.hypot(*vin.T), "bmot", "m/s",
        "outgoing minus incoming ball speed")

    # ---- A3 nearest players, A4 carrier ----------------------------------------------------
    d = np.hypot(xy[..., 0] - b[:, None, 0], xy[..., 1] - b[:, None, 1])    # NaN if either missing
    vp = velocity(xy, segs, fps)
    near_d, near_i = {}, {}
    for t, m in cols.items():
        dt = np.where(m[None, :], d, np.nan)
        allnan = ~np.isfinite(dt).any(axis=1)
        near_d[t] = np.where(allnan, np.nan, nanstat(np.nanmin, dt))
        near_i[t] = np.where(allnan, 0, np.argmin(np.where(np.isfinite(dt), dt, np.inf), axis=1))
        add("ha_near_dist_" + t, near_d[t], "near", "m", "distance from the ball to the nearest player")
        for r in hc["near_radii_m"]:
            add("ha_near_n{}m_{}".format(int(r), t), np.where(avail, (dt < r).sum(axis=1), np.nan),
                "near", "players", "players within {} m of the ball".format(r))
        rows = np.arange(n)
        v_t = vp[rows, near_i[t]]
        s_t = np.hypot(v_t[:, 0], v_t[:, 1])
        rel = np.hypot(vb[:, 0] - v_t[:, 0], vb[:, 1] - v_t[:, 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (vb * v_t).sum(axis=1) / (sb * s_t)
        moving = (sb > hc["carrier_min_speed_mps"]) & (s_t > hc["carrier_min_speed_mps"])
        add("ha_carr_relspeed_" + t, np.where(np.isfinite(near_d[t]), rel, np.nan), "carr", "m/s",
            "ball speed relative to the nearest player (ball-minus-player velocity)")
        add("ha_carr_cos_" + t, np.where(moving & np.isfinite(near_d[t]), cos, np.nan), "carr", "cosine",
            "cosine between ball and nearest-player movement (both moving)")
        back = lag_values(near_d[t], segs, 5)
        add("ha_carr_closing_" + t, (near_d[t] - back) / (5.0 / fps), "carr", "m/s",
            "change of nearest-player distance (negative = closing)")
    margin = near_d["a"] - near_d["h"]
    add("ha_near_margin", margin, "near", "m", "nearest away minus nearest home distance (+ = home closer)")

    # ---- A5 persistence ---------------------------------------------------------------------
    home_closer = np.where(np.isfinite(margin), (margin > 0).astype(np.float32), np.nan)
    for w in hc["persistence_windows_s"]:
        f = int(w * fps)
        add("ha_pers_homecloser_past{}s".format(w), window_mean(home_closer, segs, -f, 0, 0.2), "pers",
            "share", "share of the last {} s (ball available) with home closer".format(w), False)
        add("ha_pers_homecloser_next{}s".format(w), window_mean(home_closer, segs, 0, f, 0.2), "pers",
            "share", "share of the next {} s (ball available) with home closer".format(w), False)
    r = hc["persist_radius_m"]
    code = np.where(~avail, np.nan,
                    np.where((near_d["h"] <= r) & (near_d["h"] < near_d["a"]), 1.0,
                             np.where((near_d["a"] <= r) & (near_d["a"] < near_d["h"]), -1.0, 0.0)))
    add("ha_pers_run_s", _signed_run_seconds(code, segs, fps), "pers", "s",
        "seconds the same team has stayed nearest within 2 m (+ home, - away)", False)

    # ---- A6 last reliable relationship ------------------------------------------------------
    last_margin = _ffill(np.where(observed, margin, np.nan).astype(np.float32), segs)
    since_rel = _frames_since(observed & np.isfinite(margin), segs, int(hc["last_margin_cap_s"] * fps)) / fps
    add("ha_last_margin", last_margin, "last", "m", "last margin measured with an observed ball", False)
    add("ha_last_margin_age_s", np.where(np.isfinite(last_margin), since_rel, np.nan), "last", "s",
        "seconds since that last margin (capped)", False)

    # ---- A7 team context (no ball) ----------------------------------------------------------
    for t, m in cols.items():
        cnt = valid[:, m].sum(axis=1)
        gate = cnt >= gc["min_valid_players"]
        xt, yt = xy[:, m, 0], xy[:, m, 1]
        cx = nanstat(np.nanmean, xt)
        cy = nanstat(np.nanmean, yt)
        comp = nanstat(np.nanmean, np.hypot(xt - cx[:, None], yt - cy[:, None]))
        with np.errstate(invalid="ignore", divide="ignore"):
            opp = ((xt * dirs[t][:, None]) > 0).sum(axis=1) / np.maximum(cnt, 1)
        fwd = nanstat(np.nanmean, vp[:, m, 0] * dirs[t][:, None])
        spd = nanstat(np.nanmean, pdata.speed[:, m])
        for nm, v, unit, desc in (("cx", cx * dirs[t], "m", "team centroid x, + towards opponent goal"),
                                  ("opphalf", opp, "share", "share of players in the opponent half"),
                                  ("compact", comp, "m", "mean distance to team centroid"),
                                  ("fwdvel", fwd, "m/s", "mean player velocity towards the opponent goal"),
                                  ("speed", spd, "m/s", "mean player speed")):
            needs_dir = nm in ("cx", "opphalf", "fwdvel")
            mask = gate & np.isfinite(dirs[t]) if needs_dir else gate
            add("ha_ctx_{}_{}".format(nm, t), np.where(mask, v, np.nan), "ctx", unit, desc)

    # ---- windows ----------------------------------------------------------------------------
    out, out_meta = {}, []
    for name, arr in base.items():
        variants = windowed(name, arr, segs, fps, wc) if name in to_window else {name: arr}
        for col, values in variants.items():
            out[col] = values
            grp = meta[name]["group"]
            out_meta.append({"name": col, "model": "HA", "base": name, "group": GROUP_RULEBOOK[grp][0],
                             "rulebook": GROUP_RULEBOOK[grp][1], "unit": meta[name]["unit"],
                             "window": col[len(name):].lstrip("_") or "frame",
                             "description": meta[name]["description"]})
    df = pd.DataFrame(out, index=pd.Index(pdata.frame, name="frame"))
    return df, out_meta
