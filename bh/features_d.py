"""D features: players only (merged Rulebook 2.3 / 4.1-4.10, D6). Spec: docs/feature_spec_v1.md §2.

build_d_features(PlayerData) is the only entry point. It accepts nothing else, so no ball column
can reach the dead-ball model by construction (acceptance requirement 1).
"""

import numpy as np
import pandas as pd

from .data import PlayerData
from .features_common import (attack_direction, lag_diff, lag_values, nanstat, segments,
                              velocity, windowed)

GROUP_RULEBOOK = {
    "spd": ("D1 collective activity", "4.3, C7.4.10"),
    "both": ("D2 coordinated slowing", "4.3"),
    "dec": ("D3 deceleration", "4.3"),
    "shape": ("D4 shape", "4.8, C7.4.1-4"),
    "line": ("D5 defensive line", "C7.4.6"),
    "terr": ("D6 territory", "C7.4.7"),
    "stab": ("D7 shape change and stability", "4.8, C7.4.9"),
    "ko": ("D8 kick-off shape", "4.4"),
    "tight": ("D9 tight groups", "4.5"),
    "touch": ("D10 touchline clustering", "4.6"),
    "pa": ("D11 penalty-area clustering", "4.7"),
    "mut": ("D12 local mutual movement", "4.9"),
}


def _chunks(n, size=4000):
    for lo in range(0, n, size):
        yield lo, min(n, lo + size)


def build_d_features(pdata, cfg):
    """Returns (DataFrame indexed like pdata.frame, list of per-column metadata dicts)."""
    if not isinstance(pdata, PlayerData):
        raise TypeError("D features take PlayerData only (players, no ball) - Rulebook D6")
    fc, gc, wc = cfg["d_features"], cfg["gating"], cfg["windows"]
    fps, (L, W) = pdata.fps, pdata.pitch
    xy, V, team, period = pdata.xy, pdata.speed, pdata.team, pdata.period
    n = pdata.n
    segs = segments(period)
    valid = pdata.valid
    x, y = xy[..., 0], xy[..., 1]
    dir_h = attack_direction(xy, team, period)
    dirs = {"h": dir_h, "a": -dir_h}
    cols = {"h": team == "H", "a": team == "A"}

    base, meta = {}, {}

    def add(name, values, group, unit, desc):
        base[name] = np.asarray(values, dtype=np.float32)
        meta[name] = {"group": group, "unit": unit, "description": desc}

    # ---- gating (Rulebook 4.10: quality controls feature availability, it is not an input) ----
    cnt, gate, sgate = {}, {}, {}
    for t, m in cols.items():
        cnt[t] = valid[:, m].sum(axis=1)
        sv = (np.isfinite(V) & valid)[:, m].sum(axis=1)
        gate[t] = cnt[t] >= gc["min_valid_players"]
        with np.errstate(invalid="ignore", divide="ignore"):
            sgate[t] = gate[t] & (sv / np.maximum(cnt[t], 1) >= gc["min_speed_share"])

    def g(arr, t, speed=False):
        return np.where(sgate[t] if speed else gate[t], arr, np.nan).astype(np.float32)

    # ---- D1 collective activity, D3 deceleration -------------------------------------------
    med = {}
    for t, m in cols.items():
        v = V[:, m]
        nv = np.isfinite(v).sum(axis=1)
        med[t] = g(nanstat(np.nanmedian, v), t, True)
        add("d_spd_median_" + t, med[t], "spd", "m/s", "median player speed")
        add("d_spd_mean_" + t, g(nanstat(np.nanmean, v), t, True), "spd", "m/s", "mean player speed")
        with np.errstate(invalid="ignore", divide="ignore"):
            slow = (v < fc["slow_speed_mps"]).sum(axis=1) / np.maximum(nv, 1)
            act = (v > fc["active_speed_mps"]).sum(axis=1) / np.maximum(nv, 1)
        add("d_spd_slowshare_" + t, g(slow, t, True), "spd", "share", "share of players below the slow speed")
        add("d_spd_activeshare_" + t, g(act, t, True), "spd", "share", "share of players above the active speed")
        add("d_dec_median_" + t, lag_diff(med[t], segs, int(fc["deceleration_lag_s"] * fps)),
            "dec", "m/s", "median speed now minus 1 s earlier")

    # ---- D2 coordinated slowing (both teams) -------------------------------------------------
    # np.minimum / np.maximum propagate NaN: if either team is gated out, so is the pair
    add("d_both_median_min", np.minimum(med["h"], med["a"]), "both", "m/s",
        "lower of the two team median speeds")
    add("d_both_median_max", np.maximum(med["h"], med["a"]), "both", "m/s",
        "higher of the two team median speeds")
    add("d_both_slowshare_min", np.minimum(base["d_spd_slowshare_h"], base["d_spd_slowshare_a"]),
        "both", "share", "lower of the two slow shares (both teams slow together)")
    add("d_both_median_absdiff", np.abs(med["h"] - med["a"]), "both", "m/s",
        "difference between the team median speeds")

    # ---- D4 shape, D5 defensive line, D6 territory, D8 own half, D10 touchline ---------------
    cx, cy, wid, dep = {}, {}, {}, {}
    q_lo, q_hi = fc["shape_quantiles"]
    for t, m in cols.items():
        xt, yt, d = x[:, m], y[:, m], dirs[t][:, None]
        cx[t] = nanstat(np.nanmean, xt)
        cy[t] = nanstat(np.nanmean, yt)
        wid[t] = _nanq(yt, q_hi) - _nanq(yt, q_lo)
        dep[t] = _nanq(xt, q_hi) - _nanq(xt, q_lo)
        comp = nanstat(np.nanmean, np.hypot(xt - cx[t][:, None], yt - cy[t][:, None]))
        add("d_shape_width_" + t, g(wid[t], t), "shape", "m", "y spread (10th-90th percentile)")
        add("d_shape_depth_" + t, g(dep[t], t), "shape", "m", "x spread (10th-90th percentile)")
        add("d_shape_compact_" + t, g(comp, t), "shape", "m", "mean distance to team centroid")
        add("d_shape_cx_" + t, g(cx[t] * dirs[t], t), "shape", "m", "centroid x, + towards opponent goal")
        add("d_shape_abscy_" + t, g(np.abs(cy[t]), t), "shape", "m", "centroid distance from the long axis")
        # defensive line: distance to own goal line along x, 4 deepest after the single deepest
        dg = np.sort(xt * d + L / 2.0, axis=1)       # NaN sorted last
        k = fc["defensive_line_players"]
        line = nanstat(np.nanmean, dg[:, 1:1 + k])
        enough = np.isfinite(dg[:, k]) if dg.shape[1] > k else np.zeros(n, bool)
        add("d_line_height_" + t, g(np.where(enough, line, np.nan), t), "line", "m",
            "mean distance of the 4 deepest outfield players to own goal line")
        with np.errstate(invalid="ignore", divide="ignore"):
            opp = ((xt * d) > 0).sum(axis=1) / np.maximum(cnt[t], 1)
            touch = (np.abs(yt) > W / 2.0 - fc["touchline_band_m"]).sum(axis=1) / np.maximum(cnt[t], 1)
        opp = np.where(np.isfinite(dirs[t]), opp, np.nan)
        add("d_terr_opphalf_" + t, g(opp, t), "terr", "share", "share of players in the opponent half")
        add("d_ko_ownhalf_" + t, g(1.0 - opp, t), "ko", "share", "share of players in own half")
        add("d_touch_share_" + t, g(touch, t), "touch", "share", "share of players within 5 m of a touchline")

    add("d_ko_ownhalf_min", np.minimum(base["d_ko_ownhalf_h"], base["d_ko_ownhalf_a"]), "ko", "share",
        "lower of the two own-half shares (both teams in own halves)")
    add("d_ko_centre_count", np.where(gate["h"] & gate["a"],
                                     (np.hypot(x, y) <= fc["centre_circle_m"]).sum(axis=1), np.nan),
        "ko", "players", "players of either team within 9.15 m of the centre spot")

    # ---- D7 shape change and stability -------------------------------------------------------
    lag1 = int(fc["shape_change_lag_s"] * fps)
    lag2 = int(fc["stability_lag_s"] * fps)
    xy_back = lag_values(xy, segs, lag2)
    disp = np.hypot(xy[..., 0] - xy_back[..., 0], xy[..., 1] - xy_back[..., 1])
    for t, m in cols.items():
        dcx, dcy = lag_diff(cx[t], segs, lag1), lag_diff(cy[t], segs, lag1)
        add("d_stab_centroid_move_" + t, g(np.hypot(dcx, dcy), t), "stab", "m", "centroid displacement over 1 s")
        add("d_stab_width_change_" + t, g(np.abs(lag_diff(wid[t], segs, lag1)), t), "stab", "m", "absolute width change over 1 s")
        add("d_stab_depth_change_" + t, g(np.abs(lag_diff(dep[t], segs, lag1)), t), "stab", "m", "absolute depth change over 1 s")
        add("d_stab_player_move_" + t, g(nanstat(np.nanmean, disp[:, m]), t), "stab", "m",
            "mean player displacement over 2 s")

    # ---- D9 tight groups, D11 penalty areas, D12 mutual movement (pairwise, chunked) ----------
    vel = velocity(xy, segs, fps)
    tight_n = {t: np.full(n, np.nan, np.float32) for t in cols}
    tight_goal = {t: np.full(n, np.nan, np.float32) for t in cols}
    nnopp = np.full(xy.shape[:2], np.nan, np.float32)      # per player: distance to nearest opponent
    nnidx = np.full(xy.shape[:2], -1, np.int32)
    nnall = np.full(xy.shape[:2], np.nan, np.float32)      # per player: nearest other player (any team)
    is_h = cols["h"]
    for lo, hi in _chunks(n):
        p = xy[lo:hi].astype(np.float32)
        dd = np.sqrt(((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1))      # (c, S, S)
        S = dd.shape[1]
        eye = np.eye(S, dtype=bool)[None]
        dd_noself = np.where(eye, np.nan, dd)
        nnall[lo:hi] = nanstat(np.nanmin, dd_noself, axis=2)
        same = (team[:, None] == team[None, :])[None]
        for t, m in cols.items():
            within = (dd_noself < fc["tight_group_radius_m"]) & same & m[None, :, None] & m[None, None, :]
            core = within.sum(axis=2) >= fc["tight_group_min_teammates"]
            # members: core players plus anyone within the radius of a core teammate (so the two
            # ends of a line of players count too)
            tight = core | (within & core[:, None, :]).any(axis=2)
            tight_n[t][lo:hi] = tight.sum(axis=1)
            with np.errstate(invalid="ignore"):
                tx = np.where(tight, p[..., 0], np.nan)
                ty = np.where(tight, p[..., 1], np.nan)
            gx, gy = nanstat(np.nanmean, tx), nanstat(np.nanmean, ty)
            goal_x = -dirs[t][lo:hi] * L / 2.0
            tight_goal[t][lo:hi] = np.hypot(gx - goal_x, gy)
        opp = np.where((team[:, None] != team[None, :])[None], dd, np.nan)
        allnan = ~np.isfinite(opp).any(axis=2)
        nnidx[lo:hi] = np.where(allnan, -1, np.argmin(np.where(np.isfinite(opp), opp, np.inf), axis=2))
        nnopp[lo:hi] = nanstat(np.nanmin, opp, axis=2)

    persist = int(fc["tight_group_persist_s"] * fps)
    from .features_common import window_mean
    for t, m in cols.items():
        tn = np.where(gate[t], tight_n[t], np.nan)
        add("d_tight_count_" + t, tn, "tight", "players",
            "players in a tight group (>= 2 teammates within 1.2 m, or within 1.2 m of such a player)")
        add("d_tight_goaldist_" + t, g(np.where(tn > 0, tight_goal[t], np.nan), t), "tight", "m",
            "distance of the tight group's centre to own goal (NaN if no group)")
        add("d_tight_persist_" + t, window_mean((tn >= 3).astype(np.float32) * np.where(np.isfinite(tn), 1, np.nan),
                                                segs, -persist, 0),
            "tight", "share", "share of the last 1 s with a tight group of >= 3")

    # penalty areas, defined from the home team's perspective (own = area home defends)
    in_area = {}
    for side, sign in (("homeown", -1.0), ("homeopp", 1.0)):
        xd = x * dir_h[:, None]
        in_area[side] = (xd * sign > L / 2.0 - fc["penalty_area_depth_m"]) & (np.abs(y) < fc["penalty_area_half_width_m"])
    both_gate = gate["h"] & gate["a"]
    for side, ia in in_area.items():
        tot = ia.sum(axis=1).astype(np.float32)
        add("d_pa_{}_total".format(side), np.where(both_gate, tot, np.nan), "pa", "players",
            "players of both teams in the penalty area ({})".format(side))
        add("d_pa_{}_home".format(side), np.where(both_gate, ia[:, is_h].sum(axis=1), np.nan), "pa", "players",
            "home players in the penalty area ({})".format(side))
        add("d_pa_{}_away".format(side), np.where(both_gate, ia[:, ~is_h].sum(axis=1), np.nan), "pa", "players",
            "away players in the penalty area ({})".format(side))
        with np.errstate(invalid="ignore"):
            sp = nanstat(np.nanmean, np.where(ia, nnall, np.nan))
            vv = nanstat(np.nanmean, np.where(ia, V, np.nan))
        add("d_pa_{}_spacing".format(side), np.where(both_gate & (tot >= 2), sp, np.nan), "pa", "m",
            "mean nearest-neighbour distance inside the area")
        add("d_pa_{}_speed".format(side), np.where(both_gate & (tot >= 1), vv, np.nan), "pa", "m/s",
            "mean speed inside the area")

    # mutual movement
    nn_back = lag_values(nnopp, segs, 5)
    closing = (nnopp - nn_back) / (5.0 / fps)
    rows = np.arange(n)[:, None]
    safe_idx = np.where(nnidx >= 0, nnidx, 0)
    opp_vel = vel[rows, safe_idx]                                     # (n, S, 2)
    sp_p = np.hypot(vel[..., 0], vel[..., 1])
    sp_o = np.hypot(opp_vel[..., 0], opp_vel[..., 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (vel * opp_vel).sum(-1) / (sp_p * sp_o)
    moving = (sp_p > fc["following_min_speed_mps"]) & (sp_o > fc["following_min_speed_mps"]) & (nnidx >= 0)
    cos = np.where(moving, cos, np.nan)
    for t, m in cols.items():
        add("d_mut_nnopp_" + t, g(nanstat(np.nanmean, nnopp[:, m]), t), "mut", "m", "mean distance to nearest opponent")
        with np.errstate(invalid="ignore", divide="ignore"):
            close_share = (nnopp[:, m] < fc["opponent_close_m"]).sum(axis=1) / np.maximum(cnt[t], 1)
        add("d_mut_closeshare_" + t, g(close_share, t), "mut", "share", "share of players with an opponent within 2 m")
        add("d_mut_closing_" + t, g(nanstat(np.nanmean, closing[:, m]), t), "mut", "m/s",
            "mean closing speed to nearest opponent (negative = closing)")
        add("d_mut_following_" + t, g(nanstat(np.nanmean, cos[:, m]), t), "mut", "cosine",
            "mean cosine between own and nearest opponent's velocity (both moving)")

    # ---- windows -----------------------------------------------------------------------------
    out, out_meta = {}, []
    for name, arr in base.items():
        for col, values in windowed(name, arr, segs, fps, wc).items():
            out[col] = values
            grp = meta[name]["group"]
            out_meta.append({"name": col, "model": "D", "base": name, "group": GROUP_RULEBOOK[grp][0],
                             "rulebook": GROUP_RULEBOOK[grp][1], "unit": meta[name]["unit"],
                             "window": col[len(name):].lstrip("_") or "frame",
                             "description": meta[name]["description"]})
    df = pd.DataFrame(out, index=pd.Index(pdata.frame, name="frame"))
    return df, out_meta


def _nanq(a, q):
    """Row-wise NaN-ignoring quantile (linear interpolation, same as np.nanquantile's default);
    NaN for rows with no finite value. Vectorised: np.nanquantile loops row by row and took
    ~90% of the build time."""
    a = np.asarray(a, dtype=np.float32)
    s = np.sort(a, axis=1)                         # NaN sorted last
    k = np.isfinite(a).sum(axis=1)
    pos = q * np.maximum(k - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(k - 1, 0))
    frac = (pos - lo).astype(np.float32)
    rows = np.arange(a.shape[0])
    last = a.shape[1] - 1
    v = s[rows, np.clip(lo, 0, last)] * (1 - frac) + s[rows, np.clip(hi, 0, last)] * frac
    return np.where(k > 0, v, np.nan).astype(np.float32)
