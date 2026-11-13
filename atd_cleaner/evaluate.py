"""Score the cleaner on external data where the truth is known.

Inputs (all Tracking objects of the same match, same frames):
  original - clean source tracking (e.g. IDSSE as delivered)
  degraded - the same match after the BAR-like degradation (gaps, false detections, duplicates, ...)
  cleaned  - result of clean() on `degraded`

Slot convention for the degraded data: original slots keep their slot ids; injected duplicate
tracks get NEW slot ids. Identity swaps simply move positions between existing slots.

A degraded value is "corrupted" if it differs from the original by more than `tolerance_m`, or
exists where the original has none; otherwise it is "genuine".

Metrics
  ball.jump_*        removal of corrupted ball points: recall, precision, genuine points removed
  ball.fill_*        frames interpolated, share of fillable gaps filled, error vs. truth by gap length
  ball.touch_*       if touch frames are given: share of gaps containing a touch that were filled
                     (lower is better - the cleaner should refuse those)
  players.spike_*    removal of corrupted player points: recall, precision, genuine removed
  players.duplicate_* injected duplicate slot-frames removed (recall); genuine slot-frames merged away
  players.break_*    starts of lasting corruptions (>= 5 frames, e.g. identity swaps) flagged as
                     track_break within +/- 2 frames
  rmse_*             position error vs. truth: *_before_all_frames on every degraded point; before
                     and after on the same points (those that survive cleaning)
  ball.live_*        fill metrics on live frames only, if labels are given
"""

import numpy as np

from .clean import (ESTIMATED, FLAG_DUPLICATE_MERGED, FLAG_INTERPOLATED, FLAG_JUMP_REMOVED,
                    FLAG_SPEED_REMOVED, FLAG_TRACK_BREAK, UNAVAILABLE)

GAP_BINS = [(1, 2), (3, 5), (6, 12), (13, 25), (26, 10 ** 9)]


def _ratio(a, b):
    return float(a) / float(b) if b else None


def _runs(mask):
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def _align_slots(original, degraded):
    """Index of each original slot in the degraded data (-1 if absent) + mask of injected slots."""
    pos = {sid: k for k, sid in enumerate(degraded.slot_ids.tolist())}
    idx = np.array([pos.get(sid, -1) for sid in original.slot_ids.tolist()])
    injected = np.ones(degraded.n_slots, dtype=bool)
    injected[idx[idx >= 0]] = False
    return idx, injected


def evaluate(original, degraded, cleaned, tolerance_m=1.0, touch_frames=None, max_gap_frames=None,
             live=None, persistent_frames=5):
    """live: optional bool array per frame (True = ball in play, e.g. label H or A). If given, ball
    fill metrics are also reported for live frames only - important on IDSSE, where the ball is
    frozen during dead balls and gaps there are trivially easy to fill."""
    if original.n_frames != degraded.n_frames or not np.array_equal(original.frame, degraded.frame):
        raise ValueError("original and degraded must cover the same frames")
    play = original.period > 0
    out = {"frames_in_play": int(play.sum())}

    # ---- ball -------------------------------------------------------------------------------
    ob, db = original.ball.astype(np.float64), degraded.ball.astype(np.float64)
    cb, cst, cfl = cleaned["ball"].astype(np.float64), cleaned["ball_status"], cleaned["ball_flags"]
    o_ok, d_ok = np.isfinite(ob[:, 0]), np.isfinite(db[:, 0])
    err = np.hypot(*(db - ob).T)
    corrupted = d_ok & play & (~o_ok | (err > tolerance_m))
    genuine = d_ok & play & o_ok & (err <= tolerance_m)
    removed = d_ok & play & ((cfl & FLAG_JUMP_REMOVED) > 0)
    out["ball"] = {
        "corrupted_points": int(corrupted.sum()),
        "jump_recall": _ratio((removed & corrupted).sum(), corrupted.sum()),
        "jump_precision": _ratio((removed & corrupted).sum(), removed.sum()),
        "genuine_points_removed": int((removed & genuine).sum()),
    }

    filled = (cst == ESTIMATED) & ((cfl & FLAG_INTERPOLATED) > 0)
    missing = ~d_ok & play
    gap_len = np.zeros(original.n_frames, dtype=np.int64)
    gaps = _runs(missing)
    for a, b in gaps:
        gap_len[a:b + 1] = b - a + 1
    scorable = filled & o_ok
    fe = np.hypot(*(cb - ob).T)
    by_bin = {}
    for lo, hi in GAP_BINS:
        in_bin = missing & o_ok & (gap_len >= lo) & (gap_len <= hi)
        s = scorable & in_bin
        label = "{}-{}".format(lo, hi) if hi < 10 ** 8 else "{}+".format(lo)
        by_bin[label] = {"missing_frames": int(in_bin.sum()), "filled": int(s.sum()),
                         "fill_rate": _ratio(s.sum(), in_bin.sum()),
                         "median_error_m": float(np.median(fe[s])) if s.any() else None,
                         "p90_error_m": float(np.percentile(fe[s], 90)) if s.any() else None}
    fillable = missing & o_ok & (gap_len <= (max_gap_frames or 10 ** 9))
    out["ball"].update({
        "frames_filled": int(filled.sum()),
        "fill_rate_of_fillable_gaps": _ratio((filled & fillable).sum(), fillable.sum()),
        "fill_median_error_m": float(np.median(fe[scorable])) if scorable.any() else None,
        "fill_p90_error_m": float(np.percentile(fe[scorable], 90)) if scorable.any() else None,
        "fill_by_gap_length": by_bin,
    })
    if live is not None:
        live = np.asarray(live, dtype=bool)
        sl = scorable & live
        fl = fillable & live
        out["ball"].update({
            "live_fill_rate_of_fillable_gaps": _ratio((filled & fl).sum(), fl.sum()),
            "live_fill_median_error_m": float(np.median(fe[sl])) if sl.any() else None,
            "live_fill_p90_error_m": float(np.percentile(fe[sl], 90)) if sl.any() else None,
        })
    if touch_frames is not None:
        touch = np.isin(original.frame, np.asarray(touch_frames))
        with_touch = [(a, b) for a, b in gaps if touch[a:b + 1].any()]
        filled_with_touch = sum(1 for a, b in with_touch if filled[a:b + 1].all())
        out["ball"].update({"gaps_with_touch": len(with_touch),
                            "touch_gaps_filled": filled_with_touch,
                            "touch_gap_fill_rate": _ratio(filled_with_touch, len(with_touch))})

    # ---- players ----------------------------------------------------------------------------
    idx, injected = _align_slots(original, degraded)
    keep = idx >= 0
    op = original.players[:, keep].astype(np.float64)
    dp = degraded.players[:, idx[keep]].astype(np.float64)
    pf = cleaned["player_flags"][:, idx[keep]]
    cp = cleaned["players"][:, idx[keep]].astype(np.float64)
    po, pd_ = np.isfinite(op[..., 0]), np.isfinite(dp[..., 0])
    perr = np.hypot(dp[..., 0] - op[..., 0], dp[..., 1] - op[..., 1])
    playm = play[:, None]
    p_corr = pd_ & playm & (~po | (perr > tolerance_m))
    p_gen = pd_ & playm & po & (perr <= tolerance_m)
    p_rem = pd_ & playm & ((pf & FLAG_SPEED_REMOVED) > 0)
    dup_rem_genuine = p_gen & ((pf & FLAG_DUPLICATE_MERGED) > 0)

    inj_present = np.isfinite(degraded.players[:, injected, 0]) & playm
    inj_removed = inj_present & ((cleaned["player_flags"][:, injected] & FLAG_DUPLICATE_MERGED) > 0)

    # lasting corruptions (e.g. identity swaps): runs of corrupted points >= persistent_frames.
    # Short ones are spikes and are scored by spike_recall instead.
    onsets = np.zeros_like(p_corr)
    for k in range(p_corr.shape[1]):
        for a, b in _runs(p_corr[:, k]):
            if b - a + 1 >= persistent_frames and a > 0 and p_gen[a - 1, k]:
                onsets[a, k] = True
    brk = (pf & FLAG_TRACK_BREAK) > 0
    near = brk.copy()
    for sh in (1, 2):
        near[sh:] |= brk[:-sh]
        near[:-sh] |= brk[sh:]
    out["players"] = {
        "corrupted_points": int(p_corr.sum()),
        "spike_recall": _ratio((p_rem & p_corr).sum(), p_corr.sum()),
        "spike_precision": _ratio((p_rem & p_corr).sum(), p_rem.sum()),
        "genuine_points_removed_as_spikes": int((p_rem & p_gen).sum()),
        "injected_duplicate_slots": int(injected.sum()),
        "duplicate_recall": _ratio(inj_removed.sum(), inj_present.sum()),
        "genuine_points_merged_away": int(dup_rem_genuine.sum()),
        "lasting_corruption_onsets": int(onsets.sum()),
        "break_flag_rate_at_lasting_onsets": _ratio((onsets & near).sum(), onsets.sum()),
    }

    # ---- overall error ----------------------------------------------------------------------
    both_b = play & o_ok & d_ok & (cst != UNAVAILABLE)
    out["rmse_ball_before_m"] = float(np.sqrt(np.mean(err[both_b] ** 2))) if both_b.any() else None
    out["rmse_ball_after_m"] = float(np.sqrt(np.mean(fe[both_b] ** 2))) if both_b.any() else None
    cp_ok = np.isfinite(cp[..., 0])
    both_p = playm & po & pd_ & cp_ok
    cerr = np.hypot(cp[..., 0] - op[..., 0], cp[..., 1] - op[..., 1])
    out["rmse_players_before_m"] = float(np.sqrt(np.mean(perr[both_p] ** 2))) if both_p.any() else None
    out["rmse_players_after_m"] = float(np.sqrt(np.mean(cerr[both_p] ** 2))) if both_p.any() else None
    all_p = playm & po & pd_
    out["rmse_players_before_all_frames_m"] = (float(np.sqrt(np.mean(perr[all_p] ** 2)))
                                               if all_p.any() else None)
    out["player_points_kept_share"] = _ratio(both_p.sum(), all_p.sum())
    all_b = play & o_ok & d_ok
    out["rmse_ball_before_all_frames_m"] = float(np.sqrt(np.mean(err[all_b] ** 2))) if all_b.any() else None
    return out


def to_markdown(report, title="Cleaning evaluation"):
    lines = ["# " + title, ""]

    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return "{:.4f}".format(v)
        return str(v)

    for section in ("ball", "players"):
        lines += ["## " + section, "", "| metric | value |", "|---|---|"]
        for k, v in report[section].items():
            if isinstance(v, dict):
                continue
            lines.append("| {} | {} |".format(k, fmt(v)))
        lines.append("")
    bins = report["ball"].get("fill_by_gap_length", {})
    if bins:
        lines += ["## ball fill by gap length (frames)", "",
                  "| gap | missing | filled | fill rate | median err m | p90 err m |",
                  "|---|---|---|---|---|---|"]
        for label, b in bins.items():
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                label, b["missing_frames"], b["filled"], fmt(b["fill_rate"]),
                fmt(b["median_error_m"]), fmt(b["p90_error_m"])))
        lines.append("")
    lines += ["## overall", "", "| metric | value |", "|---|---|"]
    for k in ("frames_in_play", "rmse_ball_before_all_frames_m", "rmse_ball_before_m",
              "rmse_ball_after_m", "rmse_players_before_all_frames_m", "rmse_players_before_m",
              "rmse_players_after_m", "player_points_kept_share"):
        lines.append("| {} | {} |".format(k, fmt(report.get(k))))
    return "\n".join(lines) + "\n"
