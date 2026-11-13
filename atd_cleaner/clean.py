"""The cleaning steps. Pure numpy, any match, thresholds from config.

Order:
  1. ball:    remove impossible jumps (short runs reachable only at an impossible speed)
  2. ball:    interpolate short gaps - only with credible endpoints, compatible motion and no sign of
              an intervening interaction (merged Rulebook 1.7, C6.1)
  3. players: remove impossible speeds, per slot
  4. players: merge duplicate tracks (same team, near-identical position, sustained)

Status per value:  0 observed | 1 estimated | 2 unavailable
Flags (bitmask):   1 jump_removed | 2 interpolated | 8 duplicate_merged | 16 track_break | 32 speed_removed
(bit 4 is reserved for set-piece-inferred positions, which this package never creates)

Nothing here decides D, H or A. A missing value is a missing value.
"""

import numpy as np

OBSERVED, ESTIMATED, UNAVAILABLE = 0, 1, 2
FLAG_JUMP_REMOVED = 1
FLAG_INTERPOLATED = 2
FLAG_SETPIECE_INFERRED = 4          # reserved, not produced here
FLAG_DUPLICATE_MERGED = 8
FLAG_TRACK_BREAK = 16
FLAG_SPEED_REMOVED = 32

FLAG_NAMES = {1: "jump_removed", 2: "interpolated", 4: "setpiece_inferred (reserved)",
              8: "duplicate_merged", 16: "track_break", 32: "speed_removed"}
STATUS_NAMES = {0: "observed", 1: "estimated", 2: "unavailable"}


def _present(xy):
    return np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])


# ------------------------------------------------------------------------------------------------
# impossible jumps / speeds
# ------------------------------------------------------------------------------------------------

def remove_spikes(xy, fps, max_speed, min_segment_frames, period=None, max_passes=3, min_jump_m=0.0):
    """Remove short runs of points that can only be reached at an impossible speed.

    Observed points are split wherever reaching the next observed point would need more than
    max_speed (elapsed time counts, so a long gap is not a jump) AND the step is longer than
    min_jump_m (so frame-to-frame position jitter on a fast sprint is not mistaken for a jump),
    and at every period change. A
    segment shorter than min_segment_frames bounded by a speed break is a false detection and is
    removed. A break between two long segments (identity switch, or a false detection that
    persists) cannot be resolved by physics and is flagged, not removed.

    Returns (cleaned copy, removed mask, break mask).
    """
    xy = np.array(xy, dtype=np.float32, copy=True)
    n = xy.shape[0]
    removed = np.zeros(n, dtype=bool)
    breaks = np.zeros(n, dtype=bool)

    def speed_breaks(obs):
        step = np.diff(xy[obs].astype(np.float64), axis=0)
        dist = np.hypot(step[:, 0], step[:, 1])
        is_speed = (dist / (np.diff(obs) / fps) > max_speed) & (dist > min_jump_m)
        is_period = np.zeros_like(is_speed) if period is None else (np.diff(period[obs]) != 0)
        return is_speed, is_speed | is_period

    for _ in range(max_passes):
        obs = np.flatnonzero(_present(xy))
        if obs.size < 2:
            break
        is_speed, is_cut = speed_breaks(obs)
        seg = np.concatenate(([0], np.cumsum(is_cut)))
        n_seg = seg[-1] + 1
        seg_len = np.bincount(seg, minlength=n_seg)
        # a segment is suspect only if a SPEED break (not a period change) bounds it
        left_speed = np.zeros(n_seg, dtype=bool)
        right_speed = np.zeros(n_seg, dtype=bool)
        cut_idx = np.flatnonzero(is_cut)
        left_speed[seg[cut_idx + 1]] |= is_speed[cut_idx]
        right_speed[seg[cut_idx]] |= is_speed[cut_idx]
        drop_seg = (seg_len < min_segment_frames) & (left_speed | right_speed)
        drop = drop_seg[seg]
        if not drop.any():
            break
        idx = obs[drop]
        xy[idx] = np.nan
        removed[idx] = True

    obs = np.flatnonzero(_present(xy))
    if obs.size >= 2:
        is_speed, _ = speed_breaks(obs)
        breaks[obs[1:][is_speed]] = True
    return xy, removed, breaks


# ------------------------------------------------------------------------------------------------
# ball gaps
# ------------------------------------------------------------------------------------------------

def _velocity(xy, status, i, direction, window, fps):
    j = i + direction * window
    if j < 0 or j >= xy.shape[0] or status[i] != OBSERVED or status[j] != OBSERVED:
        return None
    a, b = (xy[j], xy[i]) if direction < 0 else (xy[i], xy[j])
    return (b.astype(np.float64) - a) / (window / fps)


def _nearest_team(players, team, frame, pos, radius):
    p = players[frame]
    d = np.hypot(p[:, 0] - pos[0], p[:, 1] - pos[1])
    d = np.where(np.isfinite(d), d, np.inf)
    j = int(np.argmin(d))
    return team[j] if d[j] <= radius else None


def fill_ball_gaps(ball, status, flags, conf, fps, period, th, players=None, team=None):
    """Linear interpolation over short gaps. Never across a period change or in period 0.

    A gap is skipped when:
      * it is longer than max_gap_frames
      * crossing it needs more than max_speed_mps
      * the ball's velocity before and after differs by more than max_endpoint_velocity_change_mps
        (a contact inside the gap); if velocity can't be measured, only very short gaps are filled
      * the team of the nearest player within interaction_radius_m differs at the two endpoints
        (evidence of an intervening interaction), if block_if_nearest_team_changes is on
    Modifies the arrays in place; returns the number of frames filled.
    """
    max_gap = int(th["max_gap_frames"])
    max_speed = float(th["max_speed_mps"])
    max_dv = float(th["max_endpoint_velocity_change_mps"])
    window = int(th["velocity_window_frames"])
    max_gap_no_v = int(th["max_gap_frames_without_velocity"])
    min_conf = float(th["min_estimated_confidence"])
    check_team = bool(th.get("block_if_nearest_team_changes", True)) and players is not None
    radius = float(th.get("interaction_radius_m", 2.0))

    obs = np.flatnonzero(status == OBSERVED)
    if obs.size < 2:
        return 0
    gaps = np.diff(obs) - 1
    filled = 0
    for k in np.flatnonzero((gaps >= 1) & (gaps <= max_gap)):
        a, b = obs[k], obs[k + 1]
        g = b - a - 1
        if period[a] != period[b] or period[a] <= 0:
            continue
        if np.hypot(*(ball[b].astype(np.float64) - ball[a])) / ((b - a) / fps) > max_speed:
            continue
        v_in = _velocity(ball, status, a, -1, window, fps)
        v_out = _velocity(ball, status, b, +1, window, fps)
        if v_in is not None and v_out is not None:
            if np.hypot(*(v_out - v_in)) > max_dv:
                continue
        elif g > max_gap_no_v:
            continue
        if check_team:
            ta = _nearest_team(players, team, a, ball[a], radius)
            tb = _nearest_team(players, team, b, ball[b], radius)
            if ta is not None and tb is not None and ta != tb:
                continue
        t = (np.arange(1, g + 1, dtype=np.float32) / (g + 1))[:, None]
        ball[a + 1:b] = ball[a] + t * (ball[b] - ball[a])
        status[a + 1:b] = ESTIMATED
        flags[a + 1:b] |= FLAG_INTERPOLATED
        conf[a + 1:b] = max(min_conf, 1.0 - g / (max_gap + 1.0))
        filled += int(g)
    return filled


# ------------------------------------------------------------------------------------------------
# players
# ------------------------------------------------------------------------------------------------

def clean_player_speeds(players, fps, th, period=None):
    players = np.array(players, dtype=np.float32, copy=True)
    n, s, _ = players.shape
    flags = np.zeros((n, s), dtype=np.uint8)
    for slot in range(s):
        if not np.isfinite(players[:, slot, 0]).any():
            continue
        xy, removed, breaks = remove_spikes(players[:, slot], fps, th["max_speed_mps"],
                                            th["min_segment_frames"], period,
                                            min_jump_m=float(th.get("min_jump_m", 0.0)))
        players[:, slot] = xy
        flags[removed, slot] |= FLAG_SPEED_REMOVED
        flags[breaks, slot] |= FLAG_TRACK_BREAK
    return players, flags


def _runs(frames):
    if frames.size == 0:
        return []
    cut = np.flatnonzero(np.diff(frames) != 1)
    starts = np.concatenate(([frames[0]], frames[cut + 1]))
    ends = np.concatenate((frames[cut], [frames[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def merge_duplicates(players, team, flags, th):
    """Same-team pairs within max_distance_m for >= min_frames: the higher slot is removed
    (flag duplicate_merged) and the lower keeps the mean. Opponents are never merged.

    With only_when_team_exceeds_max (default), a pair is a candidate only in frames where that
    team has more than max_players_per_team tracked players: a team cannot field a 12th player,
    so an extra track is the duplicate. Without this, teammates standing together (walls,
    celebrations, waiting at a stoppage) were merged - on clean IDSSE ~99% of such merges fell in
    dead-ball frames, erasing the player groupings the dead-ball model relies on (Rulebook 4.1, 4.5).
    Modifies players/flags in place; returns slot-frames removed."""
    max_d = float(th["max_distance_m"])
    min_frames = int(th["min_frames"])
    chunk = int(th.get("chunk_frames", 3000))
    gate = bool(th.get("only_when_team_exceeds_max", True))
    max_team = int(th.get("max_players_per_team", 11))
    n, s, _ = players.shape
    same = (team[:, None] == team[None, :]) & np.triu(np.ones((s, s), dtype=bool), k=1)
    teams = np.unique(team)
    hf, hk = [], []
    for lo in range(0, n, chunk):
        p = players[lo:lo + chunk]
        d = np.sqrt(((p[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1))
        close = (d < max_d) & same[None]
        if gate:
            valid = np.isfinite(p[:, :, 0])
            over_slot = np.zeros(valid.shape, dtype=bool)          # (frames, slots): slot's team is over
            for t in teams:
                cols = team == t
                over = valid[:, cols].sum(axis=1) > max_team
                over_slot[:, cols] = over[:, None]
            close &= over_slot[:, :, None]
        f, i, j = np.nonzero(close)
        hf.append(f + lo)
        hk.append(i.astype(np.int64) * s + j)
    if not hf:
        return 0
    f, key = np.concatenate(hf), np.concatenate(hk)
    if f.size == 0:
        return 0
    order = np.lexsort((f, key))
    f, key = f[order], key[order]
    removed = 0
    bounds = np.flatnonzero(np.diff(key)) + 1
    for part_f, part_k in zip(np.split(f, bounds), np.split(key, bounds)):
        a, b = divmod(int(part_k[0]), s)
        for start, end in _runs(part_f):
            if end - start + 1 < min_frames:
                continue
            seg = slice(start, end + 1)
            both = np.isfinite(players[seg, a, 0]) & np.isfinite(players[seg, b, 0])
            if not both.any():
                continue
            mean = (players[seg, a] + players[seg, b]) / 2.0
            players[seg, a] = np.where(both[:, None], mean, players[seg, a])
            players[seg, b] = np.where(both[:, None], np.nan, players[seg, b])
            flags[seg, b] = np.where(both, flags[seg, b] | FLAG_DUPLICATE_MERGED, flags[seg, b])
            removed += int(both.sum())
    return removed


# ------------------------------------------------------------------------------------------------
# position-based player speed (v0.2.1)
# ------------------------------------------------------------------------------------------------

def derived_speed(players, fps, period, window, max_speed=None):
    """Speed (m/s) from cleaned positions with a centred difference over +/- `window` frames.

    speed[t] = |p[t+w] - p[t-w]| / (2w / fps), only where p[t-w], p[t] and p[t+w] are all observed
    and in the same period; NaN everywhere else (never 0 - Rulebook 4.2). Values above max_speed
    are set to NaN rather than clipped: they come from an unresolved break, not from running.

    The centred window smooths the frame-to-frame position jitter of automatic tracking, so the same
    calculation gives comparable speeds on clean (IDSSE) and noisy (BAR-ATH) tracking.
    Returns (speed (n, S) float32, number of values set to NaN by the max_speed guard).
    """
    n = players.shape[0]
    w = int(window)
    out = np.full(players.shape[:2], np.nan, dtype=np.float32)
    if n <= 2 * w or w < 1:
        return out, 0
    a, mid, b = players[:-2 * w], players[w:n - w], players[2 * w:]
    ok = (np.isfinite(a[..., 0]) & np.isfinite(mid[..., 0]) & np.isfinite(b[..., 0]))
    same = (period[:-2 * w] == period[w:n - w]) & (period[w:n - w] == period[2 * w:])
    ok &= same[:, None]
    d = np.hypot(b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]) / (2.0 * w / fps)
    v = np.where(ok, d, np.nan).astype(np.float32)
    guarded = 0
    if max_speed is not None:
        too_fast = v > max_speed
        guarded = int(too_fast.sum())
        v[too_fast] = np.nan
    out[w:n - w] = v
    return out, guarded


# ------------------------------------------------------------------------------------------------
# everything
# ------------------------------------------------------------------------------------------------

def clean(tr, thresholds):
    """Run all steps on a Tracking. Returns (result dict of arrays, summary dict)."""
    bt, pt, dt = thresholds["ball"], thresholds["players"], thresholds["duplicates"]
    fps, period = tr.fps, tr.period
    n = tr.n_frames

    players, p_flags = clean_player_speeds(tr.players, fps, pt, period)
    dup_removed = merge_duplicates(players, tr.team, p_flags, dt)

    ball, removed, breaks = remove_spikes(tr.ball, fps, bt["max_speed_mps"],
                                          bt["min_segment_frames"], period)
    b_flags = np.zeros(n, dtype=np.uint8)
    b_flags[removed] |= FLAG_JUMP_REMOVED
    b_flags[breaks] |= FLAG_TRACK_BREAK
    b_status = np.where(_present(ball), OBSERVED, UNAVAILABLE).astype(np.uint8)
    b_conf = np.where(b_status == OBSERVED, 1.0, 0.0).astype(np.float32)
    w = int(bt["track_break_window_frames"])
    for i in np.flatnonzero(breaks):
        seg = slice(max(0, i - w), min(n, i + w + 1))
        b_conf[seg] = np.where(b_status[seg] == OBSERVED,
                               np.minimum(b_conf[seg], bt["track_break_confidence"]), b_conf[seg])
    filled = fill_ball_gaps(ball, b_status, b_flags, b_conf, fps, period, bt, players, tr.team)

    p_status = np.where(np.isfinite(players[:, :, 0]), OBSERVED, UNAVAILABLE).astype(np.uint8)
    speed_d, speed_guarded = derived_speed(players, fps, period, pt.get("speed_window_frames", 2),
                                           pt["max_speed_mps"])
    result = dict(ball=ball, ball_status=b_status, ball_flags=b_flags, ball_conf=b_conf,
                  players=players, player_status=p_status, player_flags=p_flags,
                  player_speed_derived=speed_d)
    summary = {
        "ball_jump_points_removed": int(removed.sum()),
        "ball_unresolved_breaks": int(breaks.sum()),
        "ball_frames_interpolated": int(filled),
        "ball_observed_frames": int((b_status == OBSERVED).sum()),
        "ball_estimated_frames": int((b_status == ESTIMATED).sum()),
        "ball_unavailable_frames": int((b_status == UNAVAILABLE).sum()),
        "player_speed_points_removed": int(((p_flags & FLAG_SPEED_REMOVED) > 0).sum()),
        "player_unresolved_breaks": int(((p_flags & FLAG_TRACK_BREAK) > 0).sum()),
        "duplicate_slot_frames_removed": int(dup_removed),
        "player_speed_derived_measurable_share": float(
            np.isfinite(speed_d).sum() / max(1, (p_status == OBSERVED).sum())),
        "player_speed_derived_guarded": int(speed_guarded),
    }
    return result, summary


def check_invariants(tr, result):
    """Consistency checks. Returns a list of problems (empty = all good)."""
    problems = []
    b, st = result["ball"], result["ball_status"]
    if np.any(_present(b) & (st == UNAVAILABLE)):
        problems.append("ball position present where status is unavailable")
    if np.any(~_present(b) & (st != UNAVAILABLE)):
        problems.append("ball status available where position is NaN")
    if np.any((result["ball_conf"] < 0) | (result["ball_conf"] > 1)):
        problems.append("ball confidence outside [0, 1]")
    if np.any(np.isfinite(result["players"][:, :, 0]) != (result["player_status"] == OBSERVED)):
        problems.append("player status does not match positions")
    raw_missing = ~np.isfinite(tr.players[:, :, 0])
    if np.any(raw_missing & np.isfinite(result["players"][:, :, 0])):
        problems.append("a player position appeared where the input had none")
    est = st == ESTIMATED
    if np.any(est & (tr.period <= 0)):
        problems.append("ball estimated outside play")
    sd = result.get("player_speed_derived")
    if sd is not None:
        if np.any(np.isfinite(sd) & (result["player_status"] != OBSERVED)):
            problems.append("derived speed present where the player position is unavailable")
        if np.any(sd[np.isfinite(sd)] < 0):
            problems.append("negative derived speed")
    return problems
