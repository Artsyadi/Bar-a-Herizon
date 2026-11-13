"""Phase 5 segmenter: frame probabilities -> H/A/D labels (merged Rulebook D4, D5, P6).

Two layers, each a two-state Viterbi with a cost per switch:
  1. dead vs live, from p_d (the dead-ball model's probability)
  2. inside each live interval found by layer 1, home vs away from p_h, decided afresh in every
     interval: nothing carries across a dead spell, so H -> D -> H stays two possessions (D5)

No minimum duration anywhere (D4): a spell survives when its evidence pays for the switches into
and out of it. Every frame gets H, A or D, never X (P6); frames outside a half (period 0) get D.

Two-state Viterbi with a symmetric switching cost c reduces to one recursion on the difference of
the two path scores: d_t = r_t + clip(d_{t-1}, -c, c), where r_t is the frame's log-odds. The
backtrace switches state exactly where the clip was active (proof in the docstring of _viterbi2).
"""

from itertools import accumulate

import numpy as np


def logodds(p, clip):
    p = np.clip(np.asarray(p, dtype=np.float64), clip, 1.0 - clip)
    return np.log(p) - np.log1p(-p)


def _viterbi2(r, cost):
    """Most likely 0/1 path for log-odds r (state 1 favoured where r > 0) with switch cost `cost`.

    Path scores: S1_t = e1_t + max(S1_{t-1}, S0_{t-1} - c), S0_t = e0_t + max(S0_{t-1}, S1_{t-1} - c).
    With d = S1 - S0 and r = e1 - e0: d_t = r_t + clip(d_{t-1}, -c, c).
    State 1 at t came from state 0 iff d_{t-1} < -c; state 0 came from state 1 iff d_{t-1} > c.
    """
    n = len(r)
    if n == 0:
        return np.zeros(0, dtype=bool)
    lo, hi = -cost, cost
    d = np.fromiter(accumulate(r.tolist(), lambda prev, x: x + (lo if prev < lo else hi if prev > hi else prev)),
                    dtype=np.float64, count=n)
    out = np.empty(n, dtype=bool)
    s = d[-1] > 0
    out[-1] = s
    dl = d.tolist()
    for t in range(n - 1, 0, -1):
        prev = dl[t - 1]
        if s and prev < lo:
            s = False
        elif (not s) and prev > hi:
            s = True
        out[t - 1] = s
    return out


def _viterbi2_tv(r, cost):
    """As _viterbi2, but cost[t] is the cost of switching between frame t-1 and frame t."""
    n = len(r)
    if n == 0:
        return np.zeros(0, dtype=bool)
    rl, cl = r.tolist(), cost.tolist()
    d = [0.0] * n
    prev = rl[0]
    d[0] = prev
    for t in range(1, n):
        c = cl[t]
        prev = rl[t] + (-c if prev < -c else c if prev > c else prev)
        d[t] = prev
    out = np.empty(n, dtype=bool)
    s = d[-1] > 0
    out[-1] = s
    for t in range(n - 1, 0, -1):
        c, p = cl[t], d[t - 1]
        if s and p < -c:
            s = False
        elif (not s) and p > c:
            s = True
        out[t - 1] = s
    return out


def _runs(mask):
    """(start, end) of each run of True in a boolean array."""
    m = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[::2], edges[1::2]))


def dead_mask(p_d, periods, cost_d, clip=1e-4):
    """Layer 1: True where play is dead (every frame outside a half is dead)."""
    p_d, periods = np.asarray(p_d), np.asarray(periods)
    r_d = logodds(p_d, clip)
    dead = np.ones(len(p_d), dtype=bool)
    for per in np.unique(periods):
        if per <= 0:
            continue
        idx = np.flatnonzero(periods == per)
        dead[idx] = _viterbi2(r_d[idx], cost_d)
    return dead


def segment(p_d, p_h, periods, cost_d, cost_h, clip=1e-4, dead=None):
    """Labels ('H', 'A', 'D') for one match/unit. p_d, p_h, periods: same length, frame order.
    `dead` (from dead_mask) may be passed to reuse layer 1 across several cost_h.
    `cost_h` is a number, or an array (one switching cost per frame, for the switch into that frame)."""
    p_d, p_h, periods = np.asarray(p_d), np.asarray(p_h), np.asarray(periods)
    if not (len(p_d) == len(p_h) == len(periods)):
        raise ValueError("p_d, p_h and periods must have the same length")
    if dead is None:
        dead = dead_mask(p_d, periods, cost_d, clip)
    labels = np.full(len(p_d), "D", dtype=object)
    r_h = logodds(p_h, clip)
    # live intervals never cross a half: a period change splits them
    live = ~dead & (periods > 0)
    cut = np.r_[False, np.diff(periods) != 0]
    for s, e in _runs(live):
        bounds = [s] + [int(i) for i in np.flatnonzero(cut[s:e]) + s if i > s] + [e]
        for a, b in zip(bounds[:-1], bounds[1:]):               # layer 2, fresh in each live interval
            home = _viterbi2(r_h[a:b], cost_h) if np.isscalar(cost_h) else _viterbi2_tv(r_h[a:b], cost_h[a:b])
            labels[a:b] = np.where(home, "H", "A")
    return labels


def change_cost(p_c, cost_h, weight, ref, clip=1e-4, min_cost=0.5):
    """Frame-dependent home/away switching cost from a change detector's probability p_c:
    cost_h + weight * (surprise - ref), surprise = -logit(p_c); ref = the typical surprise (median on
    the training data), so a typical frame keeps cost_h. weight 0 gives the constant cost_h."""
    s = -logodds(p_c, clip)
    return np.maximum(min_cost, cost_h + weight * (s - ref))
