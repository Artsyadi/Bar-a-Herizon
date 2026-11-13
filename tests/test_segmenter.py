"""Segmenter tests (step 5.1) on made-up probabilities. No match data needed.

    python tests/test_segmenter.py
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh.segmenter import _viterbi2, logodds, segment  # noqa: E402

LIVE, DEAD = 0.01, 0.99          # p_d values
HOME, AWAY = 0.95, 0.05          # p_h values


def runs(labels):
    return [(k, len(list(g))) for k, g in itertools.groupby(labels)]


def brute_force(r, cost):
    """Exhaustive best path, for checking the fast Viterbi on short inputs."""
    best, arg = -np.inf, None
    for path in itertools.product([0, 1], repeat=len(r)):
        p = np.array(path)
        score = np.where(p == 1, r, 0.0).sum() - cost * np.abs(np.diff(p)).sum()
        if score > best + 1e-12:
            best, arg = score, p
    return arg.astype(bool)


def test_fast_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 11))
        r = rng.normal(0, 3, n)
        cost = float(rng.choice([0.5, 2.0, 5.0, 12.0]))
        fast, slow = _viterbi2(r, cost), brute_force(r, cost)
        # ties can pick a different path with the same score; compare scores
        def score(p):
            p = p.astype(int)
            return np.where(p == 1, r, 0).sum() - cost * np.abs(np.diff(p)).sum()
        assert np.isclose(score(fast), score(slow)), (r, cost, fast, slow)


def test_strong_short_spell_survives_D4():
    n = 300
    p_d, p_h, per = np.full(n, LIVE), np.full(n, AWAY), np.ones(n)
    p_h[150:160] = 0.9999                          # 10 frames of very strong home evidence
    lab = segment(p_d, p_h, per, cost_d=16, cost_h=16)
    assert runs(lab) == [("A", 150), ("H", 10), ("A", 140)]


def test_weak_short_spell_is_absorbed():
    n = 300
    p_d, p_h, per = np.full(n, LIVE), np.full(n, AWAY), np.ones(n)
    p_h[150:153] = 0.7                             # 3 frames of weak home evidence
    lab = segment(p_d, p_h, per, cost_d=16, cost_h=16)
    assert runs(lab) == [("A", 300)]


def test_no_fixed_minimum_duration():
    n = 100
    p_d, p_h, per = np.full(n, LIVE), np.full(n, AWAY), np.ones(n)
    p_h[50] = 1 - 1e-6                             # one frame, log-odds clipped at +9.21
    lab = segment(p_d, p_h, per, cost_d=4, cost_h=4)   # switch in + out costs 8 < the frame's 9.21
    assert runs(lab) == [("A", 50), ("H", 1), ("A", 49)]


def test_home_dead_home_stays_two_possessions_D5():
    n = 300
    p_d, p_h, per = np.full(n, LIVE), np.full(n, HOME), np.ones(n)
    p_d[140:160] = DEAD                            # a 20-frame stoppage
    lab = segment(p_d, p_h, per, cost_d=8, cost_h=64)
    assert runs(lab) == [("H", 140), ("D", 20), ("H", 140)]


def test_fresh_decision_after_dead_ball():
    n = 300
    p_d, p_h, per = np.full(n, LIVE), np.full(n, HOME), np.ones(n)
    p_d[140:160] = DEAD
    p_h[160:] = 0.4                                # mild away evidence after the stoppage
    lab = segment(p_d, p_h, per, cost_d=8, cost_h=64)
    # nothing carries across D: the second interval follows its own evidence (A), with no switch cost
    assert runs(lab) == [("H", 140), ("D", 20), ("A", 140)]


def test_only_HAD_on_every_frame_and_halves_kept_apart():
    rng = np.random.default_rng(1)
    n = 2000
    per = np.r_[np.zeros(50), np.ones(950), np.full(950, 2), np.zeros(50)].astype(int)
    lab = segment(rng.random(n), rng.random(n), per, cost_d=8, cost_h=8)
    assert set(lab) <= {"H", "A", "D"} and len(lab) == n
    assert (lab[per == 0] == "D").all()


def test_time_varying_cost_equals_constant_and_brute_force():
    from bh.segmenter import _viterbi2_tv
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(1, 10))
        r = rng.normal(0, 3, n)
        c = rng.choice([0.5, 2.0, 6.0], n)
        assert np.array_equal(_viterbi2_tv(r, np.full(n, 3.0)), _viterbi2(r, 3.0))
        def sc(p):
            p = p.astype(int)
            return np.where(p == 1, r, 0).sum() - (c[1:] * np.abs(np.diff(p))).sum()
        best = max(sc(np.array(q)) for q in itertools.product([0, 1], repeat=n))
        assert np.isclose(sc(_viterbi2_tv(r, c)), best)


def test_cheap_switch_where_change_is_likely():
    n = 300
    p_d, p_h, per = np.full(n, LIVE), np.full(n, AWAY), np.ones(n)
    p_h[150:156] = 0.8                             # 6 frames of mild home evidence: 6 x 1.39 = 8.3
    cost = np.full(n, 16.0)
    assert runs(segment(p_d, p_h, per, 8, cost)) == [("A", 300)]
    cost[150] = cost[156] = 2.0                    # a detected change in and out
    assert runs(segment(p_d, p_h, per, 8, cost)) == [("A", 150), ("H", 6), ("A", 144)]


def test_logodds_clip():
    assert np.isclose(logodds([1.0], 1e-4)[0], np.log(1 - 1e-4) - np.log(1e-4))
    assert np.isclose(logodds([0.5], 1e-4)[0], 0.0)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:
            import traceback
            failed += 1
            print("FAIL", name, "-", repr(exc))
            traceback.print_exc(limit=2)
    print("{} passed, {} failed".format(len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
