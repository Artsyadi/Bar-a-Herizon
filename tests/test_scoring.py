"""Scoring wrapper tests on made-up labels.

    python tests/test_scoring.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh.scoring import pm, score, shift_labels, with_separators  # noqa: E402


def long_spells(n_per_period=3000, spell=60, seed=0):
    """Two periods of alternating H / D / A spells, every spell `spell` frames long."""
    pattern = np.array(["H"] * spell + ["D"] * spell + ["A"] * spell + ["D"] * spell, dtype=object)
    one = np.resize(pattern, n_per_period)
    labels = np.concatenate([one, one])
    periods = np.repeat([1, 2], n_per_period)
    return labels, periods


def test_self_is_perfect_and_constant_is_zero():
    gt, per = long_spells()
    assert score(gt, gt, per)["spell_f1_mean"] == 1.0
    assert score(gt, np.full(len(gt), "H", dtype=object), per)["spell_f1_mean"] == 0.0


def test_one_frame_offset_is_free_when_spells_are_long():
    gt, per = long_spells()
    assert score(gt, shift_labels(gt, per, 1), per)["spell_f1_mean"] == 1.0


def test_one_second_offset_costs():
    gt, per = long_spells()
    s = score(gt, shift_labels(gt, per, 25), per)["spell_f1_mean"]
    assert 0 < s < 1


def test_separator_splits_halves_and_drops_no_graded_frame():
    gt, per = long_spells()
    g, p = with_separators(gt, gt, per)
    runs = pm.valid_runs(pm.label_codes(g))
    assert len(runs) == 2 and len(g) == len(gt) + 1
    assert sum(b - a for a, b in runs) == len(gt)


def test_possession_does_not_continue_across_halftime():
    """H at the end of half 1 and H at the start of half 2 must count as two spells."""
    gt = np.array(["H"] * 100 + ["H"] * 100, dtype=object)
    per = np.repeat([1, 2], 100)
    assert score(gt, gt, per)["n_gt_spells"] == 2


def test_period_zero_is_not_graded():
    gt = np.array(["D"] * 50 + ["H"] * 100 + ["D"] * 50, dtype=object)
    per = np.array([0] * 50 + [1] * 100 + [0] * 50)
    r = score(gt, np.full(200, "A", dtype=object), per)
    assert r["n_graded"] == 100


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, "-", repr(exc))
    print("{} passed, {} failed".format(len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
