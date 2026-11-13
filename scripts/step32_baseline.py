"""Step 3.2: the organisers' baseline on external matches -> the v1 floor.

Runs the four published variants (raw, +/-25, +/-50, +/-125 frames) on the 6 development matches
(all 5 seeds) and the 3 Metrica matches. The final test match (config/split.json) is refused.

Faithfulness check (first unit): our vote must equal the organisers' own `smooth()` from
starter/baseline.py applied half by half.

Resumable; --budget keeps each call under the 180 s limit.

    PY=python
    $PY scripts/step32_baseline.py            # repeat until "all done"
    $PY scripts/step32_baseline.py --report   # tables + checkpoint verdict

Output: runs/step32_baseline/results.csv (one row per unit x variant)
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.baseline import VARIANTS, input_ball, nearest_codes, raw_players, smooth  # noqa: E402
from bh.scoring import load_labels, score  # noqa: E402

OUT = os.path.join(config.path("runs"), "step32_baseline")
RESULTS = os.path.join(OUT, "results.csv")


def units_to_run():
    sp = config.split()
    matches = list(sp["dev_matches"]) + list(sp["comparison_matches"])
    assert not set(matches) & set(sp["test_matches"]), "test match must not be scored here"
    return [(m, s) for m in matches for _, s in config.units(m)]


def organisers_smooth():
    path = os.path.join(config.path("scorer_dir"), os.pardir, "starter", "baseline.py")
    spec = importlib.util.spec_from_file_location("organisers_baseline", os.path.normpath(path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(path), os.pardir)))
    spec.loader.exec_module(mod)
    return mod.smooth


def faithful(codes, periods):
    ref = organisers_smooth()
    cuts = np.r_[0, np.flatnonzero(np.diff(periods) != 0) + 1, len(codes)]
    for _, w in VARIANTS:
        ours = smooth(codes, w, periods)
        theirs = np.concatenate([ref(codes[s:e], w) for s, e in zip(cuts[:-1], cuts[1:])])
        if not np.array_equal(ours.astype(str), theirs.astype(str)):
            return False
    return True


def run(budget):
    done = pd.read_csv(RESULTS) if os.path.isfile(RESULTS) else pd.DataFrame(columns=["unit"])
    todo = [(m, s) for m, s in units_to_run() if config.unit_id(m, s) not in set(done["unit"])]
    t0, players, n = time.time(), {}, 0
    for m, s in todo:
        if time.time() - t0 > budget:
            break
        t = time.time()
        if m not in players:
            players = {m: raw_players(m)}
        pf, xy, team = players[m]
        bf, periods, ball = input_ball(m, s)
        lf, gt = load_labels(m)
        if not (np.array_equal(pf, bf) and np.array_equal(bf, lf)):
            raise ValueError("{}: frames differ between players, ball and labels".format(m))
        codes = nearest_codes(xy, team, ball)
        if done.empty and n == 0:
            ok = faithful(codes, periods)
            print("faithfulness vs organisers' smooth():", "identical" if ok else "DIFFERENT")
            if not ok:
                sys.exit(1)
        rows = []
        for name, w in VARIANTS:
            r = score(gt, smooth(codes, w, periods), periods)
            rows.append({"unit": config.unit_id(m, s), "match": m, "seed": -1 if s is None else s,
                         "variant": name, "window": w, "spellF1": r["spell_f1_mean"],
                         "macroF1": r["macro_f1"], "alive_dead_f1": r["alive_dead_f1"],
                         "team_acc": r["team_acc_gt_alive"], "pred_spells": r["n_pred_spells"],
                         "true_spells": r["n_gt_spells"], "ball_seen": float((codes >= 0).mean())})
        os.makedirs(OUT, exist_ok=True)
        pd.DataFrame(rows).to_csv(RESULTS, mode="a", header=not os.path.isfile(RESULTS), index=False)
        n += 1
        print("{:<18} {:>5.1f}s  smooth50 spellF1 {:.4f}".format(config.unit_id(m, s), time.time() - t,
                                                               rows[2]["spellF1"]))
    left = len(todo) - n
    print("all done ({} this call)".format(n) if left == 0 else "done this call: {} | left: {}".format(n, left))


def report():
    r = pd.read_csv(RESULTS)
    pd.set_option("display.width", 200)
    expected = {config.unit_id(m, s) for m, s in units_to_run()}
    complete = set(r["unit"]) == expected and len(r) == 4 * len(expected)
    dev = r[r.match.str.startswith("idsse")]

    print("A. all variants, mean over the 30 development units (and Metrica, 3 matches):")
    a = r.assign(group=np.where(r.match.str.startswith("idsse"), "dev", "metrica")).groupby(
        ["group", "variant", "window"])[["spellF1", "macroF1", "alive_dead_f1", "team_acc", "pred_spells", "true_spells"]].mean()
    print(a.sort_index(level=["group", "window"]).round(4).to_string())

    print("\nB. the published floor (baseline_smooth50) per match: mean and worst seed")
    s50 = r[r.variant == "baseline_smooth50"]
    b = s50.groupby("match").agg(seeds=("seed", "size"), spellF1_mean=("spellF1", "mean"),
                                 spellF1_worst=("spellF1", "min"), macroF1_mean=("macroF1", "mean"),
                                 macroF1_worst=("macroF1", "min"), pred_spells=("pred_spells", "mean"),
                                 true_spells=("true_spells", "first"), ball_seen=("ball_seen", "mean"))
    print(b.round(4).to_string())
    d50 = dev[dev.variant == "baseline_smooth50"]
    print("\nv1 FLOOR (smooth50, 6 dev matches x 5 seeds): spellF1 {:.4f} (worst unit {:.4f}) | macroF1 {:.4f}".format(
        d50.spellF1.mean(), d50.spellF1.min(), d50.macroF1.mean()))
    print("STEP 3.2 CHECK:", "PASSED" if complete else "INCOMPLETE ({} of {} units)".format(
        r.unit.nunique(), len(expected)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=float, default=140.0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.budget)


if __name__ == "__main__":
    main()
