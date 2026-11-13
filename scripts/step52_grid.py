"""Step 5.2: choose the two switching costs on the 6 dev folds.

The grid is fixed in config/models.json ("segmenter") before any result. Each pair is scored with the
official scorer on all 30 dev units, using the Phase 4 out-of-fold probabilities (so every unit is
scored by models that never saw its match).

    PY=python
    $PY scripts/step52_grid.py              # resumable; repeat until "all done"
    $PY scripts/step52_grid.py --report     # grid table, best pair, edge check
    $PY scripts/step52_grid.py --choose     # write the best pair into config/models.json

Output (gitignored): runs/phase5/grid.csv (one row per pair x unit)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.models import PHASE4, models_config  # noqa: E402
from bh.scoring import score  # noqa: E402
from bh.segmenter import dead_mask, segment  # noqa: E402

OUT = os.path.join(config.PROJECT, "runs", "phase5")
GRID = os.path.join(OUT, "grid.csv")
BASELINE_FLOOR = 0.2164          # step 3.2, smooth50 on the same 30 units
FRAMEWISE = 0.3322               # step 4 combination without a segmenter


def load_oof():
    d, h = {}, {}
    for _, val in config.folds():
        d[val] = pd.read_parquet(os.path.join(PHASE4, "d", "oof_{}.parquet".format(val)))
        for m, s in config.units(val):
            uid = config.unit_id(m, s)
            h[uid] = pd.read_parquet(os.path.join(PHASE4, "ha", "oof_{}.parquet".format(uid)))
            assert np.array_equal(h[uid].frame.to_numpy(), d[val].frame.to_numpy())
    return d, h


def run(budget):
    seg = models_config()["segmenter"]
    clip = seg["prob_clip"]
    pairs = [(cd, ch) for cd in seg["grid_cost_d"] for ch in seg["grid_cost_h"]]
    done = set()
    if os.path.isfile(GRID):
        g = pd.read_csv(GRID)
        done = set(zip(g.cost_d, g.cost_h))
    todo = [p for p in pairs if p not in done]
    if not todo:
        print("all done (0 this call)")
        return
    d, h = load_oof()
    t0, n = time.time(), 0
    for cd in sorted({p[0] for p in todo}):
        if n and time.time() - t0 > budget:
            break
        dead = {val: dead_mask(d[val].p_d.to_numpy(), d[val].period.to_numpy(), cd, clip) for val in d}
        for ch in [p[1] for p in todo if p[0] == cd]:
            if n and time.time() - t0 > budget:
                break
            t = time.time()
            rows = []
            for _, val in config.folds():
                per, gt = d[val].period.to_numpy(), d[val].label.to_numpy()
                for m, s in config.units(val):
                    uid = config.unit_id(m, s)
                    lab = segment(d[val].p_d.to_numpy(), h[uid].p_h.to_numpy(), per, cd, ch, clip, dead=dead[val])
                    r = score(gt, lab, per)
                    rows.append({"cost_d": cd, "cost_h": ch, "fold": val, "seed": s, "spellF1": r["spell_f1_mean"],
                                 "macroF1": r["macro_f1"], "alive_dead_f1": r["alive_dead_f1"],
                                 "team_acc": r["team_acc_gt_alive"], "pred_spells": r["n_pred_spells"],
                                 "true_spells": r["n_gt_spells"]})
            os.makedirs(OUT, exist_ok=True)
            pd.DataFrame(rows).to_csv(GRID, mode="a", header=not os.path.isfile(GRID), index=False)
            n += 1
            print("cost_d {:>3} cost_h {:>3} | spellF1 {:.4f} | {:.1f}s".format(
                cd, ch, pd.DataFrame(rows).spellF1.mean(), time.time() - t))
    left = len(todo) - n
    print("all done ({} this call)".format(n) if left == 0 else "done this call: {} | left: {}".format(n, left))


def summary():
    g = pd.read_csv(GRID)
    return g, g.groupby(["cost_d", "cost_h"]).agg(
        spellF1=("spellF1", "mean"), spellF1_worst=("spellF1", "min"), macroF1=("macroF1", "mean"),
        alive_dead_f1=("alive_dead_f1", "mean"), pred_spells=("pred_spells", "mean"),
        true_spells=("true_spells", "mean"), units=("spellF1", "size"))


def report():
    seg = models_config()["segmenter"]
    g, s = summary()
    pd.set_option("display.width", 220)
    print("A. mean spellF1 over the 30 dev units (rows: dead/live cost, columns: home/away cost)")
    print(s.spellF1.unstack().round(4).to_string())
    print("\n   mean macroF1")
    print(s.macroF1.unstack().round(4).to_string())
    print("\n   mean predicted possessions (true: {:.0f})".format(s.true_spells.iloc[0]))
    print(s.pred_spells.unstack().round(0).to_string())
    best = s.spellF1.idxmax()
    b = s.loc[best]
    print("\nB. best pair cost_d={} cost_h={}: spellF1 {:.4f} (worst unit {:.4f}), macroF1 {:.4f}, alive/dead F1 {:.4f}, "
          "possessions {:.0f} vs {:.0f} true".format(best[0], best[1], b.spellF1, b.spellF1_worst, b.macroF1,
                                                     b.alive_dead_f1, b.pred_spells, b.true_spells))
    top = s.spellF1.sort_values(ascending=False).head(5)
    print("   top 5 pairs:", ", ".join("({}, {}) {:.4f}".format(k[0], k[1], v) for k, v in top.items()))
    edge = []
    if best[0] in (min(seg["grid_cost_d"]), max(seg["grid_cost_d"])):
        edge.append("cost_d")
    if best[1] in (min(seg["grid_cost_h"]), max(seg["grid_cost_h"])):
        edge.append("cost_h")
    print("   on the grid edge:", ", ".join(edge) if edge else "no")
    f = g[(g.cost_d == best[0]) & (g.cost_h == best[1])].groupby("fold").agg(
        spellF1=("spellF1", "mean"), spellF1_worst_seed=("spellF1", "min"), macroF1=("macroF1", "mean"),
        pred_spells=("pred_spells", "mean"), true_spells=("true_spells", "first"))
    print("\nC. best pair per fold:")
    print(f.round(4).to_string())
    print("\nD. vs the floor: baseline {:.4f} | models without segmenter {:.4f} | with segmenter {:.4f} "
          "(gain over baseline {:+.4f})".format(BASELINE_FLOOR, FRAMEWISE, b.spellF1, b.spellF1 - BASELINE_FLOOR))
    complete = len(s) == len(seg["grid_cost_d"]) * len(seg["grid_cost_h"]) and (s.units == 30).all()
    print("STEP 5.2 grid:", "complete" if complete else "INCOMPLETE", "| beats the baseline by more than 0.04:",
          b.spellF1 - BASELINE_FLOOR > 0.04)


def choose():
    _, s = summary()
    best = s.spellF1.idxmax()
    path = os.path.join(config.CONFIG, "models.json")
    cfg = json.load(open(path, encoding="utf-8"))
    cfg["segmenter"]["cost_d"], cfg["segmenter"]["cost_h"] = int(best[0]), int(best[1])
    cfg["segmenter"]["_chosen"] = ("Best mean spellF1 over the 30 dev units in the fixed grid: {:.4f} "
                                   "(roadmap 5.2, runs/phase5/grid.csv).".format(s.spellF1.max()))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    print("written: cost_d={} cost_h={}".format(*best))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--choose", action="store_true")
    a = ap.parse_args()
    report() if a.report else choose() if a.choose else run(a.budget)


if __name__ == "__main__":
    main()
