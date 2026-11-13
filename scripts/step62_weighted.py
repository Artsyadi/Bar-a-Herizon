"""Step 6.2, round 2: H/A model retrained with extra weight on rows near true possession
changes. Settings fixed in config/models.json ("transition_weighting") before any result.

    PY=python
    $PY scripts/step62_weighted.py --train     # (weight x fold); repeat until "all done"
    $PY scripts/step62_weighted.py --grid      # segmenter scoring (weight x cost_h)
    $PY scripts/step62_weighted.py --report    # table + keep/undo verdict

Outputs (gitignored): runs/phase6/weighted/w<k>/{fold_<m>.txt, oof_<unit>.parquet}, grid.csv
"""

import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.models import PHASE4, feature_names, load_tables, models_config, params, predict  # noqa: E402
from bh.pipeline import ha_features  # noqa: E402
from bh.scoring import load_labels, score  # noqa: E402
from bh.segmenter import dead_mask, segment  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step62_change import near_switch, periods_of, switches  # noqa: E402

OUT = os.path.join(config.PROJECT, "runs", "phase6", "weighted")
GRID = os.path.join(OUT, "grid.csv")
CONTROL, NOISE = 0.5525, 0.016


def oof_path(w, uid):
    return os.path.join(OUT, "w{}".format(w), "oof_{}.parquet".format(uid))


def train(budget):
    cfg = models_config()["transition_weighting"]
    names = feature_names("HA")
    p, rounds = params("ha_model")
    jobs = [(w, tr, v) for w in cfg["grid_weight"] for tr, v in config.folds()
            if not all(os.path.isfile(oof_path(w, config.unit_id(m, s))) for m, s in config.units(v))]
    t0, n = time.time(), 0
    for w, train_m, val in jobs:
        if n and time.time() - t0 > budget:
            break
        t = time.time()
        ids = [config.unit_id(m, s) for tm in train_m for m, s in config.units(tm)]
        X, y, info = load_tables("HA", ids, names)
        wt = np.ones(len(y), np.float32)
        for m in train_m:
            frames, labels = load_labels(m)
            sw = switches(labels, periods_of(m))
            row_of = pd.Series(np.arange(len(frames)), index=frames)
            k = (info.match == m).to_numpy()
            near = near_switch(row_of.loc[info.frame[k]].to_numpy(), sw, cfg["window_frames"])
            wt[k] = np.where(near == 1, w, 1.0)
        booster = lgb.train(p, lgb.Dataset(X, label=y, weight=wt, feature_name=names), num_boost_round=rounds)
        del X
        os.makedirs(os.path.join(OUT, "w{}".format(w)), exist_ok=True)
        booster.save_model(os.path.join(OUT, "w{}".format(w), "fold_{}.txt".format(val)))
        for m, s in config.units(val):
            feats, _ = ha_features(m, s)
            pd.DataFrame({"frame": feats.index.to_numpy(), "p_h": predict(booster, feats).astype(np.float32)}).to_parquet(
                oof_path(w, config.unit_id(m, s)), index=False)
        n += 1
        print("weight {:>2} fold {:<14} weighted rows {:.3f} | {:.1f}s".format(w, val, float((wt > 1).mean()), time.time() - t))
    left = len(jobs) - n
    print("all done ({} this call)".format(n) if left == 0 else "done this call: {} | left: {}".format(n, left))


def grid():
    cfg, seg = models_config()["transition_weighting"], models_config()["segmenter"]
    done = set()
    if os.path.isfile(GRID):
        g = pd.read_csv(GRID)
        done = set(zip(g.weight, g.cost_h))
    rows = []
    for _, val in config.folds():
        d = pd.read_parquet(os.path.join(PHASE4, "d", "oof_{}.parquet".format(val)))
        per, gt = d.period.to_numpy(), d.label.to_numpy()
        dead = dead_mask(d.p_d.to_numpy(), per, seg["cost_d"], seg["prob_clip"])
        for m, s in config.units(val):
            uid = config.unit_id(m, s)
            for w in cfg["grid_weight"]:
                p_h = pd.read_parquet(oof_path(w, uid)).p_h.to_numpy()
                for c in cfg["grid_cost_h"]:
                    if (w, c) in done:
                        continue
                    lab = segment(d.p_d.to_numpy(), p_h, per, seg["cost_d"], c, seg["prob_clip"], dead=dead)
                    r = score(gt, lab, per)
                    live = np.isin(gt, ["H", "A"]) & (per > 0)
                    rows.append({"weight": w, "cost_h": c, "fold": val, "seed": s, "spellF1": r["spell_f1_mean"],
                                 "macroF1": r["macro_f1"], "pred_spells": r["n_pred_spells"],
                                 "true_spells": r["n_gt_spells"],
                                 "frame_team_acc": float(((p_h[live] >= 0.5) == (gt[live] == "H")).mean())})
    if rows:
        pd.DataFrame(rows).to_csv(GRID, mode="a", header=not os.path.isfile(GRID), index=False)
    print("scored {} rows".format(len(rows)))


def report():
    g = pd.read_csv(GRID)
    s = g.groupby(["weight", "cost_h"]).agg(spellF1=("spellF1", "mean"), macroF1=("macroF1", "mean"),
                                           pred=("pred_spells", "mean"), acc=("frame_team_acc", "mean"))
    pd.set_option("display.width", 220)
    print("A. mean over 30 dev units; control (Phase 4 model, cost_h 8) spellF1 {:.4f}, frame team accuracy 0.8896".format(CONTROL))
    print(s.round(4).to_string())
    best = s.spellF1.idxmax()
    ctrl = pd.read_csv(os.path.join(config.PROJECT, "runs", "phase5", "grid.csv"))
    ctrl = ctrl[(ctrl.cost_d == 64) & (ctrl.cost_h == 8)].groupby("fold").spellF1.mean()
    bf = g[(g.weight == best[0]) & (g.cost_h == best[1])].groupby("fold").spellF1.mean()
    gain, better = s.spellF1.loc[best] - CONTROL, int((bf > ctrl).sum())
    print("\nB. best weight {} cost_h {}:".format(*best))
    print(pd.DataFrame({"control": ctrl, "best": bf, "change": bf - ctrl}).round(4).to_string())
    print("\nKEEP RULE: gain {:+.4f} (needs > {}) | folds improved {}/6 (needs >= 5) -> {}".format(
        gain, NOISE, better, "KEEP" if gain > NOISE and better >= 5 else "UNDO"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in ("train", "grid", "report"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--budget", type=float, default=65.0)
    a = ap.parse_args()
    train(a.budget) if a.train else grid() if a.grid else report() if a.report else ap.print_help()


if __name__ == "__main__":
    main()
