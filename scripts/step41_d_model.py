"""Step 4.1: D model (dead vs live), players-only features, 6-fold leave-one-match-out.

Each fold trains on the every-5th-frame D tables of 5 dev matches (one table per match: D features
do not depend on the seed) and predicts EVERY frame of the held-out match (features rebuilt on
demand). The out-of-fold probabilities are kept for the Phase 5 segmenter.

    PY=python
    $PY scripts/step41_d_model.py            # train folds; repeat until "all done"
    $PY scripts/step41_d_model.py --accept   # acceptance test (Rulebook 3.2)
    $PY scripts/step41_d_model.py --report   # checkpoint tables + verdict

Outputs (gitignored): runs/phase4/d/fold_<match>.txt        LightGBM model
                      runs/phase4/d/oof_<match>.parquet     frame, period, label, p_d, rule
                      runs/phase4/d/importance.csv           gain per fold
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.data import load_ball  # noqa: E402
from bh.models import (PHASE4, alive_f1, dead_f1, feature_names, load_tables, models_config,  # noqa: E402
                       predict, train)
from bh.pipeline import d_features  # noqa: E402
from bh.scoring import load_labels  # noqa: E402

import lightgbm as lgb  # noqa: E402

OUT = os.path.join(PHASE4, "d")
# IDSSE metadata: these two are Bundesliga, the other dev matches 2. Bundesliga
BUNDESLIGA = {"idsse_J03WMX", "idsse_J03WN1"}


def periods_of(match, frames):
    p = pd.read_parquet(config.path("raw_frames", match=match), columns=["frame", "period"]).set_index("frame")
    return p.loc[frames, "period"].fillna(0).to_numpy(np.int16)


def run(budget):
    names = feature_names("D")
    rule = models_config()["trivial_d_rule"]["feature"]
    os.makedirs(OUT, exist_ok=True)
    imp_path = os.path.join(OUT, "importance.csv")
    t0, n, todo = time.time(), 0, [f for f in config.folds()
                                   if not os.path.isfile(os.path.join(OUT, "oof_{}.parquet".format(f[1])))]
    for train_m, val in todo:
        if n and time.time() - t0 > budget:
            break
        t = time.time()
        X, y, _ = load_tables("D", train_m, names)
        booster = train("d_model", X, y, names)
        del X
        t_train = time.time() - t
        booster.save_model(os.path.join(OUT, "fold_{}.txt".format(val)))
        feats, _ = d_features(val)
        frames, labels = load_labels(val)
        assert np.array_equal(frames, feats.index.to_numpy())
        oof = pd.DataFrame({"frame": frames, "period": periods_of(val, frames), "label": labels,
                            "p_d": predict(booster, feats).astype(np.float32),
                            "rule": feats[rule].to_numpy(np.float32)})
        oof.to_parquet(os.path.join(OUT, "oof_{}.parquet".format(val)), index=False)
        imp = pd.DataFrame({"fold": val, "feature": booster.feature_name(),
                            "gain": booster.feature_importance("gain")})
        imp.to_csv(imp_path, mode="a", header=not os.path.isfile(imp_path), index=False)
        n += 1
        print("fold {:<14} train rows {:>7,} | train {:>5.1f}s | total {:>5.1f}s".format(
            val, len(y), t_train, time.time() - t))
    left = len(todo) - n
    print("all done ({} this call)".format(n) if left == 0 else "done this call: {} | left: {}".format(n, left))


def accept():
    """Rulebook 3.2: changing the ball information leaves D probabilities unchanged."""
    val = config.folds()[0][1]
    booster = lgb.Booster(model_file=os.path.join(OUT, "fold_{}.txt".format(val)))
    banned = set(json.load(open(os.path.join(config.CONFIG, "features.json")))["excluded_inputs"]["d_model"])
    inputs_ok = not any(f in banned or "ball" in f for f in booster.feature_name())
    feats, _ = d_features(val)

    def with_ball(seed, shuffle=False):
        b = load_ball(val, seed)
        cols = {"ball_x": b.xy[:, 0], "ball_y": b.xy[:, 1], "ball_status": b.status, "ball_conf": b.conf}
        if shuffle:
            perm = np.random.default_rng(0).permutation(len(b.frame))
            cols = {k: v[perm] for k, v in cols.items()}
        return feats.assign(**cols)

    p1 = predict(booster, with_ball(1))
    p_shuf = predict(booster, with_ball(1, shuffle=True))
    p3 = predict(booster, with_ball(3))
    same = np.array_equal(p1, p_shuf) and np.array_equal(p1, p3)
    print("model inputs contain no ball/quality column:", inputs_ok)
    print("fold {}: ball shuffled -> identical: {} | seed 3 ball -> identical: {} | max |diff| {:.1e}".format(
        val, np.array_equal(p1, p_shuf), np.array_equal(p1, p3),
        max(np.abs(p1 - p_shuf).max(), np.abs(p1 - p3).max())))
    print("ACCEPTANCE TEST:", "PASSED" if inputs_ok and same else "FAILED")


def report():
    thr = models_config()["decision_threshold"]
    below = models_config()["trivial_d_rule"]["dead_below_mps"]
    rows = []
    for _, val in config.folds():
        o = pd.read_parquet(os.path.join(OUT, "oof_{}.parquet".format(val)))
        o = o[o.period > 0]
        gt = o.label.to_numpy()
        pred = np.where(o.p_d.to_numpy() >= thr, "D", "H")
        rule = np.where(o.rule.to_numpy() < below, "D", "H")          # NaN -> live
        rows.append({"fold": val, "league": "BL" if val in BUNDESLIGA else "2.BL",
                     "alive_dead_F1": alive_f1(gt, pred), "dead_F1": dead_f1(gt, pred),
                     "AUC": roc_auc_score(gt == "D", o.p_d), "rule_alive_dead_F1": alive_f1(gt, rule),
                     "rule_dead_F1": dead_f1(gt, rule), "true_dead_share": float((gt == "D").mean()),
                     "pred_dead_share": float((pred == "D").mean())})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("A. per fold (every graded frame of the held-out match, threshold {}):".format(thr))
    print(t.round(4).to_string(index=False))
    print("   mean alive/dead F1 {:.4f} (rule {:.4f}) | mean dead F1 {:.4f} (rule {:.4f}) | mean AUC {:.4f}".format(
        t.alive_dead_F1.mean(), t.rule_alive_dead_F1.mean(), t.dead_F1.mean(), t.rule_dead_F1.mean(), t.AUC.mean()))
    beats = bool((t.alive_dead_F1 > t.rule_alive_dead_F1).all() and (t.dead_F1 > t.rule_dead_F1).all())
    print("   beats 'both teams slow' on every fold (alive/dead F1 and dead F1):", beats)
    print("   Bundesliga folds mean alive/dead F1 {:.4f} | 2. Bundesliga folds {:.4f}".format(
        t[t.league == "BL"].alive_dead_F1.mean(), t[t.league == "2.BL"].alive_dead_F1.mean()))

    imp = pd.read_csv(os.path.join(OUT, "importance.csv"))
    meta = pd.read_csv(os.path.join(config.PROJECT, "docs", "features_list.csv")).set_index("name")
    g = imp.groupby("feature").gain.mean()
    g = (g / g.sum()).sort_values(ascending=False)
    top = pd.DataFrame({"gain_share": g.head(20).round(4), "group": meta.loc[g.head(20).index, "group"],
                        "description": meta.loc[g.head(20).index, "description"].str[:60]})
    print("\nB. top 20 features by mean gain share across folds:")
    print(top.to_string())
    grp = g.groupby(meta.loc[g.index, "group"]).sum().sort_values(ascending=False)
    print("\n   gain share by group:\n" + grp.round(3).to_string())

    print("\nC. missing-value shortcut check (2. Bundesliga breaks): NaN share of the top 8 features, dead vs live")
    dev = config.split()["dev_matches"]
    tabs = pd.concat([pd.read_parquet(os.path.join(config.PROJECT, "data", "features", "train_d", m + ".parquet"),
                                      columns=["match", "label"] + list(g.head(8).index)) for m in dev])
    tabs["league"] = np.where(tabs.match.isin(BUNDESLIGA), "BL", "2.BL")
    tabs["state"] = np.where(tabs.label == "D", "dead", "live")
    nan = tabs.groupby(["league", "state"])[list(g.head(8).index)].agg(lambda s: s.isna().mean()).T
    print(nan.round(3).to_string())
    print("\nSTEP 4.1 numbers:", "beats rule on every fold" if beats else "DOES NOT beat the rule on every fold")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--accept", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    accept() if a.accept else report() if a.report else run(a.budget)


if __name__ == "__main__":
    main()
