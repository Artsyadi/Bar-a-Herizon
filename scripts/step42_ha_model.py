"""Step 4.2: H/A model (home vs away) on live frames, all 5 seeds, the same 6 folds as 4.1.

Each fold trains on the every-5th-frame live rows of the 5 training dev matches x 5 seeds and
predicts EVERY frame of each of the 5 seeds of the held-out match (features rebuilt on demand).
Out-of-fold probabilities are kept for the Phase 5 segmenter. Accuracy is measured on true-live
graded frames only (the segmenter, not this model, decides dead vs live).

    PY=python
    $PY scripts/step42_ha_model.py            # one fold per call; repeat until "all done"
    $PY scripts/step42_ha_model.py --report

Outputs (gitignored): runs/phase4/ha/fold_<match>.txt
                      runs/phase4/ha/oof_<unit>.parquet   frame, period, label, p_h, ball_status, rule
                      runs/phase4/ha/importance.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.data import load_ball  # noqa: E402
from bh.models import PHASE4, feature_names, load_tables, models_config, predict, train  # noqa: E402
from bh.pipeline import ha_features  # noqa: E402
from bh.scoring import load_labels  # noqa: E402

OUT = os.path.join(PHASE4, "ha")
D_OUT = os.path.join(PHASE4, "d")


def oof_path(uid):
    return os.path.join(OUT, "oof_{}.parquet".format(uid))


def run(budget):
    names = feature_names("HA")
    rule = models_config()["trivial_ha_rule"]["feature"]
    os.makedirs(OUT, exist_ok=True)
    imp_path = os.path.join(OUT, "importance.csv")
    todo = [(tr, val) for tr, val in config.folds()
            if not all(os.path.isfile(oof_path(config.unit_id(m, s))) for m, s in config.units(val))]
    t0, n = time.time(), 0
    for train_m, val in todo:
        if n and time.time() - t0 > budget:
            break
        t = time.time()
        ids = [config.unit_id(m, s) for tm in train_m for m, s in config.units(tm)]
        X, y, _ = load_tables("HA", ids, names)
        booster = train("ha_model", X, y, names)
        n_rows = len(y)
        del X, y
        t_train = time.time() - t
        booster.save_model(os.path.join(OUT, "fold_{}.txt".format(val)))
        frames, labels = load_labels(val)
        period = pd.read_parquet(os.path.join(D_OUT, "oof_{}.parquet".format(val)), columns=["frame", "period"])
        assert np.array_equal(period.frame.to_numpy(), frames)
        for m, s in config.units(val):
            feats, _ = ha_features(m, s)
            assert np.array_equal(feats.index.to_numpy(), frames)
            pd.DataFrame({"frame": frames, "period": period.period.to_numpy(), "label": labels,
                          "p_h": predict(booster, feats).astype(np.float32),
                          "ball_status": load_ball(m, s).status.astype(np.int8),
                          "rule": feats[rule].to_numpy(np.float32)}).to_parquet(
                oof_path(config.unit_id(m, s)), index=False)
        imp = pd.DataFrame({"fold": val, "feature": booster.feature_name(),
                            "gain": booster.feature_importance("gain")})
        imp.to_csv(imp_path, mode="a", header=not os.path.isfile(imp_path), index=False)
        n += 1
        print("fold {:<14} train rows {:>7,} | train {:>5.1f}s | total {:>5.1f}s".format(
            val, n_rows, t_train, time.time() - t))
    left = len(todo) - n
    print("all done ({} this call)".format(n) if left == 0 else "done this call: {} | left: {}".format(n, left))


def report():
    thr = models_config()["decision_threshold"]
    rows, by_status = [], []
    for _, val in config.folds():
        for m, s in config.units(val):
            o = pd.read_parquet(oof_path(config.unit_id(m, s)))
            o = o[(o.period > 0) & o.label.isin(["H", "A"])]
            truth = (o.label == "H").to_numpy()
            pred = o.p_h.to_numpy() >= thr
            rule = np.nan_to_num(o.rule.to_numpy(), nan=0.0) >= 0          # NaN / tie -> home
            correct = pred == truth
            rows.append({"fold": val, "seed": s, "acc": correct.mean(), "AUC": roc_auc_score(truth, o.p_h),
                         "rule_acc": (rule == truth).mean()})
            for st, name in ((0, "observed"), (1, "estimated"), (2, "unavailable")):
                k = o.ball_status.to_numpy() == st
                by_status.append({"fold": val, "seed": s, "status": name, "share": k.mean(),
                                  "acc": correct[k].mean() if k.any() else np.nan})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    f = t.groupby("fold").agg(acc_mean=("acc", "mean"), acc_worst_seed=("acc", "min"), AUC_mean=("AUC", "mean"),
                              rule_acc_mean=("rule_acc", "mean"))
    print("A. team accuracy on true-live graded frames, per fold (5 seeds each, threshold {}):".format(thr))
    print(f.round(4).to_string())
    print("   mean accuracy {:.4f} (worst unit {:.4f}) | rule (last reliable margin) {:.4f} | AUC {:.4f}".format(
        t.acc.mean(), t.acc.min(), t.rule_acc.mean(), t.AUC.mean()))
    print("   beats the rule on every unit:", bool((t.acc > t.rule_acc).all()))

    b = pd.DataFrame(by_status).groupby("status").agg(share=("share", "mean"), acc_mean=("acc", "mean"),
                                                      acc_worst_unit=("acc", "min"))
    print("\nB. accuracy by ball status (share of true-live frames):")
    print(b.loc[["observed", "estimated", "unavailable"]].round(4).to_string())

    imp = pd.read_csv(os.path.join(OUT, "importance.csv"))
    meta = pd.read_csv(os.path.join(config.PROJECT, "docs", "features_list.csv")).set_index("name")
    g = imp.groupby("feature").gain.mean()
    g = (g / g.sum()).sort_values(ascending=False)
    print("\nC. top 15 features by mean gain share:")
    print(pd.DataFrame({"gain_share": g.head(15).round(4), "group": meta.loc[g.head(15).index, "group"],
                        "description": meta.loc[g.head(15).index, "description"].str[:60]}).to_string())
    print("\n   gain share by group:\n" + g.groupby(meta.loc[g.index, "group"]).sum().sort_values(ascending=False).round(3).to_string())
    done = len(t) == 30
    print("\nSTEP 4.2 numbers:", "complete (30 units, same folds as 4.1)" if done else "INCOMPLETE ({} units)".format(len(t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.budget)


if __name__ == "__main__":
    main()
