"""Step 6.2, round 1: a change-of-possession detector -> cheaper switching where a change is
likely, dearer where it is not (e.g. in ball gaps). Settings fixed in config/models.json
("change_model", "change_cost") before any result.

    PY=python
    $PY scripts/step62_change.py --train      # LOMO detector; repeat until "all done"
    $PY scripts/step62_change.py --detector   # hit rate of the detector on its own
    $PY scripts/step62_change.py --grid       # segmenter grid (weight x cost_h); repeat
    $PY scripts/step62_change.py --report     # grid table + keep/undo verdict
    $PY scripts/step62_change.py --choose     # only if kept: write weight/cost_h to config

Outputs (gitignored): runs/phase6/change/{fold_<m>.txt, oof_<unit>.parquet, ref.json, grid.csv}
"""

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.models import PHASE4, feature_names, load_tables, models_config, predict  # noqa: E402
from bh.pipeline import ha_features  # noqa: E402
from bh.scoring import load_labels, score  # noqa: E402
from bh.segmenter import change_cost, dead_mask, logodds, segment  # noqa: E402

OUT = os.path.join(config.PROJECT, "runs", "phase6", "change")
REF = os.path.join(OUT, "ref.json")
GRID = os.path.join(OUT, "grid.csv")
CONTROL = 0.5525      # Phase 5 chosen segmenter on the same 30 units
NOISE = 0.016         # 0.04 / sqrt(6), fixed in change_cost._keep_rule


def switches(labels, periods):
    """Frame indices i where possession passes H<->A between i-1 and i inside a half."""
    l, p = np.asarray(labels), np.asarray(periods)
    live = np.isin(l, ["H", "A"])
    i = np.flatnonzero(live[1:] & live[:-1] & (l[1:] != l[:-1]) & (p[1:] == p[:-1])) + 1
    return i


def near_switch(frames_idx, sw, window):
    """1 where a switch lies within +-window of each index (sorted switch array)."""
    if len(sw) == 0:
        return np.zeros(len(frames_idx), np.int8)
    pos = np.searchsorted(sw, frames_idx)
    left = np.abs(frames_idx - sw[np.clip(pos - 1, 0, len(sw) - 1)])
    right = np.abs(sw[np.clip(pos, 0, len(sw) - 1)] - frames_idx)
    return (np.minimum(left, right) <= window).astype(np.int8)


def periods_of(match):
    return pd.read_parquet(os.path.join(PHASE4, "d", "oof_{}.parquet".format(match)), columns=["period"]).period.to_numpy() \
        if os.path.isfile(os.path.join(PHASE4, "d", "oof_{}.parquet".format(match))) else \
        pd.read_parquet(config.path("raw_frames", match=match), columns=["frame", "period"]).sort_values("frame").period.fillna(0).to_numpy()


def match_targets(match, window):
    frames, labels = load_labels(match)
    sw = switches(labels, periods_of(match))
    return frames, sw, near_switch(np.arange(len(frames)), sw, window)


def lgb_params():
    c = models_config()["change_model"]
    p = {k: v for k, v in c.items() if not k.startswith("_") and k not in ("target_window_frames", "num_boost_round")}
    return p, c["num_boost_round"], c["target_window_frames"]


def train(budget):
    names = feature_names("HA")
    params, rounds, window = lgb_params()
    os.makedirs(OUT, exist_ok=True)
    refs = json.load(open(REF)) if os.path.isfile(REF) else {}
    todo = [(tr, v) for tr, v in config.folds() if v not in refs]
    t0, n = time.time(), 0
    for train_m, val in todo:
        if n and time.time() - t0 > budget:
            break
        t = time.time()
        ids = [config.unit_id(m, s) for tm in train_m for m, s in config.units(tm)]
        X, _, info = load_tables("HA", ids, names)
        y = np.zeros(len(info), np.int8)
        for m in train_m:
            frames, _, tgt = match_targets(m, window)
            row_of = pd.Series(np.arange(len(frames)), index=frames)
            k = (info.match == m).to_numpy()
            y[k] = tgt[row_of.loc[info.frame[k]].to_numpy()]
        booster = lgb.train(params, lgb.Dataset(X, label=y, feature_name=names), num_boost_round=rounds)
        ref = float(np.median(-logodds(booster.predict(X), 1e-4)))
        del X
        booster.save_model(os.path.join(OUT, "fold_{}.txt".format(val)))
        for m, s in config.units(val):
            feats, _ = ha_features(m, s)
            pd.DataFrame({"frame": feats.index.to_numpy(), "p_c": predict(booster, feats).astype(np.float32)}).to_parquet(
                os.path.join(OUT, "oof_{}.parquet".format(config.unit_id(m, s))), index=False)
        refs[val] = {"ref": ref, "train_rows": int(len(y)), "positive_share": float(y.mean())}
        json.dump(refs, open(REF, "w"), indent=1)
        n += 1
        print("fold {:<14} rows {:>7,} positives {:.3f} ref {:.2f} | {:.1f}s".format(
            val, len(y), y.mean(), ref, time.time() - t))
    left = len(todo) - n
    print("all done ({} this call)".format(n) if left == 0 else "done this call: {} | left: {}".format(n, left))


def peaks(p, thr, sep):
    """Local maxima above thr, at least sep frames apart (greedy by height)."""
    idx = np.flatnonzero(p >= thr)
    idx = idx[np.argsort(-p[idx])]
    taken = np.zeros(len(p), bool)
    out = []
    for i in idx:
        if not taken[max(0, i - sep):i + sep + 1].any():
            out.append(i)
            taken[i] = True
    return np.sort(np.array(out, dtype=int))


def detector():
    _, _, window = lgb_params()
    rows = []
    for _, val in config.folds():
        frames, sw, tgt = match_targets(val, window)
        _, labels = load_labels(val)
        per = periods_of(val)
        live = np.isin(labels, ["H", "A"]) & (per > 0)
        # switches that start or end a short (< 2 s) possession
        bounds = np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) | (per[1:] != per[:-1])) + 1, len(labels)]
        short_sw = set()
        for a, b in zip(bounds[:-1], bounds[1:]):
            if labels[a] in ("H", "A") and b - a < 50:
                short_sw.update([a, b])
        short_sw = np.array(sorted(set(sw) & short_sw))
        for m, s in config.units(val):
            p = pd.read_parquet(os.path.join(OUT, "oof_{}.parquet".format(config.unit_id(m, s)))).p_c.to_numpy()
            r = {"fold": val, "seed": s, "AUC_live": roc_auc_score(tgt[live], p[live])}
            for thr in (0.3, 0.5, 0.7):
                pk = peaks(np.where(live, p, 0), thr, window)
                hit = near_switch(pk, sw, window) if len(pk) else np.zeros(0)
                found = near_switch(sw, pk, window) if len(pk) else np.zeros(len(sw))
                found_s = near_switch(short_sw, pk, window) if len(pk) else np.zeros(len(short_sw))
                r["precision@{}".format(thr)] = hit.mean() if len(pk) else np.nan
                r["recall@{}".format(thr)] = found.mean()
                r["recall_short@{}".format(thr)] = found_s.mean() if len(short_sw) else np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("Change detector on its own (true live frames; a detection = a peak within +-{} frames of a true switch)".format(window))
    print(t.groupby("fold").mean(numeric_only=True).drop(columns="seed").round(3).to_string())
    print("MEAN", t.mean(numeric_only=True).drop("seed").round(3).to_dict())
    print("needed for an added possession to pay off: precision above about 0.28 (= current spellF1 / 2)")


def grid(budget):
    cc, seg = models_config()["change_cost"], models_config()["segmenter"]
    refs = json.load(open(REF))
    pairs = [(w, c) for w in cc["grid_weight"] for c in cc["grid_cost_h"]]
    done = set()
    if os.path.isfile(GRID):
        g = pd.read_csv(GRID)
        done = set(zip(g.weight, g.cost_h))
    todo = [p for p in pairs if p not in done]
    data = {}
    for _, val in config.folds():
        d = pd.read_parquet(os.path.join(PHASE4, "d", "oof_{}.parquet".format(val)))
        dead = dead_mask(d.p_d.to_numpy(), d.period.to_numpy(), seg["cost_d"], seg["prob_clip"])
        units = []
        for m, s in config.units(val):
            uid = config.unit_id(m, s)
            p_h = pd.read_parquet(os.path.join(PHASE4, "ha", "oof_{}.parquet".format(uid)), columns=["p_h"]).p_h.to_numpy()
            p_c = pd.read_parquet(os.path.join(OUT, "oof_{}.parquet".format(uid))).p_c.to_numpy()
            units.append((s, p_h, p_c))
        data[val] = (d, dead, units)
    t0, n = time.time(), 0
    for w, c in todo:
        if n and time.time() - t0 > budget:
            break
        t = time.time()
        rows = []
        for val, (d, dead, units) in data.items():
            per, gt = d.period.to_numpy(), d.label.to_numpy()
            for s, p_h, p_c in units:
                cost = c if w == 0 else change_cost(p_c, c, w, refs[val]["ref"], seg["prob_clip"], cc["min_cost"])
                lab = segment(d.p_d.to_numpy(), p_h, per, seg["cost_d"], cost, seg["prob_clip"], dead=dead)
                r = score(gt, lab, per)
                rows.append({"weight": w, "cost_h": c, "fold": val, "seed": s, "spellF1": r["spell_f1_mean"],
                             "macroF1": r["macro_f1"], "pred_spells": r["n_pred_spells"], "true_spells": r["n_gt_spells"]})
        pd.DataFrame(rows).to_csv(GRID, mode="a", header=not os.path.isfile(GRID), index=False)
        n += 1
        print("weight {:>2} cost_h {:>3} | spellF1 {:.4f} | {:.1f}s".format(w, c, pd.DataFrame(rows).spellF1.mean(), time.time() - t))
    left = len(todo) - n
    print("all done ({} this call)".format(n) if left == 0 else "done this call: {} | left: {}".format(n, left))


def report():
    g = pd.read_csv(GRID)
    s = g.groupby(["weight", "cost_h"]).agg(spellF1=("spellF1", "mean"), macroF1=("macroF1", "mean"),
                                           pred=("pred_spells", "mean"), true=("true_spells", "mean"))
    pd.set_option("display.width", 220)
    print("A. mean spellF1, 30 dev units (rows: weight, columns: base cost_h); weight 0 = current segmenter")
    print(s.spellF1.unstack().round(4).to_string())
    print("\n   mean predicted possessions (true {:.0f})".format(s.true.iloc[0]))
    print(s.pred.unstack().round(0).to_string())
    best = s.spellF1.idxmax()
    ctrl = g[(g.weight == 0) & (g.cost_h == 8)].groupby("fold").spellF1.mean()
    bf = g[(g.weight == best[0]) & (g.cost_h == best[1])].groupby("fold").spellF1.mean()
    gain = s.spellF1.loc[best] - s.spellF1.loc[(0, 8)]
    better = int((bf > ctrl).sum())
    print("\nB. best weight {} cost_h {}: spellF1 {:.4f}, macroF1 {:.4f}, possessions {:.0f} | control (0, 8) {:.4f}".format(
        best[0], best[1], s.spellF1.loc[best], s.macroF1.loc[best], s.pred.loc[best], s.spellF1.loc[(0, 8)]))
    print(pd.DataFrame({"control": ctrl, "best": bf, "change": bf - ctrl}).round(4).to_string())
    keep = gain > NOISE and better >= 5
    print("\nKEEP RULE: gain {:+.4f} (needs > {}) | folds improved {}/6 (needs >= 5) -> {}".format(
        gain, NOISE, better, "KEEP" if keep else "UNDO"))


def choose():
    g = pd.read_csv(GRID)
    s = g.groupby(["weight", "cost_h"]).spellF1.mean()
    best = s.idxmax()
    path = os.path.join(config.CONFIG, "models.json")
    cfg = json.load(open(path, encoding="utf-8"))
    cfg["change_cost"]["weight"], cfg["change_cost"]["cost_h"] = int(best[0]), int(best[1])
    cfg["change_cost"]["_chosen"] = "Kept 12 Nov: mean dev spellF1 {:.4f} vs control {:.4f} (runs/phase6/change/grid.csv).".format(
        s.max(), s.loc[(0, 8)])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
        fh.write("\n")
    print("written: weight={} cost_h={}".format(*best))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in ("train", "detector", "grid", "report", "choose"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--budget", type=float, default=60.0)
    a = ap.parse_args()
    if a.train:
        train(a.budget)
    elif a.detector:
        detector()
    elif a.grid:
        grid(a.budget)
    elif a.report:
        report()
    elif a.choose:
        choose()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
