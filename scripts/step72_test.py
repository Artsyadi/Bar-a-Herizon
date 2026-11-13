"""Step 7.2: the one test. Frozen v1 (models/v1 + the chosen segmenter) on the test match
J03WPY (5 seeds) and, for comparison, the 3 Metrica matches, next to the organisers' baseline on the
same units.

Guards:
  * every fingerprint in models/v1/manifest.json is checked BEFORE and AFTER scoring
  * each group is scored once: if its results exist, they are shown and nothing is re-scored

    PY=python
    $PY scripts/step72_test.py --group test
    $PY scripts/step72_test.py --group metrica
    $PY scripts/step72_test.py --show

Output: runs/phase7/results.csv, labels per unit in runs/phase7/labels/
"""

import argparse
import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.baseline import input_ball, nearest_codes, raw_players, smooth  # noqa: E402
from bh.models import predict  # noqa: E402
from bh.pipeline import d_features, ha_features  # noqa: E402
from bh.scoring import load_labels, score  # noqa: E402
from bh.segmenter import segment  # noqa: E402

V1 = os.path.join(config.PROJECT, "models", "v1")
OUT = os.path.join(config.PROJECT, "runs", "phase7")
RESULTS = os.path.join(OUT, "results.csv")
DEV_OOF = {"spellF1": 0.5525, "macroF1": 0.8747}           # Phase 5, 30 dev units, out of fold
DEV_BASELINE = {"spellF1": 0.2164, "macroF1": 0.4568}      # step 3.2


def verify_manifest(when):
    man = json.load(open(os.path.join(V1, "manifest.json"), encoding="utf-8"))
    bad = []
    for rel, h in man["md5"].items():
        with open(os.path.join(config.ROOT, rel), "rb") as f:
            if hashlib.md5(f.read()).hexdigest() != h:
                bad.append(rel)
    print("manifest check ({}): {} files, {}".format(when, len(man["md5"]), "all unchanged" if not bad else "CHANGED: " + ", ".join(bad)))
    if bad:
        sys.exit(1)
    return man


def run(group):
    if os.path.isfile(RESULTS) and group in set(pd.read_csv(RESULTS).group):
        print("group '{}' was already scored; showing the stored results (nothing re-scored)".format(group))
        show()
        return
    man = verify_manifest("before")
    sp = config.split()
    matches = sp["test_matches"] if group == "test" else sp["comparison_matches"]
    seg = man["segmenter"]
    d_model = lgb.Booster(model_file=os.path.join(V1, "d_model.txt"))
    ha_model = lgb.Booster(model_file=os.path.join(V1, "ha_model.txt"))
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)
    rows = []
    for m in matches:
        t = time.time()
        fd, _ = d_features(m)
        frames, gt = load_labels(m)
        per = pd.read_parquet(config.path("raw_frames", match=m), columns=["frame", "period"]).set_index("frame") \
            .loc[frames, "period"].fillna(0).to_numpy()
        assert np.array_equal(fd.index.to_numpy(), frames)
        p_d = predict(d_model, fd)
        pf, xy, team = raw_players(m)
        for mm, s in config.units(m):
            uid = config.unit_id(mm, s)
            fh, _ = ha_features(mm, s)
            lab = segment(p_d, predict(ha_model, fh), per, seg["cost_d"], seg["cost_h"], seg["prob_clip"])
            pd.DataFrame({"frame": frames, "label": lab.astype(str)}).to_parquet(
                os.path.join(OUT, "labels", uid + ".parquet"), index=False)
            _, bper, ball = input_ball(mm, s)
            base = smooth(nearest_codes(xy, team, ball), 50, bper)
            for system, pred in (("v1", lab), ("baseline_smooth50", base)):
                r = score(gt, pred, per)
                rows.append({"group": group, "unit": uid, "match": m, "seed": -1 if s is None else s, "system": system,
                             "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "alive_dead_f1": r["alive_dead_f1"],
                             "team_acc": r["team_acc_gt_alive"], "pred_spells": r["n_pred_spells"],
                             "true_spells": r["n_gt_spells"]})
        print("{:<16} scored in {:.0f}s".format(m, time.time() - t))
    pd.DataFrame(rows).to_csv(RESULTS, mode="a", header=not os.path.isfile(RESULTS), index=False)
    verify_manifest("after")
    show()


def show():
    r = pd.read_csv(RESULTS)
    pd.set_option("display.width", 220)
    for group, g in r.groupby("group", sort=False):
        print("\n=== {} ===".format(group))
        t = g.pivot_table(index="unit", columns="system", values=["spellF1", "macroF1"]).round(4)
        print(t.to_string())
        s = g.groupby("system").agg(spellF1=("spellF1", "mean"), spellF1_worst=("spellF1", "min"),
                                    macroF1=("macroF1", "mean"), alive_dead_f1=("alive_dead_f1", "mean"),
                                    team_acc=("team_acc", "mean"), pred_spells=("pred_spells", "mean"),
                                    true_spells=("true_spells", "mean"))
        print(s.round(4).to_string())
    print("\nreference, dev (30 units, out of fold): v1 spellF1 {spellF1} macroF1 {macroF1}".format(**DEV_OOF),
          "| baseline spellF1 {spellF1} macroF1 {macroF1}".format(**DEV_BASELINE))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", choices=["test", "metrica"])
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.show:
        show()
    elif a.group:
        run(a.group)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
