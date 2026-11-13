"""Step 3.1 checkpoint: the scoring wrapper behaves like the official scorer should.

On one development match (the ground truth scored against altered copies of itself):
  * itself                        -> spellF1 1.0
  * delayed 1 frame               -> >= 0.98. EVALUATION.md says "a one-frame offset is free" on
                                     BAR-ATH; this IDSSE match has 12 possessions of <= 4 frames,
                                     and a 1-frame shift drops their overlap below the 0.7 IoU
                                     level. tests/test_scoring.py proves it is exactly 1.0 when
                                     every possession is long.
  * delayed 3, 8 frames, 1 s      -> falling (EVALUATION.md's figures: 0.98, 0.89, 0.65).
                                     The 1 s value is checked against arithmetic: a possession of
                                     length L shifted by s has IoU (L-s)/(L+s), so it still matches
                                     at threshold t only if L >= s(1+t)/(1-t). Averaging that share
                                     over t = 0.3..0.7 predicts spellF1 from the possession lengths
                                     alone; measured must be within 0.02. This match has short
                                     possessions (median 3 s), hence ~0.49 rather than 0.65.
  * a constant guess (all 'H')    -> 0.0
  * separator check: no possession crosses half-time

    python scripts/step31_check_scorer.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.scoring import load_labels, score, shift_labels, with_separators, pm  # noqa: E402

MATCH = "idsse_J03WMX"     # a development match


def main():
    frames, gt = load_labels(MATCH)
    periods = pd.read_parquet(config.path("raw_frames", match=MATCH), columns=["frame", "period"]) \
        .sort_values("frame")["period"].to_numpy()
    assert len(periods) == len(gt), "labels and frames differ in length"

    rows = []
    for name, pred in [("itself", gt),
                       ("1 frame late", shift_labels(gt, periods, 1)),
                       ("3 frames late", shift_labels(gt, periods, 3)),
                       ("8 frames late", shift_labels(gt, periods, 8)),
                       ("1 s late", shift_labels(gt, periods, 25)),
                       ("constant H", np.full(len(gt), "H", dtype=object))]:
        r = score(gt, pred, periods)
        rows.append({"case": name, "spellF1": round(r["spell_f1_mean"], 4),
                     "macroF1": round(r["macro_f1"], 4), "pred spells": r["n_pred_spells"],
                     "true spells": r["n_gt_spells"]})
    print(pd.DataFrame(rows).to_string(index=False))

    g, _ = with_separators(gt, gt, periods)
    runs = pm.valid_runs(pm.label_codes(g))
    print("runs seen by the scorer:", runs, "| period change at frame",
          int(np.flatnonzero(np.diff(periods) != 0)[0] + 1))

    key = np.char.add(gt.astype(str), periods.astype(str))
    cut = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1, len(gt)]
    lengths = np.array([(cut[i + 1] - cut[i]) / 25.0 for i in range(len(cut) - 1)
                        if gt[cut[i]] in ("H", "A") and periods[cut[i]] > 0])
    predicted = float(np.mean([(lengths >= (1 + t) / (1 - t)).mean() for t in (0.3, 0.4, 0.5, 0.6, 0.7)]))
    print("1 s late: measured {:.4f} | predicted from possession lengths {:.4f} | median possession {:.1f} s".format(
        rows[4]["spellF1"], predicted, float(np.median(lengths))))

    ok = (rows[0]["spellF1"] == 1.0 and rows[1]["spellF1"] >= 0.98 and rows[5]["spellF1"] == 0.0
          and rows[2]["spellF1"] >= rows[3]["spellF1"] >= rows[4]["spellF1"] and len(runs) == 2
          and abs(rows[4]["spellF1"] - predicted) <= 0.02)
    print("STEP 3.1 CHECK:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
