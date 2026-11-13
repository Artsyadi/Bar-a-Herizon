"""Step 2.3 checkpoint: H/A features on real (external) data.

  1. value ranges and NaN shares on one development unit; ball-based features are NaN exactly where
     the ball is unavailable; no ball_z
  2. sign sanity (wiring check, not tuning): on live frames with an observed ball, when the nearest
     home player is clearly closer (margin > 2 m) the label should mostly be H, and vice versa
  3. runtime and peak memory

    python scripts/step23_check_ha.py
"""

import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.data import config_features, load_ball, load_players  # noqa: E402
from bh.features_ha import build_ha_features  # noqa: E402
from bh.scoring import load_labels  # noqa: E402

MATCH, SEED = "idsse_J03WMX", 1      # a development unit


def main():
    cfg = config_features()
    ok = True
    t0 = time.time()
    p, b = load_players(MATCH), load_ball(MATCH, SEED)
    df, meta = build_ha_features(p, b, cfg)
    t_build = time.time() - t0

    unavailable = b.status == 2
    m = df["ha_near_margin"].to_numpy()
    nan_ok = bool(np.all(np.isnan(m[unavailable])))
    print("1. ball-based margin NaN on every unavailable-ball frame:", nan_ok,
          "| unavailable share {:.3f} | margin NaN share {:.3f}".format(unavailable.mean(), np.isnan(m).mean()),
          "| ball_z columns:", sum("ball_z" in c for c in df.columns))
    ok &= nan_ok and not any("ball_z" in c for c in df.columns)

    base = [x for x in meta if x["window"] == "frame"]
    rows = []
    for x in base:
        v = df[x["name"]].to_numpy()
        f = v[np.isfinite(v)]
        rows.append({"feature": x["name"], "unit": x["unit"], "nan_share": round(1 - len(f) / len(v), 3),
                     "p1": round(float(np.percentile(f, 1)), 2) if len(f) else None,
                     "median": round(float(np.median(f)), 2) if len(f) else None,
                     "p99": round(float(np.percentile(f, 99)), 2) if len(f) else None})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("   per-frame H/A features on {} seed {} ({} base, {} columns):".format(MATCH, SEED, len(base), df.shape[1]))
    print(table.to_string(index=False))
    out = os.path.join(config.path("runs"), "step23_ha_feature_ranges_{}_s{}.csv".format(MATCH, SEED))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.to_csv(out, index=False)

    frames, labels = load_labels(MATCH)
    assert np.array_equal(frames, df.index.to_numpy()), "label frames differ from feature frames"
    obs = b.status == 0
    live = np.isin(labels, ["H", "A"])
    home_clear = live & obs & (m > 2)
    away_clear = live & obs & (m < -2)
    ph = float((labels[home_clear] == "H").mean())
    pa = float((labels[away_clear] == "A").mean())
    print("2. sign sanity: label H when home clearly nearer {:.3f} (n={:,}) | label A when away clearly nearer {:.3f} (n={:,})".format(
        ph, int(home_clear.sum()), pa, int(away_clear.sum())))
    ok &= ph > 0.6 and pa > 0.6

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("3. build time {:.1f} s (incl. loading) | peak memory {:.0f} MB".format(t_build, peak_mb))
    print("STEP 2.3 CHECK:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
