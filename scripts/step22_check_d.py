"""Step 2.2 checkpoint: D features on real (external) data.

  1. leakage test (Rulebook 3.2 acceptance 1): building D features while two different seeds'
     ball data are loaded gives byte-identical output; the D loader reads no ball column
  2. attacking direction = the standardized home_attacks_right on every external match
  3. value ranges per group on one development match; NaN shares; no column mentions the ball
  4. runtime and peak memory

    python scripts/step22_check_d.py
"""

import hashlib
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.data import config_features, load_ball, load_players  # noqa: E402
from bh.features_common import attack_direction  # noqa: E402
from bh.features_d import build_d_features  # noqa: E402

MATCH = "idsse_J03WMX"    # a development match


def fingerprint(df):
    return hashlib.sha256(np.ascontiguousarray(df.to_numpy(np.float32)).tobytes()
                          + "|".join(df.columns).encode()).hexdigest()[:16]


def main():
    cfg = config_features()
    ok = True

    # 1. leakage test
    t0 = time.time()
    _ = load_ball(MATCH, 1)
    d1, meta = build_d_features(load_players(MATCH), cfg)
    t_build = time.time() - t0
    _ = load_ball(MATCH, 3)
    d3, _ = build_d_features(load_players(MATCH), cfg)
    same = fingerprint(d1) == fingerprint(d3)
    ball_cols = [c for c in d1.columns if "ball" in c]
    print("1. leakage: D features with seed-1 vs seed-3 ball loaded identical:", same,
          "| columns mentioning ball:", len(ball_cols))
    ok &= same and not ball_cols

    # 2. direction vs the standardized column, all matches
    agree = total = 0
    for m in config.all_matches():
        p = load_players(m)
        d = attack_direction(p.xy, p.team, p.period)
        given = pd.read_parquet(config.path("raw_frames", match=m), columns=["frame", "home_attacks_right"]) \
            .sort_values("frame")["home_attacks_right"].to_numpy()
        inplay = np.isfinite(d)
        agree += int(((d[inplay] > 0) == given[inplay]).sum())
        total += int(inplay.sum())
    print("2. attacking direction matches home_attacks_right on {:.4%} of {:,} frames ({} matches)".format(
        agree / total, total, len(config.all_matches())))
    ok &= agree == total

    # 3. ranges and NaN shares on the base (per-frame) columns
    base = [m for m in meta if m["window"] == "frame"]
    rows = []
    for m in base:
        v = d1[m["name"]].to_numpy()
        f = v[np.isfinite(v)]
        rows.append({"feature": m["name"], "unit": m["unit"], "nan_share": round(1 - len(f) / len(v), 3),
                     "p1": round(float(np.percentile(f, 1)), 2) if len(f) else None,
                     "median": round(float(np.median(f)), 2) if len(f) else None,
                     "p99": round(float(np.percentile(f, 99)), 2) if len(f) else None})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("3. per-frame D features on {} ({} base, {} columns with windows):".format(MATCH, len(base), d1.shape[1]))
    print(table.to_string(index=False))
    out = os.path.join(config.path("runs"), "step22_d_feature_ranges_{}.csv".format(MATCH))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.to_csv(out, index=False)

    # 4. runtime / memory
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("4. build time {:.1f} s for {:,} frames | peak memory {:.0f} MB".format(t_build, len(d1), peak_mb))
    print("STEP 2.2 CHECK:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
