"""Step 1.2: clean every external unit with atd-cleaner v0.2.1.

Units: each IDSSE match x 5 seeds (degraded ball + the shared raw player table) and each Metrica
match (raw, not degraded). The ball depends on the seed; the players don't - so the cleaned player
table is written once per match (from seed 1) and every other seed is checked to produce an
identical player result (fingerprint).

Resumable: units already done are skipped (use --force to redo). Use --budget to stop before the
180-second command limit; just run it again to continue.

    PY=python
    $PY scripts/step12_clean_all.py --budget 150        # repeat until "all done"
    $PY scripts/step12_clean_all.py --summary            # build summary.csv

Outputs (data/cleaned/, gitignored):
    frames/<unit>.parquet     cleaned ball + per-frame quality, one per unit
    players/<match>.parquet   cleaned players (incl. speed_derived_ms), one per match
    reports/<unit>.json       cleaner report + timing + player fingerprint
    summary.csv               one row per unit
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from bh import config  # noqa: E402

sys.path.insert(0, config.path("cleaner_root"))
import atd_cleaner  # noqa: E402
from atd_cleaner.io_table import frames_table, objects_table, read_table  # noqa: E402
from atd_cleaner.pipeline import load_thresholds, run  # noqa: E402


def player_fingerprint(res):
    h = hashlib.sha256()
    for key in ("players", "player_status", "player_flags", "player_speed_derived"):
        h.update(np.ascontiguousarray(res[key]).tobytes())
    return h.hexdigest()[:16]


def all_units():
    return [(m, s) for m in config.all_matches() for _, s in config.units(m)]


def clean_unit(match, seed, th):
    t0 = time.time()
    frames_in = (config.path("degraded_frames", match=match, seed=seed) if seed is not None
                 else config.path("raw_frames", match=match))
    tr = read_table(frames_in, config.path("raw_objects", match=match))
    raw_ball_observed = float(np.isfinite(tr.ball[:, 0]).mean())
    res, quality, rep = run(tr, th)
    uid = config.unit_id(match, seed)

    f_path = config.path("cleaned_frames", unit=uid)
    os.makedirs(os.path.dirname(f_path), exist_ok=True)
    frames_table(tr, res, quality, match_id=match).to_parquet(f_path, index=False)

    fp = player_fingerprint(res)
    p_path = config.path("cleaned_players", match=match)
    wrote_players = False
    if seed in (None, 1):
        os.makedirs(os.path.dirname(p_path), exist_ok=True)
        objects_table(tr, res, match_id=match).to_parquet(p_path, index=False)
        wrote_players = True

    st = res["ball_status"]
    rep.update({
        "unit": uid, "match": match, "seed": seed, "seconds": round(time.time() - t0, 1),
        "inputs": {"frames": os.path.relpath(frames_in, config.ROOT),
                   "objects": os.path.relpath(config.path("raw_objects", match=match), config.ROOT)},
        "cleaner_version": atd_cleaner.__version__,
        "ball_observed_share_raw": raw_ball_observed,
        "ball_observed_share_clean": float((st == 0).mean()),
        "ball_available_share_clean": float((st != 2).mean()),
        "player_fingerprint": fp, "wrote_players": wrote_players,
    })
    r_path = config.path("cleaned_reports", unit=uid)
    os.makedirs(os.path.dirname(r_path), exist_ok=True)
    with open(r_path, "w", encoding="utf-8") as handle:
        json.dump(rep, handle, indent=2, default=str)
    return rep


def build_summary():
    rows = []
    for p in sorted(glob.glob(config.path("cleaned_reports", unit="*"))):
        r = json.load(open(p, encoding="utf-8"))
        row = {"unit": r["unit"], "match": r["match"], "seed": r["seed"], "seconds": r["seconds"],
               "invariants_ok": not r["invariant_problems"], "cleaner_version": r["cleaner_version"],
               "speed_units": r["speed_units_check"]["verdict"],
               "ball_observed_raw": round(r["ball_observed_share_raw"], 4),
               "ball_observed_clean": round(r["ball_observed_share_clean"], 4),
               "ball_available_clean": round(r["ball_available_share_clean"], 4),
               "player_fingerprint": r["player_fingerprint"]}
        row.update(r["summary"])
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        ref = df[df.seed.isin([1]) | df.seed.isna()].set_index("match")["player_fingerprint"]
        df["players_identical_to_seed1"] = [fp == ref.get(m) for m, fp in zip(df.match, df.player_fingerprint)]
    out = config.path("cleaned_summary")
    df.to_csv(out, index=False)
    return df, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=float, default=150.0, help="stop starting new units after this many seconds")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", action="store_true", help="only build summary.csv from the reports")
    args = ap.parse_args()

    if args.summary:
        df, out = build_summary()
        print("units:", len(df), "| invariants OK:", int(df.invariants_ok.sum()),
              "| players identical to seed 1:", int(df.players_identical_to_seed1.sum()))
        print("saved:", os.path.relpath(out, config.ROOT))
        return

    th = load_thresholds(config.path("cleaner_thresholds"))
    t0 = time.time()
    todo = [(m, s) for m, s in all_units()
            if args.force or not os.path.isfile(config.path("cleaned_reports", unit=config.unit_id(m, s)))]
    # seed 1 / Metrica first, so each match's player table exists before its other seeds
    todo.sort(key=lambda u: (u[1] not in (None, 1), u[0], u[1] or 0))
    done = 0
    for match, seed in todo:
        if time.time() - t0 > args.budget:
            break
        rep = clean_unit(match, seed, th)
        done += 1
        print("{:<22} {:>5.1f}s  invariants {}  ball obs {:.3f} -> avail {:.3f}  players {}".format(
            rep["unit"], rep["seconds"], "OK" if not rep["invariant_problems"] else rep["invariant_problems"],
            rep["ball_observed_share_raw"], rep["ball_available_share_clean"], rep["player_fingerprint"]))
    left = len(todo) - done
    print("done this call: {} | left: {}".format(done, left) if left else "all done ({} units this call)".format(done))


if __name__ == "__main__":
    main()
