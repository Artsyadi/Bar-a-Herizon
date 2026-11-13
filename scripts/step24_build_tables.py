"""Step 2.4: training feature tables + the generated feature list.

  D   : one table per match (players-only features are identical for every seed), every 5th frame
  H/A : one table per unit (match x seed), every 5th frame, live frames only (label H or A)
Full-frame features are NOT stored: they rebuild in ~20 s (D) / ~3 s (H/A) when a held-out match is
scored (bh.pipeline), which saves ~2-3 GB on the D: drive.

All 10 matches are written (dev, test, Metrica); which rows train what is decided by config/split.json
in Phase 4, never here.

Resumable; --budget keeps each call under the 180 s limit.

    PY=python
    $PY scripts/step24_build_tables.py --budget 140     # repeat until "all done"
    $PY scripts/step24_build_tables.py --check          # checkpoint report

Outputs (gitignored):  data/features/train_d/<match>.parquet
                       data/features/train_ha/<unit>.parquet
Committed doc:         docs/features_list.csv   (every column: model, group, rulebook,
                                                              window, unit, description)
"""

import argparse
import glob
import json
import os
import resource
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.data import config_features  # noqa: E402
from bh.pipeline import attach_labels, d_features, ha_features, stride_mask  # noqa: E402
from bh.scoring import load_labels  # noqa: E402

OUT = os.path.join(config.PROJECT, "data", "features")
LIST = os.path.join(config.PROJECT, "docs", "features_list.csv")


def d_path(match):
    return os.path.join(OUT, "train_d", match + ".parquet")


def ha_path(uid):
    return os.path.join(OUT, "train_ha", uid + ".parquet")


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=True, compression="zstd")


def build(budget, force):
    cfg = config_features()
    stride = cfg["train_stride_frames"]
    jobs = [("D", m, None) for m in config.all_matches()]
    jobs += [("HA", m, s) for m in config.all_matches() for _, s in config.units(m)]
    t0 = time.time()
    metas, done = {}, 0
    left = []
    for kind, m, s in jobs:
        path = d_path(m) if kind == "D" else ha_path(config.unit_id(m, s))
        if os.path.isfile(path) and not force and (kind in metas or os.path.isfile(LIST)):
            continue
        if time.time() - t0 > budget:
            left.append((kind, m, s))
            continue
        t = time.time()
        df, meta = d_features(m, cfg) if kind == "D" else ha_features(m, s, cfg)
        metas.setdefault(kind, meta)
        df = attach_labels(df, m)
        if kind == "D":
            rows = stride_mask(df.index, stride)
            df["target_d"] = (df["label"] == "D").astype(np.int8)
        else:
            rows = stride_mask(df.index, stride) & df["label"].isin(["H", "A"]).to_numpy()
            df["target_h"] = (df["label"] == "H").astype(np.int8)
        df = df.loc[rows]
        df.insert(0, "match", m)
        df.insert(1, "seed", -1 if s is None else s)
        save(df, path)
        done += 1
        print("{:<3} {:<22} {:>5.1f}s  rows {:>6,}  cols {}".format(
            kind, config.unit_id(m, s), time.time() - t, len(df), df.shape[1]))
    if metas:
        write_list(metas)
    print("done this call: {} | left: {}".format(done, len(left)) if left else "all done ({} this call)".format(done))


def write_list(metas):
    rows = []
    if os.path.isfile(LIST):
        old = pd.read_csv(LIST)
        rows = old[~old["model"].isin(list(metas))].to_dict("records")
    for kind, meta in metas.items():
        rows += meta
    df = pd.DataFrame(rows)[["name", "model", "group", "rulebook", "base", "window", "unit", "description"]]
    os.makedirs(os.path.dirname(LIST), exist_ok=True)
    df.sort_values(["model", "group", "name"]).to_csv(LIST, index=False)


def check():
    cfg = config_features()
    stride = cfg["train_stride_frames"]
    ok = True
    feats = pd.read_csv(LIST)
    d_names = feats.loc[feats.model == "D", "name"].tolist()
    ha_names = feats.loc[feats.model == "HA", "name"].tolist()
    print("feature list: {} D columns, {} H/A columns | D columns mentioning ball: {}".format(
        len(d_names), len(ha_names), sum("ball" in n for n in d_names)))
    ok &= not any("ball" in n for n in d_names)

    rows = []
    rng = np.random.default_rng(0)
    for p in sorted(glob.glob(os.path.join(OUT, "train_*", "*.parquet"))):
        kind = "D" if "train_d" in p else "HA"
        df = pd.read_parquet(p)
        m = df["match"].iloc[0]
        frames, labels = load_labels(m)
        lab = pd.Series(labels, index=frames)
        expected = int(stride_mask(frames, stride).sum()) if kind == "D" else int(
            (stride_mask(frames, stride) & np.isin(labels, ["H", "A"])).sum())
        sample = rng.choice(df.index.to_numpy(), size=min(2000, len(df)), replace=False)
        aligned = bool((df.loc[sample, "label"].to_numpy() == lab.loc[sample].to_numpy()).all())
        names = d_names if kind == "D" else ha_names
        cols_ok = list(df.columns[2:2 + len(names)]) == names or set(names) <= set(df.columns)
        rows.append({"kind": kind, "file": os.path.basename(p)[:-8], "rows": len(df), "expected": expected,
                     "rows_ok": len(df) == expected, "labels_aligned": aligned, "columns_ok": cols_ok,
                     "MB": round(os.path.getsize(p) / 1e6, 1)})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(t.to_string(index=False))
    ok &= bool(t.rows_ok.all() and t.labels_aligned.all() and t.columns_ok.all())
    ok &= len(t[t.kind == "D"]) == 10 and len(t[t.kind == "HA"]) == 38
    print("total size {:.0f} MB | peak memory of this check {:.0f} MB".format(
        t.MB.sum(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    print("STEP 2.4 CHECK:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=float, default=140.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        build(args.budget, args.force)


if __name__ == "__main__":
    main()
