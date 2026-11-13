"""Step 7.1: freeze training v1.

Retrains the D and H/A models on ALL 6 dev matches with the chosen settings (config/models.json),
saves them in models/v1/, and writes a manifest of fingerprints (md5) of the models,
configs, code, cleaner thresholds and library versions. The test match and Metrica are never used.

    PY=python
    $PY scripts/step71_freeze.py --train d        # ~1 min
    $PY scripts/step71_freeze.py --train ha       # ~1 min
    $PY scripts/step71_freeze.py --repeat d       # retrain into models/v1/repeat/
    $PY scripts/step71_freeze.py --repeat ha
    $PY scripts/step71_freeze.py --check          # byte-identical? + write manifest

Checkpoint: the repeat models are byte-identical and predict identically on a dev unit.
"""

import argparse
import glob
import hashlib
import importlib.metadata as md
import json
import os
import platform
import sys
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.models import feature_names, load_tables, models_config, predict, train  # noqa: E402
from bh.pipeline import d_features, ha_features  # noqa: E402

V1 = os.path.join(config.PROJECT, "models", "v1")
MANIFEST = os.path.join(V1, "manifest.json")
CHECK_UNIT = ("idsse_J03WMX", 1)          # a dev unit for the prediction check


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit(kind, folder):
    dev = config.split()["dev_matches"]
    t = time.time()
    if kind == "d":
        names = feature_names("D")
        X, y, _ = load_tables("D", dev, names)
        booster = train("d_model", X, y, names)
    else:
        names = feature_names("HA")
        X, y, _ = load_tables("HA", [config.unit_id(m, s) for d in dev for m, s in config.units(d)], names)
        booster = train("ha_model", X, y, names)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "{}_model.txt".format(kind))
    booster.save_model(path)
    print("{} model: {:,} rows, {} features, {:.1f}s -> {} (md5 {})".format(
        kind.upper(), len(y), len(names), time.time() - t, os.path.relpath(path, config.ROOT), md5(path)[:12]))


def check():
    ok = True
    for kind in ("d", "ha"):
        a, b = os.path.join(V1, "{}_model.txt".format(kind)), os.path.join(V1, "repeat", "{}_model.txt".format(kind))
        same = md5(a) == md5(b)
        print("{} model file byte-identical on retraining: {}".format(kind.upper(), same))
        ok &= same
    m, s = CHECK_UNIT
    fd, _ = d_features(m)
    fh, _ = ha_features(m, s)
    for kind, feats in (("d", fd), ("ha", fh)):
        pa = predict(lgb.Booster(model_file=os.path.join(V1, "{}_model.txt".format(kind))), feats)
        pb = predict(lgb.Booster(model_file=os.path.join(V1, "repeat", "{}_model.txt".format(kind))), feats)
        same = np.array_equal(pa, pb)
        print("{} predictions on {} seed {} identical: {} ({} frames)".format(kind.upper(), m, s, same, len(pa)))
        ok &= same

    files = {}
    for rel in (["models/v1/d_model.txt", "models/v1/ha_model.txt",
                 "config/models.json", "config/features.json",
                 "config/split.json", "config/paths.json", "config/epts_bar_ath.json",
                 "docs/features_list.csv", "requirements.txt",
                 os.path.relpath(config.path("cleaner_thresholds"), config.ROOT).replace(os.sep, "/")]
                + sorted(os.path.relpath(p, config.ROOT).replace(os.sep, "/") for p in
                         glob.glob(os.path.join(config.PROJECT, "bh", "*.py")))
                + sorted(os.path.relpath(p, config.ROOT).replace(os.sep, "/") for p in
                         glob.glob(os.path.join(config.ROOT, "atd_cleaner", "*.py")))):
        files[rel] = md5(os.path.join(config.ROOT, rel))
    seg = models_config()["segmenter"]
    manifest = {
        "version": "training v1",
        "frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trained_on": config.split()["dev_matches"],
        "never_used_for_training_or_choices": config.split()["test_matches"] + config.split()["comparison_matches"],
        "segmenter": {"cost_d": seg["cost_d"], "cost_h": seg["cost_h"], "prob_clip": seg["prob_clip"]},
        "phase6_fixes_used": False,
        "python": platform.python_version(),
        "libraries": {p: md.version(p) for p in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm")},
        "md5": files,
        "determinism_check_passed": bool(ok),
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("manifest written: {} files fingerprinted".format(len(files)))
    print("STEP 7.1 CHECK:", "PASSED" if ok else "FAILED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", choices=["d", "ha"])
    ap.add_argument("--repeat", choices=["d", "ha"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.train:
        fit(a.train, V1)
    elif a.repeat:
        fit(a.repeat, os.path.join(V1, "repeat"))
    elif a.check:
        check()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
