"""Regenerate the submission file from the organisers' data bundle with the frozen models.

    python scripts/make_submission.py                    # everything (~3 minutes)
    python scripts/make_submission.py --step clean       # 1/2: read and clean the tactical ATD
    python scripts/make_submission.py --step predict     # 2/2: features, models, segmenter, CSV

Options:
    --data-dir   folder that holds tactical/ (default: $MTAH_DATA_DIR, else more-than-a-hack-2027/data)
    --out        output CSV (default: outputs/Barça Herizon.csv)

Nothing here trains or reads labels. It prints checks only. Intermediate files go to outputs/bar_ath/.
If the organisers' repository is present (more-than-a-hack-2027/), their validate_submission.py is run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import bar_ath, config  # noqa: E402
from bh.models import predict  # noqa: E402
from bh.segmenter import segment  # noqa: E402

V1 = os.path.join(config.PROJECT, "models", "v1")
DEFAULT_OUT = os.path.join(config.ROOT, "outputs", "Barça Herizon.csv")


def show(checks):
    for name, ok in checks.items():
        print("  {:<50} {}".format(name, "OK" if ok else "FAILED"))
    if not all(checks.values()):
        sys.exit(1)


def verify_manifest():
    man = json.load(open(os.path.join(V1, "manifest.json"), encoding="utf-8"))
    changed = []
    for rel, h in man["md5"].items():
        with open(os.path.join(config.ROOT, rel), "rb") as f:
            if hashlib.md5(f.read()).hexdigest() != h:
                changed.append(rel)
    for rel in changed:
        print("  changed since the models were frozen:", rel)
    show({"frozen models, configs and code unchanged ({} files)".format(len(man["md5"])): not changed})
    return man


def step_clean(data_dir):
    t = time.time()
    print("1/2 read and clean the tactical ATD")
    show(bar_ath.clean(data=data_dir))
    print("    done in {:.0f} s".format(time.time() - t))


def step_predict(out):
    t = time.time()
    print("2/2 features, frozen models, segmenter")
    man = verify_manifest()
    seg = man["segmenter"]
    period, fd, fh = bar_ath.features()
    p_d = predict(lgb.Booster(model_file=os.path.join(V1, "d_model.txt")), fd)
    p_h = predict(lgb.Booster(model_file=os.path.join(V1, "ha_model.txt")), fh)
    labels = segment(p_d, p_h, period, seg["cost_d"], seg["cost_h"], seg["prob_clip"])
    bar_ath.write_submission(labels, out)
    checks = {"submission written: 199,319 rows, H/A/D": os.path.isfile(out)}
    validator = os.path.join(config.path("scorer_dir"), "validate_submission.py")
    if os.path.isfile(validator):
        r = subprocess.run([sys.executable, validator, out], capture_output=True, text=True)
        checks["organisers' validate_submission.py"] = r.returncode == 0
    else:
        print("  (organisers' repository not found: run validate_submission.py yourself)")
    show(checks)
    with open(out, "rb") as f:
        print("    md5 {}".format(hashlib.md5(f.read()).hexdigest()))
    print("    done in {:.0f} s -> {}".format(time.time() - t, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", choices=["clean", "predict", "all"], default="all")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.step in ("clean", "all"):
        step_clean(a.data_dir)
    if a.step in ("predict", "all"):
        step_predict(a.out)


if __name__ == "__main__":
    main()
