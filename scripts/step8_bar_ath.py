"""Phase 8: the BAR-ATH submission with the frozen training v1. QUIET by design: prints only
pass/fail checks and timings, never values, counts or scores of the provided game. Numbers that a
person may want to read go to local files in the work folder (outside the repository).

    PY=python
    $PY scripts/step8_bar_ath.py --clean      # 8.1 read + clean
    $PY scripts/step8_bar_ath.py --coverage   # 8.2 input coverage -> work/coverage.txt
    $PY scripts/step8_bar_ath.py --predict    # 8.3 frozen v1 -> submission CSV + validator
    $PY scripts/step8_bar_ath.py --public     # 8.4 public 5-minute check -> PASS/FAIL

Submission file: <root>/outputs/submission/Barça Herizon.csv
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import bar_ath, config  # noqa: E402
from bh.models import predict  # noqa: E402
from bh.segmenter import segment  # noqa: E402

V1 = os.path.join(config.PROJECT, "models", "v1")
SUB_DIR = os.path.join(config.ROOT, "outputs", "submission")
SUB = os.path.join(SUB_DIR, "Barça Herizon.csv")
EVAL = config.path("scorer_dir")
# public-window pass rule, fixed 12 Nov before any run (catches broken plumbing, not quality)
MIN_TEAM_ACC_BOTH_ALIVE = 0.65
MIN_ALIVE_DEAD_F1 = 0.70


def verdict(checks):
    for k, v in checks.items():
        print("  {:<48} {}".format(k, "PASS" if v else "FAIL"))
    return all(checks.values())


def verify_manifest():
    man = json.load(open(os.path.join(V1, "manifest.json"), encoding="utf-8"))
    bad = [rel for rel, h in man["md5"].items()
           if hashlib.md5(open(os.path.join(config.ROOT, rel), "rb").read()).hexdigest() != h]
    print("  frozen v1 manifest ({} files) {:<22} {}".format(len(man["md5"]), "", "PASS" if not bad else "FAIL"))
    if bad:
        sys.exit(1)
    return man


def do_clean():
    t = time.time()
    ok = verdict(bar_ath.clean())
    print("8.1 {} ({:.0f}s)".format("PASSED" if ok else "FAILED", time.time() - t))


def do_coverage():
    """Input-only aggregate shares, written for a person to read; nothing printed."""
    t = time.time()
    p = bar_ath.load_players()
    b = bar_ath.load_ball()
    period, fd, fh = bar_ath.features()
    play = period > 0
    lines = ["BAR-ATH input coverage (frames inside the two halves; input only, no labels)", ""]
    lines.append("ball available after cleaning: {:.3f}".format(float((b.status[play] != 2).mean())))
    lines.append("ball observed (not estimated): {:.3f}".format(float((b.status[play] == 0).mean())))
    for t_ in ("H", "A"):
        n = p.valid[:, p.team == t_].sum(axis=1)[play]
        lines.append("team {}: median tracked players {:.0f} | share of frames with >= 7 tracked {:.3f}".format(
            t_, float(np.median(n)), float((n >= 7).mean())))
    for col in ("d_spd_slowshare_a_m5s", "d_both_slowshare_min_m5s", "d_shape_compact_h", "d_spd_median_h"):
        lines.append("dead-ball feature {:<28} blank on {:.3f} of frames".format(col, float(fd[col].isna()[play].mean())))
    for col in ("ha_pers_homecloser_past3s", "ha_ctx_compact_h_m5s", "ha_ctx_fwdvel_a_chg2s", "ha_near_margin"):
        lines.append("possession feature {:<27} blank on {:.3f} of frames".format(col, float(fh[col].isna()[play].mean())))
    lines += ["", "For comparison, the same shares on the external dev data are in runs (Phase 2 checks)."]
    path = os.path.join(bar_ath.work_dir(), "coverage.txt")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("8.2 coverage written to {} ({:.0f}s) - not printed".format(os.path.relpath(path, config.ROOT), time.time() - t))


def do_predict():
    t = time.time()
    man = verify_manifest()
    seg = man["segmenter"]
    period, fd, fh = bar_ath.features()
    p_d = predict(lgb.Booster(model_file=os.path.join(V1, "d_model.txt")), fd)
    p_h = predict(lgb.Booster(model_file=os.path.join(V1, "ha_model.txt")), fh)
    labels = segment(p_d, p_h, period, seg["cost_d"], seg["cost_h"], seg["prob_clip"])
    bar_ath.write_submission(labels, SUB)
    r = subprocess.run([sys.executable, os.path.join(EVAL, "validate_submission.py"), SUB],
                       capture_output=True, text=True)
    first = (r.stdout.strip().splitlines() or ["?"])[0].split()[0]
    checks = {"submission written (199,319 rows, H/A/D only)": os.path.isfile(SUB),
              "organisers' validate_submission.py": r.returncode == 0 and first == "OK"}
    ok = verdict(checks)
    with open(SUB, "rb") as fh_:
        md5 = hashlib.md5(fh_.read()).hexdigest()
    print("  file md5 {} (for the reproducibility check)".format(md5))
    print("8.3 {} ({:.0f}s) -> {}".format("PASSED" if ok else "FAILED", time.time() - t, os.path.relpath(SUB, config.ROOT)))


def do_public():
    """Public 5-minute window with the organisers' scorer. Full report -> local file; PASS/FAIL printed."""
    sys.path.insert(0, EVAL)
    import possession_metrics as pm
    gt_path = os.path.join(EVAL, "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    ours = pm.evaluate(gt, pm.label_codes(pm.read_label_csv(SUB)), bootstrap_n=0)
    base_path = os.path.join(EVAL, "results", "baseline_smooth50.csv")
    base = pm.evaluate(gt, pm.label_codes(pm.read_label_csv(base_path)), bootstrap_n=0) \
        if os.path.isfile(base_path) else None
    keys = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_gt_alive", "team_acc_both_alive",
            "n_gt_spells", "n_pred_spells", "n_graded")
    lines = ["Public 5-minute window (organisers' possession_metrics.evaluate). For reading by the team only.",
             "Pipeline check, NOT a tuning signal (RULES.md): ~17 possessions cannot rank anything.", ""]
    lines.append("{:<22} {:>12} {:>20}".format("metric", "v1", "baseline_smooth50"))
    for k in keys:
        lines.append("{:<22} {:>12} {:>20}".format(k, "{:.4f}".format(ours[k]) if isinstance(ours[k], float) else ours[k],
                                                   ("{:.4f}".format(base[k]) if isinstance(base[k], float) else base[k]) if base else "-"))
    lines += ["", "pass rule (fixed before running): team_acc_both_alive >= {} and alive_dead_f1 >= {}".format(
        MIN_TEAM_ACC_BOTH_ALIVE, MIN_ALIVE_DEAD_F1)]
    path = os.path.join(SUB_DIR, "public_check.txt")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    checks = {"team accuracy where both say live >= {}".format(MIN_TEAM_ACC_BOTH_ALIVE):
              ours["team_acc_both_alive"] >= MIN_TEAM_ACC_BOTH_ALIVE,
              "alive/dead F1 >= {}".format(MIN_ALIVE_DEAD_F1): ours["alive_dead_f1"] >= MIN_ALIVE_DEAD_F1}
    ok = verdict(checks)
    print("8.4 {} - full report for the team in {} (not printed)".format(
        "PASS" if ok else "FAIL", os.path.relpath(path, config.ROOT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for f in ("clean", "coverage", "predict", "public"):
        ap.add_argument("--" + f, action="store_true")
    a = ap.parse_args()
    if a.clean:
        do_clean()
    elif a.coverage:
        do_coverage()
    elif a.predict:
        do_predict()
    elif a.public:
        do_public()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
