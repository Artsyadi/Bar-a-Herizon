"""Score any external match with the OFFICIAL challenge scorer (possession_metrics.py).

The official scorer derives "runs" (the halves) from X frames in the ground truth and never scores
across a run boundary. BAR-ATH's ground truth has an X gap at half-time; the external labels don't
(frames run straight through both halves). So this wrapper inserts ONE X separator frame into both
the ground truth and the prediction at every period change, and marks period-0 frames as X in the
ground truth. No graded frame is dropped; nothing else about the scoring changes.

    from bh.scoring import score, load_labels
    gt = load_labels("idsse_J03WMX")                 # frame-ordered H/A/D strings
    result = score(gt, pred, periods)                # dict of official metrics
"""

import os
import sys

import numpy as np
import pandas as pd

from . import config

sys.path.insert(0, config.path("scorer_dir"))
import possession_metrics as pm  # noqa: E402  (the official file, unmodified)

HEADLINE = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_gt_alive",
            "n_gt_spells", "n_pred_spells", "spell_count_error",
            "gt_mean_spell_duration_s", "mean_spell_duration_s")


def load_labels(match):
    """(frames, labels) for an external match, frame-ordered, as numpy arrays."""
    df = pd.read_csv(config.path("labels", match=match))
    df = df.sort_values("frame").reset_index(drop=True)
    if not (np.diff(df["frame"].to_numpy()) == 1).all():
        raise ValueError("{}: label frames are not contiguous".format(match))
    return df["frame"].to_numpy(), df["label"].astype(str).str.upper().to_numpy()


def with_separators(gt, pred, periods):
    """Insert one X frame at every period change; period-0 frames become X in the ground truth."""
    gt = np.asarray(gt, dtype=object).copy()
    pred = np.asarray(pred, dtype=object).copy()
    periods = np.asarray(periods)
    if not (len(gt) == len(pred) == len(periods)):
        raise ValueError("gt, pred and periods must have the same length")
    gt[periods <= 0] = "X"
    cuts = np.flatnonzero(np.diff(periods) != 0) + 1
    gt = np.insert(gt, cuts, "X")
    pred = np.insert(pred, cuts, "X")
    return gt, pred


def score(gt, pred, periods, bootstrap_n=0):
    """Official metrics for one match. gt / pred: arrays of 'H','A','D' ('X' allowed in pred)."""
    g, p = with_separators(gt, pred, periods)
    return pm.evaluate(pm.label_codes(g), pm.label_codes(p), bootstrap_n=bootstrap_n)


def headline(result):
    return {k: result[k] for k in HEADLINE}


def shift_labels(labels, periods, k):
    """Labels delayed by k frames inside each period (the first k frames repeat the first label).
    Used to check the scorer reproduces EVALUATION.md's timing table."""
    out = np.asarray(labels, dtype=object).copy()
    periods = np.asarray(periods)
    for per in np.unique(periods):
        idx = np.flatnonzero(periods == per)
        seg = out[idx].copy()
        if k > 0:
            seg = np.concatenate([np.repeat(seg[:1], k), seg[:-k]])
        out[idx] = seg
    return out
