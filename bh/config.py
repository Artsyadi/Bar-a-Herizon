"""Paths and the train/dev/test split, loaded from config/ - the one place every script gets them.

    from bh.config import split, folds, units, path
    for train, val in folds():          # 6 leave-one-match-out folds over the dev matches
        ...
    path("labels", match="idsse_J03WMX")
"""

import json
import os

PACKAGE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(PACKAGE)                  # repository root
ROOT = PROJECT                                      # paths.json is relative to the repository root
CONFIG = os.path.join(PROJECT, "config")


def _load(name):
    with open(os.path.join(CONFIG, name), encoding="utf-8") as handle:
        return {k: v for k, v in json.load(handle).items() if not k.startswith("_")}


def split():
    return _load("split.json")


def paths():
    return _load("paths.json")


def path(key, **fmt):
    """Absolute path for a key in config/paths.json, formatted with match= / seed= if needed."""
    return os.path.normpath(os.path.join(ROOT, paths()[key].format(**fmt)))


def is_idsse(match):
    return match.startswith("idsse_")


def units(match):
    """The training/evaluation units of a match: (match, seed) for each IDSSE seed, or
    (match, None) for a Metrica match (not degraded)."""
    if is_idsse(match):
        return [(match, s) for s in split()["idsse_seeds"]]
    return [(match, None)]


def unit_id(match, seed):
    """File-name id of a unit: 'idsse_J03WMX_s1' or 'metrica_game_1'."""
    return match if seed is None else "{}_s{}".format(match, seed)


def all_matches():
    s = split()
    return s["dev_matches"] + s["test_matches"] + s["comparison_matches"]


def folds():
    """Leave-one-match-out over the dev matches: list of (train_matches, val_match)."""
    dev = split()["dev_matches"]
    return [([m for m in dev if m != val], val) for val in dev]


def check_split():
    """Raise if the split breaks a rule. Returns the split when it is sound."""
    s = split()
    dev, test, comp = set(s["dev_matches"]), set(s["test_matches"]), set(s["comparison_matches"])
    problems = []
    if dev & test or dev & comp or test & comp:
        problems.append("a match appears in more than one role")
    if not test:
        problems.append("no test match")
    for train, val in folds():
        if val in train:
            problems.append("fold {}: validation match also in training".format(val))
        if test & set(train) or val in test:
            problems.append("fold {}: test match used".format(val))
        train_units = {m for t in train for m, _ in units(t)}
        if train_units & {m for m, _ in units(val)}:
            problems.append("fold {}: a seed of the validation match is in training".format(val))
    if sorted(v for _, v in folds()) != sorted(dev):
        problems.append("dev matches are not each validated exactly once")
    if problems:
        raise ValueError("split problems: " + "; ".join(problems))
    return s


def input_files():
    """Every input file v1 needs, as (description, path)."""
    s = split()
    out = []
    for m in s["dev_matches"] + s["test_matches"] + s["comparison_matches"]:
        out.append(("raw frames " + m, path("raw_frames", match=m)))
        out.append(("raw objects " + m, path("raw_objects", match=m)))
        out.append(("labels " + m, path("labels", match=m)))
        if is_idsse(m):
            for seed in s["idsse_seeds"]:
                out.append(("degraded frames {} seed {}".format(m, seed),
                            path("degraded_frames", match=m, seed=seed)))
    out.append(("cleaner thresholds", path("cleaner_thresholds")))
    out.append(("official scorer", os.path.join(path("scorer_dir"), "possession_metrics.py")))
    return out
