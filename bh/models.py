"""Phase 4 helpers: training tables -> LightGBM -> out-of-fold probabilities.

    from bh.models import feature_names, load_tables, train, alive_f1
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config

FEATURES_LIST = os.path.join(config.PROJECT, "docs", "features_list.csv")
TABLES = os.path.join(config.PROJECT, "data", "features")
PHASE4 = os.path.join(config.PROJECT, "runs", "phase4")


def models_config():
    with open(os.path.join(config.CONFIG, "models.json"), encoding="utf-8") as h:
        return json.load(h)


def params(kind):
    """LightGBM params for 'd_model' / 'ha_model' (keys starting with '_' are notes)."""
    p = {k: v for k, v in models_config()[kind].items() if not k.startswith("_")}
    rounds = p.pop("num_boost_round")
    return p, rounds


def feature_names(kind):
    """Model inputs for 'D' or 'HA', from the generated feature list. Never ball columns for D."""
    f = pd.read_csv(FEATURES_LIST)
    names = f.loc[f.model == kind, "name"].tolist()
    if kind == "D":
        cfg = json.load(open(os.path.join(config.CONFIG, "features.json"), encoding="utf-8"))
        banned = set(cfg["excluded_inputs"]["d_model"])
        bad = [n for n in names if n in banned or "ball" in n]
        if bad:
            raise ValueError("D inputs include ball/quality columns: {}".format(bad[:5]))
    return names


def table_path(kind, uid):
    return os.path.join(TABLES, "train_d" if kind == "D" else "train_ha", uid + ".parquet")


def load_tables(kind, ids, names):
    """Stack training tables. Returns X (float32), y (int8), info DataFrame (match, seed, frame)."""
    target = "target_d" if kind == "D" else "target_h"
    xs, ys, infos = [], [], []
    for uid in ids:
        df = pd.read_parquet(table_path(kind, uid), columns=["match", "seed"] + names + [target])
        xs.append(df[names].to_numpy(np.float32))
        ys.append(df[target].to_numpy(np.int8))
        infos.append(pd.DataFrame({"match": df["match"].to_numpy(), "seed": df["seed"].to_numpy(),
                                   "frame": df.index.to_numpy()}))
    return np.vstack(xs), np.concatenate(ys), pd.concat(infos, ignore_index=True)


def train(kind_key, X, y, names):
    p, rounds = params(kind_key)
    ds = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=rounds)


def predict(booster, df):
    """Probabilities from a DataFrame holding (at least) the booster's features, picked by name."""
    return booster.predict(df[booster.feature_name()].to_numpy(np.float32))


def alive_f1(gt, pred):
    """The official alive/dead F1 (ALIVE positive), for label arrays over graded frames."""
    ga, pa = np.isin(gt, ["H", "A"]), np.isin(pred, ["H", "A"])
    tp, fp, fn = float((ga & pa).sum()), float((~ga & pa).sum()), float((ga & ~pa).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def dead_f1(gt, pred):
    gd, pd_ = gt == "D", pred == "D"
    tp, fp, fn = float((gd & pd_).sum()), float((~gd & pd_).sum()), float((gd & ~pd_).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
