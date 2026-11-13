"""BAR-ATH (the provided game): read, clean, predict with the frozen v1. Nothing here trains,
tunes or reads labels. Phase 8.

The tactical ATD (FIFA/EPTS metadata .xml + raw .txt) is read with atd-cleaner's EPTS reader:
  * video frame = ATD frame_id - 1 (organisers' starter/load_atd.py), 199,319 frames
  * the first team in the metadata is home (H), as in the organisers' baseline
  * halves = the published tracking blocks in video frames (games.py / DATA.md), from
    config/epts_bar_ath.json
  * pitch 105.3 x 68.0 m (games.py)
Every intermediate file goes to a work folder outside the repository (default <root>/outputs/bar_ath).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from . import config
from .data import BallData, PlayerData, config_features

FPS = 25.0
N_FRAMES = 199319


def cleaner_config():
    path = os.path.join(config.path("cleaner_root"), "config", "epts_bar_ath.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def data_dir(override=None):
    """The organisers' data bundle (folder holding tactical/), like games.py: MTAH_DATA_DIR or
    <root>/more-than-a-hack-2027/data."""
    return override or os.environ.get("MTAH_DATA_DIR") or os.path.join(config.ROOT, "more-than-a-hack-2027", "data")


def work_dir(override=None):
    d = override or os.environ.get("BH_WORK_DIR") or os.path.join(config.ROOT, "outputs", "bar_ath")
    os.makedirs(d, exist_ok=True)
    return d


def clean(data=None, work=None):
    """Read + clean BAR-ATH; write frames.parquet and players.parquet. Returns pass/fail checks only."""
    sys.path.insert(0, config.path("cleaner_root"))
    from atd_cleaner.io_epts import read_epts, read_metadata
    from atd_cleaner.io_table import frames_table, objects_table
    from atd_cleaner.pipeline import load_thresholds, run

    cfg = cleaner_config()
    tac = os.path.join(data_dir(data), "tactical")
    meta = os.path.join(tac, "BAR-ATH_tactical_atd.xml")
    raw = os.path.join(tac, "BAR-ATH_tactical_atd.txt")
    md = read_metadata(meta)
    tr = read_epts(meta, raw, pitch=cfg["pitch_m"], frame_id_offset=cfg["frame_id_offset"],
                   n_frames=cfg["n_frames"], periods=cfg["periods"], origin=cfg["origin"],
                   flip_y=cfg["flip_y"])
    res, quality, rep = run(tr, load_thresholds(config.path("cleaner_thresholds")))
    w = work_dir(work)
    frames_table(tr, res, quality, match_id="BAR-ATH").to_parquet(os.path.join(w, "frames.parquet"), index=False)
    objects_table(tr, res, match_id="BAR-ATH").to_parquet(os.path.join(w, "players.parquet"), index=False)
    checks = {
        "199,319 frames numbered 0..199318": bool(len(tr.frame) == N_FRAMES and tr.frame[0] == 0
                                                   and (np.diff(tr.frame) == 1).all()),
        "2 teams in the metadata": len(md["teams"]) == 2,
        "every tracked slot has a team in the metadata": all(str(s) in md["players"] for s in tr.slot_ids),
        "both teams have tracked slots": bool((tr.team == "H").any() and (tr.team == "A").any()),
        "2 halves set": sorted(set(np.unique(tr.period)) - {0}) == [1, 2],
        "no unreadable raw lines": tr.meta.get("lines_skipped", 1) == 0,
        "cleaner invariants": not rep["invariant_problems"],
    }
    return checks


def load_players(work=None):
    w = work_dir(work)
    f = pd.read_parquet(os.path.join(w, "frames.parquet"), columns=["frame", "period"]).sort_values("frame")
    frame = f["frame"].to_numpy(np.int64)
    period = f["period"].fillna(0).to_numpy(np.int16)
    o = pd.read_parquet(os.path.join(w, "players.parquet"),
                        columns=["frame", "slot", "team", "is_goalkeeper", "x", "y", "speed_derived_ms"])
    slots = np.array(sorted(o["slot"].unique()))
    row_of = pd.Series(np.arange(len(frame)), index=frame)
    col_of = pd.Series(np.arange(len(slots)), index=slots)
    xy = np.full((len(frame), len(slots), 2), np.nan, dtype=np.float32)
    speed = np.full((len(frame), len(slots)), np.nan, dtype=np.float32)
    r = row_of.loc[o["frame"].to_numpy()].to_numpy()
    c = col_of.loc[o["slot"].to_numpy()].to_numpy()
    xy[r, c, 0] = o["x"].to_numpy(np.float32)
    xy[r, c, 1] = o["y"].to_numpy(np.float32)
    speed[r, c] = o["speed_derived_ms"].to_numpy(np.float32)
    first = o.drop_duplicates("slot").set_index("slot")
    team = first.loc[slots, "team"].to_numpy()
    gk = first.loc[slots, "is_goalkeeper"].fillna(-1).to_numpy(np.int8)
    return PlayerData("BAR-ATH", frame, period, xy, speed, team, gk, tuple(cleaner_config()["pitch_m"]), FPS)


def load_ball(work=None):
    f = pd.read_parquet(os.path.join(work_dir(work), "frames.parquet"),
                        columns=["frame", "period", "ball_x", "ball_y", "ball_status", "ball_conf"]).sort_values("frame")
    return BallData("BAR-ATH", f["frame"].to_numpy(np.int64), f["period"].fillna(0).to_numpy(np.int16),
                    f[["ball_x", "ball_y"]].to_numpy(np.float32), f["ball_status"].to_numpy(np.int8),
                    f["ball_conf"].to_numpy(np.float32))


def features(work=None):
    from .features_d import build_d_features
    from .features_ha import build_ha_features
    cfg = config_features()
    p = load_players(work)
    fd, _ = build_d_features(p, cfg)
    fh, _ = build_ha_features(p, load_ball(work), cfg)
    return p.period, fd, fh


def write_submission(labels, path):
    """frame,label CSV, one row per video frame (SUBMISSION.md)."""
    labels = np.asarray(labels).astype(str)
    if len(labels) != N_FRAMES or not set(labels) <= {"H", "A", "D"}:
        raise ValueError("submission must have 199,319 labels from H/A/D")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path
