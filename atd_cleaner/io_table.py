"""Read and write the long-table format used in the external training pipeline (challenge_v3).

Input
  frames  (one row per frame):  frame, period, ball_x, ball_y, [ball_z], [ball_visible], ...
  objects (one row per frame x visible slot):  frame, slot, x, y, team ('H'/'A'), [speed_ms],
                                               [period], [is_goalkeeper]
  Coordinates in metres, centre origin by default. A ball is missing where ball_x/ball_y is NaN or
  ball_visible is False. Any other frame columns are carried through unchanged.

Output (same layout, plus cleaning columns) - see README "Output files".
"""

import os

import numpy as np
import pandas as pd

from .tracking import Tracking

FRAME_REQUIRED = ("frame", "period", "ball_x", "ball_y")
OBJECT_REQUIRED = ("frame", "slot", "x", "y", "team")


def read_table(frames, objects, fps=25.0, pitch=(105.0, 68.0), origin="centre"):
    """frames / objects: DataFrames or paths to .parquet / .csv files."""
    frames = _load(frames)
    objects = _load(objects)
    for name, df, cols in (("frames", frames, FRAME_REQUIRED), ("objects", objects, OBJECT_REQUIRED)):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError("{} table is missing columns {}".format(name, missing))

    frames = frames.sort_values("frame").reset_index(drop=True)
    frame_ids = frames["frame"].to_numpy(np.int64)
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("frames table has duplicate frame values")
    row_of = pd.Series(np.arange(len(frame_ids)), index=frame_ids)

    objects = objects[objects["frame"].isin(row_of.index)]
    if objects.duplicated(["frame", "slot"]).any():
        raise ValueError("objects table has duplicate (frame, slot) rows")
    teams = objects.groupby("slot")["team"].agg(lambda s: set(s.dropna()))
    mixed = [int(k) for k, v in teams.items() if len(v) != 1]
    if mixed:
        raise ValueError("slots with more than one team (or none): {}".format(mixed[:10]))
    slot_ids = np.array(sorted(teams.index))
    col_of = pd.Series(np.arange(len(slot_ids)), index=slot_ids)

    n, S = len(frame_ids), len(slot_ids)
    players = np.full((n, S, 2), np.nan, dtype=np.float32)
    speed = np.full((n, S), np.nan, dtype=np.float32)
    present = np.zeros((n, S), dtype=bool)
    r = row_of.loc[objects["frame"].to_numpy()].to_numpy()
    c = col_of.loc[objects["slot"].to_numpy()].to_numpy()
    players[r, c, 0] = objects["x"].to_numpy(np.float32)
    players[r, c, 1] = objects["y"].to_numpy(np.float32)
    if "speed_ms" in objects.columns:
        speed[r, c] = objects["speed_ms"].to_numpy(np.float32)
    present[r, c] = True

    team = np.array([next(iter(teams[s])) for s in slot_ids])
    gk = np.full(S, -1, dtype=np.int8)
    if "is_goalkeeper" in objects.columns:
        g = objects.groupby("slot")["is_goalkeeper"].max()
        gk = np.array([int(g.get(s)) if pd.notna(g.get(s)) else -1 for s in slot_ids], dtype=np.int8)

    ball = frames[["ball_x", "ball_y"]].to_numpy(np.float32).copy()
    if "ball_visible" in frames.columns:
        ball[~frames["ball_visible"].fillna(False).to_numpy(bool)] = np.nan
    ball_z = frames["ball_z"].to_numpy(np.float32) if "ball_z" in frames.columns else None

    passthrough = {col: frames[col].to_numpy() for col in frames.columns
                   if col not in ("frame", "period", "ball_x", "ball_y", "ball_z", "ball_visible")}
    tr = Tracking(frame=frame_ids, period=frames["period"].fillna(0).to_numpy(np.int16), ball=ball,
                  players=players, speed=speed, slot_ids=slot_ids, team=team, goalkeeper=gk,
                  fps=float(fps), pitch=tuple(pitch), origin=origin, ball_z=ball_z,
                  present=present, frame_columns=passthrough,
                  meta={"source": "table", "speed_unit_declared": "m/s (speed_ms column)"
                        if "speed_ms" in objects.columns else None})
    return tr.validate()


def _load(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if str(obj).endswith(".csv"):
        return pd.read_csv(obj)
    return pd.read_parquet(obj)


def frames_table(tr: Tracking, result=None, quality=None, match_id=None) -> pd.DataFrame:
    """Per-frame table: input columns, raw and cleaned ball, ball status/flags/confidence, quality."""
    df = pd.DataFrame({"frame": tr.frame, "period": tr.period})
    if match_id is not None:
        df.insert(0, "match_id", match_id)
    for col, values in tr.frame_columns.items():
        if col not in df.columns:
            df[col] = values
    df["ball_x_raw"] = tr.ball[:, 0]
    df["ball_y_raw"] = tr.ball[:, 1]
    if tr.ball_z is not None:
        df["ball_z"] = tr.ball_z
    if tr.ball_speed is not None:
        df["ball_speed_given_ms"] = tr.ball_speed
    if result is not None:
        df["ball_x"] = result["ball"][:, 0]
        df["ball_y"] = result["ball"][:, 1]
        df["ball_status"] = result["ball_status"]
        df["ball_flags"] = result["ball_flags"]
        df["ball_conf"] = result["ball_conf"]
    if quality is not None:
        for col in quality.columns:
            df[col] = quality[col].to_numpy()
    return df


def objects_table(tr: Tracking, result=None, match_id=None) -> pd.DataFrame:
    """Per (frame, slot) table, one row wherever the input had a row (removed points keep their
    row with x/y NaN and status 2, so nothing silently disappears)."""
    present = tr.present if tr.present is not None else np.isfinite(tr.players[:, :, 0])
    r, c = np.nonzero(present)
    df = pd.DataFrame({
        "frame": tr.frame[r], "period": tr.period[r], "slot": tr.slot_ids[c],
        "team": tr.team[c], "is_goalkeeper": tr.goalkeeper[c],
        "x_raw": tr.players[r, c, 0], "y_raw": tr.players[r, c, 1],
        "speed_ms": tr.speed[r, c],
    })
    if match_id is not None:
        df.insert(0, "match_id", match_id)
    if result is not None:
        df["x"] = result["players"][r, c, 0]
        df["y"] = result["players"][r, c, 1]
        df["player_status"] = result["player_status"][r, c]
        df["player_flags"] = result["player_flags"][r, c]
        if "player_speed_derived" in result:
            df["speed_derived_ms"] = result["player_speed_derived"][r, c]
    return df


def write_tables(tr, result, quality, out_dir, match_id):
    os.makedirs(out_dir, exist_ok=True)
    f_path = os.path.join(out_dir, "{}_frames_clean.parquet".format(match_id))
    o_path = os.path.join(out_dir, "{}_objects_clean.parquet".format(match_id))
    frames_table(tr, result, quality, match_id).to_parquet(f_path, index=False)
    objects_table(tr, result, match_id).to_parquet(o_path, index=False)
    return f_path, o_path
