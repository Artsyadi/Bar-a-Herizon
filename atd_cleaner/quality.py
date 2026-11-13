"""Per-frame tracking-quality columns (merged Rulebook 4.1, 4.2, 4.10).

Quality says how much to trust each frame's inputs. It is meant to control which features are
usable, never to define a stoppage: poor quality means uncertainty, not D and not live play.

Columns (one row per frame):
  home_valid_n, away_valid_n      players with an observed position after cleaning (duplicates
                                  already merged, so this is the deduplicated count)
  valid_n_change                  change in total valid players since the previous frame
  home_speed_valid_n/_share       players with a valid position-based speed this frame
  away_speed_valid_n/_share       (speed_derived_ms finite: observed at t-w, t, t+w, same period,
                                  below the speed guard); share = speed_valid_n / valid_n,
                                  NaN when valid_n = 0
  n_track_breaks                  slots with an unresolved identity/position break this frame
  n_duplicates_merged             slots removed as duplicates this frame
  n_speed_removed                 slots with an impossible-speed point removed this frame
  ball_quality                    0 observed, 1 estimated, 2 unavailable (same as ball_status)
  ball_track_break                ball has an unresolved break this frame
  goalkeepers_valid_n             valid players in goalkeeper slots (NaN if goalkeepers unknown)
"""

import numpy as np
import pandas as pd

from .clean import FLAG_DUPLICATE_MERGED, FLAG_SPEED_REMOVED, FLAG_TRACK_BREAK, OBSERVED


def quality_table(tr, result):
    p = result["players"]
    valid = result["player_status"] == OBSERVED
    flags = result["player_flags"]
    home = tr.team == "H"
    away = ~home

    if "player_speed_derived" in result:
        speed_ok = np.isfinite(result["player_speed_derived"])
    else:                                   # v0.2.0 behaviour: observed now and one frame before
        same_period = np.zeros(tr.n_frames, dtype=bool)
        same_period[1:] = (tr.period[1:] == tr.period[:-1]) & (tr.period[1:] > 0)
        speed_ok = np.zeros_like(valid)
        speed_ok[1:] = valid[1:] & valid[:-1] & same_period[1:, None]

    def count(mask, cols):
        return mask[:, cols].sum(axis=1).astype(np.int16)

    hv, av = count(valid, home), count(valid, away)
    hs, as_ = count(speed_ok, home), count(speed_ok, away)
    with np.errstate(invalid="ignore", divide="ignore"):
        h_share = np.where(hv > 0, hs / np.maximum(hv, 1), np.nan)
        a_share = np.where(av > 0, as_ / np.maximum(av, 1), np.nan)
    total = hv.astype(np.int32) + av
    change = np.concatenate(([0], np.diff(total))).astype(np.int16)

    df = pd.DataFrame({
        "home_valid_n": hv, "away_valid_n": av, "valid_n_change": change,
        "home_speed_valid_n": hs, "home_speed_valid_share": h_share.astype(np.float32),
        "away_speed_valid_n": as_, "away_speed_valid_share": a_share.astype(np.float32),
        "n_track_breaks": ((flags & FLAG_TRACK_BREAK) > 0).sum(axis=1).astype(np.int16),
        "n_duplicates_merged": ((flags & FLAG_DUPLICATE_MERGED) > 0).sum(axis=1).astype(np.int16),
        "n_speed_removed": ((flags & FLAG_SPEED_REMOVED) > 0).sum(axis=1).astype(np.int16),
        "ball_quality": result["ball_status"].astype(np.int8),
        "ball_track_break": ((result["ball_flags"] & FLAG_TRACK_BREAK) > 0),
    })
    gk = tr.goalkeeper == 1
    df["goalkeepers_valid_n"] = valid[:, gk].sum(axis=1).astype(np.float32) if gk.any() else np.nan
    return df
