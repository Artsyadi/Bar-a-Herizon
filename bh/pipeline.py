"""Features + labels for a match or unit, on demand. Used by step 2.4 (training tables) and by the
later phases (full-frame features for held-out scoring).

    from bh.pipeline import d_features, ha_features
    df, meta = d_features("idsse_J03WMX")             # full frames, D features (players only)
    df, meta = ha_features("idsse_J03WMX", seed=1)    # full frames, H/A features
"""

import numpy as np

from .data import config_features, load_ball, load_players
from .features_d import build_d_features
from .features_ha import build_ha_features
from .scoring import load_labels


def d_features(match, cfg=None):
    return build_d_features(load_players(match), cfg or config_features())


def ha_features(match, seed, cfg=None):
    return build_ha_features(load_players(match), load_ball(match, seed), cfg or config_features())


def attach_labels(df, match):
    """Add `label` (H/A/D) by frame. Raises if the frame sets differ."""
    frames, labels = load_labels(match)
    if not np.array_equal(frames, df.index.to_numpy()):
        raise ValueError("{}: label frames differ from feature frames".format(match))
    out = df.copy()
    out["label"] = labels
    return out


def stride_mask(frames, stride):
    """Training rows: frames whose index is a multiple of `stride` (the same rows for every seed)."""
    return (np.asarray(frames) % stride) == 0
