"""Check the speed field's units against speed computed from positions (merged Rulebook 3.5).

The source's speed field (already converted to m/s by the loader when a unit was declared) is
compared with speed derived from consecutive observed positions. A median ratio near 1 means the
field is in m/s; near 3.6 means it is still in km/h; anything else means the field is something
else or unreliable. Only frames where the derived speed is above min_speed are used, because ratios
of near-zero speeds are noise.
"""

import numpy as np


def derived_speed(players, fps, period):
    n = players.shape[0]
    d = np.full(players.shape[:2], np.nan, dtype=np.float32)
    step = np.hypot(*(players[1:] - players[:-1]).transpose(2, 0, 1)) * fps
    same = (period[1:] == period[:-1])[:, None]
    d[1:] = np.where(same, step, np.nan)
    return d


def resolve_speed_units(tr):
    """Make tr.speed (and tr.ball_speed) m/s based on the positions, not on the declared label.

    BAR-ATH declares its speed channel as 'kph', but its values match position-derived speed in
    m/s (ratio 0.275 = 1/3.6 after converting). So:
      * converted from kph but now 3.6x too small  -> undo the conversion
      * looks like kph                             -> convert to m/s
    Modifies tr in place; returns (check before, check after, action taken).
    """
    before = check_speed_units(tr)
    action = "none"
    r = before.get("median_ratio_given_over_derived")
    if r is not None and before["verdict"] != "consistent_mps":
        if 0.23 <= r <= 0.33:
            factor, action = 3.6, "multiplied by 3.6: values were already m/s despite the declared unit"
        elif 3.0 <= r <= 4.2:
            factor, action = 1 / 3.6, "divided by 3.6: values were km/h"
        else:
            factor = None
        if factor is not None:
            tr.speed = (tr.speed * factor).astype(np.float32)
            if tr.ball_speed is not None:
                tr.ball_speed = (tr.ball_speed * factor).astype(np.float32)
    after = check_speed_units(tr) if action != "none" else before
    tr.meta["speed_unit_effective"] = ("m/s" if after["verdict"] == "consistent_mps"
                                       else "unverified ({})".format(after["verdict"]))
    tr.meta["speed_unit_action"] = action
    return before, after, action


def check_speed_units(tr, min_speed=1.0, max_speed=12.0):
    if not np.isfinite(tr.speed).any():
        return {"verdict": "no_speed_field", "declared_unit": tr.meta.get("speed_unit_declared")}
    d = derived_speed(tr.players, tr.fps, tr.period)
    given = tr.speed
    ok = np.isfinite(d) & np.isfinite(given) & (d > min_speed) & (d < max_speed)
    if ok.sum() < 100:
        return {"verdict": "too_few_samples", "samples": int(ok.sum()),
                "declared_unit": tr.meta.get("speed_unit_declared")}
    ratio = float(np.median(given[ok] / d[ok]))
    if 0.85 <= ratio <= 1.15:
        verdict = "consistent_mps"
    elif 3.0 <= ratio <= 4.2:
        verdict = "looks_like_kph"
    else:
        verdict = "inconsistent"
    return {"verdict": verdict, "median_ratio_given_over_derived": round(ratio, 3),
            "samples": int(ok.sum()), "declared_unit": tr.meta.get("speed_unit_declared"),
            "converted_to_mps_by_loader": bool(tr.meta.get("speed_converted_to_mps", False))}
