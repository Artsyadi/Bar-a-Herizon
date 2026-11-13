"""Feature tests on made-up players (and ball, for H/A). No match data needed.

    python tests/test_features.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh.data import BallData, PlayerData, config_features  # noqa: E402
from bh.features_common import attack_direction, segments, window_mean  # noqa: E402
from bh.features_d import build_d_features  # noqa: E402

CFG = config_features()
FPS = 25.0


def make_players(n=500, speed_h=3.0, speed_a=3.0, seed=0):
    """11 v 11. Home defends -x (goalkeeper near x=-50), away defends +x, both halves the same."""
    rng = np.random.default_rng(seed)
    S = 22
    team = np.array(["H"] * 11 + ["A"] * 11)
    base = np.zeros((S, 2), np.float32)
    base[0] = [-50, 0]
    base[1:11] = np.column_stack([rng.uniform(-40, 5, 10), rng.uniform(-30, 30, 10)])
    base[11] = [50, 0]
    base[12:22] = np.column_stack([rng.uniform(-5, 40, 10), rng.uniform(-30, 30, 10)])
    t = (np.arange(n) / FPS)[:, None]
    xy = np.repeat(base[None], n, axis=0).copy()
    xy[:, 1:11, 1] += (speed_h * t).astype(np.float32)     # outfielders run along y
    xy[:, 12:22, 1] -= (speed_a * t).astype(np.float32)
    speed = np.zeros((n, S), np.float32)
    speed[:, 1:11], speed[:, 12:22] = speed_h, speed_a
    period = np.ones(n, np.int16)
    return PlayerData("synthetic", np.arange(n), period, xy, speed, team,
                      np.full(S, -1, np.int8), (105.0, 68.0), FPS)


def replace(p, **kw):
    d = dict(p.__dict__)
    d.update(kw)
    return PlayerData(**d)


# -- shared helpers --------------------------------------------------------------------------------

def test_window_mean_ignores_nan_and_respects_segments():
    x = np.arange(10, dtype=float)
    x[3] = np.nan
    segs = segments(np.array([1] * 5 + [2] * 5))
    m = window_mean(x, segs, -1, 1)
    assert np.isclose(m[1], 1.0)
    assert np.isnan(m[4])                       # segment edge: only frame 4 valid (1 of 3) < 50% -> NaN
    assert np.isclose(m[5], 5.5)                # frame 5 doesn't see frame 4 (other segment)
    m2 = window_mean(x, segs, -1, 1, min_frac=0.3)
    assert np.isclose(m2[4], 4.0)               # with a lower minimum: mean of {4}, NaN ignored


def test_fast_quantile_matches_numpy():
    from bh.features_d import _nanq
    rng = np.random.default_rng(1)
    a = rng.normal(size=(500, 11)).astype(np.float32)
    a[rng.random(a.shape) < 0.3] = np.nan
    a[7] = np.nan                                                    # an all-NaN row
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for q in (0.1, 0.5, 0.9):
            ref = np.nanquantile(a, q, axis=1)
            got = _nanq(a, q)
            assert np.allclose(got, ref, atol=1e-5, equal_nan=True)


def test_attack_direction_from_deepest_player():
    p = make_players()
    d = attack_direction(p.xy, p.team, p.period)
    assert np.all(d == 1.0)                                          # home defends -x -> attacks +x
    flipped = replace(p, xy=p.xy * np.array([-1, 1], np.float32))
    assert np.all(attack_direction(flipped.xy, flipped.team, flipped.period) == -1.0)


# -- D features ------------------------------------------------------------------------------------

def test_d_features_refuse_ball_data():
    ball = BallData("u", np.arange(3), np.ones(3), np.zeros((3, 2)), np.zeros(3), np.ones(3))
    try:
        build_d_features(ball, CFG)
        raise AssertionError("D features accepted ball data")
    except TypeError:
        pass


def test_d_speed_features_and_slowing():
    df, meta = build_d_features(make_players(speed_h=1.0, speed_a=6.0), CFG)
    assert np.allclose(df["d_spd_median_h"], 1.0) and np.allclose(df["d_spd_median_a"], 6.0)
    # goalkeepers stand still (speed 0): home 11/11 slow; away only its goalkeeper (1/11)
    assert np.allclose(df["d_spd_slowshare_h"], 1.0) and np.allclose(df["d_spd_slowshare_a"], 1 / 11)
    assert np.allclose(df["d_both_median_min"], 1.0) and np.allclose(df["d_both_median_max"], 6.0)
    assert len(df.columns) == len(meta) and all(m["model"] == "D" for m in meta)


def test_missing_speed_is_never_zero_and_gates_out():
    p = make_players()
    sp = p.speed.copy()
    sp[:, :20] = np.nan                                              # most players: no speed
    df, _ = build_d_features(replace(p, speed=sp), CFG)
    assert df["d_spd_median_h"].isna().all()                        # gated out, not 0
    assert not (df["d_spd_median_h"] == 0).any()


def test_low_coverage_gates_shape():
    p = make_players()
    xy = p.xy.copy()
    xy[:, 1:8] = np.nan                                              # home: only 4 players visible
    df, _ = build_d_features(replace(p, xy=xy), CFG)
    assert df["d_shape_width_h"].isna().all() and df["d_shape_width_a"].notna().all()


def test_kickoff_shape_and_tight_group():
    p = make_players(speed_h=0.0, speed_a=0.0)
    df, _ = build_d_features(p, CFG)
    assert (df["d_ko_ownhalf_min"] > 0.7).all()                      # synthetic teams mostly in own half
    xy = p.xy.copy()
    xy[:, 12:16] = np.array([[20, 0], [20, 0.8], [20, 1.6], [20, 2.4]], np.float32)   # 4 away in a line
    df2, _ = build_d_features(replace(p, xy=xy), CFG)
    assert (df2["d_tight_count_a"] >= 3).all()
    assert np.allclose(df2["d_tight_goaldist_a"].iloc[50], np.hypot(52.5 - 20, 1.2), atol=0.5)


def test_d_columns_have_windows_and_no_ball_names():
    df, meta = build_d_features(make_players(), CFG)
    assert any(c.endswith("_m1s") for c in df) and any(c.endswith("_chg2s") for c in df)
    assert not any("ball" in c for c in df.columns)


# -- H/A features ----------------------------------------------------------------------------------

def dribble(n=400, gap=None):
    """Home player 3 carries the ball along +y at 4 m/s; away players far away."""
    p = make_players(n=n, speed_h=0.0, speed_a=0.0)
    xy = p.xy.copy()
    t = (np.arange(n) / FPS).astype(np.float32)
    xy[:, 3, 0] = 0.0
    xy[:, 3, 1] = -20 + 4 * t
    xy[:, 12:22, 0] = 40.0                       # away outfielders well clear of the dribble
    sp = p.speed.copy()
    sp[:, 3] = 4.0
    ball = np.column_stack([xy[:, 3, 0] + 0.5, xy[:, 3, 1] + 0.3]).astype(np.float32)
    status = np.zeros(n, np.int8)
    if gap is not None:
        ball[gap[0]:gap[1]] = np.nan
        status[gap[0]:gap[1]] = 2
    pl = replace(p, xy=xy, speed=sp)
    bd = BallData("u", np.arange(n), np.ones(n, np.int16), ball, status,
                  np.where(status == 0, 1.0, 0.0).astype(np.float32))
    return pl, bd


def test_ha_dribble_gives_home_evidence():
    from bh.features_ha import build_ha_features
    pl, bd = dribble()
    df, meta = build_ha_features(pl, bd, CFG)
    mid = slice(100, 300)
    assert (df["ha_near_margin"].iloc[mid] > 5).all()                 # home much closer
    assert np.allclose(df["ha_near_dist_h"].iloc[mid], np.hypot(0.5, 0.3), atol=1e-3)
    assert (df["ha_carr_cos_h"].iloc[mid] > 0.99).all()               # moving together
    assert (df["ha_carr_relspeed_h"].iloc[mid] < 0.1).all()
    assert (df["ha_pers_homecloser_past1s"].iloc[mid] == 1).all()
    assert (df["ha_pers_run_s"].iloc[mid] > 0).all() and df["ha_pers_run_s"].iloc[299] > df["ha_pers_run_s"].iloc[150]
    assert all(m["model"] == "HA" for m in meta) and not any("ball_z" in c for c in df.columns)


def test_ha_missing_ball_is_unknown_not_invented():
    from bh.features_ha import build_ha_features
    pl, bd = dribble(gap=(200, 260))
    df, _ = build_ha_features(pl, bd, CFG)
    g = df.iloc[210:250]
    assert g["ha_near_margin"].isna().all() and g["ha_bmot_speed"].isna().all()
    assert (g["ha_last_margin"] > 5).all()                            # last reliable value carried
    assert np.isclose(df["ha_last_margin_age_s"].iloc[230], (230 - 199) / FPS, atol=1e-3)
    assert np.isclose(df["ha_ball_since_obs_s"].iloc[230], (230 - 199) / FPS, atol=1e-3)
    assert g["ha_ctx_speed_h"].notna().all()                          # team context still there


def test_ha_refuses_mismatched_frames():
    from bh.features_ha import build_ha_features
    pl, bd = dribble()
    bad = BallData("u", bd.frame + 1, bd.period, bd.xy, bd.status, bd.conf)
    try:
        build_ha_features(pl, bad, CFG)
        raise AssertionError("accepted mismatched frames")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:
            import traceback
            failed += 1
            print("FAIL", name, "-", repr(exc))
            traceback.print_exc(limit=2)
    print("{} passed, {} failed".format(len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
