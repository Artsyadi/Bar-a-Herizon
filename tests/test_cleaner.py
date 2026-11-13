"""Tests on made-up data only - no match data needed.

    python tests/test_cleaner.py      (or: pytest tests/)
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atd_cleaner import clean as C  # noqa: E402
from atd_cleaner.evaluate import evaluate  # noqa: E402
from atd_cleaner.io_epts import read_epts  # noqa: E402
from atd_cleaner.io_table import frames_table, objects_table, read_table  # noqa: E402
from atd_cleaner.pipeline import load_thresholds, run  # noqa: E402
from atd_cleaner.quality import quality_table  # noqa: E402
from atd_cleaner.speed_check import check_speed_units  # noqa: E402
from atd_cleaner.tracking import Tracking  # noqa: E402

FPS = 25.0
TH = load_thresholds()


def line(n, v=(5.0, 0.0), start=(-20.0, 0.0)):
    t = np.arange(n) / FPS
    return np.stack([start[0] + v[0] * t, start[1] + v[1] * t], 1).astype(np.float32)


def make_tracking(n=300, players=None, team=None, ball=None, period=None, speed=None):
    if players is None:
        players = np.stack([line(n, (3, 0), (-10, 5)), line(n, (0, 3), (10, -5))], 1)
    S = players.shape[1]
    team = np.array(team if team is not None else ["H", "A"][:S] + ["H"] * max(0, S - 2))
    return Tracking(frame=np.arange(n), period=np.ones(n, np.int16) if period is None else period,
                    ball=line(n) if ball is None else ball, players=players.astype(np.float32),
                    speed=np.full((n, S), np.nan, np.float32) if speed is None else speed,
                    slot_ids=np.arange(S), team=team, goalkeeper=np.full(S, -1, np.int8),
                    fps=FPS, pitch=(105.0, 68.0)).validate()


# -- cleaning -------------------------------------------------------------------------------------

def test_spike_removed_and_gap_refilled_on_line():
    b = line(300)
    b[150] += [30, 0]
    res, summ = C.clean(make_tracking(ball=b), TH)
    assert summ["ball_jump_points_removed"] == 1
    assert res["ball_flags"][150] & C.FLAG_JUMP_REMOVED
    assert res["ball_status"][150] == C.ESTIMATED            # 1-frame hole refilled on the line
    assert np.allclose(res["ball"][150], line(300)[150], atol=1e-3)


def test_long_jump_flagged_not_removed():
    b = line(300)
    b[200:] += [25, 0]
    res, summ = C.clean(make_tracking(ball=b), TH)
    assert summ["ball_jump_points_removed"] == 0
    assert res["ball_flags"][200] & C.FLAG_TRACK_BREAK
    assert res["ball_conf"][200] < 1


def test_short_gap_filled_long_gap_kept():
    b = line(300)
    b[50:58] = np.nan
    b[150:170] = np.nan
    res, summ = C.clean(make_tracking(ball=b), TH)
    assert summ["ball_frames_interpolated"] == 8
    assert (res["ball_status"][50:58] == C.ESTIMATED).all()
    assert (res["ball_status"][150:170] == C.UNAVAILABLE).all()


def test_no_fill_across_period_or_outside_play():
    b = line(300)
    b[95:105] = np.nan
    b[200:205] = np.nan
    period = np.ones(300, np.int16)
    period[100:] = 2
    period[198:210] = 0
    res, summ = C.clean(make_tracking(ball=b, period=period), TH)
    assert summ["ball_frames_interpolated"] == 0


def test_no_fill_when_velocity_flips():
    a = line(60, (10, 0))
    b2 = line(60, (-10, 0), tuple(a[-1]))
    b = np.concatenate([a, b2[1:]])
    b[55:63] = np.nan
    n = len(b)
    res, summ = C.clean(make_tracking(n=n, ball=b, players=np.full((n, 2, 2), 40.0)), TH)
    assert summ["ball_frames_interpolated"] == 0


def test_no_fill_when_nearest_team_changes():
    n = 200
    b = line(n, (2, 0))
    b[100:106] = np.nan
    p = np.full((n, 2, 2), 40.0, np.float32)
    p[:100, 0] = b[:100] + [0.5, 0]                            # home player with the ball before
    p[106:, 1] = b[106:] + [0.5, 0]                            # away player with it after
    res, summ = C.clean(make_tracking(n=n, ball=b, players=p), TH)
    assert summ["ball_frames_interpolated"] == 0
    th = {**TH, "ball": {**TH["ball"], "block_if_nearest_team_changes": False}}
    _, summ2 = C.clean(make_tracking(n=n, ball=b, players=p), th)
    assert summ2["ball_frames_interpolated"] == 6


def test_player_spike_removed_and_duplicates_merged():
    n = 100
    p = np.full((n, 4, 2), np.nan, np.float32)
    p[:, 0] = line(n, (3, 0))
    p[:, 1] = p[:, 0] + [0.1, 0]                               # duplicate of slot 0
    p[:, 2] = line(n, (0, 3), (20, 0))
    p[:, 3] = p[:, 0] + [0.1, 0]                               # opponent on the same spot
    p[40, 2] += [6, 0]                                         # 150 m/s spike
    tr = make_tracking(n=n, players=p, team=["H", "H", "H", "A"])
    th = {**TH, "duplicates": {**TH["duplicates"], "max_players_per_team": 2}}   # 3 H tracks > 2
    res, summ = C.clean(tr, th)
    # frame 40: the spike on slot 2 is removed first, so H has only 2 tracks there -> not over the
    # limit -> that one frame is not merged
    assert summ["duplicate_slot_frames_removed"] == n - 1
    assert np.isfinite(res["players"][40, 1, 0])
    assert np.isnan(np.delete(res["players"][:, 1, 0], 40)).all()
    assert np.isfinite(res["players"][:, 3, 0]).all()
    assert np.isnan(res["players"][40, 2, 0]) and res["player_flags"][40, 2] & C.FLAG_SPEED_REMOVED
    assert C.check_invariants(tr, res) == []


def test_close_teammates_kept_when_team_not_over_limit():
    """A wall / celebration: teammates side by side, team count within 11 -> no merge."""
    n = 100
    p = np.stack([line(n, (0, 0), (30, 0)), line(n, (0, 0), (30.3, 0)), line(n, (1, 0), (0, 10))], 1)
    tr = make_tracking(n=n, players=p, team=["H", "H", "A"])
    res, summ = C.clean(tr, TH)
    assert summ["duplicate_slot_frames_removed"] == 0
    assert np.isfinite(res["players"][:, :2, 0]).all()


def test_sprint_jitter_not_removed_but_real_jump_is():
    n = 100
    p = np.stack([line(n, (8, 0)), line(n, (0, 3), (20, 0))], 1)
    p[50, 0] += [0.3, 0]                                       # jitter: one ~15 m/s step, 0.6 m
    p[70, 1] += [5, 0]                                         # identity-style jump: 5 m
    res, summ = C.clean(make_tracking(n=n, players=p), TH)
    assert np.isfinite(res["players"][50, 0, 0])
    assert np.isnan(res["players"][70, 1, 0])


# -- position-based speed (v0.2.1) ------------------------------------------------------------------

def test_derived_speed_exact_on_straight_line_and_nan_at_edges():
    n = 60
    p = np.stack([line(n, (3, 4)), line(n, (6, 8), (10, 0))], 1)      # 5 m/s and 10 m/s
    tr = make_tracking(n=n, players=p)
    res, summ = C.clean(tr, TH)
    sd = res["player_speed_derived"]
    w = TH["players"]["speed_window_frames"]
    assert np.allclose(sd[w:n - w, 0], 5.0, atol=1e-3) and np.allclose(sd[w:n - w, 1], 10.0, atol=1e-3)
    assert np.isnan(sd[:w]).all() and np.isnan(sd[n - w:]).all()     # no window -> NaN, not 0
    assert C.check_invariants(tr, res) == []


def test_derived_speed_nan_around_gaps_and_period_change():
    n = 80
    p = np.stack([line(n, (3, 0)), line(n, (0, 3), (20, 0))], 1)
    p[30:33, 0] = np.nan
    period = np.ones(n, np.int16)
    period[50:] = 2
    res, _ = C.clean(make_tracking(n=n, players=p, period=period), TH)
    sd = res["player_speed_derived"]
    w = TH["players"]["speed_window_frames"]
    assert np.isnan(sd[30 - w:33 + w, 0]).all()                       # every window touching the gap
    assert np.isfinite(sd[30 - w - 1, 0]) and np.isfinite(sd[33 + w, 0])
    assert np.isnan(sd[50 - w:50 + w, 1]).all()                       # windows crossing the half
    assert not np.any(sd[np.isfinite(sd)] == 0)                       # constant motion, never 0


def test_derived_speed_guard_and_output_column():
    from atd_cleaner.clean import derived_speed
    n = 40
    p = np.stack([line(n, (20, 0))], 1)                               # 20 m/s, above the 12.5 guard
    sd, guarded = derived_speed(p, FPS, np.ones(n, np.int16), 2, 12.5)
    assert np.isnan(sd).all() and guarded == n - 4
    frames = pd.DataFrame({"frame": np.arange(n), "period": 1, "ball_x": 0.0, "ball_y": 0.0})
    objects = pd.DataFrame({"frame": np.arange(n), "slot": 1, "x": np.arange(n) * 0.2,
                            "y": 0.0, "team": "H"})                   # 5 m/s
    tr = read_table(frames, objects)
    res, _, _ = run(tr, TH)
    ot = objects_table(tr, res)
    assert "speed_derived_ms" in ot and np.allclose(ot.speed_derived_ms.iloc[5:-5], 5.0, atol=1e-3)
    assert ot.speed_ms.isna().all()                                   # source speed stays separate


# -- quality / speed --------------------------------------------------------------------------------

def test_quality_counts_and_speed_never_zero():
    n = 50
    p = np.stack([line(n), line(n, (1, 1)), line(n, (0, 2))], 1)
    p[10:20, 1] = np.nan
    tr = make_tracking(n=n, players=p, team=["H", "H", "A"])
    res, _ = C.clean(tr, TH)
    q = quality_table(tr, res)
    assert q["home_valid_n"].iloc[5] == 2 and q["home_valid_n"].iloc[15] == 1
    assert q["home_speed_valid_n"].iloc[20] == 1               # slot 1 back, but no previous frame
    assert q["home_speed_valid_n"].iloc[0] == 0                # first frame: no speed, not 0 m/s
    assert np.isnan(tr.speed).all()                            # missing speed stays NaN


def test_speed_unit_check():
    n = 400
    p = np.stack([line(n, (5, 0)), line(n, (0, 4), (0, -20))], 1)
    true = np.tile([5.0, 4.0], (n, 1)).astype(np.float32)
    assert check_speed_units(make_tracking(n=n, players=p, speed=true))["verdict"] == "consistent_mps"
    assert check_speed_units(make_tracking(n=n, players=p, speed=true * 3.6))["verdict"] == "looks_like_kph"


def test_speed_units_resolved_from_positions_not_label():
    from atd_cleaner.speed_check import resolve_speed_units
    n = 400
    p = np.stack([line(n, (5, 0)), line(n, (0, 4), (0, -20))], 1)
    true = np.tile([5.0, 4.0], (n, 1)).astype(np.float32)
    wrongly_divided = make_tracking(n=n, players=p, speed=true / 3.6)   # label said kph, values m/s
    _, after, action = resolve_speed_units(wrongly_divided)
    assert after["verdict"] == "consistent_mps" and "3.6" in action
    assert np.allclose(wrongly_divided.speed, true, atol=1e-4)
    in_kph = make_tracking(n=n, players=p, speed=true * 3.6)
    resolve_speed_units(in_kph)
    assert np.allclose(in_kph.speed, true, atol=1e-4)


# -- loaders ----------------------------------------------------------------------------------------

EPTS_META = """<?xml version="1.0"?><main><Metadata><GlobalConfig><FrameRate>25</FrameRate>
<ProviderGlobalParameters>
 <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
 <ProviderParameter><Name>first_half_end</Name><Value>2</Value></ProviderParameter>
 <ProviderParameter><Name>second_half_start</Name><Value>3</Value></ProviderParameter>
 <ProviderParameter><Name>second_half_end</Name><Value>4</Value></ProviderParameter>
</ProviderGlobalParameters></GlobalConfig>
<Sessions><Session><MatchParameters><FieldSize><Width>100</Width><Height>50</Height></FieldSize></MatchParameters></Session></Sessions>
<Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players>
 <Player id="P1" teamId="T1"><ProviderPlayerParameters><ProviderParameter><Name>position_type</Name><Value>Goalkeeper</Value></ProviderParameter></ProviderPlayerParameters></Player>
 <Player id="P2" teamId="T2"><ProviderPlayerParameters><ProviderParameter><Name>position_type</Name><Value>Unknown</Value></ProviderParameter></ProviderPlayerParameters></Player>
 <Player id="P3" teamId="T2"/>
</Players>
<Devices><Device><Sensors><Sensor><Channels>
 <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
 <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
 <Channel id="s"><Name>speed_kph</Name><Unit>kph</Unit></Channel>
</Channels><PlayerChannels>
 <PlayerChannel id="p1_x" playerId="P1" channelId="x"/><PlayerChannel id="p1_y" playerId="P1" channelId="y"/><PlayerChannel id="p1_s" playerId="P1" channelId="s"/>
 <PlayerChannel id="p2_x" playerId="P2" channelId="x"/><PlayerChannel id="p2_y" playerId="P2" channelId="y"/><PlayerChannel id="p2_s" playerId="P2" channelId="s"/>
 <PlayerChannel id="p3_x" playerId="P3" channelId="x"/><PlayerChannel id="p3_y" playerId="P3" channelId="y"/><PlayerChannel id="p3_s" playerId="P3" channelId="s"/>
</PlayerChannels></Sensor></Sensors></Device></Devices>
<DataFormatSpecifications>
 <DataFormatSpecification startFrame="1" endFrame="2" separator=":"><StringRegister name="frameCount"/>
  <SplitRegister separator=";">
   <SplitRegister separator=","><PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/><PlayerChannelRef playerChannelId="p1_s"/></SplitRegister>
   <SplitRegister separator=","><PlayerChannelRef playerChannelId="p2_x"/><PlayerChannelRef playerChannelId="p2_y"/><PlayerChannelRef playerChannelId="p2_s"/></SplitRegister>
  </SplitRegister>
  <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="s"/></SplitRegister>
 </DataFormatSpecification>
 <DataFormatSpecification startFrame="3" endFrame="4" separator=":"><StringRegister name="frameCount"/>
  <SplitRegister separator=";">
   <SplitRegister separator=","><PlayerChannelRef playerChannelId="p1_x"/><PlayerChannelRef playerChannelId="p1_y"/><PlayerChannelRef playerChannelId="p1_s"/></SplitRegister>
   <SplitRegister separator=","><PlayerChannelRef playerChannelId="p3_x"/><PlayerChannelRef playerChannelId="p3_y"/><PlayerChannelRef playerChannelId="p3_s"/></SplitRegister>
  </SplitRegister>
  <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="s"/></SplitRegister>
 </DataFormatSpecification>
</DataFormatSpecifications></Metadata></main>"""

EPTS_RAW = ("1:0.5,0.5,36;0.1,0.2,18:0.25,0.75,72\n"
            "2:0.5,0.5,36;NaN,NaN,NaN:NaN,NaN,NaN\n"
            "3:0.6,0.5,36;0.9,0.9,7.2:0.5,0.5,0\n"
            "4:0.6,0.5,36;0.9,0.9,7.2:0.5,0.5,0\n")


def test_epts_blocks_substitution_units_goalkeeper():
    with tempfile.TemporaryDirectory() as d:
        mp, rp = os.path.join(d, "m.xml"), os.path.join(d, "r.txt")
        open(mp, "w").write(EPTS_META)
        open(rp, "w").write(EPTS_RAW)
        tr = read_epts(mp, rp, origin="corner")
        tc = read_epts(mp, rp, origin="centre")
    assert list(tr.slot_ids) == ["P1", "P2", "P3"] and list(tr.team) == ["H", "A", "A"]
    assert list(tr.goalkeeper) == [1, -1, -1]
    assert np.allclose(tr.players[0, 1], [10.0, 10.0])          # 0.1 x 100, 0.2 x 50
    assert np.isnan(tr.players[2, 1]).all() and np.allclose(tr.players[2, 2], [90.0, 45.0])
    assert np.isnan(tr.players[0, 2]).all()                     # P3 not on the pitch yet
    assert np.isclose(tr.speed[0, 0], 10.0) and np.isclose(tr.speed[0, 1], 5.0)   # kph -> m/s
    assert np.isclose(tr.ball_speed[0], 20.0) and tr.ball_z is None
    assert list(tr.period) == [1, 1, 2, 2]
    assert np.allclose(tc.players[0, 0], [0.0, 0.0])            # centre spot
    assert tr.meta["speed_converted_to_mps"] is True


def test_table_roundtrip_keeps_removed_rows():
    n = 40
    frames = pd.DataFrame({"frame": np.arange(n), "period": 1, "ball_x": np.linspace(-5, 5, n),
                           "ball_y": 0.0, "ball_visible": True, "t_s": np.arange(n) / 25})
    frames.loc[10:14, "ball_visible"] = False
    rows = []
    for f in range(n):
        rows.append({"frame": f, "slot": 3, "x": -10 + 0.1 * f, "y": 0.0, "team": "H", "speed_ms": 2.5})
        rows.append({"frame": f, "slot": 8, "x": 10.0, "y": 1.0, "team": "A", "speed_ms": 0.0})
    objects = pd.DataFrame(rows)
    objects.loc[(objects.slot == 3) & (objects.frame == 20), "x"] += 8.0       # spike
    tr = read_table(frames, objects)
    assert list(tr.slot_ids) == [3, 8] and np.isnan(tr.ball[10:15]).all()
    res, q, rep = run(tr, TH)
    ft, ot = frames_table(tr, res, q), objects_table(tr, res)
    assert len(ot) == len(objects)                                              # nothing dropped
    spike = ot[(ot.slot == 3) & (ot.frame == 20)].iloc[0]
    assert np.isnan(spike.x) and spike.player_status == C.UNAVAILABLE
    assert spike.player_flags & C.FLAG_SPEED_REMOVED
    assert (ft.loc[10:14, "ball_status"] == C.ESTIMATED).all() and "t_s" in ft
    assert rep["invariant_problems"] == []


# -- evaluation -------------------------------------------------------------------------------------

def test_evaluation_counts_injected_errors():
    n = 400
    orig = make_tracking(n=n, players=np.stack([line(n, (3, 0)), line(n, (0, 3), (20, 0))], 1))
    deg_ball = orig.ball.copy()
    deg_ball[100] += [30, 0]                                    # 1 corrupted point
    deg_ball[200:206] = np.nan                                  # 6-frame gap
    dp = np.concatenate([orig.players, orig.players[:, :1] + [0.1, 0]], 1)    # injected duplicate
    deg = Tracking(frame=orig.frame, period=orig.period, ball=deg_ball, players=dp,
                   speed=np.full((n, 3), np.nan, np.float32), slot_ids=np.array([0, 1, 99]),
                   team=np.array(["H", "A", "H"]), goalkeeper=np.full(3, -1, np.int8),
                   fps=FPS, pitch=(105.0, 68.0)).validate()
    th = {**TH, "duplicates": {**TH["duplicates"], "max_players_per_team": 1}}   # 2 H tracks > 1
    res, _, _ = run(deg, th)
    rep = evaluate(orig, deg, res, max_gap_frames=12)
    assert rep["ball"]["corrupted_points"] == 1 and rep["ball"]["jump_recall"] == 1.0
    assert rep["ball"]["frames_filled"] == 7                    # 6 gap + the removed spike frame
    assert rep["ball"]["fill_median_error_m"] < 1e-3
    assert rep["players"]["injected_duplicate_slots"] == 1 and rep["players"]["duplicate_recall"] == 1.0
    assert rep["rmse_ball_after_m"] < rep["rmse_ball_before_m"]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, "-", repr(exc))
    print("{} passed, {} failed".format(len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
