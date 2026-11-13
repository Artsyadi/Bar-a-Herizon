"""Read Metrica FIFA/EPTS tracking (metadata .xml + raw .txt) for any match.

Uses the metadata's DataFormatSpecification blocks, so it handles:
  * any number of values per player (x,y or x,y,speed) and per ball (x,y / x,y,speed / x,y,z)
  * a player list that changes during the match (substitutions): each block maps line positions
    to player channels; a slot is one metadata Player
  * declared units: positions 'normalized' are scaled by the pitch size; speeds in 'kph' are
    converted to m/s (BAR-ATH declares speed_kph; DATA.md's m/s is after kloppy's conversion)
  * goalkeepers from the Player's position_type parameter (Goalkeeper -> 1, Unknown -> -1)

Raw line format:  <frame_id>:<group>;<group>;...:<ball>   with NaN for missing values.
"""

import warnings
import xml.etree.ElementTree as ET

import numpy as np

from .tracking import Tracking, to_origin

KPH = {"kph", "km/h", "kmh", "kilometers per hour"}
MPS = {"m/s", "mps", "ms", "meters per second", "metres per second"}


def _speed_factor(unit):
    u = (unit or "").strip().lower()
    if u in KPH:
        return 1.0 / 3.6
    if u in MPS:
        return 1.0
    return None


def read_metadata(meta_path):
    root = ET.parse(meta_path).getroot()
    fps = float(root.findtext(".//GlobalConfig/FrameRate") or "nan")
    width = root.findtext(".//FieldSize/Width")
    height = root.findtext(".//FieldSize/Height")
    field = (float(width), float(height)) if width and height else None

    params = {}
    for p in root.iter("ProviderParameter"):
        name, value = p.findtext("Name"), p.findtext("Value")
        if name and name.endswith(("_start", "_end")):
            params[name] = int(value) if value and value.strip().lstrip("-").isdigit() else None

    teams = [(t.get("id"), (t.findtext("Name") or "").strip()) for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]

    players = {}
    for p in root.iter("Player"):
        position = None
        for pp in p.iter("ProviderParameter"):
            if pp.findtext("Name") == "position_type":
                position = (pp.findtext("Value") or "").strip().lower()
        gk = 1 if position == "goalkeeper" else (-1 if position in (None, "", "unknown") else 0)
        players[p.get("id")] = {"team": "H" if p.get("teamId") == home_id else "A", "gk": gk}

    channels = {c.get("id"): ((c.findtext("Name") or ""), (c.findtext("Unit") or ""))
                for c in root.iter("Channel")}
    player_channels = {pc.get("id"): (pc.get("playerId"), pc.get("channelId"))
                       for pc in root.iter("PlayerChannel")}

    specs = []
    for spec in root.iter("DataFormatSpecification"):
        groups = []
        ball = [b.get("channelId") for b in spec.iter("BallChannelRef")]
        for reg in spec.iter("SplitRegister"):
            refs = [r.get("playerChannelId") for r in reg.findall("PlayerChannelRef")]
            if refs:
                groups.append([player_channels[r] for r in refs])    # [(player_id, kind), ...]
        specs.append({"start": int(spec.get("startFrame")), "end": int(spec.get("endFrame")),
                      "groups": groups, "ball": ball})
    specs.sort(key=lambda s: s["start"])
    if not specs:
        raise ValueError("{}: no DataFormatSpecification".format(meta_path))
    return {"fps": fps, "field": field, "params": params, "teams": teams, "players": players,
            "channels": channels, "specs": specs}


def _parse_lines(lines, n_fields):
    text = ",".join(ln.strip() for ln in lines).replace(":", ",").replace(";", ",")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = np.fromstring(text, sep=",", dtype=np.float64)
    if values.size == len(lines) * n_fields:
        return values.reshape(len(lines), n_fields), 0
    rows, skipped = [], 0
    for ln in lines:
        parts = ln.strip().replace(":", ",").replace(";", ",").split(",")
        try:
            if len(parts) != n_fields:
                raise ValueError
            rows.append([float(p) for p in parts])
        except ValueError:
            skipped += 1
    return np.array(rows, dtype=np.float64).reshape(-1, n_fields), skipped


def read_epts(meta_path, raw_path, pitch=None, fps=None, frame_id_offset=1, n_frames=None,
              periods=None, origin="centre", flip_y=False, chunk_lines=20000):
    """Load an EPTS pair into a Tracking.

    pitch:   (length, width) in metres; default = metadata FieldSize
    periods: list of (first, last) output frames, one per period; default = metadata halves,
             or one period covering everything if the metadata has no second half
    flip_y:  mirror y (Metrica's normalised y grows downwards from the top touchline)
    """
    md = read_metadata(meta_path)
    fps = float(fps or md["fps"])
    pitch = tuple(pitch or md["field"] or (105.0, 68.0))
    L, W = pitch
    speed_unit = md["channels"].get("s", ("", ""))[1]
    factor = _speed_factor(speed_unit)

    slot_ids = []
    for spec in md["specs"]:
        for group in spec["groups"]:
            pid = group[0][0]
            if pid not in slot_ids:
                slot_ids.append(pid)
    slot_index = {pid: k for k, pid in enumerate(slot_ids)}
    S = len(slot_ids)
    if n_frames is None:
        n_frames = max(s["end"] for s in md["specs"]) - frame_id_offset + 1

    players = np.full((n_frames, S, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, S), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_third = np.full(n_frames, np.nan, dtype=np.float32)

    layouts = []
    for spec in md["specs"]:
        col = 1
        assign = []                                  # (column, slot, kind)
        for group in spec["groups"]:
            for pid, kind in group:
                assign.append((col, slot_index[pid], kind))
                col += 1
        ball_cols = {kind: col + k for k, kind in enumerate(spec["ball"])}
        layouts.append({"n_fields": col + len(spec["ball"]), "assign": assign, "ball": ball_cols})
    starts = np.array([s["start"] for s in md["specs"]])
    ends = np.array([s["end"] for s in md["specs"]])

    skipped = 0
    with open(raw_path, encoding="utf-8") as handle:
        while True:
            lines = [ln for ln in (handle.readline() for _ in range(chunk_lines)) if ln.strip()]
            if not lines:
                break
            ids = np.array([int(ln[:ln.index(":")]) for ln in lines])
            which = np.searchsorted(starts, ids, side="right") - 1
            ok = (which >= 0) & (ids <= ends[np.clip(which, 0, len(ends) - 1)])
            for k in np.unique(which[ok]):
                sel = np.flatnonzero(ok & (which == k))
                lay = layouts[k]
                arr, bad = _parse_lines([lines[i] for i in sel], lay["n_fields"])
                skipped += bad
                f = arr[:, 0].astype(np.int64) - frame_id_offset
                keep = (f >= 0) & (f < n_frames)
                f, arr = f[keep], arr[keep]
                for col, slot, kind in lay["assign"]:
                    if kind == "x":
                        players[f, slot, 0] = arr[:, col] * L
                    elif kind == "y":
                        players[f, slot, 1] = arr[:, col] * W
                    elif kind == "s":
                        speed[f, slot] = arr[:, col]
                b = lay["ball"]
                if "x" in b:
                    ball[f, 0] = arr[:, b["x"]] * L
                if "y" in b:
                    ball[f, 1] = arr[:, b["y"]] * W
                third = "s" if "s" in b else ("z" if "z" in b else None)
                if third:
                    ball_third[f] = arr[:, b[third]]

    ball_speed = ball_z = None
    if "s" in md["channels"] and factor is not None:
        speed *= factor
    if any("s" in s["ball"] for s in md["specs"]):
        ball_speed = ball_third * factor if factor is not None else ball_third
    elif any("z" in s["ball"] for s in md["specs"]):
        ball_z = ball_third

    if flip_y:
        players[:, :, 1] = W - players[:, :, 1]
        ball[:, 1] = W - ball[:, 1]
    players = to_origin(players, pitch, "corner", origin)
    ball = to_origin(ball, pitch, "corner", origin)

    period = np.zeros(n_frames, dtype=np.int16)
    if periods:
        for k, (first, last) in enumerate(periods, start=1):
            period[max(0, first):min(n_frames, last + 1)] = k
    else:
        p = md["params"]
        halves = [(p.get("first_half_start"), p.get("first_half_end")),
                  (p.get("second_half_start"), p.get("second_half_end")),
                  (p.get("first_extra_half_start"), p.get("first_extra_half_end")),
                  (p.get("second_extra_half_start"), p.get("second_extra_half_end"))]
        halves = [(a, b) for a, b in halves if a is not None and b is not None]
        if len(halves) >= 2:
            for k, (a, b) in enumerate(halves, start=1):
                period[max(0, a - frame_id_offset):min(n_frames, b - frame_id_offset + 1)] = k
        else:
            period[:] = 1

    team = np.array([md["players"].get(pid, {"team": "H"})["team"] for pid in slot_ids])
    gk = np.array([md["players"].get(pid, {"gk": -1})["gk"] for pid in slot_ids], dtype=np.int8)
    tr = Tracking(frame=np.arange(n_frames, dtype=np.int64), period=period, ball=ball,
                  players=players, speed=speed, slot_ids=np.array(slot_ids), team=team,
                  goalkeeper=gk, fps=fps, pitch=pitch, origin=origin, ball_z=ball_z,
                  ball_speed=ball_speed, present=np.isfinite(players[:, :, 0]),
                  meta={"source": "epts", "speed_unit_declared": speed_unit or None,
                        "speed_converted_to_mps": factor is not None and "s" in md["channels"],
                        "lines_skipped": int(skipped), "flip_y": bool(flip_y),
                        "teams": md["teams"], "n_format_blocks": len(md["specs"])})
    return tr.validate()
