"""One call that runs everything on a Tracking and returns what the scripts write out."""

import json
import os

from . import __version__
from .clean import FLAG_NAMES, STATUS_NAMES, check_invariants, clean
from .quality import quality_table
from .speed_check import resolve_speed_units

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_THRESHOLDS = os.path.join(ROOT, "config", "cleaner_thresholds.json")


def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return _strip_comments(json.load(handle))


def load_thresholds(path=None):
    return load_json(path or DEFAULT_THRESHOLDS)


def run(tr, thresholds):
    """Returns (result arrays, quality DataFrame, report dict)."""
    before, after, action = resolve_speed_units(tr)
    speed_units = {"verdict": after["verdict"], "before_correction": before,
                   "after_correction": after, "action": action,
                   "declared_unit": before.get("declared_unit")}
    result, summary = clean(tr, thresholds)
    quality = quality_table(tr, result)
    problems = check_invariants(tr, result)
    report = {
        "version": __version__,
        "source": tr.meta,
        "frames": tr.n_frames, "slots": tr.n_slots, "fps": tr.fps, "pitch_m": list(tr.pitch),
        "origin": tr.origin,
        "goalkeeper_slots_known": int((tr.goalkeeper == 1).sum()),
        "speed_units_check": speed_units,
        "summary": summary,
        "invariant_problems": problems,
        "thresholds": thresholds,
        "status_codes": {str(k): v for k, v in STATUS_NAMES.items()},
        "flag_bits": {str(k): v for k, v in FLAG_NAMES.items()},
    }
    return result, quality, report


def write_report(report, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return path
