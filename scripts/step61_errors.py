"""Step 6.1: error categories (merged Rulebook C8.1) on the 30 dev units.

Uses the Phase 4 out-of-fold probabilities + the chosen Phase 5 segmenter. For each category: a
count, two examples (external matches only) and, where it can be defined, the spellF1 gained if
that category were fixed perfectly with the true labels ("oracle gain"), which shows where the
points are. Nothing here changes a model.

    python scripts/step61_errors.py

Output: runs/phase6/errors_round0.csv (counts) + printed table and examples.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config  # noqa: E402
from bh.models import PHASE4, models_config  # noqa: E402
from bh.scoring import score  # noqa: E402
from bh.segmenter import segment  # noqa: E402

FPS = 25
OUT = os.path.join(config.PROJECT, "runs", "phase6")


def runs(lab, per):
    """(start, end, label) runs inside halves (period > 0)."""
    key = np.char.add(lab.astype(str), per.astype(str))
    cut = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1, len(lab)]
    return [(int(s), int(e), lab[s]) for s, e in zip(cut[:-1], cut[1:]) if per[s] > 0]


def best_iou(a, b, t, spells):
    best = 0.0
    for s, e, u in spells:
        if u != t or e <= a or s >= b:
            continue
        best = max(best, (min(e, b) - max(s, a)) / (max(e, b) - min(s, a)))
    return best


def stamp(per, frame0, s, e):
    """'H1 12:34-12:36' style label (period, minutes:seconds from the start of that half)."""
    t0 = frame0[per[s]]
    f = lambda i: "{:02d}:{:02d}".format(int((i - t0) / FPS // 60), int((i - t0) / FPS % 60))
    return "H{} {}-{}".format(per[s], f(s), f(e))


def analyse(gt, lab, per, ball):
    out, ex = {}, {}
    frame0 = {p: int(np.flatnonzero(per == p)[0]) for p in np.unique(per) if p > 0}
    G, P = runs(gt, per), runs(lab, per)
    GP = [r for r in G if r[2] in ("H", "A")]
    PP = [r for r in P if r[2] in ("H", "A")]
    live_gt = np.isin(gt, ["H", "A"])

    # 1 reversed restart ownership: first 1 s after a true restart, majority predicted team is wrong
    n, bad, e1 = 0, 0, []
    for (s0, e0, l0), (s1, e1_, l1) in zip(G[:-1], G[1:]):
        if l0 == "D" and l1 in ("H", "A") and per[s1] == per[s0]:
            w = lab[s1:min(s1 + FPS, e1_)]
            w = w[np.isin(w, ["H", "A"])]
            if len(w) == 0:
                continue
            n += 1
            if (w == ("A" if l1 == "H" else "H")).mean() > 0.5:
                bad += 1
                e1.append(stamp(per, frame0, s1, min(s1 + FPS, e1_)))
    out["1 reversed restart ownership"], ex["1 reversed restart ownership"] = (bad, n), e1

    # 2 false D during live play: predicted D runs that are >= 80% true live
    pd_runs = [(s, e) for s, e, l in P if l == "D"]
    fd = [(s, e) for s, e in pd_runs if live_gt[s:e].mean() >= 0.8]
    out["2 false D in live play"] = (len(fd), len(pd_runs))
    ex["2 false D in live play"] = [stamp(per, frame0, s, e) for s, e in fd]

    # 3 missed stoppages (mostly quick restarts): true D runs with < 50% predicted D
    gd = [(s, e) for s, e, l in G if l == "D"]
    miss = [(s, e) for s, e in gd if (lab[s:e] == "D").mean() < 0.5]
    out["3 missed stoppages"] = (len(miss), len(gd))
    ex["3 missed stoppages"] = ["{} ({:.0f} s)".format(stamp(per, frame0, s, e), (e - s) / FPS) for s, e in miss]

    # 4 flicker: predicted possessions < 1 s that match no true possession (IoU < 0.5)
    fl = [(s, e) for s, e, t in PP if e - s < FPS and best_iou(s, e, t, GP) < 0.5]
    out["4 flicker (false short possessions)"] = (len(fl), len(PP))
    ex["4 flicker (false short possessions)"] = [stamp(per, frame0, s, e) for s, e in fl]

    # 5 short spells smoothed away: true possessions < 2 s not found (IoU < 0.5)
    sh = [(s, e) for s, e, t in GP if e - s < 2 * FPS and best_iou(s, e, t, PP) < 0.5]
    out["5 short possessions missed"] = (len(sh), len(GP))
    ex["5 short possessions missed"] = [stamp(per, frame0, s, e) for s, e in sh]

    # 6 D ending too late (and starting late): matched true / predicted D runs
    late_end, starts, ends = [], [], []
    for s, e in gd:
        ov = [(ps, pe) for ps, pe in pd_runs if pe > s and ps < e]
        if not ov:
            continue
        ps, pe = ov[0][0], ov[-1][1]
        starts.append((ps - s) / FPS)
        ends.append((pe - e) / FPS)
        if pe - e > FPS:
            late_end.append(stamp(per, frame0, e, pe))
    out["6 D ending > 1 s late"] = (len(late_end), len(ends))
    ex["6 D ending > 1 s late"] = late_end

    # 7 false transfers while the ball is unavailable: predicted H<->A switches with no true switch +-1 s
    sw_p = [i for i in range(1, len(lab)) if lab[i] in ("H", "A") and lab[i - 1] in ("H", "A")
            and lab[i] != lab[i - 1] and per[i] == per[i - 1]]
    sw_g = np.array([i for i in range(1, len(gt)) if gt[i] in ("H", "A") and gt[i - 1] in ("H", "A")
                     and gt[i] != gt[i - 1] and per[i] == per[i - 1]])
    blind = [i for i in sw_p if ball[i] == 2]
    false_blind = [i for i in blind if len(sw_g) == 0 or np.abs(sw_g - i).min() > FPS]
    out["7 false transfers, ball unavailable"] = (len(false_blind), len(blind))
    ex["7 false transfers, ball unavailable"] = [stamp(per, frame0, i, i + 1) for i in false_blind]

    # timing of correct switches (pred minus true, within +-2 s)
    lags = []
    for i in sw_p:
        if len(sw_g):
            j = sw_g[np.abs(sw_g - i).argmin()]
            if abs(i - j) <= 2 * FPS and lab[i] == gt[j]:
                lags.append((i - j) / FPS)
    return out, ex, {"d_start_lag": starts, "d_end_lag": ends, "switch_lag": lags}


def oracles(gt, lab, p_d, p_h, per, cost_d, cost_h, clip):
    """Our labels with one category fixed using the truth."""
    o = {}
    # perfect dead/live, our team decisions inside the true live intervals
    o["perfect dead/live"] = segment(p_d, p_h, per, cost_d, cost_h, clip, dead=~np.isin(gt, ["H", "A"]))
    # restart ownership: predicted possession that covers the first live second of a restart gets the true team
    x = lab.copy()
    G = runs(gt, per)
    P = runs(lab, per)
    for (s0, _, l0), (s1, _, l1) in zip(G[:-1], G[1:]):
        if l0 == "D" and l1 in ("H", "A"):
            k = min(s1 + FPS // 2, len(lab) - 1)
            for ps, pe, pl in P:
                if ps <= k < pe and pl in ("H", "A") and pl != l1:
                    x[ps:pe] = l1
    o["restart ownership right"] = x
    # short true possessions (< 2 s) inserted
    x = lab.copy()
    for s, e, t in G:
        if t in ("H", "A") and e - s < 2 * FPS:
            x[s:e] = t
    o["short possessions found (< 2 s)"] = x
    # false flicker removed (merged into what precedes it)
    x = lab.copy()
    GP = [r for r in G if r[2] in ("H", "A")]
    for s, e, t in P:
        if t in ("H", "A") and e - s < FPS and s > 0 and best_iou(s, e, t, GP) < 0.5:
            x[s:e] = x[s - 1]
    o["false flicker removed"] = x
    return o


def main():
    seg = models_config()["segmenter"]
    cd, ch, clip = seg["cost_d"], seg["cost_h"], seg["prob_clip"]
    counts, gains, timing, examples = [], [], {"d_start_lag": [], "d_end_lag": [], "switch_lag": []}, {}
    for _, val in config.folds():
        d = pd.read_parquet(os.path.join(PHASE4, "d", "oof_{}.parquet".format(val)))
        per, gt, p_d = d.period.to_numpy(), d.label.to_numpy(), d.p_d.to_numpy()
        for m, s in config.units(val):
            uid = config.unit_id(m, s)
            h = pd.read_parquet(os.path.join(PHASE4, "ha", "oof_{}.parquet".format(uid)))
            p_h = h.p_h.to_numpy()
            lab = segment(p_d, p_h, per, cd, ch, clip)
            out, ex, tm = analyse(gt, lab, per, h.ball_status.to_numpy())
            for k, (num, den) in out.items():
                counts.append({"unit": uid, "category": k, "errors": num, "out_of": den})
                if s in (None, 1):
                    examples.setdefault(k, []).extend("{} {}".format(m, e) for e in ex[k][:1])
            for k in timing:
                timing[k] += tm[k]
            base = score(gt, lab, per)["spell_f1_mean"]
            for k, x in oracles(gt, lab, p_d, p_h, per, cd, ch, clip).items():
                gains.append({"unit": uid, "fix": k, "gain": score(gt, x, per)["spell_f1_mean"] - base, "base": base})
    c = pd.DataFrame(counts)
    t = c.groupby("category").agg(errors_per_match=("errors", "mean"), out_of=("out_of", "mean"))
    t["share"] = t.errors_per_match / t.out_of
    os.makedirs(OUT, exist_ok=True)
    c.to_csv(os.path.join(OUT, "errors_round0.csv"), index=False)
    pd.set_option("display.width", 220)
    print("A. error categories (mean per unit over 30 dev units; 'out of' = how many chances)")
    print(t.round(3).to_string())
    g = pd.DataFrame(gains).groupby("fix").gain.agg(["mean", "min", "max"])
    print("\nB. spellF1 gained if one category were fixed perfectly (oracle, mean over 30 units; base {:.4f})".format(
        pd.DataFrame(gains).base.mean()))
    print(g.sort_values("mean", ascending=False).round(4).to_string())
    print("\nC. timing (seconds, predicted minus true; + = late)")
    for k, v in timing.items():
        v = np.array(v)
        print("   {:<12} n {:>6} | median {:+.2f} | mean {:+.2f} | share > +1 s {:.2f} | share < -1 s {:.2f}".format(
            k, len(v), np.median(v), v.mean(), (v > 1).mean(), (v < -1).mean()))
    print("\nD. examples (seed 1, first per match; H1/H2 = half, mm:ss from the start of that half)")
    for k in sorted(examples):
        print("  ", k + ":", "; ".join(examples[k][:3]))


if __name__ == "__main__":
    main()
