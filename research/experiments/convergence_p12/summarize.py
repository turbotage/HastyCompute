"""Summarise P1–P2 solver arms of one or more levels (run_level.sh): held-out residual vs gradients and the truth
scores (score_pair.py) at the saved gradient counts.

    rustlib/.venv/bin/python -I research/experiments/convergence_p12/summarize.py <tag> [...] [--ref <run prefix, e.g. p12_k32ref_admm1>]

Per level and encode: the held-out target = the PCG arm's held-out residual at its last iterate (the arms v2 baseline
quality); for every arm the gradients to reach it, its minimum held-out residual and where. Truth: encode 1 − 0
velocity-phase slope / corr at depth >= 5 and in the edge band 0-2, and dynamics >= 2, at r* = 0.493 (and the mean
depth >= 5 slope over the three breathing states).
"""
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
O = REPO / "data/hasty/phantom/factory_arms/ph0001/spline2d"
PY = REPO / "rustlib/.venv/bin/python"
SP = REPO / "research/experiments/spline2d/score_pair.py"


def heldout(tag):
    out = {}
    for f in sorted(O.glob(f"p12_{tag}_arms_e*.csv")):
        e = int(re.search(r"_e(\d+)\.csv", f.name).group(1))
        rows = defaultdict(list)
        for r in csv.DictReader(open(f)):
            rows[r["arm"]].append((float(r["grads"]), float(r["heldout"]), float(r["F"]) if r["F"] not in ("nan", "NaN") else float("nan")))
        out[e] = rows
    return out


def scores(tag, arm):
    pre = O / f"p12_{tag}_{arm}"
    txt = subprocess.run([str(PY), "-I", str(SP), str(pre)], capture_output=True, text=True, cwd=REPO).stdout
    res = {}
    for line in txt.splitlines():
        m = re.match(r"p12_.*_r([\d.]+)_it(\w+)_lumen\.\w+ depth 0-2 ([-\d.]+)/([-\d.]+) \| depth 5-99 ([-\d.]+)/([-\d.]+) \| dyn>=2 ([-\d.]+)/([-\d.]+)", line)
        if m:
            r, it = m.group(1), m.group(2)
            res[(r, it)] = tuple(float(m.group(i)) for i in range(3, 9))
    return res


def refdist(tag, ref_prefix, arms, labels=("5", "10", "20", "40", "best")):
    """relative distance of each arm's lumen values (all states, encodes, frames) to a reference run's"""
    import numpy as np
    sys.path.insert(0, str(REPO / "python/lib"))
    try:
        from hastycompute.store import load_any
    except Exception:
        load_any = np.load

    def load(pre, lab):
        out = []
        for r in ("0.07", "0.181", "0.493"):
            for ext in (".hv", ".npy"):
                f = O / f"{pre}_r{r}_it{lab}_lumen{ext}"
                if f.exists():
                    out.append(load_any(str(f)).ravel())
                    break
        return np.concatenate(out) if len(out) == 3 else None

    ref = load(ref_prefix, "best")
    if ref is None:
        print(f"  (no reference {ref_prefix})")
        return
    print(f"  distance of the lumen values to the reference {ref_prefix} (relative):")
    for a in arms:
        row = []
        for lab in labels:
            v = load(f"p12_{tag}_{a}", lab)
            if v is not None:
                row.append(f"{lab}: {np.linalg.norm(v - ref) / np.linalg.norm(ref):.4f}")
        print(f"    {a:9s} " + " | ".join(row))


def main(tags, ref=None):
    for tag in tags:
        ho = heldout(tag)
        print(f"=== level {tag}")
        arms = list(dict.fromkeys(a for e in ho for a in ho[e]))
        for e, rows in ho.items():
            pcg = rows.get("pcg")
            h0 = pcg[0][1] if pcg else float("nan")
            hall = min(h for a in rows for _, h, _ in rows[a])
            target = min(h for _, h, _ in pcg) if pcg else hall
            t90 = h0 - 0.9 * (h0 - hall)
            print(f"  encode {e}: held-out at the warm start {h0:.5f}; targets: PCG's best {target:.5f}, 90 % of the best reduction over all arms {t90:.5f} (best {hall:.5f})")
            fmt = lambda v: ('%5.1f' % v) if v is not None else '   — '
            for a in arms:
                r = rows.get(a, [])
                if not r:
                    continue
                hit = next((g for g, h, _ in r if h <= target), None)
                hit90 = next((g for g, h, _ in r if h <= t90), None)
                gmin, hmin, _ = min(r, key=lambda t: t[1])
                fl = [(g, f) for g, _, f in r if f == f]
                print(f"    {a:9s} grads→PCG-best {fmt(hit)} →90% {fmt(hit90)}  min held-out {hmin:.5f} at {gmin:5.1f}  last {r[-1][1]:.5f} at {r[-1][0]:.1f}" + (f"  F last {fl[-1][1]:.6e}" if fl else ""))
            same = [a for a in ("ncg", "ncg0", "ncgcap", "ncgI", "fista", "zmshr1") if a in rows]
            fs = {a: [(g, f) for g, _, f in rows[a] if f == f] for a in same}
            fs = {a: v for a, v in fs.items() if v}
            if fs:
                f0 = max(v[0][1] for v in fs.values())
                fb = min(f for v in fs.values() for _, f in v)
                line = []
                for a, v in fs.items():
                    hit = [next((g for g, f in v if f - fb <= t * (f0 - fb)), None) for t in (0.1, 0.01)]
                    line.append(f"{a} {'/'.join('%.0f' % h if h is not None else '—' for h in hit)}")
                print("    smoothed objective F_γ (γ = 0.1/bulk), gradients to 10 % / 1 % of the excess over the best value seen: " + ", ".join(line))
        print("  truth (encode 1 − 0 phase; r 0.493: depth>=5 slope/corr, edge 0-2 slope, dyn>=2 slope; mean depth>=5 slope over r):")
        for a in arms:
            sc = scores(tag, a)
            its = sorted({it for _, it in sc}, key=lambda s: (s == "best", int(s) if s.isdigit() else 0))
            parts = []
            for it in its:
                v = sc.get(("0.493", it))
                if v is None:
                    continue
                mean5 = sum(sc[(r, it)][2] for r in ("0.07", "0.181", "0.493") if (r, it) in sc) / 3
                parts.append(f"{it}: {v[2]:.3f}/{v[3]:.3f} e{v[0]:.3f} d{v[4]:.3f} m{mean5:.3f}")
            print(f"    {a:9s} " + " | ".join(parts))
        if ref:
            refdist(tag, ref, arms)


if __name__ == "__main__":
    args = sys.argv[1:]
    ref = None
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        del args[i:i + 2]
    main(args, ref)
