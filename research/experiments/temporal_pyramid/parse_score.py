"""Tables for temporal_pyramid.md §13 from run_score.sh's run directories.

    parse_score.py <tpyr_score dir>

Per run and start (tag): the pc score (pcscore_<tag>.csv: RMSE, time-mean / detail RMSE, |Δv| median / p90, share
> VENC, phase RMS; hierarchical decoder RMSE / median) at it 0, 12 and the last report, and the data term f; per
--holdout-eval run the held-out relative residual √(f(x) / f(0)); the pooled PCG's last relative residual per run.
"""
import csv
import glob
import os
import re
import sys

root = sys.argv[1]
for d in sorted(glob.glob(os.path.join(root, "*"))):
    n = os.path.basename(d)
    log = os.path.join(d, "log.txt")
    if not os.path.isfile(log) or n.startswith("smoke"):
        continue
    txt = open(log).read()
    if n.startswith("ev_"):
        f = re.findall(r"it   1: f = ([0-9.eE+-]+)", txt)
        if len(f) >= 2:
            print(f"{n}: held-out f(x) {float(f[0]):.4e}, f(0) {float(f[1]):.4e}, relative residual {(float(f[0]) / float(f[1])) ** 0.5:.4f}")
        continue
    pr = re.findall(r"pooled PCG it (\d+): ‖r‖/‖r₀‖ per group ([0-9.e+\- ]+)", txt)
    if pr:
        it, v = pr[-1]
        print(f"{n}: pooled PCG it {it} relative residual {v.split()[0]} (encode 0)")
    for c in sorted(glob.glob(os.path.join(d, "pcscore_*.csv"))):
        tag = os.path.basename(c)[len("pcscore_"):-4]
        rows = list(csv.DictReader(open(c)))
        if not rows:
            continue
        pick = [r for r in rows if r["it"] in ("0", "12")] + ([rows[-1]] if rows[-1]["it"] not in ("0", "12") else [])
        print(f"  {n} {tag}")
        for r in pick:
            g = lambda k: float(r[k])  # noqa: E731
            print(f"    it {r['it']:>3}: f {r['f']:>11}  RMSE {g('rmse'):.3f} (mean {g('mean'):.3f} / detail {g('detail'):.3f})  median {g('median'):.3f} p90 {g('p90'):.3f}  >VENC {100 * g('wrap'):.1f} %  phase {g('phase'):.3f}  | hier RMSE {g('rmse_hier'):.3f} median {g('median_hier'):.3f}")
