"""Tables from run_dev.sh's log (temporal_pyramid.md §12): per run (init × anchor, in the run order) the lumen velocity
RMSE and its split into time mean / temporal detail at iterations 0, 1, 3, 6, 12."""
import re
import sys

log = open(sys.argv[1]).read().splitlines()
inits = sys.argv[2].split(",")
anch = sys.argv[3].split(",")
names = [(i, a) for i in inits for a in anch]
runs, cur, split = [], None, None
for ln in log:
    m = re.search(r"velocity error split: time-mean RMSE ([\d.]+) m/s, temporal-detail RMSE ([\d.]+) m/s", ln)
    if m:
        md = re.search(r"median ([\d.]+), p90 ([\d.]+)", ln)
        split = (float(m.group(1)), float(m.group(2))) + ((float(md.group(1)), float(md.group(2))) if md else (float("nan"),) * 2)
        continue
    m = re.search(r"it +(\d+): f = \S+, velocity RMSE ([\d.]+) m/s", ln)
    if m:
        k = int(m.group(1))
        if k == 0:
            cur = {}
            runs.append(cur)
        cur[k] = (float(m.group(2)),) + (split or (float("nan"),) * 4)
        split = None
show = [0, 1, 3, 6, 12]
print("| init | anchor | " + " | ".join(f"it {k}: RMSE (mean / detail)" for k in show) + " |")
print("|---|---|" + "---|" * len(show))
for (i, a), r in zip(names, runs):
    cells = [f"{r[k][0]:.3f} ({r[k][1]:.3f} / {r[k][2]:.3f})" if k in r else "–" for k in show]
    print(f"| {i} | {a} | " + " | ".join(cells) + " |")

if any(r[0][3] == r[0][3] for r in runs):
    print()
    print("| init | anchor | " + " | ".join(f"it {k}: median / p90 of |v − t|" for k in show) + " |")
    print("|---|---|" + "---|" * len(show))
    for (i, a), r in zip(names, runs):
        print(f"| {i} | {a} | " + " | ".join(f"{r[k][3]:.3f} / {r[k][4]:.2f}" if k in r else "–" for k in show) + " |")
