"""summaries of t1bias fisher csv: geometric-mean kappa over tissues/B1 per (inc, model, free); usage: python t1bias_sum.py sched"""
import subprocess, sys, csv, io, collections
import numpy as np
BIN = "../../../rustlib/target/epg-check/release/t1bias"
sched = sys.argv[1]
out = subprocess.run([BIN, sched, "2", "150", "fisher"], capture_output=True, text=True, check=True).stdout
d = collections.defaultdict(list)
for r in csv.DictReader(io.StringIO(out)):
    v = np.nan if r["kappa"] == "NONID" else float(r["kappa"])
    d[(r["inc"], r["model"], r["free"])].append((r["tissue"], v))
print(f"schedule {sched}   (geometric mean κ over WM, GM, blood, CSF and B1 1/0.8; lower better)")
for k, v in d.items():
    g = np.exp(np.mean([np.log(x) for t, x in v if t != "fat"]))
    per = {t: round(float(np.exp(np.mean([np.log(x) for tt, x in v if tt == t]))), 3) for t in ("WM", "GM", "blood", "CSF")}
    print(f"  inc {k[0]:11s} {k[1]:5s} free {k[2]}: κ {g:6.3f}  {per}")
