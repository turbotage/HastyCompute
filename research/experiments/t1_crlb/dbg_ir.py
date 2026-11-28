"""consistency check: EPG signals of the IR design vs the ideal-spoiling model at the true parameters (run from this directory)"""
import csv, io
import numpy as np
import t1ir

segstr = ",".join(f"{f}:20:56" for f in (21.5, 8, 17, 3, 3, 3, 3, 3))
segs = [(np.deg2rad(float(f)), float(t) * 1e-3, int(n)) for f, t, n in (b.split(":") for b in segstr.split(","))]
sig = list(csv.DictReader(io.StringIO(t1ir.run("sig", "ir", segstr, "csf"))))
for T1 in (0.4, 1.0, 4.3):
    for inc in ("117", "150"):
        rr = sorted([r for r in sig if float(r["T1"]) == T1 and r["inc"] == inc and float(r["b1"]) == 1.0], key=lambda r: int(r["idx"]))
        d = np.array([float(r["val"]) for r in rr])
        m = t1ir.ideal_model(1.0, T1, 1.0, segs, True)
        k = (d * m).sum() / (m * m).sum()
        res = d - k * m
        print(T1, inc, "scale", round(k, 4), "rms resid % of peak", round(100 * np.sqrt((res ** 2).mean()) / d.max(), 3),
              "max|res| % of peak", round(100 * abs(res).max() / d.max(), 3), "d[:3]", d[:3].round(4), "ideal[:3]", (k * m[:3]).round(4))
