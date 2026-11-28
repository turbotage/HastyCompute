"""Bias of the ideal-spoiling (Ernst-recursion) model when the data are EPG signals with T2 and RF-spoil increment.
Runs rustlib/.../t1bias (hasty_seq::epg), fits (rho, T1, B1) of the ideal model to the last cycle, prints bias.
usage: python t1bias.py "flip:tr_ms:s,..." [label]
"""
import subprocess, sys, io, os
import numpy as np
from scipy.optimize import least_squares

BIN = "../../../rustlib/target/epg-check/release/t1bias"
TIS = {"fat": 0.29, "WM": 0.83, "GM": 1.33, "blood": 1.75, "CSF": 4.2}


def run(sched, te=2.0):
    out = subprocess.run([BIN, sched, str(te)], capture_output=True, text=True, check=True).stdout
    import csv
    return list(csv.DictReader(io.StringIO(out)))


def model(rho, T1, B1, flips, trs):
    """cyclic ideal-spoiled steady state; flips (rad), trs (s) per pulse over one cycle"""
    E = np.exp(-trs / T1)
    ca, sa = np.cos(B1 * flips), np.sin(B1 * flips)
    # affine map per pulse: m -> (m*ca - 1)*E + 1
    a, b = 1.0, 0.0
    for c, e in zip(ca, E):
        a, b = a * c * e, (b * c * e) + (1 - e)
    m = b / (1 - a)
    s = np.empty(len(flips))
    for i in range(len(flips)):
        s[i] = rho * sa[i] * m
        m = (m * ca[i] - 1) * E[i] + 1
    return s


def main():
    sched = sys.argv[1]
    rows = run(sched)
    nb = len(sched.split(","))
    print(f"schedule {sched}")
    print(f"{'tissue':6s} {'B1':>4s} {'inc':>10s} | T1 bias %   B1 bias %   resid %   (ideal model fit, truth start)")
    keys = sorted({(r["tissue"], r["b1"], r["inc"]) for r in rows}, key=lambda k: (list(TIS).index(k[0]), -float(k[1]), k[2]))
    res = {}
    for k in keys:
        rr = [r for r in rows if (r["tissue"], r["b1"], r["inc"]) == k and int(r["block"]) >= nb]
        rr.sort(key=lambda r: (int(r["block"]), int(r["idx"])))
        fl = np.deg2rad([float(r["flip_deg"]) for r in rr])
        tr = np.array([float(r["tr_ms"]) for r in rr]) * 1e-3
        d = np.array([np.hypot(float(r["re"]), float(r["im"])) for r in rr])
        skip = int(os.environ.get("SKIP", "0"))      # ignore the first `skip` pulses after every block change
        mask = np.ones(len(rr))
        for i, r in enumerate(rr):
            if int(r["idx"]) < skip: mask[i] = 0.0
        T1t, B1t = TIS[k[0]], float(k[1])
        x0 = np.array([np.log(d.max() / np.sin(B1t * fl).max() / 0.5), np.log(T1t), np.log(B1t)])
        if os.environ.get("FIXB1"):   # B1 known exactly: fit rho, T1 only
            f = lambda x: (model(np.exp(x[0]), np.exp(x[1]), B1t, fl, tr) - d) * mask / d.max()
            x0 = x0[:2]
            starts = ([0, 0], [0.3, 0.2], [-0.3, -0.2])
        else:
            f = lambda x: (model(np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), fl, tr) - d) * mask / d.max()
            starts = ([0, 0, 0], [0.3, 0.2, 0.05], [-0.3, -0.2, -0.05])
        # best of a few starts: truth, and truth-perturbed
        best = None
        for p in starts:
            r = least_squares(f, x0 + np.array(p), method="lm")
            if best is None or r.cost < best.cost:
                best = r
        T1f = np.exp(best.x[1]); B1f = np.exp(best.x[2]) if len(best.x) > 2 else B1t
        rs = np.sqrt(2 * best.cost / len(d)) * 100
        res[k] = (100 * (T1f / T1t - 1), 100 * (B1f / B1t - 1), rs)
        print(f"{k[0]:6s} {k[1]:>4s} {k[2]:>10s} | {res[k][0]:9.2f}  {res[k][1]:9.2f}  {rs:8.3f}")
    return res


if __name__ == "__main__":
    main()
