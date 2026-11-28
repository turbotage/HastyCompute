"""EPG study of inversion-prepared designs on a wide T1 range with a Gd model (CSF T2 shortens with concentration).
usage: python t1ir.py "<flips comma>" TR_ms Tp_s series [ir|none]   e.g. python t1ir.py 21.5,8,17,3,3,3,3,3 20 9 csf
 Prints per T1: sigma_R1 of ideal / EPG(T2 known) / EPG(T2 free) models (Fisher, B1 and increment averaged) and the R1 bias of
 the ideal-model fit to EPG data (B1 free, truth start), in units of the tolerance 0.02 + 0.05 R1 (1/s).
 A multi-TR design: use SEGS='f:tr:n,f:tr:n,...' env instead of the positional flips."""
import os, sys, subprocess, csv, io
import numpy as np
from scipy.optimize import least_squares

BIN = "../../../rustlib/target/epg-check/release/t1ir"
GAP = 15e-3


def segs_from_args():
    if os.environ.get("SEGS"):
        return os.environ["SEGS"]
    flips = [float(x) for x in sys.argv[1].split(",")]
    tr, tp = float(sys.argv[2]), float(sys.argv[3])
    n = int((tp - GAP) / (tr * 1e-3))
    per = n // len(flips)
    return ",".join(f"{f}:{tr}:{per}" for f in flips)


def run(mode, ir, segs, series):
    return subprocess.run([BIN, mode, ir, segs, series], capture_output=True, text=True, check=True).stdout


def ideal_model(rho, T1, B1, segs, ir):
    """cyclic ideal-spoiling steady state of one period (same recursion as the CRLB study)"""
    def go(m, record):
        out = []
        if ir:
            Eg = np.exp(-GAP / T1)
            m = np.cos(np.pi * B1) * m * Eg + (1 - Eg)
        for a, tr, n in segs:
            ca, sa, E = np.cos(B1 * a), np.sin(B1 * a), np.exp(-tr / T1)
            for _ in range(n):
                out.append(rho * sa * m)
                m = 1 + (m * ca - 1) * E
        return np.array(out), m
    _, b = go(0.0, False)
    _, ab = go(1.0, False)
    m0 = b / (1 - (ab - b))
    return go(m0, True)[0]


def main():
    segstr = segs_from_args()
    series = sys.argv[4] if len(sys.argv) > 4 and os.environ.get("SEGS") is None else (sys.argv[1] if os.environ.get("SEGS") else "csf")
    ir = "none" if "none" in sys.argv[1:] else "ir"
    print(f"design {ir} segs {segstr} series {series}")
    segs = [(np.deg2rad(float(f)), float(t) * 1e-3, int(n)) for f, t, n in (b.split(":") for b in segstr.split(","))]
    # Fisher
    rows = list(csv.DictReader(io.StringIO(run("fisher", ir, segstr, series))))
    t1s = sorted({float(r["T1"]) for r in rows})
    t2of = {float(r["T1"]): float(r["T2"]) for r in rows}
    acc = {}
    for r in rows:
        if r["sigma_R1"] == "NONID":
            continue
        acc.setdefault((float(r["T1"]), r["model"] + r["free"]), []).append(np.log(float(r["sigma_R1"])))
    g = lambda t, k: np.exp(np.mean(acc[(t, k)])) if (t, k) in acc else np.nan
    # bias of the ideal fit
    sig = list(csv.DictReader(io.StringIO(run("sig", ir, segstr, series))))
    bias = {}
    for t in t1s:
        for inc in sorted({r["inc"] for r in sig}, key=float):
            for b1 in ("1.0", "0.8"):
                rr = [r for r in sig if float(r["T1"]) == t and r["inc"] == inc and float(r["b1"]) == float(b1)]
                rr.sort(key=lambda r: int(r["idx"]))
                d = np.array([float(r["val"]) for r in rr])
                B1t = float(b1)
                f = lambda x: (ideal_model(np.exp(x[0]), np.exp(x[1]), np.exp(x[2]), segs, ir == "ir") - d) / abs(d).max()
                x0 = np.array([np.log(abs(d).max() * 2), np.log(t), np.log(B1t)])
                best = None
                for p in ([0, 0, 0], [.2, .2, .03], [-.2, -.2, -.03]):
                    res = least_squares(f, x0 + np.array(p), method="lm")
                    if best is None or res.cost < best.cost:
                        best = res
                bias.setdefault((t, inc), []).append(1.0 / np.exp(best.x[1]) - 1.0 / t)   # R1 error 1/s
    tol = lambda t: 0.02 + 0.05 / t
    print(f"{'T1':>5s} {'T2':>6s} | sigma_R1 ideal  epgT2known  epgT2free  (1/s, per unit SNR_TR) | ideal-fit R1 error 117°  150° (1/s, worst B1)  | in tol units")
    incl = sorted({k[1] for k in bias}, key=float)
    print("increments", incl)
    for t in t1s:
        bs = [max(bias[(t, i)], key=abs) for i in incl]
        print(f"{t:5.2f} {t2of[t]:6.3f} | {g(t,'ideal3'):13.3f} {g(t,'epg3'):11.3f} {g(t,'epg4'):10.3f} | " + " ".join(f"{b:+7.3f}" for b in bs) + " | tol units " + " ".join(f"{b/tol(t):+6.1f}" for b in bs))


if __name__ == "__main__":
    main()
