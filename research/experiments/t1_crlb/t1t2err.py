"""R1 bias when the fit assumes a WRONG T2 (T2 known only approximately from the B0/T2/fat sequence).
Pre-registered question (user, 2026-11-28): B0/T2/fat runs first, so T2 is a good estimate, not a free parameter. How good must it be?
Data: EPG signal of the inversion design (noise free, increment 117) at true (T1, T2_true(T1), B1), Gd series
  1/T1 = 1/T10 + r1 C, 1/T2 = 1/T20 + r2 C. Fit: (rho linear, T1, B1) with T2 = assumed rule evaluated at the candidate T1.
Assumed rules: true | baseline factor f (T20 -> f T20) | r2 wrong | T2 fixed at T20 (Gd shortening ignored).
Output: worst-B1 R1 error in units of tol = 0.02 + 0.05 R1 (1/s).
Decision: required T2 accuracy; is a tied T2(R1) rule needed or is a fixed T2 map enough.
usage: python t1t2err.py csf|gm   (flips/TR/T_p fixed to the current IR design)"""
import os, sys, subprocess, io
import numpy as np
from scipy.optimize import least_squares

BIN = "../../../rustlib/target/epg-check/release/t1ir"
GAP = 15e-3
FLIPS, TR, TP = "13.1,11.6,3,30,3,3,3,3", 20.0, 12.0
ENV = dict(os.environ, INCS="117")
nper = int((TP - GAP) / (TR * 1e-3)) // len(FLIPS.split(","))
SEGSTR = ",".join(f"{f}:{TR}:{nper}" for f in FLIPS.split(","))


def sigs(rows):
    inp = "\n".join(",".join(f"{x:.9g}" for x in r) for r in rows) + "\n"
    out = subprocess.run([BIN, "eval", "ir", SEGSTR, "csf"], input=inp, capture_output=True, text=True, check=True, env=ENV).stdout
    return np.loadtxt(io.StringIO(out), delimiter=",", ndmin=2)


SER = dict(csf=dict(T10=4.5, T20=2.0, r1=3.3, r2=5.0, grid=[0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 2.8, 3.6, 4.5]),
           gm=dict(T10=1.3, T20=0.10, r1=3.3, r2=5.0, grid=[0.3, 0.4, 0.6, 0.8, 1.0, 1.3]))


def t2_rule(T1, s, f=1.0, r2=None, fixed=False):
    c = max(0.0, (1 / T1 - 1 / s["T10"]) / s["r1"])
    T20 = f * s["T20"]
    if fixed:
        return T20
    return 1.0 / (1 / T20 + (s["r2"] if r2 is None else r2) * c)


def main():
    series = sys.argv[1] if len(sys.argv) > 1 else "csf"
    s = SER[series]
    variants = [("true", {}), ("T20 x0.8", dict(f=0.8)), ("T20 x1.25", dict(f=1.25)), ("T20 x0.5", dict(f=0.5)), ("T20 x2", dict(f=2.0)),
                ("r2 /2", dict(r2=s["r2"] / 2)), ("r2 x2", dict(r2=s["r2"] * 2)), ("T2 fixed T20", dict(fixed=True))]
    print(f"series {series} {s}  design IR T_p {TP} flips {FLIPS} TR {TR}; R1 error (1/s) and in tol units (worst over B1 1.0/0.8)")
    print(f"{'T1':>5s} {'T2true':>7s} {'tol':>6s} | " + " | ".join(f"{n:>15s}" for n, _ in variants))
    for T1 in s["grid"]:
        T2t = t2_rule(T1, s)
        tol = 0.02 + 0.05 / T1
        line = []
        for name, kw in variants:
            worst = 0.0
            for B1 in (1.0, 0.8):
                d = sigs([(T1, T2t, B1)])[0]

                def model(x):
                    t1, b1 = np.exp(x)
                    m = sigs([(t1, t2_rule(t1, s, **kw), b1)])[0]
                    return m

                def res(x):
                    m = model(x)
                    rho = (m @ d) / (m @ m)
                    return (rho * m - d) / abs(d).max()

                r = least_squares(res, np.log([T1, B1]), method="lm", x_scale=1.0, diff_step=1e-4)
                e = 1.0 / np.exp(r.x[0]) - 1.0 / T1
                if abs(e) > abs(worst):
                    worst = e
            line.append(f"{worst:+7.3f} ({worst/tol:+5.2f})")
        print(f"{T1:5.2f} {T2t:7.3f} {tol:6.3f} | " + " | ".join(f"{x:>15s}" for x in line))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
