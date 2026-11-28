"""Literature VFA (Wahlin 2026: 2/12 deg, TR 7.2 ms, 750 s, B1 from a separate map) vs our designs, B1 NOT pessimistic.
Pre-registered question (user 2026-11-28): B1 was measured beforehand with some reasonable error. How does the gap change?
(a) Fisher with a Gaussian prior on ln B1 (sigma 0 %=known ... free), sigma_R1 and L4(sigma/tol) on the narrow grid; (b) bias of R1 when the fit uses B1 off by +-5 %.
Ideal-spoiling model, unit SNR_TR, same weights as t1crlb_wide.py.  usage: GRID=narrow python t1crlb_vs_lit.py"""
import numpy as np
from scipy.optimize import least_squares
import t1crlb_wide as w

w.B1S = [1.0, 0.8]


def sigma_r1_prior(segs, prep, prior, t_scan):
    """like w.sigma_r1 with a prior sigma (relative) on B1; prior = None -> free, 0 -> known"""
    cases = [(t, b) for t in w.T1_GRID for b in w.B1S]
    nc = len(cases)
    base = np.array([[1.0, t, b] for t, b in cases]).T
    TH, hs = [], []
    for i in range(3):
        h = 1e-6 * base[i]
        tp, tm = base.copy(), base.copy()
        tp[i] += h
        tm[i] -= h
        TH += [tp, tm]
        hs.append(h)
    TH = np.concatenate(TH, axis=1)
    S, dur = w.sim(TH[0], TH[1], TH[2], segs, prep)
    reps = t_scan / dur
    out = np.zeros(nc)
    for c in range(nc):
        D = np.stack([(S[:, 2 * k * nc + c] - S[:, (2 * k + 1) * nc + c]) / (2 * hs[k][c]) * base[k, c] for k in range(3)], axis=1)
        J = D.T @ D * reps
        if prior is not None:
            J[2, 2] += 1.0 / max(prior, 1e-9) ** 2
        C = np.linalg.inv(J)
        out[c] = np.sqrt(C[1, 1]) / cases[c][0]
    return out.reshape(len(w.T1_GRID), len(w.B1S))


def l4(s):
    return float(np.mean((s / w.tol()) ** 4) ** 0.25)


def bias_b1(segs, prep, rel):
    """R1 error when T1,rho fitted with B1 fixed at (1+rel) * truth; worst over B1 true 1.0/0.8; units of tol"""
    res = np.zeros(len(w.T1_GRID))
    for i, T1 in enumerate(w.T1_GRID):
        worst = 0.0
        for B1 in w.B1S:
            d = w.sim(np.array([1.0]), np.array([T1]), np.array([B1]), segs, prep)[0][:, 0]

            def r(x):
                m = w.sim(np.array([1.0]), np.array([np.exp(x[0])]), np.array([B1 * (1 + rel)]), segs, prep)[0][:, 0]
                rho = (m @ d) / (m @ m)
                return (rho * m - d) / abs(d).max()
            f = least_squares(r, [np.log(T1)], method="lm", diff_step=1e-4)
            e = 1 / np.exp(f.x[0]) - 1 / T1
            if abs(e) > abs(worst):
                worst = e
        res[i] = worst / (0.02 + 0.05 / T1)
    return res


if __name__ == "__main__":
    lit = (w.nseg([2, 12], [7.2, 7.2], 20.0), False, 750.0, "literature VFA 2/12 deg TR 7.2, 750 s")
    flips = [13.1, 11.6, 3, 30, 3, 3, 3, 3]
    n = int((12.0 - w.T_GAP) / 0.020) // 8
    ours = ([(np.deg2rad(a), 0.020, n) for a in flips], True, 480.0, "ours IR T_p 12 s, 8 flips, TR 20 ms, 480 s")
    print("T1 grid", w.T1_GRID, "tol", np.round(w.tol()[:, 0], 3))
    for segs, prep, ts, name in (lit, ours):
        print("\n" + name)
        for pr in (None, 0.05, 0.03, 0.01, 0.0):
            s = sigma_r1_prior(segs, prep, pr, ts)
            g = np.exp(np.mean(np.log(s), axis=1))
            print(f"  B1 prior {'free' if pr is None else f'{100*pr:g} %':>6s}: L4 {l4(s):7.2f} | sigma_R1: " + " ".join(f"{t:.1f}:{v:.3f}" for t, v in zip(w.T1_GRID, g)))
        for rel in (0.05, -0.05):
            b = bias_b1(segs, prep, rel)
            print(f"  R1 bias, B1 fixed {100*rel:+.0f} % off, tol units: " + " ".join(f"{t:.1f}:{v:+.2f}" for t, v in zip(w.T1_GRID, b)))
