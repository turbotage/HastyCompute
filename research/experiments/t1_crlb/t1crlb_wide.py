"""Wide-range T1 mapping for intrathecal gadolinium (T1 0.1-4.5 s): CRLB in ABSOLUTE R1 precision.

Pre-registered question (user, 2026-11-27): concentration C = dR1 / r1, so the precision that matters is sigma_R1 = kappa_T1 / T1
(1/s, at SNR per TR = 1, scan 480 s, scaled by 1/SNR_TR), needed UNIFORMLY over T1 = 0.1 ... 4.5 s (CSF baseline 4.2-4.3 s at the
slow end, tissue/CSF with Gd toward 0.1-1 s). Which schedule family gives the lowest worst-case / L4 sigma_R1:
  N: no-prep cyclic (flip, TR) blocks, block length B as a parameter        (settled vs transient blocks)
  I: hard-180 inversion (eta = -cos(pi B1), no extra unknown) every T_p, K flip pieces, TR fixed
Decision it feeds: schedule family for the T1 sequence, and whether CSF-range accuracy forces an inversion.
Model: ideal spoiling (EPG bias is checked separately), rho, T1, B1 free, B1 true in {1.0, 0.8}, TR <= 25 ms, readout = min(15 ms, TR-1.5 ms),
measurement weight = ADC time / 9.5 ms, flip in [3, 30] deg. Relative numbers only (SNR_TR = 1).
usage: python t1crlb_wide.py [sections: N I E]  (E = evaluate old head-optimised schedules on the wide grid)"""
import sys, time, os
import numpy as np
from scipy.optimize import minimize

T1_GRID = np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.2, 3.0, 3.8, 4.3])
if os.environ.get("GRID") == "narrow":      # literature (lit_intrathecal_gd_t1.md): intracranial T1 >= ~0.4 s, CSF baseline 4.2-4.5 s
    T1_GRID = np.array([0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 2.8, 3.6, 4.5])
B1S = [1.0, 0.8]
T_SCAN = 480.0
T_GAP = 15e-3
LO, HI = 3.0, float(os.environ.get("HI", "30"))


def wt(tr):
    return min(15.0, tr * 1e3 - 1.5) / 9.5


def sim(rho, T1, B1, segs, prep):
    """cyclic steady state of one period; segs = [(flip_rad, tr_s, n)]; arrays over cases. Returns (signals (N, nc), period s)"""
    Eg = np.exp(-T_GAP / T1)

    def run(m0):
        m = m0
        if prep:
            m = np.cos(np.pi * B1) * m * Eg + (1.0 - Eg)
        out = []
        for a, tr, n in segs:
            ba = B1 * a
            ca, sa = np.cos(ba), np.sin(ba)
            E = np.exp(-tr / T1)
            s = np.sqrt(wt(tr))
            for _ in range(n):
                out.append(rho * sa * m * s)
                m = 1.0 + (m * ca - 1.0) * E
        return out, m

    _, b = run(np.zeros_like(rho))
    _, ab = run(np.ones_like(rho))
    m0 = b / (1.0 - (ab - b))
    out, _ = run(m0)
    dur = sum(tr * n for _, tr, n in segs) + (T_GAP if prep else 0.0)
    return np.array(out), dur


def sigma_r1(segs, prep):
    """sigma_R1 (1/s) for every (T1 grid, B1) -> array (len(T1_GRID), len(B1S)); 1e3 if T1 is not identifiable"""
    cases = [(t, b) for t in T1_GRID for b in B1S]
    nc = len(cases)
    base = np.array([[1.0, t, b] for t, b in cases]).T          # rho, T1, B1
    TH, hs = [], []
    for i in range(3):
        h = 1e-6 * base[i]
        tp, tm = base.copy(), base.copy()
        tp[i] += h
        tm[i] -= h
        TH += [tp, tm]
        hs.append(h)
    TH = np.concatenate(TH, axis=1)
    S, dur = sim(TH[0], TH[1], TH[2], segs, prep)
    reps = T_SCAN / dur
    out = np.zeros(nc)
    for c in range(nc):
        D = np.stack([(S[:, 2 * k * nc + c] - S[:, (2 * k + 1) * nc + c]) / (2 * hs[k][c]) * base[k, c] for k in range(3)], axis=1)
        J = D.T @ D * reps
        ev, V = np.linalg.eigh(J)
        null = ev <= 1e-11 * ev.max()
        if np.sum(V[1, null] ** 2) > 1e-6:
            out[c] = 1e3
        else:
            keep = ~null
            out[c] = np.sqrt(((V[:, keep] / ev[keep]) @ V[:, keep].T)[1, 1]) / cases[c][0]
    return out.reshape(len(T1_GRID), len(B1S))


TOL_ABS, TOL_REL = 0.02, 0.05      # tolerance on R1: 0.02 /s + 5 % of R1 (user to confirm)


def tol():
    return (TOL_ABS + TOL_REL / T1_GRID)[:, None]


def objective(segs, prep):
    s = sigma_r1(segs, prep) / tol()           # error in units of the R1 tolerance
    return float(np.mean(s ** 4) ** 0.25)      # L4 over the grid ~ worst case, smooth


def sg(x, lo, hi):
    return lo + (hi - lo) / (1 + np.exp(-np.clip(x, -30, 30)))


def optimise(make, nx, starts=4, seed=5, maxiter=70):
    f = lambda x: np.log(objective(*make(x)))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        r = minimize(f, rng.normal(0, 1.0, nx), method="L-BFGS-B", options=dict(maxiter=maxiter, eps=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.fun)), best.x


def report(name, segs, prep, extra=""):
    s = sigma_r1(segs, prep)
    g = np.exp(np.mean(np.log(s), axis=1))
    print(f"  {name}: L4(sigma_R1/tol) {objective(segs, prep):.3f}  max sigma_R1 {s.max():.4f}  {extra}")
    print("      sigma_R1 [1/s] by T1 [s]: " + "  ".join(f"{t:.1f}:{v:.3f}" for t, v in zip(T1_GRID, g)))
    print("      sigma_R1/tol:             " + "  ".join(f"{t:.1f}:{v:.2f}" for t, v in zip(T1_GRID, g / tol()[:, 0])))
    sys.stdout.flush()


def nseg(flips, trs, B):
    return [(np.deg2rad(a), r * 1e-3, max(1, int(B / (r * 1e-3)))) for a, r in zip(flips, trs)]


if __name__ == "__main__":
    t0 = time.time()
    SEC = sys.argv[1] if len(sys.argv) > 1 else "NIE"
    print("T1 grid", T1_GRID, " B1", B1S, " objective = L4 norm of sigma_R1 [1/s] at SNR_TR 1 (lower better); table = geometric mean over B1")
    if "E" in SEC:
        print("\n== E. earlier head-optimised schedules (1 s blocks) evaluated on the wide grid")
        report("2 flips x 3 TRs  3/30 deg, TR 11,20,20 (1 s blocks)", nseg([3, 3, 3, 30, 30, 30], [11, 20, 20, 11, 20, 20], 1.0), False)
        report("6 free blocks 11.6/3/3/3/30/20.7 deg, TR 20,20,20,20,16.5,20", nseg([11.6, 3, 3, 3, 30, 20.7], [20, 20, 20, 20, 16.5, 20], 1.0), False)
        report("classic VFA 3/15 deg TR 11, 8 s blocks", nseg([3, 15], [11, 11], 8.0), False)
    if "N" in SEC:
        print("\n== N. no-prep cyclic blocks, optimised flips and TRs, block length B")
        for K, B in [(4, 1.0), (4, 2.0), (6, 2.0), (6, 4.0)]:
            mk = lambda x, K=K, B=B: (nseg(sg(x[:K], LO, HI), sg(x[K:], 11, 25), B), False)
            k, x = optimise(mk, 2 * K)
            report(f"K={K} blocks of {B} s", *mk(x), extra=f"flips {np.round(sg(x[:K], LO, HI), 1)} TRs {np.round(sg(x[K:], 11, 25), 1)}  [{time.time()-t0:.0f}s]")
    if "I" in SEC:
        print("\n== I. hard-180 inversion every T_p, K flip pieces, fixed TR")
        for Tp in (6.0, 9.0, 12.0):
            for tr in (20.0,):
                K = 8
                n = int((Tp - T_GAP) / (tr * 1e-3))
                per = [n // K + (1 if i < n % K else 0) for i in range(K)]
                mk = lambda x, per=per, tr=tr: ([(np.deg2rad(a), tr * 1e-3, p) for a, p in zip(sg(x, LO, HI), per)], True)
                k, x = optimise(mk, K)
                report(f"IR T_p {Tp} s TR {tr} ms K={K}", *mk(x), extra=f"flips {np.round(sg(x, LO, HI), 1)}  [{time.time()-t0:.0f}s]")
    if "J" in SEC:
        print("\n== J. hard-180 inversion, 8 pieces of equal duration, free flip AND TR per piece")
        for Tp in (9.0, 12.0):
            K = 8
            D = (Tp - T_GAP) / K
            mk = lambda x, D=D: ([(np.deg2rad(a), r * 1e-3, max(1, int(D / (r * 1e-3)))) for a, r in zip(sg(x[:K], LO, HI), sg(x[K:], 11, 25))], True)
            k, x = optimise(mk, 2 * K, starts=5)
            report(f"IR T_p {Tp} s free TR", *mk(x), extra=f"flips {np.round(sg(x[:K], LO, HI), 1)} TRs {np.round(sg(x[K:], 11, 25), 1)}  [{time.time()-t0:.0f}s]")
    print(f"done {time.time()-t0:.0f} s")
