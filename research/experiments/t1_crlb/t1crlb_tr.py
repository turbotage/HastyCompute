"""Second modulation knob: TR (head tissues, B1 free, no prep, B1 true in {1.0, 0.8}).
Pre-registered question: at equal scan time, does modulating TR as well as flip angle lower kappa vs flip-only, and does it
reduce the T1-B1 coupling?  Blocks: each block = 1 s of TRs (n = floor(1 s / TR_k)); cyclic.
Weighting of a TR as a measurement:
  mode 'idle'  : readout fixed (9.5 ms), TR lengthened by idle time -> every TR weight 1 (ADC duty drops)
  mode 'duty'  : readout grows with TR (constant ADC duty) -> TR counts TR/11 ms measurements (noise var 11/TR)
Ideal spoiling, relative numbers only."""
import sys, time
import numpy as np
from scipy.optimize import minimize
import t1crlb2 as T

TIS = [("scalp fat", 0.29, 0.5), ("white matter", 0.83, 2.0), ("grey matter", 1.33, 2.0), ("blood", 1.75, 1.0), ("CSF", 4.2, 0.5)]
B1S = [1.0, 0.8]
TR0 = 11e-3
BLK = 1.0
LO, HI = 3.0, 30.0   # flip floor / ceiling (deg)
import os
TRMAX = float(os.environ.get("TRMAX", "80"))


def sim(rho, T1, B1, al, tr, w):
    """cyclic no-prep train; al, tr arrays per TR (rad, s); returns weighted signals (n, B)"""
    def run(m0):
        m = m0
        out = []
        for a, r in zip(al, tr):
            ba = B1 * a
            out.append(rho * np.sin(ba) * m)
            m = 1.0 + (m * np.cos(ba) - 1.0) * np.exp(-r / T1)
        return out, m
    _, b = run(np.zeros_like(rho))
    _, ab = run(np.ones_like(rho))
    m0 = b / (1 - (ab - b))
    out, _ = run(m0)
    return np.array(out) * np.sqrt(w)[:, None]


def kappa_set(flips_deg, trs_ms, mode, free=("rho", "T1", "B1")):
    al, tr = [], []
    for a, r in zip(flips_deg, trs_ms):
        n = int(BLK / (r * 1e-3))
        al += [np.deg2rad(a)] * n
        tr += [r * 1e-3] * n
    al, tr = np.array(al), np.array(tr)
    if mode == "idle":
        w = np.ones_like(tr)
    elif mode == "cap15":      # readout = min(15 ms, TR - 1.5 ms); weight = ADC time / 9.5 ms (user: readout <= ~15 ms)
        w = np.minimum(15.0, tr * 1e3 - 1.5) / 9.5
    else:
        w = tr / TR0
    dur = tr.sum()
    names = ["rho", "T1", "B1"]
    idx = [names.index(f) for f in free]
    cases = [(n, t, b, wt) for n, t, wt in TIS for b in B1S]
    base = np.array([[1.0, t, b] for _, t, b, _ in cases]).T
    nc = len(cases)
    TH, hs = [], []
    for i in idx:
        h = 1e-6 * np.abs(base[i])
        tp, tm = base.copy(), base.copy()
        tp[i] += h; tm[i] -= h
        TH += [tp, tm]; hs.append(h)
    TH = np.concatenate(TH, axis=1)
    S = sim(TH[0], TH[1], TH[2], al, tr, w)
    reps = T.T_SCAN / dur
    j = free.index("T1")
    num = den = 0.0
    for c, (n, t, b, wt) in enumerate(cases):
        D = np.stack([(S[:, (2 * k) * nc + c] - S[:, (2 * k + 1) * nc + c]) / (2 * hs[k][c]) * base[idx[k], c] for k in range(len(idx))], axis=1)
        J = D.T @ D * reps
        ev, V = np.linalg.eigh(J)
        null = ev <= 1e-11 * ev.max()
        if np.sum(V[j, null] ** 2) > 1e-6:
            k_ = 1e3
        else:
            keep = ~null
            k_ = float(np.sqrt(((V[:, keep] / ev[keep]) @ V[:, keep].T)[j, j]))
        num += wt * np.log(k_); den += wt
    return float(np.exp(num / den))


def sg(x, lo, hi):
    return lo + (hi - lo) / (1 + np.exp(-np.clip(x, -30, 30)))


def optimise(make, nx, mode, starts=6, seed=3):
    f = lambda x: np.log(kappa_set(*make(x), mode))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        r = minimize(f, rng.normal(0, 1.0, nx), method="L-BFGS-B", options=dict(maxiter=120, eps=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.fun)), best.x


if __name__ == "__main__":
    t0 = time.time()
    for mode in os.environ.get("MODES", "idle,duty").split(","):
        print(f"\n===== mode {mode} (flip floor {LO}°, ceiling {HI}°, TR in [11, 80] ms) =====")
        # flip only
        for K in (2, 3, 4):
            k, x = optimise(lambda x: (sg(x, LO, HI), [11.0] * len(x)), K, mode)
            print(f"  flip only, K={K} flips, TR 11: kappa {k:.4g} flips {np.round(np.sort(sg(x, LO, HI)), 1)}")
        # TR only, single flip
        k, x = optimise(lambda x: ([sg(x[0], LO, HI)] * 3, [11.0] + list(sg(x[1:3], 11, TRMAX))), 3, mode)
        print(f"  one flip, 3 TRs (11 + 2 free): kappa {k:.4g} flip {sg(x[0], LO, HI):.1f} TRs {np.round(sg(x[1:3], 11, TRMAX), 1)}")
        # 2 flips x 2 TRs
        mk = lambda x: ([sg(x[0], LO, HI)] * 2 + [sg(x[1], LO, HI)] * 2, [11.0, sg(x[2], 11, TRMAX), 11.0, sg(x[2], 11, TRMAX)])
        k, x = optimise(mk, 3, mode)
        print(f"  2 flips x 2 TRs (11, TR2): kappa {k:.4g} flips {np.round(sg(x[:2], LO, HI), 1)} TR2 {sg(x[2], 11, TRMAX):.1f}")
        # 2 flips x 3 TRs
        mk = lambda x: ([sg(x[0], LO, HI)] * 3 + [sg(x[1], LO, HI)] * 3, [11.0, sg(x[2], 11, TRMAX), sg(x[3], 11, TRMAX)] * 2)
        k, x = optimise(mk, 4, mode)
        print(f"  2 flips x 3 TRs: kappa {k:.4g} flips {np.round(sg(x[:2], LO, HI), 1)} TRs {np.round(sg(x[2:4], 11, TRMAX), 1)}")
        # free (flip, TR) per block, K=4 and 6
        for K in (4, 6):
            mk = lambda x, K=K: (sg(x[:K], LO, HI), sg(x[K:], 11, TRMAX))
            k, x = optimise(mk, 2 * K, mode)
            print(f"  free (flip,TR) x {K} blocks: kappa {k:.4g} flips {np.round(sg(x[:K], LO, HI), 1)} TRs {np.round(sg(x[K:], 11, TRMAX), 1)}")
        sys.stdout.flush()
    print(f"done {time.time()-t0:.0f}s")
