"""CRLB comparison of T1-encoding schedules for an RF-spoiled 3D GRE readout train (research/t1_ivim_dti_sequences.md).

Question: at the same scan time and the same TR (= the same ADC duty), which signal modulation gives the best T1
precision when T1 is fitted jointly with rho, the flip-angle scale B1 and (optionally) the preparation efficiency?

Model (ideal spoiling, TE decay absorbed in rho, noise sigma = 1 per TR per voxel, i.e. every readout is one incoherent
"measurement" of the voxel; true for a fully incoherent sampling pattern, so only RELATIVE numbers between schedules
are meaningful):
  Mz before pulse n; pulse: Mz+ = Mz cos(B a_n), signal s_n = rho sin(B a_n) Mz;  recovery Mz' = 1 + (Mz+ - 1) exp(-TR/T1)
  prep at the start of every period: inversion (Mz -> -eta Mz, then T_gap of relaxation), saturation (Mz -> (1-eta) Mz),
  or none (cyclic flip-angle schedule).
The periodic steady state is solved exactly (the period map is affine in Mz).
Output: kappa = sqrt(CRLB_T1)/T1 over a fixed scan time and SNR_TR = 1; real sigma_T1/T1 = kappa / SNR_TR.
"""
import numpy as np

TR = 11e-3          # s, RF centre to RF centre (readout 9.5 ms + RF + short tail; no encode block)
T_SCAN = 480.0      # s
T_GAP = 15e-3       # s, adiabatic inversion pulse + crusher before the first readout
ETA_INV = 0.95
ETA_SAT = 0.95


def period_signals(theta, alphas, tr, prep, t_gap=T_GAP):
    """theta: (4, B) = rho, T1, B1, eta ; alphas: (n,) rad per TR; returns signals (n, B)"""
    rho, T1, B1, eta = theta
    E = np.exp(-tr / T1)
    Eg = np.exp(-t_gap / T1)

    def run(m0):
        m = m0
        out = []
        for a in alphas:
            out.append(rho * np.sin(B1 * a) * m)
            m = 1.0 + (m * np.cos(B1 * a) - 1.0) * E
        return np.array(out), m

    _, b = run(np.zeros_like(rho))
    _, ab = run(np.ones_like(rho))
    a = ab - b
    if prep == "ir":
        c, d = -eta * Eg, 1.0 - Eg + 0.0       # relax toward 1 during the gap
        c, d = -eta * Eg, (1.0 - Eg)
    elif prep == "sr":
        c, d = (1.0 - eta), 0.0
    else:
        c, d = 1.0, 0.0
    m_start = (c * b + d) / (1.0 - c * a)
    s, _ = run(m_start)
    return s


def fisher(alphas, tr, prep, T1, B1=1.0, free=("rho", "T1", "B1"), eta=None, t_period=None):
    """Fisher information (per the whole scan) for theta=(rho,T1,B1,eta); returns CRLB std of T1 / T1 (kappa)."""
    if eta is None:
        eta = ETA_INV if prep == "ir" else ETA_SAT
    th0 = np.array([1.0, T1, B1, eta])
    names = ["rho", "T1", "B1", "eta"]
    idx = [names.index(n) for n in free]
    cols = []
    for i in idx:
        h = 1e-6 * max(abs(th0[i]), 1e-3)
        thp, thm = th0.copy(), th0.copy()
        thp[i] += h
        thm[i] -= h
        sp = period_signals(thp[:, None], alphas, tr, prep)[:, 0]
        sm = period_signals(thm[:, None], alphas, tr, prep)[:, 0]
        cols.append((sp - sm) / (2 * h))
    D = np.array(cols).T                                   # (n, p)
    J = D.T @ D
    n_per = len(alphas) * tr + (T_GAP if prep == "ir" else 0.0)
    n_period = T_SCAN / (t_period if t_period else n_per)
    J = J * n_period
    C = np.linalg.pinv(J)
    j = free.index("T1")
    return np.sqrt(C[j, j]) / T1


# tissue set for a thorax scan: (name, T1 s, weight)
TISSUE = [("fat", 0.29, 1.0), ("liver", 0.80, 1.0), ("myocardium", 1.20, 2.0), ("blood", 1.75, 2.0), ("blood+Gd", 0.40, 0.0)]


def score(alphas, tr, prep, B1=1.0, free=("rho", "T1", "B1"), tissues=TISSUE):
    """weighted mean of log kappa over the tissues (geometric mean of relative precision)"""
    num, den = 0.0, 0.0
    for _, T1, w in tissues:
        if w == 0:
            continue
        num += w * np.log(fisher(alphas, tr, prep, T1, B1, free))
        den += w
    return np.exp(num / den)


def per_tissue(alphas, tr, prep, B1=1.0, free=("rho", "T1", "B1")):
    return {n: fisher(alphas, tr, prep, T1, B1, free) for n, T1, _ in TISSUE}


def ir_design(t_period, alpha_deg, tr=TR, ramp=None):
    n = int(np.floor((t_period - T_GAP) / tr))
    a = np.full(n, np.deg2rad(alpha_deg))
    if ramp is not None:
        a = np.deg2rad(np.linspace(ramp[0], ramp[1], n))
    return a


def sr_design(t_period, alpha_deg, tr=TR):
    n = int(np.floor(t_period / tr))
    return np.full(n, np.deg2rad(alpha_deg))


def vfa_design(angles_deg, tr=TR, block=200):
    return np.concatenate([np.full(block, np.deg2rad(a)) for a in angles_deg])


if __name__ == "__main__":
    import sys
    from scipy.optimize import minimize

    free = ("rho", "T1", "B1")
    print("tissues:", [(n, T1) for n, T1, w in TISSUE if w > 0])
    print(f"TR {TR*1e3:.1f} ms, scan {T_SCAN:.0f} s; kappa = sqrt(CRLB_T1)/T1 at SNR_TR = 1 (smaller is better)\n")

    # 0. reference: B1 known (T1 + rho only) vs unknown
    print("== A. VFA steady-state (no prep), B1 unknown; angle sets (deg)")
    for ang in [(3, 15), (3, 8, 15), (2, 5, 10, 18), (4, 6, 8, 10, 12, 15, 20)]:
        al = vfa_design(ang)
        if len(ang) < 3:
            f = ("rho", "T1")
        else:
            f = free
        print(f"  {str(ang):28s} free={f}  kappa(geo-mean) {score(al, TR, 'none', free=f):.4g}   B1 known: {score(al, TR, 'none', free=('rho','T1')):.4g}")

    print("\n== B. continuous IR, fixed flip, B1 + rho free (eta fixed 0.95)")
    print("  T_period(s) alpha(deg) | kappa myo   blood   fat   liver | geo-mean")
    best = None
    for tp in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        for a in [2, 4, 6, 8, 12, 16, 20, 25]:
            al = ir_design(tp, a)
            k = per_tissue(al, TR, "ir")
            sc = score(al, TR, "ir")
            if best is None or sc < best[0]:
                best = (sc, tp, a)
            if a in (4, 8, 16):
                print(f"  {tp:5.1f}   {a:4d}     | {k['myocardium']:.4g} {k['blood']:.4g} {k['fat']:.4g} {k['liver']:.4g} | {sc:.4g}")
    print("  best fixed-flip IR:", best)

    print("\n== C. saturation recovery, fixed flip")
    best_sr = None
    for tp in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]:
        for a in [4, 8, 12, 16, 25]:
            al = sr_design(tp, a)
            sc = score(al, TR, "sr")
            if best_sr is None or sc < best_sr[0]:
                best_sr = (sc, tp, a)
    print("  best fixed-flip SR:", best_sr)

    print("\n== D. IR with optimised piecewise flip-angle schedule (K segments), B1 free")
    for tp in [2.0, 3.0, 4.0]:
        for K in [1, 2, 4, 8]:
            n = int(np.floor((tp - T_GAP) / TR))
            edges = np.linspace(0, n, K + 1).astype(int)

            def build(x):
                a = np.empty(n)
                for k in range(K):
                    a[edges[k]:edges[k + 1]] = np.deg2rad(np.clip(x[k], 0.5, 40))
                return a

            f = lambda x: np.log(score(build(x), TR, "ir"))
            r = minimize(f, np.full(K, 8.0), method="Nelder-Mead", options=dict(xatol=0.05, fatol=1e-4, maxiter=400))
            print(f"  T_period {tp:.1f} s, K={K}: kappa {np.exp(r.fun):.4g}  angles {np.round(r.x, 1)}")
            sys.stdout.flush()
