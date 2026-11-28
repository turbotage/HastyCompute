"""CRLB comparison of T1-encoding schedules, v2 (research/t1_ivim_dti_sequences.md §2).

Changes vs t1crlb.py: the prep efficiency eta (inversion or saturation) is a FREE nuisance parameter (like B1 it varies
in the thorax), parameters of all Fisher columns and all tissues are propagated in one vectorised pass, schedules are
lists of periods (prep type + piecewise-constant flip angles), bounded optimisation.

Model: ideal RF spoiling, TE decay absorbed in rho, white noise sigma = 1 per readout (= per TR), every readout an
independent measurement of the voxel (incoherent sampling). Only RELATIVE numbers between schedules are meaningful.
  pulse:   s_n = rho sin(B a_n) Mz ; Mz+ = Mz cos(B a_n) ; recovery Mz' = 1 + (Mz+ - 1) exp(-TR/T1)
  IR:      Mz -> -eta Mz, then T_GAP relaxation ;  SR: Mz -> (1-eta) Mz ;  none: nothing
kappa = sqrt(CRLB_T1)/T1 for the whole scan at SNR_TR = 1 (real precision = kappa / SNR_TR). Geometric mean over tissues.
"""
import sys
import numpy as np
from scipy.optimize import minimize

TR = 11e-3
T_SCAN = 480.0
T_GAP = 15e-3
NONID = 1e3      # kappa reported for structurally non-identifiable T1
ETA0 = {"ir": 0.95, "sr": 0.95, "none": 1.0}

TISSUE = [("fat", 0.29, 1.0), ("liver", 0.80, 1.0), ("myocardium", 1.20, 2.0), ("blood", 1.75, 2.0)]
B1_SET = [1.0, 0.7]          # true flip-angle scale in the voxel


def _prep(m, prep, eta, Eg):
    if prep == "ir":
        return -eta * Eg * m + (1.0 - Eg)
    if prep == "sr":
        return (1.0 - eta) * m
    return m


def sim_super(rho, T1, B1, eta, periods, tr=TR):
    """all args arrays (B,). periods: list of (prep, alphas[rad]) played in order, repeated. Returns signals (n_tot, B)
    and the duration of one super-period (s)."""
    E = np.exp(-tr / T1)
    Eg = np.exp(-T_GAP / T1)

    def run(m0):
        m = m0
        out = []
        for prep, al in periods:
            m = _prep(m, prep, eta, Eg)
            for a in al:
                ba = B1 * a
                out.append(rho * np.sin(ba) * m)
                m = 1.0 + (m * np.cos(ba) - 1.0) * E
        return out, m

    _, b = run(np.zeros_like(rho))
    _, ab = run(np.ones_like(rho))
    a = ab - b
    m0 = b / (1.0 - a)
    out, _ = run(m0)
    dur = sum(len(al) * tr + (T_GAP if p == "ir" else 0.0) for p, al in periods)
    return np.array(out), dur


def kappas(periods, free=("rho", "T1", "B1", "eta"), tissues=TISSUE, b1_set=B1_SET, eta_override=None):
    """kappa for each (tissue, B1): dict[(name,B1)] -> kappa"""
    names = ["rho", "T1", "B1", "eta"]
    idx = [names.index(n) for n in free]
    cases = [(n, T1, B1) for n, T1, _ in tissues for B1 in b1_set]
    nc = len(cases)
    p = len(idx)
    prep0 = periods[0][0]
    cols = []
    base = np.array([[1.0, T1, B1, (eta_override if eta_override is not None else ETA0[prep0])] for _, T1, B1 in cases]).T  # (4, nc)
    # batch: for each free param, plus and minus
    th = []
    hs = []
    for i in idx:
        h = 1e-6 * np.maximum(np.abs(base[i]), 1e-3)
        tp, tm = base.copy(), base.copy()
        tp[i] += h
        tm[i] -= h
        th += [tp, tm]
        hs.append(h)
    TH = np.concatenate(th, axis=1)  # (4, 2*p*nc)
    S, dur = sim_super(TH[0], TH[1], TH[2], TH[3], periods)
    out = {}
    D = np.zeros((S.shape[0], p, nc))
    for k in range(p):
        sp = S[:, (2 * k) * nc:(2 * k + 1) * nc]
        sm = S[:, (2 * k + 1) * nc:(2 * k + 2) * nc]
        D[:, k, :] = (sp - sm) / (2 * hs[k])
    reps = T_SCAN / dur
    j = free.index("T1")
    scale = np.array([base_scale for base_scale in [1.0] * p])
    for c, (n, T1, B1) in enumerate(cases):
        # relative parameters (theta_i / theta_i0) so kappa = sqrt(var of relative T1) and J is well scaled
        col_scale = np.array([base[i, c] for i in idx])
        Dc = D[:, :, c] * col_scale
        J = Dc.T @ Dc * reps
        w, V = np.linalg.eigh(J)
        tol = 1e-11 * w.max()
        null = w <= tol
        if np.sum(V[j, null] ** 2) > 1e-6:
            out[(n, B1)] = NONID          # T1 has weight on a null direction: not identifiable
        else:
            keep = ~null
            C = (V[:, keep] / w[keep]) @ V[:, keep].T
            out[(n, B1)] = float(np.sqrt(C[j, j]))
    return out


def score(periods, **kw):
    k = kappas(periods, **kw)
    w = {n: wt for n, _, wt in TISSUE}
    num = sum(w[n] * np.log(v) for (n, B1), v in k.items())
    den = sum(w[n] for (n, B1) in k)
    return float(np.exp(num / den))


def summary(periods, **kw):
    k = kappas(periods, **kw)
    g = lambda n: float(np.exp(np.mean([np.log(k[(n, b)]) for b in B1_SET])))
    return {n: g(n) for n, _, _ in TISSUE}


# ---- design builders
def n_ir(tp):
    return int(np.floor((tp - T_GAP) / TR))


def n_sr(tp):
    return int(np.floor(tp / TR))


def piecewise(n, vals_deg):
    K = len(vals_deg)
    edges = np.linspace(0, n, K + 1).astype(int)
    a = np.empty(n)
    for k in range(K):
        a[edges[k]:edges[k + 1]] = np.deg2rad(vals_deg[k])
    return a


def sigm(x, lo=1.0, hi=30.0):
    return lo + (hi - lo) / (1.0 + np.exp(-x))


def optimise(make, K, x0=None, label="", **kw):
    """make(angles_deg[K]) -> periods ; L-BFGS-B on logit-parametrised angles"""
    f = lambda x: np.log(score(make(sigm(x)), **kw))
    best = None
    rng = np.random.default_rng(1)
    starts = [x0] if x0 is not None else [rng.normal(0.0, 1.2, K) for _ in range(6)]
    for s0 in starts:
        r = minimize(f, s0, method="L-BFGS-B", options=dict(maxiter=100, eps=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.fun)), sigm(best.x)


if __name__ == "__main__":
    import time
    t0 = time.time()
    SEC = sys.argv[1] if len(sys.argv) > 1 else "ABCDE"
    print(f"TR {TR*1e3:.1f} ms, scan {T_SCAN:.0f} s, free = rho,T1,B1,eta ; B1 true in {B1_SET}; kappa geometric mean over tissues")
    print("tissues:", [(n, T1) for n, T1, w in TISSUE])

    if "A" in SEC:
        print("\n== A. steady-state VFA (no prep), 200 TR per angle, free rho,T1,B1 (eta absent)")
        for ang in [(3, 15), (3, 8, 15), (2, 5, 10, 18), (2, 4, 8, 14, 22)]:
            per = [("none", np.concatenate([np.full(200, np.deg2rad(a)) for a in ang]))]
            print(f"  {str(ang):22s} kappa {score(per, free=('rho','T1','B1')):.4g}   (B1 known: {score(per, free=('rho','T1')):.4g})")

    if "B" in SEC:
        print("\n== B. fixed flip, one prep per period; eta,B1,rho free")
        for prep, nfun in (("ir", n_ir), ("sr", n_sr)):
            print(f"  prep={prep}")
            for tp in [0.5, 1.0, 2.0, 3.0, 4.0]:
                row = [score([(prep, np.full(nfun(tp), np.deg2rad(a)))]) for a in [4, 8, 12, 16, 22, 30]]
                print(f"    T_p {tp:3.1f} s: " + "  ".join(f"{a:2d}deg {v:8.3g}" for a, v in zip([4, 8, 12, 16, 22, 30], row)))

    if "C" in SEC:
        print("\n== C. optimised piecewise flip angle in one period, eta,B1,rho free")
        for prep, nfun in (("ir", n_ir), ("sr", n_sr)):
            for tp in [1.0, 2.0, 3.0]:
                for K in [1, 2, 4, 8]:
                    n = nfun(tp)
                    k, ang = optimise(lambda x: [(prep, piecewise(n, x))], K)
                    print(f"  {prep} T_p {tp:3.1f} s K={K}: kappa {k:8.4g}  angles {np.round(ang, 1)}   [{time.time()-t0:.0f}s]")
                    sys.stdout.flush()

    if "D" in SEC:
        print("\n== D. alternating IR period + SR period (super-period), optimised K=4 each")
        for tp_ir, tp_sr in [(2.0, 1.0), (3.0, 1.0)]:
            K = 4
            make = lambda x: [("ir", piecewise(n_ir(tp_ir), x[:K])), ("sr", piecewise(n_sr(tp_sr), x[K:]))]
            k, ang = optimise(make, 2 * K)
            print(f"  IR {tp_ir} s + SR {tp_sr} s: kappa {k:.4g} angles {np.round(ang, 1)}")
            sys.stdout.flush()

    if "E" in SEC:
        print("\n== E. cost of nuisances, IR T_p 2 s, fixed 12 deg (and optimum 12.6)")
        per = [("ir", np.full(n_ir(2.0), np.deg2rad(12.0)))]
        for free in [("rho", "T1"), ("rho", "T1", "B1"), ("rho", "T1", "eta"), ("rho", "T1", "B1", "eta")]:
            print(f"  free {free}: kappa {score(per, free=free):.4g}   per tissue {({k: round(v, 3) for k, v in summary(per, free=free).items()})}")
        print("  same, eta_true = 0.85 (inversion worse than assumed):")
        print(f"    {score(per, free=('rho','T1','B1','eta'), eta_override=0.85):.4g}")
    print(f"done {time.time()-t0:.0f} s")


def extra():
    """F: optimum schedules vs nuisance set; G: best VFA"""
    import time
    t0 = time.time()
    print("\n== F. IR T_p 3 s, optimised K=8/12 flip schedule vs which nuisances are free")
    for free in [("rho", "T1"), ("rho", "T1", "eta"), ("rho", "T1", "B1"), ("rho", "T1", "B1", "eta")]:
        for K in (8, 12):
            n = n_ir(3.0)
            k, ang = optimise(lambda x: [("ir", piecewise(n, x))], K, free=free)
            per = [("ir", piecewise(n, ang))]
            print(f"  free {free} K={K}: kappa {k:.4g}  per tissue {({a: round(v, 3) for a, v in summary(per, free=free).items()})}  angles {np.round(ang, 1)}  [{time.time()-t0:.0f}s]")
            sys.stdout.flush()
    print("\n== G. best VFA (no prep), K angle blocks of equal time, B1 free")
    for K in (3, 4, 6):
        k, ang = optimise(lambda x: [("none", np.concatenate([np.full(200, np.deg2rad(a)) for a in x]))], K, free=("rho", "T1", "B1"))
        print(f"  K={K}: kappa {k:.4g} angles {np.round(np.sort(ang), 1)}")
    print("\n== H. IR period length, K=8, B1 + eta free")
    for tp in (1.5, 2.0, 3.0, 4.0, 6.0):
        n = n_ir(tp)
        k, ang = optimise(lambda x: [("ir", piecewise(n, x))], 8)
        print(f"  T_p {tp} s: kappa {k:.4g} angles {np.round(ang, 1)}")
        sys.stdout.flush()
