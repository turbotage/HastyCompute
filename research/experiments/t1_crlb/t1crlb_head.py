"""Head T1 CRLB (user 2026-11-27: head, T1 itself, prefers many modulation levels). Question: what do many-level schedules
and a flip floor cost against the bang-bang optimum? Same model as t1crlb2.py, head tissues, B1 true in {1.0, 0.8}."""
import sys, time
import numpy as np
from scipy.optimize import minimize
import t1crlb2 as T

T.TISSUE[:] = [("scalp fat", 0.29, 0.5), ("white matter", 0.83, 2.0), ("grey matter", 1.33, 2.0), ("blood", 1.75, 1.0), ("CSF", 4.2, 0.5)]
T.B1_SET[:] = [1.0, 0.8]
KW = dict(tissues=T.TISSUE, b1_set=T.B1_SET)


def sc(per, **kw):
    return T.score(per, **{**KW, **kw})


def opt(make, K, lo, hi=30.0, starts=4, seed=1, **kw):
    f = lambda x: np.log(sc(make(T.sigm(x, lo, hi)), **kw))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        r = minimize(f, rng.normal(0, 1.2, K), method="L-BFGS-B", options=dict(maxiter=100, eps=1e-4))
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.fun)), T.sigm(best.x, lo, hi)


if __name__ == "__main__":
    t0 = time.time()
    free = ("rho", "T1", "B1", "eta")
    print("head tissues", [(n, t) for n, t, w in T.TISSUE], "B1", T.B1_SET)
    print("\n== 1. IR, flip floor sweep, K=12 pieces, B1+eta free")
    for tp in (3.0, 6.0):
        for lo in (1.0, 3.0, 5.0, 8.0):
            n = T.n_ir(tp)
            k, a = opt(lambda x: [("ir", T.piecewise(n, x))], 12, lo)
            print(f"  T_p {tp} s floor {lo:3.0f}°: kappa {k:.4g}  distinct levels >floor+1°: {int(np.sum(a > lo + 1))}/12  angles {np.round(a,1)}  [{time.time()-t0:.0f}s]")
            sys.stdout.flush()

    print("\n== 2. many-level VFA (no prep), equal blocks, B1 free (eta absent)")
    fr = ("rho", "T1", "B1")
    for K in (3, 4, 6, 12, 24):
        ang = np.linspace(2, 30, K)
        per = [("none", np.concatenate([np.full(max(1, 2400 // K), np.deg2rad(a)) for a in ang]))]
        print(f"  K={K:2d} equally spaced 2..30°: kappa {sc(per, free=fr):.4g}")
    for K in (4, 12):
        k, a = opt(lambda x: [("none", np.concatenate([np.full(max(1, 2400 // K), np.deg2rad(v)) for v in x]))], K, 2.0, free=fr)
        print(f"  K={K:2d} optimised (floor 2°): kappa {k:.4g} angles {np.round(np.sort(a),1)}")
        sys.stdout.flush()

    print("\n== 3. IR T_p 4 s with a smooth many-level schedule (24 pieces, floor 3°), B1+eta free, vs sinusoidal/random-smooth fixed designs")
    n = T.n_ir(4.0)
    k, a = opt(lambda x: [("ir", T.piecewise(n, x))], 24, 3.0, starts=3)
    print(f"  optimised 24 pieces: kappa {k:.4g}")
    rng = np.random.default_rng(5)
    for name, f in [("linear ramp 3->25", lambda t: 3 + 22 * t), ("sine 3..25 (3 periods)", lambda t: 14 + 11 * np.sin(2 * np.pi * 3 * t)),
                    ("random smooth", None)]:
        t = np.linspace(0, 1, n)
        if f is None:
            vals = np.convolve(rng.uniform(3, 25, n + 200), np.ones(100) / 100, mode="valid")[:n]
            vals = 3 + (vals - vals.min()) / (vals.max() - vals.min()) * 22
        else:
            vals = f(t)
        print(f"  {name:26s} kappa {sc([('ir', np.deg2rad(vals))], free=free):.4g}")
    print(f"done {time.time()-t0:.0f}s")
