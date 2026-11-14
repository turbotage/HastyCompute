#!/usr/bin/env python3
"""Metric vs fixed point, MRI-like regime (peaked density, density-limited conditioning). CPU, ~10 min.

Same multi-set toy as metric_fixedpoint.py (joint soft threshold over T sets in Haar coordinates), but with the
sampling density nu ~ 1/(1 + |k|^2/k0^2), k0 = 4 (≈ 100x from centre to edge) and more shots per set, so that the
data step is limited by the density (as in the real scan, where a circulant cuts the iteration count by ~10x).

1. Fixed points x_P of T_P(x) = D(x - alpha_P P grad f(x)) for P in {I, set, circ, circset}, computed to a residual
   ||x - T_P(x)|| <= 1e-10 ||x|| with the fixed-point-preserving z-iteration (any invertible B keeps the fixed
   points; B = (I - J) + beta C J with the circulant C). Error to the truth over a lambda sweep.
2. Displacement formula: x_P - x_I = -(I - J (I - alpha_P P H))^{-1} J (alpha_P P - alpha_I) grad f(x_I), with J the
   Jacobian of D at the reference pre-denoiser point (local version of the secant identity), solved by GMRES;
   compared with the measured displacement.
3. Speed: error to the truth vs iterations for PnP-ISTA (P = I), P2nP with each fixed metric, and the accelerated
   reference iteration (converges to x_I).
Writes results/metric_fixedpoint_v2.json.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from scipy.sparse.linalg import LinearOperator, gmres  # noqa: E402

import experiments as ex  # noqa: E402
import metric_fixedpoint as mf  # noqa: E402

N, T = ex.N, mf.T
K0 = 4.0
mf.M_SETS = [1200, 1800, 2400, 3600]


def setup(seed=1):
    rng = np.random.default_rng(seed)
    xs = mf.phantom_sets()
    S = ex.make_coils(N, ex.NC)
    nu = ex.density(N, k0=K0)
    ops = [ex.Op(S, ex.draw_counts(nu, m, rng)) for m in mf.M_SETS]
    for op in ops:
        op.L = 1.0
    Lg = max(ex.power_norm(op.H) for op in ops)
    for op in ops:
        op.L = Lg
    sig = mf.SNR * np.sqrt(np.mean([op.data_rms(x) ** 2 for op, x in zip(ops, xs)]))
    nrng = np.random.default_rng(seed + 100)
    b = np.array([op.H(x) + op.AT_noise(sig, nrng) for op, x in zip(ops, xs)])
    return dict(xs=xs, S=S, ops=ops, b=b, sig=sig, nu=nu)


def Tmap(P, Mt, x, lam):
    return mf.D(x - Mt["apply"](mf.grad(P, x)), lam)


def fixed_point(P, Mt, lam, C, iters=60000, tol=1e-10, beta=0.5, accel_ok=True):
    """Fixed point of T_P. For P = I or per-set scalars: z-iteration on Phi_P with B = (I - J) + beta C J / alpha_P
    (relative to the map's own step; guarded: falls back to plain iteration if the residual grows). For the
    circulant metrics (already preconditioned): plain iteration."""
    def resid(x):
        return np.linalg.norm(x - Tmap(P, Mt, x, lam)) / np.linalg.norm(x)

    if accel_ok:
        x = mf.D(P["b"].copy(), lam)
        z = x - Mt["apply"](mf.grad(P, x))
        best = np.inf
        for k in range(iters):
            xz = mf.D(z, lam)
            F = z - (xz - Mt["apply"](mf.grad(P, xz)))
            Jf = mf.jvp(z, F, lam)
            z = z - ((F - Jf) + beta * C["apply"](Jf) / Mt["alpha"])
            if k % 25 == 0:
                xz = mf.D(z, lam)
                r = resid(xz)
                if not np.isfinite(r) or r > 10 * best:
                    print(f"    (accelerated fixed point diverged at it {k}, residual {r:.1e}: plain iteration)", flush=True)
                    break
                best = min(best, r)
                if r < tol:
                    return xz, z, k, r
    x = mf.D(P["b"].copy(), lam)
    for k in range(iters):
        z = x - Mt["apply"](mf.grad(P, x))
        xn = mf.D(z, lam)
        if k % 25 == 0:
            r = np.linalg.norm(xn - x) / np.linalg.norm(xn)
            if r < tol:
                return xn, z, k, resid(xn)
        x = xn
    return x, z, iters, resid(x)


def predicted_displacement(P, MI, Mt, lam, xI, zI):
    g = mf.grad(P, xI)
    rhs = -mf.jvp(zI, Mt["apply"](g) - MI["apply"](g), lam)
    shp = rhs.shape

    def mv(v):
        v = v.reshape(shp)
        return (v - mf.jvp(zI, v - Mt["apply"](mf.Hs(P, v)), lam)).ravel()

    A = LinearOperator((rhs.size, rhs.size), matvec=mv, dtype=float)
    d, info = gmres(A, rhs.ravel(), rtol=1e-8, restart=200, maxiter=2000)
    return d.reshape(shp), info


def main():
    t0 = time.time()
    P = setup()
    xs = P["xs"]
    kinds = ["I", "set", "circ", "circset"]
    mets = {k: mf.make_metric(P, k) for k in kinds}
    C = mets["circ"]
    print(f"setup {time.time() - t0:.1f}s; alpha {[round(mets[k]['alpha'], 4) for k in kinds]}; per-set 1/||H_t|| "
          f"{[round(1 / ex.power_norm(op.H), 3) for op in P['ops']]}", flush=True)
    lams = [6.25e-5, 1.25e-4, 2.5e-4, 5e-4, 1e-3]
    res = dict(k0=K0, M_sets=mf.M_SETS, sweep={}, disp={}, speed={})
    fps = {}
    for lam in lams:
        row = {}
        for k in kinds:
            x, z, it, r = fixed_point(P, mets[k], lam, C, accel_ok=k in ("I", "set"))
            fps[(k, lam)] = (x, z)
            row[k] = dict(err=mf.rel(x, xs), iters=it, resid=float(r))
        res["sweep"][lam] = row
        print(f"lam {lam:.2e}: " + "  ".join(f"{k} err {row[k]['err']:.4f} (resid {row[k]['resid']:.0e}, {row[k]['iters']} it)" for k in kinds), flush=True)
    best = {k: min(lams, key=lambda l: res["sweep"][l][k]["err"]) for k in kinds}
    res["best"] = {k: dict(lam=best[k], err=res["sweep"][best[k]][k]["err"]) for k in kinds}
    print("best over lambda:", {k: (best[k], round(res['best'][k]['err'], 4)) for k in kinds}, flush=True)
    lam0 = best["I"]
    xI, zI = fps[("I", lam0)]
    for k in kinds[1:]:
        xP = fps[(k, lam0)][0]
        meas = xP - xI
        pred, info = predicted_displacement(P, mets["I"], mets[k], lam0, xI, zI)
        lc = mf.leak_constants(P, mets[k], zI, lam0)
        res["disp"][k] = dict(measured=mf.rel(xP, xI), predicted=float(np.linalg.norm(pred) / np.linalg.norm(xI)),
                              pred_error=float(np.linalg.norm(pred - meas) / max(np.linalg.norm(meas), 1e-300)), gmres=int(info),
                              err_I=mf.rel(xI, xs), err_P=mf.rel(xP, xs), cos_to_truth=float(np.sum(meas * (xs - xI)) / (np.linalg.norm(meas) * np.linalg.norm(xs - xI))), **lc)
        d = res["disp"][k]
        print(f"  {k}: |x_P - x_I|/|x_I| measured {d['measured']:.4e}, predicted {d['predicted']:.4e} (relative prediction error {d['pred_error']:.2e}, gmres {info}); "
              f"err to truth {d['err_I']:.4f} -> {d['err_P']:.4f}; cos(displacement, x* - x_I) {d['cos_to_truth']:+.3f}; ell {d['ell']:.3f}, ||[J,P]|| {d['comm']:.3e}", flush=True)
    # speed: error to the truth vs iterations at lam0
    n_it = 600
    marks = (0, 10, 25, 50, 100, 200, 400, n_it - 1)
    for k in kinds:
        x = mf.D(P["b"].copy(), lam0)
        e_truth, e_own = [], []
        xf = fps[(k, lam0)][0]
        for i in range(n_it):
            x = Tmap(P, mets[k], x, lam0)
            e_truth.append(mf.rel(x, xs))
            e_own.append(mf.rel(x, xf))
        res["speed"][f"pnp_{k}"] = dict(truth=e_truth[::5], own=e_own[::5])
        print(f"  P2nP {k:8s}: err to truth at it " + ", ".join(f"{i}: {e_truth[i]:.4f}" for i in marks) + f" | to own fp at end {e_own[-1]:.1e}", flush=True)
    for k in ["circ", "set"]:
        x = mf.D(P["b"].copy(), lam0)
        z = x - mets["I"]["apply"](mf.grad(P, x))
        e_truth, e_ref = [], []
        for i in range(n_it):
            xz = mf.D(z, lam0)
            F = z - (xz - mets["I"]["apply"](mf.grad(P, xz)))
            Jf = mf.jvp(z, F, lam0)
            z = z - ((F - Jf) + 0.5 * mets[k]["apply"](Jf) / mets["I"]["alpha"])
            xz = mf.D(z, lam0)
            e_truth.append(mf.rel(xz, xs))
            e_ref.append(mf.rel(xz, xI))
        res["speed"][f"acc_{k}"] = dict(truth=e_truth[::5], ref=e_ref[::5])
        print(f"  accel ({k:4s} B, beta 0.5): err to truth at it " + ", ".join(f"{i}: {e_truth[i]:.4f}" for i in marks) + f" | to x_I at end {e_ref[-1]:.1e}", flush=True)
    with open(HERE / "results" / "metric_fixedpoint_v2.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
