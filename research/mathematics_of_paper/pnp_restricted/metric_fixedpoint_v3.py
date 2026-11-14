#!/usr/bin/env python3
"""Follow-up of metric_fixedpoint_v2.py (same MRI-like multi-set toy):
1. Exactness of the displacement identity (Theorem thm:displacement) with the secant S = mean of J_D over the segment
   [z_I, z_P] (midpoint quadrature): (I - S(I - alpha_P P H)) Delta = -S (alpha_P P - alpha_I) grad f(x_I), by GMRES,
   against the measured Delta = x_P - x_I.
2. Speed at each method's own best lambda (extended lambda grid for the circulant metrics): error to the truth vs
   iterations for P2nP (I, set, circ) and for the fixed-point-preserving accelerator of the reference P = I.
Writes results/metric_fixedpoint_v3.json.
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

import metric_fixedpoint as mf  # noqa: E402
import metric_fixedpoint_v2 as v2  # noqa: E402

NQ = 64


def secant_apply(zI, zP, v, lam):
    acc = np.zeros_like(v)
    for q in range(NQ):
        t = (q + 0.5) / NQ
        acc += mf.jvp(zI + t * (zP - zI), v, lam)
    return acc / NQ


def exact_check(P, MI, Mt, lam, xI, zI, xP, zP):
    g = mf.grad(P, xI)
    rhs = -secant_apply(zI, zP, Mt["apply"](g) - MI["apply"](g), lam)
    shp = rhs.shape
    # secant identity itself: D(z_P) - D(z_I) = S (z_P - z_I)
    sec_err = np.linalg.norm(secant_apply(zI, zP, zP - zI, lam) - (xP - xI)) / np.linalg.norm(xP - xI)

    def mv(v):
        v = v.reshape(shp)
        return (v - secant_apply(zI, zP, v - Mt["apply"](mf.Hs(P, v)), lam)).ravel()

    A = LinearOperator((rhs.size, rhs.size), matvec=mv, dtype=float)
    d, info = gmres(A, rhs.ravel(), rtol=1e-9, restart=100, maxiter=300)
    d = d.reshape(shp)
    meas = xP - xI
    return dict(secant_identity_err=float(sec_err), pred_err=float(np.linalg.norm(d - meas) / np.linalg.norm(meas)), gmres=int(info),
                measured=float(np.linalg.norm(meas) / np.linalg.norm(xI)))


def run_curve(P, step, n_it, xs, lam):
    x = mf.D(P["b"].copy(), lam)
    out = []
    for i in range(n_it):
        x = mf.D(x - step(x), lam)
        out.append(mf.rel(x, xs))
    return out


def run_accel(P, MI, C, beta, n_it, xs, xref, lam):
    x = mf.D(P["b"].copy(), lam)
    z = x - MI["apply"](mf.grad(P, x))
    tr, rf = [], []
    for i in range(n_it):
        xz = mf.D(z, lam)
        F = z - (xz - MI["apply"](mf.grad(P, xz)))
        Jf = mf.jvp(z, F, lam)
        z = z - ((F - Jf) + beta * C["apply"](Jf) / MI["alpha"])
        xz = mf.D(z, lam)
        tr.append(mf.rel(xz, xs))
        rf.append(mf.rel(xz, xref))
    return tr, rf


def main():
    t0 = time.time()
    P = v2.setup()
    xs = P["xs"]
    kinds = ["I", "set", "circ"]
    mets = {k: mf.make_metric(P, k) for k in kinds}
    C = mets["circ"]
    grids = {"I": [6.25e-5, 1.25e-4, 2.5e-4], "set": [1.25e-4, 2.5e-4, 5e-4], "circ": [1e-3, 2e-3, 4e-3]}
    res = dict(best={}, exact={}, speed={})
    fps = {}
    for k in kinds:
        for lam in grids[k]:
            x, z, it, r = v2.fixed_point(P, mets[k], lam, C, accel_ok=k in ("I", "set"))
            fps[(k, lam)] = (x, z)
            print(f"  {k} lam {lam:.2e}: err {mf.rel(x, xs):.4f} (resid {r:.0e})", flush=True)
        lb = min(grids[k], key=lambda l: mf.rel(fps[(k, l)][0], xs))
        res["best"][k] = dict(lam=lb, err=mf.rel(fps[(k, lb)][0], xs))
    print("best:", res["best"], flush=True)
    # 1. exactness at lambda = best of I, for set and circ (fixed points at that lambda)
    lamI = res["best"]["I"]["lam"]
    xI, zI = fps[("I", lamI)]
    for k in ["set", "circ"]:
        xP, zP, _, _ = v2.fixed_point(P, mets[k], lamI, C, accel_ok=k == "set")
        e = exact_check(P, mets["I"], mets[k], lamI, xI, zI, xP, zP)
        res["exact"][k] = e
        print(f"  displacement identity {k}: measured |Delta|/|x_I| {e['measured']:.3e}; secant identity error {e['secant_identity_err']:.1e}; "
              f"prediction error {e['pred_err']:.1e} (gmres {e['gmres']})", flush=True)
    # 2. speed at each method's own best lambda
    n_it = 800
    marks = (10, 50, 100, 200, 400, n_it - 1)
    for k in kinds:
        lam = res["best"][k]["lam"]
        Mt = mets[k]
        c = run_curve(P, lambda x: Mt["apply"](mf.grad(P, x)), n_it, xs, lam)
        res["speed"][f"pnp_{k}"] = dict(lam=lam, truth=c[::5])
        print(f"  P2nP {k:4s} (lam {lam:.1e}, limit err {res['best'][k]['err']:.4f}): err to truth at " + ", ".join(f"{i}: {c[i]:.4f}" for i in marks), flush=True)
    for beta in (0.5, 0.75):
        tr, rf = run_accel(P, mets["I"], C, beta, n_it, xs, xI, lamI)
        res["speed"][f"acc_circ_b{beta}"] = dict(lam=lamI, truth=tr[::5], ref=rf[::5])
        print(f"  accel ref I, circ B beta {beta} (lam {lamI:.1e}, limit err {res['best']['I']['err']:.4f}): err to truth at " + ", ".join(f"{i}: {tr[i]:.4f}" for i in marks) + f" | to x_I {rf[-1]:.1e}", flush=True)
    with open(HERE / "results" / "metric_fixedpoint_v3.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
