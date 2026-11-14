#!/usr/bin/env python3
"""Does the metric move the PnP fixed point, and can acceleration keep it? (multi-set toy, CPU, minutes)

Model: T sets (frames) of a 64x64 Shepp-Logan with a small moving blob, each with its own i.i.d. shots from the same
variable density (different shot counts M_t, so the per-set curvatures differ like 169..211 readouts or
sets × respiratory bins in the real scan), shared coils, no density compensation. Data term separable over sets,
H = blockdiag(H_t), all normalised by one common L (||H|| = 1).
Denoiser coupling the sets (not a proximal assumption, eta = 0 secant structure): orthonormal Haar, joint soft
threshold of every coefficient's vector across the sets (coarse band untouched).
  D(z)_t = W^T [ (1 - lam/||c||)_+ c ]_t,   c = W z (per coefficient, the T-vector over sets).

Metrics P (fixed; step alpha_P = 1/lambda_max(P^1/2 H P^1/2)):
  I       global scalar (reference PnP-ISTA)
  set     per-set scalars tau_t = 1/||H_t|| (the deterministic analogue of ZMSHR1's per-set tau_t)
  circ    one circulant (density of the pooled shots), shared by all sets (recon_p2np --circ-share encode)
  circset one circulant per set (its own density)
For each: fixed point x_P (long run), error to the truth over a lambda sweep, the leak constant
  ell_P = ||Pi P Pi_perp|| / lambda_min(Pi P Pi |T)   (Pi = projector onto the active coefficients x sets),
and the commutator ||[J_D, P]|| at the fixed point.
Fixed-point-preserving acceleration (z-iteration on the reference map Phi(z) = D(z) - grad f(D(z))):
  z_{k+1} = z_k - B_k (z_k - Phi(z_k)),   B_k = (I - J_k) + (alpha_P/alpha_I) P J_k,
  J_k = Jacobian of D at z_k (one JVP per iteration). Its fixed points are those of P = I; the tangent block of its
  linearisation is I - beta P_00 H_00 (the metric's rate) without the leak.

Usage: rustlib/.venv/bin/python -I research/mathematics_of_paper/pnp_restricted/metric_fixedpoint.py [--quick]
Writes results/metric_fixedpoint.json.
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

import experiments as ex  # noqa: E402

N, NC = ex.N, ex.NC
QUICK = "--quick" in sys.argv
T = 4
M_SETS = [260, 390, 520, 780]
SNR = 0.01


def phantom_sets():
    x0 = ex.shepp_logan(N)
    t = (np.arange(N) + 0.5) / N * 2 - 1
    X, Y = np.meshgrid(t, -t)
    out = []
    for k in range(T):
        cx, cy = -0.3 + 0.2 * k, 0.15
        blob = 0.25 * (((X - cx) / 0.08) ** 2 + ((Y - cy) / 0.08) ** 2 <= 1)
        out.append(x0 + blob)
    return np.array(out)


def setup(seed=1):
    rng = np.random.default_rng(seed)
    xs = phantom_sets()
    S = ex.make_coils(N, NC)
    nu = ex.density(N)
    ops = [ex.Op(S, ex.draw_counts(nu, m, rng)) for m in M_SETS]
    # common normalisation: one L for the stack (the curvatures differ between sets)
    for op in ops:
        op.L = 1.0
    Lg = max(ex.power_norm(op.H) for op in ops)
    for op in ops:
        op.L = Lg
    sig = SNR * np.sqrt(np.mean([op.data_rms(x) ** 2 for op, x in zip(ops, xs)]))
    nrng = np.random.default_rng(seed + 100)
    b = np.array([op.H(x) + op.AT_noise(sig, nrng) for op, x in zip(ops, xs)])
    return dict(xs=xs, S=S, ops=ops, b=b, sig=sig, nu=nu)


def Hs(P, x):
    return np.array([op.H(x[t]) for t, op in enumerate(P["ops"])])


def grad(P, x):
    return Hs(P, x) - P["b"]


# ---- joint soft threshold over the sets in Haar coordinates ---------------------------------------------------
def D(z, lam):
    c = ex.haar(z)
    r = np.sqrt(np.sum(c * c, 0))
    f = np.where(r > lam, 1 - lam / np.maximum(r, 1e-300), 0.0)
    f[ex.APPROX] = 1.0
    return ex.ihaar(c * f[None])


def jvp(z, v, lam):
    """J_D(z) v"""
    c, w = ex.haar(z), ex.haar(v)
    r = np.sqrt(np.sum(c * c, 0))
    act = r > lam
    rr = np.maximum(r, 1e-300)
    ch = c / rr[None]
    proj = np.sum(ch * w, 0)
    out = np.where(act[None], w - (lam / rr)[None] * (w - ch * proj[None]), 0.0)
    out[:, ex.APPROX] = w[:, ex.APPROX]
    return ex.ihaar(out)


def active_mask(z, lam):
    c = ex.haar(z)
    r = np.sqrt(np.sum(c * c, 0))
    m = r > lam
    m[ex.APPROX] = True
    return m


def proj(m, v):
    """orthogonal projector onto the active coefficients (all sets) — the tangent T of the joint threshold"""
    c = ex.haar(v)
    return ex.ihaar(c * m[None])


# ---- metrics ------------------------------------------------------------------------------------------------------
def circ_from_counts(P, counts):
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k)
    ker = np.exp(-(KX ** 2 + KY ** 2) / (2 * 1.5 ** 2))
    ker /= ker.sum()
    dens = np.real(np.fft.ifft2(np.fft.fft2(counts) * np.fft.fft2(ker)))
    cover = np.mean(np.sum(np.abs(P["S"]) ** 2, 0))
    return 1.0 / (np.maximum(dens, 0) * cover / P["ops"][0].L + 0.02)


def make_metric(P, kind):
    if kind == "I":
        ap = [None] * T
    elif kind == "set":
        ap = [1.0 / ex.power_norm(op.H) for op in P["ops"]]
    elif kind == "circ":
        pooled = sum(op.c for op in P["ops"]) / T
        p = circ_from_counts(P, pooled)
        ap = [p] * T
    elif kind == "circset":
        ap = [circ_from_counts(P, op.c) for op in P["ops"]]
    else:
        raise ValueError(kind)

    def raw(x):
        out = np.empty_like(x)
        for t in range(T):
            a = ap[t]
            if a is None:
                out[t] = x[t]
            elif np.isscalar(a):
                out[t] = a * x[t]
            else:
                out[t] = np.real(np.fft.ifft2(np.fft.fft2(x[t], norm="ortho") * a, norm="ortho"))
        return out

    def half(x):
        out = np.empty_like(x)
        for t in range(T):
            a = ap[t]
            if a is None:
                out[t] = x[t]
            elif np.isscalar(a):
                out[t] = np.sqrt(a) * x[t]
            else:
                out[t] = np.real(np.fft.ifft2(np.fft.fft2(x[t], norm="ortho") * np.sqrt(a), norm="ortho"))
        return out

    lmax = power_stack(lambda v: half(Hs(P, half(v))))
    alpha = 1.0 / lmax
    return dict(kind=kind, apply=lambda x: alpha * raw(x), raw=raw, alpha=alpha)


def power_stack(apply, iters=150, seed=0):
    v = np.random.default_rng(seed).standard_normal((T, N, N))
    lam = 0.0
    for _ in range(iters):
        w = apply(v)
        lam = np.linalg.norm(w) / np.linalg.norm(v)
        v = w / np.linalg.norm(w)
    return lam


# ---- solvers --------------------------------------------------------------------------------------------------------
def pnp(P, Mt, lam, iters, x0=None, tol=1e-11):
    x = D(P["b"].copy(), lam) if x0 is None else x0.copy()
    hist = []
    for k in range(iters):
        z = x - Mt["apply"](grad(P, x))
        xn = D(z, lam)
        d = np.linalg.norm(xn - x) / max(np.linalg.norm(xn), 1e-30)
        x = xn
        hist.append(d)
        if d < tol:
            break
    return x, z, hist


def accel(P, Mt, lam, iters, xref, alpha_ref, x0=None, beta=None):
    """z-iteration on the reference map (P = I, step alpha_ref) with B = (I - J) + beta P J"""
    beta = Mt["alpha"] / alpha_ref if beta is None else beta
    x = D(P["b"].copy(), lam) if x0 is None else x0.copy()
    z = x - alpha_ref * grad(P, x)
    err = []
    for k in range(iters):
        xz = D(z, lam)
        F = z - (xz - alpha_ref * grad(P, xz))
        Jf = jvp(z, F, lam)
        step = (F - Jf) + beta * Mt["raw"](Jf)
        z = z - step
        err.append(np.linalg.norm(D(z, lam) - xref) / np.linalg.norm(xref))
    return D(z, lam), err


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def leak_constants(P, Mt, z, lam):
    m = active_mask(z, lam)
    Pi = lambda v: proj(m, v)  # noqa: E731
    Pp = lambda v: v - Pi(v)  # noqa: E731
    R = Mt["raw"]
    # ||Pi R Pi_perp|| via power iteration on (Pi R Pp)^T (Pi R Pp) = Pp R Pi R Pp (R symmetric)
    s = np.sqrt(power_stack(lambda v: Pp(R(Pi(R(Pp(v)))))))
    # lambda_min(Pi R Pi | T) = lmax - lambda_max(lmax Pi - Pi R Pi) on T
    lm = power_stack(lambda v: Pi(R(Pi(v))))
    lo = lm - power_stack(lambda v: lm * Pi(v) - Pi(R(Pi(v))))
    # commutator with the denoiser's Jacobian: [J, R] is antisymmetric-ish; norm via power iteration on C^T C
    J = lambda v: jvp(z, v, lam)  # noqa: E731
    C = lambda v: J(R(v)) - R(J(v))  # noqa: E731
    Ct = lambda v: R(J(v)) - J(R(v))  # J, R symmetric: C^T = -C
    cn = np.sqrt(power_stack(lambda v: Ct(C(v)), iters=80))
    return dict(leak_norm=float(s), lmin_P00=float(lo), lmax_P00=float(lm), ell=float(s / max(lo, 1e-300)), comm=float(cn), d_frac=float(m.mean()))


def main():
    t0 = time.time()
    P = setup()
    xs = P["xs"]
    lams = [0.00025, 0.0005, 0.001, 0.002] if QUICK else [0.000125, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.004]
    iters = 20000 if QUICK else 40000
    kinds = ["I", "set", "circ", "circset"]
    mets = {k: make_metric(P, k) for k in kinds}
    res = dict(M_sets=M_SETS, T=T, alphas={k: mets[k]["alpha"] for k in kinds}, sweep={}, set_taus=[1.0 / ex.power_norm(op.H) for op in P["ops"]])
    print(f"setup {time.time() - t0:.1f}s; alpha {res['alphas']}; per-set tau (1/||H_t||) {np.round(res['set_taus'], 3).tolist()}")
    fp = {}
    for lam in lams:
        row = {}
        for k in kinds:
            x, z, h = pnp(P, mets[k], lam, iters)
            fp[(k, lam)] = (x, z)
            row[k] = dict(err=rel(x, xs), iters=len(h), resid=h[-1])
        res["sweep"][lam] = row
        print(f"lam {lam:.5f}: " + "  ".join(f"{k} err {row[k]['err']:.4f} ({row[k]['iters']} it, last rel. change {row[k]['resid']:.1e})" for k in kinds), flush=True)
    best = {k: min(lams, key=lambda l: res["sweep"][l][k]["err"]) for k in kinds}
    res["best"] = {k: dict(lam=best[k], err=res["sweep"][best[k]][k]["err"]) for k in kinds}
    print("best over lambda:", {k: (best[k], round(res['best'][k]['err'], 4)) for k in kinds})
    # leak constants at the reference lambda (best of P = I)
    lam0 = best["I"]
    res["lam0"] = lam0
    res["leak"] = {}
    for k in kinds[1:]:
        x, z = fp[(k, lam0)]
        res["leak"][k] = leak_constants(P, mets[k], z, lam0)
        res["leak"][k]["fp_dist_to_I"] = rel(x, fp[("I", lam0)][0])
        print(f"  {k}: ell {res['leak'][k]['ell']:.3f}, ||[J,P]|| {res['leak'][k]['comm']:.3e} (P00 in [{res['leak'][k]['lmin_P00']:.3e}, {res['leak'][k]['lmax_P00']:.3e}]), "
              f"fixed point vs P=I {res['leak'][k]['fp_dist_to_I']:.4f}, active {res['leak'][k]['d_frac']:.3f}", flush=True)
    # convergence speed at lam0: iterations of plain PnP per metric to reach 1e-4 of its own fixed point, and the
    # accelerated reference iteration (fixed point of P = I)
    xI = fp[("I", lam0)][0]
    res["speed"] = {}
    n_it = 400 if QUICK else 800
    for k in kinds:
        x, _, _ = pnp(P, mets[k], lam0, 1, tol=0)  # warm-up identical start
        errs = []
        xk = D(P["b"].copy(), lam0)
        xf = fp[(k, lam0)][0]
        for i in range(n_it):
            xk = D(xk - mets[k]["apply"](grad(P, xk)), lam0)
            errs.append(rel(xk, xf))
        res["speed"][f"pnp_{k}"] = errs[::10]
        res["speed"][f"pnp_{k}_it1e-3"] = next((i for i, e in enumerate(errs) if e < 1e-3), None)
    for k in kinds[1:]:
        for bscale in (1.0, 0.5, 0.25):
            beta = bscale * mets[k]["alpha"] / mets["I"]["alpha"]
            xa, errs = accel(P, mets[k], lam0, n_it, xI, mets["I"]["alpha"], beta=beta)
            tag = f"acc_{k}_b{bscale}"
            res["speed"][tag] = errs[::10]
            res["speed"][f"{tag}_it1e-3"] = next((i for i, e in enumerate(errs) if e < 1e-3), None)
            res["speed"][f"{tag}_final_vs_I"] = rel(xa, xI)
            print(f"  accel {k} beta x{bscale}: error to the P=I fixed point at it " + ", ".join(f"{i}: {errs[i]:.2e}" for i in (0, 50, 100, 200, 400, n_it - 1) if i < len(errs)), flush=True)
    for k in kinds:
        e = res["speed"][f"pnp_{k}"]
        print(f"  pnp {k}: error to its own fixed point at it " + ", ".join(f"{10 * i}: {e[i]:.2e}" for i in (0, 5, 10, 20, 40, len(e) - 1) if i < len(e)), flush=True)
    print("iterations to 1e-3 of the own fixed point:", {k: v for k, v in res["speed"].items() if k.endswith("it1e-3")})
    print("accelerated final vs P = I fixed point:", {k: round(v, 6) for k, v in res["speed"].items() if k.endswith("final_vs_I")})
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "metric_fixedpoint.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
