#!/usr/bin/env python3
"""Convergence of PnP / proximal schemes on a toy with 3D-centre-out-like k-space density (2026-11-10).

Problem: real 64x64 phantom, 4 coils, shots i.i.d. from nu(k) ~ 1/(1+|k|^2/k0^2)^2 (centre/edge density ~1e3, like
the 3D yarn at k-voxel scale), Haar-l1 prior. Estimator: the minimiser x* of F(x) = 1/2||Ax-y||^2 + lam ||W x||_1
(coarse band unpenalised), computed by a long ADMM run. Methods, measured in normal-operator applications (the cost
unit of a gradient pass):
  ista       x <- prox_{a lam}(x - a grad f), a = 1/L                         (estimator-preserving)
  fista      Nesterov momentum on ista                                        (estimator-preserving)
  pnp_fixed  x <- soft_theta(x - a grad f) with theta fixed (not a*lam): fixed point depends on a (Prop 6.1(4))
  admm_pcg   ADMM, x-step (H + rho I) x = A^H y + rho(z - u) by PCG with a circulant (density) preconditioner,
             n_in inner iterations; estimator-preserving whatever the preconditioner
  fista_dcw  FISTA on the density-weighted data term: a different estimator (minimiser of the weighted problem)
Usage: rustlib/.venv/bin/python -I research/experiments/convergence/rates_toy.py <paper dir with experiments.py> <out dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
import experiments as E  # noqa: E402

out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
N = E.N
rng = np.random.default_rng(5)
x0 = E.shepp_logan(N); S = E.make_coils(N, E.NC)
k = np.fft.fftfreq(N) * N; KX, KY = np.meshgrid(k, k)
SHOTS = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
K0 = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
nu = 1.0 / (1.0 + (KX ** 2 + KY ** 2) / K0 ** 2) ** 1.5; nu /= nu.sum()
counts = E.draw_counts(nu, SHOTS, rng)
op = E.Op(S, counts)                         # normalised: ||H|| = 1
sigma = 0.01 * op.data_rms(x0)
noise = op.AT_noise(sigma, np.random.default_rng(6))
b = op.H(x0) + noise                          # A^T y / L
lam = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-4
cnt = np.fft.ifftshift(counts)  # only for printing
dens = np.real(np.fft.ifft2(np.fft.fft2(counts) * np.fft.fft2(np.fft.ifftshift(np.exp(-((KX) ** 2 + (KY) ** 2) / 2.0)) / np.exp(-((KX) ** 2 + (KY) ** 2) / 2.0).sum())))
print(f"shots {SHOTS}, k0 {K0}, lam {lam}: sampled locations {int((counts > 0).sum())} of {N*N}, density centre / median over sampled {counts[0, 0] / np.median(counts[counts > 0]):.0f}")

def prox(z, t):
    return E.ihaar(E.soft(E.haar(z), t))

def F(x):
    r = op.H(x) - b  # not the objective itself; compute F via data residual surrogate below
    return None

# exact minimiser by long ADMM with an exact (many-iteration) CG x-step
def cg(apply, rhs, x, iters, M=None):
    r = rhs - apply(x); z = M(r) if M else r; p = z.copy(); rz = np.vdot(r, z).real
    for _ in range(iters):
        q = apply(p); a = rz / np.vdot(p, q).real
        x = x + a * p; r = r - a * q
        z = M(r) if M else r; rz_new = np.vdot(r, z).real
        p = z + (rz_new / rz) * p; rz = rz_new
    return x

def circ_M(rho):
    # circulant approximation of (H + rho I)^{-1}: H ~ F^H diag(c(k) * mean coil energy / L) F
    cover = np.mean(np.sum(np.abs(S) ** 2, 0))
    d = counts * cover / op.L
    def M(r):
        return np.real(np.fft.ifft2(np.fft.fft2(r, norm="ortho") / (d + rho), norm="ortho"))
    return M

def admm(rho, outer, n_in, M=None, x_star=None, x_init=None):
    x = np.zeros((N, N)) if x_init is None else x_init.copy(); z = x.copy(); u = np.zeros_like(x)
    A = lambda v: op.H(v) + rho * v
    hist, napp = [], 0
    for it in range(outer):
        x = cg(A, b + rho * (z - u), x, n_in, M); napp += n_in + 1
        z = prox(x + u, lam / rho); u = u + x - z
        if x_star is not None:
            hist.append((napp, np.linalg.norm(z - x_star) / np.linalg.norm(x_star)))
    return z, hist

t0 = time.time()
RHO_REF = float(sys.argv[6]) if len(sys.argv) > 6 else 0.05
x_star, _ = admm(RHO_REF, 4000, 30, circ_M(RHO_REF))
x_star2, _ = admm(RHO_REF / 2.5, 4000, 30, circ_M(RHO_REF / 2.5), x_init=x_star)
print(f"minimiser: ADMM runs agree to {np.linalg.norm(x_star - x_star2) / np.linalg.norm(x_star):.1e} ({time.time() - t0:.0f}s); error to truth {np.linalg.norm(x_star2 - x0) / np.linalg.norm(x0):.3f}")
x_star = x_star2
budget = 1500
res = {}

def ista(fista=False, theta=None, alpha=1.0, w=None, bw=None):
    x = np.zeros((N, N)); xp = x.copy(); t = 1.0; hist = []
    Hop = (lambda v: op.H(v)) if w is None else w
    rhs = b if bw is None else bw
    for it in range(budget):
        if fista:
            tn = (1 + np.sqrt(1 + 4 * t * t)) / 2; yk = x + ((t - 1) / tn) * (x - xp); t = tn
        else:
            yk = x
        g = Hop(yk) - rhs
        xp = x
        x = prox(yk - alpha * g, alpha * lam if theta is None else theta)
        hist.append((it + 1, np.linalg.norm(x - x_star) / np.linalg.norm(x_star)))
    return x, hist

xi, res["ista"] = ista()
xf, res["fista"] = ista(fista=True)
xp1, res["pnp_fixed_a1"] = ista(theta=lam, alpha=1.0)
xp2, res["pnp_fixed_a0.5"] = ista(theta=lam, alpha=0.5)
RHOS = [float(v) for v in sys.argv[7].split(',')] if len(sys.argv) > 7 else [0.05, 0.02]
for rho, nin in [(r, n) for r in RHOS for n in (3, 8)]:
    _, res[f"admm_pcg_rho{rho}_in{nin}"] = admm(rho, budget // (nin + 1), nin, circ_M(rho), x_star)
# density-weighted data term: weights 1/smoothed-density per k location
wk = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
def Hw(v):
    y = np.zeros((N, N))
    for c in range(E.NC):
        kk = np.fft.fft2(S[c] * v, norm="ortho") * counts * wk
        y += np.real(np.conj(S[c]) * np.fft.ifft2(kk, norm="ortho"))
    return y
Lw = E.power_norm(Hw); Hwn = lambda v: Hw(v) / Lw
# weighted A^H W y: data per location = counts * (F S x0) + noise sum -> weight 1/count gives the mean
bw = np.zeros((N, N)); rng2 = np.random.default_rng(6)
for c in range(E.NC):
    g = (rng2.standard_normal((N, N)) + 1j * rng2.standard_normal((N, N))) * sigma * np.sqrt(counts)
    kk = (np.fft.fft2(S[c] * x0, norm="ortho") * counts + g) * wk
    bw += np.real(np.conj(S[c]) * np.fft.ifft2(kk, norm="ortho"))
bw /= Lw
xd, res["fista_dcw"] = ista(fista=True, w=Hwn, bw=bw)

def first_below(h, tol):
    for napp, e in h:
        if e < tol:
            return napp
    return None

summary = {}
for kname, h in res.items():
    summary[kname] = {"final_rel_dist": h[-1][1], "apps_to_1e-1": first_below(h, 1e-1), "apps_to_1e-2": first_below(h, 1e-2), "apps_to_1e-3": first_below(h, 1e-3)}
summary["error_to_truth"] = {"x_star": float(np.linalg.norm(x_star - x0) / np.linalg.norm(x0)), "pnp_fixed_a1": float(np.linalg.norm(xp1 - x0) / np.linalg.norm(x0)), "pnp_fixed_a0.5": float(np.linalg.norm(xp2 - x0) / np.linalg.norm(x0)), "fista_dcw": float(np.linalg.norm(xd - x0) / np.linalg.norm(x0))}
summary["pnp_fixed_alpha_dependence"] = float(np.linalg.norm(xp1 - xp2) / np.linalg.norm(x_star))
(out / "rates_toy.json").write_text(json.dumps(summary, indent=1))
for kname, v in summary.items():
    print(kname, v)
