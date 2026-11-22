"""Is BFP-quantised solver state safe? (research/bfp_storage.md §3.4)

2-D toy with the real phantom: slice z = 57 of ph0001 (254 x 144), 10 virtual coils (true maps, signal-region Gram),
Cartesian variable-density undersampling R ~ 3, complex Gaussian noise at image SNR 20. Problem
    min_x 1/2 ||M F S x - y||^2 + lam/2 ||x||^2
solved by Barzilai-Borwein gradient descent (the quasi-Newton family of the recon) with the state handled in
different precisions; reference x* = CG in f64 to 1e-13. Reports ||x_k - x*|| / ||x*|| per variant against the
noise error ||x* - x_true|| / ||x_true|| and the f64 run.

Variants (Q_b = BFP round trip, 16-complex blocks along x of the flat image):
  f64            all f64
  f32            state and arithmetic f32 (numpy complex64)
  state_bfp_b    x stored as BFP between iterations, no master copy: x <- Q_b(x - a g)
  master_x_b     f32 master x, the operator reads a BFP working copy: g = A^H(A Q_b(x) - y) + lam x
  master_g_b     f32 master x, the gradient comes back as BFP: x <- x - a Q_b(g)
  pairs_b        f32 master x and g, the BB secant pair s, y stored as BFP (step length from Q_b(s), Q_b(y))
  all_b          master_x + master_g + pairs together (the proposed policy)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acc import D, NVC, q_bfp, hs  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11)
zs = 57
pd = (np.load(D + "truth_pd_water.npy")[:, :, zs] + np.load(D + "truth_pd_fat.npy")[:, :, zs]).astype(np.float64).T  # (y, x)
ny, nx = pd.shape
yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")
xt = pd * np.exp(1j * (0.8 * xx + 0.5 * yy))
sm = hs.load_any(D + "smaps_true.hv", lazy=True)
s3 = np.asarray(sm[:, :, zs, :]).astype(np.complex128)  # (x, y, 32)
s3 = np.transpose(s3, (2, 1, 0))  # (32, y, x)
mask = pd > 0.05 * pd.max()
sv = s3[:, mask]
w, U = np.linalg.eigh(sv @ sv.conj().T)
U = U[:, ::-1][:, :NVC]
S = np.einsum("ck,cyx->kyx", U.conj(), s3)
# variable-density Cartesian mask along y (phase encode), full x (readout)
ky = np.fft.fftfreq(ny) * 2
prob = np.clip(1.0 / (1 + (np.abs(ky) / 0.08) ** 2) + 0.08, 0, 1)
M = (rng.random(ny) < prob)[:, None] * np.ones((1, nx), bool)
R = M.size / M.sum()


def A(x):
    return M[None] * np.fft.fft2(S * x[None], norm="ortho")


def AH(k):
    return (np.conj(S) * np.fft.ifft2(M[None] * k, norm="ortho")).sum(0)


y0 = A(xt)
comb = np.abs(xt) * np.sqrt((np.abs(S) ** 2).sum(0))
sig = comb[mask].mean() / 20.0  # image SNR 20 (ortho FFT: same sigma in k-space)
y = y0 + M[None] * sig * (rng.standard_normal(y0.shape) + 1j * rng.standard_normal(y0.shape)) / np.sqrt(2)
lam = 0.02 * np.abs(S).max() ** 2  # mild Tikhonov (keeps the problem well posed)


def normal(x):
    return AH(A(x)) + lam * x


b = AH(y)
# CG reference
x = np.zeros_like(b)
r = b - normal(x)
p = r.copy()
rr = np.vdot(r, r).real
for it in range(2000):
    Ap = normal(p)
    a = rr / np.vdot(p, Ap).real
    x += a * p
    r -= a * Ap
    rn = np.vdot(r, r).real
    if np.sqrt(rn / np.vdot(b, b).real) < 1e-13:
        break
    p = r + rn / rr * p
    rr = rn
xs = x
nerr = np.linalg.norm(xs - xt) / np.linalg.norm(xt)
print(f"slice {ny} x {nx}, R {R:.2f}, CG iterations {it}, noise error ||x* - x_true|| / ||x_true|| = {nerr:.3e}", flush=True)


def run(variant, b_bits=None, iters=300):
    f = np.complex64 if variant != "f64" else np.complex128
    Q = (lambda v: q_bfp(v.astype(np.complex128), b_bits).astype(f)) if b_bits else (lambda v: v)
    I = lambda v: v  # noqa: E731
    qx = Q if variant in ("master_x", "all") else I
    qg = Q if variant in ("master_g", "all") else I
    qp = Q if variant in ("pairs", "all") else I
    qs = Q if variant == "state_bfp" else I
    yy_ = y.astype(f)
    bb = b.astype(f)

    def grad(x):
        xr = qx(x)
        return (AH(A(xr)) + lam * xr - bb).astype(f) if variant != "f32" else (AH(A(xr).astype(f)).astype(f) + f(lam) * xr - bb).astype(f)

    x = np.zeros_like(bb)
    g = qg(grad(x))
    a = 1.0 / (np.abs(S) ** 2).sum(0).max()
    hist = []
    for k in range(iters):
        xn = qs((x - f(a) * g).astype(f))
        gn = qg(grad(xn))
        s_, y_ = qp((xn - x).astype(f)), qp((gn - g).astype(f))
        sy = np.vdot(s_, y_).real
        a = np.vdot(s_, s_).real / sy if sy > 0 else a
        x, g = xn, gn
        if k % 10 == 9 or k == iters - 1:
            hist.append((k + 1, float(np.linalg.norm(x.astype(np.complex128) - xs) / np.linalg.norm(xs))))
    _ = yy_
    return hist


res = {"R": R, "noise_err": nerr, "cg_iters": it, "runs": {}}
for v, bl in [("f64", [None]), ("f32", [None]), ("state_bfp", [12, 16, 20, 24]), ("master_x", [12, 16, 20, 24]), ("master_g", [12, 16, 20, 24]), ("pairs", [12, 16, 20]), ("all", [12, 16, 20, 24])]:
    for bits in bl:
        h = run(v, bits)
        key = v + (f"_{bits}" if bits else "")
        res["runs"][key] = h
        print(f"{key:>14}: " + "  ".join(f"{k}:{e:.2e}" for k, e in h if k in (10, 30, 100, 200, 300)), flush=True)
json.dump(res, open(os.path.join(OUT, "state_toy.json"), "w"), indent=1)
