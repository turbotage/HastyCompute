#!/usr/bin/env python3
"""Magnitude coupling across encodes inside the PnP denoiser: does it help the velocity, and what does it cost where
the truth violates it? 2-D toy, CPU, minutes.

Truth (64x64, built on a 4x finer grid and box-averaged, so partial volume and intravoxel shear dispersion are physical):
  static tissue (modified Shepp-Logan magnitude), a vessel (disk, radius 9 px) with Poiseuille through-plane velocity
  v_max = 0.8 m/s and blood brightness 1.2, and a turbulent jet sub-region (velocity fluctuations sigma_t = 0.3 m/s on the
  fine grid) — so the voxel's encodes are x_e = mean_fine( rho e^{-i k_e v} ) with intravoxel dispersion where the
  fine-grid velocity varies.
Encodes: k = [-1.571, 1.571, 0.524] rad per m/s (one axis of the ph0001 design: reference with nonzero moment).
Sampling: the same i.i.d. shots for every encode (as the encodes of one TR group / frame share rotations), variable
density k0 = 4, 4 coils, noise. Data term separable over encodes, P = I.
Denoiser: joint soft threshold across the encodes in Haar coordinates (complex), then the magnitude prox
  |u_e| <- mean_e|u| + (|u_e| - mean_e|u|) / (1 + gamma w),  phases kept,  w = m^2 / (m^2 + delta^2) (SNR guard).
Sweep gamma in {0, 0.1, 0.3, 1, 3, 10, 100}; for each, the best lambda of a small grid. Report velocity RMSE (vessel
interior / wall ring / jet), magnitude error, and the truth's own magnitude spread per region.
Usage: rustlib/.venv/bin/python -I research/experiments/magnitude/mag_toy.py [--quick]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "research" / "mathematics_of_paper" / "pnp_restricted"))
import numpy as np  # noqa: E402

import experiments as ex  # noqa: E402

N, NC = ex.N, ex.NC
KV = np.array([-1.571, 1.571, 0.524])
E = len(KV)
QUICK = "--quick" in sys.argv


def truth(seed=3):
    rng = np.random.default_rng(seed)
    f = 4
    nf = N * f
    tissue = np.kron(ex.shepp_logan(N), np.ones((f, f)))  # static magnitude on the fine grid
    t = (np.arange(nf) + 0.5) / nf * 2 - 1
    X, Y = np.meshgrid(t, -t)
    cx, cy, R = 0.25, -0.1, 9 / (N / 2)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    vessel = r < R
    v = np.where(vessel, 0.8 * (1 - (r / R) ** 2), 0.0)
    jet = vessel & (X > cx) & (Y > cy)  # one quadrant: disturbed flow
    v = v + np.where(jet, 0.3 * rng.standard_normal(v.shape), 0.0)
    rho = np.where(vessel, 1.2, tissue)
    xf = np.stack([rho * np.exp(-1j * k * v) for k in KV])  # (E, nf, nf)
    x = xf.reshape(E, N, f, N, f).mean((2, 4))
    frac = vessel.reshape(N, f, N, f).mean((1, 3))
    vbar = (v * vessel).reshape(N, f, N, f).sum((1, 3)) / np.maximum(vessel.reshape(N, f, N, f).sum((1, 3)), 1)
    jetc = jet.reshape(N, f, N, f).mean((1, 3)) > 0.5
    return x, frac, vbar, jetc


SHOTS = "indep" if "--indep" in sys.argv else "shared"


class COp:
    def __init__(self, S, counts):
        self.S, self.c = S, counts
        self.L = 1.0
        v = np.random.default_rng(0).standard_normal((N, N)) + 0j
        for _ in range(60):
            w = self.H1(v)
            lam = np.linalg.norm(w) / np.linalg.norm(v)
            v = w / np.linalg.norm(w)
        self.L = lam

    def H1(self, x):
        y = np.zeros((N, N), complex)
        for c in range(NC):
            y += np.conj(self.S[c]) * np.fft.ifft2(np.fft.fft2(self.S[c] * x, norm="ortho") * self.c, norm="ortho")
        return y / self.L

    def AHn(self, sigma, rng):
        y = np.zeros((N, N), complex)
        for c in range(NC):
            g = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * sigma * np.sqrt(self.c)
            y += np.conj(self.S[c]) * np.fft.ifft2(g, norm="ortho")
        return y / self.L


def chaar(z):
    return ex.haar(z.real) + 1j * ex.haar(z.imag)


def cihaar(c):
    return ex.ihaar(c.real) + 1j * ex.ihaar(c.imag)


def joint_soft(z, lam):
    c = chaar(z)
    r = np.sqrt(np.sum(np.abs(c) ** 2, 0))
    f = np.where(r > lam, 1 - lam / np.maximum(r, 1e-300), 0.0)
    f[ex.APPROX] = 1.0
    return cihaar(c * f[None])


CAP = "--cap" in sys.argv


def mag_prox(u, gamma, delta):
    """--cap: per-encode weight w_e = m_e^2 / (m_e^2 + delta^2) (a weak encode is not pulled up) and the magnitude
    gain |u_e'| / |u_e| clipped to [1/2, 3/2] (bounds the expansion of the phase direction)"""
    if gamma == 0:
        return u
    m = np.abs(u)
    mb = m.mean(0)
    if CAP:
        w = m ** 2 / (m ** 2 + delta ** 2)
        mn = mb[None] + (m - mb[None]) / (1 + gamma * w)
        gain = np.clip(mn / np.maximum(m, 1e-300), 0.5, 1.5)
        return np.where(m > 0, u * gain, 0)
    w = mb ** 2 / (mb ** 2 + delta ** 2)
    mn = mb[None] + (m - mb[None]) / (1 + gamma * w[None])
    return np.where(m > 0, u * (mn / np.maximum(m, 1e-300)), 0)


def velocity(x):
    dk = KV[1:] - KV[0]
    dphi = np.stack([np.angle(x[e] * np.conj(x[0])) for e in range(1, E)])  # (E-1, N, N)
    # least squares on the phase differences (sim convention: phase = -k v)
    return -np.tensordot(dk, dphi, (0, 0)) / np.sum(dk ** 2)


def run(ops, b, lam, gamma, delta, iters, x0=None):
    x = joint_soft(b, lam) if x0 is None else x0.copy()
    for k in range(iters):
        g = np.stack([ops[e].H1(x[e]) for e in range(E)]) - b
        xn = mag_prox(joint_soft(x - g, lam), gamma, delta)
        d = np.linalg.norm(xn - x) / np.linalg.norm(xn)
        x = xn
        if d < 1e-8:
            break
    return x, k + 1


def main():
    t0 = time.time()
    xt, frac, vbar, jet = truth()
    rng = np.random.default_rng(1)
    S = ex.make_coils(N, NC)
    nu = ex.density(N, k0=4.0)
    # shots: shared by the encodes (one pattern) or independent per encode (encodes of a frame from different
    # rotations); the operator norm is normalised once (common L)
    cl = [ex.draw_counts(nu, 700, rng)] * E if SHOTS == "shared" else [ex.draw_counts(nu, 700, rng) for _ in range(E)]
    ops = [COp(S, c) for c in cl]
    Lc = max(o.L for o in ops)
    for o in ops:
        o.L = Lc
    sig = 0.02 * np.sqrt(np.mean(np.abs(xt) ** 2))
    nrng = np.random.default_rng(7)
    b = np.stack([ops[e].H1(xt[e]) + ops[e].AHn(sig, nrng) for e in range(E)])
    interior = frac > 0.99
    wall = (frac > 0.05) & (frac <= 0.99)
    tissue = (frac == 0) & (np.abs(xt[0]) > 0.05)
    reg = {"vessel interior": interior & ~jet, "jet": interior & jet, "wall": wall, "tissue": tissue}
    m = np.abs(xt)
    spread = m.std(0) / np.maximum(m.mean(0), 1e-12)
    res = dict(shots=SHOTS, truth_spread={k: dict(median=float(np.median(spread[v])), p90=float(np.percentile(spread[v], 90))) for k, v in reg.items()})
    print("truth magnitude spread across encodes:", {k: (round(v['median'] * 100, 2), round(v['p90'] * 100, 2)) for k, v in res["truth_spread"].items()}, "(median %, p90 %)")
    delta = 3 * sig
    gammas = [0, 0.3, 3, 100] if QUICK else [0, 0.1, 0.3, 1, 3, 10, 100]
    lams = [0.001, 0.002, 0.004] if QUICK else [0.0005, 0.001, 0.002, 0.004, 0.008]
    iters = 1500 if QUICK else 5000
    res["runs"] = {}
    for gm in gammas:
        best = None
        for lam in lams:
            x, it = run(ops, b, lam, gm, delta, iters)
            v = velocity(x)
            ve = {k: float(np.sqrt(np.mean((v[r] - vbar[r]) ** 2))) for k, r in reg.items()}
            me = float(np.linalg.norm((np.abs(x) - m)[:, tissue | interior | wall]) / np.linalg.norm(m[:, tissue | interior | wall]))
            score = ve["vessel interior"] + ve["jet"] + ve["wall"]  # tissue velocity reported, not scored
            if best is None or score < best[0]:
                best = (score, lam, ve, me, it)
        res["runs"][gm] = dict(lam=best[1], vel_rmse=best[2], mag_rel_err=best[3], iters=best[4])
        print(f"gamma {gm:6.1f}: best lam {best[1]:.3f}; velocity RMSE (m/s) " + ", ".join(f"{k} {v:.4f}" for k, v in best[2].items()) + f"; magnitude rel. err {best[3]:.4f} ({best[4]} it)", flush=True)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"mag_toy_{SHOTS}{'_cap' if CAP else ''}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
