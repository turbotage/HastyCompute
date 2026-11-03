"""Numerical checks of the concomitant-field structure used in ../concomitant_structure.md

  (1) Maxwell coefficients: |B| expansion vs the quadratic formula.
  (2) Linear-transform-of-base-spoke => concomitant waveforms are a linear mixing Theta_s of the prototype's
      second-moment waveforms W_0(tau): w_{s,q} = sum_p Theta_{s,qp} W_p(tau).
  (3) Numerical rank (Eckart-Young) of exp(-i Phi_conc) over (samples x voxels):
        - global (all spokes, all times)            <- what phi_lowrank's SVD sees
        - per spoke (single spoke, all times)
        - temporal family only (spoke-independent time basis, P-space sampled)
  (4) k-space stencil width of one modulated row exp(-i Phi_j(r)).

Run:  python conc_structure.py [--B0 3.0] [--n 20] [--spokes 120] [--T 48] [--device cuda]
"""
import argparse, math
import numpy as np
import torch

GAMMA = 2.6752218744e8  # rad/s/T

def cumtrapz(y, dt):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dt, axis=0)
    return out

def make_prototype(kind, T, Tr, kmax, turns=2.0, theta=0.6):
    t = np.linspace(0, Tr, T)
    u = t / Tr
    if kind == "radial":
        k0 = kmax * u[:, None] * np.array([[0.0, 0.0, 1.0]])
    elif kind == "twisty":      # cone-spiral starting at the centre
        ph = 2 * math.pi * turns * u
        d = np.stack([np.sin(theta) * np.cos(ph), np.sin(theta) * np.sin(ph), np.cos(theta) * np.ones_like(ph)], 1)
        k0 = kmax * u[:, None] * d
    elif kind == "curly":       # stronger 3D curling: two incommensurate rotations
        a = 2 * math.pi * turns * u; b = 2 * math.pi * 0.7 * turns * u
        d = np.stack([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)], 1)
        k0 = kmax * u[:, None] * d
    else:
        raise ValueError(kind)
    dt = t[1] - t[0]
    G0 = np.gradient(k0, dt, axis=0) / GAMMA          # T/m
    return t, k0, G0, dt

def rand_rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0: q[:, 0] *= -1
    return q

def S_of_r(r):
    """S(r) with B_c = G^T S G / (2 B0): u = (Gx z - Gz x/2, Gy z - Gz y/2), B_c = |u|^2/(2B0)."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    S = np.zeros(r.shape[:-1] + (3, 3))
    S[..., 0, 0] = z * z; S[..., 1, 1] = z * z; S[..., 2, 2] = (x * x + y * y) / 4
    S[..., 0, 2] = S[..., 2, 0] = -x * z / 2
    S[..., 1, 2] = S[..., 2, 1] = -y * z / 2
    return S

def phase_conc(W_s, r, B0):
    """W_s [T,3,3] second-moment of G_s; r [V,3] -> Phi [T,V]  (rad)."""
    S = S_of_r(r)                                       # [V,3,3]
    return GAMMA / (2 * B0) * np.einsum("tab,vab->tv", W_s, S)

def numrank(E, tols=(1e-2, 1e-3, 1e-4), device="cuda"):
    """smallest L with ||E - E_L||_F / ||E||_F <= tol"""
    s = torch.linalg.svdvals(torch.as_tensor(E, dtype=torch.complex64, device=device)).double().cpu().numpy()
    e2 = np.cumsum((s ** 2)[::-1])[::-1]               # tail energy from index i
    tot = e2[0]
    out = {}
    for tol in tols:
        # tail after keeping L terms = e2[L]
        L = int(np.searchsorted(-(np.sqrt(np.append(e2[1:], 0.0) / tot)), -tol))  # first L with tail<=tol
        out[tol] = L
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B0", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--spokes", type=int, default=120)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--Tr", type=float, default=2e-3)
    ap.add_argument("--fov", type=float, default=0.32)
    ap.add_argument("--offset", type=float, default=0.0, help="isocentre offset along z [m]")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    # (1) Maxwell check: |B| = |B0 z + G.r z + (Gx z - Gz x/2) x + (Gy z - Gz y/2) y|
    G = rng.normal(size=3) * 0.02; r = rng.normal(size=3) * 0.1
    B0 = a.B0
    Bz = B0 + G @ r; Bx = G[0] * r[2] - G[2] * r[0] / 2; By = G[1] * r[2] - G[2] * r[1] / 2
    exact = math.sqrt(Bz ** 2 + Bx ** 2 + By ** 2) - B0 - G @ r
    approx = (Bx ** 2 + By ** 2) / (2 * B0)
    quad = (G @ S_of_r(r) @ G) / (2 * B0)
    print(f"[1] Bc exact={exact:.4e} leading-order={approx:.4e} G^T S G/2B0={quad:.4e}  rel.err(1/B0 truncation)={(exact-approx)/approx:.2e}")
    # the (Gx^2+Gy^2)/(8B0) z^2 form from notes/MRI_Physics.tex eq. conc_maxwell:
    wrong = ((G[0] ** 2 + G[1] ** 2) / (8 * B0)) * r[2] ** 2 + G[2] ** 2 / (8 * B0) * (r[0] ** 2 + r[1] ** 2) + G[0] * G[2] / (2 * B0) * r[0] * r[2] + G[1] * G[2] / (2 * B0) * r[1] * r[2]
    print(f"    notes/non_fourier_interp_test.cpp form = {wrong:.4e}  (ratio to exact {wrong/approx:.3f})")

    ax = (np.arange(a.n) - a.n / 2 + 0.5) * a.fov / a.n
    X, Y, Z = np.meshgrid(ax, ax, ax + a.offset, indexing="ij")
    r_vox = np.stack([X, Y, Z], -1).reshape(-1, 3)
    kmax = math.pi / (a.fov / 256.0)               # 256-voxel-equivalent resolution

    for kind in ("radial", "twisty", "curly"):
        t, k0, G0, dt = make_prototype(kind, a.T, a.Tr, kmax)
        W0 = cumtrapz(np.einsum("ta,tb->tab", G0, G0), dt)           # [T,3,3]
        Rs = [rand_rot(rng) for _ in range(a.spokes)]
        # (2) mixing check on one spoke
        A = Rs[0]
        Gs = G0 @ A.T
        Ws_direct = cumtrapz(np.einsum("ta,tb->tab", Gs, Gs), dt)
        Ws_mix = np.einsum("ac,tcd,bd->tab", A, W0, A)
        assert np.allclose(Ws_direct, Ws_mix, rtol=1e-6, atol=1e-18), "W_s != A W_0 A^T"
        phis = np.stack([phase_conc(np.einsum("ac,tcd,bd->tab", A, W0, A), r_vox, B0) for A in Rs])  # [S,T,V]
        pmax = np.abs(phis).max()
        E_all = np.exp(-1j * phis).reshape(a.spokes * a.T, -1)
        # (3a) global rank on a subsample of rows (spokes x time) -- keep all voxels
        sub = rng.choice(E_all.shape[0], size=min(6000, E_all.shape[0]), replace=False)
        rk_g = numrank(E_all[sub], device=a.device)
        # (3b) single spoke
        rk_1 = numrank(np.exp(-1j * phis[0]), device=a.device)
        # (3c) 20-spoke cluster nearest in direction (angle of the rotated prototype z axis)
        z_dirs = np.stack([A[:, 2] for A in Rs]); c = z_dirs @ z_dirs[0]
        near = np.argsort(-np.abs(c))[:20]
        rk_c = numrank(np.exp(-1j * phis[near]).reshape(-1, phis.shape[-1]), device=a.device)
        print(f"[3] {kind:7s} max|Phi|={pmax:6.2f} rad | rank@1e-2/1e-3/1e-4: global {list(rk_g.values())} | one spoke {list(rk_1.values())} | 20 nearest-direction spokes {list(rk_c.values())}")
        # (3d) temporal family: rows = times, cols = (spoke,voxel) pairs -> spoke-independent time basis
        cols = rng.choice(a.spokes * r_vox.shape[0], size=4000, replace=False)
        Ecols = np.exp(-1j * phis.transpose(1, 0, 2).reshape(a.T, -1)[:, cols])
        rk_t = numrank(Ecols, device=a.device)
        print(f"    temporal-family rank (spoke-independent time basis, T={a.T}) @1e-2/1e-3/1e-4: {list(rk_t.values())}")

    # (4) stencil width for one modulated row with the strongest phase
    t, k0, G0, dt = make_prototype("twisty", a.T, a.Tr, kmax)
    W0 = cumtrapz(np.einsum("ta,tb->tab", G0, G0), dt)
    n = 48
    ax4 = (np.arange(n) - n / 2 + 0.5) * a.fov / n
    X, Y, Z = np.meshgrid(ax4, ax4, ax4, indexing="ij")
    rv = np.stack([X, Y, Z], -1).reshape(-1, 3)
    phi = phase_conc(W0[-1:], rv, B0)[0].reshape(n, n, n)
    spec = np.fft.fftshift(np.fft.fftn(np.exp(-1j * phi) * np.hanning(n)[:, None, None] ** 0))
    p = np.abs(spec) ** 2; p /= p.sum()
    idx = np.indices(p.shape) - n // 2
    rad = np.sqrt((idx ** 2).sum(0))
    ms = []
    for rr in (1, 2, 3, 4, 6, 8):
        ms.append(f"r<={rr}: {1 - p[rad <= rr].sum():.1e}")
    print(f"[4] k-space spread of exp(-i Phi) at end of readout, max|Phi|={np.abs(phi).max():.2f} rad, energy OUTSIDE radius (cells of 1/FOV): " + ", ".join(ms))

if __name__ == "__main__":
    main()
