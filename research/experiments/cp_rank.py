"""Hermitian-CP rank of the pair family {u_j u_j^H}: smallest L' such that
      u_j u_j^H  ~  sum_{l=1..L'} c_lj  d_l d_l^H      (rank-1 atoms)  =>  L' Toeplitz kernels AND L' FFT pairs per coil.
This is the structure of Fessler 2005 eq. (24) generalised to arbitrary phase (your P x R / ALS idea with R = 1).
Optimised by gradient descent on the Gram form (never forms V x V matrices); c_j eliminated in closed form."""
import argparse, math
import numpy as np, torch
from conc_structure import *

def fit(E, Lp, steps=400, lr=0.03, init=None, seed=0):
    K, V = E.shape
    g = torch.Generator(device=E.device).manual_seed(seed)
    if init is None:
        _, _, Vh = torch.linalg.svd(E, full_matrices=False)
        D0 = Vh[:Lp].conj().T if Lp <= Vh.shape[0] else None
    else:
        D0 = init
    D0 = D0.clone()
    D0 = D0 + 0.02 * torch.randn(D0.shape, dtype=D0.dtype, device=E.device, generator=g)
    D = torch.view_as_real(D0).clone().requires_grad_(True)
    opt = torch.optim.Adam([D], lr=lr)
    Eh = E.conj()
    best = (1e9, None)
    for it in range(steps):
        Dc = torch.view_as_complex(D)
        Dn = Dc / Dc.abs().pow(2).sum(0, keepdim=True).sqrt()           # unit-norm atoms
        B = (Eh @ Dn).abs() ** 2 / V                                   # [K,L'] |<u_j,d_l>|^2 (u normalised by sqrt(V))
        M = (Dn.conj().T @ Dn).abs() ** 2                              # [L',L']
        Msym = M + 1e-9 * torch.eye(Lp, dtype=M.dtype, device=M.device)
        C = torch.linalg.solve(Msym, B.T)                              # [L',K]
        # ||U_j||^2 = 1 (u normalised), residual = 1 - b^T c  (at the LS optimum)
        res = (1 - (B.T * C).sum(0)).clamp(min=0)
        loss = res.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]: best = (loss.item(), Dn.detach().clone())
    return math.sqrt(best[0]), best[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--spokes", type=int, default=50); ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--cases", default="twisty:3.0,twisty:1.0,curly:1.0")
    ap.add_argument("--Lps", default="4,6,8,12,16,24")
    a = ap.parse_args(); rng = np.random.default_rng(1)
    print("columns: L'=#kernels=#FFT pairs per coil  -> relative HS error of the pair family (sqrt of mean residual); compare R_HS / Tucker in pair_rank.py")
    for case in a.cases.split(","):
        kind, B0 = case.split(":"); B0 = float(B0)
        t, k0, G0, dt = make_prototype(kind, a.T, 2e-3, math.pi / (0.32 / 256.0))
        W0 = cumtrapz(np.einsum("ta,tb->tab", G0, G0), dt)
        ax = (np.arange(a.n) - a.n / 2 + 0.5) * 0.32 / a.n
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij"); r = np.stack([X, Y, Z], -1).reshape(-1, 3)
        Rs = [rand_rot(rng) for _ in range(a.spokes)]
        phi = np.concatenate([phase_conc(np.einsum("ac,tcd,bd->tab", A, W0, A), r, B0) for A in Rs])
        E = torch.as_tensor(np.exp(-1j * phi), dtype=torch.complex64, device=a.device)
        s = torch.linalg.svdvals(E.to(torch.complex128)).cpu().numpy() ** 2; tail = s.sum() - np.cumsum(s)
        Lt = {e: int(np.searchsorted(-np.sqrt(tail.clip(min=0) / s.sum()), -e)) + 1 for e in (1e-2, 1e-3)}
        res = []
        for Lp in map(int, a.Lps.split(",")):
            if Lp > min(E.shape): continue
            err, _ = fit(E, Lp)
            res.append(f"L'={Lp}: {err:.1e}")
        print(f"{kind:7s} B0={B0} maxPhi={np.abs(phi).max():.2f} rank(E)@1e-2,1e-3={list(Lt.values())} | CP: " + ", ".join(res))

if __name__ == "__main__":
    main()
