"""How many Toeplitz kernels / FFT pairs does the concomitant phase really need?

The normal operator is  N(r,r') = sum_j w_j e^{i k_j.(r-r')} U_j(r,r'),  U_j = u_j u_j^H,  u_j(r) = exp(-i Phi_j(r)).
Any decomposition  U_j ~ sum_l c_l(j) A_l(r,r')  gives  N ~ sum_l T_l(r-r') A_l(r,r')  with ONE Toeplitz kernel T_l per l
(kernel weights w_j c_l(j)) and  sum_l rank(A_l)  FFT pairs per coil (A_l = sum_p d_lp d'_lp^H).

The minimum number of kernels at tolerance eps is the eps-rank of the family {U_j} in Hilbert-Schmidt norm, i.e. of the Gram matrix
G2(j,j') = |<u_j,u_j'>|^2.  Compared with the Tucker route (u_j ~ sum_l c_l(j) b_l, L = rank(E)): L(L+1)/2 kernels, 2L FFTs/coil.
(The linear phase e^{ik_j.(r-r')} is exact and ignored here: this is a proxy for the operator error.)
"""
import argparse, math
import numpy as np, torch
from conc_structure import *

def eps_rank_from_eigs(ev, eps):
    ev = np.sort(ev)[::-1].clip(min=0); tot = ev.sum()
    tail = tot - np.cumsum(ev)
    return int(np.searchsorted(-(np.sqrt(tail / tot)), -eps)) + 1 if tot > 0 else 0

def run(kind, B0, n, spokes, T, dev, rng, eps=(1e-2, 1e-3)):
    t, k0, G0, dt = make_prototype(kind, T, 2e-3, math.pi / (0.32 / 256.0))
    W0 = cumtrapz(np.einsum("ta,tb->tab", G0, G0), dt)
    ax = (np.arange(n) - n / 2 + 0.5) * 0.32 / n
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij"); r = np.stack([X, Y, Z], -1).reshape(-1, 3)
    out = {}
    for label, nsp in (("1 spoke", 1), (f"{spokes} spokes", spokes)):
        Rs = [rand_rot(rng) for _ in range(nsp)]
        phi = np.concatenate([phase_conc(np.einsum("ac,tcd,bd->tab", A, W0, A), r, B0) for A in Rs])
        E = torch.as_tensor(np.exp(-1j * phi), dtype=torch.complex128, device=dev)           # [K,V]
        V = E.shape[1]
        sv = torch.linalg.svdvals(E).cpu().numpy() ** 2
        L = {e: eps_rank_from_eigs(sv, e) for e in eps}
        Gram = E @ E.conj().T
        G2 = (Gram.abs() ** 2) / V ** 2
        ev, evec = torch.linalg.eigh(G2)
        ev = ev.cpu().numpy()
        R = {e: eps_rank_from_eigs(ev, e) for e in eps}
        # atom ranks for the leading HS components (eps=1e-2 target)
        Rt = R[eps[0]]
        order = np.argsort(ev)[::-1][:Rt]
        atom_r = []
        for idx in order:
            v = evec[:, idx]
            A = E.conj().T @ (v[:, None] * E)                      # sum_j v_j u_j^* u_j^T   [V,V] (Hermitian up to conj)
            a = torch.linalg.eigvalsh((A + A.conj().T) / 2).abs().cpu().numpy()
            a = np.sort(a)[::-1]; cum = np.cumsum(a ** 2); atom_r.append(int(np.searchsorted(cum, (1 - eps[0] ** 2) * cum[-1])) + 1)
        out[label] = (np.abs(phi).max(), L, R, sum(atom_r))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--spokes", type=int, default=50); ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--B0s", default="3.0,1.0,0.5")
    a = ap.parse_args(); rng = np.random.default_rng(1)
    print("rank(E)=L (u_j ~ sum c_l b_l) -> Tucker: L(L+1)/2 kernels, 2L FFT/coil | R_HS = eps-rank of {u_j u_j^H} = min #kernels | atomFFT = sum of atom ranks (eps 1e-2) = FFT pairs/coil")
    for kind in ("twisty", "curly"):
        for B0 in map(float, a.B0s.split(",")):
            res = run(kind, B0, a.n, a.spokes, a.T, a.device, rng)
            for label, (pm, L, R, af) in res.items():
                tk = {e: L[e] * (L[e] + 1) // 2 for e in L}
                print(f"{kind:7s} B0={B0:3.1f} {label:10s} maxPhi={pm:5.2f} | L(1e-2,1e-3)={list(L.values())} Tucker kernels={list(tk.values())} | R_HS={list(R.values())} | atomFFT(1e-2)={af}")

if __name__ == "__main__":
    main()
