"""Operator-level accuracy vs FFT-pair count for different ways of representing the concomitant-field normal operator
        N(r,r') = sum_j w_j e^{ik_j.(r-r')} e^{i(Phi_j(r)-Phi_j(r'))}          (dense, tiny grid, exact reference)

Strategies (cost = FFT pairs per coil; kernels = distinct Toeplitz kernels):
  taylor-Phi(P) : u_j = e^{-i Phi_j} ~ sum_{p<=P} (-i Phi_j)^p/p!            pairs C(4+P,P)  (fields = monomials in the 4 conc. fields)
  svd-Tucker(L) : u_j ~ sum_l c_l(j) b_l(r), L = rank                         pairs L, kernels L(L+1)/2
  weyl-pair(n)  : e^{ix}, x = Phi(r)-Phi(r') = rbar^T M Delta (exact), truncated at order n   pairs C(n+3,3)  (image-domain coordinate monomials x Toeplitz)
  prodconv(L)   : one-sided product-convolution (Escande-Weiss): N(r,r-D)=TVIR(r,D) ~ sum_l u_l(r) h_l(D); SVD of TVIR   pairs L, kernels L, ONE diagonal
  cp(L')        : two-sided rank-1 atoms (Fessler IV-B style)                  pairs L', kernels L'
Error = ||N - N_approx||_F / ||N - N_0||_F  (1 = no better than ignoring the concomitant field; N_0 is Phi=0).
"""
import argparse, math, itertools
import numpy as np, torch
from conc_structure import make_prototype, cumtrapz, rand_rot, S_of_r, GAMMA

def build(kind, n, spokes, T, target_phi, dev, seed=0):
    rng = np.random.default_rng(seed)
    fov = 0.32; dx = fov / n; kmax = math.pi / dx
    t, k0, G0, dt = make_prototype(kind, T, 2e-3, kmax)
    W0 = cumtrapz(np.einsum("ta,tb->tab", G0, G0), dt)
    Rs = [rand_rot(rng) for _ in range(spokes)]
    ks = np.concatenate([k0 @ A.T for A in Rs])                        # [K,3]
    Ws = np.concatenate([np.einsum("ac,tcd,bd->tab", A, W0, A) for A in Rs])  # [K,3,3]
    def pos(shift):                                                    # integer voxel offsets -> physical coords
        g = (np.arange(n) - (n - 1) / 2)
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        return (np.stack([X, Y, Z], -1).reshape(-1, 3) + shift) * dx
    r = pos(0)
    S = S_of_r(r); Phi = np.einsum("kab,vab->kv", Ws, S)
    c = target_phi / np.abs(Phi).max()
    return dict(n=n, dx=dx, ks=ks, Ws=Ws, c=c, pos=pos, r=r, K=ks.shape[0], dev=dev)

def phi_at(P, rr):
    S = S_of_r(rr); return P["c"] * np.einsum("kab,vab->kv", P["Ws"], S)

def N_from_u(P, u_row, u_col):
    """N(r,r') = sum_j e^{ik_j r} conj(u(r)) e^{-ik_j r'} u(r');  u_row [K,Vr], u_col [K,Vc] (torch complex128)"""
    ks = torch.as_tensor(P["ks"], dtype=torch.float64, device=P["dev"])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--spokes", type=int, default=48); ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--kind", default="twisty"); ap.add_argument("--phi", type=float, default=1.0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); dev = a.device
    P = build(a.kind, a.n, a.spokes, a.T, a.phi, dev)
    n, K = P["n"], P["K"]; V = n ** 3
    td = dict(dtype=torch.complex128, device=dev)
    ks = torch.as_tensor(P["ks"], dtype=torch.float64, device=dev)

    def lin(rr):                                   # e^{-i k_j . r}  [K,V]
        return torch.exp(-1j * (ks @ torch.as_tensor(rr, dtype=torch.float64, device=dev).T)).to(torch.complex128)
    Lg = lin(P["r"])
    Phi_g = phi_at(P, P["r"])
    Ug = torch.as_tensor(np.exp(-1j * Phi_g), **td)
    def Nmat(U):                                   # a_j(r) = L_j(r) U_j(r);  N = sum_j conj(a_j(r)) a_j(r')
        A = Lg * U
        return A.conj().T @ A
    N = Nmat(Ug); N0 = Nmat(torch.ones_like(Ug))
    base = torch.linalg.norm(N - N0).item()
    print(f"kind={a.kind} n={n}^3 V={V} K={K} max|Phi|={np.abs(Phi_g).max():.2f}  ||N-N0||/||N||={base/torch.linalg.norm(N).item():.3f}")
    def err(Na): return torch.linalg.norm(N - Na).item() / base
    rows = []
    # ---- taylor in Phi
    Phi_t = torch.as_tensor(Phi_g, dtype=torch.float64, device=dev)
    for Pord in (1, 2, 3, 4, 6):
        U = sum(((-1j * Phi_t) ** p / math.factorial(p)).to(torch.complex128) for p in range(Pord + 1))
        rows.append((f"taylor-Phi(P={Pord})", math.comb(4 + Pord, Pord), err(Nmat(U))))
    # ---- svd tucker
    Uu, s, Vh = torch.linalg.svd(Ug, full_matrices=False)
    for L in (2, 4, 6, 8, 12, 16, 24):
        UL = (Uu[:, :L] * s[:L]) @ Vh[:L]
        rows.append((f"svd-Tucker(L={L})", L, err(Nmat(UL))))
    # ---- weyl pair expansion: N_n = sum_j L-terms * sum_{m<=n} (i x)^m/m!   (x = Phi(r)-Phi(r'))
    Ph = torch.as_tensor(Phi_g, dtype=torch.float64, device=dev)
    # N_n(r,r') = sum_j conj(L_j(r))... build via loop over m using elementwise pair phases: O(K V^2) memory chunked
    def weyl(nord):
        out = torch.zeros(V, V, **td)
        for j0 in range(0, K, 64):
            sl = slice(j0, min(K, j0 + 64))
            x = Ph[sl, :, None] - Ph[sl, None, :]                          # [k,V,V]
            ser = sum((1j * x) ** m / math.factorial(m) for m in range(nord + 1)).to(torch.complex128)
            A = Lg[sl]
            out += torch.einsum("kr,kRs,ks->rs", A.conj(), ser, A) if False else (A.conj()[:, :, None] * ser * A[:, None, :]).sum(0)
        return out
    for nord in (1, 2, 3, 4, 6):
        rows.append((f"weyl-pair(n={nord})", math.comb(nord + 3, 3), err(weyl(nord))))
    # ---- one-sided product-convolution via SVD of TVIR(r,D)=N(r,r-D)
    g = (np.arange(n) - (n - 1) / 2)
    Dg = np.arange(-(n - 1), n)
    nd = len(Dg)
    ext = np.arange(-(n - 1), 2 * n - 1)                                   # r' index range (voxel offsets) = r - D covers [-(n-1), 2n-2]
    Ext = np.stack(np.meshgrid(ext - (n - 1) / 2, ext - (n - 1) / 2, ext - (n - 1) / 2, indexing="ij"), -1).reshape(-1, 3) * P["dx"]
    Phi_e = phi_at(P, Ext)
    Ue = torch.as_tensor(np.exp(-1j * Phi_e), **td); Le = lin(Ext)
    Aext = Le * Ue; Ag = Lg * Ug
    Next = Ag.conj().T @ Aext                                              # [V, Vext]
    ne = len(ext)
    ix = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"), -1).reshape(-1, 3)   # voxel index of r
    DD = np.stack(np.meshgrid(Dg, Dg, Dg, indexing="ij"), -1).reshape(-1, 3)                                 # [nd^3,3]
    rp = ix[:, None, :] - DD[None, :, :]                                                                      # r' index (may be outside)
    ext_idx = ((rp[..., 0] + (n - 1)) * ne + (rp[..., 1] + (n - 1))) * ne + (rp[..., 2] + (n - 1))
    ext_idx_t = torch.as_tensor(ext_idx, device=dev)
    TV = torch.gather(Next, 1, ext_idx_t)                                  # [V, nd^3]
    valid = torch.as_tensor(((rp >= 0) & (rp < n)).all(-1), device=dev)
    Ut, st, Vt = torch.linalg.svd(TV, full_matrices=False)
    ridx = ix[:, None, :]
    def rebuild_N(TVa):
        Na = torch.zeros(V, V, **td)
        # scatter valid TV entries back to N(r, r')
        rr = torch.as_tensor(np.broadcast_to(np.arange(V)[:, None], valid.shape).copy(), device=dev)[valid]
        cc = torch.as_tensor(((rp[..., 0] * n + rp[..., 1]) * n + rp[..., 2]), device=dev)[valid]
        Na[rr, cc] = TVa[valid]
        return Na
    for L in (2, 4, 6, 8, 12, 16, 24, 32):
        TVa = (Ut[:, :L] * st[:L]) @ Vt[:L]
        rows.append((f"prodconv(L={L})", L, err(rebuild_N(TVa))))
    print(f"{'method':22s} {'FFT pairs':>9s} {'rel.err':>9s}")
    for name, pairs, e in sorted(rows, key=lambda x: (x[0].split('(')[0], x[1])):
        print(f"{name:22s} {pairs:9d} {e:9.2e}")

if __name__ == "__main__":
    main()
