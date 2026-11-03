"""Does a butterfly-style hierarchical split of the sample set reduce the total number of spatial basis functions (= FFTs per coil)?
Each group g of samples gets its own spatial basis of rank L_g (separate FFTs); a global basis has rank L.  We compare L vs sum_g L_g for
groupings by time bin (late samples carry big phase, early ones almost none) and by time bin x direction cluster."""
import math, argparse
import numpy as np, torch
from conc_structure import *

def rank_of(E, tol, dev):
    s = torch.linalg.svdvals(torch.as_tensor(E, dtype=torch.complex64, device=dev)).double().cpu().numpy() ** 2
    tail = s.sum() - np.cumsum(s)
    return int(np.searchsorted(-np.sqrt(tail.clip(min=0) / s.sum()), -tol)) + 1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=16); ap.add_argument("--spokes", type=int, default=120); ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--tol", type=float, default=1e-3); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    rng = np.random.default_rng(3)
    ax = (np.arange(a.n) - a.n / 2 + 0.5) * 0.32 / a.n
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij"); r = np.stack([X, Y, Z], -1).reshape(-1, 3)
    print(f"tol={a.tol}; columns: max|Phi| | global L | sum over G time bins | sum over G x D (time bin x direction cluster)")
    for kind, B0 in (("twisty", 1.0), ("twisty", 0.3), ("curly", 1.0), ("curly", 0.3)):
        t, k0, G0, dt = make_prototype(kind, a.T, 2e-3, math.pi / (0.32 / 256.0))
        W0 = cumtrapz(np.einsum("ta,tb->tab", G0, G0), dt)
        Rs = [rand_rot(rng) for _ in range(a.spokes)]
        phi = np.stack([phase_conc(np.einsum("ac,tcd,bd->tab", A, W0, A), r, B0) for A in Rs])     # [S,T,V]
        E = np.exp(-1j * phi)
        Lg = rank_of(E.reshape(-1, E.shape[-1])[rng.choice(a.spokes * a.T, 5000, replace=False)], a.tol, a.device)
        out = []
        for G in (2, 4, 8):
            edges = np.linspace(0, a.T, G + 1).astype(int)
            tot = sum(rank_of(E[:, edges[g]:edges[g + 1]].reshape(-1, E.shape[-1]), a.tol, a.device) for g in range(G))
            out.append(f"G={G}: {tot}")
        # direction clusters via k-means-lite on the rotated z axis
        zd = np.stack([A[:, 2] for A in Rs]); zd = zd * np.sign(zd[:, :1] + 1e-9)
        D = 4; cent = zd[rng.choice(a.spokes, D, replace=False)]
        for _ in range(10):
            lab = np.argmax(np.abs(zd @ cent.T), 1)
            for d in range(D):
                if (lab == d).any(): cent[d] = zd[lab == d].mean(0); cent[d] /= np.linalg.norm(cent[d])
        G = 4; edges = np.linspace(0, a.T, G + 1).astype(int)
        tot = 0
        for g in range(G):
            for d in range(D):
                sel = lab == d
                if sel.sum() == 0: continue
                tot += rank_of(E[sel][:, edges[g]:edges[g + 1]].reshape(-1, E.shape[-1]), a.tol, a.device)
        out.append(f"G=4,D=4: {tot}")
        print(f"{kind:7s} B0={B0:3.1f} max|Phi|={np.abs(phi).max():5.1f} | global L={Lg:3d} | " + " | ".join(out))

if __name__ == "__main__":
    main()
