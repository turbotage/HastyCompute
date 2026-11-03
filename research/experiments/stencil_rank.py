"""Tensor structure of the 3-D chirp spectrum  c(m; w) = FT_{support}[ exp(-i sum_q w_q m_q(x)) ]  restricted to a (2s+1)^3 window of fine-grid shifts m.
A stencil for a modulated NUFFT row is (ES spreading) (x) c(.; w): if c has low CP rank R, generation and application cost ~ R * (3 small 1-D convolutions)
and R separable gathers instead of one dense (2s+1)^3 object per sample.
Fields m_q = z^2, x^2+y^2, xz, yz on the normalised cube [-1,1]^3; w drawn so max|Phi| = Phi_max."""
import numpy as np, torch, math, argparse

def chirp_spectrum(w, n, M, s, phimax):
    g = (np.arange(n) - (n - 1) / 2) / (n / 2)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    m = np.stack([Z ** 2, X ** 2 + Y ** 2, X * Z, Y * Z])
    ph = np.tensordot(w, m, 1)
    ph = ph * (phimax / np.abs(ph).max())
    f = np.zeros((M, M, M), complex); f[:n, :n, :n] = np.exp(-1j * ph)
    F = np.fft.fftn(f) / n ** 3
    idx = np.arange(-s, s + 1) % M
    return F[np.ix_(idx, idx, idx)]

def cp_als(T, R, iters=200, seed=0):
    T = torch.as_tensor(T, dtype=torch.complex128); d = T.shape
    g = torch.Generator().manual_seed(seed)
    A = [torch.randn(d[i], R, generator=g, dtype=torch.float64).to(torch.complex128) for i in range(3)]
    def kr(B, C): return (B[:, None, :] * C[None, :, :]).reshape(-1, R)
    for _ in range(iters):
        for mode in range(3):
            others = [A[i] for i in range(3) if i != mode]
            Tm = T.permute(mode, *[i for i in range(3) if i != mode]).reshape(d[mode], -1)
            Kr = kr(others[0], others[1])
            G = (others[0].conj().T @ others[0]) * (others[1].conj().T @ others[1])
            A[mode] = Tm @ Kr.conj() @ torch.linalg.pinv(G.T)
    rec = torch.einsum("ar,br,cr->abc", *A)
    return (torch.linalg.norm(rec - T) / torch.linalg.norm(T)).item()

if __name__ == "__main__":
    n, M, s = 32, 64, 5
    rng = np.random.default_rng(0)
    print(f"window {(2*s+1)}^3 of the chirp spectrum; relative error of rank-R CP approximation (and Tucker ranks at 1e-3)")
    for phimax in (1.0, 2.0, 4.0, 8.0):
        for label, w in (("diag only (z2,x2+y2)", np.array([1.0, 0.7, 0, 0])), ("all 4 fields", np.array([1.0, 0.7, 0.8, -0.6]))):
            T = chirp_spectrum(w, n, M, s, phimax)
            tot = np.linalg.norm(T)
            # multilinear (Tucker) ranks at relative tolerance 1e-3
            ranks = []
            for mode in range(3):
                U = np.linalg.svd(np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1), compute_uv=False)
                tail = np.sqrt(np.cumsum((U ** 2)[::-1])[::-1] / (U ** 2).sum())
                ranks.append(int(np.sum(tail > 1e-3)))
            errs = [cp_als(T, R) for R in (1, 2, 3, 4, 6)]
            print(f"Phi={phimax:4.1f} {label:22s} energy in window {np.linalg.norm(T)**2:.4f} | Tucker ranks {ranks} | CP R=1,2,3,4,6 err: " + " ".join(f"{e:.1e}" for e in errs))
