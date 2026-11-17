"""Segmented Toeplitz normal operator: is the L x L kernel matrix low-rank per k-space point?

2-D, N = 64, spiral-out (24 interleaves, 8 ms), smooth field +-225 Hz (3.6 phase cycles), L time segments with LS
interpolators (Sutton 2003).  N_seg = sum_{l,l'} D_l^H T_{ll'} D_l', T_{ll'} Toeplitz with kernel
t_{ll'}(Delta) = sum_j conj(a_l(tau_j)) a_l'(tau_j) e^{i 2 pi k_j . Delta / N}, embedded in a 2N circulant.
Per 2N-grid frequency kappa, That(kappa) is an L x L Hermitian matrix.  We truncate its eigen-expansion to rank R and
measure ||N_R x - N_seg x|| / ||N_seg x||, plus a parametric rank-1 model rho(kappa) a(tau(|kappa|)) a(.)^H that
stores 2 reals per kappa (tau from the trajectory's |k| -> t map).
"""
import numpy as np, time

rng = np.random.default_rng(3)
N, NI, NS, T = 64, 24, 1500, 8e-3
tau = (np.arange(NS) + 0.5) / NS * T
kr = (N / 2) * tau / T                                   # |k| proportional to time (constant-velocity spiral)
turns = N / (2 * NI) * 1.0
kk = []
for i in range(NI):
    ang = 2 * np.pi * turns * tau / T + 2 * np.pi * i / NI
    kk.append(np.stack([kr * np.cos(ang), kr * np.sin(ang)], -1))
k = np.concatenate(kk)                                    # (M, 2) cycles / FOV
tj = np.tile(tau, NI)
M = len(tj)

r1 = np.arange(-N // 2, N // 2)
RX, RY = np.meshgrid(r1, r1, indexing="ij")
w = 2 * np.pi * (225 * np.tanh(RX / 14.0) * 0.6 + 120 * np.exp(-((RX - 8) ** 2 + (RY + 5) ** 2) / 90.0) - 60 * (RY / 32.0) ** 2)
w = w - (w.max() + w.min()) / 2
w = w / np.abs(w).max() * 2 * np.pi * 225
print("phase cycles", (w.max() - w.min()) / (2 * np.pi) * T)

x_white = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
x_obj = (1 / (1 + np.exp((np.sqrt((RX / 26.0) ** 2 + (RY / 22.0) ** 2) - 1) * 10))) * (1 + 0.4 * np.cos(RX / 4.0))


def seg_basis(L):
    tl = (np.arange(L) + 0.5) / L * T
    D = np.exp(-1j * w[None] * tl[:, None, None])              # (L, N, N)
    Df = D.reshape(L, -1)
    G = Df.conj() @ Df.T
    tt = np.linspace(0, T, 400)
    beta = Df.conj() @ np.exp(-1j * np.outer(w.reshape(-1), tt))  # (L, 400)
    A = np.linalg.solve(G + 1e-10 * np.trace(G).real / L * np.eye(L), beta)  # (L, 400)
    a = np.array([np.interp(tj, tt, A[l].real) + 1j * np.interp(tj, tt, A[l].imag) for l in range(L)])  # (L, M)
    return D, a, (tt, A)


def apply_exact(x):
    xf = x.reshape(-1)
    rr = np.stack([RX.reshape(-1), RY.reshape(-1)], -1)
    out = np.zeros(N * N, complex)
    for c in range(0, M, 2000):
        E = np.exp(-1j * (2 * np.pi * k[c:c + 2000] @ rr.T / N + np.outer(tj[c:c + 2000], w.reshape(-1))))
        out += E.conj().T @ (E @ xf)
    return out.reshape(N, N)


def kernels(a):
    L = a.shape[0]
    d = np.arange(-N, N)                                       # 2N lags, circulant order after ifftshift
    Ex = np.exp(1j * 2 * np.pi * np.outer(k[:, 0], d) / N)     # (M, 2N)
    Ey = np.exp(1j * 2 * np.pi * np.outer(k[:, 1], d) / N)
    Th = np.zeros((L, L, 2 * N, 2 * N), complex)
    for l in range(L):
        for lp in range(l, L):
            wts = a[l].conj() * a[lp]
            t = (Ex * wts[:, None]).T @ Ey                      # t(dx, dy)
            t[0, :] = 0; t[:, 0] = 0                            # lag -N is outside the support (|Delta| <= N-1)
            Th[l, lp] = np.fft.fft2(np.fft.ifftshift(t))
            if lp != l:
                Th[lp, l] = Th[l, lp].conj()
    return Th


def apply_toep(x, D, Th):
    L = D.shape[0]
    V = np.zeros((L, 2 * N, 2 * N), complex)
    for l in range(L):
        p = np.zeros((2 * N, 2 * N), complex)
        p[:N, :N] = D[l] * x
        V[l] = np.fft.fft2(p)
    U = np.einsum("abxy,bxy->axy", Th, V)
    out = np.zeros((N, N), complex)
    for l in range(L):
        out += D[l].conj() * np.fft.ifft2(U[l])[:N, :N]
    return out


def lowrank(Th, R):
    L = Th.shape[0]
    H = np.moveaxis(Th, (0, 1), (2, 3))                         # (2N,2N,L,L)
    lam, U = np.linalg.eigh(H)
    order = np.argsort(-np.abs(lam), axis=-1)
    lam = np.take_along_axis(lam, order, -1)
    U = np.take_along_axis(U, order[..., None, :], -1)
    lam[..., R:] = 0
    Hr = np.einsum("xyar,xyr,xybr->xyab", U, lam, U.conj())
    frac = (np.abs(np.take_along_axis(np.linalg.eigh(H)[0], order, -1))[..., :R].sum(-1) / np.abs(np.linalg.eigh(H)[0]).sum(-1))
    return np.moveaxis(Hr, (2, 3), (0, 1)), frac


def parametric(Th, tabA):
    tt, A = tabA
    L = Th.shape[0]
    kap = np.fft.fftfreq(2 * N, d=1.0 / (2 * N)) / 2.0           # cycles / FOV on the 2N grid
    KX, KY = np.meshgrid(kap, kap, indexing="ij")
    tk = np.clip(np.hypot(KX, KY) / (N / 2) * T, 0, T)
    av = np.array([np.interp(tk, tt, A[l].real) + 1j * np.interp(tk, tt, A[l].imag) for l in range(L)])  # (L,2N,2N)
    tr = np.einsum("aaxy->xy", Th).real
    rho = tr / np.maximum((np.abs(av) ** 2).sum(0), 1e-12)
    return rho[None, None] * av[:, None] * av[None, :].conj()


if __name__ == "__main__":
    for L in [6, 9, 12]:
        t0 = time.time()
        D, a, tab = seg_basis(L)
        Th = kernels(a)
        res = {}
        for nm, x in [("white", x_white), ("object", x_obj)]:
            ye = apply_exact(x)
            ys = apply_toep(x, D, Th)
            res[nm] = (ye, ys)
            print(f"L={L} {nm}: segmented-Toeplitz vs exact normal op {np.linalg.norm(ys - ye) / np.linalg.norm(ye):.2e}")
        for R in [1, 2, 3, 4]:
            ThR, frac = lowrank(Th, R)
            msg = []
            for nm, x in [("white", x_white), ("object", x_obj)]:
                ye, ys = res[nm]
                yr = apply_toep(x, D, ThR)
                msg.append(f"{nm} vs seg {np.linalg.norm(yr - ys) / np.linalg.norm(ys):.2e} vs exact {np.linalg.norm(yr - ye) / np.linalg.norm(ye):.2e}")
            print(f"  rank {R}: |lambda| captured median {np.median(frac):.4f} min {frac.min():.4f}; " + "; ".join(msg))
        ThP = parametric(Th, tab)
        msg = []
        for nm, x in [("white", x_white), ("object", x_obj)]:
            ye, ys = res[nm]
            yr = apply_toep(x, D, ThP)
            msg.append(f"{nm} vs seg {np.linalg.norm(yr - ys) / np.linalg.norm(ys):.2e}")
        print("  parametric rank-1 (2 reals/kappa): " + "; ".join(msg), f"({time.time() - t0:.0f} s)", flush=True)
