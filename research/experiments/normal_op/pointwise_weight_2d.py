"""Q2/Q4: accuracy of 'kernel-free' per-sample weights in the Toeplitz form.

If a per-sample weight g_j is a function of the k-space position, g_j = g(k_j) (here: readout time tau on one half of
an out-and-back readout, tau = tau_h(|k|); segment interpolator products a_l(tau) a_l'(tau)^*; fat c(tau)), the
weighted Toeplitz multiplier is approximated by the pointwise product  T_g(u) ~= g(u) T(u)  on the embedding grid.
In Delta-space this is a convolution of the unweighted kernel K with g^ (the transform of g), so it is exact only if
the embedding is large enough to hold K on |Delta| < N + b (b = support of g^) and g is smooth.

2-D test (cheap, exact reference by dense matrices): N x N image, M_spokes out-and-back radial readouts, tau linear
in arc length; g(tau) = exp(-i 2 pi f tau T) (f = number of phase cycles over the readout, i.e. a segment-product /
off-resonance-like weight).  Error = ||N_exact - N_approx||_2 / ||N_exact||_2 (power iteration on the difference),
for guard bands b = 0, 4, 8, 16, 32 cells (embedding 2N + 2b), per half separately (two densities) vs ignoring halves.

run: rustlib/.venv/bin/python -I research/experiments/normal_op/pointwise_weight_2d.py
"""
import json
import os

import numpy as np

rng = np.random.default_rng(3)
N = 40
nsp, nro = 160, 120          # spokes, samples per spoke (out 60 + back 60)
half = nro // 2
s = np.concatenate([np.linspace(0, 1, half, endpoint=False), np.linspace(1, 0, half, endpoint=False)])  # radius / kmax
tau = np.linspace(0, 1, nro, endpoint=False)
ang = rng.uniform(0, 2 * np.pi, nsp)
ang2 = ang + rng.uniform(-0.3, 0.3, nsp)          # back half on a slightly different angle (yarn-like curvature)
kx = np.concatenate([np.outer(np.cos(ang), s[:half]), np.outer(np.cos(ang2), s[half:])], 1) * np.pi
ky = np.concatenate([np.outer(np.sin(ang), s[:half]), np.outer(np.sin(ang2), s[half:])], 1) * np.pi
kx, ky = kx.ravel(), ky.ravel()
tt = np.tile(tau, nsp)
hh = np.tile((np.arange(nro) >= half).astype(int), nsp)

g1 = np.arange(N) - N // 2
X, Y = np.meshgrid(g1, g1, indexing="ij")
X, Y = X.ravel(), Y.ravel()


def kernel(dx, dy, wts):
    """K(Delta) = sum_j w_j exp(i k_j . Delta) on a grid of Delta (exact)"""
    out = np.zeros(dx.size, complex)
    for c in range(0, kx.size, 2000):
        out += np.exp(1j * (np.outer(dx, kx[c:c + 2000]) + np.outer(dy, ky[c:c + 2000]))) @ wts[c:c + 2000]
    return out


def exact_normal(wts):
    E = np.exp(-1j * (np.outer(kx, X) + np.outer(ky, Y)))
    return E.conj().T @ (wts[:, None] * E)


def toeplitz_apply_matrix(T):
    """dense matrix of x -> crop(IFFT(T * FFT(pad(x)))) on the embedding grid of size T.shape"""
    Mg = T.shape[0]
    cols = []
    for i in range(N * N):
        x = np.zeros((Mg, Mg), complex)
        x[X[i] % Mg, Y[i] % Mg] = 1.0
        y = np.fft.ifft2(T * np.fft.fft2(x))
        cols.append(y[X % Mg, Y % Mg])
    return np.stack(cols, 1)


def opnorm(A, it=60):
    v = rng.standard_normal(A.shape[1]) + 1j * rng.standard_normal(A.shape[1])
    for _ in range(it):
        v = A.conj().T @ (A @ v)
        v /= np.linalg.norm(v)
    return np.sqrt(np.linalg.norm(A.conj().T @ (A @ v)))


def multiplier(Mg, b, wts):
    """exact Toeplitz multiplier on an Mg grid: K on |Delta| < N + b (true values), zero beyond, DFT"""
    d = np.arange(Mg)
    d = np.where(d < Mg // 2, d, d - Mg)
    DX, DY = np.meshgrid(d, d, indexing="ij")
    m = (np.abs(DX) < N + b) & (np.abs(DY) < N + b)
    K = np.zeros((Mg, Mg), complex)
    K[m] = kernel(DX[m].astype(float), DY[m].astype(float), wts)
    return np.fft.fft2(K)          # T(u) = sum_Delta K(Delta) e^{-i u Delta}; u_m = 2 pi m / Mg


def g_on_grid(Mg, f, h):
    """g(tau_h(|u|)): tau of half h as a function of the radius"""
    u = np.fft.fftfreq(Mg) * 2 * np.pi
    UX, UY = np.meshgrid(u, u, indexing="ij")
    r = np.minimum(np.sqrt(UX ** 2 + UY ** 2) / np.pi, 1.0)
    t = r * 0.5 if h == 0 else 1.0 - 0.5 * r
    return np.exp(-1j * 2 * np.pi * f * t)


res = {}
for f in (1.0, 4.0):
    gj = np.exp(-1j * 2 * np.pi * f * tt)
    Nex = exact_normal(gj)
    nN = opnorm(Nex)
    row = {}
    for b in (0, 4, 8, 16, 32):
        Mg = 2 * N + 2 * b
        Nap = np.zeros_like(Nex)
        for h in (0, 1):
            T = multiplier(Mg, b, (hh == h).astype(float))
            Nap += toeplitz_apply_matrix(g_on_grid(Mg, f, h) * T)
        row[str(b)] = float(opnorm(Nex - Nap) / nN)
        print(f"f {f}: guard {b}: rel op-norm error {row[str(b)]:.2e}", flush=True)
    # baseline: exact weighted multiplier (no pointwise approximation), 2N grid -> checks the harness
    T = multiplier(2 * N, 0, gj)
    row["exact_weighted_kernel_2N"] = float(opnorm(Nex - toeplitz_apply_matrix(T)) / nN)
    res[f"f{f}"] = row
    print(f, row, flush=True)
od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(od, exist_ok=True)
json.dump(res, open(os.path.join(od, "pointwise_weight_2d.json"), "w"), indent=1)
