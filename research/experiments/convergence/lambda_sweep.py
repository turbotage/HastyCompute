#!/usr/bin/env python3
"""lambda sweep: best error to truth of the unweighted (BLUE data term) vs density-weighted minimisers, FISTA to
convergence. Same toy as rates_toy.py. Usage: python -I lambda_sweep.py <paper dir> <shots> <k0>"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import experiments as E  # noqa: E402
N = E.N; rng = np.random.default_rng(5)
x0 = E.shepp_logan(N); S = E.make_coils(N, E.NC)
k = np.fft.fftfreq(N) * N; KX, KY = np.meshgrid(k, k)
SH, K0 = int(sys.argv[2]), float(sys.argv[3])
nu = 1.0 / (1.0 + (KX ** 2 + KY ** 2) / K0 ** 2) ** 1.5; nu /= nu.sum()
counts = E.draw_counts(nu, SH, rng); op = E.Op(S, counts)
sigma = 0.01 * op.data_rms(x0)
rng2 = np.random.default_rng(6)
G = [(rng2.standard_normal((N, N)) + 1j * rng2.standard_normal((N, N))) * sigma * np.sqrt(counts) for _ in range(E.NC)]
def make(weights):
    def H(v):
        y = np.zeros((N, N))
        for c in range(E.NC):
            y += np.real(np.conj(S[c]) * np.fft.ifft2(np.fft.fft2(S[c] * v, norm="ortho") * counts * weights, norm="ortho"))
        return y
    L = E.power_norm(H)
    b = sum(np.real(np.conj(S[c]) * np.fft.ifft2((np.fft.fft2(S[c] * x0, norm="ortho") * counts + G[c]) * weights, norm="ortho")) for c in range(E.NC))
    return (lambda v: H(v) / L), b / L
prox = lambda z, t: E.ihaar(E.soft(E.haar(z), t))
def fista(H, b, lam, iters=3000):
    x = np.zeros((N, N)); xp = x.copy(); t = 1.0
    for _ in range(iters):
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2; yk = x + ((t - 1) / tn) * (x - xp); t = tn
        xp = x; x = prox(yk - (H(yk) - b), lam)
    return x, np.linalg.norm(x - xp) / np.linalg.norm(x)
LAMS = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else (3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7)
wk = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
for name, w in (("unweighted (BLUE data term)", np.ones_like(counts)), ("density-weighted (DCW)", wk)):
    H, b = make(w)
    row = []
    for lam in LAMS:
        x, step = fista(H, b, lam)
        row.append((lam, np.linalg.norm(x - x0) / np.linalg.norm(x0), step))
    best = min(row, key=lambda r: r[1])
    print(f"{name}: " + "  ".join(f"λ {l:.0e}: {e:.3f}" for l, e, _ in row) + f"  → best {best[1]:.3f} at λ {best[0]:.0e} (last step {best[2]:.1e})")
