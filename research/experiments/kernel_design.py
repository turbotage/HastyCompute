"""Joint design of the shared image-domain correction c(x) for a CHIRP-AWARE NUFFT (the 'different kernel' question).
Plain NUFFT: f(x)=e^{-ikx} ~ c(x) sum_m Gamma_m e^{-i u_m x}, c = 1/psi_ES.  For chirp rows f=e^{-i(kx+mu x^2/2)} the best J-tap stencils (LS) with an
ES-derived c need many more taps (stencil_width2.py).  Here c(x) is optimised jointly for a whole family  {frac(k)} x {Phi in [0, Phi_max]}
by alternating minimisation (Gamma_j: per-sample least squares; c(x): closed form per voxel).  Reports worst-case error on held-out (k, Phi)."""
import numpy as np, math, sys
from stencil_width2 import es_psi

def build(N, sigma, k, mu, J, x):
    M = int(round(sigma * N))
    f = np.exp(-1j * (k * x + 0.5 * mu * x ** 2))
    m0 = int(np.floor(k * M / (2 * math.pi))) - J // 2 + 1
    u = 2 * math.pi * (m0 + np.arange(J)) / M
    return f, np.exp(-1j * np.outer(x, u))

def solve(f, B, c):
    A = c[:, None] * B
    g, *_ = np.linalg.lstsq(A, f, rcond=None)
    return g, A @ g

def worst(N, sigma, J, c, x, Phis, fracs):
    M = int(round(sigma * N)); w = 0
    for Phi in Phis:
        mu = 2 * Phi / (N / 2) ** 2
        for fr in fracs:
            k = 2 * math.pi * (37 + fr) / M
            f, B = build(N, sigma, k, mu, J, x)
            _, a = solve(f, B, c)
            w = max(w, np.linalg.norm(a - f) / np.linalg.norm(f))
    return w

def design(N, sigma, J, x, Phis, fracs, c0, iters=40):
    M = int(round(sigma * N)); c = c0.copy()
    fam = []
    for Phi in Phis:
        mu = 2 * Phi / (N / 2) ** 2
        for fr in fracs:
            k = 2 * math.pi * (37 + fr) / M
            fam.append(build(N, sigma, k, mu, J, x))
    for it in range(iters):
        num = np.zeros(len(x), complex); den = np.zeros(len(x))
        for f, B in fam:
            g, _ = solve(f, B, c)
            h = B @ g
            num += np.conj(h) * f; den += np.abs(h) ** 2
        c = num / den
        c = c / np.abs(c).max() * np.abs(c0).max()
    return c

if __name__ == "__main__":
    N = 64; x = np.arange(N) - (N - 1) / 2
    test_phis = np.linspace(0, 1, 6) ; fr_test = (0.0, 0.13, 0.37, 0.5, 0.81)
    for sigma in (2.0, 1.5):
        print(f"\nsigma={sigma}")
        for Phimax in (2.0, 4.0):
            tp = np.linspace(0, Phimax, 9); train_phis = np.linspace(0, Phimax, 5)
            print(f" design family: Phi in [0,{Phimax}]")
            for J in (6, 8, 10, 12):
                best_es = None
                for wc in range(2, 13):
                    c = 1.0 / es_psi(N, sigma, wc, x)
                    e = worst(N, sigma, J, c, x, tp, fr_test)
                    if best_es is None or e < best_es[0]: best_es = (e, wc, c)
                cdes = design(N, sigma, J, x, train_phis, (0.0, 0.25, 0.5, 0.75), best_es[2].astype(complex))
                ed = worst(N, sigma, J, cdes, x, tp, fr_test)
                print(f"   J={J:2d}: best ES-derived c (w_c={best_es[1]}): worst err {best_es[0]:.1e}   |   jointly designed c: {ed:.1e}   (gain {best_es[0]/ed:.1f}x)")
