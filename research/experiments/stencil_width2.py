"""Stencil width for chirp-modulated NUFFT rows WITH a shared image-domain correction c(x) (as in every NUFFT: the apodization division).
Model (1-D): f(x) = exp(-i(kx + 0.5 mu x^2)),  f(x) ~ c(x) * sum_{m in window(J)} Gamma_m exp(-i u_m x)   on the image support,
c(x) = 1/psi(x) with psi the FT of an ES(w_c) kernel (independent of the sample), Gamma least-squares per sample.
Prints, for a tolerance, the smallest J (over w_c in 2..14) vs Phi and sigma (worst of two fractional k)."""
import numpy as np, math

def es_psi(N, sigma, w, x):
    M = sigma * N; du = 2 * math.pi / M; beta = 2.30 * w
    zz = np.linspace(-1, 1, 4001); ph = np.exp(beta * (np.sqrt(np.clip(1 - zz ** 2, 0, None)) - 1))
    return np.array([np.trapezoid(ph * np.exp(-1j * xi * zz * du * w / 2), zz) for xi in x]).real * (du * w / 2) / du

def err(N, sigma, k, mu, J, c, x):
    M = int(round(sigma * N))
    f = np.exp(-1j * (k * x + 0.5 * mu * x ** 2))
    m0 = int(np.floor(k * M / (2 * math.pi))) - J // 2 + 1
    ms = m0 + np.arange(J); u = 2 * math.pi * ms / M
    A = c[:, None] * np.exp(-1j * np.outer(x, u))
    g, *_ = np.linalg.lstsq(A, f, rcond=None)
    return np.linalg.norm(A @ g - f) / np.linalg.norm(f)

if __name__ == "__main__":
    N = 64; x = np.arange(N) - (N - 1) / 2
    tols = (1e-2, 1e-3, 1e-4, 1e-5)
    for sigma in (2.0, 1.5, 1.25):
        M = int(round(sigma * N))
        cs = {w: 1.0 / es_psi(N, sigma, w, x) for w in range(2, 15, 1)}
        print(f"\nsigma={sigma}: minimal J (taps/axis); best over shared-correction widths w_c;   plain NUFFT w for reference = column Phi=0")
        print("  Phi[rad] | " + " | ".join(f"tol {t:g}" for t in tols))
        for Phi in (0.0, 1.0, 2.0, 4.0, 8.0):
            mu = 2 * Phi / (N / 2) ** 2
            row = []
            for tol in tols:
                best = 99
                for wc, c in cs.items():
                    worst = 0
                    for frac in (0.0, 0.5):
                        k = 2 * math.pi * (37 + frac) / M
                        J = next((J for J in range(2, 40) if err(N, sigma, k, mu, J, c, x) < tol), 99)
                        worst = max(worst, J)
                    best = min(best, worst)
                row.append(best)
            print(f"  {Phi:7.1f} | " + " | ".join(f"{r:7d}" for r in row))
