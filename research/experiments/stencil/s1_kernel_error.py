"""1-D interpolation error of gridding kernels and their derivatives, as used by the Taylor (derivative) stencil.

For a sample at grid coordinate k = (integer) + f and image frequency xi = 2 pi x / (sigma N) in [-pi/sigma, pi/sigma]:
    S_n(f, xi) = sum_m phi^(n)(f - m) e^{i (f - m) xi}   ~   phihat(xi) (-i xi)^n
    e_n(f, xi) = |S_n / phihat(xi) - (-i xi)^n| / (pi/sigma)^n          (relative to the largest value of the term)
So e_0 is the plain NUFFT error and e_2 (e_4) multiplies chi_max (chi_max^2 / 2) in the stencil error budget.
'ls-floor' = per-offset least-squares taps (Fessler-Sutton type, scaling = ES phihat): the best any W-tap stencil can do
for that order with that deapodisation (each order designed separately).
Output: markdown tables (max over f, xi) + a CSV.
"""
import numpy as np, sys, itertools, csv
from kern import Kernel, es_beta, kb_beta

PI = np.pi
NF, NX = 48, 241


def errors(ker, sigma, nmax=4):
    W = ker.W
    xi = np.linspace(-PI / sigma, PI / sigma, NX)
    ph = ker.phihat(xi)
    fs = (np.arange(NF) + 0.5) / NF
    E = np.zeros((nmax + 1, NF, NX))
    for i, f in enumerate(fs):
        m = np.arange(int(np.ceil(f - W / 2)), int(np.floor(f + W / 2)) + 1)
        t = f - m
        d = ker.derivs_t(t, nmax)                     # (n, taps)
        ph_t = np.exp(1j * np.outer(t, xi))           # (taps, xi)
        S = d @ ph_t
        for n in range(nmax + 1):
            E[n, i] = np.abs(S[n] / ph - (-1j * xi) ** n) / (PI / sigma) ** n
    return E


def ls_floor(W, sigma, nmax=4):
    ker = Kernel("es", W, sigma)
    xi = np.linspace(-PI / sigma, PI / sigma, NX)
    ph = ker.phihat(xi)
    fs = (np.arange(NF) + 0.5) / NF
    E = np.zeros((nmax + 1, NF, NX))
    for i, f in enumerate(fs):
        m = np.arange(int(np.ceil(f - W / 2)), int(np.ceil(f - W / 2)) + W)
        t = f - m
        B = np.exp(1j * np.outer(xi, t)) / ph[:, None]      # relative
        Br = np.vstack([B.real, B.imag])
        for n in range(nmax + 1):
            tgt = (-1j * xi) ** n
            g, *_ = np.linalg.lstsq(Br, np.concatenate([tgt.real, tgt.imag]), rcond=None)
            E[n, i] = np.abs(B @ g - tgt) / (PI / sigma) ** n
    return E


def summarize(E):
    return [E[n].max() for n in range(E.shape[0])], [np.sqrt((E[n] ** 2).mean()) for n in range(E.shape[0])]


if __name__ == "__main__":
    sigmas = [1.125, 1.25, 1.5, 2.0]
    Ws = [4, 5, 6, 7, 8, 10, 12]
    rows = []
    for sigma in sigmas:
        for W in Ws:
            cands = {
                "es": Kernel("es", W, sigma),
                "es-noclamp": Kernel("es", W, sigma, clamp=None),
                "kb": Kernel("kb", W, sigma),
            }
            # beta scan (optimise the combined budget e0 + 0.7 e2 + 0.245 e4, max-norm) for ES (clamped) and KB
            for kind in ["es", "kb"]:
                b0 = es_beta(W, sigma) if kind == "es" else kb_beta(W, sigma)
                best = None
                for fac in np.linspace(0.70, 1.15, 19):
                    k = Kernel(kind, W, sigma, beta=fac * b0)
                    mx, _ = summarize(errors(k, sigma))
                    c = mx[0] + 0.7 * mx[2] + 0.245 * mx[4]
                    if best is None or c < best[0]:
                        best = (c, fac)
                cands[kind + "-opt"] = Kernel(kind, W, sigma, beta=best[1] * b0)
            for name, k in cands.items():
                mx, rms = summarize(errors(k, sigma))
                rows.append([sigma, W, name, k.beta] + mx + rms)
            mx, rms = summarize(ls_floor(W, sigma))
            rows.append([sigma, W, "ls-floor", float("nan")] + mx + rms)
            print(f"sigma {sigma} W {W} done", file=sys.stderr)
    hdr = ["sigma", "W", "kernel", "beta"] + [f"max_e{n}" for n in range(5)] + [f"rms_e{n}" for n in range(5)]
    with open("s1_kernel_error.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
    for r in rows:
        print("| " + " | ".join([f"{r[0]}", f"{r[1]}", r[2], f"{r[3]:.2f}"] + [f"{v:.1e}" for v in r[4:9]] + [f"{v:.1e}" for v in r[9:14]]) + " |")
