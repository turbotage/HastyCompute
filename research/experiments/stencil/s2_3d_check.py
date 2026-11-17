"""3-D check: chirped type-2 NUFFT with derivative stencils vs an f64 direct sum.

y_j = sum_r x(r) exp(-i kappa_j . 2 pi r / N - i chi_j(rho)),  rho = r / (N/2) in [-1,1]^3,  chi_j = rho^T Q_j rho.
Gridded: G = FFT_n[x / phihat] (n = sigma N), y_j = sum_u Gamma_j(u) G(u), Gamma_j = p_j(X) phi(k_j - u),
X = chi(i d_k) = -sum_pq Q'_pq d_p d_q (Q' = Q (sigma/pi)^2 in grid-radian units), p_j a degree-2 polynomial
(taylor: 1 - i s - s^2/2 ; cheb: Chebyshev interpolant of e^{-is} on the range of chi_j over the FOV).
Reports relative error ||y - y_direct|| / ||y_direct|| for a white random image (worst case for edge errors) and
for a smooth object (ellipsoid with soft edge, the energy concentrated near the FOV centre, as in practice).
"""
import numpy as np, itertools, sys
from kern import Kernel

rng = np.random.default_rng(7)
N, M = 32, 1500
r1 = np.arange(-N // 2, N // 2)
RX, RY, RZ = np.meshgrid(r1, r1, r1, indexing="ij")
rho = np.stack([RX, RY, RZ], -1).reshape(-1, 3) / (N / 2)

x_white = (rng.standard_normal((N,) * 3) + 1j * rng.standard_normal((N,) * 3)) / np.sqrt(2)
el = (RX / 13.0) ** 2 + (RY / 11.0) ** 2 + (RZ / 14.0) ** 2
x_obj = (1 / (1 + np.exp((np.sqrt(el) - 1) * 12))) * (1 + 0.3 * np.cos(RX / 3.0) * np.sin(RY / 4.0))
x_obj = x_obj * np.exp(1j * 0.5 * RZ / N)
kappa = rng.uniform(-N / 2, N / 2, (M, 3))          # cycles per FOV
corners = np.array(list(itertools.product([-1, 1], repeat=3)), float)
g9 = np.linspace(-1, 1, 9)
probe = np.stack(np.meshgrid(g9, g9, g9, indexing="ij"), -1).reshape(-1, 3)


def make_Q(a, kind):
    Qs = []
    for _ in range(M):
        if kind == "definite":
            Q = np.diag([0.0, 0.0, 1.0]) if rng.random() < 0.5 else np.diag([1.0, 1.0, 0.0])
        else:
            B1, B2 = rng.standard_normal((2, 3, 3))
            Q = B1 @ B1.T - B2 @ B2.T
        s = np.einsum("ni,ij,nj->n", probe, Q, probe)
        Qs.append(Q * a / max(np.abs(s).max(), 1e-12) * rng.uniform(0.2, 1.0))   # spread of |chi| over samples
    return np.array(Qs)


def direct(x, Q):
    y = np.zeros(M, complex)
    xf = x.reshape(-1)
    for c in range(0, M, 100):
        ph = 2 * np.pi * kappa[c:c + 100] @ (rho.T * (N / 2)) / N
        ch = np.einsum("ni,mij,nj->mn", rho, Q[c:c + 100], rho)
        y[c:c + 100] = np.exp(-1j * (ph + ch)) @ xf
    return y


def poly_coefs(Q, mode):
    c = np.zeros((M, 3), complex)
    for j in range(M):
        s = np.einsum("ni,ij,nj->n", probe, Q[j], probe)
        m, Mx = s.min(), s.max()
        if mode == "taylor":
            c[j] = [1, -1j, -0.5]
        elif mode == "order0":
            c[j] = [1, 0, 0]
        else:
            sc, hw = 0.5 * (m + Mx), 0.5 * (Mx - m)
            nodes = sc + hw * np.cos((2 * np.arange(3) + 1) * np.pi / 6)
            if hw < 1e-12:
                nodes = sc + np.array([-1e-6, 0, 1e-6])
            V = np.vander(nodes, 3, increasing=True).astype(complex)
            c[j] = np.linalg.solve(V, np.exp(-1j * nodes))
    return c


def monomial_coefs(Qg, c):
    """p(X) with X = -d^T Q d  ->  dict alpha -> (M,) complex coefficient."""
    out = {(0, 0, 0): c[:, 0].copy()}
    X = {}
    for p in range(3):
        for q in range(3):
            a = [0, 0, 0]; a[p] += 1; a[q] += 1; a = tuple(a)
            X[a] = X.get(a, 0) - Qg[:, p, q]
    for a, v in X.items():
        out[a] = out.get(a, 0) + c[:, 1] * v
    for (a, v), (b, w) in itertools.product(X.items(), X.items()):
        ab = tuple(np.add(a, b))
        out[ab] = out.get(ab, 0) + c[:, 2] * v * w
    return out


def gridded(x, Q, ker, sigma, c):
    n = int(round(sigma * N)); n += n % 2
    sig = n / N
    xi = 2 * np.pi * r1 / n
    ph = ker.phihat(xi)
    xp = np.zeros((n,) * 3, complex)
    idx = np.mod(r1, n)
    xp[np.ix_(idx, idx, idx)] = x / (ph[:, None, None] * ph[None, :, None] * ph[None, None, :])
    G = np.fft.fftn(xp)
    k = kappa * n / N
    W = ker.W
    i0 = np.ceil(k - W / 2).astype(int)                       # (M,3)
    cells = i0[:, :, None] + np.arange(W)                    # (M,3,W)
    D = ker.derivs_t(k[:, :, None] - cells, 4)               # (5,M,3,W)
    Qg = Q * (sig / np.pi) ** 2
    mono = monomial_coefs(Qg, c)
    wt = np.zeros((M, W, W, W), complex)
    for a, v in mono.items():
        wt += v[:, None, None, None] * D[a[0], :, 0, :, None, None] * D[a[1], :, 1, None, :, None] * D[a[2], :, 2, None, None, :]
    cm = np.mod(cells, n)
    vals = G[cm[:, 0, :, None, None], cm[:, 1, None, :, None], cm[:, 2, None, None, :]]
    return (wt * vals).sum(axis=(1, 2, 3))


if __name__ == "__main__":
    configs = [(1.25, 6), (1.25, 7), (1.25, 8), (1.25, 10), (1.5, 5), (1.5, 6), (1.5, 7), (2.0, 4), (2.0, 5), (2.0, 6)]
    print("| chirp | max chi | sigma | W | kernel | poly | white | object |")
    print("|---|---|---|---|---|---|---|---|")
    for kind, a in [("none", 0.0), ("indefinite", 0.3), ("indefinite", 0.7), ("definite", 0.7)]:
        Q = make_Q(a, kind) if a > 0 else np.zeros((M, 3, 3))
        yd = {nm: direct(xx, Q) for nm, xx in [("white", x_white), ("object", x_obj)]}
        polys = ["order0"] if a == 0 else ["taylor", "cheb"]
        cs = {p: poly_coefs(Q, p) for p in polys}
        for sigma, W in configs:
            for kk in ["es", "kb"]:
                ker = Kernel(kk, W, sigma)
                for p in polys:
                    e = []
                    for nm, xx in [("white", x_white), ("object", x_obj)]:
                        y = gridded(xx, Q, ker, sigma, cs[p])
                        e.append(np.linalg.norm(y - yd[nm]) / np.linalg.norm(yd[nm]))
                    print(f"| {kind} | {a} | {sigma} | {W} | {kk} | {p} | {e[0]:.1e} | {e[1]:.1e} |", flush=True)
        # truncation floor alone: wide kernel
        if a > 0:
            ker = Kernel("kb", 12, 2.0)
            for p in polys:
                e = []
                for nm, xx in [("white", x_white), ("object", x_obj)]:
                    y = gridded(xx, Q, ker, 2.0, cs[p])
                    e.append(np.linalg.norm(y - yd[nm]) / np.linalg.norm(yd[nm]))
                print(f"| {kind} | {a} | 2.0 | 12 (floor) | kb | {p} | {e[0]:.1e} | {e[1]:.1e} |", flush=True)
