"""Which degree-P polynomial p(s) should the derivative stencil use for e^{-i chi}?

The stencil sum_n c_n chi(i d_k)^n phi is exact on the plane wave up to E(chi(rho)) = e^{-i chi(rho)} - p(chi(rho)),
so the stencil error at image point rho depends on rho only through chi(rho).  Any p of degree P costs the same
(same derivative tables, complex coefficients).  Compare on the FOV cube rho in [-1,1]^3, chi = rho^T Q rho:
  taylor   : sum_{n<=P} (-i s)^n / n!                       (production)
  centred  : e^{-i s_c} sum (-i (s - s_c))^n / n!,  s_c = midrange of chi over the FOV (constant -> phi_0, free)
  cheb     : interpolant of e^{-is} at P+1 Chebyshev nodes of [m, M] = range of chi (near-minimax)
  ls       : least squares over the FOV points (operator-norm optimal for white images)
Reported: rms and max over the FOV of |E|, for definite Q (single concomitant monomial, worst case for Taylor) and
indefinite Q (difference of two random PSD matrices: per-sample deviations Delta C_j from the encode mean).
"""
import numpy as np

rng = np.random.default_rng(1)
g = np.linspace(-1, 1, 33)
R = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def chi_of(Q):
    return np.einsum("ni,ij,nj->n", R, Q, R)




def fact(n):
    r = 1
    for k in range(2, n + 1):
        r *= k
    return r


def errs(s, P):
    m, M = s.min(), s.max()
    sc = 0.5 * (m + M)
    tgt = np.exp(-1j * s)
    res = {}
    res["taylor"] = tgt - sum((-1j * s) ** n / fact(n) for n in range(P + 1))
    res["centred"] = tgt - np.exp(-1j * sc) * sum((-1j * (s - sc)) ** n / fact(n) for n in range(P + 1))
    nodes = sc + 0.5 * (M - m) * np.cos((2 * np.arange(P + 1) + 1) * np.pi / (2 * P + 2))
    V = np.vander(nodes, P + 1, increasing=True)
    c = np.linalg.solve(V.astype(complex), np.exp(-1j * nodes))
    res["cheb"] = tgt - np.vander(s, P + 1, increasing=True) @ c
    A = np.vander(s, P + 1, increasing=True).astype(complex)
    c2, *_ = np.linalg.lstsq(A, tgt, rcond=None)
    res["ls"] = tgt - A @ c2
    return {k: (np.sqrt(np.mean(np.abs(v) ** 2)), np.abs(v).max()) for k, v in res.items()}


def rand_psd():
    B = rng.standard_normal((3, 3))
    return B @ B.T


cases = {
    "definite z^2": lambda: np.diag([0, 0, 1.0]),
    "definite x^2+y^2": lambda: np.diag([1.0, 1.0, 0]),
    "indefinite (PSD - PSD)": lambda: rand_psd() - rand_psd(),
}
print("| case | max|chi| | P | taylor rms/max | centred rms/max | cheb rms/max | ls rms/max |")
print("|---|---|---|---|---|---|---|")
for name, mk in cases.items():
    for a in [0.3, 0.7, 1.0, 1.4, 2.0]:
        for P in [1, 2]:
            acc = {k: [] for k in ["taylor", "centred", "cheb", "ls"]}
            ntr = 1 if name.startswith("definite") else 40
            for _ in range(ntr):
                Q = mk()
                s = chi_of(Q)
                s = s * a / np.abs(s).max()
                for k, v in errs(s, P).items():
                    acc[k].append(v)
            cells = []
            for k in ["taylor", "centred", "cheb", "ls"]:
                v = np.array(acc[k])
                cells.append(f"{np.median(v[:,0]):.1e} / {np.median(v[:,1]):.1e}")
            print(f"| {name} | {a} | {P} | " + " | ".join(cells) + " |")
