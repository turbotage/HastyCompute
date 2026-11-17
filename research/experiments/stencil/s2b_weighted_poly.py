"""Energy-weighted least-squares chirp polynomial (per sample, same cost as Taylor) vs Taylor / Chebyshev, 3-D.
Weight w(rho) = |x|^2 of a coarse (8 mm-like) magnitude image: what a real scan has (no labels).  sigma 1.25, W 8."""
import numpy as np
import s2_3d_check as S
from kern import Kernel

# coarse energy weights on the probe grid (9^3): block-average |x|^2 of each test image
def probe_weights(x):
    N = x.shape[0]
    idx = np.clip(np.round((S.probe + 1) / 2 * (N - 1)).astype(int), 0, N - 1)
    e = np.abs(x) ** 2
    from scipy.ndimage import uniform_filter
    e = uniform_filter(e, 5)
    return e[idx[:, 0], idx[:, 1], idx[:, 2]] + 1e-3 * e.mean()

def wls_coefs(Q, wt):
    c = np.zeros((S.M, 3), complex)
    sw = np.sqrt(wt)
    for j in range(S.M):
        s = np.einsum("ni,ij,nj->n", S.probe, Q[j], S.probe)
        A = np.vander(s, 3, increasing=True).astype(complex) * sw[:, None]
        c[j], *_ = np.linalg.lstsq(A, np.exp(-1j * s) * sw, rcond=None)
    return c

print("| chirp | max chi | poly | white | object |")
print("|---|---|---|---|---|")
ker = Kernel("es", 8, 1.25)
for kind, a in [("indefinite", 0.7), ("definite", 0.7), ("definite", 1.0), ("indefinite", 1.4)]:
    Q = S.make_Q(a, kind)
    res = {}
    for nm, xx in [("white", S.x_white), ("object", S.x_obj)]:
        yd = S.direct(xx, Q)
        for p in ["taylor", "cheb", "wls"]:
            c = wls_coefs(Q, probe_weights(xx)) if p == "wls" else S.poly_coefs(Q, p)
            y = S.gridded(xx, Q, ker, 1.25, c)
            res[(p, nm)] = np.linalg.norm(y - yd) / np.linalg.norm(yd)
    for p in ["taylor", "cheb", "wls"]:
        print(f"| {kind} | {a} | {p} | {res[(p,'white')]:.1e} | {res[(p,'object')]:.1e} |", flush=True)
