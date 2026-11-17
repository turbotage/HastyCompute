"""Random search over integer encoding designs z_e in Z^3 (|z|_inf <= ZMAX, z_0 = 0), K = 5, 6, 7 (2026-11-17).

For an integer design the phase is kappa0 z_e . v. Lattice facts used (enc_lib docstring for the model):
- true aliases: w in (2 pi / kappa0) M*, M = lattice of the differences z_e - z_0, so the protected ball radius is
  R = (pi / kappa0) lambda_1(M*); we scale kappa0 so that R = V = 1 (the whole CRT range used, no slack);
- redundancy margin (phase distance to the nearest wrong sheet): m_min = 2 pi lambda_1(P_U Z^K), U = ker [1 | z]^T,
  computed here by enumeration of small n (verified for the finalists with enc_lib.Geometry);
- per-encode SNR needed for P_int = N_min Q(m_min / (2 s)) = 1e-3, s = 1/(sqrt 2 SNR_e) (Gaussian phase noise);
- time-normalised noise eta = sigma_v,rms * SNR_tot / V with SNR_e = SNR_tot / sqrt K.
Output: Pareto front (snr_tot_req, eta) per K, as JSON.

    nice -n 10 rustlib/.venv/bin/python research/experiments/encoding/enc_search.py OUT.json [n_per_K]
"""
import itertools
import json
import math
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc

ZMAX = 3
M1CAP = 1e9


def Q(x):
    return 0.5 * erfc(x / math.sqrt(2.0))


def gcd_minors(D):
    g = 0
    for idx in itertools.combinations(range(D.shape[0]), 3):
        g = math.gcd(g, int(round(abs(np.linalg.det(D[list(idx)])))))
        if g == 1:
            return 1
    return g


def lam1_dual(D):
    """lambda_1 of M* = {w : D w in Z^{K-1}} (M* contains Z^3, so lambda_1 <= 1)"""
    Dm = gcd_minors(D)
    if Dm == 0:
        return None
    if Dm == 1:
        return 1.0
    r = np.arange(-Dm, Dm + 1)
    W = np.array(list(itertools.product(r, r, r)), float) / Dm
    W = W[np.linalg.norm(W, axis=1) > 1e-12]
    X = W @ D.T
    ok = np.all(np.abs(X - np.round(X)) < 1e-9, axis=1)
    return float(np.linalg.norm(W[ok], axis=1).min())


def lll(B, delta=0.75):
    """LLL reduction of the rows of B (float; small integer lattices)"""
    B = np.array(B, float)
    n = len(B)

    def gso(B):
        Bs = np.zeros_like(B)
        mu = np.zeros((n, n))
        for i in range(n):
            Bs[i] = B[i]
            for j in range(i):
                d = Bs[j] @ Bs[j]
                mu[i, j] = (B[i] @ Bs[j]) / d if d > 1e-12 else 0.0
                Bs[i] -= mu[i, j] * Bs[j]
        return Bs, mu

    Bs, mu = gso(B)
    k = 1
    it = 0
    while k < n and it < 10000:
        it += 1
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q:
                B[k] -= q * B[j]
                Bs, mu = gso(B)
        if Bs[k] @ Bs[k] >= (delta - mu[k, k - 1] ** 2) * (Bs[k - 1] @ Bs[k - 1]):
            k += 1
        else:
            B[[k, k - 1]] = B[[k - 1, k]]
            Bs, mu = gso(B)
            k = max(k - 1, 1)
    return B


def margin_exact(z):
    """2 pi lambda_1(P_U Z^K) and its kissing count, via the integer kernel C of [1|z]^T (LLL on [N A | I]) and
    P_U Z^K = (Z^K cap U)^* (U = ker A^T)"""
    z = np.asarray(z, float)
    K = len(z)
    A = np.hstack([np.ones((K, 1)), z])
    Nbig = 1e4
    B = lll(np.hstack([Nbig * A, np.eye(K)]))
    C = np.array([b[4:] for b in B if np.all(np.abs(b[:4]) < 0.5)])
    r = K - np.linalg.matrix_rank(A)
    if len(C) != r or r == 0:
        return None
    Cd = np.linalg.solve(C @ C.T, C)  # dual basis rows (in U)
    Cd = lll(Cd)
    rng_ = np.arange(-3, 4)
    T = np.array(list(itertools.product(*([rng_] * r))), float)
    T = T[np.any(T != 0, axis=1)]
    L = np.linalg.norm(T @ Cd, axis=1)
    l1 = L.min()
    return 2 * math.pi * l1, int(np.sum(L < l1 * 1.01))


NBOX = {}


def nbox(K, b):
    key = (K, b)
    if key not in NBOX:
        r = np.arange(-b, b + 1)
        n = np.array(list(itertools.product(*([r] * (K - 1)))), float)
        NBOX[key] = np.hstack([np.zeros((len(n), 1)), n])
    return NBOX[key]


def evaluate(z, b):
    z = np.asarray(z, float)
    K = len(z)
    D = z[1:] - z[0]
    if np.linalg.matrix_rank(D) < 3:
        return None
    l1 = lam1_dual(D)
    if l1 is None:
        return None
    kappa0 = math.pi * l1  # R = 1
    k = kappa0 * z
    kc = k - k.mean(0)
    F = kc.T @ kc
    trFi = float(np.trace(np.linalg.inv(F)))
    A = np.hstack([np.ones((K, 1)), z])
    P = np.eye(K) - A @ np.linalg.pinv(A)
    me = margin_exact(z)
    if me is None:
        return None  # no redundancy (K = 4 type)
    mmin, nmin = me
    # SNR_e for nmin Q(mmin/(2s)) = 1e-3, s = 1/(sqrt2 snr)
    f = lambda snr: nmin * Q(mmin * math.sqrt(2) * snr / 2) - 1e-3
    snr_e = brentq(f, 0.05, 1e3)
    eta = math.sqrt(trFi / 3 / 2) * math.sqrt(K)  # sigma_v * SNR_tot / V
    m1 = kappa0 * float(np.max(0.5 * (z.max(0) - z.min(0)))) / (1.5 * math.pi)  # vs coprime7 p=2
    return dict(m1_rel=m1, z=z.astype(int).tolist(), K=K, mmin=mmin, nmin=nmin, snr_e_req=snr_e, snr_tot_req=snr_e * math.sqrt(K), eta=eta, lam1=l1)


def worker(args):
    K, n, seed = args
    rng = np.random.default_rng(seed)
    b = {5: 4, 6: 3, 7: 2}.get(K, 2)  # unused since margin_exact
    out = []
    for _ in range(n):
        z = np.vstack([np.zeros(3, int), rng.integers(-ZMAX, ZMAX + 1, size=(K - 1, 3))])
        if len({tuple(r) for r in z}) < K:
            continue
        r = evaluate(z, b)
        if r is not None:
            out.append(r)
    return out


def pareto(rows):
    rows = [r for r in rows if r["m1_rel"] <= M1CAP]
    rows = sorted(rows, key=lambda r: (r["snr_tot_req"], r["eta"]))
    front, best = [], math.inf
    for r in rows:
        if r["eta"] < best - 1e-9:
            front.append(r)
            best = r["eta"]
    return front


if __name__ == "__main__":
    out = sys.argv[1]
    nper = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    Ks = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [5, 6, 7]
    res = {}
    with Pool(8) as pool:
        allrows = {}
        for K in Ks:
            n = nper
            parts = pool.map(worker, [(K, n // 8, 1000 * K + i) for i in range(8)])
            rows = [r for p in parts for r in p]
            allrows[K] = rows
            fr = pareto(rows)
            res[K] = fr
            print(f"K={K}: {len(rows)} designs, Pareto {len(fr)}", flush=True)
            for r in fr:
                print(f"  snr_tot_req {r['snr_tot_req']:6.2f}  eta {r['eta']:.4f}  mmin {r['mmin']:.3f} x{r['nmin']}  z {r['z']}", flush=True)
    # constrained fronts: encode first moment no larger than coprime7 p=2 (same encode block), and <= 1.3x
    for cap in (1.0, 1.3):
        M1CAP = cap
        res[f"cap{cap}"] = {K: pareto(allrows[K]) for K in allrows}
    json.dump(res, open(out, "w"), indent=1)
