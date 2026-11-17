"""Velocity-encoding designs: Fisher information, wrap (alias) geometry, failure probability (2026-11-17).

Model (one voxel, white phase noise, equal magnitude in every encode):
    phi_e = theta + k_e . v + eps_e  (mod 2 pi),   e = 0..K-1,  eps_e ~ N(0, s^2),  s = 1 / (sqrt(2) SNR_e)
A = [1 | k] (K x 4) acts on x = (theta, v). A wrap hypothesis n in Z^K (n_0 = 0 wlog: n + 1 is a global phase)
corresponds to the flat A x + 2 pi n; its closest point to the true flat is x_n = A^+ 2 pi n with
    m_n = || (I - A A^+) 2 pi n ||   (the part no (theta, v) can explain: the redundancy margin)
    w_n = velocity part of x_n       (the velocity error the wrap would cause)
With a prior |v| <= V (speed ball), the distance from the true point p(v) to the *restricted* sheet n is
    d_n(v)^2 = m_n^2 + min_{|v + u| <= V} (u - w_n)^T F (u - w_n),   F = k^T (I - 11^T / K) k
(F: Fisher matrix of v per unit phase variance, theta profiled). Union bound of ML decoding failure:
    P_fail(v) ~ sum_n Q(d_n(v) / (2 s)),  averaged over v uniform in the ball.
Velocity noise (unwrapping right): Cov(v) = s^2 F^-1.
Fixed scan time: SNR_e = SNR_tot / sqrt(K) (SNR_tot: one image from all readouts).
Velocities in units of V (protected speed): designs are shapes k_hat scaled by kappa.
"""
import itertools

import numpy as np
from scipy.special import erfc


def Q(x):
    return 0.5 * erfc(x / np.sqrt(2.0))


def fisher(k):
    k = np.asarray(k, float)
    kc = k - k.mean(0)
    return kc.T @ kc  # = K Cov(k)


def sigma_v_rms(k, snr_e):
    """rms over components of the velocity std (units of v) at per-encode SNR snr_e"""
    s2 = 1.0 / (2.0 * snr_e ** 2)
    return np.sqrt(s2 * np.trace(np.linalg.inv(fisher(k))) / 3.0)


def wraps(k, wmax, chunk=2_000_000):
    """all wrap hypotheses n (n_0 = 0) whose velocity offset |w_n| <= wmax: returns (m, w) arrays"""
    k = np.asarray(k, float)
    K = k.shape[0]
    A = np.hstack([np.ones((K, 1)), k])
    Ap = np.linalg.pinv(A)
    P = np.eye(K) - A @ Ap
    d = np.linalg.norm(k[1:] - k[0], axis=1)
    B = np.ceil(d * wmax / (2 * np.pi)).astype(int) + 1
    ranges = [np.arange(-b, b + 1) for b in B]
    ms, ws = [], []
    # enumerate in chunks over the leading dims
    sizes = [len(r) for r in ranges]
    total = int(np.prod(sizes))
    # split: outer product over first j dims, inner meshgrid over rest
    j = 0
    while j < len(sizes) and int(np.prod(sizes[j:])) > chunk:
        j += 1
    inner = np.array(list(itertools.product(*ranges[j:])), dtype=np.int32) if j < len(sizes) else np.zeros((1, 0), np.int32)
    for outer in itertools.product(*ranges[:j]):
        n = np.zeros((inner.shape[0], K))
        n[:, 1:1 + j] = outer
        n[:, 1 + j:] = inner
        n *= 2 * np.pi
        x = n @ Ap.T
        w = x[:, 1:]
        r = n @ P.T
        m = np.linalg.norm(r, axis=1)
        wn = np.linalg.norm(w, axis=1)
        keep = (wn <= wmax) & ((wn > 1e-9) | (m > 1e-9))
        ms.append(m[keep])
        ws.append(w[keep])
    del total
    return np.concatenate(ms), np.concatenate(ws)


def ball_dist2(F, d, V):
    """min_{|y| <= V} (y - d)^T F (y - d) for rows d (n x 3); trust-region secular equation, vectorised"""
    lam_, Qm = np.linalg.eigh(F)
    dp = d @ Qm
    inside = np.linalg.norm(d, axis=1) <= V
    lo = np.zeros(len(d))
    hi = np.full(len(d), lam_.max() * 1e6 + 1.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        y = lam_ * dp / (lam_ + mid[:, None])
        big = (y ** 2).sum(1) > V ** 2
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    lmb = hi
    val = (lam_ * (lmb[:, None] * dp / (lam_ + lmb[:, None])) ** 2).sum(1)
    return np.where(inside, 0.0, val)


def ball_samples(nv, rng):
    u = rng.normal(size=(nv, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u * rng.uniform(size=(nv, 1)) ** (1 / 3)


class Geometry:
    """wrap geometry of design k for the speed ball |v| <= V (V = 1).

    Two velocity populations: 'int' = uniform in |v| <= V/2 (the bulk of voxels: wrong-wrap errors there come only from
    the redundancy margin m_n), 'edge' = |v| = V (peak voxels: also the exit from the unaliased cell).
    The design requirement used here: P_int <= 1e-3 and mean P_edge <= 1e-2.
    """

    def __init__(self, k, V=1.0, nv=48, seed=0, wslack=0.6, m_cut=4.5):
        self.k = np.asarray(k, float)
        self.K = self.k.shape[0]
        self.F = fisher(self.k)
        m, w = wraps(self.k, 2 * V + wslack)
        wn = np.linalg.norm(w, axis=1)
        self.true_alias = bool(np.any((m < 1e-7) & (wn < 2 * V - 1e-9)))
        inner = (wn <= 2 * V) & (m > 1e-7)
        self.m_min = float(m[inner].min()) if np.any(inner) else np.inf  # redundancy margin inside the range
        sel = m < m_cut
        m, w = m[sel], w[sel]
        self.n_sheets = int(sel.sum())
        rng = np.random.default_rng(seed)
        u = rng.normal(size=(nv, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        self.D = {}
        for name, vs in (("int", ball_samples(nv, rng) * 0.5 * V), ("edge", u * V)):
            D = []
            for v in vs:
                d2 = m ** 2 + ball_dist2(self.F, w + v, V)
                d2.sort()
                d2 = d2[:400]
                D.append(np.sqrt(np.pad(d2, (0, 400 - len(d2)), constant_values=1e6)))
            self.D[name] = np.array(D)

    def p_fail(self, snr_e, which="int"):
        s = 1.0 / (np.sqrt(2.0) * snr_e)
        return float(np.mean(np.minimum(1.0, Q(self.D[which] / (2 * s)).sum(1))))

    def ok(self, snr_e, t_int=1e-3, t_edge=1e-2):
        return (not self.true_alias) and self.p_fail(snr_e, "int") <= t_int and self.p_fail(snr_e, "edge") <= t_edge

    def snr_req(self, t_int=1e-3, t_edge=1e-2):
        """smallest per-encode SNR meeting both targets"""
        if self.true_alias or not self.ok(1e4, t_int, t_edge):
            return np.inf
        lo, hi = 0.3, 1e4
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            if self.ok(mid, t_int, t_edge):
                hi = mid
            else:
                lo = mid
        return hi


def max_m1(k):
    """largest per-axis first-moment excursion after per-axis centring (balanced layout), in units of k"""
    k = np.asarray(k, float)
    c = 0.5 * (k.max(0) + k.min(0))
    return float(np.abs(k - c).max())


def tetra():
    return np.array([[-1, -1, -1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float) / 2.0


def icosa6():
    """six icosahedral axes (one per antipodal pair), unit vectors: (0, ±1, ±psi) cyclic"""
    p = (1 + 5 ** 0.5) / 2
    v = []
    for s in (1, -1):
        v += [[0, 1, s * p], [1, s * p, 0], [s * p, 0, 1]]
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def designs():
    """shapes k_hat (rows = encodes); scale is searched separately"""
    I = np.eye(3)
    d = {}
    d["ref4"] = np.vstack([np.zeros(3), I])
    d["had4"] = tetra()
    d["jm5 (FC + tetra)"] = np.vstack([np.zeros(3), tetra()])
    d["pelc6 (+-axes)"] = np.vstack([I, -I]) / 2
    d["dual7 1:2"] = np.vstack([np.zeros(3)] + [np.vstack([2 * I[a], I[a]]) for a in range(3)])
    for p in (2, 3, 4, 5):
        d[f"coprime7 p={p}"] = np.vstack([np.zeros(3)] + [np.vstack([(p + 1) * I[a], p * I[a]]) for a in range(3)])
    for p in (2, 3):
        d[f"coprime8 p={p}"] = np.vstack([np.zeros(3), (p + 1) * I, np.zeros(3), p * I])
        d[f"coprime8H p={p}"] = np.vstack([(p + 1) * tetra(), p * tetra()])
    d["icosa6 (no ref)"] = icosa6()
    d["icosa7 (ref + 6)"] = np.vstack([np.zeros(3), icosa6()])
    return d


class Decoder:
    """maximum-likelihood (Gaussian phase noise) velocity decoder over all wrap hypotheses with |v_hat| <= vmax"""

    def __init__(self, k, vmax=1.05, chunk=2_000_000):
        self.k = np.asarray(k, float)
        K = self.k.shape[0]
        self.A = np.hstack([np.ones((K, 1)), self.k])
        self.Ap = np.linalg.pinv(self.A)
        self.P = np.eye(K) - self.A @ self.Ap
        # |v_hat| <= vmax with phi in the cube (-pi, pi]^K: the wrap part w_n must satisfy |w_n| <= vmax + c
        c = np.pi * np.abs(self.Ap[1:]).sum(1)
        c = float(np.linalg.norm(c))
        d = np.linalg.norm(self.k[1:] - self.k[0], axis=1)
        B = np.ceil(d * (vmax + c) / (2 * np.pi)).astype(int) + 1
        ranges = [np.arange(-b, b + 1) for b in B]
        keep = []
        sizes = [len(r) for r in ranges]
        j = 0
        while j < len(sizes) and int(np.prod(sizes[j:])) > chunk:
            j += 1
        inner = np.array(list(itertools.product(*ranges[j:])), dtype=np.int32)
        for outer in itertools.product(*ranges[:j]):
            n = np.zeros((inner.shape[0], K))
            n[:, 1:1 + j] = outer
            n[:, 1 + j:] = inner
            w = (2 * np.pi * n) @ self.Ap[1:].T
            sel = np.linalg.norm(w, axis=1) <= vmax + c
            keep.append(n[sel])
        self.n = np.concatenate(keep)
        self.vmax = vmax
        self.Pn = (2 * np.pi * self.n) @ self.P.T  # (Nc, K)
        self.Wn = (2 * np.pi * self.n) @ self.Ap[1:].T  # (Nc, 3)

        # group hypotheses by their residual class P n (rational designs: few classes, many alias copies each)
        key = np.round(self.Pn / 1e-6).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind="stable")
        bounds = np.searchsorted(inv[order], np.arange(inv.max() + 2))
        self.cls_P = np.array([self.Pn[order[bounds[c]]] for c in range(len(bounds) - 1)])
        self.cls_W = [self.Wn[order[bounds[c]:bounds[c + 1]]] for c in range(len(bounds) - 1)]

    def decode(self, phi, slack=0.15):
        """phi (N, K) wrapped phases -> v (N, 3): best residual class that has a hypothesis with
        |v_hat| <= vmax + slack (the slack keeps the true class eligible when noise pushes a peak voxel past vmax);
        within a class the smallest-|v| alias copy"""
        phi = np.asarray(phi, float)
        r = phi @ self.P.T
        v0 = phi @ self.Ap[1:].T
        out = np.full((len(phi), 3), np.nan)
        for s in range(0, len(phi), 512):
            res = ((r[s:s + 512, None, :] + self.cls_P[None]) ** 2).sum(-1)
            srt = np.argsort(res, axis=1)
            for i in range(len(res)):
                for c in srt[i, :64]:
                    v = v0[s + i] + self.cls_W[c]
                    nv = np.linalg.norm(v, axis=1)
                    j = int(np.argmin(nv))
                    if nv[j] <= self.vmax + slack:
                        out[s + i] = v[j]
                        break
        return out, None


class GridDecoder:
    """approximate ML decoder for any design: maximise the profiled likelihood |sum_e exp(i (phi_e - k_e . v))| over
    a grid in the ball |v| <= vmax (step 0.2 pi / max|k_e - k_mean|, finer than every basin), then unwrap against the
    winner and solve LS twice. Memory: block x grid x K complex (block chosen for <= ~100 MB)."""

    def __init__(self, k, vmax=1.15, step_frac=0.2):
        self.k = np.asarray(k, float)
        K = len(self.k)
        h = step_frac * np.pi / np.abs(self.k - self.k.mean(0)).max()
        g = np.arange(-vmax, vmax + 1e-12, h)
        G = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
        self.G = G[np.linalg.norm(G, axis=1) <= vmax]
        self.E = np.exp(-1j * (self.G @ self.k.T))  # (Ng, K)
        A = np.hstack([np.ones((K, 1)), self.k])
        self.Ap = np.linalg.pinv(A)
        self.block = max(1, int(6e6 / (len(self.G) * K)))

    def decode(self, phi):
        phi = np.asarray(phi, float)
        z = np.exp(1j * phi)
        out = np.zeros((len(phi), 3))
        for s in range(0, len(phi), self.block):
            zz = z[s:s + self.block]
            p = np.abs(zz @ self.E.T)  # (b, Ng): |sum_e z_e e^{-i k_e v}|
            v = self.G[p.argmax(1)]
            for _ in range(2):
                th = np.angle((zz * np.exp(-1j * (v @ self.k.T))).sum(1))
                pred = th[:, None] + v @ self.k.T
                phu = pred + np.angle(np.exp(1j * (phi[s:s + self.block] - pred)))
                v = phu @ self.Ap[1:].T
            out[s:s + self.block] = v
        return out, None
