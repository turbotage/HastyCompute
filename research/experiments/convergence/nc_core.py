#!/usr/bin/env python3
"""Undersampled non-Cartesian convergence toy: core (research/convergence_strategy.md, section
"Undersampled non-Cartesian toy (2026-11-17)"). CPU, numpy/scipy only, self-contained.

Model (3-D, N^3 voxels, K time sets = cardiac bins, NC smooth complex coils):
  trajectory  3-D golden-means centre-out radial half-spokes (Chan et al. 2009 golden means), N/2+1 samples per spoke at
              unit k-spacing (density ~ 1/k^2, DC sampled by every spoke, like the centre-out yarn); spokes are binned
              into the K sets at random (cardiac binning of an interleaved golden sequence) with a 1.6x spread of the
              per-set spoke counts.  Per-set acceleration R = N^3 / (samples per set).
  operator    EXACT separable NUDFT (no gridding error): A_t x = { sum_r S_c(r) x_t(r) exp(-2 pi i k_m.r / N) }_{m,c}
              and its adjoint; the normal operator A_t^H A_t is applied by an exact Toeplitz embedding on the 2N grid
              (kernel t(d) = sum_m exp(2 pi i k_m.d / N) computed by a direct separable sum).
  data        generated from a 2x finer phantom (exact NUDFT on the fine grid, no inverse crime); the truth is the
              2x2x2 box average.  Complex noise sigma = sigma_rel * rms(clean samples).  Held-out spokes (extra 10 %,
              >= 2 per set) are never used in the data term.
  prior WT    lam * || W x ||_1, W = separable orthonormal Haar: 3-D spatial (3 levels) x temporal (3 levels on K = 8);
              only the (spatial coarse x temporal DC) band is unpenalised.  Exact prox = soft threshold.
  prior WL    lam_w * ||W_s x||_1 (spatial Haar per frame) + lam_l * LLR (6^3 x K Casorati blocks, fixed grid).
Cost unit: one gradient = one A^H A of the whole stack (or one A plus one A^H).  Priors / FFT-only metrics are counted
separately.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import scipy.fft as sfft

WK = int(os.environ.get("OMP_NUM_THREADS", "4"))
N, NC = 24, 4
K = int(os.environ.get("NC_K", "8"))
LEV_S, LEV_T = 3, int(round(np.log2(K)))
SQ2 = np.sqrt(2.0)


def fft3(a):
    return sfft.fftn(a, axes=(-3, -2, -1), workers=WK)


def ifft3(a):
    return sfft.ifftn(a, axes=(-3, -2, -1), workers=WK)


def fft3o(a):
    return sfft.fftn(a, axes=(-3, -2, -1), norm="ortho", workers=WK)


def ifft3o(a):
    return sfft.ifftn(a, axes=(-3, -2, -1), norm="ortho", workers=WK)


def vdot(a, b):
    return float(np.vdot(a, b).real)


def nrm(a):
    return float(np.sqrt(vdot(a, a)))


def rel(x, xs):
    return nrm(x - xs) / nrm(xs)


# ------------------------------------------------------------------ geometry
RC = np.arange(N) - N // 2                      # coarse voxel positions (integer)
RF = (np.arange(2 * N) - N) / 2.0 - 0.25        # fine voxel centres (2 per coarse voxel: r -/+ 0.25)
D2N = np.concatenate([np.arange(N), np.arange(-N, 0)])  # Toeplitz lags in FFT order on the 2N grid


def grid3(r):
    s = r / (N / 2.0)
    return np.meshgrid(s, s, s, indexing="ij")


def golden_means(n):
    phi1, phi2 = 0.4656, 0.6823
    i = np.arange(n)
    z = 2 * np.mod(i * phi1, 1.0) - 1
    az = 2 * np.pi * np.mod(i * phi2, 1.0)
    s = np.sqrt(np.maximum(1 - z * z, 0))
    return np.stack([s * np.cos(az), s * np.sin(az), z], 1)


def spokes_to_k(dirs):
    rr = np.arange(N // 2 + 1, dtype=float)
    return (dirs[:, None, :] * rr[None, :, None]).reshape(-1, 3)


def coils_fn(X, Y, Z, nc=NC):
    S = []
    for c in range(nc):
        a = 2 * np.pi * c / nc
        cx, cy, cz = 1.4 * np.cos(a), 1.4 * np.sin(a), 0.4 * (-1) ** c
        mag = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / 1.6)
        ph = np.exp(1j * (0.7 * X * np.cos(a) + 0.7 * Y * np.sin(a) + 0.4 * Z + 1.3 * c))
        S.append(mag * ph)
    return np.array(S)


def ell(X, Y, Z, c, r):
    return ((X - c[0]) / r[0]) ** 2 + ((Y - c[1]) / r[1]) ** 2 + ((Z - c[2]) / r[2]) ** 2 <= 1


def phantom_fn(X, Y, Z, kk=K):
    """static body + organs with smooth phase; dynamic: a beating 'ventricle' (moving wall), two pulsatile vessels
    with flow phase and radius pulsation (piecewise smooth with sharp edges)."""
    base = 0.55 * ell(X, Y, Z, (0, 0, 0), (0.88, 0.78, 0.92)).astype(complex)
    for c, r, v in [((-0.4, 0.25, 0.1), (0.22, 0.3, 0.4), 0.45), ((0.42, -0.32, -0.2), (0.18, 0.15, 0.35), -0.3),
                    ((0.2, 0.48, 0.42), (0.12, 0.12, 0.12), 0.35), ((-0.3, -0.45, -0.4), (0.3, 0.12, 0.2), 0.2)]:
        base += v * ell(X, Y, Z, c, r)
    base *= np.exp(1j * (0.5 * X + 0.3 * Y - 0.25 * Z))
    out = np.zeros((kk,) + X.shape, complex)
    for t in range(kk):
        ph = 2 * np.pi * t / kk
        heart = ell(X, Y, Z, (0.05, -0.05, 0.0), np.array((0.3, 0.27, 0.3)) * (1 - 0.13 * np.sin(ph)))
        img = np.where(heart, 0.85 * np.exp(1j * 0.3 * np.sin(ph)), base)
        r1 = 0.085 * (1 + 0.18 * np.sin(ph))
        v1 = (((X - 0.35) / r1) ** 2 + ((Y - 0.3) / r1) ** 2 <= 1) & (np.abs(Z) < 0.75)
        img = np.where(v1, (0.7 + 0.3 * np.sin(ph)) * np.exp(1j * (0.2 + 1.4 * np.sin(ph))), img)
        r2 = 0.07 * (1 + 0.15 * np.sin(ph - 1.0))
        v2 = (((X + 0.1) / r2) ** 2 + ((Z - 0.35) / r2) ** 2 <= 1) & (np.abs(Y) < 0.65)
        img = np.where(v2, (0.6 + 0.35 * np.sin(ph - 1.0)) * np.exp(1j * (-0.4 + 1.1 * np.sin(ph - 1.0))), img)
        out[t] = img
    return out


# ------------------------------------------------------------------ exact separable NUDFT
def phase_mats(k, r, sign=-1.0):
    return [np.exp(sign * 2j * np.pi * np.outer(k[:, d], r) / N) for d in range(3)]


def nudft(E, X, chunk=1500):
    """X (C, n, n, n) -> (M, C): sum_r X(r) Ex[m,x] Ey[m,y] Ez[m,z]"""
    Ex, Ey, Ez = E
    C, n = X.shape[0], X.shape[1]
    Xr = X.transpose(1, 0, 2, 3).reshape(n, C * n * n)
    out = np.empty((Ex.shape[0], C), complex)
    for s in range(0, Ex.shape[0], chunk):
        e = slice(s, s + chunk)
        U = (Ex[e] @ Xr).reshape(-1, C, n, n)
        V = np.einsum("mcyz,my->mcz", U, Ey[e], optimize=True)
        out[e] = np.einsum("mcz,mz->mc", V, Ez[e], optimize=True)
    return out


def nudft_adj(E, Y, chunk=1500):
    """Y (M, C) -> (C, n, n, n), adjoint of nudft"""
    Ex, Ey, Ez = E
    M, C = Y.shape
    n = Ex.shape[1]
    acc = np.zeros((n, C * n * n), complex)
    for s in range(0, M, chunk):
        e = slice(s, s + chunk)
        V = Y[e][:, :, None] * np.conj(Ez[e])[:, None, :]
        U = np.conj(Ey[e])[:, None, :, None] * V[:, :, None, :]
        acc += np.conj(Ex[e]).T @ U.reshape(U.shape[0], C * n * n)
    return acc.reshape(n, C, n, n).transpose(1, 0, 2, 3)


def toeplitz_kernel(k):
    """t(d) = sum_m exp(+2 pi i k_m.d/N) on the 2N grid (FFT order), by a direct separable sum"""
    G = phase_mats(k, D2N, sign=+1.0)
    out = np.zeros((2 * N, 2 * N, 2 * N), complex)
    for s in range(0, k.shape[0], 1500):
        e = slice(s, s + 1500)
        XY = (G[0][e][:, :, None] * G[1][e][:, None, :]).reshape(-1, 4 * N * N)
        out += (XY.T @ G[2][e]).reshape(2 * N, 2 * N, 2 * N)
    return out


def autocorr(u):
    """sum_r u(r+d) conj(u(r)) on the 2N grid (FFT order) for an N^3 array u"""
    p = np.zeros(u.shape[:-3] + (2 * N,) * 3, complex)
    p[..., :N, :N, :N] = u
    return ifft3(np.abs(fft3(p)) ** 2)


# ------------------------------------------------------------------ Haar wavelets (orthonormal)
def _h1(a, ax):
    ev = np.take(a, np.arange(0, a.shape[ax], 2), ax)
    od = np.take(a, np.arange(1, a.shape[ax], 2), ax)
    return np.concatenate([(ev + od) / SQ2, (ev - od) / SQ2], ax)


def _interleave(ev, od, ax):
    sh = list(ev.shape)
    sh[ax] *= 2
    out = np.empty(sh, ev.dtype)
    idx = [slice(None)] * ev.ndim
    idx[ax] = slice(0, None, 2)
    out[tuple(idx)] = ev
    idx[ax] = slice(1, None, 2)
    out[tuple(idx)] = od
    return out


def _ih1b(a, ax):
    h = a.shape[ax] // 2
    lo = np.take(a, np.arange(h), ax)
    hi = np.take(a, np.arange(h, 2 * h), ax)
    return _interleave((lo + hi) / SQ2, (lo - hi) / SQ2, ax)


def haar_s(x, lev=LEV_S):
    c = x.copy()
    n = N
    for _ in range(lev):
        sub = c[..., :n, :n, :n]
        for ax in (-3, -2, -1):
            sub = _h1(sub, ax)
        c[..., :n, :n, :n] = sub
        n //= 2
    return c


def ihaar_s(c, lev=LEV_S):
    x = c.copy()
    n = N >> (lev - 1)
    for _ in range(lev):
        sub = x[..., :n, :n, :n]
        for ax in (-1, -2, -3):
            sub = _ih1b(sub, ax)
        x[..., :n, :n, :n] = sub
        n *= 2
    return x


def haar_t(c, lev=LEV_T):
    c = c.copy()
    n = c.shape[0]
    for _ in range(lev):
        c[:n] = _h1(c[:n], 0)
        n //= 2
    return c


def ihaar_t(c, lev=LEV_T):
    c = c.copy()
    n = c.shape[0] >> (lev - 1)
    for _ in range(lev):
        c[:n] = _ih1b(c[:n], 0)
        n *= 2
    return c


def W_st(x):
    return haar_t(haar_s(x))


def WH_st(c):
    return ihaar_s(ihaar_t(c))


def _slev(n=N, lev=LEV_S):
    i = np.arange(n)
    li = np.zeros(n, int)
    for j in range(1, lev + 1):
        li[(i >= n >> j) & (i < n >> (j - 1))] = j
    return li


_LS1 = _slev()
SLEV = np.zeros((N, N, N), int)                  # spatial level of a coefficient (0 = coarse, 1 = finest)
_ix, _iy, _iz = np.meshgrid(_LS1, _LS1, _LS1, indexing="ij")
_m = np.minimum(np.where(_ix == 0, 99, _ix), np.minimum(np.where(_iy == 0, 99, _iy), np.where(_iz == 0, 99, _iz)))
SLEV = np.where(_m == 99, 0, _m)
TLEV = _slev(K, LEV_T)                           # temporal level per coefficient index
COARSE_S = SLEV == 0
COARSE_ST = (TLEV == 0)[:, None, None, None] & COARSE_S[None]
PEN_ST = ~COARSE_ST
PEN_S = np.broadcast_to(~COARSE_S[None], (K, N, N, N))

# frequency bands that the Haar levels roughly occupy (for the prior-aware metric)
_kf = np.abs(np.fft.fftfreq(N) * N)
_kx, _ky, _kz = np.meshgrid(_kf, _kf, _kf, indexing="ij")
_kinf = np.maximum(np.maximum(_kx, _ky), _kz)
KBAND = np.zeros((N, N, N), int)
for _j in range(1, LEV_S + 1):
    KBAND[(_kinf > N / 2 ** (_j + 1)) & (_kinf <= N / 2 ** _j)] = _j
_wf = np.abs(np.fft.fftfreq(K) * K)
WBAND = np.zeros(K, int)
for _j in range(1, LEV_T + 1):
    WBAND[(_wf > K / 2 ** (_j + 1)) & (_wf <= K / 2 ** _j)] = _j
KRAD = np.sqrt(_kx ** 2 + _ky ** 2 + _kz ** 2)


def soft(c, thr, pen):
    m = np.abs(c)
    return np.where(pen, c * np.maximum(1 - thr / np.maximum(m, 1e-300), 0), c)


# ------------------------------------------------------------------ LLR (fixed 6^3 blocks over all K frames)
BL = 6
NB = N // BL


def to_blocks(x):
    """(K, N, N, N) -> (NB^3, BL^3, K)"""
    b = x.reshape(K, NB, BL, NB, BL, NB, BL).transpose(1, 3, 5, 2, 4, 6, 0)
    return b.reshape(NB ** 3, BL ** 3, K)


def from_blocks(b):
    b = b.reshape(NB, NB, NB, BL, BL, BL, K).transpose(6, 0, 3, 1, 4, 2, 5)
    return b.reshape(K, N, N, N)


def svt(x, thr):
    U, s, Vh = np.linalg.svd(to_blocks(x), full_matrices=False)
    s2 = np.maximum(s - thr, 0)
    return from_blocks((U * s2[:, None, :]) @ Vh), s


def nuc(x):
    return float(np.linalg.svd(to_blocks(x), compute_uv=False).sum())


# ------------------------------------------------------------------ problem
class Problem:
    def __init__(self, R, spread=1.6, sigma_rel=0.01, seed=1, ho_frac=0.1):
        rng = np.random.default_rng(seed)
        self.R = R
        spp = N // 2 + 1
        nsp = N ** 3 / R / spp
        w = np.geomspace(1 / np.sqrt(spread), np.sqrt(spread), K)
        rng.shuffle(w)
        cnt = np.maximum(1, np.round(nsp * w / w.mean())).astype(int)
        ho = np.maximum(2, np.round(ho_frac * cnt)).astype(int)
        tot = int((cnt + ho).sum())
        dirs = golden_means(tot)
        perm = rng.permutation(tot)
        Xc = grid3(RC)
        Xf = grid3(RF)
        self.S = coils_fn(*Xc)
        Sf = coils_fn(*Xf)
        sc = np.sqrt((np.abs(self.S) ** 2).sum(0)).max()
        self.S /= sc
        Sf /= sc
        xf = phantom_fn(*Xf)
        self.xt = xf.reshape(K, N, 2, N, 2, N, 2).mean((2, 4, 6))
        self.cnt, self.ho = cnt, ho
        self.E, self.Eh, self.That, self.ong, self.lam_t = [], [], [], [], []
        clean, cleanh = [], []
        pos = 0
        for t in range(K):
            ids = perm[pos:pos + cnt[t] + ho[t]]
            pos += cnt[t] + ho[t]
            kd = spokes_to_k(dirs[ids[:cnt[t]]])
            kh = spokes_to_k(dirs[ids[cnt[t]:]])
            self.E.append(phase_mats(kd, RC))
            self.Eh.append(phase_mats(kh, RC))
            clean.append(nudft(phase_mats(kd, RF), Sf * xf[t][None]) / 8.0)
            cleanh.append(nudft(phase_mats(kh, RF), Sf * xf[t][None]) / 8.0)
            tk = toeplitz_kernel(kd)
            self.That.append(fft3(tk))
            self._tk = tk
            self.lam_t.append(self._circ_diag(tk))
            self.ong.append(self._ong(kd, tk))
        self.That = np.array(self.That)
        self.lam_t = np.array(self.lam_t)
        self.M = np.array([e[0].shape[0] for e in self.E])
        allc = np.concatenate([c.ravel() for c in clean])
        self.sigma = sigma_rel * np.sqrt(np.mean(np.abs(allc) ** 2))
        cn = lambda s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)) * self.sigma / SQ2
        self.y = [c + cn(c.shape) for c in clean]
        self.yh = [c + cn(c.shape) for c in cleanh]
        self.ynorm2 = sum(vdot(v, v) for v in self.y)
        self.nQ = self.nA = self.nAH = 0
        self.nprior = 0
        self.eps = 0.0
        self.b = self.AH(self.y, count=False)
        # circulant metric pieces
        self.lam_bar = self.lam_t.mean(0)
        self.n_t = self.lam_t.sum((1, 2, 3)) / self.lam_bar.sum()
        inside = KRAD <= N / 2
        self.bulk = float(np.median(self.lam_bar[inside]))
        self.L = self.power(lambda v: self.Q(v, count=False), 50)

    # ---------- spectra
    def _circ_diag(self, tk):
        """exact diag of F H_t F^H (H_t = sum_c S_c^H T_t S_c, F the unitary N-point DFT), floored at 1e-3 max"""
        a = np.conj(autocorr(self.S)).sum(0) / N ** 3
        B = (tk * a).reshape(2, N, 2, N, 2, N).sum((0, 2, 4))
        lam = np.real(sfft.fftn(B, workers=WK))
        return np.maximum(lam, 1e-3 * lam.max())

    def _ong(self, kd, tk):
        """Ong-Uecker-Lustig l2-optimised multichannel k-space preconditioner p_{m,c} = (AA^H)_ii / sum_j |(AA^H)_ij|^2"""
        G = phase_mats(kd, D2N)
        p = np.empty((kd.shape[0], NC))
        for c in range(NC):
            g = self.S[c][None] * np.conj(self.S)
            bc = autocorr(g).sum(0)
            den = nudft(G, (tk * bc)[None])[:, 0].real
            p[:, c] = (np.abs(self.S[c]) ** 2).sum() / np.maximum(den, 1e-300)
        return p

    # ---------- operators
    def A(self, x, count=True, E=None):
        if count:
            self.nA += 1
        E = self.E if E is None else E
        return [nudft(E[t], self.S * x[t][None]) for t in range(K)]

    def AH(self, r, count=True):
        if count:
            self.nAH += 1
        return np.array([(np.conj(self.S) * nudft_adj(self.E[t], r[t])).sum(0) for t in range(K)])

    def Q(self, x, count=True):
        if count:
            self.nQ += 1
        u = np.zeros((K, NC) + (2 * N,) * 3, complex)
        u[..., :N, :N, :N] = self.S[None] * x[:, None]
        u = ifft3(fft3(u) * self.That[:, None])
        out = (np.conj(self.S)[None] * u[..., :N, :N, :N]).sum(1)
        return out + self.eps * x if self.eps else out

    def cost(self):
        return self.nQ + 0.5 * (self.nA + self.nAH)

    def reset(self):
        self.nQ = self.nA = self.nAH = self.nprior = 0

    def f(self, x, Qx=None):
        Qx = self.Q(x, count=False) if Qx is None else Qx
        return 0.5 * vdot(x, Qx) - vdot(self.b, x) + 0.5 * self.ynorm2

    def heldout(self, x):
        """held-out k-space residual / expected noise energy (1 = noise floor)"""
        r = self.A(x, count=False, E=self.Eh)
        num = sum(vdot(a - b, a - b) for a, b in zip(r, self.yh))
        return num / (sum(v.size for v in self.yh) * self.sigma ** 2)

    def power(self, op, it, seed=3):
        v = np.random.default_rng(seed).standard_normal((K, N, N, N)) + 0j
        lam = 0.0
        for _ in range(it):
            w = op(v)
            lam = nrm(w) / nrm(v)
            v = w / nrm(w)
        return lam

    # ---------- metrics (FFT-only, applied to (K, N, N, N) stacks)
    def metric(self, name, psi=None, rho=0.0):
        """name in I, C (shared circulant), Cn (shared + per-set scalars), Cnp (Cn + prior term psi(omega,k) + rho:
        joint (omega, k) diagonal with the N^{-1/2} sandwich); returns (P, P^{-1})"""
        lb, nt = self.lam_bar, self.n_t[:, None, None, None]
        if name == "I":
            return (lambda g: g.copy()), (lambda g: g.copy())
        if name == "C":
            d = lb[None] + rho
            return (lambda g: ifft3o(fft3o(g) / d)), (lambda g: ifft3o(fft3o(g) * d))
        if name == "Cn":
            d = nt * lb[None] + rho
            return (lambda g: ifft3o(fft3o(g) / d)), (lambda g: ifft3o(fft3o(g) * d))
        if name == "Cnp":
            sq = np.sqrt(nt)
            den = lb[None] + (0.0 if psi is None else psi) + rho
            def P(g):
                h = np.fft.fft(fft3o(g / sq), axis=0, norm="ortho") / den
                return ifft3o(np.fft.ifft(h, axis=0, norm="ortho")) / sq
            def Pi(g):
                h = np.fft.fft(fft3o(g * sq), axis=0, norm="ortho") * den
                return ifft3o(np.fft.ifft(h, axis=0, norm="ortho")) * sq
            return P, Pi
        raise ValueError(name)


def self_test(R=25):
    import time
    t0 = time.time()
    P = Problem(R)
    print(f"R={R}: M per set {P.M.tolist()}, setup {time.time() - t0:.1f}s, L={P.L:.3g}, bulk={P.bulk:.3g},"
          f" lam range {P.lam_bar.max() / P.lam_bar.min():.3g}, n_t {np.round(P.n_t, 2).tolist()}")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((K, N, N, N)) + 1j * rng.standard_normal((K, N, N, N))
    t0 = time.time(); q1 = P.Q(x, count=False); tq = time.time() - t0
    t0 = time.time(); q2 = P.AH(P.A(x, count=False), count=False); ta = time.time() - t0
    print(f"Toeplitz vs exact A^H A: rel {rel(q1, q2):.2e}; t(Q)={tq:.3f}s, t(A+A^H)={ta:.3f}s")
    # adjoint test
    r = [rng.standard_normal(v.shape) + 1j * rng.standard_normal(v.shape) for v in P.y]
    lhs = sum(np.vdot(a, b) for a, b in zip(P.A(x, count=False), r))
    rhs = np.vdot(x, P.AH(r, count=False))
    print(f"adjoint test {abs(lhs - rhs) / abs(lhs):.2e}")
    # exact NUDFT vs brute force on a few samples
    E = P.E[0]
    kd = None
    m = 5
    Xc = grid3(RC)
    brute = []
    for i in range(m):
        ph = E[0][i][:, None, None] * E[1][i][None, :, None] * E[2][i][None, None, :]
        brute.append((P.S[0] * x[0] * ph).sum())
    print(f"separable NUDFT vs brute {np.abs(np.array(brute) - P.A(x, count=False)[0][:m, 0]).max() / np.abs(brute).max():.2e}")
    # circulant diag probe
    t = 0
    for kv in [(0, 0, 0), (3, 1, 0), (7, 5, 2), (11, 11, 11)]:
        v = np.zeros((K, N, N, N), complex)
        e = np.zeros((N, N, N), complex); e[kv] = 1
        v[t] = ifft3o(e)
        qv = P.Q(v, count=False)
        print(f"circ diag k={kv}: probe {vdot(v, qv):.6g} formula {P.lam_t[t][kv]:.6g}")
    # wavelet orthonormality
    c = W_st(x)
    print(f"Haar ST: |Wx|/|x| {nrm(c) / nrm(x):.15f}, inv err {rel(WH_st(c), x):.2e}; spatial inv err"
          f" {rel(ihaar_s(haar_s(x)), x):.2e}; LLR roundtrip {rel(from_blocks(to_blocks(x)), x):.2e}")
    # Ong precond: diag check on one row
    print(f"Ong p range set0: {P.ong[0].min():.3g}..{P.ong[0].max():.3g}")
    print(f"truth |x| {nrm(P.xt):.3g}, sigma {P.sigma:.3g}, heldout(truth) {P.heldout(P.xt):.3f}")
    return P


if __name__ == "__main__":
    import sys
    self_test(float(sys.argv[1]) if len(sys.argv) > 1 else 25)
