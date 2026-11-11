#!/usr/bin/env python3
"""Toy experiments for the paper draft "Restricted contraction for preconditioned plug-and-play".

Model problem (CPU, minutes):
  * real 64x64 modified Shepp-Logan phantom x0;
  * 4 smooth complex receive coils; a "shot" is one k-space location seen by all coils;
  * shots drawn i.i.d. from a variable density nu(k) ~ 1/(1+|k|^2/k0^2), no density compensation;
  * denoiser D = orthonormal Haar soft-thresholding (coarse band untouched). Its secant
    D(z) - D(x0) = S (z - x0) is exactly a tangent-supported nonexpansive map (eta = 0), so every
    hypothesis of the deterministic theorems can be computed, not assumed.

Everything is normalised so that ||H|| = 1 (alpha = 1).

Experiments (each cached in results/<name>.npz; delete a file to recompute it):
  lemmaA   random check of the ZMSHR1 spectrum (real and complex)
  ista     PnP-ISTA: Hong et al. q >= 1, restricted constants delta_k, a-posteriori bound, Ostrowski rate
  zm       P2nP-D (ZMSHR1, paper parameters): spectrum of P_k against the paper's, the delta-free and the
           restricted bounds; safeguard margin
  theta    theta_1 sweep of ZMSHR1 (why tight coefficient bounds slow the method)
  propG    estimator depends on P: exact local-linear covariances + nonlinear Monte Carlo
  thmL     restricted Chernoff event vs number of shots for several tangent dimensions d

Usage:  rustlib/.venv/bin/python -I research/mathematics_of_paper/pnp_restricted/experiments.py [--quick] [--only NAME ...]
Outputs: results/*.npz, results/summary.json, figures/*.pdf
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"

N = 64
LEV = 4
NC = 4
NN = N * N
APPROX = np.zeros((N, N), bool)
APPROX[: N >> LEV, : N >> LEV] = True
APPROX_FLAT = APPROX.ravel()

# paper parameters of Hong et al. (2024), Algorithm 2
TH1, TH2, DELTA_SG = 2e-6, 200.0, 1e-8


# ----------------------------------------------------------------------------------------------------------------
# orthonormal 2-D Haar, batched over leading axes
# ----------------------------------------------------------------------------------------------------------------
def haar(x):
    c = np.array(x, dtype=float, copy=True)
    for lev in range(LEV):
        m = N >> lev
        h = m // 2
        b = c[..., :m, :m]
        p, q, r, s = b[..., 0::2, 0::2], b[..., 0::2, 1::2], b[..., 1::2, 0::2], b[..., 1::2, 1::2]
        out = np.empty_like(b)
        out[..., :h, :h] = (p + q + r + s) / 2
        out[..., :h, h:] = (p - q + r - s) / 2
        out[..., h:, :h] = (p + q - r - s) / 2
        out[..., h:, h:] = (p - q - r + s) / 2
        c[..., :m, :m] = out
    return c


def ihaar(c):
    x = np.array(c, dtype=float, copy=True)
    for lev in reversed(range(LEV)):
        m = N >> lev
        h = m // 2
        b = x[..., :m, :m]
        ll, lh, hl, hh = b[..., :h, :h], b[..., :h, h:], b[..., h:, :h], b[..., h:, h:]
        out = np.empty_like(b)
        out[..., 0::2, 0::2] = (ll + lh + hl + hh) / 2
        out[..., 0::2, 1::2] = (ll - lh + hl - hh) / 2
        out[..., 1::2, 0::2] = (ll + lh - hl - hh) / 2
        out[..., 1::2, 1::2] = (ll - lh - hl + hh) / 2
        x[..., :m, :m] = out
    return x


def soft(c, lam):
    out = np.sign(c) * np.maximum(np.abs(c) - lam, 0.0)
    out[..., APPROX] = c[..., APPROX]
    return out


def denoise(z, lam):
    return ihaar(soft(haar(z), lam))


# ----------------------------------------------------------------------------------------------------------------
# phantom, coils, sampling
# ----------------------------------------------------------------------------------------------------------------
def shepp_logan(n):
    E = [  # A, a, b, x0, y0, phi(deg): modified Shepp-Logan (Toft)
        (1.0, 0.69, 0.92, 0.0, 0.0, 0), (-0.8, 0.6624, 0.874, 0.0, -0.0184, 0),
        (-0.2, 0.11, 0.31, 0.22, 0.0, -18), (-0.2, 0.16, 0.41, -0.22, 0.0, 18),
        (0.1, 0.21, 0.25, 0.0, 0.35, 0), (0.1, 0.046, 0.046, 0.0, 0.1, 0),
        (0.1, 0.046, 0.046, 0.0, -0.1, 0), (0.1, 0.046, 0.023, -0.08, -0.605, 0),
        (0.1, 0.023, 0.023, 0.0, -0.606, 0), (0.1, 0.023, 0.046, 0.06, -0.605, 0)]
    t = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(t, -t)
    img = np.zeros((n, n))
    for A, a, b, x0, y0, phi in E:
        ph = np.deg2rad(phi)
        xr = (X - x0) * np.cos(ph) + (Y - y0) * np.sin(ph)
        yr = -(X - x0) * np.sin(ph) + (Y - y0) * np.cos(ph)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += A
    return img


def make_coils(n, nc):
    t = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(t, -t)
    centres = [(0.95, 0.0), (-0.95, 0.0), (0.0, 0.95), (0.0, -0.95)][:nc]
    S = []
    for j, (cx, cy) in enumerate(centres):
        mag = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 0.8 ** 2))
        pha = np.exp(1j * np.pi * (0.35 * (j + 1) * X / 2 - 0.25 * Y + 0.2 * j))
        S.append(mag * pha)
    S = np.array(S)
    return S / np.sqrt(np.max(np.sum(np.abs(S) ** 2, 0)))


def density(n, k0=16.0):
    k = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(k, k)
    nu = 1.0 / (1.0 + (KX ** 2 + KY ** 2) / k0 ** 2)
    return nu / nu.sum()


def draw_counts(nu, M, rng):
    idx = rng.choice(nu.size, size=M, replace=True, p=nu.ravel())
    return np.bincount(idx, minlength=nu.size).reshape(nu.shape).astype(float)


class Op:
    """Real-linear multi-coil sampling operator; H is the normalised normal operator (||H|| = 1)."""

    def __init__(self, S, counts):
        self.S, self.c, self.L = S, counts, 1.0
        self.L = power_norm(self.H)

    def H(self, x):
        y = np.zeros(np.shape(x))
        for c in range(NC):
            k = np.fft.fft2(self.S[c] * x, norm="ortho") * self.c
            y += np.real(np.conj(self.S[c]) * np.fft.ifft2(k, norm="ortho"))
        return y / self.L

    def AT_noise(self, sigma, rng):
        """A^T n / L for complex noise with N(0, sigma^2) real and imaginary parts (aggregated per location)."""
        y = np.zeros((N, N))
        for c in range(NC):
            g = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * sigma * np.sqrt(self.c)
            y += np.real(np.conj(self.S[c]) * np.fft.ifft2(g, norm="ortho"))
        return y / self.L

    def data_rms(self, x):
        tot, cnt = 0.0, 0.0
        for c in range(NC):
            k = np.fft.fft2(self.S[c] * x, norm="ortho")
            tot += np.sum(self.c * np.abs(k) ** 2)
            cnt += self.c.sum()
        return np.sqrt(tot / cnt)


def power_norm(apply, iters=200, seed=0):
    v = np.random.default_rng(seed).standard_normal((N, N))
    lam = 0.0
    for _ in range(iters):
        w = apply(v)
        lam = np.linalg.norm(w) / np.linalg.norm(v)
        v = w / np.linalg.norm(w)
    return lam


def wavelet_matrix(apply, chunk=256):
    """Matrix of x -> apply(x) in Haar coordinates, i.e. W apply W^T (dense NN x NN)."""
    G = np.empty((NN, NN))
    for s in range(0, NN, chunk):
        idx = np.arange(s, min(s + chunk, NN))
        E = np.zeros((len(idx), NN))
        E[np.arange(len(idx)), idx] = 1.0
        cols = haar(apply(ihaar(E.reshape(-1, N, N)))).reshape(len(idx), NN)
        G[:, idx] = cols.T
    return G


# ----------------------------------------------------------------------------------------------------------------
# ZMSHR1 (Hong, Xu, Hu, Fessler 2024, Algorithm 2), real inner product
# ----------------------------------------------------------------------------------------------------------------
A_GRID = np.concatenate([[0.0], np.logspace(-9, 0, 4000)])


def zmshr1(s, m, th1=TH1, th2=TH2, delta=DELTA_SG):
    ss, sm, mm = float(s.ravel() @ s.ravel()), float(s.ravel() @ m.ravel()), float(m.ravel() @ m.ravel())
    a = A_GRID
    sv = a * ss + (1 - a) * sm
    vv = a * a * ss + 2 * a * (1 - a) * sm + (1 - a) ** 2 * mm
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (sv > 0) & (th1 <= sv / ss) & (vv / sv <= th2)
    i = int(np.argmax(ok)) if ok.any() else len(a) - 1
    a = float(A_GRID[i])
    v = a * s + (1 - a) * m
    sv, vv = float(s.ravel() @ v.ravel()), float(v.ravel() @ v.ravel())
    rho, r2 = ss / sv, ss / vv
    tau = rho - np.sqrt(max(rho * rho - r2, 0.0))
    u = s - tau * v
    uv, nu_ = float(u.ravel() @ v.ravel()), float(np.linalg.norm(u))
    cosuv = uv / (nu_ * np.sqrt(vv)) if nu_ > 0 else 1.0
    fired = uv <= delta * nu_ * np.sqrt(vv)
    lam_max = tau if fired else tau + nu_ ** 2 / uv
    sin_t = np.sqrt(max(1.0 - sv * sv / (ss * vv), 0.0))
    return dict(a=a, tau=tau, u=(0 * u if fired else u), uv=uv, lam_max=lam_max, lam_min=tau, cos_uv=cosuv,
                fired=bool(fired), sin=sin_t, rho=rho, curv=sm / ss)


def zm_apply(P, g):
    if P["uv"] <= 0 or not np.any(P["u"]):
        return P["tau"] * g
    return P["tau"] * g + P["u"] * (float(P["u"].ravel() @ g.ravel()) / P["uv"])


# ----------------------------------------------------------------------------------------------------------------
# problem setup
# ----------------------------------------------------------------------------------------------------------------
def setup(M=520, lam=0.005, snr=0.01, seed=1):
    rng = np.random.default_rng(seed)
    x0 = shepp_logan(N)
    S = make_coils(N, NC)
    nu = density(N)
    counts = draw_counts(nu, M, rng)
    op = Op(S, counts)
    sigma = snr * op.data_rms(x0)
    noise = op.AT_noise(sigma, np.random.default_rng(seed + 100))
    b = op.H(x0) + noise
    return dict(rng=rng, x0=x0, S=S, nu=nu, counts=counts, op=op, sigma=sigma, noise=noise, b=b, lam=lam, M=M)


def circulant_precond(P):
    """Density-compensation-type circulant preconditioner, scaled so that ||P^1/2 H P^1/2|| = 1."""
    op, counts = P["op"], P["counts"]
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k)
    ker = np.exp(-(KX ** 2 + KY ** 2) / (2 * 1.5 ** 2))
    ker /= ker.sum()
    dens = np.real(np.fft.ifft2(np.fft.fft2(counts) * np.fft.fft2(ker)))
    cover = np.mean(np.sum(np.abs(P["S"]) ** 2, 0))
    p = 1.0 / (np.maximum(dens, 0) * cover / op.L + 0.02)
    sq = np.sqrt(p)

    def half(x):
        return np.real(np.fft.ifft2(np.fft.fft2(x, norm="ortho") * sq, norm="ortho"))

    a = 1.0 / power_norm(lambda x: half(op.H(half(x))))
    pa = p * a

    def apply(x):
        return np.real(np.fft.ifft2(np.fft.fft2(x, norm="ortho") * pa, norm="ortho"))

    return apply


def run_pnp(P, iters, mode="I", precond=None, th1=TH1, th2=TH2, x_init=None, b=None, tol=0.0, keep=False):
    op, lam = P["op"], P["lam"]
    b = P["b"] if b is None else b
    x = (b.copy() if x_init is None else x_init.copy())  # paper: A^H y initialisation
    x = denoise(x, lam)
    x_prev = g_prev = None
    out = dict(x=[], z=[], zm=[])
    for k in range(iters):
        g = op.H(x) - b
        info = None
        if mode == "I":
            step = g
        elif mode == "circ":
            step = precond(g)
        elif mode == "zm":
            if x_prev is None:
                step = g
            else:
                info = zmshr1(x - x_prev, g - g_prev, th1, th2)
                step = zm_apply(info, g)
        else:
            raise ValueError(mode)
        z = x - step
        xn = denoise(z, lam)
        if keep:
            out["x"].append(x)
            out["z"].append(z)
        out["zm"].append(info)
        x_prev, g_prev, x = x, g, xn
        if tol > 0 and np.linalg.norm(x - x_prev) <= tol * max(np.linalg.norm(x), 1e-30):
            break
    out["final"] = x
    if keep:
        out["x"].append(x)
    return out


# ----------------------------------------------------------------------------------------------------------------
# helpers on supports (Haar coordinates)
# ----------------------------------------------------------------------------------------------------------------
def supp_secant(cz, c0, lam):
    """indices where the soft-threshold secant slope between cz and c0 is nonzero, plus the coarse band"""
    return np.flatnonzero((np.abs(cz) > lam) | (np.abs(c0) > lam) | APPROX_FLAT)


def supp_coeffs(c):
    return np.flatnonzero((c != 0) | APPROX_FLAT)


def opnorm(Msub):
    """exact spectral norm via the top eigenvalue of X^T X (LAPACK MRRR, robust to clustered singular values)"""
    X = Msub.T @ Msub if Msub.shape[0] >= Msub.shape[1] else Msub @ Msub.T
    n = X.shape[0]
    return float(np.sqrt(max(sla.eigh(X, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0], 0.0)))


def lam_min_sym(X):
    n = X.shape[0]
    return float(sla.eigh(X, eigvals_only=True, subset_by_index=[0, 0])[0])


def save(name, **arrs):
    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RES / f"{name}.npz", **arrs)


def load(name):
    f = RES / f"{name}.npz"
    return dict(np.load(f, allow_pickle=True)) if f.exists() else None


# ----------------------------------------------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------------------------------------------
def exp_lemmaA(quick):
    rng = np.random.default_rng(7)
    trials = 300 if quick else 2000
    err_eig, err_cos, err_sec = 0.0, 0.0, 0.0
    for t in range(trials):
        n = int(rng.integers(5, 60))
        cplx = t % 2 == 1
        r = int(rng.integers(1, n + 1))
        B = rng.standard_normal((r, n)) + (1j * rng.standard_normal((r, n)) if cplx else 0)
        Hm = B.conj().T @ B * 10.0 ** rng.uniform(-3, 3)
        s = rng.standard_normal(n) + (1j * rng.standard_normal(n) if cplx else 0)
        if t % 5 == 0:  # secant partly in the null space
            nul = sla.null_space(Hm)
            if nul.shape[1]:
                s = s + 10 * nul @ rng.standard_normal(nul.shape[1])
        a = float(rng.uniform(0, 1)) if t % 3 == 0 else 0.0
        v = a * s + (1 - a) * (Hm @ s)
        sv = np.vdot(s, v).real
        if sv <= 0:
            continue
        ss, vv = np.vdot(s, s).real, np.vdot(v, v).real
        rho = ss / sv
        tau = rho - np.sqrt(max(rho * rho - ss / vv, 0))
        u = s - tau * v
        uv = np.vdot(u, v)
        if abs(uv) < 1e-14 * np.linalg.norm(u) * np.sqrt(vv):
            continue
        P = tau * np.eye(n) + np.outer(u, u.conj()) / uv.real
        sin_t = np.sqrt(max(1 - sv * sv / (ss * vv), 0))
        ev = np.linalg.eigvalsh(P)
        pred = np.full(n, rho * (1 - sin_t))
        pred[-1] = rho * (1 + sin_t)
        err_eig = max(err_eig, np.max(np.abs(ev - np.sort(pred))) / ev[-1])
        err_sec = max(err_sec, np.linalg.norm(P @ v - s) / np.linalg.norm(s))
        cos_pred = np.sqrt(1 - sin_t ** 2) / np.sqrt(2 * (1 + sin_t))
        err_cos = max(err_cos, abs(uv.real / (np.linalg.norm(u) * np.sqrt(vv)) - cos_pred))
    return dict(trials=trials, max_rel_eig_err=err_eig, max_cos_err=err_cos, max_secant_err=err_sec)


def exp_ista(P, GH, quick):
    iters = 1500 if quick else 4000
    lam, x0, op = P["lam"], P["x0"], P["op"]
    r = run_pnp(P, iters, "I", keep=True)
    xs, zs = r["x"], r["z"]
    c0 = haar(x0).ravel()
    Dx0 = denoise(x0, lam)
    eps_vec = Dx0 - x0
    eps0 = float(np.linalg.norm(eps_vec))
    Meps_w = haar(eps_vec - op.H(eps_vec)).ravel()
    w_w = haar(P["noise"]).ravel()  # w = alpha A^T n / L: the noise part of z_k - x0
    MW = np.eye(NN) - GH  # W M W^T for P = I
    K = len(zs)
    err = np.array([np.linalg.norm(x - x0) for x in xs])
    Anorm = np.array([np.linalg.norm(x - Dx0) for x in xs])  # a_k = x_k - D(x0) exactly (eta = 0)
    T = [supp_secant(haar(z).ravel(), c0, lam) for z in zs]  # T_k: tangent support of S_k
    dk = np.array([len(t) for t in T])
    delta = np.full(K, np.nan)
    Wk = np.array([np.linalg.norm(w_w[t]) for t in T])
    muk = np.array([np.linalg.norm(Meps_w[t]) for t in T])
    cache = {}
    for k in range(1, K):
        key = (T[k].tobytes(), T[k - 1].tobytes())
        if key not in cache:
            cache[key] = opnorm(MW[np.ix_(T[k], T[k - 1])])
            if len(cache) % 100 == 0:
                print(f"  ista: {len(cache)} distinct support pairs at k={k}", flush=True)
        delta[k] = cache[key]
    # a-posteriori bound: A_{k+1} <= delta_k A_k + mu_k + W_k (k >= 1), E_k <= A_k + eps0; start from exact A_1
    Ab = np.full(K + 1, np.nan)
    Ab[1] = Anorm[1]
    for k in range(1, K):
        Ab[k + 1] = delta[k] * Ab[k] + muk[k] + Wk[k]
    Eb = Ab + eps0
    k0 = K // 2
    dsup, Wsup, musup = np.nanmax(delta[k0:]), Wk[k0:].max(), muk[k0:].max()
    limsup_bound = (Wsup + musup) / (1 - dsup) + eps0
    # local (Ostrowski) rate at the limit: rho(J_D(z*) M) = ||Pi* M Pi*|| for the projector J_D(z*)
    zstar = haar(zs[-1]).ravel()
    Ts = np.flatnonzero((np.abs(zstar) > lam) | APPROX_FLAT)
    sub = MW[np.ix_(Ts, Ts)]
    ostrowski = float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (sub + sub.T)))))
    steps = np.array([np.linalg.norm(xs[k + 1] - xs[k]) for k in range(len(xs) - 1)])
    j = np.flatnonzero(steps > 1e-12 * steps[0])
    tail = j[int(0.6 * len(j)):]
    obs_rate = float(np.exp(np.polyfit(tail, np.log(steps[tail]), 1)[0])) if len(tail) > 10 else np.nan
    lam_min_H = float(np.linalg.eigvalsh(GH)[0])
    rankH = int(np.sum(np.linalg.eigvalsh(GH) > 1e-10))
    T0 = supp_secant(c0, c0, lam)
    return dict(err=err, Anorm=Anorm, dk=dk, delta=delta, Wk=Wk, muk=muk, Ab=Ab, Eb=Eb, eps0=eps0,
                limsup_bound=limsup_bound, dsup=dsup, Wsup=Wsup, musup=musup, final_err=err[-1],
                ostrowski=ostrowski, obs_rate=obs_rate, lam_min_H=lam_min_H, rankH=rankH, d_star=len(Ts),
                d0=len(T0), final=xs[-1], steps=steps, iters=K)


def exp_zm(P, GH, quick):
    iters = 400 if quick else 1000
    r = run_pnp(P, iters, "zm", keep=True)
    xs, infos = r["x"], r["zm"]
    rec = {k: [] for k in ["k", "a", "tau", "lam_max", "cos_uv", "fired", "sin", "rho", "curv", "dU", "restr"]}
    cache = {}
    for k, info in enumerate(infos):
        if info is None:
            continue
        U = np.union1d(supp_coeffs(haar(xs[k]).ravel()), supp_coeffs(haar(xs[k - 1]).ravel()))
        key = U.tobytes()
        if key not in cache:
            cache[key] = lam_min_sym(GH[np.ix_(U, U)])  # (1 - delta_U) L with L = 1
        lmin_U = cache[key]
        rec["k"].append(k)
        for f in ["a", "tau", "lam_max", "cos_uv", "fired", "sin", "rho", "curv"]:
            rec[f].append(info[f])
        rec["dU"].append(1.0 - lmin_U)
        rec["restr"].append((1.0 + info["sin"]) / lmin_U if lmin_U > 1e-12 else np.inf)
    out = {k: np.array(v, dtype=float) for k, v in rec.items()}
    out["err"] = np.array([np.linalg.norm(x - P["x0"]) for x in xs])
    out["final"] = xs[-1]
    out["paper_bound"] = (DELTA_SG + 1) / (DELTA_SG * TH1)
    out["dfree_bound"] = (1 + np.sqrt(1 - TH1 / TH2)) / TH1
    out["sg_margin"] = 0.5 * np.sqrt(TH1 / TH2)
    out["tau_low"] = 1.0 / (TH2 * (1 + np.sqrt(1 - TH1 / TH2)))
    return out


def exp_theta(P, quick):
    iters = 400 if quick else 1000
    ths = [2e-6, 1e-2, 1e-1, 3e-1, 6e-1]
    out = dict(theta1=np.array(ths))
    for t in ths:
        r = run_pnp(P, iters, "zm", th1=t, keep=True)
        out[f"err_{t:g}"] = np.array([np.linalg.norm(x - P["x0"]) for x in r["x"]])
        a = np.array([i["a"] for i in r["zm"] if i is not None])
        lm = np.array([i["lam_max"] for i in r["zm"] if i is not None])
        out[f"amix_{t:g}"] = float(np.mean(a > 0))
        out[f"lmax_{t:g}"] = float(np.median(lm))
    rI = run_pnp(P, iters, "I", keep=True)
    out["err_I"] = np.array([np.linalg.norm(x - P["x0"]) for x in rI["x"]])
    return out


def exp_propG(P, GH, quick, zm_final_info):
    lam, x0, op = P["lam"], P["x0"], P["op"]
    # noise in normalised gradient: A^T n / L ~ N(0, sigma^2 H / L) = N(0, (sigma^2/L) H_tilde)
    sig2 = P["sigma"] ** 2 / op.L
    c0 = haar(x0).ravel()
    T0 = supp_secant(c0, c0, lam)
    d = len(T0)
    eps_w = haar(denoise(x0, lam) - x0).ravel()
    eps_perp = eps_w.copy()
    eps_perp[T0] = 0.0
    circ = circulant_precond(P)
    Pc_w = wavelet_matrix(circ)
    Pc_w = 0.5 * (Pc_w + Pc_w.T)
    Gt = GH[np.ix_(T0, T0)]
    blue = sig2 * np.trace(np.linalg.inv(Gt))
    tauT = 1.0
    PT_w = tauT * np.eye(NN)
    PT_w[np.ix_(T0, T0)] = np.linalg.inv(Gt)
    res = dict(d=d, blue=blue)
    mats = {"I": None, "circ": Pc_w, "PT": PT_w}
    if zm_final_info is not None:
        uw = haar(zm_final_info["u"]).ravel()
        mats["zm"] = ("rank1", zm_final_info["tau"], uw, zm_final_info["uv"])
    for name, Pw in mats.items():
        if Pw is None:
            PH = GH
            PHP = GH
        elif isinstance(Pw, tuple):
            _, tau, uw, uv = Pw
            PH = tau * GH + np.outer(uw, uw @ GH) / uv
            HP = PH.T
            PHP = tau * HP + np.outer(uw, uw @ HP) / uv
        else:
            PH = Pw @ GH
            PHP = PH @ Pw
        G = PH[np.ix_(T0, T0)]
        Gi = np.linalg.inv(G)
        cov = sig2 * Gi @ PHP[np.ix_(T0, T0)] @ Gi.T
        lam_G = np.linalg.eigvals(np.eye(d) - G)
        # local affine model of soft thresholding at the truth: D(z) = Pi0 z + c, eps = D(x0) - x0 (alpha = 1)
        # bias b_P = Pi0perp eps + G_P^{-1} (Pi0 eps - Pi0 P H Pi0perp eps)
        bT = Gi @ (eps_w[T0] - PH[T0, :] @ eps_perp)
        bias2 = float(bT @ bT + eps_perp @ eps_perp)
        res[name] = dict(trace=float(np.trace(cov)), ratio=float(np.trace(cov) / blue),
                         rho_T=float(np.max(np.abs(lam_G))), min_re=float(np.min(np.real(np.linalg.eigvals(G)))),
                         bias2_lin=bias2, mse_lin=bias2 + float(np.trace(cov)))
    # nonlinear Monte Carlo (soft-threshold PnP iterated to convergence)
    R = 6 if quick else 16
    itI = 2000 if quick else 6000
    finals = {"I": [], "circ": [], "zm": []}
    rng = np.random.default_rng(1234)
    Hx0 = op.H(x0)
    for rr in range(R):
        nz = op.AT_noise(P["sigma"], rng)
        b = Hx0 + nz
        finals["I"].append(run_pnp(P, itI, "I", b=b, tol=1e-9)["final"])
        finals["circ"].append(run_pnp(P, itI, "circ", precond=circ, b=b, tol=1e-9)["final"])
        finals["zm"].append(run_pnp(P, itI // 3, "zm", b=b, tol=1e-9)["final"])
    mc = {}
    for k, v in finals.items():
        v = np.array(v)
        mean = v.mean(0)
        mc[k] = dict(mse=float(np.mean(np.sum((v - x0) ** 2, axis=(1, 2)))),
                     bias2=float(np.sum((mean - x0) ** 2)),
                     var=float(np.sum(v.var(0, ddof=1))))
    res["mc"] = mc
    res["R"] = R
    return res


def tangent_basis(kind, dt, c0, rng):
    if kind == "random":  # incoherent: span of d Gaussian images
        Q, _ = np.linalg.qr(rng.standard_normal((NN, dt)))
        return Q.T.reshape(dt, N, N)
    order = np.sort(np.abs(c0[~APPROX_FLAT]))[::-1]  # Haar: largest coefficients of the truth + coarse band
    thr = order[dt - APPROX.sum() - 1] * 0.999
    T0 = np.flatnonzero((np.abs(c0) >= thr) | APPROX_FLAT)
    E = np.zeros((len(T0), NN))
    E[np.arange(len(T0)), T0] = 1.0
    return ihaar(E.reshape(len(T0), N, N))


def exp_thmL(P, quick):
    S, nu = P["S"], P["nu"]
    c0 = haar(P["x0"]).ravel()
    cases = [("random", 40), ("random", 80), ("random", 160)] + ([] if quick else [("random", 320)])
    cases += [("haar", 60), ("haar", 120)] + ([] if quick else [("haar", 240)])
    trials = 10 if quick else 24
    t_dev, p_fail = 0.5, 0.1
    rng = np.random.default_rng(99)
    out = dict(t=t_dev, p=p_fail, cases=np.array([f"{k}{d}" for k, d in cases]))
    for kind, dt in cases:
        V = tangent_basis(kind, dt, c0, rng)
        d = V.shape[0]
        X = np.array([np.fft.fft2(S[c] * V, norm="ortho") for c in range(NC)])  # (NC, d, N, N)
        X = X.reshape(NC, d, NN).transpose(2, 0, 1)  # (NN, NC, d)
        R = np.concatenate([X.real, X.imag], axis=1)  # (NN, 2NC, d): real block of a shot
        Gom = np.einsum("wai,wbi->wab", R, R)
        Phi0 = float(np.max(np.linalg.eigvalsh(Gom)[:, -1]))
        Rn = R * np.sqrt(nu.ravel())[:, None, None]
        Hbar = np.einsum("wai,waj->ij", Rn, Rn)
        evb = np.linalg.eigvalsh(Hbar)
        lm, lp = float(evb[0]), float(evb[-1])
        Mpred = 2 * Phi0 / (t_dev ** 2 * lm) * np.log(d / p_fail)
        Ms = np.unique(np.round(np.geomspace(max(d / (2 * NC) * 0.7, 4), min(1.3 * Mpred, 60000), 12)).astype(int))
        succ, ratio_med = [], []
        for M in Ms:
            ok, rat = 0, []
            for _ in range(trials):
                cnt = draw_counts(nu, int(M), rng).ravel()
                w = np.flatnonzero(cnt)
                Rs = (R[w] * np.sqrt(cnt[w])[:, None, None]).reshape(-1, d)
                lmin = lam_min_sym(Rs.T @ Rs)
                rat.append(lmin / (M * lm))
                ok += lmin >= (1 - t_dev) * M * lm
            succ.append(ok / trials)
            ratio_med.append(np.median(rat))
        out[f"{kind}{dt}"] = dict(kind=kind, d=d, Phi0=Phi0, lam_minus=lm, lam_plus=lp, kcoh=Phi0 * 2 * NC / (d * lm),
                                  Mpred=Mpred, Ms=Ms, succ=np.array(succ), ratio=np.array(ratio_med))
        print(f"  thmL {kind}{dt}: d={d} kcoh={Phi0 * 2 * NC / (d * lm):.2f} Mpred={Mpred:.0f}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical slots 1-5 (white surface)
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 7.5,
        "pdf.fonttype": 42})


def fig_ista(r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    k = np.arange(len(r["err"]))
    ax[0].semilogy(k, r["err"], color=C[0], label=r"$\|x_k-x^{\natural}\|$ (observed)")
    ax[0].semilogy(np.arange(len(r["Eb"])), r["Eb"], color=C[1], ls="--", label="a-posteriori bound (Thm 4.4)")
    ax[0].axhline(r["limsup_bound"], color=C[2], ls="-.", label=r"$\limsup$ bound")
    ax[0].axhline(r["eps0"], color=INK2, lw=0.8, ls=":", label=r"$\varepsilon_0=\|D(x^{\natural})-x^{\natural}\|$")
    ax[0].set_xlabel("iteration $k$")
    ax[0].set_ylabel("error")
    ax[0].legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
    ax[1].plot(np.arange(len(r["delta"])), r["delta"], color=C[0], label=r"$\delta_k=\|\Pi_kM\Pi_{k-1}\|$")
    ax[1].axhline(1.0, color=C[1], ls="--", label=r"$\rho(I-\alpha H)=1$ (Hong et al. $q\geq1$)")
    lo = np.nanmin(r["delta"])
    ax[1].set_ylim(lo - 0.6 * (1 - lo), 1 + 0.15 * (1 - lo))
    ax[1].set_xlabel("iteration $k$")
    ax[1].set_ylabel("contraction constant")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "ista_restricted.pdf")
    plt.close(fig)


def fig_zm(r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    k = r["k"]
    ax[0].semilogy(k, r["lam_max"], color=C[0], label=r"$\lambda_{\max}(P_k)$")
    ax[0].semilogy(k, r["restr"], color=C[1], ls="--", label=r"restricted bound $(1+\sin\vartheta_k)/(1-\delta_{U_k})$")
    ax[0].axhline(float(r["dfree_bound"]), color=C[2], ls="-.", label=r"$\delta$-free bound $(1+\sqrt{1-\theta_1/\theta_2})/\theta_1$")
    ax[0].axhline(float(r["paper_bound"]), color=C[3], ls=":", label=r"Hong et al. $(\delta+1)/(\delta\theta_1)$")
    ax[0].set_ylim(0.5, 3 * float(r["paper_bound"]))
    ax[0].set_xlabel("iteration $k$")
    ax[0].set_ylabel("eigenvalue")
    ax[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.94), fontsize=6.5)
    ax[1].semilogy(k, r["cos_uv"], color=C[0], label=r"$\cos\angle(u_k,v_k)$")
    ax[1].axhline(float(r["sg_margin"]), color=C[1], ls="--", label=r"$\frac{1}{2}\sqrt{\theta_1/\theta_2}$ (Cor. 3.3)")
    ax[1].axhline(DELTA_SG, color=C[3], ls=":", label=r"safeguard $\delta=10^{-8}$")
    ax[1].set_xlabel("iteration $k$")
    ax[1].set_ylabel("cosine")
    ax[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.12))
    fig.tight_layout()
    fig.savefig(FIG / "zmshr1_spectrum.pdf")
    plt.close(fig)


def fig_theta(r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))
    ax.semilogy(r["err_I"], color=INK2, lw=1.0, label="PnP-ISTA ($P=I$)")
    for i, t in enumerate(r["theta1"]):
        ax.semilogy(r[f"err_{t:g}"], color=C[i], ls=LS[i], label=rf"ZMSHR1, $\theta_1={t:g}$")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\|x_k-x^{\natural}\|$")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "theta1_sweep.pdf")
    plt.close(fig)


def fig_thmL(r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    pts = {"random": ([], []), "haar": ([], [])}
    i = 0
    for key in [str(c) for c in r["cases"]]:
        q = r[key]
        if q["kind"] == "random":
            ax[0].semilogx(q["Ms"], q["succ"], color=C[i], ls=LS[i], marker="o", ms=3, label=rf"$d={int(q['d'])}$")
            ax[0].axvline(q["Mpred"], color=C[i], lw=0.8, ls=":")
            i += 1
        i90 = np.flatnonzero(np.asarray(q["succ"]) >= 0.9)
        if len(i90):
            pts[q["kind"]][0].append(q["Phi0"] / q["lam_minus"] * np.log(q["d"] / r["p"]))
            pts[q["kind"]][1].append(q["Ms"][i90[0]])
    ax[0].set_xlabel("number of shots $N_b$")
    ax[0].set_ylabel(r"$\Pr[\lambda_{\min}\geq(1-t)N_b\lambda_-]$")
    ax[0].set_title("incoherent tangent spaces", fontsize=8, color=INK2)
    ax[0].legend(loc="lower right")
    allx = np.array(pts["random"][0] + pts["haar"][0])
    g = np.geomspace(allx.min() / 1.5, allx.max() * 1.5, 10)
    ax[1].loglog(g, 2 / r["t"] ** 2 * g, color=INK2, lw=1.0, ls="--", label="lower-tail Chernoff threshold")
    xr, yr = np.array(pts["random"][0]), np.array(pts["random"][1])
    ax[1].loglog(xr, yr, "o", color=C[0], ms=5, label="incoherent (random) tangent")
    if len(xr) > 1:
        cfit = np.exp(np.mean(np.log(yr / xr)))
        ax[1].loglog(g, cfit * g, color=C[0], lw=1.0, ls="-.", label=rf"slope 1, constant {cfit:.2f}")
    ax[1].loglog(pts["haar"][0], pts["haar"][1], "s", color=C[1], ms=5, label="Haar tangent of the phantom")
    ax[1].set_xlabel(r"$\frac{\Phi_0}{\lambda_-}\log\frac{d}{p}$")
    ax[1].set_ylabel("shots for 90 % success")
    ax[1].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "thmL_chernoff.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
def to_json(x):
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size < 64 else f"<array {x.shape}>"
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    global RES, FIG
    if args.quick:
        RES, FIG = HERE / "results_quick", HERE / "figures_quick"
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    want = (lambda n: args.only is None or n in args.only)
    summ_f = RES / "summary.json"
    summary = json.loads(summ_f.read_text()) if summ_f.exists() else {}
    t0 = time.time()
    P = setup()
    c0 = haar(P["x0"]).ravel()
    print(f"setup: L={P['op'].L:.4g}, M={P['M']}, distinct={int((P['counts'] > 0).sum())}, "
          f"sigma={P['sigma']:.3g}, d0={len(supp_secant(c0, c0, P['lam']))}", flush=True)
    gh_f = RES / "GH.npy"
    if gh_f.exists():
        GH = np.load(gh_f)
    else:
        GH = wavelet_matrix(P["op"].H)
        GH = 0.5 * (GH + GH.T)
        np.save(gh_f, GH)
    print(f"GH ready ({time.time() - t0:.0f}s)", flush=True)
    summary["setup"] = dict(N=N, coils=NC, levels=LEV, M=P["M"], lam=P["lam"], sigma=P["sigma"], L=P["op"].L,
                            real_data_dim=int(2 * NC * (P["counts"] > 0).sum()), n=NN)

    if want("lemmaA"):
        summary["lemmaA"] = exp_lemmaA(args.quick)
        print("lemmaA", summary["lemmaA"], flush=True)

    if want("ista"):
        r = load("ista")
        if r is None:
            r = exp_ista(P, GH, args.quick)
            save("ista", **r)
        fig_ista(r)
        summary["ista"] = {k: to_json(r[k]) for k in ["eps0", "limsup_bound", "dsup", "Wsup", "musup", "final_err",
                                                      "ostrowski", "obs_rate", "lam_min_H", "rankH", "d_star", "d0",
                                                      "iters"]}
        summary["ista"]["d_range"] = [int(np.min(r["dk"])), int(np.max(r["dk"]))]
        print("ista", summary["ista"], f"({time.time() - t0:.0f}s)", flush=True)

    zm_final = None
    if want("zm") or want("propG"):
        r = load("zm")
        if r is None:
            r = exp_zm(P, GH, args.quick)
            save("zm", **r)
        if want("zm"):
            fig_zm(r)
        summary["zm"] = dict(max_lam_max=float(np.max(r["lam_max"])), max_restr_ratio=float(np.max(r["lam_max"] / r["restr"])),
                             min_cos_uv=float(np.min(r["cos_uv"])), fired=int(np.sum(r["fired"])),
                             a_positive=int(np.sum(r["a"] > 0)), paper_bound=float(r["paper_bound"]),
                             dfree_bound=float(r["dfree_bound"]), sg_margin=float(r["sg_margin"]),
                             min_tau=float(np.min(r["tau"])), tau_low=float(r["tau_low"]),
                             final_err=float(r["err"][-1]), max_dU=float(np.max(r["dU"])))
        print("zm", summary["zm"], f"({time.time() - t0:.0f}s)", flush=True)
        # frozen final ZMSHR1 preconditioner for Proposition 5.1
        rr = run_pnp(P, 60, "zm", x_init=r["final"])
        zm_final = [i for i in rr["zm"] if i is not None][-1]

    if want("theta"):
        r = load("theta")
        if r is None:
            r = exp_theta(P, args.quick)
            save("theta", **r)
        fig_theta(r)
        summary["theta"] = {f"{t:g}": dict(final_err=float(r[f"err_{t:g}"][-1]), amix=float(r[f"amix_{t:g}"]),
                                           median_lam_max=float(r[f"lmax_{t:g}"])) for t in r["theta1"]}
        summary["theta"]["I_final_err"] = float(r["err_I"][-1])
        print("theta", summary["theta"], f"({time.time() - t0:.0f}s)", flush=True)

    if want("propG"):
        summary["propG"] = to_json(exp_propG(P, GH, args.quick, zm_final))
        print("propG", summary["propG"], f"({time.time() - t0:.0f}s)", flush=True)

    if want("thmL"):
        r = load("thmL")
        if r is None:
            r = exp_thmL(P, args.quick)
            save("thmL", **{k: v for k, v in r.items()})
        r = {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.ndim == 0 else v) for k, v in r.items()}
        fig_thmL(r)
        summary["thmL"] = {k: to_json(v) for k, v in r.items()}
        print("thmL", summary["thmL"], f"({time.time() - t0:.0f}s)", flush=True)

    summ_f.write_text(json.dumps(to_json(summary), indent=1))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
