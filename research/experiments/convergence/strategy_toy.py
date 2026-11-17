#!/usr/bin/env python3
"""Convergence strategy toy (research/convergence_strategy.md, 2026-11-17). CPU, numpy only, self-contained.

Model: T sets (frames / knots) of an N x N complex image, NC smooth coils, Cartesian k-space with per-set sample
multiplicities w_t(k) ~ Poisson(n_t c rho(k)), rho(k) = 1/(1 + |k|^2/k0^2) (centre/edge density ~1e3, like the 3D
centre-out yarn at k-voxel scale). The set weights n_t carry the per-set sample-count spread.
  f(x)   = 1/2 sum_t ||sqrt(W_t) F S x_t - y_t||^2            (no density compensation: BLUE data term)
  quad   = f + mu/2 sum_t ||x_{t+1} - x_t||^2 + eps/2 ||x||^2  (periodic temporal coupling: stands in for a basis / LLR
                                                               curvature that couples the sets)
  l1     = quad + lam ||W x||_1   (orthonormal 2-D Haar per frame, coarse band unpenalised)
  smooth = quad + Moreau envelope_gamma of lam||W.||_1           (the RED / Huber form: metric-independent minimiser)
Cost unit: one application of A^H A to the whole stack (= one gradient pass). Priors, metrics and FFT-only
preconditioners are counted separately (they cost no NUFFT and no coil FFTs; on the real problem a coil-free
circulant application is ~1/(2 L C) of a set's gradient).

Experiments (all distances relative to the method's own target minimiser unless stated):
  E1 quadratic, coupled (mu > 0): metric {I, shared circulant, shared + per-set scalar, per-set circulant,
     prior-aware (k, omega) diagonal} x algorithm {GD, Nesterov+restart, ZMSHR1-M (global), PCG}; spread 1.6 and 16.
  E2 quadratic, uncoupled (mu = 0, ridge): per-set ZMSHR1-M vs global vs PCG; spread 16.
  E3 l1 / smooth: FISTA-I (exact prox), FISTA with a subband x set diagonal majoriser (exact prox, BARISTA-like),
     ADMM-PCG (rho grid), NCG and L-BFGS(3) with the circulant metric on the smoothed objective (exact data line
     search, prior-only bisection), ZMSHR1-M on the smoothed objective (RED-like).
  E4 unregularised semi-convergence with held-out k-space and model error: PCG-circ vs ZMSHR1-M vs GD-circ.
Usage: nice -n 10 rustlib/.venv/bin/python -I research/experiments/convergence/strategy_toy.py OUTDIR full|quick [E1,E2,E3,E4]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"
N, T, NC = 64, 8, 4
K0, EDGE = 1.5, 0.25          # density scale; mean samples per k at the edge for n_t = 1
MAXIT = 150 if QUICK else 300
rng0 = np.random.default_rng(1)
fft2 = lambda a: np.fft.fft2(a, norm="ortho")
ifft2 = lambda a: np.fft.ifft2(a, norm="ortho")
kk = np.fft.fftfreq(N) * N
KX, KY = np.meshgrid(kk, kk, indexing="ij")
RHO = 1.0 / (1.0 + (KX ** 2 + KY ** 2) / K0 ** 2)
C_EDGE = EDGE / RHO[N // 2, 0]
yy, xx = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), indexing="ij")


def coils(nc, rng):
    S = []
    for c in range(nc):
        a = 2 * np.pi * c / nc
        cx, cy = 1.3 * np.cos(a), 1.3 * np.sin(a)
        mag = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.2)
        ph = np.exp(1j * (0.8 * xx * np.cos(a) + 0.8 * yy * np.sin(a) + rng.uniform(0, 2 * np.pi)))
        S.append(mag * ph)
    S = np.array(S)
    return S / np.sqrt((np.abs(S) ** 2).sum(0)).max()


def phantom():
    base = np.zeros((N, N), complex)
    for (cx, cy, ax, ay, v) in [(0, 0, .8, .9, 1.0), (.2, -.1, .3, .25, -.4), (-.3, .3, .15, .2, .5),
                                (.35, .4, .1, .1, .6), (-.2, -.45, .25, .08, .3)]:
        base += v * (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1)
    base = base * np.exp(1j * 0.6 * (xx + 0.5 * yy))
    X = np.zeros((T, N, N), complex)
    for t in range(T):
        ph = 2 * np.pi * t / T
        r = 0.08 + 0.03 * np.sin(ph)
        cx, cy = -0.05 + 0.05 * np.cos(ph), 0.1
        vessel = (((xx - cx) / r) ** 2 + ((yy - cy) / r) ** 2 <= 1) * (0.8 + 0.4 * np.sin(ph)) * np.exp(1j * 1.2 * np.sin(ph))
        X[t] = base + vessel
    return X


class Problem:
    def __init__(self, spread, mu, eps_rel=1e-6, sigma_rel=0.01, seed=2, heldout=0.0, coil_err=0.0):
        rng = np.random.default_rng(seed)
        self.S = coils(NC, np.random.default_rng(7))
        self.n = np.geomspace(1.0 / np.sqrt(spread), np.sqrt(spread), T)
        rng.shuffle(self.n)
        self.W = np.array([rng.poisson(n * C_EDGE * RHO * (1 - heldout)) for n in self.n]).astype(float)
        self.Wh = np.array([rng.poisson(n * C_EDGE * RHO * heldout) for n in self.n]).astype(float) if heldout else None
        self.xt = phantom()
        Strue = self.S if coil_err == 0 else self.S * np.exp(1j * coil_err * (xx ** 2 - yy))  # model error: phase of maps
        clean = fft2(Strue[None] * self.xt[:, None])                       # (T, C, N, N)
        scale = np.sqrt((self.W[:, None] * np.abs(clean) ** 2).sum() / max(self.W.sum() * NC, 1))
        self.sigma = sigma_rel * scale
        noise = (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)) * self.sigma / np.sqrt(2)
        # aggregated samples: sqrt(w) * mean of w noisy samples = sqrt(w) * clean + unit-variance noise
        self.y = np.sqrt(self.W[:, None]) * clean + noise * (self.W[:, None] > 0)
        if heldout:
            nh = (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)) * self.sigma / np.sqrt(2)
            self.yh = np.sqrt(self.Wh[:, None]) * clean + nh * (self.Wh[:, None] > 0)
        self.b = self.AH(self.y)
        self.mu = mu
        self.eps = 0.0
        self.nAHA = 0
        L0 = self.power(lambda v: self.Q(v, count=False), 40)
        self.eps = eps_rel * L0
        self.L = self.power(lambda v: self.Q(v, count=False), 60)
        # coil power spectrum (the coil smear of the k-space diagonal: precond.rs use_conv)
        sh = fft2(self.S)
        self.Pc = (np.abs(sh) ** 2).sum(0) / (N * N)

    def A(self, x):
        return np.sqrt(self.W[:, None]) * fft2(self.S[None] * x[:, None])

    def AH(self, r):
        return (np.conj(self.S)[None] * ifft2(np.sqrt(self.W[:, None]) * r)).sum(1)

    def AHA(self, x, count=True):
        if count:
            self.nAHA += 1
        return (np.conj(self.S)[None] * ifft2(self.W[:, None] * fft2(self.S[None] * x[:, None]))).sum(1)

    def Lt(self, x):
        return 2 * x - np.roll(x, 1, 0) - np.roll(x, -1, 0)

    def Q(self, x, count=True):
        return self.AHA(x, count) + self.mu * self.Lt(x) + self.eps * x

    def power(self, op, it, seed=3):
        v = np.random.default_rng(seed).standard_normal((T, N, N)) + 0j
        lam = 0.0
        for _ in range(it):
            w = op(v)
            lam = np.sqrt(np.vdot(w, w).real / np.vdot(v, v).real)
            v = w / np.linalg.norm(w)
        return lam

    # ---- circulant spectra (coil-convolved density, floored at 1e-3 of the max: precond.rs) ----
    def spec(self, w):
        # diag of F H F^H: lam(k) = sum_k' w(k') Pc(k' - k) (circular cross-correlation with the coil power spectrum)
        lam = np.real(np.fft.ifft2(np.fft.fft2(w) * np.conj(np.fft.fft2(self.Pc))))
        return np.maximum(lam, 1e-3 * lam.max())

    def metrics(self):
        wbar = self.W.mean(0)
        lam_sh = self.spec(wbar)
        nt = self.W.sum((1, 2)) / self.W.sum() * T
        lam_set = np.array([self.spec(w) for w in self.W])
        omega = 2 - 2 * np.cos(2 * np.pi * np.arange(T) / T)      # eigenvalues of the periodic temporal Laplacian
        mu, eps = self.mu, self.eps
        I = lambda g: g.copy()
        shared = lambda g: ifft2(fft2(g) / lam_sh[None])
        scaled = lambda g: ifft2(fft2(g) / (nt[:, None, None] * lam_sh[None]))
        perset = lambda g: ifft2(fft2(g) / lam_set)
        sq = np.sqrt(nt)[:, None, None]
        den = lam_sh[None] + (mu * omega[:, None, None] + eps)          # joint (omega, k) diagonal

        def prior_aware(g, rho=0.0):
            h = fft2(g / sq)
            h = np.fft.fft(h, axis=0) / (den + rho)
            return ifft2(np.fft.ifft(h, axis=0)) / sq
        self.prior_aware = prior_aware
        return {"I": I, "Cshared": shared, "Cshared+n_t": scaled, "Cperset": perset, "K(omega,k)": prior_aware}


def rel(x, xs):
    return np.linalg.norm(x - xs) / np.linalg.norm(xs)


# ------------------------------------------------------------------ quadratic solvers (cost = A^H A applications)
REC = lambda x: x.copy()


def pcg(P, Mf, x0, it, rec=REC):
    x = x0.copy(); r = P.b - P.Q(x); z = Mf(r); p = z.copy(); rz = np.vdot(r, z).real
    hist = [rec(x)]
    for _ in range(it):
        q = P.Q(p); a = rz / np.vdot(p, q).real
        x = x + a * p; r = r - a * q; z = Mf(r); rz2 = np.vdot(r, z).real
        p = z + (rz2 / rz) * p; rz = rz2
        hist.append(rec(x))
    return hist


def pcg_sets(P, Mf, x0, it, rec=REC):
    """independent CG per set (its own alpha_t, beta_t): exact block-separate Krylov; only valid for separable Q"""
    dot = lambda a, b: np.real((np.conj(a) * b).sum(axis=(1, 2)))[:, None, None]
    x = x0.copy(); r = P.b - P.Q(x); z = Mf(r); p = z.copy(); rz = dot(r, z)
    hist = [rec(x)]
    for _ in range(it):
        q = P.Q(p); a = rz / np.maximum(dot(p, q), 1e-300)
        x = x + a * p; r = r - a * q; z = Mf(r); rz2 = dot(r, z)
        p = z + (rz2 / np.maximum(rz, 1e-300)) * p; rz = rz2
        hist.append(rec(x))
    return hist


def gd(P, Mf, x0, it, alpha, nesterov=False, rec=REC):
    x = x0.copy(); hist = [rec(x)]; tk = 1.0; yv = x.copy()
    for _ in range(it):
        g = P.Q(yv) - P.b
        xn = yv - alpha * Mf(g)
        if nesterov:
            if np.vdot(yv - xn, xn - x).real > 0:     # gradient restart (O'Donoghue-Candes)
                tk = 1.0
            tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            yv = xn + ((tk - 1) / tn) * (xn - x); tk = tn
        else:
            yv = xn
        x = xn
        hist.append(rec(x))
    return hist


def zmshr1_scalars(ss, sm, mm, th1=2e-6, th2=200.0):
    a = 1.0
    for i in range(2001):
        aa = i / 2000
        s_v = aa * ss + (1 - aa) * sm
        vv = aa * aa * ss + 2 * aa * (1 - aa) * sm + (1 - aa) ** 2 * mm
        if s_v > 0 and s_v / ss >= th1 and vv / s_v <= th2:
            a = aa; break
    s_v = a * ss + (1 - a) * sm
    vv = a * a * ss + 2 * a * (1 - a) * sm + (1 - a) ** 2 * mm
    tau = ss / s_v - np.sqrt(max((ss / s_v) ** 2 - ss / vv, 0.0))
    if not (np.isfinite(tau) and tau > 0):
        return 1.0, 1.0
    return a, tau


def zmshr1m(P, Mf, Minv, x0, it, alpha, grad, per_set=False, rec=REC):
    """ZMSHR1 in metric M (p2np.rs Zmshr1M / Zmshr1MSet): g scaled by alpha; per_set: (a, tau, c) per set."""
    x = x0.copy(); hist = [rec(x)]; xprev = gprev = None
    for k in range(it):
        g = alpha * grad(x)
        if k == 0:
            pg = Mf(g)
        else:
            s = x - xprev; m = g - gprev
            ms = Minv(s); mm = Mf(m)
            ax = (1, 2) if per_set else None
            def dot(a, b):
                return np.real((np.conj(a) * b).sum(axis=ax)) if per_set else np.vdot(a, b).real
            ss, sm, mmm = dot(s, ms), dot(s, m), dot(m, mm)
            if per_set:
                at = [zmshr1_scalars(ss[t], sm[t], mmm[t]) if ss[t] > 0 else (1.0, 1.0) for t in range(T)]
                a = np.array([v[0] for v in at])[:, None, None]; tau = np.array([v[1] for v in at])[:, None, None]
            else:
                a, tau = zmshr1_scalars(ss, sm, mmm)
            w = (1 - tau * a) * s - tau * (1 - a) * mm
            z = a * ms + (1 - a) * m
            wz, wg = dot(w, z), dot(w, g)
            ww, zz = dot(w, w), dot(z, z)
            ok = wz > 1e-8 * np.sqrt(ww * zz)
            c = np.where(ok, wg / np.where(ok, wz, 1.0), 0.0)
            if per_set:
                c = c[:, None, None]
            pg = tau * Mf(g) + c * w
        xprev, gprev = x, g
        x = x - pg
        hist.append(rec(x))
    return hist


def first_below(errs, thr):
    for i, e in enumerate(errs):
        if e <= thr:
            return i
    return None


def summarise(errs):
    return {"to_1e-1": first_below(errs, 1e-1), "to_1e-2": first_below(errs, 1e-2), "to_1e-3": first_below(errs, 1e-3),
            "at10": float(errs[min(10, len(errs) - 1)]), "at40": float(errs[min(40, len(errs) - 1)]),
            "final": float(errs[-1])}


def make_minv(P, mname):
    """inverse of each metric (needed by the ZMSHR1-M secant products)"""
    lam_sh = P.spec(P.W.mean(0)); nt = P.W.sum((1, 2)) / P.W.sum() * T
    if mname == "I":
        return lambda g: g.copy()
    if mname == "Cshared":
        return lambda g: ifft2(fft2(g) * lam_sh[None])
    if mname == "Cshared+n_t":
        return lambda g: ifft2(fft2(g) * (nt[:, None, None] * lam_sh[None]))
    if mname == "Cperset":
        ls = np.array([P.spec(w) for w in P.W]); return lambda g: ifft2(fft2(g) * ls)
    sq = np.sqrt(nt)[:, None, None]
    om = 2 - 2 * np.cos(2 * np.pi * np.arange(T) / T); den = lam_sh[None] + (P.mu * om[:, None, None] + P.eps)
    def Minv(g):
        h = np.fft.fft(fft2(g * sq), axis=0) * den
        return ifft2(np.fft.ifft(h, axis=0)) * sq
    return Minv


def coupled_problem(spread, mu_f, **kw):
    P = Problem(spread, mu=0.0, **kw)
    P.bulk = float(np.median(P.spec(P.W.mean(0))))
    P.mu = mu_f * P.bulk                     # temporal coupling in units of the bulk (median) k-space curvature
    P.L = P.power(lambda v: P.Q(v, count=False), 60)
    return P


def recorder(P, xs):
    nrm_e = np.sqrt(np.vdot(xs, P.Q(xs, count=False)).real)
    def rec(x):
        e = x - xs
        return (rel(x, xs), rel(x, P.xt), np.sqrt(max(np.vdot(e, P.Q(e, count=False)).real, 0)) / nrm_e)
    return rec


def E1(spread, mu_f, tag):
    P = coupled_problem(spread, mu_f)
    Ms = P.metrics()
    xs = pcg(P, Ms["K(omega,k)"], np.zeros_like(P.xt), 800)[-1]
    xs = pcg(P, Ms["K(omega,k)"], xs, 200)[-1]
    res = {"spread": spread, "mu": P.mu, "bulk": P.bulk, "L": P.L, "truth_err_of_xstar": rel(xs, P.xt),
           "n_t": P.n.tolist(), "runs": {}}
    x0 = np.zeros_like(P.xt)
    grad = lambda x: P.Q(x) - P.b
    rec = recorder(P, xs)
    for mname, Mf in Ms.items():
        lam = P.power(lambda v: Mf(P.Q(v, count=False)), 80)
        alpha = 1.0 / lam
        Minv = make_minv(P, mname)
        for aname, fn in [("GD", lambda: gd(P, Mf, x0, MAXIT, alpha, rec=rec)),
                          ("Nesterov", lambda: gd(P, Mf, x0, MAXIT, alpha, nesterov=True, rec=rec)),
                          ("ZMSHR1-M", lambda: zmshr1m(P, Mf, Minv, x0, MAXIT, alpha, grad, rec=rec)),
                          ("PCG", lambda: pcg(P, Mf, x0, MAXIT, rec=rec)),
                          ("PCG per-set", lambda: pcg_sets(P, Mf, x0, MAXIT, rec=rec))]:
            h = fn()
            errs = [v[0] for v in h]; terr = [v[1] for v in h]; eerr = [v[2] for v in h]
            r = {"lam_max(MQ)": lam, **summarise(errs), "energy_to_1e-2": first_below(eerr, 1e-2),
                 "energy_at20": eerr[min(20, len(eerr) - 1)],
                 "truth_at10": terr[min(10, len(terr) - 1)], "truth_at40": terr[min(40, len(terr) - 1)]}
            res["runs"][f"{aname}|{mname}"] = r
            print(f"[E1 {tag}] {aname:9s} {mname:12s} lamMQ {lam:9.3g}  1e-1:{r['to_1e-1']} 1e-2:{r['to_1e-2']}"
                  f" 1e-3:{r['to_1e-3']} @40 {r['at40']:.2e} energy@20 {r['energy_at20']:.2e}", flush=True)
    return res


def E2(spread):
    """uncoupled sets (separable H, ridge prior): per-set vs global ZMSHR1 scalars, per-set step via n_t"""
    P = Problem(spread, mu=0.0, eps_rel=1e-3)
    Ms = P.metrics()
    xs = pcg(P, Ms["Cperset"], np.zeros_like(P.xt), 1000)[-1]
    grad = lambda x: P.Q(x) - P.b
    rec = recorder(P, xs)
    out = {"spread": spread, "eps": P.eps, "runs": {}}
    x0 = np.zeros_like(P.xt)
    for mname in ("Cshared", "Cshared+n_t", "Cperset"):
        Mf = Ms[mname]; Minv = make_minv(P, mname)
        lam = P.power(lambda v: Mf(P.Q(v, count=False)), 80); alpha = 1 / lam
        for name, fn in [("ZMSHR1-M global", lambda: zmshr1m(P, Mf, Minv, x0, MAXIT, alpha, grad, rec=rec)),
                         ("ZMSHR1-M per-set", lambda: zmshr1m(P, Mf, Minv, x0, MAXIT, alpha, grad, per_set=True, rec=rec)),
                         ("GD", lambda: gd(P, Mf, x0, MAXIT, alpha, rec=rec)),
                         ("PCG", lambda: pcg(P, Mf, x0, MAXIT, rec=rec)),
                         ("PCG per-set", lambda: pcg_sets(P, Mf, x0, MAXIT, rec=rec))]:
            h = fn(); errs = [v[0] for v in h]
            out["runs"][f"{name}|{mname}"] = {"lam_max(MQ)": lam, **summarise(errs)}
            print(f"[E2 s{spread}] {name:18s}|{mname:12s} {out['runs'][f'{name}|{mname}']}", flush=True)
    return out


# ------------------------------------------------------------------ l1 / smoothed
def haar_fwd(x, lev=3):
    c = x.copy()
    n = N
    for _ in range(lev):
        a = c[..., :n, :n]
        lo = (a[..., 0::2, :] + a[..., 1::2, :]) / np.sqrt(2); hi = (a[..., 0::2, :] - a[..., 1::2, :]) / np.sqrt(2)
        a = np.concatenate([lo, hi], -2)
        lo = (a[..., :, 0::2] + a[..., :, 1::2]) / np.sqrt(2); hi = (a[..., :, 0::2] - a[..., :, 1::2]) / np.sqrt(2)
        c[..., :n, :n] = np.concatenate([lo, hi], -1)
        n //= 2
    return c


def haar_inv(c, lev=3):
    x = c.copy()
    n = N >> (lev - 1)
    for _ in range(lev):
        a = x[..., :n, :n].copy(); h = n // 2
        lo, hi = a[..., :, :h], a[..., :, h:]
        b = np.empty_like(a); b[..., :, 0::2] = (lo + hi) / np.sqrt(2); b[..., :, 1::2] = (lo - hi) / np.sqrt(2)
        lo, hi = b[..., :h, :], b[..., h:, :]
        a[..., 0::2, :] = (lo + hi) / np.sqrt(2); a[..., 1::2, :] = (lo - hi) / np.sqrt(2)
        x[..., :n, :n] = a
        n *= 2
    return x


COARSE = np.zeros((N, N), bool); COARSE[: N >> 3, : N >> 3] = True


def subband_ids(lev=3):
    ids = np.zeros((N, N), int); n = N; j = 1
    for _ in range(lev):
        h = n // 2
        ids[:h, h:n] = j; ids[h:n, :h] = j + 1; ids[h:n, h:n] = j + 2; j += 3; n = h
    ids[COARSE] = 0
    return ids


SB = subband_ids()


def soft(c, thr):
    m = np.abs(c)
    out = c * np.maximum(1 - thr / np.maximum(m, 1e-300), 0)
    return np.where(COARSE, c, out)


def E3(spread, mu_f):
    P = coupled_problem(spread, mu_f)
    Ms = P.metrics()
    bulk = P.bulk
    x0 = np.zeros_like(P.xt)
    REL = lambda ref: (lambda x: rel(x, ref))

    def fista_I(lam, it, rec=None):
        x = x0.copy(); yv = x.copy(); tk = 1.0; a = 1 / P.L; hist = [rec(x)] if rec else None
        for _ in range(it):
            g = P.Q(yv) - P.b
            xn = haar_inv(soft(haar_fwd(yv - a * g), a * lam))
            if np.vdot(yv - xn, xn - x).real > 0:
                tk = 1.0
            tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2; yv = xn + ((tk - 1) / tn) * (xn - x); tk = tn; x = xn
            if rec:
                hist.append(rec(x))
        return hist if rec else x

    def admm(lam, Mrho, rho, n_in, cost, rec=REL(P.xt)):
        x = x0.copy(); Qx = np.zeros_like(x); v = haar_fwd(x); u = np.zeros_like(v); hist = [rec(x)]
        P.nAHA = 0
        while P.nAHA < cost:
            rhs = P.b + rho * haar_inv(v - u)
            r = rhs - (Qx + rho * x); z = Mrho(r); p = z.copy(); rz = np.vdot(r, z).real
            for _ in range(n_in):
                Qp = P.Q(p); Ap = Qp + rho * p; a = rz / np.vdot(p, Ap).real
                x = x + a * p; Qx = Qx + a * Qp; r = r - a * Ap; z = Mrho(r); rz2 = np.vdot(r, z).real
                p = z + (rz2 / rz) * p; rz = rz2
                hist.append(rec(x))
            c = haar_fwd(x)
            v = soft(c + u, lam / rho); u = u + c - v
        return hist, x

    def pg_residual(x, lam):
        """prox-gradient fixed-point residual ||x - prox(x - g/L)|| / ||x|| (optimality of the l1 problem)"""
        g = P.Q(x, count=False) - P.b
        return np.linalg.norm(x - haar_inv(soft(haar_fwd(x - g / P.L), lam / P.L))) / np.linalg.norm(x)

    scale = np.abs(np.where(COARSE, 0, haar_fwd(P.b))).max()
    Mr1 = lambda g: P.prior_aware(g, 1.0 * bulk)
    terr = {}
    for f in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        lam = scale * f
        terr[f] = admm(lam, Mr1, bulk, 8, 600)[0][-1]
        print(f"[E3] lam {f:g}*max|Wb|: truth err {terr[f]:.4f}", flush=True)
    fbest = min(terr, key=terr.get); lam = scale * fbest
    _, xs_l1 = admm(lam, Mr1, bulk, 8, 6000 if not QUICK else 2000)
    xs_f = fista_I(lam, 8000 if not QUICK else 2000)
    out = {"spread": spread, "lam_factor": fbest, "truth_err_grid": terr, "truth_err_xs_l1": rel(xs_l1, P.xt),
           "ref_check": {"admm_vs_fista8000": rel(xs_f, xs_l1), "pg_res_admm": pg_residual(xs_l1, lam),
                         "pg_res_fista": pg_residual(xs_f, lam)}, "runs": {}}
    print(f"[E3] refs: {out['ref_check']}", flush=True)

    def egrad(x, gam):
        c = haar_fwd(x); return haar_inv(np.where(COARSE, 0, (c - soft(c, lam * gam)) / gam))

    def smooth_solver(Mf, gam, it, kind="ncg", m=3, rec=None):
        x = x0.copy(); gq = -P.b.copy(); g = gq + egrad(x, gam)
        hist = [rec(x)] if rec else None; d = None; gprev = None; Mg_prev = None; S_, Y_ = [], []
        nprior = 0
        for k in range(it):
            if kind == "ncg":
                Mg = Mf(g)
                if d is None:
                    d = -Mg
                else:
                    beta = max(0.0, np.vdot(g - gprev, Mg).real / np.vdot(gprev, Mg_prev).real)
                    d = -Mg + beta * d
                    if np.vdot(d, g).real >= 0:
                        d = -Mg
                Mg_prev = Mg
            else:
                q = g.copy(); al = []
                for s_, y_ in reversed(list(zip(S_, Y_))):
                    r_ = 1 / np.vdot(y_, s_).real; a_ = r_ * np.vdot(s_, q).real; al.append((a_, r_, s_, y_)); q = q - a_ * y_
                ck = np.vdot(S_[-1], Y_[-1]).real / np.vdot(Y_[-1], Mf(Y_[-1])).real if S_ else 1.0
                z = ck * Mf(q)
                for a_, r_, s_, y_ in reversed(al):
                    b_ = r_ * np.vdot(y_, z).real; z = z + (a_ - b_) * s_
                d = -z
                if np.vdot(d, g).real >= 0:
                    d = -Mf(g)
            Qd = P.Q(d)
            gd0 = np.vdot(gq, d).real; dQd = np.vdot(d, Qd).real; cd = haar_fwd(d); cx = haar_fwd(x)

            def dphi(a):
                c = cx + a * cd
                ge = np.where(COARSE, 0, (c - soft(c, lam * gam)) / gam)
                return gd0 + a * dQd + np.vdot(ge, cd).real
            lo, hi = 0.0, 1.0; nprior += 1
            while dphi(hi) < 0:
                lo, hi = hi, 2 * hi; nprior += 1
            for _ in range(50):
                mid = 0.5 * (lo + hi); nprior += 1
                if dphi(mid) < 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-9 * hi:
                    break
            a = 0.5 * (lo + hi)
            xn = x + a * d; gqn = gq + a * Qd; gn = gqn + egrad(xn, gam)
            if kind != "ncg":
                if np.vdot(xn - x, gn - g).real > 0:
                    S_.append(xn - x); Y_.append(gn - g)
                if len(S_) > m:
                    S_.pop(0); Y_.pop(0)
            gprev = g; x, gq, g = xn, gqn, gn
            if rec:
                hist.append(rec(x))
        return (hist if rec else x), nprior

    def fista_diag(it, rec):
        nb = SB.max() + 1
        d = np.zeros((T, nb))
        for t in range(T):
            Bn = np.zeros((nb, nb))
            for j in range(nb):
                for jj in range(j, nb):
                    v = np.random.default_rng(10 + j).standard_normal((N, N)) * (SB == jj) + 0j
                    v /= np.linalg.norm(v); lamb = 0.0
                    for _ in range(25):
                        hx = (np.conj(P.S) * ifft2(P.W[t] * fft2(P.S * haar_inv(v)))).sum(0)
                        w = haar_fwd(hx) * (SB == j)
                        hx2 = (np.conj(P.S) * ifft2(P.W[t] * fft2(P.S * haar_inv(w)))).sum(0)
                        w2 = haar_fwd(hx2) * (SB == jj)
                        lamb = np.sqrt(np.linalg.norm(w2))
                        v = w2 / max(np.linalg.norm(w2), 1e-300)
                    Bn[j, jj] = Bn[jj, j] = lamb * 1.02
            d[t] = Bn.sum(1) + 4 * P.mu + P.eps
        D = d[:, SB]
        c = np.zeros_like(P.xt); yv = c.copy(); tk = 1.0; hist = [rec(x0)]
        for _ in range(it):
            g = haar_fwd(P.Q(haar_inv(yv)) - P.b)
            cn = soft(yv - g / D, lam / D)
            if np.vdot(yv - cn, cn - c).real > 0:
                tk = 1.0
            tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2; yv = cn + ((tk - 1) / tn) * (cn - c); tk = tn; c = cn
            hist.append(rec(haar_inv(c)))
        return hist, {"d_coarse": float(d[:, 0].mean()), "d_finest_mean": float(d[:, 1:4].mean()),
                      "L": P.L, "sum_offdiag_share": None}

    IT = MAXIT
    rl1 = REL(xs_l1)
    out["runs"]["FISTA-I (exact prox)"] = {**summarise(fista_I(lam, IT, rec=rl1)), "target": "l1"}
    h, info = fista_diag(IT, rl1)
    out["runs"]["FISTA-subband x set diag (exact prox)"] = {**summarise(h), "target": "l1", **info}
    for rho_f in (0.3, 1.0, 3.0):
        for n_in in (3, 8):
            rho = rho_f * bulk
            Mrho = lambda g, rho=rho: P.prior_aware(g, rho)
            h, _ = admm(lam, Mrho, rho, n_in, IT, rec=rl1)
            out["runs"][f"ADMM-PCG rho={rho_f}*bulk inner={n_in}"] = {**summarise(h), "target": "l1"}
    for gfac in (1.0, 0.1):
        gam = gfac / bulk
        key = f"gamma={gfac}/bulk"
        xs_g, _ = smooth_solver(Ms["K(omega,k)"], gam, 2000 if not QUICK else 600, kind="lbfgs")
        xs_g2, _ = smooth_solver(Ms["K(omega,k)"], gam, 300, kind="ncg")
        out[f"smooth {key}"] = {"dist_to_l1_min": rel(xs_g, xs_l1), "truth_err": rel(xs_g, P.xt),
                                "ref_check_ncg300_vs_lbfgs": rel(xs_g2, xs_g)}
        print(f"[E3] smooth {key}: {out[f'smooth {key}']}", flush=True)
        rg = REL(xs_g)
        for mname in ("I", "Cshared", "K(omega,k)"):
            for kind in ("ncg", "lbfgs"):
                hh, npri = smooth_solver(Ms[mname], gam, IT, kind=kind, rec=lambda x: (rel(x, xs_g), rel(x, xs_l1)))
                out["runs"][f"{kind.upper()}|{mname} smooth {key}"] = {**summarise([v[0] for v in hh]),
                                                                       "target": f"smooth {key}", "prior_evals_per_it": npri / IT,
                                                                       "dist_l1_at40": hh[min(40, len(hh) - 1)][1]}
        Mf = Ms["Cshared"]; Minv = make_minv(P, "Cshared")
        lamq = P.power(lambda v: Mf(P.Q(v, count=False)), 60)
        lam_sh = P.spec(P.W.mean(0))
        for an, alpha in (("alpha=data", 1 / lamq), ("alpha=data+prior", 1 / (lamq + (1 / lam_sh.min()) / gam))):
            hh = zmshr1m(P, Mf, Minv, x0, IT, alpha, lambda x: P.Q(x) - P.b + egrad(x, gam), rec=rg)
            out["runs"][f"ZMSHR1-M|Cshared {an} smooth {key}"] = {**summarise(hh), "target": f"smooth {key}"}
    for kname, v in out["runs"].items():
        print(f"[E3] {kname:50s} {v}", flush=True)
    return out


def E4():
    """unregularised LS with model error (coil-map phase) and noise; 5 % of the samples held out"""
    P = Problem(1.6, mu=0.0, eps_rel=0.0, heldout=0.05, coil_err=0.15, sigma_rel=0.02)
    Ms = P.metrics()
    x0 = np.zeros_like(P.xt)
    grad = lambda x: P.Q(x) - P.b
    def ho(x):
        r = np.sqrt(P.Wh[:, None]) * fft2(P.S[None] * x[:, None]) - P.yh
        return float(np.vdot(r, r).real)
    rec = lambda x: (rel(x, P.xt), ho(x))
    out = {}
    bulk = float(np.median(P.spec(P.W.mean(0))))
    for muf in (0.3, 1.0, 3.0):
        P.mu = muf * bulk; Mk = P.metrics()["K(omega,k)"]
        hk = pcg(P, Mk, x0, 150, rec=rec)
        te = [v[0] for v in hk]
        out[f"regularised mu={muf}*bulk PCG|K"] = {"truth_err_converged": te[-1], "it_to_within_1pct_of_final":
                                                  next(i for i, e in enumerate(te) if abs(e - te[-1]) <= 0.01 * te[-1])}
        print(f"[E4] reg mu={muf}: {out[f'regularised mu={muf}*bulk PCG|K']}", flush=True)
    P.mu = 0.0; Ms = P.metrics()
    Mf = Ms["Cperset"]; Minv = make_minv(P, "Cperset")
    lam = P.power(lambda v: Mf(P.Q(v, count=False)), 60)
    lamI = P.power(lambda v: P.Q(v, count=False), 60)
    for name, fn in [("PCG|Cperset", lambda: pcg(P, Mf, x0, 200, rec=rec)),
                     ("ZMSHR1-M|Cperset", lambda: zmshr1m(P, Mf, Minv, x0, 200, 1 / lam, grad, rec=rec)),
                     ("GD|Cperset", lambda: gd(P, Mf, x0, 200, 1 / lam, rec=rec)),
                     ("PCG|I", lambda: pcg(P, Ms["I"], x0, 200, rec=rec)),
                     ("GD|I", lambda: gd(P, Ms["I"], x0, 200, 1 / lamI, rec=rec))]:
        h = fn()
        te = [v[0] for v in h]; he = [v[1] for v in h]
        ib, ih = int(np.argmin(te)), int(np.argmin(he))
        out[name] = {"best_truth_it": ib, "best_truth_err": te[ib], "heldout_min_it": ih, "truth_err_at_heldout_min": te[ih],
                     "truth_err_at_200": te[-1], "truth_curve(1,2,5,10,20,40,80,120,200)": [te[i] for i in (1, 2, 5, 10, 20, 40, 80, 120, 200)]}
        print(f"[E4] {name:18s} {out[name]}", flush=True)
    return out


def E5(spread=1.6, mu_f=1.0, lam_factor=3e-5):
    """PDHG / Condat-Vu with the data term dualised and a k-space (dual) preconditioner Sigma = c / w (Ong-Uecker-Lustig
    2020 idea, single-channel): DCW-like conditioning without changing the estimator; Euclidean (exact) wavelet prox."""
    P = coupled_problem(spread, mu_f)
    scale = np.abs(np.where(COARSE, 0, haar_fwd(P.b))).max(); lam = lam_factor * scale
    x0 = np.zeros_like(P.xt)
    # reference: FISTA-I 8000 (agreed with ADMM to 2e-13 in E3)
    x = x0.copy(); yv = x.copy(); tk = 1.0; a = 1 / P.L
    for _ in range(8000 if not QUICK else 2000):
        g = P.Q(yv, count=False) - P.b
        xn = haar_inv(soft(haar_fwd(yv - a * g), a * lam))
        if np.vdot(yv - xn, xn - x).real > 0:
            tk = 1.0
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2; yv = xn + ((tk - 1) / tn) * (xn - x); tk = tn; x = xn
    xs = x
    Lh = 4 * P.mu + P.eps
    wpos = (P.W > 0)
    def run(Sig, tau, it):
        x = x0.copy(); u = np.zeros_like(P.y); hist = [rel(x, xs)]
        for _ in range(it):
            g = P.AH(u) + P.mu * P.Lt(x) + P.eps * x
            xn = haar_inv(soft(haar_fwd(x - tau * g), tau * lam))
            xb = 2 * xn - x
            v = u + Sig * P.A(xb)
            u = (v - Sig * P.y) / (1 + Sig)
            x = xn
            hist.append(rel(x, xs))
        return hist
    def normSA(Sig):
        v = np.random.default_rng(4).standard_normal((T, N, N)) + 0j; lamb = 0
        for _ in range(60):
            w = P.AH(Sig * P.A(v)); lamb = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
        return lamb
    out = {"lam_factor": lam_factor, "runs": {}}
    for c in (0.1, 0.3, 1.0, 3.0):
        Sig = np.where(wpos, c / np.maximum(P.W, 1), 0.0)[:, None] * np.ones((1, NC, 1, 1))
        nrm = normSA(Sig); tau = 0.99 / (nrm + Lh / 2)
        h = run(Sig, tau, MAXIT)
        out["runs"][f"PDHG k-space precond Sigma={c}/w"] = {**summarise(h), "tau": tau}
        print(f"[E5] Sigma={c}/w tau {tau:.3g}: {out['runs'][f'PDHG k-space precond Sigma={c}/w']}", flush=True)
    LA = P.power(lambda v: P.AHA(v, count=False), 60)
    for c in (0.01, 0.1, 1.0):
        sig = c / np.sqrt(LA); tau = 0.99 / (sig * LA + Lh / 2)
        h = run(sig, tau, MAXIT)
        out["runs"][f"PDHG scalar sigma={c}/sqrt(L)"] = {**summarise(h), "tau": tau}
        print(f"[E5] scalar sigma={c}/sqrt(L) tau {tau:.3g}: {out['runs'][f'PDHG scalar sigma={c}/sqrt(L)']}", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["E1", "E2", "E3", "E4"]
    if "E1" in which:
        for spread in (1.6, 16.0):
            res[f"E1_spread{spread}"] = E1(spread, 1.0, f"s{spread}")
    if "E2" in which:
        res["E2_spread16"] = E2(16.0)
    if "E4" in which:
        res["E4"] = E4()
    if "E3" in which:
        res["E3_spread1.6"] = E3(1.6, 1.0)
    if "E5" in which:
        res["E5"] = E5()
    res["seconds"] = time.time() - t0
    (OUT / f"strategy_toy_{'_'.join(which)}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"done in {res['seconds']:.0f} s")
