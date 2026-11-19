"""CPU toy: cheap coil dimension in iterative reconstruction (research/coil_cost_reduction.md §7).

    rustlib/.venv/bin/python -I research/experiments/coil_sketch/toy.py [experiment ...]

Problem (out/slice.npz from spectrum.py): an axial 2 mm slice of ph0001 (256 × 144 after padding), the 30 global
energy-ordered virtual coils of the whitened true maps (signal-weighted Gram), two encodes (encode 1 carries the
through-plane velocity phase π v / venc), 48 golden-angle radial spokes per encode (different spokes per encode),
white noise. Exact NDFT model; the normal operator per coil is the exact Toeplitz embedding (2× grid), so one coil
application = one forward + one adjoint of one coil = 2 FFTs on the 2× grid.

Objective F = ½ Σ_c ‖A_c x − y_c‖² + λ Σ_detail env_γ(|W x|) (orthonormal Haar, 4 levels, Moreau envelope of ℓ1;
γ = 0.1 / bulk data curvature, convergence_strategy.md §0.4). Metric P = D^{-1/2} F⁻¹ (λ_circ + κ)⁻¹ F D^{-1/2}
(the circulant metric of precond.rs plus a prior-aware floor κ). Also a PnP form: x ← soft(x − α P ∇f) (§7.5).

Cost unit: one full gradient = nc coil applications. Every run is cached in out/runs/<name>.npz (resumable: rerun skips
finished runs; delete a file to redo it).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft as sfft

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
WK = int(__import__("os").environ.get("TOY_WORKERS", "4"))
NSP = 48
BUDGETS = (5, 10, 20, 40, 60, 120)
KAPPA = float(__import__("os").environ.get("TOY_KAPPA", "0.1"))  # prior-aware metric floor, × (1/γ)/D_med
FRAME = 14
GOLD = np.pi * (np.sqrt(5) - 1) / 2


# ---------------------------------------------------------------- problem
def haar_fwd(x, lev=4):
    y = x.copy()
    nx, ny = y.shape
    for _ in range(lev):
        b = y[:nx, :ny]
        a = (b[0::2] + b[1::2]) / np.sqrt(2)
        d = (b[0::2] - b[1::2]) / np.sqrt(2)
        b = np.concatenate([a, d], 0)
        a = (b[:, 0::2] + b[:, 1::2]) / np.sqrt(2)
        d = (b[:, 0::2] - b[:, 1::2]) / np.sqrt(2)
        y[:nx, :ny] = np.concatenate([a, d], 1)
        nx, ny = nx // 2, ny // 2
    return y


def haar_inv(y, lev=4):
    x = y.copy()
    sizes = []
    nx, ny = x.shape
    for _ in range(lev):
        sizes.append((nx, ny))
        nx, ny = nx // 2, ny // 2
    for nx, ny in reversed(sizes):
        b = x[:nx, :ny]
        a, d = b[:, :ny // 2], b[:, ny // 2:]
        c = np.empty_like(b)
        c[:, 0::2] = (a + d) / np.sqrt(2)
        c[:, 1::2] = (a - d) / np.sqrt(2)
        a, d = c[:nx // 2], c[nx // 2:]
        b2 = np.empty_like(c)
        b2[0::2] = (a + d) / np.sqrt(2)
        b2[1::2] = (a - d) / np.sqrt(2)
        x[:nx, :ny] = b2
    return x


class Problem:
    def __init__(self, coils="global", noise=0.03, seed=0):
        z = np.load(OUT / "slice.npz")
        raw = z["raw"]  # (254, 144, 32) whitened
        U = {"global": z["U_sig"], "heart": z["U_heart"], "box": z["U_box"]}[coils.split(":")[0]]
        nvc = int(coils.split(":")[1]) if ":" in coils else 30
        V = raw @ U[:, :nvc].conj()  # (254,144,nv)
        pad = lambda a: np.pad(a, ((1, 1), (0, 0)) + ((0, 0),) * (a.ndim - 2))
        self.maps = np.moveaxis(pad(V), -1, 0).astype(np.complex128)  # (nc, 256, 144)
        self.nc = self.maps.shape[0]
        self.n = self.maps.shape[1:]
        self.rho = pad(z["rho"]).astype(np.float64)
        self.lumen = pad(z["lumen"]) > 0.5
        self.body = pad(z["body"])
        self.heart = pad(z["heart"])
        self.venc = float(z["venc"])
        vz = pad(z["vel"][:, :, 2, FRAME])
        from scipy import ndimage
        # support (the recon's air map / support mask): body dilated by 3 voxels; unknowns outside are fixed at 0
        self.supp = ndimage.binary_dilation(self.body, iterations=3)
        self.vroi = self.lumen & self.heart
        nx, ny = self.n
        X, Y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), indexing="ij")
        phi_bg = 0.6 * X + 0.4 * Y ** 2 - 0.3 * X * Y
        s = self.rho / self.rho.max() * self.supp
        self.truth = [s * np.exp(1j * phi_bg), s * np.exp(1j * (phi_bg + np.pi * vz / self.venc))]
        # trajectory: full radial spokes, k in rad/voxel, golden angle, encode e uses spokes e·NSP ...
        nro = 2 * max(nx, ny)
        kr = np.pi * (np.arange(nro) - nro // 2) / (nro // 2)
        rng = np.random.default_rng(seed)
        self.T, self.b, self.y0n = [], [], []
        ix = np.arange(nx) - nx // 2
        iy = np.arange(ny) - ny // 2
        dx = np.fft.fftfreq(2 * nx, 1 / (2 * nx))  # circular lags
        dy = np.fft.fftfreq(2 * ny, 1 / (2 * ny))
        for e in range(2):
            th = (np.arange(NSP) + e * NSP) * GOLD
            kx = (np.cos(th)[:, None] * kr[None]).ravel()
            ky = (np.sin(th)[:, None] * kr[None]).ravel()
            Ex = np.exp(-1j * np.outer(kx, ix))
            Ey = np.exp(-1j * np.outer(ky, iy))
            Gx = np.exp(1j * np.outer(kx, dx))
            Gy = np.exp(1j * np.outer(ky, dy))
            t = Gx.T @ Gy  # t[d] = Σ_m exp(i k·d), circular layout on the 2n grid
            self.T.append(sfft.fft2(t, workers=WK))
            # data from the 32 raw (whitened) coils + white noise (same realisation for every compression), then
            # compressed: y_v = u_vᴴ y (orthonormal U: noise stays white)
            raw_p = np.moveaxis(pad(raw), -1, 0)
            ys = np.array([np.sum((Ex @ (raw_p[c] * self.truth[e])) * Ey, 1) for c in range(raw_p.shape[0])])
            if e == 0:
                sigma = noise * np.sqrt(np.mean(np.abs(ys) ** 2))
            ys = ys + sigma * (rng.standard_normal(ys.shape) + 1j * rng.standard_normal(ys.shape)) / np.sqrt(2)
            yv = U[:, :nvc].conj().T @ ys
            bs = [self.maps[c].conj() * ((Ex.conj().T * yv[c]) @ Ey.conj()) for c in range(self.nc)]
            self.b.append(np.array(bs))  # per-coil A_cᴴ y_c
            self.sigma = sigma
            self.M = kx.size
        # Toeplitz check on one coil
        x = rng.standard_normal(self.n) + 1j * rng.standard_normal(self.n)
        img = self.maps[0] * x
        yy = np.sum((Ex @ img) * Ey, 1)
        ref = self.maps[0].conj() * ((Ex.conj().T * yy) @ Ey.conj())
        got = self.normal(1, x, self.maps[:1])
        self.toeplitz_err = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        # metric
        self.D = np.sum(np.abs(self.maps) ** 2, 0)
        Dfl = np.maximum(self.D, 1e-3 * self.D[self.body].mean())
        self.Dmh = 1 / np.sqrt(Dfl)
        self.lam_c = []
        for e in range(2):
            t = sfft.ifft2(self.T[e], workers=WK)  # back to lags
            # T. Chan optimal circulant on the n grid from the 2n-grid lags
            c = np.zeros(self.n, complex)
            wx = np.concatenate([(nx - np.arange(nx)) / nx])
            for jx in range(nx):
                for sx, ax in ((jx, (nx - jx) / nx), (jx - nx, jx / nx)):
                    if ax == 0:
                        continue
                    row = t[sx % (2 * nx)]
                    jy = np.arange(ny)
                    c[jx] += ax * ((ny - jy) / ny * row[jy] + jy / ny * row[(jy - ny) % (2 * ny)])
            self.lam_c.append(np.real(sfft.fft2(c, workers=WK)))
        self.Dmed = float(np.median(self.D[self.body]))
        lc = self.lam_c[0]
        self.bulk = float(np.median(lc) * self.Dmed)
        self.gamma = 0.1 / self.bulk
        self.lam = None  # set by set_lambda
        self.kappa = KAPPA * (1 / self.gamma) / self.Dmed
        self.detail = np.ones(self.n, bool)
        self.detail[: nx >> 4, : ny >> 4] = False

    # --- operators
    def normal(self, e, x, maps, w=None):
        """Σ_j w_j m_jᴴ T_e m_j x (maps (J, nx, ny))"""
        nx, ny = self.n
        J = maps.shape[0]
        g = np.zeros((J, 2 * nx, 2 * ny), complex)
        g[:, :nx, :ny] = maps * x
        g = sfft.ifft2(sfft.fft2(g, axes=(1, 2), workers=WK) * self.T[e], axes=(1, 2), workers=WK)[:, :nx, :ny]
        g = maps.conj() * g
        if w is not None:
            g = g * np.asarray(w)[:, None, None]
        return g.sum(0)

    def precond(self, e, r):
        r = r * self.supp
        z = sfft.ifft2(sfft.fft2(self.Dmh * r, workers=WK) / (np.maximum(self.lam_c[e], 1e-3 * self.lam_c[e].max()) + self.kappa), workers=WK)
        return self.supp * self.Dmh * z

    def prec_half(self, e, r):  # B with P = B Bᴴ
        return self.supp * self.Dmh * sfft.ifft2(sfft.fft2(r, workers=WK) / np.sqrt(np.maximum(self.lam_c[e], 1e-3 * self.lam_c[e].max()) + self.kappa), workers=WK)

    def prec_half_h(self, e, r):
        return sfft.ifft2(sfft.fft2(self.Dmh * self.supp * r, workers=WK) / np.sqrt(np.maximum(self.lam_c[e], 1e-3 * self.lam_c[e].max()) + self.kappa), workers=WK)

    # --- prior (Moreau envelope of λ‖detail(Wx)‖₁)
    def prior(self, x):
        w = haar_fwd(x)
        a = np.abs(w)
        lg = self.lam * self.gamma
        h = np.where(a <= lg, a ** 2 / (2 * self.gamma), self.lam * a - self.lam * lg / 2)
        gw = np.where(a <= lg, w / self.gamma, self.lam * w / np.maximum(a, 1e-300))
        gw[~self.detail] = 0
        return float(h[self.detail].sum()), haar_inv(gw)

    def soft(self, x, thr):
        w = haar_fwd(x)
        a = np.abs(w)
        s = np.maximum(1 - thr / np.maximum(a, 1e-300), 0)
        s[~self.detail] = 1
        return haar_inv(w * s) * self.supp


# ---------------------------------------------------------------- coil subsets / sketches
class Sampler:
    """keep the first k coils; the tail is used as: 'sub' a reshuffled group of d coils with weight p (unbiased),
    'rad' s Rademacher combinations (scaled 1/√s; unbiased), 'keep' nothing (biased, deterministic)."""

    def __init__(self, P, k, d, kind="sub", seed=0):
        self.P, self.k, self.d, self.kind = P, k, d, kind
        self.rng = np.random.default_rng(seed)
        self.tail = np.arange(k, P.nc)
        self.queue = []

    def draw(self):
        P, k = self.P, self.k
        keep = P.maps[:k]
        if self.kind == "keep" or self.d == 0 or self.tail.size == 0:
            return keep, np.ones(k), k
        if self.d >= self.tail.size:
            return P.maps, np.ones(P.nc), P.nc
        if self.kind == "sub":
            p = int(np.ceil(self.tail.size / self.d))
            if not self.queue:
                perm = self.rng.permutation(self.tail)
                self.queue = [perm[i * self.d:(i + 1) * self.d] for i in range(p)]
            g = self.queue.pop(0)
            maps = np.concatenate([keep, P.maps[g]])
            w = np.concatenate([np.ones(k), np.full(g.size, float(p))])
            return maps, w, maps.shape[0]
        if self.kind == "rad":
            Phi = self.rng.choice([-1.0, 1.0], size=(self.d, self.tail.size)) / np.sqrt(self.d)
            sk = np.tensordot(Phi, P.maps[self.tail], axes=(1, 0))
            maps = np.concatenate([keep, sk])
            return maps, np.ones(maps.shape[0]), maps.shape[0]
        raise ValueError(self.kind)


# ---------------------------------------------------------------- solvers (one encode; return trace)
def lmax(P, e, maps, w, iters=25, seed=1, with_prior=True):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(P.n) + 1j * rng.standard_normal(P.n)
    lam = 0
    for _ in range(iters):
        v /= np.linalg.norm(v)
        u = P.prec_half(e, v)
        u = P.normal(e, u, maps, w) + (u / P.gamma if with_prior else 0)
        u = P.prec_half_h(e, u)
        lam = np.real(np.vdot(v, u))
        v = u
    return lam


class Tracker:
    def __init__(self, P, e, ref, budget, every=1):
        self.P, self.e, self.ref, self.budget = P, e, ref, budget
        self.cost = 0.0
        self.rows = []
        self.snaps = {}

    def add(self, c):
        self.cost += c / self.P.nc

    def log(self, x):
        r = self.ref
        err = np.linalg.norm(x - r) / np.linalg.norm(r) if r is not None else np.nan
        self.rows.append((self.cost, err))
        for b in BUDGETS:
            if self.cost <= b + 1e-9:
                self.snaps[b] = x
        return self.cost < self.budget

    def out(self, x):
        return dict(trace=np.array(self.rows), x=x)


def grad_f_full(P, e, x):
    return P.normal(e, x, P.maps) - P.b[e].sum(0)


def run_ncg(P, e, tr, x=None, maxit=10 ** 6, tol=0.0):
    x = np.zeros(P.n, complex) if x is None else x.copy()
    gf = grad_f_full(P, e, x) if np.any(x) else -P.b[e].sum(0)
    if np.any(x):
        tr.add(P.nc)
    return ncg_core(P, e, tr, x, gf, lambda d: (P.normal(e, d, P.maps), P.nc), maxit, tol)


def lsearch(P, x, d, dq, gfd):
    """exact line search on φ(t) = ½ t² dq + t gfd + R(x + t d) (data quadratic exact, prior by safeguarded secant
    on the monotone φ'); None if d is not a descent direction"""
    def dphi(t):
        return t * dq + gfd + np.real(np.vdot(P.prior(x + t * d)[1], d))
    t0, f0 = 0.0, dphi(0.0)
    if f0 >= 0:
        return None
    t1 = -f0 / max(dq, 1e-300)  # Newton step on the data curvature with the total slope (> 0)
    f1 = dphi(t1)
    nd = 0
    while f1 < 0 and nd < 60:
        nd += 1
        t0, f0 = t1, f1
        t1 *= 2
        f1 = dphi(t1)
    side, tm = 0, t1
    for _ in range(40):  # Illinois
        tm = t0 - f0 * (t1 - t0) / (f1 - f0)
        if not (t0 < tm < t1):
            tm = 0.5 * (t0 + t1)
        fm = dphi(tm)
        if fm < 0:
            t0, f0 = tm, fm
            if side == -1:
                f1 *= 0.5
            side = -1
        else:
            t1, f1 = tm, fm
            if side == 1:
                f0 *= 0.5
            side = 1
        if abs(fm) <= 1e-9 * abs(gfd) or (t1 - t0) < 1e-12 * t1:
            break
    return tm


class Inner:
    """tracker proxy for inner iterations: charges cost to the outer tracker, logs nothing (only accepted outer
    iterates are scored)"""

    def __init__(self, tr):
        self.tr = tr

    def add(self, c):
        self.tr.add(c)

    def log(self, x):
        return self.tr.cost < self.tr.budget


def model_op(P, e, sampler, circ_tail=0.0):
    maps, w, c = sampler.draw()
    if circ_tail:
        Dt = np.sqrt(np.maximum(P.D - np.sum(np.abs(maps) ** 2, 0), 0))
        lc = np.maximum(P.lam_c[e], 0)
        return lambda dd: (P.normal(e, dd, maps, w) + circ_tail * Dt * sfft.ifft2(lc * sfft.fft2(Dt * dd, workers=WK), workers=WK), c)
    return lambda dd: (P.normal(e, dd, maps, w), c)


def run_lagncg(P, e, tr, sampler, tau=0.05, circ_tail=0.0):
    """one uninterrupted NCG (no restarts) on the lagged gradient g = ∇f(x_a) + Ñ(x − x_a) + ∇R(x): the line search
    and the gradient recursion use the sketch Ñ (cost |S|); the anchor (exact gradient, cost n_c, new sketch) is
    refreshed when ‖x − x_a‖ > τ‖x‖ — recon_p2np --lagged with a coil surrogate, NCG instead of ZMSHR1."""
    x = np.zeros(P.n, complex)
    xa = x.copy()
    gf = -P.b[e].sum(0)
    op = model_op(P, e, sampler, circ_tail)
    g = gf + P.prior(x)[1]
    z = P.precond(e, g)
    d = -z
    gz = np.real(np.vdot(g, z))
    tr.log(x)
    while tr.cost < tr.budget:
        q, c = op(d)
        tr.add(c)
        t = lsearch(P, x, d, np.real(np.vdot(d, q)), np.real(np.vdot(gf, d)))
        if t is None:
            d = -z
            continue
        x = x + t * d
        gf = gf + t * q
        if np.linalg.norm(x - xa) > tau * np.linalg.norm(x):
            gf = grad_f_full(P, e, x)
            tr.add(P.nc)
            xa = x.copy()
            op = model_op(P, e, sampler, circ_tail)
        gn = gf + P.prior(x)[1]
        zn = P.precond(e, gn)
        gzn = np.real(np.vdot(gn, zn))
        beta = max(0.0, (gzn - np.real(np.vdot(g, zn))) / gz)
        g, z, gz = gn, zn, gzn
        d = -z + beta * d
        if not tr.log(x):
            break
    return x


def run_ihs_ls(P, e, tr, sampler, inner=5, flex=False, circ_tail=0.0):
    """coil sketch as a *preconditioner* of an outer NCG on the true F: the inner K NCG steps on the sketched model
    give a direction Δ ≈ −(Ñ + H_R)⁻¹ ∇F; the outer takes an exact line search on F along d (q = N d, cost n_c, which
    also yields the next exact gradient g_f + t q: no extra cost); flex: flexible PR combination with the previous
    outer direction. circ_tail β > 0: model Ñ = N_keep + β D_tail^{1/2} C D_tail^{1/2} (coil-free circulant tail)."""
    x = np.zeros(P.n, complex)
    gf = -P.b[e].sum(0)
    tr.log(x)
    dprev = gprev = sprev = None
    while tr.cost < tr.budget:
        maps, w, c = sampler.draw()
        if circ_tail:
            Dt = np.sqrt(np.maximum(P.D - np.sum(np.abs(maps) ** 2, 0), 0))
            lc = np.maximum(P.lam_c[e], 0)
            model = lambda dd, maps=maps, w=w, c=c, Dt=Dt, lc=lc: (P.normal(e, dd, maps, w) + circ_tail * Dt * sfft.ifft2(lc * sfft.fft2(Dt * dd, workers=WK), workers=WK), c)
        else:
            model = lambda dd, maps=maps, w=w, c=c: (P.normal(e, dd, maps, w), c)
        xin, _ = ncg_core(P, e, Inner(tr), x, gf.copy(), model, inner)
        s = xin - x  # ≈ −K̃⁻¹ g (the "preconditioned gradient", negated)
        g = gf + P.prior(x)[1]
        d = s
        if flex and dprev is not None:
            beta = max(0.0, np.real(np.vdot(g - gprev, -s)) / max(np.real(np.vdot(gprev, -sprev)), 1e-300))
            d = s + beta * dprev
            if np.real(np.vdot(g, d)) >= 0:
                d = s
        q = P.normal(e, d, P.maps)
        tr.add(P.nc)
        t = lsearch(P, x, d, np.real(np.vdot(d, q)), np.real(np.vdot(gf, d)))
        if t is None:
            dprev = None
            continue
        x = x + t * d
        gf = gf + t * q
        dprev, gprev, sprev = d, g, s
        if not tr.log(x):
            break
    return x


def ncg_core(P, e, tr, x, gf, napply, maxit, tol=0.0, stop=None):
    """NCG (PR+) in the metric P on q(x) data model (gradient gf maintained by gf += t N d) + prior."""
    _, gr = P.prior(x)
    g = gf + gr
    z = P.precond(e, g)
    d = -z
    gz = np.real(np.vdot(g, z))
    gz0 = gz
    tr.log(x)
    for it in range(maxit):
        if gz <= (tol ** 2) * gz0:
            break
        q, c = napply(d)
        tr.add(c)
        dq = np.real(np.vdot(d, q))
        gfd = np.real(np.vdot(gf, d))
        t = lsearch(P, x, d, dq, gfd)
        if t is None:  # not a descent direction (PR+ safeguard): restart along −z next iteration
            d = -z
            if np.real(np.vdot(g, z)) <= 0:
                break
            continue
        x = x + t * d
        gf = gf + t * q
        _, gr = P.prior(x)
        gn = gf + gr
        zn = P.precond(e, gn)
        gzn = np.real(np.vdot(gn, zn))
        beta = max(0.0, (gzn - np.real(np.vdot(g, zn))) / gz)
        g, z, gz = gn, zn, gzn
        d = -z + beta * d
        if not tr.log(x) or (stop is not None and stop(x)):
            break
    return x, gf


def run_svrg(P, e, tr, sampler, alpha, inner, pnp_thr=None):
    x = np.zeros(P.n, complex)
    tr.log(x)
    while True:
        xa = x.copy()
        mu = grad_f_full(P, e, xa) if np.any(xa) else -P.b[e].sum(0)
        if np.any(xa):
            tr.add(P.nc)
        for _ in range(inner):
            maps, w, c = sampler.draw()
            v = mu + P.normal(e, x - xa, maps, w)
            tr.add(c)
            if pnp_thr is None:
                x = x - alpha * P.precond(e, v + P.prior(x)[1])
            else:
                x = P.soft(x - alpha * P.precond(e, v), pnp_thr)
            if not tr.log(x):
                return x


def run_ihs(P, e, tr, sampler, inner=5, tau=None):
    """iterative Hessian sketch / lagged gradient with a coil-sketch surrogate: exact gradient μ at the anchor, inner
    NCG on μ + Ñ_S (x − x_a) + ∇R with one sketch per epoch (exact secant pairs inside). tau: refresh the anchor when
    ‖x − x_a‖ > τ‖x‖ (recon_p2np --lagged rule) instead of after `inner` iterations (then inner = cap)."""
    x = np.zeros(P.n, complex)
    first = True
    while tr.cost < tr.budget:
        xa = x.copy()
        if first:
            mu = -P.b[e].sum(0)
            first = False
        else:
            mu = grad_f_full(P, e, xa)
            tr.add(P.nc)
        maps, w, c = sampler.draw()
        stop = None
        if tau is not None:
            stop = lambda xx, xa=xa: np.linalg.norm(xx - xa) > tau * np.linalg.norm(xx)
        x, _ = ncg_core(P, e, tr, x, mu.copy(), lambda d: (P.normal(e, d, maps, w), c), inner, stop=stop)
    return x


# bsum for sampled gradients: Σ_keep b_c + p Σ_group b_c — the sampler must expose the group
def _sub_draw_with_b(self):
    P, k = self.P, self.k
    if self.kind != "sub" or self.d == 0 or self.d >= self.tail.size:
        maps, w, c = Sampler.draw(self)
        self.last_idx = np.arange(maps.shape[0]) if maps.shape[0] in (k, P.nc) else None
        self.last_w = w
        return maps, w, c
    p = int(np.ceil(self.tail.size / self.d))
    if not self.queue:
        perm = self.rng.permutation(self.tail)
        self.queue = [perm[i * self.d:(i + 1) * self.d] for i in range(p)]
    g = self.queue.pop(0)
    self.last_idx = np.concatenate([np.arange(k), g])
    self.last_w = np.concatenate([np.ones(k), np.full(g.size, float(p))])
    return P.maps[self.last_idx], self.last_w, self.last_idx.size


def _bsum(self, e, maps, w):
    if self.kind == "rad":
        raise ValueError("rad sketch has no unbiased data term without sketching the data: use SVRG/IHS forms")
    return np.tensordot(self.last_w, self.P.b[e][self.last_idx], axes=(0, 0))


Sampler.draw_b = _sub_draw_with_b
Sampler.bsum = _bsum


def run_sgd(P, e, tr, sampler, alpha, anneal=None, full_last=0.0, pnp_thr=None, then_ncg_at=None, x0=None):
    """keep+draw stochastic gradient (unbiased), optional anneal [(cost_from, Sampler|None)], full_last cost units of
    full gradients, then_ncg_at: switch to full NCG once this cost is spent (pyramid → curvature)."""
    x = np.zeros(P.n, complex) if x0 is None else x0.copy()
    tr.log(x)
    cur = sampler
    while tr.cost < tr.budget:
        if then_ncg_at is not None and tr.cost >= then_ncg_at:
            gf = grad_f_full(P, e, x)
            tr.add(P.nc)
            x, _ = ncg_core(P, e, tr, x, gf, lambda d: (P.normal(e, d, P.maps), P.nc), 10 ** 6)
            return x
        if anneal:
            for c0, s in anneal:
                if tr.cost >= c0:
                    cur = s
        if (full_last and tr.cost >= tr.budget - full_last) or cur is None:
            gf = grad_f_full(P, e, x)
            c = P.nc
        else:
            maps, w, c = cur.draw_b()
            gf = P.normal(e, x, maps, w) - cur.bsum(e, maps, w)
        tr.add(c)
        if pnp_thr is None:
            x = x - alpha * P.precond(e, gf + P.prior(x)[1])
        else:
            x = P.soft(x - alpha * P.precond(e, gf), pnp_thr)
        if not tr.log(x):
            break
    return x


# ---------------------------------------------------------------- experiment driver
def metrics(P, xs, refs):
    """per-encode rel. error to the fixed point, heart-region error, velocity (phase-difference) RMS vs fixed point
    and vs truth in the heart lumen ROI [cm/s]"""
    e0 = np.linalg.norm(xs[0] - refs[0]) / np.linalg.norm(refs[0])
    e1 = np.linalg.norm(xs[1] - refs[1]) / np.linalg.norm(refs[1])
    h = P.heart
    eh = np.linalg.norm((xs[0] - refs[0])[h]) / np.linalg.norm(refs[0][h])
    m = P.vroi
    pd = np.angle(xs[1][m] * np.conj(xs[0][m]))
    pr = np.angle(refs[1][m] * np.conj(refs[0][m]))
    pt = np.angle(P.truth[1][m] * np.conj(P.truth[0][m]))
    cms = P.venc / np.pi * 100
    # signal-weighted (|x*_0||x*_1|, resp. the truth magnitudes) RMS over the heart lumen ROI
    wr = np.abs(refs[0][m] * refs[1][m])
    wt = np.abs(P.truth[0][m] * P.truth[1][m])
    vfp = np.sqrt(np.sum(wr * np.angle(np.exp(1j * (pd - pr))) ** 2) / wr.sum()) * cms
    vtr = np.sqrt(np.sum(wt * np.angle(np.exp(1j * (pd - pt))) ** 2) / wt.sum()) * cms
    return dict(err0=e0, err1=e1, err=(e0 + e1) / 2, heart=eh, v_fp=vfp, v_truth=vtr)


def at_budget(trace, B):
    i = np.searchsorted(trace[:, 0], B, side="right") - 1
    return trace[max(i, 0), 1]


def main(argv):
    t0 = time.time()
    cfg = dict(coils="global", noise=0.03)
    for a in argv:
        if a.startswith("coils="):
            cfg["coils"] = a.split("=", 1)[1]
    tag = cfg["coils"].replace(":", "")
    P = Problem(**cfg)
    print(f"problem {tag}: nc {P.nc}, M {P.M}/encode, σ {P.sigma:.3g}, toeplitz err {P.toeplitz_err:.1e}, bulk {P.bulk:.3g}, "
          f"γ {P.gamma:.3g}, κ {P.kappa:.3g}  ({time.time() - t0:.0f}s)", flush=True)
    # λ: from a short sweep of the truth error of the (60-iteration NCG) solution, cached
    lf = OUT / f"lambda_{tag}.json"
    if lf.exists():
        P.lam = json.loads(lf.read_text())["lam"]
    else:
        best = None
        sw = {}
        for lam in np.geomspace(0.03, 30, 10) * P.sigma * np.sqrt(P.M) * 0.5:
            P.lam = lam
            xs = []
            for e in range(2):
                tr = Tracker(P, e, None, 60)
                xs.append(run_ncg(P, e, tr, maxit=60)[0])
            err = np.mean([np.linalg.norm(xs[e] - P.truth[e]) / np.linalg.norm(P.truth[e]) for e in range(2)])
            sw[float(lam)] = float(err)
            print(f"  λ {lam:.3g}: truth err {err:.4f}", flush=True)
            if best is None or err < best[1]:
                best = (lam, err)
        P.lam = float(best[0])
        lf.write_text(json.dumps(dict(lam=P.lam, sweep=sw)))
    print(f"λ {P.lam:.4g}", flush=True)
    # reference fixed points (smooth objective): long NCG
    rf = RUNS / f"{tag}_ref.npz"
    if rf.exists():
        refs = list(np.load(rf)["x"])
    else:
        refs = []
        for e in range(2):
            tr = Tracker(P, e, None, 10 ** 9)
            x = None
            for rnd in range(6):  # restarts with an exact gradient (no recursion drift); each round cached (resumable)
                rcf = RUNS / f"{tag}_ref_e{e}_r{rnd}.npy"
                if rcf.exists():
                    x = np.load(rcf)
                    continue
                x, _ = run_ncg(P, e, Tracker(P, e, None, 10 ** 9), x=x, maxit=300, tol=1e-7 * 10 ** (5 - rnd) if rnd < 5 else 1e-9)
                np.save(rcf, x)
            gfx = grad_f_full(P, e, x) + P.prior(x)[1]
            g0 = -P.b[e].sum(0) + P.prior(np.zeros(P.n, complex))[1]
            print(f"  ref encode {e}: ‖g‖_P/‖g0‖_P = {np.sqrt(np.real(np.vdot(gfx, P.precond(e, gfx)) / np.vdot(g0, P.precond(e, g0)))):.2e}", flush=True)
            refs.append(x)
        np.savez(rf, x=np.array(refs))
    tr_err = [np.linalg.norm(refs[e] - P.truth[e]) / np.linalg.norm(P.truth[e]) for e in range(2)]
    m_ref = metrics(P, refs, refs)
    print(f"reference: truth err {np.round(tr_err, 4)}, velocity vs truth {m_ref['v_truth']:.2f} cm/s", flush=True)

    # step sizes: full and per sampler (α stable for every subset, λ_max over draws)
    lf_full = [lmax(P, e, P.maps, None) for e in range(2)]
    stepcache = OUT / f"steps_{tag}.json"
    steps = json.loads(stepcache.read_text()) if stepcache.exists() else {}

    def alpha_for(k, d, kind, with_prior=True):
        key = f"{k}:{d}:{kind}" + ("" if with_prior else ":data")
        fkey = "full" + ("" if with_prior else ":data")
        if fkey not in steps:
            steps[fkey] = lmax(P, 0, P.maps, None, with_prior=with_prior)
        if key not in steps:
            s = Sampler(P, k, d, kind, seed=123)
            lm = 0
            for i in range(8 if kind != "keep" else 1):
                maps, w, _ = s.draw()
                lm = max(lm, lmax(P, 0, maps, w, iters=20, seed=i, with_prior=with_prior))
            steps[key] = lm
            stepcache.write_text(json.dumps(steps))
        return 1.0 / max(steps[key], steps[fkey]), steps[key] / steps[fkey]

    a_full = 1.0 / lf_full[0]
    B = 60
    exps = {}

    def E(name, fn):
        exps[name] = fn

    E("ncg_full", lambda e, tr, s: run_ncg(P, e, tr)[0])
    E("pgd_full", lambda e, tr, s: run_sgd(P, e, tr, None, a_full))
    for tau in (0.01, 0.03, 0.1):
        E(f"lagncg_rad_k12s2_t{tau}", lambda e, tr, s, tau=tau: run_lagncg(P, e, tr, Sampler(P, 12, 2, "rad", seed=s), tau))
        E(f"lagncg_rad_k8s2_t{tau}", lambda e, tr, s, tau=tau: run_lagncg(P, e, tr, Sampler(P, 8, 2, "rad", seed=s), tau))
        E(f"lagncg_circ_k8_t{tau}", lambda e, tr, s, tau=tau: run_lagncg(P, e, tr, Sampler(P, 8, 0, "keep", seed=s), tau, circ_tail=1.0))
    for K in (3, 10):
        E(f"fcg_rad_k12s2_K{K}", lambda e, tr, s, K=K: run_ihs_ls(P, e, tr, Sampler(P, 12, 2, "rad", seed=s), inner=K, flex=True))
    for k in (8, 12, 16):
        E(f"fcg_circ_k{k}_K5", lambda e, tr, s, k=k: run_ihs_ls(P, e, tr, Sampler(P, k, 0, "keep", seed=s), inner=5, flex=True, circ_tail=1.0))
    # smooth-objective first-order arms (sgd / svrg / pyramid) are left out: with the Moreau-envelope prior
    # (γ = 0.1/bulk) the gradient step is limited by the prior curvature (λ_max ratio 0.99 for every subset), which
    # P²nP does not have; first-order coil sampling is tested in the PnP form below (data-only step bound).
    for kind, k, d, Ks, lag in (("rad", 12, 2, (3, 5, 10), True), ("rad", 3, 1, (5,), False), ("rad", 8, 2, (5,), False), ("rad", 16, 2, (5,), False),
                                ("rad", 20, 2, (5,), False), ("sub", 12, 3, (5,), True), ("keep", 12, 0, (5,), True), ("keep", 20, 0, (5,), False)):
        for K in Ks:
            E(f"ihs_{kind}_k{k}s{d}_K{K}", lambda e, tr, s, kind=kind, k=k, d=d, K=K: run_ihs(P, e, tr, Sampler(P, k, d, kind, seed=s), inner=K))
        E(f"ils_{kind}_k{k}s{d}_K5", lambda e, tr, s, kind=kind, k=k, d=d: run_ihs_ls(P, e, tr, Sampler(P, k, d, kind, seed=s), inner=5))
        E(f"fcg_{kind}_k{k}s{d}_K5", lambda e, tr, s, kind=kind, k=k, d=d: run_ihs_ls(P, e, tr, Sampler(P, k, d, kind, seed=s), inner=5, flex=True))
        if lag:
            E(f"lag_{kind}_k{k}s{d}_t0.05", lambda e, tr, s, kind=kind, k=k, d=d: run_ihs(P, e, tr, Sampler(P, k, d, kind, seed=s), inner=50, tau=0.05))
    a_pnp = min(alpha_for(12, 3, "sub", False)[0], alpha_for(12, 5, "sub", False)[0], alpha_for(8, 4, "sub", False)[0])
    for k, d in ((0, 6), (4, 4), (8, 4), (12, 3), (12, 5), (12, 6), (16, 4), (20, 5)):
        print(f"  inflation keep {k} draw {d}: data-only λ_max ratio {alpha_for(k, d, 'sub', False)[1]:.2f}, "
              f"with prior curvature {alpha_for(k, d, 'sub')[1]:.2f}", flush=True)
    sel = [a for a in argv if not a.startswith("coils=")]
    refs_pnp, thr = None, None
    if any(a.startswith("pnp") for a in sel):
        pf = OUT / f"pnp_{tag}.json"
        pj = json.loads(pf.read_text()) if pf.exists() else dict(sweep={}, alpha=a_pnp)
        wd = np.abs(haar_fwd(refs[0]))[P.detail]
        for q in (20, 35, 50, 70, 80):  # threshold sweep (truth error after 300 full PnP iterations), cached per q
            if str(q) in pj["sweep"]:
                continue
            th = float(np.percentile(wd, q))
            xs = [run_sgd(P, e, Tracker(P, e, None, 300), None, a_pnp, pnp_thr=th) for e in range(2)]
            err = float(np.mean([np.linalg.norm(xs[e] - P.truth[e]) / np.linalg.norm(P.truth[e]) for e in range(2)]))
            pj["sweep"][str(q)] = [th, err]
            pf.write_text(json.dumps(pj))
            print(f"  pnp thr q{q} {th:.3g}: truth err {err:.4f}", flush=True)
        thr = min(pj["sweep"].values(), key=lambda v: v[1])[0]
        refs_pnp = []
        for e in range(2):  # reference: full PnP in cached segments of 300 iterations until the change is < 1e-9
            rf_ = RUNS / f"{tag}_ref_pnp_e{e}.npz"
            x, n, ch = (np.zeros(P.n, complex), 0, 1.0)
            if rf_.exists():
                z = np.load(rf_)
                x, n, ch = z["x"], int(z["n"]), float(z["ch"])
            while ch > 1e-6 and n < 6000:
                xn = run_sgd(P, e, Tracker(P, e, None, 300), None, a_pnp, pnp_thr=thr, x0=x)
                ch = float(np.linalg.norm(xn - x) / np.linalg.norm(xn))
                x, n = xn, n + 300
                np.savez(rf_, x=x, n=n, ch=ch)
                print(f"  pnp ref encode {e}: {n} iterations, change {ch:.2e}", flush=True)
            refs_pnp.append(x)
    E("pnp_full", lambda e, tr, s: run_sgd(P, e, tr, None, a_pnp, pnp_thr=thr))
    E("pnp_sgd_k12d3", lambda e, tr, s: run_sgd(P, e, tr, Sampler(P, 12, 3, "sub", seed=s), a_pnp, pnp_thr=thr))
    E("pnp_sgd_k8d4", lambda e, tr, s: run_sgd(P, e, tr, Sampler(P, 8, 4, "sub", seed=s), a_pnp, pnp_thr=thr))
    E("pnp_svrg_k12d3m10", lambda e, tr, s: run_svrg(P, e, tr, Sampler(P, 12, 3, "sub", seed=s), a_pnp, 10, pnp_thr=thr))
    E("pnp_svrg_k8d4m10", lambda e, tr, s: run_svrg(P, e, tr, Sampler(P, 8, 4, "sub", seed=s), a_pnp, 10, pnp_thr=thr))
    E("pnp_pyr_k12d5", lambda e, tr, s: run_sgd(P, e, tr, Sampler(P, 12, 5, "sub", seed=s), a_pnp, pnp_thr=thr,
                                                anneal=[(0, Sampler(P, 12, 5, "sub", seed=s)), (20, Sampler(P, 12, 10, "sub", seed=s + 7)), (40, None)]))
    E("pnp_pyr_fast", lambda e, tr, s: run_sgd(P, e, tr, Sampler(P, 12, 5, "sub", seed=s), a_pnp, pnp_thr=thr,
                                               anneal=[(0, Sampler(P, 12, 5, "sub", seed=s)), (5, Sampler(P, 12, 10, "sub", seed=s + 7)), (10, None)]))
    E("pnp_sgd_k12d3_last30", lambda e, tr, s: run_sgd(P, e, tr, Sampler(P, 12, 3, "sub", seed=s), a_pnp, pnp_thr=thr, full_last=30))

    names = [n for n in exps if not sel or any(n.startswith(s) for s in sel)]
    seeds = (0, 1)
    rfile = OUT / f"results_{tag}.json"
    results = json.loads(rfile.read_text())["results"] if rfile.exists() else {}
    for n in names:
        stoch = not n.endswith("_full")
        pnp = n.startswith("pnp")
        R_ = refs_pnp if pnp else refs
        Bn = 120 if pnp else B
        rows = []
        for s in (seeds if stoch else (0,)):
            f = RUNS / f"{tag}_{n}_s{s}.npz"
            if not f.exists():
                xs, trs, sn = [], [], []
                for e in range(2):
                    tr = Tracker(P, e, R_[e], Bn)
                    x = exps[n](e, tr, s)
                    xs.append(x)
                    trs.append(np.array(tr.rows))
                    sn.append(np.array([tr.snaps.get(b, x) for b in BUDGETS]))
                np.savez(f, x=np.array(xs), t0=trs[0], t1=trs[1], snaps=np.array(sn))
            z = np.load(f)
            row = {}
            for i, b in enumerate(BUDGETS):
                met = metrics(P, [z["snaps"][0][i], z["snaps"][1][i]], R_)
                row[f"err@{b}"] = met["err"]
                row[f"v@{b}"] = met["v_fp"]
                row[f"vt@{b}"] = met["v_truth"]
                row[f"heart@{b}"] = met["heart"]
            rows.append(row)
        results[n] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        results[n]["spread@40"] = float(np.std([r["err@40"] for r in rows]))
        r = results[n]
        print(f"{n:28s} " + " ".join(f"{r[f'err@{b}']:.2e}" for b in BUDGETS) + " | v " + " ".join(f"{r[f'v@{b}']:.2f}" for b in BUDGETS)
              + f" | vt@{Bn} {r[f'vt@{Bn}']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    inf = {k: v for k, v in steps.items()}
    rfile.write_text(json.dumps(dict(results=results, steps_lmax=inf, lmax_full=lf_full, lam=P.lam, sigma=P.sigma,
                                                             gamma=P.gamma, truth_err_ref=tr_err, v_truth_ref=m_ref["v_truth"]), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
