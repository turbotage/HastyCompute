#!/usr/bin/env python3
"""Undersampled non-Cartesian convergence toy: solvers + driver (research/convergence_strategy.md, section
"Undersampled non-Cartesian toy (2026-11-17)").  Model and operators: nc_core.py.

Per R (per-set acceleration; 5 .. 200):
  1. lambda: truth-optimal on a grid (exact-l1 objective, ADMM 200 gradients each).
  2. references: x*_l1 (ADMM 4000 gradients, checked by the prox-gradient residual and a second ADMM),
     x_gamma (smoothed, gamma in {1, 0.1, 0.01}/bulk; L-BFGS(5) to stationarity, checked by NCG).
  3. methods, each with a 200-gradient budget, distances to its own target minimiser (exact or smoothed):
     PCG on the unregularised quadratic (I, Cn), NCG (PR+) and L-BFGS(3) on the smoothed objective (metrics I, C, Cn,
     Cnp = Cn + prior-aware term from a 5-iteration pilot), FISTA exact prox (Euclidean; step 1/L, 0.3/L, 2/L),
     FISTA on the smoothed objective in a metric, ZMSHR1-M on the smoothed objective (production-like, RED form),
     Becker-Fadili 0SR1 proximal quasi-Newton (exact prox), ADMM-PCG (rho sweep x inner 3/8), PDHG with the Ong
     multichannel k-space preconditioner (c sweep) and scalar PDHG, Anderson(5) on the forward-backward map (exact)
     and on the preconditioned smoothed-gradient map.
Usage: OMP_NUM_THREADS=4 nice -n 10 rustlib/.venv/bin/python research/experiments/convergence/nc_toy.py OUTDIR \
          [R list, e.g. 5,10,25,50,100,200] [prior WT|WL] [quick]
Resumable: each (prior, R) writes OUTDIR/nc_<prior>_R<R>.json and is skipped when present (STOP file: stop after the
current R).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nc_core import (K, N, NC, COARSE_S, PEN_S, PEN_ST, KBAND, WBAND, SLEV, TLEV, LEV_S, LEV_T, Problem, W_st, WH_st,
                     haar_s, ihaar_s, soft, svt, nuc, to_blocks, from_blocks, vdot, nrm, rel)

IT = 200
import os
SMK = int(os.environ.get("NC_SMOKE", "0"))
if SMK:
    IT = SMK
CPS = (10, 20, 40, 100, 200)


# ------------------------------------------------------------------ priors
class PriorWT:
    """lam * ||W_st x||_1 (penalised bands)"""
    name = "WT"
    exact_prox = True

    def __init__(self, lam):
        self.lam = lam
        self.nW = 0
        self.nE = 0

    def W(self, x):
        self.nW += 1
        return W_st(x)

    def WH(self, c):
        self.nW += 1
        return WH_st(c)

    def val(self, x):
        return self.lam * float(np.abs(W_st(x))[PEN_ST].sum())

    def prox(self, x, t):
        return self.WH(soft(self.W(x), t * self.lam, PEN_ST))

    def _eg(self, c, gam):
        m = np.abs(c)
        thr = self.lam * gam
        g = np.where(m <= thr, c / gam, self.lam * c / np.maximum(m, 1e-300))
        return np.where(PEN_ST, g, 0)

    def env_val(self, x, gam):
        m = np.abs(W_st(x))[PEN_ST]
        thr = self.lam * gam
        return float(np.where(m <= thr, m * m / (2 * gam), self.lam * m - self.lam * thr / 2).sum())

    def env_grad(self, x, gam):
        self.nE += 1
        return self.WH(self._eg(self.W(x), gam))

    def line(self, x, d, gam):
        cx, cd = self.W(x), self.W(d)
        thr = self.lam * gam
        cd2 = np.abs(cd) ** 2

        def dphi(a):
            self.nE += 1
            c = cx + a * cd
            m = np.abs(c)
            q = m <= thr
            g = np.where(PEN_ST, np.where(q, c / gam, self.lam * c / np.maximum(m, 1e-300)), 0)
            d1 = float(np.real(np.conj(cd) * g).sum())
            pr = np.real(np.conj(c) * cd) / np.maximum(m, 1e-300)
            d2 = np.where(PEN_ST, np.where(q, cd2 / gam, self.lam * (cd2 - pr * pr) / np.maximum(m, 1e-300)), 0)
            return d1, float(d2.sum())

        def grad_at(a):
            self.nE += 1
            return self.WH(self._eg(cx + a * cd, gam))
        return dphi, grad_at

    def zero_frac(self, x, gam):
        """fraction of penalised coefficients in the quadratic (zeroed) region per (spatial level, temporal level)"""
        c = np.abs(W_st(x))
        thr = self.lam * gam
        z = np.zeros((LEV_S + 1, LEV_T + 1))
        for js in range(LEV_S + 1):
            for jt in range(LEV_T + 1):
                msk = (SLEV[None] == js) & (TLEV[:, None, None, None] == jt) & PEN_ST
                z[js, jt] = float((c[msk] <= thr).mean()) if msk.any() else 0.0
        return z

    def psi(self, x, gam):
        z = self.zero_frac(x, gam)
        return z[KBAND[None], WBAND[:, None, None, None]] / gam, z


class PriorWL:
    """lam_w * ||W_s x||_1 (per frame) + lam_l * sum_blocks ||B x||_* (6^3 x K, fixed grid)"""
    name = "WL"
    exact_prox = False

    def __init__(self, lam_w, lam_l):
        self.lw, self.ll = lam_w, lam_l
        self.nW = self.nE = self.nS = 0

    def val(self, x):
        return self.lw * float(np.abs(haar_s(x))[PEN_S].sum()) + self.ll * nuc(x)

    def env_val(self, x, gam):
        m = np.abs(haar_s(x))[PEN_S]
        t1 = self.lw * gam
        v = float(np.where(m <= t1, m * m / (2 * gam), self.lw * m - self.lw * t1 / 2).sum())
        s = np.linalg.svd(to_blocks(x), compute_uv=False)
        t2 = self.ll * gam
        return v + float(np.where(s <= t2, s * s / (2 * gam), self.ll * s - self.ll * t2 / 2).sum())

    def _egw(self, c, gam):
        m = np.abs(c)
        g = np.where(m <= self.lw * gam, c / gam, self.lw * c / np.maximum(m, 1e-300))
        return np.where(PEN_S, g, 0)

    def _egl(self, x, gam):
        self.nS += 1
        return (x - svt(x, self.ll * gam)[0]) / gam

    def env_grad(self, x, gam):
        self.nE += 1
        self.nW += 2
        return ihaar_s(self._egw(haar_s(x), gam)) + self._egl(x, gam)

    def line(self, x, d, gam):
        cx, cd = haar_s(x), haar_s(d)
        self.nW += 2
        t1 = self.lw * gam
        cd2 = np.abs(cd) ** 2

        def dphi(a):
            self.nE += 1
            c = cx + a * cd
            m = np.abs(c)
            q = m <= t1
            g = np.where(PEN_S, np.where(q, c / gam, self.lw * c / np.maximum(m, 1e-300)), 0)
            d1 = float(np.real(np.conj(cd) * g).sum())
            pr = np.real(np.conj(c) * cd) / np.maximum(m, 1e-300)
            d2 = float(np.where(PEN_S, np.where(q, cd2 / gam, self.lw * (cd2 - pr * pr) / np.maximum(m, 1e-300)), 0).sum())
            gl = self._egl(x + a * d, gam)
            d1 += vdot(d, gl)
            d2 += vdot(d, d) / gam      # upper curvature of the LLR envelope (Newton on it is safeguarded)
            return d1, d2

        def grad_at(a):
            self.nE += 1
            self.nW += 1
            return ihaar_s(self._egw(cx + a * cd, gam)) + self._egl(x + a * d, gam)
        return dphi, grad_at

    def psi(self, x, gam):
        c = np.abs(haar_s(x))
        zs = np.zeros(LEV_S + 1)
        for js in range(1, LEV_S + 1):
            msk = np.broadcast_to(SLEV[None] == js, c.shape)
            zs[js] = float((c[msk] <= self.lw * gam).mean())
        s = np.linalg.svd(to_blocks(x), compute_uv=False)
        zl = float((s <= self.ll * gam).mean())
        return (zs[KBAND] / gam)[None] + zl / gam, {"z_spatial": zs.tolist(), "z_llr": zl}


# ------------------------------------------------------------------ recording
class Rec:
    def __init__(self, P, target, obj, Fstar, F0, xl1=None):
        self.P, self.target, self.obj, self.Fs, self.F0 = P, target, obj, Fstar, F0
        self.xl1 = target if xl1 is None else xl1
        self.h = []
        self.cp = {}
        self.t0 = time.time()
        self.bad = False

    def __call__(self, x):
        c = self.P.cost()
        if not np.all(np.isfinite(x)) or nrm(x) > 1e3 * nrm(self.target):
            self.bad = True
            self.h.append((c, np.inf, np.inf, np.inf))
            return
        self.h.append((c, rel(x, self.target), rel(x, self.P.xt), rel(x, self.xl1)))
        for cp in CPS:
            if c >= cp and cp not in self.cp:
                self.cp[cp] = {"objgap": (self.obj(x) - self.Fs) / (self.F0 - self.Fs), "heldout": self.P.heldout(x),
                               "truth": rel(x, self.P.xt)}

    def stop(self):
        return self.bad or (self.h and self.h[-1][0] >= IT)

    def summary(self, extra=None):
        h = np.array(self.h) if self.h else np.zeros((0, 4))
        def to(thr, col=1):
            i = np.nonzero(h[:, col] <= thr)[0]
            return float(h[i[0], 0]) if len(i) else None
        def at(c):
            i = np.nonzero(h[:, 0] <= c)[0]
            return float(h[i[-1], 1]) if len(i) else None
        def tat(c):
            i = np.nonzero(h[:, 0] <= c)[0]
            return float(h[i[-1], 2]) if len(i) else None
        out = {"to_1e-1": to(1e-1), "to_1e-2": to(1e-2), "to_1e-3": to(1e-3), "at20": at(20), "at40": at(40),
               "at100": at(100), "final": float(h[-1, 1]) if len(h) else None, "truth_at20": tat(20),
               "truth_at40": tat(40), "truth_final": float(h[-1, 2]) if len(h) else None,
               "l1_to_1e-1": to(1e-1, 3), "l1_to_3e-2": to(3e-2, 3), "l1_to_1e-2": to(1e-2, 3),
               "l1_at40": float(h[np.nonzero(h[:, 0] <= 40)[0][-1], 3]) if len(h) else None,
               "diverged": self.bad, "cp": {str(k): v for k, v in self.cp.items()}, "wall_s": time.time() - self.t0,
               "curve": [(float(a), float(b), float(c_)) for a, b, c_, _ in h[:: max(1, len(h) // 60)]]}
        if extra:
            out.update(extra)
        return out


# ------------------------------------------------------------------ solvers
def linesearch(dphi, gd0, dQd):
    """root of phi'(a) = gd0 + a dQd + prior'(a) by safeguarded Newton (phi convex); returns a, evals"""
    def f(a):
        p1, p2 = dphi(a)
        return gd0 + a * dQd + p1, dQd + p2
    f0, f20 = f(0.0)
    n = 1
    if f0 >= 0:
        return 0.0, n
    a = -f0 / max(f20, 1e-300)
    lo, hi = 0.0, None
    for _ in range(40):
        d1, d2 = f(a)
        n += 1
        if abs(d1) <= 1e-12 * abs(f0):
            break
        if d1 < 0:
            lo = a
        else:
            hi = a
        an = a - d1 / max(d2, 1e-300)
        if hi is None:
            if an <= lo:
                an = 2 * max(lo, 1e-300)
        elif not (lo < an < hi):
            an = 0.5 * (lo + hi)
        if abs(an - a) <= 1e-13 * abs(a):
            a = an
            break
        a = an
    return a, n


def smooth_solver(P, pr, Mf, gam, rec, kind="ncg", m=3, x0=None, it=IT, pilot=None):
    """NCG (PR+, restarts) or L-BFGS(m) on F_gamma = f + env_gamma(R), data-exact line search (one A^H A per
    iteration). pilot = (n_pilot, metric_builder): run n_pilot iterations, then rebuild the metric from the iterate
    (prior-aware term) and restart."""
    x = np.zeros((K, N, N, N), complex) if x0 is None else x0.copy()
    gq = (P.Q(x) if x0 is not None else 0 * x) - P.b
    g = gq + pr.env_grad(x, gam)
    d = gprev = Mgprev = None
    S_, Y_ = [], []
    nls = 0
    info = {}
    for k in range(it):
        if pilot is not None and k == pilot[0]:
            Mf, info = pilot[1](x)
            d = None
            S_, Y_ = [], []
        if kind == "ncg":
            Mg = Mf(g)
            if d is None:
                d = -Mg
            else:
                beta = max(0.0, vdot(g - gprev, Mg) / vdot(gprev, Mgprev))
                d = -Mg + beta * d
                if vdot(d, g) >= 0:
                    d = -Mg
            Mgprev = Mg
        else:
            q = g.copy()
            al = []
            for s_, y_ in reversed(list(zip(S_, Y_))):
                r_ = 1 / vdot(y_, s_)
                a_ = r_ * vdot(s_, q)
                al.append((a_, r_, s_, y_))
                q = q - a_ * y_
            ck = vdot(S_[-1], Y_[-1]) / vdot(Y_[-1], Mf(Y_[-1])) if S_ else 1.0
            z = ck * Mf(q)
            for a_, r_, s_, y_ in reversed(al):
                z = z + (a_ - r_ * vdot(y_, z)) * s_
            d = -z
            if vdot(d, g) >= 0:
                d = -Mf(g)
        Qd = P.Q(d)
        dphi, grad_at = pr.line(x, d, gam)
        a, ne = linesearch(dphi, vdot(gq, d), vdot(d, Qd))
        nls += ne
        xn = x + a * d
        gqn = gq + a * Qd
        gn = gqn + grad_at(a)
        if kind != "ncg":
            sy = vdot(xn - x, gn - g)
            if sy > 1e-12 * nrm(xn - x) * nrm(gn - g):
                S_.append(xn - x)
                Y_.append(gn - g)
            if len(S_) > m:
                S_.pop(0)
                Y_.pop(0)
        gprev = g
        x, gq, g = xn, gqn, gn
        if rec is not None:
            rec(x)
            if rec.stop():
                break
    return x, {"linesearch_evals_per_it": nls / max(k + 1, 1), **({"pilot_info": info} if info else {})}


def pcg_quad(P, Mf, rec, it=100, heldout_curve=None):
    x = np.zeros((K, N, N, N), complex)
    r = P.b.copy()
    z = Mf(r)
    p = z.copy()
    rz = vdot(r, z)
    for _ in range(it):
        q = P.Q(p)
        a = rz / vdot(p, q)
        x = x + a * p
        r = r - a * q
        z = Mf(r)
        rz2 = vdot(r, z)
        p = z + (rz2 / rz) * p
        rz = rz2
        rec(x)
        if heldout_curve is not None:
            heldout_curve.append((P.heldout(x), rel(x, P.xt)))
    return x


def fista_exact(P, pr, alpha, rec, it=IT):
    x = np.zeros((K, N, N, N), complex)
    yv = x.copy()
    tk = 1.0
    for _ in range(it):
        g = P.Q(yv) - P.b
        xn = pr.prox(yv - alpha * g, alpha)
        if vdot(yv - xn, xn - x) > 0:
            tk = 1.0
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        yv = xn + ((tk - 1) / tn) * (xn - x)
        tk = tn
        x = xn
        rec(x)
        if rec.stop():
            break
    return x


def fista_smooth(P, pr, Mf, alpha, gam, rec, it=IT):
    x = np.zeros((K, N, N, N), complex)
    yv = x.copy()
    tk = 1.0
    for _ in range(it):
        g = P.Q(yv) - P.b + pr.env_grad(yv, gam)
        xn = yv - alpha * Mf(g)
        if vdot(yv - xn, xn - x) > 0:
            tk = 1.0
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        yv = xn + ((tk - 1) / tn) * (xn - x)
        tk = tn
        x = xn
        rec(x)
        if rec.stop():
            break
    return x


def zmshr1_scalars(ss, sm, mm, th1=2e-6, th2=200.0):
    a = 1.0
    for i in range(2001):
        aa = i / 2000
        s_v = aa * ss + (1 - aa) * sm
        vv = aa * aa * ss + 2 * aa * (1 - aa) * sm + (1 - aa) ** 2 * mm
        if s_v > 0 and s_v / ss >= th1 and vv / s_v <= th2:
            a = aa
            break
    s_v = a * ss + (1 - a) * sm
    vv = a * a * ss + 2 * a * (1 - a) * sm + (1 - a) ** 2 * mm
    tau = ss / s_v - np.sqrt(max((ss / s_v) ** 2 - ss / vv, 0.0))
    if not (np.isfinite(tau) and tau > 0):
        return 1.0, 1.0
    return a, tau


def zmshr1m(P, pr, Mf, Minv, alpha, gam, rec, it=IT):
    """ZMSHR1 in metric M on the smoothed objective (p2np.rs Zmshr1M, global scalars; RED-like form)"""
    x = np.zeros((K, N, N, N), complex)
    xprev = gprev = None
    for k in range(it):
        g = alpha * (P.Q(x) - P.b + pr.env_grad(x, gam))
        if k == 0:
            pg = Mf(g)
        else:
            s = x - xprev
            mv = g - gprev
            ms, mm = Minv(s), Mf(mv)
            ss, sm, mmm = vdot(s, ms), vdot(s, mv), vdot(mv, mm)
            a, tau = zmshr1_scalars(ss, sm, mmm)
            w = (1 - tau * a) * s - tau * (1 - a) * mm
            z = a * ms + (1 - a) * mv
            wz, wg = vdot(w, z), vdot(w, g)
            ok = wz > 1e-8 * nrm(w) * nrm(z)
            pg = tau * Mf(g) + (wg / wz if ok else 0.0) * w
        xprev, gprev = x, g
        x = x - pg
        rec(x)
        if rec.stop():
            break
    return x


def sr1_prox(P, pr, rec, it=IT, gbf=0.8):
    """Becker-Fadili zero-memory SR1 proximal quasi-Newton (NeurIPS 2012), exact prox in D - w w^T via 1-D root."""
    x = np.zeros((K, N, N, N), complex)
    g = P.Q(x) - P.b
    xprev = gprev = None
    nroot = 0
    for k in range(it):
        if k == 0:
            H0 = 1.0 / P.L
            u = None
        else:
            s, yv = x - xprev, g - gprev
            sy = vdot(s, yv)
            H0 = gbf * sy / vdot(yv, yv) if sy > 0 else 1.0 / P.L
            v = s - H0 * yv
            den = vdot(v, yv)
            u = v / np.sqrt(den) if den > 1e-8 * nrm(v) * nrm(yv) else None
        xh = x - H0 * g - (u * vdot(u, g) if u is not None else 0)
        if u is None:
            xn = pr.prox(xh, H0)
        else:
            # B = H^-1 = (1/H0) I - w w^T ; prox^B(xh) = prox^D(xh + alpha* D^-1 w), root of p(alpha)
            w = (u / H0) / np.sqrt(1 + vdot(u, u) / H0)
            cxh, cw = pr.W(xh), pr.W(w)
            thr = H0 * pr.lam

            def pz(al):
                pr.nE += 1
                return soft(cxh + al * H0 * cw, thr, PEN_ST)
            def p(al):
                return vdot(cw, pz(al) - cxh) - al
            lo, hi = -1.0, 1.0
            while p(lo) < 0:
                lo *= 2
            while p(hi) > 0:
                hi *= 2
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                nroot += 1
                if p(mid) > 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo <= 1e-12 * max(abs(lo), abs(hi), 1e-30):
                    break
            xn = pr.WH(pz(0.5 * (lo + hi)))
        xprev, gprev = x, g
        x = xn
        g = P.Q(x) - P.b
        rec(x)
        if rec.stop():
            break
    return x, {"root_evals_per_it": nroot / max(k + 1, 1)}


def admm(P, pr, rho, n_in, rec, it=IT, x0=None, max_cost=None):
    """ADMM-PCG: splits v_i = T_i x with orthonormal T_i (WT: W_st; WL: W_s and the LLR block map); x-update
    (Q + n_split rho I) x = b + rho sum T_i^H (v_i - u_i) by n_in warm-started PCG steps with metric 1/(n_t lam + n rho)."""
    if pr.name == "WT":
        Ts = [(W_st, WH_st, lambda c, t: soft(c, t * pr.lam, PEN_ST))]
    else:
        Ts = [(haar_s, ihaar_s, lambda c, t: soft(c, t * pr.lw, PEN_S)),
              (lambda x: x, lambda x: x, lambda c, t: svt(c, t * pr.ll)[0])]
    ns = len(Ts)
    Mr, _ = P.metric("Cn", rho=ns * rho)
    x = np.zeros((K, N, N, N), complex) if x0 is None else x0.copy()
    Qx = P.Q(x, count=False) if x0 is not None else np.zeros_like(x)
    vs = [T(x) for T, _, _ in Ts]
    us = [np.zeros_like(v) for v in vs]
    max_cost = IT if max_cost is None else max_cost
    while P.cost() < max_cost:
        rhs = P.b + rho * sum(TH(v - u) for (_, TH, _), v, u in zip(Ts, vs, us))
        r = rhs - (Qx + ns * rho * x)
        z = Mr(r)
        p = z.copy()
        rz = vdot(r, z)
        for _ in range(n_in):
            Qp = P.Q(p)
            Ap = Qp + ns * rho * p
            a = rz / vdot(p, Ap)
            x = x + a * p
            Qx = Qx + a * Qp
            r = r - a * Ap
            z = Mr(r)
            rz2 = vdot(r, z)
            p = z + (rz2 / rz) * p
            rz = rz2
            if rec is not None:
                rec(x)
                if rec.stop():
                    return x
        for i, (T, _, pxf) in enumerate(Ts):
            c = T(x)
            vs[i] = pxf(c + us[i], 1.0 / rho)
            us[i] = us[i] + c - vs[i]
    return x


def pdhg(P, pr, Sig, rec, it=IT, sig2=None):
    """PDHG / Chambolle-Pock, data term dualised with a k-space step Sig (list per set (M_t, NC) or scalar); WT: primal
    prox exact; WL: primal prox = spatial wavelet, LLR dualised too (step sig2)."""
    SA = lambda v: [s * a for s, a in zip(Sig, P.A(v, count=False))] if isinstance(Sig, list) else \
        [Sig * a for a in P.A(v, count=False)]
    rng = np.random.default_rng(4)
    v = rng.standard_normal((K, N, N, N)) + 0j
    lamb = 0
    for _ in range(40):
        w = P.AH(SA(v), count=False)
        lamb = nrm(w) / nrm(v)
        v = w / nrm(w)
    if pr.name == "WT":
        tau = 0.99 / lamb
    else:
        sig2 = lamb if sig2 is None else sig2
        tau = 0.99 / (lamb + sig2)
    x = np.zeros((K, N, N, N), complex)
    u = [np.zeros_like(y) for y in P.y]
    u2 = np.zeros_like(x)
    for _ in range(it):
        g = P.AH(u) + (u2 if pr.name == "WL" else 0)
        if pr.name == "WT":
            xn = pr.prox(x - tau * g, tau)
        else:
            xn = ihaar_s(soft(haar_s(x - tau * g), tau * pr.lw, PEN_S))
        xb = 2 * xn - x
        Ax = P.A(xb)
        if isinstance(Sig, list):
            u = [(ui + s * (a - y)) / (1 + s) for ui, s, a, y in zip(u, Sig, Ax, P.y)]
        else:
            u = [(ui + Sig * (a - y)) / (1 + Sig) for ui, a, y in zip(u, Ax, P.y)]
        if pr.name == "WL":
            vv = u2 + sig2 * xb
            u2 = vv - svt(vv, pr.ll)[0]
        x = xn
        rec(x)
        if rec.stop():
            break
    return x, {"tau": tau, "norm_SA2": lamb}


def anderson(P, Gmap, rec, m=5, it=IT, safeguard=True):
    """type-II Anderson(m) on a fixed-point map G (one gradient per G evaluation), with a residual-decrease
    safeguard (reject the extrapolation if ||G(x)-x|| grows; fall back to the plain step)."""
    x = np.zeros((K, N, N, N), complex)
    Gx = Gmap(x)
    f = Gx - x
    Xg, F = [Gx], [f]
    nrej = 0
    xplain = None
    while True:
        if len(F) >= 2:
            dF = np.array([(F[i + 1] - F[i]).ravel() for i in range(len(F) - 1)]).T
            dG = np.array([(Xg[i + 1] - Xg[i]).ravel() for i in range(len(F) - 1)]).T
            A = np.concatenate([dF.real, dF.imag], 0)
            bb = np.concatenate([f.ravel().real, f.ravel().imag])
            gam_ = np.linalg.lstsq(A, bb, rcond=1e-10)[0]
            xn = Gx - (dG @ gam_).reshape(x.shape)
        else:
            xn = Gx
        xplain = Gx
        Gn = Gmap(xn)
        fn = Gn - xn
        if safeguard and len(F) >= 2 and nrm(fn) > nrm(f):
            nrej += 1
            xn = xplain
            Gn = Gmap(xn)
            fn = Gn - xn
            Xg, F = [], []
        x, Gx, f = xn, Gn, fn
        Xg.append(Gx)
        F.append(f)
        if len(F) > m + 1:
            Xg.pop(0)
            F.pop(0)
        rec(x)
        if rec.stop():
            break
    return x, {"rejections": nrej}


# ------------------------------------------------------------------ driver
def pg_residual(P, pr, x):
    g = P.Q(x, count=False) - P.b
    return nrm(x - pr.prox(x - g / P.L, 1 / P.L)) / nrm(x)


def run_R(R, prior, out, quick=False):
    t0 = time.time()
    P = Problem(R)
    res = {"R": R, "prior": prior, "N": N, "K": K, "NC": NC, "samples_per_set": P.M.tolist(),
           "spokes_per_set": P.cnt.tolist(), "heldout_spokes": P.ho.tolist(),
           "angular_undersampling": float(4 * np.pi * (N / 2) ** 2 / P.cnt.mean()),
           "pooled_R": float(N ** 3 / P.M.sum()), "L": P.L, "bulk": P.bulk, "n_t": P.n_t.tolist(),
           "circ_range": float(P.lam_bar.max() / P.lam_bar.min()), "sigma": P.sigma,
           "frac_k_at_floor": float((P.lam_bar <= P.lam_bar.min() * 1.0000001).mean()),
           "lam0_over_bulk": float(P.lam_bar.max() / P.bulk),
           "heldout_truth": P.heldout(P.xt), "runs": {}}
    log = lambda s: print(f"[R={R:g} {prior} K={K}] {s}", flush=True)
    log(f"setup {time.time() - t0:.1f}s; spokes/set {P.cnt.tolist()}; pooled R {res['pooled_R']:.2f};"
        f" L {P.L:.3g} bulk {P.bulk:.3g}; floor frac {res['frac_k_at_floor']:.2f}; heldout(truth) {res['heldout_truth']:.3f}")
    bulk = P.bulk
    x0 = np.zeros((K, N, N, N), complex)

    # ---- lambda: truth-optimal on a grid, by continuation (large -> small lambda, warm starts), exact-l1 objective,
    #      ADMM rho = 3 bulk, 8 inner, 400 gradients per value
    if prior == "WT":
        scale = float(np.abs(W_st(P.b))[PEN_ST].max())
        grid = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5]
        mk = lambda f: PriorWT(f * scale)
    else:
        scale = float(np.abs(haar_s(P.b))[PEN_S].max())
        sl = float(np.linalg.svd(to_blocks(P.b), compute_uv=False).max())
        wt = out / f"nc_WT_R{R:g}.json"
        lwf = json.loads(wt.read_text())["lam_factor"] if wt.exists() else 3e-4
        grid = [(lwf, f) for f in (3e-2, 1e-2, 3e-3, 1e-3)] + [(lwf / 3, f) for f in (1e-2, 3e-3, 1e-3)]
        mk = lambda f: PriorWL(f[0] * scale, f[1] * sl)
    terr, xgrid = {}, {}
    xw = None
    for f in grid:
        P.reset()
        xw = admm(P, mk(f), 3.0 * bulk, 8, None, max_cost=SMK or (200 if quick else 400), x0=xw)
        xgrid[str(f)] = xw
        terr[str(f)] = rel(xw, P.xt)
        log(f"lambda {f}: truth err {terr[str(f)]:.4f}")
    ib = int(np.argmin([terr[str(f)] for f in grid]))
    fbest = grid[ib]
    pr = mk(fbest)
    res.update({"lam_factor": fbest, "lam_grid_truth_err": terr, "lam_at_grid_edge": ib in (0, len(grid) - 1)})

    # ---- references: exact l1 by two ADMM stages (rho 3 bulk / 8 inner, then rho 0.3 bulk / 3 inner), warm
    P.reset()
    nref = SMK or (1000 if quick else 3000)
    xs1 = admm(P, pr, 3.0 * bulk, 8, None, max_cost=nref, x0=xgrid[str(fbest)])
    P.reset()
    xs = admm(P, pr, 0.3 * bulk, 3, None, max_cost=nref, x0=xs1)
    Fl1 = lambda x: P.f(x) + pr.val(x)
    Fstar = Fl1(xs)
    ref = {"stage_change": rel(xs1, xs), "truth_err_xs_l1": rel(xs, P.xt), "heldout_xs_l1": P.heldout(xs),
           "F_star": Fstar}
    if pr.exact_prox:
        ref["pg_residual"] = pg_residual(P, pr, xs)
        ref["zero_frac_xs"] = float((np.abs(W_st(xs))[PEN_ST] < 1e-12 * np.abs(W_st(xs)).max()).mean())
    log(f"l1 reference: {ref}")
    gams = {"1": 1.0 / bulk, "0.1": 0.1 / bulk, "0.01": 0.01 / bulk, "0.001": 0.001 / bulk}
    xg = {}
    Mcn, _ = P.metric("Cn")
    Mcap, _ = P.metric("Cn", rho=bulk)
    for gk, gam in gams.items():
        P.reset()
        xa, _ = smooth_solver(P, pr, Mcap, gam, None, kind="lbfgs", m=5, it=SMK or (800 if quick else 1500), x0=xs)
        P.reset()
        xb, _ = smooth_solver(P, pr, Mcap, gam, None, kind="ncg", it=SMK or (300 if quick else 500), x0=xa)
        Fg = lambda x, gam=gam: P.f(x) + pr.env_val(x, gam)
        xg[gk] = xb
        ref[f"smooth_gamma{gk}"] = {"lbfgs_vs_ncg_polish": rel(xa, xb), "dist_to_l1": rel(xb, xs),
                                    "truth_err": rel(xb, P.xt), "heldout": P.heldout(xb), "F": Fg(xb)}
        if prior == "WT":
            c = np.abs(W_st(xb))[PEN_ST]
            ref[f"smooth_gamma{gk}"]["frac_quadratic_region"] = float((c <= pr.lam * gam).mean())
        log(f"smooth gamma={gk}/bulk ref: {ref[f'smooth_gamma{gk}']}")
    res["ref"] = ref
    gk0 = "0.1"
    gam0 = gams[gk0]

    def go(name, fn, target="l1", gk=gk0):
        P.reset()
        pr.nW = pr.nE = 0
        if hasattr(pr, "nS"):
            pr.nS = 0
        if target == "l1":
            rec = Rec(P, xs, Fl1, Fstar, Fl1(x0))
        else:
            gam = gams[gk]
            Fg = lambda x: P.f(x) + pr.env_val(x, gam)
            rec = Rec(P, xg[gk], Fg, Fg(xg[gk]), Fg(x0), xl1=xs)
        t1 = time.time()
        try:
            r = fn(rec)
            extra = r[1] if isinstance(r, tuple) else {}
        except (FloatingPointError, np.linalg.LinAlgError, ValueError, OverflowError, ZeroDivisionError) as e:
            rec.bad = True
            extra = {"error": repr(e)}
        s = rec.summary({"target": target if target == "l1" else f"smooth gamma={gk}/bulk", **extra,
                         "prior_transforms_per_grad": pr.nW / max(P.cost(), 1),
                         "prior_evals_per_grad": pr.nE / max(P.cost(), 1), "wall_s": time.time() - t1})
        res["runs"][name] = s
        f = lambda v: "-" if v is None else f"{v:.3g}"
        log(f"{name:46s} own 1e-1:{f(s['to_1e-1'])} 1e-2:{f(s['to_1e-2'])} 1e-3:{f(s['to_1e-3'])} @40 {f(s['at40'])}"
            f" final {f(s['final'])} | l1 1e-1:{f(s['l1_to_1e-1'])} 1e-2:{f(s['l1_to_1e-2'])} @40 {f(s['l1_at40'])}"
            f" | truth@40 {f(s['truth_at40'])} {'DIVERGED' if s['diverged'] else ''} ({s['wall_s']:.0f}s)")

    # ---- unregularised quadratic PCG (semi-convergence, held-out early stopping)
    for mn, rho in (("I", 0.0), ("Cn", 0.0), ("Cn+1bulk", bulk)):
        Mf, _ = P.metric("I" if mn == "I" else "Cn", rho=rho)
        P.reset()
        hc = []
        rec = Rec(P, xs, Fl1, Fstar, Fl1(x0))
        pcg_quad(P, Mf, rec, it=100, heldout_curve=hc)
        hc = np.array(hc)
        ib, ih = int(np.argmin(hc[:, 1])), int(np.argmin(hc[:, 0]))
        res["runs"][f"PCG unregularised|{mn}"] = {"best_truth_it": ib + 1, "best_truth_err": float(hc[ib, 1]),
                                                  "heldout_min_it": ih + 1, "truth_at_heldout_min": float(hc[ih, 1]),
                                                  "truth_at_100": float(hc[-1, 1]),
                                                  "dist_to_l1_min_at_heldout": rec.h[ih][1]}
        log(f"PCG unregularised|{mn}: {res['runs'][f'PCG unregularised|{mn}']}")

    # ---- smoothed objective: NCG / L-BFGS in metrics
    def pilot_builder(gam):
        def build(x):
            psi, z = pr.psi(x, gam)
            M, _ = P.metric("Cnp", psi=psi)
            return M, {"z": z.tolist() if hasattr(z, "tolist") else z}
        return (5, build)
    mets = {"I": P.metric("I")[0], "Cn": Mcn, "Cn+0.1bulk": P.metric("Cn", rho=0.1 * bulk)[0], "Cn+1bulk": Mcap,
            "Cn+10bulk": P.metric("Cn", rho=10 * bulk)[0], "Cn+100bulk": P.metric("Cn", rho=100 * bulk)[0]}
    for mn, Mf in mets.items():
        go(f"NCG|{mn}", lambda rec, Mf=Mf: smooth_solver(P, pr, Mf, gam0, rec), target="smooth")
    go("NCG|Cnp (pilot 5)", lambda rec: smooth_solver(P, pr, Mcn, gam0, rec, pilot=pilot_builder(gam0)), target="smooth")
    for mn in ("I", "Cn+1bulk"):
        go(f"LBFGS3|{mn}", lambda rec, mn=mn: smooth_solver(P, pr, mets[mn], gam0, rec, kind="lbfgs"), target="smooth")
    go("LBFGS3|Cnp (pilot 5)", lambda rec: smooth_solver(P, pr, Mcn, gam0, rec, kind="lbfgs", pilot=pilot_builder(gam0)),
       target="smooth")
    for gk in ("1", "0.01", "0.001"):
        for mn in ("I", "Cn+1bulk"):
            go(f"NCG|{mn} gamma={gk}", lambda rec, gk=gk, mn=mn: smooth_solver(P, pr, mets[mn], gams[gk], rec),
               target="smooth", gk=gk)
        go(f"NCG|Cnp (pilot 5) gamma={gk}", lambda rec, gk=gk: smooth_solver(P, pr, Mcn, gams[gk], rec,
                                                                             pilot=pilot_builder(gams[gk])),
           target="smooth", gk=gk)

    # ---- first-order on the smoothed objective, in metrics (step 1/lambda_max(P (Q + I/gamma)), 60 power its, x1.05)
    for mn, base, rho in (("I", "I", 0.0), ("Cn", "Cn", 0.0), ("Cn+1bulk", "Cn", bulk), ("Cn+1/gamma", "Cn", 1.0 / gam0)):
        Mf, Mi = P.metric(base, rho=rho)
        lam = 1.05 * P.power(lambda v: Mf(P.Q(v, count=False) + v / gam0), 60)
        go(f"FISTA-smooth|{mn}", lambda rec, Mf=Mf, lam=lam: fista_smooth(P, pr, Mf, 1.0 / lam, gam0, rec), target="smooth")
        go(f"ZMSHR1-M-smooth|{mn}", lambda rec, Mf=Mf, Mi=Mi, lam=lam: zmshr1m(P, pr, Mf, Mi, 1.0 / lam, gam0, rec),
           target="smooth")
        if mn in ("I", "Cn+1bulk"):
            go(f"Anderson5 smooth-gradient map|{mn}",
               lambda rec, Mf=Mf, lam=lam: anderson(P, lambda x: x - (1.0 / lam) * Mf(P.Q(x) - P.b + pr.env_grad(x, gam0)),
                                                    rec), target="smooth")

    # ---- exact l1
    if pr.exact_prox:
        for c in (1.0, 0.3, 2.0):
            go(f"FISTA-exact|I step {c}/L", lambda rec, c=c: fista_exact(P, pr, c / P.L, rec))
        go("0SR1 prox-QN (Becker-Fadili)", lambda rec: sr1_prox(P, pr, rec))
        go("Anderson5 on FB map (exact prox, 1/L)",
           lambda rec: anderson(P, lambda x: pr.prox(x - (P.Q(x) - P.b) / P.L, 1 / P.L), rec))
    for rf in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        for n_in in (3, 8):
            go(f"ADMM-PCG rho={rf}*bulk inner={n_in}", lambda rec, rf=rf, n_in=n_in: admm(P, pr, rf * bulk, n_in, rec))
    for c in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        Sig = [c * p for p in P.ong]
        go(f"PDHG Ong k-space Sigma={c}*p", lambda rec, Sig=Sig: pdhg(P, pr, Sig, rec))
    nA = np.sqrt(P.power(lambda v: P.Q(v, count=False), 30))
    for c in (1.0, 10.0, 100.0):
        go(f"PDHG scalar sigma={c}/||A||", lambda rec, c=c: pdhg(P, pr, c / nA, rec))
    res["seconds"] = time.time() - t0
    (out / f"nc_{prior}_R{R:g}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"done in {res['seconds']:.0f}s")
    return res


if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")
    out = Path(sys.argv[1]) / f"K{K}"
    out.mkdir(parents=True, exist_ok=True)
    Rs = [float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "5,10,25,50,100,200").split(",")]
    prior = sys.argv[3] if len(sys.argv) > 3 else "WT"
    quick = len(sys.argv) > 4 and sys.argv[4] == "quick"
    if quick:
        IT = 60
    for R in Rs:
        if (out / f"nc_{prior}_R{R:g}.json").exists():
            print(f"skip R={R:g} {prior} (done)", flush=True)
            continue
        if (out / "STOP").exists():
            print("STOP file present: stopping", flush=True)
            break
        run_R(R, prior, out, quick)
