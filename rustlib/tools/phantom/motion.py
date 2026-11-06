"""Motion of the atlas phantom: cardiac (`cardiac_motion.py`) composed with respiratory (`resp_motion.py`) kinematics.

    from motion import Motion, Breathing
    m = Motion(cardiac_hv, resp_hv)
    x = m.position(X, tc, r)            # atlas points X [N, 3] → positions at cardiac time tc (s from end-diastole) and
                                        # respiratory state r (0 end-expiration … 1 end-inspiration)
    v = m.velocity(X, tc, r, rdot)      # material velocity [N, 3] (m/s)
    X = m.inverse(x, tc, r)             # reference point of each spatial point (fixed-point iteration)
    b = Breathing(seed); r, rdot = b(t) # Lujan cos^{2n} breaths with breath-to-breath variability

x = Φ_r(Φ_c(X)): X_c = X + u_c(X, tc) (periodic cubic in time between the stored samples, trilinear in space, 0
outside the heart grid), x = X_c + r · S(X_c) (S: sum of the respiratory bases, trilinear, clamped at the grid edge).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402


class _Grid:
    def __init__(self, origin, h, dims):
        self.o, self.h, self.dims = np.asarray(origin, float), float(h), np.asarray(dims, int)

    def coords(self, X):
        """index coordinates (z, y, x order) for map_coordinates"""
        return ((np.asarray(X, float) - self.o) / self.h - 0.5)[:, ::-1].T

    def sample(self, F, X, mode):
        """F [3, Z, Y, X] at points X [N, 3] → [N, 3]"""
        c = self.coords(X)
        return np.stack([map_coordinates(F[k], c, order=1, mode=mode, cval=0.0) for k in range(3)], -1)


class Motion:
    def __init__(self, cardiac_hv, resp_hv=None, cardiac_scale=1.0, resp_scale=1.0):
        c = vf.read(cardiac_hv)
        self.U = np.asarray(c["U_mm"]).astype(np.float32) * 1e-3 * cardiac_scale  # [T, 3, Z, Y, X] m
        self.T = float(c["period"])
        self.nt = self.U.shape[0]
        self.gc = _Grid(c["origin_m"], c["voxel_m"], c["dims_xyz"])
        self.S = None
        if resp_hv is not None:
            r = vf.read(resp_hv)
            self.S = np.asarray(r["B"]).sum(0).astype(np.float32) * resp_scale  # [3, Z, Y, X]
            self.gr = _Grid(r["origin_m"], r["voxel_m"], r["dims_xyz"])

    def _weights(self, tc, deriv=False):
        """periodic Catmull-Rom weights over the samples around tc (and their time derivative)"""
        s = (tc % self.T) / self.T * self.nt
        i = int(np.floor(s))
        f = s - i
        if not deriv:
            w = [(-f**3 + 2 * f**2 - f) / 2, (3 * f**3 - 5 * f**2 + 2) / 2, (-3 * f**3 + 4 * f**2 + f) / 2, (f**3 - f**2) / 2]
        else:
            w = [(-3 * f**2 + 4 * f - 1) / 2, (9 * f**2 - 10 * f) / 2, (-9 * f**2 + 8 * f + 1) / 2, (3 * f**2 - 2 * f) / 2]
            w = [x * self.nt / self.T for x in w]
        return [((i - 1 + j) % self.nt, w[j]) for j in range(4)]

    def ucard_grid(self, tc, deriv=False):
        return sum(w * self.U[k] for k, w in self._weights(tc, deriv))

    def ucard(self, X, tc, deriv=False):
        return self.gc.sample(self.ucard_grid(tc, deriv), X, "constant")

    def uresp(self, X):
        return self.gr.sample(self.S, X, "nearest") if self.S is not None else np.zeros_like(np.asarray(X, float))

    def position(self, X, tc, r=0.0):
        Xc = np.asarray(X, float) + self.ucard(X, tc)
        return Xc + r * self.uresp(Xc) if r != 0.0 else Xc

    def velocity(self, X, tc, r=0.0, rdot=0.0, dt=2e-3):
        Uc = self.ucard(X, tc)
        Vc = self.ucard(X, tc, deriv=True)
        Xc = np.asarray(X, float) + Uc
        if self.S is None or (r == 0.0 and rdot == 0.0):
            return Vc
        Sx = self.uresp(Xc)
        dS = (self.uresp(Xc + dt * Vc) - self.uresp(Xc - dt * Vc)) / (2 * dt)
        return Vc + rdot * Sx + r * dS

    def inverse(self, x, tc, r=0.0, iters=12):
        x = np.asarray(x, float)
        X = x.copy()
        for _ in range(iters):
            X = x - (self.position(X, tc, r) - X)
        return X


class Breathing:
    """Lujan breaths r(t) = a_k cos^{2n}(π (t − c_k) / τ_k) inside each breath (peak = end-inspiration at c_k), with
    period τ_k ~ N(period, (cv_period · period)²) and depth a_k ~ N(1, cv_depth²), clipped to [0.5, 1.4]."""

    def __init__(self, seed=0, period=4.0, n=2, cv_period=0.1, cv_depth=0.12, t0=-10.0, t1=600.0):
        rng = np.random.default_rng(seed)
        self.n = n
        cs, taus, amps = [], [], []
        t = t0
        while t < t1:
            tau = max(0.6 * period, rng.normal(period, cv_period * period))
            cs.append(t + tau / 2)
            taus.append(tau)
            amps.append(float(np.clip(rng.normal(1.0, cv_depth), 0.5, 1.4)))
            t += tau
        self.c, self.tau, self.a = np.array(cs), np.array(taus), np.array(amps)
        self.edges = self.c - self.tau / 2

    def __call__(self, t):
        t = np.asarray(t, float)
        k = np.clip(np.searchsorted(self.edges, t, side="right") - 1, 0, len(self.c) - 1)
        ph = np.pi * (t - self.c[k]) / self.tau[k]
        cs, sn = np.cos(ph), np.sin(ph)
        r = self.a[k] * cs ** (2 * self.n)
        rdot = -self.a[k] * 2 * self.n * cs ** (2 * self.n - 1) * sn * np.pi / self.tau[k]
        return r, rdot
