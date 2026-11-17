"""Gridding kernels with derivatives up to order 4 (w.r.t. the grid coordinate t, support |t| < W/2).

ES  : psi(z) = exp(beta (sqrt(1-z^2) - 1))            (FINUFFT / segnufft.cu; derivatives sqrt-singular at |z| = 1)
KB  : psi(z) = I0(beta sqrt(1-z^2)) / I0(beta)        (entire in z on [-1, 1]; only a jump of 1/I0(beta) at the edge)
z = 2 t / W.  d/dt = (2/W) d/dz.
phihat(xi) = int phi(t) e^{i t xi} dt by Gauss-Legendre (even kernel -> real).
"""
import numpy as np
from scipy import special

PI = np.pi


def es_beta(W, sigma):  # production rule, rustlib/crates/hasty-nufft/src/kernel.rs
    return 2.30 * W if abs(sigma - 2) < 1e-12 else 0.97 * PI * (1 - 1 / (2 * sigma)) * W


def kb_beta(W, sigma):  # Beatty, Nishimura, Pauly 2005, eq. 5
    v = (W / sigma) ** 2 * (sigma - 0.5) ** 2 - 0.8
    return PI * np.sqrt(max(v, 1e-3))


def es_derivs(z, beta, clamp=0.98, nmax=4):
    """[psi, psi', .., psi''''] w.r.t. z. clamp: zero the derivatives for |z| > clamp (segnufft.cu uses 0.98)."""
    z = np.asarray(z, float)
    a = 1 - z * z
    ok = a > 0
    out = np.zeros((nmax + 1,) + z.shape)
    aa = np.where(ok, a, 1.0)
    s = np.sqrt(aa)
    f = np.where(ok, np.exp(beta * (s - 1)), 0.0)
    out[0] = f
    is1 = 1 / s; is3 = is1 / aa; is5 = is3 / aa; is7 = is5 / aa
    u1 = -beta * z * is1; u2 = -beta * is3; u3 = -3 * beta * z * is5; u4 = -3 * beta * (1 + 4 * z * z) * is7
    d = [None, u1 * f, (u2 + u1 ** 2) * f, (u3 + 3 * u1 * u2 + u1 ** 3) * f,
         (u4 + 4 * u1 * u3 + 3 * u2 ** 2 + 6 * u1 ** 2 * u2 + u1 ** 4) * f]
    keep = ok if clamp is None else (ok & (np.abs(z) <= clamp))
    for n in range(1, nmax + 1):
        out[n] = np.where(keep, d[n], 0.0)
    return out


def _Fnu(nu, u, beta):
    """F_nu(u) = I_nu(beta sqrt u) / (beta sqrt u)^nu / I0(beta), entire in u; dF_nu/du = beta^2/2 F_{nu+1}."""
    x = beta * np.sqrt(np.maximum(u, 0))
    small = x < 1e-3
    xs = np.where(small, 1.0, x)
    val = special.ive(nu, xs) * np.exp(xs - beta) / xs ** nu
    lim = (1 / (2 ** nu * special.factorial(nu))) * np.exp(-beta) / special.i0e(beta) * np.exp(-beta) * 0  # placeholder
    # series limit: I_nu(x)/x^nu -> 1/(2^nu nu!) ; divide by I0(beta) = i0e(beta) e^beta
    lim = 1 / (2 ** nu * special.factorial(nu)) / (special.i0e(beta) * np.exp(beta))
    return np.where(small, lim, val / special.i0e(beta))


def kb_derivs(z, beta, nmax=4):
    z = np.asarray(z, float)
    ok = np.abs(z) < 1
    u = np.where(ok, 1 - z * z, 0.0)
    F = [_Fnu(k, u, beta) for k in range(5)]
    b2 = beta ** 2
    d = [F[0], -b2 * z * F[1], -b2 * F[1] + b2 ** 2 * z ** 2 * F[2],
         3 * b2 ** 2 * z * F[2] - b2 ** 3 * z ** 3 * F[3],
         3 * b2 ** 2 * F[2] - 6 * b2 ** 3 * z ** 2 * F[3] + b2 ** 4 * z ** 4 * F[4]]
    return np.array([np.where(ok, d[n], 0.0) for n in range(nmax + 1)])


class Kernel:
    def __init__(self, kind, W, sigma, beta=None, clamp=0.98):
        self.kind, self.W, self.sigma, self.clamp = kind, W, sigma, clamp
        if beta is None:
            beta = es_beta(W, sigma) if kind == "es" else kb_beta(W, sigma)
        self.beta = beta

    def derivs_t(self, t, nmax=4):
        """derivatives w.r.t. grid coordinate t, shape (nmax+1,) + t.shape"""
        z = 2 * np.asarray(t, float) / self.W
        if self.kind == "es":
            d = es_derivs(z, self.beta, self.clamp, nmax)
        else:
            d = kb_derivs(z, self.beta, nmax)
        sc = (2 / self.W) ** np.arange(nmax + 1)
        return d * sc.reshape((-1,) + (1,) * z.ndim)

    def phihat(self, xi, nq=400):
        x, w = np.polynomial.legendre.leggauss(nq)
        t = 0.5 * self.W * 0.5 * (x + 1)          # [0, W/2]
        wt = 0.5 * self.W * 0.5 * w
        f = self.derivs_t(t, 0)[0]
        return 2 * (np.cos(np.outer(np.atleast_1d(xi), t)) @ (wt * f))
