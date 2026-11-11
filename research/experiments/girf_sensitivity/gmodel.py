"""Python mirror of rustlib/crates/hasty-seq/src/girf.rs (Response / Girf / apply / played moments) plus the extra
truth effects of the GIRF sensitivity study (narrow resonances, smooth ripple tables, 2nd-order low-pass, thermal
drift, an odd amplitude nonlinearity).

Conventions as in girf.rs: H(f) per axis (self), cross terms (out, in, H), B0 terms per input axis (T per T/m).
Application: every response through the zero-padded FFT of the *piecewise-linear* waveform with the alias sum
H_pl(f) = sum_m sinc^2(f dt + m) H(f + m/dt) (girf.rs `Response::at_pl`; the Rust time-domain path for Parametric
agrees with it to 1e-7 without delay and to ~0.5 % of G at corners with a fractional delay, see girf.rs tests). The
output raster starts PRE samples before the input.
"""
import numpy as np

GBAR = 42.577478e6
GAM = 2 * np.pi * GBAR
PRE = 32


# ───────────────────────────────────────────────────────────── responses
def _k(x):
    if isinstance(x, Resp):
        return x.key()
    if isinstance(x, np.ndarray):
        return f'arr{x.shape}:{hash(x.tobytes())}'
    if isinstance(x, (list, tuple)):
        return '[' + ','.join(_k(y) for y in x) + ']'
    return repr(x)


class Resp:
    def at(self, f):
        raise NotImplementedError

    def key(self):
        """exact identity of the response (recursive; arrays by content hash), used by the at_pl cache"""
        return type(self).__name__ + '(' + ','.join(f'{k}={_k(v)}' for k, v in sorted(self.__dict__.items())) + ')'

    def __repr__(self):
        return f'{type(self).__name__}({self.__dict__})'


class Ideal(Resp):
    def at(self, f):
        return np.ones_like(np.asarray(f, float), dtype=complex)


class Param(Resp):
    """e^{-i w delay} / (1 + i w tau_lp) [1 - sum a_k (i w tau_k)/(1 + i w tau_k)]"""

    def __init__(self, delay, tau_lp, eddy=()):
        self.delay, self.tau_lp, self.eddy = float(delay), float(tau_lp), [tuple(map(float, e)) for e in eddy]

    def at(self, f):
        w = 2 * np.pi * np.asarray(f, float)
        h = np.exp(-1j * w * self.delay) / (1 + 1j * w * self.tau_lp)
        e = np.ones_like(h)
        for a, tau in self.eddy:
            e = e - a * (1j * w * tau) / (1 + 1j * w * tau)
        return h * e


class Resonant(Resp):
    """H_base (1 + g 2i zeta w0 w / (w0^2 - w^2 + 2i zeta w0 w))"""

    def __init__(self, base, f0, zeta, gain):
        self.base, self.f0, self.zeta, self.gain = base, float(f0), float(zeta), float(gain)

    def at(self, f):
        w, w0 = 2 * np.pi * np.asarray(f, float), 2 * np.pi * self.f0
        bp = (2j * self.zeta * w0 * w) / (w0 * w0 - w * w + 2j * self.zeta * w0 * w)
        return self.base.at(f) * (1 + self.gain * bp)


class MultiRes(Resp):
    """base x prod_j (1 + g_j BP_j): several resonances on one axis"""

    def __init__(self, base, res):
        self.base, self.res = base, [tuple(map(float, r)) for r in res]  # (f0, zeta, gain)

    def at(self, f):
        w = 2 * np.pi * np.asarray(f, float)
        h = self.base.at(f)
        for f0, z, g in self.res:
            w0 = 2 * np.pi * f0
            h = h * (1 + g * (2j * z * w0 * w) / (w0 * w0 - w * w + 2j * z * w0 * w))
        return h


class Scaled(Resp):
    def __init__(self, base, k):
        self.base, self.k = base, float(k)

    def at(self, f):
        return self.base.at(f) * self.k


class HighPass(Resp):
    """k (i w tau)/(1 + i w tau): eddy-shaped cross / B0 term"""

    def __init__(self, tau, k):
        self.tau, self.k = float(tau), float(k)

    def at(self, f):
        iwt = 2j * np.pi * np.asarray(f, float) * self.tau
        return self.k * iwt / (1 + iwt)


class MultiHP(Resp):
    """sum of high-pass terms (k_j, tau_j)"""

    def __init__(self, terms):
        self.terms = [tuple(map(float, t)) for t in terms]  # (k, tau)

    def at(self, f):
        f = np.asarray(f, float)
        h = np.zeros(f.shape, complex)
        for k, tau in self.terms:
            iwt = 2j * np.pi * f * tau
            h = h + k * iwt / (1 + iwt)
        return h


class Table(Resp):
    """linear interpolation of re / im on |f| (ascending, >= 0), Hermitian, clamped at the ends (girf.rs semantics)"""

    def __init__(self, f, h):
        self.f, self.h = np.asarray(f, float), np.asarray(h, complex)

    def at(self, f):
        f = np.asarray(f, float)
        fa = np.abs(f)
        v = np.interp(fa, self.f, self.h.real) + 1j * np.interp(fa, self.f, self.h.imag)
        return np.where(f >= 0, v, np.conj(v))

    def key(self):
        return f'Table({hash(self.f.tobytes())},{hash(self.h.tobytes())})'


class Mul(Resp):
    """product of responses (e.g. a base chain x a smooth ripple / LP2 factor)"""

    def __init__(self, *parts):
        self.parts = list(parts)

    def at(self, f):
        h = np.ones(np.shape(f), complex)
        for p in self.parts:
            h = h * p.at(f)
        return h


class Add(Resp):
    def __init__(self, *parts):
        self.parts = list(parts)

    def at(self, f):
        h = np.zeros(np.shape(f), complex)
        for p in self.parts:
            h = h + p.at(f)
        return h


class LP2(Resp):
    """second-order low-pass 1/(1 + i w/(Q w_c) - (w/w_c)^2) (what a first-order fit cannot represent exactly)"""

    def __init__(self, fc, q):
        self.fc, self.q = float(fc), float(q)

    def at(self, f):
        x = np.asarray(f, float) / self.fc
        return 1.0 / (1 + 1j * x / self.q - x * x)


class Ripple(Resp):
    """smooth multiplicative ripple 1 + sum_j c_j phi_j(f) (complex c_j, Gaussian bumps in f, Hermitian, = 1 at DC)"""

    def __init__(self, centers, width, coef):
        self.c, self.w, self.coef = np.asarray(centers, float), float(width), np.asarray(coef, complex)

    def at(self, f):
        f = np.asarray(f, float)
        fa = np.abs(f)
        b = np.exp(-0.5 * ((fa[..., None] - self.c) / self.w) ** 2) - np.exp(-0.5 * (self.c / self.w) ** 2)
        v = 1 + (b * self.coef).sum(-1)
        return np.where(f >= 0, v, np.conj(v))

    def key(self):
        return f'Ripple({hash(self.c.tobytes())},{self.w},{hash(self.coef.tobytes())})'


def sinc2(x):
    return np.sinc(x) ** 2


def at_pl(resp, f, dt, M=40):
    """alias sum of girf.rs at_pl"""
    if isinstance(resp, Ideal):
        return np.ones(np.shape(f), complex)
    f = np.asarray(f, float)
    h = np.zeros(f.shape, complex)
    for m in range(-M, M + 1):
        w = sinc2(f * dt + m)
        h += resp.at(f + m / dt) * w
    return h


_PL = {}


def _pl_cached(r, f, dt, nfft):
    """at_pl on an rfft grid, cached across chains by the response's key (models that share responses)"""
    key = (r.key(), nfft, dt)
    if key not in _PL:
        if len(_PL) > 400:
            _PL.clear()
        _PL[key] = at_pl(r, f, dt)
    return _PL[key]


# ───────────────────────────────────────────────────────────── chain
class Girf:
    def __init__(self, axis, cross=(), b0=(None, None, None), tail=30e-3, name='', nl=0.0, nl_ref=0.07):
        self.axis = list(axis)
        self.cross = list(cross)  # (out, in, resp)
        self.b0 = list(b0)
        self.tail = tail
        self.name = name
        # odd amplitude nonlinearity: input g -> g + nl * g |g| / nl_ref before the chain (non-LTI; survives the +/- difference)
        self.nl, self.nl_ref = nl, nl_ref
        self._cache = {}

    def key(self):
        return repr([a.key() for a in self.axis]) + repr([(o, i, r.key()) for o, i, r in self.cross]) + repr([b.key() if b else None for b in self.b0]) + f'{self.nl}'

    def spectra(self, nfft, dt):
        """H_pl on the rfft grid of nfft for every response (cached)"""
        k = (nfft, dt)
        if k not in self._cache:
            f = np.fft.rfftfreq(nfft, dt)
            ax = [None if isinstance(r, Ideal) else _pl_cached(r, f, dt, nfft) for r in self.axis]
            cr = [(o, i, _pl_cached(r, f, dt, nfft)) for o, i, r in self.cross]
            b0 = [None if r is None else _pl_cached(r, f, dt, nfft) for r in self.b0]
            self._cache[k] = (ax, cr, b0)
        return self._cache[k]

    def apply(self, g, dt, ntail=None, nfft=None):
        """g: (n, 3) nominal raster samples (PL). Returns (g_out (PRE + n + ntail, 3), b0 (same, T)), raster from PRE
        samples before g[0]."""
        g = np.asarray(g, float)
        n = len(g)
        if ntail is None:
            ntail = int(np.ceil(self.tail / dt))
        L = PRE + n + ntail
        if nfft is None:
            nfft = 1 << int(np.ceil(np.log2(2 * L)))
        x = np.zeros((nfft, 3))
        x[PRE:PRE + n] = g
        if self.nl:
            x = x + self.nl * x * np.abs(x) / self.nl_ref
        X = np.fft.rfft(x, axis=0)
        ax, cr, b0s = self.spectra(nfft, dt)
        Y = np.empty_like(X)
        for a in range(3):
            Y[:, a] = X[:, a] if ax[a] is None else X[:, a] * ax[a]
        for o, i, h in cr:
            Y[:, o] += X[:, i] * h
        B = np.zeros(X.shape[0], complex)
        for a in range(3):
            if b0s[a] is not None:
                B += X[:, a] * b0s[a]
        y = np.fft.irfft(Y, nfft, axis=0)[:L]
        b = np.fft.irfft(B, nfft)[:L]
        return y, b

    # ── parameter views used by the estimator / sensitivity code
    def copy_with(self, **kw):
        d = dict(axis=self.axis, cross=self.cross, b0=self.b0, tail=self.tail, name=self.name, nl=self.nl, nl_ref=self.nl_ref)
        d.update(kw)
        return Girf(**d)


def ideal():
    return Girf([Ideal(), Ideal(), Ideal()], tail=0.0, name='ideal')


def clinical_example():
    p = lambda d: Param(d, 8e-6, [(0.002, 3e-3), (0.001, 50e-6)])
    return Girf([p(2.0e-6), p(2.5e-6), p(3.0e-6)], tail=1e-3, name='clinical')


DELAYS = [2.0e-6, 2.6e-6, 3.1e-6]
LPS = [7e-6, 8e-6, 9e-6]
EDDY = [[(0.0025, 2.5e-3), (0.001, 50e-6)], [(0.002, 3e-3), (0.0012, 80e-6)], [(0.003, 1e-3), (0.0008, 40e-6)]]
CROSS = [(0, 2, (5e-4, 1e-3)), (1, 0, (5e-4, 2e-3)), (2, 1, (4e-4, 1.5e-3))]
B0T = [(2e-5, 2e-3), (1.5e-5, 4e-3), (3e-5, 1e-3)]
RES = [(1400.0, 0.03, 0.06), (1250.0, 0.03, 0.08), (900.0, 0.03, 0.05)]


def variants():
    """girf.rs Girf::variants()"""
    par = lambda a, lp, ed: Param(DELAYS[a], LPS[a] if lp else 0.0, EDDY[a] if ed else [])
    mk = lambda lp, ed: [par(a, lp, ed) for a in range(3)]
    cross = [(o, i, HighPass(t, k)) for o, i, (k, t) in CROSS]
    b0 = [HighPass(t, k) for k, t in B0T]
    tail = 30e-3
    out = [('delays', Girf(mk(False, False), tail=tail)), ('lowpass', Girf(mk(True, False), tail=tail)),
           ('eddy', Girf(mk(True, True), tail=tail)), ('cross', Girf(mk(True, True), cross=cross, tail=tail)),
           ('b0eddy', Girf(mk(True, True), cross=cross, b0=b0, tail=tail))]
    ax = [Resonant(par(a, True, True), *RES[a]) for a in range(3)]
    out.append(('resonance', Girf(ax, cross=cross, b0=b0, tail=tail)))
    for n, g in out:
        g.name = n
    return out


def variant(name):
    return dict(variants())[name]


# ───────────────────────────────────────────────────────────── moments of a played waveform
def cum_k(g, dt):
    """gamma * cumulative integral of a PL raster waveform at the raster points (rad/m), k[0] = 0"""
    k = np.zeros_like(g)
    k[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]), 0) * dt * GAM
    return k


def k_at(g, dt, t0, times, t_from):
    """gamma * int_{t_from}^{t} g (PL raster g starting at t0) at `times` (exact for PL)"""
    k = cum_k(g, dt)

    def kk(t):
        u = (np.asarray(t, float) - t0) / dt
        i = np.clip(np.floor(u).astype(int), 0, len(g) - 2)
        s = (u - i)[..., None]
        gi, gj = g[i], g[i + 1]
        return k[i] + GAM * dt * (gi * s + 0.5 * (gj - gi) * s * s)

    return kk(times) - kk(np.array([t_from]))[0]


def phase_at(b, dt, t0, times, t_from):
    """gamma * int b dt (rad) between t_from and times (PL)"""
    return k_at(b[:, None], dt, t0, times, t_from)[:, 0]


# ───────────────────────────────────────────────────────────── single-class model errors (girf.rs `perturbations`)
CLASSES = ['delay', 'lowpass', 'eddy_a', 'eddy_tau', 'cross', 'b0', 'res_f0', 'res_zeta', 'res_gain']


def perturb_resp(r, cls, eps, role):
    """girf.rs perturbations(): one class scaled by (1 + eps) (res_f0 by 1 + eps/10); role = which class a HighPass
    amplitude belongs to ('cross' or 'b0'); absolute variants handled by the callers"""
    s = lambda on: (1 + eps) if on else 1.0
    if isinstance(r, Param):
        return Param(r.delay * s(cls == 'delay'), r.tau_lp * s(cls == 'lowpass'), [(a * s(cls == 'eddy_a'), t * s(cls == 'eddy_tau')) for a, t in r.eddy])
    if isinstance(r, HighPass):
        return HighPass(r.tau * s(cls == 'eddy_tau'), r.k * s(cls == role))
    if isinstance(r, MultiHP):
        return MultiHP([(k * s(cls == role), t * s(cls == 'eddy_tau')) for k, t in r.terms])
    if isinstance(r, Resonant):
        return Resonant(perturb_resp(r.base, cls, eps, role), r.f0 * ((1 + 0.1 * eps) if cls == 'res_f0' else 1), r.zeta * s(cls == 'res_zeta'), r.gain * s(cls == 'res_gain'))
    if isinstance(r, MultiRes):
        return MultiRes(perturb_resp(r.base, cls, eps, role), [(f0 * ((1 + 0.1 * eps) if cls == 'res_f0' else 1), z * s(cls == 'res_zeta'), g * s(cls == 'res_gain')) for f0, z, g in r.res])
    if isinstance(r, Mul):
        return Mul(*[perturb_resp(p, cls, eps, role) for p in r.parts])
    return r


def perturbed_class(g, cls, eps):
    return g.copy_with(axis=[perturb_resp(r, cls, eps, 'delay') for r in g.axis],
                       cross=[(o, i, perturb_resp(r, cls, eps, 'cross')) for o, i, r in g.cross],
                       b0=[None if r is None else perturb_resp(r, cls, eps, 'b0') for r in g.b0], name=f'{g.name}~{cls}{eps:+g}')


def with_delay_offset(g, d):
    """the same chain with every axis delayed by d (s) more (d = array of 3 or scalar): an ADC / receiver timing
    mismatch looks exactly like this"""
    d = np.broadcast_to(np.asarray(d, float), (3,))
    return g.copy_with(axis=[Mul(r, Param(d[a], 0.0)) for a, r in enumerate(g.axis)], name=f'{g.name}+delay')


def with_gain(g, eps):
    eps = np.broadcast_to(np.asarray(eps, float), (3,))
    return g.copy_with(axis=[Scaled(r, 1 + eps[a]) for a, r in enumerate(g.axis)], name=f'{g.name}*gain')


def without(g, what):
    """the chain with a class removed (what = 'cross' | 'b0' | 'res')"""
    if what == 'cross':
        return g.copy_with(cross=[], name=g.name + '-cross')
    if what == 'b0':
        return g.copy_with(b0=[None, None, None], name=g.name + '-b0')
    if what == 'res':
        ax = [r.base if isinstance(r, (Resonant, MultiRes)) else r for r in g.axis]
        return g.copy_with(axis=ax, name=g.name + '-res')
    raise ValueError(what)
