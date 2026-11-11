"""End-to-end simulation of the GIRF QA protocol (girfqa output) on a QA sphere: truth chains with spatial terms
(self, cross, B0, 2nd / 3rd-order eddy fields), a thermal network driven by the protocol's own gradient power,
amplitude nonlinearity, receiver delay, phantom / shim / coil nuisances, noise.

Reuses the girf_sensitivity study's chain models (gmodel.py, truths.py) and its nuisance draw (thinslice.Nuisance);
the protocol (frames, inputs, timeline, flow-TR arrays, v2-validation layout) comes from the Rust generator.

Signal of one frame group (axis a, slice s at p_s, PE step k_q, test polarity σ, slice-select polarity η, input u,
level ℓ, thermal state θ), relative to the central slice's FID, averaged over the group's n frames:
  s(t) = A Σ_r w_r e^{i k_q·r⊥} P̂(σ k_a(t) + κ_r t + k_ss) e^{−t/T2} e^{−i[ψ_r(t) + σ Φ_r(t)]} / W_ref + n/√n_avg
  Φ_r(t) = Σ_l k_l(t) Y_l(r),  l ∈ {B0 (Y = 1), x, y, z, 2nd order (5), 3rd order (7)}, with r_a = p_s' + dp_r (the
  slice warp −η ΔB0(r)/(γ̄ G_ss) and the RF-frequency / G_ss-scale / per-slice errors); k_l = γ ∫ (h_{l,a} ∗ g_a)
  from the input start, g_a = σ ℓ u + β g|g|/G_ref (odd nonlinearity) through the chain at θ;
  ψ_r = 2π Δf(r) t (residual shim) + γ ∫ g_a² · q(r) (concomitant), σ-even.
Exact in-plane sums on a 4 mm grid (no linearisation), on the GPU in time chunks (≤ ~1 GB).
"""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
sys.path.insert(0, os.path.join(REPO, 'research/experiments/girf_sensitivity'))
import gmodel as gm  # noqa: E402
import truths  # noqa: E402
import thinslice as ts  # noqa: E402

GAM = gm.GAM
GBAR = gm.GBAR
B0 = 3.0
R0 = 0.1  # normalisation radius of the 2nd / 3rd-order basis (m)
DEV = torch.device("cpu")  # the GPU is shared; the CPU path below is the default

# ───────────────────────────────────────────────────────────── spatial basis (scanner coordinates, m)
# index: 0 B0, 1 x, 2 y, 3 z, 4..8 2nd order, 9..15 3rd order (real solid harmonics, unnormalised, / R0^(n-1))
BASIS = ['1', 'x', 'y', 'z', 'xy', 'xz', 'yz', 'x2-y2', '2z2-x2-y2',
         'x(x2-3y2)', 'y(3x2-y2)', 'z(2z2-3x2-3y2)', 'x(4z2-x2-y2)', 'y(4z2-x2-y2)', 'z(x2-y2)', 'xyz']
NB = len(BASIS)


def basis(r):
    """(N, 3) positions → (N, 16) basis values (m; 2nd order / R0, 3rd order / R0²)"""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    o = np.ones_like(x)
    b2 = [x * y, x * z, y * z, x * x - y * y, 2 * z * z - x * x - y * y]
    b3 = [x * (x * x - 3 * y * y), y * (3 * x * x - y * y), z * (2 * z * z - 3 * x * x - 3 * y * y), x * (4 * z * z - x * x - y * y),
          y * (4 * z * z - x * x - y * y), z * (x * x - y * y), x * y * z]
    return np.stack([o, x, y, z] + [v / R0 for v in b2] + [v / R0 ** 2 for v in b3], -1)


# ───────────────────────────────────────────────────────────── protocol (girfqa output)
class Proto:
    def __init__(self, d):
        self.d = d
        self.p = json.load(open(os.path.join(d, 'protocol.json')))
        self.cols = self.p['frame_columns']
        self.F = np.load(os.path.join(d, 'frames.npy'))
        self.inputs = np.load(os.path.join(d, 'inputs.npy'))
        self.in_names = [i['name'] for i in self.p['inputs']]
        self.in_len = [i['n'] for i in self.p['inputs']]
        self.power = np.load(os.path.join(d, 'power.npy'))
        c = self.p['config']
        self.dt = c['grad_raster']
        self.dwell = c['dwell']
        self.chunk = c['adc_chunk']
        self.slices = np.array(c['slices_m'])
        self.cfg = c
        self.parts = {q['name']: i for i, q in enumerate(self.p['parts'])}
        fv = os.path.join(d, 'vframe_in.npy')
        if os.path.exists(fv):
            self.v_in = np.load(fv)
            self.v_R = np.load(os.path.join(d, 'vframe_R.npy'))
            self.v_enc = np.load(os.path.join(d, 'vframe_enc.npy')).astype(int)
            self.ft_enc = np.load(os.path.join(d, 'flowtr_enc.npy'))
            self.ft_ro = np.load(os.path.join(d, 'flowtr_ro.npy'))
            self.ft_tail = np.load(os.path.join(d, 'flowtr_tail.npy'))

    def col(self, name):
        return self.cols.index(name)

    def input(self, i):
        return self.inputs[i, :self.in_len[i]]

    def adc_times(self, t_adc):
        """sample times (s, from the input-block start) of an ADC window split into chunks as girfqa does: chunk k in
        its own block, delay 42 µs (samples on the raster), dead time after; block durations on the 4 µs raster"""
        n_tot = int(round(t_adc / self.dwell))
        d_adc = 42e-6
        out, t0 = [], 0.0
        left = n_tot
        while left > 0:
            n = min(left, self.chunk)
            out.append(t0 + d_adc + (np.arange(n) + 0.5) * self.dwell)
            dur = d_adc + n * self.dwell + 40e-6
            t0 += np.ceil(dur / self.dt - 1e-9) * self.dt
            left -= n
        return np.concatenate(out)

    def flow_tr_played(self, e, R, sigma=1.0):
        """one history TR as played on GE (encode, gap, rotated readout, gap, tail, gap), physical axes"""
        gap = np.zeros((int(self.cfg['seg_gap_samples']), 3))
        return np.concatenate([sigma * self.ft_enc[e], gap, sigma * self.ft_ro @ R.T, gap, sigma * self.ft_tail[e], gap], 0)


# ───────────────────────────────────────────────────────────── thermal network
class Thermal:
    """θ(t) [°C above the cold start] = Σ_k κ_k (P ∗ e^{−t/τ_k}/τ_k)(t), P = Σ_a G_a² (T²/m²) from power.npy;
    κ_k set so that the flow-scan power gives a steady-state rise dT_ss split by frac"""

    def __init__(self, rng, p_flow, taus=None, dT_ss=None, frac=None, third=True):
        self.tau = np.array(taus if taus is not None else [rng.uniform(120, 360), rng.uniform(900, 2400)] + ([rng.uniform(4000, 8000)] if third else []))
        dT = dT_ss if dT_ss is not None else rng.uniform(4.0, 10.0)
        f = frac if frac is not None else rng.uniform(0.35, 0.65)
        w = np.array([f, 1 - f] + ([0.0] if third else []))
        if third:
            w = np.array([f * 0.85, (1 - f) * 0.85, 0.15])  # a slow 3rd compartment the fitted model does not have
        self.kappa = dT * w / p_flow
        self.p_flow = p_flow
        self.dT_ss = dT

    def theta(self, power, t):
        """θ at times t (s) from the 1-s power bins (exact recursion of the per-bin constant input)"""
        P = power[:, 1:].sum(1)
        n = len(P)
        th = np.zeros(n + 1)
        out = np.zeros(n + 1)
        for k in range(len(self.tau)):
            a = np.exp(-1.0 / self.tau[k])
            x = 0.0
            xs = np.zeros(n + 1)
            for i in range(n):
                x = x * a + self.kappa[k] * P[i] * (1 - a)
                xs[i + 1] = x
            out += xs
        th = out
        return np.interp(t, np.arange(n + 1), th)


# ───────────────────────────────────────────────────────────── truth: chain + spatial terms + thermal coefficients
class Truth:
    def __init__(self, seed, p_flow, kind='rand'):
        rng = np.random.default_rng(seed)
        self.g = truths.draw(np.random.default_rng(5000 + seed)) if kind == 'rand' else gm.variant(kind)
        if kind != 'rand':
            self.g.tail = 0.15
        # 2nd / 3rd-order eddy fields per input axis: high-pass terms, |k| ~ 1e-4 (2nd) / 3e-5 (3rd) of G at R0
        # [assumption, CHECK against a measured 3D GIRF; Rahmer 2019 found them small]
        self.hi = {a: [(l, gm.HighPass(truths.logu(rng, 0.3e-3, 3e-3), rng.normal(0, 1e-4 if l < 9 else 3e-5))) for l in range(4, NB)] for a in range(3)}
        # per-resonance thermal coefficients (relative f0 shift per °C, gain per °C), others as truths.THERMAL ×U[0.5, 1.5]
        self.c = {k: v * rng.uniform(0.5, 1.5) for k, v in truths.THERMAL.items()}
        self.cres = {}
        for a in range(3):
            for j, _ in enumerate(self._res(a)):
                self.cres[(a, j)] = (rng.normal(-1e-3, 0.3e-3), rng.normal(5e-3, 2e-3))
        self.thermal = Thermal(rng, p_flow)
        taus = [0.02]
        for a in range(3):
            taus += [1 / (2 * np.pi * z * f0) for f0, z, _ in self._res(a)]
            r = self.g.axis[a]
            for p_ in (r.parts if isinstance(r, gm.Mul) else [r]):
                b = p_.base if isinstance(p_, gm.MultiRes) else p_
                if isinstance(b, gm.Param):
                    taus += [t for _, t in b.eddy]
        self.pad_s = min(6 * max(taus) + 0.02, 1.0)
        # receiver: delay of the GIRF protocol's ADC (4 µs dwell) vs the gradient raster, and per dwell / coil
        self.rx = dict(d4=rng.uniform(-2e-6, 2e-6), slope=rng.uniform(0.5, 2.5), jitter={d: rng.normal(0, 50e-9) for d in (2, 4, 8, 16)},
                       coil_offset=rng.normal(0, 30e-9))
        self.seed = seed

    def _res(self, a):
        r = self.g.axis[a]
        if isinstance(r, gm.Mul):
            for p in r.parts:
                if isinstance(p, gm.MultiRes):
                    return p.res
        if isinstance(r, gm.MultiRes):
            return r.res
        if isinstance(r, gm.Resonant):
            return [(r.f0, r.zeta, r.gain)]
        return []

    def resonances(self, theta=0.0):
        """list of (axis, f0, zeta, gain) at θ"""
        out = []
        for a in range(3):
            for j, (f0, z, g) in enumerate(self._res(a)):
                cf, cg = self.cres[(a, j)]
                out.append((a, f0 * (1 + cf * theta), z, g * (1 + cg * theta)))
        return out

    def chain(self, theta):
        """gm.Girf at θ (self terms per truths.thermal with per-resonance f0 / gain coefficients; cross, B0 unchanged)"""
        c = self.c

        def th(r, a):
            if isinstance(r, gm.Param):
                return gm.Param(r.delay + c['c_d'] * theta, r.tau_lp * (1 + c['c_lp'] * theta), [(x * (1 + c['c_e'] * theta), t) for x, t in r.eddy])
            if isinstance(r, gm.MultiRes):
                res = [(f0 * (1 + self.cres[(a, j)][0] * theta), z, g * (1 + self.cres[(a, j)][1] * theta)) for j, (f0, z, g) in enumerate(r.res)]
                return gm.MultiRes(th(r.base, a), res)
            if isinstance(r, gm.Resonant):
                cf, cg = self.cres[(a, 0)]
                return gm.Resonant(th(r.base, a), r.f0 * (1 + cf * theta), r.zeta, r.gain * (1 + cg * theta))
            if isinstance(r, gm.Mul):
                return gm.Mul(*[th(p, a) for p in r.parts])
            return r
        return self.g.copy_with(axis=[th(r, a) for a, r in enumerate(self.g.axis)], name=f'{self.g.name}@{theta:+.3f}')

    def outputs(self, theta, a, g_in, dt, n_out=None):  # fields on the raster dt / UP (see k_exact)
        """output fields (T/m per unit basis; B0 term in T) on the raster of g_in (n,) zero-padded to n_out samples, for
        all 16 basis functions: returns (PRE + n_out, 16) (raster from PRE samples before g_in[0]); the response tail
        after the input is included up to n_out. Only the responses driven by
        axis a are evaluated (self term at θ, cross terms (o, a), B0 term a, 2nd / 3rd order of a), each through the
        piecewise-linear alias sum (12 alias terms; the remainder is < 1e-5)."""
        if n_out is not None and n_out > len(g_in):
            g_in = np.r_[g_in, np.zeros(n_out - len(g_in))]
        # internal raster dt/UP: the piecewise-linear input is exact there, and the output fields' piecewise-linear
        # interpolation at sub-raster sample times (receiver delay) errs UP² less than on the 4 µs raster
        if UP > 1:
            g_in = np.interp(np.arange((len(g_in) - 1) * UP + 1) / UP, np.arange(len(g_in)), g_in)
            dt = dt / UP
        n = len(g_in)
        L = gm.PRE + n
        # zero padding ≥ 6 ring-down times of the narrowest resonance and of the longest eddy term (no circular
        # wrap-around of the response tail into the window)
        nfft = 1 << int(np.ceil(np.log2(L + int(np.ceil(self.pad_s / dt)))))
        x = np.zeros(nfft)
        x[gm.PRE:gm.PRE + n] = g_in
        if self.g.nl:
            x = x + self.g.nl * x * np.abs(x) / self.g.nl_ref
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(nfft, dt)
        f = np.zeros((L, NB))
        key = ('self', a, round(theta, 6), nfft, dt)
        if key not in SPEC:
            if len(SPEC) > 64:
                SPEC.clear()
            SPEC[key] = pl12(self.chain(theta).axis[a], fr, dt)
        f[:, a + 1] = np.fft.irfft(X * SPEC[key], nfft)[:L]
        for o, i, r in self.g.cross:
            if i == a:
                f[:, o + 1] += np.fft.irfft(X * pl12c(r, fr, dt, nfft), nfft)[:L]
        if self.g.b0[a] is not None:
            f[:, 0] = np.fft.irfft(X * pl12c(self.g.b0[a], fr, dt, nfft), nfft)[:L]
        for l, h in self.hi[a]:
            f[:, l] = np.fft.irfft(X * pl12c(h, fr, dt, nfft), nfft)[:L]
        return f




UP = 4
CONC = 1.0  # concomitant-field switch (ablations)
KAP_MEAN = False  # replace the in-plane varying through-slice B0 gradient by its slice mean (ablation)
NO_WARP = False  # no in-plane slice warp (ablation)


def k_exact(f, dt, t, t_from):
    """γ ∫_{t_from}^{t} of piecewise-linear fields f returned by Truth.outputs (raster dt/UP from −PRE·dt/UP), exact
    at arbitrary times (gm.k_at); dt = the protocol raster"""
    d = dt / UP
    return gm.k_at(f, d, -gm.PRE * d, t, t_from)


SPEC = {}
_PLC = {}


def pl12(r, f, dt):
    return gm.at_pl(r, f, dt, M=12)


def pl12c(r, f, dt, nfft):
    key = (r.key(), nfft, dt)
    if key not in _PLC:
        if len(_PLC) > 400:
            _PLC.clear()
        _PLC[key] = pl12(r, f, dt)
    return _PLC[key]


# ───────────────────────────────────────────────────────────── phantom / session nuisances
class Session:
    def __init__(self, seed, prot, snr_body=300.0, snr_array=1000.0, **kw):
        rng = np.random.default_rng(10000 + seed)
        tp = ts.Protocol()
        tp.slices = prot.slices
        self.nu = ts.Nuisance(rng, tp)
        self.nu.dp_rand = rng.normal(0, 5e-6, (3, len(prot.slices)))
        self.snr_body, self.snr_array = snr_body, snr_array
        # prescan errors: centre (projections) 0.1 mm, B0 map 0.3 Hz rms smooth, slice positions 1 µm, ρ map 1 % noise
        self.c_err = rng.normal(0, 0.1e-3, 3)
        self.b0_err = rng.normal(size=16) * 0.3 / 1.0
        self.p_err = rng.normal(0, 1e-6, (3, len(prot.slices)))
        self.rng = rng
        off = kw.pop('nuis_off', None)
        if off:
            z = dict(center=np.zeros(3), b0_rms_hz=1e-6, coilw=np.zeros(3), df0_hz=0.0, eps_ss=0.0, dp_rand=np.zeros((3, len(prot.slices))), k_ss=0.0)
            for k in (z if off == 'all' else {}):
                setattr(self.nu, k, z[k])
            for k in off.split(',') if off != 'all' else []:
                setattr(self.nu, k, z[k])
            self.nu._norm = None
        self.__dict__.update(kw)


def slice_grid(nu, a, p, h=4e-3):
    """in-plane grid of the slice p through the sphere (scanner coordinates) and receive weights"""
    return ts.slice_points(nu, None, a, p, h=h)


# ───────────────────────────────────────────────────────────── signal simulation (GPU)
def adc_delay(dwell, dead=40e-6, raster=4e-6, rf_raster=2e-6):
    """girfqa adc_delay_on_raster: smallest delay ≥ dead on the RF raster with (delay + dwell/2) on the gradient
    raster, else the smallest RF-raster delay ≥ dead"""
    d0 = np.ceil(dead / rf_raster - 1e-9) * rf_raster
    d = d0
    while d < d0 + 100e-6:
        x = (d + 0.5 * dwell) / raster
        if abs(x - round(x)) < 1e-6:
            return d
        d += rf_raster
    return d0


def adc_times(t_adc, dwell, chunk=12500, raster=4e-6):
    """sample times (s, from the input-block start) of a chunked ADC window as girfqa writes it"""
    n_tot = int(round(t_adc / dwell))
    d0 = adc_delay(dwell)
    out, t0, left = [], 0.0, n_tot
    while left > 0:
        n = min(left, chunk)
        out.append(t0 + d0 + (np.arange(n) + 0.5) * dwell)
        t0 += np.ceil((d0 + n * dwell + 40e-6) / raster - 1e-9) * raster
        left -= n
    return np.concatenate(out)


def rx_delay(truth, dwell, coil):
    """receiver timing of the samples vs the gradient raster (s)"""
    d = int(round(dwell * 1e6))
    return truth.rx['d4'] + truth.rx["slope"] * (dwell - 4e-6) + truth.rx['jitter'].get(d, 0.0) - truth.rx['jitter'][4] + (truth.rx['coil_offset'] if coil == 'array' else 0.0)


class Geo:
    """per-slice in-plane grids, weights, basis values, field, warp (truth) for one session"""

    def __init__(self, sess, prot, thk_ss):
        self.nu = sess.nu
        self.cache = {}
        self.prot = prot
        _, w0 = slice_grid(self.nu, 2, 0.0)
        self.wref = w0.sum()

    def get(self, a, s, eta, gss):
        key = (a, s, eta, round(gss, 9))
        if key in self.cache:
            return self.cache[key]
        nu = self.nu
        p = self.prot.slices[s]
        pos, wt = slice_grid(nu, a, p)
        pc = pos - nu.center
        fz = nu.field_hz(pc)
        fbar = (wt * fz).sum() / wt.sum()
        dp = -eta * (fz / GBAR) / gss
        p_true = p * (1 + nu.eps_ss) + nu.dp_rand[a, s] + eta * (-(fbar + nu.df0_hz) / GBAR) / gss
        r3 = pos.copy()
        r3[:, a] = p_true + (dp - (wt * dp).sum() / wt.sum())
        e_a = np.zeros(3)
        e_a[a] = 1e-3
        g = dict(pos=pos, wt=wt, fz=fz, fbar=fbar, p_true=p_true, Y=basis(r3), dw=2 * np.pi * (fz - fbar),
                 q=CONC * ((pos[:, 0] ** 2 + pos[:, 1] ** 2) / (8 * B0) if a == 2 else pos[:, 2] ** 2 / (2 * B0)),
                 kap=2 * np.pi * (nu.field_hz(pc + e_a) - fz) / 1e-3)
        if KAP_MEAN:
            g['kap'] = np.full_like(g['kap'], (wt * g['kap']).sum() / wt.sum())
        if NO_WARP:
            g['Y'] = basis(np.where(np.arange(3)[None, :] == a, p_true, pos))
        self.cache[key] = g
        return g


def second_order_set(nl):
    """columns of Φ' kept at 2nd order: the 2 cross terms, the five 2nd-order terms and the warp (the 3rd-order terms,
    |k| ~ 3e-5, enter at 1st order only: their square is < 1e-6 rad)"""
    return list(range(0, 7)) + [nl - 1]


def _pfun(x, tbw=4.0):
    """slice-profile FT (Hann-windowed sinc) and its derivative in x = k thk / 2π"""
    w = np.where(np.abs(x) <= tbw / 2, 0.5 * (1 + np.cos(2 * np.pi * x / tbw)), 0.0)
    dw = np.where(np.abs(x) <= tbw / 2, -0.5 * np.sin(2 * np.pi * x / tbw) * 2 * np.pi / tbw, 0.0)
    sc = np.sinc(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        dsc = np.where(np.abs(x) < 1e-8, 0.0, (np.cos(np.pi * x) - sc) / x)
    return sc * w, dsc * w + sc * dw


IN_PLANE = None


def sim_group(truth, sess, geo, prot, a, s, eta, g_in, off, theta, t_in0, tn, navg, pes, thk, snr0, dwell, coil, sigmas=(1.0, -1.0), kcache=None, acache=None, dt_dec=1e-4):
    """averaged complex signals of one frame group for both test polarities and every PE step:
    returns (n_sigma, n_pe, n_t). g_in: raster waveform (σ = +1) incl. any history, `off` = samples before the
    measured input block's start; tn: sample times from that start; pes: list of (2,) in-plane k (rad/m) or None.
    In-plane sum exact to 2nd order in the small in-plane phase Φ' (cross, 2nd / 3rd-order terms, slice warp;
    |Φ'| ≲ 0.05 rad, 3rd-order remainder ≲ 2e-5) and to 1st order in the through-slice off-resonance κ t;
    the static phase (shim, concomitant) on a 0.1 ms grid, everything fast at the full sample rate."""
    dt = prot.dt
    gss = prot.cfg['g_ss'] if abs(thk - prot.cfg['thk']) < 1e-9 else prot.cfg['g_ss_v']
    n_out = off + int(np.ceil(tn[-1] / dt)) + 16
    key = (id(g_in), round(theta, 6), a, n_out)
    if kcache is not None and key in kcache:
        k = kcache[key]
    else:
        k = truth.outputs(theta, a, g_in, dt, n_out)
        if kcache is not None:
            kcache[key] = k
    t_s = tn + off * dt + rx_delay(truth, dwell, coil)
    # phase accrues from the excitation (RF centre, t_in0 before the measured block): history before it is not seen;
    # exact integration of the piecewise-linear fields at the (sub-raster) sample times
    t_exc = off * dt - t_in0
    kt = k_exact(k, dt, t_s, t_exc)
    G = geo.get(a, s, eta, gss)
    others = [b for b in range(3) if b != a]
    lin = [l for l in range(NB) if l not in (0, a + 1)]  # in-plane / higher-order basis + the warp (via Y_a below)
    t_rel = t_in0 + tn
    akey = (id(g_in), a, s, eta, tuple(None if pe is None else tuple(pe) for pe in pes), len(tn), round(gss, 9))
    if acache is not None and akey in acache:
        A0, A1, A2, Ak0, Ak1, pbar = acache[akey]
    else:
        gg = np.zeros(len(g_in) + 1)
        gg[1:] = np.cumsum(0.5 * (np.r_[0, g_in[1:] ** 2] + np.r_[0, g_in[:-1] ** 2])) * dt
        tg = np.arange(len(gg)) * dt - dt
        td = np.arange(t_rel[0] - dt_dec, t_rel[-1] + 2 * dt_dec, dt_dec)
        conc_d = GAM * (np.interp(td - t_in0 + off * dt, tg, gg) - np.interp(off * dt, tg, gg))
        wt = G['wt']
        pbar = G['p_true']
        Yp = G['Y'][:, lin].copy()
        warp = G['Y'][:, a + 1] - pbar  # the slice warp enters through the self term k_a
        Yp = np.concatenate([Yp, warp[:, None]], 1)  # Nr x L'
        nl = Yp.shape[1]
        sq = second_order_set(nl)
        iu = np.triu_indices(len(sq))
        Ys = Yp[:, sq]
        feat = np.concatenate([np.ones((len(wt), 1)), Yp, Ys[:, iu[0]] * Ys[:, iu[1]] * np.where(iu[0] == iu[1], 1.0, 2.0)[None, :]], 1)
        featk = np.concatenate([np.ones((len(wt), 1)), Yp], 1) * G['kap'][:, None]
        E = np.exp(-1j * (td[:, None] * G['dw'][None, :] + conc_d[:, None] * G['q'][None, :]))  # T x Nr
        A0, A1, A2, Ak0, Ak1 = [], [], [], [], []
        for pe in pes:
            w = wt if pe is None else wt * np.exp(1j * (pe[0] * G['pos'][:, others[0]] + pe[1] * G['pos'][:, others[1]]))
            Ew = E * w[None, :]
            M = Ew @ feat  # T x (1 + L' + L'(L'+1)/2)
            Mk = Ew @ featk
            ip = lambda Z: np.stack([np.interp(t_rel, td, Z[:, j].real) + 1j * np.interp(t_rel, td, Z[:, j].imag) for j in range(Z.shape[1])], 1)
            Mi = ip(M).astype(np.complex64)
            Mki = ip(Mk).astype(np.complex64)
            A0.append(Mi[:, 0])
            A1.append(Mi[:, 1:1 + nl])
            A2.append(Mi[:, 1 + nl:])
            Ak0.append(Mki[:, 0] * t_rel.astype(np.float32))
            Ak1.append(Mki[:, 1:] * t_rel.astype(np.float32)[:, None])
        if acache is not None:
            acache[akey] = (A0, A1, A2, Ak0, Ak1, pbar)
    nl = A1[0].shape[1]
    sq = second_order_set(nl)
    iu = np.triu_indices(len(sq))
    out = np.zeros((len(sigmas), len(pes), len(tn)), complex)
    for i, sg in enumerate(sigmas):
        kl = (np.concatenate([kt[:, lin], kt[:, a + 1:a + 2]], 1) * sg).astype(np.float32)  # σ k_l for Φ' (incl. warp coefficient k_a)
        common = np.exp(-1j * sg * (kt[:, a + 1] * pbar + kt[:, 0])) * np.exp(-t_rel / sess.nu.T2) * np.exp(-2j * np.pi * G['fbar'] * t_rel) / geo.wref * (thk / 1.5e-3)
        x = (sg * kt[:, a + 1] + sess.nu.k_ss) * thk / (2 * np.pi)
        P, dP = _pfun(x)
        dP = dP * thk / (2 * np.pi)  # d/dk
        ks = kl[:, sq]
        kk = ks[:, iu[0]] * ks[:, iu[1]]
        for j in range(len(pes)):
            v = P * (A0[j] - 1j * (A1[j] * kl).sum(1) - 0.5 * (A2[j] * kk).sum(1)) + dP * (Ak0[j] - 1j * (Ak1[j] * kl).sum(1))
            out[i, j] = common * v
        sig = 1.0 / (snr0 * np.sqrt(navg))
        out[i] += sig * (sess.rng.normal(size=out[i].shape) + 1j * sess.rng.normal(size=out[i].shape)) / np.sqrt(2)
    return out
