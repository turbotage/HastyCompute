"""Thin-slice GIRF measurement (Duyn 1998 / Rahmer 2019 / Kronthaler 2021 / Bacon 2026) simulated end to end, and
its estimator.

Measurement (per gradient axis a): thin slices along a at positions p_s, test input on axis a (self + B0 terms), both
polarities sigma = +-1, FID from the slice excitation (t = 0) for T_adc, dwell 4 us (GE gradient raster), averages.
Signal of one frame (slice s, input u, polarity sigma), relative to the full slice magnetisation:
  s(t) = A_s e^{-t/T2} Phat(sigma k_a(t) + k_ss + kappa_s t) I_s(t; sigma) e^{-i [sigma (k_a(t) p_s' + phi0(t)) + psi_s(t)]} + n
  k_a = gamma int g_out,a (truth chain, rad/m); phi0 = gamma int b0_out (B0 term, rad);
  Phat = slice-profile FT (Hann-windowed sinc RF, TBW 4: real -> only magnitude, no phase bias while > 0);
  kappa_s = through-slice B0 gradient (rad/m/s), k_ss = slice-refocusing error (rad/m), T2 of the doped water;
  p_s' = p_s (1 + eps_ss) + dp_s: slice position incl. the B0 shift -dB0_mean/G_ss, the drift of the RF centre
         frequency, a G_ss gain error and a random offset;
  I_s = in-plane integral over the sphere's cross section, weighted by an effective receive weighting (1 + alpha.r),
        of e^{-i[(dw(r) - dw_mean) t + gamma c_a(t) q_a(r) + sigma (sum_c k_c(t) r_c + k_a(t) dp(r))]}: B0
        inhomogeneity (2nd/3rd-order field), concomitant field (q = z^2/(2B0) for x/y inputs, (x^2+y^2)/(8B0) for
        z; c = int g^2), in-plane cross-term gradients k_c (they meet the slice centroid -> k0 contamination), the
        slice warp by the B0 field (dp = -dB0/G_ss); linearised in the small sigma-terms (|.| < 0.1 rad);
  psi_s = static phase (mean off-resonance, receiver, concomitant mean): identical for +-, cancels in the difference.
Estimator: d_s = arg(s_+ conj(s_-)) (unwrapped against a reference chain's prediction) = -2 (k_a p_s + phi0);
weighted LS over the slices per sample -> k_a(t), phi0(t); regularised frequency-domain LS over all inputs
-> H_aa(f), H_0a(f) (increments, so the PL dwell box is the same on input and output: no dwell-time compensation
needed); optional parametric fit (delay, low-pass, 2 eddy terms, resonances) weighted by the per-frequency Fisher
information.
"""
import numpy as np
from scipy.optimize import least_squares
import gmodel as gm

GAM = gm.GAM
CUT_FACTOR = 100.0
GBAR = gm.GBAR


# ───────────────────────────────────────────────────────────── protocol and nuisances
class Protocol:
    def __init__(self, **kw):
        self.dt = 4e-6            # ADC dwell = gradient raster (GE)
        self.t_adc = 80e-3
        self.slices = np.array([-26.25, -8.75, 8.75, 26.25]) * 1e-3
        self.thk = 1.5e-3
        self.tbw = 4.0
        self.t_rf = 3e-3          # RF duration -> G_ss = tbw / (gbar thk t_rf)
        self.gmax, self.smax = 0.070, 200.0
        self.s_tri, self.s_chirp = 0.9, 0.5   # slew fractions (PNS: short triangles 0.9, chirp sustained 0.5)
        self.s_tri_long = 0.55    # triangles with ramps > 100 us (SAFE example hw: 0.9 gives 98-121 % on y)
        self.g_cap = 0.045
        self.tri_ramps = [60e-6, 100e-6, 160e-6, 240e-6]
        self.chirp = (100.0, 20e3, 40e-3)  # f0, f1, duration (log sweep)
        self.klim_frac = 0.8      # chirp / triangle |k| <= frac * k_lim, k_lim at Phat = sinc(0.4)
        self.n_avg = 50           # averages per (slice, input, polarity)
        self.t_frame = 0.1        # s per frame (ADC 80 ms + RF, spoiler, settle)
        self.snr0 = 1000.0        # per-sample SNR of the central 1.5 mm slice FID at t = 0 (one average); signal ~ thk
        self.hadamard = False     # 4-slice Hadamard multiplexing (sqrt(4) SNR at equal time)
        self.__dict__.update(kw)

    @property
    def g_ss(self):
        return self.tbw / (GBAR * self.thk * self.t_rf)

    @property
    def klim(self):
        return 0.4 * 2 * np.pi / self.thk  # rad/m

    def n_frames(self):
        return 3 * len(self.slices) * 2 * len(self.inputs()) * self.n_avg

    def minutes(self):
        return self.n_frames() * self.t_frame / 60

    def inputs(self):
        """test waveforms on the raster from the excitation (t = 0), each (name, g (n,))"""
        if hasattr(self, '_inputs'):
            return self._inputs
        dt = self.dt
        n = int(round(self.t_adc / dt))
        out = []
        kl = self.klim_frac * self.klim
        # log chirp, amplitude min(g_cap, slew, k-limit), cosine phase (k symmetric), 2-cycle ramp-in, 1 ms ramp-out
        f0, f1, T = self.chirp
        t = np.arange(n) * dt
        t_on = 1e-3
        tt = t - t_on
        L = np.log(f1 / f0)
        on = (tt >= 0) & (tt <= T)
        f = np.where(on, f0 * np.exp(np.clip(tt, 0, T) * L / T), 0)
        ph = 2 * np.pi * f0 * T / L * (np.exp(np.clip(tt, 0, T) * L / T) - 1)
        A = np.minimum.reduce([np.full(n, self.g_cap), self.s_chirp * self.smax / (2 * np.pi * np.maximum(f, 1)), 2 * np.pi * np.maximum(f, 1) * kl / GAM])
        w = np.clip(tt * f0 / 2, 0, 1) * np.clip((T - tt) / 1e-3, 0, 1)
        w = np.sin(0.5 * np.pi * w) ** 2
        g = np.where(on, A * w * np.cos(ph), 0.0)
        out.append(('chirp', g))
        # triangles with a prephaser (k from -A/2 to +A/2): prephaser = slow trapezoid at 0.5 ms, triangle at 3 ms
        for tr in self.tri_ramps:
            st = self.s_tri if tr <= 100e-6 else self.s_tri_long
            gp = min(st * self.smax * tr, self.g_cap, 2 * kl / (GAM * tr))  # |k| = area / 2 <= klim_frac k_lim
            nr = int(round(tr / dt))
            tri = np.r_[np.linspace(0, gp, nr + 1), np.linspace(gp, 0, nr + 1)[1:]]
            area = gp * tr  # T s/m
            # prephaser: trapezoid of area -area/2, slew 0.3 smax, amplitude <= 10 mT/m
            s = 0.3 * self.smax
            gpp = min(10e-3, np.sqrt(s * area / 2))
            nrr = max(1, int(np.ceil(gpp / s / dt)))
            gpp = min(gpp, s * nrr * dt)
            flat = area / 2 / gpp - nrr * dt
            nf = max(0, int(np.round(flat / dt)))
            pre = np.r_[np.linspace(0, gpp, nrr + 1), np.full(nf, gpp), np.linspace(gpp, 0, nrr + 1)]
            pre *= (area / 2) / (np.trapezoid(pre, dx=dt))
            g = np.zeros(n)
            i0 = int(round(0.5e-3 / dt)); g[i0:i0 + len(pre)] -= pre
            i1 = int(round(3e-3 / dt)); g[i1:i1 + len(tri)] += tri
            out.append((f'tri{int(tr * 1e6)}', g))
        self._inputs = out
        return out


def slice_profile_ft(k, thk, tbw):
    """normalised FT of the slice magnetisation profile (small-tip: the RF envelope, Hann-windowed sinc)"""
    x = k * thk / (2 * np.pi)
    w = np.where(np.abs(x) <= tbw / 2, 0.5 * (1 + np.cos(2 * np.pi * x / tbw)), 0.0)
    return np.sinc(x) * w


class Nuisance:
    """one draw of the physical nuisances of a session"""

    def __init__(self, rng, prot, **kw):
        self.R_ph = 0.085                      # sphere radius (17 cm QA sphere)
        self.center = rng.normal(0, 3e-3, 3)   # phantom placement
        self.T2 = rng.uniform(0.08, 0.25)
        self.b0_rms_hz = rng.uniform(2.0, 8.0)  # residual shim over the sphere
        # field: random 2nd + 3rd order solid harmonics (Cartesian monomials), scaled to the rms over the sphere
        self.b0c = rng.normal(size=16)
        self.coilw = rng.normal(size=3); self.coilw *= rng.uniform(0, 0.3) / max(np.linalg.norm(self.coilw), 1e-9)
        self.df0_hz = rng.normal(0, 3.0)        # RF centre frequency error / drift during the block (shifts every slice)
        self.eps_ss = rng.normal(0, 2e-3)       # slice-select gain (position scale) error
        self.dp_rand = rng.normal(0, 5e-6, (3, len(prot.slices)))  # residual per-slice offsets
        self.k_ss = rng.normal(0, 20.0)         # slice refocusing error, rad/m
        self.dt_adc = 0.0                       # ADC vs gradient raster offset of the GIRF protocol (s)
        self.__dict__.update(kw)
        self._norm = None

    def field_hz(self, r):
        """B0 deviation (Hz) at positions r (m, phantom-centred), unscaled monomial combination"""
        x, y, z = (r / self.R_ph).T
        mon = np.stack([x * x - y * y, 2 * z * z - x * x - y * y, x * y, x * z, y * z,
                        x * (x * x - 3 * y * y), y * (3 * x * x - y * y), z * (2 * z * z - 3 * x * x - 3 * y * y), x * (4 * z * z - x * x - y * y),
                        y * (4 * z * z - x * x - y * y), z * (x * x - y * y), x * y * z, x, y, z, 0 * x], 1)
        f = mon @ self.b0c
        if self._norm is None:
            q = np.random.default_rng(0).uniform(-1, 1, (20000, 3)); q = q[(q ** 2).sum(1) < 1] * self.R_ph
            xq, yq, zq = (q / self.R_ph).T
            m2 = np.stack([xq * xq - yq * yq, 2 * zq * zq - xq * xq - yq * yq, xq * yq, xq * zq, yq * zq,
                           xq * (xq * xq - 3 * yq * yq), yq * (3 * xq * xq - yq * yq), zq * (2 * zq * zq - 3 * xq * xq - 3 * yq * yq), xq * (4 * zq * zq - xq * xq - yq * yq),
                           yq * (4 * zq * zq - xq * xq - yq * yq), zq * (xq * xq - yq * yq), xq * yq * zq, xq, yq, zq, 0 * xq], 1) @ self.b0c
            m2 = m2 - m2.mean()
            self._norm = self.b0_rms_hz / m2.std()
        return f * self._norm


# ───────────────────────────────────────────────────────────── simulation
def slice_points(nu, prot, a, p, h=3e-3):
    """in-plane grid of slice p (scanner coordinate on axis a) through the sphere: positions (scanner frame), weights"""
    others = [b for b in range(3) if b != a]
    c = nu.center
    rr2 = nu.R_ph ** 2 - (p - c[a]) ** 2
    if rr2 <= 0:
        return np.zeros((0, 3)), np.zeros(0)
    r = np.sqrt(rr2)
    u = np.arange(-r, r + h, h)
    U, V = np.meshgrid(u, u, indexing='ij')
    m = U ** 2 + V ** 2 <= rr2
    pos = np.zeros((m.sum(), 3))
    pos[:, a] = p
    pos[:, others[0]] = c[others[0]] + U[m]
    pos[:, others[1]] = c[others[1]] + V[m]
    w = 1 + (pos - c) @ nu.coilw / nu.R_ph
    return pos, np.clip(w, 0.05, None) * h * h


def simulate(truth, prot, nu, rng, axes=(0, 1, 2), decim=25, b0=3.0):
    """averaged complex FIDs per axis: dict a -> array (n_inputs, n_slices, 2 [+,-], n_t) incl. noise, and the
    noise std of one averaged sample (relative units)"""
    dt = prot.dt
    n = int(round(prot.t_adc / dt))
    t = np.arange(n) * dt
    td = t[::decim]
    gss = prot.g_ss
    out = {}
    sig = 1.0 / (prot.snr0 * np.sqrt(prot.n_avg) * (2.0 if prot.hadamard else 1.0))
    a_ref = None
    for a in axes:
        Y = np.zeros((len(prot.inputs()), len(prot.slices), 2, n), complex)
        for iu, (name, gin) in enumerate(prot.inputs()):
            g3 = np.zeros((n, 3)); g3[:, a] = gin
            y, bfield = truth.apply(g3, dt, ntail=0)
            y, bfield = y[gm.PRE:gm.PRE + n], bfield[gm.PRE:gm.PRE + n]
            # ADC samples at t + dt_adc (offset of the receiver vs the gradient raster)
            ts = t + nu.dt_adc
            k = gm.k_at(np.r_[np.zeros((gm.PRE, 3)), y], dt, -gm.PRE * dt, ts, 0.0)  # rad/m, 3 axes
            ph0 = gm.phase_at(np.r_[np.zeros(gm.PRE), bfield], dt, -gm.PRE * dt, ts, 0.0)
            conc = np.r_[0, np.cumsum(0.5 * (gin[1:] ** 2 + gin[:-1] ** 2)) * dt]  # int g^2 (T^2 s / m^2)
            for js, p in enumerate(prot.slices):
                pos, wt = slice_points(nu, prot, a, p)
                pc = pos - nu.center
                fz = nu.field_hz(pc)
                A_s = wt.sum()
                if a_ref is None:
                    pc0, w0 = slice_points(nu, prot, a, 0.0)
                    a_ref = w0.sum()
                fbar = (wt * fz).sum() / A_s
                dw = 2 * np.pi * (fz - fbar)
                dp = -(fz / GBAR) / gss  # local slice shift (m) from the B0 field
                # in-plane concomitant geometry (scanner coordinates)
                if a == 2:
                    q = (pos[:, 0] ** 2 + pos[:, 1] ** 2) / (8 * b0)
                else:
                    q = pos[:, 2] ** 2 / (2 * b0)
                cd = conc[::decim]
                psi = dw[None, :] * td[:, None] + GAM * cd[:, None] * q[None, :]
                E = np.exp(-1j * psi) * wt[None, :]
                I0 = E.sum(1) / A_s
                others = [b for b in range(3) if b != a]
                Mc = [(E * pos[None, :, b]).sum(1) / A_s for b in others]
                Md = (E * (dp - (wt * dp).sum() / A_s)[None, :]).sum(1) / A_s
                ip = lambda z: np.interp(t, td, z.real) + 1j * np.interp(t, td, z.imag)
                I0, Md = ip(I0), ip(Md)
                Mc = [ip(m) for m in Mc]
                # through-slice B0 gradient at the slice (finite difference of the field along a)
                e_a = np.zeros(3); e_a[a] = 1e-3
                kap = 2 * np.pi * ((wt * (nu.field_hz(pc + e_a) - fz)).sum() / A_s) / 1e-3  # rad/s/m
                # slice-select polarity: the B0-induced slice shift and warp flip with the sign of G_ss
                # (prot.ss_alt: half of the averages with each sign, signals averaged)
                for jp, sgn in enumerate((1.0, -1.0)):
                    s = 0
                    sss = (1.0, -1.0) if getattr(prot, 'ss_alt', False) else (1.0,)
                    for ss in sss:
                        p_true = p * (1 + nu.eps_ss) + nu.dp_rand[a, js] + ss * (-(fbar + nu.df0_hz) / GBAR) / gss
                        I = I0 - 1j * sgn * (sum(k[:, b] * Mc[i] for i, b in enumerate(others)) + ss * k[:, a] * Md)
                        P = slice_profile_ft(sgn * k[:, a] + nu.k_ss + kap * t, prot.thk, prot.tbw)
                        s = s + (A_s / a_ref) * (prot.thk / 1.5e-3) * np.exp(-t / nu.T2) * P * I * np.exp(-1j * sgn * (k[:, a] * p_true + ph0)) / len(sss)
                    s = s * np.exp(-1j * 2 * np.pi * fbar * t)  # static mean off-resonance (common)
                    Y[iu, js, jp] = s + sig * (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
        out[a] = Y
    return out, sig


# ───────────────────────────────────────────────────────────── estimation
def phase_fit(Y, prot, sig, ref_k, ref_ph):
    """per input: weighted LS over slices of d_s = -2 (k p_s + phi0); unwrapping against a reference prediction
    (ref_k, ref_ph: (n_inputs, n_t)). Returns k_hat, phi_hat (n_inputs, n_t), their variances"""
    ni, ns, _, n = Y.shape
    p = prot.slices
    K = np.zeros((ni, n)); P0 = np.zeros((ni, n)); VK = np.zeros((ni, n)); VP = np.zeros((ni, n))
    for iu in range(ni):
        d = np.angle(Y[iu, :, 0] * np.conj(Y[iu, :, 1]))  # ns x n
        pred = -2 * (ref_k[iu][None, :] * p[:, None] + ref_ph[iu][None, :])
        # residual against the reference chain's prediction, unwrapped along time (it grows smoothly from 0 at the
        # excitation, so a reference several us off in delay still unwraps correctly)
        r = np.unwrap(np.angle(np.exp(1j * (d - pred))), axis=1)
        r -= 2 * np.pi * np.round(r[:, :25].mean(1) / (2 * np.pi))[:, None]
        d = pred + r
        m2 = np.abs(Y[iu, :, 0]) ** 2; m2b = np.abs(Y[iu, :, 1]) ** 2
        var = 0.5 * sig ** 2 * (1 / np.maximum(m2, 1e-30) + 1 / np.maximum(m2b, 1e-30))
        W = 1 / var  # ns x n
        y = -0.5 * d
        # normal equations per sample for [k, phi0]
        s00 = (W * p[:, None] ** 2).sum(0); s01 = (W * p[:, None]).sum(0); s11 = W.sum(0)
        b0 = (W * p[:, None] * y).sum(0); b1 = (W * y).sum(0)
        det = s00 * s11 - s01 ** 2
        K[iu] = (s11 * b0 - s01 * b1) / det
        P0[iu] = (s00 * b1 - s01 * b0) / det
        VK[iu] = 0.25 * s11 / det; VP[iu] = 0.25 * s00 / det
    return K, P0, VK, VP


def deconvolve(kin, kout, var, dt, nfft=None, lam=1e-12, adaptive=False):
    """H(f) on the rfft grid from increments: sum_u w_u conj(X_u) Y_u / (sum_u w_u |X_u|^2 + lam max), w_u = 1 /
    mean noise variance of the increments of input u. kin, kout: (n_inputs, n) (rad/m, or rad for the B0 term).
    Returns f, H, information sum_u w_u |X_u|^2 (the inverse variance of H up to the white-noise approximation)"""
    ni, n = kin.shape
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
    dkin, dkout = np.diff(kin, axis=1), np.diff(kout, axis=1)
    # per input: stop where the k noise has grown CUT_FACTOR x (signal down sqrt of it by T2* / dephasing); later samples add noise
    # to every bin and little information (the response there has to be observed with signal, not with noise)
    cut = np.full(ni, n - 1)
    if adaptive:
        v0 = np.median(var[:, :max(10, n // 10)], axis=1)
        for u in range(ni):
            over = np.flatnonzero(var[u, 1:] > CUT_FACTOR * v0[u])
            if len(over):
                cut[u] = max(over[0], n // 8)
        for u in range(ni):
            dkin[u, cut[u]:] = 0.0
            dkout[u, cut[u]:] = 0.0
    X = np.fft.rfft(dkin, nfft, axis=1)
    Y = np.fft.rfft(dkout, nfft, axis=1)
    w = 1.0 / (2 * np.array([var[u, :cut[u] + 1].mean() for u in range(ni)]))  # increments of white k noise: 2 var
    f = np.fft.rfftfreq(nfft, dt)
    # PSD of differenced white noise: 4 sin^2(pi f dt) var -> per-frequency information
    s2 = 4 * np.sin(np.pi * f * dt) ** 2
    num = (w[:, None] * np.conj(X) * Y).sum(0)
    den = (w[:, None] * np.abs(X) ** 2).sum(0)
    H = num / (den + lam * den.max())
    info = den / np.maximum(s2 * 0.5, 1e-30)  # 1 / var(H) per bin (white k noise of var/2 per sample per input)
    return f, H, info


# ───────────────────────────────────────────────────────────── parametric model of one axis
def par_resp(th, nres):
    """th = [(gain - 1) * 1e3, delay_us, tau_lp_us, log a1, log tau1, log a2, log tau2, (f0_hz, log zeta, gain) * nres]
    (a free DC gain: the slice positions set it, and an image-scale error is absorbed by the image anyway)"""
    g, d, tl = 1 + th[0] * 1e-3, th[1] * 1e-6, th[2] * 1e-6
    ed = [(np.exp(th[3]), np.exp(th[4])), (np.exp(th[5]), np.exp(th[6]))]
    base = gm.Scaled(gm.Param(d, tl, ed), g)
    res = [(th[7 + 3 * j], np.exp(th[8 + 3 * j]), th[9 + 3 * j]) for j in range(nres)]
    return gm.MultiRes(base, res)


def fit_axis(f, H, sd, fmax=15e3, nres_max=4, x0=None, dt=4e-6, thr=3e-3, floor=2e-4):
    """complex LS of the parametric chain to H(f) (0 < f <= fmax) with weights 1 / sqrt(sd^2 + floor^2) (the
    per-bin noise sd of H plus a model-error floor, so the high-information low frequencies do not decide the delay /
    low-pass alone); resonances added greedily from the residual peaks (0.3-6 kHz, relative size > thr). The model
    goes through the same PL alias sum as the data (at_pl on the measurement raster, 3 alias terms)."""
    m = (f > 0) & (f <= fmax)
    ff, hh = f[m], H[m]
    ww = 1.0 / np.sqrt(sd[m] ** 2 + floor ** 2) * floor
    ff, hh, ww = ff[::2], hh[::2], ww[::2]

    def resid(th, nres):
        r = (gm.at_pl(par_resp(th, nres), ff, dt, M=3) - hh) * ww / floor
        return np.r_[r.real, r.imag]

    th = np.array([0.0, 2.5, 8.0, np.log(2e-3), np.log(3e-3), np.log(1e-3), np.log(60e-6)]) if x0 is None else np.asarray(x0, float)
    lo = [-20, -5, 0, np.log(1e-6), np.log(3e-4), np.log(1e-6), np.log(5e-6)]
    hi = [20, 10, 40, np.log(0.05), np.log(0.2), np.log(0.05), np.log(3e-4)]
    best = least_squares(resid, th, args=(0,), bounds=(lo, hi), x_scale='jac', max_nfev=600)
    nres = 0
    band = (ff > 300) & (ff < 6000)
    for j in range(nres_max):
        hm = gm.at_pl(par_resp(best.x, nres), ff, dt, M=3)
        ratio = hh / hm - 1
        rel = np.where(band & (ww > 0.2), np.abs(ratio), 0)
        i = int(np.argmax(rel))
        if rel[i] < thr:
            break
        f0 = ff[i]
        lo2 = lo + [f0 * 0.9, np.log(2e-3), -0.5]
        hi2 = hi + [f0 * 1.1, np.log(0.2), 0.5]
        th = np.r_[best.x, f0, np.log(0.02), np.clip(np.real(ratio[i]), -0.4, 0.4)]
        th = np.clip(th, np.array(lo2) + 1e-9, np.array(hi2) - 1e-9)
        sol = least_squares(resid, th, args=(nres + 1,), bounds=(lo2, hi2), x_scale='jac', max_nfev=800)
        if sol.cost < 0.98 * best.cost:
            best, nres, lo, hi = sol, nres + 1, lo2, hi2
        else:
            break
    return best.x, nres, best


def fit_b0(f, H, info, fmax=5e3):
    """B0 term: sum of two high-pass terms (k_j, tau_j) plus a DC term (slice-position cross-talk, absorbable)"""
    m = (f > 0) & (f <= fmax)
    ff, hh, ww = f[m], H[m], np.sqrt(info[m]); ww = ww / ww.max()

    def model(th):
        return gm.MultiHP([(th[0], np.exp(th[1])), (th[2], np.exp(th[3]))]).at(ff) + th[4]

    def resid(th):
        r = (model(th) - hh) * ww
        return np.r_[r.real, r.imag]
    sol = least_squares(resid, [1e-5, np.log(2e-3), 1e-6, np.log(2e-4), 0.0], x_scale=[1e-5, 1, 1e-5, 1, 1e-6], max_nfev=400)
    return sol.x


def hybrid_table(f, H, info, par, f_trust):
    """nonparametric H below f_trust (blended over 20 %), parametric above, extended to 5 MHz (Table response)"""
    fe = np.r_[f[f < f_trust * 1.2], np.geomspace(f_trust * 1.2, 5e6, 400)[1:]]
    hp = par.at(fe)
    hn = np.interp(fe, f, H.real) + 1j * np.interp(fe, f, H.imag)
    b = np.clip((fe - f_trust) / (0.2 * f_trust), 0, 1)
    return gm.Table(fe, (1 - b) * hn + b * hp)
