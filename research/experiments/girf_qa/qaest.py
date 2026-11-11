"""Estimators of the GIRF QA protocol (see girf_qa_sequence.md §4):
- phase fit per time sample over slice units (positions from the slice-position calibration, both slice-select
  polarities as separate units, in-plane / higher-order contributions from the X stage subtracted) → k_a(t), φ0(t);
- frequency-domain LS over the inputs on the k increments → H_aa, H_0a (girf_sensitivity estimator), PL → continuous,
  nonparametric below the trusted band, parametric above;
- X stage: per time sample WLS over (slice, η, PE step) units of the ± phase difference on the in-plane moments
  A_l = Re(M_l / M) of the estimated slice geometry (receive weighting, B0 map, concomitant phase, PE weighting)
  → all 16 spatial coefficients; deconvolution → cross / B0 / 2nd / 3rd-order responses (2-term high-pass fits);
- resonances: survey list from the parametric fit, time-domain fits of (f0, ζ, gain, delay) on targeted data, per
  thermal set f0 tracking, a 2-compartment power-history model for f0(t);
- linearity β, receiver delay per dwell, v2-validation residuals.
"""
import numpy as np
from scipy.optimize import least_squares
import qasim as qs

gm, ts = qs.gm, qs.ts
GAM = qs.GAM
import mc as smc  # noqa: E402  (girf_sensitivity: to_continuous, band_errors)


# ───────────────────────────────────────────────────────────── chain predictions
def chain_k(chain, a, g, dt, t, out_axes=True):
    """k (rad/m, 3 axes) and B0 phase at times t (s from the start of raster g) for input g on axis a"""
    n = len(g)
    ntail = max(10, int(np.ceil(t.max() / dt)) - n + 20)
    g3 = np.zeros((n, 3))
    g3[:, a] = g
    y, b = chain.apply(g3, dt, ntail=ntail)
    k = gm.k_at(y, dt, -gm.PRE * dt, t, 0.0)
    p = gm.phase_at(b, dt, -gm.PRE * dt, t, 0.0)
    return k, p


def resp_k(resp, g, dt, t):
    """γ∫ (resp ∗ g) at times t for one response (spatial-term predictions)"""
    n = len(g)
    L = gm.PRE + n + max(10, int(np.ceil(t.max() / dt)) - n + 20)
    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    x = np.zeros(nfft)
    x[gm.PRE:gm.PRE + n] = g
    fr = np.fft.rfftfreq(nfft, dt)
    y = np.fft.irfft(np.fft.rfft(x) * gm._pl_cached(resp, fr, dt, nfft), nfft)[:L]
    return gm.phase_at(y, dt, -gm.PRE * dt, t, 0.0)


# ───────────────────────────────────────────────────────────── phase fit over units
def phase_fit_units(Yp, Ym, pos, sig, pk, pp, corr=None, unwrap=True):
    """Yp, Ym: (nu, nt) averaged signals for σ = ±1; pos (nu,) slice positions; sig (nu,) noise sd per sample;
    pk, pp (nt,): reference prediction of k_a and φ0 (unwrapping); corr (nu, nt): known phase −½·(Σ c_l Ȳ_l) part to
    remove. Returns k, φ0, var k, var φ0 (nt,)"""
    d = np.angle(Yp * np.conj(Ym))
    pred = -2 * (pk[None, :] * pos[:, None] + pp[None, :])
    if corr is not None:
        pred = pred - 2 * corr
    r = np.angle(np.exp(1j * (d - pred)))
    if unwrap:
        # residual unwrapped along time (a poor reference); with a good reference the wrapped residual is used as is,
        # which is robust to signal dropouts (in-plane dephasing) where a temporal unwrap would slip by 2π
        r = np.unwrap(r, axis=1)
        r -= 2 * np.pi * np.round(r[:, :25].mean(1) / (2 * np.pi))[:, None]
    d = pred + r
    if corr is not None:
        d = d + 2 * corr
    var = 0.5 * sig[:, None] ** 2 * (1 / np.maximum(np.abs(Yp) ** 2, 1e-30) + 1 / np.maximum(np.abs(Ym) ** 2, 1e-30))
    W = 1 / var
    y = -0.5 * d
    s00 = (W * pos[:, None] ** 2).sum(0)
    s01 = (W * pos[:, None]).sum(0)
    s11 = W.sum(0)
    b0 = (W * pos[:, None] * y).sum(0)
    b1 = (W * y).sum(0)
    det = s00 * s11 - s01 ** 2
    return (s11 * b0 - s01 * b1) / det, (s00 * b1 - s01 * b0) / det, 0.25 * s11 / det, 0.25 * s00 / det


def to_raster(t, x, v, n, dt):
    """samples at raster times t → contiguous raster arrays (0..n−1): linear interpolation over ADC gaps, 0 before the
    first sample (input lead-in), variance of the nearest sample in gaps"""
    tr = np.arange(n) * dt
    xr = np.interp(tr, t, x, left=0.0, right=x[-1])
    vr = np.interp(tr, t, v, left=v[0], right=v[-1])
    return xr, vr


def measured_mask(t, n, dt):
    """raster points that carry an ADC sample (the others are interpolated over gaps / the lead-in)"""
    m = np.zeros(n, bool)
    idx = np.round(t / dt).astype(int)
    m[idx[(idx >= 0) & (idx < n) & (np.abs(t - idx * dt) < 0.26 * dt)]] = True
    return m


# ───────────────────────────────────────────────────────────── per-axis H from k(t) of several inputs
def axis_estimate(kin, K, VK, P0, VP, dt, fit_fmax=15e3, sd_trust=2e-3, nres_max=6):
    """kin, K, VK, P0, VP: (n_inputs, n) on the raster from the input start → dict with the PL / continuous H, the
    parametric fit, the B0 term, hybrid tables (girf_sensitivity mc.estimate, per axis)"""
    n = K.shape[1]
    f, H, info = ts.deconvolve(kin, K, VK, dt)
    _, H0, info0 = ts.deconvolve(kin, P0, VP, dt)
    sd = np.sqrt((n - 1) / np.maximum(info, 1e-300))
    th, nres, sol = ts.fit_axis(f, H, sd, fmax=fit_fmax, dt=dt, nres_max=nres_max)
    par = ts.par_resp(th, nres)
    m = (f > 0) & (f <= 5e3)
    sd0 = np.sqrt((n - 1) / np.maximum(info0, 1e-300))
    fl0 = 2e-7
    ff, hh, ww = f[m][::2], H0[m][::2] / 1e-5, (fl0 / np.sqrt(sd0[m] ** 2 + fl0 ** 2))[::2]
    Hm = H[m][::2]

    def model(x):
        return gm.MultiHP([(x[0], np.exp(x[1])), (x[2], np.exp(x[3]))]).at(ff) + x[4] * Hm

    def res(x):
        r = (model(x) - hh) * ww / (fl0 / 1e-5)
        return np.r_[r.real, r.imag]
    best = None
    for x0 in ([1.0, np.log(2e-3), 0.1, np.log(3e-4), 0.0], [-1.0, np.log(2e-3), -0.1, np.log(3e-4), 0.0], [1.0, np.log(5e-3), -0.5, np.log(1e-3), 0.0]):
        sb = least_squares(res, x0, bounds=([-100, np.log(5e-5), -100, np.log(2e-5), -100], [100, np.log(0.2), 100, np.log(0.2), 100]), x_scale='jac', max_nfev=600)
        if best is None or sb.cost < best.cost:
            best = sb
    xb = np.array(best.x)
    xb[[0, 2, 4]] *= 1e-5
    bpar = gm.MultiHP([(xb[0], np.exp(xb[1])), (xb[2], np.exp(xb[3]))])
    Hc = smc.to_continuous(f, H, par, dt)
    ok = (sd < sd_trust) & (f > 0)
    f_trust = float(f[np.flatnonzero(~ok & (f > 1e3))[0]]) if (~ok & (f > 1e3)).any() else float(f[-1])
    f_trust = min(f_trust, 30e3)
    tab = ts.hybrid_table(f, Hc, info, par, f_trust)
    H0c = smc.to_continuous(f, H0 - xb[4] * H, bpar, dt)
    ok0 = (sd0 < 2e-7) & (f > 0)
    f0t = float(f[np.flatnonzero(~ok0 & (f > 200))[0]]) if (~ok0 & (f > 200)).any() else 5e3
    f0t = min(max(f0t, 500.0), 10e3)
    tab0 = ts.hybrid_table(f, H0c, info0, bpar, f0t)
    return dict(f=f, H=H, Hc=Hc, info=info, sd=sd, H0=H0, th=th, nres=nres, par=par, bpar=bpar, b0x=xb, tab=tab, tab0=tab0, f_trust=f_trust, f0_trust=f0t)


def hp_fit(f, H, info, fmax=5e3, scale=1e-4):
    """2-term high-pass fit (k_j, τ_j) of a small spatial response (cross, 2nd / 3rd order), weights ∝ √info"""
    m = (f > 0) & (f <= fmax)
    ff, hh, ww = f[m][::2], H[m][::2] / scale, np.sqrt(info[m][::2])
    ww = ww / ww.max()

    def res(x):
        r = (gm.MultiHP([(x[0], np.exp(x[1])), (x[2], np.exp(x[3]))]).at(ff) - hh) * ww
        return np.r_[r.real, r.imag]
    best = None
    for x0 in ([1.0, np.log(1e-3), 0.1, np.log(1e-4)], [-1.0, np.log(1e-3), -0.1, np.log(1e-4)], [1.0, np.log(3e-3), 0.0, np.log(3e-4)]):
        sb = least_squares(res, x0, bounds=([-1e3, np.log(2e-5), -1e3, np.log(1e-5)], [1e3, np.log(0.1), 1e3, np.log(0.1)]), x_scale='jac', max_nfev=400)
        if best is None or sb.cost < best.cost:
            best = sb
    x = best.x
    return gm.MultiHP([(x[0] * scale, np.exp(x[1])), (x[2] * scale, np.exp(x[3]))])


# ───────────────────────────────────────────────────────────── estimated slice geometry
class EstGeo:
    """the estimator's view of each slice: calibrated centre, receive weighting and B0 map (prescan, with errors),
    slice positions from the slice-position calibration (truth + 1 µm)"""

    def __init__(self, sess, geo_true, prot):
        nu = sess.nu
        self.nu_est = ts.Nuisance(np.random.default_rng(0), ts.Protocol(), center=nu.center + sess.c_err, T2=nu.T2,
                                  b0_rms_hz=nu.b0_rms_hz, b0c=nu.b0c.copy(), coilw=nu.coilw * (1 + 0.05 * sess.rng.normal(size=3)),
                                  df0_hz=0.0, eps_ss=0.0, dp_rand=np.zeros((3, 4)), k_ss=0.0)
        # B0 map error: a smooth extra field of 0.3 Hz rms
        self.extra = sess.b0_err
        self.nu_est._norm = nu._norm
        self.sess, self.gt, self.prot = sess, geo_true, prot
        self.cache = {}

    def get(self, a, s, eta, gss):
        key = (a, s, eta, round(gss, 9))
        if key in self.cache:
            return self.cache[key]
        G = self.gt.get(a, s, eta, gss)
        nu = self.nu_est
        pos, wt = qs.slice_grid(nu, a, self.prot.slices[s])
        pc = pos - nu.center
        fz = nu.field_hz(pc) + 0.3 * (pc / 0.085) @ self.extra[:3] / np.sqrt(3)
        fbar = (wt * fz).sum() / wt.sum()
        dp = -eta * (fz / qs.GBAR) / gss
        p_hat = G['p_true'] + self.sess.p_err[a, s]
        r3 = pos.copy()
        r3[:, a] = p_hat + (dp - (wt * dp).sum() / wt.sum())
        g = dict(pos=pos, wt=wt, dw=2 * np.pi * (fz - fbar), p_hat=p_hat, Y=qs.basis(r3),
                 q=(pos[:, 0] ** 2 + pos[:, 1] ** 2) / (8 * qs.B0) if a == 2 else pos[:, 2] ** 2 / (2 * qs.B0))
        g['Ybar'] = (wt[:, None] * g['Y']).sum(0) / wt.sum()
        self.cache[key] = g
        return g


def x_moments(eg, a, s, eta, gss, g_in, dt, t_rel, t_in0, pes, dt_dec=1e-3):
    """A_l(t) = Re(M_l / M) and |M| for every PE step (estimated geometry, concomitant phase of the input)"""
    G = eg.get(a, s, eta, gss)
    gg = np.zeros(len(g_in) + 1)
    gg[1:] = np.cumsum(0.5 * (np.r_[0, g_in[1:] ** 2] + np.r_[0, g_in[:-1] ** 2])) * dt
    tg = np.arange(len(gg)) * dt - dt
    td = np.arange(t_rel[0] - dt_dec, t_rel[-1] + 2 * dt_dec, dt_dec)
    conc = GAM * np.interp(td - t_in0, tg, gg)
    E = np.exp(-1j * (td[:, None] * G['dw'][None, :] + conc[:, None] * G['q'][None, :]))
    others = [b for b in range(3) if b != a]
    A, Mag = [], []
    for pe in pes:
        w = G['wt'] * np.exp(1j * (pe[0] * G['pos'][:, others[0]] + pe[1] * G['pos'][:, others[1]]))
        M = (E * w[None, :]) @ np.concatenate([np.ones((len(w), 1)), G['Y']], 1)
        Al = (M[:, 1:] / M[:, :1]).real
        A.append(np.stack([np.interp(t_rel, td, Al[:, l]) for l in range(qs.NB)], 1))
        Mag.append(np.interp(t_rel, td, np.abs(M[:, 0])))
    return A, Mag


def x_solve(D, V, A):
    """per-time WLS: D (nu, nt) half phase difference −½ arg(s₊ s₋*), V (nu, nt) its variance, A (nu, nt, L) →
    c (nt, L), var (nt, L); normal equations on a 0.5 ms grid, interpolated"""
    nu, nt, L = A.shape
    step = 125
    ti = np.arange(0, nt, step)
    if ti[-1] != nt - 1:
        ti = np.r_[ti, nt - 1]
    B = np.zeros((len(ti), L, nu))
    Cv = np.zeros((len(ti), L))
    for j, i in enumerate(ti):
        Ai = A[:, i, :]
        W = 1 / V[:, i]
        N = Ai.T @ (W[:, None] * Ai)
        Ni = np.linalg.pinv(N, rcond=1e-12)
        B[j] = Ni @ (Ai.T * W[None, :])
        Cv[j] = np.diag(Ni)
    c = np.zeros((nt, L))
    var = np.zeros((nt, L))
    tt = np.arange(nt)
    for l in range(L):
        Bl = np.stack([np.interp(tt, ti, B[:, l, u]) for u in range(nu)], 0)  # nu x nt
        c[:, l] = (Bl * D).sum(0)
        var[:, l] = np.interp(tt, ti, Cv[:, l])
    return c, var


# ───────────────────────────────────────────────────────────── resonances
def resonance_list(par, a, fmin=300.0, fmax=2500.0):
    """(axis, f0, ζ, gain) of the parametric fit's resonances in [fmin, fmax]"""
    return [(a, f0, z, g) for f0, z, g in par.res if fmin <= f0 <= fmax]


def bp(f, f0, z):
    w, w0 = 2 * np.pi * f, 2 * np.pi * f0
    return (2j * z * w0 * w) / (w0 * w0 - w * w + 2j * z * w0 * w)


def res_fit(dkin_list, dk_list, var_list, tab, res0, dt, fit_zeta=True, fit_delay=True, nfft=None, zeta_fix=None):
    """time-domain LS fit of one resonance (f0, ζ, gain) [+ a delay] on k increments of one or more inputs:
    model H(f) = tab(f) · (1 + g BP(f; f0, ζ)) / (1 + g0 BP(f; f0⁰, ζ⁰)) · e^{−i2πf δ}, PL raster factor sinc²(f dt).
    res0 = (f0, ζ, g) of the background table's own resonance (divided out). Returns (f0, ζ, g, δ), cost, jac sd"""
    f0i, zi, gi = res0
    n = max(len(x) for x in dkin_list)
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(4 * n)))
    f = np.fft.rfftfreq(nfft, dt)
    Hb = tab.at(f) / (1 + gi * bp(f, f0i, zi)) * np.sinc(f * dt) ** 2
    X = [np.fft.rfft(x, nfft) for x in dkin_list]
    # per-increment weights: 1/sd of the increment where both ends are measured samples, 0 across gaps / lead-in
    w = []
    for v, y in zip(var_list, dk_list):
        if isinstance(v, tuple):
            v, m = v
        else:
            m = np.ones(len(v), bool)
        wi = 1 / np.sqrt(v[:-1] + v[1:])
        wi[~(m[:-1] & m[1:])] = 0.0
        w.append(wi[:len(y)])

    def model(p):
        f0, lz, g, d = p
        H = Hb * (1 + g * bp(f, f0, np.exp(lz))) * np.exp(-2j * np.pi * f * d * 1e-9)
        return [np.fft.irfft(Xi * H, nfft)[:len(y)] for Xi, y in zip(X, dk_list)]

    def resid(p):
        return np.concatenate([(m - y) * wi for m, y, wi in zip(model(p), dk_list, w)])
    p0 = np.array([f0i, np.log(max(zi if zeta_fix is None else zeta_fix, 1e-3)), gi, 0.0])
    lo = [f0i * 0.98, np.log(5e-4), -0.5, -200.0 if fit_delay else -1e-6]
    hi = [f0i * 1.02, np.log(0.2), 0.5, 200.0 if fit_delay else 1e-6]
    if not fit_zeta or zeta_fix is not None:
        lo[1], hi[1] = p0[1] - 1e-9, p0[1] + 1e-9
    sol = least_squares(resid, np.clip(p0, np.array(lo) + 1e-12, np.array(hi) - 1e-12), bounds=(lo, hi), x_scale=[0.1, 0.1, 0.005, 1.0], max_nfev=200)
    J = sol.jac
    try:
        cov = np.linalg.pinv(J.T @ J) * max(2 * sol.cost / max(len(sol.fun) - 4, 1), 1.0)
        sdv = np.sqrt(np.maximum(np.diag(cov), 0))
    except Exception:
        sdv = np.full(4, np.nan)
    f0, lz, g, d = sol.x
    return (f0, np.exp(lz), g, d * 1e-9), float(sol.cost), sdv


def with_resonances(tab, a_res, new):
    """the table with its resonances (f0, ζ, g) replaced by `new` (same order); returns a Mul response"""
    parts = [tab]
    for (f0o, zo, go), (f0n, zn, gn) in zip(a_res, new):
        parts.append(_ResRatio(f0o, zo, go, f0n, zn, gn))
    return gm.Mul(*parts)


class _ResRatio(gm.Resp):
    def __init__(self, f0o, zo, go, f0n, zn, gn):
        self.f0o, self.zo, self.go, self.f0n, self.zn, self.gn = map(float, (f0o, zo, go, f0n, zn, gn))

    def at(self, f):
        f = np.asarray(f, float)
        return (1 + self.gn * bp(f, self.f0n, self.zn)) / (1 + self.go * bp(f, self.f0o, self.zo))


# ───────────────────────────────────────────────────────────── thermal model fit
def thermal_basis(power, t, taus):
    """normalised compartment responses x_k(t) = (P ∗ e^{−t/τ}/τ)(t) / P_flow-independent units (T²/m²)"""
    P = power[:, 1:].sum(1)
    out = []
    for tau in taus:
        a = np.exp(-1.0 / tau)
        x, xs = 0.0, np.zeros(len(P) + 1)
        for i, p in enumerate(P):
            x = x * a + p * (1 - a)
            xs[i + 1] = x
        out.append(np.interp(t, np.arange(len(P) + 1), xs))
    return np.stack(out, 1)


def fit_thermal(times, f0s, sds, power, p_flow):
    """f0_j(t) = F_j + A_j [w x₁(t; τ₁) + (1 − w) x₂(t; τ₂)] / p_flow for several resonances j sharing (τ₁, τ₂, w):
    returns per resonance (F_j, A_j) and the shared parameters; the flow-scan steady state is F_j + A_j"""
    J = len(f0s)

    def unpack(p):
        return np.exp(p[0]), np.exp(p[1]), 1 / (1 + np.exp(-p[2]))

    def lin(p):
        t1, t2, w = unpack(p)
        X = thermal_basis(power, np.concatenate(times), [t1, t2]) / p_flow
        return w * X[:, 0] + (1 - w) * X[:, 1]

    def resid(p):
        x = lin(p)
        r, i0 = [], 0
        FA = []
        for j in range(J):
            n = len(times[j])
            xj = x[i0:i0 + n]
            M = np.stack([np.ones(n), xj], 1) / sds[j][:, None]
            sol = np.linalg.lstsq(M, f0s[j] / sds[j], rcond=None)[0]
            FA.append(sol)
            r.append(M @ sol - f0s[j] / sds[j])
            i0 += n
        resid.FA = FA
        return np.concatenate(r)
    best = None
    for t1 in (150.0, 300.0, 600.0):
        for t2 in (1200.0, 2400.0):
            for w in (-1.0, 0.0, 1.0):
                s = least_squares(resid, [np.log(t1), np.log(t2), w], bounds=([np.log(30), np.log(300), -6], [np.log(3000), np.log(2e4), 6]), max_nfev=200)
                if best is None or s.cost < best.cost:
                    best = s
    resid(best.x)
    t1, t2, w = unpack(best.x)
    return dict(FA=[list(map(float, x)) for x in resid.FA], tau1=t1, tau2=t2, w=w, cost=float(best.cost))


def thermal_predict(fit, power, t, p_flow):
    X = thermal_basis(power, t, [fit['tau1'], fit['tau2']]) / p_flow
    x = fit['w'] * X[:, 0] + (1 - fit['w']) * X[:, 1]
    return [F + A * x for F, A in fit['FA']]
