"""Monte Carlo of the GIRF QA protocol: simulate every part of the girfqa protocol for a random truth chain and
session, estimate everything, compare with the truth, and evaluate R for the v2 readout (girf_sensitivity rcalc).
Resumable: one JSON line per draw in <OUT>/mc_<tag>.jsonl; STOP file in <OUT> ends after the current draw.

    qamc.py --tag main --ndraw 16 [--workers 6] [--proto data/hasty/seq/girf_qa/protocol] [--truth rand]
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qasim as qs  # noqa: E402
import qaest as qe  # noqa: E402
import qainputs as qi  # noqa: E402

gm, ts, smc = qs.gm, qs.ts, qe.smc
REPO = qs.REPO
OUT = os.path.join(REPO, 'data/hasty/seq/girf_qa/mc')
PROTO = os.path.join(REPO, 'data/hasty/seq/girf_qa/protocol')
EPS = 1e-6
CUT = 0.1  # rad/m: window cut of the nonparametric estimate (median k sd)


def bins_of(prot, part):
    F, c = prot.F, prot.col
    m = F[:, c('part')] == prot.parts[part]
    avg = F[:, c('avg')]
    if part == 's_cold':
        b = (avg // 2) // 3
    elif part == 's_warm':
        b = (avg - 100) // 2
    elif part == 's_hot':
        b = (avg - 200) // 2
    elif part == 'd_cool':
        b = (avg - 300) // 2
    elif part == 't_rx':
        b = np.round(F[:, c('dwell_s')] * 1e6)
    else:
        b = np.zeros(len(F))
    return m, np.nan_to_num(b, nan=-1).astype(int)


class Ctx:
    def __init__(self, d, prot, truth_kind='rand', snr_body=300.0, snr_array=1000.0, nuis_off=None):
        self.prot = prot
        self.p_flow = prot.p['flow_tr']['p_flow_T2m2']
        self.truth = qs.Truth(d, self.p_flow, truth_kind)
        self.sess = qs.Session(d, prot, snr_body=snr_body, snr_array=snr_array, nuis_off=nuis_off)
        self.geo = qs.Geo(self.sess, prot, prot.cfg['thk'])
        self.eg = qe.EstGeo(self.sess, self.geo, prot)
        self.theta = self.truth.thermal.theta(prot.power, prot.F[:, prot.col('t_exc_s')])
        self.ginputs = {}

    def g(self, iu, level, arr=None):
        key = (iu, level) if arr is None else ('arr', iu, level)
        if key not in self.ginputs:
            base = self.prot.input(iu) if arr is None else arr
            self.ginputs[key] = base * (level if level != 0 else EPS)
        return self.ginputs[key]


def sim_part(ctx, part, coil='body', snr=None, inputs_override=None, theta_override=None):
    """simulate the survey-type frames of a part: data[(bin, iu, level, a, s, eta)] = (Y± (2, n_pe, nt), navg, θ, t);
    returns data, tn (sample times from the input start), pes"""
    prot, F, c = ctx.prot, ctx.prot.F, ctx.prot.col
    m, b = bins_of(prot, part)
    idx = np.flatnonzero(m)
    snr = snr or (ctx.sess.snr_array if coil == 'array' else ctx.sess.snr_body)
    keys = {}
    for i in idx:
        k = (int(F[i, c('input')]), float(F[i, c('level')]), int(F[i, c('axis')]), int(F[i, c('slice')]), float(F[i, c('eta')]), int(b[i]))
        pe = (F[i, c('pe_u_radm')], F[i, c('pe_v_radm')])
        keys.setdefault(k, []).append((i, pe, F[i, c('sigma')]))
    data = {}
    # one thermal state per bin (shared by all its frames: chain spectra and outputs are cached per state)
    th_bin = {bb: float(np.mean(ctx.theta[idx[b[idx] == bb]])) for bb in np.unique(b[idx])}
    t_adc = F[idx[0], c('t_adc_s')]
    t_in0 = F[idx[0], c('t_in0_rel_exc_s')]
    thk = F[idx[0], c('thk_m')]
    combos = sorted({k[:5] for k in keys if k[1] != 0.0})
    kcache = {}
    for (iu, lev, a, s, eta) in combos:
        acache = {}
        for bb in sorted({k[5] for k in keys if k[:5] == (iu, lev, a, s, eta)}):
            fr = keys[(iu, lev, a, s, eta, bb)]
            dwell = F[fr[0][0], c('dwell_s')]
            tn = qs.adc_times(t_adc, dwell)
            pes_all = sorted({pe for _, pe, _ in fr if not np.isnan(pe[0])})
            pes = [np.array(p) for p in pes_all] if pes_all else [None]
            navg = max(1, sum(1 for _, _, sg in fr if sg > 0) // len(pes))
            th = th_bin[bb] if theta_override is None else theta_override
            tt = float(np.mean(F[[i for i, _, _ in fr], c('t_exc_s')]))
            gin = ctx.g(iu, lev) if inputs_override is None else ctx.g(iu, lev, inputs_override[iu])
            Y = qs.sim_group(ctx.truth, ctx.sess, ctx.geo, prot, a, s, eta, gin, 0, th, t_in0, tn, navg, pes, thk, snr, dwell, coil, kcache=kcache, acache=acache)
            data[(bb, iu, lev, a, s, eta)] = (Y, navg, th, tt, snr, dwell)
    return data, t_in0, thk, pes_all if pes_all else None, t_adc


def survey_k(ctx, data, a, iu, lev, t_adc, ref_chain, bins=None, corr=None, dwell=4e-6, unwrap=False):
    """phase fit over units (bin, slice, η) of one input → k, φ0, vars at the sample times + the raster versions"""
    units = [k for k in data if k[1] == iu and k[2] == lev and k[3] == a and (bins is None or k[0] in bins)]
    units.sort()
    prot = ctx.prot
    tn = qs.adc_times(t_adc, dwell)
    gin = ctx.g(iu, lev)
    pk, pp = qe.chain_k(ref_chain, a, gin, prot.dt, tn)
    gss = prot.cfg['g_ss']
    Yp = np.stack([data[u][0][0, 0] for u in units])
    Ym = np.stack([data[u][0][1, 0] for u in units])
    pos = np.array([ctx.eg.get(a, u[4], u[5], gss)['p_hat'] for u in units])
    sig = np.array([1 / (data[u][4] * np.sqrt(data[u][1])) for u in units])
    cr = None
    if corr is not None:
        cr = np.stack([corr(u) for u in units])
    K, P0, VK, VP = qe.phase_fit_units(Yp, Ym, pos, sig, pk[:, a], pp, cr, unwrap=unwrap)
    return K, P0, VK, VP, tn


def raster_set(ctx, ks, tn, inputs):
    """k(t) of several inputs at the sample times → contiguous raster arrays for the deconvolution"""
    dt = ctx.prot.dt
    n = int(round(tn[-1] / dt)) + 1
    kin, K, VK, P0, VP = [], [], [], [], []
    for (k, p, vk, vp), g in zip(ks, inputs):
        gi = np.zeros(n)
        gi[:min(n, len(g))] = g[:n]
        kin.append(gm.cum_k(gi[:, None], dt)[:, 0])
        a1, b1 = qe.to_raster(tn, k, vk, n, dt)
        a2, b2 = qe.to_raster(tn, p, vp, n, dt)
        K.append(a1)
        VK.append(b1)
        P0.append(a2)
        VP.append(b2)
    return np.array(kin), np.array(K), np.array(VK), np.array(P0), np.array(VP)


def estimate_survey(ctx, data, t_adc, ref, inputs, bins=None, xterms=None, a_list=(0, 1, 2), unwrap=False, cut_sd=None):
    """per axis estimate from the survey data (pooled over `bins`); xterms: {(a, l): resp} spatial terms to remove.
    unwrap: residual unwrapped along time (first pass against a generic reference); then the window is cut where the
    median k sd over the inputs exceeds cut_sd (rad/m), so 2π slips in dephased late samples cannot enter. Later
    passes (reference = the previous estimate) use the wrapped residual over the full window."""
    out = {}
    for a in a_list:
        ks = []
        for iu in inputs:
            corr = None
            if xterms is not None:
                tn = qs.adc_times(t_adc, 4e-6)
                gin = ctx.g(iu, 1.0)
                cl = {l: qe.resp_k(r, gin, ctx.prot.dt, tn) for (aa, l), r in xterms.items() if aa == a}
                gss = ctx.prot.cfg['g_ss']

                def corr(u, cl=cl, a=a, gss=gss):
                    yb = ctx.eg.get(a, u[4], u[5], gss)['Ybar']
                    return sum(cl[l] * yb[l] for l in cl)
            K, P0, VK, VP, tn = survey_k(ctx, data, a, iu, 1.0, t_adc, ref, bins, corr, unwrap=unwrap)
            ks.append((K, P0, VK, VP))
        if cut_sd is not None:
            sd = np.median(np.sqrt(np.array([x[2] for x in ks])), 0)
            over = np.flatnonzero(sd > cut_sd)
            if len(over):
                n_cut = max(over[0], len(tn) // 8)
                ks = [tuple(v[:n_cut] for v in x) for x in ks]
                tn = tn[:n_cut]
        kin, K, VK, P0, VP = raster_set(ctx, ks, tn, [ctx.g(iu, 1.0) for iu in inputs])
        out[a] = qe.axis_estimate(kin, K, VK, P0, VP, ctx.prot.dt)
    return out


def girf_from(est, xterms=None, cross=True):
    axis = [est[a]['tab'] for a in range(3)]
    b0 = [est[a]['tab0'] for a in range(3)]
    cr = []
    if xterms is not None and cross:
        for (a, l), r in xterms.items():
            if l in (1, 2, 3) and l - 1 != a:
                cr.append((l - 1, a, r))
    return gm.Girf(axis, cross=cr, b0=b0, tail=0.15, name='est')


def x_stage(ctx, data, t_adc, t_in0, pes, ref, inputs=(0, 2)):
    """phase-encoded stage → spatial responses {(a, l): MultiHP} for l ∉ {0, a} and diagnostics"""
    prot = ctx.prot
    gss = prot.cfg['g_ss']
    tn_all = qs.adc_times(t_adc, 4e-6)
    nx = int(np.searchsorted(tn_all, 45e-3))  # cross / higher-order terms are high-pass (τ ≲ 3 ms): 45 ms suffice
    tn = tn_all[:nx]
    t_rel = t_in0 + tn
    out, diag = {}, {}
    for a in range(3):
        cs = []
        for iu in inputs:
            gin = ctx.g(iu, 1.0)
            pk, pp = qe.chain_k(ref, a, gin, prot.dt, tn)
            units = sorted(k for k in data if k[1] == iu and k[3] == a)
            D, V, A = [], [], []
            for u in units:
                Y, navg, _, _, snr, _ = data[u]
                Al, Mag = qe.x_moments(ctx.eg, a, u[4], u[5], gss, gin, prot.dt, t_rel, t_in0, pes)
                p_hat = ctx.eg.get(a, u[4], u[5], gss)['p_hat']
                for q in range(len(pes)):
                    Yq = Y[:, q, :nx]
                    d = np.angle(Yq[0] * np.conj(Yq[1]))
                    pred = -2 * (pk[:, a] * Al[q][:, a + 1] + pp)
                    r = np.angle(np.exp(1j * (d - pred)))  # good reference: wrapped residual, no temporal unwrap
                    sig = 1 / (snr * np.sqrt(navg))
                    var = 0.5 * sig ** 2 * (1 / np.maximum(np.abs(Yq[0]) ** 2, 1e-30) + 1 / np.maximum(np.abs(Yq[1]) ** 2, 1e-30))
                    D.append(-0.5 * (pred + r))
                    V.append(var / 4)
                    A.append(Al[q])
            c, var = qe.x_solve(np.array(D), np.array(V), np.array(A))
            cs.append((c, var))
        n = int(round(tn[-1] / prot.dt)) + 1
        kin = []
        for iu in inputs:
            gi = np.zeros(n)
            g = ctx.g(iu, 1.0)
            gi[:min(n, len(g))] = g[:n]
            kin.append(gm.cum_k(gi[:, None], prot.dt)[:, 0])
        kin = np.array(kin)
        for l in range(qs.NB):
            if l in (0, a + 1):
                continue
            K = np.array([qe.to_raster(tn, c[:, l], v[:, l], n, prot.dt)[0] for c, v in cs])
            VK = np.array([qe.to_raster(tn, c[:, l], v[:, l], n, prot.dt)[1] for c, v in cs])
            f, H, info = ts.deconvolve(kin, K, VK, prot.dt)
            out[(a, l)] = qe.hp_fit(f, H, info, scale=1e-4 if l < 4 else 3e-5)
            diag[(a, l)] = (f[:4000], H[:4000])
    return out, diag


V2SPEC = None


def v2_spectrum(f):
    """relative |G(f)| of the v2 readout on the GE raster (rotation-invariant vector magnitude)"""
    global V2SPEC
    if V2SPEC is None:
        ro = np.load(os.path.join(PROTO, 'flowtr_ro.npy'))
        nfft = 1 << 18
        ff = np.fft.rfftfreq(nfft, 4e-6)
        S = np.sqrt(sum(np.abs(np.fft.rfft(ro[:, a], nfft)) ** 2 for a in range(3)))
        V2SPEC = (ff, S / S.max())
    return np.interp(f, *V2SPEC)


def res_targets(est, fmin=300.0, fmax=3000.0, min_exposure=0.02):
    """pass-2 targets: parametric-fit resonances in [fmin, fmax] whose exposure to the v2 readout,
    |gain| / ζ · |G_v2(f0)| (the k error a resonance leaves scales with peak height / damping at the readout's
    spectral weight), exceeds min_exposure"""
    t = []
    for a in range(3):
        for f0, z, g in est[a]['par'].res:
            if fmin <= f0 <= fmax and abs(g) > 2e-3 and abs(g) / max(z, 1e-4) * v2_spectrum(f0) > min_exposure:
                t.append((a, f0, z, g))
    return t


def track(ctx, data, t_adc, target, est_bg, ref, bins, iu=0, zeta_fix=None):
    """f0 / gain of one resonance from one thermal set's chirp data (ζ fixed to the pass-2 value)"""
    a, f0, z, g = target
    K, P0, VK, VP, tn = survey_k(ctx, data, a, iu, 1.0, t_adc, ref, bins)
    kin, Kr, VKr, _, _ = raster_set(ctx, [(K, P0, VK, VP)], tn, [ctx.g(iu, 1.0)])
    msk = qe.measured_mask(tn, Kr.shape[1], ctx.prot.dt)
    dkin = np.diff(kin[0])
    dk = np.diff(Kr[0])
    p, cost, sd = qe.res_fit([dkin], [dk], [(VKr[0], msk)], est_bg[a]['tab'], (f0, z, g), ctx.prot.dt, zeta_fix=zeta_fix)
    return p, sd


def own_res(est_a, t):
    """the resonance (f0, ζ, g) of an axis estimate's parametric fit nearest to target t (its own peak in that table)"""
    r = est_a['par'].res
    if not r:
        return (t[1], t[2], 0.0)
    f0, z, g = min(r, key=lambda x: abs(x[0] - t[1]))
    return (f0, z, g) if abs(f0 - t[1]) < max(10 * t[2] * t[1], 15.0) else (t[1], t[2], 0.0)


def match(truth_res, est_res, a):
    """nearest estimated resonance on axis a to each truth resonance"""
    out = []
    for (ta, f0, z, g) in truth_res:
        if ta != a:
            continue
        cand = [e for e in est_res if e[0] == a]
        if not cand:
            out.append(None)
            continue
        e = min(cand, key=lambda e: abs(e[1] - f0))
        out.append(e if abs(e[1] - f0) < max(10 * z * f0, 15.0) else None)
    return out


RC = []


def get_rc():
    if not RC:
        import rcalc
        RC.append(rcalc.RCalc())
    return RC[0]


def R_of(truth_chain, est_chain):
    import v2seq
    rc = get_rc()
    K, P = v2seq.trajectories(rc.v, truth_chain, rc.rots, ref=est_chain)
    q = rc.q_of(K, P)
    r = rc.Rset(q)
    return {k: float(v) for k, v in r.items()}


def v_stage(ctx, est_chain, ref_chain, theta_v):
    """v2 validation: simulate the V frames, estimate k_a(t) / φ0(t) per readout and axis, residuals vs the
    prediction of est_chain (measured) and the truth (noiseless) → q arrays for R"""
    prot = ctx.prot
    F, c = prot.F, prot.col
    m = F[:, c('part')] == prot.parts['v_v2']
    idx = np.flatnonzero(m)
    dt = prot.dt
    nr = prot.v_in.shape[0]
    t_adc = F[idx[0], c('n_adc')] * prot.dwell
    tn = qs.adc_times(t_adc, prot.dwell)
    t_in0 = F[idx[0], c('t_in0_rel_exc_s')]
    thk = F[idx[0], c('thk_m')]
    res = dict(meas=np.zeros((nr, 3, len(tn))), true=np.zeros((nr, 3, len(tn))), pmeas=np.zeros((nr, 3, len(tn))), ptrue=np.zeros((nr, 3, len(tn))), i_ro=np.zeros((nr, 3), int),
               sd=np.zeros((nr, 3, len(tn))))
    nhist = prot.v_R.shape[1] - 1
    for r in range(nr):
        for a in range(3):
            fr = idx[(F[idx, c('readout')] == r) & (F[idx, c('axis')] == a)]
            hist = np.concatenate([prot.flow_tr_played(prot.v_enc[r, j], prot.v_R[r, j]) for j in range(nhist)], 0)
            t_input = F[fr[0], c('t_exc_s')] + t_in0
            n_gap = int(round((t_input - F[fr[0], c('t_hist0_s')]) / dt)) - len(hist)
            vin = prot.v_in[r, a]
            nz = np.flatnonzero(vin)
            vin = vin[:nz[-1] + 2] if len(nz) else vin
            g_hist_axis = hist  # physical 3 axes
            full = np.concatenate([g_hist_axis, np.zeros((max(n_gap, 0), 3)), np.zeros((len(vin), 3))], 0)
            full[-len(vin):, a] = vin
            off = len(full) - len(vin)
            # truth: the measured axis' k (self + cross from the history's other axes + B0 term)
            gin_key = ('v', r, a)
            if gin_key not in ctx.ginputs:
                ctx.ginputs[gin_key] = full
            G = ctx.ginputs[gin_key]
            Ys = {}
            for s in range(len(prot.slices)):
                ss = fr[F[fr, c('slice')] == s]
                navg = max(1, int((F[ss, c('sigma')] > 0).sum()))
                Ys[s] = qs_sim_v(ctx, a, s, G, off, theta_v, t_in0, tn, navg, thk)
            # estimate
            pk, pp = v_pred(ref_chain, a, G, dt, tn, off, t_in0)
            Yp = np.stack([Ys[s][0, 0] for s in Ys])
            Ym = np.stack([Ys[s][1, 0] for s in Ys])
            gss = prot.cfg['g_ss_v']
            pos = np.array([ctx.eg.get(a, s, 1.0, gss)['p_hat'] for s in Ys])
            sig = np.full(len(Ys), 1 / (ctx.sess.snr_body * np.sqrt(navg)))
            K, P0, VK, VP = qe.phase_fit_units(Yp, Ym, pos, sig, pk, pp, unwrap=False)
            ek, ep = v_pred(est_chain, a, G, dt, tn, off, t_in0)
            tk, tp = v_true(ctx, a, G, dt, tn, off, theta_v, t_in0)
            res['meas'][r, a], res['pmeas'][r, a] = K - ek, P0 - ep
            res['sd'][r, a] = np.sqrt(VK)
            res['true'][r, a], res['ptrue'][r, a] = tk - ek, tp - ep
            enc = prot.ft_enc[prot.v_enc[r, nhist]][:, a]
            first = np.flatnonzero(enc)
            i_enc = max(int(first[0]) - 1, 0) if len(first) else 0
            res['i_ro'][r, a] = 50 + len(enc) - i_enc - 1
    return res, tn


def qs_sim_v(ctx, a, s, G, off, theta, t_in0, tn, navg, thk):
    """V-frame signal: the measured axis carries history (all axes through the chain) + the frame input on axis a;
    the truth outputs are computed for the 3-axis waveform"""
    key = ('vk', id(G), round(theta, 6), a)
    if key not in ctx.ginputs:
        ctx.ginputs[key] = v_outputs(ctx.truth, theta, G, ctx.prot.dt)
    k = ctx.ginputs[key]
    return sim_from_k(ctx, a, s, k, G, off, t_in0, tn, navg, thk)


def v_outputs(truth, theta, G, dt):
    """16-basis k of a 3-axis input: Σ over input axes of each axis' outputs (LTI superposition; the odd
    nonlinearity per axis)"""
    k = 0
    for b in range(3):
        if np.abs(G[:, b]).max() == 0:
            continue
        k = k + truth.outputs(theta, b, G[:, b], dt, len(G) + 3200)
    return k


def sim_from_k(ctx, a, s, k, G, off, t_in0, tn, navg, thk):
    """sim_group with precomputed outputs k (3-axis input): concomitant from the measured block's axis-a input"""
    prot = ctx.prot
    orig = ctx.truth.outputs
    ctx.truth.outputs = lambda th, ax, g, dt, n_out=None: k
    try:
        Y = qs.sim_group(ctx.truth, ctx.sess, ctx.geo, prot, a, s, 1.0, np.ascontiguousarray(G[:, a]), off, 0.0, t_in0, tn, navg, [None], thk, ctx.sess.snr_body, prot.dwell, 'body', kcache=None, acache=None)
    finally:
        ctx.truth.outputs = orig
    return Y


def v_pred(chain, a, G, dt, tn, off, t_in0):
    """k_a and φ0 of a chain for the 3-axis V input at the sample times (tn from the measured block start), accrued
    from the thin-slice excitation (t_in0 before the block)"""
    up = qs.UP
    ntail = (int(np.ceil(tn[-1] / dt)) + 20) * up
    x = np.arange((len(G) - 1) * up + 1) / up
    Gu = np.stack([np.interp(x, np.arange(len(G)), G[:, c]) for c in range(3)], 1)  # the same PL waveform on dt/up
    d = dt / up
    y, b = chain.apply(Gu, d, ntail=ntail)
    t = tn + off * dt
    k = gm.k_at(y, d, -gm.PRE * d, t, off * dt - t_in0)
    p = gm.phase_at(b, d, -gm.PRE * d, t, off * dt - t_in0)
    return k[:, a], p


def v_true(ctx, a, G, dt, tn, off, theta, t_in0):
    f = ctx.ginputs[('vk', id(G), round(theta, 6), a)]
    t_s = tn + off * dt + qs.rx_delay(ctx.truth, ctx.prot.dwell, 'body')
    k = qs.k_exact(f, dt, t_s, off * dt - t_in0)
    return k[:, a + 1], k[:, 0]


def v_q(res, tn, dt, sd_max=1.0):
    """V residuals (rad/m, rad) → the gram's q (48 × ns × 4) on its sample times (from the readout start); samples
    whose measured k has sd > sd_max (signal dropouts from in-plane dephasing) are masked (q = 0) in every version;
    returns the q's and the masked fraction"""
    rc = get_rc()
    out = {}
    mask = np.zeros((rc.nr, len(rc.ts)), bool)
    for r in range(min(rc.nr, res['sd'].shape[0])):
        for a in range(3):
            t_ro = (np.round(tn / dt) - res['i_ro'][r, a]) * dt
            mask[r] |= np.interp(rc.ts, t_ro, res['sd'][r, a]) > sd_max
    for name, kk, pp in (('meas', res['meas'], res['pmeas']), ('true', res['true'], res['ptrue'])):
        q = np.zeros((rc.nr, len(rc.ts), 4))
        for r in range(min(rc.nr, kk.shape[0])):
            for a in range(3):
                t_ro = (np.round(tn / dt) - res['i_ro'][r, a]) * dt
                q[r, :, a] = np.interp(rc.ts, t_ro, kk[r, a])
                q[r, :, 3] += np.interp(rc.ts, t_ro, pp[r, a])
        out[name + '_all'] = q.copy()
        q[mask] = 0.0
        out[name] = q
    out['masked_frac'] = float(mask.mean())
    return out


def log(d, *a):
    print(time.strftime('%H:%M:%S'), f'[draw {d}]', *a, flush=True)


def run_draw(job):
    tag, d, opt = job
    t0 = time.time()
    prot = qs.Proto(opt['proto'])
    ctx = Ctx(d, prot, opt['truth'])
    tr = ctx.truth
    clin = gm.clinical_example()
    rec = dict(draw=d, tag=tag, theta=dict(dT_ss=tr.thermal.dT_ss, tau=tr.thermal.tau.tolist()), rx=dict(d4=tr.rx['d4']))
    # ── 1. survey cold, X stage, survey again with the spatial terms removed
    S, t_in0, thk, _, tA = sim_part(ctx, 's_cold')
    S_in = [0, 1, 2, 3, 4]
    # pass A against a generic reference (residual unwrapped in time), pass B against pass A (wrapped residual); the
    # nonparametric H uses the window up to where the median k sd reaches CUT rad/m (the frequency-domain LS weights
    # every sample of an input alike, so dephased late samples would add noise to every bin); the narrow resonances
    # are refined over the full window by sample-weighted time-domain fits (pass 2, tracking)
    e_cold0 = estimate_survey(ctx, S, tA, clin, S_in, unwrap=True, cut_sd=CUT)
    e_cold0 = estimate_survey(ctx, S, tA, girf_from(e_cold0), S_in, cut_sd=CUT)
    g_cold0 = girf_from(e_cold0)
    X, x_in0, _, pes, tX = sim_part(ctx, 'x_3d', coil='array')
    xt, xdiag = x_stage(ctx, X, tX, x_in0, [np.array(p) for p in pes], g_cold0)
    e_cold = estimate_survey(ctx, S, tA, g_cold0, S_in, xterms=xt, cut_sd=CUT)
    g_cold = girf_from(e_cold, xt)
    rec['t_cold'] = time.time() - t0
    log(d, 'cold + X done', rec['t_cold'])
    # ── 2. warm (flow-load) survey
    W, _, _, _, tW = sim_part(ctx, 's_warm')
    e_warm = estimate_survey(ctx, W, tW, g_cold, [0, 2, 4], xterms=xt, cut_sd=CUT)
    g_warm = girf_from(e_warm, xt)
    # ── 3. resonance survey → pass-2 targets; pass-2 simulation at the flow state
    targets = res_targets(e_cold)
    pw2 = ext_power(prot.power, ctx.p_flow, n_targets=len(targets))
    p2 = []
    for j, (a, f0, z, g) in enumerate(targets):
        (g1, _), (g2, _) = qi.targeted(f0, z)
        t_j = len(prot.power) + 600 + 480 + 70 * (j + 0.5)
        th_j = float(tr.thermal.theta(pw2, [t_j])[0])
        dat = {}
        for k_in, gg in ((0, g1), (1, g2)):
            for s in range(len(prot.slices)):
                for eta in (1.0, -1.0):
                    Y = qs.sim_group(tr, ctx.sess, ctx.geo, prot, a, s, eta, gg, 0, th_j, t_in0, qs.adc_times(0.2, 4e-6), 3, [None], thk, ctx.sess.snr_body, 4e-6, 'body', kcache={}, acache={})
                    dat[(0, k_in, 1.0, a, s, eta)] = (Y, 3, th_j, t_j, ctx.sess.snr_body, 4e-6)
        ctx.ginputs[(0, 1.0)] = g1  # survey_k reads inputs by (iu, level): temporarily map 0 → narrow chirp, 1 → tone
        ctx.ginputs[(1, 1.0)] = g2
        ks = []
        for k_in in (0, 1):
            K, P0, VK, VP, tn = survey_k(ctx, dat, a, k_in, 1.0, 0.2, g_warm)
            ks.append((K, P0, VK, VP))
        kin, Kr, VKr, _, _ = raster_set(ctx, ks, tn, [g1, g2])
        msk = qe.measured_mask(tn, Kr.shape[1], prot.dt)
        del ctx.ginputs[(0, 1.0)], ctx.ginputs[(1, 1.0)]
        pj, cost, sdj = qe.res_fit([np.diff(kin[0]), np.diff(kin[1])], [np.diff(Kr[0]), np.diff(Kr[1])], [(VKr[0], msk), (VKr[1], msk)], e_cold[a]['tab'], (f0, z, g), prot.dt)
        p2.append(dict(axis=a, f0_cold=f0, zeta_cold=z, g_cold=g, f0=pj[0], zeta=pj[1], gain=pj[2], delay=pj[3], sd=sdj.tolist(), t=t_j, theta=th_j))
    rec['t_p2'] = time.time() - t0
    log(d, f'pass 2 done ({len(targets)} targets)', rec['t_p2'])
    # ── 4. thermal tracking: s_cold bins, s_warm bins, s_hot bins, d_cool sets (chirp), + the pass-2 points
    H, _, _, _, tH = sim_part(ctx, 's_hot')
    D, _, _, _, tD = sim_part(ctx, 'd_cool')
    sets = [('s_cold', S, tA, b) for b in sorted({k[0] for k in S})] + [('s_warm', W, tW, b) for b in sorted({k[0] for k in W})] + \
           [('s_hot', H, tH, b) for b in sorted({k[0] for k in H})] + [('d_cool', D, tD, b) for b in sorted({k[0] for k in D})]
    trk = []
    for j, (a, f0, z, g) in enumerate(targets):
        zz = p2[j]['zeta']
        rows = []
        for name, dat, tw, b in sets:
            pj, sdj = track(ctx, dat, tw, (a, f0, z, g), e_cold, g_cold, [b], zeta_fix=zz)
            tb = np.mean([v[3] for k, v in dat.items() if k[0] == b])
            thb = np.mean([v[2] for k, v in dat.items() if k[0] == b])
            rows.append((name, int(b), float(tb), float(thb), float(pj[0]), float(sdj[0]), float(pj[2])))
        rows.append(('p2', 0, p2[j]['t'], p2[j]['theta'], p2[j]['f0'], max(p2[j]['sd'][0], 1e-3), p2[j]['gain']))
        trk.append(rows)
    therm = None
    if targets:
        times = [np.array([r[2] for r in rows]) for rows in trk]
        f0s = [np.array([r[4] for r in rows]) for rows in trk]
        sds = [np.maximum(np.array([r[5] for r in rows]), 0.02) for rows in trk]
        therm = qe.fit_thermal(times, f0s, sds, pw2, ctx.p_flow)
    rec['t_track'] = time.time() - t0
    log(d, 'tracking + thermal fit done', rec['t_track'])
    # ── 7. linearity (β) and receiver timing per dwell
    L, _, _, _, tL = sim_part(ctx, 'l_lin')
    rec['lin'] = lin_stage(ctx, L, tL, g_cold)
    rec['rx'] = {}
    for coil in ('body', 'array'):
        T, _, _, _, tT = sim_part(ctx, 't_rx', coil=coil)
        rec['rx'][coil] = rx_stage(ctx, T, g_cold, coil)
    # ── 5. truth at the flow-scan state; estimates; R
    th_ss = tr.thermal.dT_ss * (1.0 if len(tr.thermal.tau) == 2 else 1.0)
    th_ss = float(tr.thermal.kappa.sum() * ctx.p_flow)
    truth_ss = tr.chain(th_ss)
    truth_cold = tr.chain(float(np.mean([v[2] for v in S.values()])))
    truth_warm = tr.chain(float(np.mean([v[2] for v in W.values()])))
    # flow-state model: the non-resonant thermal change interpolated per frequency between the warm and the hot survey
    # (their states bracket the flow scan's steady state x = 1), the targeted resonances moved to the thermal model's
    # steady state
    e_hot = estimate_survey(ctx, H, tH, g_warm, [0, 2], xterms=xt, cut_sd=CUT)
    xw, xh = x_state(therm, pw2, ctx.p_flow, np.mean([v[3] for v in W.values()])), x_state(therm, pw2, ctx.p_flow, np.mean([v[3] for v in H.values()]))
    rec['x_states'] = dict(warm=xw, hot=xh)
    e_flow = interp_tables(e_warm, e_hot, xw, xh, 1.0, targets)
    if targets and therm is not None:
        axis = []
        for a in range(3):
            old, new = [], []
            for j, t in enumerate(targets):
                if t[0] != a:
                    continue
                F_, A_ = therm['FA'][j]
                old.append(own_res(e_warm[a], t))
                new.append((F_ + A_, p2[j]['zeta'], p2[j]['gain']))
            axis.append(qe.with_resonances(e_flow[a]['tab'], old, new) if old else e_flow[a]['tab'])
        g_flow = g_warm.copy_with(axis=axis, name='est_flow')
    else:
        g_flow = g_warm.copy_with(axis=[e_flow[a]['tab'] for a in range(3)], name='est_flow')
    # the imaging ADC (2 µs dwell, array) samples at δ(2 µs, array); the estimate carries δ(4 µs, body) and gets the
    # measured difference (t_rx) added; 'norx' = without that correction
    d_img = qs.rx_delay(tr, 2e-6, 'array')
    d_hat = (rec['rx']['array'][2]['est_ns'] + rec['rx']['array']['abs4_ns']['est']) * 1e-9
    d4 = qs.rx_delay(tr, 4e-6, 'body')
    # sampling δ later = the trajectory advanced by δ: k(t + δ) = with_delay_offset(·, −δ). The estimate already
    # predicts k_true(t + δ4) (measured with the GIRF protocol's ADC); it is advanced by the measured δ̂ = δ_img − δ4
    T = lambda ch: gm.with_delay_offset(ch, -d_img)
    E = lambda ch: gm.with_delay_offset(ch, -d_hat)
    rec['R'] = dict(cold0_vs_cold=R_of(T(truth_cold), E(g_cold0)), cold_vs_cold=R_of(T(truth_cold), E(g_cold)), cold_vs_ss=R_of(T(truth_ss), E(g_cold)),
                    warm_vs_ss=R_of(T(truth_ss), E(g_warm)), flow_vs_ss=R_of(T(truth_ss), E(g_flow)), flow_vs_ss_norx=R_of(T(truth_ss), g_flow),
                    perfect_nocross=R_of(truth_ss, gm.without(truth_ss, 'cross')))
    rec['rx_img'] = dict(d_img_ns=d_img * 1e9, d_hat_ns=d_hat * 1e9, d4_ns=d4 * 1e9)
    rec['t_R'] = time.time() - t0
    log(d, 'R done', rec['t_R'], {k: round(v['refined'], 3) for k, v in rec['R'].items()})
    # ── 6. accuracy per class
    # the estimate predicts k_true(t + δ4) (the GIRF protocol's ADC timing): compare with the truth advanced by δ4
    A4 = lambda ch: gm.with_delay_offset(ch, -d4)
    rec['band_cold'] = smc.band_errors(A4(truth_cold), g_cold, None)
    rec['band_cold0'] = smc.band_errors(A4(truth_cold), g_cold0, None)
    rec['band_warm'] = smc.band_errors(A4(truth_warm), g_warm, None)
    rec['band_flow_ss'] = smc.band_errors(A4(truth_ss), g_flow, None)
    fe = np.geomspace(100, 5e3, 200)
    xe = {}
    for (a, l), r in xt.items():
        if l in (1, 2, 3):
            tru = [rr for o, i, rr in tr.g.cross if o == l - 1 and i == a]
            ht = tru[0].at(fe) if tru else np.zeros_like(fe, complex)
        else:
            tru = [rr for ll, rr in tr.hi[a] if ll == l]
            ht = tru[0].at(fe) if tru else np.zeros_like(fe, complex)
        he = r.at(fe)
        xe[f'{a}_{qs.BASIS[l]}'] = dict(err_rms=float(np.sqrt(np.mean(np.abs(he - ht) ** 2))), true_rms=float(np.sqrt(np.mean(np.abs(ht) ** 2))))
    rec['xterms'] = xe
    tres_cold = tr.resonances(float(np.mean([v[2] for v in S.values()])))
    tres_ss = tr.resonances(th_ss)
    rr = []
    for j, t in enumerate(targets):
        a = t[0]
        tc = min([x for x in tres_cold if x[0] == a], key=lambda x: abs(x[1] - t[1]), default=None)
        if tc is None:
            continue
        jt = [x for x in tres_cold if x[0] == a].index(tc)
        ts_ = [x for x in tres_ss if x[0] == a][jt]
        tr2 = tr.resonances(p2[j]['theta'])
        t2 = [x for x in tr2 if x[0] == a][jt]
        cf = tr.cres[(a, jt)][0]
        pred_ss = therm['FA'][j][0] + therm['FA'][j][1] if therm else np.nan
        rr.append(dict(axis=a, f0_true_cold=tc[1], zeta_true=tc[2], gain_true=tc[3], f0_p1=t[1], zeta_p1=t[2], gain_p1=t[3], f0_true_p2=t2[1], f0_p2=p2[j]['f0'], zeta_p2=p2[j]['zeta'], gain_p2=p2[j]['gain'], gain_true_p2=t2[3],
                       sd_p2=p2[j]['sd'], f0_true_ss=ts_[1], f0_pred_ss=float(pred_ss), cf=cf, dfdT=cf * tc[1], track=trk[j]))
    rec['res'] = rr
    rec['therm'] = therm
    rec['truth_res'] = [list(map(float, x)) for x in tres_cold]
    # ── 8. v2 validation at the V block's thermal state
    m = prot.F[:, prot.col('part')] == prot.parts['v_v2']
    th_v = float(np.mean(ctx.theta[m]))
    g_flow_v = g_flow
    if targets and therm is not None:
        tv = float(np.mean(prot.F[m, prot.col('t_exc_s')]))
        fpred = qe.thermal_predict(therm, pw2, np.array([tv]), ctx.p_flow)
        axis = []
        for a in range(3):
            old, new = [], []
            for j, t in enumerate(targets):
                if t[0] != a:
                    continue
                old.append(own_res(e_warm[a], t))
                new.append((float(fpred[j][0]), p2[j]['zeta'], p2[j]['gain']))
            axis.append(qe.with_resonances(e_warm[a]['tab'], old, new) if old else e_warm[a]['tab'])
        g_flow_v = g_warm.copy_with(axis=axis, name='est_flow_v')
    vres, vtn = v_stage(ctx, g_flow_v, g_warm, th_v)
    vq = v_q(vres, vtn, prot.dt)
    rc = get_rc()
    rec['V'] = dict(theta=th_v, R_meas=float(rc.Rset(vq['meas'])['enc']), R_true=float(rc.Rset(vq['true'])['enc']),
                    R_meas_ref=float(rc.Rset(vq['meas'])['refined']), R_true_ref=float(rc.Rset(vq['true'])['refined']),
                    R_true_all=float(rc.Rset(vq['true_all'])['enc']), masked_frac=vq['masked_frac'],
                    sd_median=float(np.median(vres['sd'])), sd_p90=float(np.percentile(vres['sd'], 90)),
                    dk_rms_meas=float(np.sqrt(np.mean(vres['meas'] ** 2))), dk_rms_true=float(np.sqrt(np.mean(vres['true'] ** 2))),
                    noise_rms=float(np.sqrt(np.mean((vres['meas'] - vres['true']) ** 2))))
    if opt.get('save') and targets and therm is not None:
        import girfmeas_io as gio
        tw = float(np.mean([v[3] for v in W.values()]))
        X = qe.thermal_basis(pw2, np.array([tw]), [therm['tau1'], therm['tau2']])[0] / ctx.p_flow
        x_ref = float(therm['w'] * X[0] + (1 - therm['w']) * X[1])
        own = [own_res(e_warm[t[0]], t) for t in targets]
        rxd = {k: v['est_ns'] for k, v in rec['rx']['body'].items() if isinstance(k, int)}
        doc = gio.write(os.path.join(OUT, f'measured_{tag}_d{d}.json'), e_warm, targets, own, p2, therm, x_ref, xt, rx=rxd,
                        beta=[rec['lin'][a]['beta'] for a in range(3)], p_flow=ctx.p_flow, date='simulated')
        ch = gio.read_chain(doc, x_ref + 0.5)
        fchk = [100.0, 500.0, 700.0, 1000.0, 2000.0, 5000.0]
        doc['check'] = dict(x=x_ref + 0.5, f=fchk, axes=[[[float(v.real), float(v.imag)] for v in ch.axis[a].at(np.array(fchk))] for a in range(3)],
                            b0_500=[[float(v.real), float(v.imag)] for v in [ch.b0[a].at(np.array([500.0]))[0] for a in range(3)]])
        json.dump(doc, open(os.path.join(OUT, f'measured_{tag}_d{d}.json'), 'w'))
        rec['measured_file'] = f'measured_{tag}_d{d}.json'
        rec['x_ref'] = x_ref
    rec['targets'] = [list(map(float, t)) for t in targets]
    rec['p2'] = p2
    rec['sec'] = time.time() - t0
    if opt.get('save'):
        np.savez_compressed(os.path.join(OUT, f'mc_{tag}_d{d}.npz'), **{f'Hc{a}': e_cold[a]['Hc'][:20000] for a in range(3)}, f=e_cold[0]['f'][:20000],
                            **{f'Ht{a}': truth_cold.axis[a].at(e_cold[0]['f'][:20000]) for a in range(3)}, vmeas=vres['meas'], vtrue=vres['true'], vtn=vtn)
    return rec


def x_state(therm, power, p_flow, t):
    """normalised thermal state of the fitted power model at time t (1 = the flow scan's steady state)"""
    if therm is None:
        return float('nan')
    X = qe.thermal_basis(power, np.array([t]), [therm['tau1'], therm['tau2']])[0] / p_flow
    return float(therm['w'] * X[0] + (1 - therm['w']) * X[1])


def interp_tables(e_w, e_h, xw, xh, x, targets):
    """per-axis tables at state x by linear interpolation of H(f) between the warm and the hot estimates, with each
    table's own targeted resonance divided out first (it is put back by the caller at the modelled f0)"""
    out = {}
    for a in range(3):
        tw, th_ = e_w[a]['tab'], e_h[a]['tab']
        f = np.asarray(tw.f)
        hw, hh = tw.at(f), th_.at(f)
        for t in targets:
            if t[0] != a:
                continue
            for h, e in ((hw, e_w), (hh, e_h)):
                f0, z, g = own_res(e[a], t)
                h /= (1 + g * qe.bp(f, f0, z))
        if not np.isfinite(xw) or not np.isfinite(xh) or abs(xh - xw) < 0.05:
            h = hw
        else:
            u = (x - xw) / (xh - xw)
            h = (1 - u) * hw + u * hh
        out[a] = dict(e_w[a])
        out[a]['tab'] = gm.Table(f, h)
        out[a]['par'] = e_w[a]['par']
    return out


def ext_power(power, p_flow, n_targets, gap=600, heat=480, per=70):
    """the pass-1 power timeline + an idle gap (pass-1 estimation) + h3_flow at P_flow + r_target at P_flow"""
    n = gap + heat + per * max(n_targets, 1)
    t0 = power[-1, 0]
    ext = np.zeros((n, 4))
    ext[:, 0] = t0 + 1 + np.arange(n)
    ext[gap:, 1:] = p_flow / 3
    return np.r_[power, ext]


def lin_stage(ctx, L, tL, ref):
    """odd nonlinearity β per axis from tri100 / tri240 at 4 levels: K(ℓ)/ℓ − K(1) = β (ℓ − 1) k₂"""
    out = {}
    prot = ctx.prot
    for a in range(3):
        rows_y, rows_x = [], []
        for iu in (2, 4):
            g1 = prot.input(iu)
            tn = qs.adc_times(tL, 4e-6)
            K1 = None
            for lev in (1.0, 0.75, 0.5, 0.25):
                ctx.ginputs[(iu, lev)] = g1 * lev
                K, P0, VK, VP, tn = survey_k(ctx, L, a, iu, lev, tL, ref)
                if lev == 1.0:
                    K1 = K
                    gnl = g1 * np.abs(g1) / 0.07
                    k2, _ = qe.chain_k(ref, a, gnl, prot.dt, tn)
                    continue
                m = tn < 6e-3
                rows_y.append((K / lev - K1)[m])
                rows_x.append(((lev - 1) * k2[:, a])[m])
        y, x = np.concatenate(rows_y), np.concatenate(rows_x)
        out[a] = dict(beta=float((x * y).sum() / (x * x).sum()), beta_true=float(ctx.truth.g.nl))
    return out


def rx_stage(ctx, T, ref, coil):
    """receiver timing: delay of k_a(t) per dwell relative to 4 µs (both triangles, all axes)"""
    prot = ctx.prot
    out = {}
    est = {}
    for dw in (4, 2, 8, 16):
        sub = {k: v for k, v in T.items() if k[0] == dw}
        data = []
        for a in range(3):
            for iu in (1, 2):
                K, P0, VK, VP, tn = survey_k(ctx, sub, a, iu, 1.0, 10e-3, ref, dwell=dw * 1e-6)
                data.append((a, iu, K, VK, tn))
        d = 0.0
        for it in range(3):  # Gauss–Newton on the sample-time shift δ: K(t) = k_model(t + δ)
            num, den = 0.0, 0.0
            for a, iu, K, VK, tn in data:
                pk, _ = qe.chain_k(ref, a, prot.input(iu), prot.dt, tn + d)
                dk = np.gradient(pk[:, a], tn)
                w = 1 / VK
                num += (w * (K - pk[:, a]) * dk).sum()
                den += (w * dk * dk).sum()
            d += num / den
        est[dw] = d
    for dw in (2, 8, 16):
        out[dw] = dict(est_ns=float((est[dw] - est[4]) * 1e9), true_ns=float((qs.rx_delay(ctx.truth, dw * 1e-6, coil) - qs.rx_delay(ctx.truth, 4e-6, coil)) * 1e9))
    out['abs4_ns'] = dict(est=float(est[4] * 1e9), true=float((qs.rx_delay(ctx.truth, 4e-6, coil) - qs.rx_delay(ctx.truth, 4e-6, 'body')) * 1e9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ndraw', type=int, default=16)
    ap.add_argument('--d0', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--proto', default=PROTO)
    ap.add_argument('--truth', default='rand')
    ap.add_argument('--save', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f'mc_{a.tag}.jsonl')
    done = set()
    if os.path.exists(fn):
        for line in open(fn):
            try:
                done.add(json.loads(line)['draw'])
            except Exception:
                pass
    opt = dict(proto=a.proto, truth=a.truth, save=a.save)
    jobs = [(a.tag, d, opt) for d in range(a.d0, a.d0 + a.ndraw) if d not in done]
    print(time.strftime('%H:%M:%S'), f'{len(jobs)} draws to do ({len(done)} done)', flush=True)
    stop = os.path.join(OUT, 'STOP')

    def dump(rec):
        with open(fn, 'a') as fo:
            fo.write(json.dumps(rec, default=float) + '\n')
        print(time.strftime('%H:%M:%S'), f"draw {rec['draw']}: R flow_vs_ss enc {rec['R']['flow_vs_ss']['enc']:.3f} refined {rec['R']['flow_vs_ss']['refined']:.3f} ({rec['sec']:.0f} s)", flush=True)
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            for rec in pool.imap_unordered(run_draw, jobs):
                dump(rec)
                if os.path.exists(stop):
                    pool.terminate()
                    break
    else:
        for j in jobs:
            dump(run_draw(j))
            if os.path.exists(stop):
                break


if __name__ == '__main__':
    main()
