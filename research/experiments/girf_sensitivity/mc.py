"""Monte Carlo of the thin-slice GIRF measurement -> estimate -> v2 readout residual -> R.

One draw = (truth chain, nuisance draw, noise). Estimates: 'par' (parametric chain fitted to the nonparametric H),
'np' (nonparametric H below f_trust, raster-compensated to the continuous response, parametric extension above);
cross terms are not measured by the self-term slices (left at 0 in both estimates; see the note).
Results are appended to <OUT>/mc_<tag>.jsonl (resumable: draws already present are skipped).

    mc.py --tag T --truth resonance|rand|clinical --snr 1000 --navg 50 --ndraw 40 [--workers 6] [--thk 1.5e-3]
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gmodel as gm, thinslice as ts, truths, v2seq

OUT = os.path.join(v2seq.REPO, 'data/hasty/seq/studies/girf_sensitivity_2026-11-11')
BANDS = [(10, 100), (100, 500), (500, 2000), (2000, 5000), (5000, 10000)]


def ref_predictions(prot, chain):
    n = int(round(prot.t_adc / prot.dt)); t = np.arange(n) * prot.dt
    RK, RP = {}, {}
    for a in range(3):
        K, P = [], []
        for _, g in prot.inputs():
            g3 = np.zeros((n, 3)); g3[:, a] = g
            y, b = chain.apply(g3, prot.dt, ntail=0)
            K.append(gm.k_at(y, prot.dt, -gm.PRE * prot.dt, t, 0.0)[:, a]); P.append(gm.phase_at(b, prot.dt, -gm.PRE * prot.dt, t, 0.0))
        RK[a], RP[a] = np.array(K), np.array(P)
    return RK, RP


def to_continuous(f, Hpl, par, dt, M=40):
    """continuous response from the PL-discrete one: (H_pl - sum_{m != 0} sinc^2(f dt + m) H_par(f + m/dt)) /
    sinc^2(f dt) (the 'dwell-time compensation' in the PL convention)"""
    al = np.zeros(f.shape, complex)
    for m in range(-M, M + 1):
        if m:
            al += gm.sinc2(f * dt + m) * par.at(f + m / dt)
    return (Hpl - al) / gm.sinc2(f * dt)


def estimate(Y, sig, prot, RK, RP, fit_fmax=15e3, sd_trust=2e-3):
    """per axis: k / phi0 fit, deconvolution, parametric fit, B0 fit -> (est_par, est_np, diagnostics)"""
    n = Y[0].shape[-1]
    kin = np.array([gm.cum_k(g[:, None], prot.dt)[:, 0] for _, g in prot.inputs()])
    ax_par, ax_np, b0_par, b0_np, diag = [], [], [], [], {}
    for a in range(3):
        K, P0, VK, VP = ts.phase_fit(Y[a], prot, sig, RK[a], RP[a])
        f, H, info = ts.deconvolve(kin, K, VK, prot.dt)
        _, H0, info0 = ts.deconvolve(kin, P0, VP, prot.dt)
        sd = np.sqrt((n - 1) / np.maximum(info, 1e-300))
        th, nres, sol = ts.fit_axis(f, H, sd, fmax=fit_fmax, dt=prot.dt)
        par = ts.par_resp(th, nres)
        # B0 term: two high-pass terms + c * H_aa (slice-position cross-talk k p_mean, absorbable); units 1e-5 m
        m = (f > 0) & (f <= 5e3)
        sd0 = np.sqrt((n - 1) / np.maximum(info0, 1e-300))
        fl0 = 2e-7
        ff, hh, ww = f[m][::2], H0[m][::2] / 1e-5, (fl0 / np.sqrt(sd0[m] ** 2 + fl0 ** 2))[::2]
        Hm = H[m][::2]
        from scipy.optimize import least_squares

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
        sb = best
        sb.x = np.array(sb.x); sb.x[[0, 2, 4]] *= 1e-5
        bpar = gm.MultiHP([(sb.x[0], np.exp(sb.x[1])), (sb.x[2], np.exp(sb.x[3]))])
        # nonparametric tables, continuous, trusted band by the predicted sd of H
        Hc = to_continuous(f, H, par, prot.dt)
        ok = (sd < sd_trust) & (f > 0)
        f_trust = float(f[np.flatnonzero(~ok & (f > 1e3))[0]]) if (~ok & (f > 1e3)).any() else float(f[-1])
        f_trust = min(f_trust, 30e3)
        tab = ts.hybrid_table(f, Hc, info, par, f_trust)
        H0c = to_continuous(f, H0 - sb.x[4] * H, bpar, prot.dt)
        ok0 = (sd0 < 2e-7) & (f > 0)
        f0t = float(f[np.flatnonzero(~ok0 & (f > 200))[0]]) if (~ok0 & (f > 200)).any() else 5e3
        f0t = min(max(f0t, 500.0), 10e3)
        tab0 = ts.hybrid_table(f, H0c, info0, bpar, f0t)
        ax_par.append(par); ax_np.append(tab); b0_par.append(bpar); b0_np.append(tab0)
        diag[a] = dict(th=th.tolist(), nres=int(nres), cost=float(sol.cost), f_trust=f_trust, f0_trust=f0t, b0x=sb.x.tolist(),
                       f=f, H=H, Hc=Hc, H0=H0, sd=sd)
    est_par = gm.Girf(ax_par, b0=b0_par, tail=0.15, name='est_par')
    est_np = gm.Girf(ax_np, b0=b0_np, tail=0.15, name='est_np')
    return est_par, est_np, diag


def band_errors(truth, est, diag, dt=1e-5):
    """rms |H_est - H_true| (continuous responses) per band, self terms per axis, raw and after removing the DC gain;
    B0 terms (m) per band"""
    out = {}
    f = np.r_[np.geomspace(10, 10e3, 600)]
    for a in range(3):
        ht, he = truth.axis[a].at(f), est.axis[a].at(f)
        g = (he[:20] / ht[:20]).mean()
        b0t = truth.b0[a].at(f) if truth.b0[a] is not None else np.zeros_like(f, complex)
        b0e = est.b0[a].at(f) if est.b0[a] is not None else np.zeros_like(f, complex)
        for lo, hi in BANDS:
            m = (f >= lo) & (f < hi)
            out[f'H{a}_{lo}_{hi}'] = float(np.sqrt((np.abs(he[m] - ht[m]) ** 2).mean()))
            out[f'Hg{a}_{lo}_{hi}'] = float(np.sqrt((np.abs(he[m] / g - ht[m]) ** 2).mean()))
            out[f'B{a}_{lo}_{hi}'] = float(np.sqrt((np.abs(b0e[m] - b0t[m]) ** 2).mean()))
        out[f'gain{a}'] = float(abs(g) - 1)
        # group delay at 1-3 kHz (effective delay incl. the low-pass), ns
        ff = np.linspace(1e3, 3e3, 50)
        gd = lambda r: -np.gradient(np.unwrap(np.angle(r.at(ff))), 2 * np.pi * ff).mean()
        out[f'gd{a}_ns'] = float((gd(est.axis[a]) - gd(truth.axis[a])) * 1e9)
    return out


def param_errors(truth, est_par):
    """for truths in the fitted family (girf.rs variants with one resonance per axis): parameter errors"""
    out = {}
    for a in range(3):
        t = truth.axis[a]
        tb = t.base if isinstance(t, gm.Resonant) else t
        if not isinstance(tb, gm.Param):
            return out
        e = est_par.axis[a]
        eb = e.base if isinstance(e, gm.MultiRes) else e
        eb = eb.base if isinstance(eb, gm.Scaled) else eb
        out[f'delay{a}_ns'] = (eb.delay - tb.delay) * 1e9
        out[f'lp{a}_pct'] = (eb.tau_lp / tb.tau_lp - 1) * 100 if tb.tau_lp else 0.0
        te = sorted(tb.eddy, key=lambda x: x[1]); ee = sorted(eb.eddy, key=lambda x: x[1])
        for j, ((ta, tt), (ea, et)) in enumerate(zip(te, ee)):
            out[f'eddy{a}_{j}_a_pct'] = (ea / ta - 1) * 100
            out[f'eddy{a}_{j}_tau_pct'] = (et / tt - 1) * 100
        if isinstance(t, gm.Resonant) and isinstance(e, gm.MultiRes) and e.res:
            j = int(np.argmin([abs(r[0] - t.f0) for r in e.res]))
            f0, z, g = e.res[j]
            out[f'res{a}_f0_hz'] = f0 - t.f0
            out[f'res{a}_zeta_pct'] = (z / t.zeta - 1) * 100
            out[f'res{a}_gain_pct'] = (g / t.gain - 1) * 100
        if truth.b0[a] is not None and isinstance(truth.b0[a], gm.HighPass):
            kb = est_par.b0[a].terms
            j = int(np.argmax([abs(k) for k, _ in kb]))
            out[f'b0{a}_k_pct'] = (kb[j][0] / truth.b0[a].k - 1) * 100
            out[f'b0{a}_tau_pct'] = (kb[j][1] / truth.b0[a].tau - 1) * 100
    return out


def run_draw(job):
    tag, d, opt = job
    t0 = time.time()
    import rcalc
    rng = np.random.default_rng(1000 + d)
    tadc = opt.get('tadc') or 80e-3
    prot = ts.Protocol(snr0=opt['snr'], n_avg=opt['navg'], thk=opt['thk'], hadamard=opt.get('hadamard', False), t_adc=tadc, t_frame=tadc + 0.02)
    prot.ss_alt = bool(opt.get('ss_alt'))
    if opt.get('chirp_s'):
        prot.s_chirp = opt['chirp_s']
    truth = truths.draw(np.random.default_rng(5000 + d)) if opt['truth'] == 'rand' else gm.variant(opt['truth']) if opt['truth'] != 'clinical' else gm.clinical_example()
    if opt['truth'] == 'clinical':
        truth.tail = 30e-3
    nu = ts.Nuisance(rng, prot)
    off = dict(center=np.zeros(3), b0_rms_hz=1e-6, coilw=np.zeros(3), df0_hz=0.0, eps_ss=0.0, dp_rand=np.zeros((3, 4)), k_ss=0.0)
    if opt.get('nuis') == 'off':
        nu = ts.Nuisance(np.random.default_rng(1000 + d), prot, **off)
    elif str(opt.get('nuis', '')).startswith('only:'):
        off.pop(opt['nuis'][5:])
        nu = ts.Nuisance(np.random.default_rng(1000 + d), prot, **off)
    Y, sig = ts.simulate(truth, prot, nu, rng)
    RK, RP = ref_predictions(prot, gm.clinical_example())
    est_par, est_np, diag = estimate(Y, sig, prot, RK, RP)
    rc = RCALC[0] if RCALC else None
    rec = dict(draw=d, tag=tag, opt=opt, minutes=prot.minutes(), nuis=dict(T2=nu.T2, b0_rms_hz=nu.b0_rms_hz, df0_hz=nu.df0_hz, eps_ss=nu.eps_ss, center=nu.center.tolist(), coilw=float(np.linalg.norm(nu.coilw))))
    for name, est in ([] if opt.get('no_r') else [('par', est_par), ('np', est_np), ('np_nocross_ref', None)]):
        if est is None:
            # what is left if the estimate were perfect except for the unmeasured cross terms
            est = gm.without(truth, 'cross')
        K, P = v2seq.trajectories(rc.v, truth, rc.rots, ref=est)
        q = rc.q_of(K, P)
        rec['R_' + name] = rc.Rset(q)
        rec['dk_rms_' + name] = float(np.sqrt((K[:, rc.v.recon] ** 2).sum(-1).mean()) / 2 / np.pi)
        rec['dphi_rms_' + name] = float(np.sqrt((P[:, rc.v.recon] ** 2).mean()))
    rec['band_par'] = band_errors(truth, est_par, diag)
    rec['band_np'] = band_errors(truth, est_np, diag)
    rec['param'] = param_errors(truth, est_par)
    rec['fit'] = {a: dict(th=diag[a]['th'], nres=diag[a]['nres'], f_trust=diag[a]['f_trust'], f0_trust=diag[a]['f0_trust']) for a in range(3)}
    rec['sec'] = time.time() - t0
    if d == 0 or opt.get('save_h'):
        np.savez_compressed(os.path.join(OUT, f'mc_{tag}_d{d}_H.npz'), **{f'{k}{a}': diag[a][k][:8192] for a in range(3) for k in ('f', 'H', 'Hc', 'H0', 'sd')})
    return rec


RCALC = []


def summary(rec):
    if 'R_par' in rec:
        return f"draw {rec['draw']}: R par {rec['R_par']['raw']:.3g}/{rec['R_par']['refined']:.3g}, np {rec['R_np']['raw']:.3g}/{rec['R_np']['refined']:.3g} ({rec['sec']:.0f}s)"
    return f"draw {rec['draw']}: gd np " + ' '.join('%.1f' % rec['band_np'][f'gd{a}_ns'] for a in range(3)) + f" ({rec['sec']:.0f}s)"


NO_R = [False]


def init_worker():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if NO_R[0]:
        return
    import rcalc
    RCALC.append(rcalc.RCalc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--truth', default='resonance')
    ap.add_argument('--snr', type=float, default=1000.0)
    ap.add_argument('--navg', type=int, default=50)
    ap.add_argument('--thk', type=float, default=1.5e-3)
    ap.add_argument('--ndraw', type=int, default=40)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--nuis', default='on')
    ap.add_argument('--hadamard', action='store_true')
    ap.add_argument('--chirp-s', type=float, default=0.0)
    ap.add_argument('--no-r', action='store_true', help='estimation accuracy only (no Gram / R)')
    ap.add_argument('--tadc', type=float, default=0.0, help='ADC window (s), default 80 ms; frame = window + 20 ms')
    ap.add_argument('--ss-alt', action='store_true', help='alternate the slice-select polarity over the averages')
    a = ap.parse_args()
    opt = dict(truth=a.truth, snr=a.snr, navg=a.navg, thk=a.thk, nuis=a.nuis, hadamard=a.hadamard, chirp_s=a.chirp_s, no_r=a.no_r, tadc=a.tadc, ss_alt=a.ss_alt)
    NO_R[0] = a.no_r
    fn = os.path.join(OUT, f'mc_{a.tag}.jsonl')
    done = set()
    if os.path.exists(fn):
        for line in open(fn):
            try:
                done.add(json.loads(line)['draw'])
            except Exception:
                pass
    jobs = [(a.tag, d, opt) for d in range(a.ndraw) if d not in done]
    print(time.strftime('%H:%M:%S'), f'{a.tag}: {len(jobs)} draws to do ({len(done)} done)', flush=True)
    stop = os.path.join(OUT, 'STOP')
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers, initializer=init_worker) as pool:
            for rec in pool.imap_unordered(run_draw, jobs):
                with open(fn, 'a') as fo:
                    fo.write(json.dumps(rec) + '\n')
                print(time.strftime('%H:%M:%S'), summary(rec), flush=True)
                if os.path.exists(stop):
                    print('STOP file: terminating', flush=True)
                    pool.terminate()
                    break
    else:
        init_worker()
        for j in jobs:
            rec = run_draw(j)
            with open(fn, 'a') as fo:
                fo.write(json.dumps(rec) + '\n')
            print(time.strftime('%H:%M:%S'), summary(rec), flush=True)
            if os.path.exists(stop):
                break


if __name__ == '__main__':
    main()
