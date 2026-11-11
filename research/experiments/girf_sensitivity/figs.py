"""Figures of the GIRF sensitivity study -> <OUT>/figs/*.png

    figs.py inputs | h | mc | sens | traj | all
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gmodel as gm, thinslice as ts

OUT = os.path.join(os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute'), 'data/hasty/seq/studies/girf_sensitivity_2026-11-11')
FIG = os.path.join(OUT, 'figs')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED = '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'lines.linewidth': 1.4, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 130})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def fig_inputs():
    p = ts.Protocol()
    ins = p.inputs()
    t = np.arange(len(ins[0][1])) * p.dt * 1e3
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for j, (n, g) in enumerate(ins):
        k = gm.cum_k(g[:, None], p.dt)[:, 0]
        ax[0, 0].plot(t, g * 1e3, color=C[j], lw=1, label=n)
        ax[0, 1].plot(t, k / p.klim, color=C[j], lw=1)
    ax[0, 0].set_xlim(0, 45); ax[0, 0].set_xlabel('time from excitation [ms]'); ax[0, 0].set_ylabel('test gradient [mT/m]')
    ax[0, 0].legend(ncol=2, fontsize=8)
    ax[0, 1].set_xlim(0, 45); ax[0, 1].set_xlabel('time [ms]'); ax[0, 1].set_ylabel('k / k_lim (slice dephasing limit)')
    ax[0, 1].axhline(1, color=MUTED, lw=0.8, ls='--'); ax[0, 1].axhline(-1, color=MUTED, lw=0.8, ls='--')
    nfft = 1 << 16
    f = np.fft.rfftfreq(nfft, p.dt)
    tot = 0
    for j, (n, g) in enumerate(ins):
        k = gm.cum_k(g[:, None], p.dt)[:, 0]
        X = np.abs(np.fft.rfft(np.diff(k), nfft))
        ax[1, 0].loglog(f[1:], X[1:], color=C[j], lw=0.8)
        tot = tot + X ** 2
    ax[1, 0].set_xlim(10, 50e3); ax[1, 0].set_ylim(1e-2, 1e4); ax[1, 0].set_xlabel('f [Hz]'); ax[1, 0].set_ylabel('|FFT of k increments| [rad/m]')
    # predicted sd of H per bin for 1 average at SNR 1000 (white k noise of the 4-slice fit)
    sdk = 1 / 1000 / np.sqrt(2) / np.sqrt((p.slices ** 2).sum() * 4) * 2  # rough: phase noise / lever arm
    n = len(ins[0][1])
    s2 = 4 * np.sin(np.pi * f * p.dt) ** 2
    sdH = np.sqrt(n * sdk ** 2 * s2 / np.maximum(tot, 1e-300))
    for navg, ls in ((50, '-'), (300, '--')):
        ax[1, 1].loglog(f[1:], sdH[1:] / np.sqrt(navg), color=C[0], ls=ls, label=f'{navg} averages')
    ax[1, 1].set_xlim(10, 50e3); ax[1, 1].set_ylim(1e-7, 1e-1); ax[1, 1].set_xlabel('f [Hz]'); ax[1, 1].set_ylabel('predicted sd of H(f) per bin (SNR0 1000)')
    ax[1, 1].legend()
    fig.suptitle('Thin-slice GIRF inputs (4 us raster): log chirp 0.1-20 kHz + 4 prephased triangles; k kept below the slice-dephasing limit', fontsize=9)
    save(fig, 'inputs.png')


def fig_h():
    """true vs estimated H for draw 0 of a rand and a resonance MC run"""
    import truths
    for tag in sorted({os.path.basename(p).split('_d0_')[0][3:] for p in glob.glob(os.path.join(OUT, 'mc_*_d0_H.npz'))}):
        z = np.load(os.path.join(OUT, f'mc_{tag}_d0_H.npz'))
        recs = [json.loads(l) for l in open(os.path.join(OUT, f'mc_{tag}.jsonl'))]
        r0 = [r for r in recs if r['draw'] == 0]
        if not r0:
            continue
        opt = r0[0]['opt']
        truth = truths.draw(np.random.default_rng(5000)) if opt['truth'] == 'rand' else gm.variant(opt['truth'])
        fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for a in range(3):
            f, H, Hc, sd = z[f'f{a}'], z[f'H{a}'], z[f'Hc{a}'], z[f'sd{a}']
            m = (f > 5) & (f < 20e3)
            ht = truth.axis[a].at(f[m])
            th = r0[0]['fit'][str(a)]['th']; nres = r0[0]['fit'][str(a)]['nres']
            hp = ts.par_resp(np.array(th), nres).at(f[m])
            ax[0, a].semilogx(f[m], np.abs(ht), color=MUTED, lw=2.5, alpha=0.5, label='truth')
            ax[0, a].semilogx(f[m], np.abs(Hc[m]), color=C[0], lw=0.8, label='nonparametric')
            ax[0, a].semilogx(f[m], np.abs(hp), color=C[1], lw=0.8, ls='--', label='parametric fit')
            ax[0, a].set_ylim(0.6, 1.15); ax[0, a].set_title(f'axis {"xyz"[a]}  |H|')
            ax[1, a].loglog(f[m], np.abs(Hc[m] - ht), color=C[0], lw=0.6, label='|nonpar - truth| (incl. DC gain error)')
            ax[1, a].loglog(f[m], np.abs(hp - ht), color=C[1], lw=0.6, label='|par - truth| (free DC gain)')
            ax[1, a].loglog(f[m], sd[m], color=MUTED, lw=0.8, ls=':', label='predicted sd')
            ax[1, a].set_ylim(1e-7, 1e-1); ax[1, a].set_xlabel('f [Hz]')
        ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8)
        fig.suptitle(f'MC {tag} draw 0: estimated self terms ({r0[0]["minutes"]:.0f} min of GIRF frames, SNR0 {opt["snr"]:g})', fontsize=9)
        save(fig, f'H_{tag}.png')


def load_mc():
    out = {}
    for fn in sorted(glob.glob(os.path.join(OUT, 'mc_*.jsonl'))):
        tag = os.path.basename(fn)[3:-6]
        out[tag] = [json.loads(l) for l in open(fn)]
    return out


def fig_mc():
    mc = load_mc()
    sets = [('R_rand_10', 'ADC 80 ms, 10 min'), ('R_rand_T200', 'ADC 200 ms, 11 min'), ('R_rand_T200_60', 'ADC 200 ms, 66 min'),
            ('R_rand_s300', 'ADC 80 ms, 10 min, SNR0 300'), ('R_rand_nonuis', 'ADC 80 ms, 10 min, no nuisances')]
    sets = [(t, l) for t, l in sets if t in mc]
    if not sets:
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for k, (lvl, title) in enumerate([('enc', 'global R, enc (per-encode phase poly absorbed)'), ('refined', 'global R, + field deg 1 + per-axis delay refit'),
                                      ('loc_refined', 'vessel-local R, refined')]):
        for j, (t, lab) in enumerate(sets):
            R = np.sort([r['R_np'][lvl] for r in mc[t]])
            ax[k].semilogx(R, (np.arange(len(R)) + 1) / len(R), color=C[j], lw=1.4, drawstyle='steps-post', label=lab if k == 0 else None)
        R = np.sort([r['R_np_nocross_ref'][lvl] for r in mc[sets[0][0]]])
        ax[k].semilogx(R, (np.arange(len(R)) + 1) / len(R), color=MUTED, lw=1.0, ls='--', drawstyle='steps-post', label='perfect except cross terms' if k == 0 else None)
        for x in (0.2, 0.5):
            ax[k].axvline(x, color=MUTED, lw=0.8, ls=':')
        ax[k].set_title(title, fontsize=9); ax[k].set_xlabel('R')
    ax[0].set_ylabel('fraction of MC draws with R below')
    ax[0].legend(fontsize=7, loc='lower right')
    fig.suptitle('v2 readout after GIRF correction (nonparametric estimate): random realistic chains + session nuisances + noise', fontsize=9)
    save(fig, 'R_cdf.png')


def fig_sens():
    fn = os.path.join(OUT, 'sens.json')
    if not os.path.exists(fn):
        return
    s = json.load(open(fn))
    rows = []
    for sec in [k for k in s if k.startswith('classes_')]:
        for key, r in s[sec].items():
            if key.endswith('_0.1'):
                rows.append((sec[8:], key[:-4], r))
    if not rows:
        return
    names = sorted({r[1] for r in rows}, key=lambda c: gm.CLASSES.index(c) if c in gm.CLASSES else 99)
    truths_ = sorted({r[0] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for k, lvl in enumerate(['raw', 'refined']):
        w = 0.8 / len(truths_)
        for j, tn in enumerate(truths_):
            y = [next((r[2][lvl] for r in rows if r[0] == tn and r[1] == c), np.nan) for c in names]
            ax[k].bar(np.arange(len(names)) + j * w, y, width=w * 0.92, color=C[j], label=tn)
        ax[k].set_yscale('log'); ax[k].axhline(0.2, color=MUTED, ls='--', lw=0.8)
        ax[k].set_xticks(np.arange(len(names)) + 0.4 - w / 2); ax[k].set_xticklabels(names, rotation=30, fontsize=8)
        ax[k].set_title(f'R ({lvl}) for a 10 % single-class model error (res_f0: 1 %)', fontsize=9)
    ax[0].legend(fontsize=8); ax[0].set_ylabel('R')
    save(fig, 'sens_classes.png')


def fig_traj():
    """residual trajectory of one readout: omission vs nonparametric / parametric estimate (rand0 truth, MC draw 0
    settings: SNR0 1000, 50 averages)"""
    import truths, v2seq, mc
    truth = truths.draw(np.random.default_rng(5000), 'rand0')
    rng = np.random.default_rng(1000)
    prot = ts.Protocol(snr0=1000, n_avg=50)
    nu = ts.Nuisance(rng, prot)
    Y, sig = ts.simulate(truth, prot, nu, rng)
    RK, RP = mc.ref_predictions(prot, gm.clinical_example())
    ep, en, diag = mc.estimate(Y, sig, prot, RK, RP)
    v = v2seq.V2()
    rots = v2seq.scan_rotations(48, 7, 10)
    one = {k: x[:1] for k, x in rots.items()}
    t = v.t * 1e3
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
    for j, (lab, model) in enumerate([('nominal (no GIRF)', gm.ideal()), ('nonparametric estimate', en), ('parametric estimate', ep)]):
        K, P = v2seq.trajectories(v, truth, one, ref=model)
        for a in range(3):
            ax[0 if j == 0 else 1, 0].plot(t, K[0, :, a] / 2 / np.pi, color=C[a], lw=1.0 if j < 2 else 0.8, ls='-' if j < 2 else ':', label=f'{"xyz"[a]} ({lab})' if j != 0 or True else None)
        ax[0 if j == 0 else 1, 1].plot(t, P[0] * 1e3, color=C[j + 3], lw=1, label=lab)
    kq = v.kabs / v.kabs.max()
    for r in range(2):
        for c in range(2):
            ax[r, c].axvspan(0, t[np.flatnonzero(kq > 0.1)[0]], color='#e4e3df', alpha=0.6, lw=0)
            ax[r, c].axvline(t[v.na - 1], color=MUTED, lw=0.8, ls='--')
            ax[r, c].set_xlabel('time from echo [ms]')
    ax[0, 0].set_ylabel('k error, nominal model [cycles/m]'); ax[1, 0].set_ylabel('k error after GIRF correction [cycles/m]')
    ax[0, 1].set_ylabel('k0 (B0-term) phase error [mrad]'); ax[1, 1].set_ylabel('k0 phase error [mrad]')
    ax[0, 0].legend(fontsize=7, ncol=3); ax[1, 0].legend(fontsize=7, ncol=2); ax[0, 1].legend(fontsize=7)
    fig.suptitle('v2 readout 0 of the scan, random realistic chain rand0: played minus modelled trajectory (shaded: |k| < 0.1 kmax; dashed: end of the recon window, navigator after)', fontsize=9)
    save(fig, 'traj_residual.png')


def readout_spectrum():
    import v2seq
    v = v2seq.V2()
    gb = np.r_[np.zeros((200, 3)), v.base, np.zeros((200, 3))]
    nf = 1 << 16
    f = np.fft.rfftfreq(nf, v.dt)
    S = np.sqrt((np.abs(np.fft.rfft(gb, nf, axis=0)) ** 2).sum(1))
    return f, S / S.max()


def fig_sweep():
    """R vs the frequency of a resonance (peak 5 %) whose f0 is off by 0.05 % or whose height is off by 10 %, with the
    v2 readout's amplitude spectrum"""
    fn = os.path.join(OUT, 'sens.json')
    S = json.load(open(fn))
    f, sp = readout_spectrum()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, (case, lab) in enumerate([('f0+0.05pct', 'f0 off by 0.05 %'), ('gain+10pct', 'peak height off by 10 % (0.5 % abs)')]):
        for j, z in enumerate(('0.005', '0.03')):
            d = S.get(f'res_sweep_z{z}', {})
            pts = sorted((float(key.split('_')[0]), r) for key, r in d.items() if key.endswith(case))
            if not pts:
                continue
            x = [p[0] for p in pts]
            ax[k].semilogy(x, [p[1]['enc'] for p in pts], color=C[j], marker='o', ms=3, label=f'zeta {z}: R (enc)')
            ax[k].semilogy(x, [p[1]['loc_refined'] for p in pts], color=C[j], ls=':', marker='o', ms=2, label=f'zeta {z}: local R (refined)')
        a2 = ax[k].twinx()
        a2.plot(f, sp, color=MUTED, lw=0.8, alpha=0.6)
        a2.set_ylim(0, 1.05); a2.set_ylabel('v2 readout |G(f)| (rel.)', color=MUTED); a2.grid(False)
        a2.spines['right'].set_visible(True)
        ax[k].set_xlim(250, 4000); ax[k].axhline(0.2, color=MUTED, ls='--', lw=0.8)
        ax[k].set_xlabel('resonance frequency f0 [Hz]'); ax[k].set_ylabel('R'); ax[k].set_title(lab, fontsize=9)
        ax[k].legend(fontsize=7, loc='upper right')
    fig.subplots_adjust(wspace=0.45)
    fig.suptitle('Where the v2 readout is vulnerable: one resonance (peak 5 %) on every axis of the clinical chain, model error as stated', fontsize=9)
    save(fig, 'res_sweep.png')


def fig_risk():
    """per MC draw (R_rand_10): R vs the truth's resonance exposure max_j g_j / zeta_j |G_ro(f0_j)|"""
    import truths
    f, sp = readout_spectrum()
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    for j, tag in enumerate(['R_rand_10', 'R_rand_T200']):
        fn = os.path.join(OUT, f'mc_{tag}.jsonl')
        if not os.path.exists(fn):
            continue
        rows = []
        for l in open(fn):
            r = json.loads(l)
            t = truths.draw(np.random.default_rng(5000 + r['draw']))
            risk = max(g / z * np.interp(f0, f, sp) for a in range(3) for f0, z, g in t.axis[a].parts[0].res)
            rows.append((risk, r['R_np']['enc']))
        x, y = np.array(rows).T
        ax.loglog(x, y, 'o', ms=4, color=C[j], label={'R_rand_10': 'ADC 80 ms, 50 averages', 'R_rand_T200': 'ADC 200 ms, 25 averages'}[tag])
    ax.axhline(0.2, color=MUTED, ls='--', lw=0.8); ax.axhline(0.5, color=MUTED, ls=':', lw=0.8)
    ax.set_xlabel('resonance exposure  max_j (peak_j / zeta_j) |G_readout(f0_j)|'); ax.set_ylabel('R (enc level), nonparametric estimate')
    ax.legend(fontsize=8)
    ax.set_title('Residual R per MC draw vs the truth chain\'s resonances at the readout spectrum', fontsize=9)
    save(fig, 'R_vs_resonance.png')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for n, f in [('inputs', fig_inputs), ('h', fig_h), ('mc', fig_mc), ('sens', fig_sens), ('traj', fig_traj), ('sweep', fig_sweep), ('risk', fig_risk)]:
        if what in (n, 'all'):
            f()
