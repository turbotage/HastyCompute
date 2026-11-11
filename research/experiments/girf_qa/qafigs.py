"""Figures of the GIRF QA protocol and Monte Carlo (data/hasty/seq/girf_qa/figs).

    qafigs.py protocol            # inputs, one frame, the timeline (power + example thermal state), PNS, a V frame
    qafigs.py mc <tag> [<tag>...] # R CDFs, resonance / thermal / cross-term errors, receiver timing, V residuals
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
PROTO = os.path.join(REPO, 'data/hasty/seq/girf_qa/protocol')
FIGS = os.path.join(REPO, 'data/hasty/seq/girf_qa/figs')
MC = os.path.join(REPO, 'data/hasty/seq/girf_qa/mc')
# reference palette (dataviz skill, light mode): categorical slots in fixed order; text / grid tokens
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
AX = ['x', 'y', 'z']
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.5, 'font.size': 9, 'legend.frameon': False, 'axes.titlesize': 10})
GAM = 2.6752218744e8


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name), dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(FIGS, name))


def protocol():
    p = json.load(open(os.path.join(PROTO, 'protocol.json')))
    X = np.load(os.path.join(PROTO, 'inputs.npy'))
    dt = p['config']['grad_raster']
    klim = p['config']['klim']
    # 1. inputs: g(t) and |k(t)| / k_lim
    ins = p['inputs']
    n = len(ins)
    fig, axs = plt.subplots(n, 2, figsize=(10, 1.6 * n), sharex='col')
    for i, inp in enumerate(ins):
        g = X[i, :inp['n']]
        t = np.arange(len(g)) * dt * 1e3
        k = np.r_[0, np.cumsum(0.5 * (g[1:] + g[:-1]))] * dt * GAM
        axs[i, 0].plot(t, g * 1e3, color=C[0])
        axs[i, 0].set_ylabel(f"{inp['name']}\nmT/m")
        axs[i, 1].plot(t, k / klim, color=C[1])
        axs[i, 1].set_ylabel('k / k_lim')
        axs[i, 1].set_ylim(-1, 1)
    axs[0, 0].set_title('test input on the slice axis (unit level)')
    axs[0, 1].set_title('k(t) relative to the slice limit k_lim = 0.4·2π/thk')
    axs[-1, 0].set_xlabel('time from the input-block start (ms)')
    axs[-1, 1].set_xlabel('time from the input-block start (ms)')
    save(fig, 'protocol_inputs.png')
    # 2. timeline: parts, power / P_flow, example thermal state
    pw = np.load(os.path.join(PROTO, 'power.npy'))
    pf = p['flow_tr']['p_flow_T2m2']
    P = pw[:, 1:].sum(1) / pf
    ker = np.ones(15) / 15
    Ps = np.convolve(P, ker, mode='same')
    t = pw[:, 0] / 60
    th = np.zeros(len(P))
    xs = []
    for tau, w in ((200.0, 0.5 * 0.85), (1500.0, 0.5 * 0.85), (6000.0, 0.15)):
        a = np.exp(-1 / tau)
        x, out = 0.0, np.zeros(len(P))
        for i, v in enumerate(P):
            x = x * a + v * (1 - a)
            out[i] = x
        th += 7.0 * w * out
    fig, axs = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    axs[0].fill_between(t, 0, Ps, color=C[0], alpha=0.25, lw=0)
    axs[0].plot(t, Ps, color=C[0], lw=1.0)
    axs[0].axhline(1.0, color=INK2, lw=0.8, ls='--')
    axs[0].text(t[-1], 1.02, 'flow-scan load', ha='right', va='bottom', color=INK2, fontsize=8)
    axs[0].set_ylabel('gradient power / P_flow\n(15 s mean)')
    for q in p['parts']:
        t0, t1 = q['t0_s'] / 60, (q['t0_s'] + q['duration_s']) / 60
        axs[0].axvspan(t0, t1, color=GRID, alpha=0.35, lw=0)
        axs[0].text(0.5 * (t0 + t1), 1.62, q['name'], ha='center', va='bottom', fontsize=7, color=INK2, rotation=90)
    axs[0].set_ylim(0, 1.6)
    axs[1].plot(t, th, color=C[1])
    axs[1].set_ylabel('example coil state θ (°C)\n[assumed τ 200 / 1500 / 6000 s,\n7 °C at flow load]')
    axs[1].set_xlabel('protocol time (min)')
    save(fig, 'protocol_timeline.png')
    # 3. PNS per part (one instance of every segment type), both models
    names = list(p['checks'].keys())
    safe = [p['checks'][k]['pns_safe_example_max'] for k in names]
    ge = [p['checks'][k]['pns_ge_hrmw_max'] for k in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(x - 0.2, safe, 0.38, color=C[0], label='SAFE (pypulseq example hw)')
    ax.bar(x + 0.2, ge, 0.38, color=C[1], label='GE IEC model, hrmw (TOPPE pns.m)')
    ax.axhline(80, color=INK2, lw=0.8, ls='--')
    ax.text(len(names) - 0.5, 81, '80 % (normal mode)', ha='right', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(x, names, rotation=30)
    ax.set_ylabel('max PNS (% of threshold)')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    save(fig, 'protocol_pns.png')
    # 4. one v2-validation frame: history (3 axes) + the measured TR's axis-x components
    try:
        R = np.load(os.path.join(PROTO, 'vframe_R.npy'))
        E = np.load(os.path.join(PROTO, 'vframe_enc.npy')).astype(int)
        vin = np.load(os.path.join(PROTO, 'vframe_in.npy'))
        enc = np.load(os.path.join(PROTO, 'flowtr_enc.npy'))
        ro = np.load(os.path.join(PROTO, 'flowtr_ro.npy'))
        tail = np.load(os.path.join(PROTO, 'flowtr_tail.npy'))
        gap = np.zeros((p['config']['seg_gap_samples'], 3))
        h = np.concatenate([np.concatenate([enc[E[0, j]], gap, ro @ R[0, j].T, gap, tail[E[0, j]], gap]) for j in range(R.shape[1] - 1)])
        v = vin[0, 0]
        nz = np.flatnonzero(v)
        v = v[:nz[-1] + 2]
        full = np.concatenate([h, np.zeros((int(5e-3 / dt), 3)), np.zeros((len(v), 3))])
        full[-len(v):, 0] = v
        tt = np.arange(len(full)) * dt * 1e3
        fig, ax = plt.subplots(figsize=(10, 2.8))
        for a in range(3):
            ax.plot(tt, full[:, a] * 1e3 + 0 * a, color=C[a], lw=0.8, label=f'G{AX[a]}')
        ax.axvspan(tt[len(h)], tt[-1], color=GRID, alpha=0.5, lw=0)
        ax.text(tt[len(h)] + 1, ax.get_ylim()[1] * 0.85, 'thin slice normal to x\n+ measured TR (x only)', fontsize=8, color=INK2)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('mT/m')
        ax.set_title('v2-validation frame: 10 previous TRs of the scan (pge2 segments + 117 µs gaps), then the measured readout')
        ax.legend(loc='lower left', ncol=3)
        save(fig, 'protocol_vframe.png')
    except FileNotFoundError:
        pass


def load(tags):
    recs = []
    for t in tags:
        fn = os.path.join(MC, f'mc_{t}.jsonl')
        for line in open(fn):
            recs.append(json.loads(line))
    return recs


def cdf(ax, v, color, label):
    v = np.sort(np.asarray(v))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post', color=color, label=label)


def mc(tags):
    recs = load(tags)
    tag = '_'.join(tags)
    # R CDFs at the per-encode level and after the per-axis delay refit
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    cases = [('cold0_vs_cold', 'cold, no X stage (vs cold truth)'), ('cold_vs_ss', 'cold + X, used at the flow state'), ('warm_vs_ss', 'warm + X (vs flow state)'),
             ('flow_vs_ss', 'flow-state model: warm + X + pass 2 + thermal'), ('perfect_nocross', 'perfect except cross terms')]
    for j, lev in enumerate(('enc', 'refined')):
        for i, (k, lab) in enumerate(cases):
            cdf(axs[j], [r['R'][k][lev] for r in recs], C[i], lab)
        axs[j].axvline(0.2, color=INK2, lw=0.8, ls='--')
        axs[j].set_xscale('log')
        axs[j].set_xlabel(f'R ({lev})')
    axs[0].set_ylabel('fraction of draws')
    axs[1].legend(loc='lower right', fontsize=7)
    save(fig, f'mc_{tag}_R_cdf.png')
    # resonance f0 errors in 0.4-1.05 kHz and elsewhere: pass 1 (cold), pass 2 (flow state), thermal-model prediction at the steady state
    rows = [x for r in recs for x in r['res']]
    if rows:
        f0 = np.array([x['f0_true_cold'] for x in rows])
        e1 = np.array([x['f0_p1'] - x['f0_true_cold'] for x in rows])
        e2 = np.array([x['f0_p2'] - x['f0_true_p2'] for x in rows])
        e3 = np.array([x['f0_pred_ss'] - x['f0_true_ss'] for x in rows])
        fig, ax = plt.subplots(figsize=(8, 3.4))
        for e, i, lab in ((e1, 0, 'pass 1 (survey, cold)'), (e2, 1, 'pass 2 (targeted, flow load)'), (e3, 2, 'thermal model → flow steady state')):
            ax.scatter(f0, np.abs(e) + 1e-4, s=14, color=C[i], label=lab, edgecolor=SURF, linewidth=0.5)
        ax.axhline(0.2, color=INK2, lw=0.8, ls='--')
        ax.axvspan(400, 1050, color=GRID, alpha=0.5, lw=0)
        ax.text(420, 0.25, '0.2 Hz', fontsize=8, color=INK2)
        ax.set_yscale('log')
        ax.set_xlabel('resonance f0 (Hz)')
        ax.set_ylabel('|f0 error| (Hz)')
        ax.legend(loc='upper right', fontsize=7)
        save(fig, f'mc_{tag}_res_f0.png')
        # thermal tracking of one resonance (the first draw with a target in the readout band)
        for r in recs:
            cand = [x for x in r['res'] if 400 <= x['f0_true_cold'] <= 1050]
            if cand and r['therm']:
                x = cand[0]
                trk = np.array([[row[2], row[4], row[3]] for row in x['track']])
                fig, axs = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True)
                axs[0].errorbar(trk[:, 0] / 60, trk[:, 1], yerr=[row[5] for row in x['track']], fmt='o', ms=3, color=C[0], label='per-set fit')
                axs[0].plot(trk[:, 0] / 60, x['f0_true_cold'] * (1 + x['cf'] * trk[:, 2]), '.', color=C[1], ms=5, label='truth at the set')
                axs[0].axhline(x['f0_true_ss'], color=C[2], lw=0.8, ls='--', label='truth at the flow steady state')
                axs[0].axhline(x['f0_pred_ss'], color=C[3], lw=0.8, label='model prediction')
                axs[0].set_ylabel('f0 (Hz)')
                axs[0].legend(fontsize=7, loc='best')
                axs[1].plot(trk[:, 0] / 60, trk[:, 2], 'o', ms=3, color=C[1])
                axs[1].set_ylabel('truth θ (°C)')
                axs[1].set_xlabel('protocol time (min)')
                axs[0].set_title(f"draw {r['draw']}, axis {AX[x['axis']]}, f0 {x['f0_true_cold']:.1f} Hz, ζ {x['zeta_true']:.4f}")
                save(fig, f'mc_{tag}_thermal_track.png')
                break
    # cross / spatial term errors
    keys = sorted({k for r in recs for k in r['xterms']})
    if keys:
        err = np.array([[r['xterms'][k]['err_rms'] for k in keys] for r in recs])
        tru = np.array([[r['xterms'][k]['true_rms'] for k in keys] for r in recs])
        cross = [i for i, k in enumerate(keys) if k.split('_')[1] in ('x', 'y', 'z')]
        fig, ax = plt.subplots(figsize=(9, 3.2))
        xx = np.arange(len(keys))
        ax.semilogy(xx, np.median(tru, 0), 'o', color=C[0], label='truth (rms over 0.1-5 kHz, median)')
        ax.semilogy(xx, np.median(err, 0), 's', color=C[1], label='estimation error (median)')
        ax.semilogy(xx, np.percentile(err, 90, 0), 'v', color=C[1], alpha=0.5, label='estimation error (p90)')
        ax.set_xticks(xx, [k.replace('_', ' ') for k in keys], rotation=90, fontsize=6)
        ax.set_ylabel('|H| (1, or m⁰ per basis unit)')
        ax.legend(fontsize=7)
        save(fig, f'mc_{tag}_xterms.png')


if __name__ == '__main__':
    if sys.argv[1] == 'protocol':
        protocol()
    else:
        mc(sys.argv[2:])
