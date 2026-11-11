"""Figures of the readout-physics budget (<OUT>/figs/): per-sample error energy vs time after the echo, the R budget,
local flow numbers and the warp handling."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import pb

OUT = pb.OUT
FIG = os.path.join(OUT, 'figs')
os.makedirs(FIG, exist_ok=True)
D = ['current', 'v2']
SIG2C = 32 * 39.97583861630753 ** 2


def ld(name):
    fn = os.path.join(OUT, name)
    return json.load(open(fn)) if os.path.exists(fn) else None


# 1. per-sample error energy over the readout
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax, d in zip(axs, D):
    G = ld(f'global_{d}.json')
    if G is None:
        continue
    des = pb.Design(d)
    n = np.array(G['energy']['n']); t = des.t[n] * 1e3
    ax.semilogy(t, np.array(G['energy']['E']) / SIG2C, 'k-', lw=2, label='data energy (truth model)')
    ax.axhline(1.0, color='gray', ls='--', lw=1, label='noise (per sample, coil-summed)')
    Lw = G['T1_rule']['svdn_L_rule']
    curves = [('T1', 'omission', 'B0 omitted', 'C0'), ('T1', f'svdn_L{Lw}', f'B0 svdn L={Lw}', 'C0'), ('T2', 'omission', 'concomitant (readout) omitted', 'C1'),
              ('T2', 'taylor2_residual', 'concomitant residual (Taylor 2)', 'C1'), ('T3', 'omission', 'fat as water', 'C2'), ('T3', 'one_map_mean_vs_frame', 'one fat map vs frame', 'C2'),
              ('T4', 'omission', 'R2* omitted', 'C3'), ('T4', 'table_residual', 'R2* table', 'C3'), ('T4', 'eff_complex_residual', 'R2* effective complex rate', 'C3'),
              ('T9', 'omission', 'rect voxel basis omitted', 'C5'), ('T9', 'absorbed_residual', 'rect basis, image redefined', 'C5')]
    for sec, key, lab, col in curves:
        e = (G.get(sec) or {}).get(key)
        if e is None or 'E_n' not in e:
            continue
        ls = '-' if ('omi' in key or 'omission' in key or key == 'omission') else ':'
        ax.semilogy(t, np.maximum(np.array(e['E_n']), 1e-12) / SIG2C, ls, color=col, lw=1.2, label=f'{lab} (R {e["R"]:.3g})')
    F = np.load(os.path.join(OUT, f'flow_omission_{d}_cycle.npz')) if os.path.exists(os.path.join(OUT, f'flow_omission_{d}_cycle.npz')) else None
    if F is not None:
        tf = des.t[F['n']] * 1e3
        J = ld(f'flow_omission_{d}_cycle.json')
        for k, lab, col in [('m1__all', 'blood readout M1 omitted (cycle)', 'C4'), ('perp__all', 'blood M1 perpendicular', 'C6'), ('acc__all', 'blood acceleration', 'C7'), ('pathb0__all', 'blood path B0', 'C8')]:
            kk, rr = k.split('__')
            ax.semilogy(tf, np.maximum(F[k], 1e-12) / SIG2C, '-', color=col, lw=1.0, label=f'{lab} (R {J["terms"][kk][rr]["R"]:.3g})')
    ax.set_xlabel('time after the echo (ms)'); ax.set_title(f'{d}: error energy per sample / noise (mean over readouts)')
    ax.set_ylim(1e-6, 1e6); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper right', ncol=1)
axs[0].set_ylabel('E(n) / (C sigma^2)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'error_energy_vs_time.png'), dpi=110); plt.close(fig)

# 2. global R budget
items = [('T1 B0 omitted', 'global', 'T1', 'omission'), ('T1 B0 svdn rule L', 'global', 'T1', 'RULE'), ('T2 conc. omitted', 'global', 'T2', 'omission'),
         ('T2 conc. deviation omitted', 'global', 'T2', 'deviation_omitted'), ('T2 conc. residual', 'global', 'T2', 'taylor2_residual'),
         ('T3 fat omitted', 'global', 'T3', 'omission'), ('T3 one fat map', 'global', 'T3', 'one_map_mean_vs_frame'),
         ('T4 R2* omitted', 'global', 'T4', 'omission'), ('T4 R2* table', 'global', 'T4', 'table_residual'), ('T4 R2* eff. complex', 'global', 'T4', 'eff_complex_residual'),
         ('T5+T7 blood M1 omitted (cycle)', 'flow', 'm1', 'cycle'), ('T5+T7 (systole)', 'flow', 'm1', 'systole'), ('T7 M1 perp (systole)', 'flow', 'perp', 'systole'),
         ('T6 acceleration (systole)', 'flow', 'acc', 'systole'), ('T12 path B0 (systole)', 'flow', 'pathb0', 'systole'),
         ('T9 rect basis omitted', 'global', 'T9', 'omission'), ('T9 image redefined', 'global', 'T9', 'absorbed_residual'), ('T10 GIRF 5 %', 'global', 'T10', 'girf_5pct_residual')]
fig, ax = plt.subplots(figsize=(11, 7))
y = np.arange(len(items))
for j, d in enumerate(D):
    G = ld(f'global_{d}.json')
    vals = []
    for lab, kind, a_, b_ in items:
        v = None
        if kind == 'global' and G is not None:
            if b_ == 'RULE':
                b_ = f'svdn_L{G["T1_rule"]["svdn_L_rule"]}'
            v = ((G.get(a_) or {}).get(b_) or {}).get('R')
        elif kind == 'flow':
            J = ld(f'flow_omission_{d}_{b_}.json')
            v = None if J is None else J['terms'][a_]['all']['R']
        vals.append(np.nan if v is None else v)
    ax.barh(y + 0.4 * j - 0.2, vals, height=0.4, label=d)
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=8); ax.set_xscale('log'); ax.axvline(0.2, color='k', ls='--', lw=1)
ax.text(0.21, len(items) - 0.5, 'R = 0.2 (segment rule)', fontsize=8)
ax.invert_yaxis(); ax.set_xlabel('global R (data error rms / noise rms)'); ax.legend(); ax.grid(alpha=0.3, axis='x')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'budget_global_R.png'), dpi=110); plt.close(fig)

# 3. local flow numbers
fig, axs = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
regs = ['all', 'chambers', 'vessels', 'coronary']
kinds = [('m1', 'readout M1 (T5+T7)'), ('perp', 'M1 perp (T7)'), ('acc', 'accel. (T6)'), ('pathb0', 'path B0 (T12)')]
for ax, ph in zip(axs, ['cycle', 'systole', 'diastole']):
    xx = np.arange(len(regs))
    w = 0.1
    for j, d in enumerate(D):
        J = ld(f'flow_omission_{d}_{ph}.json')
        if J is None:
            continue
        for i, (k, lab) in enumerate(kinds):
            vals = [J['terms'][k][r]['R_loc'] for r in regs]
            ax.bar(xx + (j * len(kinds) + i - 3.5) * w, vals, width=w, color=f'C{i}', alpha=1.0 if j == 1 else 0.45, label=f'{lab} {d}' if ph == 'cycle' else None)
    ax.set_xticks(xx); ax.set_xticklabels(regs); ax.set_yscale('log'); ax.axhline(0.2, color='k', ls='--', lw=1); ax.set_title(f'{ph}: R_loc (pale: current, solid: v2)')
    ax.grid(alpha=0.3, axis='y')
axs[0].legend(fontsize=7, ncol=2); axs[0].set_ylabel('R_loc')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'flow_local_R.png'), dpi=110); plt.close(fig)

# 4. warp handling
W = ld('warp_eval_matched.json')
if W:
    keys = [('omit', 'not modelled'), ('exact_trilinear', 'warp trilinear'), ('exact_cubic', 'warp cubic'), ('exact_lanczos3', 'warp Lanczos-3'), ('exact_lanczos4', 'warp Lanczos-4'),
            ('exact_ideal', 'ideal shift'), ('exact_ideal_perp', 'ideal + M1 perp'), ('exact_ideal_bias0.8', 'ideal, v_hat 0.8 v'), ('exact_ideal_noise0.05', 'ideal, v_hat + 5 cm/s noise'),
            ('exact_ideal_smooth4mm', 'ideal, v_hat smoothed')]
    fig, axs = plt.subplots(1, len(W), figsize=(5 * len(W), 6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (k_, r) in zip(axs, W.items()):
        segk = [(k, k.replace('_', ' ')) for k in r if k.startswith('seg') and not k.endswith('static') and '_cor' not in k and '_ves' not in k and 'acc' not in k]
        ks = [k for k in keys + segk if k[0] in r and 'R_over_floor' in r[k[0]]]
        ax.barh(np.arange(len(ks)), [r[k]['R_over_floor'] for k, _ in ks], color='C0')
        ax.set_yticks(np.arange(len(ks))); ax.set_yticklabels([l for _, l in ks], fontsize=8); ax.invert_yaxis(); ax.set_xscale('log')
        ax.set_title(f'{k_} (blood, global R over the static floor)', fontsize=9); ax.grid(alpha=0.3, axis='x'); ax.axvline(0.2 / r['R_loc_factor_all'], color='k', ls='--', lw=1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'warp_handling.png'), dpi=110); plt.close(fig)
print('figures written to', FIG)
