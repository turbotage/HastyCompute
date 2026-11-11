"""Figures of research/recon_physics_wiring.md -> <OUT>/figs/ (ablation, |k| bands, refit, recovery, navigator)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
OUT = os.path.join(REPO, 'data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11')
CASE = os.path.join(OUT, 'case_v2')
FIG = os.path.join(OUT, 'figs')
os.makedirs(FIG, exist_ok=True)
# reference categorical palette (dataviz skill, light surface)
S1, S2, S3, S4, S5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e3e2de', '#fcfcfb'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'axes.spines.top': False, 'axes.spines.right': False})


def load(f):
    p = os.path.join(CASE, f)
    return json.load(open(p)) if os.path.exists(p) else None


LABELS = [  # (tag with estimated GIRF, tag with the true chain, label)
    ('all', 'truthchain', 'everything on (L 15, union fat, fitted rate, concomitant, GIRF + 10 TRs)'),
    ('girf_nominal', None, 'GIRF off (nominal trajectory)'),
    ('girf_hist1', 'girf_truth_h1', 'GIRF with 1 previous TR'),
    ('girf_T80', None, 'GIRF estimate of the 80 ms protocol'),
    ('fat_mean', 't_fat_mean', 'fat: one time-mean map'),
    ('fat_none', 't_fat_none', 'fat: none'),
    ('rate_none', 't_rate_none', 'rate: none (map field)'),
    ('rate_tab', 't_rate_tab', 'rate: table R2* only'),
    ('rate_wt', 't_rate_wt', 'rate: table R2* + signal-weighted freq.'),
    ('conc_off', 't_conc_off', 'concomitant: off'),
    ('conc_mean', 't_conc_mean', 'concomitant: encode mean only'),
    ('seg_L13', 't_L13', 'segments L 13'),
    ('seg_L17', 't_L17', 'segments L 17'),
]


def ablation():
    R = load('R.json')
    if not R:
        return
    rows = [(lab, R.get(a, {}).get('R'), R.get(b, {}).get('R') if b else None) for a, b, lab in LABELS]
    rows = [r for r in rows if r[1] is not None or r[2] is not None]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    h = 0.36
    for k, (col, name) in enumerate([(S1, 'model trajectory: estimated GIRF (200 ms protocol)'), (S2, 'model trajectory: true chain')]):
        v = np.array([r[1 + k] if r[1 + k] is not None else np.nan for r in rows])
        ax.barh(y + (h / 2 if k == 0 else -h / 2), v, height=h - 0.04, color=col, label=name)
        for yy, vv in zip(y, v):
            if np.isfinite(vv):
                ax.text(vv * 1.04, yy + (h / 2 if k == 0 else -h / 2), f'{vv:.2f}', va='center', fontsize=7, color=INK2)
    ax.axvline(0.2, color=INK2, lw=0.8, ls='--')
    ax.text(0.2, y[0] + 0.7, ' tolerance 0.2', fontsize=7, color=INK2)
    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel('R = data error rms / noise rms (exact direct sum, 48 readouts, v2)')
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    ax.set_title('Operator validation: everything on and each term changed in turn', fontsize=9, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ablation.png'), dpi=150)
    plt.close(fig)


def refit():
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    any_ = False
    for f, col, lab in [('refit_exact.json', S1, 'delays + B0 amp. + resonances'), ('refit_exact_gain.json', S3, '+ per-axis gains'),
                        ('refit_exact_enc.json', S2, '+ per-encode k offset / phase'), ('refit_exact_enconly.json', S4, 'per-encode k offset / phase only')]:
        j = load(f)
        if not j:
            continue
        it = j['runs'][0]['iterations']
        ax.plot([r['iter'] for r in it], [r['R_misfit'] for r in it], '-o', color=col, lw=2, ms=4, label=lab)
        any_ = True
    if not any_:
        plt.close(fig); return
    ax.set_yscale('log')
    ax.set_xlabel('Gauss–Newton iteration (image fixed)')
    ax.set_ylabel('misfit R vs the exact data (unweighted)')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('GIRF refit on the exact data (model: estimated GIRF)', fontsize=9, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'refit_exact.png'), dpi=150); plt.close(fig)


def recovery():
    j = load('refit_recovery.json')
    if not j:
        return
    th_true = np.array([1e-7, -1e-7, 1e-7, 0.2, 0.005, 0, -0.005, 0, 0.005, 0])
    names = j['params']
    est = np.array([r['iterations'][-1]['theta'] for r in j['runs']])
    crb = np.array(j['runs'][0]['iterations'][-1]['crb'])
    z = (est - th_true[None, :len(names)]) / crb[None, :]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(names))
    ax.axhspan(-1, 1, color=GRID, zorder=0)
    for k in range(est.shape[0]):
        ax.plot(x + (k - est.shape[0] / 2) * 0.05, z[k], 'o', ms=4, color=S1, label='noise realisation' if k == 0 else None)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('(θ̂ − θ*) / CRB')
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.set_title('Parameter recovery, mismatched chain, σ noise (band: ±1 CRB)', fontsize=9, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'refit_recovery.png'), dpi=150); plt.close(fig)


def nav():
    j = load('nav_est.json')
    if not j:
        return
    rows = j['rows']
    est = np.array([r['est_dk_cyc_m'] for r in rows]); tru = np.array([r['true_mean_dk_cyc_m'] for r in rows])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for a, col, n in [(0, S1, 'x'), (1, S2, 'y'), (2, S3, 'z')]:
        ax.plot(tru[:, a], est[:, a], 'o', ms=4, color=col, label=f'k_{n}')
    lim = np.nanmax(np.abs(np.r_[est.ravel(), tru.ravel()])) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=0.8)
    ax.set_xlabel('true − model k, mean over the navigator (cycles/m)')
    ax.set_ylabel('navigator estimate (cycles/m)')
    ax.legend(fontsize=7, frameon=False)
    ax.set_title('Per-TR navigator monitor (48 TRs)', fontsize=9, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'nav_monitor.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    ablation(); refit(); recovery(); nav()
    print('figures in', FIG, os.listdir(FIG))
