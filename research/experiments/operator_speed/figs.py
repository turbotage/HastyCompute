"""Figures of the operator-speed study (research/operator_speed.md) from the JSON / log outputs in
data/hasty/studies/operator_speed_2026-11-12/:

  fig_breakdown.png   one 2 mm gradient iteration (L 13, 280 sets) per component, today vs with the adopted levers
  fig_width.png       per-set operator time (forward + adjoint, recon configuration) and NUFFT error vs kernel width W
  fig_llr.png         LLR prior: CPU (recon log, 10-core quota) vs GPU (f64 eigensolver), per call

    rustlib/.venv/bin/python research/experiments/operator_speed/figs.py
"""
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.environ.get('OPSPEED_OUT', '/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/studies/operator_speed_2026-11-12')
FIG = os.path.join(OUT, 'figs')
os.makedirs(FIG, exist_ok=True)


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else []


def variant(r):
    return dict(r.get('env', [])).get('HASTY_BENCH_VARIANT', f"s{r['sigma']}w{r['width']}")


def fig_breakdown():
    p = os.path.join(OUT, 'breakdown.json')
    if not os.path.exists(p):
        print('breakdown.json missing'); return
    b = json.load(open(p))
    cols = b['columns']
    comps = b['components']
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    left = np.zeros(len(cols))
    cmap = plt.get_cmap('tab10')
    for i, (name, vals) in enumerate(comps.items()):
        v = np.array(vals, dtype=float)
        ax.barh(cols, v, left=left, color=cmap(i % 10), label=name)
        for j, (l, w) in enumerate(zip(left, v)):
            if w > 12:
                ax.text(l + w / 2, j, f'{w:.0f}', ha='center', va='center', fontsize=8, color='white')
        left += v
    for j, t in enumerate(left):
        ax.text(t + 4, j, f'{t:.0f} s', va='center', fontsize=9)
    ax.set_xlabel('seconds per iteration (2 mm ph0001, L 13, 280 sets, 31 coils, RTX 4090 at 350 W)')
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.15)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.set_title('one P²nP iteration per component')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_breakdown.png'), dpi=150)
    plt.close(fig)


def fig_width():
    bench = load('bench_op.json')
    acc = load('accuracy_small.json')
    rows = {}
    for r in bench:
        v = variant(r)
        if r.get('sigma') == 1.25 and v.startswith('w') and '_' not in v.replace('w10_plain', 'w10plain'):
            rows[r['width']] = r
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    if rows:
        ws = sorted(rows)
        t = [rows[w]['mean_total_ms'] for w in ws]
        tp = [rows[w]['pipeline']['ms_per_set'] if rows[w].get('pipeline') else np.nan for w in ws]
        ax[0].plot(ws, t, 'o-', label='forward + adjoint (operator)')
        ax[0].plot(ws, tp, 's--', label='per-set pipeline (+ k-space staging)')
        ax[0].set_xlabel('ES kernel width W (σ 1.25)')
        ax[0].set_ylabel('ms per set')
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
    for a in acc:
        lab = 'white image' if a.get('white_x') else 'phantom image'
        pts = [(r['width'], r['eps_fwd'], r['eps_adj']) for r in a['rows'] if r['sigma'] == 1.25]
        if not pts or a.get('dc_scale', 1.0) != 1.0 or a['reference'] != [2.0, 14]:
            continue
        pts.sort()
        w, ef, ea = zip(*pts)
        ax[1].semilogy(w, ef, 'o-', label=f'forward, {lab}')
        if not a.get('white_x'):
            ax[1].semilogy(w, ea, 'x:', label='adjoint, white data')
    ax[1].set_xlabel('ES kernel width W (σ 1.25)')
    ax[1].set_ylabel('relative ℓ2 error vs σ 2, W 14')
    ax[1].axhline(0.0172 / 47, color='k', lw=0.8, ls='--')
    ax[1].text(5.1, 0.0172 / 47 * 1.15, 'R guard (+10 %) at SNR 47', fontsize=7)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_width.png'), dpi=150)
    plt.close(fig)


def fig_llr():
    p = os.path.join(OUT, 'llr_full.json')
    if not os.path.exists(p):
        print('llr_full.json missing'); return
    runs = json.load(open(p))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    labels, vals = ['CPU (recon log,\n10-core quota)', 'CPU (fix_2mm log,\nno quota)'], [165.0, 56.0]
    for r in runs:
        labels.append(f"GPU f64\nbatch {r['batch']}")
        vals.append(min(g['wall_s'] for g in r['gpu']))
    ax.bar(labels, vals, color=['#888', '#bbb'] + ['#1f77b4'] * (len(vals) - 2))
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f'{v:.1f} s', ha='center', fontsize=8)
    ax.set_ylabel('seconds per LLR call (280 images, 2 mm)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_llr.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    fig_breakdown()
    fig_width()
    fig_llr()
    print('figures in', FIG)
