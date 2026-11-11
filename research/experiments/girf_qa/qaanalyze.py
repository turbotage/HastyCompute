"""Tables from the Monte Carlo (mc_<tag>.jsonl) → <OUT>/tables_<tag>.md and summary_<tag>.json

    qaanalyze.py main
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
MC = os.path.join(REPO, 'data/hasty/seq/girf_qa/mc')
BANDS = ['10_100', '100_500', '500_2000', '2000_5000', '5000_10000']


def q(v, p):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return float(np.percentile(v, p)) if len(v) else float('nan')


def main(tags):
    recs = []
    for t in tags:
        for line in open(os.path.join(MC, f'mc_{t}.jsonl')):
            recs.append(json.loads(line))
    tag = '_'.join(tags)
    n = len(recs)
    out, md = {}, [f'# GIRF QA Monte Carlo `{tag}` ({n} draws)\n']
    # R
    md.append('## R for the v2 readout (truth at the flow scan steady state unless stated)\n')
    md.append('| estimate | level | median [p10, p90] | P(≤ 0.2) | P(≤ 0.5) |\n|---|---|---|---|---|')
    out['R'] = {}
    for k in recs[0]['R']:
        for lev in ('enc', 'refined', 'loc_refined'):
            v = [r['R'][k][lev] for r in recs]
            out['R'][f'{k}/{lev}'] = dict(median=q(v, 50), p10=q(v, 10), p90=q(v, 90), p02=float(np.mean(np.array(v) <= 0.2)), p05=float(np.mean(np.array(v) <= 0.5)))
            o = out['R'][f'{k}/{lev}']
            md.append(f"| {k} | {lev} | {o['median']:.3f} [{o['p10']:.3f}, {o['p90']:.3f}] | {o['p02']:.2f} | {o['p05']:.2f} |")
    # self-term bands and group delay
    md.append('\n## Self terms: |H_est − H_true| per band (median over draws and axes; Hg = after removing the DC gain at 10–15 Hz), group delay error\n')
    md.append('| estimate | ' + ' | '.join(f'{b} Hz H / Hg' for b in BANDS) + ' | |Δ group delay| median / p90 [ns] |\n|---|' + '---|' * (len(BANDS) + 1))
    out['bands'] = {}
    for key in ('band_cold0', 'band_cold', 'band_warm', 'band_flow_ss'):
        row = []
        for b in BANDS:
            h = [r[key][f'H{a}_{b}'] for r in recs for a in range(3)]
            hg = [r[key][f'Hg{a}_{b}'] for r in recs for a in range(3)]
            row.append(f'{q(h, 50):.1e} / {q(hg, 50):.1e}')
            out['bands'][f'{key}/{b}'] = dict(H=q(h, 50), Hg=q(hg, 50), H90=q(h, 90))
        gd = [abs(r[key][f'gd{a}_ns']) for r in recs for a in range(3)]
        out['bands'][f'{key}/gd'] = dict(median=q(gd, 50), p90=q(gd, 90))
        md.append(f'| {key[5:]} | ' + ' | '.join(row) + f' | {q(gd, 50):.1f} / {q(gd, 90):.1f} |')
    # B0 terms
    md.append('\n## B0 terms |ΔH0| (m) per band, median\n')
    md.append('| estimate | ' + ' | '.join(BANDS) + ' |\n|---|' + '---|' * len(BANDS))
    for key in ('band_cold0', 'band_cold', 'band_warm'):
        md.append(f'| {key[5:]} | ' + ' | '.join(f"{q([r[key][f'B{a}_{b}'] for r in recs for a in range(3)], 50):.1e}" for b in BANDS) + ' |')
    # cross / spatial terms
    md.append('\n## Cross and 2nd / 3rd-order terms (rms over 0.1–5 kHz)\n')
    md.append('| term class | truth rms (median) | error rms median [p90] | error / truth median |\n|---|---|---|---|')
    out['xterms'] = {}
    for cls, sel in (('cross (x, y, z outputs)', lambda b: b in ('x', 'y', 'z')), ('2nd order', lambda b: b in ('xy', 'xz', 'yz', 'x2-y2', '2z2-x2-y2')), ('3rd order', lambda b: b not in ('x', 'y', 'z', 'xy', 'xz', 'yz', 'x2-y2', '2z2-x2-y2'))):
        e = [v['err_rms'] for r in recs for k, v in r['xterms'].items() if sel(k.split('_', 1)[1])]
        t = [v['true_rms'] for r in recs for k, v in r['xterms'].items() if sel(k.split('_', 1)[1])]
        rel = [a / b for a, b in zip(e, t) if b > 0]
        out['xterms'][cls] = dict(true=q(t, 50), err=q(e, 50), err90=q(e, 90), rel=q(rel, 50))
        md.append(f'| {cls} | {q(t, 50):.1e} | {q(e, 50):.1e} [{q(e, 90):.1e}] | {q(rel, 50):.2f} |')
    # resonances
    rows = [x for r in recs for x in r['res']]
    md.append(f'\n## Resonances ({len(rows)} targeted in 0.3–2 kHz over {n} draws)\n')
    if rows:
        f0 = np.array([x['f0_true_cold'] for x in rows])
        band = (f0 >= 400) & (f0 <= 1050)
        e1 = np.abs([x['f0_p1'] - x['f0_true_cold'] for x in rows])
        e2 = np.abs([x['f0_p2'] - x['f0_true_p2'] for x in rows])
        e3 = np.abs([x['f0_pred_ss'] - x['f0_true_ss'] for x in rows])
        dT = np.abs([x['dfdT'] for x in rows])
        ez = np.abs([x['zeta_p2'] / x['zeta_true'] - 1 for x in rows])
        eg = np.abs([x['gain_p2'] / x['gain_true_p2'] - 1 for x in rows])
        md.append('| quantity | all: median [p90] | 0.4–1.05 kHz: median [max] |\n|---|---|---|')
        for name, e in (('f0 pass 1 (cold, survey) [Hz]', e1), ('f0 pass 2 (flow load, targeted) [Hz]', e2), ('f0 thermal model → flow steady state [Hz]', e3),
                        ('… in °C-equivalent (|Δf0| / |df0/dT|)', e3 / dT), ('ζ pass 2 (relative)', ez), ('gain pass 2 (relative)', eg)):
            md.append(f"| {name} | {q(e, 50):.3g} [{q(e, 90):.3g}] | {q(e[band], 50) if band.any() else float('nan'):.3g} [{(e[band].max() if band.any() else float('nan')):.3g}] |")
            out[f'res/{name}'] = dict(median=q(e, 50), p90=q(e, 90), band_median=q(e[band], 50) if band.any() else None, band_max=float(e[band].max()) if band.any() else None)
        out['res/n'] = len(rows)
        out['res/n_band'] = int(band.sum())
        # how many truth resonances in 0.4-1.05 kHz were found
        tr = [(r['draw'], x) for r in recs for x in r['truth_res'] if 400 <= x[1] <= 1050 and x[3] >= 0.01]
        found = sum(1 for d, x in tr if any(abs(t[1] - x[1]) < max(10 * x[2] * x[1], 15) and int(t[0]) == int(x[0]) for r in recs if r['draw'] == d for t in r['targets']))
        md.append(f'\nTruth resonances with peak ≥ 1 % in 0.4–1.05 kHz: {len(tr)}, targeted by pass 1: {found}.')
        out['res/truth_band'] = [len(tr), found]
    # thermal model
    th = [r['therm'] for r in recs if r['therm']]
    if th:
        md.append(f"\nThermal model fits: τ₁ median {q([t['tau1'] for t in th], 50):.0f} s, τ₂ {q([t['tau2'] for t in th], 50):.0f} s (truth τ 2–6 / 15–40 / 70–130 min).")
    # linearity, receiver
    lb = [abs(r['lin'][str(a)]['beta'] - r['lin'][str(a)]['beta_true']) for r in recs for a in range(3)]
    md.append(f'\n## Linearity and receiver timing\n\nOdd nonlinearity β: |error| median {q(lb, 50):.1e}, p90 {q(lb, 90):.1e} (truth β ≤ 3·10⁻⁴).')
    out['lin'] = dict(median=q(lb, 50), p90=q(lb, 90))
    md.append('\n| coil | dwell | |Δδ error| median / max [ns] |\n|---|---|---|')
    for coil in ('body', 'array'):
        for dw in ('2', '8', '16'):
            e = [abs(r['rx'][coil][dw]['est_ns'] - r['rx'][coil][dw]['true_ns']) for r in recs]
            md.append(f'| {coil} | {dw} µs | {q(e, 50):.2f} / {max(e):.2f} |')
            out[f'rx/{coil}/{dw}'] = dict(median=q(e, 50), max=float(max(e)))
        e = [abs(r['rx'][coil]['abs4_ns']['est'] - r['rx'][coil]['abs4_ns']['true']) for r in recs]
        md.append(f'| {coil} | 4 µs (vs the GIRF estimate) | {q(e, 50):.2f} / {max(e):.2f} |')
    e = [abs(r['rx_img']['d_hat_ns'] - (r['rx_img']['d_img_ns'] - r['rx_img']['d4_ns'])) for r in recs]
    md.append(f'| imaging (2 µs, array) vs GIRF protocol (4 µs, body) | | {q(e, 50):.2f} / {max(e):.2f} |')
    out['rx/img'] = dict(median=q(e, 50), max=float(max(e)))
    # V
    md.append('\n## v2 validation (V)\n\n| quantity | median [p10, p90] |\n|---|---|')
    for k in ('R_meas', 'R_true', 'R_meas_ref', 'R_true_ref', 'R_true_all', 'masked_frac', 'sd_median', 'sd_p90', 'dk_rms_meas', 'dk_rms_true', 'noise_rms'):
        v = [r['V'][k] for r in recs if k in r['V']]
        if v:
            md.append(f'| {k} | {q(v, 50):.3g} [{q(v, 10):.3g}, {q(v, 90):.3g}] |')
            out[f'V/{k}'] = dict(median=q(v, 50), p10=q(v, 10), p90=q(v, 90))
    rm = np.array([r['V']['R_meas'] for r in recs])
    rt = np.array([r['V']['R_true'] for r in recs])
    md.append(f'\nSpearman(R_V,meas, R_V,true) = {spearman(rm, rt):.2f}; Spearman(R_V,meas, R flow_vs_ss enc) = {spearman(rm, [r["R"]["flow_vs_ss"]["enc"] for r in recs]):.2f}.')
    json.dump(out, open(os.path.join(MC, f'summary_{tag}.json'), 'w'), indent=1)
    open(os.path.join(MC, f'tables_{tag}.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float('nan')


if __name__ == '__main__':
    main(sys.argv[1:])
