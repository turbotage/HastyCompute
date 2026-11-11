"""Tables from the MC runs (mc_*.jsonl) and sens.json -> <OUT>/tables.md and summary.json"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute'), 'data/hasty/seq/studies/girf_sensitivity_2026-11-11')
LV = ['raw', 'affine', 'enc', 'field', 'refined', 'loc_raw', 'loc_enc', 'loc_refined']
BANDS = [(10, 100), (100, 500), (500, 2000), (2000, 5000), (5000, 10000)]


def load():
    mc = {}
    for fn in sorted(glob.glob(os.path.join(OUT, 'mc_*.jsonl'))):
        recs = [json.loads(l) for l in open(fn)]
        if recs:
            mc[os.path.basename(fn)[3:-6]] = recs
    return mc


def rstats(x):
    x = np.asarray(x)
    return dict(n=len(x), median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                P02=float((x <= 0.2).mean()), P05=float((x <= 0.5).mean()))


def main():
    mc = load()
    summ = {}
    lines = ['# GIRF sensitivity: Monte Carlo tables', '']
    for tag, recs in mc.items():
        o = recs[0]['opt']
        s = summ.setdefault(tag, dict(opt=o, minutes=recs[0]['minutes'], n=len(recs)))
        lines.append(f"## {tag}: truth {o['truth']}, SNR0 {o['snr']:g}, {o['navg']} averages ({recs[0]['minutes']:.0f} min of frames), slice {o['thk'] * 1e3:g} mm, nuisances {o.get('nuis', 'on')}, hadamard {o.get('hadamard', False)}; {len(recs)} draws")
        lines.append('')
        if 'R_np' in recs[0]:
            lines.append('| estimate | level | median R | p10 | p90 | P(R<=0.2) | P(R<=0.5) |')
            lines.append('|---|---|---|---|---|---|---|')
            for est in ('np', 'par', 'np_nocross_ref'):
                for lv in LV:
                    st = rstats([r[f'R_{est}'][lv] for r in recs])
                    s[f'R_{est}_{lv}'] = st
                    lines.append(f"| {est} | {lv} | {st['median']:.3g} | {st['p10']:.3g} | {st['p90']:.3g} | {st['P02']:.2f} | {st['P05']:.2f} |")
            dk = [r['dk_rms_np'] for r in recs]
            lines.append(f"\nresidual k error rms (np): median {np.median(dk):.3g} cycles/m; k0 phase rms median {np.median([r['dphi_rms_np'] for r in recs]):.2e} rad\n")
        # band errors: mean over axes of rms error, median / p90 over draws
        lines.append('| estimate | quantity | ' + ' | '.join(f'{lo}-{hi} Hz' for lo, hi in BANDS) + ' |')
        lines.append('|---|---|' + '---|' * len(BANDS))
        for est in ('np', 'par'):
            for q, lab in (('H', '|dH| self'), ('Hg', '|dH| self, gain removed'), ('B', '|dH0| B0 term [m]')):
                vals = []
                for lo, hi in BANDS:
                    x = [np.mean([r[f'band_{est}'][f'{q}{a}_{lo}_{hi}'] for a in range(3)]) for r in recs]
                    vals.append(f'{np.median(x):.1e} ({np.percentile(x, 90):.1e})')
                    s[f'band_{est}_{q}_{lo}_{hi}'] = [float(np.median(x)), float(np.percentile(x, 90))]
                lines.append(f'| {est} | {lab} | ' + ' | '.join(vals) + ' |')
        for est in ('np', 'par'):
            gd = np.array([[r[f'band_{est}'][f'gd{a}_ns'] for a in range(3)] for r in recs])
            gn = np.array([[r[f'band_{est}'][f'gain{a}'] for a in range(3)] for r in recs])
            s[f'gd_{est}'] = dict(bias=gd.mean(0).tolist(), sd=gd.std(0).tolist(), rms=float(np.sqrt((gd ** 2).mean())))
            s[f'gain_{est}'] = dict(rms=float(np.sqrt((gn ** 2).mean())))
            lines.append(f"\n{est}: group-delay error (1-3 kHz) bias {np.round(gd.mean(0), 2).tolist()} ns, sd {np.round(gd.std(0), 2).tolist()} ns; DC gain error rms {np.sqrt((gn ** 2).mean()):.1e}")
        if recs[0].get('param'):
            keys = sorted(recs[0]['param'])
            lines.append('\n| parameter | bias | sd | rms |\n|---|---|---|---|')
            for k in keys:
                x = np.array([r['param'].get(k, np.nan) for r in recs])
                x = x[np.isfinite(x)]
                s['param_' + k] = dict(bias=float(x.mean()), sd=float(x.std()), rms=float(np.sqrt((x ** 2).mean())))
                lines.append(f'| {k} | {x.mean():.3g} | {x.std():.3g} | {np.sqrt((x ** 2).mean()):.3g} |')
        lines.append('')
    sens = os.path.join(OUT, 'sens.json')
    if os.path.exists(sens):
        S = json.load(open(sens))
        lines.append('# Deterministic sensitivity (sens.json)\n')
        for sec, d in S.items():
            if sec == 'fisher':
                lines.append(f'## fisher\n\n```\n{json.dumps(d, indent=1)}\n```\n')
                continue
            lines.append(f'## {sec}\n\n| case | raw | affine | enc | field | refined | local raw | local refined | dk rms [c/m] | dk at echo [c/m] | dphi rms [rad] |\n|---|---|---|---|---|---|---|---|---|---|---|')
            for k, r in d.items():
                lines.append(f"| {k} | {r['raw']:.3g} | {r['affine']:.3g} | {r['enc']:.3g} | {r['field']:.3g} | {r['refined']:.3g} | {r['loc_raw']:.3g} | {r['loc_refined']:.3g} | {r.get('dk_rms_cpm', np.nan):.3g} | {r.get('dk_echo_cpm', np.nan):.3g} | {r.get('dphi_rms', np.nan):.2e} |")
            lines.append('')
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(lines))
    json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
