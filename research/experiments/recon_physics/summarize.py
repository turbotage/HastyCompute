"""Collect the outputs of the recon_physics round into <OUT>/tables.md (and print it)."""
import json, os
import numpy as np

REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
OUT = os.path.join(REPO, 'data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11')
C = os.path.join(OUT, 'case_v2')


def J(f):
    p = os.path.join(C, f)
    return json.load(open(p)) if os.path.exists(p) else None


lines = ['# recon_physics_2026-11-11 tables (summarize.py)', '']
R = J('R.json') or {}
Rm = J('R_y_exact_modelfit_truth.json') or {}
if R:
    lines += ['## Operator vs exact data (R; |k| bands < 0.1 / 0.1–0.5 / 0.5–0.9 / > 0.9 kmax)', '',
              '| tag | model | fat | rate | conc | L | hist | R | bands |', '|---|---|---|---|---|---|---|---|---|']
    for t, v in R.items():
        i = v.get('info', {})
        b = ' / '.join(f'{v[k]:.3f}' for k in v if k.startswith('R_k'))
        lines.append(f"| {t} | {os.path.basename(str(i.get('model', '')))} | {i.get('fat', '')} | {i.get('rate', '')} | {i.get('conc', '')} | {i.get('L', '')} | {i.get('hist', '')} | {v['R']:.3f} | {b} |")
    lines.append('')
if Rm:
    lines += ['## Operator vs the direct sum of its own model (x[(1-f)+f c] e^{-z(tau-TE)}, true-chain records)', '']
    for t, v in Rm.items():
        lines.append(f"- {t}: R {v['R']:.4f}")
    lines.append('')
for f in sorted(os.listdir(C)):
    if f.startswith('refit_') and f.endswith('.json'):
        j = J(f)
        lines += [f'## {f}: model {os.path.basename(j["model"])}, params {j["params"]}, enc {j.get("enc")}, noise {j["noise"]}, seeds {j.get("seeds")}', '']
        for run in j['runs']:
            its = run['iterations']
            lines.append(f"- seed {run['seed']}: misfit R " + ' → '.join(f"{r['R_misfit']:.4f}" for r in its))
            last = its[-1]
            lines.append('  - θ (ns / %): ' + ', '.join(f'{n} {v:.3f}' for n, v in last['theta_natural']))
            lines.append('  - CRB (ns / %): ' + ', '.join(f'{n} {v:.4f}' for n, v in last['crb_natural']))
        if len(j['runs']) > 1:
            est = np.array([r['iterations'][-1]['theta'] for r in j['runs']])
            crb = np.array(j['runs'][0]['iterations'][-1]['crb'])
            lines.append('- over seeds: mean θ ' + ', '.join(f'{v:.4g}' for v in est.mean(0)) + '; sd ' + ', '.join(f'{v:.3g}' for v in est.std(0, ddof=1)) + '; sd / CRB ' + ', '.join(f'{v:.2f}' for v in est.std(0, ddof=1) / crb))
        lines.append(f"- seconds: {j['seconds']}")
        lines.append('')
for f in sorted(os.listdir(C)):
    if f.startswith('jac_') and f.endswith('.json'):
        j = J(f)
        rows = j['rows']
        by = {}
        for r in rows:
            by.setdefault(r['param'], []).append(r['rel'])
        lines += [f'## {f}', ''] + [f'- {p}: |J e_p − FD| / |FD| median {np.median(v):.2e}, max {np.max(v):.2e}' for p, v in by.items()] + ['']
    if f.startswith('nav_') and f.endswith('.json'):
        j = J(f)
        rows = j['rows']
        est = np.array([r['est_dk_cyc_m'] for r in rows]); tru = np.array([r['true_mean_dk_cyc_m'] for r in rows])
        sd = np.array([r['sd_dk_cyc_m'] for r in rows]); ep = np.array([r['est_dphi'] for r in rows]); tp = np.array([r['true_mean_dphi'] for r in rows])
        lines += [f'## {f}: model {os.path.basename(j["model"])}', '',
                  f'- per-TR CRB of the k offset (cycles/m) median {np.median(sd, 0).round(5).tolist()}; of the phase {np.median([r["sd_dphi"] for r in rows]) * 1e3:.3f} mrad',
                  f'- |true − model| mean navigator offset rms {np.sqrt((tru ** 2).mean(0)).round(4).tolist()} cycles/m; estimate − truth rms {np.sqrt(((est - tru) ** 2).mean(0)).round(4).tolist()}',
                  f'- correlation estimate vs truth per axis {[float(np.corrcoef(est[:, a], tru[:, a])[0, 1]) for a in range(3)]}',
                  f'- phase: true mean rms {np.sqrt((tp ** 2).mean()) * 1e3:.2f} mrad, estimate − truth rms {np.sqrt(((ep - tp) ** 2).mean()) * 1e3:.2f} mrad', '']
    if (f.startswith('cost_') or f.startswith('senscost_')) and f.endswith('.json'):
        lines += [f'## {f}', '', '```', json.dumps(J(f), indent=1), '```', '']
txt = '\n'.join(lines)
open(os.path.join(OUT, 'tables.md'), 'w').write(txt)
print(txt)
