"""Collect the budget JSONs of <OUT> into markdown tables (printed; also written to <OUT>/tables.md).

R = global noise-referenced data error; R_loc = R of the errors originating in a region times sqrt(N_obj / N_region)
(the global R the whole object would have if every region of that size behaved like this one).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import pb

OUT = pb.OUT
D = ['current', 'v2']


def ld(name):
    fn = os.path.join(OUT, name)
    return json.load(open(fn)) if os.path.exists(fn) else None


def f(x, nd=3):
    if x is None:
        return '—'
    if isinstance(x, str):
        return x
    if abs(x) >= 100:
        return f'{x:.0f}'
    if abs(x) >= 10:
        return f'{x:.1f}'
    return f'{x:.{nd}g}'


def g(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


lines = []
def out(s=''):
    print(s); lines.append(s)


# ── global terms ──
out('## Global terms (static tissue + blood frame 0, truth image), R global / R_loc blood / vessels / coronary / myocardium')
out()
out('| term | quantity | ' + ' | '.join(f'{d}: global | {d}: blood | {d}: vessels | {d}: coronary | {d}: myo' for d in D) + ' |')
out('|---|---|' + '---|' * (5 * len(D)))
rows = [('T1 B0', 'omission', ('T1', 'omission')), ('T1 B0', 'svdn at the rule L', ('T1', 'svdn_rule')),
        ('T2 concomitant', 'omission (readout part)', ('T2', 'omission')), ('T2 concomitant', 'deviation from encode mean omitted', ('T2', 'deviation_omitted')),
        ('T2 concomitant', 'residual (encode mean + ΔC Taylor 2)', ('T2', 'taylor2_residual')),
        ('T3 fat', 'omission (fat as water)', ('T3', 'omission')), ('T3 fat', 'one time-mean map vs frame map', ('T3', 'one_map_mean_vs_frame')),
        ('T3 fat', 'frame-0 map vs frame map', ('T3', 'one_map_frame0_vs_frame')), ('T3 fat', 'joint svdn (one map) at the rule L', ('T3', 'joint_rule')),
        ('T4 R2*', 'omission (amplitudes frozen at TE)', ('T4', 'omission')), ('T4 R2*', 'table R2* map', ('T4', 'table_residual')),
        ('T4 R2*', 'effective complex rate per voxel', ('T4', 'eff_complex_residual')), ('T4 R2*', 'effective real R2* per voxel', ('T4', 'eff_real_residual')),
        ('T9 voxel basis', 'omission (rect vs Dirac)', ('T9', 'omission')), ('T9 voxel basis', 'after the image redefinition', ('T9', 'absorbed_residual')),
        ('T10 GIRF (illustrative chain)', 'omission', ('T10', 'girf_omission')), ('T10 GIRF', '5 % of the error left', ('T10', 'girf_5pct_residual'))]
for term, q, (sec, key) in rows:
    cells = []
    for d in D:
        G = ld(f'global_{d}.json')
        for reg in ['', '_blood', '_vessels', '_coronary', '_myocardium']:
            J = G if reg == '' else ld(f'global_{d}{reg}.json')
            kk = key
            if key == 'svdn_rule':
                L = g(G, 'T1_rule', 'svdn_L_rule'); kk = f'svdn_L{L}'
            if key == 'joint_rule':
                L = g(G, 'T3_rule', 'svdn_L_rule'); kk = f'joint_svdn_L{L}'
            v = g(J, sec, kk, 'R' if reg == '' else 'R_loc')
            cells.append(f(v))
    out(f'| {term} | {q} | ' + ' | '.join(cells) + ' |')
out()
for d in D:
    G = ld(f'global_{d}.json')
    if G is None:
        continue
    out(f'{d}: rule L (svdn, R 0.2 x 1.5 margin): water {g(G, "T1_rule", "svdn_L_rule")}, water + fat (one map) {g(G, "T3_rule", "svdn_L_rule")}; '
        f'T4seg: no decay {g(G, "T4seg", "nodecay_svdn_L_rule")}, table R2* {g(G, "T4seg", "table_svdn_L_rule")}, effective complex rate {g(G, "T4seg", "effective_svdn_L_rule")}, '
        f'table with decay-weighted voxel weights {g(G, "T4seg", "table_svdn_L_rule_decayweighted_voxels")}; energy {g(G, "energy_summary")}')
    t1 = g(G, 'T1') or {}
    out(f'  T1 R(L): ' + ', '.join(f'{k} {v["R"]:.3g}' for k, v in t1.items() if isinstance(v, dict) and 'R' in v))
    t4 = g(G, 'T4seg') or {}
    out(f'  T4seg R(L): ' + ', '.join(f'{k} {v["R"]:.3g}' for k, v in t4.items() if isinstance(v, dict) and 'R' in v))
    out(f'  T4 fit: {g(G, "T4_fit")}')
    out(f'  T2 max deviation phase: {g(G, "T2", "max_deviation_phase_rad")}')
out()

# ── flow terms (blood sources) ──
out('## Motion of blood during the readout (source level): R global / R_loc per region')
out()
out('| design | phase | term | R all | R_loc all | R_loc chambers | R_loc vessels | R_loc coronary | R split |k|<0.1/0.1-0.5/0.5-0.9/>0.9 kmax |')
out('|---|---|---|---|---|---|---|---|---|')
for d in D:
    for ph in ['cycle', 'systole', 'diastole']:
        J = ld(f'flow_omission_{d}_{ph}.json')
        if J is None:
            continue
        for kind, nm in [('m1', 'T5+T7 readout M1 (all)'), ('disp', 'T5 displacement (parallel)'), ('perp', 'T7 M1 perpendicular'), ('acc', 'T6 acceleration (M2)'), ('pathb0', 'T12 motion through the B0 gradient')]:
            t = J['terms'].get(kind)
            if t is None:
                continue
            a = t['all']
            out(f'| {d} | {ph} | {nm} | {f(a["R"])} | {f(a["R_loc"])} | {f(t["chambers"]["R_loc"])} | {f(t["vessels"]["R_loc"])} | {f(t["coronary"]["R_loc"])} | '
                f'{f(a["R_k0.0-0.1"])} / {f(a["R_k0.1-0.5"])} / {f(a["R_k0.5-0.9"])} / {f(a["R_k0.9-9.0"])} |')
out()

# ── tissue ──
out('## Tissue motion (T8)')
out()
for d in D:
    J = ld(f'tissue_{d}.json')
    if J is None:
        continue
    for ph in ['systole', 'diastole']:
        if ph not in J:
            continue
        r = J[ph]
        out(f'{d} {ph}: myocardial speed p50/95/99 {np.round(r["speed_pct_50_95_99_myo_mm_s"], 1).tolist()} mm/s; ' + ', '.join(f'{k} R {v["R"]:.3g} R_loc {v["R_loc"]:.3g}' for k, v in r.items() if isinstance(v, dict) and 'R' in v))
out()

# ── warp handling ──
for wname in ['warp_eval.json', 'warp_eval_matched.json', 'warp_eval_shear.json']:
  W = ld(wname)
  if W:
    out(f'### {wname}')
    out('## Handling of T5/T6 by the lagged warp (R over the static floor; global R, R_loc = x factor)')
    out()
    keys = ['omit', 'exact_ideal', 'exact_nearest', 'exact_trilinear', 'exact_cubic', 'exact_lanczos3', 'exact_lanczos4', 'exact_ideal_perp', 'exact_ideal_bias0.8', 'exact_ideal_noise0.05', 'exact_ideal_smooth4mm',
            'acc_truth_omit', 'acc_truth_exact_ideal_noacc', 'acc_truth_exact_ideal_acc', 'pb0_truth_omit', 'pb0_truth_exact_ideal_nopb0', 'pb0_truth_exact_ideal_pb0map',
            'pb0_truth_exact_ideal_pb0src', 'all_truth_omit', 'all_truth_exact_lanczos3_full', 'exact_ideal_shear_cos', 'exact_ideal_shear_sinc', 'exact_lanczos3_shear_cos']
    for k_, r in W.items():
        segk = sorted([k for k in r if k.startswith('seg') and not k.endswith('static') and '_cor' not in k and '_ves' not in k and 'static' not in k])
        out(f'{k_}: L_rule {r["L_rule"]}, model voxels {r["n_model_voxels"]}, R_loc factor all {r["R_loc_factor_all"]:.2f}, vessels {r.get("R_loc_factor_ves", 0):.1f}, coronary {r["R_loc_factor_cor"]:.1f}; '
            f'static floor R {r["static"]["R"]:.3g}')
        out('  ' + ', '.join(f'{k} {r[k]["R_over_floor"]:.3g}' for k in keys + segk if k in r and 'R_over_floor' in r[k]))
        out('  vessels: ' + ', '.join(f'{k} {r[k]["R_over_floor"]:.3g}' for k in r if k.endswith('_ves') and isinstance(r[k], dict) and 'R_over_floor' in r[k]))
        out('  coronary: ' + ', '.join(f'{k} {r[k]["R_over_floor"]:.3g}' for k in r if k.endswith('_cor') and isinstance(r[k], dict) and 'R_over_floor' in r[k]))
    out()
I = ld('imgbias.json')
if I:
    out('## Image-space velocity bias of the unmodelled displacement (ideal full-sampling LSQ)')
    for k_, r in I.items():
        out(f'{k_}: ' + '; '.join(f'{reg}: n {v["n"]}, speed p50/95 {np.round(v["speed_mm_s_50_95"], 0).tolist()} mm/s, |bias| p50/95/99 {np.round(v["bias_mm_s_50_95_99"], 1).tolist()} mm/s, rms bias / rms speed {v["bias_rms_over_speed_rms"]:.3f}' for reg, v in r.items()))
open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(lines) + '\n')

# ── follow-ups ──
for d in D:
    J = ld(f'r2grad_{d}.json')
    if J:
        out(f'R2* follow-up {d}: ' + ', '.join(f'{k} {v["R"]:.3g}' for k, v in J.items() if isinstance(v, dict) and 'R' in v)
            + f"; map gradient Hz/cm p50/90/99 {np.round(J['gradient_hz_cm_pct_50_90_99'], 1).tolist()}, energy fraction with g D tmax / 4 > 1: {J['g_D_tmax_over4_gt1_energy_fraction']:.3f}")
    J = ld(f'fat_union_{d}.json')
    if J:
        out(f'fat per-frame maps {d}: union rule L {J.get("union_rule_L")}; ' + ', '.join(f'{k} {v["R"]:.3g}' for k, v in J.items() if isinstance(v, dict) and 'R' in v))
J = ld('t9check.json')
if J:
    out('T9 check (v2, 24 readouts): ' + ', '.join(f'{k} {v["R"]:.3g} (|k|<0.1 kmax {v["R_k0.0-0.1"]:.3g})' for k, v in J.items()))
open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(lines) + '\n')
