#!/usr/bin/env python3
"""Digest of the phase-2 audit (out_p2/, families ra2 / kl2 / vt2 / vs2) against the phase-1 baseline (out/):
key-shape tables (registers / warps per SM, S = spill-store bytes) per architecture and per-arch spill counts.
research/sample_kernels.md §19."""
import json, collections, sys
here = sys.argv[1] if len(sys.argv) > 1 else '.'
rows = [json.loads(l) for l in open(f'{here}/out_p2/results.jsonl')]
base = [json.loads(l) for l in open(f'{here}/out/results.jsonl')]
archs = ['sm_80', 'sm_89', 'sm_90', 'sm_100', 'sm_120']
idx = {(r['case'], r['family'], r['cfg'], r['arch']): r for r in rows + base}
def cell(r):
    if r is None: return '-'
    s = f"{r['regs']}/{r['warps_sm']}"
    if r['spill_st'] > 0: s += f" S{r['spill_st']}"
    return s
out = []
out.append('# Phase-2 kernel audit: key shapes (registers / warps per SM; S = spill-store bytes)\n')
n = collections.Counter(); c = collections.Counter()
for r in rows:
    n[(r['family'], r['arch'])] += 1
    c[(r['family'], r['arch'])] += r['spill_st'] > 0
out.append('| family | ' + ' | '.join(archs) + ' |\n|---|' + '---|' * len(archs))
for f in ['ra2_adj', 'kl2_adj', 'vt2_fwd', 'vs2_fwd']:
    out.append(f'| {f} (kernels / spilling) | ' + ' | '.join(f'{n[(f, a)]} / {c[(f, a)]}' for a in archs) + ' |')
def table(case, fam, cfgs, title):
    out.append(f'\n## {case}: {title}\n\n| variant | smem (B) | ' + ' | '.join(archs) + ' |\n|---|---|' + '---|' * len(archs))
    for lab, f, cfg in cfgs:
        rs = [idx.get((case, f, cfg, a)) for a in archs]
        sm = next((r['smem_dyn'] for r in rs if r), '')
        out.append(f'| {lab} | {sm} | ' + ' | '.join(cell(r) for r in rs) + ' |')
RV = [('base chk 16', 'ra_adj', '{p} chk16 listmax1024'), ('base chk 16, hint 4', 'ra2_adj', '{p} chk16 inplace0 stages1 lbmin4'), ('base chk 16, hint 3', 'ra2_adj', '{p} chk16 inplace0 stages1 lbmin3'),
      ('in place chk 16', 'ra2_adj', '{p} chk16 inplace1 stages1 lbmin0'), ('in place chk 12', 'ra2_adj', '{p} chk12 inplace1 stages1 lbmin0'), ('in place chk 12, hint 4', 'ra2_adj', '{p} chk12 inplace1 stages1 lbmin4'),
      ('cp.async chk 8', 'ra2_adj', '{p} chk8 inplace1 stages2 lbmin0'), ('cp.async chk 12', 'ra2_adj', '{p} chk12 inplace1 stages2 lbmin0'), ('cp.async chk 12, hint 4', 'ra2_adj', '{p} chk12 inplace1 stages2 lbmin4'),
      ('cp.async chk 16', 'ra2_adj', '{p} chk16 inplace1 stages2 lbmin0'), ('cp.async chk 24', 'ra2_adj', '{p} chk24 inplace1 stages2 lbmin0')]
KV = [('base chk 16', 'kl2_adj', '{p} chk16 stages1 lbmin0'), ('chk 12', 'kl2_adj', '{p} chk12 stages1 lbmin0'), ('chk 12, hint 4', 'kl2_adj', '{p} chk12 stages1 lbmin4'),
      ('cp.async chk 8', 'kl2_adj', '{p} chk8 stages2 lbmin0'), ('cp.async chk 12', 'kl2_adj', '{p} chk12 stages2 lbmin0'), ('cp.async chk 12, hint 4', 'kl2_adj', '{p} chk12 stages2 lbmin4'),
      ('cp.async chk 16', 'kl2_adj', '{p} chk16 stages2 lbmin0'), ('cp.async chk 24', 'kl2_adj', '{p} chk24 stages2 lbmin0')]
for case, p in [('vel2d_o3_cb5', 'nxg4 cb7'), ('vel2d_o3_cb10', 'nxg4 cb13'), ('knot1d_o2_cb5', 'nxg4 cb7'), ('static_o2_cb5', 'nxg4 cb7'), ('static_o0_10x10', 'nxg4 cb13'), ('static_o0_48x1', 'nxg2 cb3'), ('static_o0_48x1', 'nxg4 cb6')]:
    table(case, 'ra', [(l, f, c.format(p=p)) for l, f, c in RV], f'ra {p}')
    if case.startswith(('vel', 'knot')):
        table(case, 'kl', [(l, f, c.format(p=p)) for l, f, c in KV], f'kl {p}')
for case, g, cfs, f0, f1 in [('vel2d_o3_cb5', '4x4x16 smax48', ['nwc4 nws1', 'nwc4 nws2', 'nwc2 nws1'], 'vt_fwd', 'vt2_fwd'), ('vel2d_o3_cb10', '4x4x16 smax48', ['nwc7 nws1', 'nwc4 nws1'], 'vt_fwd', 'vt2_fwd'),
                             ('static_o2_cb5', '4x4x16 smax48', ['nwc4 nws1', 'nwc4 nws2'], 'vt_fwd', 'vt2_fwd'), ('static_o0_10x10', '4x4x16 smax64', ['nwc7 nws1', 'nwc4 nws1'], 'vs_fwd', 'vs2_fwd')]:
    cfgs = []
    for cf in cfs:
        cfgs.append((f'{cf} one pass', f0, f'{g} {cf} vxpad0'))
        for sch in [32, 24, 16, 12]:
            cfgs.append((f'{cf} sch {sch}', f1, f'{g} {cf} vxpad0 sch{sch}'))
    table(case, 'vt', cfgs, f'vs / vt {g}')
print('\n'.join(out))
