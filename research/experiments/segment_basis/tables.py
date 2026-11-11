import json, glob, sys, numpy as np
rows = {}
for f in sorted(glob.glob('res3_*.jsonl')):
    for l in open(f):
        r = json.loads(l); rows[(r['case'], r['variant'], r['cand'], r['L'])] = r
# scale correction for variants with a modified field (noise reference = the raw field's data)
scales = {}
for f in glob.glob('log3_*.log'):
    v = None
    for line in open(f):
        if line.startswith('variant '): v = line.split()[1].rstrip(':')
        if line.startswith('case:'): case_f = 'fat' if 'fat True' in line else 'water'
        if line.startswith('scale model/meas') and v: scales[(case_f, v)] = float(line.split()[-1])
base = {c: scales.get((c, 'raw')) for c in ('water', 'fat')}
def corr(case, var):
    s = scales.get((case, var)); b = base.get(case)
    return np.sqrt(s / b) if (s and b) else 1.0
# mean of the mixed weights (for tail -> R prediction) per case

cases = sorted({(k[0], k[1]) for k in rows})
for case, var in cases:
    cands = sorted({k[2] for k in rows if k[0] == case and k[1] == var})
    Ls = sorted({k[3] for k in rows if k[0] == case and k[1] == var})
    print(f'\n### {case}, field {var} (R = data error rms / noise rms; noise-reference correction x{corr(case, var):.3f})\n')
    print('| basis | ' + ' | '.join(f'L={L}' for L in Ls) + ' |')
    print('|---|' + '---|' * len(Ls))
    for c in cands:
        cells = []
        for L in Ls:
            r = rows.get((case, var, c, L))
            cells.append('' if r is None else f"{r['R'] * corr(case, var):.3f}")
        print(f'| {c} | ' + ' | '.join(cells) + ' |')
