#!/usr/bin/env python3
"""Per-pass bandwidth of the fftpass_bench traces (fft/*.csv): x, y, z pass time per transform (mean of forward and
inverse) and the model bytes of each pass (slabs: the x pass writes and the y pass reads the intermediate in L2)."""
import csv, glob, os, re, collections
D = os.path.dirname(os.path.abspath(__file__))
cfg = {}
for l in open(os.path.join(D, 'fft', 'fft.log')):
    m = re.match(r'== (\S+): (.*) \(', l)
    if m: cfg[m.group(1)] = list(map(int, m.group(2).split()[:8]))
print("| grid (block) | channels | slabs | x pass ms (GB/s, % peak) | y pass | z pass | transform ms |")
print("|---|---|---|---|---|---|---|")
for name, c in cfg.items():
    n, N, nch, slab = c[0:3], c[3:6], c[6], c[7]
    p = os.path.join(D, 'fft', f'{name}_cuda_gpu_trace.csv')
    g = collections.OrderedDict()
    for r in csv.DictReader(open(p)):
        if r['Name'] != 'fft_axis': continue
        k = (r['GrdX'], r['GrdY'], r['GrdZ'], r['BlkX'])
        g.setdefault(k, [0.0, 0])[0] += float(r['Duration (ns)']) * 1e-6
    reps = 12  # 6 forward + 6 inverse transforms per pass signature
    t = [v[0] / reps for v in g.values()]
    blk, xo, yo, full = N[0]*N[1]*N[2], n[0]*N[1]*N[2], n[0]*n[1]*N[2], n[0]*n[1]*n[2]
    b = [blk + (0 if slab else xo), (0 if slab else xo) + yo, yo + full]
    b = [x * nch * 8 for x in b]
    cells = [f"{ti:.2f} ({bi/ti/1e6:.0f}, {100*bi/ti/1e6/1008:.0f} %)" for ti, bi in zip(t, b)]
    print(f"| {'×'.join(map(str,n))} ({'×'.join(map(str,N))}) | {nch} | {slab} | {cells[0]} | {cells[1]} | {cells[2]} | {sum(t):.1f} |")
