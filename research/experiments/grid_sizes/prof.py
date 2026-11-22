#!/usr/bin/env python3
"""One-gradient breakdown from p/*_cuda_gpu_trace.csv: interpolation kernels, FFT passes (x pass incl. fused pad / crop),
pad / crop / zeroing / memsets."""
import csv, glob, os, collections
D = os.path.dirname(os.path.abspath(__file__))
print("| case | total ms | gather (fwd kernel) | scatter (adj kernel) | FFT passes (x incl. fused pad / crop, y, z) | pad / crop kernels | zeroing + memsets |")
print("|---|---|---|---|---|---|---|")
for p in sorted(glob.glob(os.path.join(D, 'p', '*_cuda_gpu_trace.csv'))):
    name = os.path.basename(p).replace('_cuda_gpu_trace.csv', '')
    c = collections.defaultdict(float); fft = collections.OrderedDict(); tot = 0
    for r in csv.DictReader(open(p)):
        n = r['Name']; d = float(r['Duration (ns)']) * 1e-6; tot += d
        if n == 'fft_axis':
            k = (r['GrdX'], r['GrdY'], r['GrdZ']); fft[k] = fft.get(k, 0) + d
        elif n.startswith('seg_forward'): c['f'] += d
        elif n.startswith('seg_adjoint'): c['a'] += d
        elif n in ('seg_pad', 'seg_crop'): c['pc'] += d
        else: c['z'] += d
    fv = list(fft.values())
    print(f"| {name} | {tot:.1f} | {c['f']:.1f} | {c['a']:.1f} | {sum(fv):.1f} ({' / '.join(f'{v:.1f}' for v in fv)}) | {c['pc']:.1f} | {c['z']:.1f} |")
