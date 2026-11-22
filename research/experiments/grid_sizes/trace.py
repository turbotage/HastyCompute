#!/usr/bin/env python3
"""Per-kernel breakdown of one gradient from an nsys cuda_gpu_trace CSV (opbench --profile range): kernels grouped by
(name, grid, block, dynamic smem) in order of first launch, total ms and count; memsets/memcpys summed.
usage: trace.py <case>_cuda_gpu_trace.csv"""
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
g = collections.OrderedDict()
tot = 0.0
for r in rows:
    name = r['Name']
    d = float(r['Duration (ns)']) * 1e-6
    tot += d
    if r.get('GrdX'):
        key = (name[:40], r['GrdX'], r['GrdY'], r['GrdZ'], r['BlkX'], r.get('DymSMem (MB)', r.get('DymSMem', '')))
    else:
        key = (name[:40], r.get('Bytes', ''), '', '', '', '')
    e = g.setdefault(key, [0.0, 0])
    e[0] += d; e[1] += 1
print(f"total GPU time in range {tot:.2f} ms over {len(rows)} events")
for k, (t, n) in g.items():
    print(f"{t:9.2f} ms  x{n:<5d} {k[0]:<40} grid ({k[1]},{k[2]},{k[3]}) block {k[4]} smem {k[5]}")
