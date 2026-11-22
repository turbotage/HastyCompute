#!/usr/bin/env python3
"""§18 tables from opbench --jsonl records: per case the gradient and its parts, memory, the FFT byte model and its
bandwidth, and a lower bound. usage: table.py results.jsonl [more.jsonl ...] [--peak 1008]"""
import json, re, sys
PEAK = 1008e9   # RTX 4090 DRAM peak (21 Gbit/s x 384 bit)
args = [a for a in sys.argv[1:] if not a.startswith('--')]
if '--peak' in sys.argv:
    PEAK = float(sys.argv[sys.argv.index('--peak') + 1]) * 1e9
recs = []
for p in args:
    for l in open(p):
        l = l.strip()
        if l:
            r = json.loads(l); r['_file'] = p; recs.append(r)

def fft_model(r):
    """DRAM bytes of one plain (unfused) pruned forward and inverse transform of one channel batch, and the slab flag"""
    d = r['fft_desc']
    N = r['matrix']; n = r['grid']; nch = r['channels']
    fz = float(re.search(r'z-pass columns ([0-9.]+)', d).group(1)) if 'z-pass columns' in d else 1.0
    fc = float(re.search(r'cells ([0-9.]+)', d).group(1)) if 'cells' in d else 1.0
    slab = 'two-pass L2 slabs' in d
    blk = N[0] * N[1] * N[2]; xo = n[0] * N[1] * N[2]; yo = n[0] * n[1] * N[2]; full = n[0] * n[1] * n[2]
    # forward: x R blk W xo | y R xo W yo | z R fz*yo W fc*full ; slabs: the x output stays in L2 (no W xo, no R xo)
    fwd = blk + (0 if slab else 2 * xo) + yo + fz * yo + fc * full
    inv = fc * full + fz * yo + yo + (0 if slab else 2 * xo) + blk
    return fwd * nch * 8, inv * nch * 8, slab

def short(s, n=60):
    return s if len(s) <= n else s[:n]

def row(r):
    ff, fi = r['fft_one_ms']; nf, ni = r['fft_count']
    bf, bi, slab = fft_model(r)
    bwf, bwi = bf / (ff * 1e-3), bi / (fi * 1e-3)
    kf, ka = r['kernels_ms']
    # lower bound: every transform at peak bandwidth + the measured interpolation kernels
    lb = (bf * nf + bi * ni) / PEAK * 1e3 + (kf + ka)
    return dict(m=r['matrix'], grid=r['grid'], nc=r['coils'], L=r['nseg'], o=r['order'], M=r['samples'], cb=r['coil_batch'], lb=r.get('seg_batch', r['nseg']),
                ch=r['channels'], passes=r.get('passes', -1), grad=r['gradient_ms'], fwd=r['forward_ms'], adj=r['adjoint_ms'], kf=kf, ka=ka,
                fftf=r['fft_ms'][0], ffti=r['fft_ms'][1], restf=r['rest_ms'][0], resti=r['rest_ms'][1], eng=r['engine_gib'], free=r.get('free_gib_after', float('nan')),
                bwf=bwf / PEAK, bwi=bwi / PEAK, slab=slab, lbound=lb, sk=r.get('static_knot', False), vs=r['vs'], ra=r['ra'], kl=r['kl'], tiles=r['tiles'], mt=r['multi_tiles'], f=r['_file'])

rows = [row(r) for r in recs]
hdr = "matrix | grid | coils x L | order | M | path | cb x lb = ch (passes) | gradient | fwd / adj | interp fwd / adj | FFT fwd / inv | rest fwd / inv | engine GiB | FFT BW % peak fwd / inv (slab) | lower bound"
print(hdr)
for d in rows:
    print(f"{'x'.join(map(str,d['m']))} | {'x'.join(map(str,d['grid']))} | {d['nc']}x{d['L']} | {d['o']} | {d['M']/1e6:.2f} M | {'one-bin knot' if d['sk'] else 'static'} | {d['cb']}x{d['lb']} = {d['ch']} ({d['passes']}) | {d['grad']:.1f} | {d['fwd']:.1f} / {d['adj']:.1f} | {d['kf']:.1f} / {d['ka']:.1f} | {d['fftf']:.1f} / {d['ffti']:.1f} | {d['restf']:.1f} / {d['resti']:.1f} | {d['eng']:.2f} | {100*d['bwf']:.0f} / {100*d['bwi']:.0f} ({'y' if d['slab'] else 'n'}) | {d['lbound']:.1f}")
if '--variants' in sys.argv:
    for d in rows:
        print(f"{'x'.join(map(str,d['m']))} o{d['o']} M {d['M']/1e6:.2f} {d['nc']}x{d['L']} cb {d['cb']}x{d['lb']} {'1bin' if d['sk'] else 'static'}: vs {short(d['vs'],80)} | ra {short(d['ra'],60)} | kl {short(d['kl'],70)} | tiles {short(d['tiles'],200)} | multi {short(d['mt'],200)}")
