#!/usr/bin/env python3
"""Markdown tables for §18 from a/ (before), b/ (after), c/ (planner) results.jsonl"""
import json, sys, os, collections, re
sys.path.insert(0, os.path.dirname(__file__))
D = os.path.dirname(os.path.abspath(__file__))
_t = open(os.path.join(D, 'table.py')).read()
PEAK = 1008e9
exec(_t[_t.index("def fft_model"):_t.index("rows = [row(r) for r in recs]")])
def load(t):
    p = os.path.join(D, t, 'results.jsonl')
    if not os.path.exists(p): return []
    out = []
    for l in open(p):
        if l.strip():
            r = json.loads(l); r['_file'] = p; out.append(row(r))
    return out
A, B, C = load('a'), load('b'), load('c')
mx = lambda d: '×'.join(map(str, d['m']))
gx = lambda d: '×'.join(map(str, d['grid']))
def key(d): return (mx(d), d['nc'], d['L'], d['o'], round(d['M'] / 1e4))
def best(rows):
    b = {}
    for d in rows:
        if rows is B and not d['sk']: continue
        k = key(d)
        if k not in b or d['grad'] < b[k]['grad']: b[k] = d
    return b
ba, bb = best(A), best(B)
print("### before / after (best channel batch of each)\n")
print("| matrix (grid) | coils × L | order | samples | before: batch, gradient (fwd / adj) | after: batch, gradient (fwd / adj) | after: interp fwd / adj, FFT fwd / inv, rest fwd / inv | change |")
print("|---|---|---|---|---|---|---|---|")
for k in sorted(set(ba) | set(bb), key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
    a, b = ba.get(k), bb.get(k)
    d = a or b
    fa = f"{a['cb']}×{a['lb']} = {a['ch']} ch: **{a['grad']:.0f}** ({a['fwd']:.0f} / {a['adj']:.0f})" if a else "—"
    fb = f"{b['cb']}×{b['lb']} = {b['ch']} ch: **{b['grad']:.0f}** ({b['fwd']:.0f} / {b['adj']:.0f})" if b else "—"
    br = f"{b['kf']:.0f} / {b['ka']:.0f}, {b['fftf']:.0f} / {b['ffti']:.0f}, {b['restf']:.0f} / {b['resti']:.0f}" if b else ""
    ch = f"{100*(b['grad']/a['grad']-1):+.0f} %" if a and b else ""
    print(f"| {mx(d)} ({gx(d)}) | {d['nc']} × {d['L']} | {d['o']} | {d['M']/1e6:.2f} M | {fa} | {fb} | {br} | {ch} |")
print("\n### gradient vs channel batch\n")
print("| matrix | coils × L | order | samples | path | channels per batch (coils × segments): gradient ms |")
print("|---|---|---|---|---|---|")
g = collections.defaultdict(list)
for rows in (A, B):
    for d in rows:
        g[(mx(d), d['nc'], d['L'], d['o'], round(d['M'] / 1e4), 'one-bin knot' if d['sk'] else 'static')].append(d)
for k, v in sorted(g.items()):
    if len(v) < 2: continue
    v.sort(key=lambda d: d['ch'])
    print(f"| {k[0]} | {k[1]} × {k[2]} | {k[3]} | {v[0]['M']/1e6:.2f} M | {k[5]} | " + ", ".join(f"{d['ch']} ({d['cb']}×{d['lb']}): {d['grad']:.0f}" for d in v) + " |")
print("\n### bandwidth, memory, lower bound, variants (after)\n")
print("| matrix | coils × L | order | samples | batch | engine GiB | free GiB after | FFT model BW % of 1008 GB/s fwd / inv (slabs) | lower bound ms (FFT bytes at peak + measured interp) | measured | variants |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k, b in sorted(bb.items()):
    var = ('vs/vt ' + ('on' if b['vs'] != 'None' else 'off')) + ', ra ' + ('on' if b['ra'] != 'None' else 'off') + ', kl ' + ('skip' if 'active true' in b['kl'] else 'off') + (' + kl kernel' if 'kernel Some' in b['kl'] else '')
    print(f"| {mx(b)} | {b['nc']} × {b['L']} | {b['o']} | {b['M']/1e6:.2f} M | {b['cb']}×{b['lb']} | {b['eng']:.2f} | {b['free']:.2f} | {100*b['bwf']:.0f} / {100*b['bwi']:.0f} ({'y' if b['slab'] else 'n'}) | {b['lbound']:.0f} | {b['grad']:.0f} | {var} |")
if C:
    print("\n### planner (SegSplit::planned)\n")
    print("| matrix | coils × L | budget GiB | chosen coils × segments (groups, passes) | gradient ms |")
    print("|---|---|---|---|---|")
    for d, r in zip(C, [json.loads(l) for l in open(os.path.join(D, 'c', 'results.jsonl')) if l.strip()]):
        bud = r['cmdline'].split('--plan-budget-gb ')[1].split()[0]
        print(f"| {mx(d)} | {d['nc']} × {d['L']} | {bud} | {d['cb']} × {d['lb']} ({r.get('seg_groups')}, {d['passes']}) | {d['grad']:.0f} |")
