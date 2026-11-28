"""Which vessels touch / run inside a named vessel's lumen in a phantom graph (capsule model, as pack_vessels / phantomflow).

    rustlib/.venv/bin/python -I jobs/tools/vessel_contacts.py <dir with graph.npz/json> "<name substring>" [--gap-mm 1]

For every other edge: samples whose capsule comes within gap of the target lumen; reports the depth of its centreline in
the target (negative = centreline inside the target lumen), the length of contact, the angle between the tangents, and
whether the edge is graph-adjacent to the target (shares a node: a real tributary / continuation) or not (a foreign
vessel crossing or running alongside)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

A = sys.argv[1:]
D, PAT = Path(A[0]), A[1].lower()
GAP = float(A[A.index("--gap-mm") + 1]) * 1e-3 if "--gap-mm" in A else 1e-3
z = np.load(D / "graph.npz")
names = json.loads((D / "graph.json").read_text())["edge_names"]
off, P, R, ea, eb = z["edge_offsets"], z["points"].astype(np.float64), z["radius"].astype(np.float64), z["edge_a"], z["edge_b"]
E = len(names)
eid = np.repeat(np.arange(E), np.diff(off))
tgt = np.array([PAT in n.lower() for n in names])
tnodes = set(ea[tgt]) | set(eb[tgt])
# tangents per sample
T = np.zeros_like(P)
for i in range(E):
    p = P[off[i]:off[i + 1]]
    if len(p) > 1:
        t = np.gradient(p, axis=0)
        T[off[i]:off[i + 1]] = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-30)
ti = np.flatnonzero(tgt[eid])
tree = cKDTree(P[ti])
rmax = R[ti].max()
rows = {}
for i in range(E):
    if tgt[i]:
        continue
    s = np.arange(off[i], off[i + 1])
    d, k = tree.query(P[s], k=8, distance_upper_bound=rmax + R[s].max() + GAP + 5e-3)
    ok = np.isfinite(d)
    if not ok.any():
        continue
    kk = np.where(ok, k, 0)
    j = ti[kk]
    # depth of this sample's centreline relative to the target surface (point-to-sample approx, samples ≤ 0.5 mm apart)
    depth = np.where(ok, d - R[j], np.inf)
    a = depth.argmin(1)
    dep = depth[np.arange(len(s)), a]
    jj = j[np.arange(len(s)), a]
    touch = dep < R[s] + GAP
    if not touch.any():
        continue
    seg = np.linalg.norm(np.diff(P[s], axis=0), axis=1)
    L = float(seg[(touch[1:] & touch[:-1])].sum())
    ang = np.degrees(np.arccos(np.clip(np.abs((T[s] * T[jj]).sum(1)), 0, 1)))
    adj = bool({ea[i], eb[i]} & tnodes)
    rows[i] = dict(name=names[i], adj=adj, n=int(touch.sum()), L_mm=L * 1e3, depth_min_mm=float(dep.min() * 1e3), r_mm=float(R[s][touch].mean() * 1e3),
                   ang_med=float(np.median(ang[touch])), z_mm=(float(P[s][touch][:, 2].min() * 1e3), float(P[s][touch][:, 2].max() * 1e3)),
                   tgt=sorted({names[e] for e in set(eid[jj[touch]])}))
print(f"target '{PAT}': {tgt.sum()} edges, r {R[ti].min() * 1e3:.1f}–{R[ti].max() * 1e3:.1f} mm, z {P[ti][:, 2].min() * 1e3:.0f}–{P[ti][:, 2].max() * 1e3:.0f} mm")
for i, r in sorted(rows.items(), key=lambda x: x[1]["depth_min_mm"]):
    print(f"  {'ADJ' if r['adj'] else '---'} e{i:5d} {r['name'][:44]:44s} r {r['r_mm']:4.1f} contact {r['L_mm']:5.1f} mm depth {r['depth_min_mm']:6.1f} mm angle {r['ang_med']:4.0f}° z {r['z_mm'][0]:.0f}–{r['z_mm'][1]:.0f} | {', '.join(r['tgt'])[:60]}")
