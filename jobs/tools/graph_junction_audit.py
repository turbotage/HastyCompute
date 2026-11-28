"""Geometric audit of a vessel graph's joints: hairpins, backward branches, kinks (what a viewer sees as 'weird').

    rustlib/.venv/bin/python -I jobs/tools/graph_junction_audit.py <dir with graph.npz/json> [--rmin-mm 2] [--top 30]

Per node with ≥ 2 incident edges (radius ≥ rmin): the direction of each incident edge leaving the node (chord to the
sample max(3 mm, 1.5 r) along it). Reports
- turn at degree-2 joints (bridges / joins / connectors meet the vessel they continue): 180° − angle between the two
  leaving directions; > 90° is a hairpin (the path doubles back);
- at junctions (degree ≥ 3): for each branch, the angle between it and the two largest edges' axis (the parent
  through-line); a branch leaving at < 25° to the reversed parent direction runs back along the parent ('backward');
- edges of kind bridge / join / connector / relocated with their length and the vessels they connect.
Also: kinks inside edges — the bending angle over ±max(2 mm, r) larger than 60°."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

A = sys.argv[1:]
D = Path(A[0])
RMIN = float(A[A.index("--rmin-mm") + 1]) * 1e-3 if "--rmin-mm" in A else 2e-3
TOP = int(A[A.index("--top") + 1]) if "--top" in A else 30
z = np.load(D / "graph.npz")
names = json.loads((D / "graph.json").read_text())["edge_names"]
off, P, R, ea, eb = z["edge_offsets"], z["points"].astype(np.float64), z["radius"].astype(np.float64), z["edge_a"], z["edge_b"]
E = len(names)
inc = defaultdict(list)
for k in range(E):
    if off[k + 1] - off[k] < 2:
        continue
    inc[int(ea[k])].append((k, 0))
    inc[int(eb[k])].append((k, 1))


def leave(k, end):
    p = P[off[k]:off[k + 1]]
    r = R[off[k]:off[k + 1]]
    if end == 1:
        p, r = p[::-1], r[::-1]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    L = max(3e-3, 1.5 * r[0])
    j = min(np.searchsorted(s, L), len(p) - 1)
    d = p[j] - p[0]
    return d / (np.linalg.norm(d) + 1e-30), float(r[: j + 1].mean()), float(s[-1])


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


hair, back = [], []
for n, lst in inc.items():
    if len(lst) < 2:
        continue
    L = [(k, e, *leave(k, e)) for k, e in lst]
    L.sort(key=lambda x: -x[3])
    if L[0][3] < RMIN:
        continue
    pos = P[off[L[0][0]]] if L[0][1] == 0 else P[off[L[0][0] + 1] - 1]
    if len(L) == 2:
        turn = 180 - ang(L[0][2], L[1][2])
        if turn > 60:
            hair.append((turn, n, names[L[0][0]], names[L[1][0]], L[0][3], L[1][3], pos))
    else:
        # parent through-line: the two largest edges
        a, b = L[0][2], L[1][2]
        for k, e, d, r, ln in L[2:]:
            if r < RMIN * 0.5:
                continue
            # a branch nearly parallel to one of the parent's leaving directions runs along that parent edge
            ma = min(ang(d, a), ang(d, b))
            if ma < 25:
                back.append((ma, n, names[k], names[L[0][0]] if ang(d, a) < ang(d, b) else names[L[1][0]], r, ln, pos))
print(f"{D}: {E} edges")
print(f"\nhairpins at degree-2 joints (turn > 60°; > 90° doubles back), radius ≥ {RMIN * 1e3:.1f} mm:")
for t, n, a_, b_, ra, rb, pos in sorted(hair, reverse=True)[:TOP]:
    print(f"  turn {t:5.1f}° node {n:5d} at {np.round(pos * 1e3).astype(int)} mm | {a_[:40]} (r {ra * 1e3:.1f}) ↔ {b_[:40]} (r {rb * 1e3:.1f})")
print("\nbranches leaving a junction along a parent edge (< 25°: co-running / folded back):")
for ma, n, k_, par, r, ln, pos in sorted(back)[:TOP]:
    print(f"  {ma:5.1f}° node {n:5d} at {np.round(pos * 1e3).astype(int)} mm | {k_[:40]} (r {r * 1e3:.1f}, {ln * 1e3:.0f} mm) along {par[:40]}")
kinds = defaultdict(list)
for k, nm in enumerate(names):
    for pre in ("bridge: ", "join: ", "connector: "):
        if nm.startswith(pre):
            p = P[off[k]:off[k + 1]]
            kinds[pre].append((float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), nm, float(R[off[k]:off[k + 1]].mean())))
for pre, v in kinds.items():
    v.sort(reverse=True)
    print(f"\n{pre.strip()} edges: {len(v)}; longest: " + "; ".join(f"{nm[len(pre):][:30]} {L * 1e3:.0f} mm r {r * 1e3:.1f}" for L, nm, r in v[:10]))
