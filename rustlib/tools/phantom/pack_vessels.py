"""Pack the vessel graph and its network flow for the GPU truth generator (`phantomflow`).

    python -I pack_vessels.py <export_dir> <flow_dir> <out.hv> [--wall-artery 0.12] [--wall-vein 0.06]

Contents (all SI):
- points f32 [N, 3], radius f32 [N], edge_offsets i64 [E+1]: the graph centrelines (samples of one edge are
  consecutive; consecutive samples form a capsule / truncated cone).
- cls u8 [N]: 1 systemic artery, 2 systemic vein, 3 pulmonary artery, 4 pulmonary vein, 5 other artery, 6 other vein.
- wall f32 [N]: wall thickness (fraction of r per class, ≥ 0.1 mm).
- q f32 [N, H+1, 2]: volume-flow harmonics (re, im), flow positive along the sample order; q(t) = Re[q0 + 2 Σ q_n
  e^{iω_n t}]. omega f64 [H+1], period s.
- prof f32 [NA, NX, 2]: Womersley profile shape g(ξ; α) = (1 − J0(Λξ)/J0(Λ)) / (1 − 2J1(Λ)/(ΛJ0(Λ))), Λ = α i^{3/2},
  so that u(ξ) = Q / (πr²) · g; α on [0, alpha_max] (NA points), ξ = d / r on [0, 1] (NX points); g(ξ; 0) = 2(1 − ξ²).
- edge_nodes i32 [E, 2] (graph nodes of each edge), vessel_id i32 [E] (one id per vessel name; connectors, joins and
  bridges take their vessel's id): which capsules belong together when the generator resolves contacts.
- rho, mu (blood).
Embedded branches: where a branch still runs fully inside a larger lumen of the same class (the atlas draws many
branches from their parent's axis), it owns no blood in `phantomflow` (ownership by signed depth), so its flow Q is
added to the parent samples it runs inside (from the first to the last embedded position), conserving mass until it
emerges.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import jv

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

E, F, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def opt(n, d):
    return float(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


RHO, MU = 1060.0, 3.5e-3
g = np.load(E / "graph.npz")
gj = json.loads((E / "graph.json").read_text())
fl = np.load(F / "network_flow.npz")
pts, rad, eoff, ecomp = g["points"], g["radius"], g["edge_offsets"], g["edge_comp"]
comps = {c["id"]: c for c in gj["components"]}
CODE = {"systemic_arterial": 1, "systemic_venous": 2, "pulmonary_arterial": 3, "pulmonary_venous": 4, "arterial_other": 5, "coronary": 5, "venous_other": 6}
ne = len(eoff) - 1
cls = np.zeros(len(pts), np.uint8)
for k in range(ne):
    cls[eoff[k] : eoff[k + 1]] = CODE[comps[int(ecomp[k])]["class"]]
venous = np.isin(cls, (2, 4, 6))
wall = np.maximum(np.where(venous, opt("--wall-vein", 0.06), opt("--wall-artery", 0.12)) * rad, 0.1e-3).astype(np.float32)
q = fl["q"].astype(np.complex128)
omega = fl["omega"]
assert q.shape[0] == len(pts), (q.shape, len(pts))

# Womersley profile table
amax = float(rad.max() * np.sqrt(omega.max() * RHO / MU)) * 1.05 + 1.0
NA, NX = int(np.ceil(amax / 0.125)) + 1, 129
al = np.linspace(0, amax, NA)
xi = np.linspace(0, 1, NX)
prof = np.zeros((NA, NX), np.complex128)
for i, a in enumerate(al):
    if a < 0.3:
        prof[i] = 2 * (1 - xi**2)
        continue
    L = a * np.exp(3j * np.pi / 4)
    j0 = jv(0, L)
    prof[i] = (1 - jv(0, L * xi) / j0) / (1 - 2 * jv(1, L) / (L * j0))
# check: the cross-sectional integral 2∫ g ξ dξ = 1 (unit mean velocity)
mean = 2 * np.trapezoid(prof * xi[None], xi, axis=1)
assert np.abs(mean - 1).max() < 2e-3, np.abs(mean - 1).max()
# ── embedded branches: hand their flow to the containing parent samples ──────────────────────────────────────────
from scipy.spatial import cKDTree  # noqa: E402

es = np.searchsorted(eoff, np.arange(len(pts)), side="right") - 1
cap = np.flatnonzero(es[:-1] == es[1:])  # capsule j: samples j, j+1
A, B = pts[cap].astype(np.float64), pts[cap + 1].astype(np.float64)
mid = 0.5 * (A + B)
rc = 0.5 * (rad[cap] + rad[cap + 1]).astype(np.float64)
tree = cKDTree(mid)
seglen = np.linalg.norm(B - A, axis=1)
nb = tree.query_ball_point(mid, rc + 0.5 * seglen.max() + 1e-4)
pi, pj = [], []
for i, js in enumerate(nb):
    js = np.asarray(js)
    js = js[(rc[js] < rc[i]) & (es[cap[js]] != es[cap[i]]) & (cls[cap[js]] == cls[cap[i]])]
    pi.append(np.full(len(js), i))
    pj.append(js)
pi, pj = np.concatenate(pi), np.concatenate(pj)
ab = B[pi] - A[pi]
sg = np.clip(np.einsum("ij,ij->i", mid[pj] - A[pi], ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
dd = np.linalg.norm(mid[pj] - (A[pi] + sg[:, None] * ab), axis=1)
rr = rad[cap[pi]] + sg * (rad[cap[pi] + 1] - rad[cap[pi]])
# child capsule j fully inside the cylinder of parent capsule i (not its end cap: a vessel tapering through a node
# sits in its previous piece's cap), and not a continuation of the parent through a shared node
leave = {}
for k in range(len(eoff) - 1):
    a_, b_ = int(eoff[k]), int(eoff[k + 1])
    if b_ - a_ >= 2:
        j_ = min(b_ - a_ - 1, 4)
        for nd, fr, to in ((int(g["edge_a"][k]), a_, a_ + j_), (int(g["edge_b"][k]), b_ - 1, b_ - 1 - j_)):
            d_ = pts[to].astype(np.float64) - pts[fr]
            leave[(k, nd)] = d_ / max(np.linalg.norm(d_), 1e-12)


def continuation(e1, e2):
    for nd in (int(g["edge_a"][e1]), int(g["edge_b"][e1])):
        if nd in (int(g["edge_a"][e2]), int(g["edge_b"][e2])) and (e1, nd) in leave and (e2, nd) in leave:
            if np.dot(leave[(e1, nd)], leave[(e2, nd)]) < -0.5:
                return True
    return False


emb = (dd + rc[pj] < rr) & (sg > 1e-3) & (sg < 1 - 1e-3)
emb[emb] = [not continuation(int(es[cap[a_]]), int(es[cap[b_]])) for a_, b_ in zip(pi[emb], pj[emb])]
pi, pj, depth = pi[emb], pj[emb], (dd - rr)[emb]
owner = {}
for i, j, d in zip(pi, pj, depth):
    if j not in owner or d < owner[j][1]:
        owner[j] = (i, d)
runs = {}  # (child edge, parent edge) → [parent capsule indices], [child capsules], tangent dots
for j, (i, _) in owner.items():
    key = (int(es[cap[j]]), int(es[cap[i]]))
    r_ = runs.setdefault(key, [[], [], []])
    r_[0].append(int(cap[i]))
    r_[1].append(int(cap[j]))
    tj, ti = B[j] - A[j], B[i] - A[i]
    r_[2].append(float(np.dot(tj, ti) / max(np.linalg.norm(tj) * np.linalg.norm(ti), 1e-18)))
q = q.copy()
handed = 0
for (ce, pe), (ps_, cs_, dots) in runs.items():
    lo, hi = min(ps_), max(ps_) + 1  # parent samples lo..hi (capsules lo..hi−1 and their end sample)
    sgn = 1.0 if np.mean(dots) >= 0 else -1.0
    qc = q[min(cs_)]  # the child's flow at its first embedded sample (positive along the child's order)
    q[lo : hi + 1] += sgn * qc
    handed += 1
print(f"embedded branch runs handed to their parents: {handed} ({len(owner)} capsules)")
base = [n.split(": ", 1)[1] if n.startswith(("connector: ", "join: ", "bridge: ")) else n for n in gj["edge_names"]]
vid = {n: i for i, n in enumerate(sorted(set(base)))}
vessel_id = np.array([vid[n] for n in base], np.int32)
edge_nodes = np.stack([g["edge_a"], g["edge_b"]], 1).astype(np.int32)
vf.write(
    OUT,
    {
        "points": pts.astype(np.float32), "radius": rad.astype(np.float32), "edge_offsets": eoff.astype(np.int64), "cls": cls, "wall": wall, "edge_nodes": edge_nodes, "vessel_id": vessel_id,
        "q": np.stack([q.real, q.imag], -1).astype(np.float32), "omega": omega.astype(np.float64), "period": float(fl["period"]),
        "prof": np.stack([prof.real, prof.imag], -1).astype(np.float32), "alpha_max": amax, "rho": RHO, "mu": MU,
        "source": "Z-Anatomy (CC BY-SA 4.0) centrelines; network_flow.py harmonics",
    },
)
print(f"wrote {OUT}: {len(pts)} samples, {ne} edges, {len(omega) - 1} harmonics, profile table {NA}×{NX} (α ≤ {amax:.1f}), |mean−1| {np.abs(mean - 1).max():.1e}")
