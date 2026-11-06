"""Vessel graph from the exported Z-Anatomy centrelines (`blend_export.py`): stitch the per-region curves into trees.

    python -I vessel_graph.py <export_dir>

1. Splines → polylines (points, radius, vessel name).
2. Every spline endpoint attaches to the nearest point of another spline if it lies within max(1.5 mm, 2 r) (branch
   origins and continuations); that spline is split there. Gaps beyond the tolerance stay free ends (terminals).
3. Junctions closer than 0.5 mm merge; edges are the polyline pieces between nodes.
4. Connected components (union-find), classified by their member vessels: systemic arterial (aorta), systemic venous
   (venae cavae), pulmonary arterial (pulmonary trunk), pulmonary venous (pulmonary veins), coronary, other.
5. Orientation: breadth-first from each component's root (aortic root, pulmonary trunk, the venae cavae' atrial ends,
   pulmonary veins' atrial ends), so edges point downstream for arteries and upstream (toward the heart) for veins.
Writes graph.npz (node positions, edge node pairs, per-edge polyline offsets into points / radius), graph.json
(names, classes, components, diagnostics) and figs/graph.png.
"""
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

E = Path(sys.argv[1])
v = np.load(E / "vessels.npz")
P, R, V, S = v["points"].astype(np.float64), v["radius"].astype(np.float64), v["vessel"], v["spline"]
names = [x["name"] for x in json.loads((E / "vessels.json").read_text())["vessels"]]
# radius corrections: atlas radii (bevel × point radius) that are far below adult values; target = the vessel's maximum
# radius after scaling (all its samples scale by the same factor). Sources: typical adult CT/MR diameters.
RFIX = {"Hepatic portal vein": 6.0e-3, "Superior mesenteric vein": 4.5e-3, "Splenic vein": 4.0e-3, "Left subclavian artery": 4.5e-3,
        "Right superior pulmonary vein": 6.5e-3, "Right inferior pulmonary vein": 6.0e-3, "Left superior pulmonary vein": 6.0e-3, "Left inferior pulmonary vein": 5.5e-3}
rfix_applied = {}
for vi, nm in enumerate(names):
    if nm in RFIX:
        m = V == vi
        f = RFIX[nm] / R[m].max()
        R[m] *= f
        rfix_applied[nm] = round(float(f), 3)
RMIN = {"Splenic artery": 2.0e-3, "Common hepatic artery": 2.2e-3, "Proper hepatic artery": 1.8e-3}  # thin atlas stretches
for vi, nm in enumerate(names):
    if nm in RMIN:
        m = V == vi
        R[m] = np.maximum(R[m], RMIN[nm])
        rfix_applied[nm + " (floor)"] = RMIN[nm] * 1e3
print("radius corrections:", rfix_applied)
key = V.astype(np.int64) * 1000 + S
starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
ends = np.r_[starts[1:], len(P)]
polys = [(s, e) for s, e in zip(starts, ends) if e - s >= 2]
pid = np.full(len(P), -1)
for i, (s, e) in enumerate(polys):
    pid[s:e] = i
tree = cKDTree(P)


def system_of(name):
    """(venous, pulmonary) from the vessel name: attachments and bridges stay inside one system"""
    n = name.lower()
    venous = any(w in n for w in ("vein", "venous", "vena", "venae")) or ("sinus" in n and "arter" not in n and "carotid" not in n)
    pulmonary = ("lung" in n or "pulmonary" in n) and "bronchial" not in n
    return (venous, pulmonary)


vsys = [system_of(n) for n in names]
psys = [vsys[V[s]] for s, e in polys]

# ── attachments: cut points per polyline ───────────────────────────────────────────────────────────────────────────
cuts = defaultdict(set)  # polyline -> local indices where it is split
attach = []  # (polyline, end 0/1, target polyline, target local index, distance)
for i, (s, e) in enumerate(polys):
    for end, gi in ((0, s), (1, e - 1)):
        tol = max(1.5e-3, 2.0 * R[gi])
        dd, jj = tree.query(P[gi], k=96, distance_upper_bound=tol)
        jc = np.minimum(jj, len(P) - 1)
        ok = (jj < len(P)) & (pid[jc] != i) & (pid[jc] >= 0)
        ok &= np.array([psys[pid[j]] == psys[i] if pid[j] >= 0 else False for j in jc])
        if not ok.any():
            continue
        j = jj[ok][0]
        q = pid[j]
        lj = j - polys[q][0]
        attach.append((i, end, q, lj, dd[ok][0]))
        cuts[q].add(int(lj))
        cuts[i].add(0 if end == 0 else e - s - 1)
for i, (s, e) in enumerate(polys):
    cuts[i] |= {0, e - s - 1}

# ── nodes: cut positions, merged within 0.5 mm ────────────────────────────────────────────────────────────────────
raw = []  # (polyline, local index)
for i, cs in cuts.items():
    for c in cs:
        raw.append((i, c))
rpos = np.array([P[polys[i][0] + c] for i, c in raw])
# an attaching endpoint and its target point are the same node
parent = list(range(len(raw)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    a, b = find(a), find(b)
    if a != b:
        parent[b] = a


ridx = {rc: k for k, rc in enumerate(raw)}
for i, end, q, lj, _ in attach:
    s, e = polys[i]
    union(ridx[(i, 0 if end == 0 else e - s - 1)], ridx[(q, int(lj))])
for a, b in cKDTree(rpos).query_pairs(0.5e-3):
    if psys[raw[a][0]] == psys[raw[b][0]]:  # never fuse arteries and veins running side by side
        union(a, b)
roots = sorted({find(k) for k in range(len(raw))})
node_of = {r: n for n, r in enumerate(roots)}
nid = np.array([node_of[find(k)] for k in range(len(raw))])
npos = np.zeros((len(roots), 3))
cnt = np.zeros(len(roots))
for k in range(len(raw)):
    npos[nid[k]] += rpos[k]
    cnt[nid[k]] += 1
npos /= cnt[:, None]

# ── edges: polyline pieces between consecutive cuts ────────────────────────────────────────────────────────────────
E_a, E_b, E_off, E_name, pts_out, rad_out = [], [], [0], [], [], []
for i, (s, e) in enumerate(polys):
    cs = sorted(cuts[i])
    for c0, c1 in zip(cs[:-1], cs[1:]):
        a, b = nid[ridx[(i, c0)]], nid[ridx[(i, c1)]]
        if a == b and c1 - c0 < 3:
            continue
        seg = P[s + c0 : s + c1 + 1].copy()
        seg[0], seg[-1] = npos[a], npos[b]  # snap to the merged node positions
        pts_out.append(seg)
        rad_out.append(R[s + c0 : s + c1 + 1])
        E_a.append(a)
        E_b.append(b)
        E_name.append(names[V[s]])
        E_off.append(E_off[-1] + len(seg))
E_a, E_b = np.array(E_a), np.array(E_b)
pts_out, rad_out = np.concatenate(pts_out), np.concatenate(rad_out)
ne = len(E_a)
elen = np.array([np.linalg.norm(np.diff(pts_out[E_off[k] : E_off[k + 1]], axis=0), axis=1).sum() for k in range(ne)])
erad = np.array([rad_out[E_off[k] : E_off[k + 1]].mean() for k in range(ne)])

# ── components and classes ─────────────────────────────────────────────────────────────────────────────────────────
cp = list(range(len(npos)))


def cf(a):
    while cp[a] != a:
        cp[a] = cp[cp[a]]
        a = cp[a]
    return a


for a, b in zip(E_a, E_b):
    ra, rb = cf(a), cf(b)
    if ra != rb:
        cp[rb] = ra
comp = np.array([cf(n) for n in range(len(npos))])
ecomp = comp[E_a]
cids, csize = np.unique(ecomp, return_counts=True)
order = np.argsort(-np.array([elen[ecomp == c].sum() for c in cids]))
cids = cids[order]


def classify(members):
    low = " ".join(members).lower()
    sysm = Counter(system_of(m) for m in members if not m.startswith("bridge")).most_common(1)[0][0] if members else (False, False)
    if sysm[1]:
        return "pulmonary_venous" if sysm[0] else "pulmonary_arterial"
    if "aorta" in low or "brachiocephalic trunk" in low:
        return "systemic_arterial"
    if "vena cava" in low:
        return "systemic_venous"
    if "pulmonary trunk" in low or "pulmonary artery" in low:
        return "pulmonary_arterial"
    if "pulmonary vein" in low or "vein of right lung" in low or "vein of left lung" in low:
        return "pulmonary_venous"
    if "coronary" in low or "cardiac vein" in low or "coronary sinus" in low:
        return "coronary"
    if "vein" in low or "venous" in low or "sinus" in low:
        return "venous_other"
    return "arterial_other"


comps = []
for c in cids:
    m = sorted(set(E_name[k] for k in np.flatnonzero(ecomp == c)))
    comps.append({"id": int(c), "class": classify(m), "edges": int((ecomp == c).sum()), "length_m": float(elen[ecomp == c].sum()), "vessels": len(m), "max_radius_mm": float(erad[ecomp == c].max() * 1e3), "members_sample": m[:8]})

# ── bridges: each fragment joins the nearest larger same-system tree from its largest free end (≤ 25 mm) ────────
def comp_system(c):
    ks = np.flatnonzero(ecomp == c["id"])
    return Counter(system_of(E_name[k]) for k in ks).most_common(1)[0][0]


bridges = []
deg0 = np.bincount(np.r_[E_a, E_b], minlength=len(npos))
for _ in range(3):  # a few rounds: fragments may chain through other fragments
    changed = False
    comp_len = {c["id"]: c["length_m"] for c in comps}
    by_sys = defaultdict(list)
    for c in comps:
        by_sys[comp_system(c)].append(c["id"])
    node_comp = comp
    for c in sorted(comps, key=lambda c: c["length_m"]):
        sysc = comp_system(c)
        bigger = [d for d in by_sys[sysc] if comp_len[d] > c["length_m"] and d != c["id"]]
        if not bigger:
            continue
        free = [n for n in np.flatnonzero((comp == c["id"]) & (deg0 == 1))]
        if not free:
            continue
        # radius at a free node: of its edge's end sample
        def rad_at(n):
            k = next(k for k in range(ne) if E_a[k] == n or E_b[k] == n)
            return rad_out[E_off[k]] if E_a[k] == n else rad_out[E_off[k + 1] - 1]
        n0 = max(free, key=rad_at)
        target_pts = np.flatnonzero(np.isin(ecomp[np.searchsorted(E_off, np.arange(len(pts_out)), side="right") - 1], bigger))
        if len(target_pts) == 0:
            continue
        lim_b = 60e-3 if sysc[1] else 25e-3  # lungs: segmental trees sit far from the lobar artery in this atlas
        tnames = np.array([E_name[q_] for q_ in np.searchsorted(E_off, target_pts, side="right") - 1])
        fr_names = " ".join(c["members_sample"])
        pref = None
        if "Intrarenal arteries" in fr_names:
            pref, lim_b = np.char.find(tnames.astype(str), "renal artery") >= 0, 60e-3
        elif "Intrarenal veins" in fr_names:
            pref, lim_b = np.char.find(tnames.astype(str), "renal vein") >= 0, 60e-3
        dist = np.linalg.norm(pts_out[target_pts] - npos[n0], axis=1)
        j = None
        for ok_ in ([pref] if pref is not None else []) + [rad_out[target_pts] >= 1.5 * rad_at(n0), np.ones(len(target_pts), bool)]:
            dd_ = np.where(ok_, dist, np.inf)
            if dd_.min() <= lim_b:
                j = int(np.argmin(dd_))
                break
        if j is None:
            continue
        d = dist[j]
        # nearest node of the target edge (bridge to its closer end node, keeps the graph simple)
        g = target_pts[j]
        k = np.searchsorted(E_off, g, side="right") - 1
        tn = E_a[k] if np.linalg.norm(npos[E_a[k]] - pts_out[g]) < np.linalg.norm(npos[E_b[k]] - pts_out[g]) else E_b[k]
        seg = np.linspace(npos[n0], npos[tn], max(2, int(np.ceil(np.linalg.norm(npos[tn] - npos[n0]) / 0.5e-3)) + 1))
        r = np.full(len(seg), rad_at(n0))
        E_a = np.r_[E_a, n0]
        E_b = np.r_[E_b, tn]
        pts_out = np.concatenate([pts_out, seg])
        rad_out = np.concatenate([rad_out, r])
        E_off.append(E_off[-1] + len(seg))
        E_name.append(f"bridge: {E_name[next(k for k in range(ne) if E_a[k] == n0 or E_b[k] == n0)]}")
        elen = np.r_[elen, np.linalg.norm(npos[tn] - npos[n0])]
        erad = np.r_[erad, r[0]]
        ecomp = np.r_[ecomp, ecomp[k]]
        ne += 1
        bridges.append({"from_node": int(n0), "to_node": int(tn), "length_mm": float(d * 1e3)})
        # merge components
        old = c["id"]
        newc = int(ecomp[k])
        ecomp[ecomp == old] = newc
        comp[comp == old] = newc
        changed = True
        deg0 = np.bincount(np.r_[E_a, E_b], minlength=len(npos))
    comps_new = []
    for cid in np.unique(ecomp):
        m = sorted(set(E_name[k] for k in np.flatnonzero(ecomp == cid)))
        comps_new.append({"id": int(cid), "class": classify(m), "edges": int((ecomp == cid).sum()), "length_m": float(elen[ecomp == cid].sum()), "vessels": len(m), "max_radius_mm": float(erad[ecomp == cid].max() * 1e3), "members_sample": m[:8]})
    comps = sorted(comps_new, key=lambda c: -c["length_m"])
    if not changed:
        break

# ── proximal free-end joins: a free end that is the larger-radius end of its vessel (an origin, not a distal
#    truncation) and ≥ 1.2 mm joins the nearest same-system point of a vessel at least 0.8× as large, splitting that
#    edge there (≤ max(12 mm, 3r); 30 mm in the lungs, whose segmental trees are fragmented in the atlas).
#    Heart-side ends (aortic / pulmonary root, venae cavae, coronary sinus, pulmonary veins) are roots, never joined.
HEART = np.array([0.02, -0.025, 1.29])
ROOTS = {"systemic_arterial": ["Ascending aorta"], "pulmonary_arterial": ["Bifurcation of pulmonary trunk", "Pulmonary trunk"],
         "systemic_venous": ["Superior vena cava", "Inferior vena cava (thoracic part)", "Coronary sinus"],
         "pulmonary_venous": ["Right superior pulmonary vein", "Right inferior pulmonary vein", "Left superior pulmonary vein", "Left inferior pulmonary vein"]}
EP = [pts_out[E_off[k] : E_off[k + 1]].copy() for k in range(ne)]
ER = [rad_out[E_off[k] : E_off[k + 1]].copy() for k in range(ne)]
E_a, E_b, npos = list(E_a), list(E_b), [p for p in npos]


def degree():
    return np.bincount(np.r_[E_a, E_b], minlength=len(npos))


def heart_end(nm):
    """the free end of vessel `nm` closest to the heart"""
    dg = degree()
    cand = [n for k in range(len(E_a)) if E_name[k] == nm for n in (E_a[k], E_b[k]) if dg[n] == 1]
    return min(cand, key=lambda n: np.linalg.norm(npos[n] - HEART)) if cand else None


protected = {n for v in ROOTS.values() for nm in v for n in [heart_end(nm)] if n is not None}
joins = []
dg = degree()
for n in [n for n in range(len(npos)) if dg[n] == 1 and n not in protected]:
    k = next(k for k in range(len(E_a)) if E_a[k] == n or E_b[k] == n)
    rn, ro = (ER[k][0], ER[k][-1]) if E_a[k] == n else (ER[k][-1], ER[k][0])
    if rn < 1.2e-3 or rn < 0.95 * ro:
        continue
    sy = system_of(E_name[k])
    cand = [q for q in range(len(E_a)) if q != k and E_name[q] != E_name[k] and system_of(E_name[q]) == sy and ER[q].max() >= 0.8 * rn]
    if not cand:
        continue
    allp = np.concatenate([EP[q] for q in cand])
    owner = np.concatenate([np.full(len(EP[q]), q) for q in cand])
    local = np.concatenate([np.arange(len(EP[q])) for q in cand])
    rr = np.concatenate([ER[q] for q in cand])
    d0 = np.linalg.norm(allp - npos[n], axis=1)
    d = np.where(rr >= 1.5 * rn, d0, np.inf)  # a parent much larger than the branch (aortic arch for a carotid) first
    j = int(np.argmin(d))
    if d[j] > (30e-3 if sy[1] else max(16e-3, 4 * rn)):
        d = np.where(rr >= 0.8 * rn, d0, np.inf)
        j = int(np.argmin(d))
        if d[j] > (30e-3 if sy[1] else max(12e-3, 3 * rn)):
            continue
    q, lj = int(owner[j]), int(local[j])
    if lj <= 1:
        t = E_a[q]
    elif lj >= len(EP[q]) - 2:
        t = E_b[q]
    else:  # split edge q at lj
        t = len(npos)
        npos.append(EP[q][lj].copy())
        E_a.append(t)
        E_b.append(E_b[q])
        EP.append(EP[q][lj:].copy())
        ER.append(ER[q][lj:].copy())
        E_name.append(E_name[q])
        EP[q], ER[q] = EP[q][: lj + 1], ER[q][: lj + 1]
        E_b[q] = t
    m = max(2, int(np.ceil(np.linalg.norm(npos[t] - npos[n]) / 0.5e-3)) + 1)
    E_a.append(n)
    E_b.append(t)
    EP.append(np.linspace(npos[n], npos[t], m))
    ER.append(np.full(m, rn))
    E_name.append(f"join: {E_name[k]}")
    joins.append({"vessel": E_name[k], "to": E_name[q], "r_mm": round(float(rn * 1e3), 2), "gap_mm": round(float(d[j] * 1e3), 1)})
# ── heart connectors: the atlas centrelines stop short of the heart (pulmonary veins at their lobar junctions, the
#    pulmonary trunk at its bifurcation). Straight connectors join each root vessel's heart-side end to the heart:
#    aorta / pulmonary trunk → the valve's leaflet centroid; venae cavae → nearest right-atrial vertex; each pulmonary
#    vein → nearest left-atrial vertex. The connector's heart end is the flow root of its tree.
MJ = json.loads((E / "meshes.json").read_text())["5__Cardiovascular_system"]["objects"]
MD = np.load(E / "meshes_5__Cardiovascular_system.npz")


def mesh_vertices(rx):
    import re as _re
    vs = [MD["verts"][np.unique(MD["tris"][MD["tri_offsets"][i] : MD["tri_offsets"][i + 1]])] for i, o in enumerate(MJ) if _re.search(rx, o["name"]) and o["tris"] > 0]
    return np.concatenate(vs).astype(np.float64) if vs else None


TARGET = {"av": mesh_vertices(r"^(Left coronary|Right coronary|Non-coronary) leaflet$"), "pv": mesh_vertices(r"leaflet of pulmonary valve"),
          "ra": mesh_vertices(r"^Right atrium$"), "la": mesh_vertices(r"^Left atrium$")}
CONNECT = [("Ascending aorta", "av", "systemic_arterial"), ("Bifurcation of pulmonary trunk", "pv", "pulmonary_arterial"),
           ("Superior vena cava", "ra", "systemic_venous"), ("Inferior vena cava (thoracic part)", "ra", "systemic_venous"),
           ("Right superior pulmonary vein", "la", "pulmonary_venous"), ("Right inferior pulmonary vein", "la", "pulmonary_venous"),
           ("Left superior pulmonary vein", "la", "pulmonary_venous"), ("Left inferior pulmonary vein", "la", "pulmonary_venous")]
heart_roots, connectors = [], []
for nm, tg, cl in CONNECT:
    tv = TARGET[tg]
    ks = [k for k in range(len(E_a)) if E_name[k] == nm]
    if tv is None or not ks:
        print("connector: missing", nm, tg)
        continue
    ref = tv.mean(0)
    ends = [(E_a[k], ER[k][0]) for k in ks] + [(E_b[k], ER[k][-1]) for k in ks]
    n, rn = min(ends, key=lambda e: np.linalg.norm(npos[e[0]] - ref))
    tp = ref if tg in ("av", "pv") else tv[np.argmin(np.linalg.norm(tv - npos[n], axis=1))]
    gap = float(np.linalg.norm(tp - npos[n]))
    if gap < 1.5e-3:
        heart_roots.append(int(n))
        continue
    t = len(npos)
    npos.append(tp.copy())
    m = max(2, int(np.ceil(gap / 0.5e-3)) + 1)
    E_a.append(t)  # heart end first: the connector runs away from the heart
    E_b.append(n)
    if tg in ("av", "pv"):
        # Hermite curve leaving the valve along its axis (normal of the leaflet plane, toward the vessel) and entering
        # the vessel along its own direction: the pulmonary trunk wraps around the aortic root instead of cutting it
        _, _, vt = np.linalg.svd(tv - ref, full_matrices=False)
        nv = vt[2] * np.sign(vt[2] @ (npos[n] - tp))
        kk = next(k for k in ks if E_a[k] == n or E_b[k] == n)
        q_ = EP[kk][min(4, len(EP[kk]) - 1)] - EP[kk][0] if E_a[kk] == n else EP[kk][-1 - min(4, len(EP[kk]) - 1)] - EP[kk][-1]
        lv = q_ / max(np.linalg.norm(q_), 1e-12)  # from the vessel end into the vessel: the curve arrives along it
        L_ = 1.2 * gap
        ss = np.linspace(0, 1, m)[:, None]
        h00, h10, h01, h11 = 2 * ss**3 - 3 * ss**2 + 1, ss**3 - 2 * ss**2 + ss, -2 * ss**3 + 3 * ss**2, ss**3 - ss**2
        cv = h00 * tp + h10 * L_ * nv + h01 * npos[n] + h11 * L_ * lv
        m = max(2, int(np.ceil(np.linalg.norm(np.diff(cv, axis=0), axis=1).sum() / 0.5e-3)) + 1)
        ss = np.linspace(0, 1, m)[:, None]
        h00, h10, h01, h11 = 2 * ss**3 - 3 * ss**2 + 1, ss**3 - 2 * ss**2 + ss, -2 * ss**3 + 3 * ss**2, ss**3 - ss**2
        EP.append(h00 * tp + h10 * L_ * nv + h01 * npos[n] + h11 * L_ * lv)
    else:
        EP.append(np.linspace(tp, npos[n], m))
    ER.append(np.full(len(EP[-1]), rn))
    E_name.append(f"connector: {nm}")
    heart_roots.append(t)
    connectors.append({"vessel": nm, "to": tg, "gap_mm": round(gap * 1e3, 1), "r_mm": round(float(rn * 1e3), 2)})
print("heart connectors:", connectors)

# ── root separation: the atlas places the pulmonary bifurcation and trunk within ~15 mm of the ascending-aortic axis
#    (radii 12–14 mm each). Push vectors from the overlap of the trunk / bifurcation samples with the aorta (target
#    axis distance 0.9 (r_i + r_j)) spread over the pulmonary arterial tree with a Gaussian falloff (σ 20 mm); iterated.
def _sep(moving_names, fixed_names, carry_pred, sigma=20e-3, iters=8):
    mv = [k for k in range(len(E_a)) if E_name[k] in moving_names]
    fx = [k for k in range(len(E_a)) if E_name[k] in fixed_names]
    carry = [k for k in range(len(E_a)) if carry_pred(E_name[k])]
    if not mv or not fx:
        return 0.0
    FP = np.concatenate([EP[k] for k in fx])
    FR = np.concatenate([ER[k] for k in fx])
    ft = cKDTree(FP)
    total = 0.0
    for _ in range(iters):
        C_ = np.concatenate([EP[k] for k in mv])
        CR = np.concatenate([ER[k] for k in mv])
        push = np.zeros_like(C_)
        for i_, p_ in enumerate(C_):
            for j_ in ft.query_ball_point(p_, CR[i_] + FR.max()):
                d_ = np.linalg.norm(p_ - FP[j_])
                ov = 0.9 * (CR[i_] + FR[j_]) - d_
                if ov > 0:
                    v_ = ov * (p_ - FP[j_]) / max(d_, 1e-9)
                    if np.linalg.norm(v_) > np.linalg.norm(push[i_]):
                        push[i_] = v_
        on = np.linalg.norm(push, axis=1) > 0
        if not on.any():
            break
        Cp, Pp = C_[on], push[on]
        step = 0.0
        for k in carry:
            d2 = ((EP[k][:, None, :] - Cp[None]) ** 2).sum(-1)
            w = np.exp(-d2 / (2 * sigma**2))
            u = (w @ Pp) / np.maximum(w.sum(1, keepdims=True), 1e-12) * w.max(1, keepdims=True)
            EP[k] = EP[k] + 0.5 * u
            step = max(step, float(np.linalg.norm(0.5 * u, axis=1).max()))
        total += step
    return total


_pa = {"connector: Bifurcation of pulmonary trunk", "Bifurcation of pulmonary trunk"}
_ao = {"Ascending aorta", "connector: Ascending aorta", "Aortic arch"}
_pa_comp_names = {E_name[k] for k in range(len(E_a)) if "pulmonary" in E_name[k].lower() and "vein" not in E_name[k].lower() or "of left lung" in E_name[k] or "of right lung" in E_name[k] and "vein" not in E_name[k]}
moved_pa = _sep(_pa, _ao, lambda n: n in _pa_comp_names and "vein" not in n.lower())
for k in range(len(E_a)):  # node positions follow their edges' end samples
    npos[E_a[k]] = EP[k][0]
    npos[E_b[k]] = EP[k][-1]
print(f"root separation: pulmonary trunk / bifurcation moved up to ~{moved_pa * 1e3:.1f} mm (cumulative) away from the aorta")

print(f"{len(joins)} proximal free-end joins:")
for jn in joins:
    print("  ", jn)
E_a, E_b, npos = np.array(E_a), np.array(E_b), np.array(npos)
ne = len(E_a)
E_off = list(np.r_[0, np.cumsum([len(x) for x in EP])])
pts_out, rad_out = np.concatenate(EP), np.concatenate(ER)
elen = np.array([np.linalg.norm(np.diff(EP[k], axis=0), axis=1).sum() for k in range(ne)])
erad = np.array([ER[k].mean() for k in range(ne)])
cp = list(range(len(npos)))
for a_, b_ in zip(E_a, E_b):
    ra, rb = cf(a_), cf(b_)
    if ra != rb:
        cp[rb] = ra
comp = np.array([cf(n) for n in range(len(npos))])
ecomp = comp[E_a]
comps = []
for c in np.unique(ecomp):
    m = sorted(set(E_name[k] for k in np.flatnonzero(ecomp == c)))
    comps.append({"id": int(c), "class": classify(m), "edges": int((ecomp == c).sum()), "length_m": float(elen[ecomp == c].sum()), "vessels": len(m), "max_radius_mm": float(erad[ecomp == c].max() * 1e3), "members_sample": m[:8]})
comps = sorted(comps, key=lambda c: -c["length_m"])

direction = None

# ── orientation from the heart: root nodes = largest-radius end of the component's defining vessel ────────────────
adj = defaultdict(list)
for k, (a, b) in enumerate(zip(E_a, E_b)):
    adj[a].append((b, k))
    adj[b].append((a, k))
direction = np.zeros(ne, np.int8)  # +1: a→b away from the heart (downstream for arteries, upstream for veins), −1: b→a
comp_root = {}
for c in comps:
    ks = np.flatnonzero(ecomp == c["id"])
    rs = [r for r in heart_roots if comp[r] == c["id"]]
    if not rs:  # fragment / other: the largest-radius end of its widest edge
        k0 = max(ks, key=lambda k: erad[k])
        rs = [int(E_a[k0] if rad_out[E_off[k0]] >= rad_out[E_off[k0 + 1] - 1] else E_b[k0])]
    root = rs[0]
    comp_root[c["id"]] = root
    seen = set(rs)
    dq = deque(rs)
    while dq:
        n = dq.popleft()
        for m, k in adj[n]:
            if direction[k] == 0:
                direction[k] = 1 if E_a[k] == n else -1
            if m not in seen:
                seen.add(m)
                dq.append(m)
    c["roots"] = rs
    c["root_node"] = int(root)
    c["root_pos_m"] = npos[root].round(4).tolist()

deg = np.bincount(np.r_[E_a, E_b], minlength=len(npos))
diag = {
    "polylines": len(polys), "attachments": len(attach), "nodes": int(len(npos)), "edges": int(ne), "components": len(comps),
    "terminals (degree 1)": int((deg == 1).sum()), "junctions (degree ≥ 3)": int((deg >= 3).sum()),
    "length_m": float(elen.sum()),
    "length_in_largest_5_components": [round(c["length_m"], 2) for c in comps[:5]],
    "joins": len(joins), "bridges": len(bridges), "bridge_length_mm_p50_max": [float(np.median([b["length_mm"] for b in bridges])) if bridges else 0.0, float(max([b["length_mm"] for b in bridges], default=0.0))],
    "classes_by_length_m": {k: round(sum(c["length_m"] for c in comps if c["class"] == k), 2) for k in sorted({c["class"] for c in comps})},
}
# ── de-overlap: shrink radii where non-adjacent lumens intersect (deoverlap.py) ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deoverlap import deoverlap  # noqa: E402

venous_s = np.array([system_of(E_name[k].split(": ", 1)[-1])[0] for k in range(ne) for _ in range(E_off[k + 1] - E_off[k])])
r_before = rad_out.copy()
rad_out, dor = deoverlap(pts_out, rad_out, np.asarray(E_off), E_a, E_b, npos, lambda r: np.maximum(np.where(venous_s, 0.06, 0.12) * r, 0.1e-3), floor=0.85)
chg = r_before - rad_out
shr = {}
for k in range(ne):
    a, b = E_off[k], E_off[k + 1]
    if chg[a:b].max() > 1e-5:
        nm = E_name[k].split(": ", 1)[-1]
        shr[nm] = max(shr.get(nm, 0.0), float((chg[a:b] / r_before[a:b]).max()))
dor["vessels"] = {k: round(v, 3) for k, v in sorted(shr.items(), key=lambda kv: -kv[1])}
diag["deoverlap"] = {k: v for k, v in dor.items() if k != "vessels"}
print("de-overlap:", diag["deoverlap"], "largest shrinks:", list(dor["vessels"].items())[:12])
np.savez_compressed(E / "graph.npz", node_pos=npos.astype(np.float32), edge_a=E_a.astype(np.int32), edge_b=E_b.astype(np.int32), edge_offsets=np.array(E_off, np.int64), points=pts_out.astype(np.float32), radius=rad_out.astype(np.float32), direction=direction, edge_comp=ecomp.astype(np.int32))
(E / "graph.json").write_text(json.dumps({"diagnostics": diag, "components": comps, "edge_names": E_name, "joins": joins, "connectors": connectors, "radius_corrections": rfix_applied, "deoverlap_max_shrink": dor["vessels"], "source": "Z-Anatomy (CC BY-SA 4.0)"}, indent=1))
print(json.dumps(diag, indent=1))
for c in comps[:14]:
    print(f"{c['class']:20s} edges {c['edges']:5d} length {c['length_m']:6.2f} m  vessels {c['vessels']:4d}  rmax {c['max_radius_mm']:5.1f} mm  e.g. {c['members_sample'][:3]}")

# ── figure: components coloured by class ───────────────────────────────────────────────────────────────────────────
col = {"systemic_arterial": (0.85, 0.1, 0.1), "systemic_venous": (0.15, 0.35, 0.9), "pulmonary_arterial": (0.6, 0.1, 0.8), "pulmonary_venous": (0.1, 0.7, 0.7), "coronary": (1.0, 0.6, 0.0), "venous_other": (0.5, 0.6, 0.95), "arterial_other": (0.95, 0.55, 0.55)}
cls_of = {c["id"]: c["class"] for c in comps}
fig, ax = plt.subplots(1, 2, figsize=(16, 13), gridspec_kw=dict(width_ratios=[1, 1]))
for a, (lim, t) in zip(ax, [(None, "whole body"), (((-0.16, 0.16), (1.05, 1.5)), "thorax")]):
    segs, cs, lw = [], [], []
    for k in range(ne):
        q = pts_out[E_off[k] : E_off[k + 1]][:, [0, 2]]
        segs.append(q)
        cs.append(col[cls_of[ecomp[k]]])
        lw.append(np.clip(erad[k] * 1e3 * 0.8, 0.15, 6))
    a.add_collection(LineCollection(segs, colors=cs, linewidths=lw))
    term = np.flatnonzero(deg == 1)
    a.plot(npos[term, 0], npos[term, 2], "k.", ms=1.2)
    for c in comps[:6]:
        a.plot(*npos[c["root_node"]][[0, 2]], "k*", ms=12)
    a.autoscale()
    if lim:
        a.set_xlim(*lim[0])
        a.set_ylim(*lim[1])
    a.set_aspect("equal")
    a.set_title(t)
handles = [plt.Line2D([], [], color=c, lw=3, label=k) for k, c in col.items()]
ax[0].legend(handles=handles, fontsize=8, loc="lower right")
fig.suptitle(f"Vessel graph: {len(comps)} components, {ne} edges, terminals (dots), roots (stars)")
(E / "figs").mkdir(exist_ok=True)
fig.savefig(E / "figs/graph.png", dpi=75, bbox_inches="tight")
