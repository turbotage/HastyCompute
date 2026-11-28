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
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

E = Path(sys.argv[1])
# --out DIR: write graph.npz / graph.json / figs there instead of the export dir (tests); --no-separate: skip the
# centreline separation of interpenetrating components (the graph as before 2026-11-25)
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else E
SEPARATE = "--no-separate" not in sys.argv
OUT.mkdir(parents=True, exist_ok=True)
v = load_any(E / "vessels.npz")
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
# organ roots (2026-11-28): a venous tree that drains into an organ's capillary bed, not into the heart. The hepatic
# portal vein ends at the porta hepatis; its blood reaches the IVC only through the liver sinusoids and the hepatic veins.
# Its free organ end is the tree's root (class portal_venous; network_flow: a sink at the sinusoidal pressure, the
# hepatic veins carry the liver's outflow): never bridged or joined to another vessel. Before, the bridge ran from the
# porta hepatis 27 mm back down into the IVC and junction relocation turned it into an elbow straight into the IVC — a
# portosystemic shunt carrying 20 % of the cardiac output, a hairpin in the viewer. `--portal-shunt`: the old behaviour.
ORGAN_ROOTS = {} if "--portal-shunt" in sys.argv else {"Hepatic portal vein": "portal_venous"}
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
# toward a larger vessel the tolerance is measured to its lumen wall (2026-11-26, `--attach-axis`: to its axis, as
# before): the maxillary artery's pterygoid-canal / buccal / descending-palatine branches start 0.7–1 mm outside its
# wall (2.3–2.65 mm from its axis) and were bridged to each other and 131 mm away instead
ATTACH_WALL = "--attach-axis" not in sys.argv
RMAXP = float(R.max())
for i, (s, e) in enumerate(polys):
    if names[V[s]] in ORGAN_ROOTS:  # organ-rooted vessels (portal vein) attach nothing: tributaries attach to them
        continue
    for end, gi in ((0, s), (1, e - 1)):
        tol = max(1.5e-3, 2.0 * R[gi])
        dd, jj = tree.query(P[gi], k=96, distance_upper_bound=tol + (RMAXP if ATTACH_WALL else 0.0))
        jc = np.minimum(jj, len(P) - 1)
        ok = (jj < len(P)) & (pid[jc] != i) & (pid[jc] >= 0)
        ok &= np.array([psys[pid[j]] == psys[i] if pid[j] >= 0 else False for j in jc])
        if ATTACH_WALL:
            # axis rule, or the wall rule toward a vessel at least as large
            wall = dd - np.where(R[jc] >= R[gi], R[jc], 0.0)
            ok &= (dd <= tol) | (wall <= tol)
            if not ok.any():
                continue
            j = jj[ok][int(np.argmin(np.where(R[jc[ok]] >= R[gi], wall[ok], dd[ok])))]
            dd_j = float(dd[jj == j][0])
        else:
            if not ok.any():
                continue
            j = jj[ok][0]
            dd_j = float(dd[ok][0])
        q = pid[j]
        lj = j - polys[q][0]
        attach.append((i, end, q, lj, dd_j))
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
# a node holding an attachment's target point sits on the target's axis (its target points only): the attaching end
# moves to the parent, not the parent's axis halfway to the end (2026-11-26; before: the mean of all merged points)
tgt_raw = {ridx[(q, int(lj))] for _, _, q, lj, _ in attach} if "--attach-axis" not in sys.argv else set()
has_tgt = np.zeros(len(roots), bool)
for k in tgt_raw:
    has_tgt[nid[k]] = True
for k in range(len(raw)):
    if has_tgt[nid[k]] and k not in tgt_raw:
        continue
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
    for m_ in members:
        if m_ in ORGAN_ROOTS:
            return ORGAN_ROOTS[m_]
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
    if "coronary sinus" in low:  # drains into the right atrium: a systemic venous tree with its own root (2026-11-26)
        return "systemic_venous"
    if "coronary" in low or "cardiac vein" in low:
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
_ROOT_NAMES = {"Ascending aorta", "Bifurcation of pulmonary trunk", "Pulmonary trunk", "Superior vena cava", "Inferior vena cava (thoracic part)", "Coronary sinus",
               "Right superior pulmonary vein", "Right inferior pulmonary vein", "Left superior pulmonary vein", "Left inferior pulmonary vein"}  # = ROOTS below
_HEART0 = np.array([0.02, -0.025, 1.29])


def _is_root_heart_end(n):
    """n is the heart-side free end of a root vessel (the end of that vessel's edges closest to the heart)"""
    ks = [k for k in range(ne) if E_a[k] == n or E_b[k] == n]
    if len(ks) != 1 or E_name[ks[0]] not in _ROOT_NAMES:
        return False
    nm = E_name[ks[0]]
    ends = [m for k in range(ne) if E_name[k] == nm for m in (E_a[k], E_b[k]) if np.sum((E_a == m) | (E_b == m)) == 1]
    return bool(ends) and n == min(ends, key=lambda m: np.linalg.norm(npos[m] - _HEART0))


def _split_flat(g):
    """the node at sample g of the flat edge arrays: an end node of its edge when g is within one sample of it, else
    the edge is split there (it keeps the part up to g, the rest becomes a new last edge)"""
    global E_a, E_b, E_off, pts_out, rad_out, elen, erad, ecomp, comp, npos, ne
    k = int(np.searchsorted(E_off, g, side="right") - 1)
    a, b = E_off[k], E_off[k + 1]
    if g - a <= 1:
        return int(E_a[k])
    if b - 1 - g <= 1:
        return int(E_b[k])
    t = len(npos)
    npos = np.vstack([npos, pts_out[g]])
    comp = np.r_[comp, comp[E_a[k]]]
    p2, r2 = pts_out[g:b].copy(), rad_out[g:b].copy()
    cut = b - (g + 1)
    pts_out = np.concatenate([pts_out[: g + 1], pts_out[b:], p2])
    rad_out = np.concatenate([rad_out[: g + 1], rad_out[b:], r2])
    E_off = E_off[: k + 1] + [g + 1] + [o - cut for o in E_off[k + 2 :]]
    E_off.append(E_off[-1] + len(p2))
    E_a = np.r_[E_a, t]
    E_b = np.r_[E_b, E_b[k]]
    E_b[k] = t
    E_name.append(E_name[k])
    l1 = float(np.linalg.norm(np.diff(pts_out[a : g + 1], axis=0), axis=1).sum())
    elen = np.r_[elen, max(float(elen[k]) - l1, 0.0)]
    elen[k] = l1
    erad = np.r_[erad, erad[k]]
    ecomp = np.r_[ecomp, ecomp[k]]
    ne += 1
    return t
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
        if c["class"] in ORGAN_ROOTS.values():  # an organ-rooted tree (portal system): its root is its organ end
            continue
        free = [n for n in np.flatnonzero((comp == c["id"]) & (deg0 == 1))]
        # a root vessel's heart end is the fragment's own root, never a bridge source (2026-11-26: the coronary sinus
        # was bridged by its atrial end 22 mm through the right atrium to the IVC root; the LBM's walls pinched it)
        if "--bridge-heart-ends" not in sys.argv:
            free = [n for n in free if not _is_root_heart_end(n)]

        # radius at a node: of an incident edge's end sample
        def rad_at(n):
            k = next(k for k in range(ne) if E_a[k] == n or E_b[k] == n)
            return rad_out[E_off[k]] if E_a[k] == n else rad_out[E_off[k + 1] - 1]
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
        j = None
        why = "has no free end"
        if free:
            n0 = max(free, key=rad_at)
            dist = np.linalg.norm(pts_out[target_pts] - npos[n0], axis=1)
            # ranked by the distance to the target's lumen wall (2026-11-25; the axis distance preferred a nearby
            # small vessel over the aorta's wall 8 mm closer: posterior intercostal arteries bridged to each other)
            wdist = dist - (rad_out[target_pts] if "--bridge-axis" not in sys.argv else 0.0)
            if "--bridge-nearest" not in sys.argv:
                # first choice (2026-11-25): the vessel continued along its own direction — a ray from the free end along
                # its outward tangent (over the last 3 mm) joins the first lumen ≥ 1.5× its own it enters within the
                # limit (the right coronary artery's free end, 11 mm from the aorta, points at the aortic sinus; the
                # nearest wall joined it 37 mm up the aorta, outside the root's zone)
                kf = next(k for k in range(ne) if E_a[k] == n0 or E_b[k] == n0)
                qf = pts_out[E_off[kf] : E_off[kf + 1]]
                qf = qf if E_b[kf] == n0 else qf[::-1]  # ends at n0
                arcf = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(qf[::-1], axis=0), axis=1))]
                back = qf[::-1][min(int(np.searchsorted(arcf, 3e-3)), len(qf) - 1)]
                u_ = npos[n0] - back
                if np.linalg.norm(u_) > 1e-6:
                    u_ = u_ / np.linalg.norm(u_)
                    big_ = (rad_out[target_pts] >= 1.5 * rad_at(n0)) & (dist <= lim_b + rad_out[target_pts])
                    if pref is not None:
                        big_ &= pref
                    w_ = pts_out[target_pts] - npos[n0]
                    tr = w_ @ u_
                    dr = np.linalg.norm(w_ - tr[:, None] * u_, axis=1)
                    hit = big_ & (tr > 0) & (dr < rad_out[target_pts])
                    if hit.any():
                        ent = np.where(hit, tr - np.sqrt(np.maximum(rad_out[target_pts] ** 2 - dr**2, 0.0)), np.inf)
                        if ent.min() <= lim_b:
                            j = int(np.argmin(ent))
            for ok_ in ([pref] if pref is not None else []) + [rad_out[target_pts] >= 1.5 * rad_at(n0), np.ones(len(target_pts), bool)]:
                if j is not None:
                    break
                dd_ = np.where(ok_, wdist, np.inf)
                if dd_.min() <= lim_b:
                    j = int(np.argmin(dd_))
                    break
            why = f"nearest larger tree {dist.min() * 1e3:.0f} mm from its largest free end > {lim_b * 1e3:.0f} mm"
        if j is None:
            # fallback (2026-11-25): a fragment node inside a larger same-system lumen joins there, whatever its degree
            # (the atlas draws the left upper-lobe segmental arteries' origin junction on the left pulmonary artery's
            # axis, 61 mm from their largest free end). Only when the free-end rule fails, so a fragment merely
            # crossing a lumen elsewhere keeps its free-end bridge.
            fnodes = np.flatnonzero(comp == c["id"])
            fnodes = fnodes[np.isin(fnodes, np.r_[E_a, E_b])]
            if len(fnodes):
                dd_c, jj_c = cKDTree(pts_out[target_pts]).query(npos[fnodes])
                rel = dd_c / rad_out[target_pts[jj_c]]
                if rel.min() <= 1.0:
                    q_ = int(np.argmin(rel))
                    n0 = int(fnodes[q_])
                    dist = np.linalg.norm(pts_out[target_pts] - npos[n0], axis=1)
                    j = int(jj_c[q_])
                    print(f"bridge: fragment {c['id']} ({c['length_m']:.2f} m, {c['members_sample'][:2]}) {why}: joined at a node inside a larger lumen ({dd_c[q_] * 1e3:.1f} mm from its axis)")
        if j is None:
            if c["length_m"] > 0.05:
                print(f"bridge: fragment {c['id']} ({c['length_m']:.2f} m, {c['members_sample'][:2]}) {why}: not bridged")
            continue
        d = dist[j]
        g = target_pts[j]
        if "--bridge-end-node" in sys.argv:
            # old rule: the target edge's closer end node (the bridge then ran inside the target's lumen to it: the
            # posterior intercostal bridges along the thoracic aorta, 114 mm along the graph from where they enter)
            k = np.searchsorted(E_off, g, side="right") - 1
            tn = E_a[k] if np.linalg.norm(npos[E_a[k]] - pts_out[g]) < np.linalg.norm(npos[E_b[k]] - pts_out[g]) else E_b[k]
        else:
            # the first lumen at least 1.5× the fragment's (or the target's own) that the straight bridge enters: its
            # axis sample there; the target edge is split at it
            v_ = pts_out[g] - npos[n0]
            L_ = max(float(np.linalg.norm(v_)), 1e-9)
            big = target_pts[(rad_out[target_pts] >= 1.5 * rad_at(n0)) & (dist <= L_ + rad_out[target_pts])]
            if len(big):
                w_ = pts_out[big] - npos[n0]
                t_ = np.clip(w_ @ v_ / L_**2, 0.0, 1.0)
                dseg = np.linalg.norm(w_ - t_[:, None] * v_, axis=1)
                inn = dseg < rad_out[big]
                if inn.any():
                    entry = t_ - np.sqrt(np.maximum(rad_out[big] ** 2 - dseg**2, 0.0)) / L_
                    entry = np.where(inn, entry, np.inf)
                    g2 = int(big[np.argmin(entry)])
                    if np.min(entry) < 1.0 - 1e-9:
                        g = g2
                        d = float(np.linalg.norm(pts_out[g] - npos[n0]))
            k = int(np.searchsorted(E_off, g, side="right") - 1)  # keeps its index (and component) when split
            tn = _split_flat(int(g))
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


def organ_root_nodes():
    """the free (degree-1) ends of the ORGAN_ROOTS vessels: their trees' roots (never joined, never bridged)"""
    dg_ = degree()
    out = []
    for nm in ORGAN_ROOTS:
        ends = [(n, ER[k][0] if E_a[k] == n else ER[k][-1]) for k in range(len(E_a)) if E_name[k] == nm for n in (E_a[k], E_b[k]) if dg_[n] == 1]
        if ends:
            out.append(int(max(ends, key=lambda e: e[1])[0]))
    return out


protected |= set(organ_root_nodes())
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
MD = load_any(E / "meshes_5__Cardiovascular_system.npz")


def mesh_vertices(rx):
    import re as _re
    vs = [MD["verts"][np.unique(MD["tris"][MD["tri_offsets"][i] : MD["tri_offsets"][i + 1]])] for i, o in enumerate(MJ) if _re.search(rx, o["name"]) and o["tris"] > 0]
    return np.concatenate(vs).astype(np.float64) if vs else None


TARGET = {"av": mesh_vertices(r"^(Left coronary|Right coronary|Non-coronary) leaflet$"), "pv": mesh_vertices(r"leaflet of pulmonary valve"),
          "ra": mesh_vertices(r"^Right atrium$"), "la": mesh_vertices(r"^Left atrium$")}
CONNECT = [("Ascending aorta", "av", "systemic_arterial"), ("Bifurcation of pulmonary trunk", "pv", "pulmonary_arterial"),
           ("Superior vena cava", "ra", "systemic_venous"), ("Inferior vena cava (thoracic part)", "ra", "systemic_venous"),
           ("Coronary sinus", "ra", "systemic_venous"),  # its own root (2026-11-26; was bridged through the RA to the IVC root)
           ("Right superior pulmonary vein", "la", "pulmonary_venous"), ("Right inferior pulmonary vein", "la", "pulmonary_venous"),
           ("Left superior pulmonary vein", "la", "pulmonary_venous"), ("Left inferior pulmonary vein", "la", "pulmonary_venous")]
heart_roots, connectors = [], []
# venous roots already inside their chamber (2026-11-28): the atlas draws the venae cavae and pulmonary veins a few mm
# into the atrium. The connector then ran from that end *outward* to the nearest atrial-wall vertex (SVC: 9 mm sideways
# at 103° — a ball on the atrium; pulmonary veins: out into the myocardium). Now the vessel is trimmed where its axis
# enters the chamber blood (atlas labels, --chamber-labels, default data/hasty/phantom/p0/tissue_1mm.hv) and its root
# sits one sample inside; no connector. `--venous-connectors`: the old rule.
_CHL = None
_chl_path = Path(sys.argv[sys.argv.index("--chamber-labels") + 1]) if "--chamber-labels" in sys.argv else Path(__file__).resolve().parents[3] / "data/hasty/phantom/p0/tissue_1mm.hv"
if "--venous-connectors" not in sys.argv and exists_any(_chl_path):
    _cl = load_any(_chl_path)
    _CHL = dict(lab=np.asarray(_cl["labels"]), names=list(_cl["tissues"]), o=np.asarray(_cl["origin_m"], float), h=float(_cl["voxel_m"]))


def chamber_at(q, chamber):
    """True where the atlas label at points q [n, 3] is the chamber's blood"""
    i = np.floor((q - _CHL["o"]) / _CHL["h"]).astype(int)
    sh = _CHL["lab"].shape  # [z, y, x]
    ok = (i >= 0).all(1) & (i[:, 0] < sh[2]) & (i[:, 1] < sh[1]) & (i[:, 2] < sh[0])
    out = np.zeros(len(q), bool)
    out[ok] = _CHL["lab"][i[ok, 2], i[ok, 1], i[ok, 0]] == _CHL["names"].index(chamber)
    return out


trimmed_roots = []
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
    if _CHL is not None and tg in ("ra", "la"):
        ch_ = {"ra": "blood_RA", "la": "blood_LA"}[tg]
        k_ = next(k for k in ks if E_a[k] == n or E_b[k] == n)
        q_ = EP[k_] if E_a[k_] == n else EP[k_][::-1]  # from the heart end into the vessel
        ins = chamber_at(q_, ch_)
        n_inc = sum(1 for k in range(len(E_a)) if E_a[k] == n or E_b[k] == n)
        if ins[0] and n_inc > 1:
            # several edges meet at the end node (pulmonary-vein branches): trimming one and moving the node left the
            # others ending at the old place (review 2026-11-28) — keep the end as the root, no trim, no connector
            trimmed_roots.append({"vessel": nm, "trimmed_mm": 0.0, "edges_at_root": n_inc, "connector_was_mm": round(gap * 1e3, 1)})
            heart_roots.append(int(n))
            continue
        if ins[0]:
            j_ = int(np.argmin(ins)) if not ins.all() else len(ins) - 1  # first sample outside the chamber
            j_ = min(max(j_ - 1, 0), len(q_) - 3)  # the root: one sample inside; ≥ 3 samples stay
            if j_ > 0:
                if E_a[k_] == n:
                    EP[k_], ER[k_] = EP[k_][j_:], ER[k_][j_:]
                else:
                    EP[k_], ER[k_] = EP[k_][: len(EP[k_]) - j_], ER[k_][: len(ER[k_]) - j_]
                npos[n] = q_[j_].copy()
            trimmed_roots.append({"vessel": nm, "trimmed_mm": round(float(np.linalg.norm(np.diff(q_[: j_ + 1], axis=0), axis=1).sum() * 1e3), 1), "connector_was_mm": round(gap * 1e3, 1)})
            heart_roots.append(int(n))
            continue
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
print("venous roots trimmed at their chamber entry (no connector):", trimmed_roots)

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

# ── junction relocation (2026-11-25, research/flow_phantom.md Tier B5): the atlas draws many branches running inside
#    their parent's lumen well past the point where the graph joins them (right PA ↔ superior lobar artery 11.7 mm,
#    left PA ↔ segmental arteries ~10 mm, iliac branches; bridges ending at the far node of their target edge). At every
#    junction an incident edge is a child of the larger incident edges: walking from the junction, its axis stays inside
#    a parent lumen up to the exit sample; if the exit lies further along the parent than r_parent + r_child from the
#    junction (co-running, not a branch leaving), the junction moves to the parent sample nearest the exit (splitting
#    that edge) and the child's inside part becomes a straight segment from there to the exit. Repeated until nothing
#    moves (≤ 6 rounds). `--no-relocate`: off.
def _n_trees(tag):
    """connected components over the live edges (E_a ≥ 0), counting only nodes with an edge; logged per stage — a
    stage that splits a tree is a bug (2026-11-26: the right inferior basal vein lost its junction)"""
    lab_ = {}

    def f_(x):
        while lab_.setdefault(x, x) != x:
            lab_[x] = lab_.setdefault(lab_[x], lab_[x])
            x = lab_[x]
        return x
    for a_, b_ in zip(E_a, E_b):
        if a_ < 0:
            continue
        ra_, rb_ = f_(int(a_)), f_(int(b_))
        if ra_ != rb_:
            lab_[rb_] = ra_
    n_ = len({f_(x) for x in list(lab_)})
    STAGES.append((tag, n_))
    if len(STAGES) > 1 and n_ > STAGES[-2][1]:
        print(f"WARNING graph stage '{tag}' split a tree: {STAGES[-2][1]} → {n_} components", flush=True)
        if "--allow-split" not in sys.argv:
            raise SystemExit(f"graph stage '{tag}' disconnected a tree ({STAGES[-2][1]} → {n_} components); --allow-split to continue")
    return n_


STAGES = []
_n_trees("bridges and joins")


def _reloc_check(desc):
    """after each relocation: a split names the step (debugging aid; the stage check gates)"""
    if "--reloc-debug" in sys.argv:
        n0 = STAGES[-1][1]
        STAGES.append(("_", n0))
        n1 = _n_trees(desc)
        STAGES.pop()
        STAGES.pop()
        if n1 > n0:
            print(f"   relocation split: {desc}", flush=True)
            STAGES.append(("_", n1))
relocated, dropped = [], []
if "--no-relocate" not in sys.argv:

    def _components():
        lab_ = list(range(len(npos)))

        def f_(x):
            while lab_[x] != x:
                lab_[x] = lab_[lab_[x]]
                x = lab_[x]
            return x
        for a_, b_ in zip(E_a, E_b):
            ra_, rb_ = f_(a_), f_(b_)
            if ra_ != rb_:
                lab_[rb_] = ra_
        return [f_(x) for x in range(len(npos))]

    heart_set = set(heart_roots)
    # two passes: "inside" (the child's axis within the parent's lumen; moved when the exit lies > r_p + r_c along the
    # parent) and "alongside" (the child's lumen within the parent's lumen + both walls + gap, 1.5 mm for r ≥ 2.5 mm:
    # hugging the parent's wall past the junction — aortic intercostal / lumbar / renal origins; moved when the exit
    # lies > 2 (r_p + r_c) along the parent, so a branch leaving at a normal angle stays)
    passes = [("inside", 8)] + ([("alongside", 8)] if "--no-alongside" not in sys.argv else []) + ([("hook", 4)] if "--no-hook" not in sys.argv else [])
    for _pass, _rounds in passes:
      for _round in range(_rounds):
        moved_any = False
        alive = [k for k in range(len(E_a)) if E_a[k] >= 0]
        inc = defaultdict(list)
        for k in alive:
            inc[E_a[k]].append(k)
            inc[E_b[k]].append(k)
        ncomp = _components()
        allP = np.concatenate([EP[k] for k in alive])
        allR = np.concatenate([ER[k] for k in alive])
        allE = np.concatenate([np.full(len(EP[k]), k) for k in alive])
        allL = np.concatenate([np.arange(len(EP[k])) for k in alive])
        allC = np.array([ncomp[E_a[k]] for k in allE])
        tree_all = cKDTree(allP)
        touched = set()
        for n in list(inc):
            ks = inc[n]
            if len(ks) < 3 or n in touched:
                continue
            for c in ks:
                if E_a[c] < 0 or not (E_a[c] == n or E_b[c] == n) or c in touched:
                    continue
                # the child's chain from n: through degree-2 nodes, ≤ 80 mm
                chain, pts_, rads_, node_, e_ = [], [], [], n, c
                while e_ is not None and len(pts_) < 400:
                    fw = E_a[e_] == node_
                    q_ = EP[e_] if fw else EP[e_][::-1]
                    r_ = ER[e_] if fw else ER[e_][::-1]
                    chain.append((e_, len(pts_) - 1 if pts_ else 0, fw))  # global index of this edge's first sample (shared node)
                    pts_.extend(q_ if not pts_ else q_[1:])
                    rads_.extend(r_ if not rads_ else r_[1:])
                    node_ = E_b[e_] if fw else E_a[e_]
                    nxt = [k for k in inc[node_] if k != e_ and E_a[k] >= 0]
                    e_ = nxt[0] if len(nxt) == 1 and len(inc[node_]) == 2 and nxt[0] not in [x[0] for x in chain] else None
                    if np.linalg.norm(np.diff(np.asarray(pts_), axis=0), axis=1).sum() > 80e-3:
                        break
                cs, cr = np.asarray(pts_), np.asarray(rads_)
                rc = float(cr[0])
                chain_set = {x[0] for x in chain}
                if len(cs) < 3 or chain_set & touched or node_ in heart_set or any(E_name[x].startswith("connector: ") for x in chain_set):
                    continue  # changed this round, or a heart connection: never moved or dropped
                # parents: same component, radius ≥ the child's, not on the chain
                dd, jj = tree_all.query(cs, k=24, distance_upper_bound=3 * float(allR.max()))
                ok_ = jj < len(allP)
                jc = np.minimum(jj, len(allP) - 1)
                ok_ &= (allR[jc] >= rc) & (allC[jc] == ncomp[n]) & ~np.isin(allE[jc], list(chain_set))
                dmin = np.where(ok_, dd, np.inf)
                kbest = np.argmin(dmin, axis=1)
                dpar = dmin[np.arange(len(cs)), kbest]
                jpar = jc[np.arange(len(cs)), kbest]
                if _pass in ("inside", "hook"):
                    inside = dpar < np.where(np.isfinite(dpar), allR[jpar], 0)
                else:
                    gap_ = np.where((cr >= 2.5e-3) & (allR[jpar] >= 2.5e-3), 1.5e-3, 0.12 * (cr + allR[jpar]))
                    inside = dpar < np.where(np.isfinite(dpar), allR[jpar] + cr + gap_, 0)
                if _pass == "hook":
                    # a hook (2026-11-26): the branch leaves the junction, comes back and crosses its parent's lumen within
                    # 3 (R + r) + 5 mm (the left inferior epigastric artery: out 6.7 mm, then through the external iliac
                    # 13 mm along) — it really leaves the parent where it last crosses it; the hook goes
                    if not inside[0]:
                        continue
                    arc_c = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cs, axis=0), axis=1))]
                    win = arc_c <= 3 * (float(allR[jpar[0]]) + rc) + 5e-3
                    idx_in = np.flatnonzero(inside & win)
                    if len(idx_in) < 2:
                        continue
                    kl = int(idx_in[-1])
                    if kl < 2 or kl + 1 >= len(cs) or inside[: kl + 1].all():
                        continue
                    kx = kl + 1
                else:
                    if not inside[0] or not inside[1]:
                        continue
                    kx = int(np.argmin(inside)) if not inside.all() else -1
                end_node = node_
                if kx == -1:
                    # the whole chain lies inside parents: a duplicate drawn inside another vessel, dropped when it ends
                    # free (its far node a terminal); a chain reaching another junction is left to that junction
                    if len(inc[end_node]) == 1 and _pass == "inside":
                        for e2, _, _ in chain:
                            dropped.append(E_name[e2])
                            E_a[e2] = E_b[e2] = -1
                            touched.add(e2)
                        _reloc_check(f"drop {E_name[c]}")
                        moved_any = True
                        break
                    # a chain inside its parent between two junctions (segmental arteries whose junction was drawn
                    # inside the left pulmonary artery, joined to it 27 mm away): the chain is removed and its far
                    # junction moves onto the parent at its nearest point, its other edges extended to it
                    if _pass == "inside" and len(inc[end_node]) >= 3 and "--no-collapse" not in sys.argv:
                        jf = jpar[-1]
                        q, lj = int(allE[jf]), int(allL[jf])
                        if q in chain_set or q in touched or E_a[q] < 0:
                            continue
                        # n must reach the parent without the chain, else removing it would split the tree
                        seen_, todo_ = {n}, [n]
                        while todo_:
                            x_ = todo_.pop()
                            for k_ in inc[x_]:
                                if k_ in chain_set or E_a[k_] < 0:
                                    continue
                                for y_ in (E_a[k_], E_b[k_]):
                                    if y_ not in seen_:
                                        seen_.add(y_)
                                        todo_.append(y_)
                        if E_a[q] not in seen_ and E_b[q] not in seen_:
                            continue
                        if lj <= 1:
                            t = E_a[q]
                        elif lj >= len(EP[q]) - 2:
                            t = E_b[q]
                        else:
                            t = len(npos)
                            npos.append(EP[q][lj].copy())
                            E_a.append(t)
                            E_b.append(E_b[q])
                            EP.append(EP[q][lj:].copy())
                            ER.append(ER[q][lj:].copy())
                            E_name.append(E_name[q])
                            EP[q], ER[q] = EP[q][: lj + 1], ER[q][: lj + 1]
                            E_b[q] = t
                        if t == end_node:
                            continue
                        for e2, _, _ in chain:
                            E_a[e2] = E_b[e2] = -1
                        for k_ in inc[end_node]:
                            if k_ in chain_set or E_a[k_] < 0:
                                continue
                            if E_a[k_] == end_node:
                                m_ = max(2, int(np.ceil(np.linalg.norm(EP[k_][0] - npos[t]) / 0.5e-3)) + 1)
                                EP[k_] = np.vstack([np.linspace(npos[t], EP[k_][0], m_)[:-1], EP[k_]])
                                ER[k_] = np.r_[np.full(m_ - 1, ER[k_][0]), ER[k_]]
                                E_a[k_] = t
                            if E_b[k_] == end_node:
                                m_ = max(2, int(np.ceil(np.linalg.norm(EP[k_][-1] - npos[t]) / 0.5e-3)) + 1)
                                EP[k_] = np.vstack([EP[k_], np.linspace(EP[k_][-1], npos[t], m_)[1:]])
                                ER[k_] = np.r_[ER[k_], np.full(m_ - 1, ER[k_][-1])]
                                E_b[k_] = t
                            touched.add(k_)
                        relocated.append({"pass": "collapse", "vessel": E_name[c], "parent": E_name[q], "moved_mm": round(float(np.linalg.norm(npos[t] - npos[end_node]) * 1e3), 1),
                                          "inside_mm": round(float(np.linalg.norm(np.diff(cs, axis=0), axis=1).sum() * 1e3), 1)})
                        touched.update(chain_set | {q, len(E_a) - 1})
                        touched.update((n, end_node, t))
                        _reloc_check(f"collapse {E_name[c]} into {E_name[q]}")
                        moved_any = True
                        break
                    continue
                if kx < 2:
                    continue
                # the foot: the parent sample matched by the last chain sample still inside (always a valid parent;
                # at the exit itself no parent may be among the neighbours searched)
                jf = jpar[kx - 1]
                foot = allP[jf]
                if _pass != "hook" and np.linalg.norm(foot - npos[n]) <= (allR[jf] + rc) * (1.0 if _pass == "inside" else 2.0):
                    continue  # a branch leaving normally (a hook goes even when it comes back right at its junction)
                q, lj = int(allE[jf]), int(allL[jf])
                if q in chain_set:
                    continue
                if q in touched or E_a[q] < 0:
                    continue
                if _pass == "hook":
                    # only onto the vessel the branch left: one passing through the junction (its name on two edges
                    # there, not the branch's own) — not a sibling at a bifurcation (internal / external iliac or
                    # carotid) nor the branch's own continuation
                    nm_at = [E_name[k_] for k_ in inc[n] if E_a[k_] >= 0]
                    through = {x for x in nm_at if nm_at.count(x) >= 2} - {E_name[c]}
                    if E_name[q] not in through:
                        continue
                # the parent must stay reachable from n without the chain (whose part inside the parent is removed):
                # else the move cuts the tree (the superior basal vein drains through the inferior basal vein; moving
                # the latter's junction onto the former disconnected both from the right inferior pulmonary vein)
                seen_, todo_ = {n}, [n]
                while todo_:
                    x_ = todo_.pop()
                    for k_ in inc[x_]:
                        if k_ in chain_set or E_a[k_] < 0:
                            continue
                        for y_ in (E_a[k_], E_b[k_]):
                            if y_ not in seen_:
                                seen_.add(y_)
                                todo_.append(y_)
                if E_a[q] not in seen_ and E_b[q] not in seen_:
                    continue
                if lj <= 1:
                    t = E_a[q]
                elif lj >= len(EP[q]) - 2:
                    t = E_b[q]
                else:
                    t = len(npos)
                    npos.append(EP[q][lj].copy())
                    E_a.append(t)
                    E_b.append(E_b[q])
                    EP.append(EP[q][lj:].copy())
                    ER.append(ER[q][lj:].copy())
                    E_name.append(E_name[q])
                    EP[q], ER[q] = EP[q][: lj + 1], ER[q][: lj + 1]
                    E_b[q] = t
                if t == end_node or (t == n and _pass != "hook"):
                    continue
                # the chain edge holding the exit sample keeps the part from the exit on, starting at t; the edges
                # before it (inside the parent) are removed
                ce, off_, fw = next(x for x in reversed(chain) if x[1] <= kx)
                loc = kx - off_
                q_ = EP[ce] if fw else EP[ce][::-1]
                r_ = ER[ce] if fw else ER[ce][::-1]
                if loc >= len(q_):
                    print(f"relocation debug: n {n} c {c} chain {[(x[0], x[1], x[2], len(EP[x[0]]), E_a[x[0]], E_b[x[0]]) for x in chain]} kx {kx} len(cs) {len(cs)} end {end_node}", flush=True)
                    raise SystemExit("relocation index bug")
                m_ = max(2, int(np.ceil(np.linalg.norm(q_[loc] - npos[t]) / 0.5e-3)) + 1)
                newp = np.vstack([np.linspace(npos[t], q_[loc], m_)[:-1], q_[loc:]])
                newr = np.concatenate([np.full(m_ - 1, rc), r_[loc:]])
                far = E_b[ce] if fw else E_a[ce]
                EP[ce], ER[ce], E_a[ce], E_b[ce] = newp, newr, t, far
                for e2, o2, _ in chain:
                    if o2 < off_:
                        E_a[e2] = E_b[e2] = -1
                relocated.append({"pass": _pass, "vessel": E_name[c], "parent": E_name[q], "moved_mm": round(float(np.linalg.norm(npos[t] - npos[n]) * 1e3), 1),
                                  "inside_mm": round(float(np.linalg.norm(np.diff(cs[: kx + 1], axis=0), axis=1).sum() * 1e3), 1)})
                touched.update(chain_set | {q, len(E_a) - 1})
                touched.add(n)
                _reloc_check(f"{_pass} {E_name[c]} onto {E_name[q]}")
                moved_any = True
                break
        if not moved_any:
            break
    keep_e = [k for k in range(len(E_a)) if E_a[k] >= 0]
    E_a, E_b = [E_a[k] for k in keep_e], [E_b[k] for k in keep_e]
    EP, ER, E_name = [EP[k] for k in keep_e], [ER[k] for k in keep_e], [E_name[k] for k in keep_e]
    print(f"junction relocation: {len(relocated)} co-running branches moved to where they leave their parent, {len(dropped)} duplicate edges inside a parent dropped; "
          f"largest: {sorted(relocated, key=lambda x: -x['inside_mm'])[:8]}; dropped: {sorted(set(dropped))[:12]}")
_n_trees("junction relocation")
# ── crossing joins (2026-11-25): two edges of one tree whose axes cross (closer than half the smaller radius) while
#    more than max(2 (r_i + r_j) + 2 mm, 20 mm) apart along the graph meet there in reality — a confluence the graph
#    misses (the hepatic veins pass through the IVC's axis but join it 89 mm away; the superior mesenteric vein crosses
#    itself): both edges are split at the crossing and joined by a short edge (the graph may gain a cycle; the network
#    is a general Kirchhoff solve). `--no-crossing-join`: off.
crossings = []
if "--no-crossing-join" not in sys.argv:
    import scipy.sparse as _sp
    from scipy.sparse.csgraph import dijkstra as _dij, connected_components as _cc
    for _round in range(4):
        allP = np.concatenate(EP)
        allR = np.concatenate(ER)
        allE = np.concatenate([np.full(len(EP[k]), k) for k in range(len(EP))])
        allL = np.concatenate([np.arange(len(EP[k])) for k in range(len(EP))])
        seg_ = [np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))] for q in EP]
        allA = np.concatenate(seg_)
        elen_ = np.array([x[-1] for x in seg_])
        pr = cKDTree(allP).query_pairs(0.5 * float(allR.max()), output_type="ndarray")
        i_, j_ = pr[:, 0], pr[:, 1]
        e_, f_ = allE[i_], allE[j_]
        d_ = np.linalg.norm(allP[i_] - allP[j_], axis=1)
        _, ccomp = _cc(_sp.coo_matrix((np.ones(len(E_a)), (E_a, E_b)), shape=(len(npos), len(npos))), directed=False)
        ok = (e_ != f_) & (d_ < 0.5 * np.minimum(allR[i_], allR[j_])) & (ccomp[np.asarray(E_a)[e_]] == ccomp[np.asarray(E_a)[f_]])
        i_, j_, e_, f_, d_ = i_[ok], j_[ok], e_[ok], f_[ok], d_[ok]
        if len(i_) == 0:
            break
        nn_ = len(npos)
        Gm = _sp.coo_matrix((np.r_[elen_, elen_] + 1e-9, (np.r_[E_a, E_b], np.r_[E_b, E_a])), shape=(nn_, nn_)).tocsr()
        nodes = np.unique(np.r_[np.asarray(E_a)[e_], np.asarray(E_b)[e_]])
        DGc = _dij(Gm, indices=nodes, limit=0.3)
        row = {int(n): q for q, n in enumerate(nodes)}
        Ea_, Eb_ = np.asarray(E_a), np.asarray(E_b)
        ri = np.array([row[int(x)] for x in Ea_[e_]])
        rb = np.array([row[int(x)] for x in Eb_[e_]])
        si, sj = allA[i_], allA[j_]
        le, lf = elen_[e_], elen_[f_]
        gd = np.minimum.reduce([si + DGc[ri, Ea_[f_]] + sj, si + DGc[ri, Eb_[f_]] + (lf - sj), (le - si) + DGc[rb, Ea_[f_]] + sj, (le - si) + DGc[rb, Eb_[f_]] + (lf - sj)])
        far = gd > np.maximum(2 * (allR[i_] + allR[j_]) + 2e-3, 20e-3)
        if not far.any():
            break
        # one crossing per edge pair (the closest), edges used once per round
        cand = sorted(zip(d_[far], i_[far], j_[far], gd[far]), key=lambda x: x[0])
        used, made = set(), 0
        for dd, a_, b_, g_ in cand:
            ea0, fb0 = int(allE[a_]), int(allE[b_])
            if ea0 in used or fb0 in used:
                continue
            nodes_new = []
            for k, lj in ((ea0, int(allL[a_])), (fb0, int(allL[b_]))):
                if lj <= 1:
                    nodes_new.append(E_a[k])
                elif lj >= len(EP[k]) - 2:
                    nodes_new.append(E_b[k])
                else:
                    t = len(npos)
                    npos.append(EP[k][lj].copy())
                    E_a.append(t)
                    E_b.append(E_b[k])
                    EP.append(EP[k][lj:].copy())
                    ER.append(ER[k][lj:].copy())
                    E_name.append(E_name[k])
                    EP[k], ER[k] = EP[k][: lj + 1], ER[k][: lj + 1]
                    E_b[k] = t
                    nodes_new.append(t)
            u, v = nodes_new
            if u != v and np.linalg.norm(npos[v] - npos[u]) < 0.5e-3:
                # the axes meet (closer than the 0.5 mm junction merge): v merges into u, no zero-length edge
                for k in range(len(E_a)):
                    if E_a[k] == v:
                        E_a[k] = u
                        EP[k][0] = npos[u]
                    if E_b[k] == v:
                        E_b[k] = u
                        EP[k][-1] = npos[u]
            elif u != v:
                m_ = max(2, int(np.ceil(np.linalg.norm(npos[v] - npos[u]) / 0.5e-3)) + 1)
                E_a.append(u)
                E_b.append(v)
                EP.append(np.linspace(npos[u], npos[v], m_))
                ER.append(np.full(m_, float(min(allR[a_], allR[b_]))))
                E_name.append(f"crossing: {E_name[ea0]}")
            crossings.append({"a": E_name[ea0], "b": E_name[fb0], "axis_distance_mm": round(float(dd) * 1e3, 2), "graph_distance_mm": round(float(g_) * 1e3) if np.isfinite(g_) else None})
            used.update((ea0, fb0, len(E_a) - 1))
            made += 1
        if made == 0:
            break
    print(f"crossing joins: {len(crossings)}: {crossings[:8]}")

_n_trees("crossing joins")
# ── floating junctions (2026-11-26): a junction (degree ≥ 3) whose branch ends met each other next to a same-tree
#    vessel — within 1.5 mm of its lumen surface — but reach it only far along the graph (> max(2 (R + r) +
#    10 mm, 30 mm)) is that vessel's branch point: joined to its axis there (the maxillary artery's pterygoid-canal /
#    buccal / descending-palatine group: their junction 2.3 mm from its axis, 94 mm away along the graph; the
#    atlas's maxillary is thinner there than the branches, so the per-end rules miss it; they reached it 94 mm away
#    through the buccal–facial–angular–infra-orbital anastomoses, so the graph gains that real loop).
#    OPT-IN (`--float-join`): on the atlas it also joined the facial artery's junction to the internal carotid artery
#    (ECA and ICA run side by side in the neck), a lumbar artery to the testicular artery, the left colic artery to the
#    aorta — vessels lie side by side, so a junction near another vessel does not tell an origin.
float_joins = []
if "--float-join" in sys.argv:
    import scipy.sparse as _sp2
    from scipy.sparse.csgraph import dijkstra as _dij2
    degf = np.bincount(np.r_[E_a, E_b], minlength=len(npos))
    jn = [n for n in range(len(npos)) if degf[n] >= 3]
    allP = np.concatenate(EP)
    allR = np.concatenate(ER)
    allE = np.concatenate([np.full(len(EP[k]), k) for k in range(len(EP))])
    allL = np.concatenate([np.arange(len(EP[k])) for k in range(len(EP))])
    tf = cKDTree(allP)
    elen_f = np.array([np.linalg.norm(np.diff(q, axis=0), axis=1).sum() for q in EP])
    Gf = _sp2.coo_matrix((np.r_[elen_f, elen_f] + 1e-9, (np.r_[E_a, E_b], np.r_[E_b, E_a])), shape=(len(npos), len(npos))).tocsr()
    rn_f = {}
    for k in range(len(EP)):
        rn_f[E_a[k]] = max(rn_f.get(E_a[k], 0.0), float(ER[k][0]))
        rn_f[E_b[k]] = max(rn_f.get(E_b[k], 0.0), float(ER[k][-1]))
    cand_f = []
    for n in jn:
        r_n = rn_f[n]
        tol = 1.5e-3  # fixed: 2 r let 5–8 mm lung-artery junctions join across 9–15 mm
        idx = tf.query_ball_point(npos[n], tol + r_n + float(allR.max()))
        if not idx:
            continue
        idx = np.asarray(idx)
        inc_n = {k for k in range(len(E_a)) if E_a[k] == n or E_b[k] == n}
        idx = idx[~np.isin(allE[idx], list(inc_n))]
        if not len(idx):
            continue
        gap = np.linalg.norm(allP[idx] - npos[n], axis=1) - allR[idx] - r_n
        idx, gap = idx[gap <= tol], gap[gap <= tol]
        if len(idx):
            cand_f.append((n, idx, gap))
    if cand_f:
        from scipy.sparse.csgraph import connected_components as _ccf
        _, ccf = _ccf(Gf, directed=False)
        DGf = _dij2(Gf, indices=[c[0] for c in cand_f], limit=0.2)
        Ea0, Eb0 = np.asarray(E_a).copy(), np.asarray(E_b).copy()  # the graph the distances were taken on
        len0 = np.array([len(q) for q in EP])
        used_e = set()
        for row, (n, idx, gap) in enumerate(cand_f):
            e_ = allE[idx]
            gd = np.minimum(DGf[row, Ea0[e_]] + allL[idx] * 0.5e-3, DGf[row, Eb0[e_]] + (len0[e_] - 1 - allL[idx]) * 0.5e-3)
            far = gd > np.maximum(2 * (allR[idx] + rn_f[n]) + 10e-3, 30e-3)
            far &= ccf[Ea0[e_]] == ccf[n]  # one tree (arteries never join veins)
            far &= ~np.isin(e_, list(used_e))
            if not far.any():
                continue
            q_ = int(np.argmin(np.where(far, gap, np.inf)))
            k, lj = int(allE[idx[q_]]), int(allL[idx[q_]])
            if k in used_e:
                continue
            if lj <= 1:
                t = E_a[k]
            elif lj >= len(EP[k]) - 2:
                t = E_b[k]
            else:
                t = len(npos)
                npos.append(EP[k][lj].copy())
                E_a.append(t)
                E_b.append(E_b[k])
                EP.append(EP[k][lj:].copy())
                ER.append(ER[k][lj:].copy())
                E_name.append(E_name[k])
                EP[k], ER[k] = EP[k][: lj + 1], ER[k][: lj + 1]
                E_b[k] = t
                used_e.add(len(E_a) - 1)
            used_e.add(k)
            m_ = max(2, int(np.ceil(np.linalg.norm(npos[t] - npos[n]) / 0.5e-3)) + 1)
            E_a.append(n)
            E_b.append(t)
            EP.append(np.linspace(npos[n], npos[t], m_))
            ER.append(np.full(m_, rn_f[n]))
            inc0 = next(kk for kk in range(len(E_a) - 1) if E_a[kk] == n or E_b[kk] == n)
            E_name.append(f"join: {E_name[inc0]}")
            float_joins.append({"junction_of": E_name[inc0], "onto": E_name[k], "surface_gap_mm": round(float(gap[q_]) * 1e3, 2),
                                "graph_distance_mm": None if not np.isfinite(gd[q_]) else round(float(gd[q_]) * 1e3)})
    print(f"floating junctions joined: {len(float_joins)}: {float_joins[:20]}")
_n_trees("floating junctions")
# ── self-loops and kinks (2026-11-25, research/flow_phantom.md Tier B5) ─────────────────────────────────────────────
# A self-loop edge (both ends at one node, e.g. a 34 mm "Bifurcation of pulmonary trunk" loop) whose samples lie mostly
# inside the lumens of the other vessels at that node is a drawing artefact and is dropped; otherwise it is split at its
# farthest sample into two edges (a ring). Kinks: a centreline bending tighter than its lumen (+ wall gap) overlaps
# itself; such samples are smoothed (3-point Laplacian, ends pinned) until the bend radius is ≥ 1.2 r + 1 mm or 200
# passes. Before that, a centreline folding back on itself (two samples closer than max(r) in space, > 1.5 max(r)
# apart along the edge: a zigzag in the thoracic aorta) has the fold cut out. `--no-kink-fix`: off.
loops = {"dropped": [], "split": []}
if "--no-kink-fix" not in sys.argv:
    inc_ = defaultdict(list)
    for k in range(len(E_a)):
        inc_[E_a[k]].append(k)
        inc_[E_b[k]].append(k)
    for k in [k for k in range(len(E_a)) if E_a[k] == E_b[k] and E_a[k] >= 0]:
        n = E_a[k]
        others = [q for q in inc_[n] if q != k and E_a[q] >= 0]
        inside_frac = 0.0
        if others:
            OP = np.concatenate([EP[q] for q in others])
            OR = np.concatenate([ER[q] for q in others])
            dd, jj = cKDTree(OP).query(EP[k])
            inside_frac = float((dd < OR[jj]).mean())
        if inside_frac >= 0.8:
            E_a[k] = E_b[k] = -1
            loops["dropped"].append((E_name[k], round(inside_frac, 2)))
        else:
            j = int(np.argmax(np.linalg.norm(EP[k] - npos[n], axis=1)))
            j = min(max(j, 1), len(EP[k]) - 2)
            t = len(npos)
            npos.append(EP[k][j].copy())
            E_a.append(t)
            E_b.append(n)
            EP.append(EP[k][j:].copy())
            ER.append(ER[k][j:].copy())
            E_name.append(E_name[k])
            EP[k], ER[k] = EP[k][: j + 1], ER[k][: j + 1]
            E_b[k] = t
            loops["split"].append((E_name[k], round(inside_frac, 2)))
    keep_e = [k for k in range(len(E_a)) if E_a[k] >= 0]
    E_a, E_b = [E_a[k] for k in keep_e], [E_b[k] for k in keep_e]
    EP, ER, E_name = [EP[k] for k in keep_e], [ER[k] for k in keep_e], [E_name[k] for k in keep_e]
    # loop excision: two samples of one edge closer than r in space but more than 3 r apart along it — the centreline
    # doubles back (a hairpin no smoothing undoes) — the piece between them becomes a straight segment; repeated
    n_loop = 0
    for k in range(len(EP)):
        for _ in range(20):
            q, r_ = EP[k], ER[k]
            if len(q) < 8:
                break
            ar = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
            pr = cKDTree(q).query_pairs(float(r_.max()), output_type="ndarray")
            if len(pr) == 0:
                break
            i_, j_ = pr.min(1), pr.max(1)
            dd = np.linalg.norm(q[i_] - q[j_], axis=1)
            # closer than max(r) in space and > 1.5 max(r) apart along: a fold (a smooth bend of radius ≥ r never is)
            bad = (dd < np.maximum(r_[i_], r_[j_])) & (ar[j_] - ar[i_] > 1.5 * np.maximum(r_[i_], r_[j_]))
            if not bad.any():
                break
            w_ = np.argmax(np.where(bad, ar[j_] - ar[i_], -1))  # the longest loop first
            a0, b0 = int(i_[w_]), int(j_[w_])
            m_ = max(2, int(np.ceil(np.linalg.norm(q[b0] - q[a0]) / 0.5e-3)) + 1)
            seg_ = np.linspace(q[a0], q[b0], m_)
            EP[k] = np.vstack([q[:a0], seg_, q[b0 + 1 :]])
            ER[k] = np.concatenate([r_[:a0], np.linspace(r_[a0], r_[b0], m_), r_[b0 + 1 :]])
            n_loop += 1
    loops["loops_excised"] = n_loop
    n_kink = 0
    for k in range(len(EP)):
        q = EP[k]
        if len(q) < 5:
            continue
        r_ = ER[k]
        for _ in range(200):
            a_, b_, c_ = q[:-2], q[1:-1], q[2:]
            la, lb, lc = np.linalg.norm(b_ - a_, axis=1), np.linalg.norm(c_ - b_, axis=1), np.linalg.norm(c_ - a_, axis=1)
            area2 = np.linalg.norm(np.cross(b_ - a_, c_ - b_), axis=1)
            rho = la * lb * lc / np.maximum(2 * area2, 1e-18)  # circumradius = local bend radius
            bad = rho < 1.2 * r_[1:-1] + 1e-3
            if not bad.any():
                break
            n_kink += int(_ == 0) * int(bad.sum())
            m_ = np.zeros(len(q), bool)
            m_[1:-1] = bad
            m_ = np.convolve(m_.astype(float), np.ones(5), mode="same") > 0  # smooth a little around them
            m_[0] = m_[-1] = False
            q = q.copy()
            q[1:-1] = np.where(m_[1:-1, None], 0.5 * q[1:-1] + 0.25 * (q[:-2] + q[2:]), q[1:-1])
        EP[k] = q
    loops["kink_samples"] = n_kink
    print(f"self-loops: {len(loops['dropped'])} dropped (inside their node's other vessels), {len(loops['split'])} split into rings; {n_loop} doubled-back loops excised; "
          f"kinks: {n_kink} samples bending tighter than 1.2 r + 1 mm smoothed; dropped {loops['dropped'][:6]}")
_n_trees("self-loops, folds, kinks")
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
for c_ in comps:  # an organ-rooted tree must not have swallowed a heart tree (a join / bridge merging them relabels it)
    if c_["class"] in ORGAN_ROOTS.values():
        _mem = {E_name[k] for k in np.flatnonzero(ecomp == c_["id"])}
        assert not any(("vena cava" in m_.lower() or "aorta" in m_.lower() or "pulmonary" in m_.lower()) for m_ in _mem), (c_["class"], sorted(_mem)[:20])

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
    if not rs and c["class"] in ORGAN_ROOTS.values():  # organ-rooted tree: the organ end of its root vessel
        dg_ = np.bincount(np.r_[E_a, E_b], minlength=len(npos))
        rs = [int(n) for k in ks if E_name[k] in ORGAN_ROOTS for n in (E_a[k], E_b[k]) if dg_[n] == 1][:1]
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
from deoverlap import deoverlap, separate  # noqa: E402

venous_s = np.array([system_of(E_name[k].split(": ", 1)[-1])[0] for k in range(ne) for _ in range(E_off[k + 1] - E_off[k])])
# ── separation: interpenetrating tubes of different components move apart (deoverlap.separate; research/
#    flow_phantom.md Tier B4). Shrinking cannot resolve deep overlaps, and the LBM case builder's separating walls then
#    pinch both lumens (right hilum: RPA / right superior PV, 6 m/s and 3 m/s jets). Mobility = distance to the heart
#    chambers' surface / 30 mm clipped to [0.25, 1] (coronaries and cardiac veins stay on the heart), and 0 within 5 mm
#    of a heart root rising to 1 at 20 mm (the valves sit at the connectors' heart ends); lumen gap target = walls,
#    at least 1.5 mm (the LBM builder's separation); each sample moves at most
#    mobility × (0.5 r + 2 mm). A same-class component without a heart root (a fragment that should have been joined)
#    is left in place and reported.
if SEPARATE:
    comp_s = np.repeat(ecomp, np.diff(E_off))
    heart_surf = mesh_vertices(r"^(Left|Right) (atrium|ventricle)$")
    dist_heart = cKDTree(heart_surf).query(pts_out)[0]
    dist_root = cKDTree(npos[np.array(sorted(set(heart_roots)), int)]).query(pts_out)[0]
    # heart roots pinned (the LBM valves sit at the connectors' heart ends): 0 within 5 mm, 1 from 20 mm
    mob = np.minimum(np.clip(dist_heart / 30e-3, 0.25, 1.0), np.clip((dist_root - 5e-3) / 15e-3, 0.0, 1.0))
    cap = mob * (0.5 * rad_out + 2e-3)
    cls_of_c = {c["id"]: c["class"] for c in comps}
    has_root = {c["id"]: any(comp[r] == c["id"] for r in heart_roots) for c in comps}

    def _skip(ca, cb):
        return cls_of_c.get(ca) == cls_of_c.get(cb) and not (has_root.get(ca) and has_root.get(cb))

    # target lumen gap: the walls, at least 1.5 mm (the LBM case builder separates disallowed regions by 1.5 mm and
    # carves whatever is closer out of both lumens)
    pts_out, sep_disp, sep_rep = separate(pts_out, rad_out, np.asarray(E_off), comp_s, lambda r: np.maximum(np.where(venous_s, 0.06, 0.12) * r, 0.75e-3), mob, skip_pair=_skip, cap=cap, iters=40)
    # re-join at the nodes: the end samples of all edges at a node (capped by their own radii) move to their mean
    nsum = np.zeros((len(npos), 3))
    ncnt = np.zeros(len(npos))
    ends = np.r_[np.asarray(E_off[:-1]), np.asarray(E_off[1:]) - 1]
    enode = np.r_[E_a, E_b]
    np.add.at(nsum, enode, pts_out[ends])
    np.add.at(ncnt, enode, 1)
    used = ncnt > 0
    npos[used] = nsum[used] / ncnt[used, None]
    sep_rep["node_rejoin_max_mm"] = float(np.linalg.norm(npos[enode] - pts_out[ends], axis=1).max() * 1e3)
    # spread each end correction along its edge: weight exp(−s² / 2σ²)·(1 − s/L) from end a (1 there, 0 at end b),
    # mirrored for end b, σ = max(r, 2 mm): the ends land exactly on the node, no kink
    for k in range(ne):
        a, b = E_off[k], E_off[k + 1]
        q = pts_out[a:b]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        L_ = max(arc[-1], 1e-9)
        da, db = npos[E_a[k]] - q[0], npos[E_b[k]] - q[-1]
        sa, sb = max(rad_out[a], 2e-3), max(rad_out[b - 1], 2e-3)
        wa = np.exp(-arc**2 / (2 * sa**2)) * (1 - arc / L_)
        wb = np.exp(-(L_ - arc) ** 2 / (2 * sb**2)) * (arc / L_)
        corr = wa[:, None] * da + wb[:, None] * db
        pts_out[a:b] = q + corr
        sep_disp[a:b] += corr
    dm = np.linalg.norm(sep_disp, axis=1)
    moved = {}
    for k in range(ne):
        a, b = E_off[k], E_off[k + 1]
        if dm[a:b].max() > 0.5e-3:
            nm = E_name[k].split(": ", 1)[-1]
            moved[nm] = max(moved.get(nm, 0.0), float(dm[a:b].max() * 1e3))
    sep_rep["moved_mm"] = {k: round(v, 1) for k, v in sorted(moved.items(), key=lambda kv: -kv[1])}
    sep_rep["skipped_component_pairs"] = [[a, b, cls_of_c.get(a), cls_of_c.get(b)] for a, b in sep_rep["skipped_component_pairs"]]
    diag["separation"] = {k: v for k, v in sep_rep.items() if k != "moved_mm"}
    print("separation:", diag["separation"], "largest moves (mm):", list(sep_rep["moved_mm"].items())[:12])
else:
    sep_rep = {"moved_mm": {}}
# radius continuity where one vessel continues another (2026-11-28; radius_taper.joint_taper; --joint-taper-mm 0: off,
# as before; --joint-taper-through off: degree-2 joints only): at degree-2 joints and along the parent's through-line at
# branch nodes the two ends meet at their geometric mean (IVC, aortic arch | thoracic aorta, subclavian | axillary,
# pulmonary veins stepped 17–53 % at their name boundaries)
JTAPER = float(sys.argv[sys.argv.index("--joint-taper-mm") + 1]) * 1e-3 if "--joint-taper-mm" in sys.argv else 15e-3
taper_rep = {}
if JTAPER > 0:
    from radius_taper import joint_taper  # noqa: E402

    _thr = (sys.argv[sys.argv.index("--joint-taper-through") + 1] if "--joint-taper-through" in sys.argv else "on") == "on"
    rad_out, taper_rep = joint_taper(pts_out, rad_out, E_off, E_a, E_b, len(npos), length=JTAPER, through=_thr, names=E_name)
    diag["joint_taper"] = {"joints": len(taper_rep), "length_mm": JTAPER * 1e3, "through": _thr}
    print(f"joint taper: {len(taper_rep)} joints, largest steps (mm):", sorted(taper_rep.items(), key=lambda kv: -max(kv[1]) / min(kv[1]))[:10])
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
_bad_ends = [(E_name[k], round(float(np.linalg.norm(pts_out[E_off[k]] - npos[E_a[k]]) * 1e3), 2), round(float(np.linalg.norm(pts_out[E_off[k + 1] - 1] - npos[E_b[k]]) * 1e3), 2))
              for k in range(ne) if np.linalg.norm(pts_out[E_off[k]] - npos[E_a[k]]) > 1e-4 or np.linalg.norm(pts_out[E_off[k + 1] - 1] - npos[E_b[k]]) > 1e-4]
assert not _bad_ends, f"{len(_bad_ends)} edges whose polyline does not run from node a to node b (mm off at a, b): {_bad_ends[:8]}"
np.savez_compressed(OUT / "graph.npz", node_pos=npos.astype(np.float32), edge_a=E_a.astype(np.int32), edge_b=E_b.astype(np.int32), edge_offsets=np.array(E_off, np.int64), points=pts_out.astype(np.float32), radius=rad_out.astype(np.float32), direction=direction, edge_comp=ecomp.astype(np.int32))
(OUT / "graph.json").write_text(json.dumps({"diagnostics": diag, "components": comps, "edge_names": E_name, "joins": joins, "connectors": connectors, "trimmed_roots": trimmed_roots, "radius_corrections": rfix_applied, "joint_taper": taper_rep, "deoverlap_max_shrink": dor["vessels"], "separation_moved_mm": sep_rep["moved_mm"], "relocated": relocated, "dropped_inside_parent": dropped, "loops_kinks": loops, "crossings": crossings, "floating_joins": float_joins, "source": "Z-Anatomy (CC BY-SA 4.0)"}, indent=1))
print(json.dumps(diag, indent=1))
for c in comps[:14]:
    print(f"{c['class']:20s} edges {c['edges']:5d} length {c['length_m']:6.2f} m  vessels {c['vessels']:4d}  rmax {c['max_radius_mm']:5.1f} mm  e.g. {c['members_sample'][:3]}")

# ── figure: components coloured by class ───────────────────────────────────────────────────────────────────────────
col = {"systemic_arterial": (0.85, 0.1, 0.1), "systemic_venous": (0.15, 0.35, 0.9), "pulmonary_arterial": (0.6, 0.1, 0.8), "pulmonary_venous": (0.1, 0.7, 0.7), "coronary": (1.0, 0.6, 0.0), "venous_other": (0.5, 0.6, 0.95), "portal_venous": (0.4, 0.2, 0.6), "arterial_other": (0.95, 0.55, 0.55)}
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
(OUT / "figs").mkdir(exist_ok=True)
fig.savefig(OUT / "figs/graph.png", dpi=75, bbox_inches="tight")
