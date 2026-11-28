"""Cardiac lattice-Boltzmann case of a phantom (hasty-cfd; research/flow_phantom.md "Tier B: LBM").

    rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_case.py <phantom_dir> <case_dir>
        [--labels data/hasty/phantom/p0/tissue_0.5mm_heart.hv] [--coarsen 1] [--valves leaflets|orifice]
        [--stub-mm 45,30,20,15] [--v-max <m/s, default from the anatomy>] [--u-lattice 0.08] [--cycles 4] [--warmup 2] [--bins 50]
        [--out-factor 2] [--rcr] [--av-radius-mm 16,17] [--chamber-first] [--venous inertial|pressure|inlet]
        [--av-diastasis 0.5] [--wall-mm 2] [--sep-mm 1.5] [--carve-mm 2.5] [--carve-down-mm 4] [--open-thr 0.03] [--open-abs] [--dx-mm 0.5]
        [--pulm-mtt 5] [--sys-capillary-s 2] [--sl-hinge lumen|annulus] [--motion <cardiac_motion.hv>]
        [--refine aortic,pulmonary|all] [--refine-margin-mm 6] [--refine-down-mm 10] [--refine-levels 1]
(--labels default: <phantom_dir>/atlas/tissue_0.5mm_heart.hv when the phantom has a deformed atlas, else p0's)

Blood domain in the reference pose (the atlas pose that `cardiac_motion.hv` displaces):
- the four chambers (+ the valve tissue, fluid like in phantom_4d), small components dropped;
- great-vessel stubs from each heart root (the roots phantom_4d feeds the chambers from), following the main branch
  for a length per class (systemic arterial / pulmonary arterial / systemic venous / pulmonary venous), as capsules
  of the vessel samples, continued by a slab of 3 lattice spacings beyond the cut plane (the equilibrium inlet /
  outlet nodes) and capped.
Walls the atlas labels lack (their chamber meshes touch): every blood voxel gets a side (left: LA, LV, aorta and
pulmonary-vein stubs; right: RA, RV, venae cavae and pulmonary-trunk stubs; valve tissue: the nearest chamber's),
a septal wall goes wherever the sides touch (thickened until they are disconnected), an atrioventricular wall wherever
atrium and ventricle touch except inside the valve's cylinder, and a solid collar around every valve annulus — so
the only passage between atrium and ventricle, ventricle and artery is the valve disk (checked: removing a valve's
disk disconnects them).
Signed distance (> 0 in the blood) by exact EDT, lightly smoothed, on the label grid (or coarsened ×k).
Valves:
- aortic / pulmonary at the root connectors (centre, downstream axis, vessel radius), three leaflets (4 strips each);
  --sl-hinge lumen (default): the hinge circle half a leaflet + half a cell outside the lumen, so the open cusps lie
  in the wall (cusps opening into the sinuses) and the orifice is the lumen; annulus: hinge at the lumen (before);
- mitral / tricuspid on a plane fitted to the LA–LV / RA–RV interface (normal atrium → ventricle, radius from the
  interface area ×1.1), mitral two leaflets (coaptation y = 0.2 R), tricuspid three;
- opening: 1 while the cardiac model's valve flow (Q_AO, Q_PA, Q_MV, Q_TV) exceeds 3 % of its peak (--open-abs:
  |Q|, the closing backflow included), ramped over 30 ms (periodic moving average); AV valves stay ≥ half open
  through diastasis.
Cuts at the stub ends: veins carry the cardiac model's atrial inflow Q_PV / Q_VR split by the network's mean shares,
- inertial (default): velocity inlets plus a correction driven by the cut pressure through the upstream vein's
  inertance (ℓ = 6 cm pulmonary, 10 cm caval): the wall motion decides the total, the vein's inertia keeps the
  waveform and the split;
- pressure: pressure inlets with a slow share trim (no upstream inertia: the veins slosh against each other);
- inlet: plain velocity inlets (no pressure reference behind closed valves: the LBM chamber volume change differs from
  the model's by ~10 %, the excess compresses the closed atrium / ventricle);
arteries pressure outlets (or RCR with --rcr, R1 from the characteristic impedance, C and R2 from the stub's network
mean flow and pressure).
Writes <case_dir>/case.json, sdf.npy, figs/ (domain slices with valves and cuts, opening tables).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
P0, OUT = Path(A[0]), Path(A[1])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


CH = ["blood_LA", "blood_LV", "blood_RA", "blood_RV"]
STUB_MM = [float(x) for x in opt("--stub-mm", "45,30,20,15").split(",")]
STUB = dict(zip(["systemic_arterial", "pulmonary_arterial", "systemic_venous", "pulmonary_venous"], STUB_MM))
CHAMBER_OF = {"systemic_arterial": "blood_LV", "pulmonary_arterial": "blood_RV", "systemic_venous": "blood_RA", "pulmonary_venous": "blood_LA"}
VALVES = opt("--valves", "leaflets")
VENOUS = opt("--venous", "inertial")
AV_BRIDGE = float(opt("--av-diastasis", 0.5))
# lengths are physical (mm) with lattice minimums, so a case at another resolution (or of a diffeomorphically deformed
# phantom) gets the same walls: interface walls are 2 voxels, grown to ≥ WALL_MM in total and ≥ 4 lattice cells (a
# thinner wall loses its solid nodes under smoothing, motion and sampling); forbidden regions ≥ SEP_MM and ≥ 3 cells apart
WALL_MM = float(opt("--wall-mm", 2.0))
SEP_MM = float(opt("--sep-mm", 1.5))
# proximity walls between regions that must not exchange fluid: "half" (default, 2026-11-25) carves the separation from
# both sides of the mid-surface — a voxel of b becomes wall where its distance to a plus its own depth in b is ≤ SEP + 1
# (voxels), i.e. each side gives up (SEP − gap)/2 and the gap ends ≥ SEP; "each" (before): every voxel of b within SEP
# of a becomes wall and vice versa (gap 2 SEP − gap; at the right hilum it carved the right PA / superior PV lumens to
# slivers: 6 m/s and 3 m/s jets, research/flow_phantom.md Tier B4)
SEP_SPLIT = opt("--sep-split", "half")
assert SEP_SPLIT in ("half", "each"), SEP_SPLIT
TRANSP = opt("--transpiration", "on")
assert TRANSP in ("on", "off"), TRANSP
LUMEN_MIN = float(opt("--lumen-min", 0.6))
CARVE = float(opt("--carve-mm", 2.5)) * 1e-3
CARVE_DOWN = float(opt("--carve-down-mm", 4.0)) * 1e-3  # into the downstream chamber (the inflow / outflow tract)
CARVE_RIM = float(opt("--carve-rim-mm", 2.0)) * 1e-3  # beyond the annulus radius
SPLIT_ZONE = opt("--split-zone", "passage")
assert SPLIT_ZONE in ("passage", "wide"), SPLIT_ZONE
OPEN_THR = float(opt("--open-thr", 0.03))
OPEN_ABS = "--open-abs" in A
ST18 = ndimage.generate_binary_structure(3, 2)  # face + edge neighbours: the D3Q19 links
# --debug-pt x,y,z[,r] (mm): at each build stage, what holds the voxels within r (default 3 mm) of the point (domain,
# regions, walls), to find the step that pinches a lumen
_DBG = [[float(x) * 1e-3 for x in s_.split(",")] for s_ in A[A.index("--debug-pt") + 1].split(";")] if "--debug-pt" in A else []


def _dbg(stage):
    for q_ in _DBG:
        p_, r_ = np.asarray(q_[:3]), (q_[3] if len(q_) > 3 else 3e-3)
        dom_ = globals().get("domain")
        if dom_ is None:
            continue
        lo_ = np.maximum(np.floor((p_ - r_ - lo0) / h).astype(int), 0)
        hi_ = np.minimum(np.ceil((p_ + r_ - lo0) / h).astype(int) + 1, np.array(dom_.shape[::-1]))
        zz_, yy_, xx_ = np.meshgrid(*(np.arange(lo_[d], hi_[d]) for d in (2, 1, 0)), indexing="ij")
        Xq = np.stack([xx_, yy_, zz_], -1) * h + lo0 + h / 2
        m_ = np.linalg.norm(Xq - p_, axis=-1) < r_
        sl_ = (slice(lo_[2], hi_[2]), slice(lo_[1], hi_[1]), slice(lo_[0], hi_[0]))
        dm_ = dom_[sl_][m_]
        out_ = f"debug {stage:24s} at {np.round(p_ * 1e3, 1).tolist()} r {r_ * 1e3:.1f} mm: domain {dm_.mean() * 100:.0f} %"
        lab_ = Lc[sl_][m_]
        out_ += "; labels " + ", ".join(f"{names[i_]} {np.mean(lab_ == i_) * 100:.0f}" for i_ in np.unique(lab_))
        if "region" in globals() and globals()["region"].shape == dom_.shape:
            rg_ = globals()["region"][sl_][m_]
            out_ += "; regions " + ", ".join(f"{i_}:{np.mean(rg_ == i_) * 100:.0f}" for i_ in np.unique(rg_))
        if "wall" in globals() and globals()["wall"].shape == dom_.shape:
            out_ += f"; wall {globals()['wall'][sl_][m_].mean() * 100:.0f} %"
        print(out_)
K = int(opt("--coarsen", 1))

# ── labels (reference pose) ──────────────────────────────────────────────────────────────────────────────────────────
# the phantom's own (diffeomorphically deformed) atlas when it has one (make_phantom --diffeo)
_own = P0 / "atlas" / "tissue_0.5mm_heart.hv"
TS = vf.read(opt("--labels", str(_own if _own.exists() else REPO / "data/hasty/phantom/p0/tissue_0.5mm_heart.hv")))
names = list(TS["tissues"])
ID = {n: i for i, n in enumerate(names)}
L = np.asarray(TS["labels"])  # [Z, Y, X]
lo0 = np.asarray(TS["origin_m"], float)
h0 = float(TS["voxel_m"])
blood = np.isin(L, [ID[c] for c in CH] + [ID["valve"]])
if K > 1:
    nz, ny, nx = (s // K for s in blood.shape)
    blood = blood[: nz * K, : ny * K, : nx * K].reshape(nz, K, ny, K, nx, K).mean((1, 3, 5)) >= 0.5
    Lc = L[K // 2 :: K, K // 2 :: K, K // 2 :: K][:nz, :ny, :nx]
else:
    Lc = L
h = h0 * K
# the lattice spacing: the label voxel by default; --dx-mm decouples it (the sdf is sampled trilinearly), e.g. a finer
# lattice for a resolution study on the same geometry
DX = float(opt("--dx-mm", h * 1e3)) * 1e-3
WALL_GROW = max(1, int(np.ceil((WALL_MM * 1e-3 / h - 2) / 2 - 1e-9)))
SEP = max(3, int(round(SEP_MM * 1e-3 / h)))
lab, nlab = ndimage.label(blood)
sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nlab + 1))
blood = np.isin(lab, 1 + np.flatnonzero(sizes * h**3 >= 0.25e-6))  # components ≥ 0.25 mL
print(f"chambers: {int(blood.sum())} voxels of {h * 1e3:.2f} mm ({blood.sum() * h**3 * 1e6:.0f} mL incl. valve tissue), {len(sizes)} components → {int((sizes * h**3 >= 0.25e-6).sum())}")


def ijk(p):
    return np.floor((np.asarray(p) - lo0) / h).astype(int)  # x, y, z index


def centre(i):
    return lo0 + (np.asarray(i) + 0.5) * h


# ── vessels, roots, stubs ────────────────────────────────────────────────────────────────────────────────────────────
V = vf.read(P0 / "vessels.hv")
P_ref = np.asarray(V["points"]).astype(np.float64)
R_ref = np.asarray(V["radius"]).astype(np.float64)
G = load_any(P0 / "graph.npz")
GJ = json.loads((P0 / "graph.json").read_text())
eoff, ea, eb = G["edge_offsets"], G["edge_a"], G["edge_b"]
enames = GJ["edge_names"]
net = load_any(P0 / "flow" / "network_flow.npz")
om, qn, pn = net["omega"], net["q"], net["p"]
# --motion <cardiac_motion.hv>: another cardiac model of the same phantom (tests); default the phantom's own
CM_FILE = Path(opt("--motion", str(P0 / "motion" / "cardiac_motion.hv")))
cm = vf.read(CM_FILE)
period = float(cm["period"])
adj = {}
for k in range(len(ea)):
    adj.setdefault(int(ea[k]), []).append(k)
    adj.setdefault(int(eb[k]), []).append(k)


def edge_samples(k, from_node):
    s = np.arange(eoff[k], eoff[k + 1])
    return (s, 1) if ea[k] == from_node else (s[::-1], -1)


def stub_path(root, length):
    """samples from the root along the main branch (largest mean radius) up to `length`; orientation sign per sample
    (+1: traversed a → b)"""
    path, sgn, acc, node, used = [], [], 0.0, root, set()
    while acc < length:
        cand = [k for k in adj.get(node, []) if k not in used]
        if not cand:
            break
        k = max(cand, key=lambda k_: R_ref[eoff[k_] : eoff[k_ + 1]].mean())
        used.add(k)
        s, sg = edge_samples(k, node)
        for j in s:
            if path:
                acc += np.linalg.norm(P_ref[j] - P_ref[path[-1]])
            path.append(int(j))
            sgn.append(sg)
            if acc >= length:
                break
        node = int(eb[k] if sg == 1 else ea[k])
    return np.array(path), np.array(sgn)


roots = []
for c in GJ["components"]:
    if c["class"] not in CHAMBER_OF:
        continue
    for rn in c.get("roots", []):
        ks = [k for k in range(len(ea)) if ea[k] == rn or eb[k] == rn]
        if not ks:
            continue
        s0 = int(eoff[ks[0]]) if ea[ks[0]] == rn else int(eoff[ks[0] + 1] - 1)
        ic = np.floor((P_ref[s0] - lo0) / h0).astype(int)
        rv = int(round(10e-3 / h0))
        cube = L[max(ic[2] - rv, 0) : ic[2] + rv + 1, max(ic[1] - rv, 0) : ic[1] + rv + 1, max(ic[0] - rv, 0) : ic[0] + rv + 1]
        if not np.isin(cube, [ID[x] for x in CH]).any():
            continue
        path, sgn = stub_path(rn, STUB[c["class"]] * 1e-3)
        if len(path) < 3:
            continue
        roots.append({"name": enames[ks[0]], "class": c["class"], "chamber": CHAMBER_OF[c["class"]], "path": path, "sgn": sgn, "node": int(rn)})
print("heart roots:", [(r["name"][:32], r["class"], len(r["path"])) for r in roots])

# ── vessel trees ─────────────────────────────────────────────────────────────────────────────────────────────────────
# from each heart root every branch of the vessel graph whose lumen radius is ≥ R_CUT (≥ 5 lattice cells at 0.5 mm) and
# that stays inside the label box (less a margin for the cut slab and cap): the great vessels with their CFD flow, not
# only short stubs (an aneurysm 60 mm up the ascending aorta was outside a 45 mm stub). A branch ends in a cut where
# its radius drops below R_CUT or it leaves the box; side branches whose included part is shorter than max(4 mm, 2 R)
# are left out (their network flow leaves through the tree's pressure outlet). Edges leading to another heart root
# (e.g. a venous connection between the cavae) are left out. `--stubs`: the old single path of --stub-mm per root.
R_CUT = float(opt("--tree-rmin-mm", 2.5)) * 1e-3
BOX_MARG = float(opt("--box-margin-mm", 8.0)) * 1e-3
box_lo = lo0 + BOX_MARG
box_hi = lo0 + np.array(L.shape[::-1]) * h0 - BOX_MARG
rootset = {r["node"] for r in roots}


def inside(p):
    return bool(np.all(p >= box_lo) and np.all(p <= box_hi))


def arclen(sm):
    return float(np.linalg.norm(np.diff(P_ref[sm], axis=0), axis=1).sum()) if len(sm) > 1 else 0.0


def _off_junction(seg):
    """a cut never on a junction (2026-11-26): the branch ends 1.5 r + 1 mm before its node when that node is a
    junction — a cut disk through the crotch was 30 % fluid (the left superior pulmonary vein's inlet at its lingular /
    anterior split delivered 1 % of its flow). `--cut-at-node`: as before"""
    if "--cut-at-node" in A or len(seg) < 3:
        return seg
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P_ref[seg[::-1]], axis=0), axis=1))]  # arc from the end
    keep = d >= 1.5 * R_ref[seg[-1]] + 1e-3
    n_keep = len(seg) - int(np.argmax(keep)) if keep.any() else 2
    return seg[: max(2, n_keep)]


def build_tree(r):
    """branches [{samples, edge, sgn, parent, end}] of root r, depth first from the root node"""
    if "--stubs" in A:
        return [{"samples": np.asarray(r["path"]), "edge": -1, "sgn": int(r["sgn"][-1]), "parent": -1, "end": True}]
    br, stack, used, seen = [], [(r["node"], None, -1)], set(), {r["node"]}
    while stack:
        node, via, parent = stack.pop()
        # the continuation (the largest lumen leaving the node) is always kept: only side branches are dropped for being
        # short (a 24 mm rule cut the descending aorta's last piece in the box, leaving the aorta blind-ended)
        cand_ = [k for k in adj.get(node, []) if k != via and k not in used]
        cont = max(cand_, key=lambda k_: R_ref[edge_samples(k_, node)[0][0]]) if cand_ else None
        for k in adj.get(node, []):
            if k == via or k in used:
                continue
            used.add(k)
            sm, sg = edge_samples(k, node)
            other = int(eb[k] if sg == 1 else ea[k])
            if other in rootset and other != r["node"]:
                continue
            n_in = 0
            for j in sm:
                if R_ref[j] < R_CUT or not inside(P_ref[j]):
                    break
                n_in += 1
            seg = np.asarray(sm[:n_in])
            full = n_in == len(sm)
            if full and other not in seen and len(adj.get(other, [])) > 1:
                br.append({"samples": seg, "edge": int(k), "sgn": int(sg), "parent": parent, "end": False})
                seen.add(other)
                stack.append((other, k, len(br) - 1))
            elif n_in >= 2 and (parent == -1 or k == cont or arclen(seg) >= max(4e-3, 2 * R_ref[sm[0]])):
                if full and len(adj.get(other, [])) >= 3:
                    seg2 = _off_junction(seg)
                    # the trimmed cut's cap (2.5 r beyond the plane) must stay clear of the parent's lumen: a side branch
                    # too short for that is left out (its flow leaves through the tree's pressure outlet), the root and
                    # continuation branches keep their cut at the node
                    r_par = max(R_ref[edge_samples(k_, node)[0][0]] for k_ in adj.get(node, [k]))  # the largest lumen at the branch's origin
                    if arclen(seg2) >= 2.5 * R_ref[seg2[-1]] + r_par + 1e-3:
                        seg = seg2
                    elif parent >= 0 and k != cont:
                        continue
                br.append({"samples": seg, "edge": int(k), "sgn": int(sg), "parent": parent, "end": True})
            elif k == cont and parent >= 0 and R_ref[sm[0]] >= R_CUT:
                # the continuation leaves the box right at the node: the branch into the node ends in the cut
                br[parent]["end"] = True
                if not any(b_["parent"] == parent for b_ in br):  # no side branch starts at that node
                    br[parent]["samples"] = _off_junction(br[parent]["samples"])
    # a branch whose onward edges were all left out (thinner than R_CUT at their start, outside the box, too short)
    # ends in a cut too, off its junction (2026-11-26: the coronary sinus continues into the 2.2 mm great cardiac vein;
    # with no cut its tree was a sealed pocket with no inflow). `--no-orphan-cuts`: as before
    if "--no-orphan-cuts" not in A:
        has_child = {b_["parent"] for b_ in br}
        for q_, b_ in enumerate(br):
            if not b_["end"] and q_ not in has_child and len(b_["samples"]) >= 2:
                b_["end"] = True
                b_["samples"] = _off_junction(np.asarray(b_["samples"]))
    # dead ends: an end branch without network flow (graph artefacts, e.g. a 12 mm blind stub at the pulmonary
    # bifurcation) is no outlet — its cut and cap only destroy the neighbourhood (a 4 mL cap cut off a pulmonary-vein
    # tree): dropped
    q0_ = abs(float(np.real(qn[r["path"][0], 0])))
    for b in br:
        if b["end"] and abs(float(np.real(qn[b["samples"][-1], 0]))) < max(0.3e-6, 0.005 * q0_):
            b["drop"] = True
    return br


branches = []  # global list: dict + root index
for k_, r in enumerate(roots):
    for b in build_tree(r):
        b["root"] = k_
        branches.append(b)
    mine = [b for b in branches if b["root"] == k_ and not b.get("drop")]
    vol_ = sum(np.pi * float((R_ref[b["samples"]] ** 2).mean()) * arclen(b["samples"]) for b in mine)
    nd_ = sum(b["root"] == k_ and b.get("drop", False) for b in branches)
    print(f"tree {r['name'][:30]:30s} ({r['class']}): {len(mine)} branches, {sum(b['end'] for b in mine)} cuts, ≈ {vol_ * 1e6:.0f} mL of lumen ({nd_} dead ends dropped)")

# ── cut clearance (2026-11-26): a cut's slab and flux layer (lateral < 1.3 R, −1.5 h … slab + 3 h past its plane) and
#    its capsule end must hold no other lumen: the builder removes other branches' voxels there (the middle lobar artery
#    was pinched to 0.52 r by its medial segmental branch's cut). An end branch slides its cut back along its samples
#    until clear; a side branch with no clear position is left out (its flow leaves through the tree's pressure outlet,
#    like the junction rule's), a root / continuation keeps its cut (reported). Lumens of all trees count; a ball of the
#    other sample's radius must reach > 0.5 mm into the zone. `--cut-overlap`: as before
from scipy.spatial import cKDTree  # noqa: E402

CUT_TOL = 0.5e-3
CUT_KILL = opt("--cut-kill", "geom")
assert CUT_KILL in ("geom", "owner"), CUT_KILL


def _in_capsule(Xq, poly):
    """points inside the capsule chain (pts, rads) of a branch (as rasterized)"""
    pts, rads = poly
    ins = np.zeros(len(Xq), bool)
    for a in range(len(pts) - 1):
        pa, pb = pts[a], pts[a + 1]
        ab = pb - pa
        s = np.clip(((Xq - pa) @ ab) / max(ab @ ab, 1e-18), 0, 1)
        d = np.linalg.norm(Xq - (pa + s[:, None] * ab), axis=1)
        ins |= d <= rads[a] + s * (rads[a + 1] - rads[a])
    return ins


def _cut_zone_hits(b, n_use, kd, sid_, rad_, pts_, cont_parent=None):
    """other-branch samples reaching into the cut zone of branch b cut at its sample n_use − 1 (cont_parent: the
    parent of a continuation, the same lumen: its samples behind the cut plane do not count)"""
    sm = np.asarray(b["samples"])[:n_use]
    p = P_ref[sm]
    if len(p) < 2:
        par = [x for x in branches if x["root"] == b["root"]]
        if b["parent"] < 0:
            return []
        p = np.vstack([P_ref[par[b["parent"]]["samples"][-1]], p])
    c, t = p[-1], p[-1] - p[-2]
    t = t / np.linalg.norm(t)
    R_ = float(R_ref[sm[-1]])
    slab = 3 * h
    reach = 1.3 * R_ + slab + R_ + h + float(rad_.max())
    out = []
    for j in kd.query_ball_point(c, reach):
        if sid_[j] == id(b):
            continue
        d = pts_[j] - c
        s_ = float(d @ t)
        if cont_parent is not None and sid_[j] == cont_parent and s_ < 0:
            continue
        lat = float(np.linalg.norm(d - s_ * t))
        rq = float(rad_[j]) - CUT_TOL
        thin = (-1.5 * h - rq <= s_ <= slab + 3 * h + rq) and lat < 1.3 * R_ + rq
        cap = (slab < s_ <= slab + h + R_ + rq) and lat < R_ + rq
        if thin or cap:
            out.append(j)
    return out


if "--cut-overlap" not in A:
    for _pass in range(4):
        live = [b for b in branches if not b.get("drop")]
        pts_ = np.vstack([P_ref[b["samples"]] for b in live])
        rad_ = np.concatenate([R_ref[b["samples"]] for b in live])
        sid_ = np.concatenate([np.full(len(b["samples"]), id(b), dtype=np.int64) for b in live])
        bname = {id(b): (enames[b["edge"]] if b["edge"] >= 0 else roots[b["root"]]["name"]) for b in live}
        kd_ = cKDTree(pts_)
        changed_ = False
        for b in live:
            if not b["end"]:
                continue
            n0 = len(b["samples"])
            par = [x for x in branches if x["root"] == b["root"]]
            sib = [x for x in par if x["parent"] == b["parent"] and not x.get("drop")]
            is_cont = b["parent"] < 0 or R_ref[b["samples"][0]] >= max(R_ref[x["samples"][0]] for x in sib)
            cpar = id(par[b["parent"]]) if (is_cont and b["parent"] >= 0) else None
            hits = _cut_zone_hits(b, n0, kd_, sid_, rad_, pts_, cpar)
            if not hits:
                continue
            who = sorted({bname[int(sid_[j])][:30] for j in hits})
            nm = bname[id(b)][:34]
            n_ok = None
            for n_ in range(n0 - 1, 1, -1):
                if not _cut_zone_hits(b, n_, kd_, sid_, rad_, pts_, cpar):
                    n_ok = n_
                    break
            # the slid cut keeps the junction rule's clearance from the parent (2.5 r + the parent's radius + 1 mm)
            if n_ok is not None and b["parent"] >= 0:
                r_par = float(R_ref[par[b["parent"]]["samples"][-1]])
                if arclen(np.asarray(b["samples"])[:n_ok]) < 2.5 * R_ref[b["samples"][n_ok - 1]] + r_par + 1e-3 and not is_cont:
                    n_ok = None
            if n_ok is not None:
                print(f"cut clearance: {nm} cut moved back {arclen(np.asarray(b['samples'])[n_ok - 1 :]) * 1e3:.1f} mm (its zone held {', '.join(who)})")
                b["samples"] = np.asarray(b["samples"])[:n_ok]
                changed_ = True
            elif not is_cont:
                print(f"cut clearance: {nm} left out (no clear cut position; its zone held {', '.join(who)})")
                b["drop"] = True
                changed_ = True
                # a parent left without onward branches ends in a cut itself
                if b["parent"] >= 0 and not any(x["parent"] == b["parent"] and not x.get("drop") for x in par):
                    pb = par[b["parent"]]
                    pb["end"] = True
                    pb["samples"] = _off_junction(np.asarray(pb["samples"]))
            elif not b.get("_kept"):
                b["_kept"] = True
                if "--debug-cuts" in A:
                    sm_ = np.asarray(b["samples"])
                    c_, t_ = P_ref[sm_[-1]], P_ref[sm_[-1]] - P_ref[sm_[-2]]
                    t_ = t_ / np.linalg.norm(t_)
                    for j in hits[:: max(1, len(hits) // 6)]:
                        d_ = pts_[j] - c_
                        print(f"    hit {bname[int(sid_[j])][:30]}: s {d_ @ t_ * 1e3:.1f} mm, lat {np.linalg.norm(d_ - (d_ @ t_) * t_) * 1e3:.1f} mm, r {rad_[j] * 1e3:.1f} mm; branch arclen {arclen(sm_) * 1e3:.1f} mm, R {R_ref[sm_[-1]] * 1e3:.1f} mm, {len(sm_)} samples, parent {b['parent']}")
                print(f"cut clearance: {nm} keeps its cut (root / continuation, no clear position; its zone holds {', '.join(who)})")
        if not changed_:
            break

# capsules (+ the slab beyond each cut, then capped)
stub = np.zeros_like(blood)
nzyx = np.array(blood.shape)
cuts = []
for bi, b in enumerate(branches):
    if b.get("drop"):
        continue
    r = roots[b["root"]]
    sm = b["samples"]
    p = P_ref[sm]
    rad = R_ref[sm]
    if b["parent"] >= 0 or len(sm) < 2:
        # joined to the parent's end (the shared node)
        par = [x for x in branches if x["root"] == b["root"]]
        pp = P_ref[par[b["parent"]]["samples"][-1]] if b["parent"] >= 0 else None
        if pp is not None:
            p = np.vstack([pp, p])
            rad = np.append(R_ref[par[b["parent"]]["samples"][-1]], rad)
    if len(p) < 2:
        continue
    t_end = p[-1] - p[-2]
    t_end /= np.linalg.norm(t_end)
    slab = 3 * h
    if b["end"]:
        pts = np.vstack([p, p[-1] + t_end * (slab + h)])
        rads = np.append(rad, rad[-1])
    else:
        pts, rads = p, rad
    for a in range(len(pts) - 1):
        pa, pb, ra = pts[a], pts[a + 1], max(rads[a], rads[a + 1])
        lo = np.floor((np.minimum(pa, pb) - ra - lo0) / h).astype(int) - 1
        hi = np.ceil((np.maximum(pa, pb) + ra - lo0) / h).astype(int) + 1
        lo, hi = np.maximum(lo, 0), np.minimum(hi, nzyx[::-1])
        if (hi <= lo).any():
            continue
        zz, yy, xx = np.meshgrid(*(np.arange(lo[d], hi[d]) for d in (2, 1, 0)), indexing="ij")
        X = np.stack([xx, yy, zz], -1) * h + lo0 + h / 2
        ab = pb - pa
        s = np.clip(((X - pa) @ ab) / max(ab @ ab, 1e-18), 0, 1)
        d = np.linalg.norm(X - (pa + s[..., None] * ab), axis=-1)
        rr = rads[a] + s * (rads[a + 1] - rads[a])
        stub[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]] |= d <= rr
    b["_poly"] = (pts, rads)
    if b["end"]:
        nm_ = r["name"] if b["parent"] == -1 and sum(x["root"] == b["root"] and x["end"] for x in branches) == 1 else f"{enames[b['edge']] if b['edge'] >= 0 else r['name']} #{int(sm[-1])}"
        cuts.append({"name": nm_, "class": r["class"], "center": p[-1].tolist(), "normal": t_end.tolist(), "radius": float(rad[-1]), "slab": slab,
                     "sample": int(sm[-1]), "sgn": int(b["sgn"]), "end": (p[-1] + t_end * slab).tolist(), "root": b["root"], "branch": bi})
print(f"{len(cuts)} cuts")
domain = blood | stub
_dbg("capsules")
# cut the stubs' slabs beyond the cap plane (capsule ends are round): this stub's own capsule voxels beyond the plane
# (owner = the stub of the nearest sample over its radius) and chamber blood just beyond it; another stub passing
# behind the plane keeps its lumen (the 2.5 R kill cylinder of the pulmonary bifurcation cut ate 15–35 % of the
# ascending aorta)
from scipy.spatial import cKDTree  # noqa: E402

zz, yy, xx = np.nonzero(domain)
X = np.stack([xx, yy, zz], -1) * h + lo0 + h / 2
is_stub = stub[zz, yy, xx]
# owner of a capsule voxel: the branch of the nearest sample over its radius
_pts = np.vstack([P_ref[b["samples"]] for b in branches if not b.get("drop")])
_prad = np.concatenate([R_ref[b["samples"]] for b in branches if not b.get("drop")])
_sid = np.concatenate([np.full(len(b["samples"]), bi) for bi, b in enumerate(branches) if not b.get("drop")])
owner = np.full(len(X), -1)
dd_, jj_ = cKDTree(_pts).query(X[is_stub], k=8)
owner[is_stub] = _sid[jj_[np.arange(len(jj_)), np.argmin(dd_ / _prad[jj_], axis=1)]]
for c in cuts:
    k_ = c["branch"]
    # beyond the end plane, but only up to max(4 R, 6 cells): a branch curving back towards the heart had its unbounded
    # kill cylinder run through the LA (11.8 mL removed, a pulmonary-vein tree cut off)
    _sb = (X - np.asarray(c["end"])) @ np.asarray(c["normal"])
    beyond = (_sb > 0) & (_sb <= max(4 * c["radius"], 6 * h))
    lat = np.linalg.norm((X - np.asarray(c["center"])) - ((X - np.asarray(c["center"])) @ np.asarray(c["normal"]))[:, None] * np.asarray(c["normal"]), axis=1)
    s_ = (X - np.asarray(c["center"])) @ np.asarray(c["normal"])
    # and anything not of this stub in the thin cylinder of its slab and flux layer (hasty-cfd decides slab, layer and
    # sponge membership in the reference pose within 1.25 / 1.1 R: a neighbour there would become outlet nodes)
    thin = (s_ > -1.5 * h) & (s_ <= c["slab"] + 3 * h) & (lat < 1.3 * c["radius"]) & (owner != k_)
    # a voxel inside this branch's own capsule is its own lumen whatever its nearest sample (2026-11-26: a 1.3 mm
    # continuation of the left subclavian artery past a junction lost the parent-owned voxels of its own lumen in its
    # slab, 76 % fluid, 3 m/s jets). `--cut-kill owner`: as before
    if CUT_KILL == "geom" and thin.any():
        ti_ = np.flatnonzero(thin)
        thin[ti_[_in_capsule(X[ti_], branches[k_]["_poly"])]] = False
    kill = (beyond & (((owner == k_) & (lat < 2.5 * c["radius"])) | (~is_stub & (lat < 1.5 * c["radius"])))) | thin
    spared = beyond & (lat < 2.5 * c["radius"]) & ~kill
    domain[zz[kill], yy[kill], xx[kill]] = False
    if _DBG and kill.any() and min(np.linalg.norm(X[kill] - np.asarray(q_[:3]), axis=1).min() for q_ in _DBG) < 8e-3:
        print(f"debug cut {c['name'][:30]} kills {int(kill.sum())} voxels within 8 mm of a debug point (beyond {int((kill & beyond).sum())}, thin {int((kill & thin).sum())})")
    if spared.sum() * h**3 * 1e6 > 0.05:
        print(f"cap {c['name'][:28]:28s}: {kill.sum() * h**3 * 1e6:.2f} mL removed beyond the slab, {spared.sum() * h**3 * 1e6:.2f} mL of other vessels / chambers spared")
_dbg("capsules + caps")
print(f"domain: {int(domain.sum())} voxels ({domain.sum() * h**3 * 1e6:.0f} mL), stubs {int((stub & ~blood).sum() * h**3 * 1e6)} mL")

# ── valves ───────────────────────────────────────────────────────────────────────────────────────────────────────────
tq = np.asarray(cm["flows_t"])

# velocity scale of the lattice (u_lattice at v_max): unless given, from the anatomy — the peak ejection flow through
# the narrowest lumen of each arterial stub (a stenosis, a small root), section mean, at least 2 m/s; the jet core
# (≈ 1.6× the section mean) then sits at u_lattice ≈ 0.13. A larger v_max lowers dt and with it τ0 towards ½: the
# ph0102 stenosis at v_max 5 (τ0 0.50032) had the same phase-averaged jet (3.9 vs 3.75 m/s) as at 2 (τ0 0.50079)
# but instantaneous node spikes to 22 m/s instead of 6.4
_vm = 2.0
for k_, r in enumerate(roots):
    if r["class"] in ("systemic_arterial", "pulmonary_arterial"):
        qpk = float(np.max(np.abs(np.asarray(cm["Q_AO" if r["class"] == "systemic_arterial" else "Q_PA"], float))))
        q0 = max(abs(float(np.real(qn[r["path"][0], 0]))), 1e-9)
        # each tree sample: the peak ejection × its share of the root's mean flow, through its lumen
        for b in branches:
            if b["root"] == k_ and not b.get("drop"):
                sm = b["samples"]
                v_ = qpk * np.minimum(np.abs(np.real(qn[sm, 0])) / q0, 1.0) / (np.pi * R_ref[sm] ** 2)
                _vm = max(_vm, float(v_.max()))


# --open-ramp causal (default 2026-11-27) | centred (before): see opening()
OPEN_RAMP = opt("--open-ramp", "causal")
assert OPEN_RAMP in ("causal", "centred"), "--open-ramp causal | centred"


def opening(Q, bridge=0.0):
    """1 while the flow exceeds 3 % of its peak, ramped over 30 ms; bridge > 0 (AV valves): closed gaps shorter than
    300 ms between two open phases (diastasis between E and A) stay open at least that much (the leaflets float
    half open in diastasis; a closed AV valve there seals the ventricle and the atrium into boxes the LBM compresses)"""
    Q = np.asarray(Q, float)
    # |Q|: the closing backflow (Q_AO −29 mL/s while the LV still expands) needs the valve open, a valve closed while
    # the motion still changes the chamber volume seals it into a box the LBM de-/pressurises (jet bursts on opening)
    # (--open-abs; tried on ph0001: the isovolumic pressure bursts stayed, the tricuspid's got worse)
    Qs = np.abs(Q) if OPEN_ABS else Q
    raw = (Qs > OPEN_THR * np.abs(Q).max()).astype(float)
    n = len(tq)
    if bridge > 0 and raw.any() and not raw.all():
        dtq = period / n
        k0 = int(np.argmin(raw))  # start the scan in a closed sample
        r = np.roll(raw, -k0)
        on = np.flatnonzero(r > 0.5)
        runs = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1)
        for a_, b_ in zip(runs[:-1], runs[1:]):
            if (b_[0] - a_[-1] - 1) * dtq < 0.3:
                r[a_[-1] + 1 : b_[0]] = np.maximum(r[a_[-1] + 1 : b_[0]], bridge)
        raw = np.roll(r, k0)
    w = max(1, int(round(0.030 / (period / n))))
    ker = np.ones(w) / w
    sm = np.real(np.fft.ifft(np.fft.fft(raw) * np.fft.fft(np.roll(np.pad(ker, (0, n - w)), -(w // 2)))))
    if OPEN_RAMP == "causal":
        # opening starts at the flow onset (the pressure crossover), not 15 ms before it: a valve opening into a chamber
        # that does not fill yet pushes its leaflets' volume back through the slit (2 m/s backward tricuspid jets at
        # 401 ms on full17); the closing keeps the centred ramp (leaflets close while the flow decelerates)
        sk = np.real(np.fft.ifft(np.fft.fft(raw) * np.fft.fft(np.pad(ker, (0, n - w)))))
        sm = np.minimum(sm, sk)
    return {"t": tq.tolist(), "v": np.clip(sm, 0, 1).tolist()}


def perp(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e = np.cross(n, a)
    return e / np.linalg.norm(e)


valves = []
# semilunar hinge (--sl-hinge): "lumen" (default, 2026-11-27) puts the hinge circle half a leaflet thickness + half a cell
# outside the lumen, so the open cusps lie in the wall like cusps opening into the sinuses of Valsalva and the orifice is
# the lumen (healthy aortic peak velocity 1.0–1.7 m/s); "annulus" (before): hinge at 1.05 × the lumen radius, the open
# 3-cell cusps stood inside it (aortic 2.85 of 4.4 cm² open on full17, jet 2.8 m/s at 500 mL/s)
SL_HINGE = opt("--sl-hinge", "lumen")
# --leaflets plates | membrane (hasty-cfd "membrane": zero-thickness leaflets — every lattice link crossing a leaflet or
# the closed disk bounces back — instead of partially saturated plates three cells thick, which act as sharp-edged
# orifices (EOA / GOA ≈ 0.5–0.6) and push their own volume through the valve as they swing)
LEAFLETS = opt("--leaflets", "membrane")  # plates before 2026-11-27
assert LEAFLETS in ("plates", "membrane"), "--leaflets plates | membrane"
SL_SINUS = opt("--sl-sinus", "none" if LEAFLETS == "membrane" else "carve")
assert SL_SINUS in ("none", "carve"), "--sl-sinus none | carve"
# --theta-open-deg sl,av: the open leaflets' angle from the annulus plane (default 90,85 — measured: mitral jet −12 %; 90 = parallel to the axis: a
# strip leaning in by 90° − θ reaches (90° − θ) × its length into the orifice, e.g. 0.2 R for the long mitral strips at 80°)
TH_SL, TH_AV = (float(x) for x in opt("--theta-open-deg", "90,85").split(","))  # 85,80 before 2026-11-27
assert SL_HINGE in ("lumen", "annulus"), "--sl-hinge lumen | annulus"
for r in roots:
    if r["class"] not in ("systemic_arterial", "pulmonary_arterial"):
        continue
    s0, s1 = r["path"][0], r["path"][min(2, len(r["path"]) - 1)]
    n = P_ref[s1] - P_ref[s0]
    n /= np.linalg.norm(n)
    q = cm["Q_AO"] if r["class"] == "systemic_arterial" else cm["Q_PA"]
    th_l = max(1.5e-3, 3 * DX)
    r_l = float(R_ref[s0] * 1.05)
    r_h = r_l + 0.5 * th_l + 0.5 * DX if SL_HINGE == "lumen" else r_l
    valves.append({"_root": r, "name": "aortic" if r["class"] == "systemic_arterial" else "pulmonary", "kind": VALVES, "center": P_ref[s0].tolist(), "normal": n.tolist(),
                   "e1": perp(n).tolist(), "radius": r_h, "lumen_radius": r_l, "ring": max(1e-3, 2 * DX), "thickness": th_l, "theta_open": float(np.radians(TH_SL)),
                   "leaflets": "tri:90,4", "hole": [0.85 * r_l / r_h] * 2, "opening": opening(q)})
for ca, cv, nm, q, lf in (("blood_LA", "blood_LV", "mitral", cm["Q_MV"], "bi:0.2,5,7"), ("blood_RA", "blood_RV", "tricuspid", cm["Q_TV"], "tri:90,4")):
    A_ = Lc == ID[ca]
    B_ = Lc == ID[cv]
    faces = []
    for ax in range(3):
        for sh in (-1, 1):
            nb = np.roll(B_, sh, axis=ax)
            m = A_ & nb
            idx = np.argwhere(m)
            if len(idx):
                off = np.zeros(3)
                off[ax] = 0.5 * sh
                faces.append(idx + off)
    if not faces:
        print(f"no {ca}–{cv} interface: no {nm} valve")
        continue
    F = np.vstack(faces)[:, ::-1]  # x, y, z index
    X = F * h + lo0 + h / 2
    c = X.mean(0)
    u, sv, vt = np.linalg.svd(X - c, full_matrices=False)
    nrm = vt[2]
    cA, cB = centre(np.argwhere(A_)[:, ::-1].mean(0)), centre(np.argwhere(B_)[:, ::-1].mean(0))
    if (cB - cA) @ nrm < 0:
        nrm = -nrm
    area = len(F) * h * h
    cap = float(opt("--av-radius-mm", "16,17").split(",")[0 if nm == "mitral" else 1]) * 1e-3
    rad = float(min(np.sqrt(area / np.pi) * 1.1, cap))
    # leaflets keep ≥ 3 cells (a 2-cell leaflet is mostly partially saturated cells: porous, 4 m/s through-jets at the
    # tricuspid in late E on ph0001)
    # centre on the passage: the contact centroid of irregular atlas chambers can sit beside the actual inflow tract (on
    # ph0001 the mitral disk was 7 mm off it: 31 % myocardium in the disk, 30 % 4 mm into the LV); the blood (atrium,
    # ventricle, valve tissue) centroid over the planes −3…+8 mm within 1.3 R, iterated
    e1p = perp(nrm)
    e2p = np.cross(nrm, e1p)
    c0 = c.copy()
    bids = [ID[ca], ID[cv], ID["valve"]]
    ss = np.arange(-1.3 * rad, 1.3 * rad, h / 2)
    Ag, Bg = np.meshgrid(ss, ss, indexing="ij")
    mg = Ag**2 + Bg**2 < (1.3 * rad) ** 2
    for _ in range(6):
        acc_ = []
        for offp in np.arange(-3e-3, 8.01e-3, 1e-3):
            Pp = c + offp * nrm + Ag[mg][:, None] * e1p + Bg[mg][:, None] * e2p
            li = np.clip(np.floor((Pp - lo0) / h).astype(int), 0, np.array(Lc.shape[::-1]) - 1)
            okp = np.isin(Lc[li[:, 2], li[:, 1], li[:, 0]], bids)
            acc_.append(Pp[okp] - offp * nrm)
        cn = np.vstack(acc_).mean(0)
        cn = cn - ((cn - c) @ nrm) * nrm
        moved = np.linalg.norm(cn - c)
        c = cn
        if moved < 0.2e-3:
            break
    print(f"{nm}: centred on the passage, {np.linalg.norm(c - c0) * 1e3:.1f} mm from the contact centroid")
    valves.append({"name": nm, "kind": VALVES, "center": c.tolist(), "normal": nrm.tolist(), "e1": vt[0].tolist(), "radius": rad, "ring": max(1e-3, 2 * DX),
                   "thickness": max(1.5e-3, 3 * DX), "theta_open": float(np.radians(TH_AV)), "leaflets": lf, "hole": [0.9, 0.6] if nm == "mitral" else [0.85, 0.85],
                   "opening": opening(q, AV_BRIDGE)})
    print(f"{nm}: centre {np.round(c * 1e3, 1)} mm, normal {np.round(nrm, 2)}, interface {area * 1e4:.2f} cm² → radius {rad * 1e3:.1f} mm")

# semilunar axis (--sl-axis): "channel" (default, 2026-11-27) — the line from the outflow tract's centroid 20 mm upstream
# (ventricle voxels around that depth within the tract's reach, the AV valves' inflow tracts left out) to the vessel
# centreline two lumen diameters downstream; "stub" (before) — the first two centreline samples. On full17 the stub axis
# was 30–38° off the LVOT–ascending-aorta channel, which runs straight: the flow crossed the valve obliquely and left as
# an eccentric, bicuspid-like jet (2.4–2.5 m/s, displacement 0.27).
SL_AXIS = opt("--sl-axis", "channel")
assert SL_AXIS in ("channel", "stub"), "--sl-axis channel | stub"
for v in valves:
    r = v.pop("_root", None)
    if r is None or SL_AXIS == "stub":
        continue
    c_, n0 = np.asarray(v["center"]), np.asarray(v["normal"])
    rl = float(v["lumen_radius"])
    # downstream: the centreline point at arc length 4 r_l
    pp = P_ref[r["path"]]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pp, axis=0), axis=1))]
    c_dn = np.array([np.interp(4 * rl, arc, pp[:, k]) for k in range(3)])
    # upstream: the ventricle's outflow tract around 20 mm before the valve
    vent = ID["blood_LV"] if v["name"] == "aortic" else ID["blood_RV"]
    iv = np.argwhere(Lc == vent)
    Xv = iv[:, ::-1] * h + lo0 + h / 2
    dv = -((Xv - c_) @ n0)
    latv = np.linalg.norm((Xv - c_) + dv[:, None] * n0, axis=1)
    sel = (np.abs(dv - 20e-3) < 3e-3) & (latv < 1.6 * rl + 10e-3)
    for w in valves:
        if w["name"] in ("mitral", "tricuspid"):
            cw, nw = np.asarray(w["center"]), np.asarray(w["normal"])
            ow = (Xv - cw) @ nw
            lw = np.linalg.norm((Xv - cw) - ow[:, None] * nw, axis=1)
            sel &= ~((ow > 0) & (ow < 25e-3) & (lw < w["radius"] + 4e-3))
    if sel.sum() < 20:
        print(f"{v['name']}: no outflow tract found 20 mm upstream — stub axis kept")
        continue
    c_up = Xv[sel].mean(0)
    n1 = c_dn - c_up
    n1 /= np.linalg.norm(n1)
    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    print(f"{v['name']}: channel axis {ang(n0, n1):.0f}° from the stub axis (outflow tract approach {ang(n0, c_ - c_up):.0f}°, vessel at 2 diameters {ang(n0, c_dn - c_):.0f}°)")
    v["normal"] = n1.tolist()
    v["e1"] = perp(n1).tolist()

if LEAFLETS == "membrane":
    for v in valves:
        if v["kind"] == "leaflets":
            v["membrane"] = True

# ── valve disease (valve_disease.py; --valve-disease <spec>, default the phantom's params.json "pathology" →
#    "valves"): a stenosis opens the leaflets to the angle of the target orifice area, a regurgitation keeps the closed
#    valve's regurgitant orifice open (hasty-cfd "regurg_hole"); the cardiac model's flows carry the regurgitant volume
#    and the slow E wave already. The jets enter the velocity scale.
sys.path.insert(0, str(HERE))
import valve_disease as VD  # noqa: E402

_pp = P0 / "params.json"
VDIS = VD.parse(opt("--valve-disease") if "--valve-disease" in A else (json.loads(_pp.read_text()).get("pathology", {}).get("valves") if _pp.exists() else None))
def leaflet_polys(spec):
    """the hasty-cfd leaflet presets (case.rs presets) in valve-local 2D, units of R"""
    kind, args = (spec.split(":", 1) + [""])[:2]
    nums = [float(x) for x in args.split(",") if x]
    out = []
    if kind == "tri":
        a0, st = np.radians(nums[0] if nums else 90.0), int(nums[1]) if len(nums) > 1 else 4
        for k in range(3):
            a_, b_ = a0 + k * 2 * np.pi / 3, a0 + (k + 1) * 2 * np.pi / 3
            for j in range(st):
                s0, s1 = a_ + (b_ - a_) * j / st, a_ + (b_ - a_) * (j + 1) / st
                out.append([[np.cos(s0), np.sin(s0)], [np.cos(s1), np.sin(s1)], [0.0, 0.0]])
    elif kind == "bi":
        yc = nums[0] if nums else 0.2
        na, npo = (int(nums[1]) if len(nums) > 1 else 5), (int(nums[2]) if len(nums) > 2 else 7)
        tc = np.arcsin(np.clip(yc, -0.9, 0.9))
        for a_, b_, n_ in ((tc, np.pi - tc, na), (np.pi - tc, 2 * np.pi + tc, npo)):
            for j in range(n_):
                s0, s1 = a_ + (b_ - a_) * j / n_, a_ + (b_ - a_) * (j + 1) / n_
                p0, p1 = [np.cos(s0), np.sin(s0)], [np.cos(s1), np.sin(s1)]
                out.append([p0, p1, [p1[0], yc], [p0[0], yc]])
    else:
        raise SystemExit(f"leaflets {spec}: tri | bi")
    return [np.asarray(q, float) for q in out]


def open_area(polys, theta, th_r, n=241, r_in=1.0):
    """open share of the valve disk seen along its axis with every strip turned by θ about its hinge chord (as the
    kernel: strip coordinates a along the chord, b along the inward normal m0 rotated towards n, thickness th_r in
    units of R, incl. the ½-cell ramp): a column (x, y) is blocked when some strip point (a, b) of the polygon with
    |h| ≤ th/2 projects onto it, i.e. b ∈ [(c − ½th sin θ) / cos θ, (c + ½th sin θ) / cos θ] meets the polygon's b-range at a"""
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g, indexing="ij")
    # (r_in < 1: only the lumen disk counts, the hinge circle reaching into the wall; the share stays of the full disk)
    inside = X**2 + Y**2 < r_in**2
    P = np.stack([X[inside], Y[inside]], -1)
    blocked = np.zeros(len(P), bool)
    ct, st = np.cos(theta), np.sin(theta)
    for q in polys:
        pa, d = q[0], q[1] - q[0]
        d = d / np.linalg.norm(d)
        m0 = np.array([-d[1], d[0]])
        if (q.mean(0) - pa) @ m0 < 0:
            m0 = -m0
        A_ = (q - pa) @ d
        B_ = (q - pa) @ m0
        a = (P - pa) @ d
        c = (P - pa) @ m0
        # the polygon's b-range at each a (convex polygon: intersect the edges)
        bmin = np.full(len(P), np.inf)
        bmax = np.full(len(P), -np.inf)
        for i in range(len(q)):
            a0_, a1_, b0_, b1_ = A_[i], A_[(i + 1) % len(q)], B_[i], B_[(i + 1) % len(q)]
            lo_, hi_ = min(a0_, a1_), max(a0_, a1_)
            m_ = (a >= lo_) & (a <= hi_)
            f_ = np.where(abs(a1_ - a0_) > 1e-12, (a - a0_) / (a1_ - a0_ + 1e-30), 0.0)
            bb_ = b0_ + f_ * (b1_ - b0_)
            bmin = np.where(m_, np.minimum(bmin, bb_), bmin)
            bmax = np.where(m_, np.maximum(bmax, bb_), bmax)
        lo_b = (c - 0.5 * th_r * st) / max(ct, 1e-6)
        hi_b = (c + 0.5 * th_r * st) / max(ct, 1e-6)
        blocked |= (bmax >= bmin) & (hi_b >= bmin) & (lo_b <= bmax)
        # the circular segment between the hinge chord and the annulus stays solid (the kernel's fixed ring part)
        blocked |= (a >= 0) & (a <= np.linalg.norm(q[1] - q[0])) & (c < 0)
    return float((1 - blocked.mean()) * r_in**2)


def leaf_band(v):
    """the band a leaflet blocks across its plane (m): plates their thickness + one cell (the ½-cell ramps), membranes
    half a cell (the bounce-back wall sits on the membrane, ± half a link)"""
    return 0.5 * DX if v.get("membrane") else v["thickness"] + DX


def theta_for_area(v, area_geo):
    """the opening angle whose open area (open_area, the leaflets' blocked band) equals area_geo, at most the normal angle"""
    polys = leaflet_polys(v["leaflets"])
    th_r = leaf_band(v) / v["radius"]
    tot = np.pi * v["radius"] ** 2
    th0 = v["theta_open"]
    ri = lumen_share(v)
    if open_area(polys, th0, th_r, r_in=ri) * tot <= area_geo:
        return th0, open_area(polys, th0, th_r, r_in=ri) * tot
    lo_, hi_ = 0.0, th0
    for _ in range(30):
        mid = 0.5 * (lo_ + hi_)
        if open_area(polys, mid, th_r, r_in=ri) * tot < area_geo:
            lo_ = mid
        else:
            hi_ = mid
    return hi_, open_area(polys, hi_, th_r, r_in=ri) * tot


def lumen_share(v):
    """the lumen radius / the hinge radius (semilunar hinge outside the lumen, --sl-hinge lumen; else 1)"""
    return v.get("lumen_radius", v["radius"]) / v["radius"]


def open_cm2(v, theta=None):
    """the valve's open area (cm²) at θ (default its opening angle), inside the lumen"""
    return open_area(leaflet_polys(v["leaflets"]), v["theta_open"] if theta is None else theta, leaf_band(v) / v["radius"], r_in=lumen_share(v)) * np.pi * v["radius"] ** 2 * 1e4


_vjet = 0.0
for v in valves:
    les = VDIS.get(v["name"], {})
    _ql = np.asarray(cm[{"aortic": "Q_AO", "pulmonary": "Q_PA", "mitral": "Q_MV", "tricuspid": "Q_TV"}[v["name"]]], float)
    print(f"{v['name']}: open {open_cm2(v):.2f} cm² of the {np.pi * v.get('lumen_radius', v['radius']) ** 2 * 1e4:.2f} cm² lumen at {np.degrees(v['theta_open']):.0f}° "
          f"(hinge radius {v['radius'] * 1e3:.1f} mm), model peak {_ql.max() * 1e6:.0f} mL/s → mean orifice velocity {_ql.max() / (open_cm2(v) * 1e-4):.2f} m/s")
    qv = np.asarray(cm[{"aortic": "Q_AO", "pulmonary": "Q_PA", "mitral": "Q_MV", "tricuspid": "Q_TV"}[v["name"]]], float)
    if "bicuspid" in les:
        # two cusps (strip fans as the mitral preset), the coaptation line at the offset; set before the stenosis angle
        v["leaflets"] = f"bi:{les['bicuspid']:g},6,6"
        print(f"{v['name']}: bicuspid, leaflets {v['leaflets']}")
    if "stenosis" in les:
        th0 = v["theta_open"]
        # the spec's area is the effective orifice (continuity / Gorlin, as graded clinically): geometric = EOA / 0.75;
        # the open area from the strips' actual projection incl. their thickness (2026-11-26: πR²(1 − cos θ) counted
        # the slits between strips, which 3-cell leaflets close — an "1.0 cm²" aortic valve opened ≈ 0.2 cm² and the LV
        # pressurised to a doubled density). `--stenosis-formula`: the old cos rule
        if "--stenosis-formula" in A:
            v["theta_open"] = VD.open_angle(les["stenosis"] / VD.CC, v["radius"], th0)
            a_open = np.pi * v["radius"] ** 2 * (1 - np.cos(v["theta_open"]))
        else:
            v["theta_open"], a_open = theta_for_area(v, les["stenosis"] * 1e-4 / VD.CC)
            if a_open < 0.9 * les["stenosis"] * 1e-4 / VD.CC:
                print(f"{v['name']}: WARNING the leaflets reach only {a_open * 1e4:.2f} cm² open")
        print(f"{v['name']}: normal leaflets open {open_cm2(v, th0):.2f} cm² of {np.pi * v.get('lumen_radius', v['radius']) ** 2 * 1e4:.2f} cm² at {np.degrees(th0):.0f}°")
        _vjet = max(_vjet, float(qv.max()) / (VD.CC * a_open))
        print(f"{v['name']}: stenosis {les['stenosis']:g} cm² → leaflets open to {np.degrees(v['theta_open']):.0f}° (normal {np.degrees(th0):.0f}°), jet ≈ {qv.max() / (VD.CC * a_open):.1f} m/s")
    if "regurgitation" in les:
        v["regurg_hole"] = VD.closed_hole(v["name"], les["regurgitation"], v["radius"], yc=0.2)
        _vjet = max(_vjet, float(-qv.min()) / (les["regurgitation"] * 1e-4))
        print(f"{v['name']}: regurgitation EROA {les['regurgitation']:g} cm² → closed-valve hole {np.round(v['regurg_hole'], 3).tolist()} (units of R {v['radius'] * 1e3:.1f} mm), "
              f"peak backflow {-qv.min() * 1e6:.0f} mL/s, jet ≈ {-qv.min() / (les['regurgitation'] * 1e-4):.1f} m/s")
# the geometric open area (inside the lumen) at the final opening angle, and a stenosis's target effective area: the
# report compares the LBM's effective orifice area (peak flow / peak jet velocity, the echo continuity equation) to them
for v in valves:
    v["open_area_cm2"] = float(open_cm2(v))
    if "stenosis" in VDIS.get(v["name"], {}):
        v["target_eoa_cm2"] = float(VDIS[v["name"]]["stenosis"])
REGURG_SIDES = VD.regurgitant_sides(VDIS)
V_MAX = float(opt("--v-max", np.ceil(max(_vm, 1.1 * _vjet) * 2) / 2))
print(f"velocity scale v_max {V_MAX} m/s (anatomy estimate {_vm:.2f} m/s" + (f", valve jets {_vjet:.2f} m/s" if _vjet else "") + ")")

# ── annulus openings: the valve defines the orifice. The capped mitral disk overlapped 31 % atlas myocardium (the
# LA–LV contact is irregular, 4.9 mm out of plane), the aortic one 13 % non-domain: carve each valve's cylinder
# (lateral < R, |offset| ≤ CARVE) into the blood; its voxels join the valve's two sides by the plane below
carved = {}
for v in valves:
    c_, n_ = np.asarray(v["center"]), np.asarray(v["normal"])
    rr_ = v["radius"] + CARVE_RIM + 2 * h
    lo_ = np.maximum(np.floor((c_ - rr_ - lo0) / h).astype(int) - 1, 0)
    hi_ = np.minimum(np.ceil((c_ + rr_ - lo0) / h).astype(int) + 2, np.array(domain.shape[::-1]))
    zz_, yy_, xx_ = np.meshgrid(*(np.arange(lo_[d], hi_[d]) for d in (2, 1, 0)), indexing="ij")
    Xc = np.stack([xx_, yy_, zz_], -1) * h + lo0 + h / 2
    off_ = (Xc - c_) @ n_
    lat_ = np.linalg.norm((Xc - c_) - off_[..., None] * n_, axis=-1)
    # radius + 2 mm: open leaflets (hinged at R, near-axial) must not hug the cylinder wall — moving to half open they
    # left a thin gap the fluid rushed into (3–6 m/s for a few ms at late E on ph0001); the collar ring stays solid
    m_ = (lat_ < v["radius"] + CARVE_RIM) & (off_ >= -CARVE) & (off_ <= CARVE_DOWN)
    # --sl-sinus none (default with membrane leaflets, 2026-11-27): a sinus-hinged semilunar valve's carve stops at the
    # lumen downstream of the valve plane — the R + rim pocket ended 4 mm on in a step down to the lumen (8.5 → 4.9 cm²
    # aortic on full17), a sharp-edged contraction right after the valve; membranes need no clearance there
    if SL_SINUS == "none" and "lumen_radius" in v and v["lumen_radius"] < v["radius"]:
        m_ &= (off_ <= 1.5 * h) | (lat_ < v["lumen_radius"] + 0.5 * h)
    mk = np.zeros(domain.shape, bool)
    mk[lo_[2] : hi_[2], lo_[1] : hi_[1], lo_[0] : hi_[0]] = m_
    new_ = mk & ~domain
    carved[v["name"]] = mk
    domain |= mk
    print(f"{v['name']}: annulus cylinder carved, {new_.sum() * h**3 * 1e6:.2f} mL of non-blood opened ({new_.sum() / max(mk.sum(), 1) * 100:.0f} % of the cylinder)")
carved_any = np.zeros(domain.shape, bool)
_dbg("annulus cylinders")
for m_ in carved.values():
    carved_any |= m_

# ── regions and walls ────────────────────────────────────────────────────────────────────────────────────────────────
# region per domain voxel: chambers 1–4 (LA, LV, RA, RV; valve tissue → the nearest chamber), stub k → 10 + k (stub
# voxels outside the chambers, by the nearest stub sample); near each valve the boundary between its upstream and
# downstream regions is the valve plane; walls wherever two regions touch, except venous stub ↔ its atrium (free) and a
# valve's upstream ↔ downstream inside the valve disk; a solid collar around each annulus
ST = ndimage.generate_binary_structure(3, 1)
CHI = {c: k_ + 1 for k_, c in enumerate(CH)}
region = np.zeros(domain.shape, np.int16)
for c in CH:
    region[(Lc == ID[c]) & domain] = CHI[c]
known = region > 0
_, ind = ndimage.distance_transform_edt(~known, return_indices=True)
unk = domain & ~known & blood
region[unk] = region[tuple(ind)][unk]
# stub voxels: by the nearest stub sample (distance over the sample radius); the vessel lumen takes precedence over a
# chamber label of *another* chamber (the atlas chamber meshes overlap the great-vessel roots: the aortic root inside
# LA / RA / RV labels got walled and pinched to half its lumen); its own chamber keeps its voxels (the valve plane
# below, or the free venous junction)
from scipy.spatial import cKDTree  # noqa: E402

pts = np.vstack([P_ref[b["samples"]] for b in branches if not b.get("drop")])
prad = np.concatenate([R_ref[b["samples"]] for b in branches if not b.get("drop")])
sid = np.concatenate([np.full(len(b["samples"]), 10 + b["root"]) for b in branches if not b.get("drop")])
own = {10 + k_: CHI[r["chamber"]] for k_, r in enumerate(roots)}
STUB_FIRST = "--chamber-first" not in A
sv_idx = np.argwhere(domain & ~carved_any & ((region == 0) | (stub if STUB_FIRST else False)))
if len(sv_idx):
    Xs = sv_idx[:, ::-1] * h + lo0 + h / 2
    dd, jj = cKDTree(pts).query(Xs, k=8)
    jbest = jj[np.arange(len(Xs)), np.argmin(dd / prad[jj], axis=1)]
    sreg = sid[jbest]
    cur = region[tuple(sv_idx.T)]
    # only beyond the root plane (the capsule's half-ball at the root lies in the chamber it opens into)
    root_p = {10 + k_: P_ref[r["path"][0]] for k_, r in enumerate(roots)}
    root_t = {10 + k_: (P_ref[r["path"][min(2, len(r["path"]) - 1)]] - P_ref[r["path"][0]]) / np.linalg.norm(P_ref[r["path"][min(2, len(r["path"]) - 1)]] - P_ref[r["path"][0]]) for k_, r in enumerate(roots)}
    beyond = np.einsum("ij,ij->i", Xs - np.array([root_p[x] for x in sreg]), np.array([root_t[x] for x in sreg])) > 0
    # the half-ball at the root lies within 1.5 r_root of it: a curved vessel coming back behind its root plane further
    # away is still its own lumen (2026-11-26: the coronary sinus, curving along the AV groove, lost lumen voxels
    # labelled myocardium behind its root plane and was pinched to 0.39 r)
    root_r = {10 + k_: float(R_ref[r["path"][0]]) for k_, r in enumerate(roots)}
    beyond |= np.linalg.norm(Xs - np.array([root_p[x] for x in sreg]), axis=1) > 1.5 * np.array([root_r[x] for x in sreg])
    # arterial stubs also take their own ventricle's labels beyond the valve plane (the atlas LVOT / RVOT labels reach
    # into the root: kept as ventricle they got walled off inside the lumen beyond the valve-plane split zone)
    arterial = np.array([roots[x - 10]["class"] in ("systemic_arterial", "pulmonary_arterial") for x in sreg])
    take = (cur == 0) | (beyond & ((cur != np.array([own[x] for x in sreg])) | arterial))
    claimed = {}
    for a_, b_ in zip(sreg[take & (cur > 0)], cur[take & (cur > 0)]):
        claimed[(int(a_), int(b_))] = claimed.get((int(a_), int(b_)), 0) + 1
    region[tuple(sv_idx[take].T)] = sreg[take]
    for (a_, b_), nv_ in sorted(claimed.items(), key=lambda z: -z[1]):
        if nv_ * h**3 * 1e6 >= 0.05:
            print(f"stub {roots[a_ - 10]['name'][:28]:28s} takes {nv_ * h**3 * 1e6:.2f} mL of {CH[b_ - 1]} labels")
stub_region = {r["name"]: 10 + k_ for k_, r in enumerate(roots)}
_dbg("regions (stubs)")
up_down = {}
for v in valves:
    if v["name"] == "aortic":
        up_down[v["name"]] = (CHI["blood_LV"], next(10 + k_ for k_, r in enumerate(roots) if r["class"] == "systemic_arterial"))
    elif v["name"] == "pulmonary":
        up_down[v["name"]] = (CHI["blood_RV"], next(10 + k_ for k_, r in enumerate(roots) if r["class"] == "pulmonary_arterial"))
    elif v["name"] == "mitral":
        up_down[v["name"]] = (CHI["blood_LA"], CHI["blood_LV"])
    elif v["name"] == "tricuspid":
        up_down[v["name"]] = (CHI["blood_RA"], CHI["blood_RV"])


def frame(v, idx):
    X = idx[:, ::-1] * h + lo0 + h / 2
    c, n = np.asarray(v["center"]), np.asarray(v["normal"])
    off = (X - c) @ n
    lat = np.linalg.norm((X - c) - off[:, None] * n, axis=1)
    return off, lat


for v in valves:
    U_, D_ = up_down[v["name"]]
    idx = np.argwhere(domain & (np.isin(region, [U_, D_]) | carved[v["name"]]))
    off, lat = frame(v, idx)
    # the plane split covers the valve's passage only (lateral < R + rim + 2 voxels, the carved cylinder): beyond it the
    # atrium / ventricle boundary follows the labels (walls wherever the regions touch, so the seal does not depend on
    # it). The old zone (R + 6 mm, ±20 mm; `--split-zone wide`) flipped the atrium voxels at the coronary sinus ostium,
    # 7 mm below the flat tricuspid plane, into the RV: the proximity walls then pinched the sinus to 0.39 r (2026-11-26)
    if SPLIT_ZONE == "wide":
        nb = (lat < v["radius"] + 6e-3) & (np.abs(off) < 20e-3)
    else:
        nb = (lat < v["radius"] + CARVE_RIM + 2 * h) & (np.abs(off) < 20e-3)
    region[tuple(idx[nb & (off <= 0)].T)] = U_
    region[tuple(idx[nb & (off > 0)].T)] = D_
_dbg("regions (valve planes)")
# allowed contacts
nreg = int(region.max()) + 1
allowed = np.zeros((nreg, nreg), bool)
for k_, r in enumerate(roots):
    if r["class"] in ("systemic_venous", "pulmonary_venous"):
        a_, b_ = 10 + k_, CHI[r["chamber"]]
        allowed[a_, b_] = allowed[b_, a_] = True
disk = np.zeros(domain.shape, bool)
for v in valves:
    idx = np.argwhere(domain)
    off, lat = frame(v, idx)
    m = (np.abs(off) <= 1.0 * h) & (lat < v["radius"])
    disk[tuple(idx[m].T)] = True
wall = np.zeros_like(domain)
# every lattice link (face and edge neighbours, D3Q19): regions meeting only along an edge are linked too
for sh3 in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if 1 <= abs(a) + abs(b) + abs(c) <= 2]:
    rb = np.roll(region, sh3, axis=(0, 1, 2))
    db = np.roll(disk, sh3, axis=(0, 1, 2))
    touch = (region > 0) & (rb > 0) & (region != rb)
    ok = allowed[region, rb]
    for v in valves:
        U_, D_ = up_down[v["name"]]
        pair = ((region == U_) & (rb == D_)) | ((region == D_) & (rb == U_))
        ok |= pair & (disk | db)
    wall |= touch & ~ok
# the two voxel layers at an interface are 1 mm; the smoothed SDF, the motion and the lattice sampling thin oblique
# walls below a lattice spacing, and links cross them (pinhole jets at 4–5 m/s): grow by WALL_GROW voxels each side
nw0 = int(wall.sum())
_dbg("interface walls")
if WALL_GROW > 0:
    wall = ndimage.binary_dilation(wall, ST, iterations=WALL_GROW) & domain
# proximity walls: regions that must not exchange fluid (not a venous junction, not a valve's two sides) and come
# within SEP voxels of each other, through a thin atlas gap rather than a contact, got no wall; under the motion the
# gap closed (RA ↔ aortic root, RA ↔ right superior PV, aortic ↔ pulmonary root leaked in the lattice). Every voxel
# within SEP voxels of such a region becomes wall
vp_pairs = {tuple(sorted(up_down[v["name"]])) for v in valves}
nprox = 0
if SEP > 0:
    rids = [int(x) for x in np.unique(region) if x > 0]
    for a_ in rids:
        ma = region == a_
        sl = ndimage.find_objects(ma.astype(np.int8))[0]
        sl = tuple(slice(max(s_.start - SEP - 1, 0), s_.stop + SEP + 1) for s_ in sl)
        da = ndimage.distance_transform_edt(~ma[sl])
        rb = region[sl]
        bad = np.zeros(rb.shape, bool)
        for b_ in rids:
            if b_ == a_ or allowed[a_, b_] or tuple(sorted((a_, b_))) in vp_pairs:
                continue
            bad |= rb == b_
        if SEP_SPLIT == "half":
            depth = ndimage.distance_transform_edt(bad)  # a voxel's depth in the disallowed regions (1 at their surface)
            add = bad & (da + depth <= SEP + 1) & domain[sl] & ~wall[sl]
        else:
            add = bad & (da <= SEP) & domain[sl] & ~wall[sl]
        nprox += int(add.sum())
        wall[sl] |= add
domain &= ~wall
_dbg("walls + proximity")
region[~domain] = 0
print(f"walls between regions: {nw0} interface voxels, {int(wall.sum())} after growing {WALL_GROW} voxel(s) each side ({(2 + 2 * WALL_GROW) * h * 1e3:.1f} mm) and {nprox} proximity voxels (separation ≥ {SEP} voxels = {SEP * h * 1e3:.1f} mm, split {SEP_SPLIT})")
for v in valves:
    idx = np.argwhere(domain)
    off, lat = frame(v, idx)
    col = (np.abs(off) <= max(0.75e-3, 1.5 * h)) & (lat >= v["radius"]) & (lat <= v["radius"] + 4e-3)
    domain[tuple(idx[col].T)] = False
region[~domain] = 0
_dbg("collars")
# outflow-tract funnel (--ot-funnel L_mm,k, e.g. 20,1.6; default 0 = off: measured worse, see flow_phantom.md): the atlas ventricle met the
# semilunar passage through a step (full17 LVOT ≈ 7.1 cm² up to the annulus, lumen 4.9 cm²: a sudden contraction, flow
# entering 30° off the axis, a 2.4 m/s eccentric jet in a healthy root). A real LVOT / RVOT narrows over ≈ 2 cm to the
# annulus. Within L upstream of each semilunar valve, ventricle voxels outside the cone r(d) = r_lumen (1 + (k − 1)
# (d / L)²) become wall (d the upstream distance; the plane / collar layer and the other valves' carved passages kept)
_otf = opt("--ot-funnel", "0")
if _otf not in ("0", "off"):
    OT_L, OT_K = float(_otf.split(",")[0]) * 1e-3, float(_otf.split(",")[1]) if "," in _otf else 1.6
    for v in valves:
        if v["name"] not in ("aortic", "pulmonary"):
            continue
        U_, D_ = up_down[v["name"]]
        rl = float(v.get("lumen_radius", v["radius"]))
        # kept: the other valves' carved passages and the AV valves' inflow tracts (their passage cylinder 25 mm into
        # the ventricle: the LVOT lies beside the mitral inflow, aorto-mitral continuity)
        idx = np.argwhere(domain & (region == U_))
        keep = np.zeros(len(idx), bool)
        for w in valves:
            if w is v:
                continue
            keep |= carved[w["name"]][tuple(idx.T)]
            if w["name"] in ("mitral", "tricuspid"):
                ow, lw = frame(w, idx)
                keep |= (ow > 0) & (ow < 25e-3) & (lw < w["radius"] + CARVE_RIM + 2e-3)
        off, lat = frame(v, idx)
        d = -off
        X_ = idx[:, ::-1] * h + lo0 + h / 2
        c_, n_ = np.asarray(v["center"]), np.asarray(v["normal"])
        lv_ = (X_ - c_) - off[:, None] * n_  # lateral vectors
        # the tract bends: its centre is on the valve axis at the annulus and at the cavity's own centroid at the mouth
        # (the ventricle voxels 2–4 mm around depth L within the mouth's reach, AV inflow tracts left out), the offset
        # growing ∝ (d / L)² so the flow enters the valve along its axis (full17: the LVOT approached 30–40° off it)
        mouth = (np.abs(d - OT_L) < 3e-3) & (lat < rl * OT_K + 10e-3) & ~keep
        o_L = lv_[mouth].mean(0) if mouth.sum() > 20 else np.zeros(3)
        if np.linalg.norm(o_L) > rl * OT_K:
            o_L *= rl * OT_K / np.linalg.norm(o_L)
        zone = (d > 1.5 * h) & (d <= OT_L)
        f_ = np.clip(d / OT_L, 0, 1) ** 2
        latb = np.linalg.norm(lv_ - f_[:, None] * o_L, axis=1)
        rcone = rl * (1 + (OT_K - 1) * f_)
        # (laterally up to the mouth radius + 15 mm from the bent centre: the tract's walls, not the ventricle body)
        cut = zone & (latb > rcone) & (latb < rl * OT_K + 15e-3) & ~keep
        print(f"{v['name']}: outflow tract mouth {np.linalg.norm(o_L) * 1e3:.1f} mm off the valve axis at {OT_L * 1e3:.0f} mm "
              f"({np.degrees(np.arctan2(np.linalg.norm(o_L), OT_L)):.0f}°)")
        domain[tuple(idx[cut].T)] = False
        print(f"{v['name']}: outflow-tract funnel over {OT_L * 1e3:.0f} mm (radius {rl * 1e3:.1f} → {rl * OT_K * 1e3:.1f} mm), {cut.sum() * h**3 * 1e6:.2f} mL of ventricle walled")
    region[~domain] = 0
    _dbg("outflow funnels")
# sealing: without each valve's disk, no component holds both its upstream and its downstream region near the valve
for v in valves:
    U_, D_ = up_down[v["name"]]
    d_ = domain & ~disk
    lab, nl = ndimage.label(d_, ST18)
    idx = np.argwhere(d_ & np.isin(region, [U_, D_]))
    off, lat = frame(v, idx)
    near_ = (lat < v["radius"] + 8e-3) & (np.abs(off) < 15e-3)
    cu = set(lab[tuple(idx[near_ & (region[tuple(idx.T)] == U_)].T)].tolist()) - {0}
    cd = set(lab[tuple(idx[near_ & (region[tuple(idx.T)] == D_)].T)].tolist()) - {0}
    v["sealed"] = not (cu & cd)
    print(f"{v['name']}: {'sealed' if v['sealed'] else 'LEAKS'} (upstream components {sorted(cu)[:4]}, downstream {sorted(cd)[:4]})")
# passages: open faces between each stub and the region it must drain into / fill from, against its lumen area
for k_, r in enumerate(roots):
    a_ = 10 + k_
    b_ = CHI[r["chamber"]] if r["class"] in ("systemic_venous", "pulmonary_venous") else None
    if b_ is None:
        continue
    nf = 0
    for ax in range(3):
        for sh in (-1, 1):
            nf += int(((region == a_) & (np.roll(region, sh, axis=ax) == b_) & domain & np.roll(domain, sh, axis=ax)).sum())
    lumen = np.pi * R_ref[r["path"][0]] ** 2
    print(f"passage {r['name'][:28]:28s} → {r['chamber']}: {nf * h * h * 1e4:.2f} cm² open faces vs lumen {lumen * 1e4:.2f} cm²")
# arterial stubs: open cross-section along the path against the vessel lumen (walls must not pinch them)
rname_ = {v_: k_ for k_, v_ in CHI.items()} | {10 + k_: "stub " + r["name"][:20] for k_, r in enumerate(roots)}
for k_, r in enumerate(roots):
    if r["class"] not in ("systemic_arterial", "pulmonary_arterial"):
        continue
    idx = np.argwhere(domain)
    X = idx[:, ::-1] * h + lo0 + h / 2
    prof = []
    for j in range(2, len(r["path"]) - 2, max(1, len(r["path"]) // 8)):
        p_, t_ = P_ref[r["path"][j]], P_ref[r["path"][j + 1]] - P_ref[r["path"][j - 1]]
        t_ /= np.linalg.norm(t_)
        off = (X - p_) @ t_
        lat = np.linalg.norm((X - p_) - off[:, None] * t_, axis=1)
        sel = (np.abs(off) < 0.5 * h) & (lat < 1.3 * R_ref[r["path"][j]])
        own_ = region[tuple(idx[sel].T)] == 10 + k_
        frac = own_.sum() * h * h / (np.pi * R_ref[r["path"][j]] ** 2)
        prof.append(f"{frac:.2f}")
        if frac < 0.85:
            # who holds the lumen disk here: other regions (domain) and non-domain voxels (walls, outside)
            Xa = np.argwhere(np.ones(1, bool))  # placeholder to keep names local
            lo_ = np.floor((p_ - R_ref[r["path"][j]] - lo0) / h).astype(int)
            hi_ = np.ceil((p_ + R_ref[r["path"][j]] - lo0) / h).astype(int) + 1
            zz_, yy_, xx_ = np.meshgrid(*(np.arange(max(lo_[d], 0), min(hi_[d], domain.shape[2 - d])) for d in (2, 1, 0)), indexing="ij")
            Xd = np.stack([xx_, yy_, zz_], -1).reshape(-1, 3)
            Xm = Xd * h + lo0 + h / 2
            o2 = (Xm - p_) @ t_
            l2 = np.linalg.norm((Xm - p_) - o2[:, None] * t_, axis=1)
            dsk = (np.abs(o2) < 0.5 * h) & (l2 < R_ref[r["path"][j]])
            zi, yi, xi = Xd[dsk, 2], Xd[dsk, 1], Xd[dsk, 0]
            rg = region[zi, yi, xi]
            dm = domain[zi, yi, xi]
            wl = wall[zi, yi, xi]
            st_ = stub[zi, yi, xi]
            cnt = {}
            for a_, b_, c_ in zip(rg, dm, wl):
                key = ("wall" if c_ else (rname_.get(int(a_), int(a_)) if b_ else "outside"))
                cnt[key] = cnt.get(key, 0) + 1
            print(f"   sample {j}: lumen disk ({dsk.sum()} voxels, {st_.mean() * 100:.0f} % in the capsule): " + ", ".join(f"{k__} {v__ / dsk.sum() * 100:.0f} %" for k__, v__ in sorted(cnt.items(), key=lambda z: -z[1])))
    print(f"stub {r['name'][:28]:28s} own-region section / lumen along the path (R {R_ref[r['path'][0]] * 1e3:.1f}…{R_ref[r['path'][-1]] * 1e3:.1f} mm): {' '.join(prof)}")
lab, nl = ndimage.label(domain, ST18)
sz = np.bincount(lab.ravel())[1:]
keep = 1 + np.flatnonzero(sz >= 0.05 * sz.max())  # the left and right heart; slivers cut off by the walls dropped
rname = {v_: k_ for k_, v_ in CHI.items()} | {10 + k_: "stub " + r["name"][:28] for k_, r in enumerate(roots)}
for c_ in keep:
    rr = np.bincount(region[lab == c_], minlength=nreg)
    parts = ", ".join(f"{rname.get(i_, i_)} {rr[i_] * h**3 * 1e6:.1f}" for i_ in np.argsort(-rr)[:6] if rr[i_] > 0)
    print(f"  component {c_}: {sz[c_ - 1] * h**3 * 1e6:.1f} mL: {parts}")
domain = np.isin(lab, keep)
_dbg("components kept")
print(f"domain after walls: {int(domain.sum())} voxels ({domain.sum() * h**3 * 1e6:.0f} mL), {len(keep)} components kept")

# ── lumen gate (2026-11-25): along every tree branch the wall distance at the centreline must stay ≥ LUMEN_MIN × the
#    graph radius, away from the cut planes and the heart end (a wall carved through interpenetrating atlas tubes left
#    the right PA and the right superior PV as slivers: 6 m/s and 3 m/s jets, research/flow_phantom.md Tier B4)
_dw = ndimage.distance_transform_edt(domain) * h
_cut_c = np.array([c["center"] for c in cuts]) if cuts else np.zeros((0, 3))
_cut_r = np.array([c["radius"] for c in cuts]) if cuts else np.zeros(0)
_bad = []
for b in branches:
    if b.get("drop"):
        continue
    sm = np.asarray(b["samples"])
    pp, rr = P_ref[sm], R_ref[sm]
    iv = np.floor((pp - lo0) / h).astype(int)
    ok_ = np.all((iv >= 0) & (iv < np.array(domain.shape[::-1])), 1)
    rt = P_ref[roots[b["root"]]["path"][0]]
    near = np.linalg.norm(pp - rt, axis=1) < np.maximum(2 * rr, 4e-3)
    for cc, cr in zip(_cut_c, _cut_r):
        near |= np.linalg.norm(pp - cc, axis=1) < max(2 * cr, 4e-3) + 3 * h
    use = ok_ & ~near
    if not use.any():
        continue
    ratio = _dw[iv[use, 2], iv[use, 1], iv[use, 0]] / rr[use]
    j_ = int(np.argmin(ratio))
    if ratio[j_] < LUMEN_MIN:
        _bad.append((float(ratio[j_]), enames[b["edge"]] if b["edge"] >= 0 else roots[b["root"]]["name"], float(rr[use][j_] * 1e3), (pp[use][j_] * 1e3).round(1).tolist()))
# cut disks: the open share of each cut's flux disk (lateral < R, the voxel layer just inside its plane) in its own
# region — a partly solid disk forces its flow through the rest (the left subclavian's 76 % disk: 3 m/s jets)
_cd = []
for c in cuts:
    cc_, nn_, R_ = np.asarray(c["center"]), np.asarray(c["normal"]), float(c["radius"])
    lo_ = np.maximum(np.floor((cc_ - R_ - 2 * h - lo0) / h).astype(int), 0)
    hi_ = np.minimum(np.ceil((cc_ + R_ + 2 * h - lo0) / h).astype(int) + 1, np.array(domain.shape[::-1]))
    zz_, yy_, xx_ = np.meshgrid(*(np.arange(lo_[d], hi_[d]) for d in (2, 1, 0)), indexing="ij")
    Xq = np.stack([xx_, yy_, zz_], -1).reshape(-1, 3)
    d_ = Xq * h + lo0 + h / 2 - cc_
    sq = d_ @ nn_
    lq = np.linalg.norm(d_ - sq[:, None] * nn_, axis=1)
    m_ = (sq > -1.0 * h) & (sq <= 0) & (lq < R_)
    if m_.sum() < 3:
        continue
    zi, yi, xi = Xq[m_, 2], Xq[m_, 1], Xq[m_, 0]
    _cd.append((float((domain[zi, yi, xi] & (region[zi, yi, xi] == 10 + c["root"])).mean()), c["name"]))
_cd.sort()
print("cut disks: open share in the own region, lowest: " + ", ".join(f"{n_[:30]} {f_:.2f}" for f_, n_ in _cd[:6]))
_bad.sort()
print(f"lumen gate: {len(_bad)} branches with wall distance < {LUMEN_MIN} r at the centreline" + "".join(f"\n   {x[0]:.2f} r at {x[3]} mm, {x[1][:40]} (r {x[2]:.1f} mm)" for x in _bad[:15]))
if _bad and LUMEN_MIN > 0:
    sys.exit(f"lumen gate failed: {len(_bad)} pinched branches (--lumen-min 0 to build anyway)")

# ── signed distance on the domain's bounding box (+5 mm) ─────────────────────────────────────────────────────────────
nzv = np.nonzero(domain)
pad = int(np.ceil(5e-3 / h))
b0 = np.maximum([min(a) - pad for a in nzv], 0)
b1 = np.minimum([max(a) + pad + 1 for a in nzv], domain.shape)
D = domain[b0[0] : b1[0], b0[1] : b1[1], b0[2] : b1[2]]
din = ndimage.distance_transform_edt(D)
dout = ndimage.distance_transform_edt(~D)
phi = np.where(D, din - 0.5, -(dout - 0.5)) * h
phi = ndimage.gaussian_filter(phi, 0.6)
sdf_origin = lo0 + np.array([b0[2], b0[1], b0[0]]) * h
np.save(OUT / "sdf.npy", phi.astype(np.float32))
# region map on the same grid (diagnostics: which chamber / stub a lattice node belongs to in the reference pose)
np.save(OUT / "region.npy", np.where(domain, region, 0)[b0[0] : b1[0], b0[1] : b1[1], b0[2] : b1[2]].astype(np.int16))
print(f"sdf: {phi.shape[::-1]} (x, y, z) at {h * 1e3:.2f} mm, origin {sdf_origin}")

# ── cuts ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
tt = np.linspace(0, period, 256, endpoint=False)
cut_out = []
q_ins = []
for c in cuts:
    rec = {"name": c["name"], "center": c["center"], "normal": c["normal"], "radius": c["radius"], "slab": c["slab"], "class": c["class"]}
    q_t = np.real(qn[c["sample"], 0] + 2 * (qn[c["sample"], 1:] @ np.exp(1j * np.outer(om[1:], tt))))
    q_in = -q_t if c["sgn"] == 1 else q_t  # traversed away from the heart: positive network flow leaves the domain
    q_ins.append(q_in)
    if c["class"] in ("systemic_venous", "pulmonary_venous"):
        rec |= {"kind": "inlet", "q_network_mean": float(np.mean(q_in)), "class": c["class"]}
    elif "--rcr" in A:
        p_t = np.real(pn[c["sample"], 0] + 2 * (pn[c["sample"], 1:] @ np.exp(1j * np.outer(om[1:], tt))))
        qm, pm = max(float(np.mean(-q_in)), 1e-7), float(np.mean(p_t))
        rho, cwave = 1060.0, 5.0
        r1 = rho * cwave / (np.pi * c["radius"] ** 2)
        r2 = max(pm / qm - r1, r1)
        rec |= {"kind": "windkessel", "rcr": [r1, 1.0 / (r2 * 2 * np.pi / period), r2]}
    else:
        rec |= {"kind": "pressure", "p": 0.0}
    cut_out.append(rec)
# arterial trees: every outlet but the largest gets the 1D network's outflow waveform at its sample (a velocity cut
# with a prescribed outflow: physiological splits and waveforms, the network's compliance included); the largest is a
# pressure outlet and takes the rest — the ventricle's ejection (wall motion) less the prescribed outflows, incl. the
# branches below R_CUT. Windkessel outlets would turn the pulse pressure into density swings of the weakly compressible
# lattice (an artificial compliance of V / (ρ c_s²) ≈ 0.4 mL/mmHg at c_s = 14 m/s)
tree_q = {}  # arterial root → valve flow and the outlets' summed network targets (the wall transpiration's storage)
for k_, r in enumerate(roots):
    if r["class"] not in ("systemic_arterial", "pulmonary_arterial"):
        continue
    key = "Q_AO" if r["class"] == "systemic_arterial" else "Q_PA"
    Qroot = np.interp(tt, tq, np.asarray(cm[key], float), period=period)
    ci = [i for i, c in enumerate(cuts) if c["root"] == k_]
    if ci and "--rcr" not in A:
        _qo = {i: -q_ins[i] for i in ci}
        _f = float(np.mean(Qroot)) / max(sum(float(np.mean(_qo[i])) for i in ci), 1e-12)
        tree_q[k_] = {"Qroot": Qroot, "targets": sum(_qo[i] for i in ci) * _f}
    if len(ci) == 1 or "--rcr" in A:
        for i in ci:
            ref_ = tree_q[k_]["targets"] if (TRANSP == "on" and k_ in tree_q) else Qroot
            cut_out[i]["q_ref"] = {"t": tt.tolist(), "v": ref_.tolist()}
        continue
    qout = {i: -q_ins[i] for i in ci}
    main = max(ci, key=lambda i: float(np.mean(qout[i])))
    # a regurgitant valve on this side (MR / AR left, TR / PR right): no pressure outlet, every cut prescribed — the
    # ventricle's contraction less the prescribed forward flow then has to go back through the regurgitant orifice
    # (the lattice has no physiological pressures that would split it); the venous inertial inlets keep the reference
    if ("left" if r["class"] == "systemic_arterial" else "right") in REGURG_SIDES:
        main = None
    # the branches below R_CUT (lung segmental arteries, intercostals, coronaries) carry the rest of the root's mean
    # flow: share it over all outlets in proportion to their network flow (else it all exits the pressure outlet: 48 of
    # 98 mL/s through one 3 mm segmental artery of the pulmonary tree)
    f_tree = float(np.mean(Qroot)) / max(sum(float(np.mean(qout[i])) for i in ci), 1e-12)
    for i in ci:
        qout[i] = qout[i] * f_tree
    rest = sum(qout[i] for i in ci if i != main)
    for i in ci:
        if i == main:
            # the diagnostics' reference: with the wall transpiration storing Qroot − Σ targets, the pressure outlet
            # should carry its own network target; without, it takes the storage too
            cut_out[i]["q_ref"] = {"t": tt.tolist(), "v": (qout[main] if TRANSP == "on" else Qroot - rest).tolist()}
        else:
            cut_out[i] |= {"kind": "inlet", "q": {"t": tt.tolist(), "v": (-qout[i]).tolist()}}
    if main is None:
        print(f"tree {r['name'][:30]}: outflows × {f_tree:.2f} (flow below R_CUT shared out); no pressure outlet (regurgitant valve on this side), {len(ci)} prescribed outflows "
              f"totalling {np.mean(rest) * 1e6:.1f} mL/s of {np.mean(Qroot) * 1e6:.1f}")
    else:
        print(f"tree {r['name'][:30]}: outflows × {f_tree:.2f} (flow below R_CUT shared out); pressure outlet {cut_out[main]['name'][:36]} (network mean × f {np.mean(qout[main]) * 1e6:.1f} mL/s, takes "
              f"{np.mean(Qroot - rest) * 1e6:.1f}), {len(ci) - 1} prescribed outflows totalling {np.mean(rest) * 1e6:.1f} mL/s of {np.mean(Qroot) * 1e6:.1f}")
# ── wall transpiration (2026-11-25, research/flow_phantom.md Tier B4): the arterial trees are rigid in the lattice
#    apart from the cardiac motion field's volume change, so the storage of the network's compliance, Q_valve − Σ outlet
#    targets, went through each tree's single pressure outlet (−118…+54 mL/s through a 3.4 mm segmental artery, 8.5 m/s
#    jets; the head/arm outflow fed backwards up the aorta in diastole). Each tree instead stores
#    q_w(t) = Q_valve(t) − Σ targets(t) − dV_tree/dt (motion), zero-mean, through its wall: in a band of 2 lattice units on
#    both sides (hasty-cfd kernels: normal velocity q_w w / (da/dA), tapering over the last half unit) with the weight
#    w ∝ the local graph radius (thin-wall compliance per wall area ∝ r), ∮ w dA = 1; not within 1.5 R + 3 mm of the
#    tree's cuts and valve. The pressure outlet then carries its own target up to the residual.
transp = None
if TRANSP == "on" and tree_q:
    Dc = D
    Rc = np.where(domain, region, 0)[b0[0] : b1[0], b0[1] : b1[1], b0[2] : b1[2]]
    grp = np.zeros(Dc.shape, np.uint8)
    wgt = np.zeros(Dc.shape, np.float32)
    gz, gy, gx = np.gradient(phi, h)
    gmag = np.sqrt(gz**2 + gy**2 + gx**2)
    del gz, gy, gx
    eps_ = 1.5 * h
    delta = np.where(np.abs(phi) < eps_, (1 + np.cos(np.pi * phi / eps_)) / (2 * eps_), 0.0) * gmag
    del gmag
    # the cardiac motion's Jacobian J(X, t) = det(I + ∇U) on the motion grid, for the trees' volume change
    Um = np.asarray(cm["U_mm"], np.float32) * 1e-3  # [T, 3, Z, Y, X]
    m_o = np.asarray(cm["origin_m"], float)
    m_h = float(cm["voxel_m"])
    m_t = np.asarray(cm["t"], float)
    Jt = []
    for it_ in range(Um.shape[0]):
        G_ = [[np.gradient(Um[it_, i_], m_h, axis=2 - j_) for j_ in range(3)] for i_ in range(3)]  # ∂U_i/∂X_j (x, y, z)
        F_ = [[G_[i_][j_] + (1.0 if i_ == j_ else 0.0) for j_ in range(3)] for i_ in range(3)]
        Jt.append(F_[0][0] * (F_[1][1] * F_[2][2] - F_[1][2] * F_[2][1]) - F_[0][1] * (F_[1][0] * F_[2][2] - F_[1][2] * F_[2][0]) + F_[0][2] * (F_[1][0] * F_[2][1] - F_[1][1] * F_[2][0]))
    del Um
    groups_ = []
    for gid, (k_, tqd) in enumerate(sorted(tree_q.items()), start=1):
        r = roots[k_]
        rid = 10 + k_
        Fg = Rc == rid
        din_g = ndimage.distance_transform_edt(Fg)
        dF = ndimage.distance_transform_edt(~Fg)
        dO = ndimage.distance_transform_edt(~(Dc & ~Fg))
        band = (Fg & (din_g <= 2.5)) | (~Dc & (dF <= 2.5) & (dF + 1 <= dO))
        zz_, yy_, xx_ = np.nonzero(band)
        Xb = np.stack([xx_, yy_, zz_], -1) * h + sdf_origin + h / 2
        keep_ = np.ones(len(Xb), bool)
        for c in cuts:
            if c["root"] == k_:
                keep_ &= np.linalg.norm(Xb - np.asarray(c["center"]), axis=1) > 1.5 * c["radius"] + 3e-3
        for v in valves:
            if v["name"] == ("aortic" if r["class"] == "systemic_arterial" else "pulmonary"):
                keep_ &= np.linalg.norm(Xb - np.asarray(v["center"]), axis=1) > 1.5 * v["radius"] + 3e-3
        zz_, yy_, xx_, Xb = zz_[keep_], yy_[keep_], xx_[keep_], Xb[keep_]
        tp_ = np.vstack([P_ref[b["samples"]] for b in branches if b["root"] == k_ and not b.get("drop")])
        tr_ = np.concatenate([R_ref[b["samples"]] for b in branches if b["root"] == k_ and not b.get("drop")])
        rv_ = tr_[cKDTree(tp_).query(Xb)[1]]
        dl_ = delta[zz_, yy_, xx_]
        area = float(dl_.sum() * h**3)
        C_ = float((dl_ * rv_).sum() * h**3)
        grp[zz_, yy_, xx_] = gid
        wgt[zz_, yy_, xx_] = (rv_ / C_).astype(np.float32)
        # the tree's volume under the motion: Σ J h³ over its reference voxels (every 2nd voxel per axis, × 8)
        fz, fy, fx = np.nonzero(Fg[::2, ::2, ::2])
        Xf = np.stack([fx * 2, fy * 2, fz * 2], -1) * h + sdf_origin + h / 2
        ic = ((Xf - m_o) / m_h - 0.5)[:, ::-1].T  # (z, y, x) index coordinates on the motion grid
        Vt = np.array([float(ndimage.map_coordinates(Jt[it_], ic, order=1, mode="nearest").sum()) * 8 * h**3 for it_ in range(len(m_t))])
        kf = np.fft.rfftfreq(len(m_t), d=period / len(m_t))
        dV = np.fft.irfft(np.fft.rfft(Vt) * 2j * np.pi * kf, n=len(m_t))  # periodic derivative (m³/s); m_t uniform
        assert np.allclose(np.diff(m_t), period / len(m_t), rtol=1e-6), "motion samples not uniform in time"
        dVt = np.interp(tt, m_t, dV, period=period)
        qw = tqd["Qroot"] - tqd["targets"] - dVt
        qw_mean = float(qw.mean())
        qw = qw - qw_mean
        umax_ = float(np.abs(qw).max() * wgt[zz_, yy_, xx_].max())
        print(f"transpiration group {gid} {r['name'][:30]} (region {rid}): band {len(zz_)} voxels, wall area {area * 1e4:.0f} cm², mean radius "
              f"{C_ / max(area, 1e-12) * 1e3:.1f} mm; motion volume {Vt.min() * 1e6:.1f}…{Vt.max() * 1e6:.1f} mL; q_w {qw.min() * 1e6:.0f}…{qw.max() * 1e6:.0f} mL/s "
              f"(mean removed {qw_mean * 1e6:.2f}), stored swing {np.ptp(np.cumsum(qw) * period / len(tt)) * 1e6:.1f} mL, largest wall speed {umax_ * 1e2:.2f} cm/s")
        groups_.append({"id": gid, "name": r["name"], "region": rid, "q": {"t": tt.tolist(), "v": qw.tolist()}})
    del Jt, delta
    np.save(OUT / "transp_group.npy", grp)
    np.save(OUT / "transp_weight.npy", wgt)
    transp = {"group": "transp_group.npy", "weight": "transp_weight.npy", "band": 2.0, "groups": groups_}
# venous inlets: the cardiac model's atrial inflow (Q_PV into the LA, Q_VR into the RA: the flows its volume curves are
# built from), split by the network's mean shares; the network's own stub flows miss the branches without a stub
for cls_, key in (("pulmonary_venous", "Q_PV"), ("systemic_venous", "Q_VR")):
    ins = [r_ for r_ in cut_out if r_.get("class") == cls_]
    tot = sum(max(r_["q_network_mean"], 0.0) for r_ in ins)
    Qc = np.interp(tt, tq, np.asarray(cm[key], float), period=period)
    for r_ in ins:
        share = max(r_["q_network_mean"], 0.0) / tot if tot > 0 else 1.0 / len(ins)
        if VENOUS == "inertial":
            # velocity inlet with the upstream vein's inertia: q(t) = share × Q plus a correction driven by the cut
            # pressure (hasty-cfd CutSpec::inertance); L_k = ρ ℓ / (A_group s_k), so the group's corrections split by share
            # one correction per atrium (group), damped: ζ = 0.7 against the atrium's LBM compliance V / (ρ c_s²)
            a_grp = sum(np.pi * x["radius"] ** 2 for x in ins)
            ell = {"pulmonary_venous": 0.06, "systemic_venous": 0.10}[cls_]
            L_g = 1060.0 * ell / a_grp
            cs = V_MAX / float(opt("--u-lattice", 0.08)) / np.sqrt(3.0)
            v_at = float((region == CHI["blood_LA" if cls_ == "pulmonary_venous" else "blood_RA"]).sum()) * h**3
            R_g = 2 * 0.7 * np.sqrt(L_g / (v_at / (1060.0 * cs**2)))
            r_["q"] = {"t": tt.tolist(), "v": (share * Qc).tolist()}
            r_ |= {"inertance": L_g, "resistance": R_g, "relax": 1.0, "share": {"group": cls_, "fraction": share}}
        elif VENOUS == "pressure":
            # pressure inlet: the wall motion sets the inflow (incl. the reversal at atrial contraction), a slow trim of
            # the cut pressures keeps the cycle-mean split at the shares; q_ref (outward) for the diagnostics
            r_ |= {"kind": "pressure", "p": 0.0, "share": {"group": cls_, "fraction": share}, "sponge": 3e-3}
            r_["q_ref"] = {"t": tt.tolist(), "v": (-share * Qc).tolist()}
        else:
            r_["q"] = {"t": tt.tolist(), "v": (share * Qc).tolist()}
        print(f"inlet {r_['name'][:32]:32s} ({VENOUS}) share {share * 100:.0f} % of {key} (mean {np.mean(Qc) * 1e6:.1f} mL/s; network mean at the cut {r_['q_network_mean'] * 1e6:.1f})")
        # the class stays in the record (2026-11-26: the report's cut balances per tree need it; hasty-cfd ignores it)
# upstream paths of the venous inlets (hasty-cfd CutSpec::upstream: the spin history integrates the imaging slab's
# saturation along them): the vessel graph beyond the cut, away from the heart along the main branch to its end, transit
# times from the network's mean flow; at the end a capillary segment. Systemic veins start fresh (the systemic loop is
# long and mostly outside a thoracic slab); a pulmonary vein starts from the pulmonary arteries' outflow (`from`), runs
# the arterial path to the lung of the outlet that ends nearest, the capillary bed there (the rest of the pulmonary
# mean transit time --pulm-mtt, default 5 s ≈ pulmonary blood volume / cardiac output) and its own path back
MTT_PULM = float(opt("--pulm-mtt", 5.0))
T_CAP_SYS = float(opt("--sys-capillary-s", 2.0))


def main_walk(node, exclude, length=2.0):
    """samples from `node` along the main branch (largest mean radius), not entering the edges in `exclude`"""
    path, acc, used = [], 0.0, set(exclude)
    while acc < length:
        cand = [k for k in adj.get(node, []) if k not in used]
        if not cand:
            break
        k = max(cand, key=lambda k_: R_ref[eoff[k_] : eoff[k_ + 1]].mean())
        used.add(k)
        sm, sg = edge_samples(k, node)
        for j in sm:
            if path:
                acc += np.linalg.norm(P_ref[j] - P_ref[path[-1]])
            path.append(int(j))
        node = int(eb[k] if sg == 1 else ea[k])
    return np.asarray(path, int)


def walk_from(c):
    """samples beyond cut c (its edge onwards, then the main branch) and the transit time of each segment"""
    j = c["sample"]
    k = int(np.searchsorted(eoff, j, side="right") - 1)
    if c["sgn"] == 1:
        sm, node = np.arange(j, eoff[k + 1]), int(eb[k])
    else:
        sm, node = np.arange(j, eoff[k] - 1, -1), int(ea[k])
    seg = np.concatenate([sm, main_walk(node, {k})]).astype(int)
    pts = P_ref[seg]
    dts = []
    for a in range(len(seg) - 1):
        ln = float(np.linalg.norm(pts[a + 1] - pts[a]))
        area = np.pi * max(R_ref[seg[a]], 0.5e-3) ** 2
        v = float(np.clip(abs(np.real(qn[seg[a], 0])) / area, 0.03, 1.0))
        dts.append(ln / v)
    return pts, np.asarray(dts)


walk = {i: walk_from(c) for i, c in enumerate(cuts) if c["class"] in ("systemic_venous", "pulmonary_venous", "pulmonary_arterial")}
pa_out = [i for i, c in enumerate(cuts) if c["class"] == "pulmonary_arterial"]
for i, c in enumerate(cuts):
    if c["class"] not in ("systemic_venous", "pulmonary_venous"):
        continue
    pv, tv = walk[i]
    # far upstream → cut: the venous samples reversed; segment a ends at sample a + 1 (dt is the transit to the next)
    pts_v, dt_v = pv[::-1][:-1], tv[::-1]
    ka = None
    if c["class"] == "systemic_venous":
        pts = np.vstack([pv[-1:], pts_v])
        dts = np.concatenate([[T_CAP_SYS], dt_v])
        frm = []
    else:
        ka = min(pa_out, key=lambda j_: np.linalg.norm(walk[j_][0][-1] - pv[-1])) if pa_out else None
        pa_, ta_ = walk[ka] if ka is not None else (np.zeros((0, 3)), np.zeros(0))
        # the graph's distal branches carry little mean flow (slow, clipped at 3 cm/s): when arteries + veins exceed
        # the mean transit time less a capillary bed of 0.75 s, both are scaled to fit it
        t_big = ta_.sum() + tv.sum()
        f_ = min(1.0, (MTT_PULM - 0.75) / max(t_big, 1e-9))
        t_cap = MTT_PULM - f_ * t_big
        pts = np.vstack([pa_[:-1], pv[-1:], pts_v]) if len(pa_) else np.vstack([pv[-1:], pts_v])
        dts = np.concatenate([f_ * ta_, [t_cap], f_ * dt_v])
        frm = [cut_out[j_]["name"] for j_ in pa_out]
    cut_out[i]["upstream"] = {"points": pts.tolist(), "dt": dts.tolist(), "from": frm}
    print(f"upstream {cut_out[i]['name'][:32]:32s}: {len(pts)} samples, {dts.sum():.2f} s ({dt_v.sum():.2f} s in the vein, z {pts[:, 2].min():.3f}…{pts[:, 2].max():.3f} m)"
          + (f", from {len(frm)} pulmonary outlets via {cut_out[ka]['name'][:24]}" if frm else ""))
for r_ in cut_out:
    if r_["kind"] != "inlet" and "share" not in r_:
        print(f"outlet {r_['name'][:32]:32s} r {r_['radius'] * 1e3:.1f} mm, {r_['kind']}")

_tp = P0 / "tissue_params.json"
BLOOD_MR = None
if _tp.exists():
    _tab = json.loads(_tp.read_text())
    _bl = [v_ for k_, v_ in _tab["table"].items() if k_.startswith("blood")]
    if _bl:
        BLOOD_MR = {"t1": float(np.mean([b_["t1_s"] for b_ in _bl])), "t2": float(np.mean([b_["t2_s"] for b_ in _bl])), "contrast": _tab.get("contrast")}
print(f"blood MR: {BLOOD_MR}")

# ── LBM grid: the domain's box grown by the largest displacement ─────────────────────────────────────────────────────
U = np.asarray(cm["U_mm"], np.float32)
umax = float(np.sqrt((U.astype(np.float32) ** 2).sum(1)).max()) * 1e-3
marg = umax + 3e-3
glo = sdf_origin + h / 2 - marg
ghi = sdf_origin + (np.array(phi.shape[::-1]) - 0.5) * h + marg
dx = DX
dims = (np.ceil((ghi - glo) / dx) + 1).astype(int)
# ── grid refinement (hasty-cfd refine.rs, kernels/lbm_core.h "grid refinement"): level 1 (dx / 2) over spheres around
# the named valves along their path through the cycle — the valve centre and points downstream along its normal (the
# jet; --refine-down-mm) and half a radius upstream, each moved by the cardiac motion at every 4th motion sample —
# of radius R + --refine-margin-mm. "--refine aortic,pulmonary" | "all"; --refine-levels 2 adds level 2 (dx / 4) with
# half the margin and the downstream reach
REFINE = opt("--refine", "")
refine = []
if REFINE:
    from refine_spheres import spheres as refine_spheres  # noqa: E402

    refine = refine_spheres(valves, cm, [v["name"] for v in valves] if REFINE == "all" else REFINE.split(","), float(opt("--refine-margin-mm", 6.0)) * 1e-3,
                            float(opt("--refine-down-mm", 10.0)) * 1e-3, int(opt("--refine-levels", 1)))

case = {
    "phantom": str(P0), "source": "rustlib/tools/phantom/lbm_case.py",
    "grid": {"origin": glo.tolist(), "dx": dx, "dims": dims.tolist()},
    "sdf": {"file": "sdf.npy", "origin": sdf_origin.tolist(), "h": h},
    "regions": {"file": "region.npy", "names": {str(v_): k_ for k_, v_ in CHI.items()} | {str(10 + k_): r["name"] for k_, r in enumerate(roots)},
                "allowed": [[int(a_), int(b_)] for a_, b_ in zip(*np.nonzero(allowed)) if a_ < b_],
                "valves": {v["name"]: list(up_down[v["name"]]) for v in valves}},
    "motion": {"file": str(CM_FILE.resolve())},
    "fluid": {"nu": 3.5e-3 / 1060.0, "rho": 1060.0},
    # the blood's MR properties of this phantom (its contrast state: cardiac_lbm --spin takes T1 from here)
    "blood": BLOOD_MR,
    "v_max": V_MAX, "u_lattice": float(opt("--u-lattice", 0.08)),
    "valves": valves, "cuts": cut_out,
    # the chambers' volume rates of the cardiac model (hasty-cfd Case::wall_pump, run.wall_pump; `--no-wall-pump`: as
    # before): the moving-wall bounce-back is the chambers' pump and injected ≈ 0.89 of the stroke; corrected to −dV/dt per
    # chamber, the healthy full15 case pumps 83.4–84.7 mL against the model's 83.8 (research/flow_phantom.md)
    "wall_pump": {nm_: {"t": tq.tolist(), "v": (np.asarray(cm[a_], float) - np.asarray(cm[b_], float)).tolist()}
                  for nm_, a_, b_ in (("blood_LV", "Q_MV", "Q_AO"), ("blood_LA", "Q_PV", "Q_MV"), ("blood_RV", "Q_TV", "Q_PA"), ("blood_RA", "Q_VR", "Q_TV"))},
    **({"transpiration": transp} if transp else {}),
    **({"refine": refine} if refine else {}),
    "run": {"cycles": int(opt("--cycles", 4)), "warmup": int(opt("--warmup", 2)), "bins": int(opt("--bins", 50)), "out_factor": int(opt("--out-factor", 2)),
            "acc_every": 20, "geom_every": 4, "cwale": 0.5, "inv_iters": 3, "uclip": 0.35, "diag_every": 1000, "ckpt_every": 20000,
            "wall_pump": "--no-wall-pump" not in A},
}
(OUT / "case.json").write_text(json.dumps(case, indent=1))
print(f"grid {dims.tolist()} at {dx * 1e3:.2f} mm ({np.prod(dims) / 1e6:.1f} M box nodes; max displacement {umax * 1e3:.1f} mm) → {OUT / 'case.json'}")

# ── figures ──────────────────────────────────────────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
cz, cy, cx = (np.array(phi.shape) // 2).tolist()
ext = lambda a, b: [sdf_origin[a] * 1e3, (sdf_origin[a] + phi.shape[::-1][a] * h) * 1e3, sdf_origin[b] * 1e3, (sdf_origin[b] + phi.shape[::-1][b] * h) * 1e3]  # noqa: E731
for ax, (img, (a, b), title) in zip(axs, [(phi[cz], (0, 1), "axial"), (phi[:, cy], (0, 2), "coronal"), (phi[:, :, cx], (1, 2), "sagittal")]):
    ax.imshow(img > 0, origin="lower", extent=ext(a, b), cmap="Greys", alpha=0.8)
    ax.contour(img, levels=[0], origin="lower", extent=ext(a, b), colors="#2a78d6", linewidths=0.6)
    for v in valves:
        ax.plot(v["center"][a] * 1e3, v["center"][b] * 1e3, "o", color="#eb6834", ms=6)
        ax.annotate(v["name"], (v["center"][a] * 1e3, v["center"][b] * 1e3), color="#eb6834", fontsize=7)
    for c in cut_out:
        ax.plot(c["center"][a] * 1e3, c["center"][b] * 1e3, "s", color="#1baf7a", ms=5)
    ax.set_title(f"{title} (domain, valves ●, cuts ■)", fontsize=9, loc="left")
    ax.set_xlabel("mm")
fig.tight_layout()
fig.savefig(OUT / "figs" / "lbm_case_domain.png", dpi=130)
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 3.5))
for v, col in zip(valves, ["#2a78d6", "#eb6834", "#1baf7a", "#1f1f1e"]):
    ax.plot(np.array(v["opening"]["t"]) * 1e3, v["opening"]["v"], color=col, lw=1.5, label=v["name"])
ax.set_xlabel("t from end-diastole [ms]")
ax.set_ylabel("opening")
ax.legend(frameon=False, fontsize=8)
ax.set_title("valve opening (from the cardiac model's valve flows)", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(OUT / "figs" / "lbm_case_valves.png", dpi=130)
print(f"figures: {OUT / 'figs'}")
