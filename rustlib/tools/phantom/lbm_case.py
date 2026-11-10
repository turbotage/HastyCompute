"""Cardiac lattice-Boltzmann case of a phantom (hasty-cfd; research/flow_phantom.md "Tier B: LBM").

    rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_case.py <phantom_dir> <case_dir>
        [--labels data/hasty/phantom/p0/tissue_0.5mm_heart.hv] [--coarsen 1] [--valves leaflets|orifice]
        [--stub-mm 45,30,20,15] [--v-max <m/s, default from the anatomy>] [--u-lattice 0.08] [--cycles 4] [--warmup 2] [--bins 50]
        [--out-factor 2] [--rcr] [--av-radius-mm 16,17] [--chamber-first] [--venous inertial|pressure|inlet]
        [--av-diastasis 0.5] [--wall-mm 2] [--sep-mm 1.5] [--carve-mm 2.5] [--carve-down-mm 4] [--open-thr 0.03] [--open-abs] [--dx-mm 0.5]
        [--pulm-mtt 5] [--sys-capillary-s 2]
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
CARVE = float(opt("--carve-mm", 2.5)) * 1e-3
CARVE_DOWN = float(opt("--carve-down-mm", 4.0)) * 1e-3  # into the downstream chamber (the inflow / outflow tract)
CARVE_RIM = float(opt("--carve-rim-mm", 2.0)) * 1e-3  # beyond the annulus radius
OPEN_THR = float(opt("--open-thr", 0.03))
OPEN_ABS = "--open-abs" in A
ST18 = ndimage.generate_binary_structure(3, 2)  # face + edge neighbours: the D3Q19 links
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
G = np.load(P0 / "graph.npz")
GJ = json.loads((P0 / "graph.json").read_text())
eoff, ea, eb = G["edge_offsets"], G["edge_a"], G["edge_b"]
enames = GJ["edge_names"]
net = np.load(P0 / "flow" / "network_flow.npz")
om, qn, pn = net["omega"], net["q"], net["p"]
cm = vf.read(P0 / "motion" / "cardiac_motion.hv")
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
                br.append({"samples": seg, "edge": int(k), "sgn": int(sg), "parent": parent, "end": True})
            elif k == cont and parent >= 0 and R_ref[sm[0]] >= R_CUT:
                # the continuation leaves the box right at the node: the branch into the node ends in the cut
                br[parent]["end"] = True
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
    if b["end"]:
        nm_ = r["name"] if b["parent"] == -1 and sum(x["root"] == b["root"] and x["end"] for x in branches) == 1 else f"{enames[b['edge']] if b['edge'] >= 0 else r['name']} #{int(sm[-1])}"
        cuts.append({"name": nm_, "class": r["class"], "center": p[-1].tolist(), "normal": t_end.tolist(), "radius": float(rad[-1]), "slab": slab,
                     "sample": int(sm[-1]), "sgn": int(b["sgn"]), "end": (p[-1] + t_end * slab).tolist(), "root": b["root"], "branch": bi})
print(f"{len(cuts)} cuts")
domain = blood | stub
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
    kill = (beyond & (((owner == k_) & (lat < 2.5 * c["radius"])) | (~is_stub & (lat < 1.5 * c["radius"])))) | thin
    spared = beyond & (lat < 2.5 * c["radius"]) & ~kill
    domain[zz[kill], yy[kill], xx[kill]] = False
    if spared.sum() * h**3 * 1e6 > 0.05:
        print(f"cap {c['name'][:28]:28s}: {kill.sum() * h**3 * 1e6:.2f} mL removed beyond the slab, {spared.sum() * h**3 * 1e6:.2f} mL of other vessels / chambers spared")
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
V_MAX = float(opt("--v-max", np.ceil(_vm * 2) / 2))
print(f"velocity scale v_max {V_MAX} m/s (anatomy estimate {_vm:.2f} m/s)")


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
    return {"t": tq.tolist(), "v": np.clip(sm, 0, 1).tolist()}


def perp(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e = np.cross(n, a)
    return e / np.linalg.norm(e)


valves = []
for r in roots:
    if r["class"] not in ("systemic_arterial", "pulmonary_arterial"):
        continue
    s0, s1 = r["path"][0], r["path"][min(2, len(r["path"]) - 1)]
    n = P_ref[s1] - P_ref[s0]
    n /= np.linalg.norm(n)
    q = cm["Q_AO"] if r["class"] == "systemic_arterial" else cm["Q_PA"]
    valves.append({"name": "aortic" if r["class"] == "systemic_arterial" else "pulmonary", "kind": VALVES, "center": P_ref[s0].tolist(), "normal": n.tolist(),
                   "e1": perp(n).tolist(), "radius": float(R_ref[s0] * 1.05), "ring": max(1e-3, 2 * DX), "thickness": max(1.5e-3, 3 * DX), "theta_open": float(np.radians(85)),
                   "leaflets": "tri:90,4", "hole": [0.85, 0.85], "opening": opening(q)})
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
                   "thickness": max(1.5e-3, 3 * DX), "theta_open": float(np.radians(80)), "leaflets": lf, "hole": [0.9, 0.6] if nm == "mitral" else [0.85, 0.85],
                   "opening": opening(q, AV_BRIDGE)})
    print(f"{nm}: centre {np.round(c * 1e3, 1)} mm, normal {np.round(nrm, 2)}, interface {area * 1e4:.2f} cm² → radius {rad * 1e3:.1f} mm")

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
    mk = np.zeros(domain.shape, bool)
    mk[lo_[2] : hi_[2], lo_[1] : hi_[1], lo_[0] : hi_[0]] = m_
    new_ = mk & ~domain
    carved[v["name"]] = mk
    domain |= mk
    print(f"{v['name']}: annulus cylinder carved, {new_.sum() * h**3 * 1e6:.2f} mL of non-blood opened ({new_.sum() / max(mk.sum(), 1) * 100:.0f} % of the cylinder)")
carved_any = np.zeros(domain.shape, bool)
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
    nb = (lat < v["radius"] + 6e-3) & (np.abs(off) < 20e-3)
    region[tuple(idx[nb & (off <= 0)].T)] = U_
    region[tuple(idx[nb & (off > 0)].T)] = D_
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
        add = bad & (da <= SEP) & domain[sl] & ~wall[sl]
        nprox += int(add.sum())
        wall[sl] |= add
domain &= ~wall
region[~domain] = 0
print(f"walls between regions: {nw0} interface voxels, {int(wall.sum())} after growing {WALL_GROW} voxel(s) each side ({(2 + 2 * WALL_GROW) * h * 1e3:.1f} mm) and {nprox} proximity voxels (separation ≥ {SEP} voxels = {SEP * h * 1e3:.1f} mm)")
for v in valves:
    idx = np.argwhere(domain)
    off, lat = frame(v, idx)
    col = (np.abs(off) <= max(0.75e-3, 1.5 * h)) & (lat >= v["radius"]) & (lat <= v["radius"] + 4e-3)
    domain[tuple(idx[col].T)] = False
region[~domain] = 0
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
print(f"domain after walls: {int(domain.sum())} voxels ({domain.sum() * h**3 * 1e6:.0f} mL), {len(keep)} components kept")

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
    rec = {"name": c["name"], "center": c["center"], "normal": c["normal"], "radius": c["radius"], "slab": c["slab"]}
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
for k_, r in enumerate(roots):
    if r["class"] not in ("systemic_arterial", "pulmonary_arterial"):
        continue
    key = "Q_AO" if r["class"] == "systemic_arterial" else "Q_PA"
    Qroot = np.interp(tt, tq, np.asarray(cm[key], float), period=period)
    ci = [i for i, c in enumerate(cuts) if c["root"] == k_]
    if len(ci) == 1 or "--rcr" in A:
        for i in ci:
            cut_out[i]["q_ref"] = {"t": tt.tolist(), "v": Qroot.tolist()}
        continue
    qout = {i: -q_ins[i] for i in ci}
    main = max(ci, key=lambda i: float(np.mean(qout[i])))
    # the branches below R_CUT (lung segmental arteries, intercostals, coronaries) carry the rest of the root's mean
    # flow: share it over all outlets in proportion to their network flow (else it all exits the pressure outlet: 48 of
    # 98 mL/s through one 3 mm segmental artery of the pulmonary tree)
    f_tree = float(np.mean(Qroot)) / max(sum(float(np.mean(qout[i])) for i in ci), 1e-12)
    for i in ci:
        qout[i] = qout[i] * f_tree
    rest = sum(qout[i] for i in ci if i != main)
    for i in ci:
        if i == main:
            cut_out[i]["q_ref"] = {"t": tt.tolist(), "v": (Qroot - rest).tolist()}
        else:
            cut_out[i] |= {"kind": "inlet", "q": {"t": tt.tolist(), "v": (-qout[i]).tolist()}}
    print(f"tree {r['name'][:30]}: outflows × {f_tree:.2f} (flow below R_CUT shared out); pressure outlet {cut_out[main]['name'][:36]} (network mean × f {np.mean(qout[main]) * 1e6:.1f} mL/s, takes "
          f"{np.mean(Qroot - rest) * 1e6:.1f}), {len(ci) - 1} prescribed outflows totalling {np.mean(rest) * 1e6:.1f} mL/s of {np.mean(Qroot) * 1e6:.1f}")
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
        del r_["class"]
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
case = {
    "phantom": str(P0), "source": "rustlib/tools/phantom/lbm_case.py",
    "grid": {"origin": glo.tolist(), "dx": dx, "dims": dims.tolist()},
    "sdf": {"file": "sdf.npy", "origin": sdf_origin.tolist(), "h": h},
    "regions": {"file": "region.npy", "names": {str(v_): k_ for k_, v_ in CHI.items()} | {str(10 + k_): r["name"] for k_, r in enumerate(roots)},
                "allowed": [[int(a_), int(b_)] for a_, b_ in zip(*np.nonzero(allowed)) if a_ < b_],
                "valves": {v["name"]: list(up_down[v["name"]]) for v in valves}},
    "motion": {"file": str((P0 / "motion" / "cardiac_motion.hv").resolve())},
    "fluid": {"nu": 3.5e-3 / 1060.0, "rho": 1060.0},
    # the blood's MR properties of this phantom (its contrast state: cardiac_lbm --spin takes T1 from here)
    "blood": BLOOD_MR,
    "v_max": V_MAX, "u_lattice": float(opt("--u-lattice", 0.08)),
    "valves": valves, "cuts": cut_out,
    "run": {"cycles": int(opt("--cycles", 4)), "warmup": int(opt("--warmup", 2)), "bins": int(opt("--bins", 50)), "out_factor": int(opt("--out-factor", 2)),
            "acc_every": 20, "geom_every": 4, "cwale": 0.5, "inv_iters": 3, "uclip": 0.35, "diag_every": 1000, "ckpt_every": 20000},
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
