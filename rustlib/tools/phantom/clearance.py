"""Vessel clearance of a phantom: no two vessel lumens overlap, and no vessel enters a heart chamber or the myocardium,
at any instant of the phantom's motion (research/flow_phantom.md Tier B4).

    python -I clearance.py <phantom_dir> [--audit-only] [--static] [--carve] [--cardiac-frames 2×motion samples]
        [--r-max 1.4] [--iters 80] [--labels <tissue_0.5mm_heart.hv>] [--tissue-pool 1] [--merged-runs graph|edge]
        [--dump-pairs <npz>]

Geometry as phantom_4d.py builds it, continuous in time: each centreline sample X moves to Φ(X) = Xc + r ΣB(Xc),
Xc = X + U_c(X, t_c) (cardiac field: periodic Catmull-Rom in time over its samples, trilinear, zeros outside; respiratory
field: trilinear, border), and its radius distends with the network pressure, R (1 + 0.75 Δp(t) / (Eh/r)); every
constraint uses the sample's largest radius over the cycle. Time: the cardiac cycle at `--cardiac-frames` phases (two
per motion sample) with a margin of half the relative displacement between neighbouring phases (the instants between),
and the breathing state exactly over r ∈ [0, --r-max] (a point moves linearly in r at a fixed phase; motion.Breathing
clips breath depths to [0.5, 1.4]). `--static`: the reference pose only (diagnostics).

Constraints (each with the vessels' wall thickness from pack_vessels):
- vessel / vessel: samples of different graph components, or of one component more than 2 (r_i + r_j) + 2 mm apart
  along the graph (not a junction), keep their lumens apart by the two walls, and by ≥ 1.5 mm where both radii are
  ≥ 2.5 mm (vessels the LBM resolves; its builder separates regions by 1.5 mm and carves anything closer). Exempt: a
  bifurcation's merged lumen — the branches leaving a junction, over the arc where their lumens (+ gap) overlap
  contiguously from it, followed past the branches' next nodes (`--merged-runs graph`, default; `edge`: only the
  first edges, which flagged two pulmonary vein branches still merged where the first one's 11 mm edge ends);
- vessel / chamber blood (LA, LV, RA, RV, valves; the atlas labels moved by the same field): no lumen + wall inside,
  except a tree within its own heart connection (samples of the root's component within R_root + 10 mm of the heart
  root node, own chamber and valves);
- vessel / myocardium: likewise, except in the connection zones and for the heart's own circulation — the subtrees
  that leave a trunk inside a connection zone (coronary arteries at the aortic root, cardiac veins at the right atrium;
  found from the graph, not by name), which perfuse / drain the wall and move with it.
Distances to the heart tissue at a frame: the reference signed distance × the stretch |F n| along the direction n
from the tissue (∂Φ/∂X at X; the smallest over the frames).

Solve (unless --audit-only): pushes in the reference pose — a pair apart radially from the larger vessel's axis (the
smaller moves more), a sample out of the tissue along the signed-distance gradient — each sample keeping its largest
push, spread over its component by a Gaussian (σ = max(2.5 r, 4 mm)), × mobility (0 within 5 mm of a heart root, 1 from
20 mm: the valves and venous ostia stay), capped at mobility × max(0.5 r + 2 mm, 4 mm) in total, under-relaxed and iterated;
graph nodes re-joined smoothly. What the cap leaves is resolved by shrinking the radii there (running minimum over
±1.5 r along the edge, floor --shrink-floor 0.9 of the radius before the first clearance run: velocity ∝ 1/r², a
deeper shrink would distort the flow; what remains fails the gate). Detection on a sub-sampled centreline (spacing ≤ r / 2, the sag
r / 32 added to the need); the final audit uses every sample and every frame and must find no overlap.
Writes the phantom's graph.npz (points, radius) and clearance.json; exit code 1 if the audit fails.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402
from hastycompute.store import load_any  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deoverlap import gaussian_spread  # noqa: E402

A = sys.argv[1:]
D = Path(A[0])
REPO = next(p for p in Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir())


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


AUDIT_ONLY = "--audit-only" in A
# the truth is continuous in time: the cardiac field a periodic spline of its NTU samples (the simulator moves sources
# every 10 ms), the breathing state r(t) continuous with depths clipped to [0.5, 1.4] (motion.Breathing). Checked: NC
# cardiac phases (default 2 per motion sample) with a margin of half the relative displacement between neighbouring
# phases (covers the instants between), and r ∈ [0, R_MAX] exactly (at a fixed phase a point moves linearly in r)
NC = int(opt("--cardiac-frames", 0))
R_MAX = float(opt("--r-max", 1.4))
ITERS = int(opt("--iters", 80))
G_LBM, R_LBM = 1.5e-3, 2.5e-3
ZONE = 10e-3
CARD_RULE = opt("--cardiac-rule", "epicardial")
PUSH_DIR = opt("--push-dir", "normal")  # normal | radial (before 2026-11-26)
PLATEAU = opt("--plateau", "worst")  # worst (2026-11-27) | deficit | count (before 2026-11-26)
CAP_AWARE = "--cap-blind" not in A
NEAR_REST = "--near-rest" in A
ROOT_MOB = opt("--root-mob", "great")
WATCH = tuple(opt("--watch").split("|")) if opt("--watch") else None
assert ROOT_MOB in ("great", "all"), ROOT_MOB
TREE_GAP = "--tree-gap" in A
t0 = time.time()

# ── inputs ───────────────────────────────────────────────────────────────────────────────────────────────────────────
G = dict(load_any(D / "graph.npz"))
GJ = json.loads((D / "graph.json").read_text())
P0 = np.asarray(G["points"], np.float64)
R0 = np.asarray(G["radius"], np.float64)
# the radius before any clearance run: the shrink floor is relative to it, so repeated runs (solve, carve, solve) do not
# compound (0.6 × 0.6 took the thoracic aorta to r 3.1 mm, 2026-11-25)
R_REF = np.asarray(G.get("radius_ref", G["radius"]), np.float64)
G["radius_ref"] = R_REF.astype(np.float32)
SHRINK_FLOOR = float(opt("--shrink-floor", 0.9))  # 0.9: ≤ 1.23× the velocity; what needs more fails the gate
eoff = np.asarray(G["edge_offsets"], np.int64)
ea, eb = np.asarray(G["edge_a"]), np.asarray(G["edge_b"])
npos = np.asarray(G["node_pos"], np.float64)
ecomp = np.asarray(G["edge_comp"])
ns, ne, nn = len(P0), len(ea), len(npos)
es = np.searchsorted(eoff, np.arange(ns), side="right") - 1


def ename_of(q):
    """the edge name of sample q (diagnostics)"""
    return GJ["edge_names"][int(es[q])]
comp = ecomp[es]
cls_of = {c["id"]: c["class"] for c in GJ["components"]}
V = vf.read(D / "vessels.hv")
assert len(V["points"]) == ns, f"vessels.hv has {len(V['points'])} samples, graph.npz {ns}: pack the graph first"
W = np.asarray(V["wall"], np.float64)
vcls = np.asarray(V["cls"])
net = load_any(D / "flow" / "network_flow.npz")
om, p_h = np.asarray(net["omega"]), np.asarray(net["p"])
assert len(p_h) == ns, "network_flow.npz does not match the graph"
cm = vf.read(D / "motion" / "cardiac_motion.hv")
Tp = float(cm["period"])
U_all = np.asarray(cm["U_mm"], np.float32) * 1e-3  # [T, 3, Z, Y, X]
NTU = U_all.shape[0]
mo_c, mh_c = np.asarray(cm["origin_m"], float), float(cm["voxel_m"])
rmo = vf.read(D / "motion" / "resp_motion.hv")
B_r = np.asarray(rmo["B"], np.float32).sum(0)  # [3, Z, Y, X]
mo_r, mh_r = np.asarray(rmo["origin_m"], float), float(rmo["voxel_m"])
_own = D / "atlas" / "tissue_0.5mm_heart.hv"
TS = vf.read(opt("--labels", str(_own if _own.exists() else REPO / "data/hasty/phantom/p0/tissue_0.5mm_heart.hv")))
tnames = list(TS["tissues"])
LAB = np.asarray(TS["labels"])
lo_h, h_h = np.asarray(TS["origin_m"], float), float(TS["voxel_m"])

# ── heart tissue: signed distances (m, + outside) per chamber and the myocardium, on the 0.5 mm heart labels ───────────
TISSUE = ["blood_LA", "blood_LV", "blood_RA", "blood_RV", "valve", "myocardium"]
# on the label grid (0.5 mm). --tissue-pool 2: pooled 2×2×2 by majority, EDTs 8× faster, but thin walls / valves erode
# and the test phantom's audit missed 17 % of the tissue contacts (906 vs 1 089) — not for gates
_K = int(opt("--tissue-pool", 1))
if _K > 1:
    _z, _y, _x = (n // _K * _K for n in LAB.shape)
    _lab = LAB[:_z, :_y, :_x]
else:
    _lab = LAB
_MASKS = {}
for tn in TISSUE:
    m = _lab == tnames.index(tn)
    if _K > 1:
        m = m.reshape(_z // _K, _K, _y // _K, _K, _x // _K, _K).mean((1, 3, 5)) >= 0.5
    _MASKS[tn] = m


def _edt_job(job):
    # (tissue, outside?) → the EDT of the complement / the mask, in voxels (float32); forked workers see _MASKS
    tn, outside = job
    m = _MASKS[tn]
    return job, ndimage.distance_transform_edt(~m if outside else m).astype(np.float32)


# the 12 distance transforms in parallel (fork: the masks are shared, not copied; ~100 s → ~10 s)
import multiprocessing as _mp  # noqa: E402

with _mp.get_context("fork").Pool(min(12, _mp.cpu_count())) as _pool:
    _res = dict(_pool.map(_edt_job, [(tn, o) for tn in TISSUE for o in (True, False)]))
SD = {tn: ((_res[(tn, True)] - _res[(tn, False)]) * h_h * _K).astype(np.float32) for tn in TISSUE}
del _res
h_h = h_h * _K
SMIN = np.full(ns, np.nan)  # σ_min per sample, computed once at its first position (iterations); the final audit recomputes
hbox_lo, hbox_hi = lo_h, lo_h + np.array(SD["myocardium"].shape[::-1]) * h_h
# exact tissue check (--tissue-check exact, default; 2026-11-25): the tissues' boundary voxels move with the motion like
# the vessel samples, and every (vessel sample, boundary voxel) distance is taken over every frame. The linearised check
# (`linear`: reference distance × the smallest stretch |F n| at the vessel's axis) over-estimated the compression of
# the gap to a wide vessel's wall: on the test phantom it carved 47 mL of chamber where the exact motion needs 6.4.
TISSUE_CHECK = opt("--tissue-check", "exact")
TREACH = float(opt("--tissue-reach", 3.0))  # candidates: reference distance < TREACH (need + 1 mm); 2, 3, 4 find the same pairs on the test phantom
H0 = float(TS["voxel_m"])


def boundary_voxels(L, tn):
    """(z, y, x) indices of the voxels of tissue tn with a 6-neighbour of another label (or the grid edge)"""
    m = L == tnames.index(tn)
    return np.argwhere(m & ~ndimage.binary_erosion(m, border_value=0))


BIDX = {}
BPOS = {}
if TISSUE_CHECK == "exact":
    for tn in TISSUE:
        BIDX[tn] = boundary_voxels(LAB, tn)
        BPOS[tn] = BIDX[tn][:, ::-1] * H0 + lo_h + H0 / 2
    BTREE = {tn: cKDTree(BPOS[tn]) for tn in TISSUE}
P_cur = None
BMOVE = {}  # tissue → (Xc, Br) of its boundary voxels over the frames, float32 on the device, computed on first use



# ── motion (mirrors phantom_4d.FrameMotion; grid_sample align_corners=False ⇔ voxel centres at o + (i + ½) h) ─────────
def tri(F, o, h, X, border):
    """F [C, Z, Y, X] at points X (n, 3) → (n, C)"""
    c = ((X - o) / h - 0.5)[:, ::-1].T
    return np.stack([ndimage.map_coordinates(F[k], c, order=1, mode="nearest" if border else "constant", cval=0.0) for k in range(F.shape[0])], 1)


def cr_weights(tc):
    s = (tc % Tp) / Tp * NTU
    i = int(np.floor(s))
    f = s - i
    w = [(-f**3 + 2 * f**2 - f) / 2, (3 * f**3 - 5 * f**2 + 2) / 2, (-3 * f**3 + 4 * f**2 + f) / 2, (f**3 - f**2) / 2]
    return [((i - 1 + j) % NTU, w[j]) for j in range(4)]


NC = NC or 2 * NTU
TCS = np.arange(NC) * Tp / NC
RESP = [0.0, 0.5 * R_MAX, R_MAX]  # for the tissue stretch only (the pairs use the exact minimum over r)
FRAMES = [(tc, r) for tc in TCS for r in RESP]
Uc_t = [sum(w * U_all[k] for k, w in cr_weights(tc)) for tc in TCS]  # [3, Z, Y, X] per cardiac phase


STATIC = "--static" in A  # diagnostics: the reference pose only (no motion, no distension)
if STATIC:
    FRAMES = [(0.0, 0.0)]


# torch (GPU when available) for the motion, the pair distances and the stretch: the cardiac phases as one batch
import torch  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_Uc = torch.tensor(np.stack(Uc_t), dtype=torch.float32, device=DEV)  # [NC, 3, Z, Y, X]
_Br = torch.tensor(B_r, dtype=torch.float32, device=DEV)[None]  # [1, 3, Z, Y, X]
_nc = torch.tensor([U_all.shape[4], U_all.shape[3], U_all.shape[2]], dtype=torch.float64, device=DEV)
_nr = torch.tensor([B_r.shape[3], B_r.shape[2], B_r.shape[1]], dtype=torch.float64, device=DEV)
_oc, _or = torch.tensor(mo_c, device=DEV), torch.tensor(mo_r, device=DEV)


def _gs(F, o, h, n, P, border):
    """F [B, C, Z, Y, X] at points P [B, m, 3] (world, float64) → [B, m, C] (phantom_4d's GField convention)"""
    g = (2 * (P - o) / (n * h) - 1).to(torch.float32)
    out = torch.nn.functional.grid_sample(F, g[:, None, None], mode="bilinear", padding_mode="border" if border else "zeros", align_corners=False)
    return out[:, :, 0, 0].transpose(1, 2).to(torch.float64)


def _cardiac_resp_t(X):
    """torch (NC, n, 3) cardiac positions and respiratory displacement per unit r; X torch (n, 3) float64"""
    Xb = X[None].expand(NC, -1, -1)
    Xc = Xb + _gs(_Uc, _oc, mh_c, _nc, Xb, False)
    Br = torch.cat([_gs(_Br, _or, mh_r, _nr, Xc[i : i + 16].reshape(1, -1, 3), True).reshape(-1, X.shape[0], 3) for i in range(0, NC, 16)])
    return Xc, Br


def cardiac_and_resp(X):
    """(NC, n, 3) cardiac positions Xc and the respiratory displacement per unit r at Xc (zeros when static); torch"""
    Xt = torch.as_tensor(np.asarray(X, np.float64), device=DEV)
    if STATIC:
        return Xt[None].clone(), torch.zeros_like(Xt)[None]
    return _cardiac_resp_t(Xt)


def min_dist_pairs(Xc, Br, a, b, chunk=100000):
    """per pair (a, b index rows of Xc / Br): the smallest centre distance over every cardiac phase and r ∈ [0, R_MAX]
    (exact in r), minus half the largest relative displacement between neighbouring phases (the instants between)"""
    out = np.empty(len(a), np.float64)
    wf = np.empty(len(a), np.int64)
    for s0 in range(0, len(a), chunk):
        i = torch.as_tensor(a[s0 : s0 + chunk], device=DEV)
        j = torch.as_tensor(b[s0 : s0 + chunk], device=DEV)
        rel = Xc[:, i] - Xc[:, j]  # (NC, m, 3)
        db = Br[:, i] - Br[:, j]
        bb = (db * db).sum(2)
        r_ = torch.clamp(-(rel * db).sum(2) / torch.clamp(bb, min=1e-18), 0.0, R_MAX)
        d = torch.linalg.vector_norm(rel + r_[..., None] * db, dim=2)
        dmin, k = d.min(0)
        if Xc.shape[0] > 1:
            step = torch.linalg.vector_norm(torch.diff(torch.cat([rel, rel[:1]]), dim=0), dim=2)  # periodic neighbours
            dmin = dmin - 0.5 * step.max(0).values
        out[s0 : s0 + chunk], wf[s0 : s0 + chunk] = dmin.cpu().numpy(), k.cpu().numpy()
    return out, wf


def min_dist_cross(XcA, BrA, a, XcB, BrB, b, chunk=200000):
    """min_dist_pairs between two point sets: rows a of (XcA, BrA) against rows b of (XcB, BrB)"""
    out = np.empty(len(a), np.float64)
    wf = np.empty(len(a), np.int64)
    for s0 in range(0, len(a), chunk):
        i = torch.as_tensor(a[s0 : s0 + chunk], device=DEV)
        j = torch.as_tensor(b[s0 : s0 + chunk], device=DEV)
        rel = XcA[:, i].double() - XcB[:, j].double()
        db = BrA[:, i].double() - BrB[:, j].double()
        bb = (db * db).sum(2)
        r_ = torch.clamp(-(rel * db).sum(2) / torch.clamp(bb, min=1e-18), 0.0, R_MAX)
        d = torch.linalg.vector_norm(rel + r_[..., None] * db, dim=2)
        dmin, k = d.min(0)
        if XcA.shape[0] > 1:
            step = torch.linalg.vector_norm(torch.diff(torch.cat([rel, rel[:1]]), dim=0), dim=2)
            dmin = dmin - 0.5 * step.max(0).values
        out[s0 : s0 + chunk], wf[s0 : s0 + chunk] = dmin.cpu().numpy(), k.cpu().numpy()
    return out, wf


def bmove(tn):
    if tn not in BMOVE:
        Xc_, Br_ = cardiac_and_resp(BPOS[tn])
        BMOVE[tn] = (Xc_.float(), Br_.float())
    return BMOVE[tn]


def tissue_need(tn, rows, Rm):
    """per vessel sample: the clearance to tissue tn (lumen + wall, + 0.5 mm to blood) from the axis to a voxel's
    centre, + half a voxel (its face)"""
    return Rm[rows] + W[rows] + (0.0 if tn == "myocardium" else 0.5e-3) + 0.5 * H0


def tissue_exempt(tn, rows, own, cardiac):
    if tn == "myocardium":
        return cardiac[rows] | (own[rows] != "")
    return (own[rows] == tn) | ((tn == "valve") & (own[rows] != ""))


TISSUE_REACH = {"pairs": 0, "beyond_0.75_reach": 0}


def tissue_exact(hi, hrow, Xc_, Br_, Rm, own, cardiac, sag=0.0):
    """exact vessel / tissue violations: per non-exempt sample (hi: sample indices, hrow: their rows in Xc_ / Br_) the
    largest deficit over the tissues' boundary voxels within 2 need + 2 mm (reference distance; a gap compressed below
    half of its reference never occurs in these fields: checked — pairs violating beyond 0.75 of the reach are counted)"""
    out = []
    for tn in TISSUE:
        q = np.flatnonzero(~tissue_exempt(tn, hi, own, cardiac))
        if not len(q):
            continue
        need = tissue_need(tn, hi[q], Rm) + sag * Rm[hi[q]]
        reach = TREACH * (need + 1e-3)
        nb = BTREE[tn].query_ball_point(P_cur[hi[q]], reach)
        cnt = np.array([len(x) for x in nb])
        if not cnt.sum():
            continue
        a_ = np.repeat(np.arange(len(q)), cnt)
        b_ = np.concatenate([np.asarray(x, np.int64) for x in nb if len(x)])
        XcB, BrB = bmove(tn)
        dmin, wf = min_dist_cross(Xc_, Br_, hrow[q][a_], XcB, BrB, b_)
        dref = np.linalg.norm(P_cur[hi[q]][a_] - BPOS[tn][b_], axis=1)
        bad = dmin < need[a_]
        TISSUE_REACH["pairs"] += int(bad.sum())
        TISSUE_REACH["beyond_0.75_reach"] += int((bad & (dref > 0.75 * reach[a_])).sum())
        if not bad.any():
            continue
        dfc = np.where(bad, (need[a_] - dmin) * np.clip(dref / np.maximum(dmin, 1e-4), 1.0, 3.0), -np.inf)
        best = np.full(len(q), -np.inf)
        np.maximum.at(best, a_, dfc)
        wbest = np.zeros(len(q), np.int64)
        hit = dfc >= best[a_]
        wbest[a_[hit]] = wf[hit]
        v = np.flatnonzero(best > 0)
        out.append((hi[q][v], np.full(len(v), tn, object), best[v], wbest[v]))
    return out


def _eig3_min(M):
    """smallest eigenvalue of symmetric 3×3 matrices [..., 3, 3], closed form (trigonometric; batched
    torch.linalg.eigvalsh on CUDA asks for ~1 TB of workspace for ~2 M matrices)"""
    a11, a22, a33 = M[..., 0, 0], M[..., 1, 1], M[..., 2, 2]
    a12, a13, a23 = M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    q = (a11 + a22 + a33) / 3
    p2 = (a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2 * p1
    p = torch.sqrt(torch.clamp(p2 / 6, min=1e-300))
    b11, b22, b33 = (a11 - q) / p, (a22 - q) / p, (a33 - q) / p
    b12, b13, b23 = a12 / p, a13 / p, a23 / p
    detb = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13)
    phi = torch.acos(torch.clamp(detb / 2, -1.0, 1.0)) / 3
    lmin = q + 2 * p * torch.cos(phi + 2 * np.pi / 3)
    return torch.where(p2 < 1e-24, q, lmin)


def stretch_min(X, eps=0.5e-3):
    """(n,) smallest singular value of ∂Φ/∂X at X over every frame: central differences, σ_min = √λ_min(FᵀF)"""
    if STATIC:
        return np.ones(len(X))
    out = np.full(len(X), np.inf)
    for s0 in range(0, len(X), 20000):
        Xt = torch.as_tensor(np.asarray(X[s0 : s0 + 20000], np.float64), device=DEV)
        cols = []
        for a_ in range(3):
            d = torch.zeros(3, dtype=torch.float64, device=DEV)
            d[a_] = eps
            cols.append((_cardiac_resp_t(Xt + d), _cardiac_resp_t(Xt - d)))
        m = torch.full((Xt.shape[0],), float("inf"), dtype=torch.float64, device=DEV)
        for r in RESP:
            Fm = torch.stack([((cp + r * bp) - (cn + r * bn)) / (2 * eps) for (cp, bp), (cn, bn) in cols], -1)  # (NC, n, 3, 3)
            lam = _eig3_min(Fm.transpose(-1, -2) @ Fm)
            m = torch.minimum(m, torch.sqrt(torch.clamp(lam, min=0.0)).min(0).values)
        out[s0 : s0 + 20000] = m.cpu().numpy()
    return out


def stretch_dir(X, N, eps=0.5e-3):
    """(n, k) the smallest stretch |F n_k| over every frame along k directions per point (N (n, k, 3), unit): the
    distance from a point to tissue in direction n scales by |F n|, not by the smallest singular value σ_min (which
    over-reported contacts near the heart and inflated the carving ×3)"""
    if STATIC:
        return np.ones(N.shape[:2])
    out = np.full(N.shape[:2], np.inf)
    for s0 in range(0, len(X), 20000):
        Xt = torch.as_tensor(np.asarray(X[s0 : s0 + 20000], np.float64), device=DEV)
        Nt = torch.as_tensor(np.asarray(N[s0 : s0 + 20000], np.float64), device=DEV)  # (n, k, 3)
        cols = []
        for a_ in range(3):
            d = torch.zeros(3, dtype=torch.float64, device=DEV)
            d[a_] = eps
            cols.append((_cardiac_resp_t(Xt + d), _cardiac_resp_t(Xt - d)))
        m = torch.full(Nt.shape[:2], float("inf"), dtype=torch.float64, device=DEV)
        for r in RESP:
            Fm = torch.stack([((cp + r * bp) - (cn + r * bn)) / (2 * eps) for (cp, bp), (cn, bn) in cols], -1)  # (NC, n, 3, 3)
            Fn = torch.einsum("pnij,nkj->pnki", Fm, Nt)  # (NC, n, k, 3)
            m = torch.minimum(m, torch.linalg.vector_norm(Fn, dim=-1).min(0).values)
        out[s0 : s0 + 20000] = m.cpu().numpy()
    return out


SDIR = np.full((ns, 6), np.nan)  # directional stretch toward each tissue, cached per sample (iterations)


# ── radii over the cycle (mirrors phantom_4d: Eh/r by class) ─────────────────────────────────────────────────────────
fac = np.select([vcls == 2, vcls == 3, vcls == 4, vcls == 6], [0.1, 0.35, 0.15, 0.1], 1.0)
tt_ = np.arange(48) * Tp / 48
dp = np.real(2 * (p_h[:, 1:] @ np.exp(1j * np.outer(om[1:], tt_))))  # (n, 48)


def rmax(R):
    if STATIC:
        return R.copy()
    EHR = (2.0e6 * np.exp(-2253.0 * R) + 8.65e4) * fac
    return R * np.maximum(1 + 0.75 * dp.max(1) / EHR, 1.0)


# ── heart roots, connection zones, the heart's own circulation ───────────────────────────────────────────────────────
CHAMBER_OF = {"systemic_arterial": "blood_LV", "pulmonary_arterial": "blood_RV", "systemic_venous": "blood_RA", "pulmonary_venous": "blood_LA"}
roots = []  # (node, component, chamber name)
for c in GJ["components"]:
    if c["class"] in CHAMBER_OF:
        for rn in c.get("roots", []):
            roots.append((int(rn), int(c["id"]), CHAMBER_OF[c["class"]]))
adj = {}
for k in range(ne):
    adj.setdefault(int(ea[k]), []).append(k)
    adj.setdefault(int(eb[k]), []).append(k)


def node_positions(P):
    out = npos.copy()
    for n, ks in adj.items():
        k = ks[0]
        out[n] = P[eoff[k]] if ea[k] == n else P[eoff[k + 1] - 1]
    return out


def zones(P, R):
    """per sample: the own chamber's name inside its tree's connection zone (else ''), and the heart-circulation flag.
    Zone: the root's component within R_root + 10 mm of the heart root node, plus the ostium — the run along the tree
    from the root over which the lumen (+ wall) reaches contiguously into the own chamber. Heart circulation: on the shortest-path
    tree from the root (graph loops cannot leak), the nodes whose path leaves the zone through an edge other than the
    trunk's exit (the trunk: from the root along the largest lumen) — side branches that start inside the zone"""
    own = np.array([""] * ns, dtype=object)
    cardiac = np.zeros(ns, bool)
    np_ = node_positions(P)
    for rn, cid, chn in roots:
        if rn not in adj:
            continue
        k0 = adj[rn][0]
        r_root = float(R[eoff[k0]] if ea[k0] == rn else R[eoff[k0 + 1] - 1])
        zr = r_root + ZONE
        inz = (comp == cid) & (np.linalg.norm(P - np_[rn], axis=1) <= zr)
        own[inz] = chn
        # the ostium: from the root along the tree while the lumen (+ wall) still reaches into its own chamber — a vessel
        # entering its chamber (contiguous from the root; a later contact with that chamber is a violation)
        Rm_ = rmax(R)
        seen_e, stack = set(), [(rn, None)]
        while stack:
            node, via = stack.pop()
            for k in adj.get(node, []):
                if k == via or k in seen_e:
                    continue
                seen_e.add(k)
                sm = np.arange(eoff[k], eoff[k + 1])
                sm = sm if ea[k] == node else sm[::-1]
                inside = sd_at(chn, P[sm]) < Rm_[sm] + W[sm] + 0.5e-3
                kx = len(sm) if inside.all() else int(np.argmin(inside))
                own[sm[:kx]] = chn
                if kx == len(sm):
                    stack.append((int(eb[k] if ea[k] == node else ea[k]), k))
        # the heart-circulation zone: wider than the own-chamber zone (the right coronary artery leaves the aorta
        # 31 mm from the root node in the atlas, its sinus region ~ 2 r_root); side branches are then kept only if they
        # lie on the heart (CARD_RULE "epicardial", default; "zone": the R_root + 10 mm zone alone, no test)
        zn = np.linalg.norm(np_ - np_[rn], axis=1) <= (2 * r_root + ZONE if CARD_RULE == "epicardial" else zr)
        # trunk exit edge
        node, via, seen_n, exit_e = rn, None, {rn}, None
        while True:
            cand = [k for k in adj.get(node, []) if k != via]
            if not cand:
                break
            k = max(cand, key=lambda k_: R[eoff[k_] : eoff[k_ + 1]].mean())
            other = int(eb[k] if ea[k] == node else ea[k])
            if not zn[other] or other in seen_n:
                exit_e = k
                break
            seen_n.add(other)
            node, via = other, k
        dist_, pred = dijkstra(Gg, indices=rn, return_predecessors=True)
        reach = np.flatnonzero(np.isfinite(dist_) & ~zn)
        card_node = np.zeros(nn, bool)
        group = np.full(nn, -1)
        for n in reach:
            c_, z_ = n, pred[n]
            while z_ >= 0 and not zn[z_]:
                c_, z_ = z_, pred[z_]
            if z_ < 0:
                continue
            ent = next((k for k in adj[int(z_)] if {int(ea[k]), int(eb[k])} == {int(z_), int(c_)}), None)
            card_node[n] = ent is not None and ent != exit_e
            group[n] = -1 if ent is None else ent
        if CARD_RULE == "epicardial":
            # a side branch is the heart's own circulation when it lies on the heart: ≥ 50 % of its subtree's samples
            # within lumen + 5 mm of the myocardium (the coronary arteries in their grooves; not the azygos, the
            # lung arteries or the lower-body veins a wider zone would reach through its trunk choice)
            ek = np.flatnonzero(comp[eoff[:-1]] == cid)
            g_e = np.maximum(group[ea[ek]], group[eb[ek]])
            near_myo = sd_at("myocardium", P) < rmax(R) + W + 5e-3
            for ent in np.unique(g_e[g_e >= 0]):
                ks = ek[g_e == ent]
                sm = np.concatenate([np.arange(eoff[k], eoff[k + 1]) for k in ks])
                if near_myo[sm].mean() < 0.5:
                    card_node[(group == ent)] = False
        for k in range(ne):
            if comp[eoff[k]] == cid and (card_node[ea[k]] or card_node[eb[k]]):
                cardiac[eoff[k] : eoff[k + 1]] = True
    return own, cardiac


# ── graph geodesics for same-component pairs (as deoverlap.py) ──────────────────────────────────────────────────────
seg = np.r_[0, np.linalg.norm(np.diff(P0, axis=0), axis=1)]
arc = np.zeros(ns)
elen = np.zeros(ne)
for k in range(ne):
    s_ = np.cumsum(seg[eoff[k] : eoff[k + 1]])
    s_ -= s_[0]
    arc[eoff[k] : eoff[k + 1]] = s_
    elen[k] = s_[-1]
Gg = sp.coo_matrix((np.r_[elen, elen] + 1e-9, (np.r_[ea, eb], np.r_[eb, ea])), shape=(nn, nn)).tocsr()
DG = dijkstra(Gg, indices=np.arange(nn), limit=0.12)


def need_gap(i, j, R, tree_gap=None):
    """clearance two samples' lumens need: their walls; ≥ 1.5 mm where both radii are ≥ 2.5 mm and the samples lie in
    different trees (the LBM case builder separates and carves different trees' regions by 1.5 mm; the branches of one
    tree share a region, no wall goes between them — `--tree-gap`: 1.5 mm within a tree too, as before 2026-11-26)"""
    big = (R[i] >= R_LBM) & (R[j] >= R_LBM)
    if not (TREE_GAP if tree_gap is None else tree_gap):
        big = big & (comp[i] != comp[j])
    return np.where(big, np.maximum(W[i] + W[j], G_LBM), W[i] + W[j])


def merged_runs(P, R):
    """the bifurcations' merged lumens (allowed overlaps): at every junction cluster (nodes joined by edges shorter than
    the sum of the radii at their ends), for each pair of edges leaving it, the arc lengths from the cluster along both
    edges over which their lumens overlap *contiguously from the junction* (a bifurcation is one lumen splitting up to
    its carina). Returns a dict (e, f) → list of (e leaves from its a-end, f leaves from its a-end, run on e, run on f).
    Overlaps beyond these runs (re-contact after separating), kinks and contacts further away are violations."""
    Rm = rmax(R)
    lab = list(range(nn))

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x
    for k in range(ne):
        if elen[k] < Rm[eoff[k]] + Rm[eoff[k + 1] - 1]:
            ra, rb = find(int(ea[k])), find(int(eb[k]))
            if ra != rb:
                lab[rb] = ra
    root = np.array([find(x) for x in range(nn)])
    ext = {}  # cluster → [(edge, leaves from a-end)]
    for k in range(ne):
        if root[ea[k]] == root[eb[k]]:
            continue  # internal (short) edge of a cluster
        ext.setdefault(root[ea[k]], []).append((k, True))
        ext.setdefault(root[eb[k]], []).append((k, False))
    runs = {}
    for cl, lst in ext.items():
        if len(lst) < 2:
            continue
        for x in range(len(lst)):
            for y in range(x + 1, len(lst)):
                (e, ae), (f, af) = lst[x], lst[y]
                se = np.arange(eoff[e], eoff[e + 1])
                sf = np.arange(eoff[f], eoff[f + 1])
                se, sf = (se if ae else se[::-1]), (sf if af else sf[::-1])
                d_ef, j_ef = cKDTree(P[sf]).query(P[se])
                d_fe, j_fe = cKDTree(P[se]).query(P[sf])
                ov_e = d_ef < Rm[se] + Rm[sf[j_ef]] + need_gap(se, sf[j_ef], R, tree_gap=True)
                ov_f = d_fe < Rm[sf] + Rm[se[j_fe]] + need_gap(sf, se[j_fe], R, tree_gap=True)
                ke = int(np.argmin(ov_e)) if not ov_e.all() else len(ov_e)
                kf = int(np.argmin(ov_f)) if not ov_f.all() else len(ov_f)
                if ke == 0 and kf == 0:
                    continue
                arc_e = (arc[se] if ae else elen[e] - arc[se])
                arc_f = (arc[sf] if af else elen[f] - arc[sf])
                re_ = arc_e[ke - 1] if ke > 0 else -1.0
                rf_ = arc_f[kf - 1] if kf > 0 else -1.0
                runs.setdefault((e, f), []).append((ae, af, re_, rf_))
                runs.setdefault((f, e), []).append((af, ae, rf_, re_))
    return runs


def merged_regions(P, R, l_walk=40e-3):
    """the bifurcations' merged lumens followed across nodes (`--merged-runs graph`, default): at every junction
    cluster (as in merged_runs) each exterior edge starts a side — that edge and the edges beyond it, walked outward
    up to 4 r + l_walk (40 mm) from the cluster. A side's sample belongs to the cluster's merged lumen while its lumen
    (+ gap) overlaps another side's contiguously from the junction, also past the side's next node (two pulmonary vein
    branches leaving one junction side by side for 11 mm, then the first one's next junction while they still overlap
    by 5 mm). Overlap: in the reference pose or at any frame (`MERGED_DYN`: the smallest distance over the motion —
    the carina moves; a portal vein and its bridge merged further out at some frames). The samples of a cluster's own
    short edges are its core (-1), merged with every side (a 7 mm pulmonary vein edge between two close junctions had
    no side and every contact on it counted). Returns per sample a dict cluster → side."""
    Rm = rmax(R)
    lab = list(range(nn))

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x
    for k in range(ne):
        if elen[k] < Rm[eoff[k]] + Rm[eoff[k + 1] - 1]:
            ra, rb = find(int(ea[k])), find(int(eb[k]))
            if ra != rb:
                lab[rb] = ra
    root = np.array([find(x) for x in range(nn)])
    ext, core = {}, {}
    for k in range(ne):
        if root[ea[k]] == root[eb[k]]:
            core.setdefault(root[ea[k]], []).append(k)
            continue
        ext.setdefault(root[ea[k]], []).append((k, True))
        ext.setdefault(root[eb[k]], []).append((k, False))
    mem = {}

    def build(cl, lst, lim):
        """sides walked up to lim: per side the edges (edge, outward, start distance, parent index); sample arrays and
        the nearest-sample pairs to test (each sample against the nearest sample of each other side)"""
        seen = {k for k, _ in lst}
        sides = []
        for k0, a0 in lst:
            order = [(k0, a0, 0.0, -1)]
            q = 0
            while q < len(order):
                k, a_, d0, _ = order[q]
                far = int(eb[k] if a_ else ea[k])
                d1 = d0 + elen[k]
                if d1 < lim and root[far] != cl:
                    for k2 in adj.get(far, ()):
                        if k2 in seen:
                            continue
                        seen.add(k2)
                        order.append((k2, int(ea[k2]) == far, d1, q))
                q += 1
            sides.append(order)
        samp, side_of, edge_pos = [], [], []
        for x, order in enumerate(sides):
            for q, (k, a_, _, _) in enumerate(order):
                sidx = np.arange(eoff[k], eoff[k + 1])
                sidx = sidx if a_ else sidx[::-1]
                samp.append(sidx)
                side_of.append(np.full(len(sidx), x))
                edge_pos.append((x, q, len(sidx)))
        S = np.concatenate(samp)
        SX = np.concatenate(side_of)
        pa, pb, pos = [], [], []
        trees = [cKDTree(P[S[SX == y]]) for y in range(len(sides))]
        for x in range(len(sides)):
            ix = np.flatnonzero(SX == x)
            for y in range(len(sides)):
                if y == x:
                    continue
                Sy = S[SX == y]
                _, j_ = trees[y].query(P[S[ix]])
                pa.append(S[ix])
                pb.append(Sy[j_])
                pos.append(ix)
        return {"cl": cl, "sides": sides, "samp": samp, "edge_pos": edge_pos, "n": len(S), "pa": np.concatenate(pa), "pb": np.concatenate(pb), "pos": np.concatenate(pos)}

    def evaluate(items):
        """overlap per tested pair, statically and (MERGED_DYN) at the closest frame, all clusters in one batch"""
        if not items:
            return
        pa = np.concatenate([it["pa"] for it in items])
        pb = np.concatenate([it["pb"] for it in items])
        thr = Rm[pa] + Rm[pb] + need_gap(pa, pb, R, tree_gap=True)
        ov = np.linalg.norm(P[pa] - P[pb], axis=1) < thr
        if MERGED_DYN and not STATIC and len(pa):
            u_, inv = np.unique(np.r_[pa, pb], return_inverse=True)
            Xc_, Br_ = cardiac_and_resp(P[u_])
            dmin, _ = min_dist_pairs(Xc_, Br_, inv[: len(pa)], inv[len(pa) :])
            ov |= dmin < thr
        o = 0
        for it in items:
            m = len(it["pa"])
            ovs = np.zeros(it["n"], bool)
            np.logical_or.at(ovs, it["pos"], ov[o : o + m])
            it["ov"] = ovs
            o += m

    def regions(it):
        """contiguity from the junction, edge by edge along each side's walk → [(samples, side)], any first edge full"""
        off, full, got = 0, {}, []
        for (x, q, m), sidx in zip(it["edge_pos"], it["samp"]):
            par = it["sides"][x][q][3]
            entry = True if par < 0 else full.get((x, par), False)
            o_ = it["ov"][off : off + m]
            n_in = (int(np.argmin(o_)) if not o_.all() else m) if entry else 0
            full[(x, q)] = n_in == m
            got.append((sidx[:n_in], x))
            off += m
        return got, any(full.get((x, 0), False) for x in range(len(it["sides"])))

    # first only the edges leaving each cluster; walked further only where a run covers a whole first edge
    clusters = [(cl, lst) for cl, lst in ext.items() if len(lst) >= 2]
    shallow = [build(cl, lst, 0.0) for cl, lst in clusters]
    evaluate(shallow)
    deep_need = []
    for (cl, lst), it in zip(clusters, shallow):
        got, anyfull = regions(it)
        if anyfull:
            lmax = 4 * float(max(Rm[eoff[k]] if a_ else Rm[eoff[k + 1] - 1] for k, a_ in lst)) + l_walk
            deep_need.append((cl, lst, lmax))
            continue
        for sel_, x in got:
            for s_ in sel_:
                mem.setdefault(int(s_), {})[int(cl)] = x
    deep = [build(cl, lst, lmax) for cl, lst, lmax in deep_need]
    evaluate(deep)
    for it in deep:
        got, _ = regions(it)
        for sel_, x in got:
            for s_ in sel_:
                mem.setdefault(int(s_), {})[int(it["cl"])] = x
    # the clusters' own short edges: core, merged with every side of their cluster
    for cl, ks in core.items():
        if cl in ext and len(ext[cl]) >= 1:
            for k in ks:
                for s_ in range(eoff[k], eoff[k + 1]):
                    mem.setdefault(int(s_), {})[int(cl)] = -1
    return mem


RUNS = {}
MEM = {}
_N_VIOL = 0
MERGED_MODE = opt("--merged-runs", "graph")
MERGED_DYN = "--merged-static" not in A  # merged lumens tested over the motion too (2026-11-26)


def eligible(i, j, R):
    """pairs that must not overlap: different components; or, within a tree, pairs that are neither a bifurcation's
    merged lumen (merged_runs) nor within 2 (r_i + r_j) + 2 mm of each other along the graph"""
    e, f = es[i], es[j]
    si, sj, le, lf = arc[i], arc[j], elen[e], elen[f]
    gd = np.minimum.reduce([si + DG[ea[e], ea[f]] + sj, si + DG[ea[e], eb[f]] + (lf - sj), (le - si) + DG[eb[e], ea[f]] + sj, (le - si) + DG[eb[e], eb[f]] + (lf - sj)])
    gd = np.where(e == f, np.abs(si - sj), gd)
    # near a junction (along the graph) with the radii the overlap test uses — the largest over the cycle (`--near-rest`:
    # the rest radii, as before; the left coronary artery 30 mm from the distended aortic root's sample counted)
    Rn = R if NEAR_REST else rmax(R)
    el = (comp[i] != comp[j]) | (gd > 2 * (Rn[i] + Rn[j]) + 2e-3)
    cand = np.flatnonzero(el & (comp[i] == comp[j]))
    if MERGED_MODE == "graph":
        for q in cand:
            mi, mj = MEM.get(int(i[q])), MEM.get(int(j[q]))
            if mi and mj and any(mj.get(c_, x_) != x_ for c_, x_ in mi.items()):
                el[q] = False
        return el
    for q in cand:
        for ae, af, re_, rf_ in RUNS.get((int(e[q]), int(f[q])), ()):
            ai = arc[i[q]] if ae else le[q] - arc[i[q]]
            aj = arc[j[q]] if af else lf[q] - arc[j[q]]
            if ai <= re_ + 1e-9 and aj <= rf_ + 1e-9:
                el[q] = False
                break
    return el


print(f"inputs: {ns} samples, {ne} edges, {len(roots)} heart roots, {NC} cardiac phases × r ∈ [0, {R_MAX}], heart tissue distances ({time.time() - t0:.0f}s)", flush=True)


def sd_at(tn, X):
    return tri(SD[tn][None], lo_h, h_h, X, True)[:, 0]


def sd_grad(tn, X, eps=0.5e-3):
    g = np.stack([sd_at(tn, X + np.eye(3)[a] * eps) - sd_at(tn, X - np.eye(3)[a] * eps) for a in range(3)], 1)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


# ── constraint evaluation ───────────────────────────────────────────────────────────────────────────────────────────
# the solver's centreline sample spacing: ≤ r / 2 (≥ 0.5 mm) and, by default, ≤ 2 mm (2026-11-27: on the 11 mm pulmonary
# bifurcation the r / 2 spacing of 5.5 mm missed a 1.77 mm re-contact of its branches that only the full audit found;
# the solver never pushed it). `--sub-max-mm 0`: no cap, as before
SUB_MAX = float(opt("--sub-max-mm", 2.0)) * 1e-3


def subsample(R):
    keep = np.zeros(ns, bool)
    for k in range(ne):
        a, b = eoff[k], eoff[k + 1]
        last = -np.inf
        for q in range(a, b):
            sp = max(0.5 * R[q], 0.5e-3)
            if SUB_MAX > 0:
                sp = min(sp, max(SUB_MAX, 0.5e-3))
            if arc[q] - last >= sp or q == b - 1:
                keep[q] = True
                last = arc[q]
    return np.flatnonzero(keep)


def violations(P, R, full=False):
    """vessel pairs and tissue contacts that violate clearance at some frame. Returns
    pairs: (i, j, deficit_ref, worst frame), tissue: (i, tissue, deficit_ref, worst frame)"""
    Rm = rmax(R)
    idx = np.arange(ns) if full else subsample(R)
    sag = 0.0 if full else 1.0 / 32
    global RUNS, MEM, _N_VIOL
    _N_VIOL += 1
    if MERGED_MODE == "graph":
        if full or not MEM or _N_VIOL % 4 == 1:  # the solver moves ≤ a fraction of a mm per iteration: every 4th
            MEM = merged_regions(P, R)
    else:
        RUNS = merged_runs(P, R)
    Pi = P[idx]
    Xc_, Br_ = cardiac_and_resp(Pi)
    # vessel / vessel candidates from the reference pose, allowing 35 % compression + 1 mm of relative motion: the larger
    # sample of a pair searches (radius from its own reach), so small vessels do not search far
    tree = cKDTree(Pi)
    rc = Rm[idx] + np.where(R[idx] >= R_LBM, np.maximum(W[idx], G_LBM / 2), W[idx])
    reach = 1.35 * 2 * rc + 1e-3
    nb = tree.query_ball_point(Pi, reach)
    a_ = np.repeat(np.arange(len(Pi)), [len(x) for x in nb])
    b_ = np.concatenate([np.asarray(x, np.int64) for x in nb]) if len(nb) else np.zeros(0, np.int64)
    keep = (rc[b_] < rc[a_]) | ((rc[b_] == rc[a_]) & (b_ > a_))
    a_, b_ = a_[keep], b_[keep]
    d0 = np.linalg.norm(Pi[a_] - Pi[b_], axis=1)
    keep = d0 < 1.35 * (rc[a_] + rc[b_]) + 1e-3
    cand = np.stack([a_[keep], b_[keep]], 1)
    i_, j_, d0 = idx[cand[:, 0]], idx[cand[:, 1]], d0[keep]
    el = eligible(i_, j_, R)
    cand, i_, j_, d0 = cand[el], i_[el], j_[el], d0[el]
    gap = need_gap(i_, j_, R)
    need = Rm[i_] + Rm[j_] + gap + sag * (Rm[i_] + Rm[j_])
    dmin, wf = min_dist_pairs(Xc_, Br_, cand[:, 0], cand[:, 1])
    bad = dmin < need
    pairs = (i_[bad], j_[bad], ((need - dmin) * d0 / np.maximum(dmin, 1e-6))[bad], wf[bad])
    # vessel / heart tissue
    own, cardiac = zones(P, R)
    inbox = np.all((Pi > hbox_lo + 2e-3) & (Pi < hbox_hi - 2e-3), 1)
    hi = idx[inbox]
    tis = []
    global P_cur
    P_cur = P
    if len(hi) and TISSUE_CHECK == "exact":
        tis = tissue_exact(hi, np.flatnonzero(inbox), Xc_, Br_, Rm, own, cardiac, sag)
    elif len(hi):
        Nd = np.stack([sd_grad(tn, P[hi]) for tn in TISSUE], 1)  # (m, 6, 3): away from each tissue
        if full:
            sdir = stretch_dir(P[hi], Nd)  # (m, 6) over every frame, at the current positions
        else:
            miss = np.isnan(SDIR[hi, 0])
            if miss.any():
                SDIR[hi[miss]] = stretch_dir(P[hi[miss]], Nd[miss])
            sdir = SDIR[hi]
        for kt, tn in enumerate(TISSUE):
            smin = sdir[:, kt]
            sd = sd_at(tn, P[hi])
            if tn == "myocardium":
                exempt = cardiac[hi] | (own[hi] != "")
                need_t = Rm[hi] + W[hi]
            else:
                exempt = (own[hi] == tn) | ((tn == "valve") & (own[hi] != ""))
                need_t = Rm[hi] + W[hi] + 0.5e-3
            dmin_t = sd * np.where(sd > 0, smin, 1.0)
            wf_t = np.zeros(len(hi), np.int32)
            b_ = (dmin_t < need_t) & ~exempt
            sc = np.where(dmin_t > 0, sd / np.maximum(dmin_t, 1e-9), 1.0)
            tis.append((hi[b_], np.full(b_.sum(), tn, object), ((need_t - dmin_t) * sc)[b_], wf_t[b_]))
    tis = tuple(np.concatenate([t[k] for t in tis]) if tis else np.zeros(0) for k in range(4))
    return pairs, tis


def summary(pairs, tis, R):
    s = {"pairs": int(len(pairs[0])), "pair_max_mm": float(pairs[2].max() * 1e3) if len(pairs[0]) else 0.0, "tissue": int(len(tis[0])),
         "tissue_max_mm": float(tis[2].max() * 1e3) if len(tis[0]) else 0.0}
    names = GJ["edge_names"]
    worst = {}
    for i, j, d in zip(pairs[0], pairs[1], pairs[2]):
        key = " | ".join(sorted([names[es[i]][:30], names[es[j]][:30]]))
        worst[key] = max(worst.get(key, 0.0), float(d))
    for i, tn, d in zip(tis[0], tis[1], tis[2]):
        key = f"{names[es[i]][:30]} → {tn}"
        worst[key] = max(worst.get(key, 0.0), float(d))
    s["worst"] = {k: round(v * 1e3, 2) for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:25]}
    same = comp[pairs[0]] == comp[pairs[1]] if len(pairs[0]) else np.zeros(0, bool)
    s["pairs_same_component"] = int(same.sum())
    if len(pairs[0]):
        i_, j_ = pairs[0][same], pairs[1][same]
        e_, f_ = es[i_], es[j_]
        share = (ea[e_] == ea[f_]) | (ea[e_] == eb[f_]) | (eb[e_] == ea[f_]) | (eb[e_] == eb[f_])
        si, sj, le, lf = arc[i_], arc[j_], elen[e_], elen[f_]
        gd = np.minimum.reduce([si + DG[ea[e_], ea[f_]] + sj, si + DG[ea[e_], eb[f_]] + (lf - sj), (le - si) + DG[eb[e_], ea[f_]] + sj, (le - si) + DG[eb[e_], eb[f_]] + (lf - sj)])
        gd = np.where(e_ == f_, np.abs(si - sj), gd)
        s["same_component_kinds"] = {"same_edge (bends, kinks)": int((e_ == f_).sum()), "edges_sharing_a_node (bifurcations)": int(((e_ != f_) & share).sum()),
                                     "graph_≤50mm": int(((e_ != f_) & ~share & (gd <= 0.05)).sum()), "graph_50–120mm": int(((e_ != f_) & ~share & (gd > 0.05) & (gd <= 0.12)).sum()),
                                     "graph_>120mm_or_unreached": int(((e_ != f_) & ~share & (gd > 0.12)).sum())}
    s["pairs_other_component"] = int((~same).sum())
    s["edge_pairs"] = len({tuple(sorted((int(es[i]), int(es[j])))) for i, j in zip(pairs[0], pairs[1])})
    s["tissue_by_kind"] = {tn: int((tis[1] == tn).sum()) for tn in TISSUE} if len(tis[0]) else {}
    return s


def smooth_along_edges(disp, R):
    """the displacement low-passed along every edge (Gaussian, σ = max(mean r, 2 mm) of arc length), the ends kept (graph
    nodes stay joined) and blended in over one σ: a vessel moves as a smooth rod. Without it the pushes from neighbours
    on different sides bent vessels back and forth — the solver itself folded the thoracic aorta (a 43 mm fold)"""
    out = disp.copy()
    for k in range(ne):
        a, b = int(eoff[k]), int(eoff[k + 1])
        n = b - a
        if n < 5:
            continue
        sp = max(elen[k] / (n - 1), 1e-5)
        sg = max(float(R[a:b].mean()), 2e-3) / sp
        sm = ndimage.gaussian_filter1d(disp[a:b], sg, axis=0, mode="nearest")
        i_ = np.arange(n)
        w = np.clip(np.minimum(i_, n - 1 - i_) / sg, 0.0, 1.0)
        out[a:b] = disp[a:b] + w[:, None] * (sm - disp[a:b])
    return out


# push spreading along the vessels (`--spread edge`, default; `spatial`: every sample of the component within 3 σ, as
# before): a push reaches the samples of its own edge and of the edges sharing a node with it — never a crossing branch
# of the same tree, whose opposite push the spatial average cancelled (the right coronary artery and an unnamed branch
# 0.33 mm apart both moved past their caps, together)
SPREAD = opt("--spread", "edge")
_adj_keys = np.unique(np.concatenate([[int(a) * ne + int(b) for a in ks for b in ks] for ks in adj.values()])) if adj else np.zeros(0, np.int64)


def _along_vessel(si, di):
    e_, f_ = es[si].astype(np.int64), es[di].astype(np.int64)
    return (e_ == f_) | np.isin(e_ * ne + f_, _adj_keys)


# ── solve ───────────────────────────────────────────────────────────────────────────────────────────────────────────
P, R = P0.copy(), R0.copy()
pairs, tis = violations(P, R)
before = summary(pairs, tis, R)
print(f"before: {before['pairs']} pair / {before['tissue']} tissue violations (sub-sampled), worst {before['pair_max_mm']:.2f} / {before['tissue_max_mm']:.2f} mm ({time.time() - t0:.0f}s)", flush=True)
rep = {"before": before}
if not AUDIT_ONLY:
    root_pos = np.array([npos[rn] for rn, _, _ in roots]) if roots else np.zeros((0, 3))
    dist_root, iroot = cKDTree(root_pos).query(P0) if len(roots) else (np.full(ns, np.inf), np.zeros(ns, int))
    mob = np.clip((dist_root - 5e-3) / 15e-3, 0.0, 1.0)
    # the 5–20 mm ramp anchors the great vessels where they meet the heart; a thin branch near a root (a coronary
    # artery on the aortic root, r < ½ the root's) is anchored only at its ostium: free beyond 2–6 mm (2026-11-26: the
    # left coronary artery could not leave the distending aortic root, 1.59 mm, the worst pair). `--root-mob all`: old
    if ROOT_MOB == "great" and len(roots):
        r_root = np.array([float(R0[cKDTree(P0).query(npos[rn])[1]]) for rn, _, _ in roots])
        thin_ = R0 < 0.5 * r_root[iroot]
        mob = np.where(thin_, np.clip((dist_root - 2e-3) / 4e-3, 0.0, 1.0), mob)
        print(f"root mobility: {int((thin_ & (dist_root < 20e-3)).sum())} thin-branch samples within 20 mm of a heart root anchored at their ostium only", flush=True)
    cap = mob * np.maximum(0.5 * R0 + 2e-3, 4e-3)  # small vessels may move ≥ 4 mm (the atlas places them to ~mm)
    T_ = np.zeros_like(P)
    for k in range(ne):
        a, b = eoff[k], eoff[k + 1]
        if b - a >= 2:
            T_[a:b] = np.gradient(P0[a:b], axis=0)
    T_ /= np.maximum(np.linalg.norm(T_, axis=1, keepdims=True), 1e-12)
    disp = np.zeros_like(P)
    hist, hist_w = [], []
    for it in range(ITERS):
        if len(pairs[0]) == 0 and len(tis[0]) == 0:
            break
        # plateau stop: < 2 % less total deficit over the last 10 iterations (the rest goes to the radius shrink); the
        # count alone plateaus while the overlaps still get shallower (full9: 1 310 of 1 343 violating samples ended
        # at the shrink floor, two thirds of them below their displacement cap). `--plateau count`: as before
        hist.append(len(pairs[0]) + len(tis[0]) if PLATEAU == "count" else float(np.sum(pairs[2])) + float(np.sum(tis[2])))
        # `worst` (default): also the deepest violation must have stopped improving — the total is dominated by pairs
        # that no longer move, while a single deep one still closes slowly (the left coronary artery on the aortic root
        # went 1.52 → 1.04 mm, −0.03 mm per iteration, when the total plateaued at iteration 21)
        hist_w.append(max(float(np.max(pairs[2])) if len(pairs[0]) else 0.0, float(np.max(tis[2])) if len(tis[0]) else 0.0))
        stalled = len(hist) > 10 and hist[-1] > 0.98 * hist[-11]
        if PLATEAU == "worst":
            stalled = stalled and hist_w[-1] > 0.97 * hist_w[-11]
        if stalled:
            print(f"  plateau after {it} iterations ({hist[-11]:.4g} → {hist[-1]:.4g} {'violations' if PLATEAU == 'count' else 'm total deficit'}"
                  + (f", worst {hist_w[-11] * 1e3:.2f} → {hist_w[-1] * 1e3:.2f} mm" if PLATEAU == "worst" else "") + ")", flush=True)
            break
        push = np.zeros_like(P)
        pmag = np.zeros(ns)
        cand_i, cand_v = [], []
        if len(pairs[0]):
            i, j, dref = pairs[0], pairs[1], pairs[2]
            swap = R[i] < R[j]
            L_, S_ = np.where(swap, j, i), np.where(swap, i, j)
            v = P[S_] - P[L_]
            tl = T_[L_]
            nr = v - (v * tl).sum(1, keepdims=True) * tl
            nl = np.linalg.norm(nr, axis=1)
            alt = np.cross(tl, T_[S_])
            an = np.linalg.norm(alt, axis=1)
            fb = np.cross(tl, np.where(np.abs(tl[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]))
            alt = np.where(an[:, None] > 1e-6, alt / np.maximum(an, 1e-12)[:, None], fb / np.maximum(np.linalg.norm(fb, axis=1, keepdims=True), 1e-12))
            u = np.where(nl[:, None] > 1e-7, nr / np.maximum(nl, 1e-12)[:, None], alt)
            if PUSH_DIR == "normal":
                # crossing tubes separate along their common normal: the part of the radial direction along the smaller
                # vessel's own (projected) tangent only slides the crossing along it (an unnamed right-coronary branch
                # crossing the RCA's axis at 0.3 mm stayed put); removed in proportion to sin² of the crossing angle,
                # so parallel vessels keep the radial push
                ts = T_[S_]
                tsp = ts - (ts * tl).sum(1, keepdims=True) * tl
                sn = np.linalg.norm(tsp, axis=1)
                tsp = tsp / np.maximum(sn, 1e-12)[:, None]
                ur = nr - (np.clip(sn, 0.0, 1.0) ** 2 * (nr * tsp).sum(1))[:, None] * tsp
                urn = np.linalg.norm(ur, axis=1)
                sgn = np.where((alt * v).sum(1) < 0, -1.0, 1.0)[:, None]
                u = np.where(urn[:, None] > 0.05 * np.maximum(nl, 1e-9)[:, None], ur / np.maximum(urn, 1e-12)[:, None], sgn * alt)
            # the push splits by mobility and the displacement left under each sample's cap (a capped sample's share
            # went to waste; its partner now takes it)
            free = np.clip(1.0 - np.linalg.norm(disp, axis=1) / np.maximum(cap, 1e-9), 0.0, 1.0) if CAP_AWARE else np.ones(ns)
            wL, wS = mob[L_] * free[L_] * R[S_], mob[S_] * free[S_] * R[L_]
            tot = np.maximum(wL + wS, 1e-12)
            cand_i += [L_, S_]
            cand_v += [-(dref * wL / tot)[:, None] * u, (dref * wS / tot)[:, None] * u]
        if len(tis[0]):
            ti = tis[0].astype(int)
            gdir = np.zeros((len(ti), 3))
            for tn in TISSUE:
                m_ = tis[1] == tn
                if m_.any():
                    gdir[m_] = sd_grad(tn, P[ti[m_]])
            cand_i.append(ti)
            cand_v.append(tis[2].astype(float)[:, None] * gdir)
        ci = np.concatenate(cand_i)
        cv = np.concatenate(cand_v)
        mag = np.linalg.norm(cv, axis=1)
        np.maximum.at(pmag, ci, mag)
        win = np.flatnonzero(mag >= pmag[ci])
        _, first = np.unique(ci[win], return_index=True)
        sel = win[first]
        push[ci[sel]] = cv[sel]
        on = np.flatnonzero(pmag > 0)
        step = np.zeros_like(P)
        for c in np.unique(comp[on]):
            src = on[comp[on] == c]
            dst = np.flatnonzero(comp == c)
            step[dst] = gaussian_spread(P, push, src, dst, np.maximum(2.5 * R[src], 4e-3), allow=None if SPREAD == "spatial" else _along_vessel)
        disp = disp + 0.5 * step * mob[:, None]
        disp = smooth_along_edges(disp, R0)
        dn = np.linalg.norm(disp, axis=1)
        over = dn > cap
        disp[over] *= (cap[over] / np.maximum(dn[over], 1e-12))[:, None]
        if WATCH and len(pairs[0]):
            # --watch "A|B" (diagnostics): the worst pair between two named vessels, what its samples were pushed by and how
            # far they moved against their caps
            wi, wj = pairs[0].astype(int), pairs[1].astype(int)
            na_, nb_ = np.array([ename_of(x) for x in wi]), np.array([ename_of(x) for x in wj])
            hit = ((na_ == WATCH[0]) & (nb_ == WATCH[1])) | ((na_ == WATCH[1]) & (nb_ == WATCH[0]))
            if hit.any():
                k_ = np.flatnonzero(hit)[np.argmax(pairs[2][hit])]
                a_, b_ = wi[k_], wj[k_]
                print(f"  watch it {it}: {int(hit.sum())} pairs, worst {pairs[2][k_] * 1e3:.2f} mm at frame {int(pairs[3][k_])}; "
                      + "; ".join(f"{ename_of(q)[:20]} s{q}: push {np.linalg.norm(push[q]) * 1e3:.2f} mm, step {np.linalg.norm(step[q]) * 1e3:.2f}, disp {dn[q] * 1e3:.2f} / cap {cap[q] * 1e3:.2f} mm, mob {mob[q]:.2f}" for q in (a_, b_)), flush=True)
        P = P0 + disp
        pairs, tis = violations(P, R)
        at_cap = int((dn >= cap * (1 - 1e-6)).sum() - (cap <= 0).sum())
        print(f"  it {it + 1}: {len(pairs[0])} pair / {len(tis[0])} tissue violations ({at_cap} samples at their cap), worst {pairs[2].max() * 1e3 if len(pairs[0]) else 0:.2f} / "
              f"{tis[2].max() * 1e3 if len(tis[0]) else 0:.2f} mm, displacement ≤ {dn.max() * 1e3:.1f} mm ({time.time() - t0:.0f}s)", flush=True)
    # graph nodes re-joined smoothly (mean of the end samples, spread along each edge)
    nsum = np.zeros((nn, 3))
    ncnt = np.zeros(nn)
    ends = np.r_[eoff[:-1], eoff[1:] - 1]
    enode = np.r_[ea, eb]
    np.add.at(nsum, enode, P[ends])
    np.add.at(ncnt, enode, 1)
    used = ncnt > 0
    npos2 = npos.copy()
    npos2[used] = nsum[used] / ncnt[used, None]
    for k in range(ne):
        a, b = eoff[k], eoff[k + 1]
        q = P[a:b]
        ar = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        L_ = max(ar[-1], 1e-9)
        sa, sb = max(R[a], 2e-3), max(R[b - 1], 2e-3)
        P[a:b] = q + (np.exp(-ar**2 / (2 * sa**2)) * (1 - ar / L_))[:, None] * (npos2[ea[k]] - q[0]) + (np.exp(-(L_ - ar) ** 2 / (2 * sb**2)) * (ar / L_))[:, None] * (npos2[eb[k]] - q[-1])
    # residual: shrink the radii where the cap left a deficit (pairs: ∝ r; tissue: the vessel alone)
    pairs, tis = violations(P, R, full=True)
    shrink = np.ones(ns)
    for _ in range(4):
        if len(pairs[0]) == 0 and len(tis[0]) == 0:
            break
        f_ = np.ones(ns)
        if len(pairs[0]):
            i, j, dref = pairs[0], pairs[1], pairs[2]
            Rm_ = rmax(R)
            np.minimum.at(f_, i, np.clip(1 - dref * R[i] / (R[i] + R[j]) / np.maximum(Rm_[i], 1e-9) * 1.1, 0, 1))
            np.minimum.at(f_, j, np.clip(1 - dref * R[j] / (R[i] + R[j]) / np.maximum(Rm_[j], 1e-9) * 1.1, 0, 1))
        if len(tis[0]):
            ti = tis[0].astype(int)
            np.minimum.at(f_, ti, np.clip(1 - tis[2].astype(float) / np.maximum(rmax(R)[ti], 1e-9) * 1.1, 0, 1))
        for k in np.unique(es[f_ < 1]):
            a, b = eoff[k], eoff[k + 1]
            s_, x_, r_ = arc[a:b], f_[a:b], R[a:b]
            f_[a:b] = np.array([x_[np.abs(s_ - s_[q]) <= 1.5 * r_[q]].min() for q in range(b - a)])
        R = np.maximum(R * f_, SHRINK_FLOOR * R_REF)
        R = np.minimum(R, R0)  # never grow
        shrink = R / R_REF
        TISSUE_REACH.update({"pairs": 0, "beyond_0.75_reach": 0})
        pairs, tis = violations(P, R, full=True)
    G["points"] = P.astype(np.float32)
    G["radius"] = R.astype(np.float32)
    G["node_pos"] = npos2.astype(np.float32)
    np.savez_compressed(D / "graph.npz", **G)
    dm = np.linalg.norm(P - P0, axis=1)
    names_ = GJ["edge_names"]
    shr_v = {}
    for k_ in np.unique(es[shrink < 1 - 1e-6]):
        nm = names_[k_]
        shr_v[nm] = min(shr_v.get(nm, 1.0), float(shrink[eoff[k_] : eoff[k_ + 1]].min()))
    rep["solve"] = {"max_displacement_mm": float(dm.max() * 1e3), "samples_moved_over_1mm": int((dm > 1e-3).sum()), "samples_shrunk": int((shrink < 1 - 1e-6).sum()),
                    "min_radius_factor": float(shrink.min()), "shrink_floor": SHRINK_FLOOR,
                    "vessels_shrunk": {k: round(v, 3) for k, v in sorted(shr_v.items(), key=lambda kv: kv[1])[:40]}}
    print(f"solve: displacement ≤ {dm.max() * 1e3:.1f} mm ({(dm > 1e-3).sum()} samples > 1 mm), {(shrink < 1 - 1e-6).sum()} samples shrunk (≥ {shrink.min():.2f}× the pre-clearance radius, floor {SHRINK_FLOOR}); most shrunk: {list(rep['solve']['vessels_shrunk'].items())[:6]}", flush=True)
else:
    TISSUE_REACH.update({"pairs": 0, "beyond_0.75_reach": 0})
    pairs, tis = violations(P, R, full=True)
after = summary(pairs, tis, R)
after["tissue_check"] = TISSUE_CHECK
if TISSUE_CHECK == "exact":
    after["tissue_reach_check"] = dict(TISSUE_REACH)
if opt("--dump-pairs"):  # diagnostics: the final audit's violating pairs (sample indices, deficit, frame)
    np.savez(opt("--dump-pairs"), i=pairs[0], j=pairs[1], deficit=pairs[2], frame=pairs[3], ti=tis[0].astype(np.int64), tt=np.asarray(tis[1], str), td=tis[2].astype(float))
rep["after"] = after
rep["frames"] = {"cardiac_phases": NC, "r_max": R_MAX, "static": STATIC}
(D / "clearance.json").write_text(json.dumps(rep, indent=1))
print(f"audit (every sample, {'reference pose only' if STATIC else 'every frame'}): {after['pairs']} pair / {after['tissue']} tissue violations, worst {after['pair_max_mm']:.2f} / {after['tissue_max_mm']:.2f} mm ({time.time() - t0:.0f}s)")
if TISSUE_CHECK == "exact":
    print(f"   tissue check exact: {TISSUE_REACH['pairs']} violating (sample, boundary voxel) pairs, {TISSUE_REACH['beyond_0.75_reach']} beyond 0.75 of the reach")
print(f"   pairs: {after['pairs_same_component']} within one component, {after['pairs_other_component']} between components, {after['edge_pairs']} edge pairs; tissue: {after['tissue_by_kind']}")
print(f"   within one component: {after.get('same_component_kinds')}")
for k, v in list(after["worst"].items())[:15]:
    print(f"   {v:6.2f} mm  {k}")
# ── --carve (user 2026-11-25: where a great vessel lies in another chamber, the chamber yields): each remaining
#    vessel / heart-tissue violation removes that tissue (another chamber's blood, or the myocardium for a vessel that
#    is not the heart's own circulation) within (lumen + wall + 0.5 mm) / σ_min of the vessel sample, so the clearance
#    holds under the motion too, relabelled mediastinal fat, in all three label grids: <phantom>/atlas/tissue_*.hv
#    (motion, LBM, truth and simulator pick them up). The cardiac motion must then be refitted to the carved labels.
if "--carve" in A and len(tis[0]) and TISSUE_CHECK == "exact":
    # exact carve: the violating boundary voxels are relabelled fat, then their tissue neighbours are tested in turn
    # (one 0.5 mm layer per pass) until none violates — every test with the exact motion of voxel and vessel samples
    src_dir = D / "atlas" if (D / "atlas" / "tissue_1mm.hv").exists() else REPO / "data/hasty/phantom/p0"
    Rm_ = rmax(R)
    own_, card_ = zones(P, R)
    hin = np.flatnonzero(np.all((P > hbox_lo - 40e-3) & (P < hbox_hi + 40e-3), 1))
    XcV, BrV = cardiac_and_resp(P[hin])
    vt = {}
    for tn in TISSUE:
        m_ = ~tissue_exempt(tn, hin, own_, card_)
        if m_.any():
            vt[tn] = (cKDTree(P[hin[m_]]), np.flatnonzero(m_), tissue_need(tn, hin[m_], Rm_))
    Lf = LAB.copy()
    fat = tnames.index("fat")
    tis_lab = {tnames.index(tn): tn for tn in TISSUE}

    def _violating(vox):
        """vox (n, 3) z, y, x fine indices of tissue voxels → bool per voxel: inside some sample's clearance at some frame"""
        res = np.zeros(len(vox), bool)
        if not len(vox):
            return res
        lab_ = Lf[vox[:, 0], vox[:, 1], vox[:, 2]]
        cen = vox[:, ::-1] * H0 + lo_h + H0 / 2
        XcC, BrC = cardiac_and_resp(cen)
        for li, tn in tis_lab.items():
            sel = np.flatnonzero(lab_ == li)
            if not len(sel) or tn not in vt:
                continue
            _, rows_, need_ = vt[tn]
            # each sample searches the voxels within its own reach (2 need + 2 mm)
            nb = cKDTree(cen[sel]).query_ball_point(P[hin[rows_]], TREACH * (need_ + 1e-3))
            cnt = np.array([len(x) for x in nb])
            if not cnt.sum():
                continue
            b_ = np.repeat(np.arange(len(rows_)), cnt)
            a_ = sel[np.concatenate([np.asarray(x, np.int64) for x in nb if len(x)])]
            dmin, _ = min_dist_cross(XcC, BrC, a_, XcV, BrV, rows_[b_])
            bad = dmin < need_[b_]
            res[a_[bad]] = True
        return res

    # first layer: the boundary voxels near the violating samples
    vio_s = np.unique(tis[0].astype(int))
    front = []
    for tn in TISSUE:
        nb = BTREE[tn].query_ball_point(P[vio_s], TREACH * (tissue_need(tn, vio_s, Rm_) + 1e-3))
        ids = np.unique(np.concatenate([np.asarray(x, np.int64) for x in nb] + [np.zeros(0, np.int64)]))
        if len(ids):
            front.append(BIDX[tn][ids])
    front = np.concatenate(front) if front else np.zeros((0, 3), np.int64)
    front = front[_violating(front)]
    carved_n, passes = 0, 0
    nbh = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    while len(front) and passes < 400:
        Lf[front[:, 0], front[:, 1], front[:, 2]] = fat
        carved_n += len(front)
        passes += 1
        cand = (front[:, None, :] + nbh[None]).reshape(-1, 3)
        cand = cand[np.all((cand >= 0) & (cand < np.array(Lf.shape)), 1)]
        cand = np.unique(cand, axis=0)
        cand = cand[np.isin(Lf[cand[:, 0], cand[:, 1], cand[:, 2]], list(tis_lab))]
        front = cand[_violating(cand)]
    CARVED = (Lf != LAB) & (Lf == fat)
    print(f"exact carve: {carved_n} voxels of 0.5 mm in {passes} layers", flush=True)
    (D / "atlas").mkdir(exist_ok=True)
    carve_rep = {}
    for grid in ("tissue_0.5mm_heart.hv", "tissue_1mm.hv", "tissue_2mm.hv"):
        T_ = vf.read(src_dir / grid)
        names_ = list(T_["tissues"])
        assert names_ == tnames, f"{grid}: tissue list differs from the heart labels'"
        L_ = np.asarray(T_["labels"]).copy()
        o_, h_ = np.asarray(T_["origin_m"], float), float(T_["voxel_m"])
        before_ = {tn: int((L_ == names_.index(tn)).sum()) for tn in TISSUE}
        if grid == "tissue_0.5mm_heart.hv":
            assert L_.shape == Lf.shape and np.allclose(o_, lo_h) and abs(h_ - H0) < 1e-12, "the carve needs the heart labels it audited"
            L_[CARVED] = fat
        else:
            # nearest-neighbour: a coarse voxel whose centre lies in a carved fine voxel of its own tissue
            # only the coarse voxels over the fine grid's box
            c0 = np.maximum(np.floor((lo_h - o_) / h_).astype(int), 0)
            c1 = np.minimum(np.ceil((lo_h + np.array(Lf.shape[::-1]) * H0 - o_) / h_).astype(int), np.array(L_.shape[::-1]))
            zz, yy, xx = np.meshgrid(*(np.arange(c0[a], c1[a]) for a in (2, 1, 0)), indexing="ij")
            cen = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1) * h_ + o_ + h_ / 2
            fi = np.floor((cen - lo_h) / H0).astype(np.int64)[:, ::-1]
            inn = np.all((fi >= 0) & (fi < np.array(Lf.shape)), 1)
            zc, yc, xc = zz.ravel()[inn], yy.ravel()[inn], xx.ravel()[inn]
            fz, fy, fx = fi[inn, 0], fi[inn, 1], fi[inn, 2]
            hitc = CARVED[fz, fy, fx] & (L_[zc, yc, xc] == LAB[fz, fy, fx])
            L_[zc[hitc], yc[hitc], xc[hitc]] = fat
        carve_rep[grid] = {tn: round((before_[tn] - int((L_ == names_.index(tn)).sum())) * h_**3 * 1e6, 2) for tn in TISSUE}
        out_ = dict(T_)
        out_["labels"] = L_
        out_["note"] = str(T_.get("note", "")) + " | clearance --carve (exact): tissue around great vessels relabelled fat (research/flow_phantom.md Tier B5)"
        vf.write(D / "atlas" / grid, out_)
    rep["carve_ml"] = carve_rep
    (D / "clearance.json").write_text(json.dumps(rep, indent=1))
    print(f"carved (mL removed per tissue and grid, written to {D / 'atlas'}): {carve_rep}")
elif "--carve" in A and len(tis[0]):
    src_dir = D / "atlas" if (D / "atlas" / "tissue_1mm.hv").exists() else REPO / "data/hasty/phantom/p0"
    ti = tis[0].astype(int)
    tn_v = tis[1]
    Rm_ = rmax(R)
    Nd_ = np.stack([sd_grad(tn, P[ti]) for tn in TISSUE], 1)
    sdir_ = stretch_dir(P[ti], Nd_)
    sm_ = np.maximum(sdir_[np.arange(len(ti)), [TISSUE.index(t) for t in tn_v]], 0.3)
    rc = (Rm_[ti] + W[ti] + 0.5e-3) / sm_
    (D / "atlas").mkdir(exist_ok=True)
    fat = tnames.index("fat")
    carve_rep = {}
    for grid in ("tissue_0.5mm_heart.hv", "tissue_1mm.hv", "tissue_2mm.hv"):
        T_ = vf.read(src_dir / grid)
        names_ = list(T_["tissues"])
        assert names_ == tnames, f"{grid}: tissue list differs from the heart labels'"
        L_ = np.asarray(T_["labels"]).copy()
        o_, h_ = np.asarray(T_["origin_m"], float), float(T_["voxel_m"])
        before_ = {tn: int((L_ == names_.index(tn)).sum()) for tn in TISSUE}
        for c_, r_, tn in zip(P[ti], rc, tn_v):
            lo_i = np.maximum(np.floor((c_ - r_ - o_) / h_).astype(int), 0)
            hi_i = np.minimum(np.ceil((c_ + r_ - o_) / h_).astype(int) + 1, np.array(L_.shape[::-1]))
            if (hi_i <= lo_i).any():
                continue
            zz, yy, xx = np.meshgrid(*(np.arange(lo_i[a], hi_i[a]) for a in (2, 1, 0)), indexing="ij")
            Xv = np.stack([xx, yy, zz], -1) * h_ + o_ + h_ / 2
            ball = np.linalg.norm(Xv - c_, axis=-1) <= r_
            blk = L_[lo_i[2] : hi_i[2], lo_i[1] : hi_i[1], lo_i[0] : hi_i[0]]
            blk[ball & (blk == names_.index(tn))] = fat
        carve_rep[grid] = {tn: round((before_[tn] - int((L_ == names_.index(tn)).sum())) * h_**3 * 1e6, 2) for tn in TISSUE}
        out_ = dict(T_)
        out_["labels"] = L_
        out_["note"] = str(T_.get("note", "")) + " | clearance --carve: tissue around great vessels relabelled fat (research/flow_phantom.md Tier B5)"
        vf.write(D / "atlas" / grid, out_)
    rep["carve_ml"] = carve_rep
    (D / "clearance.json").write_text(json.dumps(rep, indent=1))
    print(f"carved (mL removed per tissue and grid, written to {D / 'atlas'}): {carve_rep}")
reach_ok = TISSUE_CHECK != "exact" or TISSUE_REACH["beyond_0.75_reach"] == 0
if not reach_ok:
    print("   tissue check: violations near the edge of the candidate reach — raise --tissue-reach (the gate fails)")
sys.exit(0 if after["pairs"] == 0 and after["tissue"] == 0 and reach_ok else 1)
