"""Coronary ROIs of a box scan on its reconstruction grid, per cardiac frame, from the deformed vessel tree of the truth
(phantom_4d's frames_geom.hv: centreline points and radii per frame, at the scan's breathing state).

    rustlib/.venv/bin/python -I research/experiments/box_recon/coronary_roi.py <phantom_dir> <scan_dir> <truth_idx.npy> <out_dir>

- arteries: the edges lbm_coronary.py simulates (its NAMES: the domain of the coronary LBM truth);
- veins: edges named "... cardiac vein" or "coronary sinus" (no LBM truth: potential flow).
A voxel belongs to an ROI in frame f when its centre lies within the centreline radius of a point of that frame
(union of balls over the densely sampled centreline); `core` = within radius − 0.5 voxel (no partial volume).
Writes <out_dir>/coronary_roi.npz: idx_art / idx_vein (F-order flat, union over frames, dilated by one voxel),
in_art / in_vein [n, F] (member in frame f), core_art / core_vein [n, F], grp_art / grp_vein [n, F] (branch group of
the covering centreline point: 1 left main, 2 LAD + septal, 3 LCx, 4 RCA + inferolateral, 5 cardiac veins; 0 none),
tan_art / tan_vein [n, F, 3] (its unit centreline tangent, logical axes, distal direction), and <out_dir>/lumen_idx_ext_f32.npy =
truth_idx followed by the ROI voxels not already in it (for recon_p2np --save-lumen / framerecon --idx; frame_fit reads
the first truth_idx.size rows).
Check printed: fraction of aorta / coronary centreline voxels with truth lumen fraction > 0.5 (geometry alignment).
"""
import json
import sys
from pathlib import Path

import numpy as np
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)
from scipy import ndimage

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

PH, SC, TIDX, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
NAMES = ("Left coronary artery", "Right coronary artery", "Anterior interventricular artery", "Circumflex artery of heart",
         "Septal branches of anterior interventricular artery", "Right inferolateral branch of right coronary artery")
meta = json.loads((SC / "meta.json").read_text())
center = np.array(meta["presc_center_subject"])
res = np.array(meta["res"])
dims = np.array(meta["recon_dims"])
assert np.allclose(np.array(meta["rot_presc"]), np.eye(3)), "oblique prescription: map through rot_presc first"
c0 = center - (dims // 2) * res  # centre of voxel 0

V = vf.read(PH / "vessels.hv")
eoff = np.asarray(V["edge_offsets"]).astype(np.int64)
names = json.loads((PH / "graph.json").read_text())["edge_names"]
assert len(names) == eoff.size - 1


def edge_points(pred):
    sel = [i for i, n in enumerate(names) if pred(n)]
    return np.concatenate([np.arange(eoff[i], eoff[i + 1]) for i in sel]) if sel else np.zeros(0, np.int64), len(sel)


GROUP = {"Left coronary artery": 1, "Anterior interventricular artery": 2, "Septal branches of anterior interventricular artery": 2,
         "Circumflex artery of heart": 3, "Right coronary artery": 4, "Right inferolateral branch of right coronary artery": 4}
pgrp = np.zeros(eoff[-1], np.int8)
pedge = np.full(eoff[-1], -1, np.int64)
for i, n in enumerate(names):
    pedge[eoff[i]:eoff[i + 1]] = i
    if n in GROUP:
        pgrp[eoff[i]:eoff[i + 1]] = GROUP[n]
    elif "cardiac vein" in n.lower() or "coronary sinus" in n.lower():
        pgrp[eoff[i]:eoff[i + 1]] = 5


def tangents(pts):
    """unit tangent per point along its edge (central differences inside the edge, one-sided at the ends); the
    distal sign is fixed below per edge (away from the ostium: the end nearer the aortic root comes first)"""
    t = np.zeros_like(pts)
    for i in range(eoff.size - 1):
        a, b = eoff[i], eoff[i + 1]
        if b - a < 2:
            continue
        t[a:b] = np.gradient(pts[a:b], axis=0)
    return t / np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)


p_art, n_art = edge_points(lambda n: n in NAMES)
p_vein, n_vein = edge_points(lambda n: "cardiac vein" in n.lower() or "coronary sinus" in n.lower())
p_ao, _ = edge_points(lambda n: "aorta" in n.lower())
G = vf.read(SC / "truth_fine" / "frames_geom.hv")
P = np.asarray(G["points"])  # [F, N, 3] m
R = np.asarray(G["radius"])  # [F, N] m
F = P.shape[0]
print(f"edges: {n_art} coronary artery, {n_vein} cardiac vein; points {p_art.size} / {p_vein.size}; {F} frames")


def raster(pts, rad, shrink=0.0):
    """int32 [X, Y, Z]: index (into pts) of a point whose ball (radius rad − shrink) holds the voxel centre, −1 none"""
    m = np.full(tuple(dims), -1, np.int32)
    q = (pts - c0) / res  # continuous voxel index of each point
    r = (rad - shrink) / res[0]
    ok = r > 0
    pid = np.flatnonzero(ok).astype(np.int32)
    q, r = q[ok], r[ok]
    rmax = int(np.ceil(r.max())) if r.size else 0
    off = np.stack(np.meshgrid(*(np.arange(-rmax, rmax + 1),) * 3, indexing="ij"), -1).reshape(-1, 3)
    base = np.rint(q).astype(np.int64)
    for s in range(0, q.shape[0], 4096):
        b = base[s:s + 4096, None, :] + off[None]  # [n, k, 3]
        d2 = ((b - q[s:s + 4096, None, :]) ** 2).sum(-1)
        hit = (d2 <= r[s:s + 4096, None] ** 2) & np.all((b >= 0) & (b < dims), -1)
        v = b[hit]
        m[v[:, 0], v[:, 1], v[:, 2]] = np.broadcast_to(pid[s:s + 4096, None], hit.shape)[hit]
    return m


lum = load_any(SC / "truth_lumen_fraction.npy")
for nm, pi in (("aorta", p_ao), ("coronary artery", p_art), ("cardiac vein", p_vein)):
    q = np.rint((P[F // 2][pi] - c0) / res).astype(np.int64)
    ok = np.all((q >= 0) & (q < dims), 1)
    if ok.any():
        lf = lum[q[ok, 0], q[ok, 1], q[ok, 2]]
        print(f"{nm}: {ok.mean():.2f} of the centreline points in the box; lumen fraction > 0.5 at {(lf > 0.5).mean():.2f} of them (median {np.median(lf):.2f})")

# distal direction per edge: the edge end farther from the aortic root (mean aorta point of the mid frame) is distal
root = P[F // 2][p_ao].mean(0) if p_ao.size else P[F // 2].mean(0)
out = {}
for nm, pi in (("art", p_art), ("vein", p_vein)):
    hits = [raster(P[f][pi], R[f][pi]) for f in range(F)]
    cores = [raster(P[f][pi], R[f][pi], 0.5 * res[0]) for f in range(F)]
    inf = np.stack([h >= 0 for h in hits], -1)
    core = np.stack([c >= 0 for c in cores], -1)
    union = ndimage.binary_dilation(inf.any(-1))
    loc = np.nonzero(union)
    flat = (loc[0] + dims[0] * (loc[1] + dims[1] * loc[2])).astype(np.int64)
    grp = np.zeros((flat.size, F), np.int8)
    tan = np.zeros((flat.size, F, 3), np.float32)
    for f in range(F):
        tf = tangents(P[f].astype(np.float64))
        for i in range(eoff.size - 1):
            a, b = eoff[i], eoff[i + 1]
            if b - a >= 2 and np.linalg.norm(P[f][a] - root) > np.linalg.norm(P[f][b - 1] - root):
                tf[a:b] *= -1
        h = hits[f][loc]
        ok = h >= 0
        gp = pi[h[ok]]
        grp[ok, f] = pgrp[gp]
        tan[ok, f] = tf[gp]
    out[f"idx_{nm}"] = flat
    out[f"in_{nm}"] = inf[loc]
    out[f"core_{nm}"] = core[loc]
    out[f"grp_{nm}"] = grp
    out[f"tan_{nm}"] = tan
    print(f"{nm}: voxel-frames per group", {g: int((grp == g).sum()) for g in range(1, 6)})
    print(f"{nm}: {flat.size} voxels (union over frames, dilated); per frame {inf.sum((0, 1, 2)).min()}–{inf.sum((0, 1, 2)).max()}, core {core.sum((0, 1, 2)).min()}–{core.sum((0, 1, 2)).max()}")
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / "coronary_roi.npz", **out, dims=dims)
tidx = load_any(TIDX).astype(np.int64)
extra = np.setdiff1d(np.union1d(out["idx_art"], out["idx_vein"]), tidx)
ext = np.concatenate([tidx, extra])
assert ext.max() < 2 ** 24, "f32 index not exact"
np.save(OUT / "lumen_idx_ext_f32.npy", ext.astype(np.float32))
print(f"truth idx {tidx.size} + {extra.size} coronary voxels not in it → {ext.size} ({OUT / 'lumen_idx_ext_f32.npy'})")
