"""One LLR step applied offline to a reconstruction stack, scored on the cardiac dynamics: which singular-value rule
keeps the dynamics while removing the per-frame aliasing?

    llr_offline.py <scan dir> <stack .hv> <lumen_idx.npy> [--split] [--tpyr]

Stack: list of T images, set t = frame * E + encode (the solver's order), or with --tpyr a temporal-pyramid level
(state_tpyr_done_*.hv, group order (e · R + r) · C + c = e · F + f for one group per set).

Unshifted 8³ blocks that contain a truth lumen voxel; Casorati B³ × T per block (T = all sets, or per respiratory bin:
T / NR columns), SVD once; rules: soft / hard threshold at the block's (R+1)-th singular value. Scored like dyn_fit.py
(truth at the scan-mean breathing state for every bin): per-voxel dynamics and 8 mm ROI dynamics, depth >= 5 and
>= 2, and the energy the rule removes in the lumen voxels.
"""
import os
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python", "lib"))
from hastycompute import value_format as vf  # noqa: E402

scan, stack_path, idx_path = sys.argv[1:4]
E = 7
z = np.load(os.path.join(scan, "truth_pc_frames.npz"))
tidx, C, cov = z["idx"], z["c"], z["cov"]
NC = C.shape[2]
dims = tuple(np.load(os.path.join(scan, "truth_lumen_fraction.npy"), mmap_mode="r").shape)
nv = int(np.prod(dims))
idx_rec = np.load(idx_path).astype(np.int64)

st = vf.open_lazy(stack_path)
T = len(st)
NF = T // E
NR = NF // NC
print(f"{stack_path}: {T} sets = {NF} frames ({NR} respiratory bins) x {E} encodes")

# truth rows (covered in every frame), depth, ROI blocks
vol = np.zeros(nv, bool)
vol[tidx[cov.mean(1) >= 0.5]] = True
depth_v = ndimage.distance_transform_edt(vol.reshape(dims, order="F")).ravel(order="F")
keep_t = (cov.min(1) >= 0.99) & np.isin(tidx, idx_rec)
q_eval = tidx[keep_t]
pred = np.angle(C[keep_t, 1:, :] * np.conj(C[keep_t, :1, :]))
depth = depth_v[q_eval]
ijk4 = np.stack(np.unravel_index(q_eval, dims, order="F"), 1) // 4

# the voxels to denoise: the evaluated voxels' 8³ blocks
b8 = np.stack(np.unravel_index(q_eval, dims, order="F"), 1) // 8
ub = np.unique(b8, axis=0)
print(f"{q_eval.size} evaluated voxels in {len(ub)} blocks of 8³")
vox_blocks = []
for bx, by, bz in ub:
    xs, ys, zs = (np.arange(b * 8, min(b * 8 + 8, n)) for b, n in zip((bx, by, bz), dims))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    vox_blocks.append((X + dims[0] * (Y + dims[1] * Z)).ravel())
allq = np.unique(np.concatenate(vox_blocks))
pos = np.searchsorted(allq, q_eval)
print(f"reading {allq.size} voxels x {T} sets")
A = np.empty((allq.size, T), np.complex64)
for t in range(T):
    f, e = divmod(t, E)
    A[:, t] = st[e * NF + f if "--tpyr" in sys.argv else t].read()[allq]

# per block and column group: SVD
groups = [np.arange(T)] if "--split" not in sys.argv else [np.arange(r * NC * E, (r + 1) * NC * E) for r in range(NR)]
dec = []
for vb in vox_blocks:
    rows = np.searchsorted(allq, vb)
    for g in groups:
        M = A[np.ix_(rows, g)].astype(np.complex128)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        dec.append((rows, g, U, s, Vh))


def apply(rule, R):
    out = A.copy()
    for rows, g, U, s, Vh in dec:
        if rule == "none":
            continue
        thr = s[R] if R < len(s) else 0.0
        w = (s > thr).astype(float) if rule == "hard" else np.maximum(s - thr, 0.0)
        if rule == "hard":
            w = w * s
        out[np.ix_(rows, g)] = ((U * w) @ Vh).astype(np.complex64)
    return out


def demean(ph):
    mu = np.angle(np.exp(1j * ph).mean(-1, keepdims=True))
    return np.angle(np.exp(1j * (ph - mu)))


def stats(a, b):
    ok = np.abs(a) < np.pi / 2
    s = (a[ok] * b[ok]).sum() / (a[ok] ** 2).sum()
    c = np.corrcoef(a[ok], b[ok])[0, 1]
    return s, c, np.sqrt(np.mean(np.angle(np.exp(1j * (b - a))) ** 2))


def roi_avg(a, sel):
    _, bid, cnt = np.unique(ijk4[sel], axis=0, return_inverse=True, return_counts=True)
    bid = bid.ravel()
    keep = cnt[bid] >= 16
    _, bid = np.unique(bid[keep], return_inverse=True)
    nbk = bid.max() + 1
    w = np.bincount(bid, None, nbk)
    a = a[sel][keep]
    return np.stack([np.stack([np.bincount(bid, a[:, j, f], nbk) for f in range(a.shape[2])], -1) for j in range(a.shape[1])], 1) / w[:, None, None]


def score(out):
    x = out[pos]  # (n, T)
    res = {}
    for lo in (5, 2):
        m = depth >= lo
        Pa, Ra, Pr, Rr = [], [], [], []
        for b in range(NR):
            xs = x[:, b * NC * E:(b + 1) * NC * E].reshape(-1, NC, E).transpose(0, 2, 1)  # (n, E, NC)
            rec = np.angle(xs[:, 1:, :] * np.conj(xs[:, :1, :]))
            Pa.append(demean(pred[m]).ravel())
            Ra.append(demean(rec[m]).ravel())
            Pr.append(demean(roi_avg(pred, m)).ravel())
            Rr.append(demean(roi_avg(rec, m)).ravel())
        res[lo] = stats(np.concatenate(Pa), np.concatenate(Ra)) + stats(np.concatenate(Pr), np.concatenate(Rr))
    return res


base = np.linalg.norm(A[pos])
variants = [("none", 0)] + [(r, R) for r in ("soft", "hard") for R in ((4, 8, 16, 32) if "--split" in sys.argv else (8, 16, 32, 64, 128))]
print(f"columns per Casorati: {len(groups[0])} ({'per respiratory bin' if len(groups) > 1 else 'all sets'})")
print("rule  R    | depth>=5: dyn slope corr err | ROI slope corr err || depth>=2: dyn slope corr err | ROI corr err | removed")
for rule, R in variants:
    out = apply(rule, R)
    sc = score(out)
    rem = np.linalg.norm(out[pos] - A[pos]) / base
    a, b = sc[5], sc[2]
    print(f"{rule:5s} {R:4d} | {a[0]:.2f} {a[1]:.2f} {a[2]:.3f} | {a[3]:.2f} {a[4]:.2f} {a[5]:.3f} || {b[0]:.2f} {b[1]:.2f} {b[2]:.3f} | {b[4]:.2f} {b[5]:.3f} | {rem:.3f}")
