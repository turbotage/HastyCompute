"""Cardiac dynamics of a reconstruction's velocity phases against the ideal per-frame phase-contrast signal, with one
truth per respiratory bin (the anatomy moves ~10 mm with breathing; a truth at the scan-mean breathing state is
misregistered with every bin's images).

    dyn_fit.py <scan dir> --truth T0.npz[,T1.npz,..] [--idx lumen_idx.npy] [--bins-sep] recon ...

recon: `*_lumen.npy` (n, E, F) at the voxels of --idx (default: truth_pc_frames.npz's idx), or a full (nx, ny, nz, E, F)
npy. Recon frame f = respiratory bin f // 20, cardiac frame f % 20; bin b is scored against truth T_b (one truth for all
bins if only one is given). Truths: bias_truth.py --signal-frames [--resp R --frames-out NAME].

Per bin, on the voxels the truth covers in every cardiac frame (coverage >= 0.99), depth = distance to the edge of the
bin's own lumen (frame-mean coverage >= 0.5):
- pooled: slope / corr of arg(x_e x̄_0) against arg(C_e C̄_0) (pairs with |pred| < π/2), wrapped error rms;
- dynamics: the same after removing each voxel's circular temporal mean on both sides (only the cardiac variation);
- ROI dynamics: phases averaged over 8 mm blocks (4³ voxels, >= 16 voxels each) first, then de-meaned in time.
"""
import os
import sys

import numpy as np
from scipy import ndimage

A = sys.argv[1:]


def opt(name, default=None):
    if name in A:
        i = A.index(name)
        v = A[i + 1]
        del A[i : i + 2]
        return v
    return default


bins_sep = "--bins-sep" in A
if bins_sep:
    A.remove("--bins-sep")
truths = opt("--truth", "truth_pc_frames.npz").split(",")
idx_path = opt("--idx")
scan, names = A[0], A[1:]
dims = tuple(np.load(os.path.join(scan, "truth_lumen_fraction.npy"), mmap_mode="r").shape)
T = []
for t in truths:
    z = np.load(t if os.path.exists(t) else os.path.join(scan, t))
    T.append((z["idx"], z["c"], z["cov"], float(z["resp_state"])))
idx_rec = np.load(idx_path).astype(np.int64) if idx_path else T[0][0]
NC = T[0][1].shape[2]


def demean(ph):
    mu = np.angle(np.exp(1j * ph).mean(-1, keepdims=True))
    return np.angle(np.exp(1j * (ph - mu)))


def stats(a, b):
    ok = np.abs(a) < np.pi / 2
    s = (a[ok] * b[ok]).sum() / (a[ok] ** 2).sum()
    c = np.corrcoef(a[ok], b[ok])[0, 1]
    w = np.angle(np.exp(1j * (b - a)))
    return s, c, np.sqrt(np.mean(w**2))


def prep(b):
    """bin b's truth on the voxels it covers in every frame that the recon has: (rows into idx_rec, pred, depth, ijk)"""
    idx, C, cov, _ = T[min(b, len(T) - 1)]
    vol = np.zeros(int(np.prod(dims)), bool)
    vol[idx[cov.mean(1) >= 0.5]] = True
    depth_v = ndimage.distance_transform_edt(vol.reshape(dims, order="F")).ravel(order="F")
    allb = cov.min(1) >= 0.99
    pos = {q: i for i, q in enumerate(idx_rec)}
    rows_t = np.nonzero(allb & np.isin(idx, idx_rec))[0]
    rows_r = np.array([pos[q] for q in idx[rows_t]], np.int64)
    q = idx[rows_t]
    pred = np.angle(C[rows_t, 1:, :] * np.conj(C[rows_t, :1, :]))
    ijk = np.stack(np.unravel_index(q, dims, order="F"), 1) // 4
    return rows_r, q, pred, depth_v[q], ijk


def roi(ijk, sel, *arrs):
    _, bid, cnt = np.unique(ijk[sel], axis=0, return_inverse=True, return_counts=True)
    bid = bid.ravel()
    keep = cnt[bid] >= 16
    _, bid = np.unique(bid[keep], return_inverse=True)
    nbk = bid.max() + 1
    w = np.bincount(bid, None, nbk)
    out = []
    for a in arrs:
        a = a[sel][keep]
        out.append(np.stack([np.stack([np.bincount(bid, a[:, j, f], nbk) for f in range(a.shape[2])], -1) for j in range(a.shape[1])], 1) / w[:, None, None])
    return nbk, out


P = [prep(b) for b in range(len(T))]
for nm in names:
    lumen_only = nm.endswith("_lumen.npy")
    x = np.load(nm, mmap_mode="r")
    F = x.shape[2] if lumen_only else x.shape[4]
    nb = F // NC
    acc = {}
    for b in range(nb):
        rows_r, q, pred, depth, ijk = P[min(b, len(P) - 1)]
        fs = range(b * NC, (b + 1) * NC)
        if lumen_only:
            xs = np.asarray(x[rows_r][:, :, b * NC : (b + 1) * NC])
        else:
            xs = np.stack([np.stack([np.asarray(x[..., e, f]).ravel(order="F")[q] for e in range(x.shape[3])], 1) for f in fs], -1)
        rec = np.angle(xs[:, 1:, :] * np.conj(xs[:, :1, :]))
        res = {}
        for lo in (2, 5):
            m = depth >= lo
            res[f"pooled {lo}"] = (pred[m].ravel(), rec[m].ravel(), m.sum())
            res[f"dynamics {lo}"] = (demean(pred[m]).ravel(), demean(rec[m]).ravel(), m.sum())
            nbk, (pa, ra) = roi(ijk, m, pred, rec)
            res[f"ROI dynamics {lo}"] = (demean(pa).ravel(), demean(ra).ravel(), nbk)
        if bins_sep and nb > 1:
            print(f"== {os.path.basename(nm)} bin {b} (truth breathing state {T[min(b, len(T) - 1)][3]:.3f})")
            for k, (a, r, n) in res.items():
                s, c, rms = stats(a, r)
                print(f"  {k:16s} ({n:5d}): slope {s:.3f} corr {c:.3f} err rms {rms:.3f} rad (truth rms {np.sqrt(np.mean(a**2)):.3f})")
        for k, v in res.items():
            acc.setdefault(k, []).append(v)
    print(f"== {os.path.basename(nm)} ({F} frames, {nb} bins, truths at {[round(t[3], 3) for t in T]})")
    for k, vs in acc.items():
        a = np.concatenate([v[0] for v in vs])
        r = np.concatenate([v[1] for v in vs])
        s, c, rms = stats(a, r)
        print(f"  {k:16s} ({sum(v[2] for v in vs):5d}): slope {s:.3f} corr {c:.3f} err rms {rms:.3f} rad (truth rms {np.sqrt(np.mean(a**2)):.3f})")
