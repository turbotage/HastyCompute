# how the dynamics error of ROI-averaged phases scales with the ROI size (blocks of B voxels): incoherent errors fall
# as 1/sqrt(n), coherent (streak aliasing / bias) do not
import sys, numpy as np
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)
from scipy import ndimage
scan, idxp = sys.argv[1], sys.argv[2]
z = load_any(scan + "/truth_pc_frames.npz"); idx, C, cov = z["idx"], z["c"], z["cov"]
NC = C.shape[2]
dims = load_any(scan + "/truth_lumen_fraction.npy", mmap_mode="r").shape
idx_rec = load_any(idxp).astype(np.int64)
pos = {q: i for i, q in enumerate(idx_rec)}
vol = np.zeros(int(np.prod(dims)), bool); vol[idx[cov.mean(1) >= 0.5]] = True
depth = ndimage.distance_transform_edt(vol.reshape(dims, order="F")).ravel(order="F")[idx]
rows = np.nonzero((cov.min(1) >= 0.99) & (depth >= 2) & np.isin(idx, idx_rec))[0]
rr = np.array([pos[q] for q in idx[rows]])
pred = np.angle(C[rows, 1:] * np.conj(C[rows, :1]))
xyz = np.stack(np.unravel_index(idx[rows], dims, order="F"), 1)
def dm(a): return a - a.mean(-1, keepdims=True)
def roi(a, B):
    _, bid, cnt = np.unique(xyz // B, axis=0, return_inverse=True, return_counts=True); bid = bid.ravel()
    keep = cnt[bid] >= max(1, B ** 3 // 4); b2 = np.unique(bid[keep], return_inverse=True)[1]; nb = b2.max() + 1
    w = np.bincount(b2, None, nb); a = a[keep]
    return np.stack([np.stack([np.bincount(b2, a[:, j, f], nb) for f in range(a.shape[2])], -1) for j in range(a.shape[1])], 1) / w[:, None, None], int(w.mean())
for nm in sys.argv[3:]:
    x = load_any(nm, mmap_mode="r")
    xs = np.asarray(x[rr][:, :, :NC])
    rec = np.angle(xs[:, 1:] * np.conj(xs[:, :1]))
    line = []
    for B in [1, 2, 4, 8, 12]:
        P, n = roi(pred, B); R, _ = roi(rec, B)
        P, R = dm(P), dm(R)
        e = np.sqrt(((R - P) ** 2).mean()); c = (P * R).sum() / np.sqrt((P ** 2).sum() * (R ** 2).sum())
        s = (P * R).sum() / (P ** 2).sum()
        line.append(f"B{B} (n~{n}, {P.shape[0]} rois): err {e:.3f} / truth {np.sqrt((P**2).mean()):.3f}, corr {c:.2f}, gain {s:.2f}")
    print(nm.split("/")[-1] + " (bin 0)\n   " + "\n   ".join(line))
