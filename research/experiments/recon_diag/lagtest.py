# lag / attenuation of recon ROI dynamics vs truth: circular cross-correlation over cardiac frames, and a temporally
# blurred truth (kernel width w frames) as the model: which (lag, blur) best explains the recon?
import sys, numpy as np
from scipy import ndimage
scan, idxp = sys.argv[1], sys.argv[2]
z = np.load(scan + "/truth_pc_frames.npz"); idx, C, cov = z["idx"], z["c"], z["cov"]
NC = C.shape[2]
dims = np.load(scan + "/truth_lumen_fraction.npy", mmap_mode="r").shape
idx_rec = np.load(idxp).astype(np.int64)
pos = {q: i for i, q in enumerate(idx_rec)}
vol = np.zeros(int(np.prod(dims)), bool); vol[idx[cov.mean(1) >= 0.5]] = True
depth = ndimage.distance_transform_edt(vol.reshape(dims, order="F")).ravel(order="F")[idx]
rows = np.nonzero((cov.min(1) >= 0.99) & (depth >= 2) & np.isin(idx, idx_rec))[0]
rr = np.array([pos[q] for q in idx[rows]])
pred = np.angle(C[rows, 1:] * np.conj(C[rows, :1]))
ijk = np.stack(np.unravel_index(idx[rows], dims, order="F"), 1) // 4
_, bid, cnt = np.unique(ijk, axis=0, return_inverse=True, return_counts=True); bid = bid.ravel()
keep = cnt[bid] >= 16; bid = np.unique(bid[keep], return_inverse=True)[1]; nb = bid.max() + 1
def roi(a):  # (n, E-1, F) -> (nb, E-1, F)
    a = a[keep]; w = np.bincount(bid, None, nb)
    return np.stack([np.stack([np.bincount(bid, a[:, j, f], nb) for f in range(a.shape[2])], -1) for j in range(a.shape[1])], 1) / w[:, None, None]
def dm(a): return a - a.mean(-1, keepdims=True)
P = dm(roi(pred))
for nm in sys.argv[3:]:
    x = np.load(nm, mmap_mode="r")
    F = x.shape[2]
    for b in range(F // NC):
        xs = np.asarray(x[rr][:, :, b * NC:(b + 1) * NC])
        R = dm(roi(np.angle(xs[:, 1:] * np.conj(xs[:, :1]))))
        out = []
        for lag in range(-3, 4):
            Pl = np.roll(P, lag, -1)
            c = (Pl * R).sum() / np.sqrt((Pl ** 2).sum() * (R ** 2).sum())
            out.append(f"{lag:+d}:{c:.3f}")
        best = []
        for w in [0, 0.5, 1, 1.5, 2, 3]:
            Pb = ndimage.gaussian_filter1d(P, w, axis=-1, mode="wrap") if w else P
            s = (Pb * R).sum() / (Pb ** 2).sum()
            res = R - s * Pb
            best.append(f"w{w}: gain {s:.2f} resid {np.sqrt((res**2).mean()):.3f}")
        print(f"{nm.split('/')[-1]} bin {b}: corr by lag (truth shifted) {' '.join(out)}")
        print(f"   blur model: {' | '.join(best)}  (recon rms {np.sqrt((R**2).mean()):.3f}, truth rms {np.sqrt((P**2).mean()):.3f})")
