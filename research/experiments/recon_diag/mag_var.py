"""Magnitude variation across velocity encodes (per frame): sim truth vs reconstruction error.

    mag_var.py <scan dir> [recon npy (nx, ny, nz, ne, F) ...]

Truth: truth_pc_frames.npz (bias_truth.py --signal-frames): the ideal per-frame blood signal C_{f,e} of every LBM-
covered voxel (beat library over the scan's heartbeats and each frame's phase window, six moment-matched sub-sources per
cell: intravoxel velocity spread, spin history, temporal averaging inside the frame). Per voxel-frame:
  s = std_e |C_e| / mean_e |C_e|   (relative magnitude spread across the encodes),
  d = max_e | |C_e| / mean_e |C_e| - 1 |.
Groups: interior (blood in every frame, coverage >= 0.99, depth >= 2 voxels) by depth, frame, speed; partially covered
voxels (wall) with blood only, and with a static tissue term added (partial-volume interference):
  s_e = C_e + r_tb * B_f * (1 - cov_f),  B_f = median |C| of fully covered voxels in frame f, r_tb = tissue / blood
  brightness ratio (warm image: 2-4 voxels outside the lumen vs >= 3 inside), tissue phase 0 (static water).
Reconstructions: the same spread in static tissue (>= 5 voxels from the lumen, magnitude above the median: the truth's
spread there is ~0, so this is the reconstruction error) and in the interior lumen.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

scan = sys.argv[1]
recons = sys.argv[2:]
z = np.load(os.path.join(scan, "truth_pc_frames.npz"))
idx, C, cov = z["idx"], z["c"], z["cov"]  # (n,), (n, E, F), (n, F)
dims = tuple(int(v) for v in z["dims"])
E, F = C.shape[1], C.shape[2]
meta = json.load(open(os.path.join(scan, "meta.json")))
K = z["k"]
lumf = np.load(os.path.join(scan, "truth_lumen_fraction.npy"))
lum = lumf > 0.5
depth = ndimage.distance_transform_edt(lum)
dep = depth.ravel(order="F")[idx]
lf = lumf.ravel(order="F")[idx]
up = np.nan_to_num(np.load(os.path.join(scan, "truth_bias.npz"))["u_pc"])  # (X, Y, Z, 3, F)
spd = np.stack([np.linalg.norm(up[..., f], axis=-1).ravel(order="F")[idx] for f in range(F)], -1)  # (n, F)

A = np.abs(C)
mA = A.mean(1)
with np.errstate(invalid="ignore", divide="ignore"):
    S = A.std(1) / mA  # (n, F)
    Dm = np.abs(A / mA[:, None, :] - 1).max(1)
print(f"{scan}: {idx.size} covered voxels, {E} encodes, {F} frames; |k_e|^2 = {np.round((K ** 2).sum(1), 2).tolist()} (rad per m/s)^2")


def q(v):
    v = v[np.isfinite(v)]
    return f"median {np.median(v) * 100:.2f} %, p90 {np.percentile(v, 90) * 100:.2f} %, p99 {np.percentile(v, 99) * 100:.2f} %" if v.size else "-"


full = (cov >= 0.99).all(1)
interior = full & (dep >= 2)
print(f"interior voxels (blood in every frame, depth >= 2): {interior.sum()}")
print(f"  relative magnitude spread std_e|C_e|/mean: {q(S[interior].ravel())}; max deviation: {q(Dm[interior].ravel())}")
for lo, hi in [(2, 3), (3, 5), (5, 99)]:
    m = interior & (dep >= lo) & (dep < hi)
    print(f"  depth {lo}-{hi} ({m.sum()}): spread {q(S[m].ravel())}")
print("  per frame (median / p90 spread, interior):", " ".join(f"{f}:{np.nanmedian(S[interior, f]) * 100:.1f}/{np.nanpercentile(S[interior, f], 90) * 100:.1f}" for f in range(F)))
for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 9)]:
    m = (spd[interior] >= lo) & (spd[interior] < hi)
    print(f"  speed {lo}-{hi} m/s ({m.sum()} voxel-frames): spread {q(S[interior][m])}")
# systematic encode pattern: mean of |C_e| / mean over voxel-frames (interior)
ratio = (A / mA[:, None, :])[interior]  # (ni, E, F)
pat = np.nanmean(ratio, axis=(0, 2))
print("  mean |C_e| / mean_e |C| per encode:", np.round(pat, 4).tolist())
# fully covered voxels in partially covered frames + wall voxels: blood-only spread
wall = (~full) & (cov.max(1) > 0.2)
print(f"wall / partial voxels: {wall.sum()}")
print(f"  blood only (no tissue): spread {q(S[wall][cov[wall] > 0.2])}")
# partial-volume interference with static tissue
r_tb = 1.0
if len(recons) or os.path.exists(os.path.join(os.path.dirname(scan), "..", "pyr_diag", "warm100", "warm.npy")):
    wpath = os.path.join(scan, "..", "..", "pyr_diag", "warm100", "warm.npy")
    if os.path.exists(wpath):
        w0 = np.abs(np.load(wpath, mmap_mode="r")[..., 0])
        ring = (ndimage.distance_transform_edt(~lum) >= 2) & (ndimage.distance_transform_edt(~lum) <= 4)
        r_tb = float(np.median(w0[ring]) / np.median(w0[depth >= 3]))
print(f"  tissue / blood brightness ratio (warm image) r_tb = {r_tb:.3f}")
B = np.array([np.median(mA[full, f]) for f in range(F)])
Ct = C + (r_tb * B[None, None, :] * (1 - cov[:, None, :])).astype(C.dtype)
At = np.abs(Ct)
with np.errstate(invalid="ignore", divide="ignore"):
    St = At.std(1) / At.mean(1)
for lo, hi in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99)]:
    m = wall[:, None] & (cov >= lo) & (cov < hi)
    print(f"  blood fraction {lo}-{hi} ({m.sum()} voxel-frames): spread blood + static tissue {q(St[m])}; blood only {q(S[m])}")

for rn in recons:
    x = np.load(rn, mmap_mode="r")
    print(f"== reconstruction {rn} {x.shape}")
    far = ndimage.distance_transform_edt(~lum) >= 5
    mag0 = np.abs(np.asarray(x[..., 0, 0]))
    tis = far & (mag0 > np.median(mag0[mag0 > 0]))
    ti = np.nonzero(tis.ravel(order="F"))[0]
    rng = np.random.default_rng(0)
    ti = rng.choice(ti, min(ti.size, 200000), replace=False)
    li = idx[interior]
    for name, sel in [("static tissue", ti), ("interior lumen", li)]:
        vals = []
        for f in range(x.shape[4]):
            a = np.stack([np.abs(np.asarray(x[..., e, f]).ravel(order="F")[sel]) for e in range(E)], 0)
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append(a.std(0) / a.mean(0))
        print(f"  {name}: spread across encodes {q(np.concatenate(vals))}")
