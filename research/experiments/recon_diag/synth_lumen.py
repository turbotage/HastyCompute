"""Lumen values of an inverse-crime truth x★ for framerecon --synthetic: per (voxel of lumen_idx, encode, cardiac
frame) the warm image's magnitude with the warm encode-0 phase plus the true per-frame phase difference
arg(C_e C̄_0) (truth_pc_frames.npz); voxels the truth does not cover in a frame keep the warm value.

    synth_lumen.py <scan dir> <warm_full.npy (X, Y, Z, E), Fortran> <lumen_idx.npy> <out.npy (n, E, F) c8>
"""
import sys

import numpy as np

scan, warm_p, idx_p, out_p = sys.argv[1:5]
z = np.load(scan + "/truth_pc_frames.npz")
tidx, C, cov = z["idx"], z["c"], z["cov"]
E, F = C.shape[1], C.shape[2]
w = np.load(warm_p, mmap_mode="r")
nv = int(np.prod(w.shape[:3]))
wf = np.asarray(w).reshape(nv, E, order="F")
idx = np.load(idx_p).astype(np.int64)
out = np.repeat(wf[idx][:, :, None], F, axis=2).astype(np.complex64)
pos = {q: i for i, q in enumerate(idx)}
rows = np.array([i for i, q in enumerate(tidx) if q in pos])
rr = np.array([pos[q] for q in tidx[rows]])
dphi = np.angle(C[rows] * np.conj(C[rows, :1]))  # (m, E, F), e = 0 -> 0
mag = np.abs(wf[idx[rr]])[:, :, None]
ph0 = np.angle(wf[idx[rr], 0])[:, None, None]
val = (mag * np.exp(1j * (ph0 + dphi))).astype(np.complex64)
covered = (cov[rows] >= 0.5)[:, None, :]
out[rr] = np.where(covered, val, out[rr])
np.save(out_p, out)
print(f"{out_p}: {out.shape}, {rows.size} truth voxels of {idx.size}")
