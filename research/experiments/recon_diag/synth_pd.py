"""Inverse-crime truth x★ from the phantom's proton density (framerecon --synthetic): the static image is
PD_water + PD_fat (real) in every encode, and the lumen voxels of lumen_idx add the true per-frame blood signal
|C_{e,f}| exp(i arg(C_{e,f} C̄_{0,f})) (truth_pc_frames.npz, the same units as PD: blood is not in the static PD maps,
lumen PD median 0.06 vs 5.1 in the body, |C| median 4.6; encode 0 real).

Why (research/recon_diag_2026-11-13.md §11): the first inverse crime used the warm image as x★, but that image holds
31 % of its energy outside the body (motion ghosts / haze) and 43 % beyond half k_max (truth PD: 3.9 %), so its per-frame
aliasing swamps the lumen; this x★ has the object's own spectrum.

    synth_pd.py <scan dir> <lumen_idx.npy> <out_static.npy (X, Y, Z, E) Fortran c8> <out_lumen.npy (n, E, F) c8>
"""
import sys

import numpy as np

scan, idx_p, out_w, out_s = sys.argv[1:5]
z = np.load(scan + "/truth_pc_frames.npz")
tidx, C, cov = z["idx"], z["c"], z["cov"]
E, F = C.shape[1], C.shape[2]
pd = np.load(scan + "/truth_pd_water.npy") + np.load(scan + "/truth_pd_fat.npy")
dims = pd.shape
nv = pd.size
pdf = pd.reshape(nv, order="F").astype(np.float32)
w = np.zeros((nv, E), np.complex64, order="F")
w[:] = pdf[:, None]
np.save(out_w, np.asfortranarray(w.reshape(*dims, E, order="F")))
idx = np.load(idx_p).astype(np.int64)
pos = {q: i for i, q in enumerate(tidx)}
out = np.repeat(pdf[idx][:, None, None], E, axis=1).repeat(F, axis=2).astype(np.complex64)
hit = 0
for i, q in enumerate(idx):
    j = pos.get(int(q))
    if j is None:
        continue
    c = C[j]
    out[i] = pdf[q] + np.abs(c) * np.exp(1j * np.angle(c * np.conj(c[0:1])))
    hit += 1
np.save(out_s, np.ascontiguousarray(out))
print(f"x★ static: PD, dims {dims}; lumen: {hit} of {len(idx)} voxels + the true blood signal; median PD in lumen {np.median(pdf[idx]):.3f}, in body {np.median(pdf[pdf > 0.05 * pdf.max()]):.3f}, median |x★| in lumen {np.median(np.abs(out)):.3f}")
