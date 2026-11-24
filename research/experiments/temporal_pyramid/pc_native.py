"""Unpack the sim's per-frame PC signal truth for recon_p2np --score-pc (hasty_recon::pcscore, temporal_pyramid.md §13).

    pc_native.py <scan dir> <out dir> [truth npz name, default truth_pc_frames.npz]

Writes pc_idx.npy (flat Fortran native indices, f8), pc_c.npy ((n, E, F) c8 C-order), pc_covmin.npy (minimum LBM
coverage over the cardiac frames, f4), pc_resp_state.txt (the truth's breathing state). Scoring only: no method reads these.
"""
import os
import sys

import numpy as np

scan, out = sys.argv[1:3]
name = sys.argv[3] if len(sys.argv) > 3 else "truth_pc_frames.npz"
z = np.load(os.path.join(scan, name))
os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, "pc_idx.npy"), z["idx"].astype(np.float64))
np.save(os.path.join(out, "pc_c.npy"), np.ascontiguousarray(z["c"].astype(np.complex64)))
np.save(os.path.join(out, "pc_covmin.npy"), np.nan_to_num(z["cov"]).min(-1).astype(np.float32))
# the truth's breathing state (recon_p2np --score-pc-bins checks it against each bin's true state)
with open(os.path.join(out, "pc_resp_state.txt"), "w") as f:
    f.write(f"{float(z['resp_state']):.6f}\n")
print(f"{name}: {z['idx'].size} voxels, c {z['c'].shape}, dims {z['dims']}, resp_state {float(z['resp_state']):.3f}")
