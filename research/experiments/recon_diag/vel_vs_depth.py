"""Velocity shrinkage vs depth inside the lumen (native grid): time-mean recon vs time-mean truth, per erosion depth.

    vel_vs_depth.py <run dir> <vel npy> <scan dir>
"""
import os, sys
import numpy as np
from scipy import ndimage
run, vname, scan = sys.argv[1:4]
v = np.load(os.path.join(run, vname), mmap_mode='r'); vt = np.load(os.path.join(scan, 'truth_vel_logical.npy'), mmap_mode='r')
lf = np.load(os.path.join(scan, 'truth_lumen_fraction.npy')); lum = lf > 0.5
depth = ndimage.distance_transform_edt(lum)  # voxels to the nearest non-lumen voxel
nc = vt.shape[-1]; F = v.shape[-1]
Tm = np.mean([np.asarray(vt[..., c]) for c in range(nc)], 0)
Rm = np.mean([np.asarray(v[..., f]) for f in range(F)], 0)
for lo, hi in [(1, 1.5), (1.5, 2.5), (2.5, 4), (4, 7), (7, 99)]:
    m = (depth >= lo) & (depth < hi)
    t, r = Tm[m].ravel(), Rm[m].ravel()
    print(f"depth {lo:>4}-{hi:<3} voxels ({m.sum():6d}): time-mean slope {(r * t).sum() / (t * t).sum():.2f}, corr {np.corrcoef(r, t)[0, 1]:.2f}, "
          f"truth |v̄| rms {np.sqrt((Tm[m] ** 2).sum(-1).mean()):.2f}, recon {np.sqrt((Rm[m] ** 2).sum(-1).mean()):.2f} m/s")
