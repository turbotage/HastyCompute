"""Where is the object relative to the FOV? truth PD (diagnostic only) on the native grid: occupancy of the FOV faces."""
import os, numpy as np
from common import SCAN
w = np.load(os.path.join(SCAN, "truth_pd_water.npy")); f = np.load(os.path.join(SCAN, "truth_pd_fat.npy"))
print(w.shape, w.dtype, w.flags['F_CONTIGUOUS'])
p = np.nan_to_num(w) + np.nan_to_num(f)
if p.shape[0] != 254: p = p.T
thr = 0.05 * np.percentile(p[p > 0], 50)
o = p > thr
print("occupied fraction of each face (x-,x+,y-,y+,z-,z+):", [round(float(v), 3) for v in (o[0].mean(), o[-1].mean(), o[:, 0].mean(), o[:, -1].mean(), o[:, :, 0].mean(), o[:, :, -1].mean())])
for a, n in enumerate("xyz"):
    prof = o.mean(axis=tuple(b for b in range(3) if b != a))
    idx = np.nonzero(prof > 0.001)[0]
    print(f"{n}: object spans native index {idx.min()}..{idx.max()} of {o.shape[a]}")
# lungs: low PD inside the body at recon slice z=20 (native z ~ 40)
zn = int(round((20 - 30) * 3.8 / 2.0 + 57))
sl = p[:, :, zn]
print(f"native slice {zn}: PD median in body {np.median(sl[sl > thr]):.3g}; voxels below thr inside bbox {(sl < thr).mean():.3f}")
