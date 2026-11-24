"""Where the velocity error of a dev recon sits (temporal_pyramid.md §13), from recon_p2np --score-pc's final dump.

    pcerr.py <run dir> [tag substring ...]

pcerr_<tag>.npy: (4, nf, n) F = per (voxel, frame) |Δv| of the decoder in use, |Δv| of the hierarchical decoder,
|x_0| of the recon and |C_0| of the truth; pc_vox.npy: the scored voxels. Prints, per run: |Δv| median / p90 and the
fraction > VENC for both decoders, split by the recon's encode-0 magnitude relative to the truth's (quartiles of
|x_0| / |C_0| after one global scale fit), and by depth inside the scored region (voxels to its edge, this grid).
"""
import glob
import os
import sys

import numpy as np
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any  # noqa: E402
from scipy import ndimage  # noqa: E402

run = sys.argv[1]
subs = sys.argv[2:]
VENC = 1.5
vox = load_any(os.path.join(run, "pc_vox.npy")).astype(np.int64)
dims = (128, 72, 60)
m = np.zeros(int(np.prod(dims)), bool)
m[vox] = True
depth = ndimage.distance_transform_edt(m.reshape(dims, order="F")).ravel(order="F")[vox]
for f in sorted(glob.glob(os.path.join(run, "pcerr_*.npy")) + glob.glob(os.path.join(run, "pcerr_*.hv"))):
    tag = os.path.basename(f).split(".")[0][6:]
    if subs and not any(s in tag for s in subs):
        continue
    a = np.asarray(load_any(f)).transpose(2, 1, 0)  # (n, nf, 4)
    dv, dh, x0, c0 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    s = np.median(x0) / max(np.median(c0), 1e-30)
    r = x0 / (s * np.maximum(c0, 1e-30))
    print(f"{tag}: |Δv| median {np.median(dv):.3f} p90 {np.quantile(dv, .9):.3f} >VENC {100 * np.mean(dv > VENC):.1f} % | hier median {np.median(dh):.3f} p90 {np.quantile(dh, .9):.3f} >VENC {100 * np.mean(dh > VENC):.1f} %")
    qs = np.quantile(r, [0, .25, .5, .75, 1])
    for lo, hi in zip(qs[:-1], qs[1:]):
        k = (r >= lo) & (r <= hi)
        print(f"   |x0|/(s|C0|) {lo:5.2f}–{hi:5.2f}: median {np.median(dv[k]):.3f} >VENC {100 * np.mean(dv[k] > VENC):.1f} %")
    for lo, hi in [(1, 2), (2, 3), (3, 99)]:
        k = (depth >= lo) & (depth < hi)
        if k.any():
            print(f"   depth {lo}–{hi} ({k.sum()} voxels): median {np.median(dv[k]):.3f} >VENC {100 * np.mean(dv[k] > VENC):.1f} %, hier {np.median(dh[k]):.3f} / {100 * np.mean(dh[k] > VENC):.1f} %")
