"""Time-averaged images per encode (warm start, all data pooled) against the phase the truth predicts for them.

    warm_fit.py <warm npy (nx, ny, nz, ne)> <scan dir> [min depth voxels = 3]

The time average of encode e in a lumen voxel is ∝ Σ_c exp(−i γ M1_e · v_c) (cardiac frames c, equal weights, unit
magnitude: the sign convention found by enc_fit.py on the synthetic run). Predicted Δφ_e = arg(P_e P̄_0) against the
reconstructed arg(x_e x̄_0), voxels ≥ depth inside the lumen (lumen fraction > 0.5) where every |Δφ_pred| < π/2.
Also the linear prediction −K_e · v̄ (time-mean velocity), to show the circular-mean effect.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

wname, scan = sys.argv[1:3]
dmin = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
GAMMA = 2.6752218744e8
x = np.load(wname, mmap_mode="r")
vt = np.load(os.path.join(scan, "truth_vel_logical.npy"), mmap_mode="r")
lum = np.load(os.path.join(scan, "truth_lumen_fraction.npy")) > 0.5
meta = json.load(open(os.path.join(scan, "meta.json")))
M1 = np.array([e["m1"] for e in meta["sequence"]["encoding"]["at_te"]]) @ np.array(meta["rot_presc"])
ne, nc = x.shape[3], vt.shape[-1]
depth = ndimage.distance_transform_edt(lum)
for lo, hi in [(1, 2), (2, 3), (3, 5), (5, 99)]:
    idx = np.nonzero((depth >= lo) & (depth < hi))
    V = np.stack([np.stack([np.asarray(vt[..., a, c])[idx] for a in range(3)], -1) for c in range(nc)])  # (nc, n, 3)
    P = np.exp(-1j * GAMMA * np.einsum("cna,ea->nec", V, M1)).mean(-1)  # (n, ne)
    pred = np.angle(P[:, 1:] * np.conj(P[:, :1]))
    lin = -GAMMA * V.mean(0) @ (M1[1:] - M1[0]).T
    z = np.stack([np.asarray(x[..., e])[idx] for e in range(ne)], -1)
    rec = np.angle(z[:, 1:] * np.conj(z[:, :1]))
    ok = (np.abs(pred) < np.pi / 2).all(-1)
    a, b, l = pred[ok], rec[ok], lin[ok]
    s = (a * b).sum() / (a * a).sum()
    print(f"depth {lo}-{hi} ({ok.sum()} voxels): recon vs circular-mean prediction slope {s:.3f} corr {np.corrcoef(a.ravel(), b.ravel())[0, 1]:.3f}"
          f" | per encode slopes {[round(float((a[:, e] * b[:, e]).sum() / (a[:, e] ** 2).sum()), 3) for e in range(ne - 1)]}"
          f" | linear −K v̄ vs circular slope {(a * l).sum() / (a * a).sum():.3f}")
