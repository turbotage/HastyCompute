"""Time-averaged images per encode against the ideal frame-mean phase-contrast signal of the sim (bias_truth.py --signal).

    sig_fit.py <image npy (nx, ny, nz, ne)> <scan dir> [image npy ...]

truth_pc_signal.npz: c_mean [X, Y, Z, E] = frame mean of C_e (spin history, intravoxel velocity spread, the sim's sign).
Per depth band of the lumen (lumen fraction > 0.5, LBM-covered): slope / corr of the reconstructed Δφ_e = arg(x_e x̄_0)
against arg(c_e c̄_0) (voxels where every |Δφ_pred| < π/2), and the same for the equal-weight unit-magnitude prediction of
warm_fit.py (mean_c e^{−i K_e v_c}, truth_vel_logical) — the difference is the effect of the frame weighting.
COVMIN=c: only voxels whose LBM coverage (truth_bias.npz, blood volume fraction per frame) is ≥ c in every frame (the
prediction has no tissue term: wall / chamber motion puts tissue into partly covered voxels).
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

scan = sys.argv[2]
names = [sys.argv[1]] + sys.argv[3:]
GAMMA = 2.6752218744e8
sig = np.load(os.path.join(scan, "truth_pc_signal.npz"))
cm = sig["c_mean"]
lum = np.load(os.path.join(scan, "truth_lumen_fraction.npy")) > 0.5
covered = np.abs(cm[..., 0]) > 0
if os.environ.get("COVMIN"):
    cmin = np.nan_to_num(np.load(os.path.join(scan, "truth_bias.npz"))["coverage"]).min(-1)
    covered &= cmin >= float(os.environ["COVMIN"])
vt = np.load(os.path.join(scan, "truth_vel_logical.npy"), mmap_mode="r")
meta = json.load(open(os.path.join(scan, "meta.json")))
M1 = np.array([e["m1"] for e in meta["sequence"]["encoding"]["at_te"]]) @ np.array(meta["rot_presc"])
depth = ndimage.distance_transform_edt(lum)
bands = [(1, 2), (2, 3), (3, 5), (5, 99)]
pre = {}
for lo, hi in bands:
    idx = np.nonzero((depth >= lo) & (depth < hi) & covered)
    c = cm[idx]
    ps = np.angle(c[:, 1:] * np.conj(c[:, :1]))
    V = np.stack([np.stack([np.asarray(vt[..., a, f])[idx] for a in range(3)], -1) for f in range(vt.shape[-1])])
    P = np.exp(-1j * GAMMA * np.einsum("cna,ea->nec", V, M1)).mean(-1)
    pe = np.angle(P[:, 1:] * np.conj(P[:, :1]))
    pre[(lo, hi)] = (idx, ps, pe)
    ok = (np.abs(ps) < np.pi / 2).all(-1) & (np.abs(pe) < np.pi / 2).all(-1)
    print(f"depth {lo}-{hi} ({ok.sum()} voxels): signal-weighted vs equal-weight prediction slope {(ps[ok] * pe[ok]).sum() / (pe[ok] ** 2).sum():.3f}, "
          f"corr {np.corrcoef(ps[ok].ravel(), pe[ok].ravel())[0, 1]:.3f}")


def fit(a, b):
    return (a * b).sum() / (a * a).sum(), np.corrcoef(a.ravel(), b.ravel())[0, 1]


for name in names:
    x = np.load(name, mmap_mode="r")
    ne = x.shape[3]
    print(f"== {name}")
    for (lo, hi), (idx, ps, pe) in pre.items():
        z = np.stack([np.asarray(x[..., e])[idx] for e in range(ne)], -1)
        rec = np.angle(z[:, 1:] * np.conj(z[:, :1]))
        ok = (np.abs(ps) < np.pi / 2).all(-1)
        s1, r1 = fit(ps[ok], rec[ok])
        ok2 = (np.abs(pe) < np.pi / 2).all(-1)
        s2, r2 = fit(pe[ok2], rec[ok2])
        per = [round(float(fit(ps[ok][:, e], rec[ok][:, e])[0]), 3) for e in range(ne - 1)]
        print(f"  depth {lo}-{hi}: vs signal-weighted slope {s1:.3f} corr {r1:.3f} (per encode {per}) | vs equal-weight slope {s2:.3f} corr {r2:.3f}")
