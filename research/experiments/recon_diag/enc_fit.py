"""Effective velocity-encoding matrix of a reconstruction, fitted against the simulator truth (deep lumen voxels).

    enc_fit.py <run dir> <recon npy (nx, ny, nz, ne, F)> <scan dir> [min depth voxels = 3]

Per frame f (cardiac frame f % n_card of the truth), voxel q at ≥ depth voxels inside the lumen (lumen fraction > 0.5):
Δφ_e = arg(x_e x̄_0), e = 1..ne−1, regressed on the true velocity v (least squares, voxels where the design's
phase |K_e v| < π/2 for every e so that neither side wraps): Δφ ≈ K_est v. Compared with the design
K_e = γ (M1_e − M1_0) (meta sequence.encoding.at_te, physical = logical for rot_presc = I). Also: the fit residual
for the truth shifted by s cardiac frames (frame-alignment check) and the per-encode slope of Δφ against K_e v.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

run, rname, scan = sys.argv[1:4]
dmin = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
GAMMA = 2.6752218744e8
x = np.load(os.path.join(run, rname), mmap_mode="r")
# TRUTH=u_pc|u_vol|u_sig: a truth of truth_bias.npz (bias_truth.py: the beat library over the scan's beats) instead of
# truth_vel_logical.npy
_tk = os.environ.get("TRUTH")
vt = np.load(os.path.join(scan, "truth_bias.npz"))[_tk] if _tk else np.load(os.path.join(scan, "truth_vel_logical.npy"), mmap_mode="r")
if _tk:
    vt = np.nan_to_num(vt)
lum = np.load(os.path.join(scan, "truth_lumen_fraction.npy")) > 0.5
meta = json.load(open(os.path.join(scan, "meta.json")))
rot = np.array(meta["rot_presc"])
M1 = np.array([e["m1"] for e in meta["sequence"]["encoding"]["at_te"]]) @ rot  # physical → logical (rows: Rᵀ m)
Kd = GAMMA * (M1[1:] - M1[0])
ne, F, nc = x.shape[3], x.shape[4], vt.shape[-1]
depth = ndimage.distance_transform_edt(lum)
idx = np.nonzero(depth >= dmin)
n = idx[0].size
print(f"{run}: {n} voxels at depth ≥ {dmin}, {F} frames, {ne} encodes, truth {nc} cardiac frames")
V = np.stack([np.stack([np.asarray(vt[..., a, c])[idx] for a in range(3)], -1) for c in range(nc)])  # (nc, n, 3)
dphi, mag, card = [], [], []
for f in range(F):
    z = np.stack([np.asarray(x[..., e, f])[idx] for e in range(ne)], -1)
    dphi.append(np.angle(z[:, 1:] * np.conj(z[:, :1])))
    mag.append(np.abs(z[:, 0]))
    card.append(f % nc)
dphi, mag = np.stack(dphi), np.stack(mag)  # (F, n, ne-1), (F, n)


def fit(shift):
    vv = np.stack([V[(c + shift) % nc] for c in card])  # (F, n, 3)
    ok = (np.abs(vv @ Kd.T) < np.pi / 2).all(-1) & (mag > np.median(mag))
    A, B = vv[ok], dphi[ok]
    Kt, *_ = np.linalg.lstsq(A, B, rcond=None)
    res = B - A @ Kt
    return Kt.T, ok.sum(), np.sqrt((res ** 2).mean()), A, B


Ke, nok, rms, A, B = fit(0)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print(f"design K (rad per m/s), rows e = 1..{ne - 1}:\n{Kd}")
print(f"fitted K ({nok} samples, residual rms {rms:.3f} rad):\n{Ke}")
for e in range(ne - 1):
    p = A @ Kd[e]
    s = (p * B[:, e]).sum() / (p * p).sum()
    print(f"  encode {e + 1}: slope of Δφ on K_e v {s:.3f}, corr {np.corrcoef(p, B[:, e])[0, 1]:.3f}")
print("frame shift of the truth: residual rms (rad)", {s: round(fit(s)[2], 4) for s in (-3, -2, -1, 0, 1, 2, 3)})

# per respiratory bin (frame f: cardiac f % n_card, respiratory f // n_card) and per frame: slope of Δφ on K v
vv0 = np.stack([V[c] for c in card])
okall = (np.abs(vv0 @ Kd.T) < np.pi / 2).all(-1) & (mag > np.median(mag))
for rb in range(F // nc):
    fs = [f for f in range(F) if f // nc == rb]
    A_, B_ = vv0[fs][okall[fs]], dphi[fs][okall[fs]]
    p = A_ @ Kd.T
    s = (p * B_).sum() / (p * p).sum()
    print(f"respiratory bin {rb}: slope {s:.3f}, corr {np.corrcoef(p.ravel(), B_.ravel())[0, 1]:.3f} ({okall[fs].sum()} samples)")
line = []
for f in range(F):
    o = okall[f]
    p, b = vv0[f][o] @ Kd.T, dphi[f][o]
    line.append(f"{f}:{(p * b).sum() / (p * p).sum():.2f}/{np.corrcoef(p.ravel(), b.ravel())[0, 1]:.2f}")
print("per frame slope/corr:", " ".join(line))
