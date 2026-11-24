"""Decompose recon_p2np's ESPIRiT metric (mean 1-|<s^,s>| = 0.326) by region, from what the run saved:
the masked ESPIRiT slice smaps_axial_z20.hv, mask_maps/mask_support, warm_full (pooled image), and the true maps.
Usage: nice python region_error.py [run_dir]"""
import os, sys
import numpy as np
from scipy.ndimage import distance_transform_edt
from common import RUN, true_maps_cc, ip_err, store

run = sys.argv[1] if len(sys.argv) > 1 else RUN
Z = 20
tru = true_maps_cc()                       # [x,y,z,v]
nv = tru.shape[:3]
# 1. the 'signal voxel' set of the metric: true-map energy > 0.05 median (unwhitened-compressed here; same idea)
e = np.sum(np.abs(tru) ** 2, -1)
sig = e > 0.05 * np.median(e[e > 0])
print(f"metric voxel set: {sig.sum()} of {sig.size} voxels have true-map energy > 0.05 median (the whole FOV)")
est = store.load_any(os.path.join(run, f"smaps_axial_z{Z}.hv"))   # [x,y,v], after the support mask
mm = store.load_any(os.path.join(run, "mask_maps.hv")) > 0
sup = store.load_any(os.path.join(run, "mask_support.hv")) > 0
img = store.load_any(os.path.join(run, "warm_full.hv"))           # [x,y,z,e]
rho2 = np.sum(np.abs(img[:, :, Z, :]) ** 2, -1)
t = tru[:, :, Z, :]
e1, orth = ip_err(est, t)
nz = np.any(est != 0, -1)
s2 = sig[:, :, Z]
print(f"slice z={Z}: {s2.sum()} metric voxels, {nz.sum()} with a nonzero estimated map, {(s2 & ~nz).sum()} zero (each counts error 1)")
print(f"  slice metric {e1[s2].mean():.3f}; of it from zero-map voxels {(~nz & s2).sum() / s2.sum():.3f}")
body = sup[:, :, Z]
dist = distance_transform_edt(body, sampling=(3.97, 4.0))
regions = {
    "air (outside dilated support)": s2 & ~mm[:, :, Z],
    "body (support)": body,
    "body, <= 20 mm from the skin": body & (dist <= 20),
    "body, > 20 mm from the skin": body & (dist > 20),
    "body, > 50 mm from the skin": body & (dist > 50),
    "dilation ring (mask_maps \\ support)": mm[:, :, Z] & ~body,
}
w = rho2 / rho2[body].mean()
print(f"{'region':38s} {'voxels':>7s} {'zero':>6s} {'1-|ip|':>8s} {'nonzero':>8s} {'p90':>7s} {'|rho|^2-wt':>10s} {'orth(wt)':>9s}")
for k, r in regions.items():
    rn = r & nz
    pw = np.sum(w[rn] * e1[rn]) / max(np.sum(w[rn]), 1e-30)
    po = np.sum(w[rn] * orth[rn]) / max(np.sum(w[rn]), 1e-30)
    print(f"{k:38s} {r.sum():7d} {(r & ~nz).sum():6d} {e1[r].mean() if r.any() else np.nan:8.3f} {e1[rn].mean() if rn.any() else np.nan:8.3f} "
          f"{np.percentile(e1[rn], 90) if rn.any() else np.nan:7.3f} {pw:10.4f} {po:9.4f}")
# implied whole-volume numbers from the run log (0.326 over 552950, 135993 zero)
N, Zr, M = 552950, 135993, 0.326
print(f"whole volume from the log: zero-map voxels contribute {Zr / N:.3f} of {M}; mean over nonzero voxels {(M * N - Zr) / (N - Zr):.3f}")
# per-voxel phase convention check: the metric is invariant to a per-voxel phase; magnitude ratio |s^_c|/|s_c| spread
rn = body & nz & (dist > 20)
a = np.abs(est[rn]); b = np.abs(t[rn]) / np.linalg.norm(t[rn], axis=-1, keepdims=True)
print(f"deep body: median relative coil-profile error | |s^| - |s|/||s|| |/|s|/||s|| over coils: {np.median(np.linalg.norm(a - b, axis=-1)):.3f}")
np.save(os.path.join(os.path.dirname(__file__), "region_err_z20.npy"), np.stack([e1, orth, nz, body, dist]).astype(np.float32))
