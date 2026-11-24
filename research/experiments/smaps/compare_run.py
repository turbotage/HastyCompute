"""The run's ESPIRiT slice (smaps_axial_z20, full-ADC calibration, support-masked) on the oracle's regions."""
import os, sys, numpy as np
from scipy.ndimage import map_coordinates, binary_fill_holes, distance_transform_edt
from common import SCAN, RUN, meta, recon_positions, true_maps_cc, ip_err, store
run = sys.argv[1] if len(sys.argv) > 1 else RUN
z = 20
m = meta(); x, vox = recon_positions(m); res = m["res"][0]; td = m["recon_dims"]
g = [np.clip(x[:, :, z, k] / res + td[k] // 2, 0, td[k] - 1 - 1e-9) for k in range(3)]
samp = lambda f: map_coordinates(np.nan_to_num(np.asarray(f, np.float32)), g, order=1, mode="nearest")
pd = samp(np.load(os.path.join(SCAN, "truth_pd_water.npy"))) + samp(np.load(os.path.join(SCAN, "truth_pd_fat.npy")))
sig = pd * np.exp(-samp(np.load(os.path.join(SCAN, "truth_r2s.npy"))) * 1.6e-3)
# the oracle's threshold is over the whole volume; recompute it the same way
g3 = [np.clip(x[..., k] / res + td[k] // 2, 0, td[k] - 1 - 1e-9) for k in range(3)]
s3 = (map_coordinates(np.nan_to_num(np.load(os.path.join(SCAN, "truth_pd_water.npy"))), g3, order=1) + map_coordinates(np.nan_to_num(np.load(os.path.join(SCAN, "truth_pd_fat.npy"))), g3, order=1)) * np.exp(-map_coordinates(np.nan_to_num(np.load(os.path.join(SCAN, "truth_r2s.npy"))), g3, order=1) * 1.6e-3)
thr = 0.05 * np.median(s3[s3 > 0])
obj = sig > thr; body = binary_fill_holes(obj); lung = body & ~obj
dist = distance_transform_edt(body, sampling=[1e3 * v for v in vox[:2]])
est = store.load_any(os.path.join(run, f"smaps_axial_z{z}.hv"))
t = true_maps_cc()[:, :, z]
e1, orth = ip_err(est, t)
nz = np.any(est != 0, -1)
for k, r in {"tissue": obj, "skin<=20mm": obj & (dist <= 20), "deep>50mm": obj & (dist > 50), "lung/low-signal holes": lung}.items():
    print(f"run z={z} {k:22s} n={r.sum():5d} zero-map {np.mean(~nz[r]):.2f}  e1 over nonzero {e1[r & nz].mean():.4f} (p90 {np.percentile(e1[r & nz], 90):.4f})")
