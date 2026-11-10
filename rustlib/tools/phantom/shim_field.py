"""Unshimmed B0 field of a factory phantom for shim studies (research/shimming.md; `shimstudy`).

    rustlib/.venv/bin/python -I rustlib/tools/phantom/shim_field.py <phantom_dir> <out_dir> [--step 2] [--dev cpu|cuda]

Same field as prep_sim.py before its shim: dipole field of the χ map on the phantom's 1 mm atlas (FFT, padded, relative
to the surrounding air, B0 along z), cropped to the prescribed slab (slab.json, ± 20 mm) and the body's x, y extent,
subsampled by `--step` (mm). Writes <out_dir>/ (all [z, y, x] float32, C order):
  field_ppm.npy   field without any shim (ppm)
  signal.npy      expected signal weight: tissue PD (0 in air, airway, lung; lung parenchyma has little signal)
  heart.npy       1 inside the heart ROI: myocardium, valves, blood pools and vessels, vessel wall, plus every tissue voxel
                  within 12 mm of the myocardium (epicardial fat with the coronaries)
  box.npy         prep_sim's shim box (tissue within 0.8 × FOV in x, y and the slab ± 20 mm)
  body.npy        uint8 body mask (non-air) on this grid; body_coils.npy: prep_sim's coil-array body mask (atlas, 2 mm)
  geom.json       origin (m, x y z of voxel [0,0,0] centre), voxel (m), shape, FOV centre, slab
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
PH, OUT = Path(A[0]), Path(A[1])
OUT.mkdir(parents=True, exist_ok=True)


def opt(n, d):
    return A[A.index(n) + 1] if n in A else d


step = int(opt("--step", 2))
dev = torch.device(opt("--dev", "cpu"))
t0 = time.time()
TP = json.loads((PH / "tissue_params.json").read_text())
tab = TP["table"]
P0 = PH / "atlas" if (PH / "atlas" / "tissue_1mm.hv").exists() else REPO / "data/hasty/phantom/p0"
TS = vf.read(P0 / "tissue_1mm.hv")
names = list(TS["tissues"])
ID = {n: i for i, n in enumerate(names)}
lab = np.asarray(TS["labels"])
o = np.asarray(TS["origin_m"], float)
h = float(TS["voxel_m"])
slab = json.loads((PH / "slab.json").read_text())
z_lo, z_hi = float(slab["z_lo"]), float(slab["z_hi"])
print(f"atlas {P0} {lab.shape} at {h * 1e3:.0f} mm, slab z {z_lo * 1e3:.0f}…{z_hi * 1e3:.0f} mm")

# ── dipole field of the χ map relative to air (as prep_sim) ───────────────────────────────────────────────────────
chi_t = np.array([tab[n]["chi_ppm"] if "chi_ppm" in tab[n] else 0.0 for n in names], np.float32)
chi = torch.tensor(chi_t[lab] - float(tab["air"]["chi_ppm"]), device=dev)
pad = 64
sh = [n + 2 * pad for n in chi.shape]
F = torch.fft.rfftn(chi, s=sh)
kz = torch.fft.fftfreq(sh[0], device=dev)[:, None, None]
ky = torch.fft.fftfreq(sh[1], device=dev)[None, :, None]
kx = torch.fft.rfftfreq(sh[2], device=dev)[None, None, :]
k2 = kx**2 + ky**2 + kz**2
D = 1 / 3 - kz**2 / torch.where(k2 > 0, k2, torch.ones_like(k2))
D[0, 0, 0] = 0
field = torch.fft.irfftn(F * D, s=sh)[: chi.shape[0], : chi.shape[1], : chi.shape[2]].cpu().numpy().astype(np.float32)
del F, D, chi
print(f"dipole field {time.time() - t0:.0f}s")

# ── crop: slab ± 20 mm, body x / y extent + 10 mm ─────────────────────────────────────────────────────────────────
zi0 = max(int((z_lo - 0.02 - o[2]) / h), 0)
zi1 = min(int((z_hi + 0.02 - o[2]) / h) + 1, lab.shape[0])
body_full = lab[zi0:zi1] != ID["air"]
ys_, xs_ = np.nonzero(body_full.any(0))
yi0, yi1 = max(ys_.min() - 10, 0), min(ys_.max() + 11, lab.shape[1])
xi0, xi1 = max(xs_.min() - 10, 0), min(xs_.max() + 11, lab.shape[2])
sl = (slice(zi0, zi1, step), slice(yi0, yi1, step), slice(xi0, xi1, step))
L = lab[sl]
f = field[sl]
pd = np.array([tab[n].get("pd", 0.0) for n in names], np.float32)
no_signal = np.isin(L, [ID["air"], ID["airway"], ID["lung"]])
signal = np.where(no_signal, 0.0, pd[L]).astype(np.float32)
# heart ROI: heart labels, plus tissue within 12 mm of the myocardium (epicardial fat, coronaries)
heart_ids = [ID[n] for n in ("myocardium", "valve", "blood_LA", "blood_LV", "blood_RA", "blood_RV", "blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein", "vessel_wall")]
myo = torch.tensor(L == ID["myocardium"], dtype=torch.float32)[None, None]
r = int(round(12e-3 / (h * step)))
near = torch.nn.functional.max_pool3d(myo, 2 * r + 1, 1, r)[0, 0].numpy() > 0
heart = (np.isin(L, heart_ids) | (near & ~no_signal)).astype(np.float32)
# prep_sim's shim box: tissue within 0.8 × FOV (x, y) and the slab ± 20 mm, FOV from the body extent
zc = o[2] + (np.arange(zi0, zi1, step) + 0.5) * h
yc = o[1] + (np.arange(yi0, yi1, step) + 0.5) * h
xc = o[0] + (np.arange(xi0, xi1, step) + 0.5) * h
fovc = np.array([0.5 * (xc[0] + xc[-1]), 0.5 * (yc[0] + yc[-1]), 0.5 * (z_lo + z_hi)])
fov = np.array([xc[-1] - xc[0], yc[-1] - yc[0], 1.3 * (z_hi - z_lo)])
Z, Y, X = np.meshgrid(zc - fovc[2], yc - fovc[1], xc - fovc[0], indexing="ij")
tissue = ~np.isin(L, [ID["air"], ID["airway"], ID["lung"]])
box = ((np.abs(X) < 0.4 * fov[0]) & (np.abs(Y) < 0.4 * fov[1]) & (np.abs(Z) < 0.5 * fov[2] + 0.02) & tissue).astype(np.float32)
myo = (L == ID["myocardium"]).astype(np.float32)
for name, arr in [("field_ppm", f), ("signal", signal), ("heart", heart), ("box", box), ("myo", myo)]:
    np.save(OUT / f"{name}.npy", np.ascontiguousarray(arr, dtype=np.float32))
np.save(OUT / "body.npy", np.ascontiguousarray(L != ID["air"], dtype=np.uint8))
# the coil array's body mask exactly as prep_sim writes it (whole atlas at 2 mm)
np.save(OUT / "body_coils.npy", np.ascontiguousarray(lab[::2, ::2, ::2] != ID["air"], dtype=np.uint8))
geom = {"body_coils_origin_m": (o + h / 2).tolist(), "body_coils_voxel_m": 2 * h, "origin_m": [float(xc[0]), float(yc[0]), float(zc[0])], "voxel_m": h * step, "shape_zyx": list(L.shape), "fov_center_m": fovc.tolist(), "slab_m": [z_lo, z_hi], "atlas": str(P0)}
(OUT / "geom.json").write_text(json.dumps(geom, indent=1))
fh = f[heart > 0] * 127.7
fs = f[signal > 0] * 127.7
print(f"{L.shape} voxels at {h * step * 1e3:.0f} mm; heart ROI {int(heart.sum())}, unshimmed field (3 T) heart 1–99 %: {np.percentile(fh, 1):.0f}…{np.percentile(fh, 99):.0f} Hz, signal voxels {np.percentile(fs, 1):.0f}…{np.percentile(fs, 99):.0f} Hz ({time.time() - t0:.0f}s)")
