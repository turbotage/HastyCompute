"""Respiratory kinematics of the atlas phantom: u_r(X, t) = r(t) Σ_k D_k B_k(X), r ∈ [0, 1] (0 end-expiration).

    python -I resp_motion.py <tissue_2mm.hv> <out_dir> [--dsi 15] [--dap 7] [--dlr 3]

Basis fields on a 4 mm grid covering the trunk (z 0.85–1.55 m; outside the tissue field of view the abdomen continues
the lowest labelled slice, the neck is still), unit 1 m:
- si: internal organs (lungs, heart, mediastinum, liver, stomach, spleen, bowel, … : the filled internal cavity,
      smoothed σ 8 mm) move caudally: 1 below the diaphragm dome, ((z_apex − z) / (z_apex − z_dome))^1.5 above it
      (heart ≈ 0.7 of the diaphragm, lung apex 0), fading to 0.4 low in the abdomen. The chest wall does not move SI —
      lungs slide along it.
- ap: anterior expansion of the trunk, linear from the spine (fixed) to the anterior wall.
- lr: lateral expansion of the lower ribs (bucket handle).
Amplitudes D (end-inspiration vs end-expiration): si 15 mm, ap 7 mm, lr 3 mm by default. The waveform r(t) is made by
`motion.py` (Lujan cos^{2n} breaths with breath-to-breath variability of period and depth).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
TS, OUT = vf.read(A[0]), Path(A[1])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opt(n, d):
    return float(A[A.index(n) + 1]) if n in A else d


D = {"si": opt("--dsi", 15.0) * 1e-3, "ap": opt("--dap", 7.0) * 1e-3, "lr": opt("--dlr", 3.0) * 1e-3}
lab = np.asarray(TS["labels"])
names = list(TS["tissues"])
ID = {n: i for i, n in enumerate(names)}
h0 = float(TS["voxel_m"])
o0 = np.asarray(TS["origin_m"], float)
INTERNAL = ["lung", "airway", "myocardium", "valve", "blood_LA", "blood_LV", "blood_RA", "blood_RV", "liver", "gallbladder", "spleen", "thymus",
            "stomach_wall", "gi_content", "pancreas", "bowel_wall", "kidney"]
I = np.isin(lab, [ID[n] for n in INTERNAL if n in ID])
I = np.pad(I, 4, mode="edge")  # closing erodes at the array border: pad by edge replication, then crop
I = ndimage.binary_closing(I, iterations=3)[4:-4, 4:-4, 4:-4]
I = np.stack([ndimage.binary_fill_holes(s_) for s_ in I])  # mediastinal fat / vessels between the organs
z_dome = o0[2] + (np.nonzero((lab == ID["liver"]).any(axis=(1, 2)))[0].max() + 0.5) * h0
z_apex = o0[2] + (np.nonzero((lab == ID["lung"]).any(axis=(1, 2)))[0].max() + 0.5) * h0
yy = np.nonzero(np.isin(lab, [ID["skin"], ID["fat"], ID["muscle"]]).any(axis=(0, 2)))[0]
y_ant = o0[1] + (yy.min() + 0.5) * h0
spine = (lab == ID["bone_cortical"]) | (lab == ID["bone_marrow"])
xs = o0[0] + (np.arange(lab.shape[2]) + 0.5) * h0
sp_y = np.nonzero(spine[:, :, np.abs(xs) < 0.02].any(axis=(0, 2)))[0]
sp_yw = o0[1] + (sp_y + 0.5) * h0
y_post = float(np.median(sp_yw[sp_yw > 0.02]))  # vertebral bodies (posterior bone in the midline) ≈ the AP hinge

# 4 mm resp grid
h = 4e-3
lo = np.array([-0.25, -0.16, 0.85])
hi = np.array([0.25, 0.16, 1.55])
dims = np.round((hi - lo) / h).astype(int)
nx, ny, nz = dims
Z, Y, Xg = np.meshgrid(*(lo[a] + (np.arange(dims[a]) + 0.5) * h for a in (2, 1, 0)), indexing="ij")
# internal weight sampled from the label grid (nearest in x, y; z below the field of view: the lowest slice; above: 0)
iz = np.floor((Z - o0[2]) / h0).astype(int)
above = iz >= lab.shape[0]
iz = np.clip(iz, 0, lab.shape[0] - 1)
iy = np.clip(np.floor((Y - o0[1]) / h0).astype(int), 0, lab.shape[1] - 1)
ix = np.clip(np.floor((Xg - o0[0]) / h0).astype(int), 0, lab.shape[2] - 1)
inside_xy = (Xg >= o0[0]) & (Xg < o0[0] + lab.shape[2] * h0) & (Y >= o0[1]) & (Y < o0[1] + lab.shape[1] * h0)
Wint = np.where(above | ~inside_xy, 0.0, I[iz, iy, ix].astype(float))
Wint = ndimage.gaussian_filter(Wint, 8e-3 / h)

f_si = np.where(Z <= z_dome, 1.0, np.clip((z_apex - Z) / (z_apex - z_dome), 0, 1) ** 1.5)
f_si = np.where(Z < z_dome - 0.12, np.clip(1 - 0.6 * (z_dome - 0.12 - Z) / 0.18, 0.4, 1), f_si)
trunk = np.clip((0.22 - np.abs(Xg)) / 0.05, 0, 1)
B_si = np.zeros(Z.shape + (3,))
B_si[..., 2] = -Wint * f_si * trunk
g_ap = np.clip((y_post - Y) / (y_post - y_ant), 0, 1)
f_ap = np.clip(np.minimum((Z - 0.90) / 0.10, 1), 0.5, 1) * np.where(Z > 1.42, np.clip(1 - 0.7 * (Z - 1.42) / 0.08, 0.3, 1), 1)
B_ap = np.zeros(Z.shape + (3,))
B_ap[..., 1] = -g_ap * f_ap * trunk
f_lr = np.exp(-0.5 * ((Z - 1.22) / 0.07) ** 2)
B_lr = np.zeros(Z.shape + (3,))
B_lr[..., 0] = np.clip(Xg / 0.15, -1.3, 1.3) * f_lr * trunk
B = np.stack([B_si * D["si"], B_ap * D["ap"], B_lr * D["lr"]])  # amplitudes folded in: u_r = r(t) Σ_k B_k
for k in range(3):
    for c in range(3):
        B[k, ..., c] = ndimage.gaussian_filter(B[k, ..., c], 1.0)
Ut = B.sum(0)
geo = {"z_dome": z_dome, "z_apex": z_apex, "y_ant": y_ant, "y_post": y_post, "amplitudes_mm": {k: v * 1e3 for k, v in D.items()}}


def at(p):
    i = np.clip(((np.asarray(p) - lo) / h).astype(int), 0, dims - 1)
    return (Ut[i[2], i[1], i[0]] * 1e3).round(1).tolist()


geo["u_end_inspiration_mm"] = {"heart centre": at([0.03, -0.03, 1.29]), "liver dome": at([-0.03, -0.02, 1.25]), "lung apex": at([-0.06, 0.0, 1.42]),
                               "sternum": at([0.0, -0.11, 1.30]), "spine": at([0.0, 0.07, 1.30])}
vf.write(OUT / "resp_motion.hv", {"kind": "phantom_motion_resp", "bases": ["si", "ap", "lr"], "B": np.ascontiguousarray(np.moveaxis(B, -1, 1)).astype(np.float32),
                                   "origin_m": lo.tolist(), "voxel_m": h, "dims_xyz": dims.tolist(), **{k: float(v) for k, v in geo.items() if not isinstance(v, dict)},
                                   "note": "u_r(X, t) = r(t) Σ_k B_k(X) (amplitudes included, metres), r = 0 end-expiration, 1 end-inspiration; trilinear, clamped at the grid edge"})
(OUT / "resp_motion.json").write_text(json.dumps(geo, indent=1))
print(json.dumps(geo, indent=1))

fig, axs = plt.subplots(1, 3, figsize=(17, 6))
j0 = int((0.0 - lo[1]) / h)
i0 = int((0.03 - lo[0]) / h)
mag = np.linalg.norm(Ut, axis=-1) * 1e3
for a_, (img, ext, t_) in zip(axs, [(mag[:, j0, :][::-1], [lo[0], hi[0], lo[2], hi[2]], "coronal y = 0"), (mag[:, :, i0][::-1], [lo[1], hi[1], lo[2], hi[2]], "sagittal x = 30 mm"),
                                    (Ut[:, :, i0, 2][::-1] * 1e3, [lo[1], hi[1], lo[2], hi[2]], "u_z sagittal [mm]")]):
    im = a_.imshow(img, extent=[e * 1e3 for e in ext], cmap="viridis" if "u_z" not in t_ else "coolwarm", vmin=None if "u_z" not in t_ else -D["si"] * 1e3, vmax=None if "u_z" not in t_ else D["si"] * 1e3)
    a_.set_title(f"end-inspiration |u| [mm], {t_}" if "u_z" not in t_ else t_)
    fig.colorbar(im, ax=a_, fraction=0.04)
fig.savefig(OUT / "figs" / "03_resp_field.png", dpi=90, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)
