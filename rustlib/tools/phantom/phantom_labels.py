"""Assemble the tissue label volume of the atlas phantom from the `phantomvox` group fields, with figures.

    python -I phantom_labels.py <vox.hv> <tissues.json> <out_tissue.hv> <fig_dir> [--at x,y,z]

Label rules (later rules overwrite earlier ones; w = group winding > ½, env = enclosed by the group's shell):
  body env → fat (subcutaneous and visceral adipose / connective tissue), body w → skin, muscle, bone (cortex = within
  max(1.5 mm, 1 voxel) of the bone surface, marrow inside), cartilage, organs, stomach / bowel wall (w) and content
  (env ∧ ¬w), lung, airway (w ∨ env), heart chambers (env ∧ ¬w of the myocardial shell, chamber = the member whose
  inner surface encloses the voxel; unassigned cavity voxels take the nearest assigned chamber), myocardium,
  papillary muscles (myocardium), valves. Vessels come from the vessel graph (separate step).
Output: labels u8 [Z, Y, X] (index into the tissue table), tissue names and property columns, grid geometry.
Figures: labels, PD / T1 / T2* / χ maps, a spoiled-GRE magnitude, tissue volumes.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

V = vf.read(sys.argv[1])
TT = json.loads(Path(sys.argv[2]).read_text())
OUT = Path(sys.argv[3])
FIG = Path(sys.argv[4])
FIG.mkdir(parents=True, exist_ok=True)
cols = TT["columns"]
names = [r[0] for r in TT["tissues"]]
prop = {c: np.array([r[i] for r in TT["tissues"]], dtype=object if c in ("name", "group") else np.float64) for i, c in enumerate(cols)}
ID = {n: i for i, n in enumerate(names)}
nx, ny, nz = [int(x) for x in V["dims_xyz"]]
o = np.asarray(V["origin_m"], float)
h = float(V["voxel_m"])
G = {g["name"]: g for g in V["groups"]}


def field(gname, key):
    g = G.get(gname)
    if g is None or key not in g:
        return None
    lo = [int(x) for x in g["box_lo"]]
    d = [int(x) for x in g["box_dims"]]
    a = np.zeros((nz, ny, nx), np.uint8)
    a[lo[2] : lo[2] + d[2], lo[1] : lo[1] + d[1], lo[0] : lo[0] + d[0]] = np.asarray(g[key])
    return a


def inside(gname, key="w"):
    a = field(gname, key)
    return None if a is None else a > 64  # u8 = w × 127.5; > ½


lab = np.zeros((nz, ny, nx), np.uint8)


def put(mask, tissue):
    if mask is not None:
        lab[mask] = ID[tissue]


put(inside("body", "env"), "fat")
put(inside("body"), "skin")
put(inside("muscle"), "muscle")
b = inside("bone")
if b is not None:
    d = ndimage.distance_transform_edt(b, sampling=h)
    put(b & (d <= max(1.5e-3, h * 1.01)), "bone_cortical")
    put(b & (d > max(1.5e-3, h * 1.01)), "bone_marrow")
put(inside("cartilage"), "cartilage")
for g in ("liver", "gallbladder", "spleen", "thymus", "pancreas", "thyroid"):
    put(inside(g), g)
for g, wall in (("stomach", "stomach_wall"), ("intestine", "bowel_wall")):
    w, e = inside(g), inside(g, "env")
    if w is not None:
        put(e & ~w, "gi_content")
        put(w, wall)
put(inside("lung"), "lung")
aw, ae = inside("airway"), inside("airway", "env")
if aw is not None:
    put(aw | ae, "airway")
hw, he = inside("heart_wall"), inside("heart_wall", "env")
chamber_stats = {}
if hw is not None:
    cav = he & ~hw
    mem = field("heart_wall", "member")
    members = list(G["heart_wall"]["members"])
    tissue_of = {i + 1: {"Left atrium": "blood_LA", "Left ventricle": "blood_LV", "Right atrium": "blood_RA", "Right ventricle": "blood_RV"}[m] for i, m in enumerate(members)}
    assigned = cav & (mem > 0)
    if assigned.any() and (cav & ~assigned).any():  # unassigned cavity voxels: nearest assigned chamber
        _, idx = ndimage.distance_transform_edt(~assigned, return_indices=True)
        mem = np.where(cav & ~assigned, mem[tuple(idx)], mem)
    for k, t in tissue_of.items():
        m = cav & (mem == k)
        put(m, t)
        chamber_stats[t] = float(m.sum() * (h * 100) ** 3)
    put(hw, "myocardium")
put(inside("papillary"), "myocardium")
put(inside("valve"), "valve")

# ── write ───────────────────────────────────────────────────────────────────────────────────────────────────────────
vf.write(
    OUT,
    {
        "labels": lab,
        "tissues": names,
        **{c: prop[c].astype(np.float32) for c in cols if c not in ("name", "group")},
        "tissue_group": [str(x) for x in prop["group"]],
        "origin_m": [float(x) for x in o],
        "voxel_m": h,
        "dims_xyz": [nx, ny, nz],
        "source": "Z-Anatomy (CC BY-SA 4.0) / BodyParts3D (CC BY-SA 2.1 JP) meshes, voxelised by generalised winding number; properties: " + Path(sys.argv[2]).name,
    },
)
vol = {n: float((lab == i).sum() * (h * 100) ** 3) for i, n in enumerate(names)}
(FIG / "volumes_cm3.txt").write_text("".join(f"{n:18s} {x:10.1f}\n" for n, x in vol.items() if x > 0))
print("chambers (mL):", {k: round(v, 1) for k, v in chamber_stats.items()})

# ── figures ─────────────────────────────────────────────────────────────────────────────────────────────────────────
at = np.asarray([float(s) for s in sys.argv[sys.argv.index("--at") + 1].split(",")]) if "--at" in sys.argv else o + 0.5 * h * np.array([nx, ny, nz])
ix, iy, iz = [int(np.clip((at[a] - o[a]) / h, 0, n - 1)) for a, n in zip(range(3), (nx, ny, nz))]
ext = {
    "axial": [o[0] * 1e3, (o[0] + nx * h) * 1e3, (o[1] + ny * h) * 1e3, o[1] * 1e3],
    "coronal": [o[0] * 1e3, (o[0] + nx * h) * 1e3, o[2] * 1e3, (o[2] + nz * h) * 1e3],
    "sagittal": [o[1] * 1e3, (o[1] + ny * h) * 1e3, o[2] * 1e3, (o[2] + nz * h) * 1e3],
}
title = {"axial": f"axial z={at[2] * 1e3:.0f} mm", "coronal": f"coronal y={at[1] * 1e3:.0f} mm", "sagittal": f"sagittal x={at[0] * 1e3:.0f} mm"}


def sl(a):
    return {"axial": a[iz], "coronal": a[:, iy, :][::-1], "sagittal": a[:, :, ix][::-1]}


COL = {
    "air": "#000000", "fat": "#f2e394", "skin": "#d9a07a", "muscle": "#b5413a", "bone_cortical": "#ffffff", "bone_marrow": "#c9b28f",
    "cartilage": "#8fd3e8", "liver": "#7a3b22", "gallbladder": "#3fa34d", "spleen": "#6b2a6b", "thymus": "#e0c060", "stomach_wall": "#d67fa6",
    "gi_content": "#5a6e8c", "pancreas": "#e8a33d", "bowel_wall": "#c46f8e", "thyroid": "#b07de0", "lung": "#2b3a55", "airway": "#101828",
    "myocardium": "#8c1c13", "valve": "#ffd400", "blood_LA": "#ff5a5a", "blood_LV": "#ff2020", "blood_RA": "#4a7dff", "blood_RV": "#2050ff",
    "blood_artery": "#ff3030", "blood_vein": "#3060ff", "blood_pulm_artery": "#30a0ff", "blood_pulm_vein": "#ff6080", "vessel_wall": "#aaaaaa", "kidney": "#9a5a3a",
}
cm = ListedColormap([COL.get(n, "#ff00ff") for n in names])
fig, axs = plt.subplots(1, 3, figsize=(19, 6.8))
for a, (k, s) in zip(axs, sl(lab).items()):
    a.imshow(s, cmap=cm, vmin=-0.5, vmax=len(names) - 0.5, extent=ext[k], interpolation="nearest")
    a.set_title(title[k])
    a.set_xlabel("mm")
present = [i for i, n in enumerate(names) if vol[n] > 0]
handles = [plt.Rectangle((0, 0), 1, 1, color=COL.get(names[i], "#f0f")) for i in present]
fig.legend(handles, [names[i] for i in present], loc="center right", fontsize=9, frameon=False)
fig.subplots_adjust(right=0.88)
fig.suptitle(f"atlas phantom tissue labels, voxel {h * 1e3:.2f} mm (Z-Anatomy, CC BY-SA 4.0)")
fig.savefig(FIG / "01_labels.png", dpi=110, bbox_inches="tight")
plt.close(fig)

maps = {
    "PD": (prop["pd"][lab], "gray", 0, 1),
    "T1 [s]": (prop["t1_s"][lab], "viridis", 0, 2.6),
    "T2* [ms]": (np.minimum(prop["t2s_s"][lab], 0.2) * 1e3, "magma", 0, 100),
    "χ [ppm]": (prop["chi_ppm"][lab], "coolwarm", -11, -7),
}
fig, axs = plt.subplots(len(maps), 3, figsize=(15, 4.6 * len(maps)))
for r, (mn, (a3, cmap, lo, hi)) in enumerate(maps.items()):
    for c, (k, s) in enumerate(sl(a3).items()):
        im = axs[r, c].imshow(s, cmap=cmap, vmin=lo, vmax=hi, extent=ext[k], interpolation="nearest")
        axs[r, c].set_title(f"{mn} — {title[k]}", fontsize=9)
    fig.colorbar(im, ax=axs[r], fraction=0.015)
fig.savefig(FIG / "02_properties.png", dpi=90, bbox_inches="tight")
plt.close(fig)

# spoiled GRE, water + fat (single peak), TR 5 ms, flip 15°, TE 2.46 ms (≈ out of phase at 3 T is 1.15 ms, in phase 2.3 ms)
TR, FA, TE = 5e-3, np.deg2rad(15), 2.46e-3
fat = ID["fat"]


def spgr(t1, pd):
    e1 = np.exp(-TR / t1)
    return pd * np.sin(FA) * (1 - e1) / (1 - np.cos(FA) * e1)


ff = prop["ff"]
wsig = spgr(prop["t1_s"], prop["pd"] * (1 - ff)) * np.exp(-TE / prop["t2s_s"])
fsig = spgr(np.full_like(ff, prop["t1_s"][fat]), prop["pd"] * ff) * np.exp(-TE / prop["t2s_s"][fat])
phase = np.exp(2j * np.pi * TT["fat_shift_ppm"] * 1e-6 * 42.577e6 * TT["field_T"] * TE)
sig = np.abs(wsig + fsig * phase)
img = sig[lab]
fig, axs = plt.subplots(1, 3, figsize=(18, 6.3))
for a, (k, s) in zip(axs, sl(img).items()):
    a.imshow(s, cmap="gray", vmin=0, vmax=np.percentile(img[lab > 0], 99.5), extent=ext[k], interpolation="nearest")
    a.set_title(f"SPGR TR 5 ms, 15°, TE 2.46 ms — {title[k]}", fontsize=9)
fig.savefig(FIG / "03_spgr.png", dpi=100, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT, "and", FIG, "| volumes (cm³):", {n: round(x) for n, x in vol.items() if x > 0})
