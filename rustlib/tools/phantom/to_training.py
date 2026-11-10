"""Phantom moving truth → a 7-point dual-VENC denoiser training case (the RACLETTE tier-1 layout of
tools/raclette/build_training.py, read by tools/denoiser/dataset.py).

    rustlib/.venv/bin/python -I rustlib/tools/phantom/to_training.py <truth_dir> <out_root> <case_name> [--seed s]
        [--h-mm 2] [--margin-mm 12] [--split train|val|test] [--group <name>] [--grade Healthy]

then `hvtool pack-tree <out_root>` packs x.npy / mask.npy into case.hv, and `--index` (re)writes <out_root>/index.json
from all case.json files.

Image model per native (truth) voxel, frame f: spoiled-GRE magnitude from the phantom's tissue table with its
textures (tissue_variation.apply; flip, TR, TE drawn per case), blood × an inflow gain (drawn), times a smooth static
background phase θ(x) (Gaussian random field + linear ramp, drawn), times exp(+i k_e·v) for the 7 encodes
[ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi], k_e = π / VENC_e along the truth grid axes (VENC_lo 1 m/s, VENC_hi drawn
1.5–3.5 m/s, the RACLETTE 7p range). The case grid is h (default 2 mm): the complex mean of the native voxels in each
case voxel — the intravoxel velocity distribution dephases as in a real voxel. Lumen mask: blood fraction ≥ 0.5.
Cropped to the blood's box (any frame) + margin. x is stored / scale (p99 of |x_ref| in the body), as RACLETTE.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
sys.path.insert(0, str(HERE))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


if "--index" in A:
    root = Path(A[A.index("--index") + 1])
    cases = []
    for cj in sorted(root.glob("*/case.json")):
        c = json.loads(cj.read_text())
        cases.append({k: c[k] for k in ("variant", "subject", "group", "stenosis_grade", "peak_velocity_mps", "dims", "n_frames", "n_enc", "spacing_m")}
                     | {"sub": "", "geometry": c["subject"], "path": cj.parent.name, "split": c["split"]})
    groups = {c["group"]: c["split"] for c in cases}
    counts = {s: sum(c["split"] == s for c in cases) for s in ("train", "val", "test")}
    (root / "index.json").write_text(json.dumps({"groups": groups, "counts": counts, "cases": cases}, indent=1))
    print(f"{root / 'index.json'}: {counts}")
    sys.exit(0)

TRUTH, ROOT, NAME = Path(A[0]), Path(A[1]), A[2]
seed = int(opt("--seed", 0))
rng = np.random.default_rng(seed)
H = float(opt("--h-mm", 2.0)) * 1e-3
MARG = float(opt("--margin-mm", 12.0)) * 1e-3
OUT = ROOT / NAME
OUT.mkdir(parents=True, exist_ok=True)

T = vf.read(TRUTH / "truth4d.hv")
labels = np.asarray(T["labels"])  # [F, Z, Y, X]
vel = np.asarray(T["vel"])  # [F, 3, Z, Y, X] f16
blood = np.asarray(T["blood"])  # u8 /255
tex = np.asarray(T["tex"]) if "tex" in T else None
vox = float(T["voxel_m"])
o = np.asarray(T["origin_m"], float)
F = labels.shape[0]
tc = np.asarray(T["tc"], float)
tp = json.loads(T["tissue_params"]) if T.get("tissue_params") else json.loads((TRUTH.parent / "tissue_params.json").read_text())
names = list(T["tissues"])
tab = tp["table"]
k = int(round(H / vox))
assert abs(k * vox - H) < 1e-9, f"h {H} is not a multiple of the truth voxel {vox}"

# crop: the blood's box (any frame) + margin, to a multiple of k
anyb = (blood > 127).any(0)
nz = np.nonzero(anyb)
m = int(np.ceil(MARG / vox))
lo = np.maximum([min(a) - m for a in nz], 0)
hi = np.minimum([max(a) + m + 1 for a in nz], anyb.shape)
hi = lo + (hi - lo) // k * k
sl = tuple(slice(a, b) for a, b in zip(lo, hi))
print(f"{TRUTH}: {F} frames, crop z {lo[0]}:{hi[0]} y {lo[1]}:{hi[1]} x {lo[2]}:{hi[2]} at {vox * 1e3:.1f} mm → ×{k} = {H * 1e3:.1f} mm")

# drawn acquisition / appearance
flip = np.deg2rad(rng.uniform(8, 15))
TR = rng.uniform(4e-3, 11.5e-3)
TE = rng.uniform(1.5e-3, 2.6e-3)
gain_in = rng.uniform(1.0, 2.0)
venc_lo, venc_hi = 1.0, float(rng.uniform(1.5, 3.5))
PD = np.array([tab[n]["pd"] if n in tab else 0.0 for n in names], np.float32)
T1 = np.array([tab[n]["t1_s"] if n in tab else 1.0 for n in names], np.float32)
T2s = np.array([tab[n]["t2s_s"] if n in tab else 0.05 for n in names], np.float32)
BL = np.array([n.startswith("blood") for n in names])
inflow = np.where(BL, gain_in, 1.0).astype(np.float32)
air = np.array([n == "air" for n in names])
PD[air] = 0.0

Zc, Yc, Xc = [b - a for a, b in zip(lo, hi)]
# background phase: smooth Gaussian random field (σ ≈ 40 mm) of 0.5–2 rad peak + a linear ramp (≤ 1 rad over the box)
g = ndimage.gaussian_filter(rng.standard_normal((Zc // k, Yc // k, Xc // k)), 40e-3 / H, mode="wrap")
g *= rng.uniform(0.5, 2.0) / max(np.abs(g).max(), 1e-12)
zz, yy, xx = np.meshgrid(*(np.linspace(-0.5, 0.5, n) for n in g.shape), indexing="ij")
g += rng.uniform(-1, 1) * zz + rng.uniform(-1, 1) * yy + rng.uniform(-1, 1) * xx
theta = ndimage.zoom(g, k, order=1)[:Zc, :Yc, :Xc].astype(np.float32)
kv = np.array([0.0, np.pi / venc_lo, np.pi / venc_hi, np.pi / venc_lo, np.pi / venc_hi, np.pi / venc_lo, np.pi / venc_hi])
axis = [None, 0, 0, 1, 1, 2, 2]  # velocity component (x, y, z) of each encode


def down(a):
    """complex / real mean over k³ blocks of [Z, Y, X] → case grid"""
    Z_, Y_, X_ = a.shape
    return a.reshape(Z_ // k, k, Y_ // k, k, X_ // k, k).mean((1, 3, 5))


nzc, nyc, nxc = Zc // k, Yc // k, Xc // k
x = np.zeros((nxc, nyc, nzc, F, 7, 2), np.float32)  # C [X, Y, Z, F, E, 2]
mask = np.zeros((nxc, nyc, nzc, F), np.uint8)
body = np.zeros((nxc, nyc, nzc), bool)
vmax = 0.0
for f in range(F):
    L = labels[f][sl]
    ta = tex[f, 0][sl].astype(np.float32) if tex is not None else 0.0
    tb = tex[f, 1][sl].astype(np.float32) if tex is not None else 0.0
    pd = PD[L] * np.clip(1 + ta, 0.2, 2.0)
    t1 = T1[L] * np.clip(1 + tb / 3, 0.5, 1.5)
    t2s = T2s[L] * np.clip(1 + tb, 0.2, 2.0)
    e1 = np.exp(-TR / t1)
    mag = pd * inflow[L] * np.sin(flip) * (1 - e1) / (1 - np.cos(flip) * e1) * np.exp(-TE / t2s)
    v = vel[f][(slice(None),) + sl].astype(np.float32)  # [3, Z, Y, X]
    vmax = max(vmax, float(np.sqrt((v**2).sum(0)).max()))
    base = mag * np.exp(1j * theta)
    for e in range(7):
        xe = base if axis[e] is None else base * np.exp(1j * kv[e] * v[axis[e]])
        d = down(xe).transpose(2, 1, 0)  # [X, Y, Z]
        x[:, :, :, f, e, 0], x[:, :, :, f, e, 1] = d.real, d.imag
    mask[:, :, :, f] = (down(blood[f][sl].astype(np.float32) / 255.0) >= 0.5).transpose(2, 1, 0)
    body |= (down((~air[L]).astype(np.float32)) > 0.5).transpose(2, 1, 0)
ref = np.hypot(x[..., 0, 0], x[..., 0, 1])
scale = float(np.percentile(ref[body[..., None].repeat(F, -1)], 99))
xs = (x / scale).astype(np.float16)
np.save(OUT / "x.npy", xs)
np.save(OUT / "mask.npy", mask)
rt = float(np.abs(xs.astype(np.float32) * scale - x).max() / max(np.abs(x).max(), 1e-12))
encs = []
for e in range(7):
    d = [0.0, 0.0, 0.0]
    if axis[e] is not None:
        d[axis[e]] = 1.0
    kx = [kv[e] * c for c in d]
    encs.append({"dir_xyz": d, "k_grid": kx, "k_xyz": kx, "kv": float(kv[e]), "venc": 0.0 if axis[e] is None else (venc_lo if e % 2 else venc_hi)})
params = json.loads(T["params"]) if T.get("params") else {}
origin = o + (lo[::-1] + 0.5 * k - 0.5) * vox  # centre of case voxel 0 (x, y, z)
case = {
    "variant": "phantom_7p", "subject": NAME, "sub": "", "group": opt("--group", NAME.split("_")[0]), "split": opt("--split", "train"),
    "stenosis_grade": opt("--grade", "Healthy"), "peak_velocity_mps": round(vmax, 4),
    "dims": [int(nxc), int(nyc), int(nzc)], "orig_dims": [int(nxc), int(nyc), int(nzc)], "crop_lo": [0, 0, 0], "n_frames": F, "n_enc": 7, "n_coils": 0,
    "spacing_m": [H, H, H], "rot_grid_to_xyz": np.eye(3).tolist(), "origin_xyz_m": origin.tolist(),
    "frame_times_s": tc.tolist(), "encodings": encs, "scale": scale, "fp16_roundtrip_rel": rt,
    "lumen_voxels_per_frame": [int(v_) for v_ in mask.reshape(-1, F).sum(0)],
    "ms_parameters": {"flip_deg": float(np.rad2deg(flip)), "TR": TR, "TE": TE, "blood_inflow_gain": gain_in, "venc_hi": venc_hi, "contrast": tp.get("contrast")},
    "phase_convention": "x_e ∝ exp(+i k_e·v), k_grid and v in grid basis, k in rad/(m/s)",
    "layout": {"x": "float16 C [X,Y,Z,F,E,2] / scale", "mask": "uint8 C [X,Y,Z,F]"},
    "source": f"HastyCompute phantom {TRUTH} (Z-Anatomy / BodyParts3D atlas, CC BY-SA 4.0; hasty-cfd LBM chamber flow when the truth used it)",
    "phantom_params": params, "seed": seed,
}
(OUT / "case.json").write_text(json.dumps(case, indent=1))
print(f"{OUT}: dims {case['dims']} × {F} frames, lumen {case['lumen_voxels_per_frame'][0]} voxels, peak {vmax:.2f} m/s, VENC_hi {venc_hi:.2f}, "
      f"flip {np.rad2deg(flip):.1f}°, TR {TR * 1e3:.1f} ms, scale {scale:.3g}, fp16 round trip {rt:.1e}")
