"""Export one RACLETTE frame as a static image object for `simimg` (operator validation).

    python -I export_case.py <variant> <subject> [--frame f|peak] [--res mm] [--venc 2.5]

Writes data/hasty/external/raclette_cases/<subject>_f<frame>/:
  object.npy   complex64 [x, y, z, 4]  reference + the three encodings at `--venc` (RACLETTE images as-is)
  smaps.npy    complex64 [x, y, z, 8]  RACLETTE coil sensitivities
  mask.npy     float32   [x, y, z]     lumen mask (trilinear)
  geom.json    grid, logical → patient-LPS rotation (columns = logical axes), FOV centre (LPS, m), encodings, frame time
Everything is resampled (trilinear, complex) to an isotropic grid with the same centre and axes as the RACLETTE volume.
Fortran-ordered arrays (x fastest), like the rest of data/hasty.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pyvista_zstd
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/hasty/external/raclette_data"
args = sys.argv[1:]
variant, subject = args[0], args[1]


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


res = float(opt("--res", "2.0")) * 1e-3
venc_sel = float(opt("--venc", "2.5"))
sd = DATA / variant / subject
frames = sorted(glob.glob(str(sd / "Data" / "7p_4DFlow_*.pv")))


def load(f):
    img = pyvista_zstd.read(f)
    nx, ny, nz = img.dimensions
    n = img["real"].shape[1]
    sig = (img["real"] + 1j * img["imag"]).reshape((nx, ny, nz, n), order="F")
    return img, sig


img0, _ = load(frames[0])
vencs = np.asarray(img0.field_data["vencs"]).ravel()
dirs = np.asarray(img0.field_data["enc_dirs"]).reshape(-1, 3)
mask0 = img0["mask"].reshape(img0.dimensions, order="F") > 0
fr = opt("--frame", "peak")
if fr == "peak":
    sp = []
    for f in frames:
        _, s = load(f)
        ph = np.angle(s[..., 1:] * np.conj(s[..., :1])) * (vencs[1:] / np.pi)
        sp.append(np.abs(ph).max(axis=-1)[mask0].mean())
    fr = int(np.argmax(sp))
fr = int(fr)
img, sig = load(frames[fr])
sel = [0] + [int(np.flatnonzero((np.abs(dirs[:, a]) > 0.5) & np.isclose(vencs, venc_sel))[0]) for a in range(3)]
obj = sig[..., sel]
sens = pyvista_zstd.read(str(sd / "Data" / "sensitivity.pv"))
nc = sens["real"].shape[1]
smaps = (sens["real"] + 1j * sens["imag"]).reshape(tuple(sens.dimensions) + (nc,), order="F")
mask = (img["mask"].reshape(img.dimensions, order="F") > 0).astype(np.float32)

# geometry: voxel index -> patient LPS (m), RACLETTE convention (see Scripts/dataHandler.py::_ijk_to_lps)
spacing = np.array(img.spacing)
fd = img.field_data
ijk_to_xyz = np.asarray(fd["ijk_to_xyz"], dtype=np.float64).reshape(4, 4)
xyz_to_raf = np.asarray(fd["xyz_to_raf"], dtype=np.float64).reshape(4, 4)
R = ijk_to_xyz[:3, :3] / np.linalg.norm(ijk_to_xyz[:3, :3], axis=0)
A = np.eye(4)
A[:3, :3] = R @ np.diag(spacing)
A[:3, 3] = ijk_to_xyz[:3, 3]
ijk_to_lps = xyz_to_raf @ A  # metres
dims0 = np.array(img.dimensions)
centre_idx = (dims0 - 1) / 2.0
centre_lps = (ijk_to_lps @ np.append(centre_idx, 1.0))[:3]
rot = ijk_to_lps[:3, :3] / spacing  # columns: unit logical axes in LPS
ext = dims0 * spacing
def smooth_even(n):
    """smallest even m ≥ n with prime factors ≤ 7 (FFT-friendly; the centre convention is i − N/2)"""
    m = max(n + n % 2, 2)
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 2


dims = np.array([smooth_even(int(round(e / res))) for e in ext])

# resample: new voxel i (logical position (i - N/2)·res relative to the centre) -> old fractional index
axes = [((np.arange(dims[a]) - dims[a] // 2) * res + centre_idx[a] * spacing[a]) / spacing[a] for a in range(3)]
G = np.meshgrid(*axes, indexing="ij")
coords = np.stack([g.ravel(order="F") for g in G])


def resample(v, mode="constant"):
    """trilinear; mode "constant" (zero outside the RACLETTE volume: the object) or "nearest" (edge-extended: coil maps,
    which do not end at the volume boundary — GNL moves true positions of edge voxels outside it)"""
    def one(c):
        r = map_coordinates(c.real, coords, order=1, mode=mode, cval=0.0)
        i = map_coordinates(c.imag, coords, order=1, mode=mode, cval=0.0) if np.iscomplexobj(c) else 0
        return (r + 1j * i).reshape(tuple(dims), order="F") if np.iscomplexobj(c) else r.reshape(tuple(dims), order="F")
    if v.ndim == 3:
        return one(v)
    return np.stack([one(v[..., k]) for k in range(v.shape[-1])], axis=-1)


o = resample(obj).astype(np.complex64)
s = resample(smaps, mode="nearest").astype(np.complex64)
m = resample(mask).astype(np.float32)
# logical centre convention of the reconstruction: voxel N/2 sits at the FOV centre
fov_centre = centre_lps + rot @ (np.zeros(3))
out = REPO / "data/hasty/external/raclette_cases" / f"{subject}_f{fr}"
out.mkdir(parents=True, exist_ok=True)
for name, arr in [("object", o), ("smaps", s), ("mask", m)]:
    np.save(out / f"{name}.npy", np.asfortranarray(arr))
geom = {
    "source": f"RACLETTE {variant}/{subject} frame {fr} (CC BY-NC 4.0, DOI 10.3929/ethz-c-000799752)",
    "frame": fr,
    "frame_time_s": float(np.asarray(fd["dtime"]).ravel()[0]),
    "dims": dims.tolist(),
    "res_m": res,
    "rot_logical_to_lps": rot.tolist(),
    "fov_centre_lps_m": centre_lps.tolist(),
    "orig_dims": dims0.tolist(),
    "orig_spacing_m": spacing.tolist(),
    "encodings": [{"venc": float(vencs[k]), "dir_lps": (xyz_to_raf[:3, :3] @ dirs[k]).tolist()} for k in sel],
    "n_coils": int(nc),
}
(out / "geom.json").write_text(json.dumps(geom, indent=1))
print(f"frame {fr}: {dims0.tolist()} @ {np.round(spacing * 1e3, 3).tolist()} mm -> {dims.tolist()} @ {res * 1e3:.1f} mm, centre LPS {np.round(centre_lps * 1e3, 1).tolist()} mm, rot\n{np.round(rot, 3)}\n-> {out}")
print(f"orthonormality |RᵀR - I| = {np.abs(rot.T @ rot - np.eye(3)).max():.1e}, det {np.linalg.det(rot):.3f}; lumen voxels {int((m > 0.5).sum())}, |object| p99 {np.percentile(np.abs(o[..., 0]), 99):.3f}")
