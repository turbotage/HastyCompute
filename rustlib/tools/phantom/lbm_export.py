"""Cardiac LBM run → flow4d value for the flow viewer (plotting_website/flow), same layout as phantom_4d's
viewer_flow.hv: vel [F, 3, Z, Y, X] m/s (phase-binned Eulerian mean, lab frame), mag [F, Z, Y, X] (blood coverage of
each cell in that bin: the moving heart as the anatomy channel), mask [Z, Y, X] (covered in any bin), spacing, dt,
venc, origin_m (cell centres).

    rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_export.py <run_dir> [--out <file.hv>] [--step k]
        [--frames F] [--publish [--server URL] [--name NAME]]

--step k: average k³ output cells (coverage-weighted) — the run's cells are out_factor lattice spacings (1 mm at
0.5 mm, out_factor 2). --frames F: resample the bins to F frames (linear, periodic). Cropped to the covered box + 2.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
RUN = Path(A[0])


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


meta = json.loads((RUN / "flow_bins.json").read_text())
nb, fct, dx, T = int(meta["bins"]), int(meta["out_factor"]), float(meta["dx"]), float(meta["period"])
c1 = 8 // fct
nt, cells = int(meta["tiles"]), int(meta["cells_per_tile"])
counts = np.asarray(meta["counts"], float)
ACC = int(meta.get("acc_stride", 10))
acc = np.memmap(RUN / "flow_bins.f32", np.float32, "r", shape=(nb, nt, cells, ACC))
HAS_WSS = ACC >= 15
txyz = np.asarray(meta["tile_xyz"], np.int64).reshape(-1, 3)
cc = np.arange(cells)
gx = (txyz[:, 0:1] * c1 + (cc % c1)[None]).ravel()
gy = (txyz[:, 1:2] * c1 + ((cc // c1) % c1)[None]).ravel()
gz = (txyz[:, 2:3] * c1 + (cc // (c1 * c1))[None]).ravel()
# covered box (any bin) + 2 cells
w_any = np.zeros(nt * cells, np.float32)
for b in range(nb):
    w_any = np.maximum(w_any, np.asarray(acc[b, :, :, 0]).ravel())
cov = w_any > 0
lo = np.array([gx[cov].min(), gy[cov].min(), gz[cov].min()]) - 2
hi = np.array([gx[cov].max(), gy[cov].max(), gz[cov].max()]) + 3
k = int(opt("--step", 1))
dims = ((hi - lo) + k - 1) // k  # x, y, z
nvox = int(np.prod(dims))
sel = cov
ix, iy, iz = (gx - lo[0]) // k, (gy - lo[1]) // k, (gz - lo[2]) // k
flat = ((iz * dims[1] + iy) * dims[0] + ix)
F = int(opt("--frames", nb))
print(f"{RUN}: {nb} bins, cells {fct * dx * 1e3:.1f} mm × step {k} → {(fct * dx * k) * 1e3:.1f} mm, dims {dims.tolist()} (x y z), {F} frames, "
      f"{F * 4 * nvox * 4 / 1e9:.2f} GB in float32")


def bin_fields(b):
    a = np.asarray(acc[b]).reshape(-1, ACC)
    wsum = np.bincount(flat[sel], weights=a[sel, 0], minlength=nvox)
    u = [np.bincount(flat[sel], weights=a[sel, 1 + c], minlength=nvox) for c in range(3)]
    U = np.stack([np.where(wsum > 0, x / np.maximum(wsum, 1e-30), 0.0) for x in u])
    C = wsum / max(counts[b], 1.0) / (fct**3 * k**3)
    U *= (C >= 0.25)[None]
    W = None
    if HAS_WSS:
        # wall shear stress (Pa): mean over the wall-node samples in the cell (vector, and the mean magnitude)
        ws = np.bincount(flat[sel], weights=a[sel, 10], minlength=nvox)
        W = np.stack([np.where(ws > 0, np.bincount(flat[sel], weights=a[sel, 11 + c], minlength=nvox) / np.maximum(ws, 1e-30), 0.0) for c in range(4)])
    return U.astype(np.float32), C.astype(np.float32), W


fb = [bin_fields(b) for b in range(nb)]
tf = (np.arange(F) + 0.5) / F * nb - 0.5  # frame times in bin units (bin b centred at b + ½)
vel = np.zeros((F, 3, dims[2], dims[1], dims[0]), np.float32)
wss = np.zeros((F, 4, dims[2], dims[1], dims[0]), np.float32) if HAS_WSS else None
mag = np.zeros((F, dims[2], dims[1], dims[0]), np.float32)
for f, s in enumerate(tf):
    b0 = int(np.floor(s))
    a = s - b0
    U0, C0, W0 = fb[b0 % nb]
    U1, C1, W1 = fb[(b0 + 1) % nb]
    if HAS_WSS:
        wss[f] = ((1 - a) * W0 + a * W1).reshape(4, dims[2], dims[1], dims[0])
    vel[f] = ((1 - a) * U0 + a * U1).reshape(3, dims[2], dims[1], dims[0])
    mag[f] = np.clip((1 - a) * C0 + a * C1, 0, 1).reshape(dims[2], dims[1], dims[0])
mask = (mag.max(0) >= 0.5).astype(np.uint8)
h = fct * dx * k
origin = np.asarray(meta["origin"], float) + (lo * fct + 0.5 * fct * k - 0.5) * dx  # centre of output voxel 0
vmax = float(np.sqrt((vel**2).sum(1)).max())
venc = round(float(np.ceil(np.quantile(np.sqrt((vel**2).sum(1))[:, mask > 0], 0.999) * 1.2 * 10) / 10), 1)
fv = {"kind": "flow4d", "vel": vel, "mag": mag, "spacing": [h] * 3, "dt": T / F, "venc": venc, "mask": mask, "origin_m": origin.tolist(),
      "source": f"hasty-cfd cardiac LBM {RUN}", "note": "phase-binned Eulerian mean velocity (lab frame); mag = blood coverage of the cell in that phase"}
if HAS_WSS:
    # native LBM wall shear stress (σ = −(τ0 − ½)/τ Π^neq at the first fluid node, ≈ ½ cell from the wall; −8 % on a
    # 6-cell-radius Poiseuille pipe): vector [F, 3, Z, Y, X] and the mean magnitude [F, Z, Y, X], Pa, in wall cells
    fv["wss"] = wss[:, :3]
    fv["wss_mag"] = wss[:, 3]
    fv["note"] += "; wss / wss_mag: wall shear stress (Pa) from the LBM stress tensor at wall nodes"
out = Path(opt("--out", RUN / f"viewer_flow_{h * 1e3:.1f}mm.hv"))
vf.write(out, fv)
print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB): max speed {vmax:.2f} m/s, venc {venc} m/s, mask {int(mask.sum())} voxels")
if "--publish" in A:
    from hastycompute.v2 import Client  # noqa: E402

    cli = Client(opt("--server"))
    name = opt("--name", f"LBM {RUN.parent.parent.name} {RUN.name} {h * 1e3:.1f} mm")
    i = cli.put(fv, name=name, metadata={"kind": "flow4d", "source": "hasty-cfd", "run": str(RUN)})
    cli.show(i, "flow")
    print("viewer:", cli.viewer_url(i, "flow"))
