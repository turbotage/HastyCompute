"""Crop a window out of the LBM phantom's flow_bins.f32 (data/hasty/phantom/.../lbm/run_*/flow_bins.{json,f32}) into dense arrays for validating mesh_wss against the
LBM's own wall shear stress.

usage: python -I lbm_crop.py <run dir> <out dir> [--tiles 14] [--bins 10]

Format (flow_bins.json): f32 [bins][tiles][cells=64][stride=30]; a tile is 8³ lattice nodes = 4³ cells (out_factor 2, dx = 0.15 mm → 0.3 mm cells), cell (cz, cy, cx)
flattened z-major; per cell: Σw, Σw·u (3), Σw·u uᵀ (6), then at wall nodes Σw_wall, Σ τ_w (3, Pa), Σ|τ_w| (Pa). Reads the densest window of `tiles`³ tiles and `bins` evenly
spaced bins only (a few hundred MB at most). Writes vel.npy [F,3,Z,Y,X] (m/s), mask.npy [Z,Y,X] (cells with fluid), tau.npy [Z,Y,X] (time-mean wall |τ| in Pa, 0 away
from walls), meta.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

run, out = Path(sys.argv[1]), Path(sys.argv[2])
arg = lambda n, d: int(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d
nt, nb = arg("--tiles", 14), arg("--bins", 10)
j = json.load(open(run / "flow_bins.json"))
bins, tiles, cells, stride = j["bins"], j["tiles"], j["cells_per_tile"], j["acc_stride"]
txyz = np.array(j["tile_xyz"], dtype=np.int64).reshape(-1, 3)
assert len(txyz) == tiles, (len(txyz), tiles)
# the densest window of nt³ tiles (integral image over the tile grid)
lo, hi = txyz.min(0), txyz.max(0)
dims = hi - lo + 1
occ = np.zeros(dims, dtype=np.int32)
occ[tuple((txyz - lo).T)] = 1
ii = np.pad(occ.cumsum(0).cumsum(1).cumsum(2), ((1, 0), (1, 0), (1, 0)))
best, at = -1, None
for x in range(max(1, dims[0] - nt + 1)):
    for y in range(max(1, dims[1] - nt + 1)):
        for z in range(max(1, dims[2] - nt + 1)):
            x1, y1, z1 = min(x + nt, dims[0]), min(y + nt, dims[1]), min(z + nt, dims[2])
            c = ii[x1, y1, z1] - ii[x, y1, z1] - ii[x1, y, z1] - ii[x1, y1, z] + ii[x, y, z1] + ii[x, y1, z] + ii[x1, y, z] - ii[x, y, z]
            if c > best:
                best, at = c, (x, y, z)
w0 = lo + np.array(at)
sel = np.where(((txyz >= w0) & (txyz < w0 + nt)).all(1))[0]
print(f"window of {nt}³ tiles at {w0.tolist()} holds {len(sel)} tiles", file=sys.stderr)
mm = np.memmap(run / "flow_bins.f32", dtype=np.float32, mode="r", shape=(bins, tiles, cells, stride))
use = np.linspace(0, bins - 1, nb).astype(int)
N = nt * 4
vel = np.zeros((nb, 3, N, N, N), np.float32)
wsum = np.zeros((nb, N, N, N), np.float32)
tau_sum = np.zeros((N, N, N), np.float64)
tau_cnt = np.zeros((N, N, N), np.float64)
cz, cy, cx = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij")
cz, cy, cx = cz.ravel(), cy.ravel(), cx.ravel()
tl = txyz[sel] - w0
gx = (tl[:, 0, None] * 4 + cx[None]).ravel()
gy = (tl[:, 1, None] * 4 + cy[None]).ravel()
gz = (tl[:, 2, None] * 4 + cz[None]).ravel()
for k, b in enumerate(use):
    a = np.asarray(mm[b][sel]).reshape(-1, stride)          # [tiles·cells, 30]
    w = a[:, 0]
    ok = w > 0
    for c in range(3):
        v = np.zeros_like(w)
        v[ok] = a[ok, 1 + c] / w[ok]
        vel[k, c, gz, gy, gx] = v
    wsum[k, gz, gy, gx] = w
    ww = a[:, 10]
    wk = ww > 0
    tau = np.zeros_like(ww)
    tau[wk] = a[wk, 14] / ww[wk]
    np.add.at(tau_sum, (gz[wk], gy[wk], gx[wk]), tau[wk])
    np.add.at(tau_cnt, (gz[wk], gy[wk], gx[wk]), 1.0)
mask = (wsum.max(0) > 0).astype(np.uint8)
tau = np.where(tau_cnt > 0, tau_sum / np.maximum(tau_cnt, 1), 0).astype(np.float32)
out.mkdir(parents=True, exist_ok=True)
np.save(out / "vel.npy", vel)
np.save(out / "mask.npy", mask)
np.save(out / "tau.npy", tau)
dt = j["period"] / bins * (bins - 1) / (nb - 1) if nb > 1 else j["period"]
json.dump({"spacing_m": j["dx"] * j["out_factor"], "dt": dt, "frames": nb, "period": j["period"], "dims": [N, N, N], "lumen_cells": int(mask.sum()), "wall_cells": int((tau > 0).sum()),
           "mean_wall_tau_pa": float(tau[tau > 0].mean()) if (tau > 0).any() else 0.0}, open(out / "meta.json", "w"), indent=1)
print(open(out / "meta.json").read())
