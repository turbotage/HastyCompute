"""Smallest det(I + ∇u) of a breathing field at end-inspiration (r = 1) inside the body, and where.

    rustlib/.venv/bin/python -I jobs/tools/resp_det.py <resp_motion.hv> <tissue_2mm.hv>"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

R = load_any(sys.argv[1])
U = np.moveaxis(np.asarray(R["B"], float).sum(0), 0, -1)  # Σ bases (amplitudes folded in), [Z, Y, X, 3]
o, h = np.asarray(R["origin_m"], float), float(R["voxel_m"])
L = load_any(sys.argv[2])
lab, nm, o0, h0 = np.asarray(L["labels"]), list(L["tissues"]), np.asarray(L["origin_m"], float), float(L["voxel_m"])
J = np.zeros(U.shape[:3] + (3, 3))
for c in range(3):
    gz, gy, gx = np.gradient(U[..., c], h)
    J[..., c, 0], J[..., c, 1], J[..., c, 2] = gx, gy, gz
D = np.linalg.det(np.eye(3) + J)
Zc, Yc, Xc = np.meshgrid(*(o[a] + (np.arange(U.shape[2 - a]) + 0.5) * h for a in (2, 1, 0)), indexing="ij")
i = np.floor((np.stack([Xc, Yc, Zc], -1) - o0) / h0).astype(int)
ok = (i >= 0).all(-1) & (i[..., 0] < lab.shape[2]) & (i[..., 1] < lab.shape[1]) & (i[..., 2] < lab.shape[0])
labs = np.full(D.shape, -1)
labs[ok] = lab[i[ok][:, 2], i[ok][:, 1], i[ok][:, 0]]
body = labs > 0
d = np.where(body, D, np.inf)
order = np.argsort(d, axis=None)[:6]
print(f"det(I + ∇u) at r = 1 inside the body: min {d.min():.3f}, p0.1 {np.percentile(D[body], 0.1):.3f}")
for f in order:
    z_, y_, x_ = np.unravel_index(f, D.shape)
    print(f"   det {D[z_, y_, x_]:6.3f} at {np.round(np.array([Xc[z_, y_, x_], Yc[z_, y_, x_], Zc[z_, y_, x_]]) * 1e3).astype(int)} mm, {nm[labs[z_, y_, x_]]}")
