"""Distance between the final estimates of two runs (research/solver_choice.md §5, V1 / V2): velocity on the scored blood
voxels (pc_vox) and on the support, and the relative complex-image difference. Scoring only.
    rustlib/.venv/bin/python -I research/experiments/precond_compare/compare_runs.py <run dir a> <run dir b>
"""
import os
import sys

import numpy as np

sys.path.insert(0, "python/lib")
from hastycompute import store  # noqa: E402


def one(d, prefix):
    names = [p for p in os.listdir(d) if p.startswith(prefix)]
    assert len(names) == 1, f"{d}: {names}"
    return np.asarray(store.load_any(os.path.join(d, names[0])))


a, b = sys.argv[1], sys.argv[2]
va, vb = one(a, "vel_"), one(b, "vel_")
ra, rb = one(a, "recon_"), one(b, "recon_")
sup = np.asarray(store.load_any(os.path.join(a, "mask_support.hv"))).astype(bool)
pcv = np.asarray(store.load_any(os.path.join(a, "pc_vox.hv"))).astype(np.int64).ravel()
dims = va.shape[:3]
lum = np.zeros(int(np.prod(dims)), bool)
lum[pcv[pcv < lum.size]] = True
lum = lum.reshape(dims, order="F")
dv = va - vb  # (X, Y, Z, 3, F)
rms = lambda m: float(np.sqrt((np.linalg.norm(dv, axis=3) ** 2)[m].mean()))  # noqa: E731
dimg = float(np.linalg.norm(ra - rb) / np.linalg.norm(rb))
print(f"{os.path.basename(a)} vs {os.path.basename(b)}: velocity difference RMS {rms(lum):.4f} m/s on {lum.sum()} pc voxels, "
      f"{rms(sup):.4f} m/s on {sup.sum()} support voxels; relative image difference {dimg:.3e}")
