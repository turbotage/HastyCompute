"""Which cardiac basis modes move the aorta: U(X, t) = Σ_k c_k(t) B_k(X) exactly at the stored samples, so B_k at each
centreline point is the least-squares fit of the 48 samples on the coefficients coef_t; the largest |c_k(t) B_k(X)| over
the cycle per mode along the aorta.
    rustlib/.venv/bin/python -I jobs/tools/aorta_motion_modes.py <phantom_dir>"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
sys.path.insert(0, str(REPO / "jobs/tools"))
from hastycompute import value_format as vf  # noqa: E402

P = Path(sys.argv[1])
sys.argv = [sys.argv[0], str(P), "/tmp/_unused"]  # the path helpers of vessel_motion_check (no outputs used)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("vmc_paths", REPO / "jobs/tools/vessel_motion_check.py")
src = (REPO / "jobs/tools/vessel_motion_check.py").read_text().split("paths = {")[0]
ns: dict = {"__file__": str(REPO / "jobs/tools/vessel_motion_check.py")}
exec(compile(src.replace("OUT.mkdir(parents=True, exist_ok=True)", ""), "vmc", "exec"), ns)
Xa, s = ns["resample"](ns["aorta_path"]())
c = vf.read(P / "motion/cardiac_motion.hv")
U = np.asarray(c["U_mm"], np.float32)  # [T, 3, Z, Y, X] mm
C = np.asarray(c["coef_t"], float)
bases = list(c["bases"])
o, h = np.asarray(c["origin_m"], float), float(c["voxel_m"])
from scipy.ndimage import map_coordinates  # noqa: E402

ic = ((Xa - o) / h - 0.5)[:, ::-1].T
Ut = np.stack([np.stack([map_coordinates(U[t, a], ic, order=1, mode="constant", cval=0.0) for a in range(3)], -1) for t in range(U.shape[0])])  # [T, n, 3]
Bk, *_ = np.linalg.lstsq(C, Ut.reshape(len(C), -1), rcond=None)  # [K, n·3]
Bk = Bk.reshape(len(bases), -1, 3)
res = Ut - np.einsum("tk,kna->tna", C, Bk)
print(f"fit residual max {np.abs(res).max():.3f} mm of max |U| {np.abs(Ut).max():.1f} mm")
cont = np.array([np.linalg.norm(C[:, k, None, None] * Bk[k][None], axis=-1).max(0) for k in range(len(bases))])  # [K, n] mm
print("arc (mm) | z (mm) | " + " | ".join(bases) + " | total max |U|")
for i in range(0, len(s), 15):
    print(f"{s[i] * 1e3:6.0f} | {Xa[i, 2] * 1e3:7.0f} | " + " | ".join(f"{cont[k, i]:5.1f}" for k in range(len(bases))) + f" | {np.linalg.norm(Ut[:, i], axis=-1).max():5.1f}")
