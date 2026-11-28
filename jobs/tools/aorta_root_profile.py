"""Cardiac displacement along the aortic root / ascending aorta vs the literature fractions of the AV descent.

    rustlib/.venv/bin/python -I jobs/tools/aorta_root_profile.py <phantom_view_dir> [<label>] [--dav-mm 15.74]

Per arc length s from the aortic root (the aorta path of vessel_motion_check): peak cardiac displacement (r = 0, max over
the cycle relative to the reference) and its ratio to the AV descent; the axial stretch range of 2 mm segments. Targets
(fraction of the AV descent ≈ 15 mm): annulus 0.77 (CMR 11.6 mm), ostia ~15 mm 0.41 (CTA 6.2), mid-ascending 0.25 (3.8)."""
import sys
from pathlib import Path

import numpy as np

sys.argv = [sys.argv[0], sys.argv[1], str(Path(sys.argv[1]) / "_tmp_check")] + sys.argv[2:]
DAV = float(sys.argv[sys.argv.index("--dav-mm") + 1]) if "--dav-mm" in sys.argv else 15.74
import runpy  # noqa: E402

src = Path(__file__).with_name("vessel_motion_check.py").read_text()
src = src[: src.index('paths = {"aorta')]  # helpers only
ns = {"__file__": str(Path(__file__).with_name("vessel_motion_check.py")), "__name__": "helpers"}
exec(compile(src, "vessel_motion_check_helpers", "exec"), ns)
X, s = ns["resample"](ns["aorta_path"]())
m = ns["m"]
tc = np.arange(24) * m.T / 24
U = np.array([np.linalg.norm(m.position(X, t, 0.0) - X, axis=1) for t in tc]).max(0) * 1e3
ST = np.array([np.linalg.norm(m.position(X, t, 0.0)[2:] - m.position(X, t, 0.0)[:-2], axis=1) / np.linalg.norm(X[2:] - X[:-2], axis=1) for t in tc])
for sv in (0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 130):
    i = int(np.argmin(np.abs(s * 1e3 - sv)))
    j = min(max(i - 1, 0), ST.shape[1] - 1)
    print(f"s {sv:4d} mm: disp {U[i]:5.1f} mm = {U[i] / DAV:.2f} × AV descent; stretch {ST[:, max(j - 3, 0):j + 4].min():.2f}–{ST[:, max(j - 3, 0):j + 4].max():.2f}")
