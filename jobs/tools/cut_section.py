"""Domain cross-section at each cut plane (from the case sdf) vs the graph radius and the logged layer section."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
CASE, RUN = Path(sys.argv[1]), Path(sys.argv[2])
case = json.loads((CASE / "case.json").read_text())
sd = np.load(CASE / case["sdf"]["file"]) if (CASE / case["sdf"]["file"]).suffix == ".npy" else None
o = np.asarray(case["sdf"]["origin"], float); h = float(case["sdf"]["h"])
last = json.loads((RUN / "log.jsonl").read_text().splitlines()[-1])
sec = {c["name"]: c["section_cm2"] for c in last["cuts"]}
print("sdf", sd.shape, sd.dtype, float(sd.min()), float(sd.max()), "h", h)
rows = []
for c in case["cuts"]:
    cc, n, R = np.asarray(c["center"]), np.asarray(c["normal"]), float(c["radius"])
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    g = np.linspace(-2 * R, 2 * R, 161); dA = (g[1] - g[0]) ** 2
    A_, B_ = np.meshgrid(g, g, indexing="ij")
    out = {}
    for s_off in (-0.5e-3, -2.5e-3):
        P = cc + s_off * n + A_.ravel()[:, None] * e1 + B_.ravel()[:, None] * e2
        idx = ((P - o) / h - 0.5)[:, ::-1].T  # z, y, x voxel-centre coordinates
        v = map_coordinates(sd, idx, order=1, mode="nearest")
        lat = np.hypot(A_.ravel(), B_.ravel())
        inside = v < 0
        out[s_off] = (inside & (lat < 1.1 * R)).sum() * dA, (inside & (lat < 2 * R)).sum() * dA
    rows.append((c["name"][:34], R * 1e3, np.pi * R * R * 1e4, out[-0.5e-3][0] * 1e4, out[-0.5e-3][1] * 1e4, out[-2.5e-3][1] * 1e4, sec.get(c["name"], np.nan)))
print(f"{'cut':34s} {'R mm':>5s} {'πR²':>6s} {'in1.1R':>7s} {'in2R':>6s} {'in2R@2.5':>8s} {'layer':>6s}  (cm²)")
for r in rows:
    print(f"{r[0]:34s} {r[1]:5.1f} {r[2]:6.3f} {r[3]:7.3f} {r[4]:6.3f} {r[5]:8.3f} {r[6]:6.3f}")
