"""Synthetic beating "heart" for testing the moving blood-pool surface (flow_bloodpool, plotting_website/flow):
a chamber whose radius follows the cycle, an outflow tube with a valve (a dark, thin disc that closes in diastole and
splits the blood pool in two), a ring of myocardium (mid magnitude), air outside. Magnitude has noise and speckle.
The truth radius per frame is printed so the surface can be checked.

    rustlib/.venv/bin/python -I rustlib/tools/flow/synth_heart.py [--server URL] [--n 56] [--frames 12] [--name NAME]
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute.v2 import Client, flow_value  # noqa: E402

A = sys.argv[1:]


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


n, F = int(opt("--n", 56)), int(opt("--frames", 12))
rng = np.random.default_rng(1)
z, y, x = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
cx = cy = cz = (n - 1) / 2
xc, yc, zc = x - (cx - 6), y - cy, z - cz
radius = [9.0 + 3.0 * np.sin(2 * np.pi * t / F) for t in range(F)]
mag = np.zeros((F, n, n, n), np.float32)
vel = np.zeros((F, 3, n, n, n), np.float32)
tube_r = 3.2
for t in range(F):
    r = np.sqrt(xc**2 + yc**2 + zc**2)
    chamber = r <= radius[t]
    wall = (r > radius[t]) & (r <= radius[t] + 3.5)
    xt = x - (cx - 6)
    rad = np.sqrt(yc**2 + zc**2)
    tube = (xt > 0) & (xt < 24) & (rad <= tube_r)
    tube_wall = (xt > 0) & (xt < 24) & (rad > tube_r) & (rad <= tube_r + 2.0)
    systole = t < F // 2
    valve = (np.abs(xt - radius[0] - 4) < 1.0) & (rad <= tube_r + 1.0) & (not systole)   # closed leaflet disc in diastole
    blood = (chamber | tube) & ~valve
    m = np.full((n, n, n), 0.02, np.float32)
    m[wall | tube_wall | valve] = 0.45
    m[blood] = 1.0
    mag[t] = np.maximum(m + rng.normal(0, 0.06, m.shape), 0)
    mag[t][(rng.random(m.shape) < 0.001) & (r > radius[t] + 12)] = 1.0
    s = (np.sin(np.pi * (t + 0.5) / (F / 2)) * 0.9) if systole else 0.0
    vel[t, 0][tube & ~valve] = s * (1 - (rad[tube & ~valve] / tube_r) ** 2) + 0.02
    vel[t, 0][chamber] = 0.35 * s * (1 - (r[chamber] / radius[t]) ** 2) + 0.01
    vel[t] += rng.normal(0, 0.004, vel[t].shape).astype(np.float32)
vol_ml = [(4 / 3 * np.pi * (rad_ * 2) ** 3) * 1e-3 for rad_ in radius]
print("truth chamber radius (voxels):", [round(float(r), 2) for r in radius])
print("truth chamber volume ml (2 mm voxels):", [round(float(v), 1) for v in vol_ml])
c = Client(opt("--server"))
i = c.put(flow_value(vel, mag, (2e-3,) * 3, 0.08, layout="c"), name=opt("--name", "synthetic heart"), metadata={"kind": "flow4d"})
print("id", i)
print(c.viewer_url(i, "flow"))
