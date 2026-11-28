"""Publish the LBM truth velocity of a simulated scan to the hasty server and print the Bloch parcel viewer URL.

    python -I show_lbm.py [<scan_dir>] [--server https://host:8443] [--name text]

<scan_dir> holds truth_vel_logical.* ([X, Y, Z, 3, F] m/s on the logical grid, grid centre = isocentre), meta.json
(res_truth, mean_rr, n_frames) and optionally truth_lumen_fraction.* (vessel mask). Default: the 2 mm ph0001 10-min scan
(data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan). Needs a running hasty-serve (or HASTY_SERVER).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute.store import load_any, exists_any  # noqa: E402
from hastycompute.v2 import Client, flow_value  # noqa: E402

args = sys.argv[1:]


def opt(n, d=None):
    if n in args:
        i = args.index(n)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return d


server = opt("--server")
name = opt("--name")
scan = Path(args[0]) if args else REPO / "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan"
meta = json.loads((scan / "meta.json").read_text())
vel = np.asarray(load_any(scan / "truth_vel_logical.npy"), np.float32)
nf = vel.shape[-1]
res = meta.get("res_truth", meta.get("res", [2e-3] * 3))
dt = float(meta["mean_rr"]) / nf
mask = None
if exists_any(scan / "truth_lumen_fraction.npy"):
    lf = np.asarray(load_any(scan / "truth_lumen_fraction.npy"))
    if lf.shape == vel.shape[:3]:
        mask = (lf > 0.5).astype(np.uint8)
print(f"{scan.name}: vel {vel.shape}, spacing {res} m, {nf} frames of {dt * 1e3:.1f} ms, mask {'yes' if mask is not None else 'no'}")
c = Client(server)
i = c.put(flow_value(vel, None, res, dt, meta.get("venc"), mask), name=name or f"{scan.parent.name} LBM truth", metadata={"kind": "flow4d", "sim": str(scan)})
print("flow viewer :", c.viewer_url(i, "flow"))
print("bloch viewer:", c.viewer_url(i, "flow").replace("/flow/", "/bloch/"))
