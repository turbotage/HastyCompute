"""Publish a reconstruction run (recon_p2np output directory) to the flow viewer: the simulator truth and every
finished reconstruction, each with the reconstructed reference-encode magnitude as anatomy.

    python -I show_run.py <run_dir> [--tags tag1,tag2] [--no-truth] [--error] [--server https://host:8443]

--error also publishes recon − truth per reconstruction (overlay "speed" then shows the error magnitude).

Needs a running hasty server (hasty-serve) or HASTY_SERVER. Prints the viewer URLs.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute.v2 import Client, flow_value  # noqa: E402

args = sys.argv[1:]
run = Path(args[0]).resolve()


def opt(n, d=None):
    return args[args.index(n) + 1] if n in args else d


info = json.loads((run / "run.json").read_text())
sim = Path(info["sim_dir"])
meta = json.loads((sim / "meta.json").read_text())
nf = int(info["n_frames"])
spacing = meta.get("res", [2e-3] * 3)
dt = float(meta.get("mean_rr", 1.0)) / nf
venc = info.get("venc_range", [info.get("venc")])[0] if info.get("venc_range") else info.get("venc")
c = Client(opt("--server"))
tags = opt("--tags").split(",") if opt("--tags") else [p.stem[len("vel_"):] for p in sorted(run.glob("vel_*.npy"))]
mag = None
tv = np.load(sim / "truth_vel_logical.npy", mmap_mode="r")
on_grid = tv.shape[:3] == tuple(meta["recon_dims"])
for tag in tags:
    vel = np.load(run / f"vel_{tag}.npy")  # Fortran [X, Y, Z, 3, F]
    rec = np.load(run / f"recon_{tag}.npy", mmap_mode="r")  # [X, Y, Z, E, F] complex
    mag = np.abs(rec[:, :, :, 0, :])
    i = c.put(flow_value(vel, mag, spacing, dt, venc), name=f"{run.name} {tag}", metadata={"kind": "flow4d", "run": str(run), "tag": tag})
    print(f"{tag}: {c.viewer_url(i, 'flow')}")
    if "--error" in args and on_grid:
        i = c.put(flow_value(vel - np.asarray(tv, np.float32), mag, spacing, dt, venc), name=f"{run.name} {tag} error", metadata={"kind": "flow4d", "run": str(run), "tag": tag, "error": True})
        print(f"{tag} error: {c.viewer_url(i, 'flow')}")
if "--no-truth" not in args:
    if on_grid:
        i = c.put(flow_value(np.asarray(tv, np.float32), mag, spacing, dt, venc), name=f"{sim.name} truth", metadata={"kind": "flow4d", "sim": str(sim)})
        c.show(i, "flow")
        print(f"truth: {c.viewer_url(i, 'flow')}")
    else:
        print(f"truth velocity {tv.shape[:3]} is not on the reconstruction grid {meta['recon_dims']}: skipped")
