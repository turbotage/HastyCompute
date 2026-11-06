"""Truth of a `simphantom` scan on its reconstruction grid (what `recon_p2np` / `show_run.py` compare against).

    python -I sim_truth.py <phantom_dir> <sim_dir> [--resp mean|0]

The cardiac-binned reconstruction averages over breathing, so the truth is the gated phantom at the scan's mean
respiratory state (`--resp mean`, default) or at end-expiration (`--resp 0`). `phantom_4d.py` runs at half the
reconstruction voxel over the prescribed box (voxel centres aligned so that 2 × 2 × 2 fine voxels average to one
reconstruction voxel, index i ↦ centre + (i − n/2) res), `n_frames` cardiac frames.
Writes truth_vel_logical.npy (blood-mean velocity, [X, Y, Z, 3, F] Fortran order, m/s), truth_lumen_fraction.npy
(blood fraction incl. the heart chambers, frame-averaged), truth_magnitude.npy (spoiled GRE with the phantom's table
and textures, frame 0) into <sim_dir>.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
sys.path.insert(0, str(HERE))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
PH, SIM = Path(A[0]), Path(A[1])
resp = A[A.index("--resp") + 1] if "--resp" in A else "mean"
meta = json.loads((SIM / "meta.json").read_text())
SI = vf.read(SIM.parent / "sim_input.hv") if (SIM.parent / "sim_input.hv").exists() else None
center = np.array(meta["presc_center_subject"])
fov, res = np.array(meta["fov"]), float(meta["res"][0])
dims = np.array(meta["recon_dims"])
nf = int(meta["n_frames"])
r_state = 0.0
if resp == "mean" and SI is not None:
    rt = np.asarray(SI["breathing"]["r"])
    r_state = float(rt[: int(meta["scan_time"] / 1e-3)].mean())
fine = res / 2
lo = center - (dims // 2) * res - res / 2
hi = lo + dims * res
crop = ",".join(f"{x * 1e3:.4f}" for pair in zip(lo, hi) for x in pair)
out = SIM / "truth_fine"
subprocess.run([str(REPO / "rustlib/.venv/bin/python"), "-I", str(HERE / "phantom_4d.py"), str(PH), str(PH / "motion"), str(out), "--labels-src",
                str(REPO / "data/hasty/phantom/p0/tissue_1mm.hv"), "--vox", f"{fine * 1e3}", "--frames", str(nf), "--resp-state", f"{r_state}", "--crop", crop],
               check=True, stdout=subprocess.DEVNULL)
T = vf.read(out / "truth4d.hv")
vel = np.asarray(T["vel"]).astype(np.float32)  # [F, 3, Z, Y, X] voxel-mean (blood + tissue)
bl = np.asarray(T["blood"]).astype(np.float32) / 255
F_, _, Z, Y, X = vel.shape
assert (Z, Y, X) == (2 * dims[2], 2 * dims[1], 2 * dims[0]), (vel.shape, dims)


def box(a):
    return a.reshape(a.shape[:-3] + (Z // 2, 2, Y // 2, 2, X // 2, 2)).mean(axis=(-5, -3, -1))


isb = (bl >= 0.5).astype(np.float32)
bfrac = box(isb)  # [F, z, y, x]
vb = box(vel * isb[:, None]) / np.maximum(bfrac, 1e-6)[:, None]
vb = np.where(bfrac[:, None] >= 0.25, vb, 0).astype(np.float32)
v_xyz = np.ascontiguousarray(np.transpose(vb, (4, 3, 2, 1, 0)))  # [X, Y, Z, 3, F]
np.save(SIM / "truth_vel_logical.npy", np.asfortranarray(v_xyz))
np.save(SIM / "truth_lumen_fraction.npy", np.asfortranarray(np.transpose(bfrac.mean(0), (2, 1, 0)).astype(np.float32)))
meta["truth"] = {"resp_state": r_state, "fine_voxel_m": fine, "source": "phantom_4d gated"}
(SIM / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"truth on {dims.tolist()} at {res * 1e3:.2f} mm, {nf} frames, r = {r_state:.2f}; blood voxels {int((bfrac.mean(0) >= 0.5).sum())}")
