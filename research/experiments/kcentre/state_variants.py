"""Per-state k-centre predictions with different weights, to test whether the predicted Δf-vs-breathing slope comes from
the static steady-state weight map (template positions) rather than from the field: W = 1 (PD only, no slab) vs
W = slab only (uniform T1) vs the full SS weight (pred.npz P0)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "research/experiments/kcentre"); sys.path.insert(0, "python/lib")
import predict as P
from hastycompute.store import load_any
meta = json.loads((P.SCAN / "meta.json").read_text())
dims = np.array(meta["recon_dims"]); res = meta["res"][0]
center = np.array(meta["presc_center_subject"]); iso = np.array(meta["iso_subject"])
mask = np.load(P.BC / "mask.npy"); lin = np.flatnonzero(mask.ravel(order="F"))
ijk = np.stack(np.unravel_index(lin, tuple(dims), order="F"), 1)
xs = (ijk - dims // 2) * res + center - iso
sm = load_any(P.SCAN / "smaps_true.hv"); S = np.ascontiguousarray(sm.reshape(-1, sm.shape[-1], order="F")[lin]).astype(np.complex64); del sm
zz, prof, hzm = P.slab_profile(meta)
F = np.array([P.fatF(t) for t in P.TAU])
mm = {n: np.load(P.BC / f"{n}.npy", mmap_mode="r") for n in ("pd_water", "pd_fat", "r_water", "r_fat", "omega_hz")}
cl = lambda a: np.where(np.isfinite(a), np.asarray(a, np.float64), 0.0)
e1 = np.exp(-meta["tr"] / 1.0)  # uniform T1 = 1 s
out = {"pd": np.zeros((120, 32, 2), np.complex64), "slab": np.zeros((120, 32, 2), np.complex64), "pd_water": np.zeros((120, 32, 2), np.complex64)}
for s in range(120):
    pw, pf, rw, rf, om = (cl(mm[n][s]) for n in ("pd_water", "pd_fat", "r_water", "r_fat", "omega_hz"))
    aw = np.interp(xs[:, 2] + om / hzm, zz, prof, 0, 0); af = np.interp(xs[:, 2] + (om - 383.2) / hzm, zz, prof, 0, 0)
    for name, Ww, Wf, fat in (("pd", 1.0, 1.0, 1), ("slab", P.ss(aw, e1), P.ss(af, e1), 1), ("pd_water", 1.0, 1.0, 0)):
        b = np.stack([(Ww * pw * np.exp(-rw * t) + fat * Wf * pf * F[j] * np.exp(-rf * t)) * np.exp(-2j * np.pi * om * t) for j, t in enumerate(P.TAU)])
        out[name][s] = (b.astype(np.complex64) @ S).T
np.savez(P.OUT / "pred_state_variants.npz", **out)
rst = np.array(json.loads((P.BC / "meta.json").read_text())["states"]["r"])
p0 = np.load(P.OUT / "pred.npz")["P0"]
for name, A in list(out.items()) + [("ss (pred.npz)", p0)]:
    A = A.reshape(20, 6, 32, 2)
    rho = (A[..., 1] * np.conj(A[..., 0])).sum((0, 2)) / (np.abs(A[..., 0]) ** 2).sum((0, 2))
    df = -np.angle(rho) / (2 * np.pi * (P.TAU[1] - P.TAU[0]))
    print(f"{name:16s} implied Δf per resp state r={rst}: {np.round(df, 2)}  slope {np.polyfit(rst, df, 1)[0]:.2f} Hz/r")
