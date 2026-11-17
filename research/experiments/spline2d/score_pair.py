"""Encode 1 − 0 velocity-phase slope / corr of splinerecon outputs at each breathing state's own truth, depth bands
0–2 (vessel edges) and >= 5 (deep lumen), plus a dynamics line (temporal mean removed).
    python score_pair.py <file prefix> [...]    (files <prefix>_r{state}_it{k}_lumen.npy or .hv)"""
import sys, glob, os
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
try:  # .hv outputs (hasty_value::store) and .npy; plain np.load if the library is unavailable
    from hastycompute.store import load_any
except Exception:
    load_any = np.load
def lumen_files(pat):
    """*_lumen.npy and *_lumen.hv, one per stem (.hv preferred)"""
    got = {}
    for f in glob.glob(pat + ".npy") + glob.glob(pat + ".hv"):
        stem = f.rsplit(".", 1)[0]
        if stem not in got or f.endswith(".hv"):
            got[stem] = f
    return sorted(got.values())
S = "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/"
T = {"0.07": "truth_pc_frames_r0.07.npz", "0.181": "truth_pc_frames_r0.181.npz", "0.493": "truth_pc_frames.npz"}
lum = load_any(S + "truth_lumen_fraction.npy") > 0.5
dist = ndimage.distance_transform_edt(lum).ravel(order="F")
def st(a, b):
    ok = np.abs(a) < np.pi / 2
    return (a[ok] * b[ok]).sum() / (a[ok] ** 2).sum(), np.corrcoef(a[ok], b[ok])[0, 1]
for pre in sys.argv[1:]:
    for r, tf in T.items():
        for f in lumen_files(f"{pre}_r{r}_it*_lumen"):
            z = load_any(S + tf); idx, C, cov = z["idx"], z["c"], z["cov"]
            d = dist[idx]
            pred = np.angle(C[:, 1, :] * np.conj(C[:, 0, :])); x = load_any(f); rec = np.angle(x[:, 1, :] * np.conj(x[:, 0, :]))
            out = []
            for lo, hi in ((0, 2), (5, 99)):
                m = (cov >= 0.99) & (d[:, None] >= lo) & (d[:, None] < hi)
                out.append("depth %d-%d %.3f/%.3f" % ((lo, hi) + st(pred[m], rec[m])))
            allb = (cov.min(1) >= 0.99) & (d >= 2)
            dm = lambda p: np.angle(np.exp(1j * (p - np.angle(np.exp(1j * p).mean(-1, keepdims=True)))))
            out.append("dyn>=2 %.3f/%.3f" % st(dm(pred[allb]).ravel(), dm(rec[allb]).ravel()))
            print(os.path.basename(f), " | ".join(out), flush=True)
