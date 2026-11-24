"""Background (static-tissue) apparent velocity of finished runs: practice's recipe on the decoded velocities.
static = support voxels with magnitude above the 40th percentile and low temporal velocity std (bottom 30 %),
outside the dilated pc lumen; per axis: mean, a degree-1 polynomial fit over the static voxels, residual rms."""
import sys
import numpy as np
sys.path.insert(0, "python/lib")
from hastycompute import store
from scipy import ndimage

R = "data/hasty/phantom/factory_arms/ph0001/precond_compare/"
for run in sys.argv[1:]:
    d = R + run + "/"
    vel = np.asarray(store.load_any(d + store.glob_any(d, "vel_*")[0].name if False else d + [p for p in __import__("os").listdir(d) if p.startswith("vel_")][0]))
    sup = np.asarray(store.load_any(d + "mask_support.hv")).astype(bool)
    rec_name = [p for p in __import__("os").listdir(d) if p.startswith("recon_")][0]
    rec = store.load_any(d + rec_name)
    # magnitude of encode 0, time mean (recon (X, Y, Z, E, F))
    mag = np.abs(np.asarray(rec[..., 0, :])).mean(-1)
    pcv = np.asarray(store.load_any(d + "pc_vox.hv")).astype(np.int64).ravel()
    dims = vel.shape[:3]
    lum = np.zeros(int(np.prod(dims)), bool); lum[pcv[pcv < lum.size]] = True
    lum = ndimage.binary_dilation(lum.reshape(dims, order="F"), iterations=6)
    vm = vel.mean(-1)          # (X, Y, Z, 3) time mean
    vs = np.linalg.norm(vel.std(-1), axis=-1)
    cand = sup & ~lum & (mag > np.percentile(mag[sup], 40))
    static = cand & (vs < np.percentile(vs[cand], 30))
    idx = np.argwhere(static)
    X = np.c_[np.ones(len(idx)), (idx - np.array(dims) / 2) / np.array(dims)]
    out = []
    for a in range(3):
        y = vm[..., a][static]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        # polynomial range over the box: |c0| + Σ|c_i|/2
        out.append((y.mean(), np.abs(coef[1:]).sum() / 2, res.std(), np.sqrt((y ** 2).mean())))
    print(f"{run}: {static.sum()} static voxels; per axis (mean, poly half-range, residual rms, rms) m/s:",
          "; ".join(f"{a}: {m:+.4f} {p:.4f} {r:.4f} {q:.4f}" for a, (m, p, r, q) in zip("xyz", out)))
