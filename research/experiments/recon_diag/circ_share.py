"""How much the circulant metric's spectrum changes between encodes and between frames (CPU proxy, no GPU).

    circ_share.py <scan dir> [n_frames_checked = 4]

The circulant's density factor ρ(k) is the Fejér-smoothed sample density on the N grid (precond.rs). Proxy: sample
histogram on the N-grid frequencies (k·Δ·N/2π, rounded, periodic), smoothed with the Fejér kernel's main lobe
(Gaussian, σ = 0.5 bin). Compared per |k| shell (fraction of Nyquist):
- encode e pooled over all frames vs encode 0 pooled (median ratio of the normalised densities, p10–p90 spread);
- one set (encode 0, cardiac frame f) vs the pooled encode scaled by the sample-count ratio n_t / n_e, which is
  what recon_p2np's --circ-share encode uses (the scale goes into τ_t and α).
Cardiac frames from readouts.npy (the recon also splits each into 2 respiratory bins: another ×2 sparser).
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

scan = sys.argv[1]
nfc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
meta = json.load(open(os.path.join(scan, "meta.json")))
dims = np.array(meta["recon_dims"])
res = np.array(meta["res"])
ro = np.load(os.path.join(scan, "readouts.npy"), mmap_mode="r")
smp = np.load(os.path.join(scan, "samples.npy"), mmap_mode="r")
enc, frame = np.asarray(ro[:, 1]).astype(int), np.asarray(ro[:, 3]).astype(int)
ne = enc.max() + 1
nv = int(np.prod(dims))
fr = [np.fft.fftfreq(n) * 2 for n in dims]  # fraction of Nyquist per axis
KR = np.sqrt(fr[0][:, None, None] ** 2 + fr[1][None, :, None] ** 2 + fr[2][None, None, :] ** 2)
shells = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def density(rows):
    acc = np.zeros(nv)
    for i in range(0, len(rows), 256):
        k = np.asarray(smp[rows[i:i + 256], :, :3], np.float64).reshape(-1, 3)
        idx = [np.rint(k[:, a] * res[a] * dims[a] / (2 * np.pi)).astype(np.int64) % dims[a] for a in range(3)]
        acc += np.bincount(idx[0] * dims[1] * dims[2] + idx[1] * dims[2] + idx[2], minlength=nv)
    d = acc.reshape(dims)
    return ndimage.gaussian_filter(d, 0.5, mode="wrap"), len(rows) * smp.shape[1]


def cmp(a, b, label):
    out = []
    for lo, hi in shells:
        m = (KR >= lo) & (KR < hi) & (b > 0)
        r = a[m] / b[m]
        out.append(f"{lo:.2f}-{hi:.2f}: {np.median(r):.3f} [{np.percentile(r, 10):.2f}, {np.percentile(r, 90):.2f}]")
    print(f"  {label}: " + "  ".join(out))


pooled = {}
for e in range(ne):
    pooled[e] = density(np.flatnonzero(enc == e))
print(f"{scan}: {ne} encodes, grid {dims.tolist()}; ratio of normalised densities per |k| shell (median [p10, p90])")
d0, n0 = pooled[0]
for e in range(1, ne):
    de, n_e = pooled[e]
    cmp(de / n_e, d0 / n0, f"encode {e} vs 0 (pooled, {n_e / n0:.3f} samples)")
for f in range(0, 20, max(1, 20 // nfc)):
    rows = np.flatnonzero((enc == 0) & (frame == f))
    dt, nt = density(rows)
    cmp(dt / nt, d0 / n0, f"set enc 0 / cardiac frame {f:2d} ({len(rows)} readouts, n_t/n_e = {nt / n0:.4f}) vs pooled")
