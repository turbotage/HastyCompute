"""Per-frame velocity phases of a reconstruction against the ideal per-frame phase-contrast signal of the sim.

    frame_fit.py <scan dir> <recon npy (nx, ny, nz, ne, F) | state_*.hsk[.tmp]> [...]

Truth: truth_pc_frames.npz (bias_truth.py --signal-frames), C [n, E, 20] per cardiac frame. Recon frame f has cardiac
frame f % 20 (respiratory bin f // 20). Voxels: blood in that cardiac frame (LBM coverage >= 0.99). Per depth band:
slope / corr of arg(x_e x̄_0) against arg(C_e C̄_0) (pairs with |pred| < π/2), and the rms of the wrapped phase error
(rad, all pairs). Also systole (cardiac 2-6) vs diastole (9-17), and the magnitude spread across encodes in static
tissue (>= 5 voxels from the lumen, bright) as the reconstruction-error yardstick.

The pooled slope / corr credit the spatial pattern of the time-mean flow too (a time-averaged image scores > 0). The
"dynamics" lines remove each voxel's circular temporal mean from both sides first (voxels blood in all 20 cardiac
frames, per respiratory bin), so they score only the cardiac variation. "ROI dynamics": the same after averaging the
phases over 8 mm blocks (4³ voxels, >= 16 lumen voxels each), as a practitioner reads ROI flow curves. "oracle mean" = the truth's own temporal mean
as the reconstruction (a perfect time-averaged image): its pooled score is what no dynamics at all already gets.
"""
import os
import sys

import numpy as np
from scipy import ndimage

scan = sys.argv[1]
names = sys.argv[2:]
z = np.load(os.path.join(scan, "truth_pc_frames.npz"))
idx, C, cov = z["idx"], z["c"], z["cov"]
E, NC = C.shape[1], C.shape[2]
lum = np.load(os.path.join(scan, "truth_lumen_fraction.npy")) > 0.5
depth = ndimage.distance_transform_edt(lum).ravel(order="F")[idx]
pred = np.angle(C[:, 1:, :] * np.conj(C[:, :1, :]))  # (n, E-1, 20)
far = ndimage.distance_transform_edt(~lum) >= 5
bands = [(0, 2), (2, 3), (3, 5), (5, 99)]


def stats(a, b):
    ok = np.abs(a) < np.pi / 2
    s = (a[ok] * b[ok]).sum() / (a[ok] ** 2).sum()
    c = np.corrcoef(a[ok], b[ok])[0, 1]
    w = np.angle(np.exp(1j * (b - a)))
    return s, c, np.sqrt(np.mean(w ** 2))


def load(nm):
    """recon npy (nx, ny, nz, ne, F), or the iterate x (first stack) of a host-state raw checkpoint state_*.hsk[.tmp]:
    sets t = frame * ne + encode, one image each (F-order), memory-mapped (a partial .tmp file has x complete first)"""
    if ".hsk" not in nm:
        return np.load(nm, mmap_mode="r")
    import json
    with open(nm, "rb") as f:
        f.read(8)
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
    lens = hdr["lens"][0]
    nv = int(np.prod(lum.shape))
    assert all(l == nv for l in lens), "one image per set expected"
    ne = E
    nf = len(lens) // ne
    print(f"checkpoint {nm}: meta {hdr['meta']}, {len(lens)} sets = {nf} frames x {ne} encodes")
    m = np.memmap(nm, dtype=np.complex64, mode="r", offset=16 + n, shape=(nf, ne, nv))
    # (F, E, nv F-order) -> (nx, ny, nz, E, F)
    return m.reshape(nf, ne, *lum.shape[::-1]).transpose(4, 3, 2, 1, 0)


for nm in names:
    lumen_only = nm.endswith("_lumen.npy")  # framerecon output: (n, E, F) at the truth voxels, no tissue
    x = np.load(nm, mmap_mode="r") if lumen_only else load(nm)
    F = x.shape[2] if lumen_only else x.shape[4]
    rec = np.zeros((idx.size, E - 1, F), np.float32)
    mags = []
    if not lumen_only:
        m0 = np.abs(np.asarray(x[..., 0, 0]))
        tis = np.nonzero((far & (m0 > np.median(m0[m0 > 0]))).ravel(order="F"))[0]
        tis = np.random.default_rng(0).choice(tis, min(tis.size, 100000), replace=False)
    for f in range(F):
        if lumen_only:
            xs = np.asarray(x[:, :, f])
        else:
            xs = np.stack([np.asarray(x[..., e, f]).ravel(order="F")[idx] for e in range(E)], 1)
            a = np.stack([np.abs(np.asarray(x[..., e, f]).ravel(order="F")[tis]) for e in range(E)], 0)
            mags.append(a.std(0) / np.maximum(a.mean(0), 1e-30))
        rec[:, :, f] = np.angle(xs[:, 1:] * np.conj(xs[:, :1]))
    print(f"== {nm} ({F} frames)")
    for lo, hi in bands:
        A, B = [], []
        for f in range(F):
            c = f % NC
            m = (cov[:, c] >= 0.99) & (depth >= lo) & (depth < hi)
            A.append(pred[m, :, c].ravel())
            B.append(rec[m, :, f].ravel())
        s, cc, rms = stats(np.concatenate(A), np.concatenate(B))
        print(f"  depth {lo}-{hi}: slope {s:.3f} corr {cc:.3f} wrapped phase error rms {rms:.3f} rad")
    for name, cs in [("systole (2-6)", range(2, 7)), ("diastole (9-17)", range(9, 18))]:
        A, B = [], []
        for f in range(F):
            c = f % NC
            if c in cs:
                m = (cov[:, c] >= 0.99) & (depth >= 2)
                A.append(pred[m, :, c].ravel())
                B.append(rec[m, :, f].ravel())
        s, cc, rms = stats(np.concatenate(A), np.concatenate(B))
        print(f"  {name}, depth >= 2: slope {s:.3f} corr {cc:.3f} wrapped phase error rms {rms:.3f} rad")
    # dynamics: per-voxel circular temporal mean removed (voxels blood in all cardiac frames), per respiratory bin
    allb = cov.min(1) >= 0.99
    nb = F // NC

    def demean(ph):  # (n, E-1, NC) phases -> wrapped deviation from the circular mean over the last axis
        mu = np.angle(np.exp(1j * ph).mean(-1, keepdims=True))
        return np.angle(np.exp(1j * (ph - mu)))

    pd = demean(pred)
    for lo, hi in [(2, 99), (5, 99)]:
        m = allb & (depth >= lo) & (depth < hi)
        A = np.concatenate([pd[m].ravel() for _ in range(nb)])
        B = np.concatenate([demean(rec[m][:, :, b * NC:(b + 1) * NC]).ravel() for b in range(nb)])
        s, cc, rms = stats(A, B)
        print(f"  dynamics, depth {lo}-{hi} ({m.sum()} voxels): slope {s:.3f} corr {cc:.3f} wrapped error rms {rms:.3f} rad (truth dyn rms {np.sqrt(np.mean(pd[m] ** 2)):.3f})")
    # ROI dynamics: phases averaged over 4^3-voxel blocks of the deep lumen (>= 16 voxels), then de-meaned in time
    ijk = np.stack(np.unravel_index(idx, lum.shape, order="F"), 1) // 4
    for lo in [2, 5]:
        m = np.nonzero(allb & (depth >= lo))[0]
        _, bid, cnt = np.unique(ijk[m], axis=0, return_inverse=True, return_counts=True)
        bid = bid.ravel()
        keep = cnt[bid] >= 16
        m, bid = m[keep], bid[keep]
        _, bid = np.unique(bid, return_inverse=True)
        nbk = bid.max() + 1
        avg = lambda a: np.stack([np.bincount(bid, a[m, j], nbk) for j in range(a.shape[1])], 1) / np.bincount(bid, None, nbk)[:, None]
        Pa = np.stack([avg(pred[:, :, c]) for c in range(NC)], -1)
        Ra = np.stack([avg(rec[:, :, f]) for f in range(F)], -1)
        A = np.concatenate([demean(Pa).ravel() for _ in range(nb)])
        B = np.concatenate([demean(Ra[:, :, b * NC:(b + 1) * NC]).ravel() for b in range(nb)])
        s, cc, rms = stats(A, B)
        print(f"  ROI dynamics, depth >= {lo} ({nbk} blocks of 8 mm): slope {s:.3f} corr {cc:.3f} wrapped error rms {rms:.3f} rad (truth dyn rms {np.sqrt(np.mean(demean(Pa) ** 2)):.3f})")
    if not mags:
        continue
    mg = np.concatenate(mags)
    print(f"  static-tissue magnitude spread across encodes: median {np.median(mg) * 100:.2f} %, p90 {np.percentile(mg, 90) * 100:.2f} %")
