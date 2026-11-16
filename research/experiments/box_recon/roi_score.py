"""Coronary / cardiac-vein velocity scores of box reconstructions (lumen files on the extended index).

    rustlib/.venv/bin/python -I research/experiments/box_recon/roi_score.py <scan_dir> <box_recon_dir> <x_lumen.npy> [...]
        [--truth-vel truth_vel_logical.npy] [--roi coronary_roi.npz] [--bin B]

Lumen file: complex64 (n_ext, E, Fr) at <box_recon_dir>/lumen_idx_ext_f32.npy (framerecon --idx / recon_p2np
--save-lumen); recon frame f is cardiac frame f % 20, respiratory bin f // 20 (--bin B: score bin B only; default:
every bin separately). The truth (velocities, ROIs) is at one breathing state: compare like with like (R1 pools the
breathing states → the mean-state truth; a binned recon → the truth built at that bin's state).

Velocity decode (dual-VENC 7-point, the sim's phase −k·u): coarse u from the three high-VENC encodes (phase differences
to encode 0), the three low-VENC phase differences unwrapped against it, then least squares over all six.
Scores per branch group (coronary_roi.py: 1 left main, 2 LAD, 3 LCx, 4 RCA, 5 cardiac veins), over core voxels
(no partial volume) of each cardiac frame:
- voxel rms |v_rec − v_true| (cm/s) next to the truth's rms speed (the score of reporting zero);
- ROI-mean axial velocity (along the centreline tangent, distal positive) per frame, recon vs truth: mean |error|
  (cm/s), correlation over the 20 frames, ratio of the time means (bias); and the same for the truth's own time mean
  as the recon (a flat curve: what no dynamics scores).
Sign check: voxels of the LBM truth deep in fast blood (|u_pc| > 0.5 m/s, coverage 1 in all frames), decoded recon vs
truth_bias u_pc: slope / corr per axis (must be ≈ +1 / high on a working decode).
"""
import sys
from pathlib import Path

import numpy as np

A = sys.argv[1:]
opts = {}
for k in ("--truth-vel", "--roi", "--bin"):
    if k in A:
        i = A.index(k)
        opts[k] = A[i + 1]
        del A[i:i + 2]
SC, BR = Path(A[0]), Path(A[1])
names = A[2:]
tv = np.load(opts.get("--truth-vel", SC / "truth_vel_logical.npy"), mmap_mode="r")  # [X, Y, Z, 3, F]
roi = np.load(opts.get("--roi", BR / "coronary_roi.npz"))
ext = np.load(BR / "lumen_idx_ext_f32.npy").astype(np.int64)
order = np.argsort(ext)
X, Y, Z, _, NC = tv.shape
zb = np.load(SC / "truth_bias.npz")
K = zb["k"]  # [E, 3] rad per (m/s)
E = K.shape[0]
D = -(K[1:] - K[0])  # Δφ_e = −(k_e − k_0)·u
hi = np.argsort(np.abs(D).max(1))[:3]  # the three smallest |Δk|: high VENC
assert np.linalg.matrix_rank(D[hi]) == 3
Dpinv = np.linalg.pinv(D)


def decode(x):
    """x (n, E) complex → u (n, 3) m/s"""
    ph = np.angle(x[:, 1:] * np.conj(x[:, :1]))  # (n, 6)
    u0 = np.linalg.solve(D[hi], ph[:, hi].T).T
    pred = u0 @ D.T
    ph = pred + np.angle(np.exp(1j * (ph - pred)))
    return ph @ Dpinv.T


def pos(flat):
    """rows of the flat indices in the extended lumen file"""
    p = np.searchsorted(ext, flat, sorter=order)
    r = order[np.minimum(p, ext.size - 1)]
    assert np.all(ext[r] == flat), "ROI voxel missing from the extended index"
    return r


def xyz(flat):
    return flat % X, (flat // X) % Y, flat // (X * Y)


GN = {1: "LM", 2: "LAD", 3: "LCx", 4: "RCA", 5: "veins"}
# sign-check voxels: deep fast LBM blood
up = zb["u_pc"]  # [X, Y, Z, 3, F]
cov = zb["coverage"]
sel = np.nonzero((cov.min(-1) >= 0.999) & (np.sqrt((np.nan_to_num(up) ** 2).sum(3)).max(-1) > 0.5))
sflat = sel[0] + X * (sel[1] + Y * sel[2])
rng = np.random.default_rng(0)
pick = rng.choice(sflat.size, min(sflat.size, 20000), replace=False)
sflat = sflat[pick]
s_rows = pos(sflat)
s_true = up[sel[0][pick], sel[1][pick], sel[2][pick]]  # (n, 3, F)

for nm in names:
    x = np.load(nm, mmap_mode="r")
    Fr = x.shape[2]
    nb = Fr // NC
    bins = [int(opts["--bin"])] if "--bin" in opts else list(range(nb))
    print(f"== {nm}: {x.shape[0]} voxels, {E} encodes, {Fr} frames ({nb} respiratory bin(s))")
    for b in bins:
        # sign check
        sv = np.stack([decode(np.asarray(x[s_rows, :, b * NC + f])) for f in range(NC)], -1)  # (n, 3, F)
        line = []
        for a in range(3):
            t, r_ = s_true[:, a].ravel(), sv[:, a].ravel()
            line.append(f"{'xyz'[a]} slope {(t @ r_) / (t @ t):.2f} corr {np.corrcoef(t, r_)[0, 1]:.2f}")
        print(f"  bin {b}: sign check ({s_rows.size} fast LBM voxels): " + ", ".join(line))
        for kind in ("art", "vein"):
            idx, core, grp, tan = roi[f"idx_{kind}"], roi[f"core_{kind}"], roi[f"grp_{kind}"], roi[f"tan_{kind}"]
            rows = pos(idx)
            gx, gy, gz = xyz(idx)
            for g in sorted(set(np.unique(grp)) - {0}):
                err2, tru2, nvx = 0.0, 0.0, 0
                cur_r, cur_t = np.zeros(NC), np.zeros(NC)
                for f in range(NC):
                    m = core[:, f] & (grp[:, f] == g)
                    if not m.any():
                        cur_r[f] = cur_t[f] = np.nan
                        continue
                    vr = decode(np.asarray(x[rows[m], :, b * NC + f]))
                    vt = np.asarray(tv[gx[m], gy[m], gz[m], :, f])
                    err2 += ((vr - vt) ** 2).sum()
                    tru2 += (vt ** 2).sum()
                    nvx += m.sum()
                    tg = tan[m, f]
                    cur_r[f] = (vr * tg).sum(1).mean()
                    cur_t[f] = (vt * tg).sum(1).mean()
                ok = np.isfinite(cur_t)
                e_c = np.abs(cur_r[ok] - cur_t[ok]).mean() * 100
                e_flat = np.abs(cur_t[ok].mean() - cur_t[ok]).mean() * 100
                cc = np.corrcoef(cur_r[ok], cur_t[ok])[0, 1] if ok.sum() > 2 else np.nan
                ratio = cur_r[ok].mean() / cur_t[ok].mean() if cur_t[ok].mean() != 0 else np.nan
                print(f"  bin {b} {GN[g]:>5}: {nvx / ok.sum():6.0f} core voxels/frame; voxel rms err {np.sqrt(err2 / nvx) * 100:5.1f} cm/s "
                      f"(truth rms speed {np.sqrt(tru2 / nvx) * 100:4.1f}); axial ROI-mean: truth {cur_t[ok].mean() * 100:5.1f} cm/s "
                      f"(range {cur_t[ok].min() * 100:.1f}..{cur_t[ok].max() * 100:.1f}), |err| {e_c:4.1f} (flat-truth {e_flat:4.1f}), corr {cc:5.2f}, mean ratio {ratio:5.2f}")
