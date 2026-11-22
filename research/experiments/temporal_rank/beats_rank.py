"""Beat-to-beat variability of the LBM chamber + great-vessel flow (research/temporal_rank.md, run 5).

    rustlib/.venv/bin/python beats_rank.py [run_lib|run_cor] [max_beats]

The beat library (lbm/<run>/beats, cardiac_lbm --beats) holds, per beat and phase bin (50 per cycle), the per-cell
(1 mm) fluid coverage w, mean velocity mu and velocity covariance S. The encode signal of a cell is
w exp(-i K_e . mu - K_e S K_e^T / 2) (the Gaussian intravoxel dephasing), averaged to 2 mm. Reported: the ensemble-mean
signal's rank, the stacked ensemble's rank, and the floor: each beat's distance from the ensemble mean (no basis over
the cardiac phase alone can go below it, since all beats share one phase).
"""
import json
import sys
import time

import numpy as np

from common import LEVELS, OUT, PH, decode, encode_matrix, k_at
from experiment import eig_desc

sys.path.insert(0, str(PH.parents[4] / "rustlib/tools/phantom"))
from lbm_beats import BeatLibrary  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "run_lib"
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 99


def main():
    t0 = time.time()
    L = BeatLibrary(PH / "lbm" / RUN)
    beats = L.beats[:NB]
    K = encode_matrix()
    ne = K.shape[0]
    nb = L.bins
    X, ids = [], None
    X2, Y2 = (L.dims[0] + 1) // 2, (L.dims[1] + 1) // 2
    for bi, c in enumerate(beats):
        D = L.read(c)
        if ids is None:  # the 2 mm voxel set: every voxel any bin of the first beat touches
            allid = []
            for b in range(nb):
                idx = D[b][0]
                allid.append(((L.gz[idx] - L.lo[2]) // 2 * Y2 + (L.gy[idx] - L.lo[1]) // 2) * X2 + (L.gx[idx] - L.lo[0]) // 2)
            ids = np.unique(np.concatenate(allid))
            print(f"{len(ids)} voxels of 2 mm, {nb} bins, beats {beats}", flush=True)
        xb = np.zeros((ne, len(ids), nb), np.complex64)
        for b in range(nb):
            idx, v, _ = D[b]
            vid = ((L.gz[idx] - L.lo[2]) // 2 * Y2 + (L.gy[idx] - L.lo[1]) // 2) * X2 + (L.gx[idx] - L.lo[0]) // 2
            j = np.searchsorted(ids, vid)
            ok = (j < len(ids)) & (ids[np.minimum(j, len(ids) - 1)] == vid)
            w, mu, S = v[ok, 0], v[ok, 1:4], v[ok, 4:10]
            for e in range(ne):
                k = K[e]
                quad = k[0] ** 2 * S[:, 0] + k[1] ** 2 * S[:, 1] + k[2] ** 2 * S[:, 2] + 2 * (k[0] * k[1] * S[:, 3] + k[0] * k[2] * S[:, 4] + k[1] * k[2] * S[:, 5])
                s = w * np.exp(-1j * (mu @ k) - 0.5 * np.maximum(quad, 0))
                xb[e, :, b] = (np.bincount(j[ok], s.real, len(ids)) + 1j * np.bincount(j[ok], s.imag, len(ids))) / 8
        X.append(xb)
        del D
        print(f"beat {c} {time.time() - t0:.0f} s", flush=True)
    X = np.stack(X)  # [J, 7, V, B]
    J = len(beats)
    Xm = X.mean(0)
    # occupancy: voxels mostly fluid on average
    occ = np.abs(Xm[0]).mean(1) / np.abs(Xm[0]).mean(1).max()
    res = {"run": RUN, "beats": beats, "bins": nb, "levels": LEVELS}
    G = sum(Xm[e].conj().T @ Xm[e] for e in range(ne)).astype(np.complex128)
    w, U = eig_desc(G)
    res["K_mean"] = k_at(w, LEVELS)
    res["spectrum_mean"] = (w / w[0]).tolist()
    Gs = np.zeros((J * nb, J * nb), np.complex128)  # Gram of the stacked ensemble [7, V, J B], block by block
    for e in range(ne):
        for j in range(J):
            for k in range(j, J):
                blk = X[j, e].conj().T @ X[k, e]
                Gs[j * nb:(j + 1) * nb, k * nb:(k + 1) * nb] += blk
                if k != j:
                    Gs[k * nb:(k + 1) * nb, j * nb:(j + 1) * nb] += blk.conj().T
    ws, Us = eig_desc(Gs)
    res["K_stacked"] = k_at(ws, LEVELS)
    res["spectrum_stacked"] = (ws / ws[0]).tolist()
    # floor: each beat vs the ensemble mean (relative Frobenius over all voxels and encodes), and in velocity
    fl = [float(np.sqrt((np.abs(X[j] - Xm) ** 2).sum() / (np.abs(Xm) ** 2).sum())) for j in range(J)]
    res["floor_rel_per_beat"] = fl
    res["floor_rel_mean"] = float(np.mean(fl))
    # floor in velocity: lumen voxels (mean coverage >= 0.5 of the 8 cells)
    lum = np.abs(Xm[0]).mean(1) >= 0.5
    vm = decode(np.moveaxis(Xm[:, lum], 0, -1))
    dvs = []
    for j in range(J):
        vj = decode(np.moveaxis(X[j][:, lum], 0, -1))
        dvs.append(np.linalg.norm(vj - vm, axis=-1).ravel() * 100)
    dv = np.concatenate(dvs)
    res["floor_dv_mean"], res["floor_dv_p50"], res["floor_dv_p99"] = float(dv.mean()), float(np.median(dv)), float(np.percentile(dv, 99))
    res["speed_mean_p99"] = float(np.percentile(np.linalg.norm(vm, axis=-1) * 100, 99))
    # how well the ensemble-mean SVD basis of K represents the mean and each beat (the floor shows as the plateau)
    rows = []
    for k in [4, 8, 12, 16, 20, 24, 32, 40, 50]:
        Phi = U[:, :k]
        em = np.sqrt(sum((np.abs(Xm[e] - (Xm[e] @ Phi) @ Phi.conj().T) ** 2).sum() for e in range(ne)) / (np.abs(Xm) ** 2).sum())
        eb = np.mean([np.sqrt(sum((np.abs(X[j][e] - (X[j][e] @ Phi) @ Phi.conj().T) ** 2).sum() for e in range(ne)) / (np.abs(X[j]) ** 2).sum()) for j in range(J)])
        rows.append(dict(K=k, rel_mean=float(em), rel_beats=float(eb)))
    res["rows"] = rows
    res["n_lumen"] = int(lum.sum())
    (OUT / f"beats_{RUN}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("spectrum")}, indent=1))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
