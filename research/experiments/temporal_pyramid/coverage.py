"""Temporal coverage of k-space by the yarn trajectory (research/temporal_pyramid.md §9, §10.3).

For each k-space cell (Cartesian cells of `cell` × 1/FOV per axis), each encode and each knot interval
(respiratory bin × cardiac interval), mark whether any ADC sample of that encode falls in the cell during the
interval. coverage(cell) = fraction of (encode, interval) pairs that are hit. It is reported per |k| shell (fraction
of k_max). Low coverage at a shell = temporal detail there is not determined per knot (Theorem 4) and the LOO /
residual-deconvolution estimator falls back to the mean there (§10.3).

Inputs: the ph0001 10-min scan schedule (readouts.npy: t_exc, enc, part, frame, cardiac phase, rotation 3×3 row-major;
resp.npy) and the base readout k(t) of data/hasty/seq/flow_thorax/k_adc.npy, rotated per readout. Output: a small
text table next to this file.
"""
import os, sys, time
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SCAN = os.path.join(REPO, "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan")
SEQ = os.path.join(REPO, "data/hasty/seq/flow_thorax")

def main(cell=4, nsub=2000):
    import json
    meta = json.load(open(os.path.join(SCAN, "meta.json")))
    fov = np.array(meta["fov"]); res = np.array(meta["res"])
    kmax = 0.5 / res  # 1/m per axis
    tab = np.load(os.path.join(SCAN, "readouts.npy"))
    resp = np.load(os.path.join(SCAN, "resp.npy"))[:, 0]
    enc = tab[:, 1].astype(int); card = tab[:, 4] % 1.0
    rot = tab[:, 5:14].reshape(-1, 3, 3)
    kb = np.load(os.path.join(SEQ, "k_adc.npy"))
    kb = kb[np.linspace(0, len(kb) - 1, nsub).astype(int)]  # geometry only: subsample the ADC curve
    dk = cell / fov
    ncell = np.ceil(2 * kmax / dk).astype(int) + 1
    ne = enc.max() + 1
    nro = len(tab)
    # cells hit by every readout (unique per readout)
    t0 = time.time()
    ro_cells = []
    for i0 in range(0, nro, 2000):
        k = np.einsum("rab,sb->rsa", rot[i0:i0 + 2000], kb)  # (r, s, 3) physical k
        idx = np.floor((k + kmax) / dk).astype(int)
        idx = np.clip(idx, 0, ncell - 1)
        lin = (idx[..., 0] * ncell[1] + idx[..., 1]) * ncell[2] + idx[..., 2]
        for r in range(lin.shape[0]):
            ro_cells.append(np.unique(lin[r]))
    print(f"cells per readout: median {np.median([len(c) for c in ro_cells]):.0f} of {ncell.prod()} ({time.time() - t0:.0f}s)")
    # shell of every cell
    g = np.stack(np.meshgrid(*[(np.arange(n) + 0.5) * d - km for n, d, km in zip(ncell, dk, kmax)], indexing="ij"), -1).reshape(-1, 3)
    shell = np.linalg.norm(g / kmax, axis=1)  # fraction of k_max (anisotropic box normalised per axis)
    ever = np.zeros(ncell.prod(), bool)
    for c in ro_cells:
        ever[c] = True
    lens = np.array([len(c) for c in ro_cells])
    allc = np.concatenate(ro_cells)
    ro_of = np.repeat(np.arange(nro), lens)
    # respiratory bins by amplitude quantiles (equal counts)
    def rbins(R):
        if R == 1:
            return np.zeros(nro, int)
        q = np.quantile(resp, np.linspace(0, 1, R + 1)[1:-1])
        return np.searchsorted(q, resp)
    shells = [(0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    configs = [(1, 6), (1, 12), (1, 24), (1, 48), (4, 8), (4, 16), (4, 24), (4, 32), (4, 48), (6, 12), (6, 24), (12, 24)]
    lines = [f"yarn coverage: ph0001 sim_flow_10min, {nro} readouts, {ne} encodes; cells {cell}/FOV per axis "
             f"({ncell.tolist()}), base readout subsampled to {nsub} points; coverage = fraction of (encode, resp bin x "
             f"cardiac interval) pairs with >= 1 sample in the cell; shells in |k|/k_max (per-axis normalised)",
             "resp x card | " + " | ".join(f"{a:.2f}-{b:.2f}" for a, b in shells) + "   (mean coverage; in brackets: fraction of cells with coverage < 0.5)"]
    for R, K in configs:
        rb = rbins(R)
        interval = rb * K + np.minimum((card * K).astype(int), K - 1)
        NI = R * K
        key = (enc[ro_of] * NI + interval[ro_of]).astype(np.int64) * ncell.prod() + allc
        key = np.unique(key)
        cov = np.bincount(key % ncell.prod(), minlength=ncell.prod()) / (ne * NI)
        cells_row = []
        for a, b in shells:
            m = ever & (shell >= a) & (shell < b)
            cells_row.append(f"{cov[m].mean():.2f} ({(cov[m] < 0.5).mean():.2f})" if m.any() else "–")
        lines.append(f"{R:2d} x {K:2d}     | " + " | ".join(cells_row))
        print(lines[-1]); sys.stdout.flush()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"coverage_cell{cell}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main(cell=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
