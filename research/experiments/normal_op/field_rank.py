"""Q2: how many time segments the rate map needs, and how much polynomial absorption (shim-like, degree <= 4) helps.

Matrix E[v, tau] = exp(-(i omega(v) + R2*(v)) (tau - tau_c)) over object voxels v and readout times tau of the
yarn ADC (1.6 ... 9.58 ms); L(eps) = smallest rank with relative Frobenius error <= eps (SVD = optimal Tucker-side
segmentation; also the segment count of the Toeplitz form with tau-pointwise segment weights).
Fessler's diagonal 'L approach' segments the pair factor exp(i (omega(v) - omega(v')) tau): L'(eps) from the SVD
of exp(i d tau) over the pair-difference distribution d (sampled pairs).
Regions: whole torso FOV object, and a 160 mm cube centred on the heart (blood centroid). Field: field_ppm_recon
(the sim's B0 map, 2 mm grid), R2*: truth_r2s (as a stand-in for an estimated R2* map).

run: systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 rustlib/.venv/bin/python -I \
     research/experiments/normal_op/field_rank.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

GAMMA_HZ = 42.577e6
TOLS = (1e-2, 1e-3, 1e-4)
DEGS = (-1, 1, 2, 4) if len(sys.argv) < 2 else tuple(int(a) for a in sys.argv[1].split(","))


def ranks(E):
    s = np.linalg.svd(E, compute_uv=False)
    tail = np.sqrt(np.maximum(np.cumsum((s ** 2)[::-1])[::-1] / np.sum(s ** 2), 0))
    tail = np.append(tail, 0)
    return {f"{t:g}": int(np.argmax(tail <= t)) for t in TOLS}


def polyfit_residual(f, pos, wts, deg):
    if deg < 0:
        return f
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            for k in range(deg + 1 - i - j):
                cols.append(pos[:, 0] ** i * pos[:, 1] ** j * pos[:, 2] ** k)
    A = np.stack(cols, 1)
    sw = np.sqrt(wts)
    c, *_ = np.linalg.lstsq(A * sw[:, None], f * sw, rcond=None)
    return f - A @ c


def main():
    S = common.SCAN
    meta = json.load(open(os.path.join(S, "meta.json")))
    b0, res = meta["b0"], np.array(meta["res"])
    fppm = np.load(os.path.join(S, "field_ppm_recon.npy")).astype(np.float64)
    r2s = np.load(os.path.join(S, "truth_r2s.npy")).astype(np.float64)
    pd = np.load(os.path.join(S, "truth_pd_water.npy")) + np.load(os.path.join(S, "truth_pd_fat.npy"))
    lum = np.load(os.path.join(S, "truth_lumen_fraction.npy"))
    obj = pd > 0.05 * pd.max()
    dims = np.array(fppm.shape)
    grid = np.stack(np.meshgrid(*[(np.arange(n) - n / 2) * r for n, r in zip(dims, res)], indexing="ij"), -1)
    heart = (grid * lum[..., None]).reshape(-1, 3).sum(0) / lum.sum()
    omega = 2 * np.pi * GAMMA_HZ * b0 * 1e-6 * fppm
    tau = np.linspace(1.6e-3, 1.6e-3 + 7.98e-3, 400)
    tc = tau.mean()
    rng = np.random.default_rng(0)
    out = {"heart_centre_mm": (heart * 1e3).tolist(), "tau_ms": [tau[0] * 1e3, tau[-1] * 1e3]}
    regions = {"torso": obj, "box160": obj & np.all(np.abs(grid - heart) <= 0.08, -1)}
    for name, m in regions.items():
        v = np.nonzero(m.ravel())[0]
        v = rng.choice(v, size=min(25000, v.size), replace=False)
        pos = grid.reshape(-1, 3)[v]
        w = pd.ravel()[v].astype(np.float64)
        om, r2 = omega.ravel()[v], r2s.ravel()[v]
        reg = {"voxels": int(m.sum()), "omega_hz_1_99": (np.percentile(om, [1, 99]) / 2 / np.pi).tolist(),
               "r2s_1_99": np.percentile(r2, [1, 50, 99]).tolist(),
               "omega_hz_abs_pct_50_90_99": (np.percentile(np.abs(om), [50, 90, 99]) / 2 / np.pi).tolist()}
        for deg in DEGS:
            omr = polyfit_residual(om, pos, w, deg)
            key = "raw" if deg < 0 else f"minus_deg{deg}"
            E = np.exp(-1j * np.outer(omr, tau - tc))
            Er = E * np.exp(-np.outer(r2, tau - tau[0]))
            Er /= np.linalg.norm(Er, axis=1, keepdims=True)          # per-voxel relative error (unit rows)
            d = omr[rng.integers(0, omr.size, 25000)] - omr[rng.integers(0, omr.size, 25000)]
            Ep = np.exp(1j * np.outer(d, tau - tc))
            reg[key] = {"hz_1_99": (np.percentile(omr, [1, 99]) / 2 / np.pi).tolist(),
                        "L_b0": ranks(E), "L_b0_r2s": ranks(Er), "Lprime_pair_fessler": ranks(Ep)}
            print(name, key, reg[key], flush=True)
        out[name] = reg
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "field_rank.json" if len(sys.argv) < 2 else f"field_rank_{sys.argv[1]}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
