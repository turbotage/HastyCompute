"""Extract the near-k-centre ADC samples of every readout (out start + in end) from the flow scan.

Out: samples 0..11 (|k| < 0.5 * 2π/FOV_max, t 1.600-1.622 ms after excitation).
In:  samples 3969..3989 (|k| < 0.5 * 2π/FOV_max, t 9.538-9.578 ms).
k = 0 estimates per readout and coil:
  exact : sample 0 (k = 0 exactly) and sample 3989 (|k| = 0.005 rad/m);
  fit   : least-squares s_n = a + g . k_n over the window (min-norm, rcond 1e-2), intercept a.
Clean data (data_clean.hv) and the same with recon-level noise added here (σ = meta noise_sigma per coil, Ψ-correlated,
σ/√2 per component, the simulator's / recon's noise model; own RNG, so not the recon's noise realisation).
Output (data/hasty/phantom/factory_arms/ph0001/kcentre/): k0.npz.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "python/lib")
from hastycompute.store import load_any  # noqa: E402

SCAN = Path("data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan")
OUT = Path("data/hasty/phantom/factory_arms/ph0001/kcentre")
IO = np.arange(0, 12)
II = np.arange(3969, 3990)
B = 256


def fit_k0(s, k, rcond=1e-2):
    """s [nr, n, nc], k [nr, n, 3] -> intercept [nr, nc] of s = a + g.k (per readout pinv)"""
    X = np.concatenate([np.ones(k.shape[:2] + (1,)), k], axis=2)
    # scale k columns to O(1) so rcond acts on comparable columns
    sc = np.maximum(np.abs(k).max(axis=1, keepdims=True), 1e-6)
    X[:, :, 1:] /= sc
    P = np.linalg.pinv(X, rcond=rcond)  # [nr, 4, n]
    return np.einsum("rn,rnc->rc", P[:, 0, :], s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SCAN / "meta.json").read_text())
    nr, nc = meta["n_readouts"], meta["n_coils"]
    sigma = meta["noise_sigma"]
    psi = np.array(meta["psi"], float)
    L = np.linalg.cholesky(psi)
    rng = np.random.default_rng(20261123)
    dc = load_any(SCAN / "data_clean.hv", lazy=True)
    sm = load_any(SCAN / "samples.hv", lazy=True)
    res = {n: np.zeros((nr, nc, 2), np.complex64) for n in ("exact", "fit", "fit_noisy", "exact_noisy")}
    kin = np.zeros((nr, len(II), 3), np.float32)
    kout = np.zeros((nr, len(IO), 3), np.float32)
    raw_keep = 2048
    raw_out = np.zeros((raw_keep, len(IO), nc), np.complex64)
    raw_in = np.zeros((raw_keep, len(II), nc), np.complex64)
    t_s = None
    t0 = time.time()
    for i0 in range(0, nr, B):
        i1 = min(nr, i0 + B)
        d = dc[i0:i1]
        so, si = np.ascontiguousarray(d[:, IO, :]), np.ascontiguousarray(d[:, II, :])
        del d
        ks = sm[i0:i1]
        ko, ki = ks[:, IO, :3].astype(np.float64), ks[:, II, :3].astype(np.float64)
        if t_s is None:
            t_s = (ks[0, IO, 3].astype(float), ks[0, II, 3].astype(float))
        kout[i0:i1], kin[i0:i1] = ko, ki
        if i0 < raw_keep:
            n = min(raw_keep, i1) - i0
            raw_out[i0:i0 + n], raw_in[i0:i0 + n] = so[:n], si[:n]

        def noisy(s):
            z = (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)) * (sigma / np.sqrt(2))
            return s + (z @ L.T).astype(np.complex64)

        no, ni = noisy(so), noisy(si)
        res["exact"][i0:i1, :, 0], res["exact"][i0:i1, :, 1] = so[:, 0], si[:, -1]
        res["exact_noisy"][i0:i1, :, 0], res["exact_noisy"][i0:i1, :, 1] = no[:, 0], ni[:, -1]
        res["fit"][i0:i1, :, 0], res["fit"][i0:i1, :, 1] = fit_k0(so, ko), fit_k0(si, ki)
        res["fit_noisy"][i0:i1, :, 0], res["fit_noisy"][i0:i1, :, 1] = fit_k0(no, ko), fit_k0(ni, ki)
        if (i0 // B) % 20 == 0:
            print(f"{i1}/{nr} {time.time() - t0:.0f} s", flush=True)
    np.savez(OUT / "k0.npz", **res, t_out=t_s[0], t_in=t_s[1], idx_out=IO, idx_in=II,
             kabs_out=np.linalg.norm(kout, axis=2).mean(0), kabs_in=np.linalg.norm(kin, axis=2).mean(0), sigma=sigma)
    np.savez(OUT / "raw_first2048.npz", out=raw_out, inn=raw_in)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
