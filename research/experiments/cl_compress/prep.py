"""Small per-scan tables for the coil x segment compression study (CPU, light).

Writes prep.npz next to this script:
  Wh       32x32   Psi^{-1/2} (meta psi), as hasty_recon::data::whitener
  cc_clean 32x32   eigenvectors (columns, descending) of the Gram of the whitened noise-free data of encode 0
                   (<= 2e5 samples: every readout stride, all ADC samples), eig_clean / eig_noisy the spectra
                   (noisy = clean + M sigma^2 I: the sim adds white noise of sigma per whitened coil)
  tau      3990    ADC times (s), tw_E per-ADC whitened data energy (coil sum, mean over 200 readouts of all encodes)
  kap      3990    |k| (rad/m) of the base readout, sigma (per whitened coil), wbar 3990x4 encode-0 mean concomitant
                   coefficients [z^2, x^2+y^2, xz, yz] (rad/m^2) per ADC sample (sched_nominal cols 9..12)
Run: rustlib/.venv/bin/python -I research/experiments/cl_compress/prep.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir())
sys.path.insert(0, str(ROOT / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

S = ROOT / "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan"
OUT = Path(__file__).resolve().parent / "prep.npz"


def main():
    meta = json.load(open(S / "meta.json"))
    psi = np.array(meta["psi"], np.float64)
    ev, U = np.linalg.eigh(psi)
    Wh = (U / np.sqrt(np.maximum(ev, 1e-12))) @ U.T
    sigma = float(meta["noise_sigma"])
    ro = np.load(S / "readouts.npy")
    enc = ro[:, 1].astype(int)
    r0 = np.flatnonzero(enc == 0)
    dcl = load_any(S / "data_clean.hv", lazy=True)
    sch = load_any(S / "sched_nominal.hv", lazy=True)
    smp = load_any(S / "samples.hv", lazy=True)
    na, nc = dcl.shape[1], dcl.shape[2]
    # coil Gram as splinerecon: encode-0 readouts with stride so that <= 2e5 samples enter
    stride = max(len(r0) * na // 200_000, 1)
    sub = r0[::stride]
    G = np.zeros((nc, nc), np.complex128)
    M = 0
    for r in sub:
        y = np.asarray(dcl[int(r)]).astype(np.complex128) @ Wh.T  # na x nc whitened
        G += y.T @ y.conj()
        M += y.shape[0]
    e, V = np.linalg.eigh(G)
    o = np.argsort(e)[::-1]
    e, V = np.maximum(e[o], 0), V[:, o]
    en = e + M * sigma ** 2
    # per-ADC whitened data energy, 200 readouts over the scan (as segment_basis/segeval.py)
    ros2 = np.linspace(0, dcl.shape[0] - 1, 200).astype(int)
    Em = np.zeros(na)
    ss = 0.0
    for r in ros2:
        y = np.asarray(dcl[int(r)]).astype(np.complex128)
        ss += (np.abs(y) ** 2).sum()
        Em += (np.abs(y @ Wh.T) ** 2).sum(1)
    Em /= len(ros2)
    k0 = np.asarray(smp[0], np.float64)
    tau = k0[:, 3]
    kap = np.linalg.norm(k0[:, :3], axis=1)
    # encode-0 mean concomitant coefficients (every 4th readout of encode 0)
    wb = np.zeros((na, 4))
    rr = r0[::4]
    for r in rr:
        wb += np.asarray(sch[int(r)], np.float64)[:, 9:13]
    wb /= len(rr)
    np.savez(OUT, Wh=Wh, cc_clean=V, eig_clean=e, eig_noisy=en, M_gram=M, tau=tau, kap=kap, tw_E=Em, sigma=sigma, wbar=wb,
             b0=meta["b0"], dims=np.array(meta["recon_dims"]), res=0.002)

    def count(eig, f):
        c = np.cumsum(eig) / eig.sum()
        return int(np.searchsorted(c, f) + 1)

    print(f"coil Gram: {M} samples from {len(sub)} readouts; 99.9 % energy: clean {count(e, 0.999)}, noisy {count(en, 0.999)} of {nc}")
    print("clean spectrum rel:", np.array2string(e / e.sum(), precision=2, max_line_width=200))
    print(f"tau [{tau[0]*1e3:.3f}, {tau[-1]*1e3:.3f}] ms, sigma {sigma:.2f}, mean E(n)/(C sigma^2) {Em.mean()/(nc*sigma**2):.1f}, "
          f"E(0)/(C sigma^2) {Em[0]/(nc*sigma**2):.3g}")
    print("wbar range per term over the readout (rad/m^2):", (wb.max(0) - wb.min(0)).round(3))


if __name__ == "__main__":
    main()
