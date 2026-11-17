"""Monte Carlo with exact complex noise and the grid ML decoder (enc_lib.GridDecoder; low memory, any design),
interior population |v| <= V/2, failure = |v_hat - v| > max(6 sigma_v, 0.1 V); against the Gaussian union bound
(2026-11-17). The class decoder of enc_mc.py needs > 10 GB for p >= 3 and 8H; this one needs < 1 GB.

    nice -n 10 rustlib/.venv/bin/python research/experiments/encoding/enc_mc_grid.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from enc_lib import Geometry, GridDecoder, ball_samples, designs, sigma_v_rms  # noqa: E402
from enc_search import lam1_dual  # noqa: E402

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    d = designs()
    A = np.array([[0, 0, 0], [0, -1, -1], [0, 2, -2], [0, 1, 0], [2, 1, 1], [2, 0, -1], [3, -1, -1]], float)
    S6 = np.array([[0, 0, 0], [2, -2, 0], [-1, -2, -1], [0, 0, -2], [2, 1, -3], [2, -1, -3]], float)
    cases = {
        "coprime7 p=2": 0.92 * math.pi * d["coprime7 p=2"],
        "lattice A": 0.92 * math.pi * lam1_dual(A[1:]) * A,
        "search6 b": 0.92 * math.pi * lam1_dual(S6[1:]) * S6,
        "8H f1.2": 1.2 * math.pi * d["coprime8H p=2"],
    }
    n = 20000
    for name, k in cases.items():
        g = Geometry(k)
        dec = GridDecoder(k, vmax=1.15)
        for snr in (3.0, 4.0):
            v = ball_samples(n, rng) * 0.5
            s = np.exp(1j * (rng.uniform(-np.pi, np.pi, (n, 1)) + v @ k.T))
            s = s + (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)) / (math.sqrt(2) * snr)
            vh, _ = dec.decode(np.angle(s))
            thr = max(6 * sigma_v_rms(k, snr) * math.sqrt(3), 0.1)
            pm = float(np.mean(np.linalg.norm(vh - v, axis=1) > thr))
            print(f"{name:14s} K={len(k)} SNR_e {snr}: MC int {pm:.2e} | Gaussian bound {g.p_fail(snr, 'int'):.2e}", flush=True)
