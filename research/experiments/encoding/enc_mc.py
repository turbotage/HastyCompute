"""Monte-Carlo check of the wrap-failure union bound (enc_lib.Geometry) with exact complex Gaussian noise and the
ML decoder (enc_lib.Decoder, prior |v| <= the design's unaliased radius), interior (|v| <= V/2) and edge (|v| = V)
populations (2026-11-17). Failure = |v_hat - v| > max(6 sigma_v, 0.1 V).

    nice -n 10 rustlib/.venv/bin/python research/experiments/encoding/enc_mc.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from enc_lib import Decoder, Geometry, ball_samples, designs, sigma_v_rms  # noqa: E402


def mc(k, dec, snr_e, pop, n, rng):
    if pop == "int":
        v = ball_samples(n, rng) * 0.5
    else:
        u = rng.normal(size=(n, 3))
        v = u / np.linalg.norm(u, axis=1, keepdims=True)
    theta = rng.uniform(-np.pi, np.pi, size=(n, 1))
    s = np.exp(1j * (theta + v @ k.T))
    s = s + (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)) / (np.sqrt(2) * snr_e)
    vh, _ = dec.decode(np.angle(s))
    err = np.linalg.norm(vh - v, axis=1)
    sv = sigma_v_rms(k, snr_e) * np.sqrt(3)
    return float(np.mean(~(err <= max(6 * sv, 0.1))))


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    d = designs()
    # (name, kappa = f * pi, decoder prior radius V = 1; Decoder.decode adds a 0.15 slack)
    cases = [("coprime7 p=2", 0.92, 1.0), ("coprime7 p=3", 0.92, 1.0), ("coprime8H p=2", 1.2, 1.0), ("ref4", 0.9, 1.0)]
    for name, f, vmax in cases:
        k = np.pi * f * d[name]
        g = Geometry(k)
        dec = Decoder(k, vmax=vmax)
        for snr in (3.0, 4.0, 6.0):
            pm_i = mc(k, dec, snr, "int", 12000, rng)
            pm_e = mc(k, dec, snr, "edge", 6000, rng)
            print(f"{name:14s} f={f} SNR_e {snr}: MC int {pm_i:.2e} edge {pm_e:.2e} | Gaussian bound int {g.p_fail(snr, 'int'):.2e} edge {g.p_fail(snr, 'edge'):.2e}", flush=True)
