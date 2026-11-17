"""Sphere-packing (tube-volume) bound on the time-normalised velocity noise of K-point encodings (2026-11-17).

Hypothesis sheet: {A (theta, v) mod 2 pi : theta in [0, 2 pi), |v| <= V}, a 4-dim piece of volume
    vol_4 = sqrt(det A^T A) 2 pi (4 pi / 3) V^3 = K^2 sqrt(det Cov k) (8 pi^2 / 3) V^3
in the K-torus of volume (2 pi)^K. ML decoding with margin d needs disjoint normal tubes of radius d/2 (r = K - 4
normal dimensions): vol_4 omega_r (d/2)^r <= (2 pi)^K. With Cov v = Cov(k)^-1 / (2 SNR_e^2 K), SNR_e = SNR_tot/sqrt K,
the geometric-mean time-normalised noise eta_g = det(Cov v)^(1/6) SNR_tot / V obeys
    eta_g >= 2^(-1/2) [K^2 (8 pi^2 / 3) omega_r (d/2)^r / (2 pi)^K]^(1/3),   d/2 = z sqrt(K) / (sqrt 2 SNR_tot)
(z: the one-sided Gaussian quantile for the target failure rate; edge effects of the sheet piece neglected).
Prints the bound and the designs' eta_g at their verified SNR requirement (verify.json).

    rustlib/.venv/bin/python research/experiments/encoding/enc_bound.py verify.json
"""
import json
import math
import sys

import numpy as np
from scipy.special import gamma


def omega(r):
    return math.pi ** (r / 2) / gamma(r / 2 + 1)


def eta_bound(K, snr_tot, z=3.3):
    r = K - 4
    h = z * math.sqrt(K) / (math.sqrt(2) * snr_tot)
    return 2 ** -0.5 * (K ** 2 * (8 * math.pi ** 2 / 3) * omega(r) * h ** r / (2 * math.pi) ** K) ** (1 / 3)


if __name__ == "__main__":
    print("bound eta_g(K, SNR_tot):")
    for snr in (6, 8, 10, 15, 20, 30):
        print(f"  SNR_tot {snr:3d}: " + "  ".join(f"K={K}: {eta_bound(K, snr):.3f}" for K in (4, 5, 6, 7, 8)))
    if len(sys.argv) > 1:
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from enc_lib import fisher
        rows = json.load(open(sys.argv[1]))
        print("designs (eta_g at the verified SNR_tot requirement, vs bound at that SNR):")
        for r in rows:
            if not np.isfinite(r["snr_tot_req"]):
                continue
            z = np.asarray(r["z"], float)
            # eta_g from eta_rms needs the shape: recompute (scale cancels in the ratio eta_g / eta_rms)
            F = fisher(z)
            Ci = np.linalg.inv(F)
            ratio = np.linalg.det(Ci) ** (1 / 6) / math.sqrt(np.trace(Ci) / 3)
            eg = r["eta"] * ratio
            b = eta_bound(r["K"], r["snr_tot_req"])
            print(f"  {r['name']:30s} f={r['f']:.2f} SNR_tot {r['snr_tot_req']:6.2f} eta_g {eg:.3f} bound {b:.3f} ratio {eg / b:.2f}")
