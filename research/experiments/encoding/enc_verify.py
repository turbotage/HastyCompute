"""Full-enumeration verification of search finalists and the reference designs (2026-11-17).

For each design shape and slack factor f (kappa = f * pi * lambda_1(M*) for integer designs, i.e. protected radius
R = V / f): enc_lib.Geometry with both criteria (P_int <= 1e-3 over |v| <= V/2, mean P_edge <= 1e-2 at |v| = V),
the required SNR_tot = sqrt(K) SNR_e, eta = sigma_v,rms SNR_tot / V, and the largest first-moment excursion per axis
relative to coprime7 p=2 (balanced layout; encode duration ~ (M1 ratio)^(1/3) when slew-limited).

    nice -n 10 rustlib/.venv/bin/python research/experiments/encoding/enc_verify.py search.json out.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from enc_lib import Geometry, designs, max_m1, sigma_v_rms  # noqa: E402
from enc_search import lam1_dual  # noqa: E402


def scale_of(z):
    z = np.asarray(z, float)
    if np.allclose(z, np.round(z)):
        l1 = lam1_dual(z[1:] - z[0])
        return math.pi * l1
    return None


def run(name, shape, base, fs, ref_m1):
    rows = []
    K = len(shape)
    for f in fs:
        k = f * base * np.asarray(shape, float)
        g = Geometry(k)
        se = g.snr_req()
        eta = sigma_v_rms(k, 1.0) * math.sqrt(K)
        row = dict(name=name, K=K, f=f, snr_e_req=se, snr_tot_req=se * math.sqrt(K), eta=eta, m_min=g.m_min, m1_rel=max_m1(k) / ref_m1, z=np.asarray(shape).tolist())
        rows.append(row)
        print(f"{name:34s} K={K} f={f:.2f}  SNR_tot_req {row['snr_tot_req']:6.2f}  eta {eta:.4f}  m_min {g.m_min:.3f}  M1/M1(cp7) {row['m1_rel']:.2f}", flush=True)
    return rows


if __name__ == "__main__":
    search = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    d = designs()
    ref_m1 = max_m1(math.pi * d["coprime7 p=2"])
    fs = (1.0, 0.92, 0.85)
    rows = []
    for name in ("ref4", "jm5 (FC + tetra)", "dual7 1:2", "coprime7 p=2", "coprime7 p=3", "coprime7 p=4", "coprime8 p=2", "coprime8H p=2"):
        s = d[name]
        if name in ("ref4", "jm5 (FC + tetra)", "coprime8H p=2"):
            rows += run(name, s, math.pi, (1.2, 1.0, 0.9, 0.8, 0.7, 0.6), ref_m1)  # non-integer shapes: scale scan
        else:
            rows += run(name, s, scale_of(s), fs, ref_m1)
    # irrational: icosahedral, scale scan
    for name in ("icosa6 (no ref)", "icosa7 (ref + 6)"):
        rows += run(name, d[name], math.pi, (1.6, 1.3, 1.0, 0.8, 0.6), ref_m1)
    # finalists from the search: per K, designs with snr_tot_req in bands
    for K in ("5", "6", "7"):
        front = search[K]
        pick = []
        for lo, hi in ((6, 7.5), (7.5, 9), (9, 11), (11, 14)):
            c = [r for r in front if lo <= r["snr_tot_req"] < hi]
            if c:
                pick.append(min(c, key=lambda r: r["eta"]))
        for r in pick:
            z = np.array(r["z"], float)
            rows += run(f"search K={K} @{r['snr_tot_req']:.1f}", z, scale_of(z), fs, ref_m1)
    json.dump(rows, open(out, "w"), indent=1)
