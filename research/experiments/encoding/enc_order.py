"""Encode-order cross-talk when the K encodes of one rotation share a frame (2026-11-17).

phi_e = theta + k_e . v(t_e), v(t) = v0 + a t, t_e = slot(e) * TR. The LS decode gives v0 + G a with
G = [A^+]_v diag(t) k (3 x 3). The part t_c I (t_c = tr G / 3) is a pure time stamp (absorbed by velocity_time);
the rest, E = G - t_c I, mixes accelerations of the other axes in: error rms = ||E||_F / sqrt 3 * a_rms for isotropic
accelerations. Calibrated to the measured 0.087 m/s of coprime7 in the natural order (recon_diag_2026-11-13 §7).
Also reports the encode-train length K*TR (motion between first and last encode).

    nice -n 10 rustlib/.venv/bin/python research/experiments/encoding/enc_order.py [z-json of extra designs ...]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from enc_lib import designs  # noqa: E402


def xtalk(k, order):
    K = len(k)
    t = np.empty(K)
    t[list(order)] = np.arange(K)  # order[s] = encode in slot s
    A = np.hstack([np.ones((K, 1)), k])
    G = np.linalg.pinv(A)[1:] @ (t[:, None] * k)
    tc = np.trace(G) / 3
    return np.linalg.norm(G - tc * np.eye(3)) / np.sqrt(3), tc


def best(k):
    K = len(k)
    nat = xtalk(k, tuple(range(K)))
    allv = [(xtalk(k, o)[0], o) for o in itertools.permutations(range(K))]
    allv.sort()
    return nat, allv[0]


if __name__ == "__main__":
    d = designs()
    names = ["ref4", "had4", "jm5 (FC + tetra)", "coprime7 p=2", "coprime7 p=3", "dual7 1:2", "icosa6 (no ref)", "icosa7 (ref + 6)"]
    shapes = {n: d[n] for n in names}
    for j, a in enumerate(sys.argv[1:]):
        shapes[f"extra{j} {a}"] = np.array(json.loads(a), float)
    ref = xtalk(np.pi * d["coprime7 p=2"], tuple(range(7)))[0]
    scale = 0.087 / ref  # m/s per unit metric (TR 11.3 ms, ph0001 accelerations)
    for n, s in shapes.items():
        k = np.pi * s  # cross-talk metric is scale-free (G is invariant to scaling k)
        (xn, tcn), (xb, ob) = best(k)
        print(f"{n:40s} K={len(s)}  natural {xn * scale:.4f} m/s (t_c {tcn:.2f} TR)  best {xb * scale:.4f} m/s order {ob}", flush=True)
