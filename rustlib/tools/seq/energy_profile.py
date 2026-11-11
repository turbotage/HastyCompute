"""Measured k-space data energy per ADC sample of a simulated scan (noise-free data_clean.npy), and the energy profile
vs |k| for the outgoing and the returning half of an out-and-back readout. Reads a strided subset of readouts through a
memory map (no copy of the store). Writes <out>/energy_profile.npz.

    energy_profile.py <scan dir> <out dir> [--readouts 448] [--split SAMPLE]
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("out")
ap.add_argument("--readouts", type=int, default=448)
ap.add_argument("--split", type=int, default=None, help="first sample of the return leg (default: argmax |k|)")
a = ap.parse_args()

d = np.load(os.path.join(a.scan, "data_clean.npy"), mmap_mode="r")
s = np.load(os.path.join(a.scan, "samples.npy"), mmap_mode="r")
nr, na, nc = d.shape
idx = np.linspace(0, nr - 1, a.readouts).round().astype(int)
e = np.zeros(na)
for i in idx:
    y = np.asarray(d[i])  # (na, nc)
    e += (np.abs(y) ** 2).sum(axis=1)
e /= len(idx) * nc
k = np.asarray(s[idx[0], :, :3]).astype(np.float64)
tau = np.asarray(s[idx[0], :, 3]).astype(np.float64)
kr = np.linalg.norm(k, axis=1)
split = a.split if a.split is not None else int(np.argmax(kr))
kmax = kr.max()
nb = 50
edges = np.linspace(0, kmax * 1.0001, nb + 1)
rc = 0.5 * (edges[1:] + edges[:-1])


def prof(sel):
    b = np.digitize(kr[sel], edges) - 1
    m = np.full(nb, np.nan)
    for j in range(nb):
        w = b == j
        if w.any():
            m[j] = e[sel][w].mean()
    return m


out_leg = np.zeros(na, bool)
out_leg[:split] = True
p_out, p_ret = prof(out_leg), prof(~out_leg)
os.makedirs(a.out, exist_ok=True)
np.savez(os.path.join(a.out, "energy_profile.npz"), e=e, tau=tau, kr=kr, split=split, rc=rc, p_out=p_out, p_ret=p_ret, readouts=idx)
tot = e.sum()
print(f"{len(idx)} readouts × {nc} coils, {na} samples, split {split} (τ {tau[split]*1e3:.2f} ms)")
print("energy fraction per readout quarter:", [round(float(e[q * na // 4:(q + 1) * na // 4].sum() / tot), 4) for q in range(4)])
print("energy fraction out leg / return leg:", round(float(e[:split].sum() / tot), 4), round(float(e[split:].sum() / tot), 4))
for f in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.95]:
    j = int(f * nb)
    print(f"  |k|/kmax {rc[j]/kmax:.2f}: energy per sample out {p_out[j]:.4g}, return {p_ret[j]:.4g}, ratio {p_ret[j]/p_out[j]:.3f}")
