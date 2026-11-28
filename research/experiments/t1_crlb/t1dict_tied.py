"""Rank of the EPG dictionary when T2 is KNOWN (B0/T2/fat map) and only follows R1 through the Gd model.
Entries: T1 (log grid 0.25-4.8 s) x B1 (0.7-1.2) x baseline T2_0 set; T2 = 1/(1/T2_0 + r2 C), C = (1/T1 - 1/T1_0)/r1 (T1_0 = 4.5 CSF, 1.3 tissue).
Question (user 2026-11-28): how much does known T2 shrink K?  usage: python t1dict_tied.py"""
import os, subprocess, io
import numpy as np
import t1dict

BIN = t1dict.BIN
FLIPS, TR, TP = "13.1,11.6,3,30,3,3,3,3", 20.0, 12.0
nper = int((TP - 15e-3) / (TR * 1e-3)) // 8
SEGSTR = ",".join(f"{f}:{TR}:{nper}" for f in FLIPS.split(","))
ENV = dict(os.environ, INCS="117")


def build(t20s, t10s):
    rows = []
    for T10, T20 in zip(t10s, t20s):
        for t1 in np.exp(np.linspace(np.log(0.25), np.log(4.8), 40)):
            c = max(0.0, (1 / t1 - 1 / T10) / 3.3)
            t2 = 1.0 / (1 / T20 + 5.0 * c)
            for b1 in np.linspace(0.7, 1.2, 6):
                rows.append((t1, t2, b1))
    inp = "\n".join(",".join(f"{x:.9g}" for x in r) for r in rows) + "\n"
    out = subprocess.run([BIN, "eval", "ir", SEGSTR, "csf"], input=inp, capture_output=True, text=True, check=True, env=ENV).stdout
    return np.loadtxt(io.StringIO(out), delimiter=",", ndmin=2).T


for name, t20s, t10s in [("T2_0 = 2.0 (CSF only)", [2.0], [4.5]),
                         ("T2_0 = 0.1 (tissue only)", [0.1], [1.3]),
                         ("T2_0 in {2.0 CSF, 0.1 tissue}", [2.0, 0.1], [4.5, 1.3]),
                         ("T2_0 in {2.0, 1.0, 0.3, 0.1, 0.06} (voxel-wise map)", [2.0, 1.0, 0.3, 0.1, 0.06], [4.5, 3.0, 2.0, 1.3, 0.9])]:
    t1dict.ranks(build(t20s, t10s), name)
