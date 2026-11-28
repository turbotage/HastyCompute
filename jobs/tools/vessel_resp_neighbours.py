"""Breathing displacement of a vessel vs the tissue around it, along the vessel (does it slide against its neighbours?).

    rustlib/.venv/bin/python -I jobs/tools/vessel_resp_neighbours.py <phantom dir> <tissue_2mm.hv> <edge-name regex> [--step-mm 5]

Per sample (every --step-mm along the matched edges, ordered by z): the SI displacement at end-inspiration (r = 1,
Σ resp bases) at the lumen centre, the shell (1–6 mm outside the wall) internal-organ fraction as resp_motion's
embedded-vessel rule sees it, and the median SI displacement of each tissue in a wider shell (2–15 mm outside the wall)
with its voxel share. A vessel whose own displacement differs from the solid organ beside it by several mm slides
against that organ (or the organ is held by a tether / keep rule)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

A = sys.argv[1:]
D, TSF, PAT = Path(A[0]), A[1], re.compile(A[2])
STEP = float(A[A.index("--step-mm") + 1]) * 1e-3 if "--step-mm" in A else 5e-3
INTERNAL = ["lung", "airway", "myocardium", "valve", "blood_LA", "blood_LV", "blood_RA", "blood_RV", "liver", "gallbladder", "spleen", "thymus",
            "stomach_wall", "gi_content", "pancreas", "bowel_wall", "kidney"]
R = load_any(D / "motion/resp_motion.hv")
U = np.moveaxis(np.asarray(R["B"], float).sum(0), 0, -1)  # [Z, Y, X, 3]
o, h = np.asarray(R["origin_m"], float), float(R["voxel_m"])
TS = load_any(TSF)
lab, nm, o0, h0 = np.asarray(TS["labels"]), list(TS["tissues"]), np.asarray(TS["origin_m"], float), float(TS["voxel_m"])
ID = {n: i for i, n in enumerate(nm)}
I_org = np.isin(lab, [ID[n] for n in INTERNAL if n in ID])
g = np.load(D / "graph.npz")
names = json.loads((D / "graph.json").read_text())["edge_names"]
P, Rr, off = np.asarray(g["points"], float), np.asarray(g["radius"], float), np.asarray(g["edge_offsets"])


def usi(p):
    """SI displacement at points p [N, 3] (trilinear on the resp grid, cell centres)."""
    c = (p - o) / h - 0.5
    return ndimage.map_coordinates(U[..., 2], [c[:, 2], c[:, 1], c[:, 0]], order=1, mode="nearest")


rows = []
for e, n in enumerate(names):
    if not PAT.search(n):
        continue
    p, r = P[off[e]:off[e + 1]], Rr[off[e]:off[e + 1]]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    for t in np.arange(0, s[-1], STEP):
        j = int(np.searchsorted(s, t))
        q, rq = p[j], r[j]
        ro = rq + 15e-3
        i0 = np.maximum(np.floor((q - ro - o0) / h0).astype(int), 0)
        i1 = np.minimum(np.ceil((q + ro - o0) / h0).astype(int) + 1, [lab.shape[2], lab.shape[1], lab.shape[0]])
        zz, yy, xx = np.meshgrid(*(np.arange(i0[d], i1[d]) for d in (2, 1, 0)), indexing="ij")
        X = np.stack([xx, yy, zz], -1) * h0 + o0 + h0 / 2
        dd = np.linalg.norm(X - q, axis=-1)
        sh1 = (dd > rq + 1e-3) & (dd <= rq + 6e-3)
        fr = float(I_org[zz[sh1], yy[sh1], xx[sh1]].mean()) if sh1.any() else np.nan
        sh = (dd > rq + 2e-3) & (dd <= ro)
        L, Xs = lab[zz[sh], yy[sh], xx[sh]], X[sh]
        ds = usi(Xs)
        per = []
        for li in np.unique(L):
            m = L == li
            if m.mean() >= 0.08:
                per.append((float(m.mean()), nm[li], float(np.median(ds[m]))))
        per.sort(reverse=True)
        rows.append((q[2], n, float(usi(q[None])[0]), fr, rq, per))
rows.sort()
print(f"{D}: SI displacement at end-inspiration (mm, negative = caudal) of /{PAT.pattern}/ vs tissue 2–15 mm around it")
for z, n, u, fr, rq, per in rows:
    print(f"z {z * 1e3:6.0f} r {rq * 1e3:4.1f} {n[:28]:28s} vessel {u * 1e3:6.1f} | shell organ {fr:4.2f} | "
          + ", ".join(f"{t} {f * 100:.0f}% {d * 1e3:+.1f}" for f, t, d in per[:4]))
