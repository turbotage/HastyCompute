"""Steady-state (spoiled GRE) weighted water / fat maps of a simulated scan's static tissue on its reconstruction grid.

    ss_maps.py <sim_input.hv> <scan dir> <out dir>

simphantom's truth_pd_{water,fat}.npy are raw PD sums; the simulator applies the steady state sin α (1−E1)/(1−cos α E1)
per source with the local flip angle. Here: the same deposition (index round((x − centre)/res + n/2), as simphantom),
weighted by the steady state at the nominal flip inside the slab (rect profile: an approximation of the sinc slab, its
transition band ignored) and 0 outside it. Checks the raw-PD deposition against truth_pd_water.npy.
Writes ss_water.npy, ss_fat.npy (float32, recon grid) and ss_maps.json.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(REPO, "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

hv, scan, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
meta = json.load(open(os.path.join(scan, "meta.json")))
v = vf.open_lazy(hv)
t = v["tissue"]
pos = t["pos"].read().astype(np.float64)
pd = t["pd"].read().astype(np.float64)
t1 = t["t1"].read().astype(np.float64)
sh = t["shift_ppm"].read()
presc = v["presc"]
c = np.array(presc["center"])
res = float(presc["res"])
dims = [int(round(f / res)) for f in presc["fov"]]
slab = presc["slab"]
tr, flip = float(meta["tr"]), np.deg2rad(float(meta["flip_deg"]))
idx = np.rint((pos - c) / res + np.array([d // 2 for d in dims])).astype(np.int64)
ok = np.all((idx >= 0) & (idx < np.array(dims)), axis=1)
q = idx[ok, 0] + dims[0] * (idx[ok, 1] + dims[1] * idx[ok, 2])
zrel = (pos[ok] - np.array(slab["center"])) @ np.array(slab["normal"])
inslab = np.abs(zrel) <= 0.5 * slab["thk"]
e1 = np.exp(-tr / np.maximum(t1[ok], 1e-4))
ss = np.sin(flip) * (1 - e1) / (1 - np.cos(flip) * e1) * inslab
fat = sh[ok] != 0
n = dims[0] * dims[1] * dims[2]
raw_w = np.bincount(q[~fat], weights=pd[ok][~fat], minlength=n)
ref = np.load(os.path.join(scan, "truth_pd_water.npy")).reshape(-1, order="F")
err = np.abs(raw_w - ref).max() / ref.max()
print(f"deposition check vs truth_pd_water: max rel diff {err:.2e}")
assert err < 1e-4, "deposition does not reproduce simphantom's truth map"
sw = np.bincount(q[~fat], weights=(pd[ok] * ss)[~fat], minlength=n).reshape(dims, order="F").astype(np.float32)
sf = np.bincount(q[fat], weights=(pd[ok] * ss)[fat], minlength=n).reshape(dims, order="F").astype(np.float32)
np.save(os.path.join(out, "ss_water.npy"), sw)
np.save(os.path.join(out, "ss_fat.npy"), sf)
info = dict(tr=tr, flip_deg=float(np.rad2deg(flip)), slab_thk=slab["thk"], frac_pd_in_slab=float((pd[ok] * inslab).sum() / pd[ok].sum()),
            fat_share_pd=float(pd[ok][fat].sum() / pd[ok].sum()), fat_share_ss=float(sf.sum() / (sf.sum() + sw.sum())),
            fat_share_ss_energy=float((sf.astype(np.float64) ** 2).sum() / ((sf.astype(np.float64) ** 2).sum() + (sw.astype(np.float64) ** 2).sum())))
json.dump(info, open(os.path.join(out, "ss_maps.json"), "w"), indent=1)
print(info)
