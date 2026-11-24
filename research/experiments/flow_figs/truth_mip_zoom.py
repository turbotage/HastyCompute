"""1 mm fine truth zooms (blood fraction × |v|, slab MIPs) on regions where the full MIPs (truth_mip.py) show
brightness steps along vessels. Usage: truth_mip_zoom.py <frame> <x0> <x1> <y0> <y1> <z0> <z1> <axis y|x|z> <name>
(mm = voxels on the 1 mm grid; the slab MIP is over the given range of the projection axis).
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "python/lib")
from hastycompute import store  # noqa: E402

S = "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/truth_fine/truth4d.hv"
f, x0, x1, y0, y1, z0, z1 = map(int, sys.argv[1:8])
ax_, name = sys.argv[8], sys.argv[9]
t = store.vf.open_lazy(S)
v = store.LazyArray(t["vel"])[f][:, z0:z1, y0:y1, x0:x1].astype(np.float32)
b = store.LazyArray(t["blood"])[f][z0:z1, y0:y1, x0:x1].astype(np.float32) / 255.0
lab = store.LazyArray(t["labels"])[f][z0:z1, y0:y1, x0:x1]
sp = np.sqrt((v * v).sum(0))
w = b * sp
k = {"z": 0, "y": 1, "x": 2}[ax_]
ext = {"y": [x0, x1, z0, z1], "x": [y0, y1, z0, z1], "z": [x0, x1, y0, y1]}[ax_]
fig, a = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
r = a[0].imshow(w.max(k), origin="lower", cmap="gray", extent=ext, interpolation="nearest")
a[0].set_title(f"frame {f}: slab MIP blood × |v| along {ax_}")
fig.colorbar(r, ax=a[0], shrink=0.8)
r = a[1].imshow(sp.max(k), origin="lower", cmap="magma", extent=ext, interpolation="nearest")
a[1].set_title("slab MIP |v| (all tissue)")
fig.colorbar(r, ax=a[1], shrink=0.8, label="m/s")
r = a[2].imshow(b.max(k), origin="lower", cmap="gray", extent=ext, interpolation="nearest", vmin=0, vmax=1)
a[2].set_title("slab MIP blood fraction")
fig.colorbar(r, ax=a[2], shrink=0.8)
fig.savefig("research/experiments/flow_figs/" + name, dpi=100)
names = t["tissues"]
u, c = np.unique(lab[b > 0.5], return_counts=True)
print({names[i]: int(n) for i, n in zip(u, c)})
print("max |v| in blood", float(sp[b > 0.5].max()) if (b > 0.5).any() else 0)
