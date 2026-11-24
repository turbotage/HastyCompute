"""The descending-aorta cross-section of truth_aorta_section.py in reconstructions (4 mm grid, gated runs): |v| at
peak systole (frame 6) for the truth, gc_llr_rm (binned, true ratemap) and t_cubic_rm (exact-time cubic)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "python/lib")
from hastycompute import store  # noqa: E402
S = "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/"
R = "data/hasty/phantom/factory_arms/ph0001/precond_compare/"
O = "research/experiments/flow_figs/"
vt = np.asarray(store.load_any(S + "truth_vel_logical.hv"))
spt = np.linalg.norm(vt, axis=3)
fs, (i, j, kz) = 6, (124, 62, 15)
runs = ["gc_llr_rm", "t_cubic_rm"]
vel = {r: np.asarray(store.load_any(R + r + "/" + [f for f in os.listdir(R + r) if f.startswith("vel_")][0])) for r in runs}
dr = np.array(vel[runs[0]].shape[:3]); dn = np.array(spt.shape[:3]); vr = np.array([3.96875, 4.0, 3.8])
# native voxel (i, j, kz) → recon voxel (centred grids)
p = np.round((np.array([i, j, kz]) - dn // 2) * 2.0 / vr + dr // 2).astype(int)
w = 12
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
vm = float(spt[i - w:i + w, j - w:j + w, kz, fs].max())
r0 = ax[0].imshow(spt[i - w:i + w, j - w:j + w, kz, fs].T, origin="lower", cmap="Blues", vmin=0, vmax=vm, interpolation="nearest", extent=[0, 2 * w * 2.0, 0, 2 * w * 2.0])
ax[0].set_title("truth (2 mm)", fontsize=10)
wr = 6
for a, r in zip(ax[1:], runs):
    s = np.linalg.norm(vel[r][..., fs], axis=3)
    a.imshow(s[p[0] - wr:p[0] + wr, p[1] - wr:p[1] + wr, p[2]].T, origin="lower", cmap="Blues", vmin=0, vmax=vm, interpolation="nearest", extent=[0, 2 * wr * vr[0], 0, 2 * wr * vr[1]])
    a.set_title(f"{r} (4 mm)", fontsize=10)
for a in ax:
    a.set_xlabel("mm", fontsize=8)
    a.tick_params(labelsize=7)
fig.colorbar(r0, ax=ax, shrink=0.85, label="|v| (m/s), frame 6")
fig.suptitle("Descending aorta cross-section at peak systole: truth vs reconstructions", fontsize=10)
fig.savefig(O + "recon_aorta_section.png", dpi=130)
print("recon voxel", p)
