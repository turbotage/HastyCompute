"""Truth velocity of the 10-min ph0001 flow simulation (scan/truth_vel_logical, native 2 mm grid, 20 cardiac
frames): peak-systole slices through the fastest voxel, a diastolic frame, line profiles across the vessel there
(to look for banding) and the waveform at that voxel. Figures to research/experiments/flow_figs/.
    rustlib/.venv/bin/python -I research/experiments/flow_figs/truth_flow_figs.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "python/lib")
from hastycompute import store  # noqa: E402

S = "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/"
O = "research/experiments/flow_figs/"
v = np.asarray(store.load_any(S + "truth_vel_logical.hv"))  # (X, Y, Z, 3, F)
sp = np.linalg.norm(v, axis=3)  # (X, Y, Z, F)
p99 = [np.percentile(sp[..., f][sp[..., f] > 0.05], 99) for f in range(sp.shape[3])]
fs = int(np.argmax(p99))
fd = int((fs + sp.shape[3] // 2) % sp.shape[3])
# the fastest voxel at peak systole, excluding the box edge
sf = sp[..., fs].copy()
sf[:8], sf[-8:], sf[:, :8], sf[:, -8:], sf[:, :, :8], sf[:, :, -8:] = 0, 0, 0, 0, 0, 0
q = np.unravel_index(np.argmax(sf), sf.shape)
print(f"peak systole frame {fs} (99th pct |v| {p99[fs]:.2f} m/s), diastolic frame {fd}; fastest voxel {q} |v| {sf[q]:.2f} m/s")
vmax = float(np.percentile(sp[..., fs][sp[..., fs] > 0.05], 99.5))
h = 2.0  # mm

def slices(f, name):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    imgs = [(sp[:, :, q[2], f].T, "axial (x–y)", "x (mm)", "y (mm)"),
            (sp[:, q[1], :, f].T, "coronal (x–z)", "x (mm)", "z (mm)"),
            (sp[q[0], :, :, f].T, "sagittal (y–z)", "y (mm)", "z (mm)")]
    for a, (im, t, xl, yl) in zip(ax, imgs):
        r = a.imshow(im, origin="lower", cmap="Blues", vmin=0, vmax=vmax, extent=[0, im.shape[1] * h, 0, im.shape[0] * h], interpolation="nearest")
        a.set_title(f"{t}, frame {f}", fontsize=10)
        a.set_xlabel(xl, fontsize=9)
        a.set_ylabel(yl, fontsize=9)
        a.tick_params(labelsize=8)
    fig.colorbar(r, ax=ax, shrink=0.85, label="|v| (m/s)")
    fig.savefig(O + name, dpi=130)
    plt.close(fig)

slices(fs, "truth_speed_systole.png")
slices(fd, "truth_speed_diastole.png")
# zoom on the vessel around q (60 mm box), axial and coronal, systole, with the vz component (diverging)
w = 15
sl = (slice(max(q[0] - w, 0), q[0] + w), slice(max(q[1] - w, 0), q[1] + w))
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
r0 = ax[0].imshow(sp[sl[0], sl[1], q[2], fs].T, origin="lower", cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest", extent=[0, 2 * w * h, 0, 2 * w * h])
ax[0].set_title(f"|v| zoom, axial, frame {fs}", fontsize=10)
fig.colorbar(r0, ax=ax[0], shrink=0.85, label="|v| (m/s)")
vz = v[sl[0], sl[1], q[2], 2, fs].T
m = float(np.abs(vz).max())
r1 = ax[1].imshow(vz, origin="lower", cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest", extent=[0, 2 * w * h, 0, 2 * w * h])
ax[1].set_title(f"v_z zoom, axial, frame {fs}", fontsize=10)
fig.colorbar(r1, ax=ax[1], shrink=0.85, label="v_z (m/s)")
for a in ax:
    a.set_xlabel("mm", fontsize=9)
    a.tick_params(labelsize=8)
fig.savefig(O + "truth_vessel_zoom.png", dpi=130)
plt.close(fig)
# profiles across the vessel through q along x and y (systole): bands would show as plateaus / steps
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for a, axis, lab in [(ax[0], 0, "x"), (ax[1], 1, "y")]:
    idx = [q[0], q[1], q[2]]
    rng = np.arange(max(q[axis] - w, 0), min(q[axis] + w, sp.shape[axis]))
    prof = [sp[tuple(idx[:axis] + [i] + idx[axis + 1:])][fs] for i in rng]
    a.plot((rng - q[axis]) * h, prof, color="#2a6db5", lw=2, marker="o", ms=4)
    a.set_xlabel(f"{lab} offset from the fastest voxel (mm)", fontsize=9)
    a.set_ylabel("|v| (m/s)", fontsize=9)
    a.grid(alpha=0.25)
    a.tick_params(labelsize=8)
fig.suptitle(f"|v| profiles across the vessel at peak systole (frame {fs}), native 2 mm voxels", fontsize=10)
fig.savefig(O + "truth_vessel_profiles.png", dpi=130)
plt.close(fig)
# waveform at q and its 2-voxel neighbourhood mean
fig, a = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
nb = sp[q[0] - 2:q[0] + 3, q[1] - 2:q[1] + 3, q[2] - 2:q[2] + 3, :].reshape(-1, sp.shape[3]).mean(0)
a.plot(range(sp.shape[3]), sp[q][...] if sp[q].ndim else sp[q[0], q[1], q[2], :], color="#2a6db5", lw=2, marker="o", ms=4, label="fastest voxel")
a.plot(range(sp.shape[3]), nb, color="#7a7a7a", lw=2, marker="s", ms=4, label="5³ neighbourhood mean")
a.set_xlabel("cardiac frame", fontsize=9)
a.set_ylabel("|v| (m/s)", fontsize=9)
a.legend(fontsize=8, frameon=False)
a.grid(alpha=0.25)
fig.savefig(O + "truth_waveform.png", dpi=130)
plt.close(fig)
