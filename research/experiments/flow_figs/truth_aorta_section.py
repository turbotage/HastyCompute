"""Cross-section of the descending aorta in the ph0001 truth velocity (axial slice 30 mm above the box bottom):
|v| zoom at peak systole and three other frames, plus x / y profiles through the lumen centre."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "python/lib")
from hastycompute import store  # noqa: E402
S = "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/"
O = "research/experiments/flow_figs/"
v = np.asarray(store.load_any(S + "truth_vel_logical.hv"))
sp = np.linalg.norm(v, axis=3)
h = 2.0
kz = 15  # 30 mm above the bottom
# the lumen: the strongest flow in this slice at peak systole, near x ≈ 265 mm (the descending aorta in the coronal view)
fs = 6
sl = sp[:, :, kz, fs]
xs = slice(110, 160)
i, j = np.unravel_index(np.argmax(sl[xs]), sl[xs].shape)
i += 110
print(f"descending aorta centre (voxel) {i}, {j}, slice {kz}; |v| {sl[i, j]:.2f} m/s at frame {fs}")
w = 12
frames = [2, 4, 6, 10]
fig, ax = plt.subplots(1, len(frames) + 1, figsize=(16, 3.6), constrained_layout=True)
vm = float(sp[i - w:i + w, j - w:j + w, kz, :].max())
for a, f in zip(ax, frames):
    r = a.imshow(sp[i - w:i + w, j - w:j + w, kz, f].T, origin="lower", cmap="Blues", vmin=0, vmax=vm, interpolation="nearest", extent=[0, 2 * w * h, 0, 2 * w * h])
    a.set_title(f"frame {f}", fontsize=10)
    a.set_xlabel("mm", fontsize=8)
    a.tick_params(labelsize=7)
fig.colorbar(r, ax=ax[:len(frames)], shrink=0.85, label="|v| (m/s)")
a = ax[-1]
for axis, lab, col in [(0, "x", "#2a6db5"), (1, "y", "#7a7a7a")]:
    rng = np.arange(-w, w)
    prof = [sp[i + d, j, kz, fs] if axis == 0 else sp[i, j + d, kz, fs] for d in rng]
    a.plot(rng * h, prof, color=col, lw=2, marker="o", ms=3, label=f"across {lab}")
a.set_title(f"profiles, frame {fs}", fontsize=10)
a.set_xlabel("offset from the centre (mm)", fontsize=8)
a.set_ylabel("|v| (m/s)", fontsize=8)
a.legend(fontsize=7, frameon=False)
a.grid(alpha=0.25)
a.tick_params(labelsize=7)
fig.suptitle("Descending aorta, axial cross-section (truth velocity, native 2 mm voxels)", fontsize=10)
fig.savefig(O + "truth_aorta_section.png", dpi=130)
