"""Figures of a `phantomvox` label volume: orthogonal slices of the labels and of each group's winding number.

    python -I label_figs.py <labels.hv> <fig_dir> [--at x,y,z]      (slice position in m; default: the grid centre)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

v = vf.read(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
lab = np.asarray(v["labels"])  # [Z, Y, X]
names = list(v["groups"])
o = np.asarray(v["origin_m"], float)
h = float(v["voxel_m"])
nz, ny, nx = lab.shape
at = np.asarray([float(s) for s in sys.argv[sys.argv.index("--at") + 1].split(",")]) if "--at" in sys.argv else o + 0.5 * h * np.array([nx, ny, nz])
ix, iy, iz = [int(np.clip((at[a] - o[a]) / h, 0, n - 1)) for a, n in zip(range(3), (nx, ny, nz))]
ext = {  # imshow extents in mm: axial (x, y), coronal (x, z), sagittal (y, z)
    "ax": [o[0] * 1e3, (o[0] + nx * h) * 1e3, (o[1] + ny * h) * 1e3, o[1] * 1e3],
    "co": [o[0] * 1e3, (o[0] + nx * h) * 1e3, o[2] * 1e3, (o[2] + nz * h) * 1e3],
    "sa": [o[1] * 1e3, (o[1] + ny * h) * 1e3, o[2] * 1e3, (o[2] + nz * h) * 1e3],
}


def slices(a):
    return {"ax": a[iz], "co": a[:, iy, :][::-1], "sa": a[:, :, ix][::-1]}


base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
cm = ListedColormap(np.vstack([[0, 0, 0, 1], base[: len(names)]]))
fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
for a, (k, s) in zip(axs, slices(lab).items()):
    im = a.imshow(s, cmap=cm, vmin=-0.5, vmax=len(names) + 0.5, extent=ext[k], interpolation="nearest")
    a.set_title({"ax": f"axial z={at[2] * 1e3:.0f} mm", "co": f"coronal y={at[1] * 1e3:.0f} mm", "sa": f"sagittal x={at[0] * 1e3:.0f} mm"}[k])
    a.set_xlabel("mm")
cb = fig.colorbar(im, ax=axs, ticks=range(len(names) + 1), fraction=0.02)
cb.ax.set_yticklabels(["outside"] + names)
fig.suptitle(f"{Path(sys.argv[1]).name}: labels (last group with winding > ½), voxel {h * 1e3:.2f} mm")
fig.savefig(out / "labels.png", dpi=110, bbox_inches="tight")
plt.close(fig)

W = v["winding"]
n = len(names)
fig, axs = plt.subplots(n, 3, figsize=(13, 3.6 * n))
for r, g in enumerate(names):
    w = np.asarray(W[g]).astype(np.float32) / 127.5
    for c, (k, s) in enumerate(slices(w).items()):
        a = axs[r, c]
        im = a.imshow(s, cmap="magma", vmin=0, vmax=1.5, extent=ext[k], interpolation="nearest")
        a.contour(s[::-1] if k != "ax" else s, levels=[0.5], colors="c", linewidths=0.6, extent=ext[k] if k == "ax" else [ext[k][0], ext[k][1], ext[k][2], ext[k][3]], origin="upper" if k == "ax" else "lower")
        a.set_title(f"{g} {k}", fontsize=9)
        a.tick_params(labelsize=7)
    fig.colorbar(im, ax=axs[r], fraction=0.015)
fig.savefig(out / "winding.png", dpi=80, bbox_inches="tight")
plt.close(fig)
vol = {g: float((lab == i + 1).sum() * (h * 100) ** 3) for i, g in enumerate(names)}
(out / "volumes_cm3.txt").write_text("\n".join(f"{g:12s} {x:9.1f}" for g, x in vol.items()) + "\n")
print("wrote", out, "slice at", at, "voxel", (ix, iy, iz))
