"""Figures for an `opval` run directory: truth / recon / error per encode and model, velocity errors, convergence.

    python -I opval_figs.py <run_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

run = Path(sys.argv[1])
info = json.loads((run / "run.json").read_text())
met = json.loads((run / "metrics.json").read_text())
sim = Path(info["sim_dir"])
figs = run / "figs"
figs.mkdir(exist_ok=True)
venc = info["venc"]
truth = np.load(sim / "truth_object.npy")  # x,y,z,4
lum = np.load(sim / "truth_lumen_fraction.npy") > 0.5
tvel = np.load(sim / "truth_vel_logical.npy")[..., 0]  # x,y,z,3
z = int(np.argmax(lum.sum(axis=(0, 1))))
y = int(np.argmax(lum.sum(axis=(0, 2))))
models = [m for m in info["models"] if (run / f"recon_{m.replace('+', '_')}.npy").exists()]
entries = []


def save(fig, name, cap):
    fig.savefig(figs / name, dpi=80)
    plt.close(fig)
    entries.append((name, cap))


def show(ax, img, title, cmap="gray", vmin=None, vmax=None):
    ax.imshow(img.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


ref = np.abs(truth[..., 0])
hi = np.percentile(ref[ref > 0], 99.5)
for m in models:
    rec = np.load(run / f"recon_{m.replace('+', '_')}.npy")
    fig, ax = plt.subplots(4, 6, figsize=(18, 11))
    for e in range(4):
        t, r = truth[:, :, z, e], rec[:, :, z, e]
        err = np.abs(r - t)
        show(ax[e, 0], np.abs(t), f"|truth| enc {e}", vmin=0, vmax=hi)
        show(ax[e, 1], np.abs(r), f"|recon| enc {e} ({m})", vmin=0, vmax=hi)
        show(ax[e, 2], err, f"|recon − truth| ×10 (rel {met[m]['rel_error_final'][e]:.1e})", cmap="magma", vmin=0, vmax=hi / 10)
        show(ax[e, 3], np.angle(t), "∠truth", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        show(ax[e, 4], np.angle(r), "∠recon", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        show(ax[e, 5], np.angle(r * np.conj(t)) * (np.abs(t) > 0.02 * hi), "∠(recon·truth*) (rad, ±0.2)", cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    fig.suptitle(f"model {m}: L = {met[m]['nseg']}, order {met[m]['order']}, GNL {met[m]['gnl']} — axial z = {z}", fontsize=10)
    save(fig, f"01_{m.replace('+', '_')}_images.png", f"Model `{m}`: truth / recon / error magnitude (×10) and phases, all encodes, slice z={z}.")

# velocity per model (all components), lumen-restricted error maps
fig, ax = plt.subplots(len(models), 7, figsize=(21, 3.0 * len(models)), squeeze=False)
for i, m in enumerate(models):
    rec = np.load(run / f"recon_{m.replace('+', '_')}.npy")
    v = np.stack([-np.angle(rec[..., a + 1] * np.conj(rec[..., 0])) * venc / np.pi for a in range(3)], axis=-1)
    d = (v - tvel + venc) % (2 * venc) - venc
    show(ax[i, 0], tvel[:, :, z, 0], "truth v0", cmap="RdBu_r", vmin=-venc, vmax=venc)
    for a in range(3):
        show(ax[i, 1 + a], v[:, :, z, a], f"{m}: v{a}", cmap="RdBu_r", vmin=-venc, vmax=venc)
        show(ax[i, 4 + a], d[:, :, z, a] * (np.abs(truth[:, :, z, 0]) > 0.02 * hi), f"{m}: v{a} − truth (±0.1 m/s), RMSE lumen {met[m]['vel_rmse_lumen']:.3f}", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    for a in ax[i]:
        a.contour(lum[:, :, z].T, levels=[0.5], colors="y", linewidths=0.5)
save(fig, "02_velocity.png", "Velocity from the encode phases (truth, recon, error ±0.1 m/s) per model; lumen in yellow.")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for m in models:
    for e, cur in enumerate(met[m]["rel_error_curves"]):
        k, err = np.array(cur).T
        ax[0].semilogy(k, err, label=f"{m} enc {e}" if e == 0 else None, color=f"C{models.index(m)}", alpha=0.4 + 0.15 * e)
ax[0].set_xlabel("PCGLS iteration")
ax[0].set_ylabel("relative image error (support)")
ax[0].legend(fontsize=8)
ax[0].grid(True, which="both", alpha=0.3)
names = models
ax[1].bar(np.arange(len(names)) - 0.2, [met[m]["vel_rmse_lumen"] for m in names], 0.4, label="lumen RMSE")
ax[1].bar(np.arange(len(names)) + 0.2, [met[m]["vel_rmse_static"] for m in names], 0.4, label="static-tissue RMSE")
ax[1].set_xticks(range(len(names)))
ax[1].set_xticklabels(names)
ax[1].set_ylabel("velocity error (m/s)")
ax[1].set_yscale("log")
ax[1].legend(fontsize=8)
save(fig, "03_convergence.png", "Relative image error vs iteration per model/encode; velocity RMSE per model.")

with open(figs / "index.md", "w") as fh:
    fh.write(f"# {run.name}\n\n`{info['cmdline']}`\n\n| model | L | order | GNL | rel. error (enc 0..3) | vel RMSE lumen | vel bias lumen | vel RMSE static |\n|---|---|---|---|---|---|---|---|\n")
    for m in models:
        r = met[m]
        fh.write(f"| {m} | {r['nseg']} | {r['order']} | {r['gnl']} | {' '.join(f'{v:.1e}' for v in r['rel_error_final'])} | {r['vel_rmse_lumen']:.4f} | {' '.join(f'{v:+.4f}' for v in r['vel_bias_lumen'])} | {r['vel_rmse_static']:.4f} |\n")
    fh.write("\n")
    for n, cap in entries:
        fh.write(f"## {n}\n{cap}\n\n![{n}]({n})\n\n")
print(f"{len(entries)} figures in {figs}")
