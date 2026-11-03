"""Figures for one reconstruction run directory written by `recon_p2np` (see its --out).

    python -I run_figs.py <run_dir>

Reads run.json, the arrays in <run_dir> (Fortran-ordered .npy: smaps [x,y,z,coil], mask_*, warm [x,y,z,encode],
recon_<tag> [x,y,z,encode,frame], vel_<tag> [x,y,z,component,frame], metrics_<tag>.csv) and the simulator truth from
run.json["sim_dir"]; writes <run_dir>/figs/*.png and figs/index.md. Velocities are compared on the voxel grid (with
--gnl the truth at the true positions differs slightly; the driver's RMSE uses the true positions).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

run = Path(sys.argv[1])
info = json.loads((run / "run.json").read_text())
sim = Path(info["sim_dir"])
figs = run / "figs"
figs.mkdir(exist_ok=True)
venc = info["venc"]
vlim = 0.75 * venc
truth = np.load(sim / "truth_vel_logical.npy")  # [x,y,z,3,nf]
frac = np.load(sim / "truth_lumen_fraction.npy")
lum = frac > 0.9
nf = truth.shape[-1]
zs = int(np.argmax(lum.sum(axis=(0, 1))))  # axial slice with the most lumen
ys = int(np.argmax(lum.sum(axis=(0, 2))))  # coronal slice
comp_names = ["x", "y", "z"]
speed = np.sqrt((truth ** 2).sum(axis=3))
peak = int(np.argmax([speed[..., f][lum].mean() for f in range(nf)]))
frames = sorted({0, max(peak - 2, 0), peak, min(peak + 2, nf - 1), nf // 2, nf - 1})
entries = []


def axial(a):
    return a[:, :, zs].T


def coronal(a):
    return a[:, ys, :].T


def outline(ax, sl=axial, mask=lum, color="y"):
    ax.contour(sl(mask.astype(float)), levels=[0.5], colors=color, linewidths=0.6)


def save(fig, name, caption):
    fig.savefig(figs / name, dpi=80)
    plt.close(fig)
    entries.append((name, caption))


def gshow(ax, img, ref=None, log=False):
    """grey image, robust scale: 0 .. 99.5th percentile of `ref` (default the image) inside the coil-map mask"""
    r = img if ref is None else ref
    vals = r[np.isfinite(r) & (r > 0)]
    hi = np.percentile(vals, 99.5) if vals.size else 1.0
    if log:
        ax.imshow(np.log10(np.maximum(img, hi * 1e-3)), cmap="gray", vmin=np.log10(hi * 1e-3), vmax=np.log10(hi), origin="lower", aspect="auto", interpolation="nearest")
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=hi, origin="lower", aspect="auto", interpolation="nearest")


def vimshow(ax, img, title, lim=vlim, cmap="RdBu_r"):
    ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.axis("off")


# ---------------------------------------------------------------- masks over the warm start
smaps = np.load(run / "smaps.npy")
m_sup = np.load(run / "mask_support.npy") > 0.5
m_map = np.load(run / "mask_maps.npy") > 0.5
warm = np.load(run / "warm.npy") if (run / "warm.npy").exists() else None
base = np.abs(warm).sum(axis=3) if warm is not None else np.sqrt((np.abs(smaps) ** 2).sum(axis=3))
fig, ax = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={"width_ratios": [1, 1.4]})
inner = base * m_map
for row, log in [(0, False), (1, True)]:
    for a, sl, t in [(ax[row, 0], axial, f"axial z={zs}"), (ax[row, 1], coronal, f"coronal y={ys}")]:
        gshow(a, sl(base), ref=inner[lum | m_sup] if not log else None, log=log)
        outline(a, sl, m_sup, "lime")
        outline(a, sl, m_map, "c")
        outline(a, sl, lum, "y")
        a.set_title(f"warm |x| summed over encodes ({'log10, 1e-3..max' if log else 'linear, 99.5% inside support'}), {t}\ngreen: support, cyan: coil-map mask, yellow: true lumen", fontsize=8)
        a.axis("off")
save(fig, "01_masks.png", "Object support (green) and coil-map mask (cyan) over the warm-start magnitude; true lumen in yellow. "
     "The lumen must lie inside the coil-map mask.")

# ---------------------------------------------------------------- coil maps
nvc = smaps.shape[3]
fig, ax = plt.subplots(2, nvc, figsize=(1.6 * nvc, 3.6))
for c in range(nvc):
    gshow(ax[0, c], axial(np.abs(smaps[..., c])))
    ax[1, c].imshow(axial(np.angle(smaps[..., c])), cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower")
    ax[0, c].set_title(f"vc {c}", fontsize=7)
    for r in range(2):
        outline(ax[r, c], axial, m_map, "c")
        ax[r, c].axis("off")
save(fig, "02_coilmaps.png", "Virtual coil maps (magnitude top, phase bottom), axial slice, after normalisation and masking.")

# ---------------------------------------------------------------- warm start
tavg = truth.mean(axis=-1)
if warm is not None:
    wv = np.stack([-np.angle(warm[..., a + 1] * np.conj(warm[..., 0])) * venc / np.pi for a in range(3)], axis=3)
    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for e in range(4):
        gshow(ax[0, e], axial(np.abs(warm[..., e])), ref=np.abs(warm[..., e])[m_sup])
        ax[0, e].set_title(f"warm |x|, encode {e}", fontsize=8)
        ax[1, e].imshow(axial(np.angle(warm[..., e])), cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower")
        ax[1, e].set_title(f"warm phase, encode {e}", fontsize=8)
    for a in range(3):
        vimshow(ax[2, a], axial(wv[..., a]), f"warm v{comp_names[a]} (time average)")
    vimshow(ax[2, 3], axial(tavg[..., int(np.argmax(np.abs(tavg[lum]).mean(axis=0)))]), "truth, time-average (dominant comp.)")
    for r in ax.flat:
        outline(r)
        r.axis("off")
    save(fig, "03_warm.png", "Warm start (time-averaged PCGLS per encode): magnitude, phase, velocity vs the time-averaged truth.")

# ---------------------------------------------------------------- truth
dom = int(np.argmax([np.abs(truth[..., a, :][lum]).mean() for a in range(3)]))
cols = 5
rows = int(np.ceil(nf / cols))
fig, ax = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
for f in range(rows * cols):
    a = ax.flat[f]
    if f < nf:
        vimshow(a, axial(truth[..., dom, f]), f"truth v{comp_names[dom]}, frame {f}" + (" (peak)" if f == peak else ""))
        outline(a)
    else:
        a.axis("off")
save(fig, "04_truth.png", f"Truth velocity (component {comp_names[dom]}, the dominant one in this slice) for every frame.")

# ---------------------------------------------------------------- per run
for tag in info["runs"]:
    if not (run / f"vel_{tag}.npy").exists():
        continue
    vel = np.load(run / f"vel_{tag}.npy")
    rec = np.load(run / f"recon_{tag}.npy")
    # velocity per frame: truth / recon / error, all components
    fig, ax = plt.subplots(len(frames), 9, figsize=(18, 2.1 * len(frames)))
    for r, f in enumerate(frames):
        for a in range(3):
            vimshow(ax[r, 3 * a], axial(truth[..., a, f]), f"truth v{comp_names[a]} f{f}")
            vimshow(ax[r, 3 * a + 1], axial(vel[..., a, f]), f"recon v{comp_names[a]} f{f}")
            vimshow(ax[r, 3 * a + 2], axial(vel[..., a, f] - truth[..., a, f]), "error", lim=vlim / 2)
            for k in range(3):
                outline(ax[r, 3 * a + k])
    save(fig, f"10_{tag}_velocity.png", f"{tag}: velocity per frame (truth / recon / error, all components), axial z={zs}; peak systole = frame {peak}.")
    # magnitude and phase of the reference encode per frame
    fig, ax = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows))
    for f in range(rows * cols):
        a = ax.flat[f]
        if f < nf:
            gshow(a, axial(np.abs(rec[..., 0, f])), ref=np.abs(rec[..., 0, :])[m_sup])
            a.set_title(f"|x| encode 0, frame {f}", fontsize=7)
            outline(a)
        a.axis("off")
    save(fig, f"11_{tag}_magnitude.png", f"{tag}: reconstructed magnitude of the reference encode, every frame.")
    # convergence
    with open(run / f"metrics_{tag}.csv") as fh:
        rowsm = list(csv.DictReader(fh))
    it = np.array([int(r["iter"]) for r in rowsm])
    fv = np.array([float(r["f"]) for r in rowsm])
    rm = np.array([float(r["rmse"]) for r in rowsm])
    pf = np.array([[float(r[f"frame{f}"]) for f in range(nf)] for r in rowsm])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ok = np.isfinite(fv)
    ax[0].semilogy(it[ok], fv[ok], "o-")
    ax[0].set_title("data fidelity f(x)")
    ax[0].set_xlabel("iteration")
    ax[1].plot(it, rm, "o-")
    ax[1].set_title("velocity RMSE in the lumen (m/s)")
    ax[1].set_xlabel("iteration")
    im = ax[2].imshow(pf.T, aspect="auto", origin="lower", cmap="magma", extent=[it[0] - 0.5, it[-1] + 0.5, -0.5, nf - 0.5])
    ax[2].set_title("per-frame RMSE (m/s)")
    ax[2].set_xlabel("report index → iteration")
    ax[2].set_ylabel("frame")
    plt.colorbar(im, ax=ax[2])
    save(fig, f"12_{tag}_convergence.png", f"{tag}: convergence (fidelity, lumen velocity RMSE, per-frame RMSE).")
    # velocity-time curves at lumen points (dominant component)
    idx = np.argwhere(lum[:, :, zs])
    pts = idx[np.linspace(0, len(idx) - 1, 6).astype(int)] if len(idx) else []
    fig, ax = plt.subplots(1, len(pts), figsize=(3 * max(len(pts), 1), 3), squeeze=False)
    for k, (i, j) in enumerate(pts):
        a = ax[0, k]
        a.plot(truth[i, j, zs, dom, :], "k-", label="truth")
        a.plot(vel[i, j, zs, dom, :], "r.-", label="recon")
        if warm is not None:
            a.axhline(wv[i, j, zs, dom], color="b", ls="--", label="warm")
        a.set_title(f"({i},{j},{zs}) v{comp_names[dom]}", fontsize=8)
        a.set_ylim(-venc, venc)
        if k == 0:
            a.legend(fontsize=7)
    save(fig, f"13_{tag}_curves.png", f"{tag}: velocity-time curves at six lumen points (dominant component).")
    # coronal view at peak systole
    fig, ax = plt.subplots(3, 3, figsize=(12, 6))
    for a in range(3):
        vimshow(ax[a, 0], coronal(truth[..., a, peak]), f"truth v{comp_names[a]} f{peak}")
        vimshow(ax[a, 1], coronal(vel[..., a, peak]), f"recon v{comp_names[a]}")
        vimshow(ax[a, 2], coronal(vel[..., a, peak] - truth[..., a, peak]), "error", lim=vlim / 2)
        for k in range(3):
            outline(ax[a, k], coronal)
            ax[a, k].set_aspect("auto")
    save(fig, f"14_{tag}_coronal.png", f"{tag}: coronal slice y={ys} at peak systole.")

with open(figs / "index.md", "w") as fh:
    fh.write(f"# {run.name}\n\n`{info['cmdline']}`\n\n")
    for name, cap in entries:
        fh.write(f"## {name}\n{cap}\n\n![{name}]({name})\n\n")
print(f"{len(entries)} figures in {figs}")
