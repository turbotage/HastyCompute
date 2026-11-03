"""Inspect one RACLETTE subject and the cohort metadata; figures + a summary into an output directory.

    python -I inspect_subject.py <variant> <subject> <out_dir>
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyvista_zstd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/hasty/external/raclette_data"
variant, subject, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
figs = out / "figs"
figs.mkdir(parents=True, exist_ok=True)
sd = DATA / variant / subject
lines = []


def log(s):
    print(s)
    lines.append(s)


def grid(img, key, n):
    nx, ny, nz = img.dimensions
    return img[key].reshape((nx, ny, nz) + ((n,) if n else ()), order="F")


# ---------------------------------------------------------------- MRI frames
frames = sorted(glob.glob(str(sd / "Data" / "7p_4DFlow_*.pv")))
f0 = pyvista_zstd.read(frames[0])
nenc = f0["real"].shape[1]
dims = f0.dimensions
log(f"{variant}/{subject}: {len(frames)} frames, dims {dims}, spacing {np.round(np.array(f0.spacing) * 1e3, 3)} mm, {nenc} encodings")
for k in f0.field_data.keys():
    v = np.asarray(f0.field_data[k])
    log(f"  field {k}: {(np.round(v, 4).tolist() if v.dtype.kind in 'fiu' else v.tolist()) if v.size <= 32 else v.shape}")
sig = np.stack([grid(pyvista_zstd.read(f), "real", nenc) + 1j * grid(pyvista_zstd.read(f), "imag", nenc) for f in frames], axis=-1)  # x,y,z,enc,t
mask = grid(f0, "mask", 0) > 0
dt = [float(np.asarray(pyvista_zstd.read(f).field_data["dtime"]).ravel()[0]) for f in frames]
log(f"  signal: |x| p50/p99/max = {np.percentile(np.abs(sig[..., 0, :]), [50, 99, 100]).round(4).tolist()}, lumen voxels {mask.sum()}, frame times {dt[0]:.3f}..{dt[-1]:.3f} s")
sens = pyvista_zstd.read(str(sd / "Data" / "sensitivity.pv"))
nc = sens["real"].shape[1]
S = grid(sens, "real", nc) + 1j * grid(sens, "imag", nc)
rss = np.sqrt((np.abs(S) ** 2).sum(-1))
log(f"  coil maps: {nc} coils, dims {sens.dimensions}, RSS p1/p50/p99 = {np.percentile(rss, [1, 50, 99]).round(4).tolist()}, zero-voxel fraction {(rss == 0).mean():.3f}")
vencs = np.asarray(f0.field_data["vencs"]).ravel()
dirs = np.asarray(f0.field_data["enc_dirs"]).reshape(-1, 3)
log(f"  vencs {vencs.round(3).tolist()}\n  enc_dirs {dirs.round(3).tolist()}")

# velocity from the encodes (phase difference to the reference, per encode / venc) at the systolic peak
mag = np.abs(sig[..., 0, :])
lum_mag = [mag[..., t][mask].mean() for t in range(len(frames))]
ph = np.angle(sig[..., 1:, :] * np.conj(sig[..., :1, :]))  # x,y,z,enc-1,t
vel_enc = ph * (vencs[1:] / np.pi)[None, None, None, :, None]
spd = np.abs(vel_enc).max(axis=3)
peak = int(np.argmax([spd[..., t][mask].mean() for t in range(len(frames))]))
log(f"  peak frame {peak} (t = {dt[peak]:.3f} s); max |v_enc| in lumen at peak {spd[..., peak][mask].max():.2f} m/s")

c = np.array(np.nonzero(mask)).mean(axis=1).round().astype(int)
planes = [("axial(z)", lambda a: a[:, :, c[2]].T), ("coronal(y)", lambda a: a[:, c[1], :].T), ("sagittal(x)", lambda a: a[c[0], :, :].T)]


def gshow(ax, img, log_=False):
    v = img[np.isfinite(img) & (img > 0)]
    hi = np.percentile(v, 99.5) if v.size else 1
    if log_:
        ax.imshow(np.log10(np.maximum(img, hi * 1e-3)), cmap="gray", vmin=np.log10(hi * 1e-3), vmax=np.log10(hi), origin="lower", aspect="auto")
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=hi, origin="lower", aspect="auto")


fig, ax = plt.subplots(3, 3, figsize=(15, 13))
for r, (name, sl) in enumerate(planes):
    gshow(ax[r, 0], sl(mag[..., peak]))
    ax[r, 0].contour(sl(mask.astype(float)), levels=[0.5], colors="y", linewidths=0.6)
    gshow(ax[r, 1], sl(mag[..., peak]), log_=True)
    ax[r, 2].imshow(sl(np.angle(sig[..., 0, peak])), cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower", aspect="auto")
    ax[r, 0].set_title(f"|ref| {name}, frame {peak} (linear, lumen in yellow)", fontsize=8)
    ax[r, 1].set_title("|ref| log10", fontsize=8)
    ax[r, 2].set_title("phase of reference encode", fontsize=8)
for a in ax.flat:
    a.axis("off")
fig.tight_layout()
fig.savefig(figs / "01_magnitude_phase.png", dpi=80)
plt.close(fig)

fig, ax = plt.subplots(3, nc, figsize=(2.2 * nc, 7))
for k in range(nc):
    gshow(ax[0, k], planes[0][1](np.abs(S[..., k])))
    ax[1, k].imshow(planes[0][1](np.angle(S[..., k])), cmap="twilight", vmin=-np.pi, vmax=np.pi, origin="lower", aspect="auto")
    gshow(ax[2, k], planes[1][1](np.abs(S[..., k])))
    ax[0, k].set_title(f"coil {k}", fontsize=8)
for a in ax.flat:
    a.axis("off")
fig.suptitle("coil sensitivities: |S| axial, phase axial, |S| coronal", fontsize=9)
fig.savefig(figs / "02_coilmaps.png", dpi=80)
plt.close(fig)

fig, ax = plt.subplots(2, nenc - 1, figsize=(2.6 * (nenc - 1), 5.5))
for e in range(nenc - 1):
    for r, (name, sl) in enumerate(planes[:2]):
        ax[r, e].imshow(sl(vel_enc[..., e, peak] * mask), cmap="RdBu_r", vmin=-1.5, vmax=1.5, origin="lower", aspect="auto")
        ax[r, e].set_title(f"enc {e + 1} venc {vencs[e + 1]:.2f} dir {dirs[e + 1].round(2).tolist()}\n{name}", fontsize=7)
        ax[r, e].axis("off")
fig.suptitle(f"velocity along each encoding (phase diff × venc / π) at peak frame {peak}, lumen only", fontsize=9)
fig.savefig(figs / "03_velocity_encodes.png", dpi=80)
plt.close(fig)

# ---------------------------------------------------------------- CFD ground truth
gts = sorted(glob.glob(str(sd / "GroundTruth" / "*.pv")))
if gts:
    g0 = pyvista_zstd.read(gts[0])
    log(f"  ground truth: {len(gts)} frames, dims {g0.dimensions}, spacing {np.round(np.array(g0.spacing) * 1e3, 3)} mm, origin {np.round(np.array(g0.origin) * 1e3, 1).tolist()} mm, fields {list(g0.point_data.keys())}")
    gt_t = [float(np.asarray(pyvista_zstd.read(f).field_data["dtime"]).ravel()[0]) for f in gts[:: max(len(gts) // 10, 1)]]
    log(f"  ground-truth times (subsampled) {np.round(gt_t, 3).tolist()}")
    ig = int(np.argmin([abs(t - dt[peak]) for t in [float(np.asarray(pyvista_zstd.read(f).field_data['dtime']).ravel()[0]) for f in gts]]))
    gp = pyvista_zstd.read(gts[ig])
    U = grid(gp, "U", 3)
    gm = grid(gp, "mask", 0) > 0
    sp = np.sqrt((U ** 2).sum(-1))
    log(f"  GT frame {ig}: lumen voxels {gm.sum()}, max |U| {sp[gm].max():.2f} m/s, mean {sp[gm].mean():.2f}")
    cg = np.array(np.nonzero(gm)).mean(axis=1).round().astype(int)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(sp[:, :, cg[2]].T, origin="lower", cmap="magma", vmin=0, vmax=np.percentile(sp[gm], 99.5))
    ax[0].set_title(f"CFD |U| axial, frame {ig}", fontsize=8)
    ax[1].imshow(sp[:, cg[1], :].T, origin="lower", cmap="magma", vmin=0, vmax=np.percentile(sp[gm], 99.5), aspect="auto")
    ax[1].set_title("CFD |U| coronal", fontsize=8)
    ax[2].imshow(spd[:, :, c[2], peak].T * mask[:, :, c[2]].T, origin="lower", cmap="magma", vmin=0, vmax=np.percentile(sp[gm], 99.5))
    ax[2].set_title(f"MRI max|v_enc| axial, frame {peak}", fontsize=8)
    ax[3].plot(dt, lum_mag / np.max(lum_mag), "k.-", label="lumen |ref| (norm.)")
    ax[3].plot(dt, [spd[..., t][mask].mean() for t in range(len(frames))], "r.-", label="lumen mean max|v_enc| (m/s)")
    ax[3].legend(fontsize=7)
    ax[3].set_xlabel("t (s)")
    for a in ax[:3]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(figs / "04_ground_truth.png", dpi=80)
    plt.close(fig)

# ---------------------------------------------------------------- cohort
meta = json.loads((sd / "meta.json").read_text())
log(f"  meta: {json.dumps({k: meta[k] for k in meta if k in ('source_geometry', 'source_background', 'breathing_motion')})}, acquisition {json.dumps(meta.get('acquisition', {}))[:300]}")
log(f"  wall motion: {json.dumps(meta.get('wall_motion', {}))[:300]}")
for v in ["7p_2Venc", "7p_2Venc_Breathing", "7p_2Venc_Super", "13p_2Venc_ICOSA"]:
    ms = [json.loads(p.read_text()) for p in sorted((DATA / v).glob("*/meta*.json")) if p.name in ("meta.json", "meta_mid.json")]
    if not ms:
        continue
    grades = Counter(m.get("hemodynamics", {}).get("stenosis_grade") for m in ms)
    bgs = Counter(m.get("source_background") for m in ms)
    res = Counter(tuple(np.round(m.get("acquisition", {}).get("spatial_resolution_mm", [0, 0, 0]), 2)) for m in ms)
    wm = sum(bool(m.get("wall_motion", {}).get("wall_motion_simulated")) for m in ms)
    log(f"cohort {v}: {len(ms)} subjects, grades {dict(grades)}, {len(bgs)} distinct backgrounds, resolutions {dict(res)}, wall motion in {wm}")
(out / "summary.txt").write_text("\n".join(lines) + "\n")
with open(figs / "index.md", "w") as fh:
    fh.write(f"# RACLETTE {variant}/{subject}\n\n```\n" + "\n".join(lines) + "\n```\n\n")
    for n, cap in [("01_magnitude_phase.png", "Reference-encode magnitude (linear, log) and phase, three planes through the lumen centroid; lumen mask in yellow."),
                   ("02_coilmaps.png", "Coil sensitivity maps (magnitude/phase axial, magnitude coronal)."),
                   ("03_velocity_encodes.png", "Velocity along each encoding direction at peak systole (lumen only)."),
                   ("04_ground_truth.png", "CFD ground truth |U| vs MRI-derived speed; lumen magnitude and speed over the cycle.")]:
        fh.write(f"## {n}\n{cap}\n\n![{n}]({n})\n\n")
