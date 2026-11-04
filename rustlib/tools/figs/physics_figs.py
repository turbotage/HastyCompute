"""Figures of what the reconstruction uses for one simulation (arrays written by `physfigs`).

    python -I physics_figs.py <physics_dir> [<sim_dir>]

Writes <physics_dir>/figs/*.png and index.md: B0, gradient nonlinearity, concomitant fields, sequence, sampling
coverage, coils and noise. The lumen (simulator truth, fraction > 0.9) is outlined in yellow.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P = Path(sys.argv[1])
S = Path(sys.argv[2]) if len(sys.argv) > 2 else P.parent
F = P / "figs"
F.mkdir(exist_ok=True)
J = json.loads((P / "summary.json").read_text())
ld = lambda n: np.load(P / n)  # noqa: E731
lum = None
if (S / "truth_lumen_fraction.npy").exists():
    fr = np.load(S / "truth_lumen_fraction.npy")
    lum = (fr if fr.ndim == 3 else fr[..., 0]) > 0.9
    if lum.shape != tuple(J["dims"]):
        lum = None
X, Y, Z = J["dims"]
zs = int(np.argmax(lum.sum(axis=(0, 1)))) if lum is not None else Z // 2
ys = int(np.argmax(lum.sum(axis=(0, 2)))) if lum is not None else Y // 2
lines = [f"# Reconstruction inputs: {Path(J['sim']).name}\n", f"_{J['note']}_\n"]


def views(fig, axs, vol, title, cmap="RdBu_r", sym=True, lim=None, unit=""):
    a = np.asarray(vol, np.float64)
    lim = lim or float(np.percentile(np.abs(a), 99.5)) or 1.0
    kw = dict(cmap=cmap, vmin=-lim if sym else 0, vmax=lim, origin="lower", interpolation="nearest")
    for ax, sl, t, ext in [(axs[0], a[:, :, zs].T, f"axial z={zs}", None), (axs[1], a[:, ys, :].T, f"coronal y={ys}", None)]:
        im = ax.imshow(sl, aspect="auto" if t.startswith("coronal") else "equal", **kw)
        if lum is not None:
            m = lum[:, :, zs].T if t.startswith("axial") else lum[:, ys, :].T
            ax.contour(m.astype(float), levels=[0.5], colors="y", linewidths=0.6)
        ax.set_title(f"{title}: {t}", fontsize=8)
        ax.axis("off")
    fig.colorbar(im, ax=list(axs), shrink=0.8, label=unit)


def save(fig, name, caption):
    fig.savefig(F / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    lines.append(f"## {caption}\n\n![{name}](figs/{name})\n")


def s(k, unit, nd=3):
    v = J[k]
    return f"p1 {v['p1']:.{nd}g}, median {v['p50']:.{nd}g}, p99 {v['p99']:.{nd}g}, max |·| {v['max_abs']:.{nd}g} {unit}"


# ── B0 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
b0 = ld("b0_hz.npy")
fig = plt.figure(figsize=(14, 4.2))
ax = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
views(fig, ax[:2], b0, "B0 off-resonance", unit="Hz")
ax[2].hist(b0.ravel(), bins=200, color="0.4", log=True, label="FOV")
if lum is not None:
    ax[2].hist(b0[lum], bins=100, color="tab:orange", log=True, alpha=0.8, label="lumen")
ax[2].set_xlabel("Hz")
ax[2].legend(fontsize=8)
save(fig, "01_b0.png", f"B0 (field map × γ̄B0) the operator applies. FOV: {s('b0_hz_fov', 'Hz')}; lumen: {s('b0_hz_lumen', 'Hz')}. "
     f"Over the {J['readout_s'] * 1e3:.2f} ms readout the worst voxel accrues {J['b0_readout_phase_rad_p99']:.1f} rad.")

# ── GNL ────────────────────────────────────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7.5))
ax = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
views(fig, ax[:2], ld("gnl_disp_mm.npy"), "GNL displacement", cmap="magma", sym=False, unit="mm")
views(fig, ax[2:], ld("gnl_vscale_pct.npy"), "velocity scale error", cmap="magma", sym=False, unit="%")
save(fig, "02_gnl.png", f"Gradient nonlinearity: voxel displacement (true − nominal) — FOV {s('gnl_disp_mm_fov', 'mm')}, lumen {s('gnl_disp_mm_lumen', 'mm')}; "
     f"velocity-encoding scale error max|σ(J)−1| — FOV {s('gnl_vscale_pct_fov', '%')}, lumen {s('gnl_vscale_pct_lumen', '%')}.")

# ── concomitant ────────────────────────────────────────────────────────────────────────────────────────────────────
vo = ld("conc_voffset.npy")
ne1 = vo.shape[3]
fig, axs = plt.subplots(2, ne1, figsize=(3.2 * ne1, 6.2), squeeze=False)
lim = float(np.percentile(np.abs(vo), 99.5)) or 1e-3
for e in range(ne1):
    for r, (sl, t) in enumerate([(vo[:, :, zs, e].T, "axial"), (vo[:, ys, :, e].T, "coronal")]):
        im = axs[r, e].imshow(sl, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower", aspect="equal" if r == 0 else "auto")
        if lum is not None:
            axs[r, e].contour((lum[:, :, zs].T if r == 0 else lum[:, ys, :].T).astype(float), levels=[0.5], colors="y", linewidths=0.5)
        axs[r, e].set_title(f"encode {e + 1} {t}", fontsize=8)
        axs[r, e].axis("off")
fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7, label="m/s")
save(fig, "03_concomitant_velocity.png", f"Concomitant (Maxwell) phase of each encode relative to the reference at the first ADC sample, as velocity along the encode: FOV {s('conc_voffset_mps_fov', 'm/s')}; lumen {s('conc_voffset_mps_lumen', 'm/s')}.")
fig = plt.figure(figsize=(10, 4.2))
ax = [fig.add_subplot(1, 2, i + 1) for i in range(2)]
views(fig, ax, ld("conc_ro_rad.npy"), "concomitant phase over the readout", cmap="magma", sym=False, unit="rad")
save(fig, "04_concomitant_readout.png", f"Concomitant phase accrued during the readout (reference encode, middle partition): FOV {s('conc_readout_rad_fov', 'rad')}; lumen {s('conc_readout_rad_lumen', 'rad')}.")

# ── sequence ───────────────────────────────────────────────────────────────────────────────────────────────────────
kt = ld("k_traj.npy")
g, t = ld("ro_g.npy"), ld("ro_t.npy")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for r in range(kt.shape[0]):
    ax[0].plot(kt[r, :, 0], kt[r, :, 1], lw=0.4)
ax[0].plot(kt[0, :, 0], kt[0, :, 1], "k", lw=0.8)
ax[0].set_aspect("equal")
ax[0].set_title(f"k-space: {J['n_interleaves']} spiral interleaves (middle partition), cycles/m", fontsize=8)
ax[1].plot(t * 1e3, g * 1e3)
ax[1].legend(["Gx", "Gy", "Gz"], fontsize=7)
ax[1].set_xlabel("ms")
ax[1].set_ylabel("mT/m")
ax[1].set_title(f"readout gradient (dwell {J['dwell_s'] * 1e6:.0f} µs, {J['n_adc']} samples, {J['readout_s'] * 1e3:.2f} ms)", fontsize=8)
slew = np.linalg.norm(np.diff(g, axis=0), axis=1) / np.diff(t).mean()
ax[2].plot(t[1:] * 1e3, slew, "0.3", lw=0.7)
ax[2].set_xlabel("ms")
ax[2].set_ylabel("T/m/s")
ax[2].set_title("|slew rate|", fontsize=8)
save(fig, "05_sequence.png", f"Sequence: stack of spirals, {J['n_interleaves']} interleaves × {J['n_partitions']} kz partitions, TR {J['tr_s'] * 1e3:.2f} ms, TE {J['te_s'] * 1e3:.2f} ms, "
     f"{J['encoding']} (VENC {J['venc']} m/s; kv per encode {[round(k, 2) for k in J['kv_rad_per_mps']]} rad/(m/s)), {J['scan_time_s']:.0f} s, {J['n_readouts']} readouts.")

# ── coverage ───────────────────────────────────────────────────────────────────────────────────────────────────────
cov, covz, cp = ld("coverage.npy"), ld("coverage_kz.npy"), ld("card_phase.npy")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
im = ax[0].imshow(cov.T, aspect="auto", origin="lower", cmap="viridis")
fig.colorbar(im, ax=ax[0])
ax[0].set_xlabel("frame")
ax[0].set_ylabel("encode")
ax[0].set_title("readouts per frame and encode", fontsize=8)
im = ax[1].imshow(covz.T, aspect="auto", origin="lower", cmap="viridis")
fig.colorbar(im, ax=ax[1])
ax[1].set_xlabel("frame")
ax[1].set_ylabel("kz partition")
ax[1].set_title("reference encode: interleaves per partition and frame", fontsize=8)
ax[2].hist(cp, bins=100, color="0.4")
ax[2].set_xlabel("cardiac phase")
ax[2].set_title(f"cardiac phase of the readouts (RR {J['rr_mean_s']:.2f} s ± jitter)", fontsize=8)
save(fig, "06_coverage.png", f"Sampling per frame: {J['readouts_per_frame_encode']:.1f} readouts per frame and encode against {J['n_interleaves'] * J['n_partitions']} for a full volume → acceleration R ≈ {J['acceleration_per_frame']:.1f}.")

# ── coils / noise ──────────────────────────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.2))
ax = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
views(fig, ax[:2], ld("coil_sos.npy"), "coil sensitivity (whitened, root-sum-of-squares)", cmap="gray", sym=False)
im = ax[2].imshow(ld("psi.npy"), cmap="RdBu_r")
fig.colorbar(im, ax=ax[2])
ax[2].set_title(f"noise covariance Ψ ({J['coils']} coils)", fontsize=8)
save(fig, "07_coils.png", "Coil sensitivities (Biot–Savart, after Ψ^{-1/2} whitening) and the noise covariance.")

(P / "index.md").write_text("\n".join(lines))
print(f"{len(list(F.glob('*.png')))} figures in {F}")
