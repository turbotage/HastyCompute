"""Figures of a `recon_b0` run (B0 / R2* / fat maps from the simulated multi-echo train).
    rustlib/.venv/bin/python -I rustlib/tools/b0/plot_b0recon.py <run>
01_maps        axial / coronal / sagittal: ω estimate, truth, error, R2*, fat fraction, water + fat magnitude
02_omega_err   histogram of ω − truth in signal voxels, estimate vs truth
03_echoes      echo images (magnitude, phase) of the first TE shift at a few echo times
04_conc        the scan's own concomitant field as an equivalent off-resonance (removed before the fit)
05_selfgate    (free breathing) surrogate vs the true breathing state, bins
06_bins        (free breathing) ω and |water| + |fat| per breathing bin, and ω(bin) − ω(bin 0): the dynamic maps
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def main():
    run = Path(sys.argv[1])
    S = json.loads((run / "summary.json").read_text())
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    L = lambda n: np.load(run / f"{n}.npy")  # noqa: E731  (x, y, z) arrays
    om4, tru, r24, ff4, wm4, fm4, sup = (L(n) for n in ("maps_omega_hz", "truth_omega_hz", "maps_r2", "maps_ff", "maps_water", "maps_fat", "support"))
    if om4.ndim == 3:
        om4, r24, ff4, wm4, fm4 = (a[..., None] for a in (om4, r24, ff4, wm4, fm4))
    nb = om4.shape[3]
    om, r2, ff, wm, fm = (a[..., 0] for a in (om4, r24, ff4, wm4, fm4))  # bin 0 = end-expiration
    mag = wm + fm
    m99 = np.nanpercentile(mag[sup > 0], 99)
    sig = (sup > 0) & (mag > 0.1 * m99)
    nx, ny, nz = om.shape
    cuts = {"axial": (lambda a: a[:, :, nz // 2].T), "coronal": (lambda a: a[:, ny // 2, :].T[::-1]), "sagittal": (lambda a: a[nx // 2, :, :].T[::-1])}
    err = np.where(sig, om - tru, np.nan)
    lim = max(20.0, float(np.nanpercentile(np.abs(tru[sig]), 99)))
    cols = [("ω estimate (Hz)", np.where(sig, om, np.nan), "RdBu_r", (-lim, lim)),
            ("ω truth (Hz)", np.where(sig, tru, np.nan), "RdBu_r", (-lim, lim)),
            ("ω − truth (Hz)", err, "RdBu_r", (-10, 10)),
            ("R2* (1/s)", np.where(sig, r2, np.nan), "magma", (0, 150)),
            ("fat fraction", np.where(sig, ff, np.nan), "viridis", (0, 1)),
            ("|water| + |fat|", np.where(sup > 0, mag, np.nan), "gray", (0, m99))]
    fig, axs = plt.subplots(3, len(cols), figsize=(3.1 * len(cols), 9.0))
    for r, (cn, cf) in enumerate(cuts.items()):
        for c, (tl, a, cm, (lo, hi)) in enumerate(cols):
            ax = axs[r, c]
            im = ax.imshow(cf(a), cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(tl, fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(cn, fontsize=9, color=INK)
            if r == 2:
                fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
    e = S["omega_err_hz"]
    fig.suptitle(f"B0 / R2* / fat from the multi-echo train ({S['n_sets']} echo images, {S['iters']} P²nP iterations, LLR κ {S['kappa']}); "
                 f"ω error: median {e['median']:.2f} Hz, RMSE {e['rmse']:.2f} Hz, |·| p95 {e['abs_p95']:.2f} Hz, > 150 Hz: {e['gt150']} voxels", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(figs / "01_maps.png", dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    d = (om - tru)[sig]
    axs[0].hist(np.clip(d, -20, 20), bins=200, color="#2a78d6")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("ω − truth (Hz), clipped to ±20", fontsize=9)
    axs[0].set_title(f"{d.size} signal voxels", fontsize=9)
    sub = np.random.default_rng(0).choice(d.size, min(d.size, 40000), replace=False)
    axs[1].scatter(tru[sig][sub], om[sig][sub], s=1, color="#2a78d6", alpha=0.3)
    axs[1].plot([-lim, lim], [-lim, lim], color=INK, lw=0.8)
    axs[1].set_xlabel("truth (Hz)", fontsize=9)
    axs[1].set_ylabel("estimate (Hz)", fontsize=9)
    for ax in axs:
        ax.grid(True, color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(figs / "02_omega_err.png", dpi=100)
    plt.close(fig)

    E = L("echoes_mag_phase")
    ne = E.shape[3] // 2
    pick = sorted(set([0, 1, 2, ne // 4, ne // 2, ne - 1]))
    te = S["te"]
    fig, axs = plt.subplots(2, len(pick), figsize=(2.8 * len(pick), 6))
    vmax = np.percentile(E[..., 0][sup > 0], 99.5)
    for c, e_ in enumerate(pick):
        axs[0, c].imshow(E[:, :, nz // 2, 2 * e_].T, cmap="gray", vmin=0, vmax=vmax)
        axs[1, c].imshow(np.where(sup[:, :, nz // 2] > 0, E[:, :, nz // 2, 2 * e_ + 1], np.nan).T, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axs[0, c].set_title(f"echo {e_}: TE {te[e_] * 1e3:.2f} ms", fontsize=9)
        for r in range(2):
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
    axs[0, 0].set_ylabel("magnitude", fontsize=9)
    axs[1, 0].set_ylabel("phase (conc. corrected)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "03_echoes.png", dpi=100)
    plt.close(fig)

    # concomitant pseudo off-resonance: d(w̄·m)/dt over the first TE shift's echoes
    w = np.array(S["wbar"])[:ne]
    t = np.array(te[:ne])
    slope = np.polyfit(t, w, 1)[0] / (2 * np.pi)  # Hz per monomial unit
    res = S["res"]
    xs = (np.arange(nx) - nx // 2) * res
    ys_ = (np.arange(ny) - ny // 2) * res
    zs = (np.arange(nz) - nz // 2) * res
    X, Y, Z = np.meshgrid(xs, ys_, zs, indexing="ij")
    mono = [Z * Z, X * X + Y * Y, X * Z, Y * Z]
    conc = sum(slope[a] * mono[a] for a in range(4))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (cn, cf) in zip(axs, cuts.items()):
        im = ax.imshow(cf(np.where(sup > 0, conc, np.nan)), cmap="RdBu_r", vmin=-np.nanmax(np.abs(conc)), vmax=np.nanmax(np.abs(conc)))
        ax.set_title(cn, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.04, label="Hz")
    fig.suptitle("the B0 scan's own concomitant field as an equivalent off-resonance (removed from the echo images before the fit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "04_conc.png", dpi=100)
    plt.close(fig)
    if (run / "selfgate.npy").exists():
        G = np.load(run / "selfgate.npy")
        fig, ax = plt.subplots(figsize=(13, 3.6))
        ax.plot(G[:, 0], G[:, 1], color="#2a78d6", lw=0.8, label="self-gating surrogate (FID, first crossing)")
        if np.isfinite(G[:, 2]).any():
            ax.plot(G[:, 0], G[:, 2], color=INK, lw=0.8, ls="--", label="true breathing state r(t)")
        if nb > 1:
            ax.scatter(G[:, 0], -0.15 - 0.05 * G[:, 3], s=2, c=G[:, 3], cmap="viridis", label="bin")
        ax.set_xlabel("s", fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        r = S.get("resp", {})
        ax.set_title(f"breathing: leading PC {100 * r.get('gate_explained', 0):.0f} % of the band-passed FID variance, "
                     f"correlation with truth {r.get('gate_corr_truth') if r.get('gate_corr_truth') is None else round(r['gate_corr_truth'], 3)}; "
                     f"{nb} bins ({r.get('mode', '')})", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(figs / "05_selfgate.png", dpi=100)
        plt.close(fig)
    if nb > 1:
        magb = wm4 + fm4
        fig, axs = plt.subplots(3, nb, figsize=(2.9 * nb, 8.4), squeeze=False)
        for b in range(nb):
            sb = (sup > 0) & (magb[..., b] > 0.1 * m99)
            for r_, (a, cm, lo, hi, tl) in enumerate([(np.where(sup > 0, magb[..., b], np.nan), "gray", 0, m99, "|water|+|fat|"),
                                                      (np.where(sb, om4[..., b], np.nan), "RdBu_r", -lim, lim, "ω (Hz)"),
                                                      (np.where(sb & sig, om4[..., b] - om, np.nan), "RdBu_r", -15, 15, "ω − ω(bin 0) (Hz)")]):
                im = axs[r_, b].imshow(cuts["coronal"](a), cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
                axs[r_, b].set_xticks([])
                axs[r_, b].set_yticks([])
                if b == 0:
                    axs[r_, b].set_ylabel(tl, fontsize=9)
                if r_ == 2:
                    fig.colorbar(im, ax=axs[r_, b], orientation="horizontal", fraction=0.05, pad=0.04)
            mr = S.get("resp", {}).get("bin_mean_true_r", [None] * nb)[b]
            axs[0, b].set_title(f"bin {b}" + (f" (true r {mr:.2f})" if mr is not None and mr == mr else ""), fontsize=9)
        fig.suptitle("dynamic maps per breathing bin (coronal); the last row shows motion + field change against end-expiration", fontsize=10)
        fig.tight_layout()
        fig.savefig(figs / "06_bins.png", dpi=100)
        plt.close(fig)
    print("wrote", figs)


if __name__ == "__main__":
    main()
