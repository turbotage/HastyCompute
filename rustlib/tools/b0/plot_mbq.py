"""Figures of an `mbq_b0` run (model-based B0 / R2* / fat).
    rustlib/.venv/bin/python -I rustlib/tools/b0/plot_mbq.py <run>
mbq_01_maps     axial / coronal / sagittal: ω, R2*, fat fraction (estimate, truth, error), |water| + |fat|
mbq_02_conv     objective (total and data term) per iteration
mbq_03_hist     error histograms in signal voxels
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, GRID = "#1f1f1e", "#e4e3dd"


def main():
    run = Path(sys.argv[1])
    S = json.loads((run / "summary.json").read_text())
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    L = lambda n: np.load(run / f"{n}.npy")  # noqa: E731
    om, r2, ff, wm, fm, sup = (L(n) for n in ("maps_omega_hz", "maps_r2", "maps_ff", "maps_water", "maps_fat", "support"))
    tom, tr2, tff = (L(n) for n in ("truth_omega_hz", "truth_r2", "truth_ff"))
    mag = wm + fm
    m99 = np.nanpercentile(mag[sup > 0], 99)
    sig = (sup > 0) & (mag > 0.1 * m99)
    nx, ny, nz = om.shape
    cuts = {"axial": (lambda a: a[:, :, nz // 2].T), "coronal": (lambda a: a[:, ny // 2, :].T[::-1]), "sagittal": (lambda a: a[nx // 2, :, :].T[::-1])}
    lim = max(20.0, float(np.nanpercentile(np.abs(tom[sig]), 99)))
    M = lambda a: np.where(sig, a, np.nan)  # noqa: E731
    cols = [("ω (Hz)", M(om), "RdBu_r", (-lim, lim)), ("ω truth", M(tom), "RdBu_r", (-lim, lim)), ("ω − truth", M(om - tom), "RdBu_r", (-10, 10)),
            ("R2* (1/s)", M(r2), "magma", (0, 150)), ("R2* truth", M(tr2), "magma", (0, 150)), ("R2* − truth", M(r2 - tr2), "RdBu_r", (-30, 30)),
            ("FF", M(ff), "viridis", (0, 1)), ("FF truth", M(tff), "viridis", (0, 1)), ("FF − truth", M(ff - tff), "RdBu_r", (-0.2, 0.2)),
            ("|w| + |f|", np.where(sup > 0, mag, np.nan), "gray", (0, m99))]
    fig, axs = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 8.2))
    for r, (cn, cf) in enumerate(cuts.items()):
        for c, (tl, a, cm, (lo, hi)) in enumerate(cols):
            ax = axs[r, c]
            im = ax.imshow(cf(a), cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(tl, fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(cn, fontsize=9)
            if r == 2:
                fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)
    e = S["errors"]
    fmt = lambda k: "n/a" if e[k] is None else f"median {e[k]['median']:.3g}, RMSE {e[k]['rmse']:.3g}, |·| p95 {e[k]['abs_p95']:.3g}"  # noqa: E731
    fig.suptitle(f"model-based B0 / R2* / fat ({S['n_spline']} spline coefficients, {S['knot_mm']} mm knots, {S['res'] * 1e3:.1f} mm grid): "
                 f"ω {fmt('omega_hz')} Hz; R2* {fmt('r2')} 1/s; FF {fmt('ff')}", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(figs / "mbq_01_maps.png", dpi=100)
    plt.close(fig)

    rows = list(csv.DictReader(open(run / "metrics.csv")))
    if rows:
        fig, ax = plt.subplots(figsize=(9, 4))
        it = np.array([int(r["iter"]) for r in rows])
        ax.semilogy(it, [float(r["f"]) for r in rows], color="#2a78d6", label="objective")
        ax.semilogy(it, [float(r["f_data"]) for r in rows], color="#eb6834", label="data term")
        ax.set_xlabel("iteration (ρ-only phase, then joint)")
        ax.grid(True, color=GRID)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figs / "mbq_02_conv.png", dpi=100)
        plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (a, t, lab, lm) in zip(axs, [(om, tom, "ω − truth (Hz)", 30), (r2, tr2, "R2* − truth (1/s)", 100), (ff, tff, "FF − truth", 0.5)]):
        dd = (a - t)[sig & np.isfinite(t) & np.isfinite(a)]
        ax.hist(np.clip(dd, -lm, lm), bins=200, color="#2a78d6")
        ax.set_yscale("log")
        ax.set_xlabel(lab)
        ax.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(figs / "mbq_03_hist.png", dpi=100)
    plt.close(fig)
    print("wrote", figs)


if __name__ == "__main__":
    main()
