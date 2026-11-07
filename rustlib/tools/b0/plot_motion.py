"""Figures of `motionfit` on a breathing-resolved recon_b0 run.
    rustlib/.venv/bin/python -I rustlib/tools/b0/plot_motion.py <run>
motion_01_field   fitted displacement S at r = 1 (mm): SI / AP / LR components, coronal and sagittal
motion_02_bins    per bin: measured magnitude, reference pulled back along r_b S, residual with and without the model
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    run = Path(sys.argv[1])
    M = json.loads((run / "motion.json").read_text())
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    S = np.load(run / "motion_S_mm.npy")
    P = np.load(run / "motion_pred.npy")
    B = np.load(run / "motion_bins.npy")
    sup = np.load(run / "support.npy") > 0
    nx, ny, nz, nb = B.shape
    cor = lambda a: a[:, ny // 2, :].T[::-1]  # noqa: E731
    sag = lambda a: a[nx // 2, :, :].T[::-1]  # noqa: E731
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    lim = max(1.0, float(np.nanpercentile(np.abs(S[sup]), 99.5)))
    for r, (nm, cf) in enumerate([("coronal", cor), ("sagittal", sag)]):
        for c, (a, lab) in enumerate([(2, "SI (z)"), (1, "AP (y)"), (0, "LR (x)")]):
            im = axs[r, c].imshow(cf(np.where(sup, S[..., a], np.nan)), cmap="RdBu_r", vmin=-lim, vmax=lim)
            axs[r, c].set_title(f"{nm}: S {lab} (mm at r = 1)", fontsize=9)
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            fig.colorbar(im, ax=axs[r, c], fraction=0.04)
    t = M.get("truth") or {}
    fig.suptitle(f"shared respiratory field: relative residual {M['rel_resid']:.3f}, min Jacobian {M['min_jacobian']:.3f}"
                 + (f"; vs phantom truth: RMSE {t['rmse_mm']:.2f} mm (truth RMS {t['truth_rms_mm']:.2f}), corr {t['corr']:.3f}, "
                    f"SI max {t['si_max_est_mm']:.1f} / {t['si_max_true_mm']:.1f} mm" if t else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "motion_01_field.png", dpi=100)
    plt.close(fig)
    fig, axs = plt.subplots(4, nb, figsize=(2.6 * nb, 10), squeeze=False)
    v = np.percentile(B[..., 0][sup], 99)
    for b in range(nb):
        rows = [(B[..., b], "gray", 0, v, "measured"), (P[..., b], "gray", 0, v, "reference pulled back"),
                (B[..., b] - B[..., 0], "RdBu_r", -0.5 * v, 0.5 * v, "measured − bin 0"), (B[..., b] - P[..., b], "RdBu_r", -0.5 * v, 0.5 * v, "measured − model")]
        for r, (a, cm, lo, hi, lab) in enumerate(rows):
            axs[r, b].imshow(cor(np.where(sup, a, np.nan)), cmap=cm, vmin=lo, vmax=hi)
            axs[r, b].set_xticks([])
            axs[r, b].set_yticks([])
            if b == 0:
                axs[r, b].set_ylabel(lab, fontsize=9)
        axs[0, b].set_title(f"bin {b}, r {M['r_gate'][b]:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "motion_02_bins.png", dpi=100)
    plt.close(fig)
    print("wrote", figs)


if __name__ == "__main__":
    main()
