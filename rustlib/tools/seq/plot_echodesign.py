"""Figures of an `echodesign` run (B0 / R2* / fat echo design by the CRLB per unit scan time).
    rustlib/.venv/bin/python -I rustlib/tools/seq/plot_echodesign.py <run>
01_<tissue>.png: scan-time factor vs the reference (K = 16 trapezoid, all echoes) against the train length K, for ω (= R2*,
                 the two are the real / imaginary part of one complex rate), fat fraction and the swap / alias margin;
                 idealised long readouts as dashed reference levels.
02_tissues.png:  the same factors per tissue at K = 16 and for the long readouts.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SER = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]  # variants, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
VARIANTS = [("Trapezoid", 1, "all", "trapezoid, all echoes"), ("Trapezoid", 2, "all", "trapezoid + ½-lobe TE shift"),
            ("Trapezoid", 2, "even", "+ ½ shift, flow-comp. echoes only"), ("Sine", 1, "all", "sine lobes")]
METRICS = [("om", "ω and R2*"), ("ff", "fat fraction"), ("swap", "water–fat swap / ω alias")]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def tf(row, tissue, m):
    for t in row["tissues"]:
        if t["tissue"] == tissue:
            v = t["time_factor"][m]
            return v if np.isfinite(v) else np.nan
    return np.nan


def main():
    run = Path(sys.argv[1])
    s = json.loads((run / "echodesign.json").read_text())
    rows = s["rows"]
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    tissues = [t["name"] for t in s["tissues"]]
    for ti, tissue in enumerate(["myocardium", "blood", "fat", "lung"]):
        fig, axs = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
        for c, res in enumerate([2.0, 3.0]):
            for r_, (m, mlab) in enumerate(METRICS):
                ax = axs[r_, c]
                style(ax)
                for vi, (shape, nsh, sel, lab) in enumerate(VARIANTS):
                    pts = sorted((r["info"]["K"], tf(r, tissue, m)) for r in rows
                                 if r["res_mm"] == res and r["info"].get("shape") == shape and r["info"].get("n_shift") == nsh and r["echoes"] == sel)
                    if not pts:
                        continue
                    k, v = np.array(pts).T
                    ax.plot(k, v, "-o", color=SER[vi], lw=2, ms=5, label=lab, mec="white", mew=0.8)
                for r in rows:
                    if r["res_mm"] == res and r["info"].get("t_ro_ms"):
                        v = tf(r, tissue, m)
                        ax.axhline(v, color=MUTED, ls="--", lw=1)
                        ax.text(33, v, f"long {r['info']['t_ro_ms']:.0f} ms (idealised)", color=MUTED, fontsize=7, va="bottom", ha="right")
                ax.axhline(1.0, color=INK, lw=0.8)
                ax.set_yscale("log")
                ax.set_ylim(0.1, 300)
                ax.set_xscale("log", base=2)
                ax.set_xticks([4, 8, 16, 32])
                ax.set_xticklabels(["4", "8", "16", "32"])
                if c == 0:
                    ax.set_ylabel(f"{mlab}\nscan-time factor", color=INK, fontsize=9)
                if r_ == 0:
                    ax.set_title(f"{res:.0f} mm", color=INK, fontsize=10)
                if r_ == 2:
                    ax.set_xlabel("lobes per TR (K)", color=INK, fontsize=9)
        axs[0, 0].legend(fontsize=7.5, frameon=False, loc="upper right")
        fig.suptitle(f"{tissue}: scan time needed for the same precision, relative to K = 16 trapezoid (lower is better)", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(figs / f"01_{tissue.replace(' ', '_')}.png", dpi=110)
        plt.close(fig)

    # per tissue at K = 16 and the long readouts
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    sel = [(f"K=16 {lab}", lambda r, sh=sh, n=n, e=e: r["res_mm"] == 2.0 and r["info"].get("K") == 16 and r["info"].get("shape") == sh and r["info"].get("n_shift") == n and r["echoes"] == e)
           for sh, n, e, lab in VARIANTS]
    sel += [(f"long {t:.0f} ms (idealised)", lambda r, t=t: r["res_mm"] == 2.0 and r["info"].get("t_ro_ms") == t) for t in (4.0, 8.0)]
    cols = SER + ["#9a9890", "#5f5e58"]
    x = np.arange(len(tissues))
    w = 0.8 / len(sel)
    for ax, (m, mlab) in zip(axs, METRICS):
        style(ax)
        for i, (lab, f) in enumerate(sel):
            r = next(r for r in rows if f(r))
            v = [tf(r, t, m) for t in tissues]
            ax.bar(x + (i - (len(sel) - 1) / 2) * w, v, w * 0.92, color=cols[i], label=lab)
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 300)
        ax.set_xticks(x)
        ax.set_xticklabels(tissues, fontsize=8, rotation=20)
        ax.set_title(mlab, color=INK, fontsize=10)
    axs[0].set_ylabel("scan-time factor vs K = 16 trapezoid (2 mm)", color=INK, fontsize=9)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "02_tissues.png", dpi=110)
    plt.close(fig)
    print("wrote", figs)


if __name__ == "__main__":
    main()
