"""Figure of a `spoilcheck` run: steady-state signal / ideal, TR-to-TR fluctuation and worst velocity error per spoiling
variant, tissue and RF increment; plus |S|/ideal traces of a few variants.
    rustlib/.venv/bin/python -I rustlib/tools/seq/plot_spoilcheck.py <run>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SER = ["#2a78d6", "#eb6834", "#1baf7a"]  # tissues, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def main():
    run = Path(sys.argv[1])
    s = json.loads((run / "spoilcheck.json").read_text())
    rows = s["rows"]
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    tissues = list(dict.fromkeys(r["tissue"] for r in rows))
    incs = sorted(set(r["rf_inc_deg"] for r in rows), reverse=True)
    fig, axs = plt.subplots(1, 3, figsize=(17, 0.42 * len(variants) + 2.2), sharey=True)
    y = {v: len(variants) - 1 - i for i, v in enumerate(variants)}
    for ax, key, lab, logx in [(axs[0], "rel_signal", "steady-state |S| / ideal spoiled", False),
                               (axs[1], "fluct", "TR-to-TR fluctuation of |S| (std / mean)", True),
                               (axs[2], "v_err_max", "worst pair velocity error (mm/s)", True)]:
        for r in rows:
            ti = tissues.index(r["tissue"])
            val = r[key] * (1e3 if key == "v_err_max" else 1.0)
            if logx:
                val = max(val, 1e-3)
            off = (ti - 1) * 0.22
            ax.scatter(val, y[r["variant"]] + off, s=26, color=SER[ti], marker="o" if r["rf_inc_deg"] == incs[0] else "^",
                       edgecolors="white", linewidths=0.6, zorder=3)
        if logx:
            ax.set_xscale("log")
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].axvline(1.0, color=MUTED, lw=0.8)
    axs[0].set_yticks(list(y.values()))
    axs[0].set_yticklabels(list(y.keys()), fontsize=7)
    handles = [plt.Line2D([], [], ls="", marker="o", color=SER[i], label=t) for i, t in enumerate(tissues)]
    handles += [plt.Line2D([], [], ls="", marker=m, color=MUTED, label=f"RF increment {inc:g}°") for m, inc in zip(["o", "^"], incs)]
    axs[2].legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"Spoiling check (multi-isochromat steady state, actual TR moments): {s['res']*1e3:.0f} mm voxel, flip {s['flip_deg']:.0f}°, "
                 f"T_adc {s['t_adc']*1e3:.0f} ms, {s['scheme']}, VENC_lo {s['venc_lo']} m/s", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(run / "spoilcheck.png", dpi=130)
    plt.close(fig)

    tr = np.load(run / "traces.npy")
    pick = [i for i, r in enumerate(rows) if r["tissue"] == "blood static" and r["rf_inc_deg"] == incs[0]]
    fig, ax = plt.subplots(figsize=(13, 4))
    for j, i in enumerate(pick):
        t = tr[i][np.isfinite(tr[i])]
        ax.plot(np.arange(len(t))[-700:], t[-700:], lw=0.6, label=rows[i]["variant"])
    ax.axhline(1.0, color=MUTED, lw=0.8)
    ax.set_xlabel("TR index (last 700)", fontsize=8, color=MUTED)
    ax.set_ylabel("|S| / ideal", fontsize=8, color=MUTED)
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.set_title(f"static blood, RF increment {incs[0]:g}°", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(run / "traces.png", dpi=130)
    plt.close(fig)
    print(f"wrote {run / 'spoilcheck.png'}, {run / 'traces.png'}")


if __name__ == "__main__":
    main()
