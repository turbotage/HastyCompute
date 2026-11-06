"""Side-by-side TR gradient diagrams of several `seqdesign` runs (e.g. a readout-length sweep).
    rustlib/.venv/bin/python -I rustlib/tools/seq/plot_sweep.py <out_dir> <run> [<run> ...]
Writes <out_dir>/sweep_tr.png (one row per run: physical gradients of one encode, first rotation; |k|/kmax of every
encode), <out_dir>/sweep_spokes.png (base-spoke k projections), <out_dir>/sweep.json, and prints a table.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AX = ["#2a78d6", "#eb6834", "#1baf7a"]  # x, y, z in fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
PART = {"encode": "#eef3fb", "tail": "#f6efe9"}


def tidy(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in map(Path, sys.argv[2:]):
        s = json.loads((r / "summary.json").read_text())
        runs.append((s["best"]["t_adc"], r, s))
    runs.sort(key=lambda x: x[0])
    dt = runs[0][2]["problem"]["dt"]
    tmax = max(s["best"]["tr"] for _, _, s in runs) * 1e3
    enc = 1  # x, low VENC

    fig, axs = plt.subplots(len(runs), 2, figsize=(16, 1.9 * len(runs)), sharex=True, gridspec_kw={"width_ratios": [3, 2]})
    rows = []
    for i, (_, run, s) in enumerate(runs):
        b, pr = s["best"], s["problem"]
        g = np.load(run / "tr_phys.npy")[enc]
        k = np.load(run / "k_tr.npy")
        t = np.arange(g.shape[0]) * dt * 1e3
        for col in (0, 1):
            ax = axs[i, col]
            ax.axvspan(b["i_enc"] * dt * 1e3, b["i_ro"] * dt * 1e3, color=PART["encode"], lw=0)
            ax.axvspan(b["i_tail"] * dt * 1e3, b["n_tr"] * dt * 1e3, color=PART["tail"], lw=0)
            tidy(ax)
        ax = axs[i, 0]
        for q in range(3):
            ax.plot(t, g[:, q] * 1e3, lw=0.8, color=AX[q], label="Gxyz"[q + 1] if i == 0 else None)
        ax.set_ylabel(f"T_adc {b['t_adc']*1e3:.1f} ms\nmT/m", fontsize=8, color=INK)
        ax.text(0.995, 0.93, f"TR {b['tr']*1e3:.2f} ms · N {b['n_eff']:.0f} · N·TR {b['objective']:.2f} s · PNS {b['pns']:.1f}% ({b['pns_part']})",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
        ax = axs[i, 1]
        kr = np.linalg.norm(k, axis=2) / pr["kmax"]
        for e in range(kr.shape[0]):
            ax.plot(t[: kr.shape[1]], kr[e], lw=0.6, color=AX[0] if e == enc else "#b9b8b0")
        ax.axhline(1.0, color="#c0392b", lw=0.8)
        ax.set_ylim(0, max(2.5, float(np.nanmax(kr)) * 1.05))
        ax.set_ylabel("|k| / kmax", fontsize=8, color=MUTED)
        kc = b.get("kmax_check", {})
        rows.append({"t_adc": b["t_adc"], "n": b["n_eff"], "tr": b["tr"], "te": b["te"], "n_tr": b["objective"], "scan_all_enc_s": b["scan_nyquist_s"],
                     "duty": b["t_adc"] / b["tr"], "duty_all": (b["tr"] - b["i_enc"] * dt) / b["tr"], "pns": b["pns"], "pns_part": b["pns_part"], "slew": b["slew"],
                     "tail_ms": b["n_tail"] * dt * 1e3, "aniso": b["aniso"], "shape": b["shape"], "readout_kmax": kc.get("readout_max"), "tail_kmax": kc.get("tail_max"),
                     "outside": (kc.get("readout_outside", 0) or 0) + (kc.get("tail_outside", 0) or 0), "encode_kmax": kc.get("encode_max"),
                     "gm1_end_99": b["readout_gamma_m1"][-1]})
    axs[0, 0].legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    axs[0, 0].set_title("physical gradients, encode 1 (x, low VENC), first rotation — shaded: encode (navigator ADC), tail (spoiler / M1 block, flagged)", fontsize=9, color=INK, loc="left")
    axs[0, 1].set_title("|k|/kmax of every encode (blue: encode 1); red line = kmax", fontsize=9, color=INK, loc="left")
    for c in (0, 1):
        axs[-1, c].set_xlabel("ms from TR start", fontsize=8, color=MUTED)
        axs[-1, c].set_xlim(0, tmax)
    pr = runs[0][2]["problem"]
    fig.suptitle(f"Readout-length sweep: {pr['res']*1e3:.1f} mm, FOV {pr['fov']*1e3:.0f} mm, {pr['scheme']}, VENC {pr['venc']} m/s, "
                 f"PNS ≤ {pr['pns_target']}% [{pr['pns_hw']}]", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "sweep_tr.png", dpi=120)
    plt.close(fig)

    # base-spoke shapes
    fig, axs = plt.subplots(2, len(runs), figsize=(2.2 * len(runs), 4.6))
    for i, (_, run, s) in enumerate(runs):
        k = np.load(run / "k_adc.npy") / s["problem"]["kmax"]
        n = k.shape[0]
        for row, (a, c) in enumerate([(0, 1), (0, 2)]):
            ax = axs[row, i]
            ax.plot(k[: n // 2, a], k[: n // 2, c], lw=0.4, color=AX[0])
            ax.plot(k[n // 2:, a], k[n // 2:, c], lw=0.4, color=AX[1])
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#c0392b", lw=0.6))
            ax.set_aspect("equal")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(f"{s['best']['t_adc']*1e3:.1f} ms", fontsize=8, color=INK)
        axs[0, 0].set_ylabel("ky (xy plane)", fontsize=7, color=MUTED)
        axs[1, 0].set_ylabel("kz (xz plane)", fontsize=7, color=MUTED)
    fig.suptitle("base spoke k / kmax (blue: out leg, orange: return leg; red circle = kmax)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "sweep_spokes.png", dpi=130)
    plt.close(fig)

    (out / "sweep.json").write_text(json.dumps(rows, indent=1))
    print(" T_adc     N   TR ms  TE ms  N·TR s  ×7 min  duty  duty_all  tail ms  PNS part      slew (enc, ro, tail)  |k|max ro/tail  outside  enc|k|  γM1@99%  shape")
    for r in rows:
        print(f" {r['t_adc']*1e3:5.2f} {r['n']:5.0f} {r['tr']*1e3:6.2f} {r['te']*1e3:6.2f} {r['n_tr']:7.2f} {r['scan_all_enc_s']/60:7.2f} {r['duty']:5.2f} {r['duty_all']:8.2f}"
              f" {r['tail_ms']:7.2f} {r['pns']:5.1f} {r['pns_part']:8} {r['slew'][0]:.2f} {r['slew'][1]:.2f} {r['slew'][2]:.2f}"
              f"      {r['readout_kmax']:.4f}/{r['tail_kmax']:.4f} {r['outside']:5d}  {r['encode_kmax']:.2f}  {r['gm1_end_99']:7.2f}"
              f"  kg0 {r['shape']['kg0']:.3f} A {r['shape']['wobble']:.3f} ω {r['shape']['wobble_freq']:.2f}")
    print(f"wrote {out / 'sweep_tr.png'}, {out / 'sweep_spokes.png'}")


if __name__ == "__main__":
    main()
