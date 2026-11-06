"""Figures + table of an `adcstudy` run (readout-length sweep).
    rustlib/.venv/bin/python -I rustlib/tools/seq/plot_adcstudy.py <run> [<run> ...]
Writes <run>/figs/adcstudy.png and prints a table (best shape per T_adc).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def load(run):
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("feasible")]
    shapes = []
    for r in rows:
        if r["shape"] not in shapes:
            shapes.append(r["shape"])
    best = {}
    for r in rows:
        t = r["t_adc_budget"]
        if t not in best or r["n_tr"] < best[t]["n_tr"]:
            best[t] = r
    return rows, shapes, [best[t] for t in sorted(best)]


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.set_xlabel("ADC window T_adc (ms)", fontsize=8, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    for run in map(Path, sys.argv[1:]):
        rows, shapes, best = load(run)
        figs = run / "figs"
        figs.mkdir(exist_ok=True)
        t = np.array([b["t_adc"] for b in best]) * 1e3
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))

        ax = axs[0, 0]
        for i, sh in enumerate(shapes):
            rr = sorted([r for r in rows if r["shape"] == sh], key=lambda r: r["t_adc"])
            ax.plot([r["t_adc"] * 1e3 for r in rr], [r["n_tr"] for r in rr], "-o", ms=4, lw=1.6, color=SERIES[i % 4],
                    label=f"kg0 {sh[0]:g}, A {sh[1]:g}, ω {sh[2]:g}")
        style(ax, "Nyquist-complete time per encode N·TR", "s")
        ax.legend(fontsize=7, frameon=False)

        ax = axs[0, 1]
        i0 = int(np.argmin(np.abs(t - 3.0)))
        ax.plot(t, np.sqrt([b["duty"] / best[i0]["duty"] for b in best]), "-o", ms=4, lw=1.6, color=SERIES[0], label="ADC on readout only")
        ax.plot(t, np.sqrt([b["duty_all"] / best[i0]["duty"] for b in best]), "--o", ms=4, lw=1.6, color=SERIES[1],
                label="ADC on encode + readout + tail, all samples (upper bound)")
        if "duty_all_in_kmax" in best[0]:
            ax.plot(t, np.sqrt([b["duty_all_in_kmax"] / best[i0]["duty"] for b in best]), ":o", ms=4, lw=1.6, color=SERIES[2],
                    label="same, only samples with |k| ≤ kmax")
        style(ax, f"scan-time-normalised SNR, rel. to {t[i0]:.1f} ms readout-only  (∝ √duty)", "×")
        ax.legend(fontsize=7, frameon=False)

        ax = axs[0, 2]
        to = np.array([b["t_shell_outer"] for b in best])
        for j, (t2s, lab) in enumerate([(0.040, "T2* 40 ms"), (0.020, "T2* 20 ms"), (0.010, "T2* 10 ms")]):
            ax.plot(t, np.exp(-to / t2s), "-o", ms=4, lw=1.6, color=SERIES[j], label=lab)
        style(ax, "T2* weight of the outer k shell (0.9–1 kmax), rel. to TE", "signal fraction")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, frameon=False)

        ax = axs[1, 0]
        ax.plot(t, 1.0 / (4.0 * to), "-o", ms=4, lw=1.6, color=SERIES[0], label="off-resonance (or field-map error) for π/2 at the outer shell")
        style(ax, "off-resonance tolerance at the k edge (fat at 3 T ≈ 440 Hz)", "Hz")
        ax.set_yscale("log")
        ax.axhline(440, color=MUTED, lw=0.8, ls=":")
        ax.legend(fontsize=7, frameon=False)

        ax = axs[1, 1]
        ax.plot(t, [b["gm1_first100"] for b in best], "-o", ms=4, lw=1.6, color=SERIES[0], label="first reach of kmax")
        ax.plot(t, [b["gm1_end"] for b in best], "--o", ms=4, lw=1.6, color=SERIES[1], label="ADC end")
        style(ax, "readout first moment γ|M1| about the echo (flow displacement)", "rad per m/s")
        ax.set_yscale("log")
        ax.legend(fontsize=7, frameon=False)

        ax = axs[1, 2]
        ax.plot(t, [b["conc_15cm"] for b in best], "-o", ms=4, lw=1.6, color=SERIES[0])
        style(ax, "readout concomitant phase scale at 15 cm (isotropic part)", "rad")

        pr = (run / "problem.txt").read_text().split("|")
        fig.suptitle(f"{run.name}: res {float(pr[0])*1e3:.1f} mm, FOV {float(pr[1])*1e3:.0f} mm, {pr[2]}, VENC {float(pr[3]):.3f} m/s — "
                     "best shape per T_adc except top-left; PNS ≤ 80 % on EXAMPLE hardware", fontsize=9, color=INK)
        fig.tight_layout()
        fig.savefig(figs / "adcstudy.png", dpi=130)
        plt.close(fig)

        print(f"\n{run.name}")
        print(" T_adc ms     N   TR ms  N·TR s  ×enc min  duty  duty_all  t_outer ms  enc ms  tail ms  PNS part   slew(e,r,t)       γM1 end  conc 15cm  k_enc/kmax  tail≤kmax  shape")
        for b in best:
            print(f" {b['t_adc']*1e3:7.2f} {b['n']:6.0f} {b['tr']*1e3:6.2f} {b['n_tr']:7.2f} {b['scan_nyquist_all_enc']/60:8.2f} {b['duty']:5.2f} {b['duty_all']:8.2f}"
                  f" {b['t_shell_outer']*1e3:10.2f} {b['encode']*1e3:7.2f} {b['tail']*1e3:7.2f} {b['pns']:5.1f} {b['pns_part']:8}"
                  f" {b['slew'][0]:.2f},{b['slew'][1]:.2f},{b['slew'][2]:.2f} {b['gm1_end']:9.2f} {b['conc_15cm']:9.2f} {b['k_enc_max_over_kmax']:10.2f}"
                  f" {b['tail_inside_kmax']:9.2f}  enc≤kmax {b.get('enc_inside_kmax', float('nan')):.2f}"
                  f" duty_all≤kmax {b.get('duty_all_in_kmax', float('nan')):.2f}  {b['shape']}")
        print(f"wrote {figs / 'adcstudy.png'}")


if __name__ == "__main__":
    main()
