"""Figures of the GIRF-robust encode study (`encrobust`, WP1 step 3).
    rustlib/.venv/bin/python -I rustlib/tools/seq/plot_encrobust.py [data/hasty/seq/studies/encrobust]
Writes <run>/figs/encrobust_*.png:
  01_waveforms   encode x_lo (axis x) of the compared designs: G(t) and slew
  02_spectrum    |γ̄ Ĝ(f)| of (encode x_lo − reference) on x, the design band and the variant resonances
  03_sensitivity γ̄|E(τ)| of the same difference (what eddy / low-pass terms of time constant τ leave at the echo)
  04_tradeoff    background velocity at 10 cm (body scale) against TE: unmodelled / after modelling
  05_f0scan      background velocity against the frequency of one resonance (6 %, ζ 0.03) on every axis
  06_traces      k error of (encode x_lo − reference) over the first ms of the readout (full illustrative chain)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
# designs in fixed order: two-lobe (reference), robust, robust at 1.25× slew, tuned to the known resonances
DCOL = ["#1f1f1e", "#2a78d6", "#1baf7a", "#eb6834"]
GBAR = 2.6752218744e8 / (2 * np.pi)


def style(ax, xlabel=None, ylabel=None):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "data/hasty/seq/studies/encrobust")
    J = json.loads((run / "encrobust.json").read_text())
    names = J["designs"]
    rows = {r["design"]: r for r in J["rows"]}
    W = np.load(run / "enc_waves.npy")
    sens, taus = np.load(run / "sens.npy"), np.load(run / "sens_tau.npy")
    spec, freqs = np.load(run / "spec.npy"), np.load(run / "spec_f.npy")
    tr, wts = np.load(run / "traces.npy"), np.load(run / "weights.npy")
    dt = J["dt"]
    figs = run / "figs"
    figs.mkdir(exist_ok=True)

    def pick(prefix, te_extra):
        best = None
        for i, n in enumerate(names):
            if n.startswith(prefix) and n[len(prefix):len(prefix) + 1] in "+-":
                ex = float(n[len(prefix):].replace("ms", ""))
                if best is None or abs(ex - te_extra) < abs(best[1] - te_extra):
                    best = (i, ex)
        return best[0] if best else None

    sel = [(names.index("two-lobe"), "two-lobe (current)")]
    for prefix, ex, lab in [("robust", 0.5, "robust +0.5 ms"), ("robusts1.25", 0.5, "robust ×1.25 slew +0.5 ms"), ("tuned", 0.2, "tuned to known resonances +0.2 ms")]:
        i = pick(prefix, ex)
        if i is not None:
            sel.append((i, lab + f"  [{names[i]}]"))

    # 01 waveforms
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    for k, (i, lab) in enumerate(sel):
        g = W[i, 1, :, 0]
        n = int(np.max(np.nonzero(np.abs(W[i]).sum(axis=(0, 2)))[0])) + 1
        t = np.arange(n + 1) * dt * 1e3
        axs[0].plot(t, g[: n + 1] * 1e3, color=DCOL[k], lw=1.6, label=lab)
        axs[1].plot(t[1:], np.diff(g[: n + 1]) / dt, color=DCOL[k], lw=1.2)
    style(axs[0], ylabel="G_x [mT/m]")
    style(axs[1], "time from encode start [ms]", "slew_x [T/m/s]")
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("Encode x_lo, axis x (the reference encode is its negative); M0 = 0, same M1", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(figs / "encrobust_01_waveforms.png", dpi=150)
    plt.close(fig)

    # 02 spectrum
    fig, ax = plt.subplots(figsize=(8, 4))
    band = J["robust_opts"]["band"]
    if band:
        ax.axvspan(band[0], band[1], color="#f3ecdc", lw=0, zorder=0, label="robust design band")
    for f0 in (900, 1250, 1400):
        ax.axvline(f0, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.text(1410, 0.55, "variant resonances\nz 0.9 / y 1.25 / x 1.4 kHz", color=MUTED, fontsize=7, va="top", transform=ax.get_xaxis_transform())
    for k, (i, lab) in enumerate(sel):
        ax.plot(freqs, spec[i], color=DCOL[k], lw=1.4, label=lab)
    ax.set_xlim(0, 3000)
    style(ax, "frequency [Hz]", "γ̄|Ĝ(f)| of (x_lo − ref), x  [cycles/m per Hz⁻¹]")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "encrobust_02_spectrum.png", dpi=150)
    plt.close(fig)

    # 03 sensitivity
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (i, lab) in enumerate(sel):
        ax.loglog(taus * 1e6, np.abs(sens[i]), color=DCOL[k], lw=1.4, label=lab)
    style(ax, "time constant τ [µs]", "γ̄|E(τ)| [cycles/m per unit eddy amplitude]")
    ax.set_title("k offset at the echo from a first-order term with time constant τ (× its amplitude a);\nfor τ ≫ the block → γ̄ΔM1/τ, the same for every design", color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "encrobust_03_sensitivity.png", dpi=150)
    plt.close(fig)

    # 04 trade-off against TE
    fams = [("robust", DCOL[1], "robust, resonance anywhere 0.4–2 kHz (mean)", "scan"), ("robusts1.25", DCOL[2], "robust ×1.25 slew, resonance 0.4–2 kHz (mean)", "scan"),
            ("tuned", DCOL[3], "tuned, on the known resonances (variant chain)", "resonance")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for m, (ax, title) in enumerate(zip(axs, ["unmodelled", "modelled (RSS of 10 % single-parameter GIRF errors)"])):
        for prefix, col, lab, what in fams:
            pts = []
            for n in names:
                if n.startswith(prefix) and n[len(prefix):len(prefix) + 1] in "+-":
                    r = rows[n]
                    if what == "scan":
                        v = r["res_scan"]["mean_modelled" if m else "mean"]
                    else:
                        c = next(c for c in r["chains"] if c["girf"] == "resonance")
                        v = c["v_bg_body_modelled" if m else "v_bg_body"]
                    pts.append((r["te"] * 1e3, v * 1e3))
            pts.sort()
            if pts:
                ax.plot(*zip(*pts), color=col, lw=1.6, marker="o", ms=4, label=lab)
        # eddy floor (the robust family on the eddy-only chain) and the two-lobe reference
        pts = sorted((rows[n]["te"] * 1e3, next(c for c in rows[n]["chains"] if c["girf"] == "eddy")["v_bg_body_modelled" if m else "v_bg_body"] * 1e3) for n in names if n.startswith("robust+") or n.startswith("robust-"))
        ax.plot(*zip(*pts), color=MUTED, lw=1.2, ls="--", label="eddy currents only (variant chain)")
        r0 = rows["two-lobe"]
        ax.plot([r0["te"] * 1e3], [r0["res_scan"]["mean_modelled" if m else "mean"] * 1e3], "s", color=DCOL[0], ms=6, label="two-lobe, resonance 0.4–2 kHz (mean)")
        c0 = next(c for c in r0["chains"] if c["girf"] == "resonance")
        ax.plot([r0["te"] * 1e3], [c0["v_bg_body_modelled" if m else "v_bg_body"] * 1e3], "D", color=DCOL[0], ms=6, mfc="white", label="two-lobe, known resonances (variant chain)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PLAIN)
        ax.yaxis.set_minor_formatter(PLAIN)
        ax.tick_params(axis="y", which="minor", labelsize=7, colors=MUTED)
        style(ax, "TE [ms]", "background velocity at 10 cm [mm/s]")
        ax.set_title(title, color=INK, fontsize=9, loc="left")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "encrobust_04_tradeoff.png", dpi=150)
    plt.close(fig)

    # 05 f0 scan
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for m, (ax, title) in enumerate(zip(axs, ["unmodelled", "modelled (f0 ±1 %, ζ, height ±10 %, RSS)"])):
        for k, (i, lab) in enumerate(sel[:3]):
            s = rows[names[i]]["res_scan"]
            ax.plot(s["f0"], np.array(s["v_bg_body_modelled" if m else "v_bg_body"]) * 1e3, color=DCOL[k], lw=1.4, label=lab)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PLAIN)
        style(ax, "resonance frequency f0 [Hz] (6 % peak, ζ 0.03, all axes)", "background velocity at 10 cm [mm/s]")
        ax.set_title(title, color=INK, fontsize=9, loc="left")
    axs[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "encrobust_05_f0scan.png", dpi=150)
    plt.close(fig)

    # 06 traces
    fig, ax = plt.subplots(figsize=(8, 4))
    t = wts[:, 0] * 1e6
    wb = wts[:, 1] / wts[:, 1].max()
    for k, (i, lab) in enumerate(sel):
        ax.plot(t, tr[i, 1, :, 0], color=DCOL[k], lw=1.4, label=lab)
    lo, hi = ax.get_ylim()
    ax.fill_between(t, lo, lo + (hi - lo) * wb, color="#f3ecdc", lw=0, zorder=0, label="body-scale weight (scaled)")
    ax.set_xlim(0, 1000)
    style(ax, "time from the echo [µs]", "Δk_x of (x_lo − ref) [cycles/m]")
    ax.set_title("k error the encode leaves in the readout, full illustrative chain (unmodelled)", color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "encrobust_06_traces.png", dpi=150)
    plt.close(fig)
    print(f"figures: {figs}")


if __name__ == "__main__":
    main()
