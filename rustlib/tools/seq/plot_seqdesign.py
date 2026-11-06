"""Figures of a `seqdesign` run: base spoke, TR waveforms, PNS, density, coverage, GIRF error, design-space scan.
    rustlib/.venv/bin/python -I rustlib/tools/seq/plot_seqdesign.py <run>
Writes <run>/figs/*.png.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GAMMA_BAR = 2.6752218744e8 / (2 * np.pi)


def main():
    run = Path(sys.argv[1])
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    s = json.loads((run / "summary.json").read_text())
    pr, b = s["problem"], s["best"]
    dt = pr["dt"]
    title = (f"{pr['res']*1e3:.1f} mm, FOV {pr['fov']*1e3:.0f} mm, T_adc {b['t_adc']*1e3:.2f} ms, VENC {pr['venc']} m/s {pr['scheme']}, "
             f"PNS ≤ {pr['pns_target']}% [{pr['pns_hw']}]")

    # ── base spoke ──
    k = np.load(run / "k_adc.npy")
    g = np.load(run / "spoke_grad.npy")
    t = np.arange(g.shape[0]) * dt * 1e3
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(k[:, 0], k[:, 1], k[:, 2], lw=0.6)
    ax.set_title("base spoke k(t) (cycles/m)")
    for i, (a, bb) in enumerate([(0, 1), (0, 2), (1, 2)]):
        ax = fig.add_subplot(2, 3, 2 + i)
        ax.plot(k[:, a], k[:, bb], lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"k projection {'xyz'[a]}{'xyz'[bb]}")
    ax = fig.add_subplot(2, 3, 5)
    for q in range(3):
        ax.plot(t, g[:, q] * 1e3, lw=0.8, label="xyz"[q])
    ax.plot(t, np.linalg.norm(g, axis=1) * 1e3, "k", lw=1, label="|G|")
    ax.axvline(b["i_adc"] * dt * 1e3, color="r", ls="--", lw=0.8, label="ADC end")
    ax.set_xlabel("ms from echo")
    ax.set_ylabel("mT/m")
    ax.legend(fontsize=7)
    ax.set_title("base readout + tail (rotation 0)")
    ax = fig.add_subplot(2, 3, 6)
    sl = np.diff(g, axis=0) / dt
    ax.plot(t[1:], np.linalg.norm(sl, axis=1), lw=0.8)
    ax.axhline(pr["smax"] * b["slew"][1], color="r", ls="--", lw=0.8, label="readout slew cap")
    ax.axvline(b["i_adc"] * dt * 1e3, color="r", ls=":", lw=0.8)
    ax.set_ylabel("|dG/dt| T/m/s")
    ax.set_xlabel("ms from echo")
    ax.legend(fontsize=7)
    fig.suptitle(f"base spoke  kg0={b['shape']['kg0']:.2f} A={b['shape']['wobble']:.2f} ω={b['shape']['wobble_freq']:.2f}; "
                 f"N={b.get('n_eff', b['n_design']):.0f}{' out-and-back' if b.get('ret') is not None else ''}, aniso {b['aniso']:.2f}, curve {b['curve_length']:.0f} /m\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "spoke.png", dpi=130)
    plt.close(fig)

    # ── TR waveforms ──
    trp = np.load(run / "tr_phys.npy")
    ne, n, _ = trp.shape
    tt = np.arange(n) * dt * 1e3
    ktr = np.load(run / "k_tr.npy") if (run / "k_tr.npy").exists() else None
    nrow = ne + (ktr is not None)
    fig, axs = plt.subplots(nrow, 1, figsize=(12, 2.2 * nrow), sharex=True)
    if ktr is not None:
        ax = axs[-1]
        for e in range(ne):
            ax.plot(tt[: ktr.shape[1]], np.linalg.norm(ktr[e], axis=1) / pr["kmax"], lw=0.7, label=f"enc {e}")
        ax.axhline(1.0, color="r", lw=0.8)
        for key, c in [("i_enc", "g"), ("i_ro", "r"), ("i_tail", "m")]:
            ax.axvline(b[key] * dt * 1e3, color=c, ls="--", lw=0.6)
        ax.set_ylabel("|k| / kmax")
        ax.set_ylim(0, max(1.1, float(np.nanmax(np.linalg.norm(ktr, axis=2))) / pr["kmax"] * 1.05))
        ax.legend(fontsize=6, ncol=7)
    for e in range(ne):
        for q in range(3):
            axs[e].plot(tt, trp[e, :, q] * 1e3, lw=0.8, label="xyz"[q])
        for key, c in [("i_enc", "g"), ("i_ro", "r"), ("i_tail", "m")]:
            axs[e].axvline(b[key] * dt * 1e3, color=c, ls="--", lw=0.6)
        axs[e].set_ylabel(f"enc {e}\nmT/m")
    axs[0].legend(fontsize=7, ncol=3)
    axs[-1].set_xlabel("ms (TR start = RF dead time start); dashed: encode, echo, tail; bottom: |k|/kmax (encode = navigator only)")
    fig.suptitle(f"TR {b['tr']*1e3:.3f} ms, TE {b['te']*1e3:.3f} ms (physical axes, first rotation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "tr.png", dpi=130)
    plt.close(fig)

    # ── PNS ──
    fig, axs = plt.subplots(2, 1, figsize=(13, 6))
    for ax, name, lab in [(axs[0], "pns_trace", "first 4 rotations"), (axs[1], "pns_worst", "window ending at the worst TR")]:
        p = np.load(run / f"{name}.npy")
        tp = np.arange(p.shape[0]) * dt * 1e3
        for q in range(3):
            ax.plot(tp, p[:, q], lw=0.6, label="xyz"[q])
        ax.plot(tp, p[:, 3], "k", lw=0.9, label="combined")
        ax.axhline(pr["pns_target"], color="r", ls="--", lw=0.8, label="target")
        ax.axhline(100, color="r", lw=0.8)
        ax.set_ylabel("SAFE %")
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7, ncol=5)
    axs[1].set_xlabel("ms")
    fig.suptitle(f"PNS of the played sequence: max {b['pns']:.1f}% in {b['pns_part']}; slew factors (enc, ro, tail) "
                 f"{', '.join(f'{x:.2f}' for x in b['slew'])}", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "pns.png", dpi=130)
    plt.close(fig)

    # ── density + coverage + GIRF ──
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    d = np.load(run / "density.npy")
    axs[0].semilogy(d[:, 0], d[:, 2], label="spokes needed per shell")
    n_eff = b.get("n_eff", b["n_design"])
    axs[0].axhline(n_eff, color="r", ls="--", label=f"design N {n_eff:.0f}")
    axs[0].axhline(np.pi * (2 * pr["kmax"] * pr["fov"]) ** 2, color="gray", ls=":", label="radial πM²")
    axs[0].set_xlabel("k radius (cycles/m)")
    axs[0].legend(fontsize=7)
    axs[0].set_title(f"N* = {b['n_star']:.0f}")
    c = np.load(run / "coverage.npy")
    for i, name in enumerate(["super-Fibonacci", "Fibonacci×golden", "golden means"]):
        axs[1].plot(c[i, :, 0], c[i, :, 1], label=f"{name} (N)")
        axs[1].plot(c[i, :, 0], c[i, :, 3], ls="--", label=f"{name} (N/8 contiguous)")
    axs[1].axhline(np.exp(-1), color="gray", ls=":", label="Poisson e⁻¹")
    axs[1].axhline(np.exp(-1 / 8), color="gray", ls="-.", label="Poisson e^(−1/8)")
    axs[1].set_xlabel("k radius (cycles/m)")
    axs[1].set_ylabel("empty Nyquist cells")
    axs[1].legend(fontsize=6)
    axs[1].set_title("coverage of concrete rotation sets")
    ge = np.load(run / "girf_err.npy")
    tg = np.arange(ge.shape[0]) * dt * 1e3
    axs[2].plot(tg, np.linalg.norm(ge, axis=1) * pr["fov"])
    axs[2].set_xlabel("ms from echo")
    axs[2].set_ylabel("|Δk| (Nyquist cells 1/FOV)")
    axs[2].set_title(f"GIRF (clinical example) trajectory error, max {b['girf_max_err_cells']:.2f} over 16 rotations", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "density_coverage_girf.png", dpi=130)
    plt.close(fig)

    # ── design space ──
    gr = np.load(run / "grid.npy")  # kg0, wobble, freq, n, tr, pns, aniso, objective
    ev = [json.loads(l) for l in (run / "evals.jsonl").read_text().splitlines() if l.strip()]
    ev = [e for e in ev if e["objective"] > 0]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    sc = axs[0].scatter([e["aniso"] for e in ev], [e["objective"] for e in ev], c=[e["wobble"] for e in ev], cmap="viridis", s=14)
    axs[0].scatter([b["aniso"]], [b["objective"]], marker="*", s=200, c="r", label="optimum")
    plt.colorbar(sc, ax=axs[0], label="meander amplitude A")
    axs[0].set_xlabel("concomitant anisotropy")
    axs[0].set_ylabel("N·TR (s) per encode")
    axs[0].legend(fontsize=8)
    axs[0].set_title("all evaluations: efficiency vs concomitant isotropy")
    ok = np.isfinite(gr[:, 7])
    for kg0 in np.unique(gr[:, 0]):
        for fr in np.unique(gr[:, 2]):
            m = ok & (gr[:, 0] == kg0) & ((gr[:, 2] == fr) | (gr[:, 1] == 0))
            if m.sum() > 1:
                o = np.argsort(gr[m, 1])
                axs[1].plot(gr[m, 1][o], gr[m, 7][o], marker="o", ms=3, label=f"kg0 {kg0:g}, ω {fr:g}")
    axs[1].set_xlabel("meander amplitude A")
    axs[1].set_ylabel("N·TR (s) per encode")
    axs[1].legend(fontsize=6, ncol=2)
    axs[1].set_title("grid")
    fig.tight_layout()
    fig.savefig(figs / "design_space.png", dpi=130)
    plt.close(fig)
    print(f"wrote {figs}")


if __name__ == "__main__":
    main()
