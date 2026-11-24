#!/usr/bin/env python3
"""Figures for `acousticsim` (rustlib/crates/hasty-seq/src/bin/acousticsim.rs).

    python3 tools/seq/plot_acoustic.py <acousticsim out dir>

Writes into <dir>/figs: overview.png (dB(A) per TR over the whole sequence + octave bands, events marked),
model.png (transfer function per axis), and per window w<i>_*.png (gradients, pressure, spectrogram with the
resonances, spectrum with the strongest lines and their TR harmonic / resonance).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXC = ["tab:red", "tab:green", "tab:blue"]
P0 = 20e-6


def H(model, a, f):
    s = 0
    for m in model["axes"][a]:
        u = f / m["f0"]
        jq = 1j * u / m["q"]
        s = s + m["peak"] * jq / (1 - u * u + jq)
    return s * model["b0"] / model["b0_ref"]


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-30) / P0)


def overview(d, S, fig_dir, unc):
    sig = np.load(d / "signatures.npy")
    t = np.load(d / "tr_t.npy")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(t, sig[:, 7], lw=0.6, color="k")
    a1.set_ylabel("dB(A) per TR\n(TR repeated)")
    for c in range(7):
        a2.plot(t, sig[:, c], lw=0.6, label=f"{int(S['bands_hz'][c])} Hz")
    a2.set_ylabel("octave band dB SPL")
    a2.set_xlabel("sequence time (s)")
    a2.legend(ncol=7, fontsize=7, loc="lower center")
    col = {"start": "tab:green", "loudest": "tab:red", "change": "tab:orange"}
    for e in S["events"]:
        for a in (a1, a2):
            a.axvline(e["t"], color=col[e["kind"]], lw=0.8, ls="--")
        a1.text(e["t"], a1.get_ylim()[1], f" {e['kind']}", rotation=90, va="top", fontsize=7, color=col[e["kind"]])
    for w in S["windows"]:
        a1.axvspan(w["t_start"], w["t_end"], color="tab:blue", alpha=0.3)
    a1.set_title(f"{Path(S['seq']).name}: {S['n_tr']} TRs, {S['duration_s']:.1f} s{unc}")
    fig.tight_layout()
    fig.savefig(fig_dir / "overview.png", dpi=130)
    plt.close(fig)


def main(d):
    d = Path(d)
    S = json.loads((d / "summary.json").read_text())
    fig_dir = d / "figs"
    fig_dir.mkdir(exist_ok=True)
    model = S["model"]
    unc = "" if S["calibrated"] else "  (toy model: levels indicative only)"

    # ── model
    f = np.logspace(np.log10(50), np.log10(8000), 800)
    fig, ax = plt.subplots(figsize=(8, 4))
    for a, n in enumerate("xyz"):
        ax.semilogx(f, db(H(model, a, f) * 0.01), AXC[a], label=f"{n}: 10 mT/m sine")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("dB SPL at the ear")
    ax.set_title("pressure per axis, 10 mT/m gradient sine" + (" (+ protection)" if S["protection_applied"] else "") + unc, fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "model.png", dpi=130)
    plt.close(fig)

    if (d / "signatures.npy").exists():
        overview(d, S, fig_dir, unc)

    # ── windows
    for w in S["windows"]:
        n = w["name"]
        p = np.load(d / f"{n}_p.npy")
        pa = np.load(d / f"{n}_pa.npy")
        g = np.load(d / f"{n}_g.npy")
        fs = w["fs"]
        tt = np.arange(len(p)) / fs
        tg = np.arange(len(g)) * w["g_dt"]
        fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
        a = fig.add_subplot(gs[0, 0])
        for k in range(3):
            a.plot(tg, g[:, k] * 1e3, AXC[k], lw=0.6, label="xyz"[k])
        for t0 in w["tr_t0"]:
            a.axvline(t0, color="gray", lw=0.3)
        a.set_ylabel("gradient (mT/m)")
        a.legend(ncol=3, fontsize=7)
        a.set_title(f"{n}: {w['note']}\nTR {w['tr_first']}–{w['tr_last']}, t = {w['t_start']:.3f}–{w['t_end']:.3f} s{unc}", fontsize=9)
        b = fig.add_subplot(gs[1, 0], sharex=a)
        b.plot(tt, p, "k", lw=0.4)
        b.set_ylabel("pressure (Pa)")
        c = b.twinx()
        # short-time dB(A), 10 ms rms
        k = int(0.01 * fs)
        ms = np.convolve(pa**2, np.ones(k) / k, mode="same")
        c.plot(tt, 10 * np.log10(np.maximum(ms, 1e-30) / P0**2), "tab:orange", lw=0.8)
        c.set_ylabel("dB(A), 10 ms", color="tab:orange")
        s = fig.add_subplot(gs[2, 0], sharex=a)
        nfft = 1024
        _, _, _, im = s.specgram(p, NFFT=nfft, Fs=fs, noverlap=nfft * 7 // 8, cmap="magma", scale="dB")
        lo = im.get_array().max() - 70
        im.set_clim(lo, lo + 70)
        s.set_ylim(0, 6000)
        for ax_i in range(3):
            for m in model["axes"][ax_i]:
                if m["q"] > 5:
                    s.axhline(m["f0"], color=AXC[ax_i], lw=0.5, ls=":")
        s.set_ylabel("Hz (dotted: resonances)")
        s.set_xlabel("time in window (s)")
        e = fig.add_subplot(gs[0:2, 1])
        tob = np.array(w["third_octave"])
        e.bar(np.log10(tob[:, 0]), tob[:, 1], width=0.09, color="tab:gray")
        e.set_xticks(np.log10([100, 250, 500, 1000, 2000, 4000, 8000]))
        e.set_xticklabels(["100", "250", "500", "1k", "2k", "4k", "8k"])
        e.set_ylim(max(tob[:, 1].max() - 60, 0), tob[:, 1].max() + 4)
        e.set_ylabel("1/3-octave dB SPL")
        e.set_title(f"LAeq {w['laeq']:.1f} dB(A)  LAFmax {w['lafmax']:.1f}  Lpeak {w['lpeak_db']:.1f}", fontsize=9)
        tx = fig.add_subplot(gs[2, 1])
        tx.axis("off")
        rows = ["line Hz  dB(A) axis  resonance  ×TR rate"]
        for l in w["lines"][:9]:
            r = l["resonance"]
            hs = "" if l["tr_harmonic"] is None else f"  {l['tr_harmonic']:.2f}×"
            rows.append(f"{l['f']:7.1f}  {l['db_a']:5.1f}  {l['axis']}  {r['axis']}{r['f0']:.0f} ({r['offset_half_bw']:+.1f} hw){hs}")
        tx.text(0, 1, "\n".join(rows), va="top", family="monospace", fontsize=7.5)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{n}.png", dpi=120)
        plt.close(fig)
    print(f"figures in {fig_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
