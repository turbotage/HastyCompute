"""Resonance-aware readout study: table of the seqdesign runs (scan efficiency, PNS, readout resonance exposure) and the
girfcheck full-sequence numbers of each (resonance chain), plus a figure of the readout spectra.
    rustlib/.venv/bin/python -I rustlib/tools/seq/compare_resaware.py <name>=<seqdesign dir>:<girfcheck dir> ...
Writes data/hasty/seq/studies/resaware/compare.json and figs/resaware_spectra.png.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
COL = ["#1f1f1e", "#2a78d6", "#1baf7a", "#eb6834"]
GBAR = 2.6752218744e8 / (2 * np.pi)


def spectrum(g, dt, f):
    h = dt * np.sinc(f * dt) ** 2
    t = np.arange(len(g)) * dt
    e = np.exp(-2j * np.pi * np.outer(f, t))
    return h * np.sqrt(sum(np.abs(e @ g[:, a]) ** 2 for a in range(3)))


def main():
    out = Path("data/hasty/seq/studies/resaware")
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, arg in enumerate(sys.argv[1:]):
        name, rest = arg.split("=", 1)
        sd, gd = rest.split(":", 1)
        s = json.loads((Path(sd) / "summary.json").read_text())
        g = json.loads((Path(gd) / "girfcheck.json").read_text())
        b = s["best"]
        res = next(r for r in g["rows"] if r["girf"] == "resonance")["flow"]["full_sequence"]
        eddy = next(r for r in g["rows"] if r["girf"] == "eddy")["flow"]["full_sequence"]
        row = {
            "design": name, "dir": sd, "n_eff": b["n_eff"], "tr_ms": b["tr"] * 1e3, "scan_nyquist_s": b["scan_nyquist_s"], "pns": b["pns"],
            "shape": b["shape"], "exposure_cells": g["readout_exposure_cells"],
            "resonance": {"coherent_body": res["v_bg_body"], "coherent_vessel": res["v_bg_vessel"], "rms_body": res["fluct_rms_body"], "rms_vessel": res["fluct_rms_vessel"],
                          "modelled": {k2: res["modelled"][k2] for k2 in ("v_bg_body", "v_bg_vessel", "fluct_rms_body", "fluct_rms_vessel")}},
            "eddy_coherent_body": eddy["v_bg_body"],
        }
        rows.append(row)
        spg = np.load(Path(sd) / "spoke_grad.npy")[: b["i_adc"] + 1]
        f = np.linspace(0, 3000, 601)
        ax.plot(f, GBAR * spectrum(spg, 1e-5, f), color=COL[k % len(COL)], lw=1.4, label=f"{name}: Nyquist scan {b['scan_nyquist_s']:.1f} s, exposure {row['exposure_cells']['variant_resonances']:.3f} cells")
        print(f"{name:10s} N {b['n_eff']:.0f} TR {b['tr']*1e3:.2f} ms Nyquist scan {b['scan_nyquist_s']:.1f} s PNS {b['pns']:.1f}% exposure {row['exposure_cells']['variant_resonances']:.3f} | "
              f"resonance chain: coherent {res['v_bg_body']*1e3:.0f}/{res['v_bg_vessel']*1e3:.0f}, rms {res['fluct_rms_body']*1e3:.0f}/{res['fluct_rms_vessel']*1e3:.0f} mm/s; "
              f"modelled rms {res['modelled']['fluct_rms_body']*1e3:.0f}/{res['modelled']['fluct_rms_vessel']*1e3:.0f}")
    for f0 in (900, 1250, 1400):
        ax.axvline(f0, color=MUTED, lw=0.8, ls="--")
    ax.text(1410, 0.95, "variant resonances\nz 0.9 / y 1.25 / x 1.4 kHz", color=MUTED, fontsize=7, va="top", transform=ax.get_xaxis_transform())
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("frequency [Hz]", color=INK, fontsize=9)
    ax.set_ylabel("γ̄|Ĝ(f)| of the base readout [cycles/m per Hz⁻¹]", color=INK, fontsize=9)
    ax.set_title("Readout spectrum (vector magnitude, the same for every rotation)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "figs" / "resaware_spectra.png", dpi=150)
    (out / "compare.json").write_text(json.dumps(rows, indent=1))
    print(f"→ {out / 'compare.json'}, {out / 'figs' / 'resaware_spectra.png'}")


if __name__ == "__main__":
    main()
