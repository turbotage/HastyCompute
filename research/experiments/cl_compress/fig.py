"""Error vs FFT-channel count per variant (out/<region>_ratemap.json) -> figs/err_vs_channels.png"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
(HERE / "figs").mkdir(exist_ok=True)
regions = [r for r in ("box", "object") if (HERE / f"out/{r}_ratemap.json").exists()]
keys = [("eu", "PD energy, uniform time"), ("em", "PD energy, noise-referenced time"), ("uu", "uniform support, uniform time")]
fig, axs = plt.subplots(len(regions), len(keys), figsize=(5 * len(keys), 4 * len(regions)), squeeze=False)
for i, reg in enumerate(regions):
    o = json.load(open(HERE / f"out/{reg}_ratemap.json"))
    C = o["C"]
    for j, (k, lab) in enumerate(keys):
        r = o["res"][k]
        ax = axs[i, j]
        d = np.array(r["d_tail"])
        ax.semilogy(np.arange(len(d)), d, label="(d) joint SVD", lw=2)
        s = np.array(r["seg_tail"])
        ax.semilogy(C * np.arange(len(s)), s, "o-", ms=3, label="coil-common SVD segments (C·L)")
        b = np.array(r["bc_tail"])
        ax.semilogy(np.arange(len(b)), b, label="(b') coil-adaptive segments")
        for nm, key, st in (("fixed nodes, count LS (today)", "fixed_err", "s--"), ("fixed nodes, PD-energy LS", "fixed_e_err", "^--")):
            Ls = sorted(int(x) for x in r[key])
            ax.semilogy([C * L for L in Ls], [r[key][str(L)] for L in Ls], st, ms=3, label=nm)
        if "d_cross_u_tail" in r:
            x = np.array(r["d_cross_u_tail"])
            ax.semilogy(np.arange(len(x)), x, ":", label="(d) basis, error on uniform weights")
        for e in (1e-2, 1e-3, 1e-4):
            ax.axhline(e, color="0.7", lw=0.5)
        ax.set_xlim(0, 800)
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel("FFT channels per knot image (= grid interpolations per sample)")
        ax.set_ylabel("relative weighted error")
        ax.set_title(f"{reg} ({o['nv']} vox, 2 mm): {lab}", fontsize=9)
        if i == 0 and j == 0:
            ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "figs/err_vs_channels.png", dpi=110)
print("wrote figs/err_vs_channels.png")
