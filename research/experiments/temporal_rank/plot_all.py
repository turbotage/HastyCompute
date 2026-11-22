"""Figures of the temporal-rank study from the out/*.json results."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import LEVELS, OUT  # noqa: E402


def load(n):
    p = OUT / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_spectra():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    d = load("diag_regions")
    t = load("truth_cardiac")
    if d:
        for tag, r in d.items():
            s = np.sqrt(np.array(r["spectrum_all"]))
            ax[0].semilogy(np.arange(1, len(s) + 1), s, label=tag)
    for n, lab in [("regime_truth_smooth", "truth, continuous-motion magnitude"), ("regime_overlay_ref", "network-flow overlay reference")]:
        r = load(n)
        if r:
            s = np.sqrt(np.array(r["spectrum_shared"]))
            ax[0].semilogy(np.arange(1, len(s) + 1), s, "--", label=lab)
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("singular value / first")
    ax[0].set_title("cardiac spectra (heart box, 7 encodes pooled)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    if t:
        rows = t["rows"]
        for b, st in [("svd_shared", "-o"), ("fourier", "-s"), ("bspline3", "-^")]:
            rr = [r for r in rows if r["basis"] == b]
            ax[1].semilogy([r["K"] for r in rr], [r["rel_blood"] for r in rr], st, label=f"{b} rel. error (lumen)")
            ax[1].semilogy([r["K"] for r in rr], [r["dv_mean"] / 100 for r in rr], st, alpha=0.5, label=f"{b} mean |dv| (m/s)")
        ax[1].set_xlabel("K (basis images per encode)")
        ax[1].set_title("ph0001 truth: error vs K")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_spectra.png", dpi=110)


def fig_regimes():
    import glob
    names = sorted(p.split("regime_")[1][:-5] for p in glob.glob(str(OUT / "regime_*.json")))
    if not names:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for n in names:
        r = load(f"regime_{n}")
        ax[0].plot(LEVELS, r["K_shared"], "-o", label=n)
        rr = [x for x in r["rows"] if x["basis"] == "svd_shared"]
        ax[1].semilogy([x["K"] for x in rr], [x["dv_mean"] for x in rr], "-o", label=n)
    ax[0].set_xscale("log")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("relative Frobenius error level (heart box)")
    ax[0].set_ylabel("K (shared SVD basis)")
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("K (shared SVD basis)")
    ax[1].set_ylabel("mean |dv| in the lumen (cm/s)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_regimes.png", dpi=110)


def fig_joint():
    import glob
    fs = sorted(glob.glob(str(OUT / "joint_*.json")))
    if not fs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for f in fs:
        r = json.loads(open(f).read())
        s = np.sqrt(np.array(r["spectrum"]))
        ax[0].semilogy(np.arange(1, len(s) + 1), s, label=r["tag"])
        rows = r["rows"]
        for b, st in [("svd_joint", "-o")]:
            rr = [x for x in rows if x["basis"] == b]
            ax[1].semilogy([x["K"] for x in rr], [x["rel_blood"] for x in rr], st, label=f"{r['tag']} {b}")
        kn = [x for x in rows if x["basis"].startswith("knots")]
        ax[1].semilogy([x["K"] for x in kn], [x["rel_blood"] for x in kn], "x", ms=9, label=f"{r['tag']} knots (lin x cubic)")
        sp = [x for x in rows if x["basis"].startswith("svd_sep")]
        ax[1].semilogy([x["K"] for x in sp], [x["rel_blood"] for x in sp], "+", ms=9, label=f"{r['tag']} separable SVD")
    ax[0].set_xlim(0, 150)
    ax[0].set_xlabel("k")
    ax[0].set_title("joint breathing x cardiac spectra")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("relative error, lumen")
    ax[1].legend(fontsize=6)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_joint.png", dpi=110)


if __name__ == "__main__":
    fig_spectra()
    fig_regimes()
    fig_joint()
