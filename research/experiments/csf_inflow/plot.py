"""Figures for the CSF inflow simulator. Usage: .venv/bin/python research/experiments/csf_inflow/plot.py  (run from repo root)
Paper numbers (Söderström et al. 2026, Table 2, D = 6 cm²/min): estimated-vs-prescribed regression r, α, β. Units of α are
not stated in the text I read; plotted as mm/s (assumption)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.abspath(__file__))
L = lambda *p: np.load(os.path.join(R, *p))
COL = {"cardiac": "#c0392b", "respiratory": "#2471a3", "low": "#1e8449"}
FREQ = {"cardiac": 0.93, "respiratory": 0.23, "low": 0.077}
PAPER = {"cardiac": (0.9368, 2.2039, 0.9504), "respiratory": (0.9724, -0.0559, 0.8100), "low": (0.9907, 0.2061, 0.7197)}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
names = ["cardiac", "respiratory", "low"]

# 1: estimated vs prescribed (phantom analysis)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
for a, n in zip(ax, names):
    x, y = L("phantomfit", f"pf_{n}_true.npy"), L("phantomfit", f"pf_{n}_est.npy")
    b, c = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    xx = np.linspace(0, x.max() * 1.05, 10)
    a.plot(xx, xx, color="0.7", lw=1, label="identity")
    pr, pa, pb = PAPER[n]
    a.plot(xx, pa + pb * xx, "--", color="k", lw=1.2, label=f"paper phantom fit\nr={pr:.3f}, α={pa:.2f}, β={pb:.2f}" + (" (≤1.5 mm/s)" if n == "low" else ""))
    a.plot(xx, c + b * xx, color=COL[n], lw=1.5, alpha=.6)
    a.plot(x, y, "o", color=COL[n], label=f"simulation\nr={r:.4f}, α={c:.2f}, β={b:.2f}")
    a.set_title(f"{n} ({FREQ[n]} Hz)")
    a.set_xlabel("prescribed velocity amplitude (mm/s)")
    a.set_ylabel("estimated (mm/s)")
    a.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("Phantom analysis: estimated vs prescribed velocity (sim: Monte-Carlo truth, 1 % noise, quasi-stationary estimator)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_estimated_vs_prescribed.png")); plt.close(fig)

# 2: signal modulation vs amplitude
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
for a, n in zip(ax, names):
    x, m = L("phantom", f"phantom_{n}_amp_mm_s.npy"), L("phantom", f"phantom_{n}_modulation.npy")
    a.plot(x, m * 100, "o-", color=COL[n])
    a.plot(x, m[-1] * 100 * x / x[-1], ":", color="0.5", lw=1, label="straight line through the top level")
    a.set_title(f"{n}: {FREQ[n]} Hz, appears at {abs(FREQ[n]-round(FREQ[n]/0.5)*0.5):.3f} Hz (TR 2 s)")
    a.set_xlabel("velocity amplitude (mm/s)"); a.set_ylabel("signal modulation at that frequency (% of mean)")
    if n == "low":
        a.axvline(1.5, color="k", ls="--", lw=1)
        a.text(1.52, 1, "paper: phantom estimates\nbreak down above 1.5 mm/s\n(sim: stays linear)", fontsize=7.5, va="bottom")
    a.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_modulation_vs_amplitude.png")); plt.close(fig)

# 3: aliasing
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, tag, tr in zip(ax, ["tr2.0", "tr0.5"], [2.0, 0.5]):
    f, s = L("alias", f"alias_{tag}_f.npy"), L("alias", f"alias_{tag}_amp.npy")
    a.plot(f, s * 100, color="k", lw=1)
    a.axvspan(0.01, 0.1, color=COL["low"], alpha=.12, label="low-frequency band (0.01–0.1 Hz)")
    a.axvline(0.5 / tr, color="0.5", ls=":", label=f"Nyquist {0.5/tr:.2f} Hz")
    a.set_title(f"cardiac 0.93 Hz, 12 mm/s, TR {tr} s"); a.set_xlabel("frequency (Hz)"); a.set_ylabel("amplitude (% of mean)")
    a.legend(fontsize=8, frameon=False)
a2 = ax[0]; a2.set_xlim(0, 0.25)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_aliasing.png")); plt.close(fig)

# 4: knee + round-trip time series
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.6, 1.6]})
v, k = L("knee", "knee_v_mm_s.npy"), L("knee", "knee_signal_rel_fresh.npy")
ax[0].plot(v, k, color="k"); ax[0].axvline(3.4 / 2.0, color="0.5", ls=":"); ax[0].text(1.75, 0.55, "thickness / TR", fontsize=8, rotation=90)
ax[0].set_xlim(0, 6); ax[0].set_xlabel("constant plug velocity (mm/s)"); ax[0].set_ylabel("bottom-slice signal / fresh-spin signal"); ax[0].set_title("inflow saturation")
for a, tag, ttl in zip(ax[1:], ["sinusoidal_truth", "nonsinusoidal_truth"], ["sinusoidal truth", "non-sinusoidal cardiac truth"]):
    d, m = L("roundtrip", f"rt_{tag}_data.npy"), L("roundtrip", f"rt_{tag}_model.npy")
    t = np.arange(len(d)) * 2.0
    a.plot(t, d, color="0.6", lw=.8, label="simulated fMRI signal (MC truth + 2 % noise)")
    a.plot(t, m, color="#c0392b", lw=1.1, label="fit (physiology-regressor model)")
    a.set_xlabel("time (s)"); a.set_ylabel("normalised signal"); a.set_title(ttl); a.legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(R, "fig4_knee_and_timeseries.png")); plt.close(fig)

# 5: sweeps
sw = [("dispersion_cm2_min", "dispersion D (cm²/min)", 6.0), ("t1_s", "CSF T1 (s)", 4.738), ("flip_deg", "flip angle (°)", 80), ("gap_mm", "slice gap (mm)", 0.5), ("order_0seq_1inter", "slice order (0 seq, 1 interleaved)", 1)]
fig, ax = plt.subplots(1, 5, figsize=(15, 3.3))
for a, (tag, lab, ref) in zip(ax, sw):
    d = L("sweeps", f"sweep_{tag}.npy")
    a.plot(d[:, 0], d[:, 1] * 100, "o-", color=COL["cardiac"], label="cardiac 12 mm/s")
    a.plot(d[:, 0], d[:, 2] * 100, "s-", color=COL["respiratory"], label="respiratory 1.2 mm/s")
    a.axvline(ref, color="0.7", ls=":"); a.set_xlabel(lab); a.set_ylabel("modulation (% of mean)")
ax[0].legend(fontsize=7.5, frameon=False)
fig.suptitle("Sensitivity of the inflow modulation to model/acquisition parameters (dotted: paper value)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig5_sweeps.png")); plt.close(fig)
print("ok")

# 6: mismatch study
import json
J = json.load(open(os.path.join(R, "mismatch", "summary.json")))
T = L("mismatch", "mismatch_table.npy")
names = J["names"]; tv = np.array(J["truth_v_mm_s"])
ratio = T[:, :3] / tv
seeds = [i for i, n in enumerate(names) if n.startswith("noise seed")]
others = [i for i, n in enumerate(names) if i not in seeds]
fig, ax = plt.subplots(1, 3, figsize=(14, 7.2), sharey=True)
y = np.arange(len(others))[::-1]
for k, (a, n) in enumerate(zip(ax, ["cardiac", "respiratory", "slow"])):
    lo, hi = ratio[seeds, k].min(), ratio[seeds, k].max()
    a.axvspan(lo, hi, color="0.85", label="noise-seed range (same truth)")
    a.axvline(1, color="k", lw=1)
    a.plot(ratio[others, k], y, "o", color=COL[["cardiac", "respiratory", "low"][k]])
    a.set_title(f"{n}  (truth {tv[k]} mm/s)"); a.set_xlabel("estimated / true amplitude")
    a.set_xlim(0, max(1.8, ratio[:, k].max() * 1.05)); a.grid(axis="x", color="0.92")
ax[0].set_yticks(y); ax[0].set_yticklabels([names[i] for i in others], fontsize=8)
ax[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("What moves the estimated velocity: model mismatch, timing errors, confounds, non-plug flow (fit quality R² stays 0.82–0.83 for the model mismatches)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig6_mismatch.png")); plt.close(fig)
print("fig6 ok")
