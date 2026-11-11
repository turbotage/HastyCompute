"""Where a readout spends its time in k-space: |k|(τ), radial speed, ADC time per radial shell, sampling density per
reconstruction set relative to Nyquist, and the data energy over τ. CPU only, reads seqdesign run directories.

    kspace_time.py <out dir> <energy_profile.npz> LABEL=RUN [LABEL=RUN ...] [--scan 600] [--frames 40,200]

Definitions (FOV D, Nyquist cell 1/D³, Δk = 1/D):
  time density   ADC seconds per unit radius of one spoke, s(r) = dτ/d|k| summed over its visits of the shell;
                 uniform-in-volume sampling has s(r) ∝ r².
  sample density reconstruction samples per Nyquist cell for one set of N readouts:
                 N · (#samples of a spoke in the shell) / (4π r² Δr D³). Sets the noise per cell (SNR, Fisher info).
  curve density  curve length per unit volume relative to Nyquist, N · l(r) / D² (l = path length per volume of one
                 spoke, rotation-averaged). < 1 means the curves are further apart than Δk: local acceleration
                 R(r) = D² / (N l(r)). Sets the aliasing. Same definition as hasty-seq metrics::density.
  set            one encode × one frame: N = (scan / TR) / 7 encodes / frames (uniform binning assumed).
Energy: measured per-sample energy of the simulated scan (current design, noise-free, energy_profile.py) and, for other
designs, the readout_metrics.py model E = P(|k|)·D(|k|, τ) built from that measurement.
"""
import json
import math
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout_metrics as rm  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical slots (white surface)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.8, "savefig.dpi": 150,
})

args = [a for a in sys.argv[1:]]
scan, frames = 600.0, [40, 200]
if "--scan" in args:
    i = args.index("--scan"); scan = float(args[i + 1]); del args[i:i + 2]
if "--frames" in args:
    i = args.index("--frames"); frames = [int(x) for x in args[i + 1].split(",")]; del args[i:i + 2]
out, prof = args[0], args[1]
runs = [a.split("=", 1) for a in args[2:]]
os.makedirs(os.path.join(out, "figs"), exist_ok=True)
em = rm.energy_model(prof)
zprof = np.load(prof)

NB = 40
res = {}
for lab, run in runs:
    R = rm.load_run(run)
    f, b = R["f"], R["b"]
    kmax, fov, tr = f["kmax"], R["s"]["problem"]["fov"], f["tr"]
    tau, r, mask = R["tau"], R["kr"] / kmax, R["mask"]
    # dense raster for curve length (k_adc is the spoke on the raster upsampled by b["upsample"])
    kd = np.load(os.path.join(run, "k_adc.npy"))
    up = b["upsample"]
    td = R["te"] - (b.get("i_echo", 0) * R["dt"]) + np.arange(len(kd)) * R["dt"] / up
    seg = np.diff(kd, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    seg_mid = np.linalg.norm(0.5 * (kd[1:] + kd[:-1]), axis=1) / kmax
    seg_t = 0.5 * (td[1:] + td[:-1])
    rec_ranges = b.get("recon", [[0, b["i_adc"]]])
    seg_rec = np.zeros(len(seg), bool)
    for a0, a1 in rec_ranges:
        seg_rec[a0 * up:a1 * up] = True
    edges = np.linspace(0, 1.0, NB + 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    dr = (edges[1] - edges[0]) * kmax
    vol = 4.0 / 3.0 * math.pi * ((edges[1:] * kmax) ** 3 - (edges[:-1] * kmax) ** 3)
    # time per shell (reconstruction samples): samples × dwell
    bi = np.clip(np.digitize(r, edges) - 1, 0, NB - 1)
    ns = np.bincount(bi[mask], minlength=NB).astype(float)
    t_shell = ns * rm.DWELL
    # curve length per shell (reconstruction part)
    bj = np.clip(np.digitize(seg_mid, edges) - 1, 0, NB - 1)
    l_shell = np.bincount(bj[seg_rec], weights=seg_len[seg_rec], minlength=NB)
    l1 = l_shell / vol
    n_ro = scan / tr
    n_enc = f["n_enc"]
    sets = {fr: n_ro / n_enc / fr for fr in frames}
    sets[1] = n_ro / n_enc
    cells = vol * fov ** 3  # Nyquist cells per shell
    samp = {fr: n * ns / cells for fr, n in sets.items()}
    curve = {fr: n * l1 / fov ** 2 for fr, n in sets.items()}
    # radial speed (|k|/kmax per ms) on the 2 µs samples
    vr = np.gradient(r, tau) * 1e-3
    vpath = np.linalg.norm(np.gradient(kd, axis=0), axis=1) / (R["dt"] / up)  # cycles/m per s
    # energy over τ (model built on the measured current-design spectrum)
    e = em["p_meas"](r) * em["decay"](r, tau)
    e = np.where(mask, e, 0.0)
    res[lab] = dict(run=run, tau=tau, r=r, mask=mask, vr=vr, rc=rc, t_shell=t_shell, ns=ns, l1=l1, vol=vol, samp=samp,
                    curve=curve, sets=sets, tr=tr, kmax=kmax, fov=fov, e=e, td=td, vpath=vpath, dr=dr,
                    t_rec=float(mask.sum() * rm.DWELL), t_adc=float(len(tau) * rm.DWELL))

# ── numbers ──
summ = {}
bands = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
for lab, q in res.items():
    tot = q["t_shell"].sum()
    tb = []
    for a0, a1 in zip(bands[:-1], bands[1:]):
        w = (q["rc"] >= a0) & (q["rc"] < a1)
        tb.append(dict(band=f"{a0:.2f}-{a1:.2f}", time_frac=float(q["t_shell"][w].sum() / tot), vol_frac=float(a1 ** 3 - a0 ** 3)))
    i_max = int(np.argmax(q["r"] * q["mask"]))
    s = dict(t_recon_ms=q["t_rec"] * 1e3, t_adc_ms=q["t_adc"] * 1e3, tr_ms=q["tr"] * 1e3, readouts_per_set={str(k): v for k, v in q["sets"].items()},
             tau_kmax_ms=float(q["tau"][i_max] * 1e3), tau_r50_ms=float(q["tau"][np.argmax(q["r"] >= 0.5)] * 1e3),
             tau_r90_ms=float(q["tau"][np.argmax(q["r"] >= 0.9)] * 1e3), bands=tb)
    for fr in [1] + frames:
        s[f"curve_density_rel_nyquist_{fr}fr"] = {f"{x:.2f}": float(np.interp(x, q["rc"], q["curve"][fr])) for x in (0.05, 0.1, 0.25, 0.5, 0.75, 0.99)}
        s[f"samples_per_cell_{fr}fr"] = {f"{x:.2f}": float(np.interp(x, q["rc"], q["samp"][fr])) for x in (0.05, 0.1, 0.25, 0.5, 0.75, 0.99)}
    ec = np.cumsum(q["e"]) / q["e"].sum()
    s["energy_tau_ms"] = {p: float(q["tau"][np.argmax(ec >= p)] * 1e3) for p in (0.5, 0.9, 0.99, 0.999)}
    summ[lab] = s
# measured energy of the current design (simulated scan)
em_e, em_tau, em_k = zprof["e"], zprof["tau"], zprof["kr"] / zprof["kr"].max()
ecm = np.cumsum(em_e) / em_e.sum()
summ["measured_current"] = dict(
    energy_tau_ms={p: float(em_tau[np.argmax(ecm >= p)] * 1e3) for p in (0.5, 0.87, 0.9, 0.99, 0.999)},
    return_leg_share=float(em_e[int(zprof["split"]):].sum() / em_e.sum()),
    energy_lowk_share={f"<{x}": float(em_e[em_k < x].sum() / em_e.sum()) for x in (0.05, 0.1, 0.25, 0.5)},
)
json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
np.savez(os.path.join(out, "kspace_time.npz"), **{f"{lab}__{k}": v for lab, q in res.items() for k, v in q.items() if isinstance(v, np.ndarray)})
print(json.dumps(summ, indent=1))

labs = list(res)
col = {lab: C[i] for i, lab in enumerate(labs)}

# ── fig 1: |k|(τ) and radial speed ──
fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, constrained_layout=True)
for lab, q in res.items():
    m = q["mask"]
    t = q["tau"] * 1e3
    ax[0].plot(t[m], q["r"][m], color=col[lab], label=f"{lab} (reconstruction)")
    if (~m).any():
        ax[0].plot(t[~m], q["r"][~m], color=col[lab], ls=(0, (2, 1.5)), lw=1.4, label=f"{lab} (navigator, not reconstructed)")
    ax[1].plot(t[m], q["vr"][m], color=col[lab], label=lab)
ax[0].set_ylabel("|k| / kmax")
ax[0].set_ylim(0, 1.05)
ax[0].set_title("Radius over time after the RF centre (kmax = 250 /m, 2 mm)", loc="left", fontsize=9.5, color=INK)
ax[0].legend(fontsize=8, loc="center right")
ax[1].set_ylabel("d(|k|/kmax)/dτ  [1/ms]")
ax[1].set_xlabel("τ after the RF centre [ms]")
ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].set_title("Radial speed (negative = moving back toward the centre)", loc="left", fontsize=9.5, color=INK)
fig.savefig(os.path.join(out, "figs", "radius_time.png"))
plt.close(fig)

# ── fig 2: time per shell, cumulative time vs volume ──
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
for lab, q in res.items():
    ax[0].plot(q["rc"], q["t_shell"] * 1e3 / (q["rc"][1] - q["rc"][0]), color=col[lab], label=lab)
    ct = np.concatenate([[0], np.cumsum(q["t_shell"])]) / q["t_shell"].sum()
    ax[1].plot(np.linspace(0, 1, NB + 1), ct, color=col[lab], label=lab)
xx = np.linspace(0, 1, 200)
ax[1].plot(xx, xx ** 3, color=INK2, ls=(0, (3, 2)), lw=1.2, label="uniform in volume (∝ r³)")
ax[0].set_xlabel("|k| / kmax"); ax[0].set_ylabel("ADC time per unit radius [ms]")
ax[0].set_title("Time spent per shell (one readout)", loc="left", fontsize=9.5, color=INK)
ax[1].set_xlabel("|k| / kmax"); ax[1].set_ylabel("fraction of reconstruction time below |k|")
ax[1].set_title("Cumulative time vs k-space volume", loc="left", fontsize=9.5, color=INK)
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8, loc="upper left")
fig.savefig(os.path.join(out, "figs", "time_per_shell.png"))
plt.close(fig)

# ── fig 3: density per set relative to Nyquist ──
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
dash = {frames[0]: "-", frames[-1]: (0, (3, 1.5))}
for lab, q in res.items():
    for fr in frames:
        ax[0].plot(q["rc"], q["curve"][fr], color=col[lab], ls=dash.get(fr, ":"), label=f"{lab}, {fr} frames ({q['sets'][fr]:.0f} readouts/set)")
        ax[1].plot(q["rc"], q["samp"][fr], color=col[lab], ls=dash.get(fr, ":"), label=f"{lab}, {fr} frames")
for a in ax:
    a.set_yscale("log"); a.set_xlabel("|k| / kmax"); a.axhline(1.0, color=INK2, lw=0.9)
ax[0].set_ylabel("curve density / Nyquist  (= 1/R)")
ax[0].set_title("Curve density per set (aliasing): 1 = Nyquist", loc="left", fontsize=9.5, color=INK)
ax[1].set_ylabel("samples per Nyquist cell (2 µs dwell)")
ax[1].set_title("Sample density per set (noise)", loc="left", fontsize=9.5, color=INK)
ax[0].legend(fontsize=7)
fig.savefig(os.path.join(out, "figs", "density_per_set.png"))
plt.close(fig)

# ── fig 4: data energy over τ ──
fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True, constrained_layout=True)
ax[0].semilogy(em_tau * 1e3, em_e / em_e.max(), color=col[labs[0]], lw=1.2, label=f"{labs[0]}: measured (ph0001 sim, 32 coils)")
for lab, q in res.items():
    if lab == labs[0]:
        continue
    t = q["tau"] * 1e3
    ax[0].semilogy(t[q["mask"]], (q["e"] / q["e"].max())[q["mask"]], color=col[lab], lw=1.2, label=f"{lab}: model from the measured spectrum")
ax[0].set_ylabel("energy per sample (rel. max)")
ax[0].set_ylim(1e-7, 2)
ax[0].set_title("Data energy per sample over τ", loc="left", fontsize=9.5, color=INK)
ax[0].legend(fontsize=8)
ax[1].plot(em_tau * 1e3, ecm, color=col[labs[0]], label=f"{labs[0]}: measured")
for lab, q in res.items():
    if lab == labs[0]:
        continue
    ax[1].plot(q["tau"] * 1e3, np.cumsum(q["e"]) / q["e"].sum(), color=col[lab], label=f"{lab}: model")
ax[1].set_ylabel("cumulative energy share"); ax[1].set_xlabel("τ after the RF centre [ms]")
ax[1].set_title("Cumulative energy share", loc="left", fontsize=9.5, color=INK)
ax[1].legend(fontsize=8, loc="center right")
fig.savefig(os.path.join(out, "figs", "energy_time.png"))
plt.close(fig)
print("figs →", os.path.join(out, "figs"))
