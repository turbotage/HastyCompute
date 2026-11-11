"""Signal budget of a readout: what the late samples lose, and the steady-state SNR trade-off of the readout length.

    signal_budget.py <out dir> <scan dir> <energy_profile.npz> LABEL=RUN [LABEL=RUN ...]

Part 1 (phantom, static tissue of the scan's truth maps on the reconstruction grid):
  m_c(x, Δτ) = s_c(x) [ w(x) e^{−R2*Δτ} e^{i2π f Δτ} + p_f(x) e^{−R2*Δτ} Σ_p a_p e^{i2π(f+δ_p)Δτ} ],
  f = field_ppm_recon · γ̄B0, six-peak fat (δ_p, a_p), Δτ = time after the echo (k = 0 at TE).
  Energy in the low-k ball |k| < q·kmax (FFT per coil, summed over coils) relative to Δτ = 0, for
    full      R2* + B0 + fat (what the late low-k samples see),
    r2s       R2* only (true signal loss; no phase),
    phase     B0 + fat only (no loss: Parseval keeps the total, the energy only moves in k),
  and the total energy Σ|m|² (Parseval; = R2* loss only).
  Compared with the measured return-leg / out-leg energy ratio at the same |k| of the simulated scan.
Part 2 (analytic + designed waveforms): spoiled-GRE steady state S(α, TR) = sin α (1−E1)/(1−cos α E1), Ernst
  cos α_E = E1. SNR per unit scan time of an unbiased (T2*-deconvolved) reconstruction, from the Fisher information per
  Nyquist cell: I(cell) ∝ (S²/TR) Σ_{samples in the cell} e^{−2Δτ/T2*}; variance ∝ mean over cells of 1/I. And the
  "accept the blur" SNR ∝ S √(T_rec/TR) · mean_k e^{−Δτ/T2*}. Designs use their own Δτ(|k|); the analytic curves
  assume centre-out with time ∝ k-space volume (Δτ = u T_rec, u uniform) and the overhead T_ov = TR − T_rec.
"""
import json
import math
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402
import scipy.fft as sfft  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import readout_metrics as rm  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.8, "savefig.dpi": 150,
})

out, scan, prof = sys.argv[1], sys.argv[2], sys.argv[3]
runs = [a.split("=", 1) for a in sys.argv[4:]]
os.makedirs(os.path.join(out, "figs"), exist_ok=True)
meta = json.load(open(os.path.join(scan, "meta.json")))
GB0 = rm.GBAR * meta["b0"] / 1e6  # Hz per ppm (1 ppm of the Larmor frequency)
FAT_PPM = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
FAT_A = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
KMAX = 250.0
summ = {}
if os.environ.get("SKIP_PHANTOM") is not None and os.path.exists(os.path.join(out, "summary.json")):
    summ = {k: v for k, v in json.load(open(os.path.join(out, "summary.json"))).items() if k.startswith("phantom") or k == "maps"}

# ── part 1: phantom ──
if os.environ.get("SKIP_PHANTOM") is None:
    md = os.environ.get("MAPS")  # steady-state weighted maps (ss_maps.py); default: raw PD truth maps
    if md:
        w = np.load(os.path.join(md, "ss_water.npy")).astype(np.float64)
        pf = np.load(os.path.join(md, "ss_fat.npy")).astype(np.float64)
    else:
        w = np.load(os.path.join(scan, "truth_pd_water.npy")).astype(np.float64)
        pf = np.load(os.path.join(scan, "truth_pd_fat.npy")).astype(np.float64)
    summ["maps"] = md or "raw PD"
    r2 = np.nan_to_num(np.load(os.path.join(scan, "truth_r2s.npy")).astype(np.float64), nan=0.0)
    fppm = np.nan_to_num(np.load(os.path.join(scan, "field_ppm_recon.npy")).astype(np.float64), nan=0.0)
    sm = np.load(os.path.join(scan, "smaps_true.npy"), mmap_mode="r")
    dims, res = w.shape, meta["res"]
    kk = [sfft.fftfreq(n, d=r) for n, r in zip(dims, res)]
    kr = np.sqrt(kk[0][:, None, None] ** 2 + kk[1][None, :, None] ** 2 + kk[2][None, None, :] ** 2)
    QS = (0.02, 0.05, 0.1)
    balls = {q: kr < q * KMAX for q in QS}
    f_hz = fppm * GB0
    dts = np.array([0.0, 0.05, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 7.93, 9.49]) * 1e-3
    # radial shells as in energy_profile.py (50 bins to kmax), only the low-k ones
    NSH = 50
    shb = np.floor(kr / KMAX * NSH).astype(np.int64)
    shsel = shb < 6
    shb_s = shb[shsel]
    Esh = {v: np.zeros((len(dts), 6)) for v in ("full", "r2s", "phase")}
    body = (w + pf) > 0
    print(f"grid {dims}, res {res}, {body.sum()} tissue voxels; field 1–99 % {np.percentile(f_hz[body], 1):.0f}…{np.percentile(f_hz[body], 99):.0f} Hz")
    variants = ("full", "r2s", "phase")
    E = {v: {q: np.zeros(len(dts)) for q in QS} for v in variants}
    Etot = np.zeros(len(dts))
    nc = sm.shape[-1]
    smc = [np.ascontiguousarray(sm[..., c]) for c in range(nc)]

    def image(dt, v):
        dec = np.exp(-r2 * dt) if v in ("full", "r2s") else 1.0
        if v == "r2s":
            return ((w + pf) * dec).astype(np.complex128)
        ph = np.exp(2j * np.pi * f_hz * dt)
        fat = sum(a * np.exp(2j * np.pi * d * GB0 * dt) for a, d in zip(FAT_A, FAT_PPM))
        return (w + pf * fat) * ph * dec

    for i, dt in enumerate(dts):
        for v in variants:
            m = image(dt, v)
            if v == "r2s":
                Etot[i] = sum(float((np.abs(s) ** 2 * np.abs(m) ** 2).sum()) for s in smc)
            for s in smc:
                y = sfft.fftn((s * m).astype(np.complex64), workers=4)
                p = np.abs(y) ** 2
                for q in QS:
                    E[v][q][i] += float(p[balls[q]].sum())
                Esh[v][i] += np.bincount(shb_s, weights=p[shsel], minlength=6)
        print(f"Δτ {dt*1e3:5.2f} ms: " + ", ".join(f"{v} {E[v][0.1][i]/E[v][0.1][0]:.3f}" for v in variants) + f", total {Etot[i]/Etot[0]:.3f}")
    # measured: return / out ratio at the same |k| (low k), current design
    z = np.load(prof)
    rc, p_out, p_ret = z["rc"] / z["kr"].max(), z["p_out"], z["p_ret"]
    meas = {f"{x:.3f}": float(r) for x, r in zip(rc, p_ret / p_out) if x < 0.12 and np.isfinite(r)}
    i_o, i_r = 1, int(np.argmin(np.abs(dts - 7.93e-3)))
    shell_ratio = {v: (Esh[v][i_r] / Esh[v][i_o]).tolist() for v in Esh}
    print("shell |k|/kmax centres:", [round((j + 0.5) / NSH, 3) for j in range(6)])
    for v_ in shell_ratio:
        print(f"  model return/out ({v_}):", [round(x, 3) for x in shell_ratio[v_]])
    print("  measured return/out:", {k_: round(x, 3) for k_, x in meas.items()})
    summ["phantom_shell_ratio_7.93_over_0.05"] = dict(centres=[(j + 0.5) / NSH for j in range(6)], **shell_ratio)
    summ["phantom"] = dict(dt_ms=(dts * 1e3).tolist(), total=(Etot / Etot[0]).tolist(),
                           lowk={v: {str(q): (E[v][q] / E[v][q][0]).tolist() for q in QS} for v in variants},
                           measured_return_over_out=meas)
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    lab = {"full": "R2* + B0 + fat", "r2s": "R2* only (true loss)", "phase": "B0 + fat only (no loss)"}
    for j, v in enumerate(variants):
        ax[0].plot(dts * 1e3, E[v][0.1] / E[v][0.1][0], color=C[j], label=lab[v], marker="o", ms=4)
    ax[0].plot(dts * 1e3, Etot / Etot[0], color=INK2, ls=(0, (3, 2)), lw=1.3, label="total energy, all k (Parseval)")
    mx = [float(x) for x in meas]
    ax[0].scatter([7.95] * len(mx), list(meas.values()), color=C[3], s=16, zorder=3, label="measured return/out, |k| < 0.12 kmax")
    ax[0].set_xlabel("Δτ after the echo [ms]"); ax[0].set_ylabel("energy in |k| < 0.1 kmax (rel. Δτ = 0)")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Low-k energy of static tissue over time (ph0001, 32 coils)", loc="left", fontsize=9.5, color=INK)
    ax[0].legend(fontsize=7.5, loc="lower left")
    for j, q in enumerate(QS):
        ax[1].plot(dts * 1e3, E["full"][q] / E["full"][q][0], color=C[j], label=f"|k| < {q:g} kmax", marker="o", ms=4)
    ax[1].set_xlabel("Δτ after the echo [ms]"); ax[1].set_ylabel("energy (R2* + B0 + fat), rel. Δτ = 0")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Smaller balls lose more: B0 moves energy outward", loc="left", fontsize=9.5, color=INK)
    ax[1].legend(fontsize=8)
    fig.savefig(os.path.join(out, "figs", "lowk_decay_split.png"))
    plt.close(fig)

# ── part 2: steady state and SNR vs readout length ──
TIS = {  # T1, T2* from the phantom's tissue table (post-contrast, gadolinium)
    "arterial blood": (0.388, 0.090), "venous blood": (0.388, 0.030), "myocardium": (1.037, 0.0256),
    "liver": (0.555, 0.0238), "fat": (0.371, 0.0341),
}
FLIP = math.radians(meta["flip_deg"])


def spgr(a, tr, t1):
    e1 = math.exp(-tr / t1)
    return math.sin(a) * (1 - e1) / (1 - math.cos(a) * e1)


def ernst(tr, t1):
    return math.acos(math.exp(-tr / t1))


designs = {}
for lab_, run in runs:
    R = rm.load_run(run)
    m = R["mask"]
    designs[lab_] = dict(tau=R["tau"][m] - R["te"], r=R["kr"][m] / R["kmax"], tr=R["f"]["tr"], t_rec=float(m.sum() * rm.DWELL))
NB = 40
edges = np.linspace(0, 1, NB + 1)
vol = edges[1:] ** 3 - edges[:-1] ** 3


def design_snr(d, t2s, s):
    """unbiased and blurred SNR (relative units) of a design: Fisher per radial bin (cells ∝ volume)"""
    b = np.clip(np.digitize(d["r"], edges) - 1, 0, NB - 1)
    info = np.bincount(b, weights=np.exp(-2 * d["tau"] / t2s), minlength=NB) * rm.DWELL / d["tr"]  # per unit scan time
    var = (vol / np.maximum(info / vol, 1e-300)).sum()  # Σ_bins cells × 1/I_cell (cells ∝ vol)
    ub = s * math.sqrt(1.0 / var)
    psf = (np.bincount(b, weights=np.exp(-d["tau"] / t2s), minlength=NB) / np.maximum(np.bincount(b, minlength=NB), 1) * vol).sum()
    bl = s * math.sqrt(d["t_rec"] / d["tr"]) * psf
    return ub, bl


def analytic(t_rec, t_ov, t1, t2s, a=None):
    tr = t_ov + t_rec
    s = spgr(ernst(tr, t1) if a is None else a, tr, t1)
    x = 2 * t_rec / t2s
    y = t_rec / t2s
    ub = s * math.sqrt(t_rec / tr * x / math.expm1(x))
    bl = s * math.sqrt(t_rec / tr) * (-math.expm1(-y) / y)
    return ub, bl


tab = {}
ref = list(designs)[0]
for tn, (t1, t2s) in TIS.items():
    row = {}
    for lab_, d in designs.items():
        aE = ernst(d["tr"], t1)
        sE, s12 = spgr(aE, d["tr"], t1), spgr(FLIP, d["tr"], t1)
        ubE, blE = design_snr(d, t2s, sE)
        ub12, bl12 = design_snr(d, t2s, s12)
        row[lab_] = dict(ernst_deg=math.degrees(aE), s12_over_sE=s12 / sE, sE2_over_tr=sE ** 2 / d["tr"],
                         ub_ernst=ubE, bl_ernst=blE, ub_12=ub12, bl_12=bl12)
    r0 = row[ref]["ub_ernst"]
    b0 = row[ref]["bl_ernst"]
    for lab_ in row:
        row[lab_]["ub_ernst_rel"] = row[lab_]["ub_ernst"] / r0
        row[lab_]["ub_12_rel"] = row[lab_]["ub_12"] / r0
        row[lab_]["bl_ernst_rel"] = row[lab_]["bl_ernst"] / b0
        row[lab_]["bl_12_rel"] = row[lab_]["bl_12"] / b0
    # optimum readout length of the analytic centre-out family (Ernst, overhead of the v2 family)
    t_ov = 4.6e-3
    grid = np.linspace(1e-3, 120e-3, 2400)
    u = np.array([analytic(t, t_ov, t1, t2s)[0] for t in grid])
    row["opt_t_rec_ms_unbiased"] = float(grid[np.argmax(u)] * 1e3)
    row["opt_rel_to_v2_unbiased"] = float(u.max() / analytic(9.49e-3, t_ov, t1, t2s)[0])
    tab[tn] = row
summ["snr"] = tab
summ["flip_deg"] = meta["flip_deg"]
json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)

print("\nflip", meta["flip_deg"], "deg; SNR per unit scan time relative to", ref, "(Ernst flip per design)")
for tn, row in tab.items():
    s = "  ".join(f"{l}: αE {row[l]['ernst_deg']:.1f}° S12/SE {row[l]['s12_over_sE']:.3f} SE²/TR {row[l]['sE2_over_tr']:.3f} "
                  f"unbiased {row[l]['ub_ernst_rel']:.3f} (12°: {row[l]['ub_12_rel']:.3f}) blurred {row[l]['bl_ernst_rel']:.3f}" for l in designs)
    print(f"{tn:15s} {s}\n{'':15s} optimum T_rec (unbiased, centre-out, T_ov 4.6 ms) {row['opt_t_rec_ms_unbiased']:.1f} ms, ×{row['opt_rel_to_v2_unbiased']:.3f} vs 9.49 ms")

# figure: SNR vs T_rec, analytic centre-out family, normalised to the reference design's exact value
fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
tg = np.linspace(1e-3, 40e-3, 400)
for j, (tn, (t1, t2s)) in enumerate(TIS.items()):
    r0u, r0b = tab[tn][ref]["ub_ernst"], tab[tn][ref]["bl_ernst"]
    ub = np.array([analytic(t, 4.6e-3, t1, t2s)[0] for t in tg]) / r0u
    bl = np.array([analytic(t, 4.6e-3, t1, t2s)[1] for t in tg]) / r0b
    ax[0].plot(tg * 1e3, ub, color=C[j], label=f"{tn} (T2* {t2s*1e3:.0f} ms)")
    ax[1].plot(tg * 1e3, bl, color=C[j], label=tn)
    for k, (lab_, d) in enumerate(designs.items()):
        mk = ["o", "s", "^"][k % 3]
        ax[0].scatter([d["t_rec"] * 1e3], [tab[tn][lab_]["ub_ernst_rel"]], color=C[j], marker=mk, s=22, zorder=3, edgecolor="white", linewidth=0.6)
        ax[1].scatter([d["t_rec"] * 1e3], [tab[tn][lab_]["bl_ernst_rel"]], color=C[j], marker=mk, s=22, zorder=3, edgecolor="white", linewidth=0.6)
for k, lab_ in enumerate(designs):
    ax[1].scatter([], [], color=INK2, marker=["o", "s", "^"][k % 3], s=22, label=f"design: {lab_}")
ax[0].set_title("Unbiased (T2*-deconvolved) SNR per scan time\ncentre-out, overhead 4.6 ms, Ernst flip per TR", loc="left", fontsize=9.5, color=INK)
ax[1].set_title("SNR accepting the T2* blur\nsame family", loc="left", fontsize=9.5, color=INK)
for a in ax:
    a.set_xlabel("reconstruction window T_rec [ms]")
    a.axhline(1.0, color=INK2, lw=0.8)
ax[0].set_ylabel(f"SNR relative to '{ref}'")
ax[0].legend(fontsize=7.5, loc="lower right")
ax[1].legend(fontsize=7.5, loc="lower right")
fig.savefig(os.path.join(out, "figs", "snr_vs_readout.png"))
plt.close(fig)
print("figs →", os.path.join(out, "figs"))
