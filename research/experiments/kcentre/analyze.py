"""k-centre signals: measured (k0.npz) vs physics predictions (pred.npz); figures + numbers (numbers.json)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SCAN = Path("data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan")
KC = Path("data/hasty/phantom/factory_arms/ph0001/kcentre")
FIG = Path("research/experiments/kcentre/figs")
FIG.mkdir(parents=True, exist_ok=True)
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cfe0f6", "#2a78d6", "#0d2f5c"])
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 120, "figure.dpi": 120})

z = np.load(KC / "k0.npz")
p = np.load(KC / "pred.npz")
ro = np.load(SCAN / "readouts.npy")
resp = np.load(SCAN / "resp.npy")
phys = np.load(SCAN / "physio.npy")
nro = ro.shape[0]
enc = ro[:, 1].astype(int)
tex = ro[:, 0]
r = resp[:, 0]
card = ro[:, 4]
bel = phys[:, 3]
tau = p["tau"]
DT = tau[1] - tau[0]
meas = z["exact"]  # [nro, nc, 2] clean, sample 0 and 3989
measn = z["exact_noisy"]
fitn = z["fit_noisy"]
sub = p["sub"]
state = p["state"]
nc = meas.shape[1]


def rho_coil(s):
    """per-coil in/out ratio from [n, nc, 2] (Σ over readouts)"""
    return (s[..., 1] * np.conj(s[..., 0])).sum(0) / (np.abs(s[..., 0]) ** 2).sum(0)


def rho_all(s):
    return (s[..., 1] * np.conj(s[..., 0])).sum() / (np.abs(s[..., 0]) ** 2).sum()


def implied(rho):
    return -np.angle(rho) / (2 * np.pi * DT), -np.log(np.abs(rho)) / DT


pred_state_all = p["P0"][state]  # [nro, nc, 2]
variants = {
    "static maps, PD only": p["static_pd"][None],
    "static maps, steady-state weight": p["static_ss"][None],
    "per state (resp × cardiac)": pred_state_all[sub],
    "state + concomitant": p["Pconc"],
    "state + flow": p["Pflow"],
    "state + concomitant + flow": p["Pfull"],
}
water_only = {"static maps, PD only": p["static_pd_water"][None], "static maps, steady-state weight": p["static_ss_water"][None]}
num = {"dt_ms": DT * 1e3}
mr = rho_all(meas[sub])
num["measured_subset"] = dict(zip(("df_hz", "r_eff"), map(float, implied(mr))))
num["measured_all"] = dict(zip(("df_hz", "r_eff"), map(float, implied(rho_all(meas)))))
num["measured_all_noisy_exact"] = dict(zip(("df_hz", "r_eff"), map(float, implied(rho_all(measn)))))
num["measured_all_noisy_fit"] = dict(zip(("df_hz", "r_eff"), map(float, implied(rho_all(fitn)))))
for k, v in variants.items():
    num[k] = dict(zip(("df_hz", "r_eff"), map(float, implied(rho_all(v)))))
for k, v in water_only.items():
    num[k + " (water only)"] = dict(zip(("df_hz", "r_eff"), map(float, implied(rho_all(v)))))
ts = p["tstat"]  # per state: [mean decay, mean Δf, fat signal fraction]
tw = ts[state[sub]].mean(0)
num["truth_signal_weighted_state_mean"] = {"decay_rate": float(tw[0]), "df_hz": float(tw[1]), "fat_signal_frac": float(tw[2])}
for k in ("static_pd_truth", "static_ss_truth"):
    num[k] = dict(zip(("decay_rate", "df_hz", "fat_signal_frac"), map(float, p[k])))

# per-coil ratios: measured on the subset vs each prediction
rc_m = rho_coil(meas[sub])
rc_p = {k: rho_coil(v) for k, v in variants.items()}
w_c = (np.abs(meas[sub][..., 0]) ** 2).mean(0)
w_c = w_c / w_c.max()

# coil-map consistency: complex scale α fitted per readout between measured and full prediction (out sample)
def coil_consistency(m, pr):
    a = (np.conj(pr) * m).sum(1) / (np.abs(pr) ** 2).sum(1)
    res = np.linalg.norm(m - a[:, None] * pr, axis=1) / np.linalg.norm(m, axis=1)
    return res


for j, nm in ((0, "out"), (1, "in")):
    num[f"coil_consistency_{nm}_full"] = float(np.median(coil_consistency(meas[sub][..., j], p["Pfull"][..., j])))
    num[f"coil_consistency_{nm}_state"] = float(np.median(coil_consistency(meas[sub][..., j], pred_state_all[sub][..., j])))
    num[f"coil_consistency_{nm}_static_ss"] = float(np.median(coil_consistency(meas[sub][..., j], np.broadcast_to(p["static_ss"][:, j], (sub.size, nc)))))

VK = ["static maps, PD only", "static maps, steady-state weight", "per state (resp × cardiac)", "state + concomitant + flow"]
VC = [C4, C2, C3, C1]

# (a) phase
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
for k, c in zip(VK, VC):
    ax[0].scatter(np.angle(rc_m), np.angle(rc_m) + np.angle(rc_p[k] * np.conj(rc_m)), s=10 + 40 * w_c, color=c, edgecolor="white", linewidth=0.5, label=k, zorder=3)
lim = [-np.pi - 0.3, np.pi + 0.3]
ax[0].plot(lim, lim, color=INK2, lw=0.8, ls="--", zorder=1)
ax[0].set(xlabel="measured arg(s_in / s_out) per coil (rad)", ylabel="predicted arg(s_in / s_out) (rad, unwrapped to measured)",
          title="(a) phase of the in/out k-centre ratio, 32 coils")
ax[0].legend(fontsize=7, loc="upper left")
labels = ["measured"] + VK
vals = [num["measured_subset"]["df_hz"]] + [num[k]["df_hz"] for k in VK]
cols = [INK] + VC
ax[1].barh(range(len(vals))[::-1], vals, color=cols, height=0.6)
for i, v in enumerate(vals):
    ax[1].text(max(v, 0), len(vals) - 1 - i, f" {v:.1f} Hz", va="center", ha="left", fontsize=7.5, color=INK)
ax[1].axvline(num["truth_signal_weighted_state_mean"]["df_hz"], color=INK2, ls=":", lw=1)
ax[1].text(num["truth_signal_weighted_state_mean"]["df_hz"], -0.9, "truth signal-weighted\nmean Δf (water+fat) ", fontsize=7, color=INK2, ha="right")
ax[1].set_yticks(range(len(vals))[::-1], labels, fontsize=7.5)
ax[1].set(xlabel="implied mean off-resonance, coil-combined (Hz)", title="implied Δf = −arg ρ / (2π Δt)")
ax[1].set_ylim(-1.3, len(vals) - 0.4)
fig.tight_layout()
fig.savefig(FIG / "a_ratio_phase.png")
plt.close(fig)

# (b) magnitude
fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
for k, c in zip(VK, VC):
    ax[0].scatter(np.abs(rc_m), np.abs(rc_p[k]), s=10 + 40 * w_c, color=c, edgecolor="white", linewidth=0.5, label=k, zorder=3)
lim = [0.02, 4]
ax[0].plot(lim, lim, color=INK2, lw=0.8, ls="--", zorder=1)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set(xlabel="measured |s_in / s_out| per coil", ylabel="predicted |s_in / s_out|", title="(b) magnitude of the in/out ratio, 32 coils")
ax[0].legend(fontsize=7, loc="upper left")
vals = [num["measured_subset"]["r_eff"]] + [num[k]["r_eff"] for k in VK]
ax[1].barh(range(len(vals))[::-1], vals, color=cols, height=0.6)
for i, v in enumerate(vals):
    ax[1].text(v, len(vals) - 1 - i, f" {v:.0f} /s", va="center", fontsize=7.5, color=INK)
td = num["truth_signal_weighted_state_mean"]["decay_rate"]
ax[1].axvline(td, color=INK2, ls=":", lw=1)
ax[1].text(td, -0.9, f" truth signal-weighted mean decay rate {td:.0f} /s", fontsize=7, color=INK2)
ax[1].set_yticks(range(len(vals))[::-1], labels, fontsize=7.5)
ax[1].set(xlabel="implied effective decay rate −ln|ρ| / Δt (1/s)", title="decay + dephasing (coil-combined)")
ax[1].set_ylim(-1.3, len(vals) - 0.4)
fig.tight_layout()
fig.savefig(FIG / "b_ratio_magnitude.png")
plt.close(fig)

# (c) time series, encode 0, three coils (strongest, median, weak)
order = np.argsort(-w_c)
coils = [order[0], order[nc // 2], order[-3]]
e0 = np.flatnonzero(enc == 0)
fig, ax = plt.subplots(3, 4, figsize=(11, 6.4), sharex=True)
for i, c in enumerate(coils):
    for j, nm in ((0, "out"), (1, "in")):
        s = meas[e0, c, j]
        med = np.median(s.real) + 1j * np.median(s.imag)
        m_ = np.abs(s) / np.abs(med)
        ph = np.angle(s * np.conj(med))
        for col, y, lab in ((2 * j, m_, f"|s_{nm}| / median"), (2 * j + 1, ph, f"arg s_{nm} − median (rad)")):
            win = tex[e0] < 120
            sc = ax[i, col].scatter(tex[e0][win], y[win], c=r[e0][win], cmap=SEQ, s=2.5, linewidth=0, rasterized=True)
            ax[i, col].set_ylabel(lab, fontsize=7.5)
            if i == 0:
                ax[i, col].set_title(f"{nm}-centre {'magnitude' if col % 2 == 0 else 'phase'}", fontsize=8.5)
    ax[i, 0].text(0.01, 0.97, f"coil {c}", transform=ax[i, 0].transAxes, va="top", fontsize=7.5, color=INK)
for a in ax[-1]:
    a.set_xlabel("scan time (s)")
cb = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
cb.set_label("true breathing state r")
fig.suptitle("(c) k-centre signals, first 120 s of the scan, reference encode (enc 0), coloured by true breathing state", fontsize=9)
fig.savefig(FIG / "c_timeseries.png", dpi=85)
plt.close(fig)


# (c2) correlations per coil: with bellows (Pearson) and cardiac phase (R² of a 2-harmonic fit); reference encode
def r2_card(y, ph):
    X = np.stack([np.ones_like(ph)] + [f(2 * np.pi * h * ph) for h in (1, 2) for f in (np.cos, np.sin)], 1)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def r2_lin(y, x):
    return np.corrcoef(y, x)[0, 1] ** 2


feat = {}
for j, nm in ((0, "out"), (1, "in")):
    s = meas[e0][..., j]
    med = np.median(s.real, 0) + 1j * np.median(s.imag, 0)
    feat[f"|s_{nm}|"] = np.abs(s) / np.abs(med)
    feat[f"arg s_{nm}"] = np.angle(s * np.conj(med))
rr0 = (meas[e0][..., 1] * np.conj(meas[e0][..., 0])).sum(1) / (np.abs(meas[e0][..., 0]) ** 2).sum(1)
cc = {}
for k, y in feat.items():
    cc[k] = (np.array([r2_lin(y[:, c], bel[e0]) for c in range(nc)]), np.array([r2_card(y[:, c], card[e0]) for c in range(nc)]),
             np.array([r2_lin(y[:, c], r[e0]) for c in range(nc)]))
df0 = -np.angle(rr0) / (2 * np.pi * DT)
num["corr_enc0_coilcombined_df"] = {"r2_bellows": float(r2_lin(df0, bel[e0])), "r2_true_r": float(r2_lin(df0, r[e0])), "r2_cardiac": float(r2_card(df0, card[e0]))}
num["corr_enc0_median_over_coils"] = {k: {"r2_bellows": float(np.median(v[0])), "r2_cardiac": float(np.median(v[1])), "r2_true_r": float(np.median(v[2]))} for k, v in cc.items()}
num["corr_enc0_max_over_coils"] = {k: {"r2_bellows": float(np.max(v[0])), "r2_cardiac": float(np.max(v[1]))} for k, v in cc.items()}
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
names = list(cc)
for a, ii, ttl in ((ax[0], 0, "R² with the respiratory bellows"), (ax[1], 1, "R² with cardiac phase (2-harmonic fit)")):
    data = [cc[k][ii] for k in names]
    a.boxplot(data, orientation="horizontal", widths=0.5, showfliers=False, medianprops={"color": C2, "lw": 1.5}, boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
    for i, d in enumerate(data):
        a.scatter(d, np.full(d.size, i + 1) + np.random.default_rng(i).uniform(-0.15, 0.15, d.size), s=6, color=C1, zorder=3, linewidth=0)
    a.set_yticks(range(1, len(names) + 1), names)
    a.set(xlabel="R² per coil (32 coils)", title=ttl, xlim=(0, 1))
fig.suptitle("(c) per-coil k-centre features vs physiology, reference encode", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "c2_correlations.png")
plt.close(fig)

# (d) implied off-resonance per readout vs breathing
rho_r = (meas[..., 1] * np.conj(meas[..., 0])).sum(1) / (np.abs(meas[..., 0]) ** 2).sum(1)
df_r = -np.angle(rho_r) / (2 * np.pi * DT)
rho_s = (pred_state_all[..., 1] * np.conj(pred_state_all[..., 0])).sum(1) / (np.abs(pred_state_all[..., 0]) ** 2).sum(1)
df_s = -np.angle(rho_s) / (2 * np.pi * DT)
pf = p["Pfull"]
df_f = -np.angle((pf[..., 1] * np.conj(pf[..., 0])).sum(1) / (np.abs(pf[..., 0]) ** 2).sum(1)) / (2 * np.pi * DT)
re_r = -np.log(np.abs(rho_r)) / DT
num["df_per_readout"] = {"mean": float(df_r.mean()), "std": float(df_r.std()), "r2_true_r": float(r2_lin(df_r, r)), "r2_bellows": float(r2_lin(df_r, bel)),
                         "slope_hz_per_r": float(np.polyfit(r, df_r, 1)[0]), "pred_state_slope_hz_per_r": float(np.polyfit(r, df_s, 1)[0]),
                         "pred_state_r2_true_r": float(r2_lin(df_s, r)), "pred_full_vs_meas_r2_subset": float(r2_lin(df_r[sub], df_f)),
                         "pred_state_vs_meas_r2": float(r2_lin(df_r, df_s))}
# rotation-only part: residual after removing the full prediction
num["df_per_readout"]["meas_minus_full_std_subset"] = float((df_r[sub] - df_f).std())
num["df_per_readout"]["meas_minus_full_mean_subset"] = float((df_r[sub] - df_f).mean())
num["reff_per_readout"] = {"mean": float(re_r.mean()), "std": float(re_r.std()), "r2_true_r": float(r2_lin(re_r, r)), "r2_bellows": float(r2_lin(re_r, bel))}
fig, ax = plt.subplots(1, 3, figsize=(11.5, 4.3))
ax[0].scatter(tex, df_r, c=r, cmap=SEQ, s=0.5, linewidth=0, rasterized=True)
ax[0].set(xlabel="scan time (s)", ylabel="implied Δf per readout (Hz)", title="(d) implied off-resonance over the scan")
ax[0].set_xlim(0, 60)
ax[0].set_ylim(0, 12)
ax[0].text(0.01, 0.02, "first 60 s; colour = breathing state", transform=ax[0].transAxes, fontsize=7, color=INK2)
ax[1].scatter(r, df_r, s=0.5, color="#9bbfea", linewidth=0, rasterized=True, label="measured, every readout")
ax[1].scatter(r[sub], df_f, s=1.5, color=C2, alpha=0.6, linewidth=0, rasterized=True)
bins = np.linspace(r.min(), r.max(), 15)
bc = 0.5 * (bins[1:] + bins[:-1])
ib = np.digitize(r, bins) - 1
mb = np.array([df_r[ib == i].mean() if (ib == i).any() else np.nan for i in range(bc.size)])
sb = np.array([df_s[ib == i].mean() if (ib == i).any() else np.nan for i in range(bc.size)])
ax[1].plot(bc, mb, color=C1, lw=2, marker="o", ms=4, label="measured, binned mean")
ax[1].plot(bc, sb, color=C3, lw=2, marker="s", ms=4, label="predicted per state (no conc.), binned")
ax[1].set(xlabel="true breathing state r", ylabel="implied Δf (Hz)", title="vs breathing")
fb = np.array([df_f[ib[sub] == i].mean() if (ib[sub] == i).any() else np.nan for i in range(bc.size)])
ax[1].plot(bc, fb, color=C2, lw=2, marker="^", ms=4, label="predicted: state + conc. + flow, binned")
from matplotlib.lines import Line2D
hs = [Line2D([], [], color="#9bbfea", marker="o", ls="", ms=4, label="measured, every readout"),
      Line2D([], [], color=C2, marker="o", ls="", ms=3, alpha=0.6, label="predicted full, subset readouts"),
      Line2D([], [], color=C1, marker="o", lw=2, ms=4, label="measured, binned mean"),
      Line2D([], [], color=C2, marker="^", lw=2, ms=4, label="predicted full, binned mean"),
      Line2D([], [], color=C3, marker="s", lw=2, ms=4, label="predicted per state (no conc./flow), binned")]
ax[1].legend(handles=hs, fontsize=6.3, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
ax[2].scatter(df_f, df_r[sub], s=1.5, color=C1, linewidth=0, rasterized=True)
l2 = [min(df_f.min(), df_r[sub].min()), max(df_f.max(), df_r[sub].max())]
ax[2].plot(l2, l2, color=INK2, lw=0.8, ls="--")
ax[2].set(xlabel="predicted Δf per readout, full (Hz)", ylabel="measured (Hz)", title=f"per readout: R² = {num['df_per_readout']['pred_full_vs_meas_r2_subset']:.2f}")
fig.tight_layout()
fig.savefig(FIG / "d_df_vs_resp.png")
plt.close(fig)

rst = np.unique(np.round(r[np.isfinite(r)], 0))
ts6 = ts.reshape(20, 6, 3).mean(0)
num["truth_state_df_vs_r"] = {"r": [0.0, 0.28, 0.56, 0.84, 1.12, 1.4], "signal_weighted_df_hz": list(map(float, ts6[:, 1])),
                              "decay_rate": list(map(float, ts6[:, 0])), "fat_frac": list(map(float, ts6[:, 2]))}
num["df_slope_subset"] = {"measured": float(np.polyfit(r[sub], df_r[sub], 1)[0]), "pred_full": float(np.polyfit(r[sub], df_f, 1)[0])}

# (e) per encode: k-centre phase difference against the reference encode (pairs of the same rotation group)
ng = nro // 7
G = np.arange(ng)
gsub = sub.reshape(-1, 7)[:, 0] // 7


def dphi(s, e, j, groups):
    a = s[groups * 7 + e][:, :, j]
    b = s[groups * 7][:, :, j]
    return np.angle((a * np.conj(b)).sum())


def dphi_pred(P, e, j):
    P7 = P.reshape(-1, 7, nc, 2)
    return np.angle((P7[:, e, :, j] * np.conj(P7[:, 0, :, j])).sum())


rows = []
for e in range(1, 7):
    for j, nm in ((0, "out"), (1, "in")):
        rows.append({"enc": e, "centre": nm, "meas_all": float(dphi(meas, e, j, G)), "meas_subset": float(dphi(meas, e, j, gsub)),
                     "pred_full": float(dphi_pred(p["Pfull"], e, j)), "pred_flow": float(dphi_pred(p["Pflow"], e, j)),
                     "pred_conc": float(dphi_pred(p["Pconc"], e, j)), "pred_state": float(dphi_pred(pred_state_all[sub], e, j))})
num["encode_dphi"] = rows
par = p["par"].reshape(-1, 7, 2, 7)
num["encode_dm1_out"] = [list(map(float, (par[0, e, 0, :3] - par[0, 0, 0, :3]))) for e in range(7)]
num["encode_dw_out"] = [list(map(float, (par[:, e, 0, 3:] - par[:, 0, 0, 3:]).mean(0))) for e in range(7)]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, nm in zip(ax, ("out", "in")):
    rr = [x for x in rows if x["centre"] == nm]
    x = np.arange(len(rr))
    wdt = 0.2
    for i, (key, c, lab) in enumerate((("meas_all", INK, "measured (all readouts)"), ("pred_full", C1, "predicted: conc. + flow"),
                                       ("pred_flow", C2, "predicted: flow only"), ("pred_conc", C3, "predicted: concomitant only"))):
        a.bar(x + (i - 1.5) * wdt, [q[key] for q in rr], width=wdt * 0.9, color=c, label=lab)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(x, [f"enc {q['enc']}" for q in rr])
    a.set(title=f"{nm}-centre (τ = {tau[0 if nm == 'out' else 1] * 1e3:.2f} ms)", ylabel="coil-combined phase vs enc 0 (rad)")
ax[0].legend(fontsize=7)
fig.suptitle("(e) k-centre phase difference of each encode against the reference encode, same rotation group", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "e_encode_phase.png")
plt.close(fig)

json.dump(num, open(KC / "numbers.json", "w"), indent=1)
json.dump(num, open(Path("research/experiments/kcentre") / "numbers.json", "w"), indent=1)
print(json.dumps(num, indent=1))
