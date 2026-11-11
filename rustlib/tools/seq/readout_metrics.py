"""Readout-design metrics from `seqdesign` runs (research/readout_redesign.md), computed CPU-side from the designed
waveforms: timing, sampling density, time of sampling per k radius, data-energy-weighted time statistics, phase
cycles, segment counts, T2* weighting, readout first moment, PNS / limits and moment balance.

    readout_metrics.py <energy_profile.npz> <out dir> LABEL=RUN [LABEL=RUN ...]

Energy model. The per-sample data energy is E(|k|, τ) = P(|k|) · D(|k|, τ):
  P  = the measured energy per ADC sample vs |k| of the outgoing leg of the simulated 10-min scan (factory_arms/ph0001,
       current out-and-back design, noise-free data, 32 coils; tools/seq/energy_profile.py), or ∝ 1 / (|k|² + k0²)
       (the "typical" |k|^-2 spectrum, k0 = 1 / 0.3 m), scaled to the same signal / noise energy on the current design;
  D  = the extra loss of low-k energy when the centre is revisited late (B0 / T2* dephasing of the bulk signal):
       exp(−(τ − τ_out(|k|)) / T_d) for |k| < 0.1 kmax, T_d fitted to the measured return / out energy ratio of the
       current design; 1 elsewhere.
Noise: σ² per sample and coil = noise_sigma² of the scan (meta.json).

Segment counts:
  L_rule  = ceil(1.3 · span · T + 3.5): recon_p2np's `--nseg auto` (T = span of the reconstruction samples);
  L_svd   = the smallest rank whose weighted model error, averaged over field offsets uniform over the span, is below
            η² × the noise energy of the readout's reconstruction samples: Σ_n E_n |e^{i2πfτ_n} − rank-L approx|² ≤
            η² Σ_n σ² (η = 0.1: model error energy 1 % of the noise energy), i.e. `--seg-basis svdw` with the noise as
            the yardstick. Water: f uniform over the span; water + fat: the same span and its copy shifted by the main
            fat peak (−434 Hz at 3 T).
"""
import json
import math
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

GAMMA = 2.6752218744e8
GBAR = GAMMA / (2 * np.pi)
DWELL = 2e-6
SPANS = {"1-99 %": 488.0, "worst voxel": 1236.0}
FAT = -434.0
T2S = {"lung": 1.5e-3, "myocardium": 25e-3, "blood": 40e-3}
ETA = 0.1
ETAS = (0.1, 0.03)


def load_run(run):
    s = json.load(open(os.path.join(run, "summary.json")))
    f = json.load(open(os.path.join(run, "final.json")))
    b = s["best"]
    dt, t_rf = f["dt"], f["t_rf"]
    up = b["upsample"]
    k = np.load(os.path.join(run, "k_adc.npy"))  # spoke, upsampled raster, cycles/m (base frame)
    tk = np.arange(len(k)) * dt / up
    t_adc = b["i_adc"] * dt
    na = int(math.floor(t_adc / DWELL + 1e-9))
    tn = np.arange(na) * DWELL
    kn = np.stack([np.interp(tn, tk, k[:, a]) for a in range(3)], 1)
    kr = np.linalg.norm(kn, axis=1)
    te_ro = b["i_ro"] * dt - t_rf  # readout start after the RF centre
    tau = te_ro + tn
    rec = b.get("recon", [[0, b["i_adc"]]])
    mask = np.zeros(na, bool)
    for a, c in rec:
        mask |= (tn >= a * dt - 1e-12) & (tn < c * dt - 1e-12)
    # first moment (RF-centre origin) along the TR for encode 0 / rotation 0, at the samples; readout part = minus the echo
    tm = np.load(os.path.join(run, "tr_moments.npy"))[0]  # (n_tr+1, 6): k, M1
    ttr = np.arange(len(tm)) * dt
    m1 = np.stack([np.interp(b["i_ro"] * dt + tn, ttr, tm[:, 3 + a]) for a in range(3)], 1)
    i_echo = b.get("i_echo", 0)
    m1e = np.array([np.interp(b["i_ro"] * dt + i_echo * dt, ttr, tm[:, 3 + a]) for a in range(3)])
    dm1 = GAMMA * np.linalg.norm(m1 - m1e, axis=1)
    # TR totals per encode (rotation 0): M0 as k and M1
    tmall = np.load(os.path.join(run, "tr_moments.npy"))
    k_end = tmall[:, b["n_tr"], :3]
    m1_end = tmall[:, b["n_tr"], 3:]
    den = np.load(os.path.join(run, "density.npy"))  # r, l1, need
    return dict(s=s, b=b, f=f, dt=dt, tau=tau, kr=kr, mask=mask, dm1=dm1, te=te_ro + i_echo * dt, k_end=k_end, m1_end=m1_end, den=den,
                kmax=f["kmax"], na=na)


def energy_model(prof):
    z = np.load(prof)
    kr, e, split = z["kr"], z["e"], int(z["split"])
    kmax = kr.max()
    rc, p_out, p_ret = z["rc"] / kmax, z["p_out"], z["p_ret"]
    good = np.isfinite(p_out) & (p_out > 0)
    lr, lp = rc[good], np.log(p_out[good])
    tau = z["tau"]
    # τ at which the reference out leg reaches each |k| (monotone on the out leg)
    ko = kr[:split] / kmax
    to = tau[:split]
    ko_m = np.maximum.accumulate(ko)

    def tau_out(x):
        return np.interp(x, ko_m, to)

    # decay of the low-k energy between the out and the return visit of the reference design
    low = (kr / kmax) < 0.1
    e_out = e[:split][low[:split]].sum()
    e_ret = e[split:][low[split:]].sum()
    dtau = (tau[split:][low[split:]] * e[split:][low[split:]]).sum() / e_ret - (tau[:split][low[:split]] * e[:split][low[:split]]).sum() / e_out
    t_d = dtau / math.log(e_out / e_ret)

    def p_meas(x):
        return np.exp(np.interp(x, lr, lp, left=lp[0], right=lp[-1]))

    def decay(x, t):
        d = np.exp(-np.maximum(0.0, t - tau_out(x)) / t_d)
        return np.where(x < 0.1, d, 1.0)

    return dict(p_meas=p_meas, decay=decay, t_d=t_d, e_ref=e, tau_ref=tau, kr_ref=kr / kmax, split=split, e_ret_frac=e[split:].sum() / e.sum())


def shortest_window(t, w, frac):
    o = np.argsort(t)
    t, w = t[o], w[o]
    c = np.concatenate([[0.0], np.cumsum(w)])
    tot = c[-1]
    best = (t[-1] - t[0], t[0], t[-1])
    j = 0
    for i in range(len(t)):
        while j < len(t) and c[j + 1] - c[i] < frac * tot:
            j += 1
        if j >= len(t):
            break
        if t[j] - t[i] < best[0]:
            best = (t[j] - t[i], t[i], t[j])
    return best


def sv2(tau, w, fgrid):
    # squared singular values of G[f, n] = √w_n e^{i2π f τ_n}: eigenvalues of the Gram matrix G Gᴴ (n_f × n_f)
    tt = tau - tau.mean()
    g = np.exp(2j * np.pi * fgrid[:, None] * tt[None, :])
    gram = (g * w[None, :]) @ g.conj().T
    return np.clip(np.sort(np.linalg.eigvalsh(gram))[::-1], 0.0, None)


def l_svd(tau, e, sigma2, fgrid, eta=ETA, lmax=60):
    # weighted rank: error energy per f (mean over the grid) vs η² × the noise energy of the samples
    s2 = sv2(tau, e, fgrid)
    tail = np.concatenate([np.cumsum(s2[::-1])[::-1], [0.0]]) / len(fgrid)
    target = eta ** 2 * sigma2 * len(tau)
    for L in range(lmax):
        if tail[L] <= target:
            return L
    return lmax


def fgrid_for(span, fat, n=240):
    w = np.linspace(-span / 2, span / 2, n)
    return np.concatenate([w, w + FAT]) if fat else w


def metrics(r, em, sigma2, scale2):
    b, tau, kr, mask = r["b"], r["tau"], r["kr"] / r["kmax"], r["mask"]
    tr = b["tr"]
    out = {}
    out["TR ms"] = tr * 1e3
    out["readouts / 10 min"] = int(600.0 / tr)
    t_rec = b.get("t_recon", b["t_adc"])
    t_rs = b.get("t_recon_samples", b["t_adc"])
    out["recon window ms"] = t_rec * 1e3
    out["ADC ms (all)"] = b["i_adc"] * r["dt"] * 1e3
    out["recon duty"] = t_rs / tr
    out["ADC duty"] = b["i_adc"] * r["dt"] / tr
    out["TE (echo) ms"] = r["te"] * 1e3
    out["N (Nyquist spokes)"] = b["n_eff"]
    out["N·TR·7 s"] = b["n_eff"] * tr * 7
    out["Nyquist sets / 10 min"] = 600.0 / (b["n_eff"] * tr * 7)
    den = r["den"]
    rr = den[:, 0] / r["kmax"]
    need = den[:, 2]
    for x in (0.25, 0.5, 0.75):
        out[f"density × Nyquist @{x:.2f}kmax"] = b["n_eff"] / np.interp(x, rr, need)
    out["coverage empty outer (full set)"] = b["coverage"]["super"]["empty_outer"]
    # τ per k radius (reconstruction samples)
    t_r, k_r = tau[mask], kr[mask]
    for lo, hi in ((0.0, 0.05), (0.45, 0.55), (0.95, 1.01)):
        sel = (k_r >= lo) & (k_r < hi)
        out[f"τ @|k|∈[{lo},{hi}) ms"] = (t_r[sel].min() * 1e3, t_r[sel].max() * 1e3) if sel.any() else (np.nan, np.nan)
    res = {}
    for name, p in (("meas", em["p_meas"]), ("k^-2", lambda x: scale2 / (x ** 2 + (1 / (0.3 * 250)) ** 2))):
        e = p(k_r) * em["decay"](k_r, t_r)
        tbar = (e * t_r).sum() / e.sum()
        sd = math.sqrt((e * (t_r - tbar) ** 2).sum() / e.sum())
        w999 = shortest_window(t_r, e, 0.999)
        q = dict(tbar=tbar, sd=sd, w999=w999[0], snr_energy=e.sum() / (len(e) * sigma2))
        # energy after the first 1 ms of the window (late energy)
        q["late frac"] = e[t_r > t_r.min() + 1e-3].sum() / e.sum()
        q["above noise"] = float((e > sigma2).mean())
        for t2n, t2 in T2S.items():
            q[f"T2* err {t2n}"] = math.sqrt((e * (1 - np.exp(-(t_r - r["te"]) / t2)) ** 2).sum() / e.sum())
        ls = {}
        for sn, sp in SPANS.items():
            for fat in (False, True):
                for eta in ETAS:
                    ls[(sn, fat, eta)] = l_svd(t_r, e, sigma2, fgrid_for(sp, fat), eta)
        q["L_svd"] = ls
        res[name] = q
    out["energy"] = res
    # uniform-weight (energy-agnostic) rank at a fixed relative error 1e-3 — the window effect alone
    lu = {}
    for sn, sp in SPANS.items():
        for fat in (False, True):
            s2 = sv2(t_r, np.ones_like(t_r), fgrid_for(sp, fat))
            rel = np.sqrt(np.concatenate([np.cumsum(s2[::-1])[::-1], [0.0]]) / s2.sum())
            lu[(sn, fat)] = int(np.argmax(rel <= 1e-3))
    out["L_unif(1e-3)"] = lu
    lr = {}
    for sn, sp in SPANS.items():
        for fat in (False, True):
            lr[(sn, fat)] = max(2, math.ceil(1.3 * (sp + (abs(FAT) if fat else 0.0)) * t_rec + 3.5))
    out["L_rule"] = lr
    for d in (100.0, 300.0, 500.0, abs(FAT)):
        out[f"cycles {d:.0f} Hz (window / E99.9)"] = (d * t_rec, d * res["meas"]["w999"])
    # T2* weighting at the k-centre (echo) and at kmax (mean over the outer shell samples)
    outer = k_r > 0.95
    for t2n, t2 in T2S.items():
        out[f"T2* {t2n} centre / kmax"] = (math.exp(-r["te"] / t2), float(np.exp(-t_r[outer] / t2).mean()) if outer.any() else np.nan)
    # readout first moment: at the first reach of kmax, and energy-weighted RMS
    dm1 = r["dm1"][mask]
    i_k = np.argmax(k_r >= 0.99) if (k_r >= 0.99).any() else len(k_r) - 1
    e = em["p_meas"](k_r) * em["decay"](k_r, t_r)
    out["γ|M1_ro| @kmax rad/(m/s)"] = dm1[i_k]
    out["γ|M1_ro| E-RMS rad/(m/s)"] = math.sqrt((e * dm1 ** 2).sum() / e.sum())
    out["γ|M1_ro| end rad/(m/s)"] = b.get("readout_gamma_m1_end", np.nan)
    # PNS / limits / moments
    out["PNS %"] = b["pns"]
    out["PNS part"] = b["pns_part"]
    out["slew factors"] = b["slew"]
    out["|G| max mT/m"] = b["g_vec"] * 1e3
    out["|S| max T/m/s"] = b["s_vec"]
    out["tail ms"] = b["n_tail"] * r["dt"] * 1e3
    ke, m1e = r["k_end"], r["m1_end"]
    out["TR k_net /m (all enc)"] = (float(np.linalg.norm(ke, axis=1).min()), float(np.linalg.norm(ke, axis=1).max()))
    out["TR M1 spread across enc (γ, rad/(m/s))"] = float(GAMMA * np.linalg.norm(m1e - m1e[0], axis=1).max())
    out["max |k|/kmax readout"] = b["kmax_check"]["readout_max"]
    return out


def fmt(v):
    if isinstance(v, tuple):
        return " / ".join(fmt(x) for x in v)
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        a = abs(v)
        return f"{v:.3g}" if a < 10 else (f"{v:.1f}" if a < 1000 else f"{v:.0f}")
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
    return str(v)


def main():
    prof, out_dir = sys.argv[1], sys.argv[2]
    runs = [a.split("=", 1) for a in sys.argv[3:]]
    os.makedirs(out_dir, exist_ok=True)
    em = energy_model(prof)
    sigma2 = 39.97583861630753 ** 2
    rs = {lab: load_run(run) for lab, run in runs}
    # |k|^-2 profile scaled to the measured signal / noise energy on the reference (first) run
    ref = rs[runs[0][0]]
    kref = ref["kr"][ref["mask"]] / ref["kmax"]
    tref = ref["tau"][ref["mask"]]
    e_meas = (em["p_meas"](kref) * em["decay"](kref, tref)).sum()
    e_k2 = (1 / (kref ** 2 + (1 / (0.3 * 250)) ** 2) * em["decay"](kref, tref)).sum()
    scale2 = e_meas / e_k2
    print(f"energy model: T_d (low-k revisit) {em['t_d']*1e3:.2f} ms; measured return-leg energy fraction {em['e_ret_frac']:.3f}")
    # model check on the reference: predicted return-half fraction
    pr = em["p_meas"](kref) * em["decay"](kref, tref)
    half = len(pr) // 2
    print(f"  model on the reference run: return-half energy fraction {pr[half:].sum()/pr.sum():.3f} (measured {em['e_ret_frac']:.3f})")
    allm = {lab: metrics(r, em, sigma2, scale2) for lab, r in rs.items()}
    labs = [l for l, _ in runs]
    rows = []
    keys = [k for k in allm[labs[0]] if k not in ("energy", "L_svd", "L_rule", "L_unif(1e-3)")]
    lines = ["| metric | " + " | ".join(labs) + " |", "|---|" + "---|" * len(labs)]
    for k in keys:
        lab = k.replace("|", "\\|")  # labels such as τ @|k| … must not split the markdown row
        lines.append(f"| {lab} | " + " | ".join(fmt(allm[l][k]) for l in labs) + " |")
    for prof_name in ("meas", "k^-2"):
        for k, lab in (("tbar", "E-weighted mean τ ms"), ("sd", "E-weighted SD of τ ms"), ("w999", "99.9 %-energy window ms"), ("late frac", "energy fraction > 1 ms after window start"), ("above noise", "fraction of samples with E > σ²")):
            sc = 1e3 if k not in ("late frac", "above noise") else 1.0
            lines.append(f"| {lab} ({prof_name}) | " + " | ".join(fmt(allm[l]["energy"][prof_name][k] * sc) for l in labs) + " |")
        for t2n in T2S:
            lines.append(f"| unmodelled-T2* data error, {t2n} ({prof_name}) | " + " | ".join(fmt(allm[l]["energy"][prof_name][f"T2* err {t2n}"]) for l in labs) + " |")
    for sn in SPANS:
        for fat in (False, True):
            tag = f"{sn}{' + fat' if fat else ''}"
            lines.append(f"| L rule ({tag}) | " + " | ".join(str(allm[l]["L_rule"][(sn, fat)]) for l in labs) + " |")
            lines.append(f"| L uniform SVD 1e-3 ({tag}) | " + " | ".join(str(allm[l]["L_unif(1e-3)"][(sn, fat)]) for l in labs) + " |")
            for prof_name in ("meas", "k^-2"):
                for eta in ETAS:
                    lines.append(f"| L svdw η={eta} ({tag}, {prof_name}) | " + " | ".join(str(allm[l]["energy"][prof_name]["L_svd"][(sn, fat, eta)]) for l in labs) + " |")
    table = "\n".join(lines)
    print(table)
    open(os.path.join(out_dir, "metrics_table.md"), "w").write(table + "\n")

    def js(v):
        if isinstance(v, dict):
            return {(" ".join(map(str, k)) if isinstance(k, tuple) else k): js(x) for k, x in v.items()}
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, tuple):
            return [js(x) for x in v]
        return v

    json.dump({"runs": dict(runs), "t_d": em["t_d"], "metrics": js(allm)}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    figures(rs, em, labs, out_dir)


def figures(rs, em, labs, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fd = os.path.join(out_dir, "figs")
    os.makedirs(fd, exist_ok=True)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # 1: |k|/kmax vs τ (reconstruction samples solid, navigator dotted)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, l in enumerate(labs):
        r = rs[l]
        kr = r["kr"] / r["kmax"]
        t = r["tau"] * 1e3
        m = r["mask"]
        ax[0].plot(np.where(m, t, np.nan), np.where(m, kr, np.nan), color=cols[i % 10], lw=1.2, label=l)
        ax[0].plot(np.where(~m, t, np.nan), np.where(~m, kr, np.nan), color=cols[i % 10], lw=0.8, ls=":")
        e = em["p_meas"](kr[m]) * em["decay"](kr[m], r["tau"][m])
        o = np.argsort(t[m])
        ax[1].plot(t[m][o], np.cumsum(e[o]) / e.sum(), color=cols[i % 10], lw=1.2, label=l)
        den = r["den"]
        ax[2].plot(den[:, 0] / r["kmax"], r["b"]["n_eff"] / den[:, 2], color=cols[i % 10], lw=1.2, label=l)
    ax[0].set(xlabel="τ after RF centre (ms)", ylabel="|k| / kmax", title="time of sampling per k radius (dotted: navigator ADC)")
    ax[1].set(xlabel="τ after RF centre (ms)", ylabel="cumulative data energy", title="measured-spectrum energy vs τ", ylim=(0.8, 1.002))
    ax[2].set(xlabel="|k| / kmax", ylabel="density / Nyquist (N spokes)", title="sampling density vs radius", yscale="log")
    for a in ax:
        a.grid(alpha=0.3)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "readout_tau_energy_density.png"), dpi=130)
    plt.close(fig)
    # 2: per-sample energy vs τ (log), all designs
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, l in enumerate(labs):
        r = rs[l]
        kr = r["kr"] / r["kmax"]
        m = r["mask"]
        e = em["p_meas"](kr) * em["decay"](kr, r["tau"])
        ax.semilogy(r["tau"][m] * 1e3, e[m] / 39.97583861630753 ** 2, color=cols[i % 10], lw=0.8, label=l)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set(xlabel="τ after RF centre (ms)", ylabel="E / σ² per sample and coil", title="data energy per sample (measured spectrum model); dashed: noise")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fd, "energy_per_sample.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
