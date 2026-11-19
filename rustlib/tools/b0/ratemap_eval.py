"""Accuracy of a ratemap run (recon_b0 / ratemap_fit output) against the phantom truth — evaluation only.

    python -I ratemap_eval.py <run> [--no-warp] [--tag name]

Signal-carrying voxels: truth density (water + fat + blood: 0.85 × lumen fraction, table units) > 5 % of its 99th percentile; statistics weighted
by it. Regions (from truth, for the table only): heart box (±45 mm around the centroid of the largest
blood-pool component), liver (largest component of liver-like tissue below the heart), chest wall (non-lung tissue
anterior of the heart centre within 25 mm of lung), lung (truth decay rate > 500 s⁻¹), all tissue.
Per respiratory bin the truth (given at the scan's mean breathing state) is warped to the bin's mean true state with
the phantom's respiratory motion field (pure advection, as the simulator) unless --no-warp.
Field error Δ = ω − ω_true: weighted median, p95 / p99 of |Δ|, swap fraction (|Δ| > 150 Hz) and its swap-like part
(||Δ| − 434| < 100 Hz); phase over the 8 ms readout = 2π · 0.008 · |Δ| (1 Hz → 0.050 rad). Coverage = weight with a
finite estimate. Writes <run>/ratemap_eval.json and prints a markdown table.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))

GBAR = 2.6752218744e8 / (2 * np.pi)
T_ADC = 0.008


def down(x, g):
    """centred triangle (1 − |o|/g) at fine index g·j, normalised (as recon_b0 `down`)"""
    if g == 1:
        return x
    w = 1.0 - np.abs(np.arange(-(g - 1), g)) / g
    for a in range(3):
        num = ndimage.convolve1d(x, w, axis=a, mode="constant")
        den = ndimage.convolve1d(np.ones_like(x), w, axis=a, mode="constant")
        x = np.take(num / den, np.arange(x.shape[a] // g) * g, axis=a)
    return x


def wq(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][min(np.searchsorted(c, q * c[-1]), len(v) - 1)])


def stats(d_om, d_r2, d_ff, w):
    ok = np.isfinite(d_om) & (w > 0)
    tot = w.sum()
    if ok.sum() < 10:
        return {"n": int(ok.sum())}
    a, ww = d_om[ok], w[ok]
    ab = np.abs(a)
    out = {
        "n": int(ok.sum()),
        "coverage": float(ww.sum() / tot),
        "om_median_hz": wq(a, ww, 0.5),
        "om_abs_p50_hz": wq(ab, ww, 0.5),
        "om_abs_p95_hz": wq(ab, ww, 0.95),
        "om_abs_p99_hz": wq(ab, ww, 0.99),
        "swap_frac": float(ww[ab > 150].sum() / ww.sum()),
        "swaplike_frac": float(ww[np.abs(ab - 434) < 100].sum() / ww.sum()),
    }
    out["phase_p95_rad"] = 2 * np.pi * T_ADC * out["om_abs_p95_hz"]
    r = d_r2[ok]
    rf = np.isfinite(r)
    out["r2_median"] = wq(r[rf], ww[rf], 0.5) if rf.sum() > 10 else None
    out["r2_abs_p50"] = wq(np.abs(r[rf]), ww[rf], 0.5) if rf.sum() > 10 else None
    out["r2_abs_p95"] = wq(np.abs(r[rf]), ww[rf], 0.95) if rf.sum() > 10 else None
    f = d_ff[ok]
    ff_ = np.isfinite(f)
    out["ff_median"] = wq(f[ff_], ww[ff_], 0.5) if ff_.sum() > 10 else None
    out["ff_abs_p95"] = wq(np.abs(f[ff_]), ww[ff_], 0.95) if ff_.sum() > 10 else None
    return out


def main():
    args = sys.argv[1:]
    warp = "--no-warp" not in args
    args = [a for a in args if a != "--no-warp"]
    tag = None
    if "--tag" in args:
        i = args.index("--tag")
        tag = args[i + 1]
        del args[i : i + 2]
    run = Path(args[0])
    s = json.load(open(run / "summary.json"))
    sim = Path(s["sim_dir"])
    if not sim.exists():
        sim = REPO / "data/hasty" / Path(*sim.parts[sim.parts.index("phantom") :])
    meta = json.load(open(sim / "meta.json"))
    dims = tuple(s["dims"])
    res = float(s["res"])
    res_f = float(meta["res"][0])
    g = int(round(res / res_f))
    b0 = float(meta["b0"])
    ld = lambda n: np.nan_to_num(np.load(sim / n).astype(np.float64))  # noqa: E731
    pw, pf = ld("truth_pd_water.npy"), ld("truth_pd_fat.npy")
    pd = pw + pf
    tr = {
        "om": ld("field_ppm_recon.npy") * GBAR * b0 * 1e-6,
        "r2": ld("truth_r2s.npy"),
        "pf": pf,
        "pd": pd,
        "lum": ld("truth_lumen_fraction.npy"),
    }
    # signal weight: tissue density + blood pool (the truth density maps exclude the blood, simulated as particles;
    # blood density ≈ 0.85 × lumen fraction in table units); density-weighted triangle downsampling
    # density units: the median water-tissue truth density ↔ the table's ≈ 0.70
    scale = np.median(pw[(pw > 0) & (pf == 0) & (tr["r2"] > 0) & (tr["r2"] < 60)]) / 0.70
    wgt = pd + 0.85 * scale * tr["lum"]
    tg = {}
    wg = down(wgt, g)
    pdg = down(pd, g)
    tg["om"] = down(tr["om"] * wgt, g) / np.maximum(wg, 1e-12)
    tg["r2"] = np.where(pdg > 0.5 * wg, down(tr["r2"] * pd, g) / np.maximum(pdg, 1e-12), np.nan)
    tg["ff"] = down(pf, g) / np.maximum(wg, 1e-12)
    tg["pd"] = wg
    tg["lum"] = down(tr["lum"], g)
    om_raw = down(tr["om"], g)
    assert tg["pd"].shape == dims, (tg["pd"].shape, dims)
    pdg = wg
    p99 = np.percentile(pdg[pdg > 0], 99)
    sig = pdg > 0.05 * p99
    lung = sig & (np.nan_to_num(tg["r2"]) > 500)
    # regions
    lab, n = ndimage.label(tg["lum"] > 0.5)
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    hc = np.argwhere(lab == big).mean(0)
    half = 45e-3 / res
    ii = np.indices(dims)
    hbox = np.all([np.abs(ii[a] - hc[a]) <= half for a in range(3)], 0) & sig
    # liver: largest component of fat-free tissue with the liver's decay rate (36–55 s⁻¹) ≥ 20 mm below the heart centre
    r2n = np.nan_to_num(tg["r2"])
    liv = sig & (tg["ff"] < 0.02) & (r2n > 36) & (r2n < 55) & (tg["lum"] < 0.5) & (ii[2] < hc[2] - 20e-3 / res)
    liv = ndimage.binary_opening(liv, iterations=max(1, 2 // g))
    lab2, _ = ndimage.label(liv)
    liver = lab2 == (np.argmax(np.bincount(lab2.ravel())[1:]) + 1) if lab2.max() > 0 else liv
    dl = ndimage.distance_transform_edt(~lung) * res
    chest = sig & ~lung & (dl < 25e-3) & ~hbox & ~liver & (ii[1] < hc[1])
    regions = {"all tissue": sig & ~lung, "heart box": hbox & ~lung, "liver": liver, "chest wall": chest, "lung": lung}
    print("regions (voxels):", {k: int(v.sum()) for k, v in regions.items()}, "heart centre (vox)", np.round(hc, 1))

    om = np.load(run / "maps_omega_hz.npy")
    r2 = np.load(run / "maps_r2.npy")
    ff = np.load(run / "maps_ff.npy")
    nb = om.shape[3]
    r_ref = (meta.get("truth") or {}).get("resp_state", 0.5)
    rb = s.get("resp", {}).get("bin_mean_true_r") or [r_ref] * nb
    if warp and nb > 1:
        import importlib  # noqa: F401
        from hastycompute import value_format as vf

        rm = vf.read(Path(str(sim.parent.parent / "motion" / "resp_motion.hv")))
        Bf = np.asarray(rm["B"]).sum(0)  # [3, Z, Y, X] subject displacement per unit r
        ro, rh = np.asarray(rm["origin_m"], float), float(rm["voxel_m"])
        assert np.allclose(np.asarray(meta["rot_presc"]), np.eye(3)), "warp assumes an axis-aligned prescription"
        c = np.asarray(meta["presc_center_subject"], float)
        P = np.stack([(ii[a] - dims[a] // 2) * res + c[a] for a in range(3)], -1)

        def S(Q):
            idx = [(Q[..., a] - ro[a]) / rh - 0.5 for a in (2, 1, 0)]
            return np.stack([ndimage.map_coordinates(Bf[k], idx, order=1, mode="nearest") for k in range(3)], -1)

    rows = []
    for b in range(nb):
        T = dict(tg)
        rr = None
        if warp and nb > 1:
            rr = float(rb[b])
            X = P.copy()
            for _ in range(6):
                X = P - rr * S(X)
            Q = X + r_ref * S(X)
            idx = [(Q[..., a] - c[a]) / res + dims[a] // 2 for a in range(3)]
            T = {k: ndimage.map_coordinates(tg[k], idx, order=1, mode="nearest") for k in ("om", "r2", "ff")}
            T["pd"] = tg["pd"]
        for name, m in regions.items():
            w = np.where(m, tg["pd"], 0.0)
            st = stats(om[..., b] - T["om"], r2[..., b] - T["r2"], ff[..., b] - T["ff"], w)
            st.update({"bin": b, "region": name, "r_true": rr})
            rows.append(st)
    # the original recon_b0 metric for continuity: fitted magnitude > 10 % of its p99 (includes non-tissue voxels)
    wm, fm = np.load(run / "maps_water.npy")[..., 0], np.load(run / "maps_fat.npy")[..., 0]
    mag = wm + fm
    v = np.isfinite(om[..., 0])
    sel = v & (mag > 0.1 * np.percentile(mag[v], 99))
    d = (om[..., 0] - om_raw)[sel]
    legacy = {"n": int(sel.sum()), "rmse": float(np.sqrt((d**2).mean())), "median": float(np.median(d)), "gt150": int((np.abs(d) > 150).sum()), "outside_tissue_frac": float((tg["pd"][sel] <= 0).mean()), "outside_signal_frac": float((~sig[sel]).mean())}
    out = {"run": str(run), "rows": rows, "legacy_metric": legacy, "regions": {k: int(v.sum()) for k, v in regions.items()}, "warp": warp}
    fn = run / (f"ratemap_eval_{tag}.json" if tag else "ratemap_eval.json")
    json.dump(out, open(fn, "w"), indent=1)
    print(f"legacy metric (fit-magnitude mask): n {legacy['n']}, RMSE {legacy['rmse']:.1f} Hz, median {legacy['median']:.1f}, >150 Hz {legacy['gt150']}, outside tissue {100 * legacy['outside_tissue_frac']:.0f} %")
    print("| bin | region | cover | Δω med | |Δω| p50 | p95 | p99 | >150 Hz | swap-like | phase p95 (rad, 8 ms) | ΔR med | |ΔR| p95 | Δff med | |Δff| p95 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    f = lambda x, p=1: "–" if x is None else f"{x:.{p}f}"  # noqa: E731
    for r in rows:
        if "coverage" not in r:
            print(f"| {r['bin']} | {r['region']} | – |")
            continue
        print(f"| {r['bin']} | {r['region']} | {f(r['coverage'], 2)} | {f(r['om_median_hz'])} | {f(r['om_abs_p50_hz'])} | {f(r['om_abs_p95_hz'])} | {f(r['om_abs_p99_hz'])} | {f(100 * r['swap_frac'])} % | {f(100 * r['swaplike_frac'])} % | {f(r['phase_p95_rad'], 2)} | {f(r['r2_median'])} | {f(r['r2_abs_p95'])} | {f(r['ff_median'], 3)} | {f(r['ff_abs_p95'], 3)} |")
    print("wrote", fn)


if __name__ == "__main__":
    main()
