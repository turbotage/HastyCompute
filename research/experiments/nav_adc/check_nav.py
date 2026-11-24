"""Checks of the navigator-ADC smoke simulations (nav_on / nav_off / nav_girf) against the flow_thorax design."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/turbotage/Documents/GitHub/HastyCompute")
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

O = REPO / "data/hasty/phantom/factory_arms/ph0001/nav_smoke"
D = REPO / "data/hasty/seq/flow_thorax"
on, off, gi = O / "nav_on", O / "nav_off", O / "nav_girf"

m = json.load(open(on / "meta.json"))
nav = m["nav"]
print("meta nav blocks:", [(b["name"], b["n"], round(b["t_first"] * 1e3, 3), round(b["t_last"] * 1e3, 3), b["events"]) for b in nav["blocks"]])
print("rules:", nav["rules"])
n_ro, na, nc = m["n_readouts"], m["n_adc"], m["n_coils"]
ro = np.load(on / "readouts.npy")
enc = ro[:, 1].astype(int)

# 1. files and shapes
for b in nav["blocks"]:
    for f in (b["data"], b["samples"]):
        assert (on / f).exists(), f
    d = load_any(on / b["data"])
    s = load_any(on / b["samples"])
    print(f"{b['name']}: data {d.shape} {d.dtype}, samples {s.shape}")
    assert d.shape == (n_ro, b["n"], nc) and s.shape == (n_ro, b["n"], 4)
    # 2. times: τ = window times after the RF centre, dwell 2 µs, every readout
    t = s[:, :, 3]
    tw = b["t_first"] + 2e-6 * np.arange(b["n"])
    print(f"  τ: {t[0, 0] * 1e3:.4f}–{t[0, -1] * 1e3:.4f} ms, max |τ − window| {np.abs(t - tw[None]).max():.2e} s")
    assert np.abs(t - tw[None]).max() < 1e-8
    # 3. k vs the design: k_nav_<block>.npy (hasty_seq::nav, γ̄ M0) and, independently, k_tr.npy at raster points
    kn = np.load(D / f"k_nav_{b['name']}.npy")  # [n_enc][n][3] cycles/m
    ks = s[:, :, :3] / (2 * np.pi)
    err = np.abs(ks - kn[enc]).max()
    kmax = np.abs(kn).max()
    print(f"  k vs k_nav_{b['name']}.npy: max |Δk| {err:.3e} cycles/m (|k| up to {kmax:.0f}; f32 storage)")
    assert err < 2e-4 * kmax
    ktr = np.load(D / "k_tr.npy")  # [n_enc][n_tr + 1][3] cycles/m at the 10 µs raster from the TR start
    t_tr = s[0, :, 3] + 150e-6  # from the TR start (RF centre at 150 µs)
    ir = np.round(t_tr / 10e-6).astype(int)
    on_r = np.abs(t_tr - ir * 10e-6) < 1e-9
    e2 = np.abs(ks[:, on_r] - ktr[enc][:, ir[on_r]]).max()
    print(f"  k vs k_tr.npy at the {on_r.sum()} samples on raster points: max |Δk| {e2:.3e} cycles/m")
    assert e2 < 2e-4 * kmax
    # rotation independence (encode block physical; FlowComp tail): readouts of one encode share their k path
    for e in range(7):
        r = np.where(enc == e)[0]
        if len(r) > 1:
            assert np.abs(ks[r] - ks[r[0]]).max() < 2e-4 * kmax

# 4. readout data unchanged by the navigator (nav off run)
a = load_any(on / "data_clean.hv")
b_ = load_any(off / "data_clean.hv")
a, b_ = np.asarray(a), np.asarray(b_)
print("data_clean on/off shapes", a.shape, b_.shape)
same = np.array_equal(a, b_)
rel = np.abs(a - b_).max() / np.abs(a).max()
print(f"readout data nav on vs off: bitwise equal {same}, max |Δ| / max |y| {rel:.2e}")
mon, moff = json.load(open(on / "meta.json")), json.load(open(off / "meta.json"))
print("noise sigma on/off", mon["noise_sigma"], moff["noise_sigma"], "kspace_step", mon["storage"].get("kspace_step"), moff["storage"].get("kspace_step"))
assert "nav" not in moff and not list(off.glob("nav_*"))
for f in ("samples.hv", "sched_nominal.hv", "readouts.npy"):
    print(f, "identical" if (on / f).read_bytes() == (off / f).read_bytes() else "DIFFERENT")
assert rel < 1e-5

# 5. physical sanity: signal level of the navigator vs the readout
de = np.asarray(load_any(on / "nav_encode.hv"))
dt_ = np.asarray(load_any(on / "nav_tail.hv"))
rss = lambda x: np.sqrt((np.abs(x) ** 2).sum(-1))  # noqa: E731
print(f"rss over coils, mean over readouts: readout sample 0 (TE, k = 0) {rss(a[:, 0]).mean():.3e}; encode sample 0 "
      f"(τ 0.15 ms, k = 0) {rss(de[:, 0]).mean():.3e}; encode mean {rss(de).mean():.3e}; tail first {rss(dt_[:, 0]).mean():.3e}, last {rss(dt_[:, -1]).mean():.3e}")
assert np.isfinite(de).all() and np.isfinite(dt_).all()

# 6. GIRF run: played tables exist, played k differs a little from the nominal
if (gi / "meta.json").exists():
    mg = json.load(open(gi / "meta.json"))
    for b in mg["nav"]["blocks"]:
        sp = load_any(gi / b["samples_played"])
        sn = load_any(gi / b["samples"])
        dk = np.abs(sp[:, :, :3] - sn[:, :, :3]).max() / (2 * np.pi)
        print(f"girf run {b['name']}: played {sp.shape}, max |k_played − k_nominal| {dk:.2f} cycles/m, data {load_any(gi / b['data']).shape}")
print("all checks passed")
