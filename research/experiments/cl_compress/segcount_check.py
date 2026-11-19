"""Sanity check of the time-segment count (lead, 2026-11-18): why do we need L > 10 when Fessler et al. (2005) reach
~1e-4 with L = 8 on a ~100 Hz field and an 18.9 ms readout?

Segments are the SVD of the space-time matrix E[r, τ] = s(r, τ) (one row per voxel, one column per readout time),
weighted by the signal energy w(r): the spatial functions are the left singular vectors and the interpolators a_l(τ) the
right ones. This is cheap: only the T × T Gram G = Eᴴ diag(w) E is needed (one pass over the voxels, T ≈ 200 times),
then an eigendecomposition of G.

Per voxel the time function is the ratemap signal:
  water:  e^{−(R(r) + i ω(r)) (τ − τ0)}
  fat:    Σ_p α_p e^{−i 2π f_p τ} · e^{−(R + i ω)(τ − τ0)}   (six-peak spectrum)
  mixed:  (1 − ff) · water + ff · fat (folded into one basis), or water and fat as two species with their own bases.

Errors: relative Frobenius (signal-weighted) and the worst signal-carrying voxel.
Run: flock <cpu lock> systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 \
     rustlib/.venv/bin/python -I research/experiments/cl_compress/segcount_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir())
sys.path.insert(0, str(ROOT / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

S = ROOT / "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan"
meta = json.load(open(S / "meta.json"))
B0 = float(meta.get("b0", 3.0))
HZ_PER_PPM = 42.577478 * B0
FAT_PPM = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
FAT_A = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
T = 200
TAU = np.linspace(1.60e-3, 9.58e-3, T)
TAU0 = TAU[0]


def arr(name):
    return np.asarray(load_any(S / name)).astype(np.float64)


def rank_report(E, w, label, eps=(1e-2, 1e-3, 1e-4)):
    """E: voxels × T complex (time functions), w: voxel weights (signal energy). Returns L(ε) Frobenius and worst-voxel."""
    sw = np.sqrt(w)[:, None]
    A = E * sw
    G = A.conj().T @ A
    ev, V = np.linalg.eigh(G)
    ev, V = ev[::-1].clip(0), V[:, ::-1]
    tot = ev.sum()
    tail = np.concatenate([np.cumsum(ev[::-1])[::-1][1:], [0.0]]) / tot  # tail[L-1] = energy beyond L
    out = {"label": label}
    for e in eps:
        Lf = int(np.argmax(np.sqrt(tail) <= e)) + 1
        out[f"L_frob_{e:g}"] = Lf
    # worst voxel among the signal-carrying ones (w ≥ 1 % of max): relative residual of its row
    sig = w >= 0.01 * w.max()
    rn = np.einsum("ij,ij->i", E[sig], E[sig].conj()).real
    P = E[sig] @ V
    cum = np.cumsum(np.abs(P) ** 2, axis=1)
    for e in eps:
        Lw = None
        for L in range(1, E.shape[1] + 1):
            r = np.sqrt(np.clip(rn - cum[:, L - 1], 0, None) / rn)
            if r.max() <= e:
                Lw = L
                break
        out[f"L_worst_{e:g}"] = Lw
    return out


def main():
    dims = meta["recon_dims"]
    fld = arr("field_ppm_recon.npy").reshape(-1) * HZ_PER_PPM          # Hz (x fastest per hastycompute convention)
    r2 = arr("truth_r2s.npy").reshape(-1)                              # s⁻¹ (decay part of the ratemap)
    pdw = arr("truth_pd_water.npy").reshape(-1)
    pdf = arr("truth_pd_fat.npy").reshape(-1)
    lum = arr("truth_lumen_fraction.npy").reshape(-1)
    n = fld.size
    assert r2.size == n and pdw.size == n, (fld.size, r2.size, pdw.size)
    nx, ny, nz = dims
    idx = np.arange(n)
    x, y, z = idx % nx, (idx // nx) % ny, idx // (nx * ny)
    vox = np.array(meta["fov"]) / np.array(dims)
    ql = lum > 0.5
    c = np.array([x[ql].mean(), y[ql].mean(), z[ql].mean()])
    half = 0.080 / vox
    box = (np.abs(x - c[0]) <= half[0]) & (np.abs(y - c[1]) <= half[1]) & (np.abs(z - c[2]) <= half[2])
    res = {"b0_T": B0, "tau_ms": [TAU[0] * 1e3, TAU[-1] * 1e3], "regions": {}}
    for name, m in (("box160", box), ("object", (pdw + pdf) > 0.02 * (pdw + pdf).max())):
        m = m & ((pdw + pdf) > 0)
        f, R, w_w, w_f = fld[m], r2[m], pdw[m], pdf[m]
        wtot = w_w + w_f
        ff = np.where(wtot > 0, w_f / np.maximum(wtot, 1e-30), 0.0)
        tt = TAU[None, :] - TAU0
        water = np.exp(-(R[:, None] + 2j * np.pi * f[:, None]) * tt)
        fatspec = (FAT_A[None, :] * np.exp(-2j * np.pi * (FAT_PPM * HZ_PER_PPM)[None, :] * TAU[:, None])).sum(1)
        nodecay = np.exp(-2j * np.pi * f[:, None] * tt)
        # signal-energy-weighted field percentiles
        o = np.argsort(f)
        cw = np.cumsum((wtot ** 2)[o]) / (wtot ** 2).sum()
        pct = {p: float(f[o][np.searchsorted(cw, p / 100)]) for p in (1, 5, 50, 95, 99)}
        reg = {"voxels": int(m.sum()), "field_hz_signal_weighted_pct": pct, "cases": []}
        e2 = wtot ** 2
        reg["cases"].append(rank_report(nodecay, e2, "water, off-resonance only"))
        reg["cases"].append(rank_report(water, e2, "water, off-resonance + decay"))
        mixed = (1 - ff)[:, None] * water + ff[:, None] * water * fatspec[None, :]
        reg["cases"].append(rank_report(mixed, e2, "water + fat folded into one basis (+ decay)"))
        sel = w_w > 0
        reg["cases"].append(rank_report(water[sel], w_w[sel] ** 2, "water species alone (+ decay)"))
        self_f = w_f > 0.02 * w_f.max()
        reg["cases"].append(rank_report((water * fatspec[None, :])[self_f], w_f[self_f] ** 2, "fat species alone (+ decay)"))
        res["regions"][name] = reg
        print(json.dumps({name: reg}, indent=1), flush=True)
    # Fessler-like control: uniform field over ±50 Hz, 18.855 ms readout, uniform weights, no decay
    tf = np.linspace(0, 18.855e-3, 400)
    fs = np.linspace(-50, 50, 4001)
    Ef = np.exp(-2j * np.pi * fs[:, None] * tf[None, :])
    res["fessler_like_100Hz_18.9ms"] = rank_report(Ef, np.ones(len(fs)), "uniform ±50 Hz, 18.855 ms")
    print(json.dumps(res["fessler_like_100Hz_18.9ms"]), flush=True)
    out = Path(__file__).resolve().parent / "out/segcount_check.json"
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
