"""Convergence check of the k-space simulation (research/sim_convergence.md): the same readouts simulated with one
discretisation setting changed at a time (simphantom --window …), each compared with a reference run.

    python -I sim_conv.py <out_dir> --ref <name>=<scan_dir> --var <name>=<scan_dir>[@<ref name>] … [--sigma 39.98]
        [--frame-readouts 74] [--k-region name=K …] [--region <var name>=<region>] [--title …]

Per variant, over the reconstruction window samples (meta sequence.recon_adc) of all readouts and coils:
- e_σ = rms(y_var − y_ref) / σ: the change per sample in units of the receiver noise σ;
- e_rel = ‖y_var − y_ref‖ / ‖y_ref‖;
- e_σ per |k| / kmax band;
- an image-domain ratio for a change confined to K voxels, coherent over the R readouts of one reconstructed frame
  (one encode, one cardiac bin: --frame-readouts): with A the frame's forward operator, d = A Δx and A^H A ≈ α I,
  ‖Δx‖² = ‖d‖² / α and the noise variance per voxel is σ² / α, so the rms change over the K voxels in units of the
  per-voxel noise is ‖d_frame‖ / (σ √K), ‖d_frame‖² ≈ R / n_test · ‖d_test‖² (independent of coil scaling).
  K per region: --k-region all=<N recon voxels> blood=<blood voxels> coronary=<…>; --region picks a variant's.
Writes <out_dir>/conv.json, conv.md and figs/ (bars of the image-domain ratio, e_σ per k band, one readout's |y| and
|d| over the samples).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A = sys.argv[1:]
OUT = Path(A[0])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opts(n):
    return [A[i + 1] for i, a in enumerate(A) if a == n]


def opt(n, d):
    v = opts(n)
    return v[-1] if v else d


def load(d):
    d = Path(d)
    f = d / "data_clean.npy" if (d / "data_clean.npy").exists() else d / "data_clean.part.npy"
    y = np.load(f, mmap_mode="r")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    smp = np.load(d / "samples.npy", mmap_mode="r")
    return np.asarray(y), meta, np.asarray(smp)


refs = {}
for r in opts("--ref"):
    n, p = r.split("=", 1)
    refs[n] = load(p)
first = next(iter(refs.values()))
meta0, smp0 = first[1], first[2]
sigma = float(opt("--sigma", meta0.get("noise_sigma", 1.0)))
R = int(opt("--frame-readouts", 74))
K = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in opts("--k-region")}
region = {kv.split("=")[0]: kv.split("=")[1] for kv in opts("--region")}
na = smp0.shape[1]
# reconstruction window (the rest, e.g. v2's rewinder, is navigator data)
mask = np.zeros(na, bool)
ra = (meta0.get("sequence") or {}).get("recon_adc")
if ra:
    for a, b in ra:
        mask[a:b] = True
else:
    mask[:] = True
kabs = np.linalg.norm(smp0[:, :, :3], axis=2)  # rad/m
kmax = float(np.percentile(kabs[:, mask], 99.9))
bands = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
rows = []
for v in opts("--var"):
    n, rest = v.split("=", 1)
    p, rn = rest.split("@", 1) if "@" in rest else (rest, None)
    rn = rn or next(iter(refs))
    yv, mv, sv = load(p)
    yr = refs[rn][0]
    assert yv.shape == yr.shape, f"{n}: {yv.shape} vs {rn} {yr.shape}"
    assert np.allclose(sv[:, :, :3], refs[rn][2][:, :, :3], atol=1e-3), f"{n}: other samples than {rn}"
    d = (yv - yr)[:, mask, :]
    ref_m = yr[:, mask, :]
    nt = d.shape[0]
    e_sig = float(np.sqrt(np.mean(np.abs(d) ** 2)) / sigma)
    e_rel = float(np.linalg.norm(d) / np.linalg.norm(ref_m))
    kb = kabs[:, mask] / kmax
    eb = []
    for lo, hi in bands:
        s_ = (kb >= lo) & (kb < hi)
        eb.append(float(np.sqrt(np.mean(np.abs(d[s_]) ** 2)) / sigma) if s_.any() else float("nan"))
    Kall = {"all": float(np.prod(meta0.get("recon_dims", [1]))), **K}
    reg = region.get(n, "all")
    kreg = Kall.get(reg, Kall["all"])
    imgs = {k_: float(np.linalg.norm(d) * np.sqrt(R / nt) / (sigma * np.sqrt(v_))) for k_, v_ in Kall.items()}
    img = imgs[reg] if reg in imgs else imgs["all"]
    rows.append({"name": n, "ref": rn, "e_sigma": e_sig, "e_rel": e_rel, "e_sigma_bands": eb, "region": reg, "K": kreg, "img_ratio": img, "img_ratio_by_region": imgs,
                 "readouts": nt, "ref_rms_sigma": float(np.sqrt(np.mean(np.abs(ref_m) ** 2)) / sigma)})
    print(f"{n:28s} vs {rn:10s}: e_σ {e_sig:9.3e}  e_rel {e_rel:9.3e}  image ratio ({reg}, K {kreg:.3g}) {img:9.3e}  all regions {{{', '.join(f'{k_}: {v_:.2e}' for k_, v_ in imgs.items())}}}  bands {['%.2e' % x for x in eb]}")

res = {"sigma": sigma, "frame_readouts": R, "kmax_rad_m": kmax, "bands": bands, "K": K, "rows": rows,
       "refs": {n: str(p) for n, p in (r.split("=", 1) for r in opts("--ref"))}, "vars": opts("--var")}
(OUT / "conv.json").write_text(json.dumps(res, indent=1))
regs = list(rows[0]["img_ratio_by_region"]) if rows else []
md = ["| change | vs | e_σ = rms(d)/σ | ‖d‖/‖y‖ | " + " | ".join(f"image ratio, {k_} (K {rows[0]['img_ratio_by_region'] and ({'all': float(np.prod(meta0.get('recon_dims', [1]))), **K}[k_]):.3g})" for k_ in regs) + " | e_σ per \\|k\\|/kmax band " + " / ".join(f"{a:.2g}–{b:.2g}" for a, b in bands) + " |",
      "|---|---|---|---|" + "---|" * len(regs) + "---|"]
for r in rows:
    md.append(f"| {r['name']} | {r['ref']} | {r['e_sigma']:.2e} | {r['e_rel']:.2e} | " + " | ".join(("**" if k_ == r["region"] else "") + f"{r['img_ratio_by_region'][k_]:.2e}" + ("**" if k_ == r["region"] else "") for k_ in regs) + " | " + " / ".join(f"{x:.1e}" for x in r["e_sigma_bands"]) + " |")
(OUT / "conv.md").write_text(opt("--title", "") + "\n\n" + "\n".join(md) + "\n")

# figures
names = [r["name"] for r in rows]
fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
yy = np.arange(len(rows))[::-1]
ax.barh(yy, [max(r["img_ratio"], 1e-6) for r in rows], color="#3b6ea5", height=0.6)
ax.set_xscale("log")
ax.axvline(0.1, color="#888", lw=1, ls="--")
ax.axvline(1.0, color="#444", lw=1)
ax.set_yticks(yy, [f"{r['name']} ({r['region']})" for r in rows], fontsize=8)
ax.set_xlabel("rms change over the region's voxels / per-voxel noise (one encode, one cardiac frame)")
ax.set_title("simulation convergence: image-domain size of each change (dashed 0.1, solid 1)", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "figs" / "01_image_ratio.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
xc = [0.5 * (a + min(b, 1)) for a, b in bands]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000080", "#808000"]
for i, r in enumerate(rows):
    ax.plot(xc, r["e_sigma_bands"], marker="o", ms=4, lw=1.5, color=COL[i % len(COL)], ls="-" if i < 10 else "--", label=r["name"])
ax.set_yscale("log")
ax.axhline(1.0, color="#444", lw=1)
ax.set_xlabel("|k| / kmax (band centre)")
ax.set_ylabel("rms change per sample / σ")
ax.legend(fontsize=7, ncol=2)
ax.set_title("change per sample by k-space radius", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "figs" / "02_bands.png", dpi=110)
plt.close(fig)

# one readout (the first), root-sum-of-squares over coils
fig, ax = plt.subplots(figsize=(9, 4))
i0 = 0
ix = np.nonzero(mask)[0]
yr = next(iter(refs.values()))[0]
ax.semilogy(ix, np.sqrt((np.abs(yr[i0][mask]) ** 2).sum(1)), color="#222", lw=1, label="|y| reference (rss over coils)")
for i, (v, r) in enumerate(zip(opts("--var"), rows)):
    n, rest = v.split("=", 1)
    p = rest.split("@", 1)[0]
    yv = load(p)[0]
    ax.semilogy(ix, np.maximum(np.sqrt((np.abs(yv[i0][mask] - refs[r["ref"]][0][i0][mask]) ** 2).sum(1)), 1e-3), lw=0.8, color=COL[i % len(COL)], label=f"|d| {n}")
ax.axhline(sigma * np.sqrt(yr.shape[2]), color="#c00", lw=1, ls="--", label="σ √coils (noise rss)")
ax.set_xlabel("ADC sample (reconstruction window)")
ax.legend(fontsize=6, ncol=2)
ax.set_title("readout 0: signal and change per sample", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "figs" / "03_readout.png", dpi=110)
plt.close(fig)
print(f"wrote {OUT}/conv.json, conv.md, figs/")
