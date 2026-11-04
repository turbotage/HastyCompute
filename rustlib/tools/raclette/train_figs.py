"""Quick-look montages of the converted RACLETTE training cases, to catch broken conversions.

    python -I train_figs.py

Per variant (and sub-level) under data/hasty/train/raclette/figs/: |reference| and speed at the most-lumen axial slice
and the frame of peak lumen speed, lumen contour in yellow; index.md lists the montages and case statistics.
Velocity per voxel: least squares over the highest-VENC encodes, phases relative to the k = 0 encode.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402
TRAIN = REPO / "data/hasty/train/raclette"
FIGS = TRAIN / "figs"
FIGS.mkdir(exist_ok=True)


def velocity(xs, enc):
    """xs [..., E] complex; v [..., 3] in the grid basis from the highest-VENC encodes (x_e ∝ exp(+i k_e·v))"""
    k = np.array([e["k_grid"] for e in enc])
    venc = np.array([e["venc"] for e in enc])
    ref = int(np.flatnonzero(np.linalg.norm(k, axis=1) == 0)[0])
    sel = np.flatnonzero(np.isclose(venc, venc.max()) & (np.linalg.norm(k, axis=1) > 0))
    ph = np.angle(xs[..., sel] * np.conj(xs[..., ref:ref + 1]))
    return ph @ np.linalg.pinv(k[sel]).T, float(venc.max())


groups = defaultdict(list)
for f in sorted(TRAIN.glob("*/*/case.json")) + sorted(TRAIN.glob("*/*/*/case.json")):
    c = json.loads(f.read_text())
    groups[(c["variant"], c["sub"])].append((f.parent, c))

lines = ["# RACLETTE training cases (tier 1)\n"]
for (v, sub), cases in sorted(groups.items()):
    n = len(cases)
    nc = int(np.ceil(np.sqrt(n * 1.5)))
    nr = int(np.ceil(n / nc))
    figs = {k: plt.subplots(nr, nc, figsize=(1.6 * nc, 1.5 * nr), squeeze=False) for k in ["mag", "speed"]}
    stats = []
    for i, (d, c) in enumerate(cases):
        if (d / "case.hv").exists():  # compressed, chunked: read only the slice
            t = vf.open_lazy(d / "case.hv")
            x, m = t["x"], t["mask"]
        else:
            x, m = np.load(d / "x.npy", mmap_mode="r"), np.load(d / "mask.npy", mmap_mode="r")
        z = int(np.argmax(np.asarray(m[:, :, :, 0]).sum(axis=(0, 1))))
        sl = np.asarray(x[:, :, z]).astype(np.float32)
        sl = sl[..., 0] + 1j * sl[..., 1]  # [X, Y, F, E]
        vel, venc = velocity(sl, c["encodings"])
        sp = np.linalg.norm(vel, axis=-1)  # [X, Y, F]
        lm = np.asarray(m[:, :, z, :]) > 0
        t = int(np.argmax([sp[..., f][lm[..., f]].mean() if lm[..., f].any() else 0 for f in range(sp.shape[-1])]))
        a0 = np.abs(sl[..., t, 0])
        stats.append((d.relative_to(TRAIN), c["group"], c["stenosis_grade"], c["dims"], c["n_frames"], c["n_enc"], float(sp[..., t][lm[..., t]].max()), c["peak_velocity_mps"], c["fp16_roundtrip_rel"]))
        for key, img, kw in [("mag", a0, dict(cmap="gray", vmin=0, vmax=np.percentile(a0, 99.5))), ("speed", sp[..., t], dict(cmap="inferno", vmin=0, vmax=venc))]:
            ax = figs[key][1][i // nc, i % nc]
            ax.imshow(img.T, origin="lower", interpolation="nearest", **kw)
            ax.contour(lm[..., t].T.astype(float), levels=[0.5], colors="y", linewidths=0.4)
            ax.set_title(f"{c['subject'][-4:]} t{t} {c['stenosis_grade'][:6]}", fontsize=5)
    tag = f"{v}{'_' + sub if sub else ''}"
    for key, (fig, axs) in figs.items():
        for a in axs.ravel():
            a.axis("off")
        fig.suptitle(f"{tag}: {'|x_ref|' if key == 'mag' else 'speed (m/s, 0..VENC_max)'} at peak lumen-speed frame, most-lumen slice", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / f"{key}_{tag}.png", dpi=90)
        plt.close(fig)
    lines.append(f"## {tag} ({n} cases)\n\n![mag](mag_{tag}.png)\n\n![speed](speed_{tag}.png)\n")
    lines.append("| case | background | grade | dims | F | E | max lumen speed (slice) | meta peak v | fp16 rt |\n|---|---|---|---|---|---|---|---|---|")
    lines += [f"| {a} | {b} | {g} | {dm} | {F} | {E} | {s:.2f} | {pv:.2f} | {rt:.1e} |" for a, b, g, dm, F, E, s, pv, rt in stats]
    lines.append("")
(FIGS / "index.md").write_text("\n".join(lines))
print(f"{sum(len(c) for c in groups.values())} cases, {len(groups)} montages -> {FIGS}")
