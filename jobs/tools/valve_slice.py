"""Slices of an LBM case's blood domain (sdf > 0) through a valve axis: the plane holding the valve's axis and a second
valve's centre (default aortic + mitral), and the plane across the axis at a few depths. Several cases side by side.
   rustlib/.venv/bin/python -I jobs/tools/valve_slice.py <out.png> <case_dir> [<case_dir> …] [--valve aortic] [--with mitral]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

A = sys.argv[1:]


def opt(n, d):
    return A[A.index(n) + 1] if n in A else d


VN, WN = opt("--valve", "aortic"), opt("--with", "mitral")
pos = [a for i, a in enumerate(A) if not a.startswith("--") and (i == 0 or not A[i - 1].startswith("--"))]
out, cases = Path(pos[0]), [Path(p) for p in pos[1:]]
fig, axs = plt.subplots(len(cases), 4, figsize=(17, 4.3 * len(cases)), squeeze=False)
for row, cd in zip(axs, cases):
    case = json.loads((cd / "case.json").read_text())
    sd = case["sdf"] if isinstance(case["sdf"], dict) else {}
    S = np.load(cd / "sdf.npy")
    org = np.asarray(sd.get("origin", case["grid"]["origin"]), float)
    hh = float(sd.get("h", case["grid"]["dx"]))
    v = next(x for x in case["valves"] if x["name"] == VN)
    w = next(x for x in case["valves"] if x["name"] == WN)
    c, n = np.asarray(v["center"]), np.asarray(v["normal"])
    t = np.asarray(w["center"]) - c
    t -= (t @ n) * n
    t /= np.linalg.norm(t)
    b = np.cross(n, t)
    rl = float(v.get("lumen_radius", v["radius"]))

    def samp(P):
        g = (P - org) / hh - 0.5
        return ndimage.map_coordinates(S, [g[..., 2], g[..., 1], g[..., 0]], order=1, cval=-1.0)

    s_ = np.arange(-40e-3, 40e-3, 0.25e-3)
    U, V = np.meshgrid(s_, s_, indexing="xy")
    P = c + U[..., None] * t + V[..., None] * n
    row[0].imshow(samp(P) > 0, origin="lower", extent=[s_[0] * 1e3, s_[-1] * 1e3, s_[0] * 1e3, s_[-1] * 1e3], cmap="Greys_r")
    row[0].axhline(0, color="r", lw=0.6)
    for sgn in (-1, 1):
        row[0].plot([sgn * rl * 1e3] * 2, [-25, 25], "c--", lw=0.6)
    wc = (np.asarray(w["center"]) - c)
    row[0].plot(wc @ t * 1e3, wc @ n * 1e3, "y+", ms=12)
    row[0].set_title(f"{cd.parent.name}: plane of the {VN} axis (up) and the {WN} centre (+)")
    row[0].set_xlabel("mm across")
    row[0].set_ylabel("mm along the valve axis (downstream +)")
    areas = []
    for k, dep in enumerate((-12e-3, -6e-3, -2e-3)):
        Pc = c + dep * n + U[..., None] * t + V[..., None] * b
        m = samp(Pc) > 0
        lab, _ = ndimage.label(m)
        kk = lab[lab.shape[0] // 2, lab.shape[1] // 2]
        mm = lab == kk if kk > 0 else np.zeros_like(m)
        areas.append(mm.sum() * (0.25e-3) ** 2 * 1e4)
        row[k + 1].imshow(m.astype(float) + mm, origin="lower", extent=[s_[0] * 1e3, s_[-1] * 1e3, s_[0] * 1e3, s_[-1] * 1e3], cmap="viridis")
        row[k + 1].add_patch(plt.Circle((0, 0), rl * 1e3, fill=False, color="r", lw=0.8))
        row[k + 1].set_title(f"{dep * 1e3:+.0f} mm: axis component {areas[-1]:.2f} cm² (lumen {np.pi * rl**2 * 1e4:.2f})")
    print(f"{cd}: {VN} axis-connected area at −12 / −6 / −2 mm: {', '.join(f'{a:.2f}' for a in areas)} cm² (lumen {np.pi * rl**2 * 1e4:.2f})")
fig.tight_layout()
fig.savefig(out, dpi=90)
print(out)
