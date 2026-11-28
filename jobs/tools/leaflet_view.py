"""Open leaflets of an LBM case's valves seen along the valve axis: the blocked columns of lbm_case.open_area (strips
turned by θ about their hinge chords, thickness + one cell) and the strips' far edges, per valve.
   rustlib/.venv/bin/python -I jobs/tools/leaflet_view.py <case.json> <out.png> [dx_mm=1] [theta_deg_override]"""
import ast
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

src = (Path(__file__).resolve().parents[2] / "rustlib/tools/phantom/lbm_case.py").read_text()
ns = {"np": np}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("leaflet_polys",):
        exec(compile(ast.Module([node], []), "lbm_case", "exec"), ns)
case = json.loads(Path(sys.argv[1]).read_text())
dx = float(sys.argv[3]) * 1e-3 if len(sys.argv) > 3 else 1e-3
th_over = np.radians(float(sys.argv[4])) if len(sys.argv) > 4 else None
vs = case["valves"]
fig, axs = plt.subplots(2, len(vs), figsize=(4.2 * len(vs), 8.4), squeeze=False)
for k, v in enumerate(vs):
    R = v["radius"]
    th = th_over if th_over is not None else v["theta_open"]
    th_r = (v["thickness"] + dx) / R
    polys = ns["leaflet_polys"](v["leaflets"])
    g = np.linspace(-1, 1, 301)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], -1)
    blocked = np.zeros(len(P), bool)
    ct, st = np.cos(th), np.sin(th)
    ax2 = axs[1, k]
    for q in polys:
        pa, d = q[0], q[1] - q[0]
        d = d / np.linalg.norm(d)
        m0 = np.array([-d[1], d[0]])
        if (q.mean(0) - pa) @ m0 < 0:
            m0 = -m0
        A_, B_ = (q - pa) @ d, (q - pa) @ m0
        a, c = (P - pa) @ d, (P - pa) @ m0
        bmin, bmax = np.full(len(P), np.inf), np.full(len(P), -np.inf)
        for i in range(len(q)):
            a0_, a1_, b0_, b1_ = A_[i], A_[(i + 1) % len(q)], B_[i], B_[(i + 1) % len(q)]
            lo_, hi_ = min(a0_, a1_), max(a0_, a1_)
            m_ = (a >= lo_) & (a <= hi_)
            f_ = np.where(abs(a1_ - a0_) > 1e-12, (a - a0_) / (a1_ - a0_ + 1e-30), 0.0)
            bb_ = b0_ + f_ * (b1_ - b0_)
            bmin, bmax = np.where(m_, np.minimum(bmin, bb_), bmin), np.where(m_, np.maximum(bmax, bb_), bmax)
        lo_b, hi_b = (c - 0.5 * th_r * st) / max(ct, 1e-6), (c + 0.5 * th_r * st) / max(ct, 1e-6)
        blocked |= (bmax >= bmin) & (hi_b >= bmin) & (lo_b <= bmax)
        blocked |= (a >= 0) & (a <= np.linalg.norm(q[1] - q[0])) & (c < 0)
        # side view: the strip's far edge height (b_max · sin θ) and inward reach (b_max · cos θ) per strip
        bm = B_.max()
        tip = pa + d * 0.5 * np.linalg.norm(q[1] - q[0]) + m0 * bm * ct
        axs[0, k].plot([pa[0], tip[0]], [pa[1], tip[1]], "c-", lw=0.8)
        ax2.plot([np.linalg.norm(pa + d * 0.5 * np.linalg.norm(q[1] - q[0])) * R * 1e3, np.linalg.norm(tip) * R * 1e3], [0, bm * st * R * 1e3], "k-", lw=0.8)
    inside = (X**2 + Y**2 < 1).ravel()
    img = np.where(inside, np.where(blocked, 1.0, 0.0), np.nan).reshape(X.shape)
    rl = v.get("lumen_radius", R) / R
    axs[0, k].imshow(img.T, origin="lower", extent=[-1, 1, -1, 1], cmap="Greys", vmin=0, vmax=1.5)
    axs[0, k].add_patch(plt.Circle((0, 0), rl, fill=False, color="r", lw=0.8))
    axs[0, k].set_title(f"{v['name']} {v['leaflets']} at {np.degrees(th):.0f}°: open {100 * (1 - blocked[inside & ((X**2 + Y**2) < rl**2).ravel()].mean()) * rl**2:.0f} % of the disk")
    ax2.set_xlabel("radius (mm)")
    ax2.set_ylabel("height above the hinge plane (mm)")
    ax2.set_title("strips: hinge (radius, 0) → far edge")
    ax2.axvline(rl * R * 1e3, color="r", lw=0.6)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=80)
print(sys.argv[2])
