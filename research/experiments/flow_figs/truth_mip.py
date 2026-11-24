"""Velocity-weighted MIPs (PC-angiogram style) of the 10-min ph0001 flow simulation truth, to look for banding in the
simulated flow. Two versions:
  * 1 mm fine truth (truth_fine/truth4d.hv: per-frame velocity and blood fraction, before any MR simulation):
    MIP of blood fraction × |v|;
  * 2 mm native truth (truth_vel_logical + truth_pc_signal c0_abs, the simulated reference-encode signal magnitude,
    spin history included): MIP of |m| × |v|.
Frames 0, 3, 6 (peak systole), 10, 16 and the time average of the weighted speed. Projections along the three axes.
    systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
        rustlib/.venv/bin/python -I research/experiments/flow_figs/truth_mip.py
"""
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "python/lib")
from hastycompute import store  # noqa: E402

S = "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/"
O = "research/experiments/flow_figs/"
FR = [0, 3, 6, 10, 16]


def figure(mips, avg, h, tc, title, name, axes_names):
    """mips: {frame: [mip0, mip1, mip2]} images already oriented (rows = vertical axis), avg: [3 images]."""
    rows = [(f"frame {f} (t = {1e3 * tc[f]:.0f} ms)", mips[f]) for f in FR] + [("time average", avg)]
    vmax = float(np.percentile(np.concatenate([m.ravel() for f in FR for m in mips[f]]), 99.95))
    vavg = float(np.percentile(np.concatenate([m.ravel() for m in avg]), 99.95))
    widths = [m.shape[1] for m in avg]
    fig, ax = plt.subplots(len(rows), 3, figsize=(16, 3.1 * len(rows)), gridspec_kw={"width_ratios": widths},
                           constrained_layout=True)
    for i, (lab, ims) in enumerate(rows):
        for j, im in enumerate(ims):
            a = ax[i, j]
            a.imshow(im, origin="lower", cmap="gray", vmin=0, vmax=vavg if lab == "time average" else vmax,
                     extent=[0, im.shape[1] * h, 0, im.shape[0] * h], interpolation="nearest")
            a.set_title(f"{lab}: MIP along {axes_names[j]}", fontsize=9)
            a.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11)
    fig.savefig(O + name, dpi=110)
    plt.close(fig)
    print("wrote", O + name)


# ---- 1 mm fine truth: arrays (F, [3,] z, y, x)
t = store.vf.open_lazy(S + "truth_fine/truth4d.hv")
vel, blood = store.LazyArray(t["vel"]), store.LazyArray(t["blood"])
tc = store.LazyArray(t["tc"]).read()
nf = vel.shape[0]
bmax = None
mips, acc = {}, None
for f in range(nf):
    v = vel[f].astype(np.float32)  # (3, z, y, x)
    b = blood[f].astype(np.float32)
    if bmax is None:
        bmax = float(b.max())
    w = (b / bmax) * np.sqrt((v * v).sum(0))  # (z, y, x)
    acc = w if acc is None else acc + w
    if f in FR:
        # coronal: project along y (rows z, cols x); sagittal: along x (rows z, cols y); axial: along z (rows y, cols x)
        mips[f] = [w.max(1), w.max(2), w.max(0)]
    print("fine frame", f, "max |v|·blood", float(w.max()))
acc /= nf
figure(mips, [acc.max(1), acc.max(2), acc.max(0)], 1.0, tc,
       "1 mm fine truth: MIP of blood fraction × |v| (no MR simulation)", "truth_mip_1mm.png",
       ["y (coronal)", "x (sagittal)", "z (axial)"])
del acc, mips

# ---- 2 mm native truth: arrays (x, y, z, ...)
vn = np.asarray(store.load_any(S + "truth_vel_logical.hv"))  # (X, Y, Z, 3, F)
c0 = np.load(S + "truth_pc_signal.npz")["c0_abs"]  # (X, Y, Z, F)
sp = np.linalg.norm(vn, axis=3) * c0
sp /= np.percentile(c0[c0 > 0], 99)
mips = {f: [sp[..., f].max(1).T, sp[..., f].max(0).T, sp[..., f].max(2).T] for f in FR}
avg = sp.mean(3)
figure(mips, [avg.max(1).T, avg.max(0).T, avg.max(2).T], 2.0, tc,
       "2 mm native truth: MIP of simulated reference-encode magnitude × |v| (spin history included)",
       "truth_mip_2mm.png", ["y (coronal)", "x (sagittal)", "z (axial)"])
