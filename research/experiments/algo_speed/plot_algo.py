"""Convergence per cost of recon_p2np solver variants (algo_<tag>.csv, research/algo_speed.md).

    plot_algo.py <out.png> <run dir>[:<tag substring>] ...

Every algo_*.csv of the given run directories (optionally only tags containing the substring). Panels against the
cumulative gradient cost (full-gradient units) and against wall time: data term f (exact where an exact value of the
iteration exists), relative velocity RMSE (both decoders), slope of the recon vs truth (all, per-frame deviations).
Prints a markdown table of the last row and of the row at a common cost.
"""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
runs = []
for arg in sys.argv[2:]:
    d, _, sub = arg.partition(":")
    for p in sorted(glob.glob(os.path.join(d, "algo_*.csv"))):
        tag = os.path.basename(p)[5:-4]
        if sub and sub not in tag:
            continue
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        a = {k: np.array([float(r[k]) if r[k] not in ("", "NaN", "nan") else np.nan for r in rows]) for k in rows[0]}
        runs.append((os.path.basename(d.rstrip("/")) + "/" + tag, a))

def fx(a):
    """f per row: the exact value when it belongs to the row's iteration, else the reported one"""
    f = a["f"].copy()
    ex = (a["f_exact_iter"] + 1 == a["iter"]) & np.isfinite(a["f_exact"])
    f[ex] = a["f_exact"][ex]
    return f

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for name, a in runs:
    it = a["iter"]; m = it > 0
    c = a["cost"]
    for j, (key, lab) in enumerate([("f", "f (data term)"), ("rel", "rel. velocity RMSE"), ("slope_dev", "slope of per-frame deviations")]):
        y = fx(a) if key == "f" else a[key]
        ax[0, j].plot(c[m], y[m], marker=".", label=name)
        ax[1, j].plot(a["seconds"][m], y[m], marker=".", label=name)
        ax[0, j].set_xlabel("cost (gradient-equivalents)"); ax[1, j].set_xlabel("wall time (s)")
        ax[0, j].set_title(lab); ax[1, j].set_title(lab)
        if key == "f":
            ax[0, j].set_yscale("log"); ax[1, j].set_yscale("log")
ax[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(f"figure: {out}\n")
print("| run | iterations | cost | f | rel. vel. RMSE (ML / hier) | slope | slope dev. (corr) | slope mean | wall s |")
print("|---|---|---|---|---|---|---|---|---|")
for name, a in runs:
    i = len(a["iter"]) - 1
    print(f"| {name} | {int(a['iter'][i])} | {a['cost'][i]:.1f} | {fx(a)[i]:.4e} | {a['rel'][i]:.3f} / {a['rel_h'][i]:.3f} | {a['slope'][i]:.3f} | "
          f"{a['slope_dev'][i]:.3f} ({a['corr_dev'][i]:.2f}) | {a['slope_mean'][i]:.3f} | {a['seconds'][i]:.0f} |")
