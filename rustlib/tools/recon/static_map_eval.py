"""Static / dynamic map of a recon_p2np run (--static-lock) against the phantom's motion truth.

    rustlib/.venv/bin/python -I static_map_eval.py <run_dir> [<phantom_dir>]

Truth per reconstruction voxel x (subject frame, index i ↦ centre + (i − n/2) res; displacements evaluated at the atlas
pose X ≈ x, small against their spatial scale):
- d_card: peak deviation of the cardiac displacement from its cycle mean, max_t |u_c(x, t) − mean_t u_c(x, t)| (mm);
- d_resp: end-expiration → end-inspiration displacement |Σ_k B_k(x)| (mm);
- blood: truth lumen fraction (frame-averaged, chambers included) > 0.1;
- visible motion: the displacement changes the image, d · |∇PD| > 5 % of the bright-tissue PD (a uniform organ interior
  that moves is image-static: locking it is harmless).
Classes (cardiac / respiratory): moving = blood or visible motion; static = no blood, d < 0.3 mm (cardiac) / 1 mm
(respiratory). Reports the locked fraction (w > 0.5) per class and writes figs/21_<tag>_static_truth.png and
static_eval_<tag>.json into the run directory.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

run = Path(sys.argv[1])
info = json.loads((run / "run.json").read_text())
sim = Path(info["sim_dir"])
meta = json.loads((sim / "meta.json").read_text())
ph = Path(sys.argv[2]) if len(sys.argv) > 2 else sim.parent.parent
dims = np.array(meta["recon_dims"])
res = float(meta["res"][0])
centre = np.array(meta["presc_center_subject"])
assert np.allclose(np.array(meta["rot_presc"]), np.eye(3)), "oblique prescriptions: not handled"
# voxel centres in the subject frame, [x, y, z, 3]
g = [centre[a] + (np.arange(dims[a]) - dims[a] / 2) * res for a in range(3)]
X = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1)


def sample(field_zyx, origin, vox, pts):
    """trilinear sample of a [z, y, x] grid at pts [..., 3] (subject m); 0 outside (cardiac) / clamped (resp)"""
    idx = [(pts[..., 2 - a] - origin[2 - a]) / vox for a in range(3)]  # z, y, x indices
    return map_coordinates(field_zyx, idx, order=1, mode="nearest")


c = vf.read(ph / "motion/cardiac_motion.hv")
U = np.asarray(c["U_mm"], dtype=np.float32)  # [t, comp, z, y, x] mm
Um = U - U.mean(axis=0, keepdims=True)
dev = np.sqrt((Um ** 2).sum(axis=1)).max(axis=0)  # [z, y, x]
inside = np.zeros_like(dev)
inside[1:-1, 1:-1, 1:-1] = 1.0
d_card = sample(dev, c["origin_m"], c["voxel_m"], X) * (sample(inside, c["origin_m"], c["voxel_m"], X) > 0.5)
r = vf.read(ph / "motion/resp_motion.hv")
B = np.asarray(r["B"], dtype=np.float32).sum(axis=0)  # [comp, z, y, x] m
d_resp = 1e3 * np.sqrt(sum(sample(B[k], r["origin_m"], r["voxel_m"], X) ** 2 for k in range(3)))

lum = np.load(sim / "truth_lumen_fraction.npy") > 0.1
pd = np.load(sim / "truth_pd_water.npy")
if (sim / "truth_pd_fat.npy").exists():
    pd = pd + np.load(sim / "truth_pd_fat.npy")
grad = np.sqrt(sum(np.gradient(pd, res * 1e3, axis=a) ** 2 for a in range(3)))  # per mm
bright = np.percentile(pd[pd > 0], 95) if (pd > 0).any() else 1.0
vis_card = lum | (d_card * grad > 0.05 * bright)
vis_resp = lum | (d_resp * grad > 0.05 * bright)
stat_card = ~lum & (d_card < 0.3)
stat_resp = ~lum & (d_resp < 1.0)
body = pd > 0.05 * bright

out = {}
for mp in sorted(run.glob("static_map_*.npy")):
    tag = mp.stem[len("static_map_"):]
    m = np.load(mp)
    w_card, w_resp = m[..., 3] > 0.5, m[..., 4] > 0.5
    frac = lambda w, cls: float(w[cls].mean()) if cls.any() else float("nan")  # noqa: E731
    e = {
        "cardiac": {"static_truth_locked": frac(w_card, stat_card & body), "visible_moving_locked": frac(w_card, vis_card), "blood_locked": frac(w_card, lum), "n_static": int((stat_card & body).sum()), "n_visible_moving": int(vis_card.sum())},
        "respiratory": {"static_truth_locked": frac(w_resp, stat_resp & body), "visible_moving_locked": frac(w_resp, vis_resp), "blood_locked": frac(w_resp, lum), "n_static": int((stat_resp & body).sum()), "n_visible_moving": int(vis_resp.sum())},
        "locked_of_body": [frac(w_card, body), frac(w_resp, body)],
    }
    out[tag] = e
    print(f"{tag}:")
    for k in ("cardiac", "respiratory"):
        v = e[k]
        print(f"  {k:<11}: static truth locked {100 * v['static_truth_locked']:5.1f} % (n {v['n_static']}), visibly moving locked {100 * v['visible_moving_locked']:5.2f} % (n {v['n_visible_moving']}), blood locked {100 * v['blood_locked']:5.2f} %")
    # figure: truth classes vs the map, axial / coronal through the heart
    zs = int(np.argmax(vis_card.sum(axis=(0, 1))))
    ys = int(np.argmax(vis_card.sum(axis=(0, 2))))
    figs = run / "figs"
    figs.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 4, figsize=(15, 6))
    for row, sl in enumerate((lambda a: a[:, :, zs].T, lambda a: a[:, ys, :].T)):
        ax[row, 0].imshow(sl(d_card), cmap="magma", vmin=0, vmax=8, origin="lower", aspect="auto")
        ax[row, 0].set_title("cardiac displacement dev (mm)", fontsize=8)
        ax[row, 1].imshow(sl(m[..., 3]), cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto")
        ax[row, 1].contour(sl(vis_card.astype(float)), levels=[0.5], colors="r", linewidths=0.6)
        ax[row, 1].set_title("w_card (red: visibly moving truth)", fontsize=8)
        ax[row, 2].imshow(sl(d_resp), cmap="magma", vmin=0, vmax=15, origin="lower", aspect="auto")
        ax[row, 2].set_title("respiratory displacement (mm)", fontsize=8)
        ax[row, 3].imshow(sl(m[..., 4]), cmap="viridis", vmin=0, vmax=1, origin="lower", aspect="auto")
        ax[row, 3].contour(sl(vis_resp.astype(float)), levels=[0.5], colors="r", linewidths=0.6)
        ax[row, 3].set_title("w_resp (red: visibly moving truth)", fontsize=8)
        for a in ax[row]:
            a.axis("off")
    c_, r_ = e["cardiac"], e["respiratory"]
    fig.suptitle(f"{tag}: locked — static truth {100 * c_['static_truth_locked']:.1f} % / {100 * r_['static_truth_locked']:.1f} %, visibly moving {100 * c_['visible_moving_locked']:.2f} % / {100 * r_['visible_moving_locked']:.2f} % (cardiac / respiratory)", fontsize=9)
    fig.savefig(figs / f"21_{tag}_static_truth.png", dpi=80)
    plt.close(fig)
    (run / f"static_eval_{tag}.json").write_text(json.dumps(e, indent=1))
if not out:
    print(f"no static_map_*.npy in {run}")
