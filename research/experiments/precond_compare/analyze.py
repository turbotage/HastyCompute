"""Preconditioner comparison: per-arm table (time split, held-out residual curve, pc score) and the fixed-point check.
usage: PYTHONPATH=python/lib rustlib/.venv/bin/python analyze.py <run root> arm [arm ...]"""
import sys, re, glob, os, itertools
import numpy as np
from hastycompute.store import load_any, exists_any

root, arms = sys.argv[1], sys.argv[2:]


def tag_of(d):
    f = sorted(glob.glob(os.path.join(d, "holdout_*.csv")))
    return os.path.basename(f[0])[len("holdout_"):-4] if f else None


def csv(p):
    with open(p) as fh:
        h = fh.readline().strip().split(",")
        rows = [dict(zip(h, l.strip().split(","))) for l in fh if l.strip()]
    return rows


res = {}
for a in arms:
    d = os.path.join(root, a)
    tag = tag_of(d)
    log = open(os.path.join(d, "log.txt")).read()
    t = [tuple(map(float, m)) for m in re.findall(r"time it\s+(\d+): gradient ([\d.]+)s, step ([\d.]+)s, prior ([\d.]+)s, report ([\d.]+)s", log)]
    t = np.array(t) if t else np.zeros((0, 5))
    ho = csv(os.path.join(d, f"holdout_{tag}.csv"))
    it = np.array([int(r["it"]) for r in ho])
    rel = np.array([float(r["rel"]) for r in ho])
    sec = np.array([float(r["seconds"]) for r in ho])
    pc = csv(os.path.join(d, f"pcscore_{tag}.csv"))
    peak = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", log)
    res[a] = dict(tag=tag, t=t, it=it, rel=rel, sec=sec, pc=pc[-1] if pc else None, peak=int(peak[-1]) / 2**20 if peak else float("nan"), d=d)

print("| arm | s/it (gradient + prior + step + report) | held-out rel. residual it 0 / 10 / 20 / 40 / 60 | min (it) | within 1 % of min: it, wall s | pc RMSE / median m/s | peak RSS GiB |")
print("|---|---|---|---|---|---|---|")
for a, r in res.items():
    t = r["t"]
    if len(t):
        g, s_, p_, rp = t[:, 1].mean(), t[:, 2].mean(), t[:, 3].mean(), t[:, 4].mean()
        ts = f"{g + s_ + p_ + rp:.1f} ({g:.1f} + {p_:.1f} + {s_:.1f} + {rp:.1f})"
    else:
        ts = "-"
    at = lambda k: f"{r['rel'][r['it'] == k][0]:.5f}" if (r["it"] == k).any() else "-"
    i0 = int(np.argmin(r["rel"]))
    mn = r["rel"][i0]
    j = int(np.nonzero(r["rel"] <= 1.01 * mn)[0][0])
    pc = r["pc"]
    pcs = f"{float(pc['rmse']):.3f} / {float(pc['median']):.3f} (it {pc['it']})" if pc else "-"
    print(f"| {a} | {ts} | {' / '.join(at(k) for k in (0, 10, 20, 40, 60))} | {mn:.5f} ({r['it'][i0]}) | {r['it'][j]}, {r['sec'][j]:.0f} | {pcs} | {r['peak']:.1f} |")
print()
print("held-out curves (it: rel):")
for a, r in res.items():
    print(f"  {a}: " + ", ".join(f"{k}: {v:.5f}" for k, v in zip(r["it"], r["rel"])))
print()

# fixed-point check: final encode-0 images (all voxels and the coil-map support) and velocities (support, pc voxels)
imgs, vels = {}, {}
for a, r in res.items():
    pr, pv = os.path.join(r["d"], f"recon_{r['tag']}.npy"), os.path.join(r["d"], f"vel_{r['tag']}.npy")
    if exists_any(pr):
        imgs[a] = np.asarray(load_any(pr)[:, :, :, 0, :])
    if exists_any(pv):
        vels[a] = np.asarray(load_any(pv))
if imgs:
    a0 = next(iter(imgs))
    sup = np.asarray(load_any(os.path.join(res[a0]["d"], "mask_support.npy"))).astype(bool)
    pcv = np.asarray(load_any(os.path.join(res[a0]["d"], "pc_vox.npy"))).astype(np.int64)
    pcm = np.zeros(sup.size, bool)
    pcm[pcv] = True
    pcm = pcm.reshape(sup.shape, order="F")
    print(f"fixed-point check: support {sup.sum()} voxels, pc voxels {pcm.sum()}")
    print("| pair | encode-0 image, all voxels | encode-0 image, support | velocity, support | velocity, pc voxels |")
    print("|---|---|---|---|---|")
    rd = lambda x, y, m: np.linalg.norm((x - y)[m]) / np.linalg.norm(y[m])
    for a, b in itertools.combinations(imgs, 2):
        xa, xb = imgs[a], imgs[b]
        row = [rd(xa, xb, np.ones(xa.shape, bool)), rd(xa, xb, np.broadcast_to(sup[..., None], xa.shape))]
        if a in vels and b in vels:
            va, vb = vels[a], vels[b]
            row += [rd(va, vb, np.broadcast_to(sup[..., None, None], va.shape)), rd(va, vb, np.broadcast_to(pcm[..., None, None], va.shape))]
        else:
            row += [float("nan")] * 2
        print(f"| {a} vs {b} | " + " | ".join(f"{v:.2e}" for v in row) + " |")
