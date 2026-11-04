"""Checks of the tier-1 dataset (CPU): encode-aligned resampling, exact O_h action on the 7p layout, noise calibration.

    python -I test_dataset.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: E402,F401  (caches inside the repo; before torch)
import torch  # noqa: E402
from dataset import TRAIN, Patches, Tier1, collate, layout  # noqa: E402

torch.set_num_threads(4)


def velocity(x, kv):
    """7p layout [7, ...] -> v [..., 3] from the high-VENC encodes"""
    return np.stack([np.angle(x[2 + 2 * a] * np.conj(x[0])) / kv[2 + 2 * a] for a in range(3)], -1)


ds = Tier1(split="test", patch=33, frames=4, lumen_frac=1.0)
mk = Patches(33, sigma_profile=0.0)
get = lambda: mk(collate([ds[0]]))
one = lambda r: {k: (v[0] if torch.is_tensor(v) else v) for k, v in r.items()}
ds.cases, ds.p = ds.cases[:1], np.ones(1)
m = ds.meta(ds.cases[0]["path"])
print("case", ds.cases[0]["path"], m["dims"], "spacing mm", np.round(np.array(m["spacing_m"]) * 1e3, 3))

# (a) patch centre = a native voxel: B-spline interpolation is exact there, so v_patch(centre) = R v_grid(native)
ds.symmetry = lambda rng: (np.arange(3), np.zeros(3, bool))
seen = {}
orig_choose = ds.choose
ds.choose = lambda rng, m_, mk_: (lambda r: (seen.update(centre=r[1], frames=r[0]), r)[1])(orig_choose(rng, m_, mk_))
b0 = one(get())
from dataset import _Case  # noqa: E402

x = _Case(TRAIN / ds.cases[0]["path"]).x  # case.hv (lazy) or npy
c, fr = seen["centre"], seen["frames"]
nat = np.asarray(x[c[0], c[1], c[2]], np.float32)[fr]  # [T, E, 2]
nat = nat[..., 0] + 1j * nat[..., 1]
order, kv = layout(m["encodings"])
kg = np.array([m["encodings"][e]["k_grid"] for e in order])
v_grid = np.stack([np.linalg.lstsq(kg[[2, 4, 6]], np.angle(nat[t, [order[2], order[4], order[6]]] * np.conj(nat[t, order[0]])), rcond=None)[0] for t in range(len(fr))])
R = np.array(m["rot_grid_to_xyz"])
v_pat = velocity(b0["clean"].numpy()[:, :, 16, 16, 16], b0["kv"].numpy())
err_a = np.abs(v_pat - v_grid @ R.T).max()
print(f"(a) centre voxel: v_patch {np.round(v_pat[0], 4)} vs R v_grid {np.round((v_grid @ R.T)[0], 4)} m/s, max diff {err_a:.1e}")
assert err_a < 1e-3

# (b) O_h: v'(g r) = G v(r) on the canonical layout, including the conjugation for flips
v0 = velocity(b0["clean"].numpy(), b0["kv"].numpy())  # [T, P, P, P, 3]
lum0 = b0["mask"].numpy()
worst = 0.0
for perm, flip in [((1, 0, 2), (0, 0, 0)), ((0, 1, 2), (1, 0, 1)), ((2, 0, 1), (0, 1, 1))]:
    perm, flip = np.array(perm), np.array(flip, bool)
    ds.symmetry = lambda rng, perm=perm, flip=flip: (perm, flip)
    b = one(get())
    v = velocity(b["clean"].numpy(), b["kv"].numpy())
    G = np.zeros((3, 3))
    for new, old in enumerate(perm):
        G[new, old] = -1.0 if flip[new] else 1.0
    exp = np.transpose(np.einsum("ij,tabcj->tabci", G, v0), (0, *(perm + 1), 4))
    lm = np.transpose(lum0, (0, *(perm + 1)))
    for a in range(3):
        if flip[a]:
            exp, lm = np.flip(exp, a + 1), np.flip(lm, a + 1)
    assert np.array_equal(lm, b["mask"].numpy())
    d = (v - exp)[lm]
    err = float(np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(exp[lm] ** 2)))
    worst = max(worst, err)
    print(f"(b) perm {perm} flip {flip.astype(int)}: lumen velocity rel. error {err:.1e}")
assert worst < 1e-5, worst

# (c) training configuration: shapes, noise calibration, cost per sample
ds = Tier1(split="train", patch=32, frames=12, h=(1.8e-3, 2.4e-3), length=12)
mk = Patches(32)
tw = tp = 0.0
est, vf = [], []
for j in range(3):
    t0 = time.time()
    raw = collate([ds[4 * j + i] for i in range(4)])
    t1 = time.time()
    bt = mk(raw)
    t2 = time.time()
    tw, tp = tw + t1 - t0, tp + t2 - t1
    for i in range(4):
        va = bt["valid"][i]
        n = (bt["noisy"][i] - bt["clean"][i])[:, :, va]
        est.append(float((n.abs() ** 2 / bt["sigma"][i][va] ** 2).mean()))
        vf.append(float(va.float().mean()))
print(f"(c) batch {tuple(bt['clean'].shape)}, E|n|²/σ² = {np.mean(est):.3f}, valid fraction {np.mean(vf):.2f}; "
      f"worker {tw / 12 * 1e3:.0f} ms/sample, resample+augment {tp / 12 * 1e3:.0f} ms/sample (CPU, 4 threads)")
print("ok")
