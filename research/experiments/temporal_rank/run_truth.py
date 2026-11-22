"""Run 1: the cardiac rank of ph0001's own truth (100 cardiac frames, end-expiration), plus the known-rank control.

    rustlib/.venv/bin/python run_truth.py
"""
import pickle
import time

import numpy as np

from common import CACHE, LEVELS, k_at, synthetic_rank_control
from experiment import Context, analyse, save, table, grams, eig_desc
from rank_lib import GROUPS, Fields, Vessels, build_X, coarse_masks, coarse_vessel_map


def control():
    """a rank-7 signal (spectrum 2^-k) must give K = 7 at every level below the 7th singular value's share, and the
    projection of the noisy version onto its top 7 must leave only the noise"""
    out = {}
    X = synthetic_rank_control(rank=7)
    w, _ = eig_desc((X.conj().T @ X))
    out["exact_K"] = k_at(w, LEVELS + [1e-6])
    out["exact_sv"] = (np.sqrt(w[:9] / w[0])).tolist()
    Xn = synthetic_rank_control(rank=7, noise=1e-3)
    w2, _ = eig_desc((Xn.conj().T @ Xn))
    out["noisy_sv"] = (np.sqrt(w2[:10] / w2[0])).tolist()
    return out


def context(F, Ve):
    p = CACHE / "context.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    b2, cat = coarse_masks(F)
    vmap = coarse_vessel_map(F, Ve, GROUPS)
    p.write_bytes(pickle.dumps((b2, cat, vmap)))
    return b2, cat, vmap


if __name__ == "__main__":
    t0 = time.time()
    c = control()
    print("control", c, flush=True)
    F = Fields()
    print(f"fields {F.fs} -> {F.cs} ({F.nv} voxels), {time.time() - t0:.0f} s", flush=True)
    Ve = Vessels()
    b2, cat, vmap = context(F, Ve)
    print(f"context {time.time() - t0:.0f} s; vessel voxels per group (frame 0): " + ", ".join(f"{g} {len(v[0][0])}" for g, v in vmap.items()), flush=True)
    X = build_X(F)
    print(f"X built {time.time() - t0:.0f} s", flush=True)
    ctx = Context(X, b2, vmap)
    print(f"signal voxels {ctx.sig.sum()}, lumen {ctx.bl.sum()}, decoded {len(ctx.idxB)}", flush=True)
    res = analyse(X, ctx, "ph0001 truth, r = 0, 100 cardiac frames")
    res["control"] = c
    save(res, "truth_cardiac")
    print("K shared", res["K_shared"], "K blood", res["K_blood_shared"], "per encode", res["K_per_encode"])
    print(table(res))
    print(f"total {time.time() - t0:.0f} s")
