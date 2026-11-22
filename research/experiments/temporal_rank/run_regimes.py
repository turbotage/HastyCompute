"""Run 3: cardiac regime sweep (research/temporal_rank.md). Each run builds the encode images of a perturbed truth and
measures its temporal rank and the errors of the SVD and B-spline bases. Resumable: a run whose out/regime_<name>.json
exists is skipped (touch out/STOP to stop between runs).

    rustlib/.venv/bin/python run_regimes.py [names...]
"""
import json
import pickle
import sys
import time

import numpy as np

from common import CACHE, OUT, encode_matrix
from experiment import Context, analyse, save, table
from rank_lib import Fields, Vessels, build_X
from regimes import Overlay, SmoothMagnitude

KS_Q = [4, 6, 8, 12, 16, 20, 24, 32, 48]
BS_Q = [12, 16, 20, 24, 32]


def runs():
    R = {}
    R["truth_smooth"] = dict(mag="smooth")  # truth velocity, continuous-motion magnitude
    R["overlay_ref"] = dict(mag="smooth", overlay={})
    for s in [0.25, 0.5, 0.75, 1.5, 2.0]:
        R[f"venc_x{s}"] = dict(mag="smooth", overlay={}, kscale=s)
    for t in [25, 50, 100]:
        R[f"tau_{t}ms"] = dict(mag="smooth", overlay={"tau_ms": t})
    for s in [0.1, 0.2, 0.4]:
        R[f"shape_{s}"] = dict(mag="smooth", overlay={"shape": s})
    for ramp in [60, 30, 15]:
        R[f"coronary_ramp{ramp}"] = dict(mag="smooth", overlay={"coronary": {"ramp_ms": ramp, "peak_ratio": 2.5}})
    R["coronary_ramp15_v2"] = dict(mag="smooth", overlay={"coronary": {"ramp_ms": 15, "peak_ratio": 2.5, "v_scale": 2.0}})
    R["combined_tau50_shape0.2_cor30"] = dict(mag="smooth", overlay={"tau_ms": 50, "shape": 0.2, "coronary": {"ramp_ms": 30, "peak_ratio": 2.5}})
    return R


if __name__ == "__main__":
    want = sys.argv[1:]
    t0 = time.time()
    F = Fields()
    b2, cat, vmap = pickle.loads((CACHE / "context.pkl").read_bytes())
    sm = None
    ov = None
    Ve = None
    for name, cfg in runs().items():
        if want and name not in want:
            continue
        if (OUT / f"regime_{name}.json").exists():
            continue
        if (OUT / "STOP").exists():
            print("STOP", flush=True)
            break
        if cfg.get("mag") == "smooth" and sm is None:
            sp = CACHE / "smooth_mag.npy"
            if not sp.exists():
                np.save(sp, SmoothMagnitude(F).m)
            sm = np.load(sp)
            smf = lambda f: sm[f].astype(np.float32)  # noqa: E731
            print(f"smooth magnitude ready {time.time() - t0:.0f} s", flush=True)
        vel = None
        if "overlay" in cfg:
            if ov is None:
                Ve = Vessels()
                ov = Overlay(F, Ve)
                print(f"overlay ready ({len(ov.pts)} centreline points) {time.time() - t0:.0f} s", flush=True)
            Q = ov.waveforms(cfg["overlay"])
            vel = ov.velocity(Q)
        K = encode_matrix() * cfg.get("kscale", 1.0)
        X = build_X(F, vel=vel, mag=smf if cfg.get("mag") == "smooth" else None, K=K)
        ctx = Context(X, b2, vmap)
        res = analyse(X, ctx, name, Ks=KS_Q, bspl=BS_Q, quick=True)
        res["config"] = cfg
        # v_max / VENC_lo of the run: the 99th percentile blood speed of the truth decode, over VENC_lo = 1 m/s / kscale
        from experiment import truth_refs
        vt, _ = truth_refs(X, ctx)
        okm = ctx.b2[ctx.idxB] >= 0.5
        sp = np.linalg.norm(vt, axis=-1)[okm]
        res["speed_p99_over_venc_lo"] = float(np.percentile(sp, 99))  # decoded with the unscaled K: s v
        res["speed_max_over_venc_lo"] = float(sp.max())
        save(res, f"regime_{name}")
        print(f"== {name}: K_shared {res['K_shared']} K_blood {res['K_blood_shared']} ({time.time() - t0:.0f} s)", flush=True)
        print(table(res), flush=True)
        del X
