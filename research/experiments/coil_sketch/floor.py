"""Stochastic-PnP neighbourhood: start keep+draw SGD / SVRG at the full-coil PnP fixed point x* and measure how far
the iterates wander (coil_cost_reduction.md §6). Needs toy.py's PnP caches (out/pnp_global.json, runs/global_ref_pnp_e*.npz).

    rustlib/.venv/bin/python -I research/experiments/coil_sketch/floor.py [iters]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
pj = json.loads((OUT / "pnp_global.json").read_text())
thr = min(pj["sweep"].values(), key=lambda v: v[1])[0]
alpha = pj["alpha"]
P = toy.Problem()
P.lam = json.loads((OUT / "lambda_global.json").read_text())["lam"]
res = {}
ff = OUT / "floor.json"
if ff.exists():
    res = json.loads(ff.read_text())
for name, k, d, kind in (("sgd_k12d3", 12, 3, "sgd"), ("sgd_k8d4", 8, 4, "sgd"), ("sgd_k12d6", 12, 6, "sgd"), ("svrg_k12d3m10", 12, 3, "svrg"), ("full", 0, 0, "full")):
    if name in res:
        continue
    out = []
    for e in range(2):
        xs = np.load(OUT / "runs" / f"global_ref_pnp_e{e}.npz")["x"]
        x = xs.copy()
        s = toy.Sampler(P, k, d, "sub", seed=5)
        dist = []
        xa, mu = None, None
        for it in range(ITERS):
            if kind == "full":
                gf = toy.grad_f_full(P, e, x)
            elif kind == "sgd":
                maps, w, _ = s.draw_b()
                gf = P.normal(e, x, maps, w) - s.bsum(e, maps, w)
            else:
                if it % 10 == 0:
                    xa, mu = x.copy(), toy.grad_f_full(P, e, x)
                maps, w, _ = s.draw()
                gf = mu + P.normal(e, x - xa, maps, w)
            x = P.soft(x - alpha * P.precond(e, gf), thr)
            dist.append(np.linalg.norm(x - xs) / np.linalg.norm(xs))
        out.append(dist)
    dist = np.mean(out, 0)
    res[name] = dict(rms_last_half=float(np.sqrt(np.mean(dist[ITERS // 2:] ** 2))), at_end=float(dist[-1]), trace=dist[::10].tolist())
    ff.write_text(json.dumps(res, indent=1))
    print(name, f"rms distance from x* (last half) {res[name]['rms_last_half']:.3e}, end {res[name]['at_end']:.3e}", flush=True)
