"""Pointwise step-size inflation and gradient-variance factors of keep+draw coil subsets (coil_cost_reduction.md §5).

    rustlib/.venv/bin/python -I research/experiments/coil_sketch/inflation.py

Under the circulant (pointwise-diagonal coil) approximation N ≈ D^{1/2} C D^{1/2}, the sampled operator
Ñ_S = N_keep + p N_G has λ_max(P Ñ_S) / λ_max(P N) ≈ max_r ρ_S(r), ρ_S = (D_keep + p D_G) / D, D_X = Σ_{c∈X} |s_c|².
The noise part of the per-coil gradients at the fixed point has energy ∝ ‖s_c‖², so the variance of the reshuffled
estimator relative to the full noise gradient is ≈ (p − 1) E_tail / E.
Uses the axial slice of out/slice.npz (raw whitened maps) with the global and the heart-box orderings.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
z = np.load(OUT / "slice.npz")
raw = z["raw"]
body, heart = z["body"], z["heart"]
rng = np.random.default_rng(0)
res = {}
for name, U, nv in (("global30", z["U_sig"], 30), ("heart18", z["U_heart"], 18), ("heart30", z["U_heart"], 30)):
    V = raw @ U[:, :nv].conj()
    E = np.abs(V) ** 2
    D = E.sum(-1)
    rows = []
    for k in (0, 4, 8, 12, 16):
        for d in (2, 3, 4, 6):
            tail = np.arange(k, nv)
            if d >= tail.size:
                continue
            p = int(np.ceil(tail.size / d))
            Dk = E[..., :k].sum(-1)
            mx_b, mx_h = [], []
            for _ in range(6):
                perm = rng.permutation(tail)
                for i in range(p):
                    g = perm[i * d:(i + 1) * d]
                    rho = (Dk + p * E[..., g].sum(-1)) / D
                    mx_b.append(rho[body].max())
                    mx_h.append(rho[heart].max())
            Et = E[..., k:][body].sum() / E[body].sum()
            Eth = E[..., k:][heart].sum() / E[heart].sum()
            rows.append(dict(k=k, d=d, p=p, cost=(k + d) / nv, infl_body_mean=float(np.mean(mx_b)), infl_body_max=float(np.max(mx_b)),
                             infl_heart_max=float(np.max(mx_h)), tail_E_body=float(Et), var_factor_body=float((p - 1) * Et),
                             tail_E_heart=float(Eth), var_factor_heart=float((p - 1) * Eth)))
    res[name] = rows
    print(name)
    print("  k  d  p  cost  infl(body mean/max)  infl(heart max)  tailE body/heart  (p-1)tailE body/heart")
    for r in rows:
        print(f"  {r['k']:2d} {r['d']:2d} {r['p']:2d}  {r['cost']:.2f}   {r['infl_body_mean']:6.2f} / {r['infl_body_max']:6.2f}      {r['infl_heart_max']:6.2f}"
              f"        {r['tail_E_body']:.3f} / {r['tail_E_heart']:.3f}      {r['var_factor_body']:.2f} / {r['var_factor_heart']:.2f}")
(OUT / "inflation.json").write_text(json.dumps(res, indent=1))
