#!/usr/bin/env python3
"""Check of Proposition prop:strength (the scalar part of a metric is a change of strength), same multi-set toy as
metric_fixedpoint_v2/v3.py (joint Haar soft threshold across 4 sets).

At the reference limit (P_I = I, best lambda_I) take the correction c = x_I - z_I, split by the active set into
c_par (tangent T) and c_perp. For each metric P (per-set scalars, shared circulant):
  q = <c_par, alpha_P P c_par> / |c_par|^2       (scalar part of the metric on the in-tangent correction)
  predicted strength lambda_P = q lambda_I / alpha_I
  defects, relative to q |c_par|:
    non-scalar  |Pi (alpha_P P - q) c_par|        (orthogonal to c_par)
    leak        |Pi alpha_P P c_perp|
    shrinkage   |J_D(z_I) c_par - c_par| / |c_par|  (Lambda c_par = 0 at first order for the joint threshold)
Then the metric's fixed point at lambda_P: |x_P - x_I| / |x_I| and the error to the truth, against the fixed point at
the common strength lambda_I and the best-tuned strength of the grid (metric_fixedpoint_v3.json).
Writes results/metric_strength.json.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import metric_fixedpoint as mf  # noqa: E402
import metric_fixedpoint_v2 as v2  # noqa: E402


def dot(a, b):
    return float(np.sum(a * b))


def main():
    t0 = time.time()
    P = v2.setup()
    xs = P["xs"]
    mets = {k: mf.make_metric(P, k) for k in ["I", "set", "circ"]}
    C = mets["circ"]
    best = json.load(open(HERE / "results" / "metric_fixedpoint_v3.json"))["best"]
    lamI = best["I"]["lam"]
    aI = mets["I"]["alpha"]
    xI, zI, _, r = v2.fixed_point(P, mets["I"], lamI, C, accel_ok=True)
    print(f"reference: lambda_I {lamI:.2e}, alpha_I {aI:.3e}, err to truth {mf.rel(xI, xs):.4f} (resid {r:.0e})", flush=True)
    c = xI - zI
    m = mf.active_mask(zI, lamI)
    cpar = mf.proj(m, c)
    cperp = c - cpar
    shrink = np.linalg.norm(mf.jvp(zI, cpar, lamI) - cpar) / np.linalg.norm(cpar)
    print(f"  |c_par| / |c| {np.linalg.norm(cpar) / np.linalg.norm(c):.3f}; |J_D c_par - c_par| / |c_par| {shrink:.1e}", flush=True)
    res = dict(lam_I=lamI, alpha_I=aI, err_I=mf.rel(xI, xs), shrink=shrink, metrics={})
    for k in ["set", "circ"]:
        Mt = mets[k]
        Pc = Mt["apply"](cpar)
        q = dot(cpar, Pc) / dot(cpar, cpar)
        lamP = q * lamI / aI
        nonscalar = np.linalg.norm(mf.proj(m, Pc) - q * cpar) / (q * np.linalg.norm(cpar))
        leak = np.linalg.norm(mf.proj(m, Mt["apply"](cperp))) / (q * np.linalg.norm(cpar))
        xP, _, _, rp = v2.fixed_point(P, Mt, lamP, C, accel_ok=k == "set")
        d_pred = mf.rel(xP, xI)
        out = dict(q=q, lam_pred=lamP, lam_best=best[k]["lam"], err_best=best[k]["err"], nonscalar=nonscalar, leak=leak,
                   disp_pred=d_pred, err_pred=mf.rel(xP, xs), resid=rp)
        xC, _, _, rc = v2.fixed_point(P, Mt, lamI, C, accel_ok=k == "set")
        out.update(disp_common=mf.rel(xC, xI), err_common=mf.rel(xC, xs))
        res["metrics"][k] = out
        print(f"  {k}: q / alpha_I {q / aI:.3f} -> predicted lambda {lamP:.2e} (grid best {best[k]['lam']:.2e}); defects rel. to q|c_par|: "
              f"non-scalar {nonscalar:.3f}, leak {leak:.3f}", flush=True)
        print(f"    |x_P - x_I| / |x_I|: at the predicted lambda {d_pred:.2e} (err to truth {out['err_pred']:.4f}, resid {rp:.0e}); "
              f"at the common lambda_I {out['disp_common']:.2e} (err {out['err_common']:.4f}, resid {rc:.0e}); grid best err {best[k]['err']:.4f}", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "metric_strength.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
