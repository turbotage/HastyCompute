"""Q4 toy: deterministic compression with the heart-box Gram vs the global Gram (coil_cost_reduction.md §5).

    rustlib/.venv/bin/python -I research/experiments/coil_sketch/box_toy.py

Same slice problem as toy.py, same raw 32-coil data and noise realisation, compressed with different U / counts.
Every variant reconstructs the whole FOV (out-of-box signal modelled, nothing subtracted). λ = toy.py's global λ.
Reports, against the 30-coil global fixed point x*_30 and the truth, inside the heart bbox: magnitude-weighted error,
velocity RMS (signal-weighted, cm/s), and the cost ratio n/30.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
lam = json.loads((OUT / "lambda_global.json").read_text())["lam"]
ref = np.load(OUT / "runs" / "global_ref.npz")["x"]


def solve(P, tag):
    xs = []
    for e in range(2):  # cached per encode (resumable)
        f = OUT / "runs" / f"box_{tag}_e{e}.npy"
        if f.exists():
            xs.append(np.load(f))
            continue
        x, _ = toy.run_ncg(P, e, toy.Tracker(P, e, None, 10 ** 9), maxit=250)
        x, _ = toy.run_ncg(P, e, toy.Tracker(P, e, None, 10 ** 9), x=x, maxit=250)
        np.save(f, x)
        xs.append(x)
    return xs


rows = []
for coils in sys.argv[1:] or ("global:30", "global:18", "heart:11", "heart:18", "heart:22"):
    tag = coils.replace(":", "")
    # "ctrl" = global:30 solved like the others (500 NCG iterations): the solver-error floor of the comparison
    P = toy.Problem(coils="global:30" if coils == "ctrl" else coils)
    P.lam = lam
    xs = list(ref) if coils == "global:30" else solve(P, tag)
    h = P.heart
    m_fp = toy.metrics(P, xs, list(ref))
    m_tr = toy.metrics(P, xs, P.truth)
    eh_fp = np.mean([np.linalg.norm((xs[e] - ref[e])[h]) / np.linalg.norm(ref[e][h]) for e in range(2)])
    eh_tr = np.mean([np.linalg.norm((xs[e] - P.truth[e])[h]) / np.linalg.norm(P.truth[e][h]) for e in range(2)])
    ebody_tr = np.mean([np.linalg.norm((xs[e] - P.truth[e])[P.body]) / np.linalg.norm(P.truth[e][P.body]) for e in range(2)])
    r = dict(coils=coils, cost=P.nc / 30, heart_vs_fp30=eh_fp, heart_vs_truth=eh_tr, body_vs_truth=ebody_tr,
             v_vs_fp30=m_fp["v_fp"], v_vs_truth=m_tr["v_fp"])
    rows.append(r)
    print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)
bf = OUT / "box_toy.json"
old = {r["coils"]: r for r in json.loads(bf.read_text())} if bf.exists() else {}
old.update({r["coils"]: r for r in rows})
bf.write_text(json.dumps(list(old.values()), indent=1))
