"""Cost (full-gradient equivalents) to reach given distances to the fixed point, per arm, from the cached toy runs.

    rustlib/.venv/bin/python -I research/experiments/coil_sketch/summarize.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
R = json.loads((OUT / "results_global.json").read_text())["results"]


def first_cost(tr, thr):
    i = np.flatnonzero(tr[:, 1] <= thr)
    return tr[i[0], 0] if i.size else np.inf


rows = {}
for f in sorted(glob.glob(str(OUT / "runs" / "global_*_s[0-9].npz"))):
    name = re.match(r".*/global_(.*)_s\d\.npz", f).group(1)
    z = np.load(f)
    rows.setdefault(name, []).append((z["t0"], z["t1"]))
thr_s = (0.1, 0.05, 0.03, 0.02)
thr_p = (0.5, 0.3, 0.2, 0.1)
out = {}
for name, runs in rows.items():
    th = thr_p if name.startswith("pnp") else thr_s
    # an encode pair reaches the threshold when both encodes do: max over encodes, mean over seeds
    cs = [[max(first_cost(t0, t), first_cost(t1, t)) for t in th] for t0, t1 in runs]
    c = np.mean(cs, 0)
    out[name] = dict(thr=list(th), cost=c.tolist())
base_s = out["ncg_full"]["cost"]
base_p = out["pnp_full"]["cost"] if "pnp_full" in out else None
print("smooth objective (Moreau-wavelet), cost to reach rel. distance to x* " + " / ".join(map(str, thr_s)) + "  [speed-up vs ncg_full]")
for name in sorted(out, key=lambda n: (n.startswith("pnp"), n)):
    c = out[name]["cost"]
    base = base_p if name.startswith("pnp") else base_s
    if name.startswith("pnp") and name == sorted(n for n in out if n.startswith("pnp"))[0]:
        print("PnP form, thresholds " + " / ".join(map(str, thr_p)) + "  [vs pnp_full at the same α]")
    sp = ["{:.2f}".format(b / v) if np.isfinite(v) and np.isfinite(b) else "-" for b, v in zip(base, c)]
    r = R.get(name, {})
    v = r.get("v@60" if not name.startswith("pnp") else "v@120", np.nan)
    print(f"  {name:26s} " + " ".join(f"{x:6.1f}" if np.isfinite(x) else "     ∞" for x in c) + "  [" + " ".join(sp) + f"]  v_fp {v:.2f} cm/s")
(OUT / "summary.json").write_text(json.dumps(out, indent=1))
