"""Murray-consistency floor for the atlas radii, from a network solution (iterate with `network_flow.py`).

    python -I murray_floor.py <export_dir> <flow_dir> [--f 0.75]

Atlas radii are sometimes far too thin for the flow their subtree carries (thin stretches, fragmented lung trees).
Murray's law r = c Q^{1/3}; c per class from the great vessels (systemic arteries c = 2.35 mm, pulmonary arteries
2.15 mm, veins 2.6 mm per (mL/s)^{1/3}). Each edge's radius is raised to at least f · c · |Q̄|^{1/3} (mean flow of the
edge). The atlas radii are kept as `radius_atlas` in graph.npz, so repeated runs floor from them (no compounding).
"""
import json
import sys
from pathlib import Path

import numpy as np

E, F = Path(sys.argv[1]), Path(sys.argv[2])
f = float(sys.argv[sys.argv.index("--f") + 1]) if "--f" in sys.argv else 0.75
g = dict(np.load(E / "graph.npz"))
gj = json.loads((E / "graph.json").read_text())
q = np.load(F / "network_flow.npz")["q"]
ratl = g.get("radius_atlas", g["radius"]).astype(np.float64)
eoff, ecomp = g["edge_offsets"], g["edge_comp"]
comps = {c["id"]: c for c in gj["components"]}
C = {"systemic_arterial": 2.35e-3, "pulmonary_arterial": 2.15e-3, "systemic_venous": 2.6e-3, "pulmonary_venous": 2.6e-3}
r = ratl.copy()
changed = {}
for k in range(len(eoff) - 1):
    c = C.get(comps[int(ecomp[k])]["class"])
    if c is None:
        continue
    a, b = int(eoff[k]), int(eoff[k + 1])
    qm = abs(float(q[a:b, 0].real.mean()))
    floor = f * c * (qm / 1e-6) ** (1 / 3)
    if floor > ratl[a:b].min():
        r[a:b] = np.maximum(ratl[a:b], floor)
        nm = gj["edge_names"][k]
        changed[nm] = max(changed.get(nm, 1.0), float(floor / ratl[a:b].min()))
g["radius_atlas"] = ratl.astype(np.float32)
g["radius"] = r.astype(np.float32)
np.savez_compressed(E / "graph.npz", **g)
gj["murray_floor"] = {"f": f, "c_m": C, "raised": {k: round(v, 2) for k, v in sorted(changed.items(), key=lambda kv: -kv[1])}}
(E / "graph.json").write_text(json.dumps(gj, indent=1))
print(f"Murray floor (f {f}): {len(changed)} vessels raised; largest: {[(k, round(v, 2)) for k, v in sorted(changed.items(), key=lambda kv: -kv[1])[:10]]}")
