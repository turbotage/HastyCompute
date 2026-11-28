"""Murray-consistency floor for the atlas radii, from a network solution (iterate with `network_flow.py`).

    python -I murray_floor.py <export_dir> <flow_dir> [--f 0.75]

Atlas radii are sometimes far too thin for the flow their subtree carries (thin stretches, fragmented lung trees).
Murray's law r = c Q^{1/3}; c per class from the great vessels (systemic arteries c = 2.35 mm, pulmonary arteries
2.15 mm, veins 2.6 mm per (mL/s)^{1/3}). Each edge's radius is raised to at least f · c · |Q̄|^{1/3} (mean flow of the
edge). The atlas radii are kept as `radius_atlas` in graph.npz, so repeated runs floor from them (no compounding).

Velocity floor (2026-11-26, default; `--no-vfloor`: Murray only). Murray's cube law holds for the small vessels; the
large ones are not WSS-optimal (measured flow ∝ r^2…2.5 from the aorta to its main branches) and what bounds them in a
healthy adult is the section-mean velocity at peak flow. The 0.75 Murray floor still let the right pulmonary artery
taper to 6.1 mm with the whole right-lung flow (1.9 m/s). So the radius is also raised to √(Q̂ / (π v_cap)), Q̂ the
edge's peak |Q(t)|, with v_cap per class a healthy resting peak section-mean velocity (systemic arteries 1.2,
pulmonary arteries 0.95, systemic veins 0.5, pulmonary veins 0.6 m/s): after the clearance solver's shrink floor
(0.9 r, ×1.23 velocity) they stay within flow_check.py's limits (1.6 / 1.2 / 0.7 / 0.9). Lesions (stenoses,
aneurysms, diffeomorphic variants) come after the floors, so they are not undone.
"""
import json
import sys
from pathlib import Path

import numpy as np
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)

E, F = Path(sys.argv[1]), Path(sys.argv[2])
f = float(sys.argv[sys.argv.index("--f") + 1]) if "--f" in sys.argv else 0.75
g = dict(load_any(E / "graph.npz"))
gj = json.loads((E / "graph.json").read_text())
q = np.asarray(load_any(F / "network_flow.npz")["q"])
ratl = g.get("radius_atlas", g["radius"]).astype(np.float64)
eoff, ecomp = g["edge_offsets"], g["edge_comp"]
comps = {c["id"]: c for c in gj["components"]}
C = {"systemic_arterial": 2.35e-3, "pulmonary_arterial": 2.15e-3, "systemic_venous": 2.6e-3, "portal_venous": 2.6e-3, "pulmonary_venous": 2.6e-3}
VCAP = {"systemic_arterial": 1.2, "pulmonary_arterial": 0.95, "systemic_venous": 0.5, "portal_venous": 0.5, "pulmonary_venous": 0.6}  # m/s
VFLOOR = "--no-vfloor" not in sys.argv
NET = load_any(F / "network_flow.npz")
if VFLOOR:
    _tt = np.linspace(0, float(NET["period"]), 64, endpoint=False)
    _E = np.exp(1j * np.asarray(NET["omega"])[1:, None] * _tt[None, :])
r = ratl.copy()
changed = {}
FIXED = np.asarray(g.get("variant_fixed", np.zeros(len(eoff) - 1, np.int8)))  # calibres set by a vessel variant
for k in range(len(eoff) - 1):
    c = C.get(comps[int(ecomp[k])]["class"])
    if c is None or FIXED[k]:
        continue
    a, b = int(eoff[k]), int(eoff[k + 1])
    qm = abs(float(q[a:b, 0].real.mean()))
    floor = f * c * (qm / 1e-6) ** (1 / 3)
    if VFLOOR:
        qe = q[a:b].mean(0)
        qpk = float(np.abs(np.real(qe[0] + 2 * (qe[1:] @ _E))).max())
        floor = max(floor, float(np.sqrt(qpk / (np.pi * VCAP[comps[int(ecomp[k])]["class"]]))))
    if floor > ratl[a:b].min():
        r[a:b] = np.maximum(ratl[a:b], floor)
        nm = gj["edge_names"][k]
        changed[nm] = max(changed.get(nm, 1.0), float(floor / ratl[a:b].min()))
# radius continuity after the floor (2026-11-28; --taper-mm 0: off, as before): the floor is one value per edge, so
# where it raises an edge and not its continuation the lumen steps (export_full19: 317 joints > 5 %) — taper the joints
# (radius_taper.joint_taper: degree-2 joints and the parent's through-line at branch nodes; vessel-variant calibres
# untouched). Repeated runs floor from radius_atlas, then taper: no compounding.
TAPER = float(sys.argv[sys.argv.index("--taper-mm") + 1]) * 1e-3 if "--taper-mm" in sys.argv else 15e-3
ntap = 0
if TAPER > 0:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from radius_taper import joint_taper  # noqa: E402

    r, _trep = joint_taper(g["points"], r, eoff, g["edge_a"], g["edge_b"], len(g["node_pos"]), length=TAPER, fixed=FIXED)
    ntap = len(_trep)
g["radius_atlas"] = ratl.astype(np.float32)
g["radius"] = r.astype(np.float32)
np.savez_compressed(E / "graph.npz", **g)
gj["murray_floor"] = {"f": f, "c_m": C, "v_cap_m_s": VCAP if VFLOOR else None, "taper_mm": TAPER * 1e3, "tapered_joints": ntap, "raised": {k: round(v, 2) for k, v in sorted(changed.items(), key=lambda kv: -kv[1])}}
(E / "graph.json").write_text(json.dumps(gj, indent=1))
print(f"Murray floor (f {f}): {len(changed)} vessels raised, {ntap} joints tapered; largest: {[(k, round(v, 2)) for k, v in sorted(changed.items(), key=lambda kv: -kv[1])[:10]]}")
