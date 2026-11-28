"""Physiology check of a phantom's network flow (research/flow_phantom.md Tier B5): named-vessel mean flows against
adult resting ranges, and the peak cross-sectional mean velocity per vessel class.

    rustlib/.venv/bin/python -I rustlib/tools/phantom/flow_check.py <phantom_or_export_dir> [--json out.json]

Reads <dir>/graph.npz / graph.json and <dir>/flow/network_flow.npz. A vessel's flow is that of its edge carrying the
largest mean |Q| (the most proximal piece carries the whole vessel's flow); `.l` / `.r` pairs are checked each.
Ranges are rounded textbook values for an adult at rest (cardiac output ~5 L/min), scaled with the phantom's cardiac
output; they flag gross errors (a sinus draining nothing, a lung taking two thirds), not 10 % differences.
Peak velocity: |Q(t)| / (π r²) per centreline sample over the cycle (the network's cross-sectional mean, not the
centre-line peak), against a class limit. Exit code 1 when anything is out of range.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

A = sys.argv[1:]
D = Path(A[0])
G = load_any(D / "graph.npz")
GJ = json.loads((D / "graph.json").read_text())
NET = load_any(D / "flow" / "network_flow.npz")
NJ = json.loads((D / "flow" / "network_flow.json").read_text())
names = [n.split(": ", 1)[1] if n.startswith(("connector: ", "bridge: ", "crossing: ", "join: ")) else n for n in GJ["edge_names"]]
eoff = np.asarray(G["edge_offsets"])
R = np.asarray(G["radius"], np.float64)
ec = np.asarray(G["edge_comp"])
cls_of = {c["id"]: c["class"] for c in GJ["components"]}
q_h = np.asarray(NET["q"])
W = np.asarray(NET["omega"])
period = float(NET["period"])
emq = np.abs(np.asarray(NET["edge_mean_q"])) * 1e6  # mL/s
CO = float(NJ["co_l_min"]) / 60 * 1e3  # mL/s
sc = CO / (5.0 / 60 * 1e3)

# mean flow [mL/s] at 5 L/min; pairs ".l"/".r" each
RANGES = {
    "Ascending aorta": (0.97 * 83.3, 1.03 * 83.3),
    "Brachiocephalic trunk": (9, 18), "Left common carotid artery": (4.5, 9), "Right common carotid artery": (4.5, 9),
    "Left subclavian artery": (3, 8), "Internal carotid artery": (3.5, 6.5), "External carotid artery": (1.2, 3.5),
    "Vertebral artery": (1.0, 2.6), "Basilar artery": (1.8, 3.4), "Middle cerebral artery (M1-segment)": (1.8, 3.6),
    "Anterior cerebral artery": (0.8, 1.9), "Posterior cerebral artery": (0.6, 1.4),
    "Coeliac trunk": (6, 12), "Superior mesenteric artery": (5, 10), "Left renal artery": (6, 12), "Right renal artery": (6, 12),
    "Common iliac artery": (4.5, 10), "Femoral artery": (3.5, 8),
    "Left coronary artery": (1.2, 3.5), "Right coronary artery": (0.8, 2.5),
    "Superior vena cava": (0.22 * 83.3, 0.40 * 83.3), "Inferior vena cava (thoracic part)": (0.55 * 83.3, 0.78 * 83.3),
    "Hepatic portal vein": (11, 22), "Internal jugular vein": (4, 10), "Left renal vein": (6, 12), "Right renal vein": (6, 12),
    "Azygos vein": (0.8, 4), "Superior sagittal sinus": (3.5, 7.5), "Transverse sinus": (2.5, 6), "Straight sinus": (0.7, 2.2),
}
VLIM = {"systemic_arterial": 1.6, "pulmonary_arterial": 1.2, "systemic_venous": 0.7, "portal_venous": 0.7, "pulmonary_venous": 0.9}  # m/s

by_name = defaultdict(list)
for k, n in enumerate(names):
    by_name[n].append(k)
rows, bad = [], []
for key, (lo, hi) in RANGES.items():
    lo, hi = lo * sc, hi * sc
    hits = [n for n in by_name if n == key or (n.rsplit(".", 1)[0] == key and n.rsplit(".", 1)[-1] in ("l", "r"))]
    if not hits:
        rows.append((key, None, lo, hi, "missing"))
        bad.append(key)
        continue
    for n in sorted(hits):
        q = float(emq[by_name[n]].max())
        st = "ok" if lo <= q <= hi else ("LOW" if q < lo else "HIGH")
        rows.append((n, q, lo, hi, st))
        if st != "ok":
            bad.append(n)
# per lung: arterial in = venous out
pa = {s: float(emq[by_name.get(f"{s} pulmonary artery", [])].max()) if by_name.get(f"{s} pulmonary artery") else 0.0 for s in ("Left", "Right")}
pv = {s: sum(float(emq[by_name[f"{s} {u} pulmonary vein"]].max()) for u in ("superior", "inferior") if f"{s} {u} pulmonary vein" in by_name) for s in ("Left", "Right")}

# peak cross-sectional velocity per class
tt = np.linspace(0, period, 64, endpoint=False)
E = np.exp(1j * W[1:, None] * tt[None, :])
vmax = defaultdict(lambda: (0.0, ""))
es = np.searchsorted(eoff, np.arange(len(R)), side="right") - 1
for s0 in range(0, len(R), 50000):
    sl = slice(s0, min(s0 + 50000, len(R)))
    qt = np.real(q_h[sl, :1] + 2 * (q_h[sl, 1:] @ E))
    v = np.abs(qt).max(1) / (np.pi * R[sl] ** 2)
    for c in set(ec[es[sl]].tolist()):
        cl = cls_of.get(c, "other")
        m = ec[es[sl]] == c
        if m.any():
            j = int(np.argmax(np.where(m, v, -1)))
            if v[j] > vmax[cl][0]:
                vmax[cl] = (float(v[j]), names[es[sl][j]])
print(f"cardiac output {CO:.1f} mL/s (ranges scaled ×{sc:.2f})")
for n, q, lo, hi, st in rows:
    print(f"  {n[:44]:44s} {('-' if q is None else f'{q:7.2f}'):>7s} mL/s  [{lo:6.2f}, {hi:6.2f}]  {st}")
print(f"  lungs: arterial in L {pa['Left']:.1f} / R {pa['Right']:.1f} mL/s (right share {pa['Right'] / max(pa['Left'] + pa['Right'], 1e-9):.2f}); "
      f"venous out L {pv['Left']:.1f} / R {pv['Right']:.1f} (right share {pv['Right'] / max(pv['Left'] + pv['Right'], 1e-9):.2f})")
for cl, (v, n) in sorted(vmax.items()):
    lim = VLIM.get(cl)
    st = "ok" if lim is None or v <= lim else "HIGH"
    if st != "ok":
        bad.append(f"velocity {cl}")
    print(f"  peak velocity {cl:20s} {v:5.2f} m/s at {n[:40]}" + (f" (limit {lim})  {st}" if lim else ""))
out = {"co_ml_s": CO, "vessels": {n: {"q_ml_s": q, "range": [lo, hi], "status": st} for n, q, lo, hi, st in rows},
       "lungs": {"arterial": pa, "venous": pv}, "peak_velocity": {k: {"m_s": v, "at": n} for k, (v, n) in vmax.items()}, "out_of_range": bad}
if "--json" in A:
    Path(A[A.index("--json") + 1]).write_text(json.dumps(out, indent=1))
print(f"out of range: {len(bad)}: {bad}")
sys.exit(1 if bad else 0)
