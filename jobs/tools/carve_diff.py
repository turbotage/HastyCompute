"""Compare the carved heart labels (phantom atlas) with the uncarved p0 labels: changes per label, and near given points."""
import sys
from pathlib import Path
import numpy as np
REPO = Path.cwd()
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf
a = vf.read(str(REPO / "data/hasty/phantom/p0/tissue_0.5mm_heart.hv"))
b = vf.read(sys.argv[1])
names = list(a["tissues"])
La, Lb = np.asarray(a["labels"]), np.asarray(b["labels"])
h = float(a["voxel_m"]); lo = np.asarray(a["origin_m"], float)
ch = [names.index(c) for c in ("blood_LA", "blood_LV", "blood_RA", "blood_RV", "valve", "myocardium") if c in names]
d = La != Lb
print("changed voxels", int(d.sum()), f"{d.sum() * h**3 * 1e6:.2f} mL")
for c in ch:
    m = (La == c) & d
    tgt = np.bincount(Lb[m], minlength=len(names))
    top = np.argsort(-tgt)[:3]
    print(f"  {names[c]:12s} lost {m.sum() * h**3 * 1e6:6.2f} mL → " + ", ".join(f"{names[t]} {tgt[t] * h**3 * 1e6:.2f}" for t in top if tgt[t]))
for s in sys.argv[2:]:
    p = np.array([float(x) for x in s.split(",")]) * 1e-3
    idx = np.argwhere(d)
    X = idx[:, ::-1] * h + lo + h / 2
    r = np.linalg.norm(X - p, axis=1)
    for rad in (10e-3, 20e-3, 30e-3):
        m = r < rad
        fr = La[tuple(idx[m].T)]
        print(f"  within {rad*1e3:.0f} mm of {s}: {m.sum() * h**3 * 1e6:.3f} mL changed; from " + ", ".join(f"{names[t]} {np.sum(fr == t) * h**3 * 1e6:.3f}" for t in np.unique(fr)))
