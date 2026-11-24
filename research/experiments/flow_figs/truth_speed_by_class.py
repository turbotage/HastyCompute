"""Speed statistics of the 1 mm fine truth per blood class and frame (blood fraction > 0.5): 99th percentile and
max |v|, and the location / label / waveform of the fastest pulmonary voxel at end-diastole (frame 0).
    systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
        rustlib/.venv/bin/python -I research/experiments/flow_figs/truth_speed_by_class.py
"""
import sys

import numpy as np

sys.path.insert(0, "python/lib")
from hastycompute import store  # noqa: E402

S = "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/truth_fine/truth4d.hv"
t = store.vf.open_lazy(S)
names = t["tissues"]
vel, blood, labels = (store.LazyArray(t[k]) for k in ("vel", "blood", "labels"))
tc = store.LazyArray(t["tc"]).read()
cls = [n for n in names if n.startswith("blood_")]
print("frame t_ms " + " ".join(f"{c[6:]:>18s}" for c in cls) + "   (p99 / max m/s)")
q = None
wave = []
for f in range(len(tc)):
    v = vel[f].astype(np.float32)
    sp = np.sqrt((v * v).sum(0))
    del v
    b = blood[f] > 127
    lab = labels[f]
    row = []
    for c in cls:
        m = b & (lab == names.index(c))
        s = sp[m]
        row.append(f"{np.percentile(s, 99):6.2f} /{s.max():6.2f}    " if s.size else " " * 18)
    print(f"{f:5d} {1e3 * tc[f]:4.0f} " + " ".join(row))
    if f == 0:
        pul = b & np.isin(lab, [names.index("blood_pulm_artery"), names.index("blood_pulm_vein")])
        q = np.unravel_index(np.argmax(np.where(pul, sp, 0)), sp.shape)
        print(f"   fastest pulmonary voxel at frame 0: (z, y, x) = {q}, {names[lab[q]]}, |v| {sp[q]:.2f} m/s")
    wave.append(float(sp[q]))
print("waveform at that voxel (m/s):", " ".join(f"{w:.2f}" for w in wave))
