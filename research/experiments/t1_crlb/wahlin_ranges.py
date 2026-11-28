"""Concentration ranges at the measured time points of the Wahlin 2026 intrathecal data (dataset1). usage: python -I wahlin_ranges.py <dir>
Conversions to T1 assume baseline T1_0 (CSF 4.3 s, cortex 1.5 s, WM 0.85 s, subcortical 1.1 s) and r1 = 3.4 /mM/s; unit of C assumed mM (checked below)."""
import sys
import numpy as np
import scipy.io as sio
ds = sio.loadmat(sys.argv[1] + "/dataset1.mat", squeeze_me=True, struct_as_record=False)["dataset1"]
P = ds[0]
print("it[0:3], it[-1] (h):", P.it[:3], P.it[-1], " points:", P.points, " times at points (h):", P.it[P.points.astype(int) - 1] if P.points.max() <= len(P.it) else "?")
print("injectionTimes", P.injectionTimes)
R1 = 3.4
T10 = dict(csf=4.3, ctx=1.5, wm=0.85, sub=1.1)
def stats(name, key, t10):
    allv = []
    for s in ds:
        pts = s.points.astype(int) - 1
        allv.append(np.asarray(getattr(s, key))[:, pts])
    a = np.concatenate([x.reshape(x.shape[0], -1) for x in allv], axis=0)   # ROIs from all subjects x points
    # per point
    npts = len(P.points)
    print(f"\n{name}: ROI-timepoints {a.shape}")
    for j in range(npts):
        col = np.concatenate([np.asarray(getattr(s, key))[:, s.points.astype(int)[j] - 1] for s in ds])
        q = np.percentile(col, [1, 10, 50, 90, 99, 100])
        t1 = 1.0 / (1.0 / t10 + R1 * q)
        print(f"  point {j}: C pct 1/10/50/90/99/max = " + " ".join(f"{v:7.4f}" for v in q) + " | T1 (s) = " + " ".join(f"{v:5.2f}" for v in t1))
stats("CSF next to cortex ROIs", "concCsf_Ctx", T10["csf"])
stats("cortex", "concCtx", T10["ctx"])
stats("white matter", "concWm", T10["wm"])
stats("subcortical", "concSubcor", T10["sub"])
