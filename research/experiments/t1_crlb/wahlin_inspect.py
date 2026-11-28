"""Inspect Wahlin 2026 repo data (dataset1 = intrathecal): concentration ranges in CSF and tissue. usage: python -I wahlin_inspect.py <dir>"""
import sys
import numpy as np
import scipy.io as sio
d = sys.argv[1]
try:
    m = sio.loadmat(d + "/dataset1.mat", squeeze_me=True, struct_as_record=False)
    ds = m["dataset1"]
    print(type(ds), getattr(ds, "shape", None))
    p = ds[0]
    for f in p._fieldnames:
        v = getattr(p, f)
        print(f, getattr(v, "shape", v if np.isscalar(v) else type(v)), (np.asarray(v).dtype if hasattr(v, "dtype") else ""))
except NotImplementedError as e:
    print("v7.3", e)
