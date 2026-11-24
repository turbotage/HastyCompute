import sys, os, numpy as np
sys.path.insert(0, "python/lib")
from hastycompute import store
from scipy import ndimage
R = "data/hasty/phantom/factory_arms/ph0001/precond_compare/fx_dry_bfo/"
S = "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/"
am = np.asarray(store.load_any(R + [f for f in os.listdir(R) if f.startswith("air_map") and f.endswith(".hv")][0]))
w = am[..., 0]
dr = np.array(w.shape); vr = np.array([3.96875, 4.0, 3.8])
pw = np.asarray(store.load_any(S + "truth_pd_water.npy")); pf = np.asarray(store.load_any(S + "truth_pd_fat.npy"))
pd = np.nan_to_num(pw) + np.nan_to_num(pf)
dn = np.array(pd.shape); vn = 2.0
pdb = ndimage.uniform_filter(pd, size=2)
idx = np.indices(w.shape).reshape(3, -1).T.astype(float)
nat = (idx - dr // 2) * vr / vn + dn // 2
g = ndimage.map_coordinates(pdb, nat.T, order=1, mode="constant").reshape(w.shape)
med = np.median(g[g > 0]); rel = g / med
z = w >= 0.5
flag = z & (rel > 0.05)
depth = ndimage.distance_transform_edt(z, sampling=vr)  # mm to the nearest non-zeroed voxel
d = depth[flag]
print(f"zeroed {z.sum()}, flagged {flag.sum()} (approximate grid mapping, GNL ignored)")
for lo, hi in [(0, 4.5), (4.5, 8.5), (8.5, 12.5), (12.5, 1e9)]:
    m = (d > lo) & (d <= hi)
    print(f"  depth {lo:>4}–{hi:<6} mm: {m.sum():5d} voxels, median rel PD {np.median(rel[flag][m]) if m.any() else float('nan'):.3f}")
# where along head–foot (z) and is it outside the body (lungs vs exterior)? body = filled support of truth
body = ndimage.binary_fill_holes(rel > 0.05)
inside = flag & ndimage.binary_erosion(body, iterations=2)
print(f"  flagged inside the (eroded, filled) true body: {inside.sum()}; at its surface: {(flag & ~ndimage.binary_erosion(body, iterations=2)).sum()}")
c = np.argwhere(body).mean(0)
p = np.argwhere(flag & ~ndimage.binary_erosion(body, iterations=2))
rel_pos = (p - c) * vr
for a, name in enumerate(["x (left-right)", "y (ant-post)", "z (head-foot)"]):
    h, e = np.histogram(rel_pos[:, a], bins=6)
    print(f"  surface flagged along {name}: " + ", ".join(f"[{e[i]:+.0f},{e[i+1]:+.0f}] mm: {h[i]}" for i in range(6)))
