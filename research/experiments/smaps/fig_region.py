"""Figure + intensity check for region_error.py output (slice z=20)."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import RUN, store
e1, orth, nz, body, dist = np.load("region_err_z20.npy")
nz, body = nz > 0, body > 0
img = store.load_any(os.path.join(RUN, "warm_full.hv"))[:, :, 20, :]
r = np.sqrt(np.sum(np.abs(img) ** 2, -1))
med = np.median(r[body & nz])
zb = body & ~nz
print(f"|rho| at body voxels with zero ESPIRiT map: median {np.median(r[zb]) / med:.3f} x body median; nonzero-map body p5 {np.percentile(r[body & nz], 5) / med:.3f}")
print(f"zero-map body voxels deeper than 50 mm: {(zb & (dist > 50)).sum()}, their |rho| median {np.median(r[zb & (dist > 50)]) / med:.3f}")
f, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, (t, v, kw) in zip(ax, [("|pooled image| z=20", r.T, dict(cmap="gray")), ("ESPIRiT map nonzero (white) / body (grey)", (nz * 1.0 + body * 0.5).T, dict(cmap="gray")),
                               ("1-|<s^,s>| (nonzero maps)", np.where(nz, e1, np.nan).T, dict(vmin=0, vmax=0.2, cmap="magma")), ("dist to skin mm", np.where(body, dist, np.nan).T, dict(cmap="viridis"))]):
    im = a.imshow(v, origin="lower", **kw); a.set_title(t, fontsize=9); plt.colorbar(im, ax=a, fraction=0.03)
plt.tight_layout(); plt.savefig("region_err_z20.png", dpi=80)
