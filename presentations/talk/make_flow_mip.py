"""Speed MIPs from the simulated ph0001 phantom (CFD-driven truth4d.hv). Run with rustlib/.venv python."""
import sys, numpy as np
sys.path.insert(0, 'python/lib')
from hastycompute import value_format as vf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
T = vf.open_lazy('data/hasty/phantom/factory_arms/ph0001/simblood_lib/truth4d.hv')
V = T['vel']; B = T['blood']
F = V.shape[0]
best, bs = 0, -1
for f in range(0, F, 4):
    v = np.asarray(V[f], dtype=np.float32)
    s = np.sqrt((v**2).sum(0)).mean()
    if s > bs: bs, best = s, f
print('peak frame', best)
v = np.asarray(V[best], dtype=np.float32)           # [3,Z,Y,X]
bl = np.asarray(B[best]) > 40                        # blood voxels
sp = np.sqrt((v**2).sum(0)) * bl
np.savez_compressed('/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/cb025116-ccf3-4585-a796-b2f667b63bab/scratchpad/mip.npz', frame=best)
fig, ax = plt.subplots(1, 3, figsize=(10, 4.2), facecolor='black')
vmax = np.percentile(sp[sp > 0], 99.5)
for a, (axis, name) in zip(ax[:2], [(1, 'coronal'), (2, 'sagittal')]):
    m = sp.max(axis)
    a.imshow(m, origin='lower', cmap='magma', vmin=0, vmax=vmax); a.set_title(name + ' speed MIP', color='w', fontsize=10); a.axis('off')
# direction-coded coronal MIP: colour = |v_x|,|v_y|,|v_z|, brightness = speed
idx = sp.argmax(1)                                   # y of max speed per (z,x)
zz, xx = np.meshgrid(np.arange(sp.shape[0]), np.arange(sp.shape[2]), indexing='ij')
comp = np.abs(v[:, zz, idx, xx]); sm = np.maximum(sp.max(1), 1e-9)
rgb = np.clip((comp / sm).transpose(1, 2, 0) ** 1.0, 0, 1) * np.clip(sp.max(1) / vmax, 0, 1)[..., None] ** 0.7
ax[2].imshow(rgb, origin='lower'); ax[2].set_title('coronal MIP, colour = direction (R:x G:y B:z)', color='w', fontsize=8); ax[2].axis('off')
plt.tight_layout(); plt.savefig('presentations/talk/figs/flow_mip.png', dpi=170, facecolor='black')
