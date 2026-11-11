"""Robust coprime decoding: ML over ±vext per axis, but an aliased solution (outside the first stage's ±range) is
accepted only if its phase cost beats the in-range optimum by a margin. Unweighted cost Σ_e (1 − cos(φ_e − k_e v));
the magnitude-weighted variant scales each voxel's cost by its |z_0|² / median (an SNR² proxy), so the margin is in
SNR-normalised units. Native or coarse grid, lumen > 0.9; frames given or all.

    decode_margin.py <run dir> <recon npy> <scan dir> <seqdesign run>
"""
import json, os, sys
import numpy as np
run, rname, scan, design = sys.argv[1:5]
G = 2.6752218744e8
s = json.load(open(os.path.join(design, 'summary.json'))); b = s['best']
m1 = np.load(os.path.join(design, 'tr_moments.npy'))[:, b['i_ro'] + b.get('i_echo', 0), 3:]
k = G * (m1[1:] - m1[0]); ax = np.argmax(np.abs(k), 1); kk = k[np.arange(len(k)), ax]
X = np.load(os.path.join(run, rname), mmap_mode='r')
lf = np.load(os.path.join(scan, 'truth_lumen_fraction.npy')); lum = lf > 0.9
vt = np.load(os.path.join(scan, 'truth_vel_logical.npy'), mmap_mode='r')
assert X.shape[:3] == lf.shape
grid = np.linspace(-3, 3, 3001); kmin = np.abs(kk).min(); r1 = np.pi / kmin
inr = np.abs(grid) <= r1 + 1e-9
F = X.shape[-1]; nc = vt.shape[-1]
frames = [3, 5, 12, 23, 25, 32]
margins = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 1e9]
res = {m: {'u': [], 'w': []} for m in margins}
for f in frames:
    z = np.stack([np.asarray(X[..., e, f])[lum] for e in range(X.shape[3])], 1)
    phi = -np.angle(z[:, 1:] * np.conj(z[:, :1]))
    mag = np.abs(z[:, 0]); wv = (mag / np.median(mag)) ** 2
    t = np.asarray(vt[..., f % nc])[lum]
    for m in margins:
        for kind in ('u', 'w'):
            v = np.zeros_like(t)
            for a in range(3):
                es = np.where(ax == a)[0]
                cost = sum(1 - np.cos(phi[:, e:e + 1] - kk[e] * grid[None]) for e in es)
                if kind == 'w':
                    cost = cost * wv[:, None]
                ia = np.argmin(cost, 1); ii = np.argmin(np.where(inr[None], cost, np.inf), 1)
                ca, ci = cost[np.arange(len(ia)), ia], cost[np.arange(len(ii)), ii]
                v0 = np.where(ci - ca > m, grid[ia], grid[ii])
                v[:, a] = sum(kk[e] * (phi[:, e] + 2 * np.pi * np.round((kk[e] * v0 - phi[:, e]) / (2 * np.pi))) for e in es) / sum(kk[e] ** 2 for e in es)
            res[m][kind].append(np.sqrt(((v - t) ** 2).sum(1).mean()))
print('frames', frames, '(cardiac 3, 5 systole; 12 diastole; both resp bins)')
for m in margins:
    lab = 'in-range only (= hierarchical ±1.5)' if m > 1e8 else f'margin {m}'
    print(f"{lab:38s} unweighted rms {np.round(res[m]['u'], 3)} mean {np.mean(res[m]['u']):.3f} | |z0|²-weighted {np.round(res[m]['w'], 3)} mean {np.mean(res[m]['w']):.3f}")
