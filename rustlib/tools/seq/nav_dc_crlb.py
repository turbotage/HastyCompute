#!/usr/bin/env python
"""CRLB of rigid pose from per-coil DC samples alone (FID-navigator style, research/motion_navigators.md §3).
d s_c(0)/d t_a = sum rho d_a S_c ;  d s_c(0)/d r_a = sum rho (e_a x (x - c)) . grad S_c ; nuisance: global phase + amplitude per sample.
Coils: the phantom's true (Sum|S|^2 = 1 normalised) maps; noise as in nav_crlb.py (image SNR 20 at 2 mm, 10 min, M = 2e8)."""
import numpy as np, json
SCAN = 'data/hasty/phantom/factory_arms/ph0001/sim_b0_2.5min/scan'
h = 2e-3
rho = (np.load(f'{SCAN}/truth_pd_water.npy') + np.load(f'{SCAN}/truth_pd_fat.npy')).astype(np.float32)
sm = np.load(f'{SCAN}/smaps_true.npy', mmap_mode='r')
nc = sm.shape[-1]
com = [(rho.sum(axis=tuple(j for j in range(3) if j != a)) * np.arange(rho.shape[a])).sum() / rho.sum() for a in range(3)]
ax = [((np.arange(rho.shape[a]) - com[a]) * h).astype(np.float32) for a in range(3)]
X = np.meshgrid(*ax, indexing='ij')
D = np.zeros((nc, 8), np.complex128)
for c in range(nc):
    S = np.asarray(sm[..., c])
    g = [np.gradient(S, h, axis=a) for a in range(3)]
    s0 = (rho * S).sum()
    for a in range(3):
        D[c, a] = (rho * g[a]).sum()
    for a, (b1, b2) in enumerate([(1, 2), (2, 0), (0, 1)]):      # e_a x r = (r_b2 e_b1 - r_b1 e_b2) pattern
        D[c, 3 + a] = (rho * (X[b1] * g[b2] - X[b2] * g[b1]) * (1 if True else 1)).sum()
    D[c, 6] = 1j * s0
    D[c, 7] = s0
M = 2.0e8
rho_ref = float(np.percentile(rho[rho > 0.1 * rho.max()], 90))
sigma = rho_ref * np.sqrt(M) / 20
F = 2 / sigma ** 2 * np.real(D.conj().T @ D)
A, B, C = F[:6, :6], F[:6, 6:], F[6:, 6:]
Fp = A - B @ np.linalg.solve(C, B.T)
for name, rate in [('flow: 1 DC/TR at TE (71 /s)', 1 / 14.1e-3), ('flow: + rewinder DC (142 /s)', 2 / 14.1e-3), ('B0 train: 25 crossings/TR (900 /s)', 25 / 27.75e-3)]:
    for T in (0.1, 1.0, 10.0):
        cov = np.linalg.inv(Fp * rate * T)
        sd = np.sqrt(np.diag(cov)) * np.array([1e3] * 3 + [100.0] * 3)
        print(f'{name:38s} T={T:5.1f}s  t[mm]={np.round(sd[:3], 3)}  rot@100mm[mm]={np.round(sd[3:], 3)}')
print('single-sample DC |s| per coil (min,max)', np.abs(D[:, 7]).min(), np.abs(D[:, 7]).max(), 'sigma', sigma)
