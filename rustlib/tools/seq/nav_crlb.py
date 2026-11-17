#!/usr/bin/env python
"""Cramér–Rao bound of rigid-body pose from the navigator ADC of flow_thorax_v2 (research/motion_navigators.md §3).

Model (coil-free, unit-sensitivity single channel with the coil-combined noise sigma; conservative, a multi-coil
model only adds information): s(k) = sum_x rho(x) exp(-2 pi i k.x). A rigid move (translation t, rotation about the
body's centre of mass by small angle r) gives, exactly in k-space,
    d s / d t_a = -2 pi i k_a s(k),        d s / d r_a = 2 pi i (e_a x k) . S(k),   S_j(k) = sum_x rho(x) x_j exp(-2 pi i k.x)
so only s and the three first-moment images are needed. Nuisance per TR (profiled out by Schur complement): global
phase, global amplitude, and a linear phase in time (B0 drift / frequency). Samples with any |k_i| > kmax (the 2 mm
truth grid's Nyquist) are dropped: the bound ignores the information above kmax (lower bound on information).

usage: nav_crlb.py [--run data/hasty/seq/flow_thorax_v2] [--scan <sim scan dir>] [--out DIR]
"""
import json, sys, os
import numpy as np
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)

GAMMA = 42.577e6  # Hz/T
args = sys.argv[1:]
def take(n, d):
    if n in args:
        i = args.index(n); v = args[i + 1]; del args[i:i + 2]; return v
    return d
RUN = take('--run', 'data/hasty/seq/flow_thorax_v2')
SCAN = take('--scan', 'data/hasty/phantom/factory_arms/ph0001/sim_b0_2.5min/scan')
OUT = take('--out', 'data/hasty/seq/studies/nav_crlb')
os.makedirs(OUT, exist_ok=True)

sm = json.load(open(f'{RUN}/summary.json'))['best']
dt, dwell = 10e-6, 2e-6
ie, ir, it, n = sm['i_enc'], sm['i_ro'], sm['i_tail'], sm['n_tr']
TR = n * dt
KMAX = 250.0
h = 2e-3
tr_grad = load_any(f'{RUN}/tr_grad.npy')          # [7][n+1][3] T/m, physical axes
ne = tr_grad.shape[0]

rho = (load_any(f'{SCAN}/truth_pd_water.npy') + load_any(f'{SCAN}/truth_pd_fat.npy')).astype(np.float32)
nx, ny, nz = rho.shape
com = [(rho.sum(axis=tuple(j for j in range(3) if j != a)) * np.arange(rho.shape[a])).sum() / rho.sum() for a in range(3)]
ax = [(np.arange(rho.shape[a]) - com[a]) * h for a in range(3)]
print('grid', rho.shape, 'com idx', np.round(com, 1), 'sum rho', rho.sum())

def fields(kp):
    """s(k) and S_j(k) for k [K,3] in cycles/m (separable contraction)"""
    out = []
    for w in (None, 0, 1, 2):
        r = rho if w is None else rho * ax[w].astype(np.float32).reshape([-1 if i == w else 1 for i in range(3)])
        ex = np.exp(-2j * np.pi * np.outer(kp[:, 0], ax[0])).astype(np.complex64)
        ey = np.exp(-2j * np.pi * np.outer(kp[:, 1], ax[1])).astype(np.complex64)
        ez = np.exp(-2j * np.pi * np.outer(kp[:, 2], ax[2])).astype(np.complex64)
        t = ex @ r.reshape(nx, -1).astype(np.complex64)
        t = t.reshape(-1, ny, nz)
        out.append(np.einsum('kyz,ky,kz->k', t, ey, ez, optimize=True))
    return out

def derivs(kp, tk):
    s, sx, sy, sz = fields(kp)
    S = np.stack([sx, sy, sz], 1)
    d = np.zeros((len(kp), 6 + 3), np.complex128)
    for a in range(3):
        d[:, a] = -2j * np.pi * kp[:, a] * s
        e = np.zeros(3); e[a] = 1
        d[:, 3 + a] = 2j * np.pi * (np.cross(e, kp) * S).sum(1)
    d[:, 6] = 1j * s                     # phase
    d[:, 7] = s                          # amplitude
    d[:, 8] = -2j * np.pi * tk * s       # frequency (Hz) drift
    return d, s

def fisher(d, sigma, wt):
    return wt * 2.0 / sigma ** 2 * np.real(d.conj().T @ d)

def profile(F):
    A, B, C = F[:6, :6], F[:6, 6:], F[6:, 6:]
    return A - B @ np.linalg.solve(C, B.T)

def kpath(e, i0, i1):
    G = tr_grad[e]
    # piecewise-linear G on the raster: trapezoid rule
    k = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (G[1:] + G[:-1]), 0) * dt * GAMMA])
    t = np.arange(len(k)) * dt
    sl = slice(i0, i1 + 1)
    kk = k[sl]
    return kk, t[sl] - t[ir]  # time relative to the echo

def keep(kp):
    return np.all(np.abs(kp) <= KMAX, 1)

res = {'sigma_ref': {}, 'parts': {}}
Fp = {}   # profiled Fisher per unit sigma^2=1 (multiply by 1/sigma^2 later): store with sigma=1
wt = (dt / dwell)  # raster -> actual samples
for part, (i0, i1) in {'encode': (ie, ir), 'tail': (it, n)}.items():
    Fe = []
    for e in range(ne):
        kk, tk = kpath(e, i0, i1)
        m = keep(kk)
        d, s = derivs(kk[m], tk[m])
        F = profile(fisher(d, 1.0, wt))
        Fe.append(F)
        if part == 'encode': print(part, e, 'samples kept', m.sum(), '/', len(m), 'diag', np.round(np.sqrt(np.diag(F)[:3]), 1))
    Fp[part] = Fe
# readout (rotated base spokes): average over the stored rotations
ks = load_any(f'{RUN}/k_spokes.npy')
Fr = []
for r in range(ks.shape[0]):
    kk = ks[r]
    m = keep(kk)
    d, s = derivs(kk[m], np.arange(len(kk))[m] * dt)
    Fr.append(profile(fisher(d, 1.0, wt)))
Fp['readout'] = Fr

# noise: image SNR (2 mm voxel, 10-min scan, M recon samples) -> per-sample sigma:  SNR = rho_ref sqrt(M) / sigma
M = 2.0e8
rho_ref = float(np.percentile(rho[rho > 0.1 * rho.max()], 90))
print('rho_ref', rho_ref)
LEVER = 0.1
sc = np.diag([1, 1, 1, LEVER, LEVER, LEVER])  # rotation (rad) -> displacement (m) at 100 mm
tab = []
for snr in (10, 20, 40):
    sigma = rho_ref * np.sqrt(M) / snr
    for name, parts in {'encode': ['encode'], 'tail': ['tail'], 'encode+tail': ['encode', 'tail'], 'readout (rotations avg)': ['readout']}.items():
        # information per second: encodes cycle (mean over the 7), one TR each
        if name.startswith('readout'):
            F_tr = np.mean(Fp['readout'], 0)
            per_s = F_tr / TR
        else:
            F_set = sum(sum(Fp[p]) for p in parts)    # one set of 7 consecutive TRs (all 7 encodes + their tails)
            per_s = F_set / (ne * TR)
        for T in (0.1, 1.0, 10.0):
            Fm = per_s * T / sigma ** 2
            cov = np.linalg.inv(Fm)
            sd = np.sqrt(np.diag(cov)) * np.array([1e3] * 3 + [LEVER * 1e3] * 3)   # mm (rot at 100 mm lever)
            tab.append({'snr': snr, 'part': name, 'T_s': T, 'sd_mm': [round(float(x), 4) for x in sd]})
json.dump({'rho_ref': rho_ref, 'M': M, 'TR_ms': TR * 1e3, 'lever_m': LEVER, 'table': tab}, open(f'{OUT}/crlb.json', 'w'), indent=1)
for r in tab:
    if r['snr'] == 20:
        print(f"{r['part']:26s} T={r['T_s']:5.1f}s  sd [mm] t={r['sd_mm'][:3]}  rot@100mm={r['sd_mm'][3:]}")
