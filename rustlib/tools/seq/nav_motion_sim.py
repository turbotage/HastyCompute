#!/usr/bin/env python
"""Rigid pose from the encode-window navigator under the phantom's *respiration* and an added bulk move
(research/motion_navigators.md §4): what a rigid-body navigator sees when the truth is non-rigid.

rho_r(x) = rho(x - r u(x)) (u = the phantom's respiratory field, `motion/resp_motion.hv`, r in [0,1]); optional bulk rigid
move on top. The nav signal of the 7 encode k-paths (|k_i| <= kmax) is simulated with coil-free complex noise at the
image SNR given, then the 6 pose parameters (about the centre of mass) are fitted by variable projection (a complex gain
per encode), coarse to fine in |k|. Reports pose [mm, deg] and the misfit chi2 = residual energy / noise energy per sample
(1 = the rigid model explains the data to the noise; the excess is the non-rigid part).

PYTHONPATH must contain `lz4` (hv reader).  usage: nav_motion_sim.py [--snr 20] [--out DIR]
"""
import json, sys, os
sys.path.insert(0, 'python/lib')
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from hastycompute import value_format as vf

args = sys.argv[1:]
def take(n, d):
    if n in args:
        i = args.index(n); v = args[i + 1]; del args[i:i + 2]; return v
    return d
SNR = float(take('--snr', 20))
OUT = take('--out', 'data/hasty/seq/studies/nav_crlb')
RUN, SCAN = 'data/hasty/seq/flow_thorax_v2', 'data/hasty/phantom/factory_arms/ph0001/sim_b0_2.5min/scan'
PH = 'data/hasty/phantom/factory_arms/ph0001'
GAMMA, dt, h, KMAX = 42.577e6, 10e-6, 2e-3, 250.0
sm = json.load(open(f'{RUN}/summary.json'))['best']
ie, ir = sm['i_enc'], sm['i_ro']
G = np.load(f'{RUN}/tr_grad.npy')
rho0 = (np.load(f'{SCAN}/truth_pd_water.npy') + np.load(f'{SCAN}/truth_pd_fat.npy')).astype(np.float32)
nx, ny, nz = rho0.shape
meta = json.load(open(f'{SCAN}/meta.json'))
cen = np.array(meta['presc_center_subject'])
pos = [cen[a] + (np.arange(rho0.shape[a]) - (rho0.shape[a] - 1) / 2) * h for a in range(3)]

# respiratory field on the truth grid
rm = vf.read(f'{PH}/motion/resp_motion.hv')
Bs = rm['B'].sum(0)                        # [3 comp, z, y, x] metres at r = 1
lo, hb = np.array(rm['origin_m']), rm['voxel_m']
X, Y, Z = np.meshgrid(*pos, indexing='ij')
idx = np.stack([(Z - lo[2]) / hb - 0.5, (Y - lo[1]) / hb - 0.5, (X - lo[0]) / hb - 0.5])
U = np.stack([ndimage.map_coordinates(Bs[c], idx, order=1, mode='nearest') for c in range(3)])   # c = x, y, z
print('u at heart-ish', U[:, nx // 2, ny // 2, nz // 2] * 1e3, 'max |u| mm', np.abs(U).max() * 1e3)

def warp(rho, r, T=None):
    """rho_r(x) = rho(x - r u(x)); optional rigid (R, t) about the grid centre of mass applied afterwards"""
    ci = [np.arange(n, dtype=np.float32) for n in rho.shape]
    g = np.stack(np.meshgrid(*ci, indexing='ij')).astype(np.float32)
    g -= r * U / h
    return ndimage.map_coordinates(rho, g, order=1, mode='constant')

com = [(rho0.sum(axis=tuple(j for j in range(3) if j != a)) * np.arange(rho0.shape[a])).sum() / rho0.sum() for a in range(3)]
ax = [(np.arange(rho0.shape[a]) - com[a]) * h for a in range(3)]

def s_of(rho, kp):
    ex = np.exp(-2j * np.pi * np.outer(kp[:, 0], ax[0])).astype(np.complex64)
    ey = np.exp(-2j * np.pi * np.outer(kp[:, 1], ax[1])).astype(np.complex64)
    ez = np.exp(-2j * np.pi * np.outer(kp[:, 2], ax[2])).astype(np.complex64)
    t = (ex @ rho.reshape(nx, -1).astype(np.complex64)).reshape(-1, ny, nz)
    return np.einsum('kyz,ky,kz->k', t, ey, ez, optimize=True)

# encode k paths
Ks, ENC = [], []
for e in range(G.shape[0]):
    k = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (G[e][1:] + G[e][:-1]), 0) * dt * GAMMA])[ie:ir + 1]
    m = np.all(np.abs(k) <= KMAX, 1)
    Ks.append(k[m]); ENC.append(np.full(m.sum(), e))
K = np.vstack(Ks); ENC = np.concatenate(ENC); print('nav samples', len(K))
wt = dt / 2e-6
M = 2.0e8
rho_ref = float(np.percentile(rho0[rho0 > 0.1 * rho0.max()], 90))
sigma = rho_ref * np.sqrt(M) / SNR
rng = np.random.default_rng(1)

def model(th, rho_ref_nav, kk, en):
    """rigid model of the reference nav field: s0(R^T k) exp(-2 pi i k.t), th = [t(3) m, rotvec(3) rad]"""
    t, rv = th[:3] * 1e-3, th[3:]
    R = Rotation.from_rotvec(rv).as_matrix()
    kr = kk @ R                         # R^T k (row vectors)
    return s_of(rho_ref_nav, kr) * np.exp(-2j * np.pi * kk @ t)

def fit(y, kk, en, th0, kcut_list=(30, 80, 160, 1e9)):
    th = th0.copy()
    for kc in kcut_list:
        sel = np.linalg.norm(kk, axis=1) <= kc
        if sel.sum() < 20: continue
        def res(p):
            m = model(p, rho0, kk[sel], en[sel]); yy = y[sel]; e = en[sel]
            r = np.empty(0)
            out = yy.copy()
            for q in range(G.shape[0]):
                w = e == q
                if w.any():
                    g = np.vdot(m[w], yy[w]) / np.vdot(m[w], m[w]); out[w] = yy[w] - g * m[w]
            return np.concatenate([out.real, out.imag]) / sigma * np.sqrt(wt)
        th = least_squares(res, th, method='lm', x_scale=np.array([1, 1, 1, .01, .01, .01]), diff_step=1e-3, max_nfev=60).x
    return th

def chi2(y, kk, en, th):
    m = model(th, rho0, kk, en); tot = 0.0
    for q in range(G.shape[0]):
        w = en == q
        g = np.vdot(m[w], y[w]) / np.vdot(m[w], m[w]); tot += (np.abs(y[w] - g * m[w]) ** 2).sum()
    return tot / (sigma ** 2 * len(kk))   # per complex sample: E = 1 for pure noise at the sampled density (no wt)

def apply_rigid(rho, tmm, rotdeg):
    """move the (already warped) object rigidly: x' = R (x - com) + com + t; resample rho on the grid"""
    R = Rotation.from_rotvec(np.deg2rad(rotdeg)).as_matrix()
    ci = np.stack(np.meshgrid(*[np.arange(n, dtype=np.float64) for n in rho.shape], indexing='ij')).reshape(3, -1)
    c = np.array(com)[:, None]
    src = R.T @ (ci - c - np.array(tmm)[:, None] / (h * 1e3)) + c
    return ndimage.map_coordinates(rho, src, order=1, mode='constant').reshape(rho.shape).astype(np.float32)

if '--diag' in args:
    # rotation check: exact k-space model vs grid-resampled rotation (2 deg about x), noise-free-ish
    th_t = np.array([0, 0, 0, np.deg2rad(2), 0, 0.0])
    ya = model(th_t, rho0, K, ENC) + sigma / np.sqrt(2) * (rng.standard_normal(len(K)) + 1j * rng.standard_normal(len(K)))
    print('model-generated  ->', np.round(fit(ya, K, ENC, np.zeros(6)), 3), flush=True)
    yb = s_of(apply_rigid(rho0, (0, 0, 0), (2, 0, 0)), K) + sigma / np.sqrt(2) * (rng.standard_normal(len(K)) + 1j * rng.standard_normal(len(K)))
    print('grid-resampled   ->', np.round(fit(yb, K, ENC, np.zeros(6)), 3), flush=True)
    print('grid-resampled, k<=120 ->', np.round(fit(yb, K, ENC, np.zeros(6), kcut_list=(30, 80, 120)), 3), flush=True)
    sys.exit()
rows = []
cases = [('r=0', 0.0, (0, 0, 0), (0, 0, 0)), ('r=0.25', 0.25, (0, 0, 0), (0, 0, 0)), ('r=0.5', 0.5, (0, 0, 0), (0, 0, 0)), ('r=1', 1.0, (0, 0, 0), (0, 0, 0)),
         ('bulk 5mm AP, 2deg about x', 0.0, (0, 5, 0), (2, 0, 0)), ('bulk 10mm RL', 0.0, (10, 0, 0), (0, 0, 0)), ('r=0.5 + bulk 5mm AP 2deg', 0.5, (0, 5, 0), (2, 0, 0))]
for name, r, t, rd in cases:
    rr = warp(rho0, r) if r else rho0
    if any(t) or any(rd): rr = apply_rigid(rr, t, rd)
    y = s_of(rr, K) + sigma / np.sqrt(2) * (rng.standard_normal(len(K)) + 1j * rng.standard_normal(len(K)))
    th = fit(y, K, ENC, np.zeros(6))
    c2 = chi2(y, K, ENC, th); c0 = chi2(y, K, ENC, np.zeros(6))
    rows.append({'case': name, 'pose_t_mm': np.round(th[:3], 2).tolist(), 'pose_rot_deg': np.round(np.rad2deg(th[3:]), 3).tolist(), 'chi2_rigid': round(c2, 2), 'chi2_nomotion': round(c0, 2)})
    print(rows[-1], flush=True)
json.dump({'snr': SNR, 'sigma': sigma, 'rows': rows}, open(f'{OUT}/motion_sim_snr{int(SNR)}.json', 'w'), indent=1)
