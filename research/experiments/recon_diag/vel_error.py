"""Structure of the velocity error of a recon_p2np run against the simulator truth (native grid, or a coarse grid of
the same FOV: truth box-averaged, lumen-weighted).

    vel_error.py <run dir> <vel npy name> <scan dir> [frames...]

Lumen = truth lumen fraction > 0.9. Per frame: slope of recon vs truth per component (least squares through 0),
error split by truth speed, lumen fraction, recon magnitude; fraction of voxels with |err| > 1 m/s (wrap-like).
Frames are resp-major (frame f ↔ cardiac f % 20). Writes <run>/diag_vel_error.json.
"""
import json, os, sys
import numpy as np

run, vname, scan = sys.argv[1:4]
fr_sel = [int(x) for x in sys.argv[4:]] or None
v = np.load(os.path.join(run, vname), mmap_mode='r')          # (nx, ny, nz, 3, F)
vt = np.load(os.path.join(scan, 'truth_vel_logical.npy'), mmap_mode='r')  # (nx, ny, nz, 3, 20)
lf = np.load(os.path.join(scan, 'truth_lumen_fraction.npy'))
if v.shape[:3] != lf.shape:
    # coarse grid of the same FOV (grid index i ↦ (i − n/2) h): box-average the native truth into each coarse voxel
    N, n = lf.shape, v.shape[:3]
    idx = []
    for a_ in range(3):
        j = np.arange(N[a_]); h = N[a_] / n[a_]
        idx.append(np.clip(np.floor((j - N[a_] // 2) / h + n[a_] // 2 + 0.5).astype(int), 0, n[a_] - 1))
    lin = (idx[0][:, None, None] * n[1] + idx[1][None, :, None]) * n[2] + idx[2][None, None, :]
    lin = lin.ravel(); cnt = np.bincount(lin, minlength=np.prod(n)).astype(float)
    down = lambda x: (np.bincount(lin, weights=x.ravel(), minlength=np.prod(n)) / np.maximum(cnt, 1)).reshape(n)
    vt_full = vt
    vt = np.stack([np.stack([down(np.asarray(vt_full[..., k, c]) * lf) / np.maximum(down(lf), 1e-6) for k in range(3)], -1)
                   for c in range(vt_full.shape[-1])], -1)
    lf = down(lf)
    print(f'coarse grid {n}: truth box-averaged (lumen-weighted velocity) from {N}')
lum = lf > 0.9
F = v.shape[-1]; nc = vt.shape[-1]
rec = [f for f in os.listdir(run) if f.startswith('recon_') and f.endswith('.npy')]
X = np.load(os.path.join(run, rec[0]), mmap_mode='r') if rec else None   # (nx, ny, nz, 7, F)
out = {}
for f in (fr_sel or range(F)):
    c = f % nc
    a = np.asarray(v[..., f])[lum]; b = np.asarray(vt[..., c])[lum]
    e = a - b
    sp = np.linalg.norm(b, axis=1); en = np.linalg.norm(e, axis=1)
    slope = [float((a[:, k] * b[:, k]).sum() / (b[:, k] ** 2).sum()) for k in range(3)]
    corr = [float(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in range(3)]
    bins = [0, 0.25, 0.5, 1.0, 1.5, 9]
    by_speed = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (sp >= lo) & (sp < hi)
        if m.any():
            by_speed.append(dict(speed=f'{lo}-{hi}', n=int(m.sum()), rms_err=float(np.sqrt((en[m] ** 2).mean())), mean_speed=float(sp[m].mean()),
                                 recon_speed=float(np.linalg.norm(a[m], axis=1).mean())))
    lfv = lf[lum]
    by_lf = [dict(lf=f'{lo}-{hi}', n=int(((lfv >= lo) & (lfv < hi)).sum()), rms_err=float(np.sqrt((en[(lfv >= lo) & (lfv < hi)] ** 2).mean())))
             for lo, hi in [(0.9, 0.99), (0.99, 1.01)] if ((lfv >= lo) & (lfv < hi)).any()]
    d = dict(frame=f, cardiac=c, n=int(lum.sum()), rms=float(np.sqrt((en ** 2).mean())), truth_rms_speed=float(np.sqrt((sp ** 2).mean())),
             slope=slope, corr=corr, frac_err_gt_1=float((en > 1.0).mean()), err_p50_p90_p99=[float(np.percentile(en, q)) for q in (50, 90, 99)],
             by_speed=by_speed, by_lumen_fraction=by_lf)
    if X is not None:
        mag = np.abs(np.asarray(X[..., 0, f]))[lum]
        q = np.quantile(mag, [0, 0.25, 0.5, 0.75, 1.0])
        d['by_magnitude_quartile'] = [float(np.sqrt((en[(mag >= q[i]) & (mag <= q[i + 1])] ** 2).mean())) for i in range(4)]
    out[f] = d
    print(f"frame {f:2d} (card {c:2d}): rms {d['rms']:.3f} m/s vs truth rms speed {d['truth_rms_speed']:.3f}; slope {np.round(slope, 2)}, corr {np.round(corr, 2)}; "
          f"|e|>1 m/s {100 * d['frac_err_gt_1']:.1f} %; p50/p90/p99 {np.round(d['err_p50_p90_p99'], 2)}")
    print('   by speed:', [(s['speed'], s['n'], round(s['rms_err'], 2), round(s['mean_speed'], 2), round(s['recon_speed'], 2)) for s in by_speed])
    if 'by_magnitude_quartile' in d:
        print('   rms by |x| quartile (low→high):', np.round(d['by_magnitude_quartile'], 2), ' by lumen fraction:', [(x['lf'], x['n'], round(x['rms_err'], 2)) for x in by_lf])
json.dump(out, open(os.path.join(run, 'diag_vel_error.json'), 'w'), indent=1)
