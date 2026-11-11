"""Does a reconstruction track each frame's truth or the time-averaged truth? (coarse or native grid)

    vel_vs_mean.py <run dir> <vel npy> <scan dir>

Lumen-weighted box-averaged truth as in vel_error.py. Per cardiac frame: rms error vs the frame truth and vs the
time-mean truth; the fraction of the truth's temporal deviation (v_f − v̄) the recon reproduces: slope and corr of
recon deviation vs truth deviation.
"""
import os, sys
import numpy as np
sys.argv_saved = list(sys.argv)
run, vname, scan = sys.argv[1:4]
v = np.load(os.path.join(run, vname), mmap_mode='r'); vt = np.load(os.path.join(scan, 'truth_vel_logical.npy'), mmap_mode='r')
lf = np.load(os.path.join(scan, 'truth_lumen_fraction.npy'))
N, n = lf.shape, v.shape[:3]
if N != n:
    idx = []
    for a in range(3):
        j = np.arange(N[a]); h = N[a] / n[a]
        idx.append(np.clip(np.floor((j - N[a] // 2) / h + n[a] // 2 + 0.5).astype(int), 0, n[a] - 1))
    lin = ((idx[0][:, None, None] * n[1] + idx[1][None, :, None]) * n[2] + idx[2][None, None, :]).ravel()
    cnt = np.bincount(lin, minlength=np.prod(n)).astype(float)
    down = lambda x: (np.bincount(lin, weights=x.ravel(), minlength=np.prod(n)) / np.maximum(cnt, 1)).reshape(n)
    dl = down(lf)
    vt = np.stack([np.stack([down(np.asarray(vt[..., k, c]) * lf) / np.maximum(dl, 1e-6) for k in range(3)], -1) for c in range(vt.shape[-1])], -1)
    lf = dl
lum = lf > 0.9
nc = vt.shape[-1]; F = v.shape[-1]
T = np.stack([np.asarray(vt[..., c])[lum] for c in range(nc)], 0)          # (nc, nl, 3)
Tm = T.mean(0)
for rb in range(F // nc):
    R = np.stack([np.asarray(v[..., rb * nc + c])[lum] for c in range(nc)], 0)
    Rm = R.mean(0)
    e_frame = np.sqrt(((R - T) ** 2).sum(-1).mean()); e_mean = np.sqrt(((R - Tm[None]) ** 2).sum(-1).mean())
    dR, dT = (R - Rm[None]).ravel(), (T - Tm[None]).ravel()
    sl = (dR * dT).sum() / (dT ** 2).sum(); co = np.corrcoef(dR, dT)[0, 1]
    slm = (Rm.ravel() * Tm.ravel()).sum() / (Tm.ravel() ** 2).sum(); com = np.corrcoef(Rm.ravel(), Tm.ravel())[0, 1]
    print(f"resp bin {rb}: rms vs frame truth {e_frame:.3f}, vs time-mean truth {e_mean:.3f} m/s; temporal deviation: slope {sl:.2f} corr {co:.2f} "
          f"(recon dev rms {np.sqrt((dR**2).mean()):.3f} vs truth {np.sqrt((dT**2).mean()):.3f}); time-mean: slope {slm:.2f} corr {com:.2f}")
