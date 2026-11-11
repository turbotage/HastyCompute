"""Re-decode velocities of a recon_p2np run from its complex images with a maximum-likelihood search over the full
unaliased range of the design (coprime dual VENC: ±3 m/s per axis), instead of the hierarchical FlowDecoder (range of
the least sensitive group, ±1.5 m/s). Then the same error statistics as vel_error.py / vel_vs_mean.py.

    redecode.py <run dir> <recon npy> <scan dir> <seqdesign run> [--vext 3.0]

Assumes axis-separable encodes (each k_e along one logical axis; true for coprime7 / referenced layouts) and the
identity prescription (logical = physical); GNL J⁻¹ is ignored (small near isocentre).
"""
import json, os, sys
import numpy as np

run, rname, scan, design = sys.argv[1:5]
vext = float(sys.argv[sys.argv.index('--vext') + 1]) if '--vext' in sys.argv else 3.0
G = 2.6752218744e8
s = json.load(open(os.path.join(design, 'summary.json'))); b = s['best']
tm = np.load(os.path.join(design, 'tr_moments.npy'))
m1 = tm[:, b['i_ro'] + b.get('i_echo', 0), 3:]
k = G * (m1[1:] - m1[0])                                  # (ne-1, 3) rad per m/s
ax = np.argmax(np.abs(k), 1); assert np.allclose(np.abs(k).sum(1), np.abs(k).max(1)), 'axis-separable encodes only'
kk = k[np.arange(len(k)), ax]
X = np.load(os.path.join(run, rname), mmap_mode='r')     # (nx, ny, nz, ne, F)
lf = np.load(os.path.join(scan, 'truth_lumen_fraction.npy'))
vt = np.load(os.path.join(scan, 'truth_vel_logical.npy'), mmap_mode='r')
N, n = lf.shape, X.shape[:3]
if N != n:
    idx = []
    for a in range(3):
        j = np.arange(N[a]); h = N[a] / n[a]
        idx.append(np.clip(np.floor((j - N[a] // 2) / h + n[a] // 2 + 0.5).astype(int), 0, n[a] - 1))
    lin = ((idx[0][:, None, None] * n[1] + idx[1][None, :, None]) * n[2] + idx[2][None, None, :]).ravel()
    cnt = np.bincount(lin, minlength=np.prod(n)).astype(float)
    down = lambda x: (np.bincount(lin, weights=x.ravel(), minlength=np.prod(n)) / np.maximum(cnt, 1)).reshape(n)
    dl = down(lf)
    vt = np.stack([np.stack([down(np.asarray(vt[..., kk_, c]) * lf) / np.maximum(dl, 1e-6) for kk_ in range(3)], -1) for c in range(vt.shape[-1])], -1)
    lf = dl
lum = lf > 0.9
grid = np.linspace(-vext, vext, int(2 * vext / 0.002) + 1)
F = X.shape[-1]; nc = vt.shape[-1]


def decode(z):  # z (nl, ne) complex → (nl, 3) velocity, ML per axis over the grid then LS refinement
    phi = -np.angle(z[:, 1:] * np.conj(z[:, :1]))            # (nl, ne-1)
    v = np.zeros((z.shape[0], 3))
    for a in range(3):
        es = np.where(ax == a)[0]
        cost = np.zeros((z.shape[0], grid.size))
        for e in es:
            cost += 1 - np.cos(phi[:, e:e + 1] - kk[e] * grid[None, :])
        v0 = grid[np.argmin(cost, 1)]
        # LS refine: unwrap each encode against v0
        num = np.zeros(z.shape[0]); den = 0.0
        for e in es:
            p = phi[:, e] + 2 * np.pi * np.round((kk[e] * v0 - phi[:, e]) / (2 * np.pi))
            num += kk[e] * p; den += kk[e] ** 2
        v[:, a] = num / den
    return v


def hier(z):  # the Rust FlowDecoder's hierarchical scheme for this layout (first stage = least sensitive group)
    phi = -np.angle(z[:, 1:] * np.conj(z[:, :1]))
    v = np.zeros((z.shape[0], 3))
    order = np.argsort(np.abs(kk)); kmin = np.abs(kk[order[0]])
    first = [e for e in range(len(kk)) if abs(abs(kk[e]) - kmin) < 1e-6 * kmin]
    for a in range(3):
        es1 = [e for e in first if ax[e] == a]; v[:, a] = sum(phi[:, e] / kk[e] for e in es1) / len(es1)
        es = np.where(ax == a)[0]; num = 0; den = 0
        for e in es:
            p = phi[:, e] + 2 * np.pi * np.round((kk[e] * v[:, a] - phi[:, e]) / (2 * np.pi)); num = num + kk[e] * p; den += kk[e] ** 2
        v[:, a] = num / den
    return v


print(f"k_e (rad per m/s) {np.round(kk, 3)} on axes {ax}; ML range ±{vext} m/s; lumen voxels {lum.sum()}")
res = {}
for f in range(F):
    z = np.stack([np.asarray(X[..., e, f])[lum] for e in range(X.shape[3])], 1)
    vm, vh = decode(z), hier(z)
    t = np.asarray(vt[..., f % nc])[lum]
    em, eh = np.linalg.norm(vm - t, axis=1), np.linalg.norm(vh - t, axis=1)
    sl = [(vm[:, a] * t[:, a]).sum() / (t[:, a] ** 2).sum() for a in range(3)]
    res[f] = dict(rms_ml=float(np.sqrt((em ** 2).mean())), rms_hier=float(np.sqrt((eh ** 2).mean())), slope_ml=[float(x) for x in sl],
                  frac_changed=float((np.abs(vm - vh).max(1) > 0.5).mean()), truth_rms=float(np.sqrt((t ** 2).sum(1).mean())))
    if f % 20 in (0, 3, 5, 12):
        r = res[f]
        print(f"frame {f:2d}: rms ML {r['rms_ml']:.3f} vs hierarchical {r['rms_hier']:.3f} m/s (truth rms speed {r['truth_rms']:.2f}); slope ML {np.round(sl, 2)}; voxels re-decoded {100 * r['frac_changed']:.1f} %")
tot_ml = np.sqrt(np.mean([r['rms_ml'] ** 2 for r in res.values()])); tot_h = np.sqrt(np.mean([r['rms_hier'] ** 2 for r in res.values()]))
tr = np.sqrt(np.mean([r['truth_rms'] ** 2 for r in res.values()]))
print(f"all frames: rms ML {tot_ml:.3f} (rel {tot_ml / tr:.3f}) vs hierarchical {tot_h:.3f} (rel {tot_h / tr:.3f}) m/s")
json.dump(res, open(os.path.join(run, 'diag_redecode.json'), 'w'), indent=1)
