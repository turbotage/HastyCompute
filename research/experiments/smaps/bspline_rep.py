"""How many parameters do the (whitened, compressed) true maps need? Weighted LS fit of a tensor cubic B-spline
(knot spacing h) to the true maps over the object (truth support, diagnostic only) and the resulting map error
1-|<s_fit,s>| (per voxel, normalised) in the object, near the skin and deep. Same basis as hasty_recon::smapfit
(no penalty here: pure representation error). Also: the same fit applied to the oracle ESPIRiT slice maps is not
possible (slices only), so this bounds the basis error only."""
import numpy as np, os, time
from scipy.ndimage import map_coordinates, binary_fill_holes, distance_transform_edt
from common import SCAN, DIMS, meta, recon_positions, true_maps_cc


def axis_basis(n, vox, h):
    span = int(np.floor((n - 1) * vox / h)); k = span + 4
    u = np.minimum(np.arange(n) * vox / h, span + 1 - 1e-9); j = np.minimum(np.floor(u).astype(int), span); t = u - j
    w = np.stack([(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6], 1)
    B = np.zeros((n, k))
    for d in range(4):
        B[np.arange(n), j + d] = w[:, d]
    return B


m = meta(); x, vox = recon_positions(m); res = m["res"][0]; td = m["recon_dims"]
g = [np.clip(x[..., k] / res + td[k] // 2, 0, td[k] - 1 - 1e-9) for k in range(3)]
samp = lambda f: map_coordinates(np.nan_to_num(np.asarray(f, np.float32)), g, order=1, mode="nearest")
pd = samp(np.load(os.path.join(SCAN, "truth_pd_water.npy"))) + samp(np.load(os.path.join(SCAN, "truth_pd_fat.npy")))
r2 = samp(np.load(os.path.join(SCAN, "truth_r2s.npy")))
sig = pd * np.exp(-r2 * 1.6e-3)
obj = sig > 0.05 * np.median(sig[sig > 0])
body = np.stack([binary_fill_holes(obj[:, :, z]) for z in range(DIMS[2])], 2)
dist = np.stack([distance_transform_edt(body[:, :, z], sampling=[1e3 * v for v in vox[:2]]) for z in range(DIMS[2])], 2)
tru = true_maps_cc(); nc = tru.shape[-1]
tn = (tru / np.maximum(np.linalg.norm(tru, axis=-1, keepdims=True), 1e-30)).astype(np.complex64)
w = body.astype(float)  # weight: the body (lungs included)
def B_apply(c, Bs):  # c [kx,ky,kz,nc] -> [nx,ny,nz,nc]
    c = np.tensordot(Bs[0], c, axes=([1], [0]))
    c = np.moveaxis(np.tensordot(Bs[1], c, axes=([1], [1])), 0, 1)
    return np.moveaxis(np.tensordot(Bs[2], c, axes=([1], [2])), 0, 2)


def BT_apply(v, Bs):
    v = np.tensordot(Bs[0].T, v, axes=([1], [0]))
    v = np.moveaxis(np.tensordot(Bs[1].T, v, axes=([1], [1])), 0, 1)
    return np.moveaxis(np.tensordot(Bs[2].T, v, axes=([1], [2])), 0, 2)


for h in [0.016, 0.024, 0.032, 0.048, 0.064]:
    t0 = time.time()
    Bs = [axis_basis(DIMS[a], vox[a], h).astype(np.float32) for a in range(3)]
    K = np.prod([b.shape[1] for b in Bs])
    wv = w[..., None].astype(np.float32)
    A = lambda c: BT_apply(wv * B_apply(c, Bs), Bs) + 1e-7 * c
    b = BT_apply(wv * tn, Bs)
    c = np.zeros_like(b); r = b.copy(); p = r.copy(); rr = np.vdot(r, r).real; r0 = rr
    for it in range(60):
        Ap = A(p); al = rr / np.vdot(p, Ap).real
        c += al * p; r -= al * Ap; rn = np.vdot(r, r).real
        if rn < 1e-12 * r0: break
        p = r + (rn / rr) * p; rr = rn
    fit = B_apply(c, Bs)
    e1 = 1 - np.abs(np.sum(np.conj(fit) * tn, -1)) / np.maximum(np.linalg.norm(fit, axis=-1), 1e-30)
    o = obj
    print(f"h {h * 1e3:.0f} mm: {K} coefficients/coil (CG {it + 1}, rel res {np.sqrt(rn / r0):.1e}); 1-|<s_fit,s>| object mean {e1[o].mean():.2e} p99 {np.percentile(e1[o], 99):.2e}; "
          f"skin<=20mm {e1[o & (dist <= 20)].mean():.2e}; deep>50mm {e1[o & (dist > 50)].mean():.2e}; lungs/holes {e1[body & ~o].mean():.2e} ({time.time() - t0:.0f}s)", flush=True)
