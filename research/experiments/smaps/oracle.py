"""Inverse-crime ESPIRiT controls (CPU): calibration = exact Cartesian k-space of s_c * rho (true maps, truth image),
so any map error is ESPIRiT + calibration-choice error, not data-path error. Variants switch one factor at a time.

  nice python oracle.py --tag A0                       # current settings: 24^3 on the FOV, kernel 6, eig crop 0.8
  nice python oracle.py --tag B --fext 1.5             # calibration on a 1.5x FOV (same k extent: 36^3 points, kernel 9 -> --kernel)
  nice python oracle.py --tag C --cal 32               # larger calibration
  nice python oracle.py --tag D --rate                 # ratemap (decay + field) phase over the calibration samples' times
Prints region metrics on axial slices; the truth is used only as the yardstick and to synthesise the oracle data."""
import argparse, os, time, json
import numpy as np
from scipy.ndimage import map_coordinates, binary_fill_holes, distance_transform_edt
from common import SCAN, DIMS, meta, recon_positions, true_maps_cc, ip_err, store, RUN
from espirit_py import centred_fft, crop_centre, calibrate, maps_on_slices

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="A0"); ap.add_argument("--cal", type=int, default=24); ap.add_argument("--kernel", type=int, default=6)
ap.add_argument("--fext", type=float, default=1.0); ap.add_argument("--eig", type=float, default=0.8); ap.add_argument("--rate", action="store_true")
ap.add_argument("--tbins", type=int, default=12); ap.add_argument("--mix-return", action="store_true", help="--rate: calibration = count-weighted mix of the out half and the return half at the same k (the pre-2026-11-23 full-ADC calibration)"); ap.add_argument("--rows", type=int, default=8000); ap.add_argument("--zs", default="2,20,40,57")
ap.add_argument("--cg", type=int, default=0, help="emulate the CG truncation filter of calib_from_noncart (iterations; needs psf.npy)")
a = ap.parse_args()
t0 = time.time()
m = meta(); x, vox = recon_positions(m)
res = m["res"][0]; td = m["recon_dims"]
g = [np.clip(x[..., k] / res + td[k] // 2, 0, td[k] - 1 - 1e-9) for k in range(3)]
samp = lambda f: map_coordinates(np.nan_to_num(np.asarray(f, np.float32)), g, order=1, mode="nearest")
pd = samp(np.load(os.path.join(SCAN, "truth_pd_water.npy"))) + samp(np.load(os.path.join(SCAN, "truth_pd_fat.npy")))
r2 = samp(np.load(os.path.join(SCAN, "truth_r2s.npy")))
T0 = 1.6e-3
rho = pd * np.exp(-r2 * T0)
tru = true_maps_cc()
nc = tru.shape[-1]
cal = [int(round(a.cal * a.fext / 2)) * 2] * 3
cal = [min(cal[k], int(round(DIMS[k] * a.fext / 2)) * 2) for k in range(3)]
pad = [int(round(DIMS[k] * a.fext / 2)) * 2 for k in range(3)]
f = tuple(pad[k] / DIMS[k] for k in range(3))


def calib_of(img):  # img [x,y,z,nc] on DIMS -> centred calibration on the (extended) FOV
    if pad != list(DIMS):
        o = [(pad[k] - DIMS[k]) // 2 + (DIMS[k] // 2 - DIMS[k] // 2) for k in range(3)]
        big = np.zeros(tuple(pad) + (img.shape[-1],), np.complex64)
        # keep the voxel at index N//2 at index P//2 (centred conventions)
        o = [pad[k] // 2 - DIMS[k] // 2 for k in range(3)]
        big[o[0]:o[0] + DIMS[0], o[1]:o[1] + DIMS[1], o[2]:o[2] + DIMS[2]] = img
        img = big
    return crop_centre(centred_fft(img), cal)


if not a.rate:
    calib = calib_of(tru * rho[..., None])
else:
    # sample time of each calibration point from the out-half trajectory: tau(|k|) of the first encode-0 readouts
    S = store.load_any(os.path.join(SCAN, "samples.hv"), lazy=True)
    rd = [np.asarray(S[i]) for i in range(0, 7 * 40, 7)]
    fov = [td[k] * res * f[k] for k in range(3)]
    kk = np.meshgrid(*[2 * np.pi * (np.arange(cal[k]) - cal[k] // 2) / fov[k] for k in range(3)], indexing="ij")
    kmag = np.sqrt(sum(v ** 2 for v in kk))
    edges = np.linspace(0, kmag.max() * 1.0001, 64); mid = 0.5 * (edges[1:] + edges[:-1])
    def tcurve(part):
        kt = np.concatenate([r[part] for r in rd]); kr = np.linalg.norm(kt[:, :3], axis=1); tau = kt[:, 3]
        idx = np.digitize(kr, edges)
        tb = np.array([np.median(tau[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(edges))])
        cnt = np.array([np.sum(idx == i) for i in range(1, len(edges))], float)
        ok = ~np.isnan(tb)
        return np.interp(kmag, mid[ok], tb[ok]), np.interp(kmag, mid, cnt)
    tcal, c1 = tcurve(slice(0, 1991))
    tret, c2 = tcurve(slice(1991, 3990))
    wmix = c2 / np.maximum(c1 + c2, 1e-9)
    if a.mix_return:
        print(f"return-half times {tret.min() * 1e3:.3f}..{tret.max() * 1e3:.3f} ms, return share of the samples: median {np.median(wmix):.2f}")
    print(f"calibration sample times {tcal.min() * 1e3:.3f}..{tcal.max() * 1e3:.3f} ms")
    fppm = samp(np.load(os.path.join(SCAN, "field_ppm_recon.npy")))
    w = 2 * np.pi * 42.577478e6 * m["b0"] * 1e-6 * fppm
    def calib_at(tcal_):
        tb_edges = np.linspace(tcal_.min(), tcal_.max() * 1.000001, a.tbins + 1)
        tbin = np.clip(np.digitize(tcal_, tb_edges) - 1, 0, a.tbins - 1)
        out = np.zeros(tuple(cal) + (nc,), np.complex64)
        for b in range(a.tbins):
            if not np.any(tbin == b):
                continue
            tc = 0.5 * (tb_edges[b] + tb_edges[b + 1])
            img = tru * (pd * np.exp(-(r2 + 1j * w) * tc) * np.exp(1j * w * T0))[..., None]  # image defined at T0
            cb = calib_of(img.astype(np.complex64))
            out[tbin == b] = cb[tbin == b]
        return out
    body_ = pd > 0.05 * np.median(pd[pd > 0])
    print(f"field over the object: |f| p50 {np.median(np.abs(w[body_])) / 2 / np.pi:.0f} Hz, p99 {np.percentile(np.abs(w[body_]), 99) / 2 / np.pi:.0f} Hz")
    calib = calib_at(tcal)
    if a.mix_return:
        calib = (1 - wmix[..., None]) * calib + wmix[..., None] * calib_at(tret)
if a.cg:
    from cgfilter import cg_filter
    calib = cg_filter(calib, a.cg, f)
t1 = time.time()
kern, sv, keep = calibrate(calib, kernel=a.kernel, max_rows=a.rows)
t2 = time.time()
zs = [int(v) for v in a.zs.split(",")]
mp, eig = maps_on_slices(kern, None, DIMS, zs, f)
t3 = time.time()
# regions (truth-derived, diagnostic only)
sig = np.abs(rho)
thr = 0.05 * np.median(sig[sig > 0])
rows = {}
print(f"[{a.tag}] cal {cal} on {f[0]:.2f}x FOV, kernel {a.kernel}, {keep} kernels of {kern.shape[0] * kern.shape[1] * kern.shape[2] * nc}; "
      f"calib {t1 - t0:.0f}s, gram+eig {t2 - t1:.0f}s, maps {t3 - t2:.0f}s")
for i, z in enumerate(zs):
    obj = sig[:, :, z] > thr
    body = binary_fill_holes(obj)
    lung = body & ~obj
    dist = distance_transform_edt(body, sampling=[1e3 * v for v in vox[:2]])
    t = tru[:, :, z]; s = mp[i]
    e1, orth = ip_err(s, t)
    crop = eig[i] > a.eig
    wgt = rho[:, :, z] ** 2
    def stat(r):
        rr = r & crop
        return dict(n=int(r.sum()), cropped=float((r & ~crop).mean()) if r.any() else 0, e1=float(e1[rr].mean()) if rr.any() else np.nan,
                    e1_p90=float(np.percentile(e1[rr], 90)) if rr.any() else np.nan, orth_w=float((wgt[rr] * orth[rr]).sum() / max(wgt[rr].sum(), 1e-30)),
                    e1_nocrop=float(e1[r].mean()) if r.any() else np.nan)
    reg = {"tissue": obj, "skin<=20mm": obj & (dist <= 20), "deep>50mm": obj & (dist > 50), "lung/low-signal holes": lung}
    rows[z] = {k: stat(v) for k, v in reg.items()}
    print(f"  z={z:2d} " + " | ".join(f"{k}: n={v['n']} crop {v['cropped']:.2f} e1 {v['e1']:.4f} (p90 {v['e1_p90']:.4f}, no-crop {v['e1_nocrop']:.4f}) orth_w {v['orth_w']:.1e}" for k, v in rows[z].items()))
out = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out, exist_ok=True)
json.dump({"args": vars(a), "cal": cal, "keep": keep, "f": f, "rows": rows}, open(os.path.join(out, f"oracle_{a.tag}.json"), "w"), indent=1)
np.save(os.path.join(os.environ.get("SMAPS_CACHE", "/tmp"), f"oracle_{a.tag}_maps.npy"), np.stack([mp[i] for i in range(len(zs))]).astype(np.complex64))
np.save(os.path.join(os.environ.get("SMAPS_CACHE", "/tmp"), f"oracle_{a.tag}_eig.npy"), eig.astype(np.float32))
