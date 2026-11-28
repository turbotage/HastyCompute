"""Score the data-only blood pool (flow_bloodpool) against the truth of a phantom, and measure how separable the tissue
classes are in its magnitude (the question for a wall shell: is the epicardial edge in the data?).

    rustlib/.venv/bin/python -I rustlib/tools/flow/score_bloodpool.py --flow <viewer_flow.hv> --truth <truth4d.hv> --server URL
        [--step 2] [--opts '{"min_voxels": 200}'] [--save-seq out_dir]

flow value: flow4d (vel [F,3,Z,Y,X], mag [F,Z,Y,X]) on the grid of truth4d box-averaged by --step (phantom_4d.py --viewer-step).
The method never sees the truth; truth4d's `blood` (u8 fraction × 255) and `labels` are used here only for scoring.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute.store import load_any  # noqa: E402
from hastycompute.v2 import Client  # noqa: E402

A = sys.argv[1:]


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


k = int(opt("--step", 2))
flow = load_any(opt("--flow"))
tr = load_any(opt("--truth"))
c = Client(opt("--server"))
F = flow["vel"].shape[0]
if "--snr" in A:
    # recon-like noise: complex Gaussian noise on the magnitude image (blood SNR = --snr), velocity noise venc / (π √2 SNR_local) per
    # component, so air and dark tissue have huge velocity noise like real phase data
    snr = float(opt("--snr"))
    rng = np.random.default_rng(7)
    mg = np.asarray(flow["mag"], np.float32)
    tb = np.asarray(tr["blood"])
    kk = k
    bsel = (tb[:, : tb.shape[1] // kk * kk : kk, : tb.shape[2] // kk * kk : kk, : tb.shape[3] // kk * kk : kk] >= 128)
    sig = float(np.median(mg[bsel]) / snr)
    noisy = np.sqrt((mg + rng.normal(0, sig, mg.shape)) ** 2 + rng.normal(0, sig, mg.shape) ** 2).astype(np.float32)
    venc = float(flow.get("venc") or 1.5)
    sv = (venc / (np.pi * np.sqrt(2) * np.maximum(mg / sig, 1.0))).astype(np.float32)
    vel = np.asarray(flow["vel"], np.float32)
    flow = dict(flow)
    flow["vel"] = vel + sv[:, None] * rng.normal(0, 1, vel.shape).astype(np.float32)
    flow["mag"] = noisy
    print("noise: blood SNR %.1f, magnitude sigma %.4g, velocity sigma at blood %.3f m/s" % (snr, sig, venc / (np.pi * np.sqrt(2) * snr)))
fid = c.put(flow, name="score_bloodpool flow", metadata={"kind": "flow4d"})
options = json.loads(opt("--opts", "{}"))
options["return_mask"] = True
outs, msg = c.execute("flow_bloodpool", [fid], options, "score_bloodpool surfaces")
m = json.loads(msg)
mask = np.asarray(c.get(outs[1])) > 0
seq = c.get(outs[0])
print("thresholds: air %.4g  mag %.4g  speed %.4g m/s | source %s" % (m["air_thr"], m["mag_thr"], m["speed_thr"], m["source"]))
print("components per frame:", m["components"])
print("links:", [int(x) for x in m["links"]])

# truth on the flow grid
blood = np.asarray(tr["blood"])
Fz, Z, Y, X = blood.shape
zz, yy, xx = Z // k, Y // k, X // k
bf = blood[:, : zz * k, : yy * k, : xx * k].reshape(Fz, zz, k, yy, k, xx, k).astype(np.float32).mean(axis=(2, 4, 6)) / 255.0
assert bf.shape == mask.shape, (bf.shape, mask.shape)
truth = bf >= 0.5
sp = float(flow["spacing"][0]) * 1e3
vox_ml = (sp ** 3) * 1e-3
print("\nframe  truth_ml  mask_ml  surf_ml   dice  precision recall | surface distance to the truth boundary (mm): mean  p95  max")
tm = m["surface_volume_ml"]
frames = seq["frames"]
rows = []
for f in range(F):
    t, p = truth[f], mask[f]
    inter = (t & p).sum()
    dice = 2 * inter / max(1, t.sum() + p.sum())
    # signed distance to the truth boundary (mm, + outside) at the mesh vertices
    sd = (ndi.distance_transform_edt(~t) - ndi.distance_transform_edt(t)) * sp
    v = np.asarray(frames[f]["vertices"]) / sp                       # voxel coordinates (x, y, z)
    d = ndi.map_coordinates(sd, [v[:, 2], v[:, 1], v[:, 0]], order=1, mode="nearest")
    rows.append((f, t.sum() * vox_ml, p.sum() * vox_ml, tm[f], dice, inter / max(1, p.sum()), inter / max(1, t.sum()), np.abs(d).mean(), np.percentile(np.abs(d), 95), np.abs(d).max(), d.mean()))
    print("%4d  %8.1f %8.1f %8.1f  %6.3f  %6.3f  %6.3f |  %5.2f %5.2f %5.2f   (signed mean %+.2f)" % rows[-1])
r = np.array(rows)
print("mean dice %.3f  precision %.3f  recall %.3f  surface |d| mean %.2f mm  p95 %.2f mm  signed mean %+.2f mm" % (r[:, 4].mean(), r[:, 5].mean(), r[:, 6].mean(), r[:, 7].mean(), r[:, 8].mean(), r[:, 10].mean()))
if "--save-seq" in A:
    Path(opt("--save-seq")).mkdir(parents=True, exist_ok=True)
    np.save(Path(opt("--save-seq")) / "mask.npy", mask)

# ── class separability in the magnitude (is the wall in the data?) ──────────────────────────────────────────────────────
names = list(tr["tissues"])
lab = np.asarray(tr["labels"])[:, : zz * k : k, : yy * k : k, : xx * k : k]       # central sample of each box
mag = np.asarray(flow["mag"])
scale = np.percentile(mag[truth], 50)                                              # blood = 1
print("\nmagnitude by truth class (frame-pooled, blood median = 1): class  voxels  mean  std")
stats = {}
for li in np.unique(lab):
    sel = lab == li
    if sel.sum() < 200:
        continue
    nm = names[int(li)] if 0 <= int(li) < len(names) else f"label{int(li)}"
    stats[nm] = (int(sel.sum()), float(mag[sel].mean() / scale), float(mag[sel].std() / scale))
for nm, (n, mu, sd_) in sorted(stats.items(), key=lambda x: -x[1][1]):
    print("  %-22s %9d  %6.3f  %6.3f" % (nm, n, mu, sd_))
# epicardial / endocardial neighbourhood: what is within 1 voxel outside the myocardium, and how far its magnitude is from the myocardium's
mi = [i for i, n_ in enumerate(names) if n_ == "myocardium"]
if mi:
    myo = np.isin(lab, mi)
    ring = ndi.binary_dilation(myo, structure=ndi.generate_binary_structure(4, 1), iterations=1) & ~myo
    ring[:] &= ndi.binary_dilation(myo, iterations=1) & ~myo
    mu_m, sd_m = mag[myo].mean() / scale, mag[myo].std() / scale
    print("\nneighbours of the myocardium (1 voxel): class  share  mean  contrast to myocardium in units of its std")
    for li in np.unique(lab[ring]):
        sel = ring & (lab == li)
        nm = names[int(li)] if 0 <= int(li) < len(names) else f"label{int(li)}"
        print("  %-22s %5.1f %%  %6.3f  %+6.2f" % (nm, 100 * sel.sum() / ring.sum(), mag[sel].mean() / scale, (mag[sel].mean() / scale - mu_m) / max(sd_m, 1e-6)))
    print("  (myocardium mean %.3f std %.3f)" % (mu_m, sd_m))

# ── wall shell (flow_wall_shell) against the truth wall: myocardium ─────────────────────────────────────────────────────
if "--wall" in A:
    wopts = json.loads(opt("--wall-opts", "{}"))
    wouts, wmsg = c.execute("flow_wall_shell", [fid, outs[0]], wopts, "score_bloodpool wall")
    wseq = c.get(wouts[0])
    wfr = wseq["frames"]
    sel_frames = [int(x) for x in opt("--wall-frames", "0,5,10,15").split(",")]
    tlab = np.asarray(tr["labels"])
    tblood = blood
    voxel_mm = float(tr["voxel_m"]) * 1e3
    myo_ids = [i for i, n_ in enumerate(names) if n_ in ("myocardium",)]
    print("\nwall shell vs truth (R = blood ∪ myocardium; frames %s); mm = distance of the outer vertex to the boundary of R, + = outside" % sel_frames)
    allrows = []
    for f in sel_frames:
        myo = np.isin(tlab[f], myo_ids)
        R = myo | (tblood[f] >= 128)
        zs, ys, xs = np.where(myo)
        pad = int(40 / voxel_mm)
        lo = [max(0, a.min() - pad) for a in (zs, ys, xs)]
        hi = [min(n_, a.max() + pad) for a, n_ in zip((zs, ys, xs), R.shape)]
        sl = tuple(slice(l_, h_) for l_, h_ in zip(lo, hi))
        Rc = R[sl]
        sd = (ndi.distance_transform_edt(~Rc) - ndi.distance_transform_edt(Rc)) * voxel_mm
        dmyo = ndi.distance_transform_edt(~myo[sl]) * voxel_mm
        lab_c = tlab[f][sl]
        inner = np.asarray(frames[f]["vertices"]) / voxel_mm
        outer = np.asarray(wfr[f]["vertices"]) / voxel_mm
        nrm = outer - inner
        nl = np.linalg.norm(nrm, axis=1, keepdims=True)
        sc = {k_: np.asarray(v_) for k_, v_ in wfr[f]["scalars"].items()}
        def at(vol, p):
            q = [p[:, 2] - lo[0], p[:, 1] - lo[1], p[:, 0] - lo[2]]
            ok = (q[0] >= 0) & (q[1] >= 0) & (q[2] >= 0) & (q[0] < vol.shape[0] - 1) & (q[1] < vol.shape[1] - 1) & (q[2] < vol.shape[2] - 1)
            out = np.full(len(p), np.nan)
            out[ok] = ndi.map_coordinates(vol, [q[0][ok], q[1][ok], q[2][ok]], order=1)
            return out
        d_out = at(sd, outer)
        d_myo_inner = at(dmyo, inner)                    # distance of the blood-pool vertex to the true myocardium
        near = d_myo_inner <= 2.5                         # vertices that sit on the myocardium
        unit = nrm / np.maximum(nl, 1e-6)
        probe = outer + 3.0 / voxel_mm * unit             # 3 mm behind the estimated outer surface: what is there?
        pi = np.round(probe[:, ::-1] - np.array(lo)).astype(int)
        okp = (pi >= 0).all(1) & (pi < np.array(lab_c.shape)).all(1)
        beyond = np.full(len(probe), -1)
        beyond[okp] = lab_c[pi[okp, 0], pi[okp, 1], pi[okp, 2]]
        conf = sc["confidence"]; supp = sc["support"]; th = sc["thickness_mm"]
        good = near & (conf >= 4) 
        far = (d_myo_inner > 8.0) & np.isfinite(d_myo_inner)
        if "--features" in A:
            q = lambda a_: " ".join("%7.2f" % v_ for v_ in np.percentile(a_, [10, 25, 50, 75, 90])) if len(a_) else "   none"
            print("  features of confident vertices (p10 p25 p50 p75 p90): near the myocardium: %d, far (> 8 mm): %d" % ((near & (conf >= 4)).sum(), (far & (conf >= 4)).sum()))
            for nm_ in ("chord_mm", "thickness_mm", "wall_rel", "out_rel", "confidence"):
                print("    %-13s near %s | far %s" % (nm_, q(sc[nm_][near & (conf >= 4)]), q(sc[nm_][far & (conf >= 4)])))
            sh_n = (near & (supp >= 0.5)).sum() / max(1, near.sum()); sh_f_all = (far & (supp >= 0.5)).sum(); n_far = max(1, far.sum())
            print("    SHOWN (support >= 0.5): %.1f %% of the vertices on the myocardium; %.1f %% of the vertices far (> 8 mm) from it  (%d of %d)" % (100 * sh_n, 100 * sh_f_all / n_far, sh_f_all, far.sum()))
            for thr in (20, 25, 30, 35, 40, 50):
                kept_n = (near & (conf >= 4) & (sc["chord_mm"] >= thr)).sum(); kept_f = (far & (conf >= 4) & (sc["chord_mm"] >= thr)).sum()
                print("    chord >= %2d mm keeps %5.1f %% of near, %5.1f %% of far" % (thr, 100 * kept_n / max(1, (near & (conf >= 4)).sum()), 100 * kept_f / max(1, (far & (conf >= 4)).sum())))
        shown = near & (supp >= 0.5)
        def row(sel, label):
            if sel.sum() == 0:
                return "  %-34s none" % label
            d = d_out[sel]; d = d[np.isfinite(d)]
            return "  %-34s %6d vertices  |d| median %.2f  p90 %.2f  mean signed %+.2f mm   thickness median %.1f mm" % (label, sel.sum(), np.median(np.abs(d)), np.percentile(np.abs(d), 90), d.mean(), np.median(th[sel]))
        print("frame %d: %d of %d blood-pool vertices lie on the true myocardium (≤ 2.5 mm)" % (f, near.sum(), len(near)))
        print(row(good, "confident edge (t ≥ 4)"))
        print(row(shown, "support ≥ 0.5 (what the viewer shows)"))
        print(row(near, "all of them (incl. extrapolated)"))
        print("   share with a confident edge: %.1f %%, support ≥ 0.5: %.1f %%" % (100 * good.sum() / max(1, near.sum()), 100 * shown.sum() / max(1, near.sum())))
        for li in np.unique(beyond[near]):
            sel = near & (beyond == li)
            if sel.sum() < 30:
                continue
            nm = names[int(li)] if 0 <= int(li) < len(names) else "outside"
            print(row(sel & (conf >= 4), "behind the wall: %s (confident)" % nm)[:150], "| share confident %.0f %%" % (100 * (sel & (conf >= 4)).sum() / sel.sum()))
        allrows.append((np.median(np.abs(d_out[good][np.isfinite(d_out[good])])) if good.sum() else np.nan, 100 * good.sum() / max(1, near.sum())))
    print("median |d| over frames (confident): %.2f mm; share of myocardial vertices with a confident edge: %.0f %%" % (np.nanmean([r_[0] for r_ in allrows]), np.mean([r_[1] for r_ in allrows])))
    if "--save-seq" in A:
        import pickle
        (Path(opt("--save-seq")) / "wall_id.txt").write_text(wouts[0])
