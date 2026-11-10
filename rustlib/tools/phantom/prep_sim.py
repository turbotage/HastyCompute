"""Prepare the k-space simulation of a factory phantom (input of `simphantom`).

    python -I prep_sim.py <phantom_dir> <blood_frames.hv> <out_dir> [--center x,y,z] [--fov fx,fy,fz] [--res r]
        [--scan-time 60] [--margin 25] [--src-step 1] [--slab auto|z0,z1] [--fov auto]
        [--src-roi heart|x0,x1,y0,y1,z0,z1 (mm) --src-step-out k]

Prescription like a scanner's: `--slab auto` excites an axial slab from 10 mm below the heart (a bit of the IVC) to
5 mm above the aortic arch (this phantom's labels and vessel graph; the RF slab of simphantom and cardiac_lbm --spin),
and everything in x, y (arms included where the atlas has them); `--fov auto`: the body's x, y extent in the slab +
10 mm, z = 1.3 × the slab (its transition bands). The prescription records the slab (centre, thickness).
`--src-roi` / `--src-step-out k`: tissue sources at `--src-step` (1 mm) inside the box (`heart`: the 0.5 mm heart
label box: heart and great vessels) and one per k³ block outside it — the direct-sum cost scales with the sources.

- tissue sources (Lagrangian, reference = atlas pose): every 1 mm voxel of the reference labels within the slab
  (± margin for motion) that is not air / airway / a heart chamber, weighted by 1 − (vessel lumen fraction at the
  reference, from `phantomflow` on the phantom's vessels); water and fat (fraction ff, −3.4 ppm) as separate sources;
  PD / T1 / T2* from the phantom's table with its textures (tissue_variation.apply);
- B0: dipole field of the χ map (FFT, padded, field relative to the surrounding air), 2nd-order shim over the tissue
  in the prescribed box (as the scene simulator does); sampled per tissue source at its reference position
  (Lagrangian) and per blood voxel at its frame position;
- blood frames (Eulerian, from `phantom_4d --blood-export` at r = 0): positions, blood-mean velocity, PD × fraction,
  T1 / T2* per blood class, field;
- breathing r(t), ṙ(t) at 1 ms over the scan (motion.Breathing with the phantom's parameters);
- the body mask (2 mm) for the coil array, prescription (axial: logical = subject axes), isocentre (= FOV centre).
`--keep-box heart:M | x0,x1,y0,y1,z0,z1 (mm)` (development): tissue and blood sources only inside the box (as if the
outer volume were suppressed: no arms, small FOV); `heart:M` = the bounding box of the heart and great-vessel labels in
the slab, dilated by M mm. The FOV becomes the box + 10 mm on every side; the B0 field still comes from the whole body.
`--src-octree` (multi-resolution sources): label-homogeneous blocks of 8 / 4 / 2 mm outside the heart (and 12 mm around
it), without vessel lumen, become box elements (edge `size_m`, the block's mean field and its linear gradient
`field_grad_ppm_m`; simphantom integrates each box analytically: h³ Π sinc(h (k + ∇ω τ) / 2)); every other voxel is a
1 mm box. Replaces --src-step / --src-step-out.
Writes <out_dir>/sim_input.hv.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
sys.path.insert(0, str(HERE))
from hastycompute import value_format as vf  # noqa: E402
from motion import Breathing  # noqa: E402
from tissue_variation import modes as tex_modes  # noqa: E402

A = sys.argv[1:]
PH, BF, OUT = Path(A[0]), Path(A[1]), Path(A[2])
OUT.mkdir(parents=True, exist_ok=True)


def opt(n, d):
    return A[A.index(n) + 1] if n in A else d


dev = torch.device("cuda")
t0 = time.time()
center = np.array([float(x) for x in opt("--center", "0.02,-0.03,1.31").split(",")])
fov = np.array([float(x) for x in (opt("--fov", "0.40,0.40,0.16") if opt("--fov", "") != "auto" else "0.40,0.40,0.16").split(",")])
res = float(opt("--res", 0.0025))
scan_time = float(opt("--scan-time", 60))
margin = float(opt("--margin", 25)) * 1e-3
P = json.loads((PH / "params.json").read_text())
TP = json.loads((PH / "tissue_params.json").read_text())
# the phantom's own (diffeomorphically deformed) atlas when it has one (make_phantom --diffeo)
P0 = PH / "atlas" if (PH / "atlas" / "tissue_1mm.hv").exists() else REPO / "data/hasty/phantom/p0"
TS = vf.read(P0 / "tissue_1mm.hv")
names = list(TS["tissues"])
ID = {n: i for i, n in enumerate(names)}
tab = TP["table"]
lab_full = np.asarray(TS["labels"])
o_full = np.asarray(TS["origin_m"], float)
h = float(TS["voxel_m"])

# ── prescription: slab from the anatomy, FOV from the body ───────────────────────────────────────────────────────────
slab = None
if opt("--slab", None) == "auto":
    sys.path.insert(0, str(HERE))
    from slab import auto_slab  # noqa: E402

    slab = auto_slab(PH, TS)
elif opt("--slab", None):
    slab = tuple(float(x) * 1e-3 for x in opt("--slab", None).split(","))
if slab is not None:
    center[2] = 0.5 * (slab[0] + slab[1])
    print(f"slab z {slab[0] * 1e3:.0f}…{slab[1] * 1e3:.0f} mm ({(slab[1] - slab[0]) * 1e3:.0f} mm)")
if opt("--fov", None) == "auto":
    zr = (slice(None) if slab is None else slice(max(int((slab[0] - o_full[2]) / h), 0), int((slab[1] - o_full[2]) / h) + 1))
    bm = lab_full[zr] != ID["air"]
    ys_, xs_ = np.nonzero(bm.any(0))
    xlo, xhi = o_full[0] + xs_.min() * h - 10e-3, o_full[0] + (xs_.max() + 1) * h + 10e-3
    ylo, yhi = o_full[1] + ys_.min() * h - 10e-3, o_full[1] + (ys_.max() + 1) * h + 10e-3
    fz = 1.3 * (slab[1] - slab[0]) if slab is not None else fov[2]
    fov = np.array([xhi - xlo, yhi - ylo, fz])
    center[0], center[1] = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    edge_x = bool((lab_full[zr][:, :, 0] != ID["air"]).any() or (lab_full[zr][:, :, -1] != ID["air"]).any())
    print(f"FOV {np.round(fov * 1e3).tolist()} mm at {np.round(center * 1e3, 1).tolist()} mm" + (" — the body reaches the atlas x edge: arms cut off (rebuild the atlas wider)" if edge_x else ""))

# ── development box: tissue / blood only inside it, FOV from it ───────────────────────────────────────────────────
keep_box = None
if opt("--keep-box", None):
    kb = opt("--keep-box", None)
    if kb.startswith("heart"):
        marg = float(kb.split(":")[1]) * 1e-3 if ":" in kb else 0.03
        HID = [ID[n] for n in ("myocardium", "valve", "blood_LA", "blood_LV", "blood_RA", "blood_RV", "blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein", "vessel_wall")]
        zr_ = slice(None) if slab is None else slice(max(int((slab[0] - o_full[2]) / h), 0), int((slab[1] - o_full[2]) / h) + 1)
        hz_, hy_, hx_ = np.nonzero(np.isin(lab_full[zr_], HID))
        hz_ = hz_ + (zr_.start or 0)
        lo_ = o_full + np.array([hx_.min(), hy_.min(), hz_.min()]) * h - marg
        hi_ = o_full + np.array([hx_.max() + 1, hy_.max() + 1, hz_.max() + 1]) * h + marg
    else:
        v_ = [float(x) * 1e-3 for x in kb.split(",")]
        lo_, hi_ = np.array(v_[0::2]), np.array(v_[1::2])
    if slab is not None:
        lo_[2], hi_[2] = max(lo_[2], slab[0] - 0.02), min(hi_[2], slab[1] + 0.02)
    keep_box = (lo_, hi_)
    # no tissue outside the box (outer volume suppressed): the FOV is the box + 10 mm on every side, z included
    center[:] = 0.5 * (lo_ + hi_)
    fov[:] = hi_ - lo_ + 0.02
    print(f"keep box {np.round(lo_ * 1e3).tolist()}…{np.round(hi_ * 1e3).tolist()} mm (outer volume suppressed): FOV {np.round(fov * 1e3).tolist()} mm at {np.round(center * 1e3, 1).tolist()} mm")


def in_box(P):
    """P: [n, 3] positions (m) → bool, all True without --keep-box"""
    if keep_box is None:
        return np.ones(len(P), bool)
    return np.all((P >= keep_box[0]) & (P < keep_box[1]), 1)


# ── B0: dipole field of the χ map (whole 1 mm thorax grid), shim ───────────────────────────────────────────────────
chi_t = np.array([tab[n]["chi_ppm"] if "chi_ppm" in tab[n] else 0.0 for n in names], np.float32)
chi_out = float(tab["air"]["chi_ppm"])
chi = torch.tensor(chi_t[lab_full] - chi_out, device=dev)
pad = 64
sh = [n + 2 * pad for n in chi.shape]
F = torch.fft.rfftn(chi, s=sh)
kz = torch.fft.fftfreq(sh[0], device=dev)[:, None, None]
ky = torch.fft.fftfreq(sh[1], device=dev)[None, :, None]
kx = torch.fft.rfftfreq(sh[2], device=dev)[None, None, :]
k2 = kx**2 + ky**2 + kz**2
D = 1 / 3 - kz**2 / torch.where(k2 > 0, k2, torch.ones_like(k2))
D[0, 0, 0] = 0
field = torch.fft.irfftn(F * D, s=sh)[: chi.shape[0], : chi.shape[1], : chi.shape[2]]  # ppm, B0 along z (SI)
del F, D, chi
zg, yg, xg = [torch.arange(n, device=dev, dtype=torch.float32) for n in lab_full.shape]
Zc = o_full[2] + (zg + 0.5) * h
Yc = o_full[1] + (yg + 0.5) * h
Xc = o_full[0] + (xg + 0.5) * h
LZ, LY, LX = torch.meshgrid(Zc - center[2], Yc - center[1], Xc - center[0], indexing="ij")
labt = torch.tensor(lab_full, device=dev)
tissue = ~torch.isin(labt, torch.tensor([ID["air"], ID["airway"], ID["lung"]], device=dev))
box = (LX.abs() < 0.4 * fov[0]) & (LY.abs() < 0.4 * fov[1]) & (LZ.abs() < 0.5 * fov[2] + 0.02) & tissue
xs_, ys_, zs_ = LX[box], LY[box], LZ[box]
Bm = torch.stack([torch.ones_like(xs_), xs_, ys_, zs_, xs_**2, ys_**2, zs_**2, xs_ * ys_, xs_ * zs_, ys_ * zs_], 1).double()
coef = torch.linalg.solve(Bm.T @ Bm, Bm.T @ field[box].double())  # 10 shim terms: normal equations
poly = sum(c * t for c, t in zip(coef.float(), [torch.ones_like(LX), LX, LY, LZ, LX**2, LY**2, LZ**2, LX * LY, LX * LZ, LY * LZ]))
field = field - poly
fs = field[box]
print(f"B0 after shim in the box: p1 {float(torch.quantile(fs[::50], 0.01)):.2f}, p50 {float(torch.quantile(fs[::50], 0.5)):.2f}, p99 {float(torch.quantile(fs[::50], 0.99)):.2f} ppm "
      f"(±{float(torch.quantile(fs[::50].abs(), 0.99)) * 127.7:.0f} Hz at 3 T) ({time.time() - t0:.0f}s)")
del LX, LY, LZ, box, Bm, poly

# ── tissue sources (reference pose) ────────────────────────────────────────────────────────────────────────────────
z0 = int(np.floor((center[2] - fov[2] / 2 - margin - o_full[2]) / h))
z1 = int(np.ceil((center[2] + fov[2] / 2 + margin - o_full[2]) / h))
z0, z1 = max(z0, 0), min(z1, lab_full.shape[0])
lab = labt[z0:z1]
nzc, nyc, nxc = lab.shape
crop_o = o_full + np.array([0, 0, z0 * h])
# vessel lumen fraction at the reference pose (static voxelisation of this phantom's vessels)
ccen = crop_o + np.array([nxc, nyc, nzc]) * h / 2
subprocess.run([str(REPO / "rustlib/target/release/phantomflow"), str(PH / "vessels.hv"), str(OUT / "ref_vessels.hv"), "--center", ",".join(f"{x:.6f}" for x in ccen),
                "--size", ",".join(f"{x:.6f}" for x in np.array([nxc, nyc, nzc]) * h), "--vox", f"{h}", "--sub", "3", "--harmonics", "0"], check=True, stdout=subprocess.DEVNULL)
RV = vf.read(OUT / "ref_vessels.hv")
lumen = torch.zeros(nzc * nyc * nxc, device=dev)
lumen[torch.tensor(np.asarray(RV["idx"]).astype(np.int64), device=dev)] = torch.tensor(np.asarray(RV["blood"]), device=dev)
lumen = lumen.view(nzc, nyc, nxc)
CH = [ID[c] for c in ("blood_LA", "blood_LV", "blood_RA", "blood_RV")]
keep = ~torch.isin(lab, torch.tensor([ID["air"], ID["airway"]] + CH, device=dev)) & (lumen < 0.999)
# --src-step k (development phantoms): one source per k³ block (the block's first voxel's tissue, PD × k³, placed at the
# block centre) → k³ fewer tissue sources for fast method iteration; final evaluations use k = 1
step = int(opt("--src-step", 1))
step_out = int(opt("--src-step-out", step))
roi = opt("--src-roi", None)
octree = "--src-octree" in A
if keep_box is not None:
    zc_ = crop_o[2] + (torch.arange(nzc, device=dev) + 0.5) * h
    yc_ = crop_o[1] + (torch.arange(nyc, device=dev) + 0.5) * h
    xc_ = crop_o[0] + (torch.arange(nxc, device=dev) + 0.5) * h
    inb = ((zc_ >= keep_box[0][2]) & (zc_ < keep_box[1][2]))[:, None, None] & ((yc_ >= keep_box[0][1]) & (yc_ < keep_box[1][1]))[None, :, None] & ((xc_ >= keep_box[0][0]) & (xc_ < keep_box[1][0]))[None, None, :]
    keep &= inb
zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=dev) for n in keep.shape), indexing="ij")
el_grad = None
if octree:
    # blocks of 8 / 4 / 2 voxels (coarsest first): one label, all kept, no lumen, away from the heart → box elements
    fl_ = field[z0:z1].float()
    HID = [ID[n] for n in ("myocardium", "valve", "blood_LA", "blood_LV", "blood_RA", "blood_RV", "blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein", "vessel_wall")]
    heartm = torch.isin(lab, torch.tensor(HID, device=dev)).float()[None, None]
    r12 = int(round(12e-3 / h))
    near = torch.nn.functional.max_pool3d(heartm, 2 * r12 + 1, 1, r12)[0, 0] > 0
    covered = torch.zeros_like(keep)
    el_list = []  # (zi, yi, xi block origin, B)
    for Bk in (8, 4, 2):
        Z_, Y_, X_ = (n // Bk * Bk for n in keep.shape)
        def blk(t):
            return t[:Z_, :Y_, :X_].reshape(Z_ // Bk, Bk, Y_ // Bk, Bk, X_ // Bk, Bk)
        lb_ = blk(lab)
        homo = (lb_.amax((1, 3, 5)) == lb_.amin((1, 3, 5)))
        ok = homo & blk(keep).all(5).all(3).all(1) & ~blk(near).any(5).any(3).any(1) & (blk(lumen).amax((1, 3, 5)) == 0) & ~blk(covered).any(5).any(3).any(1)
        bz, by, bx = torch.nonzero(ok, as_tuple=True)
        el_list.append((bz * Bk, by * Bk, bx * Bk, Bk))
        m_ = ok[:, None, :, None, :, None].expand(-1, Bk, -1, Bk, -1, Bk).reshape(Z_, Y_, X_)
        covered[:Z_, :Y_, :X_] |= m_
        print(f"  octree {Bk} mm: {len(bz) / 1e6:.3f} M elements ({int(m_.sum()) / 1e6:.2f} M voxels)")
    rz, ry, rx = torch.nonzero(keep & ~covered, as_tuple=True)
    el_list.append((rz, ry, rx, 1))
    # element origins, sizes, mean field and gradient (block halves; 1 mm: central differences)
    zi = torch.cat([e[0] for e in el_list]); yi = torch.cat([e[1] for e in el_list]); xi = torch.cat([e[2] for e in el_list])
    sstep = torch.cat([torch.full((len(e[0]),), float(e[3]), device=dev) for e in el_list])
    fmean, grad = [], []
    for (bz_, by_, bx_, Bk) in el_list:
        if Bk == 1:
            fmean.append(fl_[bz_, by_, bx_])
            gz_, gy_, gx_ = torch.gradient(fl_, spacing=h)
            grad.append(torch.stack([gx_[bz_, by_, bx_], gy_[bz_, by_, bx_], gz_[bz_, by_, bx_]], 1))
            continue
        Z_, Y_, X_ = (n // Bk * Bk for n in keep.shape)
        fb = fl_[:Z_, :Y_, :X_].reshape(Z_ // Bk, Bk, Y_ // Bk, Bk, X_ // Bk, Bk)[bz_ // Bk, :, by_ // Bk, :, bx_ // Bk, :]  # [n, B, B, B] (z, y, x)
        hb = Bk // 2
        fmean.append(fb.mean((1, 2, 3)))
        gxb = (fb[:, :, :, hb:].mean((1, 2, 3)) - fb[:, :, :, :hb].mean((1, 2, 3))) / (hb * h)
        gyb = (fb[:, :, hb:, :].mean((1, 2, 3)) - fb[:, :, :hb, :].mean((1, 2, 3))) / (hb * h)
        gzb = (fb[:, hb:, :, :].mean((1, 2, 3)) - fb[:, :hb, :, :].mean((1, 2, 3))) / (hb * h)
        grad.append(torch.stack([gxb, gyb, gzb], 1))
    el_field = torch.cat(fmean)
    el_grad = torch.cat(grad)
    keep_oct = True
    print(f"tissue elements: {len(zi) / 1e6:.2f} M (octree) instead of {int(keep.sum()) / 1e6:.2f} M voxels")
if octree:
    pass
elif roi is not None and step_out != step:
    if roi == "heart":
        HT = vf.read(P0 / "tissue_0.5mm_heart.hv")
        rlo = np.asarray(HT["origin_m"], float)
        rhi = rlo + np.asarray(HT["dims_xyz"], float) * float(HT["voxel_m"])
    else:
        v_ = [float(x) * 1e-3 for x in roi.split(",")]
        rlo, rhi = np.array(v_[0::2]), np.array(v_[1::2])
    # ROI in crop voxel indices, on the coarse lattice so that the inner and outer blocks tile the crop exactly
    ilo = np.floor((rlo - crop_o) / h / step_out).astype(int) * step_out
    ihi = np.ceil((rhi - crop_o) / h / step_out).astype(int) * step_out
    inr = (xx >= ilo[0]) & (xx < ihi[0]) & (yy >= ilo[1]) & (yy < ihi[1]) & (zz >= ilo[2]) & (zz < ihi[2])
    keep &= torch.where(inr, (zz % step == 0) & (yy % step == 0) & (xx % step == 0), (zz % step_out == 0) & (yy % step_out == 0) & (xx % step_out == 0))
    stp = torch.where(inr, torch.tensor(float(step), device=dev), torch.tensor(float(step_out), device=dev))
    print(f"tissue sources: {step} mm in the ROI {np.round(rlo * 1e3).tolist()}…{np.round(rhi * 1e3).tolist()} mm, one per {step_out}³ mm outside")
elif step > 1:
    keep &= (zz % step == 0) & (yy % step == 0) & (xx % step == 0)
    stp = torch.full_like(zz, step, dtype=torch.float32)
else:
    stp = torch.ones_like(zz, dtype=torch.float32)
if not octree:
    zi, yi, xi = torch.nonzero(keep, as_tuple=True)
    sstep = stp[zi, yi, xi]
off = 0.5 * sstep
X = torch.stack([crop_o[0] + (xi + off) * h, crop_o[1] + (yi + off) * h, crop_o[2] + (zi + off) * h], 1).float()
del zz, yy, xx
if not octree:
    del stp
L = lab[zi, yi, xi].long()
w = 1 - lumen[zi, yi, xi]
ta = torch.zeros(len(L), device=dev)
tb = torch.zeros(len(L), device=dev)
for nm_, (sa_, sb_, l0_, l1_) in TP.get("texture", {}).items():
    if nm_ in ID and (sa_ > 0 or sb_ > 0):
        sel = L == ID[nm_]
        if bool(sel.any()):
            ka, aa, pa = (torch.tensor(x, device=dev, dtype=torch.float32) for x in tex_modes(TP["texture_seed"], ID[nm_], sa_, (l0_, l1_), which=0))
            kb, ab, pb = (torch.tensor(x, device=dev, dtype=torch.float32) for x in tex_modes(TP["texture_seed"], ID[nm_], sb_, (l0_, l1_), which=1))
            Xs = X[sel].double().float()
            ta[sel] = torch.cos(Xs @ ka.T + pa) @ aa
            tb[sel] = torch.cos(Xs @ kb.T + pb) @ ab


def col(k):
    return torch.tensor([float(tab[n].get(k, 0.0)) for n in names], device=dev)


PD, T1, T2S, FF = col("pd")[L], col("t1_s")[L], col("t2s_s")[L], col("ff")[L]
pd = PD * torch.clamp(1 + ta, 0.2, 2.0) * w * sstep**3
t1 = T1 * torch.clamp(1 + tb / 3, 0.5, 1.5)
t2s = T2S * torch.clamp(1 + tb, 0.2, 2.0)
fld = el_field if octree else field[z0:z1][zi, yi, xi]
fat = tab["fat"]
wat_m = pd * (1 - FF)
fat_sel = FF > 0.15  # separate fat sources only for fat-rich tissue (adipose, marrow, thymus); else water only
src = {
    "pos": torch.cat([X, X[fat_sel]]).cpu().numpy(),
    "pd": torch.cat([wat_m, (pd * FF)[fat_sel]]).cpu().numpy().astype(np.float32),
    "t1": torch.cat([t1, torch.full((int(fat_sel.sum()),), float(fat["t1_s"]), device=dev)]).cpu().numpy().astype(np.float32),
    "t2s": torch.cat([t2s, torch.full((int(fat_sel.sum()),), float(fat["t2s_s"]), device=dev)]).cpu().numpy().astype(np.float32),
    "field_ppm": torch.cat([fld, fld[fat_sel]]).cpu().numpy().astype(np.float32),
    "shift_ppm": torch.cat([torch.zeros_like(fld), torch.full((int(fat_sel.sum()),), -3.4, device=dev)]).cpu().numpy().astype(np.float32),
}
if octree:
    src["size_m"] = torch.cat([sstep * h, (sstep * h)[fat_sel]]).cpu().numpy().astype(np.float32)
    src["field_grad_ppm_m"] = torch.cat([el_grad, el_grad[fat_sel]]).cpu().numpy().astype(np.float32)
nz_ = src["pd"] > 1e-4
src = {k: v[nz_] for k, v in src.items()}
print(f"tissue sources: {len(src['pd']) / 1e6:.2f} M ({int(fat_sel.sum()) / 1e6:.2f} M fat) ({time.time() - t0:.0f}s)")

# ── blood frames ───────────────────────────────────────────────────────────────────────────────────────────────────
B = vf.read(BF)
bo, bh = np.asarray(B["origin_m"], float), float(B["voxel_m"])
bnx, bny, bnz = (int(x) for x in B["dims_xyz"])
inflow = 1.5
fgrid = field.permute(2, 1, 0).contiguous()  # [X, Y, Z]


def field_at(pos):
    i = np.clip(np.floor((pos - o_full) / h).astype(int), 0, np.array(lab_full.shape[::-1]) - 1)
    return field[torch.tensor(i[:, 2], device=dev), torch.tensor(i[:, 1], device=dev), torch.tensor(i[:, 0], device=dev)].cpu().numpy()


frames = []
for fr in B["frames"]:
    idx = np.asarray(fr["idx"]).astype(np.int64)
    pos = np.stack([bo[0] + (idx % bnx + 0.5) * bh, bo[1] + ((idx // bnx) % bny + 0.5) * bh, bo[2] + (idx // (bnx * bny) + 0.5) * bh], 1)
    lb = np.asarray(fr["label"])
    frac = np.asarray(fr["frac"]).astype(np.float32) / 255
    if keep_box is not None:
        ib_ = in_box(pos)
        pos, lb, frac, fvel_ = pos[ib_], lb[ib_], frac[ib_], np.asarray(fr["vel"])[ib_]
    else:
        fvel_ = np.asarray(fr["vel"])
    bl_pd = np.array([tab[n]["pd"] for n in names], np.float32)[lb] * frac * inflow
    frames.append({"tc": float(fr["tc"]), "pos": pos.astype(np.float32), "vel": fvel_.astype(np.float32), "pd": bl_pd,
                   "t1": np.array([tab[n]["t1_s"] for n in names], np.float32)[lb], "t2s": np.array([tab[n]["t2s_s"] for n in names], np.float32)[lb],
                   "field_ppm": field_at(pos).astype(np.float32)})
print(f"blood frames: {len(frames)}, {np.mean([len(f_['pd']) for f_ in frames]) / 1e6:.2f} M voxels each (voxel {bh * 1e3:.1f} mm) ({time.time() - t0:.0f}s)")

# ── breathing, body mask, prescription ─────────────────────────────────────────────────────────────────────────────
# field on the reconstruction grid (logical = subject axes, centred, index i ↦ (i − n/2) res)
rd = np.round(fov / res).astype(int)
gx = center[0] + (np.arange(rd[0]) - rd[0] // 2) * res
gy = center[1] + (np.arange(rd[1]) - rd[1] // 2) * res
gz = center[2] + (np.arange(rd[2]) - rd[2] // 2) * res
GZ, GY, GX = np.meshgrid(gz, gy, gx, indexing="ij")
field_recon = field_at(np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)).reshape(rd[2], rd[1], rd[0]).astype(np.float32)
bp = P["breathing"]
tb_ = np.arange(0, scan_time + 2.0, 1e-3)
r_, rd_ = Breathing(P["seed"], period=bp["period_s"], n=bp["n"], cv_period=bp["cv_period"], cv_depth=bp["cv_depth"])(tb_)
body = (lab_full[::2, ::2, ::2] != ID["air"]).astype(np.uint8)
mo = PH / "motion"
vf.write(OUT / "sim_input.hv", {
    "tissue": src, "blood": frames, "blood_voxel_m": bh, "tissue_voxel_m": h,
    "breathing": {"dt": 1e-3, "r": r_.astype(np.float32), "rdot": rd_.astype(np.float32)},
    "cardiac_motion": str(mo / "cardiac_motion.hv"), "resp_motion": str(mo / "resp_motion.hv"),
    "period": float(B["period"]), "presc": {"center": center.tolist(), "fov": fov.tolist(), "res": res, "rot": np.eye(3).tolist(),
              **({"slab": {"center": [float(center[0]), float(center[1]), float(0.5 * (slab[0] + slab[1]))], "thk": float(slab[1] - slab[0]), "normal": [0.0, 0.0, 1.0]}} if slab is not None else {})}, "iso": center.tolist(),
    "body": body, "body_origin_m": (o_full + h / 2).tolist(), "body_voxel_m": 2 * h, "scan_time": scan_time, "phantom": PH.name, "params": json.dumps(P),
    "field_recon": field_recon, "recon_dims_xyz": rd.tolist(),
    # the constant inflow gain in the blood PD (simphantom divides it out when an LBM spin history gives M_z)
    "blood_inflow_gain": inflow,
})
print(f"wrote {OUT / 'sim_input.hv'} ({time.time() - t0:.0f}s)")
