"""Prepare the k-space simulation of a factory phantom (input of `simphantom`).

    python -I prep_sim.py <phantom_dir> <blood_frames.hv> <out_dir> [--center x,y,z] [--fov fx,fy,fz] [--res r]
        [--scan-time 60] [--margin 25]

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
fov = np.array([float(x) for x in opt("--fov", "0.40,0.40,0.16").split(",")])
res = float(opt("--res", 0.0025))
scan_time = float(opt("--scan-time", 60))
margin = float(opt("--margin", 25)) * 1e-3
P = json.loads((PH / "params.json").read_text())
TP = json.loads((PH / "tissue_params.json").read_text())
P0 = REPO / "data/hasty/phantom/p0"
TS = vf.read(P0 / "tissue_1mm.hv")
names = list(TS["tissues"])
ID = {n: i for i, n in enumerate(names)}
tab = TP["table"]
lab_full = np.asarray(TS["labels"])
o_full = np.asarray(TS["origin_m"], float)
h = float(TS["voxel_m"])

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
zi, yi, xi = torch.nonzero(keep, as_tuple=True)
X = torch.stack([crop_o[0] + (xi + 0.5) * h, crop_o[1] + (yi + 0.5) * h, crop_o[2] + (zi + 0.5) * h], 1).float()
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
pd = PD * torch.clamp(1 + ta, 0.2, 2.0) * w
t1 = T1 * torch.clamp(1 + tb / 3, 0.5, 1.5)
t2s = T2S * torch.clamp(1 + tb, 0.2, 2.0)
fld = field[z0:z1][zi, yi, xi]
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
    bl_pd = np.array([tab[n]["pd"] for n in names], np.float32)[lb] * frac * inflow
    frames.append({"tc": float(fr["tc"]), "pos": pos.astype(np.float32), "vel": np.asarray(fr["vel"]).astype(np.float32), "pd": bl_pd,
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
    "period": float(B["period"]), "presc": {"center": center.tolist(), "fov": fov.tolist(), "res": res, "rot": np.eye(3).tolist()}, "iso": center.tolist(),
    "body": body, "body_origin_m": (o_full + h / 2).tolist(), "body_voxel_m": 2 * h, "scan_time": scan_time, "phantom": PH.name, "params": json.dumps(P),
    "field_recon": field_recon, "recon_dims_xyz": rd.tolist(),
})
print(f"wrote {OUT / 'sim_input.hv'} ({time.time() - t0:.0f}s)")
