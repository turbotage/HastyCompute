"""Moving ground truth of the atlas phantom (P1) on the GPU: cardiac + respiratory motion, moving / distending vessels,
potential flow in the heart chambers, frame by frame on one grid.

    python -I phantom_4d.py <p0_dir> <motion_dir> <out_dir> --labels-src <tissue_*.hv> [--frames 20] [--resp-state 0]
        [--realtime t0,t1 --seed s] [--crop xmin,xmax,ymin,ymax,zmin,zmax (mm)] [--vox 1] [--sub 3]
        [--viewer-step k] [--publish] [--server URL]

Frames: cardiac-gated (`--frames` per cycle at a fixed respiratory state r, 0 = end-expiration) or real time
(`--realtime t0,t1`: frames over [t0, t1] s, cardiac time t mod RR, breathing from `motion.Breathing`).
Per frame (all on the GPU with torch, except connected-component labelling of the chamber crops):
1. vessels: centreline samples moved by the motion, radii distended by the network pressure Δr/r = 0.75 Δp / (Eh/r),
   wall velocity = the material velocity → `phantomflow --frames` (blood fraction, voxel-mean blood velocity).
2. tissue: labels pulled back through the inverse motion (fixed point, trilinear motion fields, nearest label in the
   high-resolution label source); Eulerian tissue velocity v(X(x)).
3. chambers: potential flow v = ∇φ (Δφ = 0, matrix-free Jacobi-CG on the voxel Laplacian, per chamber group).
   Boundary normal velocity: walls = tissue velocity, corrected per chamber so that the wall flux equals the chamber's
   dV/dt (the staircase capture is a gate); ostia = each heart root's network flow spread uniformly over the faces to
   that vessel (samples within 25 mm of the root); downstream of the aortic / pulmonary valve plane the vessel owns the
   voxel; mitral / tricuspid faces open while the filling flow is positive; valve leaflet voxels are fluid of the
   nearest chamber; chambers are cleaned (opening of radius ≈ 2 mm, largest component).
Voxel velocity = blood fraction × blood velocity + (1 − blood fraction) × tissue velocity (air: 0).
Gates: per-component mass imbalance, staircase wall capture, mitral / tricuspid flux vs the filling model, chamber
speed (max, p99.9), vessel flux through planes moving with the vessel (relative velocity) vs the network Q(tc).
`--blood-export`: also blood_frames.hv (sparse blood voxels per frame, for the k-space simulator `simphantom`).
Writes <out_dir>/frames_geom.hv, flow_frames.hv, truth4d.hv (labels u8 [F, Z, Y, X], vel f16 [F, 3, Z, Y, X] m/s,
blood u8 [F, Z, Y, X] /255, tc, r, t), gates.json, figs/, and with --viewer-step a flow-viewer value (box-averaged).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fnn  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
sys.path.insert(0, str(HERE))
from hastycompute import value_format as vf  # noqa: E402
from motion import Breathing  # noqa: E402
from tissue_variation import modes as tex_modes  # noqa: E402

A = sys.argv[1:]
P0, MO, OUT = Path(A[0]), Path(A[1]), Path(A[2])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


PARAMS = json.loads((P0 / "params.json").read_text()) if (P0 / "params.json").exists() else {}
TPAR = json.loads((P0 / "tissue_params.json").read_text()) if (P0 / "tissue_params.json").exists() else {}
dev = torch.device("cuda")
t_start = time.time()
NF = int(opt("--frames", 20))
vox = float(opt("--vox", 1)) * 1e-3
crop = np.array([float(x) for x in opt("--crop", "-130,160,-125,95,1150,1470").split(",")]) * 1e-3
lo, hi = crop[0::2], crop[1::2]
dims = np.round((hi - lo) / vox).astype(int)
nx, ny, nz = (int(d) for d in dims)
origin = lo
center = lo + dims * vox / 2
O_t = torch.tensor(origin, device=dev, dtype=torch.float32)


# ── motion fields on the GPU ───────────────────────────────────────────────────────────────────────────────────────
class GField:
    """a vector / scalar field on a regular grid, trilinear at points (grid_sample, voxel centres at o + (i + ½) h)"""

    def __init__(self, F, o, h, padding, shape_xyz=None):
        self.F = F  # [C, Z, Y, X] float32 (cuda)
        self.o = torch.tensor(np.asarray(o, float), device=dev, dtype=torch.float32)
        sx = shape_xyz if shape_xyz is not None else (F.shape[3], F.shape[2], F.shape[1])
        self.n = torch.tensor(sx, device=dev, dtype=torch.float32)
        self.h = float(h)
        self.pad = padding

    def at(self, P, F=None):
        F = self.F if F is None else F
        g = 2 * (P - self.o) / (self.n * self.h) - 1
        out = Fnn.grid_sample(F[None], g.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode=self.pad, align_corners=False)
        return out.view(F.shape[0], -1).T


cm = vf.read(MO / "cardiac_motion.hv")
Tp = float(cm["period"])
U_all = torch.tensor(np.asarray(cm["U_mm"]), device=dev)  # [T, 3, Z, Y, X] float16 mm
NTU = U_all.shape[0]
GC = GField(None, cm["origin_m"], cm["voxel_m"], "zeros", (U_all.shape[4], U_all.shape[3], U_all.shape[2]))
rm = vf.read(MO / "resp_motion.hv")
GR = GField(torch.tensor(np.asarray(rm["B"]).sum(0), device=dev, dtype=torch.float32), rm["origin_m"], rm["voxel_m"], "border")


def cr_weights(tc, deriv=False):
    s = (tc % Tp) / Tp * NTU
    i = int(np.floor(s))
    f = s - i
    if not deriv:
        w = [(-f**3 + 2 * f**2 - f) / 2, (3 * f**3 - 5 * f**2 + 2) / 2, (-3 * f**3 + 4 * f**2 + f) / 2, (f**3 - f**2) / 2]
    else:
        w = [(-3 * f**2 + 4 * f - 1) / 2, (9 * f**2 - 10 * f) / 2, (-9 * f**2 + 8 * f + 1) / 2, (3 * f**2 - 2 * f) / 2]
        w = [x * NTU / Tp for x in w]
    return [((i - 1 + j) % NTU, w[j]) for j in range(4)]


class FrameMotion:
    """motion at one frame (tc, r, ṙ): positions / velocities of reference points, inverse map"""

    def __init__(self, tc, r, rdot):
        self.Uc = sum(float(w) * U_all[k].float() for k, w in cr_weights(tc)) * 1e-3
        self.Vc = sum(float(w) * U_all[k].float() for k, w in cr_weights(tc, True)) * 1e-3
        self.r, self.rdot = float(r), float(rdot)

    def position(self, X):
        Xc = X + GC.at(X, self.Uc)
        return Xc + self.r * GR.at(Xc) if self.r != 0 else Xc

    def velocity(self, X, dt=2e-3):
        Vc = GC.at(X, self.Vc)
        if self.r == 0 and self.rdot == 0:
            return Vc
        Xc = X + GC.at(X, self.Uc)
        return Vc + self.rdot * GR.at(Xc) + self.r * (GR.at(Xc + dt * Vc) - GR.at(Xc - dt * Vc)) / (2 * dt)

    def inverse(self, x, iters=12):
        X = x.clone()
        for _ in range(iters):
            X = x - (self.position(X) - X)
        return X


# ── frames ─────────────────────────────────────────────────────────────────────────────────────────────────────────
if opt("--realtime"):
    t0, t1 = (float(x) for x in opt("--realtime").split(","))
    tabs = np.linspace(t0, t1, NF, endpoint=False)
    bp = PARAMS.get("breathing", {})
    rr, rd = Breathing(int(opt("--seed", 0)), period=bp.get("period_s", 4.0), n=bp.get("n", 2), cv_period=bp.get("cv_period", 0.1), cv_depth=bp.get("cv_depth", 0.12))(tabs)
    frames = [(float(t % Tp), float(r_), float(d_), float(t)) for t, r_, d_ in zip(tabs, rr, rd)]
else:
    rs = float(opt("--resp-state", 0.0))
    frames = [(f * Tp / NF, rs, 0.0, f * Tp / NF) for f in range(NF)]
print(f"{len(frames)} frames on a {nx}×{ny}×{nz} grid of {vox * 1e3:.2f} mm ({nx * ny * nz / 1e6:.1f} M voxels)")

# ── 1. vessel geometry per frame → GPU voxelisation ────────────────────────────────────────────────────────────────
V = vf.read(P0 / "vessels.hv")
P_ref = np.asarray(V["points"]).astype(np.float64)
R_ref = np.asarray(V["radius"]).astype(np.float64)
cls = np.asarray(V["cls"])
net = np.load(P0 / "flow" / "network_flow.npz")
om = net["omega"]
p_h = net["p"]
qn = net["q"]
fac = np.select([cls == 2, cls == 3, cls == 4, cls == 6], [0.1, 0.35, 0.15, 0.1], 1.0)
EHR = (2.0e6 * np.exp(-2253.0 * R_ref) + 8.65e4) * fac
Pref_t = torch.tensor(P_ref, device=dev, dtype=torch.float32)
fp, fr, fw = [], [], []
for tc, r_, rdot, _ in frames:
    fm = FrameMotion(tc, r_, rdot)
    fp.append(fm.position(Pref_t).cpu().numpy())
    fw.append(fm.velocity(Pref_t).cpu().numpy())
    dp = np.real(2 * (p_h[:, 1:] @ np.exp(1j * om[1:] * tc)))
    fr.append(R_ref * (1 + 0.75 * dp / EHR))
fp, fr, fw = np.array(fp, np.float32), np.array(fr, np.float32), np.array(fw, np.float32)
vf.write(OUT / "frames_geom.hv", {"tc": np.array([f[0] for f in frames]), "r": np.array([f[1] for f in frames]), "points": fp, "radius": fr, "wvel": fw})
swing = float((fr[:, cls == 1].max(0) / fr[:, cls == 1].min(0) - 1).max())
print(f"vessel geometry: {time.time() - t_start:.0f}s; max arterial radius swing {swing * 100:.1f} %")
subprocess.run([str(REPO / "rustlib/target/release/phantomflow"), str(P0 / "vessels.hv"), str(OUT / "flow_frames.hv"), "--center", ",".join(f"{x:.6f}" for x in center),
                "--size", ",".join(f"{x:.6f}" for x in dims * vox), "--vox", f"{vox}", "--sub", opt("--sub", "3"), "--frames", str(OUT / "frames_geom.hv")],
               check=True, stdout=subprocess.DEVNULL)
FV = vf.read(OUT / "flow_frames.hv")
print(f"vessels voxelised: {time.time() - t_start:.0f}s")

# ── static data on the GPU ─────────────────────────────────────────────────────────────────────────────────────────
TS = vf.read(opt("--labels-src", str(P0 / "tissue_1mm.hv")))
tnames = list(TS["tissues"])
ID = {n: i for i, n in enumerate(tnames)}
LSRC = torch.tensor(np.asarray(TS["labels"]), device=dev)
LS_o = torch.tensor(np.asarray(TS["origin_m"], float), device=dev, dtype=torch.float32)
LS_h = float(TS["voxel_m"])
LS_n = torch.tensor(np.asarray(TS["labels"]).shape[::-1], device=dev)
CH = ["blood_LA", "blood_LV", "blood_RA", "blood_RV"]
CHID = torch.tensor([ID[c] for c in CH], device=dev)
# textures (per-phantom, resolution-free Fourier modes in the reference frame)
TEX = {}
for nm_, (sa_, sb_, l0_, l1_) in TPAR.get("texture", {}).items():
    if nm_ in ID and (sa_ > 0 or sb_ > 0):
        ka, aa, pa = tex_modes(TPAR["texture_seed"], ID[nm_], sa_, (l0_, l1_), which=0)
        kb, ab, pb = tex_modes(TPAR["texture_seed"], ID[nm_], sb_, (l0_, l1_), which=1)
        TEX[ID[nm_]] = [torch.tensor(x, device=dev, dtype=torch.float32) for x in (ka, aa, pa, kb, ab, pb)]
print(f"textures for {len(TEX)} tissue classes" if TEX else "no textures")
VCLS = {1: "blood_artery", 2: "blood_vein", 3: "blood_pulm_artery", 4: "blood_pulm_vein", 5: "blood_artery", 6: "blood_vein"}
tq = np.arange(128) * Tp / 128
Q_MV, Q_TV = np.asarray(cm["Q_MV"]), np.asarray(cm["Q_TV"])
QF = {k_: np.asarray(cm[k_]) for k_ in ("Q_MV", "Q_TV", "Q_AO", "Q_PA", "Q_PV", "Q_VR")}
tmot = np.asarray(cm["t"])
Vmot = {c: np.asarray(cm[f"V_{c}"]) for c in CH}
iz_, iy_, ix_ = torch.meshgrid(*(torch.arange(n, device=dev) for n in (nz, ny, nx)), indexing="ij")
XG = torch.stack([ix_, iy_, iz_], -1).float() * vox + O_t + vox / 2  # [Z, Y, X, 3] world (x, y, z)
del iz_, iy_, ix_

# heart roots
EXP = P0 if (P0 / "graph.npz").exists() else REPO / "data/hasty/external/zanatomy/export"  # per-phantom graph (radius jitter)
G = np.load(EXP / "graph.npz")
GJ = json.loads((EXP / "graph.json").read_text())
eoff_g, ea_g, eb_g = G["edge_offsets"], G["edge_a"], G["edge_b"]
enames = GJ["edge_names"]
CHAMBER_OF = {"systemic_arterial": "blood_LV", "pulmonary_arterial": "blood_RV", "systemic_venous": "blood_RA", "pulmonary_venous": "blood_LA"}
roots = []
for c in GJ["components"]:
    if c["class"] not in CHAMBER_OF:
        continue
    for rn in c.get("roots", []):
        for k in range(len(ea_g)):
            if ea_g[k] == rn or eb_g[k] == rn:
                s0 = int(eoff_g[k]) if ea_g[k] == rn else int(eoff_g[k + 1] - 1)
                s1 = s0 + 2 if ea_g[k] == rn else s0 - 2
                # a heart root lies at a chamber (fragments' roots are fed by their main tree, not by the heart)
                ic = np.floor((P_ref[s0] - np.asarray(TS["origin_m"], float)) / LS_h).astype(int)
                rv_ = int(round(10e-3 / LS_h))
                cube = np.asarray(TS["labels"])[max(ic[2] - rv_, 0) : ic[2] + rv_ + 1, max(ic[1] - rv_, 0) : ic[1] + rv_ + 1, max(ic[0] - rv_, 0) : ic[0] + rv_ + 1]
                if not np.isin(cube, [ID[c_] for c_ in CH]).any():
                    continue
                lut = torch.zeros(len(P_ref) + 1, dtype=torch.bool, device=dev)  # last entry: "no sample"
                near = np.flatnonzero((np.linalg.norm(P_ref - P_ref[s0], axis=1) < 25e-3) & (cls == cls[s0]))
                lut[torch.tensor(near, device=dev)] = True
                roots.append({"name": enames[k], "chamber": CHAMBER_OF[c["class"]], "s0": s0, "s1": s1, "q_in": -qn[s0] if ea_g[k] == rn else qn[s0], "lut": lut,
                              "valve": c["class"] in ("systemic_arterial", "pulmonary_arterial")})
print("heart roots:", [(r["name"][:28], r["chamber"]) for r in roots])
GATE_V = ["Thoracic aorta", "Brachiocephalic trunk", "Left common carotid artery", "Right pulmonary artery", "Inferior vena cava (thoracic part)", "Superior vena cava"]
gate_samples = {}
for nm in GATE_V:
    ks = [k for k in range(len(enames)) if enames[k] == nm]
    if ks:
        k = max(ks, key=lambda k: eoff_g[k + 1] - eoff_g[k])
        gate_samples[nm] = int((eoff_g[k] + eoff_g[k + 1]) // 2)


# ── GPU helpers ────────────────────────────────────────────────────────────────────────────────────────────────────
SIDES = [(ax, sh) for ax in range(3) for sh in (-1, 1)]


def nbr(a, ax, sh):
    """neighbour value at offset sh (±1) along array axis ax; 0 / False beyond the grid"""
    out = torch.zeros_like(a)
    n = a.shape[ax]
    if sh == 1:
        out.narrow(ax, 0, n - 1).copy_(a.narrow(ax, 1, n - 1))
    else:
        out.narrow(ax, 1, n - 1).copy_(a.narrow(ax, 0, n - 1))
    return out


def comp_of(ax):
    return 2 - ax  # array axis (z, y, x) → velocity component (x, y, z)


def erode(m, r):
    x = m.float()[None, None]
    for _ in range(r):
        x = -Fnn.max_pool3d(-x, 3, 1, 1)
    return x[0, 0] > 0.5


def dilate(m, r):
    x = m.float()[None, None]
    for _ in range(r):
        x = Fnn.max_pool3d(x, 3, 1, 1)
    return x[0, 0] > 0.5


def bbox(m, pad=2):
    nzr = torch.nonzero(m)
    if len(nzr) == 0:
        return None
    lo_ = (nzr.min(0).values - pad).clamp(min=0)
    hi_ = torch.minimum(nzr.max(0).values + pad + 1, torch.tensor(m.shape, device=dev))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo_, hi_))


def largest_component(m):
    bb = bbox(m, 1)
    if bb is None:
        return m
    lab, n = ndimage.label(m[bb].cpu().numpy())
    if n <= 1:
        return m
    keep = lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
    out = torch.zeros_like(m)
    out[bb] = torch.tensor(keep, device=dev)
    return out


def cg(Aop, b, M, rtol=1e-7, maxiter=40000):
    x = torch.zeros_like(b)
    r = b.clone()
    z = r / M
    p = z.clone()
    rz = (r * z).sum()
    bn = b.norm().clamp(min=1e-30)
    it = 0
    for it in range(maxiter):
        Ap = Aop(p)
        alpha = rz / (p * Ap).sum()
        x += alpha * p
        r -= alpha * Ap
        if it % 25 == 0 and float(r.norm() / bn) < rtol:
            break
        z = r / M
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it, float(r.norm() / bn)


def potential_flow(Fl, g, wall):
    """v = ∇φ on the fluid mask Fl (cropped tensors). g[(ax, sh)]: outward normal velocity on boundary faces,
    wall[(ax, sh)]: wall faces (absorb each component's imbalance). Returns φ, velocity [Z, Y, X, 3], imbalance."""
    h2 = vox * vox
    nbF = {s: nbr(Fl, *s) for s in SIDES}
    b = torch.zeros(Fl.shape, device=dev, dtype=torch.float64)
    nwall = torch.zeros(Fl.shape, device=dev, dtype=torch.float64)
    deg = torch.zeros(Fl.shape, device=dev, dtype=torch.float64)
    for s in SIDES:
        bnd = Fl & ~nbF[s]
        b += torch.where(bnd, g[s].double(), 0.0) * h2
        nwall += (bnd & wall[s]).double()
        deg += (Fl & nbF[s]).double()
    lab, ncomp = ndimage.label(Fl.cpu().numpy())
    labt = torch.tensor(lab, device=dev).view(-1)
    imb = torch.zeros(ncomp + 1, device=dev, dtype=torch.float64).index_add_(0, labt, b.view(-1))
    cnt = torch.zeros(ncomp + 1, device=dev, dtype=torch.float64).index_add_(0, labt, nwall.view(-1))
    corr = torch.where(cnt > 0, imb / cnt.clamp(min=1), 0.0)  # m³/s per wall face
    corr[0] = 0
    labt = labt.view(Fl.shape)
    bc = torch.where(Fl, b - corr[labt] * nwall, 0.0)
    tot = torch.zeros(ncomp + 1, device=dev, dtype=torch.float64).index_add_(0, labt.view(-1), bc.view(-1))
    nvc = torch.zeros(ncomp + 1, device=dev, dtype=torch.float64).index_add_(0, labt.view(-1), Fl.double().view(-1))
    bc = torch.where(Fl, bc - (tot / nvc.clamp(min=1))[labt], 0.0)  # exact zero sum per component
    inner = {s: Fl & nbF[s] for s in SIDES}

    def Aop(x):  # Σ_j (φ_i − φ_j) h over fluid neighbours
        y = torch.zeros_like(x)
        for s in SIDES:
            y += torch.where(inner[s], x - nbr(x, *s), 0.0)
        return y * vox

    M = torch.where(Fl, deg * vox, 1.0).clamp(min=vox)
    phi, its, res = cg(Aop, bc, M)
    vel = torch.zeros(Fl.shape + (3,), device=dev, dtype=torch.float64)
    cv = corr[labt] / h2
    for ax in range(3):
        acc = torch.zeros(Fl.shape, device=dev, dtype=torch.float64)
        for sh in (-1, 1):
            s = (ax, sh)
            u_out = torch.where(nbF[s], (nbr(phi, *s) - phi) / vox, g[s].double() - torch.where(wall[s], cv, 0.0))
            acc += 0.5 * u_out * sh
        vel[..., comp_of(ax)] = torch.where(Fl, acc, 0.0)
    return phi, vel, float(imb[1:].sum()), its, res


def face_flux(phi, A_, B_):
    tot = 0.0
    for s in SIDES:
        f_ = A_ & nbr(B_, *s)
        tot += float(((nbr(phi, *s) - phi) / vox)[f_].sum()) * vox * vox
    return tot


# ── per frame ──────────────────────────────────────────────────────────────────────────────────────────────────────
NFR = len(frames)
BLOOD_ONLY = "--blood-only" in A  # only the sparse blood frames (k-space simulator input), no dense truth
if BLOOD_ONLY and "--blood-export" not in A:
    A.append("--blood-export")
FS = 1 if BLOOD_ONLY else NFR
labels4d = np.zeros((FS, nz, ny, nx), np.uint8)
vel4d = np.zeros((FS, 3, nz, ny, nx), np.float16)
blood4d = np.zeros((FS, nz, ny, nx), np.uint8)
tex4d = np.zeros((FS, 2, nz, ny, nx), np.float16) if TEX else None
blood_frames = []
gates = []
r_open = max(1, int(round(2.0e-3 / vox)))  # chamber cleaning radius ≈ 2 mm at every grid
nvox = nz * ny * nx
ONLY = {int(x) for x in opt("--only").split(",")} if opt("--only") else None
for f, (tc, r_, rdot, tabs_f) in enumerate(frames):
    if ONLY is not None and f not in ONLY:
        continue
    tf = time.time()
    fm = FrameMotion(tc, r_, rdot)
    # 2. tissue: pull labels back, tissue velocity
    xs = XG.view(-1, 3)
    L = torch.zeros(len(xs), dtype=torch.uint8, device=dev)
    vt = torch.zeros((len(xs), 3), device=dev)
    if TEX:
        TA = torch.zeros(len(xs), dtype=torch.float16, device=dev)
        TB = torch.zeros(len(xs), dtype=torch.float16, device=dev)
    for a in range(0, len(xs), 1 << 22):
        x = xs[a : a + (1 << 22)]
        X = fm.inverse(x)
        i = torch.floor((X - LS_o) / LS_h).long()
        ok = ((i >= 0) & (i < LS_n)).all(1)
        Li = torch.zeros(len(x), dtype=torch.uint8, device=dev)
        Li[ok] = LSRC[i[ok, 2], i[ok, 1], i[ok, 0]]
        L[a : a + len(x)] = Li
        vt[a : a + len(x)] = fm.velocity(X)
        if TEX:
            for t_, (ka, aa, pa, kb, ab, pb) in TEX.items():
                sel = Li == t_
                if bool(sel.any()):
                    Xs = X[sel]
                    TA[a : a + len(x)][sel] = (torch.cos(Xs @ ka.T + pa) @ aa).half()
                    TB[a : a + len(x)][sel] = (torch.cos(Xs @ kb.T + pb) @ ab).half()
    L = L.view(nz, ny, nx)
    vt = vt.view(nz, ny, nx, 3)
    air = (L == ID["air"]) | (L == ID["airway"])
    vt[air] = 0
    # 1. vessels of this frame (sparse → dense)
    fr_ = FV["frames"][f]
    idx = torch.tensor(np.asarray(fr_["idx"]).astype(np.int64), device=dev)

    def dense(vals, shape_tail=(), dtype=torch.float32, fill=0):
        out = torch.full((nvox,) + shape_tail, fill, device=dev, dtype=dtype)
        out[idx] = torch.tensor(np.asarray(vals), device=dev).to(dtype)
        return out.view((nz, ny, nx) + shape_tail)

    Bf = dense(fr_["blood"])
    VB = dense(fr_["vel"], (3,))
    CL = dense(fr_["cls"], dtype=torch.uint8)
    smp = np.asarray(fr_["sample"]).astype(np.int64)
    smp[smp == 0xFFFFFFFF] = len(P_ref)
    S_ = dense(smp, dtype=torch.long, fill=len(P_ref))
    WL = dense(fr_["wall"])
    Pf = torch.tensor(fp[f], device=dev, dtype=torch.float32)
    # outflow tracts: downstream of the aortic / pulmonary valve plane the vessel owns the voxel
    for r in roots:
        if not r["valve"]:
            continue
        pv = Pf[r["s0"]]
        nv = Pf[r["s1"]] - pv
        nv = nv / nv.norm()
        own = r["lut"][S_] & (Bf >= 0.5) & (((XG - pv) @ nv) > 0) & (L == ID[r["chamber"]])
        L[own] = ID["blood_artery"] if r["chamber"] == "blood_LV" else ID["blood_pulm_artery"]
    # chamber labels for the flow solve: valve leaflets → nearest chamber, cleaning
    Lc = L.clone()
    chm = torch.isin(Lc, CHID)
    unass = Lc == ID["valve"]
    for _ in range(8):
        if not bool(unass.any()):
            break
        for s in SIDES:
            nl = nbr(torch.where(chm, Lc, torch.zeros_like(Lc)), *s)
            take = unass & torch.isin(nl, CHID)
            Lc[take] = nl[take]
            unass &= ~take
        chm = torch.isin(Lc, CHID)
    for k_ in CHID.tolist():
        mk = Lc == k_
        if not bool(mk.any()):
            continue
        keep = largest_component(dilate(erode(mk, r_open), r_open) & mk)
        Lc[mk & ~keep] = ID["myocardium"]
    chamber = torch.isin(Lc, CHID)
    # final labels: vessels over tissue, not inside the chambers
    Lf = L.clone()
    isb = (Bf >= 0.5) & ~chamber
    for c_, n_ in VCLS.items():
        Lf[isb & (CL == c_)] = ID[n_]
    Lf[(Bf < 0.5) & (WL >= 0.5) & ~chamber] = ID["vessel_wall"]
    vessel_blood = (Bf >= 0.5) & ~chamber
    vbm = VB / Bf.clamp(min=1e-6)[..., None]
    # 3. boundary data: walls (tissue velocity), ostia (root flows)
    g, wall = {}, {}
    for s in SIDES:
        c3 = comp_of(s[0])
        g[s] = 0.5 * (vt[..., c3] + nbr(vt[..., c3], *s)) * s[1]
        wall[s] = torch.ones((nz, ny, nx), dtype=torch.bool, device=dev)
    ost_acc = {s: torch.zeros((nz, ny, nx), device=dev) for s in SIDES}
    ost_report = []
    for r in roots:
        qin = float(np.real(r["q_in"][0] + 2 * (r["q_in"][1:] @ np.exp(1j * om[1:] * tc))))
        ch = Lc == ID[r["chamber"]]
        near_v = vessel_blood & r["lut"][S_]
        fcs = [(s, ch & nbr(near_v, *s)) for s in SIDES]
        nfc = sum(int(m_.sum()) for _, m_ in fcs)
        if nfc == 0:  # no contact: the 20 chamber boundary faces nearest to the root
            pr = Pf[r["s0"]]
            allf = []
            for s in SIDES:
                bm = ch & ~nbr(ch, *s)
                pts = torch.nonzero(bm)
                d_ = (XG[bm] - pr).norm(dim=1)
                allf += [(float(d_[i_]), s, tuple(pts[i_].tolist())) for i_ in torch.argsort(d_)[:20].tolist()]
            allf.sort(key=lambda x: x[0])
            fcs = []
            for _, s, (z_, y_, x_) in allf[:20]:
                m_ = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
                m_[z_, y_, x_] = True
                fcs.append((s, m_))
            nfc = 20
        gout = -qin / (nfc * vox * vox)
        for s, m_ in fcs:
            ost_acc[s] += torch.where(m_, gout, 0.0)
            wall[s] &= ~m_
        ost_report.append((r["name"][:24], round(qin * 1e6, 1), nfc))
    for s in SIDES:
        g[s] = torch.where(wall[s], g[s], ost_acc[s])
    qmv, qtv = float(np.interp(tc, tq, Q_MV, period=Tp)), float(np.interp(tc, tq, Q_TV, period=Tp))
    open_mv, open_tv = qmv > 1e-6, qtv > 1e-6
    groups = [["blood_LA", "blood_LV"]] if open_mv else [["blood_LA"], ["blood_LV"]]
    groups += [["blood_RA", "blood_RV"]] if open_tv else [["blood_RA"], ["blood_RV"]]
    qi = {k_: float(np.interp(tc, tq, QF[k_], period=Tp)) for k_ in QF}  # the flows that define the volume curves
    dvdt = {"blood_LV": qi["Q_MV"] - qi["Q_AO"], "blood_LA": qi["Q_PV"] - qi["Q_MV"], "blood_RV": qi["Q_TV"] - qi["Q_PA"], "blood_RA": qi["Q_VR"] - qi["Q_TV"]}
    VCH = torch.zeros((nz, ny, nx, 3), device=dev)
    PHI = torch.zeros((nz, ny, nx), device=dev, dtype=torch.float64)
    comps = []
    for grp in groups:
        fl_full = torch.isin(Lc, torch.tensor([ID[c] for c in grp], device=dev))
        bb = bbox(fl_full, 2)
        if bb is None or int(fl_full.sum()) < 8:
            continue
        Fl = fl_full[bb]
        gg = {s: g[s][bb].clone() for s in SIDES}
        ww = {s: wall[s][bb] for s in SIDES}
        cap = {}
        for c in grp:  # per chamber: walls carry exactly dV/dt
            msk = {s: (Lc[bb] == ID[c]) & ~nbr(Fl, *s) & ww[s] for s in SIDES}
            W = sum(float(gg[s][msk[s]].sum()) for s in SIDES) * vox * vox
            N = sum(int(msk[s].sum()) for s in SIDES)
            if N:
                for s in SIDES:
                    gg[s] = torch.where(msk[s], gg[s] + (dvdt[c] - W) / (N * vox * vox), gg[s])
            cap[c] = (round(W * 1e6, 1), round(dvdt[c] * 1e6, 1))
        phi, vel_, imb, its, res = potential_flow(Fl, gg, ww)
        VCH[bb] = torch.where(Fl[..., None], vel_.float(), VCH[bb])
        PHI[bb] = torch.where(Fl, phi, PHI[bb])
        inflow = sum(float(torch.where(Fl & ~nbr(Fl, *s) & ~ww[s], -gg[s], 0.0).sum()) for s in SIDES) * vox * vox
        comps.append({"chambers": grp, "imbalance_ml_s": imb * 1e6, "ostia_inflow_ml_s": inflow * 1e6, "staircase_wall_flux_vs_dVdt_ml_s": cap, "cg_iters": its, "cg_res": res})
    mv_area = sum(int(((Lc == ID["blood_LA"]) & nbr(Lc == ID["blood_LV"], *s_)).sum()) for s_ in SIDES) * vox * vox
    tv_area = sum(int(((Lc == ID["blood_RA"]) & nbr(Lc == ID["blood_RV"], *s_)).sum()) for s_ in SIDES) * vox * vox
    mv_flux = face_flux(PHI, Lc == ID["blood_LA"], Lc == ID["blood_LV"]) if open_mv else 0.0
    tv_flux = face_flux(PHI, Lc == ID["blood_RA"], Lc == ID["blood_RV"]) if open_tv else 0.0
    # assemble
    bfrac = torch.where(chamber, torch.ones_like(Bf), Bf)
    vblood = torch.where(chamber[..., None], VCH, vbm)
    vel = bfrac[..., None] * vblood + (1 - bfrac)[..., None] * vt
    vel[air & (Bf <= 0)] = 0
    spd = VCH.norm(dim=-1)[chamber]
    if "--debug" in A:  # dump the boundary data around the fastest chamber voxel
        sp_all = torch.where(chamber, VCH.norm(dim=-1), 0.0)
        zc_, yc_, xc_ = np.unravel_index(int(sp_all.argmax()), sp_all.shape)
        sl_ = (slice(max(zc_ - 8, 0), zc_ + 9), slice(max(yc_ - 8, 0), yc_ + 9), slice(max(xc_ - 8, 0), xc_ + 9))
        np.savez(OUT / f"debug_frame{f}.npz", Lc=Lc[sl_].cpu().numpy(), L=L[sl_].cpu().numpy(), spd=sp_all[sl_].cpu().numpy(), PHI=PHI[sl_].cpu().numpy(),
                 VCH=VCH[sl_].cpu().numpy(), vt=vt[sl_].cpu().numpy(), Bf=Bf[sl_].cpu().numpy(), center=np.array([zc_, yc_, xc_]),
                 **{f"g_{a_}_{b_}": g[(a_, b_)][sl_].cpu().numpy() for a_, b_ in SIDES}, **{f"w_{a_}_{b_}": wall[(a_, b_)][sl_].cpu().numpy() for a_, b_ in SIDES})
    # vessel gates: flux through a plane moving with the vessel (velocity relative to the wall) vs network Q(tc)
    vg = {}
    GVB = GField(VB.permute(3, 0, 1, 2).contiguous(), origin, vox, "zeros")
    GBF = GField(Bf[None], origin, vox, "zeros")
    for nm, sm in gate_samples.items():
        c0 = Pf[sm]
        tt_ = Pf[sm + 2] - Pf[sm - 2]
        tt_ = tt_ / tt_.norm()
        ref = torch.tensor([1.0, 0, 0], device=dev) if abs(float(tt_[0])) < 0.9 else torch.tensor([0, 1.0, 0], device=dev)
        u = torch.linalg.cross(tt_, ref)
        u = u / u.norm()
        w2 = torch.linalg.cross(tt_, u)
        rad = float(fr[f, sm])
        ds = min(vox / 3, rad / 10)
        ax_ = torch.arange(-1.2 * rad, 1.2 * rad + ds / 2, ds, device=dev)
        U_, W_ = torch.meshgrid(ax_, ax_, indexing="ij")
        m_ = U_**2 + W_**2 <= (1.2 * rad) ** 2
        pts = c0 + U_[m_, None] * u + W_[m_, None] * w2
        if bool(((pts < O_t + vox) | (pts > O_t + torch.tensor(dims * vox, device=dev, dtype=torch.float32) - vox)).any()):
            continue
        wv = torch.tensor(fw[f, sm], device=dev)
        flux = float(((GVB.at(pts) - GBF.at(pts) * wv) @ tt_).sum()) * ds * ds  # blood velocity relative to the wall
        qref = float(np.real(qn[sm, 0] + 2 * (qn[sm, 1:] @ np.exp(1j * om[1:] * tc))))
        vg[nm] = (round(flux * 1e6, 2), round(qref * 1e6, 2))
    if "--blood-export" in A:  # sparse blood voxels for the k-space simulator (simphantom)
        bsel = (bfrac > 0.02).view(-1)
        bidx = torch.nonzero(bsel).squeeze(1)
        blood_frames.append({"tc": tc, "r": r_, "idx": bidx.int().cpu().numpy(), "frac": (bfrac.view(-1)[bidx] * 255).round().to(torch.uint8).cpu().numpy(),
                             "vel": vblood.view(-1, 3)[bidx].half().cpu().numpy(), "label": Lf.view(-1)[bidx].cpu().numpy()})
    fs_ = min(f, FS - 1)
    labels4d[fs_] = Lf.cpu().numpy()
    vel4d[fs_] = vel.permute(3, 0, 1, 2).cpu().numpy().astype(np.float16)
    blood4d[fs_] = (bfrac * 255).round().to(torch.uint8).cpu().numpy()
    if TEX:
        nob = (bfrac < 0.5).view(-1)
        tex4d[fs_, 0] = torch.where(nob, TA, torch.zeros_like(TA)).view(nz, ny, nx).cpu().numpy()
        tex4d[fs_, 1] = torch.where(nob, TB, torch.zeros_like(TB)).view(nz, ny, nx).cpu().numpy()
    sp_sub = spd[:: max(1, len(spd) // 2000000)]
    gates.append({"tc": tc, "r": r_, "t": tabs_f, "ostia": ost_report, "Q_MV_ml_s": qmv * 1e6, "mitral_flux_ml_s": mv_flux * 1e6, "Q_TV_ml_s": qtv * 1e6,
                  "tricuspid_flux_ml_s": tv_flux * 1e6, "mitral_face_area_cm2": mv_area * 1e4, "tricuspid_face_area_cm2": tv_area * 1e4, "components": comps, "chamber_volumes_ml": {c: float((L == ID[c]).sum()) * vox**3 * 1e6 for c in CH},
                  "chamber_speed_m_s": {"max": float(spd.max()), "p999": float(torch.quantile(sp_sub, 0.999))},
                  "vessel_flux_ml_s": vg, "secs": time.time() - tf})
    caps = [round(v[0] / v[1], 2) for c in comps for v in c["staircase_wall_flux_vs_dVdt_ml_s"].values() if abs(v[1]) > 30]
    print(f"frame {f:2d} tc {tc * 1e3:4.0f} ms r {r_:.2f}: MV {qmv * 1e6:6.1f} vs {mv_flux * 1e6:6.1f} ({mv_area * 1e4:.1f} cm²), TV {qtv * 1e6:6.1f} vs {tv_flux * 1e6:6.1f} mL/s, "
          f"imbalance {[round(c['imbalance_ml_s'], 1) for c in comps]}, capture {caps}, CG {[c['cg_iters'] for c in comps]}, "
          f"chamber max {gates[-1]['chamber_speed_m_s']['max']:.2f} p99.9 {gates[-1]['chamber_speed_m_s']['p999']:.2f} m/s, vessels {vg} ({time.time() - tf:.0f}s)", flush=True)
    del L, vt, Bf, VB, CL, S_, Lc, chamber, vel, VCH, PHI, g, wall, ost_acc
    torch.cuda.empty_cache()

if ONLY is not None:
    print("--only: frames", sorted(ONLY), "processed; no outputs written")
    sys.exit(0)
if blood_frames:
    vf.write(OUT / "blood_frames.hv", {"frames": blood_frames, "origin_m": origin.tolist(), "voxel_m": vox, "dims_xyz": [nx, ny, nz], "period": Tp, "tissues": tnames,
                                        "note": "per frame: blood voxels (fraction > 0.02): idx (z·ny·nx + y·nx + x), frac u8 /255, vel f16 blood-mean m/s, label (tissue table index)"})
    print(f"blood frames: {sum(len(b['idx']) for b in blood_frames) / len(blood_frames) / 1e6:.2f} M voxels per frame")
if BLOOD_ONLY:
    (OUT / "gates.json").write_text(json.dumps({"frames": gates, "grid": {"origin": origin.tolist(), "voxel": vox, "dims": [nx, ny, nz]}}, indent=1))
    print(f"blood-only: wrote {OUT} ({time.time() - t_start:.0f}s)")
    sys.exit(0)
vf.write(OUT / "truth4d.hv", {"labels": labels4d, "vel": vel4d, "blood": blood4d, "tissues": tnames, "tc": np.array([f[0] for f in frames]), "r": np.array([f[1] for f in frames]),
                               "t": np.array([f[3] for f in frames]), "origin_m": origin.tolist(), "voxel_m": vox, "dims_xyz": [nx, ny, nz], "period": Tp,
                               **({"tex": tex4d} if TEX else {}), "tissue_params": json.dumps(TPAR) if TPAR else "", "params": json.dumps(PARAMS) if PARAMS else "",
                               "note": "vel: voxel-mean velocity (blood fraction × blood + rest × tissue), m/s, float16 [F, 3, Z, Y, X]; blood: u8 fraction × 255; labels: tissue + vessels; tex: [F, 2 (a: PD, b: T2*/T1), Z, Y, X] texture values (tissue_variation.apply with tissue_params.table); Z-Anatomy CC BY-SA 4.0"})
(OUT / "gates.json").write_text(json.dumps({"frames": gates, "grid": {"origin": origin.tolist(), "voxel": vox, "dims": [nx, ny, nz]}}, indent=1))

tab = TPAR.get("table") or {n_: {"pd": float(TS["pd"][i_]), "t1_s": float(TS["t1_s"][i_]), "t2s_s": float(TS["t2s_s"][i_])} for i_, n_ in enumerate(tnames)}
PDt = np.array([tab[n_]["pd"] for n_ in tnames])
T1t = np.array([tab[n_]["t1_s"] for n_ in tnames])
T2t = np.array([max(tab[n_]["t2s_s"], 1e-4) for n_ in tnames])
inflow = np.ones(len(tnames))
for n_ in ("blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein", "blood_LA", "blood_LV", "blood_RA", "blood_RV"):
    inflow[ID[n_]] = 1.5

def spgr(f_):  # TR 5 ms, 15°, TE 2.5 ms, with the textures
    L_ = labels4d[f_]
    ta, tb = (tex4d[f_, 0].astype(np.float32), tex4d[f_, 1].astype(np.float32)) if TEX else (0.0, 0.0)
    pd, t1, t2s = PDt[L_] * np.clip(1 + ta, 0.2, 2), T1t[L_] * np.clip(1 + tb / 3, 0.5, 1.5), T2t[L_] * np.clip(1 + tb, 0.2, 2)
    e1 = np.exp(-5e-3 / t1)
    return pd * inflow[L_] * np.sin(np.deg2rad(15)) * (1 - e1) / (1 - np.cos(np.deg2rad(15)) * e1) * np.exp(-2.5e-3 / t2s)


fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
m0 = spgr(0)
iz0, iy0, ix0 = [int(np.argmin(np.abs(origin[a] + (np.arange(n) + 0.5) * vox - c))) for a, n, c in ((2, nz, 1.29), (1, ny, -0.03), (0, nx, 0.03))]
for a_, (img, ttl) in zip(axs, ((m0[iz0], "axial"), (m0[:, iy0, :][::-1], "coronal"), (m0[:, :, ix0][::-1], "sagittal"))):
    a_.imshow(img, cmap="gray", vmin=0, vmax=float(np.percentile(m0[m0 > 0], 99.5)) if (m0 > 0).any() else 1)
    a_.set_title(f"spoiled GRE (TR 5 ms, 15°, TE 2.5 ms) — {ttl}; contrast {TPAR.get('contrast', 'native')}, texture gain {TPAR.get('texture_gain', 0):.2f}", fontsize=8)
    a_.axis("off")
fig.savefig(OUT / "figs" / "03_magnitude.png", dpi=90, bbox_inches="tight")
plt.close(fig)

# ── figures ────────────────────────────────────────────────────────────────────────────────────────────────────────
tcs = np.array([g_["tc"] for g_ in gates])
fig, axs = plt.subplots(1, 3, figsize=(21, 5))
axs[0].plot(tq * 1e3, Q_MV * 1e6, "k-", label="Q_MV (filling model)")
axs[0].plot(tcs * 1e3, [g_["mitral_flux_ml_s"] for g_ in gates], "ro", label="mitral flux (chamber flow)")
axs[0].plot(tq * 1e3, Q_TV * 1e6, "k--", label="Q_TV")
axs[0].plot(tcs * 1e3, [g_["tricuspid_flux_ml_s"] for g_ in gates], "bs", label="tricuspid flux")
axs[0].legend(fontsize=8)
axs[0].set_title("AV-valve flow: model vs potential-flow solution [mL/s]")
for c in CH:
    axs[1].plot(tcs * 1e3, [g_["chamber_volumes_ml"][c] for g_ in gates], "o", label=c)
    axs[1].plot(tmot * 1e3, Vmot[c] * 1e6, "k:", lw=0.8)
axs[1].set_title("chamber volumes of the pulled-back labels (dots) vs targets (dotted) [mL]")
axs[1].legend(fontsize=8)
for nm in gate_samples:
    pts_ = [(g_["tc"], *g_["vessel_flux_ml_s"][nm]) for g_ in gates if nm in g_["vessel_flux_ml_s"]]
    if pts_:
        pts_ = np.array(sorted(pts_))
        l_ = axs[2].plot(pts_[:, 0] * 1e3, pts_[:, 2], "-", lw=1)[0]
        axs[2].plot(pts_[:, 0] * 1e3, pts_[:, 1], "o", color=l_.get_color(), label=nm[:26])
axs[2].set_title("vessel flux through planes moving with the vessel (dots) vs network Q (lines) [mL/s]")
axs[2].legend(fontsize=7)
for a_ in axs:
    a_.set_xlabel("cardiac time [ms]")
fig.savefig(OUT / "figs" / "01_gates.png", dpi=90, bbox_inches="tight")
plt.close(fig)
fpk = int(np.argmax([g_["chamber_speed_m_s"]["p999"] for g_ in gates]))
i_es = int(np.argmin([g_["chamber_volumes_ml"]["blood_LV"] for g_ in gates]))
for f, tl in ((0, "first frame"), (i_es, "end-systole"), (fpk, "peak chamber flow")):
    spd = np.linalg.norm(vel4d[f].astype(np.float32), axis=0)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    iz, iy, ix = [int(np.argmin(np.abs(origin[a] + (np.arange(n) + 0.5) * vox - c))) for a, n, c in ((2, nz, 1.30), (1, ny, -0.03), (0, nx, 0.03))]
    for a_, (img, ttl) in zip(axs, ((spd[iz], "axial"), (spd[:, iy, :][::-1], "coronal"), (spd[:, :, ix][::-1], "sagittal"))):
        im = a_.imshow(img, cmap="inferno", vmin=0, vmax=1.0)
        a_.set_title(f"{tl} (tc {gates[f]['tc'] * 1e3:.0f} ms, r {gates[f]['r']:.2f}): |v| {ttl}")
        a_.axis("off")
    fig.colorbar(im, ax=axs, fraction=0.02, label="m/s")
    fig.savefig(OUT / "figs" / f"02_speed_{tl.replace(' ', '_')}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)

if ONLY is not None:
    sys.exit(0)
if opt("--viewer-step"):
    k = int(opt("--viewer-step"))
    cz, cy, cx = nz // k, ny // k, nx // k

    def box(a):
        a = a[..., : cz * k, : cy * k, : cx * k]
        return a.reshape(a.shape[:-3] + (cz, k, cy, k, cx, k)).mean(axis=(-5, -3, -1))

    isbl = blood4d >= 128
    b_ = box(isbl.astype(np.float32))
    vb_ = np.stack([box(vel4d[:, c].astype(np.float32) * isbl) for c in range(3)], 1)
    vb_ = np.where(b_[:, None] >= 0.25, vb_ / np.maximum(b_, 1e-6)[:, None], 0).astype(np.float32)
    mag = np.stack([box(spgr(f_)) for f_ in range(NFR)]).astype(np.float32)
    fv = {"kind": "flow4d", "vel": vb_, "mag": mag, "spacing": [vox * k] * 3, "dt": float(np.diff([f[3] for f in frames]).mean()) if NFR > 1 else Tp,
          "venc": 1.5, "mask": (b_.max(0) >= 0.5).astype(np.uint8), "origin_m": origin.tolist()}
    vf.write(OUT / "viewer_flow.hv", fv)
    if "--publish" in A:
        from hastycompute.v2 import Client  # noqa: E402

        cli = Client(opt("--server"))
        i = cli.put(fv, name=f"phantom P1 {'real time' if opt('--realtime') else 'gated'} {vox * k * 1e3:.1f} mm (truth {vox * 1e3:.1f} mm)", metadata={"kind": "flow4d", "phantom": str(OUT)})
        cli.show(i, "flow")
        print("viewer:", cli.viewer_url(i, "flow"))
print(f"wrote {OUT} ({time.time() - t_start:.0f}s)")
