"""Cardiac kinematics of the atlas phantom: a displacement field u(X, t) = Σ_k a_k(t) B_k(X) whose chamber volumes
follow volume curves that are mass-consistent with the 1D network.

    python -I cardiac_motion.py <tissue_2mm.hv> <tissues.json> <export_dir> <flow_dir> <out_dir>
        [--dav 14] [--twist 9] [--la-max 65] [--ra-max 80] [--fe 0.65] [--samples 48]

Reference configuration X = the atlas pose. Basis fields (on a 2 mm grid around the heart, smoothed, all multiplied
by a taper T that is 1 in the heart and falls to 0 over 25 mm outside it):
- av:    AV-plane descent along the LV long axis (unit: metres of annulus displacement); ventricles shorten linearly
         from apex (fixed) to base, the atria stretch from the AV plane to their roof (50 mm above).
- twist: rotation about the long axis, apex +1, base −0.4 (unit: radians at the apex).
  av and twist are Helmholtz-projected: + ∇φ with Δφ = −div B on the myocardium / valves, so they change volume only
  in the cavities (and the lungs) and the wall thickens as the base descends.
- la, lv, ra, rv: chamber volume modes B_c = ∇Γ_c, ΔΓ_c = 1_c (free-space Poisson by FFT): divergence 1 inside
  chamber c, 0 elsewhere — the myocardium is incompressible to first order (unit: volumetric strain of the chamber).
All fields are tapered to 0 over the outer 15 mm of the grid (the potential fields decay as 1/r² into the lungs).
Volume curves (t = 0: onset of ejection = end-diastole):
  V_LV = EDV + ∫(Q_MV − Q_AO), V_LA = V_LA,0 + ∫(Q_PV − Q_MV), and the right heart alike with Q_TV, Q_PA, the venous
  return. Q_AO, Q_PA, Q_PV and the venous return come from the network; mitral / tricuspid inflow are E and A lobes
  (E/A volume 65/35 and 60/40). EDV = the atlas cavity volume; the atria peak at `--la-max` / `--ra-max`.
The AV descent and the twist follow the LV ejection fraction curve; (lv, rv, la, ra) are solved at 128 times per
cycle (Newton) so that ∫_chamber det(I + ∇u) dX equals the four target volumes. Coefficients are stored as heart-rate
harmonics (continuous time).
Writes <out_dir>/cardiac_motion.hv and figures (volume curves, coefficients, displacement at end-systole, deformed
labels at end-diastole / end-systole).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
TS, TT, EXP, FLD, OUT = vf.read(A[0]), json.loads(Path(A[1]).read_text()), Path(A[2]), Path(A[3]), Path(A[4])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opt(n, d):
    return float(A[A.index(n) + 1]) if n in A else d


D_AV, TWIST = opt("--dav", 14.0) * 1e-3, np.deg2rad(opt("--twist", 9.0))
lab_full = np.asarray(TS["labels"])
names = list(TS["tissues"])
ID = {n: i for i, n in enumerate(names)}
h = float(TS["voxel_m"])
o_full = np.asarray(TS["origin_m"], float)
CH = ["blood_LA", "blood_LV", "blood_RA", "blood_RV"]
heart_ids = [ID[n] for n in CH + ["myocardium", "valve"]]
heart_full = np.isin(lab_full, heart_ids)
zz, yy, xx = np.nonzero(heart_full)
m = int(np.ceil(50e-3 / h))
lo = np.maximum([xx.min() - m, yy.min() - m, zz.min() - m], 0)
hi = np.minimum([xx.max() + m + 1, yy.max() + m + 1, zz.max() + m + 1], lab_full.shape[::-1])
lab = lab_full[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]
o = o_full + lo * h
nz, ny, nx = lab.shape
Zc, Yc, Xc = np.meshgrid(*(o[a] + (np.arange(n) + 0.5) * h for a, n in ((2, nz), (1, ny), (0, nx))), indexing="ij")
X = np.stack([Xc, Yc, Zc], -1)  # [Z, Y, X, 3] world coordinates
mask = {n: lab == ID[n] for n in CH}
heart = np.isin(lab, heart_ids)


def centroid(mk):
    return X[mk].mean(0)


def interface(a, b):
    """centroid of the faces between chamber a and b voxels"""
    pts = []
    for ax in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[ax], s1[ax] = slice(0, -1), slice(1, None)
        for p, q in ((a, b), (b, a)):
            f = mask[p][tuple(s0)] & mask[q][tuple(s1)]
            pts.append(0.5 * (X[tuple(s0)][f] + X[tuple(s1)][f]))
    pts = np.concatenate(pts)
    return pts.mean(0), len(pts)


mitral, n_mv = interface("blood_LA", "blood_LV")
tricuspid, n_tv = interface("blood_RA", "blood_RV")
cLV, cRV, cLA, cRA = (centroid(mask[n]) for n in ("blood_LV", "blood_RV", "blood_LA", "blood_RA"))
P = X[mask["blood_LV"]]
_, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
e1 = vt[0] * np.sign(vt[0] @ (mitral - cLV))  # long axis, apex → base
zeta_lv = (P - mitral) @ e1
apex = mitral + e1 * zeta_lv.min()
Lv = float(-zeta_lv.min())
zeta = (X - apex) @ e1  # 0 at the apex, Lv at the mitral plane
rvec = (X - apex) - zeta[..., None] * e1
rho = np.linalg.norm(rvec, axis=-1)
rhat = rvec / np.maximum(rho, 1e-9)[..., None]

# taper: 1 in the heart, smoothstep to 0 over 25 mm outside
d_out = ndimage.distance_transform_edt(~heart, sampling=h)
s = np.clip(d_out / 25e-3, 0, 1)
T = 1 - s * s * (3 - 2 * s)

def smooth(B, sig=1.0):
    return np.stack([ndimage.gaussian_filter(B[..., c], sig) for c in range(3)], -1)


def poisson(f):
    """free-space solution of Δψ = f on the grid (zero padding to twice the size), returns ψ"""
    sh = [2 * n for n in f.shape]
    F = np.fft.rfftn(f, sh)
    k = [2 * np.pi * np.fft.fftfreq(n, h) for n in sh[:-1]] + [2 * np.pi * np.fft.rfftfreq(sh[-1], h)]
    K2 = k[0][:, None, None] ** 2 + k[1][None, :, None] ** 2 + k[2][None, None, :] ** 2
    K2[0, 0, 0] = 1.0
    F = -F / K2
    F[0, 0, 0] = 0.0
    return np.fft.irfftn(F, sh)[: f.shape[0], : f.shape[1], : f.shape[2]]


def grad_field(psi):
    gz, gy, gx = np.gradient(psi, h)
    return np.stack([gx, gy, gz], -1)


def div_field(Bf):
    return np.gradient(Bf[..., 0], h, axis=2) + np.gradient(Bf[..., 1], h, axis=1) + np.gradient(Bf[..., 2], h, axis=0)


tissue_inc = np.isin(lab, [ID["myocardium"], ID["valve"]])  # incompressible heart tissue


def project(Bf):
    """Helmholtz correction: add ∇φ, Δφ = −div B on the heart tissue, so B changes volume only outside the tissue"""
    return Bf + grad_field(poisson(-div_field(Bf) * tissue_inc))


# grid-boundary taper (the potential fields decay as 1/r²; the grid keeps 50 mm around the heart)
edge = np.minimum.reduce([np.minimum(np.arange(n), np.arange(n)[::-1]) for n in (nz,)])
dist_edge = np.minimum.reduce(np.meshgrid(*(np.minimum(np.arange(n), n - 1 - np.arange(n)) * h for n in (nz, ny, nx)), indexing="ij"))
se = np.clip(dist_edge / 15e-3, 0, 1)
TG = se * se * (3 - 2 * se)
# AV-plane descent (kinematic, then projected)
H_atr = 50e-3
w_av = np.where(zeta <= Lv, np.clip(zeta / Lv, 0, 1), np.clip(1 - (zeta - Lv) / H_atr, 0, 1))
B_av = project(smooth(-(w_av * T)[..., None] * e1))
# twist (kinematic, then projected)
th = np.where(zeta <= Lv, 1 - 1.4 * np.clip(zeta / Lv, 0, 1), -0.4 * np.clip(1 - (zeta - Lv) / 15e-3, 0, 1))
B_tw = project(smooth((th * T)[..., None] * np.cross(e1, rvec)))
# chamber volume modes: B_c = ∇Γ_c, ΔΓ_c = 1_c (divergence 1 in the chamber, 0 elsewhere)
B_ch = {n: grad_field(poisson(mask[n].astype(float))) for n in CH}
BASES = {"av": B_av * TG[..., None], "twist": B_tw * TG[..., None], **{n[6:].lower(): B_ch[n] * TG[..., None] for n in CH}}
KN = list(BASES)
B = np.stack([BASES[k] for k in KN])  # [K, Z, Y, X, 3]


REG = CH + ["myocardium"]
regmask = {**mask, "myocardium": lab == ID["myocardium"]}
V0 = {r: regmask[r].sum() * h**3 for r in REG}


# ── target volume curves ───────────────────────────────────────────────────────────────────────────────────────────
net = np.load(FLD / "network_flow.npz")
Tp = float(net["period"])
om = net["omega"]
NH = len(om) - 1
tt_net = net["t"]
NT = 128
tt = np.arange(NT) * Tp / NT


def resample(w):
    return np.interp(tt, tt_net, w, period=Tp)


Q_AO, Q_PA = resample(net["Q_aorta"]), resample(net["Q_pulmonary"])
g = np.load(EXP / "graph.npz")
gj = json.loads((EXP / "graph.json").read_text())
eoff, ea, eb = g["edge_offsets"], g["edge_a"], g["edge_b"]
qh = net["q"]


def inflow_at_roots(cls):
    """flow from the vessels into the heart at the roots of all components of a class, as harmonics"""
    tot = np.zeros(qh.shape[1], complex)
    for c in gj["components"]:
        if c["class"] != cls:
            continue
        for rn in c.get("roots", [c["root_node"]]):
            for k in range(len(ea)):
                if ea[k] == rn:
                    tot -= qh[eoff[k]]  # sample order leaves the node: flow into the node is −q
                elif eb[k] == rn:
                    tot += qh[eoff[k + 1] - 1]
    return tot


def synth(c, t):
    return np.real(c[0] + 2 * (c[1:] @ np.exp(1j * om[1:, None] * np.asarray(t)[None])))


Q_PV = synth(inflow_at_roots("pulmonary_venous"), tt)
Q_VR = synth(inflow_at_roots("systemic_venous"), tt)
SV_L, SV_R = Q_AO.mean() * Tp, Q_PA.mean() * Tp


def lobe(t0, w):
    x = (tt - t0) / w
    return np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 2, 0.0)


TEJ_L = 0.30 * (Tp / 0.857) ** 0.5
TEJ_R = 0.33 * (Tp / 0.857) ** 0.5


def filling(tej, ivrt, fe, sv):
    E = lobe(tej + ivrt, 0.26 * Tp)
    Aw = lobe(Tp - 0.15 * Tp, 0.14 * Tp)
    q = fe * E / E.mean() + (1 - fe) * Aw / Aw.mean()
    return q / q.mean() * sv / Tp


FE = opt("--fe", 0.65)  # E-wave share of the LV filling (RV: 0.05 less)
Q_MV = filling(TEJ_L, 0.08, FE, SV_L)
Q_TV = filling(TEJ_R, 0.06, FE - 0.05, SV_R)
dt = Tp / NT


def integ(q):
    c = np.cumsum(q) * dt
    return np.r_[0, c[:-1]]


EDV_L, EDV_R = V0["blood_LV"], V0["blood_RV"]
tgt = {
    "blood_LV": EDV_L + integ(Q_MV - Q_AO),
    "blood_RV": EDV_R + integ(Q_TV - Q_PA),
}
la = integ(Q_PV - Q_MV)
ra = integ(Q_VR - Q_TV)
tgt["blood_LA"] = la - la.max() + opt("--la-max", 65.0) * 1e-6
tgt["blood_RA"] = ra - ra.max() + opt("--ra-max", 80.0) * 1e-6
for k in tgt:
    tgt[k] = tgt[k] - (tgt[k][-1] + (tgt[k][1] - tgt[k][0]) - tgt[k][0]) * 0  # periodic by construction (means match)
e_L = (EDV_L - tgt["blood_LV"]) / (EDV_L - tgt["blood_LV"].min())  # 0 at ED, 1 at ES

# ── calibration: modes first (Newton on the four chamber volumes), then a nonlinear incompressibility correction of
#    the heart tissue per time sample: U ← U + ∇ψ, Δψ = −log det(I + ∇U) on myocardium / valves, chamber volumes re-fit
NT = int(opt("--samples", 48))
tt_s = np.arange(NT) * Tp / NT
sel = (np.arange(len(tt)) * NT) // len(tt)
iu = [KN.index(k) for k in ("lv", "rv", "la", "ra")]
CHV = ("blood_LV", "blood_RV", "blood_LA", "blood_RA")
idx_reg = {r: np.nonzero(regmask[r]) for r in REG}


def grad_at(U, r):
    """∇U at the voxels of region r: [n, 3, 3]"""
    zi, yi, xi = idx_reg[r]
    g = np.zeros((len(zi), 3, 3))
    for c in range(3):
        gz, gy, gx = np.gradient(U[..., c], h)
        g[:, c, 0], g[:, c, 1], g[:, c, 2] = gx[zi, yi, xi], gy[zi, yi, xi], gz[zi, yi, xi]
    return g


def vol_U(U, regs):
    gr = [np.gradient(U[..., c], h) for c in range(3)]  # once for all regions: [c][d/dz, d/dy, d/dx]
    out = {}
    for r in regs:
        zi, yi, xi = idx_reg[r]
        g = np.empty((len(zi), 3, 3))
        for c in range(3):
            g[:, c, 0], g[:, c, 1], g[:, c, 2] = gr[c][2][zi, yi, xi], gr[c][1][zi, yi, xi], gr[c][0][zi, yi, xi]
        out[r] = float(np.linalg.det(np.eye(3)[None] + g).sum() * h**3)
    return out


def tgt_at(t):
    return np.array([np.interp(t, tt, tgt[c], period=Tp) for c in CHV])


Bstack = np.stack([BASES[k] for k in KN])
U_t = np.zeros((NT, nz, ny, nx, 3), np.float32)
coef = np.zeros((NT, len(KN)))
achieved = {r: np.zeros(NT) for r in REG}
a = np.zeros(len(KN))
eL_s = np.interp(tt_s, tt, e_L, period=Tp)
for i, t in enumerate(tt_s):
    a[KN.index("av")] = D_AV * eL_s[i]
    a[KN.index("twist")] = TWIST * eL_s[i]
    target = tgt_at(t)
    extra = np.zeros((nz, ny, nx, 3))
    for outer in range(5):
        for it in range(12):  # Newton on the four volume modes, with the current correction held fixed
            U = np.einsum("k,kzyxc->zyxc", a, Bstack) + extra
            v = vol_U(U, CHV)
            f = np.array([v[c] for c in CHV]) - target
            if np.abs(f).max() < 2e-9:
                break
            Jm = np.zeros((4, 4))
            for j, kj in enumerate(iu):
                v2 = vol_U(U + 1e-3 * Bstack[kj], CHV)
                Jm[:, j] = (np.array([v2[c] for c in CHV]) - target - f) / 1e-3
            a[iu] -= np.linalg.solve(Jm, f)
        U = np.einsum("k,kzyxc->zyxc", a, Bstack) + extra
        if outer == 4:
            break
        zi, yi, xi = idx_reg["myocardium"]
        ld = np.zeros((nz, ny, nx))
        for r in ("myocardium",):
            ld[idx_reg[r]] = np.log(np.maximum(np.linalg.det(np.eye(3)[None] + grad_at(U, r)), 1e-3))
        vm = tissue_inc & ~regmask["myocardium"]
        if vm.any():
            gv = np.zeros((vm.sum(), 3, 3))
            for c in range(3):
                gz, gy, gx = np.gradient(U[..., c], h)
                gv[:, c, 0], gv[:, c, 1], gv[:, c, 2] = gx[vm], gy[vm], gz[vm]
            ld[vm] = np.log(np.maximum(np.linalg.det(np.eye(3)[None] + gv), 1e-3))
        extra = extra + grad_field(poisson(-ld * tissue_inc)) * TG[..., None]
    U_t[i] = U
    coef[i] = a
    v = vol_U(U, REG)
    for r in REG:
        achieved[r][i] = v[r]
    if i % 8 == 0:
        print(f"t {t * 1e3:5.0f} ms: myocardium {v['myocardium'] / V0['myocardium'] * 100 - 100:+.2f} %, volumes {[round(v[c] * 1e6, 1) for c in CHV]} target {[round(x * 1e6, 1) for x in target]}", flush=True)
tt128 = tt
tt = tt_s
for c in list(tgt):
    tgt[c] = np.interp(tt_s, tt128, tgt[c], period=Tp)
e_L = eL_s
err = max(np.abs(achieved[c] - tgt[c]).max() for c in tgt) * 1e6
ch = np.fft.rfft(coef, axis=0)[: min(NH, NT // 2) + 1] / NT
B = Bstack
# ── displacement summaries ─────────────────────────────────────────────────────────────────────────────────────────
i_es = int(np.argmin(tgt["blood_LV"]))
U_es = U_t[i_es].astype(np.float64)
U_ed = U_t[0].astype(np.float64)
dmag = np.linalg.norm(U_es - U_ed, axis=-1)
summary = {
    "period_s": Tp, "geometry_m": {"apex": apex.tolist(), "mitral": mitral.tolist(), "tricuspid": tricuspid.tolist(), "long_axis": e1.tolist(), "lv_length": Lv,
                                   "faces_mitral": n_mv, "faces_tricuspid": n_tv},
    "volumes_ml": {c: {"min": float(tgt[c].min() * 1e6), "max": float(tgt[c].max() * 1e6), "atlas": float(V0[c] * 1e6)} for c in tgt},
    "ef": {"LV": float(SV_L / EDV_L), "RV": float(SV_R / EDV_R)}, "sv_ml": {"L": SV_L * 1e6, "R": SV_R * 1e6},
    "volume_fit_max_err_ml": err,
    "myocardium_volume_change_pct": float(np.abs(achieved["myocardium"] / V0["myocardium"] - 1).max() * 100),
    "displacement_es_vs_ed_mm": {"heart_p50": float(np.median(dmag[heart]) * 1e3), "heart_p99": float(np.percentile(dmag[heart], 99) * 1e3)},
    "coef_range": {k: [float(coef[:, j].min()), float(coef[:, j].max())] for j, k in enumerate(KN)},
}
(OUT / "cardiac_motion.json").write_text(json.dumps(summary, indent=1))
vf.write(OUT / "cardiac_motion.hv", {
    "kind": "phantom_motion_cardiac", "U_mm": np.ascontiguousarray(np.moveaxis(U_t, -1, 1) * 1e3).astype(np.float16),  # [T, 3, Z, Y, X] mm
    "t": tt, "period": Tp, "origin_m": [float(x) for x in o], "voxel_m": h, "dims_xyz": [nx, ny, nz],
    "bases": KN, "coef_t": coef, **{f"V_{c}": tgt[c] for c in tgt}, **{f"V_{c}_achieved": achieved[c] for c in REG},
    "flows_t": tt_net[:0] if False else np.arange(128) * Tp / 128, "Q_MV": Q_MV, "Q_TV": Q_TV, "Q_AO": Q_AO, "Q_PA": Q_PA, "Q_PV": Q_PV, "Q_VR": Q_VR,
    "apex": apex, "mitral": mitral, "tricuspid": tricuspid, "long_axis": e1,
    "note": "displacement u(X, t) of the atlas pose X (t from the onset of ejection = end-diastole): U_mm at the samples t, periodic; interpolate in time (periodic cubic) and trilinearly in space (0 outside the grid)",
})
print(json.dumps(summary, indent=1))

# ── figures ────────────────────────────────────────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(15, 9))
ax = axs[0, 0]
for c, col in zip(CH, ("tab:orange", "tab:red", "tab:cyan", "tab:blue")):
    ax.plot(tt * 1e3, tgt[c] * 1e6, color=col, lw=2, label=f"{c[6:]} target")
    ax.plot(tt * 1e3, achieved[c] * 1e6, "k:", lw=1)
ax.plot(tt * 1e3, achieved["myocardium"] * 1e6 / 3, "m--", label="myocardium / 3")
ax.set_title(f"chamber volumes [mL] (dotted: ∫det∇Φ of the deformed atlas chamber; max error {err:.2e} mL)")
ax.legend(fontsize=8)
ax = axs[0, 1]
for qn_, q_ in (("Q_AO", Q_AO), ("Q_MV", Q_MV), ("Q_PV", Q_PV), ("Q_PA", Q_PA), ("Q_TV", Q_TV), ("Q_VR", Q_VR)):
    ax.plot(np.arange(128) * Tp / 128 * 1e3, q_ * 1e6, label=qn_)
ax.set_title("flows [mL/s]: outflow (network), filling (E/A), venous return (network)")
ax.legend(fontsize=8)
ax = axs[1, 0]
for j, k in enumerate(KN):
    sc = 1e3 if k == "av" else (np.rad2deg(1) if k == "twist" else 100)
    ax.plot(tt * 1e3, coef[:, j] * sc, label=f"{k} [{'mm' if k == 'av' else ('deg' if k == 'twist' else '%')}]")
ax.set_title("basis coefficients")
ax.legend(fontsize=8)
ax = axs[1, 1]
ix = int(np.argmin(np.abs(o[0] + (np.arange(nx) + 0.5) * h - cLV[0])))
im = ax.imshow(dmag[:, :, ix][::-1] * 1e3, cmap="viridis", extent=[o[1] * 1e3, (o[1] + ny * h) * 1e3, o[2] * 1e3, (o[2] + nz * h) * 1e3])
ax.contour(np.isin(lab[:, :, ix], heart_ids)[::-1], levels=[0.5], colors="w", linewidths=0.5, extent=[o[1] * 1e3, (o[1] + ny * h) * 1e3, (o[2] + nz * h) * 1e3, o[2] * 1e3])
fig.colorbar(im, ax=ax, label="mm")
ax.set_title(f"|u(ES) − u(ED)| sagittal x = {cLV[0] * 1e3:.0f} mm")
for a_ in axs[:, :].flat[:3]:
    a_.set_xlabel("t from ejection onset [ms]")
fig.tight_layout()
fig.savefig(OUT / "figs" / "01_cardiac_volumes.png", dpi=95)
plt.close(fig)


# deformed labels at ED and ES (inverse map by fixed-point iteration, nearest label)
def warp_labels(U):
    from scipy.ndimage import map_coordinates

    Xs = X.copy()
    for _ in range(6):
        idx = ((Xs - o) / h - 0.5)[..., ::-1]  # z, y, x index coordinates
        u = np.stack([map_coordinates(U[..., c], np.moveaxis(idx, -1, 0), order=1, mode="nearest") for c in range(3)], -1)
        Xs = X - u
    idx = np.rint((Xs - o) / h - 0.5).astype(int)[..., ::-1]
    for a_, n_ in enumerate((nz, ny, nx)):
        idx[..., a_] = np.clip(idx[..., a_], 0, n_ - 1)
    return lab[idx[..., 0], idx[..., 1], idx[..., 2]]


COL = {"blood_LA": "#ff8080", "blood_LV": "#ff2020", "blood_RA": "#80a0ff", "blood_RV": "#2050ff", "myocardium": "#8c1c13", "valve": "#ffd400", "lung": "#2b3a55"}
from matplotlib.colors import ListedColormap  # noqa: E402

cmap = ListedColormap(["#000000"] + list(COL.values()))
code = {ID[n]: i + 1 for i, n in enumerate(COL)}


def colour(L_):
    out = np.zeros(L_.shape, int)
    for k_, v_ in code.items():
        out[L_ == k_] = v_
    return out


fig, axs = plt.subplots(2, 3, figsize=(16, 10))
iz = int(np.argmin(np.abs(o[2] + (np.arange(nz) + 0.5) * h - cLV[2])))
iy = int(np.argmin(np.abs(o[1] + (np.arange(ny) + 0.5) * h - cLV[1])))
for r, (U, tl) in enumerate(((U_ed, "end-diastole"), (U_es, "end-systole"))):
    W = warp_labels(U)
    for c_, (sl, ttl) in enumerate(((W[iz], "axial"), (W[:, iy, :][::-1], "coronal"), (W[:, :, ix][::-1], "sagittal"))):
        axs[r, c_].imshow(colour(sl), cmap=cmap, vmin=0, vmax=len(COL), interpolation="nearest")
        axs[r, c_].set_title(f"{tl}: {ttl}")
        axs[r, c_].axis("off")
fig.suptitle("deformed atlas heart (labels pulled back through the motion field)")
fig.tight_layout()
fig.savefig(OUT / "figs" / "02_cardiac_ed_es.png", dpi=90)
plt.close(fig)
print("wrote", OUT)
