"""Cardiac kinematics of the atlas phantom: a displacement field u(X, t) = Σ_k a_k(t) B_k(X) whose chamber volumes
follow volume curves that are mass-consistent with the 1D network.

    python -I cardiac_motion.py <tissue_2mm.hv> <tissues.json> <export_dir> <flow_dir> <out_dir>
        [--dav 14] [--twist 9] [--la-max 65] [--ra-max 80] [--fe 0.65] [--samples 48] [--tether on|off]

Reference configuration X = the atlas pose. Basis fields (on a 2 mm grid around the heart, smoothed, all multiplied
by a taper T that is 1 in the heart and falls to 0 over --taper-mm (25) outside it):
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
  return. Q_AO, Q_PA, Q_PV and the venous return come from the network; mitral / tricuspid inflow are E and A waves
  (--fe the E share of the volume; --filling physio: E rises 90 ms and decelerates over DT 180 ms, A a 150 ms lobe
  ending 40 ms before ejection, × √(T / 0.857 s); --filling sin2: the earlier sin² lobes). EDV = the atlas cavity
  volume; the atria peak at `--la-max` / `--ra-max`.
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
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)
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
# --taper-mm: width of the smoothstep that takes the AV-descent and twist fields from 1 in the heart to 0 outside
# (default 25 mm: the base descends ~15 mm, so the tissue around the great vessels was compressed to ~0.5 of its length,
# research/flow_phantom.md Tier B5); the grid keeps max(50 mm, taper + 25 mm) around the heart
TAPER = opt("--taper-mm", 25.0) * 1e-3
# --outside harmonic (default, 2026-11-25) | taper: how the AV-descent and twist fields leave the heart — the harmonic
# extension of the heart's field to zero ≥ 80 mm away (motion_inclusions.harmonic_extension), or the old smoothstep over
# --taper-mm. The taper compressed the tissue around the great vessels to ≈ 0.5 (descending aorta 0.57 → 0.85 with the
# extension; within 40 mm of the heart stretch p1 0.53 → 0.71, det min 0.26 → 0.42; research/flow_phantom.md Tier B5).
# --insulate-vessels (experimental, folds: det J −7): chamber modes with the great vessels as insulated walls.
OUTSIDE = (A[A.index("--outside") + 1] if "--outside" in A else "harmonic")
assert OUTSIDE in ("harmonic", "taper"), OUTSIDE
INSUL = "--insulate-vessels" in A
INCL = OUTSIDE == "harmonic" or INSUL
m = int(np.ceil(max(50e-3, TAPER + 25e-3, 80e-3 if INCL else 0.0) / h))
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
s = np.clip(d_out / TAPER, 0, 1)
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
net = load_any(FLD / "network_flow.npz")
Tp = float(net["period"])
om = net["omega"]
NH = len(om) - 1
tt_net = net["t"]
NT = 128
tt = np.arange(NT) * Tp / NT


def resample(w):
    return np.interp(tt, tt_net, w, period=Tp)


Q_AO, Q_PA = resample(net["Q_aorta"]), resample(net["Q_pulmonary"])
g = load_any(EXP / "graph.npz")
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


FILLING = A[A.index("--filling") + 1] if "--filling" in A else "physio"
S_HR = (Tp / 0.857) ** 0.5  # heart-rate scale of the diastolic intervals (as the ejection times)


def filling(tej, ivrt, fe, sv, dt_e=None):
    """E and A waves. --filling physio (default, 2026-11-27): healthy adult Doppler shapes — E rises over 90 ms and
    decelerates linearly over the deceleration time DT = 180 ms (AV stenosis: DT = PHT / 0.29), A is a 150 ms sin² lobe
    ending 40 ms before ejection (isovolumic contraction), all × √(T / 0.857 s). With fe 0.65–0.80 this gives E/A peak
    1.1–2 and a peak filling rate ≈ 3–4 EDV/s; E and A sum where they overlap (fast heart rates).
    --filling sin2 (before): E a symmetric sin² lobe of 0.26 T, A one of 0.14 T ending at 0.99 T — the A lobe peaked above
    E (E/A < 1, the impaired-relaxation pattern) and the LV filled at ≈ 5 EDV/s; the stenosis E rose over 80 ms."""
    if FILLING == "physio":
        t0 = tej + ivrt
        x = tt - t0
        rise = 0.09 * S_HR
        dte = dt_e if dt_e is not None else 0.18 * S_HR
        E = np.where((x > 0) & (x < rise), np.sin(0.5 * np.pi * np.clip(x / rise, 0, 1)) ** 2, 0.0)
        E = np.where(x >= rise, np.clip(1 - (x - rise) / dte, 0, None), E)
        E = np.where(tt > Tp - 0.05, 0.0, E)
        aw, a_end = 0.15 * S_HR, Tp - 0.04 * S_HR
        Aw = lobe(a_end - aw, aw)
        q = fe * E / E.mean() + (1 - fe) * Aw / Aw.mean()
        return q / q.mean() * sv / Tp
    assert FILLING == "sin2", f"--filling {FILLING}: physio | sin2"
    if dt_e is None:
        E = lobe(tej + ivrt, 0.26 * Tp)
    else:
        t0 = tej + ivrt
        x = tt - t0
        E = np.where((x > 0) & (x < 0.08), np.sin(0.5 * np.pi * np.clip(x / 0.08, 0, 1)) ** 2, 0.0)
        E = np.where(x >= 0.08, np.clip(1 - (x - 0.08) / dt_e, 0, None), E)
        E = np.where(tt > Tp - 0.05, 0.0, E)
    Aw = lobe(Tp - 0.15 * Tp, 0.14 * Tp)
    q = fe * E / E.mean() + (1 - fe) * Aw / Aw.mean()
    return q / q.mean() * sv / Tp


FE = opt("--fe", 0.65)  # E-wave share of the LV filling (RV: the same with --filling physio, 0.05 less with sin2)
# valve disease (valve_disease.py; --valve-disease <spec>, --p-sys / --p-pul mean arterial pressures in mmHg): MR / TR
# send RVol back to the atrium over isovolumic contraction, ejection and isovolumic relaxation, so the ventricle fills
# SV + RVol; AR / PR are in the network's root flow already (Q_AO / Q_PA reverse in diastole); MS / TS slow the E wave
sys.path.insert(0, str(Path(__file__).resolve().parent))
import valve_disease as VD  # noqa: E402

VDIS = VD.parse(A[A.index("--valve-disease") + 1] if "--valve-disease" in A else None)
P_SYS_, P_PUL_ = opt("--p-sys", 93.0), opt("--p-pul", 15.0)


def av_inflow(valve, tej, ivrt, fe, sv, p_mean):
    les = VDIS.get(valve, {})
    dt_e = VD.pht_s(les["stenosis"]) / 0.29 if "stenosis" in les else None
    if "regurgitation" in les:
        qr = VD.regurgitant_flow(valve, les["regurgitation"], tt, Tp, tej, p_mean)
        rv = float(qr.sum() * Tp / NT)
        return filling(tej, ivrt, fe, sv + rv, dt_e) - qr
    return filling(tej, ivrt, fe, sv, dt_e)


Q_MV = av_inflow("mitral", TEJ_L, 0.08, FE, SV_L, P_SYS_)
# (the RV's E share: the LV's with the physio shapes — healthy tricuspid E/A ≥ 1 as the mitral; sin2: 0.05 less)
Q_TV = av_inflow("tricuspid", TEJ_R, 0.06, FE if FILLING == "physio" else FE - 0.05, SV_R, P_PUL_)
if VDIS:
    print(f"valve disease: {VD.describe(VDIS, Tp, P_SYS_, P_PUL_)}")
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
# the atria peak at --la-max / --ra-max; an AV regurgitation adds its RVol to the atrial swing, and the atrium
# enlarges rather than emptying further: the maximum rises so that the minimum stays at least the healthy share of the
# atlas volume (0.43 LA, 0.45 RA: ph0001 26 / 61 mL, 40 / 89 mL)
la_max_, ra_max_ = opt("--la-max", 65.0) * 1e-6, opt("--ra-max", 80.0) * 1e-6
if "regurgitation" in VDIS.get("mitral", {}):
    la_max_ = max(la_max_, 0.43 * V0["blood_LA"] + (la.max() - la.min()))
if "regurgitation" in VDIS.get("tricuspid", {}):
    ra_max_ = max(ra_max_, 0.45 * V0["blood_RA"] + (ra.max() - ra.min()))
tgt["blood_LA"] = la - la.max() + la_max_
tgt["blood_RA"] = ra - ra.max() + ra_max_
for k in tgt:
    tgt[k] = tgt[k] - (tgt[k][-1] + (tgt[k][1] - tgt[k][0]) - tgt[k][0]) * 0  # periodic by construction (means match)
# a ventricle ejecting SV + RVol from the atlas EDV must keep a physiological end-systolic volume (regurgitation with
# a normal-sized ventricle: lower the forward SV or dilate the ventricle with a diffeomorphic variant)
for k_, edv_ in (("blood_LV", EDV_L), ("blood_RV", EDV_R)):
    if tgt[k_].min() < 0.2 * edv_:
        sys.exit(f"{k_}: end-systolic volume {tgt[k_].min() * 1e6:.0f} mL < 0.2 EDV ({edv_ * 1e6:.0f} mL): stroke + regurgitant volume too large for the atlas ventricle")
e_L = (EDV_L - tgt["blood_LV"]) / (EDV_L - tgt["blood_LV"].min())  # 0 at ED, 1 at ES

# ── outside the heart (--outside harmonic, default): the AV descent and twist are the heart's kinematic fields extended
#    harmonically outside the heart to zero ≥ 80 mm away instead of a 25 mm smoothstep; the chamber volume modes stay the
#    free-space ∇Γ (a chamber's volume change pushes the surrounding tissue, vessels included, outward).
#    --insulate-vessels (experimental): chamber modes with the great vessels (r ≥ --incl-rmin mm, outside the heart) as
#    insulated walls and the tube interiors filled harmonically — it folds the field (det J −7: potential flow around
#    voxelised tubes is singular at their corners and concentrates in narrow gaps). A first version moved the tubes
#    rigidly with their root, fading along the vessel: the displacement jumped at the tube walls (det J −16).
if INCL:
    import time as _time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from motion_inclusions import harmonic_extension, insulated_poisson_mode, vessel_inclusions  # noqa: E402
    _t0 = _time.time()
    if INSUL:
        incl, w_in, rpos, info = vessel_inclusions(X, heart, g, gj, r_min=opt("--incl-rmin", 3.5) * 1e-3, decay=opt("--incl-decay", 40.0) * 1e-3)
        print(f"insulated vessels: {info}, {incl.sum() * h**3 * 1e6:.0f} mL of vessel tubes outside the heart", flush=True)

    newB = {}
    for k in KN:
        if k in ("av", "twist") and OUTSIDE == "taper":
            newB[k] = BASES[k]
        elif k in ("av", "twist"):
            # the heart's kinematic field, extended harmonically outside the heart (zero at the grid boundary, ≥ 80 mm
            # away) instead of the 25 mm smoothstep: the base descent fades over the whole margin
            Bh = B_av if k == "av" else B_tw
            ext, its = harmonic_extension(Bh, heart)
            newB[k] = ext
            print(f"  basis {k}: CG iterations {its} ({_time.time() - _t0:.0f}s)", flush=True)
        elif not INSUL:
            newB[k] = BASES[k]  # free-space chamber mode
        else:
            # the chamber's volume change flows around the great vessels (insulated walls); the tube interiors are
            # filled harmonically from their surroundings, so a vessel moves with the tissue around it (continuous)
            Bp, it_ = insulated_poisson_mode(mask["blood_" + k.upper()].astype(float), incl, h)
            ext, its = harmonic_extension(Bp, ~incl)
            newB[k] = ext
            print(f"  basis {k}: CG iterations {it_} (insulated mode) + {its} (tube fill) ({_time.time() - _t0:.0f}s)", flush=True)
    BASES = newB

# ── tethering (--tether on, default 2026-11-28; off: as before): the skeleton and the great vessels lying on it
#    (radius ≥ --tether-rmin 3 mm, wall within --tether-gap 8 mm of bone, ≥ --tether-free 100 mm along the vessels from
#    the heart; outside the heart) stay put under every basis
#    field; the heart keeps its motion; between them a harmonic shear layer at most --tether-band 15 mm thick
#    (motion_inclusions.tether). The chamber volumes are unchanged (the heart is kept)
TETHER = (A[A.index("--tether") + 1] if "--tether" in A else "on")
assert TETHER in ("on", "off"), TETHER
if TETHER == "on":
    import time as _time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from motion_inclusions import graph_path_distance, heart_roots, tether, tether_mask  # noqa: E402
    _t0 = _time.time()
    _bone = np.isin(lab, [ID[n] for n in ("bone_cortical", "bone_marrow") if n in ID])
    _gj = json.loads((EXP / "graph.json").read_text())
    # path seeds: the connectors and the heart roots (2026-11-28; `--tether-seeds connectors`: connectors only, as before —
    # the IVC root never had a connector, so its tree counted from the SVC through the azygos, 636 mm)
    _pd = graph_path_distance(g, _gj["edge_names"], heart_roots(_gj) if (A[A.index("--tether-seeds") + 1] if "--tether-seeds" in A else "roots") == "roots" else ())
    TM = tether_mask(_bone, o, h, np.asarray(g["points"], float), np.asarray(g["radius"], float), rmin=opt("--tether-rmin", 3.0) * 1e-3, gap=opt("--tether-gap", 8.0) * 1e-3, exclude=heart,
                     pathd=_pd, free_path=opt("--tether-free", 100.0) * 1e-3, weights=True)
    # ── great-artery roots (2026-11-28): the aortic / pulmonary roots lie inside the atlas ventricle labels and moved
    #    with the full AV-plane descent up to the sinotubular junction (ph0001: 15 mm), stretching the ascending aorta
    #    up to 30 % in systole. Measured with an AV descent ≈ 15 mm: aortic annulus 11.6 ± 2.9 mm (CMR), coronary ostia
    #    6.2 ± 2.0 mm, mid-ascending aorta 3.8 ± 1.9 mm (4D CTA, Weber et al. 2014). The root lumen + 2 mm of wall of each
    #    arterial heart tree (path distance s from its valve, vessels r ≥ --tether-rmin) gets a prescribed motion: in the
    #    AV basis g(s) along the descent direction −e1, i.e. g(s) · D_AV in phase with the AV descent, and no motion in
    #    the other bases (twist, chamber volumes), g = interp(s; --root-s 0,15,50,80 mm → --root-g 0.77,0.45,0.25,0.10).
    #    Aortic tree only (--root-classes). A first version scaled the local motion by a hold weight 1 − g / |B_av|: the chamber-volume modes and the Newton
    #    refit put the motion back (0.73 instead of 0.45 at 15 mm). `--root-hold off`: off (roots move with the ventricle).
    RT = None
    if (A[A.index("--root-hold") + 1] if "--root-hold" in A else "on") == "on":
        _rs = np.array([float(x) for x in (A[A.index("--root-s") + 1] if "--root-s" in A else "0,15,50,80").split(",")]) * 1e-3
        _rg = np.array([float(x) for x in (A[A.index("--root-g") + 1] if "--root-g" in A else "0.77,0.45,0.25,0.10").split(",")])
        _cls = {c["id"]: c["class"] for c in _gj["components"]}
        # --root-classes: the aortic tree only by default — prescribing the pulmonary trunk's descent out to its
        # bifurcation folded the lung around the left pulmonary artery (det J −0.36); the trunk moved realistically
        # without it (7 mm, stretch 0.99–1.04)
        _rcl = (A[A.index("--root-classes") + 1] if "--root-classes" in A else "systemic_arterial").split(",")
        _art_roots = [int(r) for c in _gj["components"] if c["class"] in _rcl for r in c.get("roots", [])]
        _pda = graph_path_distance(g, _gj["edge_names"], _art_roots, connectors=False)  # s = 0 at the valve only
        _gp, _gr = np.asarray(g["points"], float), np.asarray(g["radius"], float)
        _ecls = np.repeat([_cls.get(int(c_), "") for c_ in g["edge_comp"]], np.diff(g["edge_offsets"]))
        _sel = np.isin(_ecls, _rcl) & (_pda <= _rs[-1]) & (_gr >= opt("--tether-rmin", 3.0) * 1e-3)
        RS = np.full(lab.shape, np.inf, np.float32)  # path distance of the nearest root point covering the cell
        RD = np.full(lab.shape, np.inf, np.float32)
        for p_, r_, s_ in zip(_gp[_sel], _gr[_sel], _pda[_sel]):
            rr_ = r_ + 2e-3
            i0 = np.maximum(np.floor((p_ - rr_ - o) / h).astype(int), 0)
            i1 = np.minimum(np.ceil((p_ + rr_ - o) / h).astype(int) + 1, [nx, ny, nz])
            if np.any(i1 <= i0):
                continue
            zz, yy, xx = np.meshgrid(*(np.arange(i0[d], i1[d]) for d in (2, 1, 0)), indexing="ij")
            dd = np.linalg.norm(np.stack([xx, yy, zz], -1) * h + o + h / 2 - p_, axis=-1)
            upd = (dd <= rr_) & (dd < RD[zz, yy, xx])
            RD[zz[upd], yy[upd], xx[upd]] = dd[upd]
            RS[zz[upd], yy[upd], xx[upd]] = s_
        RM = np.isfinite(RS) & ~np.isin(lab, [ID[x] for x in ("blood_RA", "blood_LA", "blood_RV") if x in ID])  # the other chambers keep their modes
        RT = np.zeros(lab.shape + (3,), np.float32)
        RT[RM] = -np.interp(RS[RM], _rs, _rg)[:, None] * e1[None, :]
        TM = np.where(RM, 1.0, TM).astype(np.float32)
        print(f"root hold: {RM.sum() * h**3 * 1e6:.0f} mL of great-artery root (path ≤ {_rs[-1] * 1e3:.0f} mm) moves g(s) × the AV descent", flush=True)
    # --tether-keep none (default): the shear layer reaches into the heart too — where a chamber wall lies a few mm from
    # bone or a tethered vessel (the RV under the sternum, the LA on the descending aorta) the wall moves less and the
    # Newton fit below takes each chamber's volume elsewhere; heart: the heart keeps all its motion (the thin gaps
    # between it and the bone then fold: det J −0.18 on ph0001)
    _keep = heart if (A[A.index("--tether-keep") + 1] if "--tether-keep" in A else "none") == "heart" else None
    _its = 0
    for k in KN:
        BASES[k], it_ = tether(BASES[k], TM, _keep, h, band=opt("--tether-band", 30.0) * 1e-3, target=RT if (k == "av" and RT is not None) else None)
        _its += it_
    print(f"tether: {(TM > 0).sum() * h**3 * 1e6:.0f} mL tethered ({((TM > 0) & ~_bone).sum() * h**3 * 1e6:.0f} mL vessel lumen), CG iterations {_its} ({_time.time() - _t0:.0f}s)", flush=True)
    TM = TM > 0

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
# the fit on the GPU (torch, float64; 2026-11-25: the numpy loop took ~6 min, 14 000 full-grid gradient evaluations);
# --cpu-fit: the numpy loop (the reference)
try:
    import torch
    FIT_GPU = "--cpu-fit" not in A and torch.cuda.is_available()
except ImportError:
    FIT_GPU = False
if FIT_GPU:
    dev = torch.device("cuda")
    Bt = torch.tensor(Bstack, dtype=torch.float64, device=dev)  # [K, Z, Y, X, 3]
    TGt = torch.tensor(TG, dtype=torch.float64, device=dev)
    inc_t = torch.tensor(tissue_inc, device=dev)
    vm_t = inc_t & ~torch.tensor(regmask["myocardium"], device=dev)
    idx_t = {r: torch.tensor(np.ravel_multi_index(idx_reg[r], (nz, ny, nx)), device=dev) for r in REG}
    sh_ = [2 * nz, 2 * ny, 2 * nx]
    kz_, ky_ = (2 * np.pi * torch.fft.fftfreq(n_, h, dtype=torch.float64, device=dev) for n_ in sh_[:2])
    kx_ = 2 * np.pi * torch.fft.rfftfreq(sh_[2], h, dtype=torch.float64, device=dev)
    K2_ = kz_[:, None, None] ** 2 + ky_[None, :, None] ** 2 + kx_[None, None, :] ** 2
    K2_[0, 0, 0] = 1.0

    def _jac(U):
        """∇U as [Z·Y·X, 3, 3] (row c = component, column = d/dx, d/dy, d/dz; np.gradient's edge order)"""
        J = torch.empty((nz, ny, nx, 3, 3), dtype=torch.float64, device=dev)
        for c in range(3):
            gz, gy, gx = torch.gradient(U[..., c], spacing=h, dim=(0, 1, 2))
            J[..., c, 0], J[..., c, 1], J[..., c, 2] = gx, gy, gz
        return J.reshape(-1, 3, 3)

    def _det(F):
        return (F[:, 0, 0] * (F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1]) - F[:, 0, 1] * (F[:, 1, 0] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 0])
                + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0]))

    I3 = torch.eye(3, dtype=torch.float64, device=dev)[None]

    def vol_t(U, regs):
        J = _jac(U)
        return {r: float(_det(I3 + J[idx_t[r]]).sum()) * h**3 for r in regs}

    def poisson_t(f):
        F_ = torch.fft.rfftn(f, s=sh_)
        F_ = -F_ / K2_
        F_[0, 0, 0] = 0.0
        return torch.fft.irfftn(F_, s=sh_)[:nz, :ny, :nx]

    def grad_field_t(psi):
        gz, gy, gx = torch.gradient(psi, spacing=h, dim=(0, 1, 2))
        return torch.stack([gx, gy, gz], -1)

    for i, t in enumerate(tt_s):
        a[KN.index("av")] = D_AV * eL_s[i]
        a[KN.index("twist")] = TWIST * eL_s[i]
        target = tgt_at(t)
        extra = torch.zeros((nz, ny, nx, 3), dtype=torch.float64, device=dev)
        for outer in range(5):
            for it in range(12):  # Newton on the four volume modes, with the current correction held fixed
                U = torch.einsum("k,kzyxc->zyxc", torch.as_tensor(a, device=dev), Bt) + extra
                v = vol_t(U, CHV)
                f = np.array([v[c] for c in CHV]) - target
                if np.abs(f).max() < 2e-9:
                    break
                Jm = np.zeros((4, 4))
                for j, kj in enumerate(iu):
                    v2 = vol_t(U + 1e-3 * Bt[kj], CHV)
                    Jm[:, j] = (np.array([v2[c] for c in CHV]) - target - f) / 1e-3
                a[iu] -= np.linalg.solve(Jm, f)
            U = torch.einsum("k,kzyxc->zyxc", torch.as_tensor(a, device=dev), Bt) + extra
            if outer == 4:
                break
            Jf = _jac(U)
            sel_ = torch.flatten(inc_t).nonzero()[:, 0]
            ld = torch.zeros(nz * ny * nx, dtype=torch.float64, device=dev)
            ld[sel_] = torch.log(torch.clamp(_det(I3 + Jf[sel_]), min=1e-3))
            extra = extra + grad_field_t(poisson_t(-ld.reshape(nz, ny, nx))) * TGt[..., None]
        U_t[i] = U.to(torch.float32).cpu().numpy()
        coef[i] = a
        v = vol_t(U, REG)
        for r in REG:
            achieved[r][i] = v[r]
        if i % 8 == 0:
            print(f"t {t * 1e3:5.0f} ms: myocardium {v['myocardium'] / V0['myocardium'] * 100 - 100:+.2f} %, volumes {[round(v[c] * 1e6, 1) for c in CHV]} target {[round(x * 1e6, 1) for x in target]}", flush=True)
    del Bt
    torch.cuda.empty_cache()
else:
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
# strain outside the heart (research/flow_phantom.md Tier B5): the great vessels, lungs and mediastinum around the heart
# should not be compressed or folded by the heart's motion — smallest singular value of I + ∇u and smallest det over the
# cycle, outside the heart tissue within 40 mm of it (every 2nd voxel per axis)
near = (d_out > 0) & (d_out <= 40e-3)
near[1::2] = False
near[:, 1::2] = False
near[:, :, 1::2] = False
smin_t, detmin_t = [], []
for i in range(NT):
    g = np.empty((int(near.sum()), 3, 3))
    for c in range(3):
        gz, gy, gx = np.gradient(U_t[i][..., c].astype(np.float64), h)
        g[:, c, 0], g[:, c, 1], g[:, c, 2] = gx[near], gy[near], gz[near]
    Fm = np.eye(3)[None] + g
    sv = np.linalg.svd(Fm, compute_uv=False)
    smin_t.append(sv[:, -1])
    detmin_t.append(np.linalg.det(Fm))
smin_v = np.min(smin_t, axis=0)
summary["outside_heart_40mm"] = {"stretch_min": float(smin_v.min()), "stretch_p1": float(np.percentile(smin_v, 1)), "stretch_p5": float(np.percentile(smin_v, 5)),
                                 "det_min": float(np.min(detmin_t)), "outside": OUTSIDE, "taper_mm": TAPER * 1e3 if OUTSIDE == "taper" else None, "insulated_vessels": INSUL}
print(f"outside the heart (≤ 40 mm): smallest stretch over the cycle min {smin_v.min():.2f}, p1 {np.percentile(smin_v, 1):.2f}, p5 {np.percentile(smin_v, 5):.2f}; det min {np.min(detmin_t):.2f} (outside: {OUTSIDE}{', insulated vessels' if INSUL else ''})", flush=True)
if np.min(detmin_t) <= 0 or "--fold-report" in A:
    # where: the folded / most compressed cells (position, label, distance to the heart and to the tether set)
    _dm = np.min(detmin_t, axis=0)
    _idx = np.argsort(_dm)[:12]
    _pos = X[near][_idx]
    _lb = lab[near][_idx]
    _dt = ndimage.distance_transform_edt(~TM, sampling=h)[near][_idx] if TETHER == "on" else np.full(len(_idx), np.nan)
    for _p, _l, _d, _v, _do in zip(_pos, _lb, _dt, _dm[_idx], d_out[near][_idx]):
        print(f"   det {_v:6.2f} at {np.round(_p * 1e3, 1)} mm, {names[_l]}, {_do * 1e3:.0f} mm from the heart, {_d * 1e3:.0f} mm from the tether", flush=True)
assert np.min(detmin_t) > 0, "the cardiac motion folds outside the heart (det J ≤ 0)"
# the filling waves (Doppler-like numbers: E / A peaks, peak filling rate in EDV/s)
_fill = {}
for _nm, _q in (("mitral", Q_MV), ("tricuspid", Q_TV)):
    _qq = np.asarray(_q, float)
    _tq = np.arange(len(_qq)) / len(_qq) * Tp  # (the flows' own time grid: tt is the output samples by now)
    _split = Tp - 0.25 * S_HR if FILLING == "physio" else 0.75 * Tp
    _e, _a = _qq[_tq < _split], _qq[_tq >= _split]
    _fill[_nm] = {"E_peak_ml_s": float(_e.max() * 1e6), "A_peak_ml_s": float(_a.max() * 1e6), "E_over_A": float(_e.max() / max(_a.max(), 1e-12))}
summary["filling"] = {"shape": FILLING, "fe": FE, **_fill}
print(f"filling ({FILLING}): mitral E {_fill['mitral']['E_peak_ml_s']:.0f} / A {_fill['mitral']['A_peak_ml_s']:.0f} mL/s (E/A {_fill['mitral']['E_over_A']:.2f}), tricuspid E/A {_fill['tricuspid']['E_over_A']:.2f}")
summary["valve_disease"] = {"spec": VDIS, "regurgitant_volume_ml": {k: v * 1e6 for k, v in VD.totals(VDIS, Tp, P_SYS_, P_PUL_).items()}} if VDIS else None
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
