"""Diffeomorphic variants of the phantom atlas: one smooth, invertible map φ of the whole thorax, applied to every
anatomical input of the factory (tissue label volumes, vessel graph), so that every later step (network flow, cardiac
and respiratory motion, moving truth, LBM case) re-derives its physics on the deformed anatomy unchanged.

    rustlib/.venv/bin/python -I rustlib/tools/phantom/diffeo.py <out_atlas_dir> --spec <spec.json | preset> [--seed s]
        [--src-atlas data/hasty/phantom/p0] [--src-graph data/hasty/external/zanatomy/export] [--grid-mm 2]

φ is the time-1 flow of a stationary velocity field v (φ = exp(v) by scaling and squaring, φ⁻¹ = exp(−v)): a flow of
a Lipschitz field is a diffeomorphism, so tissues never tear, overlap or change topology. v lives on a grid over the
atlas box (+ margin) and is windowed to zero at its edges: φ is the identity outside the thorax box, where the vessel
graph continues (head, arms, abdomen, legs). Components (all in mm, none tied to a region of the atlas except through
the vessel names a feature is placed on):
- habitus: a Gaussian random field (correlation `rbf_mm`, peak `amp_mm`) plus an anisotropic scaling about the box
  centre (`log_scale` LR, AP, SI);
- features (list), placed on a named vessel of the graph at a fraction `at` of its arclength:
  - "bulge" / "narrow": radial dilation of the lumen by `factor` (> 1 / < 1) over `length_mm` along the vessel
    (anisotropic Gaussian kernel: σ_r = 1.5 R across, σ_a = length / 2 along, fixed tangent: smooth everywhere);
  - "rotate": a rotation by `deg` about `axis` within a ball of radius `radius_mm` around the point (heart axis).
Labels: L'(x) = L(φ⁻¹(x)) (nearest), on each label grid. Graph: points φ(p), radii × the transverse stretch
sqrt(det J / |J t|). Gates: min det J > 0 on the grid, inverse consistency |φ(φ⁻¹(x)) − x|. Figures: labels before /
after, det J. Writes the spec and the gates to diffeo.json.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
OUT = Path(A[0])


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


SRC_ATLAS = Path(opt("--src-atlas", REPO / "data/hasty/phantom/p0"))
SRC_GRAPH = Path(opt("--src-graph", REPO / "data/hasty/external/zanatomy/export"))
GH = float(opt("--grid-mm", 3.0)) * 1e-3
VOLS = ["tissue_2mm", "tissue_1mm", "tissue_0.5mm_heart"]

PRESETS = {
    # body habitus only
    "habitus": {"habitus": {"amp_mm": 8.0, "rbf_mm": 70.0, "log_scale": [0.06, 0.05, 0.02]}, "features": []},
    # ascending aortic aneurysm (dilated ascending aorta) + habitus
    "aneurysm": {"habitus": {"amp_mm": 6.0, "rbf_mm": 80.0, "log_scale": [0.04, -0.03, 0.0]},
                 "features": [{"kind": "bulge", "vessel": "Ascending aorta", "at": 0.6, "factor": 1.45, "length_mm": 40.0}]},
    # supravalvular narrowing of the ascending aorta + dilated pulmonary trunk + heart axis rotation
    "narrow": {"habitus": {"amp_mm": 6.0, "rbf_mm": 80.0, "log_scale": [-0.04, 0.04, 0.01]},
               "features": [{"kind": "narrow", "vessel": "Ascending aorta", "at": 0.35, "factor": 0.7, "length_mm": 18.0},
                            {"kind": "bulge", "vessel": "Bifurcation of pulmonary trunk", "at": 0.3, "factor": 1.25, "length_mm": 30.0}]},
}
sp = opt("--spec", "habitus")
SPEC = json.loads(Path(sp).read_text()) if Path(sp).exists() else json.loads(json.dumps(PRESETS[sp]))
SPEC["seed"] = int(opt("--seed", SPEC.get("seed", 1)))
SPEC["preset"] = None if Path(sp).exists() else sp
rng = np.random.default_rng(SPEC["seed"])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figs").mkdir(exist_ok=True)

# ── grid over the atlas box ──────────────────────────────────────────────────────────────────────────────────────────
ref = vf.read(SRC_ATLAS / "tissue_2mm.hv")
o_at = np.asarray(ref["origin_m"], float)
ext = np.asarray(ref["dims_xyz"], float) * float(ref["voxel_m"])
# the field is tapered to zero in a margin around the atlas box (it stays full strength over the whole atlas, whose
# edges cut through the body: a taper inside it would compress the body there)
MARG = float(opt("--margin-mm", 150.0)) * 1e-3
g0 = o_at - MARG
nx = np.ceil((ext + 2 * MARG) / GH).astype(int) + 1  # x, y, z
gz, gy, gx = np.meshgrid(*(g0[a] + GH * np.arange(nx[a]) for a in (2, 1, 0)), indexing="ij")
X = np.stack([gx, gy, gz], -1).astype(np.float32)  # [Z, Y, X, 3]
del gx, gy, gz
cen = o_at + ext / 2
print(f"grid {nx.tolist()} at {GH * 1e3:.1f} mm over {(ext * 1e3).round().tolist()} mm (+{MARG * 1e3:.0f} mm), centre {cen.round(3).tolist()}")


def window(Xg, ramp=MARG - 0.01):
    """1 inside, smooth (C²) ramp to 0 at the box edges over `ramp` m"""
    w = np.ones(Xg.shape[:-1], np.float32)
    for a in range(3):
        lo, hi = g0[a], g0[a] + GH * (nx[a] - 1)
        d = np.minimum(Xg[..., a] - lo, hi - Xg[..., a]) / ramp
        t = np.clip(d, 0, 1)
        w *= t**3 * (10 - 15 * t + 6 * t * t)
    return w


W = window(X)

# ── velocity field ───────────────────────────────────────────────────────────────────────────────────────────────────
V = np.zeros_like(X)
hab = SPEC.get("habitus", {})
if hab.get("amp_mm", 0) > 0:
    s = hab.get("rbf_mm", 70.0) * 1e-3 / GH
    n = np.stack([ndimage.gaussian_filter(rng.standard_normal(X.shape[:-1]), s, mode="constant") for _ in range(3)], -1)
    n *= W[..., None]
    n *= hab["amp_mm"] * 1e-3 / np.sqrt((n**2).sum(-1)).max()
    V += n
ls = np.asarray(hab.get("log_scale", [0, 0, 0]), float)
if np.abs(ls).max() > 0:
    # the flow of v = diag(ls)(x − c) scales by e^{ls} about c; windowed
    V += (X - cen) * ls * W[..., None]

G = np.load(SRC_GRAPH / "graph.npz")
GJ = json.loads((SRC_GRAPH / "graph.json").read_text())
eoff = G["edge_offsets"]
names = GJ["edge_names"]
P = G["points"].astype(np.float64)
R = G["radius"].astype(np.float64)


def vessel_point(name, frac):
    """the point at arclength fraction `frac` of the longest edge named `name` (exact or after 'connector: '), its
    tangent and radius"""
    ks = [k for k, nm in enumerate(names) if nm == name]
    if not ks:
        ks = [k for k, nm in enumerate(names) if nm.endswith(": " + name) or nm.startswith(name)]
    if not ks:
        raise SystemExit(f"no vessel named {name!r}")
    k = max(ks, key=lambda k_: eoff[k_ + 1] - eoff[k_])
    p = P[eoff[k] : eoff[k + 1]]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    i = int(np.clip(np.searchsorted(s, frac * s[-1]), 1, len(p) - 2))
    t = p[i + 1] - p[i - 1]
    return p[i], t / np.linalg.norm(t), float(R[eoff[k] + i]), s[-1]


def radial_eps(factor, sr, r0):
    """ε of the radial field dρ/dt = ε ρ e^{−ρ²/2σ²} that takes ρ = r0 to factor · r0 at t = 1 (bisection, RK4)"""
    def end(eps):
        r = r0
        for _ in range(64):
            f = lambda q: eps * q * np.exp(-q * q / (2 * sr * sr))  # noqa: E731
            k1 = f(r)
            k2 = f(r + k1 / 128)
            k3 = f(r + k2 / 128)
            k4 = f(r + k3 / 64)
            r += (k1 + 2 * k2 + 2 * k3 + k4) / 384
        return r
    lo, hi = (-3.0, 0.0) if factor < 1 else (0.0, 3.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if end(mid) < factor * r0:  # end(ε) increases with ε for either sign
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


feat_log = []
for f in SPEC.get("features", []):
    c, t, r0, L_ = vessel_point(f["vessel"], f.get("at", 0.5))
    d = X - c
    if f["kind"] in ("bulge", "narrow"):
        sr, sa = 1.5 * r0, 0.5 * f.get("length_mm", 30.0) * 1e-3
        ax = d @ t
        perp = d - ax[..., None] * t
        rho2 = (perp**2).sum(-1)
        eps = radial_eps(f["factor"], sr, r0)
        V += eps * perp * np.exp(-rho2 / (2 * sr * sr) - ax**2 / (2 * sa * sa))[..., None] * W[..., None]
        feat_log.append({**f, "center": c.tolist(), "tangent": t.tolist(), "radius_mm": r0 * 1e3, "eps": eps, "vessel_length_mm": L_ * 1e3})
    elif f["kind"] == "rotate":
        axis = np.asarray(f.get("axis", t), float)
        axis /= np.linalg.norm(axis)
        om = np.deg2rad(f["deg"]) * axis
        rb = f.get("radius_mm", 60.0) * 1e-3
        V += np.cross(om, d) * np.exp(-(d**2).sum(-1) / (2 * rb * rb))[..., None] * W[..., None]
        feat_log.append({**f, "center": c.tolist()})
    else:
        raise SystemExit(f"unknown feature {f['kind']}")
    print(f"feature {f['kind']} on {f['vessel']} at {f.get('at', 0.5)}: centre {np.round(c, 4).tolist()}, R {r0 * 1e3:.1f} mm", (f", ε {feat_log[-1]['eps']:.3f}" if "eps" in feat_log[-1] else ""))


# ── exp(±v) by scaling and squaring ──────────────────────────────────────────────────────────────────────────────────
def interp(F, Xq):
    """trilinear F [Z, Y, X, 3] at lab points Xq [..., 3]; zero (identity map) outside the grid"""
    idx = [(Xq[..., a] - g0[a]) / GH for a in (2, 1, 0)]
    return np.stack([ndimage.map_coordinates(F[..., c], idx, order=1, mode="constant", cval=0.0) for c in range(3)], -1)


def expmap(Vg, k=8):
    u = Vg / 2.0**k
    for _ in range(k):
        u = u + interp(u, X + u)
    return u


Lip = max(np.abs(np.gradient(V[..., c], GH, axis=a)).max() for c in range(3) for a in range(3))
U_f = expmap(V)
U_i = expmap(-V)
inv_err = np.sqrt(((U_i + interp(U_f, X + U_i)) ** 2).sum(-1)).max()
J = np.stack([np.stack(np.gradient(U_f[..., c], GH, axis=(2, 1, 0)), -1) for c in range(3)], -2) + np.eye(3)
detJ = np.linalg.det(J)
disp = np.sqrt((U_f**2).sum(-1))
gates = {"lipschitz_v": float(Lip), "det_j_min": float(detJ.min()), "det_j_max": float(detJ.max()), "max_disp_mm": float(disp.max() * 1e3),
         "inverse_consistency_mm": float(inv_err * 1e3)}
print("gates:", {k: round(v, 4) for k, v in gates.items()})
if gates["det_j_min"] <= 0.05:
    raise SystemExit("det J ≤ 0.05: the field is too strong for this grid")

# ── labels: L'(x) = L(φ⁻¹(x)) ────────────────────────────────────────────────────────────────────────────────────────
# every target grid samples the finest atlas volume that contains φ⁻¹(x) (0.5 mm heart crop, then 1 mm, then 2 mm):
# nearest-neighbour resampling of a grid onto itself at sub-voxel shifts would break one-voxel walls
SRC = []
for nm in ["tissue_0.5mm_heart", "tissue_1mm", "tissue_2mm"]:
    if (SRC_ATLAS / f"{nm}.hv").exists():
        T = vf.read(SRC_ATLAS / f"{nm}.hv")
        SRC.append((nm, T, np.asarray(T["labels"]), float(T["voxel_m"]), np.asarray(T["origin_m"], float)))
assert all(list(t[1]["tissues"]) == list(SRC[0][1]["tissues"]) for t in SRC), "atlas volumes with different tissue lists"


def sample_labels(Xs):
    out = np.zeros(Xs.shape[:-1], SRC[-1][2].dtype)
    done = np.zeros(Xs.shape[:-1], bool)
    for k_, (_, _, lab, h, o) in enumerate(SRC):
        Z, Y, Xn = lab.shape
        last = k_ == len(SRC) - 1
        ins = np.ones(Xs.shape[:-1], bool) if last else np.all([(Xs[..., a] >= o[a] + h) & (Xs[..., a] < o[a] + (n_ - 1) * h) for a, n_ in zip((0, 1, 2), (Xn, Y, Z))], axis=0)
        m = ins & ~done
        if m.any():
            # beyond the atlas edge (last source): the nearest edge voxel (the body continues past the box)
            ii = [np.clip(np.floor((Xs[..., a][m] - o[a]) / h).astype(int), 0, n_ - 1) for a, n_ in zip((2, 1, 0), (Z, Y, Xn))]
            out[m] = lab[ii[0], ii[1], ii[2]]
            done |= m
    return out


for nm in VOLS:
    src = SRC_ATLAS / f"{nm}.hv"
    if not src.exists():
        continue
    T = vf.read(src)
    lab = np.asarray(T["labels"])
    h = float(T["voxel_m"])
    o = np.asarray(T["origin_m"], float)
    Z, Y, Xn = lab.shape
    out = np.empty_like(lab)
    for z0 in range(0, Z, 32):
        zs = np.arange(z0, min(z0 + 32, Z))
        qz, qy, qx = np.meshgrid(zs, np.arange(Y), np.arange(Xn), indexing="ij")
        Xq = np.stack([o[0] + (qx + 0.5) * h, o[1] + (qy + 0.5) * h, o[2] + (qz + 0.5) * h], -1)
        out[zs[0] : zs[-1] + 1] = sample_labels(Xq + interp(U_i, Xq))
    T2 = dict(T)
    T2["labels"] = out
    T2["source"] = f"{T.get('source', '')} | diffeo.py {SPEC.get('preset') or sp} seed {SPEC['seed']}"
    vf.write(OUT / f"{nm}.hv", T2)
    ch = [(int(np.sum(lab == i)), int(np.sum(out == i))) for i in range(1, int(lab.max()) + 1)]
    tis = list(T["tissues"])
    big = sorted(range(len(ch)), key=lambda i: -ch[i][0])[:6]
    print(f"{nm}: {lab.shape} → volume change " + ", ".join(f"{tis[i + 1] if i + 1 < len(tis) else i + 1} {(ch[i][1] / max(ch[i][0], 1) - 1) * 100:+.1f} %" for i in big))

# ── graph: φ(p), radii × transverse stretch ──────────────────────────────────────────────────────────────────────────
Jg = J  # [Z, Y, X, 3, 3]
Pn = P + interp(U_f, P)
Jp = np.stack([np.stack([ndimage.map_coordinates(Jg[..., i, j], [(P[:, a] - g0[a]) / GH for a in (2, 1, 0)], order=1, mode="nearest") for j in range(3)], -1) for i in range(3)], -2)
outside = np.any([(P[:, a] < g0[a]) | (P[:, a] > g0[a] + GH * (nx[a] - 1)) for a in range(3)], axis=0)
Jp[outside] = np.eye(3)
Tg = np.zeros_like(P)
for k in range(len(eoff) - 1):
    s = slice(eoff[k], eoff[k + 1])
    p = P[s]
    if len(p) >= 2:
        Tg[s] = np.gradient(p, axis=0)
Tg /= np.maximum(np.linalg.norm(Tg, axis=1, keepdims=True), 1e-12)
# the lumen radius after φ: a ring of 8 wall points around each sample (⊥ the tangent), mapped through φ, its mean
# distance from the mapped centre perpendicular to the mapped tangent (finite strain: the centreline's local stretch
# overstates a dilation whose kernel peaks on the axis)
e1 = np.cross(Tg, np.where(np.abs(Tg[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]))
e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
e2 = np.cross(Tg, e1)
Tn = np.einsum("nij,nj->ni", Jp, Tg)
Tn /= np.maximum(np.linalg.norm(Tn, axis=1, keepdims=True), 1e-12)
rr = np.zeros(len(P))
for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
    q = P + R[:, None] * (np.cos(th) * e1 + np.sin(th) * e2)
    dq = q + interp(U_f, q) - Pn
    dq -= (dq * Tn).sum(1, keepdims=True) * Tn
    rr += np.linalg.norm(dq, axis=1) / 8
stretch = np.where(R > 0, rr / np.maximum(R, 1e-12), 1.0)
g2 = dict(G)
g2["points"] = Pn.astype(G["points"].dtype)
g2["radius"] = (R * stretch).astype(G["radius"].dtype)
if "radius_atlas" in g2:
    g2["radius_atlas"] = (G["radius_atlas"].astype(np.float64) * stretch).astype(G["radius_atlas"].dtype)
g2["node_pos"] = (G["node_pos"] + interp(U_f, G["node_pos"].astype(np.float64))).astype(G["node_pos"].dtype)
np.savez_compressed(OUT / "graph.npz", **g2)
shutil.copy(SRC_GRAPH / "graph.json", OUT / "graph.json")
moved = np.linalg.norm(Pn - P, axis=1)
print(f"graph: {len(P)} samples, {int((moved > 1e-6).sum())} moved (max {moved.max() * 1e3:.1f} mm), radius × {stretch.min():.3f}…{stretch.max():.3f}")
for f in feat_log:
    if "eps" in f:
        c = np.asarray(f["center"])
        j = int(np.argmin(np.linalg.norm(P - c, axis=1)))
        f["radius_after_mm"] = float(R[j] * stretch[j] * 1e3)
        print(f"  {f['kind']} {f['vessel']}: R {R[j] * 1e3:.1f} → {R[j] * stretch[j] * 1e3:.1f} mm (asked × {f['factor']})")

np.save(OUT / "u_forward_mm.npy", (U_f * 1e3).astype(np.float16))
np.save(OUT / "u_inverse_mm.npy", (U_i * 1e3).astype(np.float16))
(OUT / "diffeo.json").write_text(json.dumps({"spec": SPEC, "features": feat_log, "gates": gates, "grid": {"origin": g0.tolist(), "h": GH, "dims_xyz": nx.tolist()},
                                             "src_atlas": str(SRC_ATLAS), "src_graph": str(SRC_GRAPH)}, indent=1))

# ── figures ──────────────────────────────────────────────────────────────────────────────────────────────────────────
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T0, T1 = vf.read(SRC_ATLAS / "tissue_2mm.hv"), vf.read(OUT / "tissue_2mm.hv")
l0, l1 = np.asarray(T0["labels"]), np.asarray(T1["labels"])
h2, o2 = float(T0["voxel_m"]), np.asarray(T0["origin_m"], float)
fig, axs = plt.subplots(3, 3, figsize=(13, 11))
cz, cy, cx = [int(v) for v in (np.array(l0.shape) // 2)]
fz = [f for f in feat_log if "center" in f]
if fz:
    c = np.asarray(fz[0]["center"])
    cx, cy, cz = [int(np.clip((c[a] - o2[a]) / h2, 0, l0.shape[2 - a] - 1)) for a in range(3)]
cmap = plt.get_cmap("tab20", 31)
for r_, (title, sl) in enumerate([("axial", lambda a: a[cz]), ("coronal", lambda a: a[:, cy]), ("sagittal", lambda a: a[:, :, cx])]):
    axs[r_, 0].imshow(sl(l0), origin="lower", cmap=cmap, vmin=0, vmax=30, interpolation="nearest")
    axs[r_, 1].imshow(sl(l1), origin="lower", cmap=cmap, vmin=0, vmax=30, interpolation="nearest")
    gzz = np.clip(((np.array([cz, cy, cx]) * h2 + o2[::-1] + h2 / 2) - g0[::-1]) / GH, 0, np.array(detJ.shape) - 1).astype(int)
    dj = [detJ[gzz[0]], detJ[:, gzz[1]], detJ[:, :, gzz[2]]][r_]
    im = axs[r_, 2].imshow(dj, origin="lower", cmap="RdBu_r", vmin=0.6, vmax=1.4)
    for c_, t_ in enumerate(["atlas", "deformed", "det J (deformation grid)"]):
        axs[r_, c_].set_title(f"{title}: {t_}", fontsize=9, loc="left")
fig.colorbar(im, ax=axs[:, 2], shrink=0.6)
fig.suptitle(f"diffeo {SPEC.get('preset') or sp} seed {SPEC['seed']}: max |u| {gates['max_disp_mm']:.1f} mm, det J {gates['det_j_min']:.2f}…{gates['det_j_max']:.2f}", fontsize=10)
fig.savefig(OUT / "figs" / "diffeo_labels.png", dpi=110)
print(f"wrote {OUT}")
