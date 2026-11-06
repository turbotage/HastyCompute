"""Assemble the phantom ground truth on one grid: tissue labels (`phantom_labels.py`) + vessel lumens / walls and
blood velocity (`phantomflow`), gates against the network solution, figures, and an optional flow-viewer export.

    python -I phantom_truth.py <flow.hv> <tissue.hv> <tissues.json> <export_dir> <flow_dir> <out_dir>
        [--frames 20] [--viewer-crop xmin,xmax,ymin,ymax,zmin,zmax (mm)] [--viewer-step k] [--publish] [--server URL]

Writes <out_dir>/labels.hv (final u8 labels, tissue table), gates.json, figs/, and with --viewer-crop a flow value
viewer_flow.hv (blood-mean velocity [F, 3, Z, Y, X], spoiled-GRE magnitude, mask, spacing, dt); --publish sends it to
the hasty server (flow viewer).
Gates:
- plane flux (mass conservation): Φ(t) = ∫ v·n dA through a disk (1.6 r) across the vessel — voxel-mean velocity,
  trilinear, voxels owned by edges whose lumen crosses the plane entirely inside the disk (the vessel, duplicate
  splines, branches still inside it; not a pressed neighbour cut by the disk) — vs Σ network Q(t) of those crossings; relative RMS over the cycle;
- divergence: |∇·v| h / |v| in voxels fully inside the lumen (all 6 neighbours too), at peak systole.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
FL, TS, TT, EXP, FLD, OUT = vf.read(A[0]), vf.read(A[1]), json.loads(Path(A[2]).read_text()), Path(A[3]), Path(A[4]), Path(A[5])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


NF = int(opt("--frames", 20))
nx, ny, nz = [int(x) for x in FL["dims_xyz"]]
h = float(FL["voxel_m"])
o = np.asarray(FL["origin_m"], float)
assert [int(x) for x in TS["dims_xyz"]] == [nx, ny, nz] and abs(float(TS["voxel_m"]) - h) < 1e-12, "tissue and flow grids differ"
idx = np.asarray(FL["idx"]).astype(np.int64)
assert np.all(np.diff(idx) > 0)
blood, wall, cls, samp = np.asarray(FL["blood"]), np.asarray(FL["wall"]), np.asarray(FL["cls"]), np.asarray(FL["sample"]).astype(np.int64)
samp[samp == 0xFFFFFFFF] = -1  # wall-only voxels
vh = np.asarray(FL["vh"])  # [N, H+1, 3, 2]
vh = vh[..., 0] + 1j * vh[..., 1]
om = np.asarray(FL["omega"], float)
T = float(FL["period"])
tt = np.arange(NF) * T / NF


def synth(c, t):
    """c [..., H+1] → [..., len(t)]"""
    return np.real(c[..., :1] + 2 * (c[..., 1:] @ np.exp(1j * om[1:, None] * np.asarray(t)[None, :])))


# ── labels ─────────────────────────────────────────────────────────────────────────────────────────────────────────
names = list(TS["tissues"])
ID = {n: i for i, n in enumerate(names)}
lab = np.asarray(TS["labels"]).copy()
flat = lab.reshape(-1)
chamber = np.isin(flat[idx], [ID[n] for n in ("blood_LA", "blood_LV", "blood_RA", "blood_RV")])
VCLS = {1: "blood_artery", 2: "blood_vein", 3: "blood_pulm_artery", 4: "blood_pulm_vein", 5: "blood_artery", 6: "blood_vein"}
isb = (blood >= 0.5) & ~chamber
for c, n in VCLS.items():
    flat[idx[isb & (cls == c)]] = ID[n]
isw = (blood < 0.5) & (wall >= 0.5) & ~chamber
flat[idx[isw]] = ID["vessel_wall"]
vf.write(OUT / "labels.hv", {"labels": lab, "tissues": names, **{k: np.asarray(TS[k]) for k in ("pd", "ff", "t1_s", "t2_s", "t2s_s", "chi_ppm")},
                              "origin_m": list(map(float, o)), "voxel_m": h, "dims_xyz": [nx, ny, nz],
                              "source": "Z-Anatomy (CC BY-SA 4.0) meshes and centrelines; phantomvox + phantomflow"})
vol = {n: float((lab == i).sum() * (h * 100) ** 3) for i, n in enumerate(names) if (lab == i).any()}

# ── geometry of the graph (centreline samples → edges, names) ──────────────────────────────────────────────────────
g = np.load(EXP / "graph.npz")
gj = json.loads((EXP / "graph.json").read_text())
enames = gj["edge_names"]
eoff, P, R = g["edge_offsets"], g["points"].astype(np.float64), g["radius"].astype(np.float64)
edge_of_sample = np.searchsorted(eoff, np.arange(len(P)), side="right") - 1
net = np.load(FLD / "network_flow.npz")
qn = net["q"][:, : vh.shape[1]]


def rows_of(lin):
    """sparse row of each linear voxel index (−1 if absent)"""
    r = np.searchsorted(idx, lin)
    r = np.minimum(r, len(idx) - 1)
    return np.where(idx[r] == lin, r, -1)


def sample_vh(pts, own_edges):
    """trilinear voxel-mean velocity harmonics at points [M, 3]; voxels not owned by own_edges count as 0"""
    f = (pts - o) / h - 0.5
    i0 = np.floor(f).astype(np.int64)
    w = f - i0
    acc = np.zeros((len(pts), vh.shape[1], 3), complex)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ii = i0 + [dx, dy, dz]
                ok = np.all((ii >= 0) & (ii < [nx, ny, nz]), axis=1)
                lin = np.where(ok, (ii[:, 2] * ny + ii[:, 1]) * nx + ii[:, 0], 0)
                r = rows_of(lin)
                good = ok & (r >= 0)
                good[good] &= np.isin(edge_of_sample[np.maximum(samp[r[good]], 0)], own_edges) & (samp[r[good]] >= 0)
                wt = (w[:, 0] if dx else 1 - w[:, 0]) * (w[:, 1] if dy else 1 - w[:, 1]) * (w[:, 2] if dz else 1 - w[:, 2])
                acc[good] += wt[good, None, None] * vh[r[good]]
    return acc  # [M, H+1, 3]


deg_ = np.bincount(np.r_[g["edge_a"], g["edge_b"]])
adj_ = {}
for k_ in range(len(enames)):
    for n_ in (int(g["edge_a"][k_]), int(g["edge_b"][k_])):
        adj_.setdefault(n_, []).append(k_)
elen_ = np.array([np.linalg.norm(np.diff(P[eoff[k_] : eoff[k_ + 1]], axis=0), axis=1).sum() for k_ in range(len(enames))])


def chain_dist(n, k_from, limit=0.2):
    """geodesic distance from node n (leaving edge k_from) through degree-2 nodes to the nearest junction or end"""
    d = 0.0
    while deg_[n] == 2 and d < limit:
        k_next = [k_ for k_ in adj_[n] if k_ != k_from][0]
        d += elen_[k_next]
        n = int(g["edge_b"][k_next]) if int(g["edge_a"][k_next]) == n else int(g["edge_a"][k_next])
        k_from = k_next
    return d


GATE = ["Ascending aorta", "Thoracic aorta", "Brachiocephalic trunk", "Left common carotid artery", "Left subclavian artery",
        "connector: Bifurcation of pulmonary trunk", "Right pulmonary artery", "Left pulmonary artery", "Superior vena cava",
        "Inferior vena cava (thoracic part)", "Right superior pulmonary vein", "Left inferior pulmonary vein", "Right coronary artery", "Azygos vein",
        "Internal thoracic arteries.l", "Hepatic portal vein"]
gates, curves = {}, {}
lo_w, hi_w = o, o + h * np.array([nx, ny, nz])
for nm in GATE:
    ks = [k for k in range(len(enames)) if enames[k] == nm]
    if not ks:
        continue
    k = max(ks, key=lambda k: (eoff[k + 1] - eoff[k]) * R[eoff[k] : eoff[k + 1]].mean())
    s0, s1 = int(eoff[k]), int(eoff[k + 1])
    # gate sample: farthest (geodesic distance / r, through degree-2 nodes) from any junction or end, among the samples
    # whose centreline voxel the edge itself owns (not still inside a parent, not in a junction zone)
    arc_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[s0:s1], axis=0), axis=1))]
    Da, Db = chain_dist(int(g["edge_a"][k]), k), chain_dist(int(g["edge_b"][k]), k)
    clear = np.minimum(arc_ + Da, arc_[-1] - arc_ + Db) / R[s0:s1]
    sm = None
    for cand_s in s0 + np.argsort(-clear):
        iv = np.floor((P[cand_s] - o) / h).astype(int)
        if np.any(iv < 0) or np.any(iv >= [nx, ny, nz]):
            continue
        rw = rows_of(np.array([(iv[2] * ny + iv[1]) * nx + iv[0]]))[0]
        if rw >= 0 and samp[rw] >= 0 and edge_of_sample[samp[rw]] == k:
            sm = cand_s
            break
    if sm is None:
        continue
    c = P[sm]
    r = R[sm]
    if np.any(c - 2 * r < lo_w) or np.any(c + 2 * r > hi_w) or s1 - s0 < 5:
        continue
    t = P[min(sm + 2, s1 - 1)] - P[max(sm - 2, s0)]
    t /= np.linalg.norm(t)
    u = np.cross(t, [1, 0, 0] if abs(t[0]) < 0.9 else [0, 1, 0])
    u /= np.linalg.norm(u)
    w2 = np.cross(t, u)
    # every edge whose centreline crosses the plane inside the disk (the vessel, parallel duplicate splines, branches
    # running along it before they leave its lumen): their voxels count for the flux, their Q for the reference
    qref = np.zeros(qn.shape[1], complex)
    own = set()
    near = np.flatnonzero(np.linalg.norm(P - c, axis=1) < 1.6 * r + 1e-3)
    for k2 in np.unique(edge_of_sample[near]):
        a2, b2 = int(eoff[k2]), int(eoff[k2 + 1])
        x = (P[a2:b2] - c) @ t
        for q0 in np.flatnonzero((x[:-1] < 0) != (x[1:] < 0)):
            if np.linalg.norm(P[a2 + q0] - c) + R[a2 + q0] <= 1.6 * r + 0.5 * h:  # its whole lumen lies in the disk
                qref += np.sign((P[a2 + q0 + 1] - P[a2 + q0]) @ t) * qn[a2 + q0]
                own.add(int(k2))
    own = sorted(own)
    ds = min(h / 4, r / 12)
    ax = np.arange(-1.6 * r, 1.6 * r + ds / 2, ds)
    U, V = np.meshgrid(ax, ax, indexing="ij")
    m = U**2 + V**2 <= (1.6 * r) ** 2
    pts = c + U[m, None] * u + V[m, None] * w2
    vs = sample_vh(pts, own)
    flux_h = (vs @ t).sum(0) * ds * ds  # [H+1]
    phi = synth(flux_h, tt)
    qq = synth(qref, tt)
    err = float(np.sqrt(np.mean((phi - qq) ** 2)) / max(np.sqrt(np.mean(qq**2)), 1e-12))
    gates[nm] = {"edges_crossing": len(own), "r_mm": round(r * 1e3, 2), "r_over_h": round(r / h, 1), "flux_rel_rms": round(err, 4), "mean_q_ml_s": round(float(qq.mean() * 1e6), 2), "mean_phi_ml_s": round(float(phi.mean() * 1e6), 2)}
    curves[nm] = (phi, qq)

# divergence at peak systole (time of the maximum aortic network flow)
ka = [k for k in range(len(enames)) if enames[k] == "Ascending aorta"]
t_pk = tt[np.argmax(synth(qn[(eoff[ka[0]] + eoff[ka[0] + 1]) // 2], tt))] if ka else 0.12
vpk = synth(vh.transpose(0, 2, 1), [t_pk])[..., 0]  # [N, 3] voxel-mean velocity
full = blood >= 1.0 - 1e-6
iz, iy, ix = idx // (nx * ny), (idx // nx) % ny, idx % nx
div = np.zeros(len(idx))
ok = full.copy()
for a, (d, n) in enumerate(((1, nx), (nx, ny), (nx * ny, nz))):
    pos = (ix, iy, iz)[a]
    rp, rm = rows_of(idx + d), rows_of(idx - d)
    good = (pos + 1 < n) & (pos > 0) & (rp >= 0) & (rm >= 0)
    good[good] &= full[rp[good]] & full[rm[good]]
    ok &= good
    div[good] += (vpk[rp[good], a] - vpk[rm[good], a]) / (2 * h)
sp = np.linalg.norm(vpk, axis=1)
sel = ok & (sp > 0.02)
rel = np.abs(div[sel]) * h / sp[sel]
gates["divergence"] = {"t_s": round(float(t_pk), 3), "voxels": int(sel.sum()), "rel_median": float(np.median(rel)) if sel.any() else None,
                       "rel_p95": float(np.percentile(rel, 95)) if sel.any() else None}
summary = {"grid": {"dims_xyz": [nx, ny, nz], "voxel_mm": h * 1e3, "origin_m": o.tolist(), "sub": int(FL["sub"])}, "frames": NF, "period_s": T,
           "blood_ml": float(blood.sum() * h**3 * 1e6), "wall_ml": float(wall.sum() * h**3 * 1e6), "gates": gates, "volumes_cm3": vol}
(OUT / "gates.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(gates, indent=1))

# ── figures ────────────────────────────────────────────────────────────────────────────────────────────────────────
vfr = synth(vh.transpose(0, 2, 1), tt)  # [N, 3, F] voxel-mean
bm = np.where(blood[:, None, None] > 0.25, vfr / np.maximum(blood, 1e-6)[:, None, None], 0)  # blood-mean
spd = np.linalg.norm(bm, axis=1)  # [N, F]
fpk = int(np.argmin(np.abs(tt - t_pk)))
dense = np.zeros(nx * ny * nz, np.float32)
dense[idx] = spd[:, fpk]
D = dense.reshape(nz, ny, nx)
mipc = D.max(axis=1)[::-1]
mips = D.max(axis=2)[::-1]
ext_c = [o[0] * 1e3, (o[0] + nx * h) * 1e3, o[2] * 1e3, (o[2] + nz * h) * 1e3]
ext_s = [o[1] * 1e3, (o[1] + ny * h) * 1e3, o[2] * 1e3, (o[2] + nz * h) * 1e3]
vmax = float(np.percentile(spd[:, fpk][blood > 0.5], 99.5))
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
for a, (im_, ex, tl) in zip(axs, [(mipc, ext_c, "coronal MIP"), (mips, ext_s, "sagittal MIP")]):
    im = a.imshow(im_, cmap="inferno", vmin=0, vmax=vmax, extent=ex, interpolation="nearest")
    a.set_title(f"{tl} of blood speed at peak systole (t = {tt[fpk] * 1e3:.0f} ms)")
    a.set_xlabel("mm")
fig.colorbar(im, ax=axs, fraction=0.02, label="m/s")
fig.savefig(OUT / "figs" / "01_speed_mip.png", dpi=100, bbox_inches="tight")
plt.close(fig)

n = len(curves)
if n:
    cols = 4
    rws = (n + cols - 1) // cols
    fig, axs = plt.subplots(rws, cols, figsize=(4.2 * cols, 3.2 * rws), squeeze=False)
    for a, (nm, (phi, qq)) in zip(axs.flat, curves.items()):
        a.plot(tt * 1e3, qq * 1e6, "k-", lw=2, label="network Q(t)")
        a.plot(tt * 1e3, phi * 1e6, "r--", lw=1.5, label="plane flux")
        a.set_title(f"{nm[:34]} (r/h {gates[nm]['r_over_h']}): {gates[nm]['flux_rel_rms'] * 100:.1f} %", fontsize=8)
        a.tick_params(labelsize=7)
    for a in list(axs.flat)[n:]:
        a.axis("off")
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(f"gate: plane flux of the voxelised truth vs network flow [mL/s], voxel {h * 1e3:.2f} mm, {int(FL['sub'])}³ sub-samples")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "02_flux_gate.png", dpi=90)
    plt.close(fig)

# velocity profile across the ascending aorta over the cycle
if "Ascending aorta" in gates:
    k = max(ka, key=lambda k: eoff[k + 1] - eoff[k])
    sm = (eoff[k] + eoff[k + 1]) // 2
    c, r = P[sm], R[sm]
    t = P[sm + 2] - P[sm - 2]
    t /= np.linalg.norm(t)
    u = np.cross(t, [1, 0, 0])
    u /= np.linalg.norm(u)
    xs = np.linspace(-1.2 * r, 1.2 * r, 121)
    own = [k2 for k2 in range(len(enames)) if enames[k2] == "Ascending aorta"]
    vs = sample_vh(c + xs[:, None] * u, own) @ t  # [M, H+1] voxel-mean along the line
    prof = synth(vs, tt)
    fig, a = plt.subplots(figsize=(8, 5))
    for f in range(0, NF, max(1, NF // 10)):
        a.plot(xs * 1e3, prof[:, f] * 100, label=f"{tt[f] * 1e3:.0f} ms")
    a.axvline(-r * 1e3, color="k", lw=0.5)
    a.axvline(r * 1e3, color="k", lw=0.5)
    a.set_xlabel("mm across the ascending aorta")
    a.set_ylabel("axial voxel-mean velocity [cm/s]")
    a.set_title("Womersley profiles over the cycle (voxelised truth)")
    a.legend(fontsize=7, ncol=2)
    fig.savefig(OUT / "figs" / "03_aorta_profiles.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

# ── viewer export ──────────────────────────────────────────────────────────────────────────────────────────────────
if opt("--viewer-crop"):
    c6 = np.array([float(x) for x in opt("--viewer-crop").split(",")]) * 1e-3
    k = int(opt("--viewer-step", 1))
    lo = np.clip(np.floor((c6[0::2] - o) / h).astype(int), 0, [nx, ny, nz])
    hi = np.clip(np.ceil((c6[1::2] - o) / h).astype(int), 0, [nx, ny, nz])
    hi = lo + (hi - lo) // k * k
    cx, cy, cz = (hi - lo) // k
    inb = (ix >= lo[0]) & (ix < hi[0]) & (iy >= lo[1]) & (iy < hi[1]) & (iz >= lo[2]) & (iz < hi[2])
    cl = (((iz[inb] - lo[2]) // k) * cy + (iy[inb] - lo[1]) // k) * cx + (ix[inb] - lo[0]) // k
    vsum = np.zeros((NF, 3, cz * cy * cx), np.float32)
    bsum = np.zeros(cz * cy * cx, np.float32)
    for a in range(3):
        for f in range(NF):
            np.add.at(vsum[f, a], cl, vfr[inb, a, f])
    np.add.at(bsum, cl, blood[inb])
    bsum /= k**3
    vsum /= k**3
    vel = np.where(bsum > 0.25, vsum / np.maximum(bsum, 1e-6), 0).reshape(NF, 3, cz, cy, cx).astype(np.float32)
    # magnitude: spoiled GRE of the tissue labels (TR 5 ms, 15°), blood with ×1.5 inflow enhancement, box-averaged
    pd, t1, ff = np.asarray(TS["pd"]), np.asarray(TS["t1_s"]), np.asarray(TS["ff"])
    e1 = np.exp(-5e-3 / t1)
    sig = pd * np.sin(np.deg2rad(15)) * (1 - e1) / (1 - np.cos(np.deg2rad(15)) * e1)
    for n_ in ("blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein", "blood_LA", "blood_LV", "blood_RA", "blood_RV"):
        sig[ID[n_]] *= 1.5
    L = lab[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]
    mag = sig[L].reshape(cz, k, cy, k, cx, k).mean(axis=(1, 3, 5)).astype(np.float32)
    mask = (bsum.reshape(cz, cy, cx) >= 0.5).astype(np.uint8)
    fv = {"kind": "flow4d", "vel": vel, "mag": np.repeat(mag[None], NF, 0), "spacing": [h * k] * 3, "dt": T / NF, "venc": 1.5, "mask": mask,
          "origin_m": list(map(float, o + lo * h)), "source": "atlas flow phantom P0 (Z-Anatomy CC BY-SA 4.0)"}
    vf.write(OUT / "viewer_flow.hv", fv)
    print(f"viewer flow: {[cx, cy, cz]} voxels of {h * k * 1e3:.2f} mm, {NF} frames → {OUT / 'viewer_flow.hv'}")
    if "--publish" in A:
        from hastycompute.v2 import Client  # noqa: E402

        cli = Client(opt("--server"))
        i = cli.put(fv, name=f"phantom P0 truth {h * k * 1e3:.1f} mm", metadata={"kind": "flow4d", "phantom": str(OUT)})
        cli.show(i, "flow")
        print("viewer:", cli.viewer_url(i, "flow"))
print("wrote", OUT)
