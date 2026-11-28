"""Report of a cardiac LBM run (hasty-cfd `cardiac_lbm`): gates and figures.

    rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_report.py <case_dir> <run_dir> [--snap <snap_xxx.f32>]

From log.jsonl: cut fluxes against their targets (inlets: the prescribed inflow, outlets: the cardiac model's valve
flow), mean density deviation and peak speed over time.
From flow_bins.f32 (phase-binned mean velocity on the 1 mm output cells): per valve and bin the flux through the valve
disk relative to the annulus motion (vs the cardiac model's Q_AO / Q_PA / Q_MV / Q_TV), the jet speed (largest and 95th
percentile speed on the disk and 1 cm downstream), leakage while closed; the fastest well-covered cells per bin and
where they are; speed MIPs at peak systole, E, diastasis and A.
--snap: slices of a probe snapshot (cardiac_lbm --probe … --snap) through its fastest node.
Writes <run_dir>/report/{report.json, *.png}.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
CASE, RUN = Path(A[0]), Path(A[1])
OUT = RUN / "report"
OUT.mkdir(parents=True, exist_ok=True)


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


case = json.loads((CASE / "case.json").read_text())


# ── probe snapshot slices through its fastest node ──────────────────────────────────────────────────────────────────
def snap_slices(sp_, org_):
    hdr = json.loads(sp_.with_suffix(".json").read_text())
    dx = float(hdr["dx"])
    n = int(hdr["nodes"])
    D = np.fromfile(sp_, np.float32).reshape(6, n)
    tx = np.asarray(hdr["tile_xyz"], np.int64).reshape(-1, 3)
    li = np.arange(512)
    g = np.stack([(tx[:, None, 0] * 8 + li % 8).ravel(), (tx[:, None, 1] * 8 + li // 8 % 8).ravel(), (tx[:, None, 2] * 8 + li // 64).ravel()], 1)
    fl = D[5].astype(np.uint8)
    spd = np.sqrt(D[1] ** 2 + D[2] ** 2 + D[3] ** 2) * (fl & 1 > 0)
    k = int(np.argmax(spd))
    gk = g[k]
    print(f"snapshot {sp_.name}: fastest {spd[k]:.2f} m/s at g {gk.tolist()} (x {np.round((org_ + gk * dx) * 1e3, 1).tolist()} mm), ρ {D[0, k]:.3f}, B {D[4, k]:.2f}, flags {fl[k]:#04x}")
    gmin = g.min(0)
    shp = g.max(0) - gmin + 1
    fig, ax = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for a_, axis in enumerate((0, 1, 2)):
        sel = g[:, axis] == gk[axis]
        o_ = [d for d in (0, 1, 2) if d != axis]
        img = np.full((shp[o_[1]], shp[o_[0]]), np.nan)
        rho = np.full_like(img, np.nan)
        img[g[sel, o_[1]] - gmin[o_[1]], g[sel, o_[0]] - gmin[o_[0]]] = np.where(fl[sel] & 1, spd[sel], np.nan)
        rho[g[sel, o_[1]] - gmin[o_[1]], g[sel, o_[0]] - gmin[o_[0]]] = np.where(fl[sel] & 1, D[0, sel], np.nan)
        win = 40
        cy, cx = gk[o_[1]] - gmin[o_[1]], gk[o_[0]] - gmin[o_[0]]
        sl = (slice(max(cy - win, 0), cy + win), slice(max(cx - win, 0), cx + win))
        im = ax[0, a_].imshow(img[sl], origin="lower", cmap="magma", vmin=0, vmax=max(1.0, float(np.nanmax(img[sl]))))
        fig.colorbar(im, ax=ax[0, a_], shrink=0.7)
        ax[0, a_].set_title(f"speed, slice {'xyz'[axis]} = {gk[axis]} (±{win} nodes)", fontsize=9, loc="left")
        im = ax[1, a_].imshow(rho[sl], origin="lower", cmap="RdBu_r")
        fig.colorbar(im, ax=ax[1, a_], shrink=0.7)
        ax[1, a_].set_title("ρ", fontsize=9, loc="left")
    fig.savefig(OUT / f"{sp_.stem}_slices.png", dpi=100)
    plt.close(fig)


if opt("--snap"):
    snap_slices(Path(opt("--snap")), np.asarray(json.loads(Path(opt("--snap")).with_suffix(".json").read_text())["origin"], float))
    if not (RUN / "flow_bins.json").exists():
        sys.exit(0)
meta = json.loads((RUN / "flow_bins.json").read_text())
cm = vf.read(Path(case["motion"]["file"]))
T = float(cm["period"])
tq = np.asarray(cm["flows_t"], float)


# ── motion (as hasty-cfd motion.rs: periodic Catmull-Rom in time, trilinear in space) ───────────────────────────────
class Motion:
    def __init__(self, cm):
        self.U = cm["U_mm"]  # [nt, 3, Z, Y, X] f16, mm
        self.nt = self.U.shape[0]
        self.o = np.asarray(cm["origin_m"], float)
        self.h = float(cm["voxel_m"])
        self.T = float(cm["period"])

    def _tri(self, k, x):
        g = (np.asarray(x, float) - self.o) / self.h - 0.5  # voxel centres at o + (i + ½) h (hasty-cfd motion.rs)
        i0 = np.floor(g).astype(int)
        f = g - i0
        Z, Y, X = self.U.shape[2:]
        out = np.zeros(3)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ii = i0 + [dx, dy, dz]
                    if 0 <= ii[0] < X and 0 <= ii[1] < Y and 0 <= ii[2] < Z:
                        w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2])
                        out += w * np.asarray(self.U[k, :, ii[2], ii[1], ii[0]], float)
        return out * 1e-3

    def disp(self, x, t):
        s = (t % self.T) / self.T * self.nt
        k = int(np.floor(s))
        f = s - k
        p = [self._tri((k + j) % self.nt, x) for j in (-1, 0, 1, 2)]
        c = [-0.5 * f**3 + f**2 - 0.5 * f, 1.5 * f**3 - 2.5 * f**2 + 1, -1.5 * f**3 + 2 * f**2 + 0.5 * f, 0.5 * f**3 - 0.5 * f**2]
        return sum(ci * pi for ci, pi in zip(c, p))

    def fwd(self, x, t):
        return np.asarray(x, float) + self.disp(x, t)

    def _tri_v(self, k, P):
        g = (np.asarray(P, float) - self.o) / self.h - 0.5
        i0 = np.floor(g).astype(int)
        f = g - i0
        Z, Y, X = self.U.shape[2:]
        out = np.zeros((len(P), 3))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                    ok = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < X) & (iy < Y) & (iz < Z)
                    w = np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) * np.where(dz, f[:, 2], 1 - f[:, 2]) * ok
                    u = np.asarray(self.U[k, :, np.clip(iz, 0, Z - 1), np.clip(iy, 0, Y - 1), np.clip(ix, 0, X - 1)], float)  # [n, 3] (the scalar k joins the advanced indices)
                    out += w[:, None] * u
        return out * 1e-3

    def disp_v(self, P, t):
        s = (t % self.T) / self.T * self.nt
        k = int(np.floor(s))
        f = s - k
        c = [-0.5 * f**3 + f**2 - 0.5 * f, 1.5 * f**3 - 2.5 * f**2 + 1, -1.5 * f**3 + 2 * f**2 + 0.5 * f, 0.5 * f**3 - 0.5 * f**2]
        return sum(ci * self._tri_v((k + j) % self.nt, P) for ci, j in zip(c, (-1, 0, 1, 2)))

    def material_disk(self, c, n, e1, R, t, nr=24, nth=72, dt=1e-3):
        """the reference disk (centre c, normal n, radius R) carried by the tissue to time t: element centres, area
        vectors (∂X/∂ρ × ∂X/∂θ dρ dθ, along the moved normal) and the tissue velocity there"""
        n = np.asarray(n, float)
        e1 = np.asarray(e1, float)
        e2 = np.cross(n, e1)
        rho = (np.arange(nr) + 0.5) / nr * R
        th = np.arange(nth) / nth * 2 * np.pi
        Rg, Tg = np.meshgrid(rho, th, indexing="ij")
        P = c + Rg.ravel()[:, None] * (np.cos(Tg).ravel()[:, None] * e1 + np.sin(Tg).ravel()[:, None] * e2)
        er = np.cos(Tg).ravel()[:, None] * e1 + np.sin(Tg).ravel()[:, None] * e2
        et = -np.sin(Tg).ravel()[:, None] * e1 + np.cos(Tg).ravel()[:, None] * e2
        dr, dth = R / nr, 2 * np.pi / nth
        eps = 0.25 * dr
        X = P + self.disp_v(P, t)
        Xr = (P + eps * er + self.disp_v(P + eps * er, t) - (P - eps * er + self.disp_v(P - eps * er, t))) / (2 * eps)
        Xt = (P + eps * et + self.disp_v(P + eps * et, t) - (P - eps * et + self.disp_v(P - eps * et, t))) / (2 * eps)  # per unit arc length
        dA = np.cross(Xr, Xt) * (Rg.ravel() * dr * dth)[:, None]
        um = (self.disp_v(P, t + dt) - self.disp_v(P, t - dt)) / (2 * dt)
        return X, dA, um

    def vel(self, x, t, dt=1e-3):
        return (self.disp(x, t + dt) - self.disp(x, t - dt)) / (2 * dt)

    def frame(self, c, n, e1, t, eps=2e-3):
        """valve centre, normal, in-plane axes and annulus velocity at time t (the tissue carries the disk)"""
        n = np.asarray(n, float)
        e1 = np.asarray(e1, float)
        e2 = np.cross(n, e1)
        a1 = (self.fwd(c + eps * e1, t) - self.fwd(c - eps * e1, t)) / (2 * eps)
        a2 = (self.fwd(c + eps * e2, t) - self.fwd(c - eps * e2, t)) / (2 * eps)
        nn = np.cross(a1, a2)
        nn /= np.linalg.norm(nn)
        b1 = a1 / np.linalg.norm(a1)
        return self.fwd(c, t), nn, b1, np.cross(nn, b1), self.vel(c, t)


MO = Motion(cm)

# ── log: cut fluxes against targets ──────────────────────────────────────────────────────────────────────────────────
log = [json.loads(line) for line in (RUN / "log.jsonl").read_text().splitlines() if line.strip()]
tl = np.array([d["t"] for d in log])
ncut = len(case["cuts"])
# the cut flux: the exact link flux through the slab interface where the run logs it (hasty-cfd node_diag, 2026-11-26),
# else the u-based first-layer flux (2–3 % low at a velocity inlet); the layer value is kept as layer_mean_ml_s
_has_link = all("q_link_ml_s" in c for c in log[-1]["cuts"]) if log else False
qm = np.array([[c["q_link_ml_s" if _has_link else "q_out_ml_s"] for c in d["cuts"]] for d in log])
qlay = np.array([[c["q_out_ml_s"] for c in d["cuts"]] for d in log])
qt = np.array([[np.nan if c["target_ml_s"] is None else c["target_ml_s"] for c in d["cuts"]] for d in log])
drho = np.array([d["drho_mean"] for d in log])
vmax = np.array([d["v_max_m_s"] for d in log])
names = [c["name"].replace("connector: ", "") for c in case["cuts"]]
cyc = tl / T
warm = float(meta.get("warmup", 2))
late = cyc >= warm
rep = {"run": str(RUN), "cycles_logged": float(cyc[-1]) if len(cyc) else 0, "cuts": {}, "valves": {}}
for k in range(ncut):
    m, t_ = qm[late, k], qt[late, k]
    ok = np.isfinite(t_)
    r = {"mean_ml_s": float(m.mean()) if len(m) else None, "layer_mean_ml_s": float(qlay[late, k].mean()) if late.any() else None, "flux": "link" if _has_link else "layer"}
    if ok.sum() > 3:
        r.update({"target_mean_ml_s": float(t_[ok].mean()), "ratio_of_means": float(m[ok].mean() / t_[ok].mean()) if abs(t_[ok].mean()) > 1e-6 else None,
                  "corr": float(np.corrcoef(m[ok], t_[ok])[0, 1]), "rms_err_ml_s": float(np.sqrt(np.mean((m[ok] - t_[ok]) ** 2))), "peak_ml_s": float(m[np.argmax(np.abs(m))]),
                  "target_peak_ml_s": float(t_[ok][np.argmax(np.abs(t_[ok]))])})
    rep["cuts"][names[k]] = r
rep["drho_mean_by_cycle"] = [float(drho[(cyc >= c) & (cyc < c + 1)].mean()) for c in range(int(np.ceil(cyc[-1]))) if ((cyc >= c) & (cyc < c + 1)).any()]
rep["drho_range_late"] = [float(drho[late].min()), float(drho[late].max())] if late.any() else None
rep["vmax_late_m_s"] = float(vmax[late].max()) if late.any() else None

_nr = max(3, int(np.ceil((ncut + 1) / 4)))  # one panel per cut (+ the density panel)
fig, ax = plt.subplots(_nr, 4, figsize=(17, 10 * _nr / 3), constrained_layout=True)
for k in range(ncut):
    a = ax.flat[k]
    a.plot(tl * 1e3, qm[:, k], lw=1, color="#1f5fbf", label="LBM")
    a.plot(tl * 1e3, qt[:, k], lw=1, color="#d05a1a", ls="--", label="target")
    a.set_title(names[k][:34], fontsize=8, loc="left")
    a.set_xlabel("t [ms]", fontsize=7)
    a.set_ylabel("out [mL/s]", fontsize=7)
    a.tick_params(labelsize=7)
    for c in range(int(cyc[-1]) + 1):
        a.axvline(c * T * 1e3, color="0.85", lw=0.6)
ax.flat[0].legend(fontsize=7)
ax.flat[8].plot(tl * 1e3, drho, lw=1, color="#1f5fbf")
ax.flat[8].set_title("mean ρ − 1 over the fluid", fontsize=8, loc="left")
ax.flat[9].plot(tl * 1e3, vmax, lw=1, color="#1f5fbf")
ax.flat[9].set_title("largest node speed [m/s]", fontsize=8, loc="left")
for a in ax.flat[10:]:
    a.axis("off")
fig.savefig(OUT / "log_cuts.png", dpi=110)
plt.close(fig)

# ── bins: mean velocity on the output cells ──────────────────────────────────────────────────────────────────────────
nb, fct, dx = int(meta["bins"]), int(meta["out_factor"]), float(meta["dx"])
c1 = 8 // fct
nt, cells = int(meta["tiles"]), int(meta["cells_per_tile"])
counts = np.asarray(meta["counts"], float)
ACC = int(meta.get("acc_stride", 10))
acc = np.memmap(RUN / "flow_bins.f32", np.float32, "r", shape=(nb, nt, cells, ACC))
txyz = np.asarray(meta["tile_xyz"], np.int64).reshape(-1, 3)
cc = np.arange(cells)
czz, cyy, cxx = cc // (c1 * c1), (cc // c1) % c1, cc % c1
gx = (txyz[:, 0:1] * c1 + cxx[None]).ravel()
gy = (txyz[:, 1:2] * c1 + cyy[None]).ravel()
gz = (txyz[:, 2:3] * c1 + czz[None]).ravel()
dims = (int(gz.max()) + 1, int(gy.max()) + 1, int(gx.max()) + 1)
flat = (gz * dims[1] + gy) * dims[2] + gx
org = np.asarray(meta["origin"], float)
hc = fct * dx
o_cell = org + (0.5 * fct - 0.5) * dx  # centre of cell 0


def field(b):
    a = np.asarray(acc[b]).reshape(-1, ACC)
    w = a[:, 0]
    U = np.zeros((3, dims[0] * dims[1] * dims[2]), np.float32)
    C = np.zeros(dims[0] * dims[1] * dims[2], np.float32)
    ok = w > 0
    U[:, flat[ok]] = (a[ok, 1:4] / w[ok, None]).T
    C[flat[ok]] = w[ok] / max(counts[b], 1) / fct**3  # fluid fraction of the cell over the bin's samples
    return U.reshape(3, *dims), C.reshape(dims)


def sample(U, C, P):
    """trilinear sample of the cell field at points P [n, 3] (x, y, z); coverage-weighted"""
    g = (P - o_cell) / hc
    i0 = np.floor(g).astype(int)
    f = g - i0
    out = np.zeros((len(P), 3))
    cov = np.zeros(len(P))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx_ in (0, 1):
                ix, iy, iz = i0[:, 0] + dx_, i0[:, 1] + dy, i0[:, 2] + dz
                ok = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < dims[2]) & (iy < dims[1]) & (iz < dims[0])
                w = (np.where(dx_, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) * np.where(dz, f[:, 2], 1 - f[:, 2])) * ok
                cw = np.zeros(len(P))
                cw[ok] = C[iz[ok], iy[ok], ix[ok]]
                for a in range(3):
                    v = np.zeros(len(P))
                    v[ok] = U[a, iz[ok], iy[ok], ix[ok]]
                    out[:, a] += w * cw * v
                cov += w * cw
    return out / np.maximum(cov, 1e-6)[:, None], cov


VQ = {"aortic": "Q_AO", "pulmonary": "Q_PA", "mitral": "Q_MV", "tricuspid": "Q_TV"}
tb = (np.arange(nb) + 0.5) / nb * T
vrec = {v["name"]: {"q": [], "q_mat": [], "q_model": [], "jet_max": [], "jet_p95": [], "open": [], "jet_at": [], "un_mean": [], "un_max": [], "fluid_cm2": [], "ecc": []} for v in case["valves"]}
hot = []
filled = counts > 0
print(f"{int(filled.sum())}/{nb} bins with samples")
# cut cross-check: the binned field's flux through each cut's material disk 2.5 mm inside the domain (outward), against
# the log's flux-layer value: a biased flux layer would show in every prescribed cut and in the valve balance
_cutq = {c["name"]: [] for c in case["cuts"]}
CUT_DEPTHS = [float(x) * 1e-3 for x in opt("--cut-depths-mm", "2.5").split(",")]


def _perp(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e = np.cross(n, a)
    return e / np.linalg.norm(e)


for b in range(nb):
    if not filled[b]:
        continue
    U, C = field(b)
    for c_ in case["cuts"]:
        n_ = np.asarray(c_["normal"], float)
        qd_ = []
        for dep_ in CUT_DEPTHS:
            Xm_, dAm_, um_ = MO.material_disk(np.asarray(c_["center"], float) - dep_ * n_, n_, _perp(n_), float(c_["radius"]), tb[b], nr=8, nth=24)
            u2, cov2 = sample(U, C, Xm_)
            qd_.append(float(((u2 - um_) * dAm_).sum(1) @ np.minimum(cov2 / 0.6, 1.0)) * 1e6)
        _cutq[c_["name"]].append(qd_)
    sp = np.sqrt((U**2).sum(0))
    well = C >= 0.6
    s_ = np.where(well, sp, 0)
    i = np.unravel_index(np.argmax(s_), dims)
    x = o_cell + np.array([i[2], i[1], i[0]]) * hc
    near = min(((v["name"], float(np.linalg.norm(x - MO.fwd(v["center"], tb[b])))) for v in case["valves"]), key=lambda z: z[1])
    nearc = min(((c["name"], float(np.linalg.norm(x - MO.fwd(c["center"], tb[b])))) for c in case["cuts"]), key=lambda z: z[1])
    hot.append({"bin": b, "t_ms": tb[b] * 1e3, "max_m_s": float(s_.max()), "p999_m_s": float(np.quantile(sp[well], 0.999)), "x_mm": (x * 1e3).round(1).tolist(),
                "valve": near[0], "valve_mm": near[1] * 1e3, "cut": nearc[0][:30], "cut_mm": nearc[1] * 1e3})
    for v in case["valves"]:
        c, n, b1, b2, vel = MO.frame(np.asarray(v["center"]), v["normal"], v["e1"], tb[b])
        R = float(v["radius"])
        rr, th = np.meshgrid(np.linspace(0, R, 24), np.linspace(0, 2 * np.pi, 72, endpoint=False), indexing="ij")
        ring = (rr + R / 46).ravel()  # ring centres of the area elements
        dA = (2 * np.pi * ring / 72) * (R / 24)
        P = c + ring[:, None] * (np.cos(th).ravel()[:, None] * b1 + np.sin(th).ravel()[:, None] * b2)
        ok = ring <= R
        u, cov = sample(U, C, P[ok])
        q = float(((u - vel) @ n * np.minimum(cov / 0.6, 1.0) * dA[ok]).sum())
        jet, jat = [], []
        for s_off in (0.0, 0.004, 0.008, 0.012):
            uu, cv = sample(U, C, P[ok] + s_off * n)
            jet.append(np.sqrt((uu**2).sum(1))[cv > 0.5])
            # where the plane's maximum is: (depth mm, radius / R, normal component share of |u|)
            if (cv > 0.5).any():
                k_ = int(np.argmax(np.where(cv > 0.5, np.sqrt((uu**2).sum(1)), -1)))
                jat.append((float(np.sqrt((uu[k_]**2).sum())), s_off * 1e3, float(ring[ok][k_] / R), float(((uu[k_] - vel) @ n) / max(np.sqrt((uu[k_]**2).sum()), 1e-9))))
        jet = np.concatenate(jet) if jet else np.zeros(1)
        jat = max(jat) if jat else (0.0, 0.0, 0.0, 0.0)
        # the through-flow on the disk: mean normal velocity over its fluid area, its maximum, the fluid area
        un_ = (u - vel) @ n
        wf_ = np.minimum(cov / 0.6, 1.0) * dA[ok]
        vrec[v["name"]]["jet_at"].append(jat[1:])
        vrec[v["name"]]["un_mean"].append(float((un_ * wf_).sum() / max(wf_.sum(), 1e-12)))
        vrec[v["name"]]["un_max"].append(float(np.max(np.abs(un_[cov > 0.5]))) if (cov > 0.5).any() else 0.0)
        vrec[v["name"]]["fluid_cm2"].append(float(wf_.sum() * 1e4))
        # eccentricity as graded in 4D-flow studies, on the valve plane and one lumen diameter downstream (≈ the
        # sinotubular junction for the aortic valve): normalised flow displacement (forward-flux centroid offset /
        # diameter), jet angle (flux-weighted velocity vs the axis, deg), the lumen share carrying half the forward flux
        Rl_ = float(v.get("lumen_radius", R))
        ecc_ = []
        gg_ = np.arange(-2 * Rl_, 2 * Rl_ + 1e-9, 0.5e-3)
        G1_, G2_ = np.meshgrid(gg_, gg_, indexing="ij")
        for s_off in (0.0, 2 * Rl_):
            Pg = c + s_off * n + G1_[..., None] * b1 + G2_[..., None] * b2
            uu, cv = sample(U, C, Pg.reshape(-1, 3))
            un2 = (np.clip((uu - vel) @ n, 0, None) * np.minimum(cv / 0.6, 1.0)).reshape(G1_.shape)
            fl_ = (cv > 0.5).reshape(G1_.shape)
            if un2.max() <= 0:
                ecc_.append((0.0, 0.0, 0.0))
                continue
            # the lumen: the fluid component holding the largest forward flux (the vessel, not a neighbour)
            labl, _ = ndimage.label(fl_)
            k_ = labl.flat[int(np.argmax(un2))]
            lum = labl == k_
            w2 = np.where(lum, un2, 0.0)
            ca_ = np.array([G1_[lum].mean(), G2_[lum].mean()])
            cf_ = np.array([(G1_ * w2).sum(), (G2_ * w2).sum()]) / w2.sum()
            deq = 2 * np.sqrt(lum.sum() * 0.25e-6 / np.pi)
            disp = float(np.linalg.norm(cf_ - ca_) / deq)
            uw = (uu - vel)[lum.ravel()] * w2[lum][:, None]
            um = uw.sum(0)
            ang = float(np.degrees(np.arccos(np.clip(um @ n / max(np.linalg.norm(um), 1e-12), -1, 1))))
            wl = np.sort(w2[lum])[::-1]
            cum = np.cumsum(wl)
            a50 = float((np.searchsorted(cum, 0.5 * cum[-1]) + 1) / lum.sum())
            ecc_.append((disp, ang, a50))
        vrec[v["name"]]["ecc"].append(ecc_)
        o = np.interp(tb[b], np.asarray(v["opening"]["t"]), np.asarray(v["opening"]["v"]), period=T)
        vrec[v["name"]]["q"].append(q * 1e6)
        # through the material valve surface (the reference disk carried by the tissue, the cardiac model's Q
        # definition: the flux into the chamber as a material region), relative to the tissue velocity; the rigid disk
        # above moves with the centre only, and a deforming closed sheet sweeps volume through it
        Xm_, dAm_, um_ = MO.material_disk(np.asarray(v["center"], float), v["normal"], v["e1"], R, tb[b])
        u2, cov2 = sample(U, C, Xm_)
        vrec[v["name"]]["q_mat"].append(float(((u2 - um_) * dAm_).sum(1) @ np.minimum(cov2 / 0.6, 1.0)) * 1e6)
        vrec[v["name"]]["q_model"].append(float(np.interp(tb[b], tq, np.asarray(cm[VQ[v["name"]]], float), period=T)) * 1e6)
        vrec[v["name"]]["jet_max"].append(float(jet.max()) if jet.size else 0.0)
        vrec[v["name"]]["jet_p95"].append(float(np.quantile(jet, 0.95)) if jet.size else 0.0)
        vrec[v["name"]]["open"].append(float(o))
tbf = tb[filled]
for nm, r in vrec.items():
    q, qm_, op = np.asarray(r["q"]), np.asarray(r["q_model"]), np.asarray(r["open"])
    qmat = np.asarray(r["q_mat"])
    closed = op < 0.05
    rep["valves"][nm] = {"stroke_ml": float(q.sum() * T / nb) if len(q) else None, "stroke_model_ml": float(qm_.sum() * T / nb) if len(q) else None,
                         "peak_q_ml_s": float(q.max()) if len(q) else None, "peak_q_model_ml_s": float(qm_.max()) if len(q) else None,
                         "corr": float(np.corrcoef(q, qm_)[0, 1]) if len(q) > 3 else None,
                         # the jet while the valve is ≥ half open (a closed AV valve's cylinder holds the passing outflow
                         # in systole); jet_max_any_m_s: over every bin as before
                         "jet_max_m_s": float(max(np.where(op >= 0.5, r["jet_max"], 0))) if r["jet_max"] else None,
                         "jet_p95_peak_m_s": float(max(np.where(op >= 0.5, r["jet_p95"], 0))) if r["jet_p95"] else None,
                         "jet_max_any_m_s": float(max(r["jet_max"])) if r["jet_max"] else None,
                         # when / where the jet maximum is (time ms, depth mm past the disk, radius / R, normal share of
                         # |u|), and the through-flow at the time of peak model flow: mean / max normal velocity over the
                         # disk's fluid area (cm²)
                         "jet_where": ({"t_ms": float(tbf[int(np.argmax(np.where(op >= 0.5, r["jet_max"], 0)))] * 1e3), "depth_mm": r["jet_at"][int(np.argmax(np.where(op >= 0.5, r["jet_max"], 0)))][0],
                                        "radius_share": r["jet_at"][int(np.argmax(np.where(op >= 0.5, r["jet_max"], 0)))][1], "normal_share": r["jet_at"][int(np.argmax(np.where(op >= 0.5, r["jet_max"], 0)))][2]} if r["jet_max"] else None),
                         "at_peak_flow": ({"t_ms": float(tbf[int(np.argmax(qm_))] * 1e3), "un_mean_m_s": r["un_mean"][int(np.argmax(qm_))], "un_max_m_s": r["un_max"][int(np.argmax(qm_))],
                                           "fluid_cm2": r["fluid_cm2"][int(np.argmax(qm_))], "jet_max_m_s": r["jet_max"][int(np.argmax(qm_))],
                                           "eccentricity": {f"{lab}": {"displacement": e_[0], "jet_angle_deg": e_[1], "half_flux_area_share": e_[2]}
                                                            for lab, e_ in zip(("plane", "1 diameter"), r["ecc"][int(np.argmax(qm_))])}} if len(qm_) else None),
                         "closed_bins": int(closed.sum()), "closed_mean_q_ml_s": float(q[closed].mean()) if closed.any() else None,
                         "closed_rms_q_ml_s": float(np.sqrt((q[closed] ** 2).mean())) if closed.any() else None,
                         "stroke_mat_ml": float(qmat.sum() * T / nb) if len(qmat) else None,
                         "corr_mat": float(np.corrcoef(qmat, qm_)[0, 1]) if len(qmat) > 3 else None,
                         "closed_mean_q_mat_ml_s": float(qmat[closed].mean()) if closed.any() else None,
                         "closed_rms_q_mat_ml_s": float(np.sqrt((qmat[closed] ** 2).mean())) if closed.any() else None,
                         # against the model while closed (a regurgitant valve's backflow; ≈ the above for a healthy one)
                         "closed_rms_err_mat_ml_s": float(np.sqrt(((qmat - qm_)[closed] ** 2).mean())) if closed.any() else None,
                         "closed_model_peak_ml_s": float(np.abs(qm_[closed]).max()) if closed.any() else None,
                         "regurgitant_ml": float(-np.clip(qmat, None, 0)[closed].sum() * T / nb) if closed.any() else None,
                         "regurgitant_model_ml": float(-np.clip(qm_, None, 0)[closed].sum() * T / nb) if closed.any() else None}
rep["hot_cells"] = hot
# valve gate (2026-11-26): on the material disk (the cardiac model's definition) a closed valve passes ≤ 3 mL/s rms and
# the stroke is within 5 % of the model's. The rigid disk (moving with the centre only) is kept for comparison: a
# deforming closed sheet sweeps volume through it (the 11 mL/s "mitral leak" of the full12 phantom; material 0.2)
_gate = []
# the net stroke exactly, from the link-flux cut balances over the cycle (each tree's volume is periodic): aortic ←
# systemic arterial outflows, pulmonary ← pulmonary arterial, mitral ← pulmonary venous inflows, tricuspid ← systemic
# venous (cut classes from case.json; older cases without them: material disk only)
_cls = {"aortic": "systemic_arterial", "pulmonary": "pulmonary_arterial", "mitral": "pulmonary_venous", "tricuspid": "systemic_venous"}
# the exact link flux through each material valve plane, logged by newer runs (hasty-cfd node_diag): its late-cycle
# mean × the period is the net stroke through the valve itself
for vi_, v_ in enumerate(case["valves"]):
    ql_ = [d_["valves"][vi_].get("q_link_ml_s") for d_, l_ in zip(log, late) if l_ and vi_ < len(d_.get("valves", []))]
    if ql_ and all(x is not None for x in ql_):
        rep["valves"][v_["name"]]["stroke_link_ml"] = float(np.mean(ql_) * T)
_regurg = {v["name"] for v in case["valves"] if v.get("regurg_hole")}
for nm, r_ in rep["valves"].items():
    if all("class" in c for c in case["cuts"]):
        tot = sum(rep["cuts"][c["name"].replace("connector: ", "")]["mean_ml_s"] or 0.0 for c in case["cuts"] if c["class"] == _cls[nm])
        r_["stroke_cuts_ml"] = float(abs(tot) * T)
for nm, r_ in rep["valves"].items():
    if r_["stroke_mat_ml"] is None:
        continue
    # closed: a competent valve passes ≤ 3 mL/s rms (the model's closing backflow lobe, ≈ 1 mL, falls in the scheduled
    # closed phase and is not asked for); a regurgitant one follows the model's backflow within max(3 mL/s, 15 % of
    # its peak); the net stroke within 5 % — from the cut balance where available (exact), else the material disk
    if nm in _regurg:
        tol_c = max(3.0, 0.15 * (r_["closed_model_peak_ml_s"] or 0.0))
        ok_c = r_["closed_rms_err_mat_ml_s"] is None or r_["closed_rms_err_mat_ml_s"] <= tol_c
    else:
        ok_c = r_["closed_rms_q_mat_ml_s"] is None or r_["closed_rms_q_mat_ml_s"] <= 3.0
    _st = r_.get("stroke_link_ml", r_.get("stroke_cuts_ml", r_["stroke_mat_ml"]))
    ok_s = abs(_st / r_["stroke_model_ml"] - 1) <= 0.05
    r_["gate"] = bool(ok_c and ok_s)
    _rg = f", regurgitant {r_['regurgitant_ml']:.1f} / {r_['regurgitant_model_ml']:.1f} mL" if (r_["regurgitant_model_ml"] or 0) > 1.0 else ""
    _cr = r_["closed_rms_err_mat_ml_s"] if nm in _regurg else r_["closed_rms_q_mat_ml_s"]
    _gate.append(f"{nm} {'ok' if r_['gate'] else 'FAIL'} (closed rms {'-' if _cr is None else round(_cr, 2)} mL/s{' vs model' if nm in _regurg else ''}, "
                 f"stroke {_st:.1f} / {r_['stroke_model_ml']:.1f} mL ({'valve link' if 'stroke_link_ml' in r_ else ('cuts' if 'stroke_cuts_ml' in r_ else 'material disk')}; material disk {r_['stroke_mat_ml']:.1f}"
                 f"{', cuts ' + format(r_['stroke_cuts_ml'], '.1f') if 'stroke_cuts_ml' in r_ else ''}){_rg})")
rep["valve_gate"] = all(r_.get("gate", True) for r_ in rep["valves"].values())
# tree mass balance (exact link fluxes, late cycles): in through the valve plane, out through the tree's cuts, the
# tree region's mass change and the prescribed wall-transpiration mean; the rest is mass the lattice loses or makes
# inside the tree (e.g. a transpiration realised with a phase-dependent efficiency)
if all("class" in c for c in case["cuts"]) and log and "q_link_ml_s" in log[-1]["valves"][0]:
    _lt = [d_ for d_, l_ in zip(log, late) if l_]
    for vn_, cl_, rg_ in (("aortic", "systemic_arterial", "connector: Ascending aorta"), ("pulmonary", "pulmonary_arterial", "connector: Bifurcation of pulmonary trunk")):
        vi_ = next((i for i, v_ in enumerate(case["valves"]) if v_["name"] == vn_), None)
        if vi_ is None:
            continue
        q_in = np.mean([d_["valves"][vi_]["q_link_ml_s"] for d_ in _lt])
        ci_ = [i for i, c in enumerate(case["cuts"]) if c["class"] == cl_]
        q_out = np.mean([sum(d_["cuts"][i]["q_link_ml_s"] for i in ci_) for d_ in _lt])
        tr_ = np.mean([d_["transpiration_ml_s"].get(rg_, 0.0) for d_ in _lt]) if isinstance(_lt[0].get("transpiration_ml_s"), dict) else 0.0
        mm_ = [d_["regions"][rg_]["volume_ml"] * (1 + d_["regions"][rg_]["drho"]) for d_ in _lt if rg_ in d_.get("regions", {})]
        dm_ = (mm_[-1] - mm_[0]) / max(_lt[-1]["t"] - _lt[0]["t"], 1e-9) if len(mm_) > 1 else 0.0
        res_ = q_in - q_out - tr_ - dm_
        rep.setdefault("tree_balance", {})[vn_] = {"valve_in_ml_s": float(q_in), "cuts_out_ml_s": float(q_out), "transpiration_mean_ml_s": float(tr_), "mass_change_ml_s": float(dm_), "residual_ml_s": float(res_)}
        print(f"tree balance {vn_}: valve {q_in:.1f} in, cuts {q_out:.1f} out, transpiration mean {tr_:.1f}, mass change {dm_:.1f} → unaccounted {res_:.1f} mL/s ({res_ / max(q_in, 1e-9) * 100:.1f} %)")
print("valve gate: " + "; ".join(_gate))
# effective orifice area as echo grades it (continuity: peak flow / peak jet velocity at the time of peak flow) against
# the geometric open area: healthy valves ≈ 0.9–1 (a flat jet through the whole orifice); a stenosis's EOA is its spec
_eoa = []
for v in case["valves"]:
    rv_ = rep["valves"][v["name"]]
    apf = rv_.get("at_peak_flow") or {}
    if apf.get("jet_max_m_s") and rv_.get("peak_q_model_ml_s"):
        eoa = rv_["peak_q_model_ml_s"] * 1e-6 / apf["jet_max_m_s"] * 1e4
        rv_["eoa_cm2"] = float(eoa)
        if v.get("open_area_cm2"):
            rv_["goa_cm2"] = float(v["open_area_cm2"])
            rv_["eoa_over_goa"] = float(eoa / v["open_area_cm2"])
        if v.get("target_eoa_cm2"):
            rv_["target_eoa_cm2"] = float(v["target_eoa_cm2"])
        _eoa.append(f"{v['name']} EOA {eoa:.2f} cm²" + (f" / GOA {v['open_area_cm2']:.2f} ({eoa / v['open_area_cm2']:.2f})" if v.get("open_area_cm2") else "")
                    + (f", target {v['target_eoa_cm2']:.2f}" if v.get("target_eoa_cm2") else ""))
if _eoa:
    print("effective orifice: " + "; ".join(_eoa))
# lattice health: a chamber whose prescribed motion has no way out (a valve that does not open enough) compresses the
# weakly compressible lattice (2026-11-26: an aortic stenosis whose leaflets opened 0.24 cm² doubled the LV density,
# 23 600 mmHg, 33 m/s); per region the late-cycle density range must stay within ±5 %
_rb = []
for nm_ in (log[-1].get("regions") or {}):
    dr_ = np.array([d_["regions"][nm_]["drho"] if nm_ in d_.get("regions", {}) else 0.0 for d_, l_ in zip(log, late) if l_])
    if dr_.size and (dr_.max() > 0.05 or dr_.min() < -0.05):
        _rb.append(f"{nm_} {dr_.min():+.3f}…{dr_.max():+.3f}")
rep["density_gate"] = not _rb
print("density gate (|ρ − 1| ≤ 0.05 per region, late cycles): " + ("ok" if not _rb else "FAIL " + "; ".join(_rb)))
for c_ in case["cuts"]:
    nm_ = c_["name"].replace("connector: ", "")
    if _cutq[c_["name"]] and nm_ in rep["cuts"]:
        qq_ = np.mean(np.asarray(_cutq[c_["name"]]), 0)
        rep["cuts"][nm_]["binned_disk_mean_ml_s"] = float(qq_[0])
        if len(CUT_DEPTHS) > 1:
            rep["cuts"][nm_]["binned_disk_by_depth"] = {f"{d_ * 1e3:g}": float(x_) for d_, x_ in zip(CUT_DEPTHS, qq_)}
_rat = [r_["binned_disk_mean_ml_s"] / r_["mean_ml_s"] for r_ in rep["cuts"].values() if r_.get("binned_disk_mean_ml_s") is not None and abs(r_["mean_ml_s"] or 0) > 1.0]
if _rat:
    print(f"cut cross-check (binned field 2.5 mm inside / flux layer, |q| > 1 mL/s): median {np.median(_rat):.3f}, 10–90 % {np.quantile(_rat, 0.1):.3f}–{np.quantile(_rat, 0.9):.3f} over {len(_rat)} cuts")
(OUT / "report.json").write_text(json.dumps(rep, indent=1))
# --flow-slices: per valve at its bin of peak model flow, the speed (m/s, in-plane arrows) in the plane of the valve axis
# and the forward-flux displacement direction one diameter downstream, the perpendicular plane through the axis, and
# the cross-section one diameter downstream (report/flow_slices.png)
# --flow-slices-t <ms>: the same at that cardiac time for every valve (report/flow_slices_<ms>.png)
if "--flow-slices" in A or "--flow-slices-t" in A:
    vs_ = case["valves"]
    t_fix = float(opt("--flow-slices-t")) * 1e-3 if "--flow-slices-t" in A else None
    fig, axs = plt.subplots(len(vs_), 3, figsize=(15, 4.6 * len(vs_)), squeeze=False)
    fidx = np.flatnonzero(filled)
    for row, v in zip(axs, vs_):
        r = vrec[v["name"]]
        bpk = int(fidx[int(np.argmax(np.asarray(r["q_model"])))]) if t_fix is None else int(fidx[int(np.argmin(np.abs(tb[fidx] - t_fix)))])
        U, C = field(bpk)
        c, n, b1, b2, vel = MO.frame(np.asarray(v["center"]), v["normal"], v["e1"], tb[bpk])
        Rl_ = float(v.get("lumen_radius", v["radius"]))
        # the displacement direction one diameter downstream
        rr2, th2 = np.meshgrid(np.linspace(0, Rl_, 16), np.linspace(0, 2 * np.pi, 48, endpoint=False), indexing="ij")
        Pd = c + 2 * Rl_ * n + rr2.ravel()[:, None] * (np.cos(th2).ravel()[:, None] * b1 + np.sin(th2).ravel()[:, None] * b2)
        ud, cd = sample(U, C, Pd)
        wd = np.clip((ud - vel) @ n, 0, None) * (cd > 0.5) * (rr2.ravel() + 1e-4)
        dvec = ((Pd - c - 2 * Rl_ * n) * wd[:, None]).sum(0) / max(wd.sum(), 1e-12)
        t_ = dvec - (dvec @ n) * n
        t_ = t_ / np.linalg.norm(t_) if np.linalg.norm(t_) > 1e-6 else b1
        p_ = np.cross(n, t_)
        ss = np.arange(-35e-3, 45e-3, 0.5e-3)
        uu_ = np.arange(-35e-3, 35e-3, 0.5e-3)
        vmax_ = 0
        for k, (ax_, lab_) in enumerate(((t_, "displacement direction"), (p_, "perpendicular"))):
            G1, G2 = np.meshgrid(uu_, ss, indexing="xy")
            P = c + G2[..., None] * n + G1[..., None] * ax_
            uvel, cv = sample(U, C, P.reshape(-1, 3))
            sp_ = np.where(cv > 0.5, np.linalg.norm(uvel, axis=1), np.nan).reshape(G1.shape)
            vmax_ = max(vmax_, np.nanmax(sp_) if np.isfinite(sp_).any() else 0)
            im = row[k].imshow(sp_, origin="lower", extent=[uu_[0] * 1e3, uu_[-1] * 1e3, ss[0] * 1e3, ss[-1] * 1e3], cmap="inferno", vmin=0, vmax=2.5)
            ua = (uvel @ ax_).reshape(G1.shape)
            un = (uvel @ n).reshape(G1.shape)
            st_ = 6
            row[k].quiver(G1[::st_, ::st_] * 1e3, G2[::st_, ::st_] * 1e3, np.where(np.isfinite(sp_), ua, 0)[::st_, ::st_], np.where(np.isfinite(sp_), un, 0)[::st_, ::st_], color="c", scale=25, width=0.002)
            row[k].axhline(0, color="w", lw=0.6)
            for sg in (-1, 1):
                row[k].axvline(sg * Rl_ * 1e3, color="w", lw=0.4, ls="--")
            row[k].set_title(f"{v['name']} {tb[bpk] * 1e3:.0f} ms: axis × {lab_} (mm; downstream up)")
        G1, G2 = np.meshgrid(uu_, uu_, indexing="xy")
        P = c + 2 * Rl_ * n + G1[..., None] * t_ + G2[..., None] * p_
        uvel, cv = sample(U, C, P.reshape(-1, 3))
        unn = np.where(cv > 0.5, (uvel - vel) @ n, np.nan).reshape(G1.shape)
        im2 = row[2].imshow(unn, origin="lower", extent=[uu_[0] * 1e3, uu_[-1] * 1e3, uu_[0] * 1e3, uu_[-1] * 1e3], cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        row[2].add_patch(plt.Circle((0, 0), Rl_ * 1e3, fill=False, color="k", lw=0.6))
        row[2].set_title(f"through-plane velocity {2 * Rl_ * 1e3:.0f} mm downstream (m/s)")
        fig.colorbar(im, ax=row[1], fraction=0.04)
        fig.colorbar(im2, ax=row[2], fraction=0.04)
    fig.tight_layout()
    fn_ = "flow_slices.png" if t_fix is None else f"flow_slices_{t_fix * 1e3:.0f}.png"
    fig.savefig(OUT / fn_, dpi=80)
    plt.close(fig)
    print(f"flow slices → {OUT / fn_}")

if len(tbf):
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    for k, (nm, r) in enumerate(vrec.items()):
        a = ax[0, k]
        a.plot(tbf * 1e3, r["q"], color="#1f5fbf", lw=1.4, label="LBM (rigid disk, relative to annulus centre)")
        a.plot(tbf * 1e3, r["q_mat"], color="#5a9a2a", lw=1.2, label="LBM (material disk, relative to tissue)")
        a.plot(tbf * 1e3, r["q_model"], color="#d05a1a", ls="--", lw=1.2, label="cardiac model")
        a.fill_between(tbf * 1e3, 0, np.asarray(r["open"]) * max(np.abs(r["q_model"])), color="0.9", label="opening (scaled)")
        a.set_title(f"{nm} valve flux [mL/s]", fontsize=9, loc="left")
        a.set_xlabel("t in cycle [ms]", fontsize=8)
        b_ = ax[1, k]
        b_.plot(tbf * 1e3, r["jet_max"], color="#1f5fbf", lw=1.2, label="max")
        b_.plot(tbf * 1e3, r["jet_p95"], color="#5a9a2a", lw=1.2, label="95th pct")
        b_.axhline(1.5 if nm in ("aortic", "pulmonary") else 1.0, color="0.6", ls=":", lw=1)
        b_.set_title(f"{nm}: speed on disk and 0–12 mm downstream [m/s]", fontsize=9, loc="left")
        b_.set_xlabel("t in cycle [ms]", fontsize=8)
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    fig.savefig(OUT / "valves.png", dpi=110)
    plt.close(fig)

    # speed MIPs at four phases
    qa, qmv = np.asarray(cm["Q_AO"], float), np.asarray(cm["Q_MV"], float)
    half = len(tq) // 2
    t_sys = tq[np.argmax(qa)]
    dia = tq > tq[np.argmax(qa)] + 0.15
    tmv = tq[dia]
    e_i = np.argmax(np.where(tmv < tmv.min() + 0.25, qmv[dia], -1))
    t_e = tmv[e_i]
    t_a = tq[np.argmax(np.where(tq > t_e + 0.2, qmv, -1))]
    t_d = 0.5 * (t_e + t_a)
    phases = [("peak systole", t_sys), ("E wave", t_e), ("diastasis", t_d), ("A wave", t_a)]
    fig, ax = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
    vm = 1.5
    for k, (lab, t_) in enumerate(phases):
        b = int(t_ / T * nb) % nb
        U, C = field(b)
        sp = np.sqrt((U**2).sum(0)) * (C >= 0.3)
        for r_, axis in enumerate((0, 1, 2)):
            m = sp.max(axis)
            ext = None
            im = ax[r_, k].imshow(m, origin="lower", cmap="magma", vmin=0, vmax=vm, extent=ext)
            ax[r_, k].set_title(f"{lab} (bin {b}, {tb[b] * 1e3:.0f} ms) MIP along {'zyx'[axis]}, max {m.max():.2f} m/s", fontsize=8, loc="left")
            ax[r_, k].set_xticks([])
            ax[r_, k].set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.6, label="speed [m/s]")
    fig.savefig(OUT / "speed_mips.png", dpi=100)
    plt.close(fig)

print(json.dumps({k: rep[k] for k in ("cuts", "valves", "drho_mean_by_cycle", "drho_range_late", "vmax_late_m_s")}, indent=1))
print("hot cells per bin (max m/s, where):")
for hc_ in hot[:: max(1, len(hot) // 25)]:
    print(f"  bin {hc_['bin']:2d} {hc_['t_ms']:5.0f} ms: {hc_['max_m_s']:.2f} (p99.9 {hc_['p999_m_s']:.2f}) at {hc_['x_mm']} — {hc_['valve']} {hc_['valve_mm']:.0f} mm, {hc_['cut']} {hc_['cut_mm']:.0f} mm")
print(f"→ {OUT}")
