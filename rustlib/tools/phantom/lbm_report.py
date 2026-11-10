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
qm = np.array([[c["q_out_ml_s"] for c in d["cuts"]] for d in log])
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
    r = {"mean_ml_s": float(m.mean()) if len(m) else None}
    if ok.sum() > 3:
        r.update({"target_mean_ml_s": float(t_[ok].mean()), "ratio_of_means": float(m[ok].mean() / t_[ok].mean()) if abs(t_[ok].mean()) > 1e-6 else None,
                  "corr": float(np.corrcoef(m[ok], t_[ok])[0, 1]), "rms_err_ml_s": float(np.sqrt(np.mean((m[ok] - t_[ok]) ** 2))), "peak_ml_s": float(m[np.argmax(np.abs(m))]),
                  "target_peak_ml_s": float(t_[ok][np.argmax(np.abs(t_[ok]))])})
    rep["cuts"][names[k]] = r
rep["drho_mean_by_cycle"] = [float(drho[(cyc >= c) & (cyc < c + 1)].mean()) for c in range(int(np.ceil(cyc[-1]))) if ((cyc >= c) & (cyc < c + 1)).any()]
rep["drho_range_late"] = [float(drho[late].min()), float(drho[late].max())] if late.any() else None
rep["vmax_late_m_s"] = float(vmax[late].max()) if late.any() else None

fig, ax = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
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
vrec = {v["name"]: {"q": [], "q_model": [], "jet_max": [], "jet_p95": [], "open": []} for v in case["valves"]}
hot = []
filled = counts > 0
print(f"{int(filled.sum())}/{nb} bins with samples")
for b in range(nb):
    if not filled[b]:
        continue
    U, C = field(b)
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
        jet = []
        for s_off in (0.0, 0.004, 0.008, 0.012):
            uu, cv = sample(U, C, P[ok] + s_off * n)
            jet.append(np.sqrt((uu**2).sum(1))[cv > 0.5])
        jet = np.concatenate(jet) if jet else np.zeros(1)
        o = np.interp(tb[b], np.asarray(v["opening"]["t"]), np.asarray(v["opening"]["v"]), period=T)
        vrec[v["name"]]["q"].append(q * 1e6)
        vrec[v["name"]]["q_model"].append(float(np.interp(tb[b], tq, np.asarray(cm[VQ[v["name"]]], float), period=T)) * 1e6)
        vrec[v["name"]]["jet_max"].append(float(jet.max()) if jet.size else 0.0)
        vrec[v["name"]]["jet_p95"].append(float(np.quantile(jet, 0.95)) if jet.size else 0.0)
        vrec[v["name"]]["open"].append(float(o))
tbf = tb[filled]
for nm, r in vrec.items():
    q, qm_, op = np.asarray(r["q"]), np.asarray(r["q_model"]), np.asarray(r["open"])
    closed = op < 0.05
    rep["valves"][nm] = {"stroke_ml": float(q.sum() * T / nb) if len(q) else None, "stroke_model_ml": float(qm_.sum() * T / nb) if len(q) else None,
                         "peak_q_ml_s": float(q.max()) if len(q) else None, "peak_q_model_ml_s": float(qm_.max()) if len(q) else None,
                         "corr": float(np.corrcoef(q, qm_)[0, 1]) if len(q) > 3 else None,
                         "jet_max_m_s": float(max(r["jet_max"])) if r["jet_max"] else None, "jet_p95_peak_m_s": float(max(r["jet_p95"])) if r["jet_p95"] else None,
                         "closed_bins": int(closed.sum()), "closed_mean_q_ml_s": float(q[closed].mean()) if closed.any() else None,
                         "closed_rms_q_ml_s": float(np.sqrt((q[closed] ** 2).mean())) if closed.any() else None}
rep["hot_cells"] = hot
(OUT / "report.json").write_text(json.dumps(rep, indent=1))

if len(tbf):
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
    for k, (nm, r) in enumerate(vrec.items()):
        a = ax[0, k]
        a.plot(tbf * 1e3, r["q"], color="#1f5fbf", lw=1.4, label="LBM (disk, relative to annulus)")
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
