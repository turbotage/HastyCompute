"""Phase-binned LBM flux through material disks at stations along named vessels (cycle mean, outward along the
sample order), to locate where flow is lost between a valve and the cuts.
    python -I stations.py <phantom> <case> <run> "<edge name>[,<edge name>…]" [every_mm]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "python/lib")
from hastycompute.store import load_any
from hastycompute import value_format as vf
PH, CASE, RUN = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
want = sys.argv[4].split(",")
every = float(sys.argv[5]) * 1e-3 if len(sys.argv) > 5 else 10e-3
case = json.loads((CASE / "case.json").read_text())
meta = json.loads((RUN / "flow_bins.json").read_text())
cm = vf.read(case["motion"]["file"] if Path(case["motion"]["file"]).is_absolute() else str(CASE / case["motion"]["file"]))
U_ = cm["U_mm"]; nt_ = U_.shape[0]; mo = np.asarray(cm["origin_m"], float); mh = float(cm["voxel_m"]); T = float(cm["period"])
def tri(k, P):
    g = (P - mo) / mh - 0.5; i0 = np.floor(g).astype(int); f = g - i0; Z, Y, X = U_.shape[2:]; out = np.zeros((len(P), 3))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                ok = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < X) & (iy < Y) & (iz < Z)
                w = np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) * np.where(dz, f[:, 2], 1 - f[:, 2]) * ok
                out += w[:, None] * np.asarray(U_[k, :, np.clip(iz, 0, Z - 1), np.clip(iy, 0, Y - 1), np.clip(ix, 0, X - 1)], float)
    return out * 1e-3
def disp(P, t):
    s = (t % T) / T * nt_; k = int(np.floor(s)); f = s - k
    c = [-0.5 * f**3 + f**2 - 0.5 * f, 1.5 * f**3 - 2.5 * f**2 + 1, -1.5 * f**3 + 2 * f**2 + 0.5 * f, 0.5 * f**3 - 0.5 * f**2]
    return sum(ci * tri((k + j) % nt_, P) for ci, j in zip(c, (-1, 0, 1, 2)))
def mdisk(c, n, R, t, nr=12, nth=36, dt=1e-3):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0]); e1 = np.cross(n, a); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    rho = (np.arange(nr) + 0.5) / nr * R; th = np.arange(nth) / nth * 2 * np.pi; Rg, Tg = np.meshgrid(rho, th, indexing="ij")
    er = np.cos(Tg).ravel()[:, None] * e1 + np.sin(Tg).ravel()[:, None] * e2; et = -np.sin(Tg).ravel()[:, None] * e1 + np.cos(Tg).ravel()[:, None] * e2
    P = c + Rg.ravel()[:, None] * er; eps = 0.25 * R / nr
    Xr = (P + eps * er + disp(P + eps * er, t) - P + eps * er - disp(P - eps * er, t)) / (2 * eps)
    Xt = (P + eps * et + disp(P + eps * et, t) - P + eps * et - disp(P - eps * et, t)) / (2 * eps)
    dA = np.cross(Xr, Xt) * (Rg.ravel() * (R / nr) * (2 * np.pi / nth))[:, None]
    return P + disp(P, t), dA, (disp(P, t + dt) - disp(P, t - dt)) / (2 * dt)
nb, fct, dx = int(meta["bins"]), int(meta["out_factor"]), float(meta["dx"]); c1 = 8 // fct
ntl, cells = int(meta["tiles"]), int(meta["cells_per_tile"]); counts = np.asarray(meta["counts"], float); ACC = int(meta.get("acc_stride", 10))
acc = np.memmap(RUN / "flow_bins.f32", np.float32, "r", shape=(nb, ntl, cells, ACC))
txyz = np.asarray(meta["tile_xyz"], np.int64).reshape(-1, 3); cc = np.arange(cells)
czz, cyy, cxx = cc // (c1 * c1), (cc // c1) % c1, cc % c1
gx = (txyz[:, 0:1] * c1 + cxx[None]).ravel(); gy = (txyz[:, 1:2] * c1 + cyy[None]).ravel(); gz = (txyz[:, 2:3] * c1 + czz[None]).ravel()
dims = (int(gz.max()) + 1, int(gy.max()) + 1, int(gx.max()) + 1); flat = (gz * dims[1] + gy) * dims[2] + gx
org = np.asarray(meta["origin"], float); hc = fct * dx; o_cell = org + (0.5 * fct - 0.5) * dx
def field(b):
    a = np.asarray(acc[b]).reshape(-1, ACC); w = a[:, 0]; U = np.zeros((3, dims[0] * dims[1] * dims[2]), np.float32); C = np.zeros(dims[0] * dims[1] * dims[2], np.float32)
    ok = w > 0; U[:, flat[ok]] = (a[ok, 1:4] / w[ok, None]).T; C[flat[ok]] = w[ok] / max(counts[b], 1) / fct**3
    return U.reshape(3, *dims), C.reshape(dims)
def sample(U, C, P):
    g = (P - o_cell) / hc; i0 = np.floor(g).astype(int); f = g - i0; out = np.zeros((len(P), 3)); cov = np.zeros(len(P))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx_ in (0, 1):
                ix, iy, iz = i0[:, 0] + dx_, i0[:, 1] + dy, i0[:, 2] + dz
                ok = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < dims[2]) & (iy < dims[1]) & (iz < dims[0])
                w = (np.where(dx_, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1]) * np.where(dz, f[:, 2], 1 - f[:, 2])) * ok
                cw = np.zeros(len(P)); cw[ok] = C[iz[ok], iy[ok], ix[ok]]
                for a in range(3):
                    v = np.zeros(len(P)); v[ok] = U[a, iz[ok], iy[ok], ix[ok]]; out[:, a] += w * cw * v
                cov += w * cw
    return out / np.maximum(cov, 1e-6)[:, None], cov
G = load_any(PH / "graph.npz"); GJ = json.loads((PH / "graph.json").read_text())
eo = np.asarray(G["edge_offsets"]); Pg = np.asarray(G["points"], float); Rg_ = np.asarray(G.get("radius_ref", G["radius"]), float)
st = []
for k, nm in enumerate(GJ["edge_names"]):
    if nm in want:
        a, b = int(eo[k]), int(eo[k + 1]); L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pg[a:b], axis=0), axis=1))]
        for s0 in np.arange(every / 2, L[-1] - 2e-3, every):
            j = a + int(np.searchsorted(L, s0)); j = min(max(j, a + 1), b - 2)
            t_ = Pg[j + 1] - Pg[j - 1]; st.append((nm, k, j, Pg[j], t_ / np.linalg.norm(t_), Rg_[j]))
tb = (np.arange(nb) + 0.5) / nb * T
Q = np.zeros(len(st)); COV = np.zeros(len(st))
for b in range(nb):
    if counts[b] <= 0: continue
    U, C = field(b)
    for i, (nm, k, j, p, t_, R) in enumerate(st):
        X, dA, um = mdisk(p, t_, R, tb[b]); u, cov = sample(U, C, X)
        Q[i] += float(((u - um) * dA).sum(1) @ np.minimum(cov / 0.6, 1.0)) / nb; COV[i] += cov.mean() / nb
for i, (nm, k, j, p, t_, R) in enumerate(st):
    print(f"{nm[:28]:28s} edge {k:4d} sample {j:6d} R {R * 1e3:5.1f} mm  mean flux {Q[i] * 1e6:7.2f} mL/s  coverage {COV[i]:.2f}  at {np.round(p * 1e3, 0).tolist()}")
