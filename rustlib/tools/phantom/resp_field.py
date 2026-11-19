"""Off-resonance resolved in breathing AND cardiac time (simphantom `--resp-field`), and the truth ratemap per state.

    python -I resp_field.py <phantom_dir> <sim_input.hv> <out.hv> [--k 6] [--q 20] [--truth-dir D] [--advected-only]
                            [--rmax auto|r] [--iters 36] [--check 8] [--device cuda|cpu]

States: a grid (r_k, c_q), r_k = linspace(0, r_max, K) (r_max: the largest r of the input's breathing table, the whole
scan; r = 0 end-expiration, r = 1 nominal end-inspiration), c_q = q T / Q template cardiac time (periodic). Q = 1: no
cardiac deformation (breathing only, c = 0 = the reference pose).
Field of a state (prep_sim's model): the χ map deformed by the full motion x = X + u_c(X, c) + r S(X + u_c) (pulled back
by damped fixed-point iteration X ← X − ½ (φ(X) − x): the cardiac strain reaches |∇u| ≈ 1, where the plain iteration
oscillates; χ trilinear), the dipole field (pad 64, B0 along z), minus the SAME 2nd-order shim fitted
once on the reference pose (end-expiration; the scanner shims once and holds it).
Per source the change of its off-resonance relative to the static (prep_sim) value:
- tissue (Lagrangian, reference position X): Δ_kq(X) = F_kq(X + u_c(X, c_q) + r_k S(·)) − F_ref(X), table [Q, K, n] (f16);
  simphantom interpolates bilinearly in (r, c) at each bin's breathing state and cardiac time;
- blood (Eulerian frames at their own cardiac time c_f, r = 0 positions p): Δ_k(p) = F(r_k, c_f)(p + r_k S(p)) − F_ref(p),
  linear in c between the two cardiac states around c_f, table [n_f, K] per frame; simphantom interpolates in r.
Truth ratemap (--truth-dir, evaluation / oracle): at every state, lab frame, on the reconstruction grid, masked to the
voxels that hold signal in any state (mask.npy, recon grid [x, y, z] F order); files [S, nv] float32, state s = q K + k:
pd_water, pd_fat, ff, r_water (1/s), r_fat (1/s) (PD-weighted mean of 1/T2*), omega_hz (PD-weighted source
off-resonance, field part, no chemical shift), field_hz (the lab field at the voxel centre; not with --advected-only);
meta.json with the grid. Tissue sources at their state position, blood from the frame nearest c_q at p + r S(p).
--advected-only: no field recomputation (the input's static field advected with the sources: an existing dataset's
ratemap per state), only the truth maps. research/resp_field_sim.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
PH, SI, OUT = Path(A[0]), Path(A[1]), Path(A[2])


def opt(n, d):
    return A[A.index(n) + 1] if n in A else d


K, Q = int(opt("--k", 6)), int(opt("--q", 20))
iters = int(opt("--iters", 36))
n_check = int(opt("--check", 8))
advected = "--advected-only" in A
TD = Path(opt("--truth-dir", "")) if opt("--truth-dir", "") else None
dev = torch.device(opt("--device", "cuda"))
torch.set_num_threads(8)
t0 = time.time()
GHZ = 42.577478e6 * 3.0 * 1e-6  # Hz per ppm at 3 T

V = vf.read(SI)
presc = V["presc"]
center, fov, res = np.asarray(presc["center"], float), np.asarray(presc["fov"], float), float(presc["res"])
period = float(V["period"])
r_tab = np.asarray(V["breathing"]["r"], float)
rmax = float(r_tab.max()) if opt("--rmax", "auto") == "auto" else float(opt("--rmax", "auto"))
rk = np.linspace(0.0, rmax, K) if K > 1 else np.zeros(1)
cq = np.arange(Q) * period / Q
S_n = K * Q
print(f"states: r_k = {np.round(rk, 3).tolist()} (breathing r ∈ [{r_tab.min():.3f}, {r_tab.max():.3f}]), "
      f"{Q} cardiac phases over T = {period * 1e3:.1f} ms{' (no cardiac deformation)' if Q == 1 else ''}; {S_n} states")


def T(x, dt=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dt, device=dev)


def sampler(vol, o, hh, pad):
    """trilinear of vol [C, Z, Y, X] (tensor, centres o + (i + ½) hh) at P [N, 3] (x, y, z; m) → [N, C];
    pad 'border' = clamped (simphantom's S grid), 'zeros' = 0 outside (its cardiac grids)"""
    n = torch.tensor(vol.shape[1:][::-1], dtype=torch.float64, device=dev)
    o_ = torch.tensor(np.asarray(o, float), dtype=torch.float64, device=dev)
    v5 = vol[None]

    def f(P):
        g = (2 * ((P.double() - o_) / hh - 0.5) / (n - 1) - 1).float()
        return Fn.grid_sample(v5, g.view(1, 1, 1, -1, 3), mode="bilinear", padding_mode=pad, align_corners=True).view(vol.shape[0], -1).T

    def chunked(P, m=4_000_000):
        return torch.cat([f(P[i : i + m]) for i in range(0, len(P), m)]) if len(P) else torch.zeros(0, vol.shape[0], device=dev)

    return chunked


# ── motion: respiratory S (m at r = 1, clamped), cardiac u_c (Catmull-Rom of the samples, 0 outside its grid) ──────
R = vf.read(Path(V["resp_motion"]))
S_at = sampler(T(np.asarray(R["B"]).sum(0).astype(np.float32)), R["origin_m"], float(R["voxel_m"]), "border")
C = vf.read(Path(V["cardiac_motion"]))
U = np.asarray(C["U_mm"]).astype(np.float32) * 1e-3
nu = U.shape[0]


def ucard_grid(tc):
    s = (tc % period) / period * nu
    i = int(np.floor(s))
    f = s - i
    w = [(-f**3 + 2 * f**2 - f) / 2, (3 * f**3 - 5 * f**2 + 2) / 2, (-3 * f**3 + 4 * f**2 + f) / 2, (f**3 - f**2) / 2]
    return sum(w[j] * U[(i - 1 + j) % nu] for j in range(4))


def u_sampler(c):
    if Q == 1:
        return None
    return sampler(T(ucard_grid(c)), C["origin_m"], float(C["voxel_m"]), "zeros")


def move(P, r, u_at):
    """x = X + u_c(X) + r S(X + u_c) (P on the device, float64)"""
    X = P + u_at(P).double() if u_at is not None else P
    return X + r * S_at(X).double() if r != 0.0 else X


# ── sources ──────────────────────────────────────────────────────────────────────────────────────────────────────────
ts = V["tissue"]
tpos = T(np.asarray(ts["pos"]).reshape(-1, 3), torch.float64)
nt = len(tpos)
t_pd, t_t1, t_t2s = (np.asarray(ts[k], np.float64) for k in ("pd", "t1", "t2s"))
t_fat = np.asarray(ts["shift_ppm"]) != 0 if "shift_ppm" in ts else np.zeros(nt, bool)
t_fld = np.asarray(ts["field_ppm"], np.float64)
t_size = np.asarray(ts["size_m"], np.float64) if "size_m" in ts else np.zeros(nt)
frames = V["blood"]
bpos = [np.asarray(f["pos"], np.float32).reshape(-1, 3) for f in frames]
btc = np.array([float(f["tc"]) for f in frames])
nbf = len(frames)
alpha, TR = np.deg2rad(12.0), 0.01
E1 = np.exp(-TR / np.maximum(t_t1, 1e-4))
wt = t_pd * np.sin(alpha) * (1 - E1) / (1 - np.cos(alpha) * E1)  # signal weight (spoiled GRE, 12°, TR 10 ms) of the stats

tab_t = np.zeros((Q, K, nt), np.float16)
tab_b = [np.zeros((len(p), K), np.float32) for p in bpos]

# ── truth ratemap: the voxels with signal in any state (recon grid) ───────────────────────────────────────────────────────────────
if TD is not None:
    TD.mkdir(parents=True, exist_ok=True)
    dims = np.round(fov / res).astype(int)  # x, y, z
    nvox = int(np.prod(dims))
    nsub_t = np.maximum(np.round(t_size / res), 1).astype(int)
    b_pd = [np.asarray(f["pd"], np.float64) for f in frames]
    b_t2s = [np.asarray(f["t2s"], np.float64) for f in frames]
    b_fld = [np.asarray(f["field_ppm"], np.float64) for f in frames]
    nearest_f = [int(np.argmin(np.abs(((btc - c + period / 2) % period) - period / 2))) for c in cq]

    def vox_of(x):
        """nearest recon voxel (simphantom's truth maps), F-order linear index, −1 outside"""
        i = torch.round((x - T(center, torch.float64)) / res + T(dims // 2, torch.float64)).long()
        ok = ((i >= 0) & (i < T(dims, torch.int64))).all(1)
        return torch.where(ok, i[:, 0] + dims[0] * (i[:, 1] + dims[1] * i[:, 2]), torch.full_like(i[:, 0], -1))

    def state_sources(q, k):
        """(voxel index, pd, is_fat, 1/T2*, field ppm) of every tissue / blood (sub-)source in state (q, k)"""
        r, c = float(rk[k]), float(cq[q])
        x = move(tpos, r, u_sampler(c))
        fld = t_fld + (tab_t[q, k].astype(np.float64) if not advected else 0.0)
        out = []
        for ns in np.unique(nsub_t):
            m = nsub_t == ns
            mt = torch.as_tensor(m, device=dev)
            for o in [((np.array([a, b, cc]) + 0.5) / ns - 0.5) for a in range(ns) for b in range(ns) for cc in range(ns)] if ns > 1 else [np.zeros(3)]:
                off = T(o[None] * t_size[m][:, None], torch.float64)
                out.append((vox_of(x[mt] + off), t_pd[m] / ns**3, t_fat[m], 1 / np.maximum(t_t2s[m], 1e-4), fld[m]))
        f = nearest_f[q]
        if len(bpos[f]):
            p = T(bpos[f], torch.float64)
            out.append((vox_of(p + r * S_at(p).double()), b_pd[f], np.zeros(len(p), bool), 1 / np.maximum(b_t2s[f], 1e-4), b_fld[f] + (tab_b[f][:, k] if not advected else 0.0)))
        return out

    # pass 1: the voxels with signal in any state
    occ = torch.zeros(nvox, dtype=torch.bool, device=dev)
    for q in range(Q):
        for k in range(K):
            for vi, pd_, *_ in state_sources(q, k):
                vv = vi[(vi >= 0) & torch.as_tensor(pd_ > 0, device=dev)]
                occ[vv] = True
    mask = occ.cpu().numpy()
    sel = np.flatnonzero(mask)
    nv = len(sel)
    remap = torch.full((nvox,), -1, dtype=torch.long, device=dev)
    remap[torch.as_tensor(sel, device=dev)] = torch.arange(nv, device=dev)
    np.save(TD / "mask.npy", mask.reshape(tuple(dims), order="F"))
    names_q = ["pd_water", "pd_fat", "ff", "r_water", "r_fat", "omega_hz"] + ([] if advected else ["field_hz"])
    # --q-from Q0: resume the truth maps at cardiac phase Q0 (the arrays of an earlier, interrupted run are kept)
    q_from = int(opt("--q-from", 0))
    mm = {n: (np.lib.format.open_memmap(TD / f"{n}.npy", mode="r+") if q_from > 0 else np.lib.format.open_memmap(TD / f"{n}.npy", mode="w+", dtype=np.float32, shape=(S_n, nv))) for n in names_q}
    assert all(m.shape == (S_n, nv) for m in mm.values()), "--q-from: existing arrays have another shape"

# ── fields ───────────────────────────────────────────────────────────────────────────────────────────────────────────
if not advected:
    TP = json.loads((PH / "tissue_params.json").read_text())
    P0 = PH / "atlas" if (PH / "atlas" / "tissue_1mm.hv").exists() else REPO / "data/hasty/phantom/p0"
    TS = vf.read(P0 / "tissue_1mm.hv")
    names = list(TS["tissues"])
    ID = {n: i for i, n in enumerate(names)}
    tab = TP["table"]
    lab_full = np.asarray(TS["labels"])
    o_full = np.asarray(TS["origin_m"], float)
    h = float(TS["voxel_m"])
    chi_t = np.array([tab[n]["chi_ppm"] if "chi_ppm" in tab[n] else 0.0 for n in names], np.float32)
    chi0 = T(chi_t[lab_full] - float(tab["air"]["chi_ppm"]))
    NZ, NY, NX = chi0.shape
    pad = 64
    sh = [n + 2 * pad for n in chi0.shape]
    kz = torch.fft.fftfreq(sh[0], device=dev)[:, None, None]
    ky = torch.fft.fftfreq(sh[1], device=dev)[None, :, None]
    kx = torch.fft.rfftfreq(sh[2], device=dev)[None, None, :]
    k2 = kx**2 + ky**2 + kz**2
    D = 1 / 3 - kz**2 / torch.where(k2 > 0, k2, torch.ones_like(k2))
    D[0, 0, 0] = 0
    del k2, kx, ky, kz

    def dipole(chi):
        return torch.fft.irfftn(torch.fft.rfftn(chi, s=sh) * D, s=sh)[:NZ, :NY, :NX].contiguous()

    # the shim at the reference pose (prep_sim's 10-term fit over the tissue in the prescribed box), held for every state
    dip0 = dipole(chi0)
    zg, yg, xg = [torch.arange(n, dtype=torch.float32, device=dev) for n in (NZ, NY, NX)]
    LZ, LY, LX = torch.meshgrid(o_full[2] + (zg + 0.5) * h - center[2], o_full[1] + (yg + 0.5) * h - center[1], o_full[0] + (xg + 0.5) * h - center[0], indexing="ij")
    labt = T(lab_full, torch.int64)
    tissue = ~torch.isin(labt, T([ID["air"], ID["airway"], ID["lung"]], torch.int64))
    box = (LX.abs() < 0.4 * fov[0]) & (LY.abs() < 0.4 * fov[1]) & (LZ.abs() < 0.5 * fov[2] + 0.02) & tissue
    xs_, ys_, zs_ = LX[box], LY[box], LZ[box]
    Bm = torch.stack([torch.ones_like(xs_), xs_, ys_, zs_, xs_**2, ys_**2, zs_**2, xs_ * ys_, xs_ * zs_, ys_ * zs_], 1).double()
    coef = torch.linalg.solve(Bm.T @ Bm, Bm.T @ dip0[box].double())
    poly = sum(c * t for c, t in zip(coef.float(), [torch.ones_like(LX), LX, LY, LZ, LX**2, LY**2, LZ**2, LX * LY, LX * LZ, LY * LZ]))
    del LX, LY, LZ, box, Bm, xs_, ys_, zs_, tissue, labt
    Fref = dip0 - poly
    del dip0
    Fref_at = sampler(Fref[None], o_full, h, "border")
    chi_at = sampler(chi0[None], o_full, h, "border")
    t_ref = Fref_at(tpos)[:, 0]
    XC = torch.tensor(o_full[0] + (np.arange(NX) + 0.5) * h, device=dev)
    YC = torch.tensor(o_full[1] + (np.arange(NY) + 0.5) * h, device=dev)
    ZC = torch.tensor(o_full[2] + (np.arange(NZ) + 0.5) * h, device=dev)
    print(f"reference field and shim ({time.time() - t0:.0f}s)")

    def state_field(r, c):
        """the lab field of state (r, c), ppm [Z, Y, X]; the reference (r = 0, no cardiac deformation) unchanged"""
        u_at = u_sampler(c)
        if r == 0.0 and u_at is None:
            return Fref, 0.0
        chik = torch.empty_like(chi0)
        res_max, n_bad = 0.0, 0
        for z0 in range(0, NZ, 24):
            z1 = min(z0 + 24, NZ)
            Zs, Ys, Xs = torch.meshgrid(ZC[z0:z1], YC, XC, indexing="ij")
            x = torch.stack([Xs.ravel(), Ys.ravel(), Zs.ravel()], 1)
            X = x.clone()
            for _ in range(iters):
                X = X - 0.5 * (move(X, r, u_at) - x)
            rs = (move(X, r, u_at) - x).norm(dim=1)
            res_max = max(res_max, float(rs.max()))
            n_bad += int((rs > 1e-4).sum())
            chik[z0:z1] = chi_at(X)[:, 0].view(z1 - z0, NY, NX)
        if n_bad:
            print(f"  pull-back (r {r:.2f}, c {c * 1e3:.0f} ms): {n_bad} voxels with residual > 0.1 mm (max {res_max * 1e3:.2f} mm)")
        return dipole(chik) - poly, res_max

    def tissue_delta(F, r, c):
        return (sampler(F[None], o_full, h, "border")(move(tpos, r, u_sampler(c)))[:, 0] - t_ref).cpu().numpy()

    if TD is not None:
        Gi = torch.as_tensor(sel, device=dev)
        gx, gy, gz = Gi % dims[0], (Gi // dims[0]) % dims[1], Gi // (dims[0] * dims[1])
        Gp = torch.stack([center[0] + (gx - dims[0] // 2) * res, center[1] + (gy - dims[1] // 2) * res, center[2] + (gz - dims[2] // 2) * res], 1).double()
        gi = torch.clamp(torch.floor((Gp - T(o_full, torch.float64)) / h).long(), torch.zeros(3, dtype=torch.long, device=dev), T(np.array(lab_full.shape[::-1]) - 1, torch.int64))
        fr_sel = T(np.asarray(V["field_recon"], np.float32).ravel()[sel], torch.float64)  # C [z, y, x] ravel = F [x, y, z]
        fref_g = Fref[gi[:, 2], gi[:, 1], gi[:, 0]].double()
    # blood frames: the two cardiac states around each frame's c_f (weights linear in c, periodic)
    fq = btc / period * Q
    bq0 = np.floor(fq).astype(int) % Q
    bw1 = (fq - np.floor(fq)) if Q > 1 else np.zeros(nbf)
    b_ref = [None] * nbf
    resid = []
    for q in range(Q):
        for k, r in enumerate(rk):
            tk = time.time()
            F, rm = state_field(float(r), float(cq[q]))
            resid.append(rm)
            tab_t[q, k] = tissue_delta(F, float(r), float(cq[q]))
            if TD is not None:
                mm["field_hz"][q * K + k] = ((fr_sel + (F[gi[:, 2], gi[:, 1], gi[:, 0]].double() - fref_g)) * GHZ).float().cpu().numpy()
            Fa = sampler(F[None], o_full, h, "border")
            for f in range(nbf):
                w = (1 - bw1[f]) if bq0[f] == q else (bw1[f] if Q > 1 and (bq0[f] + 1) % Q == q else 0.0)
                if w == 0.0 or len(bpos[f]) == 0:
                    continue
                p = T(bpos[f], torch.float64)
                if b_ref[f] is None:
                    b_ref[f] = Fref_at(p)[:, 0]
                tab_b[f][:, k] += (w * (Fa(p + float(r) * S_at(p).double())[:, 0] - b_ref[f])).cpu().numpy()
            del F, Fa
            if k == K - 1 or q == 0:
                print(f"state q {q} k {k} (c {cq[q] * 1e3:.0f} ms, r {r:.2f}): pull-back residual {rm * 1e6:.1f} µm, tissue Δ p1/p50/p99 "
                      f"{(np.percentile(tab_t[q, k, ::11].astype(np.float32), [1, 50, 99]) * GHZ).round(1).tolist()} Hz ({time.time() - tk:.1f}s)", flush=True)
    print(f"field states done ({time.time() - t0:.0f}s); pull-back residual max {max(resid) * 1e6:.1f} µm, p90 {np.percentile(resid, 90) * 1e6:.1f} µm")

# ── checks and stats (tissue, signal-weighted) ───────────────────────────────────────────────────────────────────────
stats = {}
if not advected:
    def wpct(x, w, qs):
        o = np.argsort(x)
        cw = np.cumsum(w[o])
        return [float(x[o][min(np.searchsorted(cw, qq / 100 * cw[-1]), len(x) - 1)]) for qq in qs]

    def bilerp(r, c):
        f = np.clip(r / rk[-1] * (K - 1), 0, K - 1 - 1e-9) if K > 1 else 0.0
        j, a = int(f), f - int(f)
        j1 = min(j + 1, K - 1)
        g = (c % period) / period * Q
        q, b = int(np.floor(g)) % Q, g - np.floor(g)
        q1 = (q + 1) % Q
        tb = lambda qq, kk: tab_t[qq, kk].astype(np.float64)  # noqa: E731
        return (1 - a) * (1 - b) * tb(q, j) + a * (1 - b) * tb(q, j1) + (1 - a) * b * tb(q1, j) + a * b * tb(q1, j1)

    lab_t = lab_full[tuple(np.clip(np.floor((tpos.cpu().numpy() - o_full) / h).astype(int), 0, np.array(lab_full.shape[::-1]) - 1)[:, ::-1].T)]
    tp_np = tpos.cpu().numpy()
    hz_, hy_, hx_ = np.nonzero(lab_full == ID["myocardium"])
    hzr = (o_full[2] + hz_.min() * h, o_full[2] + (hz_.max() + 1) * h)
    hy_min, hxc = o_full[1] + hy_.min() * h, o_full[0] + (hx_.mean() + 0.5) * h
    liver_top = o_full[2] + (np.nonzero((lab_full == ID["liver"]).any((1, 2)))[0].max() + 1) * h
    WALL = [ID[n] for n in ("fat", "skin", "muscle", "bone_cortical", "bone_marrow", "cartilage")]
    regions = {
        "all tissue": np.ones(nt, bool),
        "heart (myocardium, valves, walls)": np.isin(lab_t, [ID[n] for n in ("myocardium", "valve", "vessel_wall")]),
        "liver dome (top 30 mm of the liver)": (lab_t == ID["liver"]) & (tp_np[:, 2] >= liver_top - 0.03),
        "chest wall (anterior to the heart, |x − x_heart| < 80 mm, heart z range)": np.isin(lab_t, WALL) & (tp_np[:, 1] < hy_min) & (np.abs(tp_np[:, 0] - hxc) < 0.08) & (tp_np[:, 2] >= hzr[0]) & (tp_np[:, 2] <= hzr[1]),
    }
    stats["regions"] = {}
    for name, m in regions.items():
        w = wt[m]
        e = {}
        # breathing: Δ(r = 1, c) − Δ(0, c), pooled over the cardiac phases
        d = np.concatenate([(bilerp(1.0, c) - bilerp(0.0, c))[m] for c in cq[:: max(Q // 5, 1)]]) * GHZ
        ww = np.tile(w, len(cq[:: max(Q // 5, 1)]))
        e["breath_end_insp_minus_end_exp_hz"] = dict(zip(["p5", "p50", "p95"], wpct(d, ww, [5, 50, 95])), abs_p95=wpct(np.abs(d), ww, [95])[0])
        if Q > 1:
            # cardiac: Δ(0, c_q) − its cycle mean, pooled over q
            mean0 = tab_t[:, 0].astype(np.float64).mean(0)
            dc = np.concatenate([(tab_t[q, 0].astype(np.float64) - mean0)[m] for q in range(Q)]) * GHZ
            wc = np.tile(w, Q)
            e["cardiac_minus_cycle_mean_at_end_exp_hz"] = dict(zip(["p5", "p50", "p95"], wpct(dc, wc, [5, 50, 95])), abs_p95=wpct(np.abs(dc), wc, [95])[0])
        stats["regions"][name] = e
        print(f"{name}: breathing Δ(r=1) − Δ(0): {', '.join(f'{k} {v:+.1f}' for k, v in e['breath_end_insp_minus_end_exp_hz'].items())} Hz"
              + (f"; cardiac Δ(c) − mean: {', '.join(f'{k} {v:+.1f}' for k, v in e['cardiac_minus_cycle_mean_at_end_exp_hz'].items())} Hz" if Q > 1 else ""))
    # interpolation error: halved grids from the nodes, and the full grid at off-grid check states (direct fields)
    heart = regions["heart (myocardium, valves, walls)"]
    errs = {}
    if Q >= 4 and Q % 2 == 0:
        e = np.concatenate([(tab_t[q, k].astype(np.float64) - 0.5 * (tab_t[q - 1, k].astype(np.float64) + tab_t[(q + 1) % Q, k].astype(np.float64))) for q in range(1, Q, 2) for k in range(K)]) * GHZ
        ww, hh = np.tile(wt, (Q // 2) * K), np.tile(heart, (Q // 2) * K)
        errs[f"cardiac Q/2 = {Q // 2}"] = {"abs_p95_hz": wpct(np.abs(e), ww, [95])[0], "abs_p99_hz": wpct(np.abs(e), ww, [99])[0], "heart_abs_p95_hz": wpct(np.abs(e[hh]), ww[hh], [95])[0]}
    if K >= 3:
        ks = range(1, K - 1)
        e = np.concatenate([(tab_t[q, k].astype(np.float64) - 0.5 * (tab_t[q, k - 1].astype(np.float64) + tab_t[q, k + 1].astype(np.float64))) for q in range(Q) for k in ks]) * GHZ
        ww, hh = np.tile(wt, Q * len(ks)), np.tile(heart, Q * len(ks))
        errs[f"breathing 2 Δr = {2 * rk[1]:.2f}"] = {"abs_p95_hz": wpct(np.abs(e), ww, [95])[0], "abs_p99_hz": wpct(np.abs(e), ww, [99])[0], "heart_abs_p95_hz": wpct(np.abs(e[hh]), ww[hh], [95])[0]}
    rng = np.random.default_rng(0)
    ec = []
    for i in range(n_check):
        r = float(rng.uniform(0, rk[-1])) if K > 1 else 0.0
        c = float(rng.uniform(0, period)) if Q > 1 else 0.0
        F, _ = state_field(r, c)
        ec.append((tissue_delta(F, r, c) - bilerp(r, c)) * GHZ)
        del F
    if ec:
        e = np.concatenate(ec)
        ww, hh = np.tile(wt, len(ec)), np.tile(heart, len(ec))
        errs[f"grid {K}×{Q} at {len(ec)} random states"] = {"abs_p95_hz": wpct(np.abs(e), ww, [95])[0], "abs_p99_hz": wpct(np.abs(e), ww, [99])[0], "heart_abs_p95_hz": wpct(np.abs(e[hh]), ww[hh], [95])[0]}
    stats["interpolation_error"] = errs
    for k_, v_ in errs.items():
        print(f"interpolation error, {k_}: |e| p95 {v_['abs_p95_hz']:.2f} Hz, p99 {v_['abs_p99_hz']:.2f} Hz, heart p95 {v_['heart_abs_p95_hz']:.2f} Hz (signal-weighted)")
    vf.write(OUT, {
        "kind": "resp_field", "mode": "rc" if Q > 1 else "r", "r": rk.tolist(), "c": cq.tolist(), "period": period, "units": "ppm",
        "sim_input": str(SI.resolve()),
        "shim": "2nd-order (10 terms) fitted on the reference pose (end-expiration) over the tissue in the prescribed box, held for every state",
        "definition": "tissue Δ_qk(X) = F_qk(X + u_c(X, c_q) + r_k S) − F_ref(X); blood Δ_k(p) = F(r_k, c_f)(p + r_k S(p)) − F_ref(p) (linear in c between states)",
        "n_tissue": nt, "blood_counts": [len(p) for p in bpos],
        "tissue_dppm": tab_t, "blood_dppm": tab_b, "stats": json.dumps(stats),
    })
    print(f"wrote {OUT} ({time.time() - t0:.0f}s)", flush=True)

# ── truth ratemap per state: the tissue maps advected to the state, the sources' ω ─────────────────────────────────
if TD is not None:
    for q in range(q_from, Q):
        for k in range(K):
            s = q * K + k
            acc = {n: torch.zeros(nv, dtype=torch.float64, device=dev) for n in ("pw", "pf", "rw", "rf", "om")}
            for vi, pd_, fat_, rr_, fl_ in state_sources(q, k):
                ok = (vi >= 0) & torch.as_tensor(pd_ > 0, device=dev)  # same filter as pass 1's mask (zero-pd sources off-mask → remap −1)
                j = remap[vi[ok]]
                okn = ok.cpu().numpy()
                pdv, fatv = T(pd_[okn], torch.float64), torch.as_tensor(fat_[okn], device=dev)
                acc["pw"].index_add_(0, j[~fatv], pdv[~fatv])
                acc["pf"].index_add_(0, j[fatv], pdv[fatv])
                acc["rw"].index_add_(0, j[~fatv], (pdv * T(rr_[okn], torch.float64))[~fatv])
                acc["rf"].index_add_(0, j[fatv], (pdv * T(rr_[okn], torch.float64))[fatv])
                acc["om"].index_add_(0, j, pdv * T(fl_[okn], torch.float64) * GHZ)
            tot = acc["pw"] + acc["pf"]
            nan = torch.tensor(float("nan"), dtype=torch.float64, device=dev)
            row = {"pd_water": acc["pw"], "pd_fat": acc["pf"], "ff": torch.where(tot > 0, acc["pf"] / tot, nan),
                   "r_water": torch.where(acc["pw"] > 0, acc["rw"] / acc["pw"], nan), "r_fat": torch.where(acc["pf"] > 0, acc["rf"] / acc["pf"], nan),
                   "omega_hz": torch.where(tot > 0, acc["om"] / tot, nan)}
            for n in names_q:
                if n != "field_hz":
                    mm[n][s] = row[n].float().cpu().numpy()
        print(f"truth maps: cardiac phase {q + 1}/{Q} ({time.time() - t0:.0f}s)", flush=True)
    for n in names_q:
        mm[n].flush()
    meta = {
        "kind": "truth_ratemap_bc", "sim_input": str(SI.resolve()), "field": "advected static field (no recomputation)" if advected else str(OUT.resolve()),
        "states": {"r": rk.tolist(), "c_s": cq.tolist(), "period_s": period, "K": K, "Q": Q, "index": "s = q K + k (cardiac phase q outer, breathing amplitude k inner)"},
        "grid": {"dims_xyz": dims.tolist(), "res_m": res, "center_m": center.tolist(), "voxel_index": "i ↦ centre + (i − n/2) res (simphantom's recon grid)",
                 "mask": "mask.npy: bool over the grid, F order [x, y, z] linear index (x fastest); nv = mask.sum() voxels, the rows' columns in that order"},
        "files": {n: f"{n}.npy [S, nv] float32" for n in names_q},
        "definitions": {
            "pd_water / pd_fat": "PD sums of the water / fat sources in the voxel (nearest voxel, box elements split), as truth_pd_*; blood (water) from the frame nearest c_q",
            "ff": "pd_fat / (pd_water + pd_fat)", "r_water / r_fat": "PD-weighted mean of 1/T2* (1/s) of the water / fat sources",
            "omega_hz": "PD-weighted mean of the sources' field off-resonance (Hz at 3 T, no chemical shift): the ω the signal sees",
            "field_hz": "the lab field at the voxel centre (nearest 1 mm voxel of the state's field, Hz)",
            "NaN": "no source of that kind in the voxel in that state",
        },
        "n_voxels_masked": int(nv), "stats": stats,
    }
    (TD / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"truth ratemap: {S_n} states × {nv} voxels → {TD} ({time.time() - t0:.0f}s)")
