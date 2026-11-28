"""Respiratory kinematics of the atlas phantom: u_r(X, t) = r(t) Σ_k D_k B_k(X), r ∈ [0, 1] (0 end-expiration).

    python -I resp_motion.py <tissue_2mm.hv> <out_dir> [--dsi 15] [--dap 7] [--dlr 3] [--graph <graph.npz>] [--tether on|off]
                             [--vessel-rod-mm 30 (0: off, as before 2026-11-28)]

Basis fields on a 4 mm grid covering the trunk (z 0.85–1.55 m; outside the tissue field of view the abdomen continues
the lowest labelled slice, the neck is still), unit 1 m:
- si: internal organs (lungs, heart, mediastinum, liver, stomach, spleen, bowel, … : the filled internal cavity,
      smoothed σ 8 mm) move caudally: 1 below the diaphragm dome, ((z_apex − z) / (z_apex − z_dome))^1.5 above it
      (heart ≈ 0.7 of the diaphragm, lung apex 0), fading to 0.4 low in the abdomen. The chest wall does not move SI —
      lungs slide along it.
- ap: anterior expansion of the trunk, linear from the spine (fixed) to the anterior wall.
- lr: lateral expansion of the lower ribs (bucket handle).
Amplitudes D (end-inspiration vs end-expiration): si 15 mm, ap 7 mm, lr 3 mm by default. The waveform r(t) is made by
`motion.py` (Lujan cos^{2n} breaths with breath-to-breath variability of period and depth).
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
TS, OUT = vf.read(A[0]), Path(A[1])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opt(n, d):
    return float(A[A.index(n) + 1]) if n in A else d


D = {"si": opt("--dsi", 15.0) * 1e-3, "ap": opt("--dap", 7.0) * 1e-3, "lr": opt("--dlr", 3.0) * 1e-3}
lab = np.asarray(TS["labels"])
names = list(TS["tissues"])
ID = {n: i for i, n in enumerate(names)}
h0 = float(TS["voxel_m"])
o0 = np.asarray(TS["origin_m"], float)
INTERNAL = ["lung", "airway", "myocardium", "valve", "blood_LA", "blood_LV", "blood_RA", "blood_RV", "liver", "gallbladder", "spleen", "thymus",
            "stomach_wall", "gi_content", "pancreas", "bowel_wall", "kidney"]
I = np.isin(lab, [ID[n] for n in INTERNAL if n in ID])
# vessel lumens embedded in internal organs move with them (2026-11-28; --embedded-vessels off: as before). The atlas
# labels hold no vessels (a lumen is the background tissue, fat), so the IVC in the liver's groove, open toward the
# crus, lay outside the internal cavity: with the σ 8 mm blur the intrahepatic IVC moved 0–6 mm against 8–13 mm of the
# liver beside it and kinked 16° below the atrium. A lumen sample (graph, r ≥ 2 mm, every 2 mm) whose shell (1–6 mm
# outside the wall), averaged ±30 mm along its edge, is ≥ 20 % internal organ adds its lumen to the cavity; tethered
# arteries stay held by the tether.
EMB = 0
if "--graph" in A and (A[A.index("--embedded-vessels") + 1] if "--embedded-vessels" in A else "on") == "on":
    _g0 = np.load(A[A.index("--graph") + 1])
    _P0, _R0, _off0 = np.asarray(_g0["points"], float), np.asarray(_g0["radius"], float), np.asarray(_g0["edge_offsets"])
    I_org = I.copy()

    def _box(p_, ro_):
        i0 = np.maximum(np.floor((p_ - ro_ - o0) / h0).astype(int), 0)
        i1 = np.minimum(np.ceil((p_ + ro_ - o0) / h0).astype(int) + 1, [lab.shape[2], lab.shape[1], lab.shape[0]])
        if np.any(i1 <= i0):
            return None
        zz, yy, xx = np.meshgrid(*(np.arange(i0[d], i1[d]) for d in (2, 1, 0)), indexing="ij")
        return zz, yy, xx, np.linalg.norm(np.stack([xx, yy, zz], -1) * h0 + o0 + h0 / 2 - p_, axis=-1)

    # organ fraction of the shell per sample (every 4th, 2 mm), then a running mean over ±20 mm along each edge: a
    # vessel is embedded over whole stretches, not voxel by voxel (a 10 mm gap of crus between liver and kidney level
    # left the IVC static there, stretch 0.63)
    for e_ in range(len(_off0) - 1):
        ii = np.arange(_off0[e_], _off0[e_ + 1], 4)
        if len(ii) == 0 or _R0[ii].max() < 2e-3:
            continue
        fr = np.full(len(ii), np.nan)
        bx = []
        for j_, i_ in enumerate(ii):
            b_ = _box(_P0[i_], _R0[i_] + 6e-3)
            bx.append(b_)
            if b_ is None:
                continue
            zz, yy, xx, dd = b_
            shell = (dd > _R0[i_] + 1e-3) & (dd <= _R0[i_] + 6e-3)
            if shell.any():
                fr[j_] = I_org[zz[shell], yy[shell], xx[shell]].mean()
        if np.all(np.isnan(fr)):
            continue
        fr = np.where(np.isnan(fr), 0.0, fr)
        frs = ndimage.uniform_filter1d(fr, size=31, mode="nearest")
        for j_, i_ in enumerate(ii):
            if frs[j_] >= 0.2 and _R0[i_] >= 2e-3 and bx[j_] is not None:
                zz, yy, xx, dd = bx[j_]
                lum = dd <= _R0[i_] + h0 / 2
                EMB += int((~I[zz[lum], yy[lum], xx[lum]]).sum())
                I[zz[lum], yy[lum], xx[lum]] = True
    print(f"embedded vessels: {EMB * h0**3 * 1e6:.0f} mL of vessel lumen joined the internal cavity", flush=True)
I = np.pad(I, 4, mode="edge")  # closing erodes at the array border: pad by edge replication, then crop
I = ndimage.binary_closing(I, iterations=3)[4:-4, 4:-4, 4:-4]
I = np.stack([ndimage.binary_fill_holes(s_) for s_ in I])  # mediastinal fat / vessels between the organs
z_dome = o0[2] + (np.nonzero((lab == ID["liver"]).any(axis=(1, 2)))[0].max() + 0.5) * h0
z_apex = o0[2] + (np.nonzero((lab == ID["lung"]).any(axis=(1, 2)))[0].max() + 0.5) * h0
yy = np.nonzero(np.isin(lab, [ID["skin"], ID["fat"], ID["muscle"]]).any(axis=(0, 2)))[0]
y_ant = o0[1] + (yy.min() + 0.5) * h0
spine = (lab == ID["bone_cortical"]) | (lab == ID["bone_marrow"])
xs = o0[0] + (np.arange(lab.shape[2]) + 0.5) * h0
sp_y = np.nonzero(spine[:, :, np.abs(xs) < 0.02].any(axis=(0, 2)))[0]
sp_yw = o0[1] + (sp_y + 0.5) * h0
y_post = float(np.median(sp_yw[sp_yw > 0.02]))  # vertebral bodies (posterior bone in the midline) ≈ the AP hinge

# 4 mm resp grid
h = 4e-3
lo = np.array([-0.25, -0.16, 0.85])
hi = np.array([0.25, 0.16, 1.55])
dims = np.round((hi - lo) / h).astype(int)
nx, ny, nz = dims
Z, Y, Xg = np.meshgrid(*(lo[a] + (np.arange(dims[a]) + 0.5) * h for a in (2, 1, 0)), indexing="ij")
# internal weight sampled from the label grid (nearest in x, y; z below the field of view: the lowest slice; above: 0)
iz = np.floor((Z - o0[2]) / h0).astype(int)
above = iz >= lab.shape[0]
iz = np.clip(iz, 0, lab.shape[0] - 1)
iy = np.clip(np.floor((Y - o0[1]) / h0).astype(int), 0, lab.shape[1] - 1)
ix = np.clip(np.floor((Xg - o0[0]) / h0).astype(int), 0, lab.shape[2] - 1)
inside_xy = (Xg >= o0[0]) & (Xg < o0[0] + lab.shape[2] * h0) & (Y >= o0[1]) & (Y < o0[1] + lab.shape[1] * h0)
Wint = np.where(above | ~inside_xy, 0.0, I[iz, iy, ix].astype(float))
Wint = ndimage.gaussian_filter(Wint, 8e-3 / h)

f_si = np.where(Z <= z_dome, 1.0, np.clip((z_apex - Z) / (z_apex - z_dome), 0, 1) ** 1.5)
f_si = np.where(Z < z_dome - 0.12, np.clip(1 - 0.6 * (z_dome - 0.12 - Z) / 0.18, 0.4, 1), f_si)
trunk = np.clip((0.22 - np.abs(Xg)) / 0.05, 0, 1)
B_si = np.zeros(Z.shape + (3,))
B_si[..., 2] = -Wint * f_si * trunk
g_ap = np.clip((y_post - Y) / (y_post - y_ant), 0, 1)
f_ap = np.clip(np.minimum((Z - 0.90) / 0.10, 1), 0.5, 1) * np.where(Z > 1.42, np.clip(1 - 0.7 * (Z - 1.42) / 0.08, 0.3, 1), 1)
B_ap = np.zeros(Z.shape + (3,))
B_ap[..., 1] = -g_ap * f_ap * trunk
f_lr = np.exp(-0.5 * ((Z - 1.22) / 0.07) ** 2)
B_lr = np.zeros(Z.shape + (3,))
B_lr[..., 0] = np.clip(Xg / 0.15, -1.3, 1.3) * f_lr * trunk
B = np.stack([B_si * D["si"], B_ap * D["ap"], B_lr * D["lr"]])  # amplitudes folded in: u_r = r(t) Σ_k B_k
for k in range(3):
    for c in range(3):
        B[k, ..., c] = ndimage.gaussian_filter(B[k, ..., c], 1.0)
# tethering (--tether on, default 2026-11-28; off: as before): the vertebral column and the great vessels lying on it (radius
# ≥ 3 mm, wall within 8 mm of bone; centrelines from --graph <graph.npz>, else bone only) stay put; the organs slide
# past them in a harmonic shear layer ≤ --tether-band 40 mm (motion_inclusions.tether; 12 mm, 3 cells of this grid, folded:
# det J −0.35). The retroperitoneal aorta moved
# 5 mm caudally with ±15–20 % stretch from the organs' smoothed mask
TETHER = (A[A.index("--tether") + 1] if "--tether" in A else "on")
assert TETHER in ("on", "off"), TETHER
tinfo = None
if TETHER == "on":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from motion_inclusions import tether, tether_mask  # noqa: E402
    from motion_inclusions import graph_path_distance, heart_roots  # noqa: E402
    gp = A[A.index("--graph") + 1] if "--graph" in A else None
    gz = np.load(gp) if gp else None
    _gj = json.loads((Path(gp).parent / "graph.json").read_text()) if gp else None
    # path seeds: connectors + heart roots (`--tether-seeds connectors`: as before 2026-11-28)
    pd_ = graph_path_distance(gz, _gj["edge_names"], heart_roots(_gj) if (A[A.index("--tether-seeds") + 1] if "--tether-seeds" in A else "roots") == "roots" else ()) if gp else None
    # the vertebral column only: the rib cage and the sternum move with the breathing (the ap and lr modes; all bone
    # tethered folded the posterior chest wall, det J 0.08) — midline bone (|x| < 30 mm), per slice its most posterior
    # 60 mm (the sternum, ≈ 150 mm anterior, out)
    vert = spine & (np.abs(xs) < 0.03)[None, None, :]
    yv = np.arange(lab.shape[1])
    for zi in range(lab.shape[0]):
        ys = yv[vert[zi].any(axis=1)]
        if len(ys):
            vert[zi, : max(0, ys.max() - int(round(0.06 / h0)))] = False
    # below the label field of view the trunk continues its lowest slice (as the organs' weight above): the bone too
    kz = max(0, int(np.ceil((o0[2] - lo[2]) / h0)))
    spine_x = np.concatenate([np.repeat(vert[:1], kz, axis=0), vert]) if kz else vert
    o_x = o0 - np.array([0.0, 0.0, kz * h0])
    # vessels tethered in breathing: the arteries only (2026-11-28; `--tether-veins on`: veins too, as before). The aorta
    # lies retrocrural on the vertebral bodies; the IVC lies in the liver's bare area and moves with the liver (its
    # infrarenal part is damped by the spine's shear layer). With the veins held, the intrahepatic IVC moved 3–5 mm
    # against ~10 mm of liver, compressed to 0.58 and kinked 16° below the right atrium.
    _tp, _tr, _tpd = (gz["points"], gz["radius"], pd_) if gz is not None else (None, None, None)
    if gz is not None and (A[A.index("--tether-veins") + 1] if "--tether-veins" in A else "off") == "off":
        _cls = {c["id"]: c["class"] for c in _gj["components"]}
        _art = np.repeat([_cls.get(int(c_), "").endswith("arterial") for c_ in gz["edge_comp"]], np.diff(gz["edge_offsets"]))
        _tp, _tr, _tpd = np.asarray(gz["points"])[_art], np.asarray(gz["radius"])[_art], np.asarray(pd_)[_art]
    # below the label field of view only the great retroperitoneal arteries (edge median radius ≥ --tether-below-rmin-mm
    # 4: aorta, iliacs ≥ 4.4 mm; the SMA ≤ 3.5 mm runs in the mesentery) are held without bone (2026-11-28; 3: as
    # before — the SMA was held there too). Per edge, not per point: a per-point radius test switched the hold on and
    # off along the left external iliac (4.4–4.7 mm, 151 of 211 points free at 5 mm)
    _tre = None
    if gz is not None:
        _off_t = np.asarray(gz["edge_offsets"])
        _tre = np.repeat([float(np.median(gz["radius"][a_:b_])) if b_ > a_ else 0.0 for a_, b_ in zip(_off_t[:-1], _off_t[1:])], np.diff(_off_t))
        if (A[A.index("--tether-veins") + 1] if "--tether-veins" in A else "off") == "off":
            _tre = _tre[_art]
    TM0 = tether_mask(spine_x, o_x, h0, _tp, _tr, pathd=_tpd, below_z=o0[2], weights=True, below_rmin=opt("--tether-below-rmin-mm", 4.0) * 1e-3, below_radii=_tre)
    # onto the 4 mm grid: a cell is tethered when any 2 mm sample inside it is (max pooling; `--tether-pool mean`: the
    # mean of the eight samples — tested 2026-11-28, det J 0.08, the IVC's still spot was not the pooling)
    _pool_max = (A[A.index("--tether-pool") + 1] if "--tether-pool" in A else "max") == "max"
    TM = np.zeros(Z.shape, np.float32)
    for dz_ in (-1, 1):
        for dy_ in (-1, 1):
            for dx_ in (-1, 1):
                pz, py, px = Z + dz_ * h / 4, Y + dy_ * h / 4, Xg + dx_ * h / 4
                jz, jy, jx = (np.floor((pz - o_x[2]) / h0).astype(int), np.floor((py - o_x[1]) / h0).astype(int), np.floor((px - o_x[0]) / h0).astype(int))
                inb = (jz >= 0) & (jz < TM0.shape[0]) & (jy >= 0) & (jy < TM0.shape[1]) & (jx >= 0) & (jx < TM0.shape[2])
                if _pool_max:
                    TM[inb] = np.maximum(TM[inb], TM0[jz[inb], jy[inb], jx[inb]])
                else:
                    TM[inb] += TM0[jz[inb], jy[inb], jx[inb]] / 8
    # solid abdominal organs keep their breathing motion (2026-11-28; --tether-keep liver,spleen,kidney; "none": as
    # before): the liver slides on the posterior wall and carries the IVC in its bare area. With the spine's 40 mm shear
    # layer reaching into them, the intrahepatic IVC moved 0–6 mm against 8–13 mm of the liver beside it and the
    # IVC–atrium junction kinked 16°. Keep cells: 4 mm cells whose eight 2 mm samples are all organ, not tethered;
    # the correction is 0 there, so the shear concentrates in the retroperitoneal fat between them and the spine.
    _kn = [x for x in (A[A.index("--tether-keep") + 1] if "--tether-keep" in A else "liver,spleen,kidney").split(",") if x and x != "none"]
    KEEP = None
    if _kn:
        _org = np.isin(lab, [ID[x] for x in _kn if x in ID])
        KEEP = np.ones(Z.shape, bool)
        for dz_ in (-1, 1):
            for dy_ in (-1, 1):
                for dx_ in (-1, 1):
                    pz, py, px = Z + dz_ * h / 4, Y + dy_ * h / 4, Xg + dx_ * h / 4
                    jz, jy, jx = (np.floor((pz - o0[2]) / h0).astype(int), np.floor((py - o0[1]) / h0).astype(int), np.floor((px - o0[0]) / h0).astype(int))
                    inb = (jz >= 0) & (jz < lab.shape[0]) & (jy >= 0) & (jy < lab.shape[1]) & (jx >= 0) & (jx < lab.shape[2])
                    v_ = np.zeros(Z.shape, bool)
                    v_[inb] = _org[jz[inb], jy[inb], jx[inb]]
                    KEEP &= v_
        KEEP &= TM <= 0
    its = 0
    for k in range(3):
        B[k], it_ = tether(B[k], TM, KEEP, h, band=opt("--tether-band", 40.0) * 1e-3)
        its += it_
    tinfo = {"tethered_ml": float((TM > 0).sum() * h**3 * 1e6), "kept_ml": float(KEEP.sum() * h**3 * 1e6) if KEEP is not None else 0.0, "keep": _kn, "graph": gp, "cg_iterations": its}
    print(f"tether: {tinfo}", flush=True)
# vessel rod (2026-11-28; --vessel-rod-mm 0: off, as before). A vessel is a continuous tube: it moves with the internal
# organs it touches and bends smoothly through the stretches where it runs in fat between them. Without it, the IVC
# between the pancreas (≈ 6 mm SI) and the liver (≈ 12 mm) lay in retroperitoneal fat beside the crus and moved 0.3 mm
# — a still spot with 34 % axial compression above it, and 0–2.5 mm at the renal level (markers near the renal
# arteries move 5–7 mm, the kidneys 2.5–20 mm). Per vessel of the classes --vessel-rod-classes (systemic_venous,
# portal_venous) and median radius ≥ --vessel-rod-rmin-mm 4, every 2 mm along it: a = the median breathing motion of the internal-organ voxels in the shell 1–6 mm outside its wall, weighted by
# their share f of the shell, blended with the field at the centreline (weight --vessel-rod-w0 0.02); samples in
# tethered cells (hold ≥ 0.5) are anchored at the tethered field (arteries only, with --vessel-rod-classes all). The centreline motion u minimises
#   ∫ (f + w0) |u − a|² ds + ℓ² ∫ |∂s u|² ds,  ℓ = --vessel-rod-mm 30
# on the vessel graph (edges joined at their nodes) — where organs touch it the vessel follows them over ≈ ℓ / √f,
# across gaps it interpolates. Its lumen cells (4 mm grid, within r of a sample) are then held at u with the tether's
# harmonic correction (≤ --vessel-rod-band 16 mm of the held cells — the tethered cells count as held, unchanged, so
# the band also lies around them, where the correction is ≈ 0; kept organs keep their motion).
# Tried first on all classes, r ≥ 3 mm: det J 0.04 — a 3.7 mm pulmonary vein in the fat at the lung base pulled to the
# lung's 10 mm against the static chest wall, a 3 mm suprarenal artery held near the aorta against the kept liver — and
# the ascending aorta smoothed toward the tethered descending aorta (resp 7.9 → 5.4 mm): small vessels ride on their
# tissue, arteries have the tether and the root hold.
ROD = opt("--vessel-rod-mm", 30.0) * 1e-3
rinfo = None
if ROD > 0 and "--graph" in A:
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import factorized

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from motion_inclusions import tether  # noqa: E402
    _g = np.load(A[A.index("--graph") + 1])
    _P, _R, _off = np.asarray(_g["points"], float), np.asarray(_g["radius"], float), np.asarray(_g["edge_offsets"])
    _ea, _eb = np.asarray(_g["edge_a"]), np.asarray(_g["edge_b"])
    _rmin, _w0, _ds = opt("--vessel-rod-rmin-mm", 4.0) * 1e-3, opt("--vessel-rod-w0", 0.02), 2e-3
    _rc = (A[A.index("--vessel-rod-classes") + 1] if "--vessel-rod-classes" in A else "systemic_venous,portal_venous").split(",")
    _gjr = json.loads((Path(A[A.index("--graph") + 1]).parent / "graph.json").read_text())
    _ccls = {c["id"]: c["class"] for c in _gjr["components"]}
    _ec = np.asarray(_g["edge_comp"])
    assert _rc == ["all"] or set(_rc) <= set(_ccls.values()), f"--vessel-rod-classes {_rc}: not in {sorted(set(_ccls.values()))}"
    assert _w0 > 0, "--vessel-rod-w0 must be > 0 (the data weight where no organ touches the vessel)"
    _Ic = np.isin(lab, [ID[n] for n in INTERNAL if n in ID])  # the organs themselves (no embedded lumens, no closing)
    _TMr = TM if TETHER == "on" else np.zeros(Z.shape, np.float32)

    def _samp(F, q):
        """F [Z, Y, X, 3] on the 4 mm grid at points q [N, 3] (trilinear, clamped)"""
        c = (q - lo) / h - 0.5
        return np.stack([ndimage.map_coordinates(F[..., d], [c[:, 2], c[:, 1], c[:, 0]], order=1, mode="nearest") for d in range(3)], -1)

    def _cell(q):
        return np.clip(np.floor((q - lo) / h).astype(int), 0, dims - 1)

    sq, sr, sa, links, ends = [], [], [], [], {}
    for e_ in range(len(_off) - 1):
        p_, r_ = _P[_off[e_]:_off[e_ + 1]], _R[_off[e_]:_off[e_ + 1]]
        if len(p_) < 2 or np.median(r_) < _rmin or (_rc != ["all"] and _ccls.get(int(_ec[e_]), "") not in _rc):
            continue
        s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p_, axis=0), axis=1))]
        j_ = np.unique(np.r_[np.searchsorted(s_, np.arange(0, s_[-1], _ds)), len(p_) - 1])
        ins = np.all((p_[j_] > lo + h) & (p_[j_] < hi - h), axis=1)  # inside the resp grid (head, legs: out)
        if not ins.any():
            continue
        gi = np.full(len(j_), -1)
        gi[ins] = len(sq) + np.arange(ins.sum())
        sq.extend(p_[j_[ins]])
        sr.extend(r_[j_[ins]])
        sa.extend([_ccls.get(int(_ec[e_]), "").endswith("arterial")] * int(ins.sum()))
        for a_ in range(len(j_) - 1):
            if ins[a_] and ins[a_ + 1]:
                links.append((gi[a_], gi[a_ + 1], max(s_[j_[a_ + 1]] - s_[j_[a_]], 2e-4)))
        if ins[0]:
            ends.setdefault(("n", int(_ea[e_])), []).append(gi[0])
        if ins[-1]:
            ends.setdefault(("n", int(_eb[e_])), []).append(gi[-1])
    sq, sr, sa = np.asarray(sq), np.asarray(sr), np.asarray(sa, bool)
    N = len(sq)
    rinfo = {"length_mm": ROD * 1e3, "classes": _rc, "samples": 0}
    if N:  # no vessel of the classes in the grid: nothing to hold
        nodes = {k_: N + m_ for m_, k_ in enumerate(k for k, v in ends.items() if len(v) > 1)}
        for k_, m_ in nodes.items():
            for i_ in ends[k_]:
                links.append((i_, m_, max(float(np.linalg.norm(sq[i_] - _g["node_pos"][k_[1]])), 5e-4)))
        M = N + len(nodes)
        # organ share f and organ motion per sample (median over the shell's organ voxels, each basis)
        f_ = np.zeros(N)
        a_org = np.zeros((3, N, 3))
        a_fld = np.stack([_samp(B[k], sq) for k in range(3)])
        for i_ in range(N):
            q_, r0 = sq[i_], sr[i_]
            b0 = np.maximum(np.floor((q_ - r0 - 6e-3 - o0) / h0).astype(int), 0)
            b1 = np.minimum(np.ceil((q_ + r0 + 6e-3 - o0) / h0).astype(int) + 1, [lab.shape[2], lab.shape[1], lab.shape[0]])
            if np.any(b1 <= b0):
                continue
            zz, yy, xx = np.meshgrid(*(np.arange(b0[d], b1[d]) for d in (2, 1, 0)), indexing="ij")
            X_ = np.stack([xx, yy, zz], -1) * h0 + o0 + h0 / 2
            dd = np.linalg.norm(X_ - q_, axis=-1)
            sh = (dd > r0 + 1e-3) & (dd <= r0 + 6e-3)
            if not sh.any():
                continue
            og = _Ic[zz[sh], yy[sh], xx[sh]]
            f_[i_] = og.mean()
            if og.any():
                Xo = X_[sh][og]
                for k in range(3):
                    a_org[k, i_] = np.median(_samp(B[k], Xo), axis=0)
        ci = _cell(sq)
        # arteries (with --vessel-rod-classes all) are anchored where tethered; a vein beside a tethered artery is not
        anchor = sa & (_TMr[ci[:, 2], ci[:, 1], ci[:, 0]] >= 0.5)
        wd = np.where(anchor, 1e6, f_ + _w0)
        a_t = np.where(anchor[None, :, None], a_fld, (f_[None, :, None] * a_org + _w0 * a_fld) / (f_ + _w0)[None, :, None])
        li, lj, ll = (np.array([x[c] for x in links], dtype=(int if c < 2 else float)) for c in range(3))
        # data weight per sample: half of its neighbouring intervals along the vessel (node links excluded)
        ds_ = np.zeros(N)
        sm_ = (li < N) & (lj < N)
        np.add.at(ds_, li[sm_], ll[sm_] / 2)
        np.add.at(ds_, lj[sm_], ll[sm_] / 2)
        ds_ = np.where(ds_ > 0, ds_, _ds / 2)
        dw = np.r_[wd * ds_, np.zeros(M - N)]
        kw = ROD**2 / ll
        Am = coo_matrix((np.r_[dw, kw, kw, -kw, -kw], (np.r_[np.arange(M), li, lj, li, lj], np.r_[np.arange(M), li, lj, lj, li])), shape=(M, M)).tocsc()
        solve = factorized(Am)
        u_rod = np.zeros((3, N, 3))
        for k in range(3):
            for c in range(3):
                u_rod[k, :, c] = solve(np.r_[dw[:N] * a_t[k, :, c], np.zeros(M - N)])[:N]
        # hold the lumen cells (centre within r of a sample; at least the sample's own cell) at the rod motion
        acc = np.zeros((3,) + Z.shape + (3,))
        cnt = np.zeros(Z.shape)
        for i_ in range(N):
            q_, r0 = sq[i_], sr[i_]
            b0 = np.maximum(np.floor((q_ - r0 - lo) / h - 0.5).astype(int), 0)
            b1 = np.minimum(np.ceil((q_ + r0 - lo) / h - 0.5).astype(int) + 1, dims)
            zz, yy, xx = np.meshgrid(*(np.arange(b0[d], b1[d]) for d in (2, 1, 0)), indexing="ij")
            inn = np.linalg.norm(np.stack([xx, yy, zz], -1) * h + lo + h / 2 - q_, axis=-1) <= r0
            zz, yy, xx = zz[inn], yy[inn], xx[inn]
            if not len(zz):
                zz, yy, xx = ci[i_, 2:3], ci[i_, 1:2], ci[i_, 0:1]
            acc[:, zz, yy, xx] += u_rod[:, i_][:, None, :]
            cnt[zz, yy, xx] += 1
        H = (cnt > 0) & (_TMr <= 0)
        Th = np.where(H | (_TMr > 0), 1.0, 0.0).astype(np.float32)
        kp = (KEEP if (TETHER == "on" and KEEP is not None) else np.zeros(Z.shape, bool)) & ~H
        its = 0
        for k in range(3):
            tgt = np.where(H[..., None], acc[k] / np.maximum(cnt, 1)[..., None], B[k])
            B[k], it_ = tether(B[k], Th, kp, h, band=opt("--vessel-rod-band", 16.0) * 1e-3, target=tgt)
            its += it_
        dsi = (u_rod - a_fld).sum(0)[:, 2] * 1e3
        rinfo = {"length_mm": ROD * 1e3, "classes": _rc, "rmin_mm": _rmin * 1e3, "samples": int(N), "anchored": int(anchor.sum()), "held_ml": float(H.sum() * h**3 * 1e6),
                 "si_change_mm": [float(dsi.min()), float(np.percentile(dsi, 5)), float(np.percentile(dsi, 95)), float(dsi.max())], "cg_iterations": its}
        print(f"vessel rod: {rinfo}", flush=True)
    else:
        print(f"vessel rod: no samples ({_rc}, r ≥ {_rmin * 1e3:.1f} mm)", flush=True)
Ut = B.sum(0)
geo = {"z_dome": z_dome, "z_apex": z_apex, "y_ant": y_ant, "y_post": y_post, "amplitudes_mm": {k: v * 1e3 for k, v in D.items()}, "tether": tinfo, "vessel_rod": rinfo}


def at(p):
    i = np.clip(((np.asarray(p) - lo) / h).astype(int), 0, dims - 1)
    return (Ut[i[2], i[1], i[0]] * 1e3).round(1).tolist()


geo["u_end_inspiration_mm"] = {"heart centre": at([0.03, -0.03, 1.29]), "liver dome": at([-0.03, -0.02, 1.25]), "lung apex": at([-0.06, 0.0, 1.42]),
                               "sternum": at([0.0, -0.11, 1.30]), "spine": at([0.0, 0.07, 1.30])}
vf.write(OUT / "resp_motion.hv", {"kind": "phantom_motion_resp", "bases": ["si", "ap", "lr"], "B": np.ascontiguousarray(np.moveaxis(B, -1, 1)).astype(np.float32),
                                   "origin_m": lo.tolist(), "voxel_m": h, "dims_xyz": dims.tolist(), **{k: float(v) for k, v in geo.items() if isinstance(v, (int, float))},
                                   "note": "u_r(X, t) = r(t) Σ_k B_k(X) (amplitudes included, metres), r = 0 end-expiration, 1 end-inspiration; trilinear, clamped at the grid edge"})
(OUT / "resp_motion.json").write_text(json.dumps(geo, indent=1))
print(json.dumps(geo, indent=1))

fig, axs = plt.subplots(1, 3, figsize=(17, 6))
j0 = int((0.0 - lo[1]) / h)
i0 = int((0.03 - lo[0]) / h)
mag = np.linalg.norm(Ut, axis=-1) * 1e3
for a_, (img, ext, t_) in zip(axs, [(mag[:, j0, :][::-1], [lo[0], hi[0], lo[2], hi[2]], "coronal y = 0"), (mag[:, :, i0][::-1], [lo[1], hi[1], lo[2], hi[2]], "sagittal x = 30 mm"),
                                    (Ut[:, :, i0, 2][::-1] * 1e3, [lo[1], hi[1], lo[2], hi[2]], "u_z sagittal [mm]")]):
    im = a_.imshow(img, extent=[e * 1e3 for e in ext], cmap="viridis" if "u_z" not in t_ else "coolwarm", vmin=None if "u_z" not in t_ else -D["si"] * 1e3, vmax=None if "u_z" not in t_ else D["si"] * 1e3)
    a_.set_title(f"end-inspiration |u| [mm], {t_}" if "u_z" not in t_ else t_)
    fig.colorbar(im, ax=a_, fraction=0.04)
fig.savefig(OUT / "figs" / "03_resp_field.png", dpi=90, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)
