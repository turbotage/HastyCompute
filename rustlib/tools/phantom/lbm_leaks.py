"""Lattice leaks of a cardiac LBM case: D3Q19 links between fluid nodes of regions that must not exchange fluid.

    rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_leaks.py <case_dir> <snap_xxx.f32> [<snap…>]

Snapshots from `cardiac_lbm --probe … --snap …`. Each fluid node (not valve solid) is mapped back to the reference pose
through the case's motion (fixed-point inverse) and takes the region of the builder's region map (`region.npy`:
chambers 1–4, stubs 10+). A link between two regions is allowed when the pair is a venous junction or a valve's two
sides within 1.5 radii of the valve centre; every other link between different regions is a leak through a wall.
Prints the leaking links per region pair with their positions (clustered at 3 mm).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
CASE = Path(A[0])
case = json.loads((CASE / "case.json").read_text())
R = case["regions"]
reg = np.load(CASE / R["file"])
so = np.asarray(case["sdf"]["origin"], float)
sh = float(case["sdf"]["h"])
names = {int(k): v for k, v in R["names"].items()}
allowed = {tuple(sorted(p)) for p in R["allowed"]}
vpairs = {}
for v in case["valves"]:
    ud = tuple(sorted(R["valves"][v["name"]]))
    vpairs[ud] = v
cm = vf.read(Path(case["motion"]["file"]))
U = np.asarray(cm["U_mm"], np.float32) * 1e-3  # [nt, 3, Z, Y, X]
nt = U.shape[0]
mo = np.asarray(cm["origin_m"], float)
mh = float(cm["voxel_m"])
T = float(cm["period"])
Zm, Ym, Xm = U.shape[2:]


def disp(P, t):
    s = (t % T) / T * nt
    k = int(np.floor(s))
    f = s - k
    cw = [-0.5 * f**3 + f**2 - 0.5 * f, 1.5 * f**3 - 2.5 * f**2 + 1, -1.5 * f**3 + 2 * f**2 + 0.5 * f, 0.5 * f**3 - 0.5 * f**2]
    g = (P - mo) / mh - 0.5  # voxel centres at o + (i + ½) h (hasty-cfd motion.rs)
    i0 = np.floor(g).astype(int)
    fr = g - i0
    out = np.zeros_like(P)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                ok = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < Xm) & (iy < Ym) & (iz < Zm)
                w = np.where(dx, fr[:, 0], 1 - fr[:, 0]) * np.where(dy, fr[:, 1], 1 - fr[:, 1]) * np.where(dz, fr[:, 2], 1 - fr[:, 2]) * ok
                ixc, iyc, izc = np.clip(ix, 0, Xm - 1), np.clip(iy, 0, Ym - 1), np.clip(iz, 0, Zm - 1)
                for j, c_ in enumerate(cw):
                    out += (w * c_)[:, None] * U[(k - 1 + j) % nt, :, izc, iyc, ixc]
    return out


Q18 = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if 1 <= abs(a) + abs(b) + abs(c) <= 2])
for sp in A[1:]:
    sp = Path(sp)
    hdr = json.loads(sp.with_suffix(".json").read_text())
    n = int(hdr["nodes"])
    D = np.fromfile(sp, np.float32).reshape(6, n)
    tx = np.asarray(hdr["tile_xyz"], np.int64).reshape(-1, 3)
    li = np.arange(512)
    g = np.stack([(tx[:, None, 0] * 8 + li % 8).ravel(), (tx[:, None, 1] * 8 + li // 8 % 8).ravel(), (tx[:, None, 2] * 8 + li // 64).ravel()], 1)
    org = np.asarray(hdr["origin"], float)
    dx = float(hdr["dx"])
    t = float(hdr["t"])
    fl = D[5].astype(np.uint8)
    fluid = ((fl & 1) > 0) & (D[4] < 0.99)
    gf = g[fluid]
    x = gf * dx + org
    X = x.copy()
    for _ in range(6):
        X = x - disp(X, t)
    si = np.floor((X - so) / sh).astype(int)
    ok = np.all((si >= 0) & (si < np.array(reg.shape[::-1])), 1)
    r = np.zeros(len(X), np.int16)
    r[ok] = reg[si[ok, 2], si[ok, 1], si[ok, 0]]
    # unlabelled (smoothing margin): nearest labelled voxel within 2 voxels
    miss = (r == 0) & ok
    if miss.any():
        best = np.full(miss.sum(), 99.0)
        for dz in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dx_ in (-2, -1, 0, 1, 2):
                    q = si[miss] + [dx_, dy, dz]
                    okq = np.all((q >= 0) & (q < np.array(reg.shape[::-1])), 1)
                    rv = np.zeros(len(q), np.int16)
                    rv[okq] = reg[q[okq, 2], q[okq, 1], q[okq, 0]]
                    d2 = dx_**2 + dy**2 + dz**2
                    upd = (rv > 0) & (d2 < best)
                    sub = r[miss]
                    sub[upd] = rv[upd]
                    r[miss] = sub
                    best[upd] = d2
    gmin = gf.min(0)
    shp = gf.max(0) - gmin + 3
    idx = np.full(tuple(shp[::-1]), -1, np.int64)
    gl = gf - gmin + 1
    idx[gl[:, 2], gl[:, 1], gl[:, 0]] = np.arange(len(gf))
    leaks = {}
    for q in Q18:
        nb = gl + q
        j = idx[nb[:, 2], nb[:, 1], nb[:, 0]]
        has = j >= 0
        a_, b_ = r[has], r[j[has]]
        diff = (a_ > 0) & (b_ > 0) & (a_ != b_)
        ii = np.flatnonzero(has)[diff]
        jj = j[has][diff]
        for i_, j_ in zip(ii, jj):
            pr = tuple(sorted((int(r[i_]), int(r[j_]))))
            if pr in allowed:
                continue
            if pr in vpairs:
                v = vpairs[pr]
                if np.linalg.norm(X[i_] - np.asarray(v["center"])) < 1.5 * v["radius"]:
                    op = np.interp(t, np.asarray(v["opening"]["t"]), np.asarray(v["opening"]["v"]), period=T)
                    if op < 0.02 and i_ < j_:
                        # closed valve: where the link passes, in the valve frame (reference pose)
                        c_, n_ = np.asarray(v["center"]), np.asarray(v["normal"])
                        e1 = np.asarray(v["e1"])
                        e2 = np.cross(n_, e1)
                        d_ = 0.5 * (X[i_] + X[j_]) - c_
                        rr = np.linalg.norm(d_ - (d_ @ n_) * n_) / v["radius"]
                        ang = np.degrees(np.arctan2(d_ @ e2, d_ @ e1))
                        leaks.setdefault(("closed " + v["name"],), []).append((0.5 * (x[i_] + x[j_]), rr, ang, d_ @ n_ * 1e3))
                    continue
            if i_ < j_:
                leaks.setdefault(pr, []).append(0.5 * (x[i_] + x[j_]))
    print(f"{sp.name} t {t * 1e3:.0f} ms: {len(gf)} fluid nodes, {int((r == 0).sum())} without region; leaking links by region pair:")
    for pr, pts in sorted(leaks.items(), key=lambda z: -len(z[1])):
        if len(pr) == 1:
            rr = np.array([p_[1] for p_ in pts])
            an = np.array([p_[2] for p_ in pts])
            of = np.array([p_[3] for p_ in pts])
            h_, e_ = np.histogram(rr, bins=[0, 0.3, 0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5])
            print(f"   {pr[0]}: {len(pts)} links across the closed valve; r/R histogram {dict(zip([f'{a:.1f}-{b:.1f}' for a, b in zip(e_[:-1], e_[1:])], h_.tolist()))}; "
                  f"angle quartiles {np.round(np.quantile(an, [0.25, 0.5, 0.75]), 0).tolist()}°, axial offset {np.round(np.quantile(of, [0.1, 0.5, 0.9]), 1).tolist()} mm")
            continue
        P_ = np.asarray(pts)
        # clusters: greedy 3 mm
        cl = []
        left = np.ones(len(P_), bool)
        while left.any() and len(cl) < 5:
            c0 = P_[np.flatnonzero(left)[0]]
            m = left & (np.linalg.norm(P_ - c0, axis=1) < 3e-3)
            cl.append((m.sum(), P_[m].mean(0)))
            left &= ~m
        print(f"   {names.get(pr[0], pr[0])[:22]:22s} ↔ {names.get(pr[1], pr[1])[:22]:22s}: {len(pts):5d} links; at " + ", ".join(f"{np.round(c * 1e3, 1).tolist()} ({k})" for k, c in cl))
