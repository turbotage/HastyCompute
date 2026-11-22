"""Joint respiratory x cardiac rank in the heart box (research/temporal_rank.md, run 3).

    rustlib/.venv/bin/python joint_rank.py [--amp A] [--venous ALPHA] [--rdot] [--R 12] [--stride 2] [--tag NAME]

The simulator moves the end-expiration cardiac truth with the respiratory displacement u(X, r) = r sum_k B_k(X)
(motion/resp_motion.hv, 4 mm, trilinear); a 2 mm voxel at breathing state r and cardiac frame c is the complex mean of
its 8 fine sub-points pulled back to X = x - u(X, r) (3 fixed-point steps) and sampled (trilinear) in frame c.
States: R breathing amplitudes at the equal-probability quantile midpoints of the scan's breathing trace (sim_input
breathing r) x the cardiac frames 0, stride, 2 stride, ...
Knobs: --amp scales the respiratory displacement; --venous ALPHA modulates the systemic-venous vessel blood (SVC, IVC,
hepatic, azygos, coronary sinus) velocity by (1 + ALPHA (2 q - 1)) with q the state's breathing quantile (inspiration
fills the right heart: venous return up); --rdot adds the respiratory tissue velocity rdot B(X) with rdot the mean
|rdot| of the state's breathing bin, alternating sign (a hysteresis floor estimate).
Voxels: all lumen / vessel voxels plus a 1-in-4 random subset of the rest (Gram weight 4).
"""
import json
import pickle
import sys
import time

import numpy as np
from scipy import ndimage

from common import CACHE, CBOX, LEVELS, OUT, PH, cardiac_basis, decode, encode_matrix, k_at, orth, resp_hats
from experiment import eig_desc
from rank_lib import ORG, GROUPS, Fields, Vessels, assign_fine, tissues

A = sys.argv[1:]


def opt(name, d, typ=float):
    return typ(A[A.index(name) + 1]) if name in A else d


AMP, VEN, R, STRIDE = opt("--amp", 1.0), opt("--venous", 0.0), opt("--R", 12, int), opt("--stride", 2, int)
RDOT = "--rdot" in A
TAG = opt("--tag", f"joint_amp{AMP}_ven{VEN}{'_rdot' if RDOT else ''}", str)


def resp_field():
    from hastycompute import value_format as vf
    Rm = vf.open_lazy(PH / "motion/resp_motion.hv")
    B = np.asarray(Rm["B"].read()).sum(0)  # [3 (x y z), z, y, x]
    return B, np.array(Rm["origin_m"]), float(Rm["voxel_m"])


def sample_B(B, org, vox, P):
    """trilinear Bsum at world points P [N, 3] (x, y, z) -> [N, 3]"""
    c = ((P - org) / vox)[:, ::-1].T  # z, y, x
    return np.stack([ndimage.map_coordinates(B[a], c, order=1, mode="nearest") for a in range(3)], 1)


def main():
    t0 = time.time()
    F = Fields()
    b2, cat, vmap = pickle.loads((CACHE / "context.pkl").read_bytes())
    sc = 0.0
    S = None
    from hastycompute import value_format as vf
    SI = vf.open_lazy(PH / "sim_flow_10min/sim_input.hv")
    r_tr = np.asarray(SI["breathing"]["r"].read())[:600000]
    rd_tr = np.asarray(SI["breathing"]["rdot"].read())[:600000]
    qs = (np.arange(R) + 0.5) / R
    r_states = np.quantile(r_tr, qs)
    edges = np.quantile(r_tr, np.linspace(0, 1, R + 1))
    rdot_mean = np.array([np.abs(rd_tr[(r_tr >= edges[j]) & (r_tr <= edges[j + 1])]).mean() for j in range(R)])
    frames = list(range(0, F.nf, STRIDE))
    C = len(frames)
    # voxel subset
    rng = np.random.default_rng(1)
    ev = np.zeros(F.nv, bool)
    for per in vmap.values():
        for ci, w, t in per:
            ev[ci] = True
    keep = (b2.max(1) >= 0.25) | ev
    rest = np.flatnonzero(~keep)
    pick = rest[rng.uniform(size=len(rest)) < 0.25]
    vox = np.sort(np.concatenate([np.flatnonzero(keep), pick]))
    wv = np.where(keep[vox], 1.0, 4.0)
    nV = len(vox)
    print(f"{nV} voxels ({keep.sum()} kept + {len(pick)} sampled), states {R} x {C}", flush=True)
    # sub-point world positions (fine voxel centres) of the chosen 2 mm voxels
    cz, cy, cx = np.unravel_index(vox, F.cs)
    off = np.array([(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)])  # z, y, x
    fz = (2 * cz[:, None] + off[None, :, 0]).ravel()
    fy = (2 * cy[:, None] + off[None, :, 1]).ravel()
    fx = (2 * cx[:, None] + off[None, :, 2]).ravel()
    P0 = ORG + (np.stack([fx, fy, fz], 1) + F.f0[::-1]) * 1e-3  # world x y z
    B, borg, bvox = resp_field()
    # pulled-back cache coordinates per breathing state
    c0 = np.array([c[0] for c in CBOX])  # z y x
    coords, rvel = [], []
    for r in r_states:
        X = P0.copy()
        for _ in range(4):
            X = P0 - AMP * r * sample_B(B, borg, bvox, X)
        coords.append((((X - ORG) / 1e-3)[:, ::-1] - c0).T.astype(np.float32))  # cache index z y x
        rvel.append(sample_B(B, borg, bvox, X).T.astype(np.float32) * AMP)
    print(f"pull-back {time.time() - t0:.0f} s; max displacement {AMP * r_states.max() * np.abs(B).max() * 1e3:.1f} mm", flush=True)
    # venous groups for the respiratory modulation: per frame a fine mask of systemic-venous vessel blood (cache box)
    ven_names = ("vena cava", "Hepatic veins", "Azygos", "azygos", "Coronary sinus", "cardiac vein", "brachiocephalic vein", "Brachiocephalic vein")
    Ve = Vessels()
    K = encode_matrix()
    ne = K.shape[0]
    X = np.zeros((ne, nV, R * C), np.complex64)  # state index s = j * C + i (breathing outer)
    bs = np.zeros((nV, R * C), np.float32)
    mcache = np.load(CACHE / "m.npy", mmap_mode="r")
    if "--staircase" not in A:  # continuous-motion magnitude over the whole cache box (regimes.SmoothMagnitude)
        sp = CACHE / "smooth_mag_cbox.npy"
        if not sp.exists():
            from regimes import SmoothMagnitude

            class _Box:
                pass
            bx = _Box()
            bx.fs = tuple(c[1] - c[0] for c in CBOX)
            bx.f0 = np.array([c[0] for c in CBOX])
            bx.nf = F.nf
            bx.m = mcache
            np.save(sp, SmoothMagnitude(bx).m)
        mcache = np.load(sp, mmap_mode="r")
        print(f"continuous-motion magnitude ready {time.time() - t0:.0f} s", flush=True)
    vcache = np.load(CACHE / "v.npy", mmap_mode="r")
    bcache = np.load(CACHE / "blood.npy", mmap_mode="r")
    from rank_lib import abox_in_cache
    for i, f in enumerate(frames):
        m = mcache[f].astype(np.float32)
        v = vcache[f].astype(np.float32)
        bl = bcache[f].astype(np.float32) / 255
        vm = None
        if VEN:
            sel, pi, d, rr = assign_fine(F, Ve, f)
            isv = np.array([any(s in Ve.names[e] for s in ven_names) for e in Ve.eid[pi]])
            vm = np.zeros(m.shape, np.float32)
            sl = abox_in_cache()
            sub = np.zeros(F.fs, np.float32).reshape(-1)
            sub[sel[isv]] = 1.0
            vm[sl] = sub.reshape(F.fs)
        for j in range(R):
            cc = coords[j]
            ms = ndimage.map_coordinates(m, cc, order=1, mode="nearest")
            vs = np.stack([ndimage.map_coordinates(v[a], cc, order=1, mode="nearest") for a in range(3)])
            if VEN:
                g = 1 + VEN * (2 * qs[j] - 1)
                ws = ndimage.map_coordinates(vm, cc, order=1, mode="nearest")
                vs = vs * (1 + (g - 1) * ws)[None]
            if RDOT:
                sgn = 1.0 if (i % 2 == 0) else -1.0
                vs = vs + sgn * rdot_mean[j] * rvel[j]
            bs[:, j * C + i] = ndimage.map_coordinates(bl, cc, order=1, mode="nearest").reshape(nV, 8).mean(1)
            for e in range(ne):
                ph = K[e, 0] * vs[0] + K[e, 1] * vs[1] + K[e, 2] * vs[2]
                X[e, :, j * C + i] = (ms * np.exp(-1j * ph)).reshape(nV, 8).mean(1)
        if i % 10 == 0:
            print(f"frame {f} {time.time() - t0:.0f} s", flush=True)
    analyse_joint(X, wv, bs, r_states, edges, qs, r_tr, frames, F.nf, t0)


def proj_err(X, Phi, wv, mask):
    num = den = 0.0
    for e in range(X.shape[0]):
        Xe = X[e]
        Pr = (Xe @ Phi) @ Phi.conj().T
        r2 = (np.abs(Xe - Pr) ** 2).sum(1) * wv
        x2 = (np.abs(Xe) ** 2).sum(1) * wv
        num += r2[mask].sum()
        den += x2[mask].sum()
    return float(np.sqrt(num / den))


def dv_err(X, Phi, bs, lumen):
    XB = X[:, lumen]
    P = np.stack([(XB[e] @ Phi) @ Phi.conj().T for e in range(X.shape[0])], -1)
    vt = decode(np.moveaxis(XB, 0, -1))
    vp = decode(P)
    ok = bs[lumen] >= 0.5
    dv = np.linalg.norm(vp - vt, axis=-1)[ok] * 100
    return float(dv.mean()), float(np.percentile(dv, 99)), float((dv > 10).mean())


def analyse_joint(X, wv, bs, r_states, edges, qs, r_tr, frames, nf, t0):
    R, C = len(r_states), len(frames)
    ne = X.shape[0]
    sw = np.sqrt(wv)[:, None]
    G = sum((X[e] * sw).conj().T @ (X[e] * sw) for e in range(ne)).astype(np.complex128)
    w, U = eig_desc(G)
    res = {"tag": TAG, "amp": AMP, "venous": VEN, "rdot": RDOT, "R": R, "C": C, "r_states": r_states.tolist(), "levels": LEVELS}
    res["spectrum"] = (w / w[0]).tolist()
    res["K_joint"] = k_at(w, LEVELS)
    lumen = bs.mean(1) >= 0.5
    allm = np.ones(X.shape[1], bool)
    # separate axes: cardiac Gram within each breathing state, breathing Gram within each cardiac frame
    Xr = X.reshape(ne, X.shape[1], R, C)
    Gc = sum(np.einsum("vjc,vjd->cd", (Xr[e] * sw[:, :, None]).conj(), Xr[e] * sw[:, :, None]) for e in range(ne))
    Gr = sum(np.einsum("vjc,vkc->jk", (Xr[e] * sw[:, :, None]).conj(), Xr[e] * sw[:, :, None]) for e in range(ne))
    wc, Uc = eig_desc(Gc.astype(np.complex128))
    wr, Ur = eig_desc(Gr.astype(np.complex128))
    res["K_cardiac_pooled"], res["K_resp_pooled"] = k_at(wc, LEVELS), k_at(wr, LEVELS)
    res["spectrum_cardiac"], res["spectrum_resp"] = (wc / wc[0]).tolist(), (wr / wr[0]).tolist()
    rows = []
    phase = np.array(frames) / nf
    for k in [4, 8, 12, 16, 20, 24, 32, 48, 64, 96]:
        if k > R * C:
            continue
        Phi = U[:, :k]
        e_all, e_bl = proj_err(X, Phi, wv, allm), proj_err(X, Phi, wv, lumen)
        dm, dp, dg = dv_err(X, Phi, bs, lumen)
        rows.append(dict(basis="svd_joint", K=k, rel_all=e_all, rel_blood=e_bl, dv_mean=dm, dv_p99=dp, dv_gt10=dg))
    # separable SVD (kr x kc) and the knot baseline (linear hats at r quantiles x periodic cubic B-splines)
    for kr, kc in [(2, 8), (2, 12), (3, 12), (3, 16), (4, 16), (4, 24), (6, 16), (6, 24)]:
        Phi = np.kron(Ur[:, :kr], Uc[:, :kc])
        e_all, e_bl = proj_err(X, Phi, wv, allm), proj_err(X, Phi, wv, lumen)
        dm, dp, dg = dv_err(X, Phi, bs, lumen)
        rows.append(dict(basis=f"svd_sep {kr}x{kc}", K=kr * kc, rel_all=e_all, rel_blood=e_bl, dv_mean=dm, dv_p99=dp, dv_gt10=dg))
    for nk, mc in [(4, 16), (6, 16), (6, 24), (8, 16), (8, 24), (12, 16)]:
        knots = np.quantile(r_tr, np.linspace(0, 1, nk))
        Br = resp_hats(r_states, knots)
        Bc = cardiac_basis(phase, mc, 3)
        Phi = orth(np.kron(Br, Bc)).astype(np.complex128)
        e_all, e_bl = proj_err(X, Phi, wv, allm), proj_err(X, Phi, wv, lumen)
        dm, dp, dg = dv_err(X, Phi, bs, lumen)
        rows.append(dict(basis=f"knots {nk}lin x {mc}cub", K=nk * mc, rel_all=e_all, rel_blood=e_bl, dv_mean=dm, dv_p99=dp, dv_gt10=dg))
    res["rows"] = rows
    res["seconds"] = time.time() - t0
    (OUT / f"{TAG}.json").write_text(json.dumps(res, indent=1))
    print(f"K_joint {res['K_joint']}  K_cardiac {res['K_cardiac_pooled']}  K_resp {res['K_resp_pooled']}")
    for r in rows:
        print(f"{r['basis']:22s} K={r['K']:3d} rel_all {r['rel_all']:.3g} rel_blood {r['rel_blood']:.3g} dv_mean {r['dv_mean']:.3g} dv_p99 {r['dv_p99']:.3g} gt10 {r['dv_gt10']:.3g}")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
