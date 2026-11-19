"""Q1: is the knot-pair Toeplitz multiplier separable,  T_kk'(u) ~= sum_r G^(r)_kk' K_r(u)  with small R?

Exact Toeplitz multiplier of knot pair (k, k') on the 2N-embedding grid (no DCW, unit weights):
    T_kk'(u) = sum_j beta_k(t_j) beta_k'(t_j) D(u - k_j),   D = Dirichlet kernel of the 2N grid (~ sinc, 1 cell).
We grid the encode-0 samples of the ph0001 10-min yarn scan (real per-readout k, cardiac phase, bellows) with a
proxy kernel D (nearest cell = no smoothing / trilinear = mild smoothing; a truncated-sinc check on few cells is in
knot_gram_sinc.py) onto a stratified cell subset of the 2N grid and SVD the [cells x pairs] matrix (Gram over pairs).

Knot basis = target config: 6 linear respiratory knots at bellows quantiles x 16 periodic cubic cardiac knots.
Proxies (cell = 1 / (2 FOV)):
  box1mm : 160 mm box at 1 mm, k scaled x2 (1 mm yarn with the same samples per readout), 2N = 320
  box2mm : 160 mm box at 2 mm (native k), 2N = 160
  box4mm : 160 mm box at 4 mm, samples with |k| <= 125 cyc/m only, 2N = 80   (pyramid coarse level)
Output: json with Frobenius error vs R (uniform cell weight = white image, and k^-4 weighted = phantom-like image),
per-cell spectral-norm errors, per-shell ranks, samples per cell.

run: systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 rustlib/.venv/bin/python -I \
     research/experiments/normal_op/knot_gram_rank.py [proxy] [kernel]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
PROXY = {"box1mm": (2.0, 0.16, 160, None), "box2mm": (1.0, 0.16, 80, None), "box4mm": (1.0, 0.16, 40, 125.0)}


def pairs_per_readout(kidx, kw):
    """36 unordered knot pairs per readout; Frobenius weight sqrt(2) on off-diagonal pairs"""
    n = kidx.shape[0]
    a, b = np.triu_indices(8)
    ka, kb = kidx[:, a], kidx[:, b]
    lo, hi = np.minimum(ka, kb), np.maximum(ka, kb)
    key = lo * 1000 + hi
    val = kw[:, a] * kw[:, b] * np.where(lo == hi, 1.0, np.sqrt(2.0))
    # same knot can appear twice when the window wraps (M=16, 4 slots: no) -> fine
    ukeys, inv = np.unique(key, return_inverse=True)
    return inv.reshape(n, 36).astype(np.int32), val.astype(np.float32), ukeys


def main():
    proxy = sys.argv[1] if len(sys.argv) > 1 else "box1mm"
    kern = sys.argv[2] if len(sys.argv) > 2 else "tri"
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
    scale, fov, N, kcut = PROXY[proxy]
    t0 = time.time()
    s, ro_idx = common.encode_samples(0, cols=[0, 1, 2])
    nro, na = s.shape[:2]
    kidx, kw = common.knot_weights_per_readout(ro_idx)
    pid, pval, ukeys = pairs_per_readout(kidx, kw)
    P = len(ukeys)
    kc = s.reshape(-1, 3) / (2 * np.pi)                          # cyc/m
    rsamp = np.repeat(np.arange(nro, dtype=np.int32), na)
    if kcut is not None:
        m = np.linalg.norm(kc, axis=1) <= kcut
        kc, rsamp = kc[m], rsamp[m]
    u = kc * scale * 2 * fov                                     # 2N-grid cell units
    del s, kc
    n2 = 2 * N
    print(f"{proxy} {kern}: {u.shape[0]} samples, {P} knot pairs, 2N = {n2}, |u|max {np.abs(u).max():.1f}  ({time.time()-t0:.0f}s)", flush=True)

    # stratified cell subset: all cells with |u| <= rc, fraction `frac` elsewhere
    rng = np.random.default_rng(1)
    g = np.arange(-N, N, dtype=np.float32)
    rc = 0.1 * N
    ncell = n2 ** 3
    lookup = np.full(ncell, -1, np.int32)
    sel, wrow, rad = [], [], []
    for ix in range(n2):
        xx, yy, zz = np.meshgrid(g[ix:ix + 1], g, g, indexing="ij")
        r = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2).ravel()
        keep = (r <= rc) | (rng.random(r.size) < frac)
        keep &= r <= 1.05 * N * (1.0 if kcut is None else 1.0)
        ids = ix * n2 * n2 + np.nonzero(keep)[0]
        sel.append(ids)
        wrow.append(np.where(r[keep] <= rc, 1.0, 1.0 / frac))
        rad.append(r[keep])
    sel = np.concatenate(sel)
    wrow = np.concatenate(wrow)
    rad = np.concatenate(rad)
    lookup[sel] = np.arange(sel.size, dtype=np.int32)
    ns = sel.size
    print(f"cell subset {ns} ({ns/ncell*100:.2f} %), accumulator {ns*P*4/1e9:.2f} GB", flush=True)
    acc = np.zeros((ns, P), np.float32)
    cnt = np.zeros(ns, np.float64)

    if kern == "nearest":
        corners = [(None, None)]
    else:
        corners = [(c, None) for c in range(8)]
    chunk = 2_000_000
    for c0 in range(0, u.shape[0], chunk):
        uu = u[c0:c0 + chunk]
        rr = rsamp[c0:c0 + chunk]
        if kern == "nearest":
            base = np.rint(uu).astype(np.int64)
            lst = [(base, np.ones(uu.shape[0], np.float32))]
        else:
            f0 = np.floor(uu)
            fr = (uu - f0).astype(np.float32)
            base0 = f0.astype(np.int64)
            lst = []
            for c in range(8):
                o = np.array([(c >> 2) & 1, (c >> 1) & 1, c & 1])
                wgt = np.prod(np.where(o[None, :] == 1, fr, 1 - fr), 1)
                lst.append((base0 + o[None, :], wgt))
        for base, wgt in lst:
            b = base + N
            ok = np.all((b >= 0) & (b < n2), 1)
            cid = (b[:, 0] * n2 + b[:, 1]) * n2 + b[:, 2]
            cid[~ok] = 0
            row = lookup[cid]
            hit = ok & (row >= 0)
            if not hit.any():
                continue
            row = row[hit]
            rh = rr[hit]
            wv = wgt[hit]
            np.add.at(cnt, row, wv)
            rows = np.repeat(row, 36)
            cols = pid[rh].ravel()
            vals = (pval[rh] * wv[:, None]).ravel()
            np.add.at(acc, (rows, cols), vals)
        print(f"  {min(c0+chunk, u.shape[0])}/{u.shape[0]}  ({time.time()-t0:.0f}s)", flush=True)
    del u, rsamp, lookup

    res = {"proxy": proxy, "kernel": kern, "P": int(P), "N": N, "subset_cells": int(ns), "frac": frac}
    # samples per cell (all pairs together: cnt is the gridded sample density)
    rn = rad / N
    shells = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0)]
    res["density_per_shell"] = [float(np.average(cnt[(rn >= a) & (rn < b)])) for a, b in shells]
    res["density_mean_ball"] = float(np.average(cnt[rn < 1.0], weights=wrow[rn < 1.0]))

    def spectrum(wts, mask=None):
        A = acc if mask is None else acc[mask]
        w = wts if mask is None else wts[mask]
        G = (A.astype(np.float64) * w[:, None]).T @ A.astype(np.float64)
        lam = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
        return lam, G

    def rank_at(lam, tols=(1e-1, 1e-2, 1e-3, 1e-4)):
        tail = np.sqrt(np.maximum(np.cumsum(lam[::-1])[::-1] / lam.sum(), 0))  # tail[R] = err keeping R terms
        tail = np.append(tail, 0.0)
        out = {}
        for t in tols:
            out[f"{t:g}"] = int(np.argmax(tail <= t))
        return out, tail

    phi = 1.0 / (1.0 + (rad / 2.0) ** 2) ** 2                     # k^-4 image spectrum, u0 = 2 cells
    lam_u, G_u = spectrum(wrow)
    lam_p, _ = spectrum(wrow * phi)
    ru, tail_u = rank_at(lam_u)
    rp, tail_p = rank_at(lam_p)
    res["rank_frob_uniform"] = ru
    res["rank_frob_kspec"] = rp
    res["frob_err_uniform_R"] = {str(R): float(tail_u[R]) for R in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if R < len(tail_u)}
    res["frob_err_kspec_R"] = {str(R): float(tail_p[R]) for R in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if R < len(tail_p)}
    res["shell_rank_uniform"] = {}
    for a, b in shells:
        m = (rn >= a) & (rn < b)
        if m.sum() > 100:
            lam_s, _ = spectrum(wrow, m)
            res["shell_rank_uniform"][f"{a}-{b}"] = rank_at(lam_s)[0]

    # per-cell spectral-norm error of the K x K knot matrix with the uniform-Gram basis
    w_, V = np.linalg.eigh(G_u)
    V = V[:, ::-1]
    lo = (ukeys // 1000).astype(int)
    hi = (ukeys % 1000).astype(int)
    off = lo != hi
    pick = rng.choice(ns, size=min(4000, ns), replace=False, p=(wrow / wrow.sum()))
    A = acc[pick].astype(np.float64)
    K = 96

    def mats(X):
        Mt = np.zeros((X.shape[0], K, K))
        v = X / np.where(off, np.sqrt(2.0), 1.0)[None, :]
        Mt[:, lo, hi] = v
        Mt[:, hi, lo] = v
        return Mt

    T = mats(A)
    nT = np.abs(np.linalg.eigvalsh(T)).max(1)
    res["cell_rank_1e-2"] = float(np.mean([(np.abs(np.linalg.eigvalsh(t)) > 1e-2 * max(n, 1e-30)).sum() for t, n in zip(T[:500], nT[:500])]))
    res["cell_rank_1e-4"] = float(np.mean([(np.abs(np.linalg.eigvalsh(t)) > 1e-4 * max(n, 1e-30)).sum() for t, n in zip(T[:500], nT[:500])]))
    res["cell_opnorm_err_R"] = {}
    for R in (1, 4, 16, 64, 256, 512):
        if R >= P:
            continue
        Vr = V[:, :R]
        E = mats(A - (A @ Vr) @ Vr.T)
        nE = np.abs(np.linalg.eigvalsh(E)).max(1)
        res["cell_opnorm_err_R"][str(R)] = {"max_over_cells_rel_max": float(nE.max() / nT.max()),
                                           "median_rel_own_cell": float(np.median(nE / np.maximum(nT, 1e-30)))}
    res["seconds"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"knot_gram_{proxy}_{kern}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
