"""Remove physically impossible lumen overlaps between vessels that are not adjacent in the graph.

The atlas radii (bevel depth × point radius) make some neighbours intersect (ascending aorta ↔ pulmonary trunk,
iliac artery ↔ vein). Overlaps near a junction are legitimate (the child leaves the parent's lumen). A pair of
centreline samples i, j conflicts when their tubes plus walls intersect, |p_i − p_j| < r_i + r_j + w_i + w_j, neither axis
lies inside the other's lumen (|p_i − p_j| ≥ max(r_i, r_j): side by side, not co-running / crossing — the atlas draws
some branches along their parent's axis), and their graph-geodesic distance exceeds 2 (r_i + r_j) + 2 mm. Each conflicting sample gets the factor (|p_i − p_j| − w_i − w_j) /
(r_i + r_j) (both sides shrink alike). Factors take a running minimum over ±1.5 r along the edge, are smoothed, and
are floored at `floor` × the original radius.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def deoverlap(P, R, eoff, ea, eb, npos, wall, floor=0.4, iters=3):
    P = np.asarray(P, np.float64)
    R = np.asarray(R, np.float64).copy()
    R0 = R.copy()
    ne = len(ea)
    es = np.searchsorted(eoff, np.arange(len(P)), side="right") - 1
    seg = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    arc = np.zeros(len(P))
    elen = np.zeros(ne)
    for k in range(ne):
        a, b = eoff[k], eoff[k + 1]
        s = np.cumsum(seg[a:b])
        s -= s[0]
        arc[a:b] = s
        elen[k] = s[-1]
    nn = len(npos)
    G = sp.coo_matrix((np.r_[elen, elen] + 1e-9, (np.r_[ea, eb], np.r_[eb, ea])), shape=(nn, nn)).tocsr()
    lim = 0.12
    D = dijkstra(G, indices=np.arange(nn), limit=lim)
    report = {"pairs": 0}
    for _ in range(iters):
        W = wall(R)
        i, j, d = candidate_pairs(P, R + W)  # = the old query_pairs(2 max(r + w)) filtered by d < r_i + r_j + w_i + w_j
        hit = d >= np.maximum(R[i], R[j])
        i, j, d = i[hit], j[hit], d[hit]
        e, f = es[i], es[j]
        si, sj = arc[i], arc[j]
        le, lf = elen[e], elen[f]
        gd = np.minimum.reduce([si + D[ea[e], ea[f]] + sj, si + D[ea[e], eb[f]] + (lf - sj), (le - si) + D[eb[e], ea[f]] + sj, (le - si) + D[eb[e], eb[f]] + (lf - sj)])
        gd = np.where(e == f, np.abs(si - sj), gd)
        bad = gd > 2 * (R[i] + R[j]) + 2e-3
        i, j, d = i[bad], j[bad], d[bad]
        if len(i) == 0:
            break
        fac = np.clip((d - W[i] - W[j]) / (R[i] + R[j]), 0.0, 1.0)
        F = np.ones(len(P))
        np.minimum.at(F, i, fac)
        np.minimum.at(F, j, fac)
        # running minimum over ±1.5 r along each edge, then a 3-pass box smoothing; floor
        G2 = F.copy()
        for k in np.unique(es[F < 1]):
            a, b = eoff[k], eoff[k + 1]
            s, x, r = arc[a:b], F[a:b], R[a:b]
            y = np.array([x[np.abs(s - s[q]) <= 1.5 * r[q]].min() for q in range(b - a)])
            for _ in range(3):
                y = np.minimum(y, np.convolve(np.pad(y, 2, mode="edge"), np.ones(5) / 5, mode="valid")) if len(y) > 4 else y
            G2[a:b] = y
        report["pairs"] += int(len(i))
        R = np.maximum(R * G2, floor * R0)
    report["samples_shrunk"] = int((R < R0 * (1 - 1e-6)).sum())
    report["at_floor"] = int((R <= floor * R0 * (1 + 1e-6)).sum())
    return R, report


def gaussian_spread(P, push, src, dst, sig, allow=None):
    """the pushes of samples `src` spread over samples `dst` (one component): per dst sample the Gaussian-weighted
    average push (σ = sig per src sample, within 3σ) × the largest weight — vectorised (one neighbour query, bincounts).
    allow(src_samples, dst_samples) → bool mask: only those (source, target) pairs spread (e.g. along the same vessel:
    two crossing branches of one tree otherwise average each other's opposite pushes away)
    """
    import itertools
    out = np.zeros((len(dst), 3))
    if len(src) == 0 or len(dst) == 0:
        return out
    nb = cKDTree(P[dst]).query_ball_point(P[src], 3 * sig)
    lens = np.fromiter((len(x) for x in nb), np.int64, len(nb))
    tot = int(lens.sum())
    if tot == 0:
        return out
    cols = np.fromiter(itertools.chain.from_iterable(nb), np.int64, tot)
    rows = np.repeat(np.arange(len(src)), lens)
    if allow is not None:
        ok = allow(src[rows], dst[cols])
        rows, cols = rows[ok], cols[ok]
    w = np.exp(-((P[dst[cols]] - P[src[rows]]) ** 2).sum(1) / (2 * sig[rows] ** 2))
    den = np.bincount(cols, w, len(dst))
    num = np.stack([np.bincount(cols, w * push[src[rows], k], len(dst)) for k in range(3)], 1)
    wmx = np.zeros(len(dst))
    np.maximum.at(wmx, cols, w)
    m = den > 0
    out[m] = num[m] / den[m, None] * wmx[m, None]
    return out


def candidate_pairs(P, reach):
    """all sample pairs with |p_i − p_j| < reach_i + reach_j (i, j, distance), each once: every sample searches within
    twice its own reach and keeps partners of smaller reach (ties by index)"""
    import itertools
    nb = cKDTree(P).query_ball_point(P, 2 * reach)
    lens = np.fromiter((len(x) for x in nb), np.int64, len(nb))
    j = np.fromiter(itertools.chain.from_iterable(nb), np.int64, int(lens.sum()))
    i = np.repeat(np.arange(len(P)), lens)
    keep = (reach[j] < reach[i]) | ((reach[j] == reach[i]) & (j > i))
    i, j = i[keep], j[keep]
    d = np.linalg.norm(P[i] - P[j], axis=1)
    hit = d < reach[i] + reach[j]
    return i[hit], j[hit], d[hit]


def inter_overlaps(P, R, W, comp):
    """Sample pairs of different components whose tubes plus walls intersect: (i, j, distance, required distance).
    Each sample searches within twice its own reach r + w and keeps partners of smaller reach (ties by index): every
    intersecting pair (d < reach_i + reach_j ≤ 2 max(reach)) is found once — instead of one 2 max(r + w) ≈ 30 mm query
    for all samples (hundreds of millions of candidates; 10 s per call, 2026-11-25)."""
    import itertools
    reach = R + W
    t = cKDTree(P)
    nb = t.query_ball_point(P, 2 * reach)
    lens = np.fromiter((len(x) for x in nb), np.int64, len(nb))
    j = np.fromiter(itertools.chain.from_iterable(nb), np.int64, int(lens.sum()))
    i = np.repeat(np.arange(len(P)), lens)
    keep = ((reach[j] < reach[i]) | ((reach[j] == reach[i]) & (j > i))) & (comp[i] != comp[j])
    i, j = i[keep], j[keep]
    d = np.linalg.norm(P[i] - P[j], axis=1)
    need = R[i] + R[j] + W[i] + W[j]
    hit = d < need
    return i[hit], j[hit], d[hit], need[hit]


def _skip_mask(skip_pair, ca, cb):
    """skip_pair on component-id arrays: vectorised (`skip_pair.vectorized`) or per unique pair (the scalar form)"""
    if getattr(skip_pair, "vectorized", False):
        return np.asarray(skip_pair(ca, cb), bool)
    keys, inv = np.unique(np.stack([ca, cb], 1), axis=0, return_inverse=True)
    vals = np.array([bool(skip_pair(int(a), int(b))) for a, b in keys])
    return vals[np.ravel(inv)]


def separate(P, R, eoff, comp, wall, mobility, skip_pair=None, cap=None, tol=0.2e-3, iters=30, relax=0.5):
    """Move centrelines of different graph components apart where their tubes (plus walls) interpenetrate.

    Shrinking cannot resolve deep overlaps (the atlas puts e.g. the right superior pulmonary vein 4–7 mm from the right
    pulmonary artery's axis, radii 5.3 and 7.4 mm), and a separating wall carved through the overlap later pinches both
    lumens. Each conflicting pair (i, j) needs |p_i − p_j| ≥ r_i + r_j + w_i + w_j. The push direction is radial from
    the larger vessel's axis (the component of p_small − p_large normal to the larger vessel's tangent; for an exact
    axis crossing the normal of both tangents). The deficit is split so the smaller vessel moves more, scaled by each
    sample's mobility (reduced near the heart roots): share_i ∝ m_i r_j. Each sample keeps its largest push; the pushes
    of one component are spread over that component's samples by a Gaussian of width σ = max(2.5 r, 4 mm) around each
    pushed sample (normalised average × the largest weight, as in the root separation), so a vessel bends smoothly and
    the other component is not dragged along. The spread step is multiplied by the destination sample's own mobility
    (a function of position, so samples meeting at a graph node move alike), and each sample's total displacement is
    capped at `cap` (n,) (None: no cap); what a cap leaves is reported and falls to the radius shrink. Under-relaxed
    (`relax`), iterated until the largest deficit < `tol`. Samples meeting at a node can get different caps (different
    radii): the caller re-joins them (mean of the end samples).

    P (n, 3) samples (edge-major, offsets `eoff`), R radii, comp per sample, wall(R) → wall thickness per sample,
    mobility (n,) in (0, 1], skip_pair(ci, cj) → True to leave a component pair alone (e.g. a same-system fragment
    that should be joined, not pushed). Returns (P_new, displacement (n, 3), report).
    """
    P0 = np.asarray(P, np.float64).copy()
    P = P0.copy()
    R = np.asarray(R, np.float64)
    comp = np.asarray(comp)
    mobility = np.asarray(mobility, np.float64)
    W = wall(R)
    n = len(P)
    # unit tangents per sample (central differences within each edge)
    T = np.zeros_like(P)
    for k in range(len(eoff) - 1):
        a, b = int(eoff[k]), int(eoff[k + 1])
        if b - a < 2:
            continue
        T[a:b] = np.gradient(P[a:b], axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    i0, j0, d0, need0 = inter_overlaps(P, R, W, comp)
    rep = {"pairs_before": int(len(i0)), "max_deficit_before_mm": float((need0 - d0).max() * 1e3) if len(i0) else 0.0}
    disp = np.zeros_like(P)
    skipped = set()
    it_done = 0
    for it in range(iters):
        i, j, d, need = inter_overlaps(P, R, W, comp)
        if skip_pair is not None and len(i):
            ok = ~_skip_mask(skip_pair, comp[i], comp[j])
            if (~ok).any():
                sk = np.unique(np.sort(np.stack([comp[i][~ok], comp[j][~ok]], 1), axis=1), axis=0)
                skipped.update((int(a), int(b)) for a, b in sk)
            i, j, d, need = i[ok], j[ok], d[ok], need[ok]
        if len(i) == 0 or (need - d).max() < tol:
            break
        it_done = it + 1
        # orient each pair as (large L, small S)
        swap = R[i] < R[j]
        L = np.where(swap, j, i)
        S_ = np.where(swap, i, j)
        v = P[S_] - P[L]
        tl = T[L]
        nrm = v - (v * tl).sum(1, keepdims=True) * tl
        nl = np.linalg.norm(nrm, axis=1)
        # exact axis crossing / co-linear: the normal of both tangents, else any normal of the larger tangent
        alt = np.cross(tl, T[S_])
        an = np.linalg.norm(alt, axis=1)
        fallback = np.cross(tl, np.where(np.abs(tl[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]))
        alt = np.where(an[:, None] > 1e-6, alt / np.maximum(an, 1e-12)[:, None], fallback / np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-12))
        u = np.where(nl[:, None] > 1e-6 * need[:, None], nrm / np.maximum(nl, 1e-12)[:, None], alt)
        # deficit along u: the distance normal to the larger axis must reach `need`
        dn = np.where(nl > 1e-6 * need, nl, 0.0)
        deficit = np.maximum(need - dn, 0.0)
        mL, mS = mobility[L], mobility[S_]
        wL, wS = mL * R[S_], mS * R[L]
        tot = np.maximum(wL + wS, 1e-12)
        pushL = -(deficit * wL / tot)[:, None] * u
        pushS = (deficit * wS / tot)[:, None] * u
        push = np.zeros_like(P)
        pmag = np.zeros(n)
        idx = np.r_[L, S_]
        pv = np.r_[pushL, pushS]
        mag = np.linalg.norm(pv, axis=1)
        np.maximum.at(pmag, idx, mag)  # the largest push per sample wins (deterministic: first of the maxima)
        win = np.flatnonzero(mag >= pmag[idx])
        _, first = np.unique(idx[win], return_index=True)
        sel = win[first]
        push[idx[sel]] = pv[sel]
        on = np.flatnonzero(pmag > 0)
        step = np.zeros_like(P)
        for c in np.unique(comp[on]):
            src = on[comp[on] == c]
            dst = np.flatnonzero(comp == c)
            step[dst] = gaussian_spread(P, push, src, dst, np.maximum(2.5 * R[src], 4e-3))
        step *= mobility[:, None]
        disp = disp + relax * step
        if cap is not None:
            dn_ = np.linalg.norm(disp, axis=1)
            over = dn_ > cap
            disp[over] *= (cap[over] / dn_[over])[:, None]
        P = P0 + disp
    i1, j1, d1, need1 = inter_overlaps(P, R, W, comp)
    if skip_pair is not None and len(i1):
        ok = ~_skip_mask(skip_pair, comp[i1], comp[j1])
        i1, j1, d1, need1 = i1[ok], j1[ok], d1[ok], need1[ok]
    dm = np.linalg.norm(disp, axis=1)
    rep.update({"iterations": it_done, "pairs_after": int(len(i1)), "max_deficit_after_mm": float((need1 - d1).max() * 1e3) if len(i1) else 0.0,
                "max_displacement_mm": float(dm.max() * 1e3), "samples_moved_over_1mm": int((dm > 1e-3).sum()),
                "samples_at_cap": int((dm >= cap * (1 - 1e-6)).sum()) if cap is not None else 0,
                "skipped_component_pairs": sorted(skipped)})
    return P, disp, rep
