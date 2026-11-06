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
        t = cKDTree(P)
        pairs = t.query_pairs(float(2 * R.max() + 2 * W.max()), output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(P[i] - P[j], axis=1)
        hit = (d < R[i] + R[j] + W[i] + W[j]) & (d >= np.maximum(R[i], R[j]))
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
