"""Radius continuity where one vessel continues another (2026-11-28).

The atlas radius is estimated per named mesh, RFIX scales single names, and make_phantom jitters each name by its own
lognormal factor — so where one name continues another the lumen stepped: IVC thoracic 8.2 | abdominal 10.1 mm (24 %,
the jitter), aortic arch 9.4 | thoracic aorta 11.0 (17 %), left subclavian 3.3 | axillary 2.4 (37 %, RFIX), right
superior | inferior pulmonary vein 2.8 | 4.4 (53 %) — a collar in the viewer and an expansion / contraction in the flow
that no vessel has.

joint_taper: at every node joining exactly two edges, and (through=True) at every branch node whose two largest edges
form a through-line (leaving directions ≥ 120° apart, every other edge < 0.8 × the smaller of them — the parent vessel
passing a side branch), the two ends meet at the geometric mean of their end radii when they differ by > 5 %; each
edge's radius ramps multiplicatively back to its own over max(length, 2 r), at most 45 % of the edge. Side branches are
untouched. The end radius is the intercept at the node of a linear fit over min(max(2 mm, r), 0.25 L) — review
2026-11-28: a median over max(2 mm, r) sat ≈ r/2 from the node and, on short edges, reached into the other end's ramp;
the factor rm / r_end applied to the node sample then reversed or worsened 30 / 37 joints of export_full19 (abdominal
aorta 8.2 | 9.0 → 8.9 | 8.3 mm, an 11.3 | 11.3 mm joint → 10.9 | 11.6)."""
import numpy as np


def _leave(p, s_, r_end):
    """unit direction leaving the node along an edge (chord to the sample max(3 mm, 1.5 r) away)"""
    i = int(np.argmin(s_))
    far = np.flatnonzero(s_ >= max(3e-3, 1.5 * r_end))
    j = int(far[np.argmin(s_[far])]) if len(far) else int(np.argmax(s_))  # first sample past t (else the farthest)
    d = p[j] - p[i]
    return d / (np.linalg.norm(d) + 1e-30)


def joint_taper(pts, rad, E_off, E_a, E_b, n_nodes, length=15e-3, through=True, names=None, fixed=None):
    """pts [N, 3], rad [N] (modified copy returned), edges by offsets E_off and end nodes E_a, E_b. Returns
    (rad', report {joint label: [r1_mm, r2_mm]}). fixed [E] (optional): edges whose calibre is set on purpose (vessel
    variants) — joints involving them are left alone."""
    rad = np.asarray(rad, np.float64).copy()
    pts = np.asarray(pts, np.float64)
    E_off, E_a, E_b = np.asarray(E_off), np.asarray(E_a), np.asarray(E_b)
    inc = [[] for _ in range(n_nodes)]
    for k in range(len(E_a)):
        if E_off[k + 1] - E_off[k] < 2:
            continue
        inc[int(E_a[k])].append((k, 0))
        inc[int(E_b[k])].append((k, 1))
    rep = {}
    for n_, lst in enumerate(inc):
        if len(lst) < 2 or (len(lst) > 2 and not through):
            continue
        info = []
        for k, end in lst:
            a, b = E_off[k], E_off[k + 1]
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts[a:b], axis=0), axis=1))]
            s_ = arc if end == 0 else arc[-1] - arc  # distance from the node along the edge
            r_ = rad[a:b]
            w_ = s_ <= max(min(max(2e-3, float(r_[np.argmin(s_)])), 0.25 * arc[-1]), 1e-12)
            r_end = float(np.polyval(np.polyfit(s_[w_], r_[w_], 1), 0.0)) if w_.sum() >= 3 else float(r_[np.argmin(s_)])
            info.append((k, a, b, s_, r_end, arc[-1]))
        if len(info) > 2:
            info.sort(key=lambda x: -x[4])
            if info[2][4] >= 0.8 * info[1][4]:
                continue  # no clear parent through-line (a bifurcation of similar branches)
            d0 = _leave(pts[info[0][1]:info[0][2]], info[0][3], info[0][4])
            d1 = _leave(pts[info[1][1]:info[1][2]], info[1][3], info[1][4])
            if np.degrees(np.arccos(np.clip(d0 @ d1, -1, 1))) < 120:
                continue
            info = info[:2]
        if info[0][0] == info[1][0]:
            continue  # a loop closing on itself
        if fixed is not None and (fixed[info[0][0]] or fixed[info[1][0]]):
            continue
        r1, r2 = info[0][4], info[1][4]
        if max(r1, r2) / min(r1, r2) <= 1.05:
            continue
        rm = np.sqrt(r1 * r2)
        for k, a, b, s_, r_end, L_ in info:
            Lt = max(min(max(length, 2 * r_end), 0.45 * L_), 1e-4)
            rad[a:b] *= (rm / r_end) ** np.clip(1 - s_ / Lt, 0, 1)
        lab = f"node {n_}" + ("" if names is None else f" {names[info[0][0]].split(': ', 1)[-1][:40]} | {names[info[1][0]].split(': ', 1)[-1][:40]}")
        rep[lab] = [round(r1 * 1e3, 2), round(r2 * 1e3, 2)]
    return rad, rep
