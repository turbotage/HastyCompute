"""Solvers for the cardiac motion outside the heart (research/flow_phantom.md Tier B5).

Used by cardiac_motion.py:
- `harmonic_extension` (default, `--outside harmonic`): the AV-descent and twist fields are the heart's kinematic fields
  extended harmonically outside the heart to zero at the grid boundary (≥ 80 mm away) instead of a 25 mm smoothstep, so
  the tissue around the great vessels is not compressed by the fade (descending aorta stretch 0.57 → 0.85; within
  40 mm of the heart p1 0.53 → 0.71, det min 0.26 → 0.42);
- `insulated_poisson_mode` + `vessel_inclusions` (experimental, `--insulate-vessels`): chamber volume modes solved with
  the great vessels as insulated walls. It folds the field (det J −7): potential flow around voxelised tubes is
  singular at their corners and concentrates in narrow gaps. A first version moved the tubes rigidly with their tree's
  root, fading along the vessel (`vessel_inclusions` weights); the displacement jumped at the tube walls (det J −16).
  Sliding interfaces would need a discontinuous displacement, which one continuous field (phantom_4d's label pull-back,
  the LBM's moving lattice) cannot hold.
Solver: Jacobi-preconditioned CG on the 7-point Laplacian with cell types (unknown / Dirichlet / insulated; Dirichlet 0
outside the grid; walled-in pockets pinned), torch on the GPU when available.
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

UNKNOWN, DIRICHLET, INSULATED = 0, 1, 2


def _torch():
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch, dev


def masked_laplace(ctype, val, rhs=None, x0=None, tol=1e-6, maxit=6000):
    """solve Σ_faces c_f (u_i − u_j) = rhs_i on the UNKNOWN cells: u_j = val_j on DIRICHLET cells and 0 outside the
    grid; faces to INSULATED cells do not conduct (c_f = 0). ctype, val, rhs, x0: [Z, Y, X]. Returns u (val on the
    DIRICHLET cells, the solution on UNKNOWN, 0 on INSULATED) and the iteration count."""
    torch, dev = _torch()
    # unknown cells that reach neither the grid edge nor a Dirichlet cell through conducting faces (pockets walled in
    # by insulated cells, single cells included: zero conductance) make the system singular: pinned to 0
    ctype = np.asarray(ctype).copy()
    unk_np = ctype == UNKNOWN
    lab, nlab = ndimage.label(unk_np)
    if nlab:
        reach = np.zeros(nlab + 1, bool)
        edge = np.zeros_like(unk_np)
        edge[[0, -1]] = True
        edge[:, [0, -1]] = True
        edge[:, :, [0, -1]] = True
        reach[np.unique(lab[edge & unk_np])] = True
        nd = ndimage.binary_dilation(ctype == DIRICHLET) & unk_np
        reach[np.unique(lab[nd])] = True
        reach[0] = True
        pocket = ~reach[lab]
        if pocket.any():
            ctype[pocket] = DIRICHLET
            val = np.where(pocket, 0.0, val)
    T = torch.as_tensor(ctype, device=dev)
    V = torch.as_tensor(np.asarray(val, np.float64), device=dev)
    unk = T == UNKNOWN
    dirm = T == DIRICHLET
    ins = T == INSULATED
    shape = T.shape

    def shift(a, ax, d, fill):
        out = torch.full_like(a, fill)
        src = [slice(None)] * 3
        dst = [slice(None)] * 3
        if d > 0:
            src[ax], dst[ax] = slice(0, -1), slice(1, None)
        else:
            src[ax], dst[ax] = slice(1, None), slice(0, -1)
        out[tuple(dst)] = a[tuple(src)]
        return out

    # per face direction: conductance (neighbour not insulated; outside the grid conducts to 0), neighbour unknown
    faces = []
    diag = torch.zeros(shape, dtype=torch.float64, device=dev)
    b = torch.zeros(shape, dtype=torch.float64, device=dev) if rhs is None else torch.as_tensor(np.asarray(rhs, np.float64), device=dev).clone()
    for ax in range(3):
        for d in (-1, 1):
            nins = shift(ins, ax, d, False)
            nunk = shift(unk, ax, d, False)
            ndir = shift(dirm, ax, d, False)
            nval = shift(V * dirm, ax, d, 0.0)
            cond = ~nins
            diag += cond
            b += torch.where(cond & ndir, nval, torch.zeros((), dtype=torch.float64, device=dev))
            faces.append((ax, d, cond & nunk))
    diag = torch.where(unk, diag, torch.ones((), dtype=torch.float64, device=dev))
    b = torch.where(unk, b, torch.zeros((), dtype=torch.float64, device=dev))

    def A(x):
        y = diag * x
        for ax, d, m in faces:
            y -= torch.where(m, shift(x, ax, d, 0.0), torch.zeros((), dtype=torch.float64, device=dev))
        return torch.where(unk, y, torch.zeros((), dtype=torch.float64, device=dev))

    x = torch.zeros(shape, dtype=torch.float64, device=dev) if x0 is None else torch.where(unk, torch.as_tensor(np.asarray(x0, np.float64), device=dev), torch.zeros((), dtype=torch.float64, device=dev))
    r = b - A(x)
    z = r / diag
    p = z.clone()
    rz = float((r * z).sum())
    bn = float(torch.linalg.vector_norm(b)) or 1.0
    it = 0
    for it in range(1, maxit + 1):
        Ap = A(p)
        alpha = rz / float((p * Ap).sum())
        x += alpha * p
        r -= alpha * Ap
        if float(torch.linalg.vector_norm(r)) < tol * bn:
            break
        z = r / diag
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    out = torch.where(unk, x, torch.where(dirm, V, torch.zeros((), dtype=torch.float64, device=dev)))
    res = float(torch.linalg.vector_norm(r)) / bn
    assert np.isfinite(res) and res < 1e3 * tol, f"masked_laplace: relative residual {res:.2e} after {it} iterations"
    return out.cpu().numpy(), it


def harmonic_extension(field, fixed, tol=1e-6):
    """components of field [Z, Y, X, 3] kept on `fixed`, harmonic elsewhere, 0 outside the grid"""
    ctype = np.where(fixed, DIRICHLET, UNKNOWN).astype(np.uint8)
    out = np.empty_like(field)
    its = []
    for c in range(field.shape[-1]):
        out[..., c], it = masked_laplace(ctype, np.where(fixed, field[..., c], 0.0), x0=np.where(fixed, 0.0, field[..., c]), tol=tol)
        its.append(it)
    return out, its


def insulated_poisson_mode(src, insulated, h, tol=1e-7):
    """B = ∇Γ with ΔΓ = src (1 in the chamber) on the grid less the insulated cells (zero normal flux at their faces,
    Γ = 0 outside the grid); Γ is extended into the insulated cells from the nearest free cell so the gradient at the
    wall is tangential"""
    ctype = np.where(insulated, INSULATED, UNKNOWN).astype(np.uint8)
    G, it = masked_laplace(ctype, np.zeros(src.shape), rhs=-src * h * h, tol=tol)
    if insulated.any():
        idx = ndimage.distance_transform_edt(insulated, return_distances=False, return_indices=True)
        G = G[tuple(idx)]
    gz, gy, gx = np.gradient(G, h)
    return np.stack([gx, gy, gz], -1), it


def vessel_inclusions(X, heart, g, gj, r_min=3.5e-3, decay=40e-3):
    """inclusion mask on the grid (world coordinates X [Z, Y, X, 3]) and per inclusion cell the fading weight
    exp(−s / decay) and its tree's heart-root position. g / gj: the phantom's graph.npz / graph.json."""
    P = np.asarray(g["points"], np.float64)
    R = np.asarray(g["radius"], np.float64)
    eoff, ea, eb = np.asarray(g["edge_offsets"]), np.asarray(g["edge_a"]), np.asarray(g["edge_b"])
    npos = np.asarray(g["node_pos"], np.float64)
    ns, ne, nn = len(P), len(ea), len(npos)
    seg = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    arc = np.zeros(ns)
    elen = np.zeros(ne)
    for k in range(ne):
        s_ = np.cumsum(seg[eoff[k] : eoff[k + 1]])
        s_ -= s_[0]
        arc[eoff[k] : eoff[k + 1]] = s_
        elen[k] = s_[-1]
    roots = [int(rn) for c in gj["components"] if c["class"] in ("systemic_arterial", "pulmonary_arterial", "systemic_venous", "pulmonary_venous") for rn in c.get("roots", [])]
    Gm = coo_matrix((np.r_[elen, elen] + 1e-9, (np.r_[ea, eb], np.r_[eb, ea])), shape=(nn, nn)).tocsr()
    dist, _, src = dijkstra(Gm, indices=roots, min_only=True, return_predecessors=True)
    es = np.searchsorted(eoff, np.arange(ns), side="right") - 1
    da = dist[ea[es]] + arc
    db = dist[eb[es]] + (elen[es] - arc)
    s = np.minimum(da, db)
    root_node = np.where(da <= db, src[ea[es]], src[eb[es]])
    ok = (R >= r_min) & np.isfinite(s) & (root_node >= 0)
    lo = X[0, 0, 0] - 1e-9
    hi = X[-1, -1, -1] + 1e-9
    ok &= np.all((P > lo - R[:, None]) & (P < hi + R[:, None]), 1)
    sel = np.flatnonzero(ok)
    incl = np.zeros(X.shape[:3], bool)
    w = np.zeros(X.shape[:3])
    rootpos = np.zeros(X.shape[:3] + (3,))
    if len(sel) == 0:
        return incl, w, rootpos, {"samples": 0}
    tree = cKDTree(P[sel])
    d, j = tree.query(X.reshape(-1, 3), distance_upper_bound=float(R[sel].max()))
    hit = np.isfinite(d)
    jj = np.minimum(j, len(sel) - 1)
    inside = hit & (d < R[sel][jj])
    inside = inside.reshape(X.shape[:3]) & ~heart
    incl[inside] = True
    jsel = sel[jj.reshape(X.shape[:3])[inside]]
    w[inside] = np.exp(-s[jsel] / decay)
    rootpos[inside] = npos[root_node[jsel]]
    info = {"samples": int(len(sel)), "cells": int(inside.sum()), "roots": len(roots)}
    return incl, w, rootpos, info


# ── tethering (2026-11-28): the skeleton and the great vessels lying on it do not follow the heartbeat or the breathing
#    organs. The harmonic extension of the AV descent carried 13 mm onto the descending aorta behind the left atrium
#    while the aorta above and below stayed put — a kink (user, 2026-11-28; ph0001 stretch 0.6–1.5, bending change 23°);
#    measured 4D CTA: displacement 6.2 mm at the coronary ostia, 3.8 mm mid-ascending, the least in the descending aorta
#    (Weber et al. 2014). The breathing organs pulled the retroperitoneal aorta 5 mm caudally with ±15–20 % stretch.

def graph_path_distance(gz, names, roots=(), connectors=True):
    """per centreline point of the vessel graph (graph.npz arrays, graph.json edge names): the distance along the
    vessels from the heart — the nearest point of a `connector:` edge or a heart root node (`roots`: graph.json heart-class
    components' roots; venous roots inside their chamber have no connector since 2026-11-28); inf for a tree without one.
    connectors=False: from `roots` only (the root-hold profile: s = 0 at the valve, not also at the aorta's atlas end)"""
    import heapq
    off, ea, eb, P = gz["edge_offsets"], gz["edge_a"], gz["edge_b"], np.asarray(gz["points"], float)
    seg = [np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[off[i]:off[i + 1]], axis=0), axis=1))]) for i in range(len(names))]
    adj = {}
    for i in range(len(names)):
        L = float(seg[i][-1])
        adj.setdefault(int(ea[i]), []).append((int(eb[i]), L))
        adj.setdefault(int(eb[i]), []).append((int(ea[i]), L))
    dist = {}
    q = []
    for i, n in enumerate(names):
        if connectors and n.startswith("connector:"):  # both ends (the attachment run counts as the heart)
            for v in (int(ea[i]), int(eb[i])):
                dist[v] = 0.0
                q.append((0.0, v))
    for v in roots:
        dist[int(v)] = 0.0
        q.append((0.0, int(v)))
    heapq.heapify(q)
    while q:
        d, u = heapq.heappop(q)
        if d > dist.get(u, np.inf):
            continue
        for v, L in adj.get(u, []):
            if d + L < dist.get(v, np.inf):
                dist[v] = d + L
                heapq.heappush(q, (d + L, v))
    out = np.full(len(P), np.inf)
    for i in range(len(names)):
        da, db = dist.get(int(ea[i]), np.inf), dist.get(int(eb[i]), np.inf)
        s = seg[i]
        out[off[i]:off[i + 1]] = np.minimum(da + s, db + (s[-1] - s))
    return out


def tether_mask(bone, origin, h, points=None, radii=None, rmin=3e-3, gap=8e-3, exclude=None, pathd=None, free_path=100e-3, ramp=30e-3, below_z=None, weights=False, below_rmin=None, below_radii=None):
    """tethered cells of a grid (voxel centres origin + (i + ½) h, [Z, Y, X]): bone, and the lumens of the vessels of
    radius ≥ rmin whose wall lies within `gap` of bone (centreline points from the vessel graph) and that lie at least
    `free_path` along the vessels from the heart (pathd, graph_path_distance: the great-vessel roots — the ascending aorta
    passes 2–10 mm behind the sternum — move with the heart; the descending aorta, ≥ 140 mm along, lies on the spine);
    minus `exclude`. The vessels' hold ramps in over free_path ± `ramp` (smoothstep; a yes / no start bent the arch 10–13°);
    below `below_z` (the label field of view) the vessels need no bone (retroperitoneal: aorta, iliacs, infrarenal cava)
    — those of radius ≥ `below_rmin` (default rmin) only: the mesenteric arteries (SMA r 2.7–3.5 mm) run in the
    mesentery and move with the bowel; held, the SMA stood still for 36 mm and then jumped 5 mm within 10 mm.
    `below_radii` (per point, e.g. its edge's median radius): the radius that test uses (default the point's own).
    weights: return the hold W ∈ [0, 1] per cell (bone 1) instead of the mask"""
    T = np.asarray(bone, bool).copy()
    W = T.astype(np.float32)
    if points is not None and len(points):
        db = ndimage.distance_transform_edt(~T, sampling=h)
        P = np.asarray(points, float)
        R = np.asarray(radii, float)
        ijk = np.floor((P - np.asarray(origin, float)) / h).astype(int)  # x, y, z
        shp = np.array(T.shape[::-1])
        ok = (R >= rmin) & np.all((ijk >= 0) & (ijk < shp), axis=1)
        wv = np.ones(len(P))
        if pathd is not None:
            t_ = np.clip((np.asarray(pathd) - (free_path - ramp)) / (2 * ramp), 0, 1)
            wv = t_ * t_ * (3 - 2 * t_)
            ok &= wv > 0
        near = np.zeros(len(P), bool)
        near[ok] = db[ijk[ok, 2], ijk[ok, 1], ijk[ok, 0]] - R[ok] <= gap
        if below_z is not None:
            near |= ok & (P[:, 2] < below_z) & ((R if below_radii is None else np.asarray(below_radii, float)) >= (rmin if below_rmin is None else below_rmin))
        ok &= near
        V = np.zeros(T.shape, np.float32)
        for p, r, w in zip(P[ok], R[ok], wv[ok]):
            lo = np.maximum(np.floor((p - r - origin) / h).astype(int), 0)
            hi = np.minimum(np.floor((p + r - origin) / h).astype(int) + 1, shp)
            if np.any(hi <= lo):
                continue
            cs = [origin[a] + (np.arange(lo[a], hi[a]) + 0.5) * h - p[a] for a in range(3)]
            d2 = cs[2][:, None, None] ** 2 + cs[1][None, :, None] ** 2 + cs[0][None, None, :] ** 2
            sub = V[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
            np.maximum(sub, np.where(d2 <= r * r, np.float32(w), np.float32(0)), out=sub)
        W = np.maximum(W, V)
        T |= V > 0
    if exclude is not None:
        T &= ~np.asarray(exclude, bool)
        W = np.where(T, W, 0.0).astype(np.float32)
    return W if weights else T


def tether(field, T, keep, h, band=15e-3, tol=1e-6, target=None):
    """field [Z, Y, X, 3] → field − C: C = W field on the tethered cells (T: a mask, W = 1, or the hold W ∈ [0, 1] of
    tether_mask(weights=True)), 0 on `keep` and beyond `band` of them, harmonic in between — the tethered tissue keeps
    (1 − W) of its motion, `keep` (the heart) all, the transition is a smooth shear layer at most `band` thick.
    target [Z, Y, X, 3] (optional): C = W (field − target), i.e. the tethered cells move to (1 − W) field + W target
    (a prescribed motion instead of rest). Returns (field', CG iterations)."""
    Wt = np.asarray(T, np.float32)
    T = Wt > 0
    d = ndimage.distance_transform_edt(~T, sampling=h)
    keep = np.zeros(T.shape, bool) if keep is None else np.asarray(keep, bool)
    unk = (d <= band) & ~T & ~keep
    ctype = np.where(unk, UNKNOWN, DIRICHLET).astype(np.uint8)
    out = np.empty_like(field)
    its = 0
    for c in range(3):
        fc = field[..., c] if target is None else field[..., c] - target[..., c]
        if not np.any(fc[T]):  # nothing to hold (a component a basis does not have): the CG has b = 0
            out[..., c] = field[..., c]
            continue
        C, it = masked_laplace(ctype, np.where(T, Wt * fc, 0.0), tol=tol)
        out[..., c] = field[..., c] - C
        its += it
    return out, its


HEART_CLASSES = ("systemic_arterial", "pulmonary_arterial", "systemic_venous", "pulmonary_venous")


def heart_roots(gj):
    """root nodes of the heart-class components (graph.json dict)"""
    return [int(r) for c in gj["components"] if c["class"] in HEART_CLASSES for r in c.get("roots", [c.get("root_node")]) if r is not None]

