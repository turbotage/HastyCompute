"""Shaded 3-D views of the vessel lumen (the capsule model of pack_vessels / phantomflow) in a box, one colour per vessel.

    rustlib/.venv/bin/python -I jobs/tools/vessel_render.py <graph_dir> <out.png> --center x,y,z --half hx,hy,hz
        [--vox 0.5] [--views ant,right,post,left,ant-right,ant-left] [--states 0:0,0:1] [--motion <phantom_dir>]
        [--title text] [--top 12]

Coordinates in mm (atlas world: x patient left, y posterior, z superior). The lumen is the union of truncated cones
between consecutive centreline samples; each voxel is owned by the vessel whose surface is deepest (as phantomflow).
Views are orthographic first-hit renders of that signed distance (sub-voxel hit depth, normals from the depth map,
Lambert + rim shading). --states t:r (cardiac time in s, breathing r in [0, 1]) deforms the centrelines with the
phantom's motion fields (radii kept); one row per state. No VTK / X server needed."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
A = sys.argv[1:]


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


G, OUT = Path(A[0]), Path(A[1])
C = np.array([float(v) for v in opt("--center").split(",")]) * 1e-3
H = np.array([float(v) for v in opt("--half").split(",")]) * 1e-3
VOX = float(opt("--vox", 0.5)) * 1e-3
VIEWS = opt("--views", "ant,right,post,left,ant-right,ant-left").split(",")
STATES = [tuple(float(x) for x in s.split(":")) for s in opt("--states", "0:0").split(",")]
TOP = int(opt("--top", 12))
RMIN = float(opt("--rmin-mm", 0)) * 1e-3  # edges of smaller mean radius are left out
ORGANS = [o for o in opt("--organs", "").split(",") if o]  # atlas labels drawn as pale surfaces (e.g. blood_RA,liver)
ONLY = opt("--only")  # regex: only vessels whose name matches
z = np.load(G / "graph.npz")
gj = json.loads((G / "graph.json").read_text())
names = gj["edge_names"]
base = [n.split(": ", 1)[1] if n.startswith(("connector: ", "join: ", "bridge: ")) else n for n in names]
off, P0, R = z["edge_offsets"], z["points"].astype(np.float64), z["radius"].astype(np.float64)
E = len(names)
eid = np.repeat(np.arange(E), np.diff(off))
lo, hi = C - H, C + H
import re  # noqa: E402

rmean = np.array([R[off[k]:off[k + 1]].mean() if off[k + 1] > off[k] else 0 for k in range(E)])
keep_e = (rmean >= RMIN) & np.array([bool(re.search(ONLY, nm)) if ONLY else True for nm in base])
shape = tuple(int(v) for v in np.ceil(2 * H / VOX))
mot = None
if opt("--motion"):
    sys.path.insert(0, str(REPO / "rustlib/tools/phantom"))
    from motion import Motion  # noqa: E402

    md = Path(opt("--motion")) / "motion"
    mot = Motion(md / "cardiac_motion.hv", md / "resp_motion.hv")


def rasterise(P):
    """signed distance (m) and owning edge id on the box grid"""
    sdf = np.full(shape, 4 * VOX, np.float32)
    own = np.full(shape, -1, np.int32)
    near = np.all((P > lo - R[:, None] - 2 * VOX) & (P < hi + R[:, None] + 2 * VOX), axis=1)
    near &= keep_e[eid]
    for e in np.unique(eid[near]):
        s0, s1 = off[e], off[e + 1]
        for k in range(s0, s1 - 1):
            if not (near[k] or near[k + 1]):
                continue
            a, b, ra, rb = P[k], P[k + 1], R[k], R[k + 1]
            rm = max(ra, rb) + 2 * VOX
            i0 = np.maximum(np.floor((np.minimum(a, b) - rm - lo) / VOX).astype(int), 0)
            i1 = np.minimum(np.ceil((np.maximum(a, b) + rm - lo) / VOX).astype(int), shape)
            if np.any(i1 <= i0):
                continue
            ax = [lo[d] + (np.arange(i0[d], i1[d]) + 0.5) * VOX for d in range(3)]
            X = np.stack(np.meshgrid(*ax, indexing="ij"), -1)
            ab = b - a
            L2 = ab @ ab
            t = np.clip(((X - a) @ ab) / max(L2, 1e-18), 0, 1) if L2 > 0 else np.zeros(X.shape[:3])
            d = np.linalg.norm(X - (a + t[..., None] * ab), axis=-1) - (ra + t * (rb - ra))
            sl = tuple(slice(i0[q], i1[q]) for q in range(3))
            cur = sdf[sl]
            m = d < cur
            cur[m] = d[m]
            own[sl][m] = e
    for oi, on in enumerate(ORGANS):
        f = organ_field(on)
        m = f < sdf
        sdf[m] = f[m]
        own[m] = -10 - oi
    return sdf, own


_LAB = {}


def organ_field(name):
    """label mask of the phantom's atlas (1 mm) on the box grid as a pseudo distance: (0.5 − mask) · 2 vox"""
    if not _LAB:
        sys.path.insert(0, str(REPO / "python/lib"))
        from hastycompute.store import load_any  # noqa: E402

        lab = load_any(Path(opt("--labels")))
        _LAB.update(lab=np.asarray(lab["labels"]), names=list(lab["tissues"]), o=np.array(lab["origin_m"]), h=float(lab["voxel_m"]))
    msk = (_LAB["lab"] == _LAB["names"].index(name)).astype(np.float32)
    ax = [lo[d] + (np.arange(shape[d]) + 0.5) * VOX for d in range(3)]
    X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    ijk = ((X - _LAB["o"]) / _LAB["h"] - 0.5)[:, ::-1].T  # labels are [z, y, x], voxel centres at o + (i + ½) h
    v = ndimage.map_coordinates(ndimage.gaussian_filter(msk, 0.6), ijk, order=1, cval=0).reshape(shape)
    return ((0.5 - v) * 2 * _LAB["h"]).astype(np.float32)


# camera: view direction (camera looks along v), up = +z unless looking along z
VD = {"ant": (0, 1, 0), "post": (0, -1, 0), "right": (1, 0, 0), "left": (-1, 0, 0), "sup": (0, 0, -1), "inf": (0, 0, 1),
      "ant-right": (1, 1, 0), "ant-left": (-1, 1, 0), "post-right": (1, -1, 0), "post-left": (-1, -1, 0)}


def render(sdf, own, view):
    v = np.array(VD[view], float)
    v /= np.linalg.norm(v)
    up = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([0, -1.0, 0])
    u = np.cross(v, up)  # camera right (anterior view: patient right on the viewer's left)
    u /= np.linalg.norm(u)
    w = np.cross(u, v)
    ext = np.linalg.norm(H) * 1.0
    n = int(np.ceil(2 * ext / VOX))
    s = (np.arange(n) + 0.5) * VOX - ext
    # rays: pixel (i along w downward, j along u), depth along v
    U, W = np.meshgrid(s, -s, indexing="xy")
    img_d = np.full(U.shape, np.inf)
    img_o = np.full(U.shape, -1, np.int64)
    prev = None
    C0 = C
    for kd, dd in enumerate(s):
        X = C0[None, None, :] + U[..., None] * u + W[..., None] * w + dd * v
        ijk = ((X - lo) / VOX - 0.5).reshape(-1, 3).T
        val = ndimage.map_coordinates(sdf, ijk, order=1, cval=4 * VOX).reshape(U.shape)
        if prev is not None:
            hit = (val < 0) & ~np.isfinite(img_d)
            if hit.any():
                f = prev[hit] / np.maximum(prev[hit] - val[hit], 1e-12)
                img_d[hit] = dd - VOX + f * VOX
                oo = ndimage.map_coordinates(own.astype(np.float64), ijk, order=0, cval=-1).reshape(U.shape)
                img_o[hit] = oo[hit].astype(np.int64)
        prev = val
    fin = np.isfinite(img_d)
    D = np.where(fin, img_d, np.nan)
    Df = np.where(fin, img_d, ndimage.grey_dilation(np.where(fin, img_d, -np.inf), size=3))
    Df = np.where(np.isfinite(Df), Df, np.nanmax(D) if fin.any() else 0)
    gy, gx = np.gradient(ndimage.gaussian_filter(Df, 0.7), VOX)
    # camera frame (u right, w up, toward the camera): rows run down w, so dD/dw = −gy
    nrm = np.stack([-gx, gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    L1 = np.array([-0.4, 0.5, 0.77])
    L1 /= np.linalg.norm(L1)
    lam = np.clip(nrm @ L1, 0, 1)
    spec = np.clip(nrm @ ((L1 + np.array([0, 0, 1.0])) / np.linalg.norm(L1 + np.array([0, 0, 1.0]))), 0, 1) ** 24
    # depth cue: nearer = brighter
    dc = 1 - 0.25 * (D - np.nanmin(D)) / max(np.nanmax(D) - np.nanmin(D), 1e-9) if fin.any() else D
    shade = np.where(fin, (0.32 + 0.68 * lam) * dc + 0.25 * spec, np.nan)
    return shade, img_o, fin, ext


rows = []
for (t, r) in STATES:
    P = P0 if mot is None else mot.position(P0, t, r)
    rows.append(((t, r), rasterise(P)))
# colours: vessels by visible area over all panels
panels = [[render(sdf, own, vw) for vw in VIEWS] for _, (sdf, own) in rows]
cnt = {}
for row in panels:
    for shade, o, fin, _ in row:
        ids, c = np.unique(o[fin], return_counts=True)
        for i, k in zip(ids, c):
            if i >= 0:
                cnt[base[i]] = cnt.get(base[i], 0) + k
order = sorted(cnt, key=cnt.get, reverse=True)
pal = plt.get_cmap("tab20").colors
OCOL = [(0.95, 0.85, 0.75), (0.80, 0.90, 0.95), (0.90, 0.80, 0.90), (0.85, 0.95, 0.80), (0.95, 0.95, 0.75)]
col = {nm: pal[(2 * k) % 20 if k < 10 else (2 * (k - 10) + 1) % 20] for k, nm in enumerate(order[:TOP])}
asp = H[2] / max(H[0], H[1])
fig, axs = plt.subplots(len(rows), len(VIEWS), figsize=(4.6 * len(VIEWS), 4.6 * asp * len(rows) + 1.4), squeeze=False)
for ri, ((t, r), _) in enumerate(rows):
    for vi, vw in enumerate(VIEWS):
        shade, o, fin, ext = panels[ri][vi]
        rgb = np.ones(shade.shape + (3,))
        cc = np.array([col.get(base[i], (0.62, 0.62, 0.62)) if i >= 0 else OCOL[(-i - 10) % len(OCOL)] for i in o[fin]])
        rgb[fin] = np.clip(cc * shade[fin, None], 0, 1)
        ax = axs[ri, vi]
        ax.imshow(np.clip(rgb, 0, 1), extent=(-ext * 1e3, ext * 1e3, -ext * 1e3, ext * 1e3), interpolation="bilinear")
        hz = max(H[0], H[1]) * 1e3
        ax.set_xlim(-hz, hz)
        ax.set_ylim(-H[2] * 1e3, H[2] * 1e3)
        ax.set_title(f"{vw}" + (f"  t {t:.2f} s r {r:.1f}" if mot is not None else ""), fontsize=9)
        ax.tick_params(labelsize=7)
from matplotlib.patches import Patch  # noqa: E402

fig.legend(handles=[Patch(color=col[nm], label=nm[:40]) for nm in order[:TOP]] + [Patch(color=(0.62, 0.62, 0.62), label="other")] + [Patch(color=OCOL[i % len(OCOL)], label=f"atlas {o}") for i, o in enumerate(ORGANS)],
           loc="lower center", ncol=min(5, TOP + 1), fontsize=8, frameon=False)
fig.suptitle(opt("--title", f"{G}: lumen around {np.round(C * 1e3).astype(int).tolist()} mm"), fontsize=10, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=105)
print(OUT, {nm: int(cnt[nm]) for nm in order[:TOP]})
