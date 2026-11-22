"""Signal construction and error metrics of the temporal-rank study (research/temporal_rank.md)."""
import json

import numpy as np

from common import ABOX, CACHE, CBOX, NF, PH, decode, encode_matrix, read_frame, BLOOD_LABELS, CHAMBERS, VESSEL_BLOOD

ORG = np.array([-0.13, -0.125, 1.15])  # simblood fine grid origin (x, y, z), 1 mm
TISSUES = None


def tissues():
    global TISSUES
    if TISSUES is None:
        from hastycompute import value_format as vf
        TISSUES = list(vf.open_lazy(PH / "simblood_lib/truth4d.hv")["tissues"])
    return TISSUES


def abox_in_cache():
    return tuple(slice(a[0] - c[0], a[1] - c[0]) for a, c in zip(ABOX, CBOX))


class Fields:
    """the fine fields over the analysis box, all 100 frames, in RAM (f16 / u8, ~2.8 GB)"""

    def __init__(self, frames=None):
        sl = abox_in_cache()
        fr = slice(None) if frames is None else frames
        self.m = np.ascontiguousarray(np.load(CACHE / "m.npy", mmap_mode="r")[(fr,) + sl])
        self.v = np.ascontiguousarray(np.load(CACHE / "v.npy", mmap_mode="r")[(fr, slice(None)) + sl])
        self.lab = np.ascontiguousarray(np.load(CACHE / "lab.npy", mmap_mode="r")[(fr,) + sl])
        self.blood = np.ascontiguousarray(np.load(CACHE / "blood.npy", mmap_mode="r")[(fr,) + sl])
        self.nf = self.m.shape[0]
        self.fs = self.m.shape[1:]
        self.cs = tuple(s // 2 for s in self.fs)
        self.nv = int(np.prod(self.cs))
        # fine index (z, y, x) of the box corner on the full simblood grid
        self.f0 = np.array([a[0] for a in ABOX])


def box2(a):
    """2 x 2 x 2 mean over the last three axes"""
    s = a.shape[:-3]
    z, y, x = a.shape[-3:]
    return a.reshape(s + (z // 2, 2, y // 2, 2, x // 2, 2)).mean(axis=(-5, -3, -1))


def build_X(F, vel=None, mag=None, K=None, frames=None):
    """complex encode images on the 2 mm voxels: X [7, V, T] complex64, x_e = box2(m exp(-i K_e . v)); `vel(f)` /
    `mag(f)` override the fine velocity [3, z, y, x] / magnitude of frame f (regime perturbations)"""
    K = encode_matrix() if K is None else K
    fr = range(F.nf) if frames is None else frames
    X = np.empty((K.shape[0], F.nv, len(fr)), np.complex64)
    for j, f in enumerate(fr):
        v = F.v[f].astype(np.float32) if vel is None else vel(f)
        m = F.m[f].astype(np.float32) if mag is None else mag(f)
        for e in range(K.shape[0]):
            ph = K[e, 0] * v[0] + K[e, 1] * v[1] + K[e, 2] * v[2]
            X[e, :, j] = box2(m * np.exp(-1j * ph).astype(np.complex64)).reshape(-1)
    return X


def coarse_masks(F):
    """per 2 mm voxel and frame: blood fraction b2 [V, T]; label category of the blood-weighted majority: 1 chamber,
    2 vessel blood, 0 other; time-mean magnitude"""
    tn = tissues()
    ch = np.array([n in CHAMBERS for n in tn])
    vb = np.array([n in VESSEL_BLOOD for n in tn])
    b2 = np.empty((F.nv, F.nf), np.float32)
    cat = np.zeros((F.nv, F.nf), np.uint8)
    for f in range(F.nf):
        b = F.blood[f].astype(np.float32) / 255
        lab = F.lab[f]
        b2[:, f] = box2(b).reshape(-1)
        fc = box2(b * ch[lab]).reshape(-1)
        fv = box2(b * vb[lab]).reshape(-1)
        cat[:, f] = np.where(fc >= fv, 1, 2) * (b2[:, f] > 0.05)
    return b2, cat


# ---------------------------------------------------------------- vessels (centrelines of frames_geom)
class Vessels:
    def __init__(self):
        from hastycompute import value_format as vf
        self.G = vf.open_lazy(PH / "simblood_lib/frames_geom.hv")
        V = vf.open_lazy(PH / "vessels.hv")
        self.eo = np.asarray(V["edge_offsets"][:])
        self.names = json.loads((PH / "graph.json").read_text())["edge_names"]
        n = int(self.eo[-1])
        self.eid = (np.searchsorted(self.eo, np.arange(n), side="right") - 1).astype(np.int32)
        self.q = np.load(PH / "flow/network_flow.npz")["q"]  # [P, 26] flow harmonics (m^3/s)
        self.omega = np.load(PH / "flow/network_flow.npz")["omega"]
        self.period = float(np.load(PH / "flow/network_flow.npz")["period"])

    def frame(self, f):
        p = read_frame(self.G["points"], f)
        r = read_frame(self.G["radius"], f)
        w = read_frame(self.G["wvel"], f)
        return p, r, w

    def tangent(self, p):
        """unit tangent per point from its edge neighbours"""
        n = len(p)
        i = np.arange(n)
        a = np.where((i - 1 >= self.eo[self.eid]), i - 1, i)
        b = np.where((i + 1 < self.eo[self.eid + 1]), i + 1, i)
        t = p[b] - p[a]
        return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def assign_fine(F, Ve, f, margin=1.5e-3, extra_labels=()):
    """fine vessel-blood voxels of frame f -> nearest centreline point within radius + margin: (flat fine idx, point
    idx, distance, radius)"""
    from scipy.spatial import cKDTree
    tn = tissues()
    vb = np.array([n in VESSEL_BLOOD or n in extra_labels for n in tn])
    lab, bl = F.lab[f], F.blood[f]
    sel = np.flatnonzero((vb[lab] & (bl > 5)).reshape(-1))
    z, y, x = np.unravel_index(sel, F.fs)
    pos = ORG + (np.stack([x, y, z], 1) + F.f0[::-1]) * 1e-3
    p, r, _ = Ve.frame(f)
    lo, hi = pos.min(0) - 0.03, pos.max(0) + 0.03
    inb = np.flatnonzero(np.all((p >= lo) & (p <= hi), 1))
    tree = cKDTree(p[inb])
    d, j = tree.query(pos, k=1)
    pi = inb[j]
    ok = d <= r[pi] + margin
    return sel[ok], pi[ok], d[ok], r[pi[ok]]


def coarse_vessel_map(F, Ve, groups, frames=None):
    """per frame: for each named group, the 2 mm voxels (flat idx), their blood weight and blood-weighted mean tangent
    (for the flow integral Q_n(t) = sum_v w_v t_v . v_v): {group: [(idx, w, t[k,3]) per frame]}"""
    fr = range(F.nf) if frames is None else frames
    out = {g: [] for g in groups}
    gname = {}
    for g, names in groups.items():
        for n in names:
            gname[n] = g
    cs = F.cs
    for f in fr:
        sel, pi, d, r = assign_fine(F, Ve, f)
        p, _, _ = Ve.frame(f)
        tg = Ve.tangent(p)
        nm = np.array([gname.get(Ve.names[e], "") for e in Ve.eid[pi]])
        b = F.blood[f].reshape(-1)[sel].astype(np.float32) / 255
        z, y, x = np.unravel_index(sel, F.fs)
        cidx = np.ravel_multi_index((z // 2, y // 2, x // 2), cs)
        for g in groups:
            k = nm == g
            if not k.any():
                out[g].append((np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 3))))
                continue
            ci, inv = np.unique(cidx[k], return_inverse=True)
            w = np.bincount(inv, b[k]) / 8.0
            t = np.stack([np.bincount(inv, b[k] * tg[pi[k], a]) for a in range(3)], 1)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
            out[g].append((ci, w, t))
    return out


def flows(vdec, vmap, T):
    """Q_g(t) = sum_v w_v t_v . v_dec(v, t) (m/s x voxel units) from decoded velocity [V, T, 3] (full V) or a dict
    {group: [T] arrays}"""
    Q = {}
    for g, per in vmap.items():
        Q[g] = np.array([(w * np.einsum("ka,ka->k", t, vdec[ci, j])).sum() if len(ci) else 0.0 for j, (ci, w, t) in enumerate(per[:T])])
    return Q


# ---------------------------------------------------------------- bases and metrics
def project(X, Phi):
    """X [V, T] onto the column span of Phi [T, K] (orthonormal): X Phi Phi^H"""
    return (X @ Phi) @ Phi.conj().T


def svd_basis(G, k):
    w, U = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    return U[:, o[:k]], np.maximum(w[o], 0)


def eval_basis(X, Phis, sig, bl_mask, b2, vmap=None, vtrue=None, Qtrue=None, idxB=None, catB=None):
    """errors of the projection of every encode e onto Phis[e] ([T, K] orthonormal):
    rel_sig: ||X - PX||_F / ||X||_F over the signal voxels (all encodes); rel_blood: same over the lumen voxels;
    vox_p99 / vox_max: per-voxel relative error over time (all encodes) in the lumen; dv_*: |v_dec(PX) - v_dec(X)|
    (cm/s) over the lumen voxel-frames with b2 >= 0.5; Q: per named group max_t |dQ| / max_t |Q| and |d mean Q| / max|Q|"""
    ne = X.shape[0]
    num_s = den_s = num_b = den_b = 0.0
    vnum = np.zeros(int(bl_mask.sum()))
    vden = np.zeros(int(bl_mask.sum()))
    XB = []
    for e in range(ne):
        Xe = X[e]
        P = project(Xe, Phis[e]).astype(np.complex64)
        R = Xe - P
        r2 = (np.abs(R) ** 2).sum(1)
        x2 = (np.abs(Xe) ** 2).sum(1)
        num_s += r2[sig].sum()
        den_s += x2[sig].sum()
        num_b += r2[bl_mask].sum()
        den_b += x2[bl_mask].sum()
        vnum += r2[bl_mask]
        vden += x2[bl_mask]
        XB.append(P[idxB])
    out = {"rel_sig": float(np.sqrt(num_s / den_s)), "rel_blood": float(np.sqrt(num_b / den_b))}
    rv = np.sqrt(vnum / np.maximum(vden, 1e-30))
    out["vox_p99"], out["vox_max"] = float(np.percentile(rv, 99)), float(rv.max())
    XB = np.stack(XB, -1)  # [nB, T, 7]
    vd = decode(XB)
    if vtrue is not None:
        okm = b2[idxB] >= 0.5
        dv = np.linalg.norm(vd - vtrue, axis=-1)[okm] * 100
        out["dv_mean"], out["dv_p99"], out["dv_max"] = float(dv.mean()), float(np.percentile(dv, 99)), float(dv.max())
        out["dv_gt10"] = float((dv > 10).mean())
        if catB is not None:  # split: chamber blood (LBM) vs vessel blood
            dva = np.linalg.norm(vd - vtrue, axis=-1) * 100
            for c, nm in [(1, "ch"), (2, "ves")]:
                mk = okm & (catB[:, None] == c)
                if mk.any():
                    out[f"dv_mean_{nm}"], out[f"dv_p99_{nm}"] = float(dva[mk].mean()), float(np.percentile(dva[mk], 99))
    if vmap is not None and Qtrue is not None:
        full = np.zeros((X.shape[1], X.shape[2], 3), np.float32)
        full[idxB] = vd
        Q = flows(full, vmap, X.shape[2])
        out["Q"] = {}
        for g in Q:
            mx = np.abs(Qtrue[g]).max()
            if mx > 0:
                out["Q"][g] = (float(np.abs(Q[g] - Qtrue[g]).max() / mx), float(abs(Q[g].mean() - Qtrue[g].mean()) / mx))
    return out


GROUPS = {
    "Asc. aorta": ["Ascending aorta", "connector: Ascending aorta"],
    "Aortic arch": ["Aortic arch"],
    "Desc. aorta": ["Thoracic aorta"],
    "Pulm. trunk + LPA/RPA": ["Bifurcation of pulmonary trunk", "connector: Bifurcation of pulmonary trunk", "Left pulmonary artery", "Right pulmonary artery"],
    "SVC": ["Superior vena cava", "connector: Superior vena cava"],
    "IVC + hepatic": ["Inferior vena cava (thoracic part)", "Hepatic veins", "bridge: Hepatic veins"],
    "Pulm. veins": ["Right superior pulmonary vein", "Left superior pulmonary vein", "Right inferior pulmonary vein", "Left inferior pulmonary vein",
                    "connector: Right superior pulmonary vein", "connector: Left superior pulmonary vein", "connector: Right inferior pulmonary vein", "connector: Left inferior pulmonary vein"],
    "RCA": ["Right coronary artery", "bridge: Right coronary artery", "Right inferolateral branch of right coronary artery"],
    "LAD": ["Anterior interventricular artery", "Left coronary artery"],
    "LCx": ["Circumflex artery of heart"],
    "Cardiac veins + CS": ["Great cardiac vein", "Middle cardiac vein", "Coronary sinus", "bridge: Coronary sinus"],
}
