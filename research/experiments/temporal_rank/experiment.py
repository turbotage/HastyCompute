"""One temporal-rank experiment on a set of encode images X [7, V, T]: spectra, truncation K at the error levels, and the
full metric table for the SVD bases (shared over encodes and per encode) and the generic bases (periodic cubic B-splines,
Fourier) at equal K."""
import json
import time

import numpy as np

from common import LEVELS, OUT, cardiac_basis, decode, fourier_basis, k_at, orth
from rank_lib import GROUPS, eval_basis, flows

KS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48]
BSPL = [8, 12, 16, 20, 24, 32, 48]


class Context:
    """masks fixed per geometry: signal voxels, lumen voxels, the voxels decoded for velocity / flow (lumen + named
    vessels), vessel map"""

    def __init__(self, X0, b2, vmap, sig_frac=0.1, cat=None):
        a = np.abs(X0[0]).mean(1)
        self.sig = a > sig_frac * np.percentile(a, 99)
        self.b2 = b2
        self.bl = b2.mean(1) >= 0.5
        ev = np.zeros(X0.shape[1], bool)
        for per in vmap.values():
            for ci, w, t in per:
                ev[ci] = True
        self.idxB = np.flatnonzero(self.bl | ev | (b2.max(1) >= 0.5))
        self.vmap = vmap
        self.catB = None
        if cat is None:
            p = __import__("common").CACHE / "context.pkl"
            if p.exists():
                import pickle
                cat = pickle.loads(p.read_bytes())[1]
        if cat is not None:  # time-majority category of the decoded voxels: 1 chamber, 2 vessel
            cb = cat[self.idxB]
            n1, n2 = (cb == 1).sum(1), (cb == 2).sum(1)
            self.catB = np.where((n1 == 0) & (n2 == 0), 0, np.where(n1 >= n2, 1, 2))


def truth_refs(X, ctx):
    XB = np.stack([X[e][ctx.idxB] for e in range(X.shape[0])], -1)
    vt = decode(XB)
    full = np.zeros((X.shape[1], X.shape[2], 3), np.float32)
    full[ctx.idxB] = vt
    return vt, flows(full, ctx.vmap, X.shape[2])


def grams(X, w=None):
    """per-encode Grams X_e^H X_e [7, T, T] (voxel weights w)"""
    G = []
    for e in range(X.shape[0]):
        Xe = X[e] if w is None else X[e] * np.sqrt(w)[:, None]
        G.append((Xe.conj().T @ Xe).astype(np.complex128))
    return np.stack(G)


def eig_desc(G):
    w, U = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    return np.maximum(w[o], 0), U[:, o]


def analyse(X, ctx, label, phase=None, Ks=KS, bspl=BSPL, gram_w=None, extra_bases=None, quick=False):
    t0 = time.time()
    T = X.shape[2]
    phase = np.arange(T) / T if phase is None else phase
    vt, Qt = truth_refs(X, ctx)
    G = grams(X, gram_w)
    Gs = G.sum(0)
    ws, Us = eig_desc(Gs)
    res = {"label": label, "T": T, "levels": LEVELS}
    res["spectrum_shared"] = (ws / ws[0]).tolist()
    res["K_shared"] = k_at(ws, LEVELS)
    per = [eig_desc(G[e]) for e in range(X.shape[0])]
    res["K_per_encode"] = [k_at(w, LEVELS) for w, _ in per]
    res["spectrum_per_encode"] = [(w / w[0]).tolist() for w, _ in per]
    # blood-only spectrum (the lumen voxels' Gram)
    Gb = grams(X[:, ctx.bl])
    wb, Ub = eig_desc(Gb.sum(0))
    res["K_blood_shared"] = k_at(wb, LEVELS)
    res["spectrum_blood"] = (wb / wb[0]).tolist()
    Qmax = {g: float(np.abs(q).max()) for g, q in Qt.items()}
    res["Q_peak"] = Qmax
    rows = []

    def ev(name, Phis, k):
        r = eval_basis(X, Phis, ctx.sig, ctx.bl, ctx.b2, ctx.vmap, vt, Qt, ctx.idxB, ctx.catB)
        r.update(basis=name, K=k)
        rows.append(r)

    for k in Ks:
        if k > T:
            continue
        ev("svd_shared", [Us[:, :k]] * X.shape[0], k)
        if not quick:
            ev("svd_per_encode", [U[:, :k] for _, U in per], k)
            ev("fourier", [orth(fourier_basis(phase, k)).astype(np.complex128)] * X.shape[0], k)
    for m in bspl:
        if m > T:
            continue
        ev("bspline3", [orth(cardiac_basis(phase, m, 3)).astype(np.complex128)] * X.shape[0], m)
    for name, (k, Phi) in (extra_bases or {}).items():
        ev(name, [Phi] * X.shape[0], k)
    res["rows"] = rows
    res["seconds"] = time.time() - t0
    res["_Us"] = Us  # not serialised
    return res


def save(res, name):
    r = {k: v for k, v in res.items() if not k.startswith("_")}
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1))


def table(res, keys=("rel_sig", "rel_blood", "vox_p99", "dv_mean", "dv_p99", "dv_mean_ch", "dv_mean_ves", "dv_p99_ves"), groups=("Asc. aorta", "Pulm. trunk + LPA/RPA", "SVC", "RCA", "LAD", "LCx", "Cardiac veins + CS")):
    lines = []
    hdr = f"{'basis':16s} {'K':>3s} " + " ".join(f"{k:>9s}" for k in keys) + " | " + " ".join(f"{g[:8]:>8s}" for g in groups)
    lines.append(hdr)
    for r in res["rows"]:
        q = r.get("Q", {})
        lines.append(f"{r['basis']:16s} {r['K']:3d} " + " ".join(f"{r.get(k, float('nan')):9.3g}" for k in keys) + " | " + " ".join(f"{q.get(g, (float('nan'),))[0]:8.3g}" for g in groups))
    return "\n".join(lines)
