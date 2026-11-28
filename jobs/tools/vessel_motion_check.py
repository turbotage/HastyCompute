"""Vessel kinematics check: the phantom's cardiac × respiratory motion along the great vessels' centrelines.

    rustlib/.venv/bin/python -I jobs/tools/vessel_motion_check.py <phantom_dir> <out_dir> [--samples 16]

Per vessel path (the aorta from its root connector to the iliac bifurcation by graph search; the named great vessels
edge by edge) and per state (cardiac time × respiratory r ∈ {0, 1}): the displacement along the arc length, the axial
stretch of 2 mm segments (deformed / reference length) and the kink — the change of the bending angle over ±5 mm
(angle between the tangents 5 mm before and after a point, deformed minus reference). A real vessel bends smoothly:
|Δθ| of a few degrees, stretch within a few %. Writes figures and a JSON summary."""
import heapq
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rustlib/tools/phantom"))
from motion import Motion  # noqa: E402

A = sys.argv[1:]


def opt(n, d):
    return A[A.index(n) + 1] if n in A else d


P, OUT = Path(A[0]), Path(A[1])
OUT.mkdir(parents=True, exist_ok=True)
NS = int(opt("--samples", 16))
g = json.loads((P / "graph.json").read_text())
z = np.load(P / "graph.npz")
names, off, pts, ea, eb = g["edge_names"], z["edge_offsets"], z["points"], z["edge_a"], z["edge_b"]
m = Motion(P / "motion/cardiac_motion.hv", P / "motion/resp_motion.hv")


def edge_pts(i, frm=None):
    p = pts[off[i]:off[i + 1]]
    return p if frm is None or ea[i] == frm else p[::-1]


def aorta_path():
    """node path root → bifurcation over the aorta-named edges (Dijkstra on length)"""
    # every edge of mean radius ≥ 6 mm (the arch carries other names); aorta-named edges preferred (length × 1, others × 3)
    Rm = z["radius"]
    ids = [i for i in range(len(names)) if Rm[off[i]:off[i + 1]].mean() >= 6e-3]
    adj = {}
    for i in ids:
        L = float(np.linalg.norm(np.diff(pts[off[i]:off[i + 1]], axis=0), axis=1).sum()) * (1.0 if "aorta" in names[i].lower() else 3.0)
        adj.setdefault(int(ea[i]), []).append((int(eb[i]), i, L))
        adj.setdefault(int(eb[i]), []).append((int(ea[i]), i, L))
    # root: the end of "connector: Ascending aorta" with the lowest degree; target: the lowest z node
    conn = [i for i in ids if names[i] == "connector: Ascending aorta"][0]
    deg = {k: len(v) for k, v in adj.items()}
    root = int(ea[conn]) if deg.get(int(ea[conn]), 0) <= deg.get(int(eb[conn]), 0) else int(eb[conn])
    aor = [i for i in ids if "abdominal aorta" in names[i].lower()]
    zmin = {n: min(pts[off[i]:off[i + 1]][:, 2].min() for (_, i, _) in adj[n] if i in aor) for n in adj if any(i in aor for (_, i, _) in adj[n])}
    tgt = min(zmin, key=zmin.get)
    dist, prev, q = {root: 0.0}, {}, [(0.0, root)]
    while q:
        d, u = heapq.heappop(q)
        if u == tgt:
            break
        for v, i, L in adj[u]:
            if d + L < dist.get(v, 1e9):
                dist[v], prev[v] = d + L, (u, i)
                heapq.heappush(q, (d + L, v))
    seq, u = [], tgt
    while u != root:
        pu, i = prev[u]
        seq.append((pu, i))
        u = pu
    seq.reverse()
    print("aorta path: " + " → ".join(dict.fromkeys(names[i] for _, i in seq)))
    out = [edge_pts(i, frm) for frm, i in seq]
    return np.vstack([o if k == 0 else o[1:] for k, o in enumerate(out)])


def resample(p, ds=1e-3):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    keep = np.concatenate([[True], np.diff(s) > 1e-9])
    p, s = p[keep], s[keep]
    sn = np.arange(0, s[-1], ds)
    return np.stack([np.interp(sn, s, p[:, a]) for a in range(3)], -1), sn


def metrics(X, x, w=5):
    """X, x [n, 3] reference / deformed at 1 mm spacing: stretch of 2 mm segments, kink (deg) over ±w mm"""
    st = np.linalg.norm(x[2:] - x[:-2], axis=1) / np.linalg.norm(X[2:] - X[:-2], axis=1)

    def bend(p):
        a = p[w:-w] - p[:-2 * w]
        b = p[2 * w:] - p[w:-w]
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))

    return st, bend(x) - bend(X)


def chain_path(pat):
    """the edges whose name contains `pat`, chained end to end from the lowest end (a simple path through them)"""
    ids = [i for i in range(len(names)) if pat in names[i].lower()]
    adj = {}
    for i in ids:
        adj.setdefault(int(ea[i]), []).append((int(eb[i]), i))
        adj.setdefault(int(eb[i]), []).append((int(ea[i]), i))
    ends = [n for n, v in adj.items() if len(v) == 1] or list(adj)
    zof = {n: min(pts[off[i]:off[i + 1]][:, 2].min() for _, i in adj[n]) for n in ends}
    u, seen, out = min(ends, key=zof.get), set(), []
    while True:
        nxt = [(v, i) for v, i in adj[u] if i not in seen]
        if not nxt:
            break
        v, i = max(nxt, key=lambda x: off[x[1] + 1] - off[x[1]])
        seen.add(i)
        out.append(edge_pts(i, u))
        u = v
    return np.vstack([o if k == 0 else o[1:] for k, o in enumerate(out)]) if out else None


paths = {"aorta (root → bifurcation)": aorta_path()}
_ivc = chain_path("inferior vena cava")
if _ivc is not None:
    paths["IVC (confluence → right atrium)"] = _ivc
for nm in ("Superior vena cava", "Inferior vena cava (thoracic part)", "Brachiocephalic trunk", "Left common carotid artery", "Right common carotid artery", "Left subclavian artery", "Bifurcation of pulmonary trunk", "Left coronary artery"):
    ids = [i for i, n in enumerate(names) if n == nm]
    if ids:
        i = max(ids, key=lambda k: off[k + 1] - off[k])
        paths[nm] = edge_pts(i)
tc = np.arange(NS) * m.T / NS
summary = {}
fig, axs = plt.subplots(4, 1, figsize=(12, 15))
for name, p in paths.items():
    X, s = resample(p)
    if len(X) < 14:
        continue
    res = {}
    for r in (0.0, 1.0):
        U, ST, K = [], [], []
        for t in tc:
            x = m.position(X, t, r)
            U.append(np.linalg.norm(x - X, axis=1))
            st, k = metrics(X, x)
            ST.append(st)
            K.append(k)
        U, ST, K = np.array(U), np.array(ST), np.array(K)
        # cardiac motion alone (r = 0): the displacement relative to t = 0 (end-diastole) as well
        res[f"r{r:.0f}"] = {"disp_max_mm": float(U.max() * 1e3), "stretch_min": float(ST.min()), "stretch_max": float(ST.max()), "kink_max_deg": float(np.abs(K).max()),
                            "kink_at_mm": float(s[5:-5][np.unravel_index(np.abs(K).argmax(), K.shape)[1]] * 1e3)}
        if name.startswith("aorta"):
            lab = "breathing r = 1 (end-inspiration)" if r else "cardiac only (r = 0)"
            axs[0].plot(s * 1e3, U.max(0) * 1e3, label=f"max over the cycle, {lab}")
            axs[1].plot(s[1:-1] * 1e3, ST.max(0), label=f"max stretch, {lab}")
            axs[1].plot(s[1:-1] * 1e3, ST.min(0), "--", label=f"min stretch, {lab}")
            axs[2].plot(s[5:-5] * 1e3, np.abs(K).max(0), label=f"max |Δ bend| over ±5 mm, {lab}")
    # respiratory displacement alone at t = 0
    x0, x1 = m.position(X, 0.0, 0.0), m.position(X, 0.0, 1.0)
    res["resp_only_disp_mm"] = {"max": float(np.linalg.norm(x1 - x0, axis=1).max() * 1e3), "min": float(np.linalg.norm(x1 - x0, axis=1).min() * 1e3)}
    if name.startswith("aorta"):
        axs[3].plot(s * 1e3, (x1 - x0)[:, 2] * 1e3, label="SI (z) displacement end-inspiration − end-expiration")
        axs[3].plot(s * 1e3, X[:, 2] * 1e3 - X[0, 2] * 1e3, "k:", lw=0.8, label="reference z − z(root) (mm, for orientation)")
        res["profile"] = {"s_mm": (s * 1e3)[::5].round(1).tolist(), "z_mm": (X[:, 2] * 1e3)[::5].round(1).tolist(), "resp_dz_mm": ((x1 - x0)[:, 2] * 1e3)[::5].round(2).tolist()}
    if name.startswith("IVC"):
        res["profile"] = {"s_mm": (s * 1e3)[::5].round(1).tolist(), "z_mm": (X[:, 2] * 1e3)[::5].round(1).tolist(), "resp_dz_mm": ((x1 - x0)[:, 2] * 1e3)[::5].round(2).tolist(),
                          "card_disp_max_mm": (np.array([np.linalg.norm(m.position(X, t, 0.0) - X, axis=1) for t in tc]).max(0) * 1e3)[::5].round(2).tolist()}
    summary[name] = res
    print(f"{name:40s} " + " | ".join(f"{k}: disp ≤ {v['disp_max_mm']:.1f} mm, stretch {v['stretch_min']:.3f}–{v['stretch_max']:.3f}, kink ≤ {v['kink_max_deg']:.1f}° at {v['kink_at_mm']:.0f} mm" for k, v in res.items() if k in ("r0", "r1")) + f" | resp only {res['resp_only_disp_mm']['min']:.1f}–{res['resp_only_disp_mm']['max']:.1f} mm")
for ax, yl in zip(axs, ["displacement (mm)", "axial stretch (2 mm segments)", "|Δ bending angle| (deg)", "mm"]):
    ax.set_xlabel("arc length along the aorta from the root (mm)")
    ax.set_ylabel(yl)
    ax.legend(fontsize=8, frameon=False)
axs[0].set_title(f"{P.name}: motion along the aorta", loc="left")
fig.tight_layout()
fig.savefig(OUT / "aorta_motion.png", dpi=110)
(OUT / "vessel_motion.json").write_text(json.dumps(summary, indent=1))
print(OUT / "aorta_motion.png")
