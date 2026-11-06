"""Linear 1D pulsatile flow on the atlas vessel graph (`vessel_graph.py`), solved per heart-rate harmonic (CPU only).

    python -I network_flow.py <export_dir> <out_dir> [--hr 70] [--sv 75] [--harmonics 25]

Model (classic linearised 1D / transmission-line theory, e.g. Womersley 1957; Olufsen 2000 for the wall stiffness):
- Every graph edge is cut into ≤ 5 mm π-segments: series impedance Z_L·l with the Womersley longitudinal impedance
  Z_L = iωρ / (πr²(1 − F10(α))), F10 = 2J1(z)/(zJ0(z)), z = α i^{3/2} (Poiseuille 8μ/πr⁴ + inertance at α < 1), shunt
  admittance iωC'·l/2 at each end with C' = 3πr²/(2 Eh/r), Eh/r = k1 e^{k2 r} + k3 (× a factor for pulmonary vessels
  and veins). The graph may contain cycles (anastomoses, venous plexuses): nodal admittance solve Y(ω) p = I.
- Systemic / pulmonary arteries: the root (aortic / pulmonary valve) gets the ventricular outflow Q(t) as a current
  source; each terminal ends in a 3-element Windkessel (Rc, R, C) to the venous / left-atrial pressure. Terminal
  resistances come from a DC solve with Murray flows (∝ r³ of the terminal) and are rescaled until the mean root
  pressure is the target; the peripheral compliance (total minus the network's own) is split ∝ flow.
- Veins: the atrial ends (venae cavae, coronary sinus / pulmonary veins) carry the atrial pressure waveform (a, c, v
  waves) as a pressure boundary; terminals are capillary sources with the Murray share of the cardiac output and no
  pulsation — the caval / pulmonary-venous pulsatility comes from the atrial waves through the compliant veins.
- Arterial fragments not connected to their main tree are fed by the main tree's pressure at the nearest node and draw
  their root flow from it there (coupled fixed-point iteration), so the main root carries the full ventricular output.
Output (<out_dir>/network_flow.npz + .json + figs/): per centreline sample of graph.npz the complex flow and pressure
harmonics (flow positive along the sample order), per-edge summaries, waveforms, and gates (Kirchhoff residuals).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.special import jv  # noqa: E402

E = Path(sys.argv[1])
OUT = Path(sys.argv[2])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opt(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


HR = opt("--hr", 70.0)
SV = opt("--sv", 75.0) * 1e-6  # m³
P_SYS, P_PUL = opt("--p-sys", 93.0), opt("--p-pul", 15.0)  # mean arterial pressures [mmHg]
C_SCALE = opt("--ctot-scale", 1.0)  # total arterial compliance factor (both circulations)
TERR_SIGMA, TERR_SEED = opt("--terr-sigma", 0.0), int(opt("--seed", 0))  # lognormal jitter of the territory fractions
NH = int(opt("--harmonics", 25))
T = 60.0 / HR
W = 2 * np.pi / T * np.arange(NH + 1)  # angular frequencies, W[0] = 0
RHO, MU = 1060.0, 3.5e-3
MMHG = 133.322
K1, K2, K3 = 2.0e6, -2253.0, 8.65e4  # Olufsen 2000 (SI): Eh/r0 = k1 exp(k2 r0) + k3
SEG = 5e-3
PHI_VE = 0.15  # wall viscoelastic loss tangent (≈ 8.5°; damps the high-frequency reflections)

g = np.load(E / "graph.npz")
gj = json.loads((E / "graph.json").read_text())
npos, ea, eb, eoff = g["node_pos"].astype(np.float64), g["edge_a"], g["edge_b"], g["edge_offsets"]
pts, rad, ecomp = g["points"].astype(np.float64), g["radius"].astype(np.float64), g["edge_comp"]
names = gj["edge_names"]
comps = {c["id"]: c for c in gj["components"]}
nn, ne = len(npos), len(ea)
deg = np.bincount(np.r_[ea, eb], minlength=nn)
# polyline sample order a → b (checked)
bad = sum(np.linalg.norm(pts[eoff[k]] - npos[ea[k]]) > 1e-4 or np.linalg.norm(pts[eoff[k + 1] - 1] - npos[eb[k]]) > 1e-4 for k in range(ne))
assert bad == 0, f"{bad} edges whose polyline does not run from node a to node b"

# ── waveforms (t = 0: onset of ventricular ejection) ──────────────────────────────────────────────────────────────────
NT = 1024
tt = np.arange(NT) * T / NT


def ejection(tej, a, b, back, sv):
    x = tt / tej
    q = np.where(x < 1, np.clip(x, 0, 1) ** a * np.clip(1 - x, 0, 1) ** b, 0.0)
    q /= q.max()
    tb = 0.05
    q += np.where((tt > tej) & (tt < tej + tb), -back * np.sin(np.pi * (tt - tej) / tb), 0.0)
    return q * sv / (q.sum() * T / NT)


def atrial(mean, a, c, v, tej):
    def bump(t0, w):
        d = (tt - t0 + T / 2) % T - T / 2
        return np.exp(-0.5 * (d / w) ** 2)

    p = a * bump(T - 0.06, 0.04) + c * bump(0.03, 0.02) + v * bump(tej + 0.07, 0.06)
    return (p - p.mean() + mean) * MMHG


TEJ_L = 0.30 * (T / 0.857) ** 0.5  # ejection time shortens with heart rate (≈ √T)
TEJ_R = 0.33 * (T / 0.857) ** 0.5
WAVE = {
    "Q_aorta": ejection(TEJ_L, 1.2, 2.0, 0.06, SV),
    "Q_pulmonary": ejection(TEJ_R, 1.6, 1.6, 0.03, SV),
    "P_RA": atrial(4.0, 3.0, 1.5, 3.0, TEJ_L),
    "P_LA": atrial(8.0, 3.0, 1.0, 5.0, TEJ_L),
}


def harm(x):
    return np.fft.rfft(x)[: NH + 1] / NT  # x(t) = Re Σ_n c_n e^{iω_n t}, c_0 real mean, c_n (n>0) ×2 below


def synth(c, t=tt):
    """c [..., NH+1] harmonics → real signal [..., len(t)]"""
    c = np.asarray(c)
    return np.real(c[..., :1] + 2 * (c[..., 1:] @ np.exp(1j * W[1:, None] * t[None, :])))


H = {k: harm(v) for k, v in WAVE.items()}

# ── classes and boundary nodes ─────────────────────────────────────────────────────────────────────────────────────
CLASS = {
    "systemic_arterial": dict(kind="artery", ehr=1.0, inflow="Q_aorta", p_mean=P_SYS, p_out=5.0, c_tot=1.3 * C_SCALE),
    "pulmonary_arterial": dict(kind="artery", ehr=0.35, inflow="Q_pulmonary", p_mean=P_PUL, p_out=8.0, c_tot=5.0 * C_SCALE),
    "systemic_venous": dict(kind="vein", ehr=0.1, p_root="P_RA"),
    "pulmonary_venous": dict(kind="vein", ehr=0.15, p_root="P_LA"),
}
edges_of = defaultdict(list)
for k in range(ne):
    edges_of[int(ecomp[k])].append(k)
heart_c = np.array([0.02, -0.025, 1.29])


def node_r(n):
    """radius at node n (of the incident edges' end samples, max)"""
    r = 0.0
    for k in node_edges[n]:
        r = max(r, rad[eoff[k]] if ea[k] == n else rad[eoff[k + 1] - 1])
    return max(r, 0.1e-3)


node_edges = defaultdict(list)
for k in range(ne):
    node_edges[int(ea[k])].append(k)
    node_edges[int(eb[k])].append(k)


def venous_roots(cid):
    return [int(r) for r in comps[cid].get("roots", [comps[cid]["root_node"]])]


# main component per arterial class = the one containing the root vessel; the others are fed fragments
main_of = {}
for cid, c in comps.items():
    if c["class"] in ("systemic_arterial", "pulmonary_arterial"):
        if c["class"] not in main_of or c["length_m"] > comps[main_of[c["class"]]]["length_m"]:
            main_of[c["class"]] = cid

# ── π-segments ─────────────────────────────────────────────────────────────────────────────────────────────────────
seg_u, seg_v, seg_l, seg_r, seg_edge, seg_s0, seg_s1 = [], [], [], [], [], [], []
n_int = nn
for k in range(ne):
    s0, s1 = int(eoff[k]), int(eoff[k + 1])
    p = pts[s0:s1]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    L = d[-1]
    m = max(1, int(np.ceil(L / SEG)))
    cut = np.unique(np.searchsorted(d, np.linspace(0, L, m + 1)).clip(0, len(p) - 1))
    cut[0], cut[-1] = 0, len(p) - 1
    if len(cut) < 2:
        cut = np.array([0, len(p) - 1])
    nodes = [int(ea[k])] + list(range(n_int, n_int + len(cut) - 2)) + [int(eb[k])]
    n_int += len(cut) - 2
    for j in range(len(cut) - 1):
        i0, i1 = cut[j], cut[j + 1]
        rr = rad[s0 + i0 : s0 + i1 + 1]
        seg_u.append(nodes[j])
        seg_v.append(nodes[j + 1])
        seg_l.append(max(d[i1] - d[i0], 1e-4))
        seg_r.append(max(float(np.mean(rr**-4) ** -0.25), 0.1e-3))
        seg_edge.append(k)
        seg_s0.append(s0 + i0)
        seg_s1.append(s0 + i1)
seg_u, seg_v, seg_l, seg_r, seg_edge = map(np.asarray, (seg_u, seg_v, seg_l, seg_r, seg_edge))
seg_s0, seg_s1 = np.asarray(seg_s0), np.asarray(seg_s1)
NN = n_int
seg_cls = np.array([comps[int(ecomp[k])]["class"] for k in seg_edge])
ehr_fac = np.array([CLASS.get(c, {"ehr": 1.0})["ehr"] for c in seg_cls])
EHR = (K1 * np.exp(K2 * seg_r) + K3) * ehr_fac
CP = 3 * np.pi * seg_r**2 / (2 * EHR)  # compliance per length [m²/Pa]


def z_long(w, r):
    if w == 0:
        return 8 * MU / (np.pi * r**4)
    alpha = r * np.sqrt(w * RHO / MU)
    z = np.zeros_like(r, dtype=complex)
    lo = alpha < 1.0
    z[lo] = 8 * MU / (np.pi * r[lo] ** 4) + 1j * w * 4 / 3 * RHO / (np.pi * r[lo] ** 2)
    hi = ~lo
    zz = alpha[hi] * np.exp(3j * np.pi / 4)
    f10 = 2 * jv(1, zz) / (zz * jv(0, zz))
    z[hi] = 1j * w * RHO / (np.pi * r[hi] ** 2 * (1 - f10))
    return z


def assemble(w, extra_diag):
    ys = 1.0 / (z_long(w, seg_r) * seg_l)
    yt = 1j * w * CP * seg_l / 2 / (1 + 1j * PHI_VE * (w > 0))  # viscoelastic wall: complex modulus E(1 + iφ)
    rows = np.r_[seg_u, seg_v, seg_u, seg_v]
    cols = np.r_[seg_u, seg_v, seg_v, seg_u]
    vals = np.r_[ys + yt, ys + yt, -ys, -ys]
    Y = sp.coo_matrix((vals, (rows, cols)), shape=(NN, NN)).tocsr()
    return Y + sp.diags(extra_diag), ys, yt


def solve(Y, inj, dir_nodes, dir_vals, active):
    """Y p = inj on the active nodes, p fixed on dir_nodes"""
    p = np.zeros(NN, complex)
    p[dir_nodes] = dir_vals
    free = np.flatnonzero(active & ~np.isin(np.arange(NN), dir_nodes))
    rhs = inj[free] - Y[free][:, dir_nodes] @ p[dir_nodes]
    p[free] = spl.spsolve(Y[free][:, free].tocsc(), rhs)
    return p


seg_comp = ecomp[seg_edge]
node_comp = np.full(NN, -1)
node_comp[seg_u] = seg_comp
node_comp[seg_v] = seg_comp
term_nodes = np.flatnonzero(deg == 1)

# per component boundary setup
setup = {}
for cid, c in comps.items():
    cfg = CLASS.get(c["class"])
    if cfg is None:
        continue
    terms = [int(n) for n in term_nodes if node_comp[n] == cid]
    if cfg["kind"] == "artery":
        if cid == main_of[c["class"]]:
            root, fed = int(c["root_node"]), None
        else:  # fragment: its largest-radius terminal near the main tree is the root
            root = int(c["root_node"])
            mainn = np.flatnonzero(node_comp[:nn] == main_of[c["class"]])
            fed = int(mainn[np.argmin(np.linalg.norm(npos[mainn] - npos[root], axis=1))])
        terms = [n for n in terms if n != root]
        setup[cid] = dict(cfg=cfg, root=root, fed=fed, terms=terms)
    else:
        roots = venous_roots(cid)
        setup[cid] = dict(cfg=cfg, roots=roots, terms=[n for n in terms if n not in roots])

# flow shares: first-order supply / drainage territories (fractions of the cardiac output, rounded physiology:
# brain ~14 %, coronary ~4.5 %, kidneys ~19 %, splanchnic ~22 %, legs ~16 %, arms ~ part of the subclavians), split
# inside each territory ∝ r³ (Murray); terminals outside every territory share the rest ∝ r³
TERR = {
    # (≈ 7 % of the output is left for the unnamed terminals: chest / abdominal wall, bronchial, oesophageal, spinal)
    "systemic_arterial": {"Left coronary artery": 0.025, "Right coronary artery": 0.02, "Right common carotid artery": 0.065, "Left common carotid artery": 0.065,
                          "Right subclavian artery": 0.06, "Left subclavian artery": 0.06, "Coeliac trunk": 0.11, "Superior mesenteric artery": 0.11,
                          "Inferior mesenteric artery": 0.025, "Right renal artery": 0.095, "Left renal artery": 0.095, "Common iliac artery.r": 0.10, "Common iliac artery.l": 0.10},
    "systemic_venous": {"Hepatic portal vein": 0.20, "Right renal vein": 0.095, "Left renal vein": 0.095, "Internal jugular vein.l": 0.065, "Internal jugular vein.r": 0.065,
                        "Left subclavian vein": 0.05, "Right subclavian vein": 0.05, "Common iliac vein.l": 0.08, "Common iliac vein.r": 0.08, "Coronary sinus": 0.045, "Azygos vein": 0.03},
    # lungs: right ~55 %, left ~45 % (the disconnected left-upper segmental fragment takes the rest)
    "pulmonary_arterial": {"Right pulmonary artery": 0.55, "Left pulmonary artery": 0.37},
    "pulmonary_venous": {"Right superior pulmonary vein": 0.30, "Right inferior pulmonary vein": 0.25, "Left superior pulmonary vein": 0.25, "Left inferior pulmonary vein": 0.20},
}
from scipy.sparse.csgraph import dijkstra  # noqa: E402

elen_g = np.array([np.linalg.norm(np.diff(pts[eoff[k] : eoff[k + 1]], axis=0), axis=1).sum() for k in range(ne)])
Gw = sp.coo_matrix((np.r_[elen_g, elen_g] + 1e-9, (np.r_[ea, eb], np.r_[eb, ea])), shape=(nn, nn)).tocsr()
edge_of = {}
for k in range(ne):
    edge_of.setdefault((int(min(ea[k], eb[k])), int(max(ea[k], eb[k]))), k)
if TERR_SIGMA > 0:  # per-phantom territory fractions: lognormal jitter, the unassigned rest kept ≥ 4 %
    rng_t = np.random.default_rng(TERR_SEED)
    for cls_, tv in TERR.items():
        tot0 = sum(tv.values())
        for k_ in sorted(tv):
            tv[k_] *= float(np.exp(rng_t.normal(0, TERR_SIGMA)))
        tot = sum(tv.values())
        cap = min(tot0, 0.96) if cls_.endswith("arterial") and cls_.startswith("systemic") else tot0
        for k_ in tv:
            tv[k_] *= cap / tot
share, territory = {}, {}
for cls in CLASS:
    cids = [cid for cid, s in setup.items() if comps[cid]["class"] == cls]
    ts = [(cid, n) for cid in cids for n in setup[cid]["terms"]]
    if not ts:
        continue
    terr = TERR.get(cls, {})
    tname = {}
    if terr:
        srcs = sorted({r for cid in cids for r in ([setup[cid]["root"]] if "root" in setup[cid] else setup[cid]["roots"])})
        _, pred, _src = dijkstra(Gw, indices=srcs, return_predecessors=True, min_only=True)
        for cid, n in ts:
            m, hit = n, None
            while pred[m] >= 0:
                k = edge_of[(int(min(m, pred[m])), int(max(m, pred[m])))]
                if names[k] in terr:
                    hit = names[k]  # keep walking: the match closest to the root wins
                m = pred[m]
            tname[n] = hit
    sealed = {n for _, n in ts if any("portal" in names[k].lower() for k in node_edges[n])}  # intrahepatic portal ends: no capillary source
    r3 = {n: node_r(n) ** 3 for _, n in ts if n not in sealed}
    for n in sealed:
        share[n] = 0.0
    groups = defaultdict(list)
    for n in r3:
        groups[tname.get(n)].append(n)
    rest = 1.0 - sum(terr[t] for t in groups if t is not None)
    for t, ns in groups.items():
        frac = terr[t] if t is not None else rest
        tot = sum(r3[n] for n in ns)
        for n in ns:
            share[n] = frac * r3[n] / tot
            territory[n] = t
    missing = sorted(set(terr) - set(groups))
    if missing:
        print(f"{cls}: territories without terminals: {missing}")
CO = SV / T

# ── DC calibration of the arterial terminals ───────────────────────────────────────────────────────────────────────
Y0, _, _ = assemble(0.0, np.zeros(NN))
term_R, term_C, term_Rc = {}, {}, {}
p_dc = np.zeros(NN)
mean_root_p = {}
for cls in ("systemic_arterial", "pulmonary_arterial"):
    cfg = CLASS[cls]
    cids = [cid for cid in setup if comps[cid]["class"] == cls]
    cids.sort(key=lambda c: c != main_of[cls])  # main first (fragments use its pressure)
    dp = (cfg["p_mean"] - cfg["p_out"]) * MMHG
    for cid in cids:
        s = setup[cid]
        act = node_comp == cid
        q = np.array([share[n] * CO for n in s["terms"]])
        inj = np.zeros(NN, complex)
        inj[s["terms"]] = -q
        proot = dp if s["fed"] is None else p_dc[s["fed"]]
        p = solve(Y0, inj, [s["root"]], [proot], act).real
        for n, qq in zip(s["terms"], q):
            term_R[n] = max(p[n], 0.25 * dp) / qq
        p_dc[act] = p[act]
    # rescale so that the root mean pressure is the target with the root current = CO; fed fragments draw their root
    # flow from the main tree at their feeding node (coupled by fixed-point iteration)
    qf = {cid: CO * sum(share[n] for n in setup[cid]["terms"]) for cid in cids if setup[cid]["fed"] is not None}
    for it in range(10):
        for cid in cids:
            s = setup[cid]
            act = node_comp == cid
            diag = np.zeros(NN)
            diag[s["terms"]] = [1 / term_R[n] for n in s["terms"]]
            Yd = (Y0 + sp.diags(diag)).tocsr()
            if s["fed"] is None:
                inj = np.zeros(NN, complex)
                inj[s["root"]] = CO
                for c2, q2 in qf.items():
                    inj[setup[c2]["fed"]] -= q2
                p = solve(Yd, inj, [], [], act).real
                f = p[s["root"]] / dp
                for n in s["terms"]:
                    term_R[n] /= f
            else:
                p = solve(Yd, np.zeros(NN, complex), [s["root"]], [p_dc[s["fed"]]], act).real
                qf[cid] = float((Yd[s["root"]] @ p)[0])  # current entering the fragment at its root
            p_dc[act] = p[act]
    for cid in cids:
        s = setup[cid]
        net_C = float((CP * seg_l)[seg_comp == cid].sum())
        cterm = max(cfg["c_tot"] * 1e-6 / MMHG - net_C, 0.2 * cfg["c_tot"] * 1e-6 / MMHG)
        for n in s["terms"]:
            k = node_edges[n][0]
            j = np.flatnonzero((seg_edge == k) & ((seg_u == n) | (seg_v == n)))[0]
            A = np.pi * seg_r[j] ** 2
            zc = RHO * np.sqrt(A / (RHO * CP[j])) / A  # ρc/A with c = sqrt(A/(ρC'))
            term_Rc[n] = min(zc, 0.3 * term_R[n])
            term_C[n] = cterm * share[n] / sum(share[m] for m in s["terms"])
        mean_root_p[cid] = p_dc[s["root"]] / MMHG + cfg["p_out"]

bad_R = [n for n, r in term_R.items() if not np.isfinite(r) or r <= 0]
assert not bad_R, f"non-finite terminal resistances at {bad_R[:5]}"
print("calibrated:", {comps[c]["class"] + f"#{c}": round(v, 1) for c, v in mean_root_p.items()})
# ── harmonic solves ────────────────────────────────────────────────────────────────────────────────────────────────
P = np.zeros((NH + 1, NN), complex)
QS = np.zeros((NH + 1, len(seg_u)), complex)
resid = []
order = sorted(setup, key=lambda c: (comps[c]["class"] not in main_of or c != main_of.get(comps[c]["class"]),))
for h, w in enumerate(W):
    diag = np.zeros(NN, complex)
    for cid, s in setup.items():
        if s["cfg"]["kind"] == "artery":
            for n in s["terms"]:
                rc, r, c = term_Rc[n], term_R[n] - term_Rc[n], term_C[n]
                diag[n] = 1.0 / (rc + r / (1 + 1j * w * r * c))
    Yh, ys, yt = assemble(w, diag)
    Yh = Yh.tocsr()
    art = [cid for cid in order if setup[cid]["cfg"]["kind"] == "artery"]
    ven = [cid for cid in order if setup[cid]["cfg"]["kind"] != "artery"]
    qfh, injs = {}, {}
    for it in range(5 if any(setup[c]["fed"] is not None for c in art) else 1):  # fragments ↔ main tree
        for cid in art:
            s = setup[cid]
            act = node_comp == cid
            inj = np.zeros(NN, complex)
            if s["fed"] is None:
                inj[s["root"]] = H[s["cfg"]["inflow"]][h]
                for c2, q2 in qfh.items():
                    if comps[c2]["class"] == comps[cid]["class"]:
                        inj[setup[c2]["fed"]] -= q2
                p = solve(Yh, inj, [], [], act)
            else:
                p = solve(Yh, inj, [s["root"]], [P[h, s["fed"]]], act)
                qfh[cid] = complex((Yh[s["root"]] @ p)[0])
            P[h, act] = p[act]
            injs[cid] = inj
    for cid in ven:
        s = setup[cid]
        act = node_comp == cid
        inj = np.zeros(NN, complex)
        if h == 0:
            inj[s["terms"]] = [share[n] * CO for n in s["terms"]]
        p = solve(Yh, inj, s["roots"], [H[s["cfg"]["p_root"]][h]] * len(s["roots"]), act)
        P[h, act] = p[act]
        injs[cid] = inj
    for cid in order:
        s = setup[cid]
        act = node_comp == cid
        inj = injs[cid]
        dn = s.get("roots", [s.get("root")] if s.get("fed") is not None else [])
        free = np.flatnonzero(act & ~np.isin(np.arange(NN), dn))
        r_ = (Yh[free] @ P[h] - inj[free])
        scale = max(np.abs(inj[free]).max(), float(np.abs(ys).max() * np.abs(P[h, act]).max()), 1e-30)
        resid.append((float(np.abs(r_).max() / scale), comps[cid]["class"] + f"#{cid}", h))
    QS[h] = ys * (P[h, seg_u] - P[h, seg_v])
# arterial gauge: pressure relative to the outflow pressure → absolute
for cid, s in setup.items():
    if s["cfg"]["kind"] == "artery":
        P[0, node_comp == cid] += s["cfg"]["p_out"] * MMHG

# Kirchhoff at DC on graph nodes: Σ series flows into node = injection
kir = np.zeros(NN)
np.add.at(kir, seg_u, -QS[0].real)
np.add.at(kir, seg_v, QS[0].real)
bnodes = {n for s_ in setup.values() for n in s_.get("roots", []) + ([s_["root"]] if "root" in s_ else []) + ([s_["fed"]] if s_.get("fed") is not None else []) + s_["terms"]}
junc = np.array([n for n in np.flatnonzero(deg >= 2) if n not in bnodes])
kir_rel = float(np.abs(kir[junc]).max() / CO)

# ── per-sample fields ──────────────────────────────────────────────────────────────────────────────────────────────
ns = len(pts)
q_s = np.zeros((ns, NH + 1), np.complex64)
p_s = np.zeros((ns, NH + 1), np.complex64)
for j in range(len(seg_u)):
    a, b = seg_s0[j], seg_s1[j]
    q_s[a : b + 1] = QS[:, j]
    x = np.linspace(0, 1, b - a + 1)[:, None]
    p_s[a : b + 1] = (1 - x) * P[:, seg_u[j]] + x * P[:, seg_v[j]]
flow_ok = np.array([comps[int(ecomp[k])]["class"] in CLASS for k in range(ne)])
mean_q = np.array([float(QS[0, seg_edge == k].real.mean()) if (seg_edge == k).any() else 0.0 for k in range(ne)])
TS = tt[::4]  # 256 samples per cycle for the summaries / figures
qt_seg = synth(QS.T, TS)  # [nseg, 256]
amp = np.array([np.ptp(qt_seg[seg_edge == k].mean(0)) for k in range(ne)])
pi_edge = amp / np.maximum(np.abs(mean_q), 1e-12)

np.savez_compressed(OUT / "network_flow.npz", q=q_s, p=p_s, omega=W, period=T, hr=HR, sv=SV, t=tt, **WAVE,
                    edge_mean_q=mean_q, edge_pulsatility=pi_edge, edge_has_flow=flow_ok)
summary = {
    "hr_bpm": HR, "sv_ml": SV * 1e6, "co_l_min": CO * 6e4, "harmonics": NH, "segments": int(len(seg_u)), "unknowns": int(NN),
    "mean_root_pressure_mmHg": {comps[c]["class"] + f"#{c}": round(v, 1) for c, v in mean_root_p.items()},
    "solve_residual_max": max(resid)[0], "solve_residual_worst": list(max(resid)[1:]), "kirchhoff_dc_rel_max": kir_rel,
    "terminals": {cls: sum(len(s["terms"]) for c, s in setup.items() if comps[c]["class"] == cls) for cls in CLASS},
    "venous_roots": {comps[c]["class"] + f"#{c}": s["roots"] for c, s in setup.items() if "roots" in s},
}
KEY = ["Ascending aorta", "Brachiocephalic trunk", "Left common carotid artery", "Left subclavian artery", "Thoracic aorta", "Abdominal aorta",
       "Coeliac trunk", "Superior mesenteric artery", "Left renal artery", "Common iliac artery.l", "Femoral artery.l", "Brachial artery.l",
       "Left coronary artery", "Right coronary artery", "connector: Bifurcation of pulmonary trunk", "Left pulmonary artery", "Right pulmonary artery",
       "connector: Right superior pulmonary vein", "connector: Left inferior pulmonary vein", "connector: Right inferior pulmonary vein", "connector: Left superior pulmonary vein", "Superior vena cava", "Inferior vena cava (thoracic part)",
       "Inferior vena cava (abdominal part)", "Hepatic portal vein", "Internal jugular vein.l", "Left renal vein", "Azygos vein"]
table = {}
for nm in KEY:
    ks = [k for k in range(ne) if names[k] == nm]
    if not ks:
        continue
    k = max(ks, key=lambda k: rad[eoff[k] : eoff[k + 1]].mean())
    js = np.flatnonzero(seg_edge == k)
    jm = js[len(js) // 2]
    q = qt_seg[jm]
    sgn = 1.0 if np.mean(q) >= 0 else -1.0
    pt = synth(P[:, [seg_u[jm]]].T, TS)[0] / MMHG
    table[nm] = {"r_mm": round(float(seg_r[jm] * 1e3), 2), "mean_ml_s": round(float(sgn * q.mean() * 1e6), 2), "peak_ml_s": round(float((sgn * q).max() * 1e6), 1),
                 "min_ml_s": round(float((sgn * q).min() * 1e6), 1), "mean_v_cm_s": round(float(sgn * q.mean() / (np.pi * seg_r[jm] ** 2) * 100), 1),
                 "peak_v_cm_s": round(float((sgn * q).max() / (np.pi * seg_r[jm] ** 2) * 100), 1),
                 "p_mean_mmHg": round(float(pt.mean()), 1), "p_sys_dia_mmHg": [round(float(pt.max()), 1), round(float(pt.min()), 1)], "seg": int(jm), "sign": sgn}
summary["vessels"] = table
(OUT / "network_flow.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "vessels"}, indent=1))
for nm, d in table.items():
    print(f"{nm:38s} r {d['r_mm']:5.2f} mm  Q {d['mean_ml_s']:7.2f} mL/s (peak {d['peak_ml_s']:7.1f}, min {d['min_ml_s']:7.1f})  v̄ {d['mean_v_cm_s']:6.1f} peak {d['peak_v_cm_s']:6.1f} cm/s  p {d['p_mean_mmHg']:5.1f} {d['p_sys_dia_mmHg']}")

# ── figures ────────────────────────────────────────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
groups = [
    ("systemic arteries — flow [mL/s]", ["Ascending aorta", "Thoracic aorta", "Abdominal aorta", "Brachiocephalic trunk", "Left common carotid artery", "Left subclavian artery", "Left renal artery", "Femoral artery.l"]),
    ("pulmonary — flow [mL/s]", ["connector: Bifurcation of pulmonary trunk", "Left pulmonary artery", "Right pulmonary artery", "connector: Right superior pulmonary vein", "connector: Right inferior pulmonary vein", "connector: Left superior pulmonary vein", "connector: Left inferior pulmonary vein"]),
    ("systemic veins — flow toward the heart [mL/s]", ["Superior vena cava", "Inferior vena cava (thoracic part)", "Inferior vena cava (abdominal part)", "Hepatic portal vein", "Internal jugular vein.l", "Azygos vein"]),
]
for a, (ttl, vs) in zip(axs.flat, groups):
    for nm in vs:
        if nm in table:
            d = table[nm]
            a.plot(TS * 1e3, d["sign"] * qt_seg[d["seg"]] * 1e6, label=f"{nm} (r {d['r_mm']} mm)")
    a.axhline(0, color="k", lw=0.5)
    a.set_title(ttl)
    a.set_xlabel("t from ejection onset [ms]")
    a.legend(fontsize=7)
a = axs[1, 1]
for nm in ["Ascending aorta", "Abdominal aorta", "Brachial artery.l", "Femoral artery.l", "connector: Bifurcation of pulmonary trunk", "Superior vena cava"]:
    if nm in table:
        a.plot(TS * 1e3, synth(P[:, [seg_u[table[nm]["seg"]]]].T, TS)[0] / MMHG, label=nm)
a.plot(tt * 1e3, WAVE["P_RA"] / MMHG, "k:", label="RA (boundary)")
a.plot(tt * 1e3, WAVE["P_LA"] / MMHG, "k--", label="LA (boundary)")
a.set_title("pressure [mmHg]")
a.set_xlabel("t from ejection onset [ms]")
a.legend(fontsize=7)
fig.suptitle(f"1D pulsatile network on the Z-Anatomy vessel graph: HR {HR:.0f}, SV {SV * 1e6:.0f} mL, CO {CO * 6e4:.2f} L/min")
fig.tight_layout()
fig.savefig(OUT / "figs" / "01_waveforms.png", dpi=100)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(16, 10))
for a, (val, lab, cm, lim) in zip(axs, [(np.log10(np.abs(mean_q) * 1e6 + 1e-3), "log10 |mean Q| [mL/s]", "viridis", (-2, 2)), (np.clip(pi_edge, 0, 20), "pulsatility (peak-to-peak / |mean|)", "magma", (0, 8))]):
    segs = [pts[eoff[k] : eoff[k + 1], [0, 2]] for k in range(ne) if flow_ok[k]]
    vv = [val[k] for k in range(ne) if flow_ok[k]]
    lw = [max(0.3, 2 * rad[eoff[k] : eoff[k + 1]].mean() * 1e3 / 3) for k in range(ne) if flow_ok[k]]
    lc = LineCollection(segs, array=np.asarray(vv), cmap=cm, linewidths=lw)
    lc.set_clim(*lim)
    a.add_collection(lc)
    a.set_aspect("equal")
    a.autoscale()
    a.set_title(lab)
    fig.colorbar(lc, ax=a, fraction=0.03)
fig.savefig(OUT / "figs" / "02_network.png", dpi=90, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT)
