"""Coronary-artery lattice-Boltzmann case of a phantom: a nested fine domain (local refinement by zones) for the
coronary tree, run by the same `cardiac_lbm` as the 0.5 mm heart case, one-way coupled to it through the pressures.

    rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_coronary.py <phantom_dir> <case_dir> [--dx-mm 0.15]
        [--sdf-mm 0.25] [--rmin-mm 0.6] [--v-max 1.0] [--u-lattice 0.08] [--kappa-l 1.0] [--kappa-r 0.5] [--tau-im 1.0]
        [--cycles 6] [--warmup 3] [--bins 50]

Domain: the coronary arteries of the vessel graph (edges named Left / Right coronary artery, Anterior interventricular,
Circumflex artery of heart, Septal branches, Right inferolateral branch), from the first sample of each main coronary
edge (the ostium; the atlas graph joins the RCA to the aorta by a bridge from the left ostium) along every branch of
lumen radius ≥ --rmin-mm (≥ 4 lattice cells at 0.15 mm), as capsules of the graph samples; signed distance on a
--sdf-mm grid (the lattice samples it trilinearly). Solved in the heart's reference frame (static geometry, case "frame":
"reference"): the lab-frame velocity is F v_rel + u_wall of the cardiac motion (mapped by phantom_4d); a moving lattice
would have to cover the vessels' swept volume (≈ 20× the lumen).
Boundary conditions:
- ostia: pressure cuts (p = 0: the epicardial pressure drop is a few mmHg, the absolute aortic pressure stays out of
  the weakly compressible lattice);
- outlets (radius below the cut-off, graph ends): velocity cuts with the outflow of a lumped coronary outlet model
  (Kim et al. 2010, Ann Biomed Eng 38:3195): P_ao → R_a → [C_a] → R_am → [C_im, pressurised by P_im] → R_v → P_v,
  P_im = κ P_LV (left territories, κ --kappa-l) or κ P_RV (right, --kappa-r): the intramyocardial pump that makes the
  left coronary flow diastolic. Per outlet the total resistance gives the network's mean flow at its sample
  ((P̄_ao − P_v) / q̄), split R_a : R_am : R_v = 0.32 : 0.52 : 0.16 (Kim), time constants R_a C_a = 0.03 s and
  R_am C_im = --tau-im (1 s: with κ_L = 1 the left systolic / diastolic flow ratio is 0.37, physiological 0.3–0.5;
  0.15 s gave 1.1, no intramyocardial pump). P_ao is the network's aortic-root pressure waveform; P_LV / P_RV are built from the cardiac
  model's valve phases (ejection: the root pressure + 3 mmHg, filling: 6 mmHg, isovolumic phases: smooth ramps).
Writes <case_dir>/case.json (+ sdf.npy), the outlet flows and pressures in coronary_lpm.json, figures.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

A = sys.argv[1:]
P0, OUT = Path(A[0]), Path(A[1])
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


DX = float(opt("--dx-mm", 0.15)) * 1e-3
HS = float(opt("--sdf-mm", 0.25)) * 1e-3
RMIN = float(opt("--rmin-mm", 0.6)) * 1e-3
MMHG = 133.322
NAMES = ("Left coronary artery", "Right coronary artery", "Anterior interventricular artery", "Circumflex artery of heart",
         "Septal branches of anterior interventricular artery", "Right inferolateral branch of right coronary artery")
LEFT = ("Left coronary artery", "Anterior interventricular artery", "Circumflex artery of heart", "Septal branches of anterior interventricular artery")

V = vf.read(P0 / "vessels.hv")
P = np.asarray(V["points"], float)
R = np.asarray(V["radius"], float)
G = np.load(P0 / "graph.npz")
GJ = json.loads((P0 / "graph.json").read_text())
eoff, ea, eb = G["edge_offsets"], G["edge_a"], G["edge_b"]
names = GJ["edge_names"]
net = np.load(P0 / "flow" / "network_flow.npz")
om, qn, pn = net["omega"], net["q"], net["p"]
cm = vf.read(P0 / "motion" / "cardiac_motion.hv")
period = float(cm["period"])
adj = {}
for k in range(len(ea)):
    adj.setdefault(int(ea[k]), []).append(k)
    adj.setdefault(int(eb[k]), []).append(k)
cor = {k for k, n in enumerate(names) if n in NAMES}


def samples(k, node):
    s = np.arange(eoff[k], eoff[k + 1])
    return (s, 1) if ea[k] == node else (s[::-1], -1)


# ostia: the main coronary edges' node that touches a non-coronary edge (the aorta or a bridge to it)
starts = []
for main in ("Left coronary artery", "Right coronary artery"):
    ks = [k for k in cor if names[k] == main]
    best = None
    for k in ks:
        for nd in (int(ea[k]), int(eb[k])):
            if any(j not in cor for j in adj[nd]):
                cand = (float(R[eoff[k] : eoff[k + 1]].max()), k, nd)
                best = max(best, cand) if best else cand
    if best:
        starts.append((main, best[1], best[2]))
print("ostia:", [(m, int(k), int(nd), np.round(P[eoff[k] if ea[k] == nd else eoff[k + 1] - 1] * 1e3, 1).tolist()) for m, k, nd in starts])

# trees: coronary-named edges from each ostium, radius ≥ RMIN
branches = []
for main, k0, nd0 in starts:
    stack, used = [(nd0, None, -1, k0)], set()
    while stack:
        node, via, parent, only = stack.pop()
        for k in ([only] if only is not None else adj.get(node, [])):
            if k == via or k in used or k not in cor:
                continue
            used.add(k)
            sm, sg = samples(k, node)
            n_in = 0
            for j in sm:
                if R[j] < RMIN:
                    break
                n_in += 1
            seg = sm[:n_in]
            other = int(eb[k] if sg == 1 else ea[k])
            if n_in < 2:
                continue
            more = n_in == len(sm) and any(j != k and j in cor for j in adj.get(other, []))
            branches.append({"samples": seg, "edge": int(k), "sgn": int(sg), "parent": parent, "end": not more, "tree": main})
            if more:
                stack.append((other, k, len(branches) - 1, None))
# a branch whose children were all below RMIN is an end too
for i, b in enumerate(branches):
    if not b["end"] and not any(c["parent"] == i for c in branches):
        b["end"] = True
ends = [b for b in branches if b["end"]]
vol = sum(np.pi * float((R[b["samples"]] ** 2).mean()) * float(np.linalg.norm(np.diff(P[b["samples"]], axis=0), axis=1).sum()) for b in branches)
print(f"coronary tree: {len(branches)} branches, {len(ends)} outlets, {vol * 1e6:.2f} mL of lumen (R ≥ {RMIN * 1e3:.2f} mm)")

# ── pressures: aortic root (network), LV / RV from the valve phases ─────────────────────────────────────────────────
tt = np.linspace(0, period, 512, endpoint=False)
tq = np.asarray(cm["flows_t"])


def wave(h_, s_):
    return np.real(h_[s_, 0] + 2 * (h_[s_, 1:] @ np.exp(1j * np.outer(om[1:], tt))))


root = int(eoff[starts[0][1]]) if starts else 0
p_ao = wave(pn, root)
p_ao = p_ao if p_ao.mean() > 1000 else p_ao * MMHG  # network pressures in Pa (mmHg if small)
pa_samp = [k for k, n in enumerate(names) if n == "connector: Bifurcation of pulmonary trunk"]
p_pa = wave(pn, int(eoff[pa_samp[0]])) if pa_samp else 0.18 * p_ao
p_pa = p_pa if p_pa.mean() > 300 else p_pa * MMHG


def ventricle(q_out, q_in, p_art, edp, tau=0.04):
    """ventricular pressure from the outflow / inflow valve phases: ejection = arterial + 3 mmHg, filling = edp,
    isovolumic contraction: a smooth rise from edp to the arterial pressure at opening, relaxation: exponential decay
    from the arterial pressure at closure towards 0 (τ), not below edp − 4 mmHg"""
    qo = np.interp(tt, tq, np.asarray(q_out, float), period=period)
    qi = np.interp(tt, tq, np.asarray(q_in, float), period=period)
    ej, fi = qo > 0.03 * qo.max(), qi > 0.03 * qi.max()
    p = np.full(len(tt), edp)
    p[ej] = p_art[ej] + 3 * MMHG
    n = len(tt)
    dt_ = period / n
    for i in range(n):
        if ej[i] or fi[i]:
            continue
        # previous and next valve events (cyclic)
        j = 1
        while not (ej[(i - j) % n] or fi[(i - j) % n]):
            j += 1
        jn = 1
        while not (ej[(i + jn) % n] or fi[(i + jn) % n]):
            jn += 1
        if fi[(i - j) % n] and ej[(i + jn) % n]:  # isovolumic contraction
            a = j / (j + jn)
            p[i] = edp + (p_art[(i + jn) % n] + 3 * MMHG - edp) * (0.5 - 0.5 * np.cos(np.pi * a))
        else:  # isovolumic relaxation (or a gap): decay from the last pressure
            p0 = p_art[(i - j) % n] + 3 * MMHG if ej[(i - j) % n] else edp
            p[i] = max(p0 * np.exp(-j * dt_ / tau), edp - 4 * MMHG)
    return p


p_lv = ventricle(cm["Q_AO"], cm["Q_MV"], p_ao, 6 * MMHG)
p_rv = ventricle(cm["Q_PA"], cm["Q_TV"], p_pa, 4 * MMHG)
print(f"P_ao {p_ao.min() / MMHG:.0f}/{p_ao.max() / MMHG:.0f} mmHg, P_LV {p_lv.min() / MMHG:.0f}…{p_lv.max() / MMHG:.0f}, P_RV {p_rv.min() / MMHG:.0f}…{p_rv.max() / MMHG:.0f}")

# ── lumped coronary outlet model, periodic steady state ──────────────────────────────────────────────────────────────
PV = 5 * MMHG


def lpm(qbar, p_im):
    """flow into the outlet (through R_a) over the cycle for a target mean flow qbar"""
    Rt = (p_ao.mean() - PV) / max(qbar, 1e-12)
    Ra, Ram, Rv = 0.32 * Rt, 0.52 * Rt, 0.16 * Rt
    Ca, Cim = 0.03 / Ra, float(opt("--tau-im", 1.0)) / Ram
    n = len(tt)
    sub = 8
    h_ = period / n / sub
    pa_i = lambda t: np.interp(t, tt, p_ao, period=period)  # noqa: E731
    pim_i = lambda t: np.interp(t, tt, p_im, period=period)  # noqa: E731
    # states: P1 (at C_a), Pim_c = P2 − P_im (charge of C_im)
    x = np.array([p_ao.mean() * 0.8, 20 * MMHG])

    def f(t, x):
        P1, dC = x
        P2 = dC + pim_i(t)
        qa = (pa_i(t) - P1) / Ra
        q12 = (P1 - P2) / Ram
        qv = (P2 - PV) / Rv
        return np.array([(qa - q12) / Ca, (q12 - qv) / Cim])

    q = np.zeros(n)
    for cyc in range(8):
        for i in range(n):
            for s_ in range(sub):
                t = (i * sub + s_) * h_
                k1 = f(t, x)
                k2 = f(t + h_ / 2, x + h_ / 2 * k1)
                k3 = f(t + h_ / 2, x + h_ / 2 * k2)
                k4 = f(t + h_, x + h_ * k3)
                x = x + h_ / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            if cyc == 7:
                q[i] = (pa_i(i * sub * h_) - x[0]) / Ra
    return q * qbar / max(q.mean(), 1e-15)  # exact mean


cuts, lpm_out = [], []
for b in ends:
    s_ = int(b["samples"][-1])
    qbar = abs(float(np.real(qn[s_, 0])))
    left = names[b["edge"]] in LEFT
    p_im = float(opt("--kappa-l", 1.0)) * p_lv if left else float(opt("--kappa-r", 0.5)) * p_rv
    q_t = lpm(qbar, p_im)
    p = P[b["samples"]]
    t_end = p[-1] - p[-2]
    t_end /= np.linalg.norm(t_end)
    cuts.append({"name": f"{names[b['edge']]} #{s_}", "kind": "inlet", "center": p[-1].tolist(), "normal": t_end.tolist(), "radius": float(R[s_]), "slab": 3 * DX,
                 "q": {"t": tt.tolist(), "v": (-q_t).tolist()}})
    lpm_out.append({"name": cuts[-1]["name"], "left": left, "q_mean_ml_min": qbar * 6e7, "q_sys_dia": float(q_t[p_lv > 0.5 * p_lv.max()].mean() / max(q_t[p_lv < 0.2 * p_lv.max()].mean(), 1e-15))})
for main, k0, nd0 in starts:
    sm, sg = samples(k0, nd0)
    p0, p1 = P[sm[0]], P[sm[min(3, len(sm) - 1)]]
    nrm = (p0 - p1) / np.linalg.norm(p0 - p1)  # outward = back into the aorta
    cuts.append({"name": f"ostium {main}", "kind": "pressure", "p": 0.0, "center": p0.tolist(), "normal": nrm.tolist(), "radius": float(R[sm[0]]), "slab": 3 * DX,
                 "sponge": 0.0})
tot = sum(abs(float(np.real(qn[int(b["samples"][-1]), 0]))) for b in ends)
print(f"{len(ends)} outlets, network mean {tot * 6e7:.0f} mL/min; left systolic / diastolic flow ratio "
      f"{np.median([x['q_sys_dia'] for x in lpm_out if x['left']] or [np.nan]):.2f}, right {np.median([x['q_sys_dia'] for x in lpm_out if not x['left']] or [np.nan]):.2f}")
(OUT / "coronary_lpm.json").write_text(json.dumps({"t": tt.tolist(), "p_ao_mmHg": (p_ao / MMHG).tolist(), "p_lv_mmHg": (p_lv / MMHG).tolist(), "p_rv_mmHg": (p_rv / MMHG).tolist(),
                                                   "outlets": lpm_out}, indent=1))

# ── signed distance of the capsules on the sdf grid ──────────────────────────────────────────────────────────────────
from scipy.spatial import cKDTree  # noqa: E402

segs = []
for i, b in enumerate(branches):
    sm = b["samples"]
    pp = P[sm]
    rr = R[sm]
    if b["parent"] >= 0:
        q0 = branches[b["parent"]]["samples"][-1]
        pp, rr = np.vstack([P[q0], pp]), np.append(R[q0], rr)
    if b["end"]:
        te = pp[-1] - pp[-2]
        te /= np.linalg.norm(te)
        pp, rr = np.vstack([pp, pp[-1] + te * 4 * DX]), np.append(rr, rr[-1])  # the outlet slab beyond the cut
    for a in range(len(pp) - 1):
        segs.append((pp[a], pp[a + 1], rr[a], rr[a + 1]))
for main, k0, nd0 in starts:  # the inlet slab beyond the ostium (into the aorta)
    sm, sg = samples(k0, nd0)
    c = next(c for c in cuts if c["name"] == f"ostium {main}")
    segs.append((P[sm[0]], P[sm[0]] + np.asarray(c["normal"]) * 4 * DX, R[sm[0]], R[sm[0]]))
S0 = np.array([s[0] for s in segs])
S1 = np.array([s[1] for s in segs])
RA = np.array([s[2] for s in segs])
RB = np.array([s[3] for s in segs])
lo = np.minimum(S0, S1).min(0) - RA.max() - max(4e-3, 26 * DX)
hi = np.maximum(S0, S1).max(0) + RA.max() + max(4e-3, 26 * DX)
dims = np.ceil((hi - lo) / HS).astype(int) + 1
# max over capsules of (radius − distance): the signed distance, true out to SD_OUT (the solver's tile activation keeps
# sdf > −16 lattice cells = 2.4 mm; a clipped −3 h everywhere made the whole 110 mm box active: 329 M nodes), −1 m beyond
SD_OUT = max(3.5e-3, 24 * DX)
sdf = np.full(dims[::-1], -1.0, np.float32)
for a_, b_, ra_, rb_ in segs:
    r_ = max(ra_, rb_)
    i0 = np.maximum(np.floor((np.minimum(a_, b_) - r_ - SD_OUT - lo) / HS).astype(int), 0)
    i1 = np.minimum(np.ceil((np.maximum(a_, b_) + r_ + SD_OUT - lo) / HS).astype(int) + 1, dims)
    zz, yy, xx = np.meshgrid(*(np.arange(i0[d], i1[d]) for d in (2, 1, 0)), indexing="ij")
    X = np.stack([xx, yy, zz], -1) * HS + lo
    ab = b_ - a_
    t = np.clip(((X - a_) @ ab) / max(ab @ ab, 1e-18), 0, 1)
    d = np.linalg.norm(X - (a_ + t[..., None] * ab), axis=-1)
    blk = sdf[i0[2] : i1[2], i0[1] : i1[1], i0[0] : i1[0]]
    v_ = (ra_ + t * (rb_ - ra_) - d).astype(np.float32)
    np.maximum(blk, np.where(v_ >= -SD_OUT, v_, -1.0).astype(np.float32), out=blk)
np.save(OUT / "sdf.npy", sdf)
print(f"sdf {dims.tolist()} at {HS * 1e3:.2f} mm: {int((sdf > 0).sum())} voxels inside ({(sdf > 0).sum() * HS**3 * 1e6:.2f} mL)")

_tp = P0 / "tissue_params.json"
BLOOD_MR = None
if _tp.exists():
    _bl = [v_ for k_, v_ in json.loads(_tp.read_text())["table"].items() if k_.startswith("blood")]
    if _bl:
        BLOOD_MR = {"t1": float(np.mean([b_["t1_s"] for b_ in _bl])), "t2": float(np.mean([b_["t2_s"] for b_ in _bl]))}
# the reference frame of the heart: a moving lattice would have to cover the swept volume of the epicardial vessels
# (≈ 20× the lumen: 320 M nodes, 87 GB at 0.15 mm); with every outlet's flux prescribed and a loop-free tree the frame's
# accelerations (≈ 2 m/s²) only add pressure, so the flow relative to the vessel is solved on the static reference
# geometry and mapped to the lab by the cardiac motion (v = F v_rel + u_wall, phantom_4d)
glo = lo - 2e-3
ghi = lo + dims * HS + 2e-3
gdims = (np.ceil((ghi - glo) / DX) + 1).astype(int)
case = {
    "phantom": str(P0), "source": "rustlib/tools/phantom/lbm_coronary.py", "kind": "coronary",
    "grid": {"origin": glo.tolist(), "dx": DX, "dims": gdims.tolist()},
    "sdf": {"file": "sdf.npy", "origin": lo.tolist(), "h": HS},
    "motion": {"still": True, "period": period},
    "frame": {"kind": "reference", "motion": str((P0 / "motion" / "cardiac_motion.hv").resolve())},
    "fluid": {"nu": 3.5e-3 / 1060.0, "rho": 1060.0},
    "blood": BLOOD_MR,
    "v_max": float(opt("--v-max", 1.0)), "u_lattice": float(opt("--u-lattice", 0.08)),
    "valves": [], "cuts": cuts,
    "run": {"cycles": int(opt("--cycles", 6)), "warmup": int(opt("--warmup", 3)), "bins": int(opt("--bins", 50)), "out_factor": 2,
            "acc_every": 20, "geom_every": 4, "cwale": 0.5, "inv_iters": 3, "uclip": 0.35, "diag_every": 1000, "ckpt_every": 20000},
}
(OUT / "case.json").write_text(json.dumps(case, indent=1))
print(f"grid {gdims.tolist()} at {DX * 1e3:.2f} mm (reference frame) → {OUT / 'case.json'}")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
axs[0].plot(tt * 1e3, p_ao / MMHG, label="P_ao (network root)")
axs[0].plot(tt * 1e3, p_lv / MMHG, label="P_LV (valve phases)")
axs[0].plot(tt * 1e3, p_rv / MMHG, label="P_RV")
axs[0].set_xlabel("t [ms]")
axs[0].set_ylabel("mmHg")
axs[0].legend(frameon=False, fontsize=8)
for c, o in zip(cuts, lpm_out):
    axs[1].plot(tt * 1e3, -np.asarray(c["q"]["v"]) * 6e7, lw=0.8, color="#1f77b4" if o["left"] else "#d62728")
axs[1].set_xlabel("t [ms]")
axs[1].set_ylabel("outlet flow [mL/min] (blue left, red right)")
fig.tight_layout()
fig.savefig(OUT / "figs" / "coronary_lpm.png", dpi=110)
