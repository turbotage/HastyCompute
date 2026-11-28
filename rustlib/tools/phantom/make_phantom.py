"""Phantom factory: one seed → one complete moving flow phantom on the atlas backbone.

    python -I make_phantom.py <seed> [--out data/hasty/phantom/factory] [--truth gated_1mm,realtime_1mm] [--nominal]
        [--diffeo <preset|spec.json>[:seed]] [--valves <valve-disease spec, valve_disease.py>]
        [--vessel-variant <preset|spec.json, vessel_variants.py>]

Seed 9001 is the held-out phantom (never in training / validation); `--nominal` uses the reference parameters (the
P0 / P1 phantom). Sampled per phantom (params.json):
- heart: HR 55–85 bpm, LV EF 0.55–0.70 with SV = EF × the atlas LV EDV (--sv-mode draw: SV 60–85 mL, EF 0.54–0.76),
  E-wave share 0.65–0.80 (--fe-range), AV-plane descent
  11–16 mm, twist 6–12°;
- circulation: mean systemic 85–100 mmHg, mean pulmonary 12–18 mmHg, compliance ×0.75–1.3, territory fractions with
  15 % lognormal jitter;
- vessels: per vessel (name) radius × lognormal (σ 6 %), joints tapered (radius_taper.py; --radius-taper-mm 0: off),
  then the Murray floor and the network;
- breathing: period 3.2–5.5 s, cos^{2n} with n ∈ {1, 2, 3}, period / depth variability 5–15 % / 8–20 %, diaphragm
  10–20 mm, AP 4–9 mm, LR 2–4 mm;
- tissue (`tissue_variation.py`): jittered PD / T1 / T2 / T2* / fat fraction, contrast native / gadolinium /
  ferumoxytol, texture gain 0.5–1.5 and texture seed.
--diffeo: a diffeomorphic variant of the atlas first (`diffeo.py`: habitus / aneurysm / narrow presets or a spec):
the deformed label volumes and vessel graph go to <phantom>/atlas and replace p0 and the Z-Anatomy graph in every
later step (prep_sim, sim_truth and lbm_case pick <phantom>/atlas up by themselves).
Steps: graph copy + radius jitter → network (+ 2 Murray-floor rounds) → pack → cardiac motion → respiratory motion →
clearance (--clearance: no overlap at any instant: solve, network, pack, audit gate; --allow-overlap) → tissue parameters → moving truth (`phantom_4d.py`) for each requested `--truth` (gated_1mm, realtime_1mm,
gated_0.5mm_heart, gated_2mm).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any, exists_any, glob_any  # noqa: E402,F401  (.npy or the .hv written instead)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from tissue_variation import sample_tissue  # noqa: E402

A = sys.argv[1:]
SEED = int(A[0])


def opt(n, d=None):
    return A[A.index(n) + 1] if n in A else d


PY = [str(REPO / "rustlib/.venv/bin/python"), "-I"]
EXP = REPO / "data/hasty/external/zanatomy/export"
P0 = REPO / "data/hasty/phantom/p0"
TT = REPO / "rustlib/crates/hasty-phantom/data/atlas_tissues_3t.json"
D = Path(opt("--out", REPO / "data/hasty/phantom/factory")) / (f"ph{SEED:04d}" + ("_nominal" if "--nominal" in A else ""))
D.mkdir(parents=True, exist_ok=True)
t0 = time.time()
DIFFEO = opt("--diffeo")


def run_parallel(*cmds):
    """independent steps at once (e.g. cardiac and respiratory motion); each must succeed"""
    procs = [subprocess.Popen([str(a) for a in c], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for c in cmds]
    outs = []
    for c, p_ in zip(cmds, procs):
        o_, e_ = p_.communicate()
        if p_.returncode != 0:
            print(o_[-3000:], e_[-3000:])
            raise SystemExit(f"step failed: {[str(a) for a in c[:3]]}")
        outs.append(o_)
    return outs


def run(*args, quiet=True):
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"step failed: {args[:3]}")
    return r.stdout


# ── parameters ─────────────────────────────────────────────────────────────────────────────────────────────────────
# E-wave share of the LV filling volume: healthy adults (20–60 y) have E/A peak velocity ≈ 1–2, i.e. fe ≈ 0.65–0.80 with
# cardiac_motion's physio E/A shapes (2026-11-27); --fe-range 0.55,0.75 is the earlier range (its low end gives E/A < 1)
FE_RANGE = tuple(float(x) for x in (A[A.index("--fe-range") + 1] if "--fe-range" in A else "0.65,0.80").split(","))
rng = np.random.default_rng(SEED)
if "--nominal" in A:
    P = {"heart": {"hr": 70.0, "sv_ml": 75.0, "fe": 0.65, "dav_mm": 14.0, "twist_deg": 9.0},
         "circulation": {"p_sys": 93.0, "p_pul": 15.0, "ctot_scale": 1.0, "terr_sigma": 0.0},
         "vessels": {"radius_sigma": 0.0},
         "breathing": {"period_s": 4.0, "n": 2, "cv_period": 0.1, "cv_depth": 0.12, "dsi_mm": 15.0, "dap_mm": 7.0, "dlr_mm": 3.0}}
else:
    P = {"heart": {"hr": float(rng.uniform(55, 85)), "sv_ml": float(rng.uniform(60, 85)), "fe": float(rng.uniform(*FE_RANGE)),
                   "dav_mm": float(rng.uniform(11, 16)), "twist_deg": float(rng.uniform(6, 12))},
         "circulation": {"p_sys": float(rng.uniform(85, 100)), "p_pul": float(rng.uniform(12, 18)), "ctot_scale": float(rng.uniform(0.75, 1.3)), "terr_sigma": 0.15},
         "vessels": {"radius_sigma": 0.06},
         "breathing": {"period_s": float(rng.uniform(3.2, 5.5)), "n": int(rng.integers(1, 4)), "cv_period": float(rng.uniform(0.05, 0.15)),
                       "cv_depth": float(rng.uniform(0.08, 0.2)), "dsi_mm": float(rng.uniform(10, 20)), "dap_mm": float(rng.uniform(4, 9)), "dlr_mm": float(rng.uniform(2, 4))}}
P["seed"] = SEED
if DIFFEO:
    P["diffeo"] = {"spec": DIFFEO.split(":")[0], "seed": int(DIFFEO.split(":")[1]) if ":" in DIFFEO else SEED}
P["held_out"] = SEED == 9001
# --valves <spec> (valve_disease.py: e.g. "mr=0.3", "as=0.9,ar=0.15"): valve disease, consistent in the network (AR /
# PR root backflow), the cardiac model (MR / TR backflow, AV-stenosis filling) and the LBM case (orifice, hole)
import valve_disease as VD  # noqa: E402

VDIS = VD.parse(opt("--valves"))
if VDIS:
    P["pathology"] = {"valves": VDIS}
P["tissue"] = sample_tissue(rng, json.loads(TT.read_text())) if "--nominal" not in A else {"contrast": "native", "texture_gain": 0.0, "texture_seed": 0, "texture": {},
                                                                                               "table": {r[0]: dict(zip(json.loads(TT.read_text())["columns"], r)) for r in json.loads(TT.read_text())["tissues"]}}
(D / "params.json").write_text(json.dumps(P, indent=1))
(D / "tissue_params.json").write_text(json.dumps(P["tissue"], indent=1))
print(f"phantom {D.name}: HR {P['heart']['hr']:.0f}, SV {P['heart']['sv_ml']:.0f} mL, breathing {P['breathing']['period_s']:.1f} s / {P['breathing']['dsi_mm']:.0f} mm, contrast {P['tissue']['contrast']}")

# ── diffeomorphic atlas variant ──────────────────────────────────────────────────────────────────────────────────────
# --base DIR: a finished phantom as the source (its cleared graph and carved atlas; radii final: no jitter, no Murray
# floor) — with --diffeo: is a cleared geometry still clear after a diffeomorphism? (research/flow_phantom.md Tier B5)
BASE = Path(opt("--base")) if opt("--base") else None
assert BASE is None or DIFFEO, "--base needs --diffeo"
if DIFFEO:
    if not (D / "atlas" / "diffeo.json").exists():
        src = []
        if BASE is not None:
            src = ["--src-graph", BASE, "--src-atlas", BASE / "atlas" if (BASE / "atlas" / "tissue_1mm.hv").exists() else P0]
        elif opt("--graph"):
            src = ["--src-graph", opt("--graph")]
        print(run(*PY, HERE / "diffeo.py", D / "atlas", "--spec", P["diffeo"]["spec"], "--seed", P["diffeo"]["seed"], *src).strip().splitlines()[-6:])
    P0 = D / "atlas"
    EXP = D / "atlas"
    print(f"diffeo {DIFFEO}: gates {json.loads((D / 'atlas' / 'diffeo.json').read_text())['gates']} ({time.time() - t0:.0f}s)")

# ── stroke volume from the anatomy (--sv-mode ef, default 2026-11-27): the sampled SV's position in its 60–85 mL range
#    picks the LV ejection fraction in 0.55–0.70 (healthy adults ≈ 0.62 ± 0.05) and SV = EF × the LV EDV of this atlas /
#    diffeomorphism (minus the LV side's regurgitant volume); the RV ejects the same SV. --sv-mode draw: the sampled SV
#    as before (EF 0.54–0.76 on the atlas heart, 0.75 for seed 1)
SV_MODE = opt("--sv-mode", "ef")
assert SV_MODE in ("ef", "draw"), "--sv-mode ef | draw"
from hastycompute import value_format as _vf  # noqa: E402

_ts = _vf.read(P0 / "tissue_2mm.hv")
_nm = list(_ts["tissues"])
_L = np.asarray(_ts["labels"])
_h3 = float(_ts["voxel_m"]) ** 3
_edv = {c_: float((_L == _nm.index(c_)).sum() * _h3) for c_ in ("blood_LV", "blood_RV")}
del _ts, _L
_rv = VD.totals(VDIS, 60.0 / P["heart"]["hr"], P["circulation"]["p_sys"], P["circulation"]["p_pul"]) if VDIS else {}
if SV_MODE == "ef" and "ef_lv" not in P["heart"]:
    _u = float(np.clip((P["heart"]["sv_ml"] - 60.0) / 25.0, 0.0, 1.0))
    P["heart"]["ef_lv"] = 0.55 + 0.15 * _u
    P["heart"]["sv_ml_sampled"] = P["heart"]["sv_ml"]
    P["heart"]["sv_ml"] = (P["heart"]["ef_lv"] * _edv["blood_LV"] - _rv.get("mitral", 0) - _rv.get("aortic", 0)) * 1e6
    P["heart"]["edv_ml"] = {k_: v_ * 1e6 for k_, v_ in _edv.items()}
    (D / "params.json").write_text(json.dumps(P, indent=1))
    print(f"stroke volume: LV EF {P['heart']['ef_lv']:.2f} × EDV {_edv['blood_LV'] * 1e6:.0f} mL → forward SV {P['heart']['sv_ml']:.1f} mL "
          f"(RV EF {(P['heart']['sv_ml'] * 1e-6 + _rv.get('tricuspid', 0) + _rv.get('pulmonary', 0)) / _edv['blood_RV']:.2f}; sampled SV {P['heart']['sv_ml_sampled']:.0f} mL)")

# ── valve disease: the ventricle ejects SV + RVol from the atlas EDV; the forward SV is capped so that the
#    end-systolic volume stays ≥ 0.25 EDV (a larger regurgitation needs a dilated ventricle: --diffeo lv_overload)
if VDIS:
    _cap = min(0.75 * _edv["blood_LV"] - _rv.get("mitral", 0) - _rv.get("aortic", 0), 0.75 * _edv["blood_RV"] - _rv.get("tricuspid", 0) - _rv.get("pulmonary", 0)) * 1e6
    P["pathology"]["regurgitant_volume_ml"] = {k_: v_ * 1e6 for k_, v_ in _rv.items()}
    P["pathology"]["edv_ml"] = {k_: v_ * 1e6 for k_, v_ in _edv.items()}
    if P["heart"]["sv_ml"] > _cap:
        if _cap < 30.0:
            raise SystemExit(f"valve disease {VD.describe(VDIS)}: forward SV would be {_cap:.0f} mL with EDV {_edv['blood_LV'] * 1e6:.0f} / {_edv['blood_RV'] * 1e6:.0f} mL — dilate the ventricle (--diffeo lv_overload) or lower the regurgitation")
        P["heart"]["sv_ml_sampled"] = P["heart"]["sv_ml"]
        P["heart"]["sv_ml"] = _cap
    (D / "params.json").write_text(json.dumps(P, indent=1))
    print(f"valve disease: {VD.describe(VDIS, 60.0 / P['heart']['hr'], P['circulation']['p_sys'], P['circulation']['p_pul'])}; EDV LV {_edv['blood_LV'] * 1e6:.0f} mL, forward SV {P['heart']['sv_ml']:.0f} mL")

# the Murray floor tapers the joints after flooring (--radius-taper-mm 0: neither taper, as before 2026-11-28)
MTAPER = ["--taper-mm", "0"] if float(opt("--radius-taper-mm", 15.0)) == 0 else []
# ── vessels: graph copy with radius jitter ─────────────────────────────────────────────────────────────────────────
# --graph DIR: the vessel graph from DIR instead of the Z-Anatomy export (a rebuilt graph under test)
GRAPH = EXP if DIFFEO else (BASE if BASE is not None else Path(opt("--graph", EXP)))  # with --diffeo: the deformed graph
g = dict(load_any(GRAPH / "graph.npz"))
gj = json.loads((GRAPH / "graph.json").read_text())
# --vessel-variant <preset|spec.json> (vessel_variants.py: circle-of-Willis variants, hypoplastic / absent / added
# segments): applied to the copy before the jitter, the network and the clearance solver
if opt("--vessel-variant") and BASE is None:
    import vessel_variants as VV  # noqa: E402

    _vs = VV.load_spec(opt("--vessel-variant"))
    g, gj = VV.apply(g, gj, _vs)
    P.setdefault("pathology", {})["vessels"] = {"spec": opt("--vessel-variant"), "ops": _vs["ops"]}
    (D / "params.json").write_text(json.dumps(P, indent=1))
g.pop("radius_atlas", None)
if P["vessels"]["radius_sigma"] > 0 and BASE is None:
    names = gj["edge_names"]
    base = [n.split(": ", 1)[1] if n.startswith(("connector: ", "join: ", "bridge: ")) else n for n in names]
    fac = {n: float(np.exp(rng.normal(0, P["vessels"]["radius_sigma"]))) for n in sorted(set(base))}
    eoff = g["edge_offsets"]
    r = g["radius"].astype(np.float64)
    for k in range(len(names)):
        r[eoff[k] : eoff[k + 1]] *= fac[base[k]]
    # each name has its own factor: where one name continues another the jitter stepped the lumen (IVC thoracic ×0.91 |
    # abdominal ×1.14: 8.2 | 10.1 mm in factory_test19) — taper the joints again (radius_taper.joint_taper, 2026-11-28;
    # --radius-taper-mm 0: off, as before)
    if float(opt("--radius-taper-mm", 15.0)) > 0:
        from radius_taper import joint_taper  # noqa: E402

        r, _trep = joint_taper(g["points"], r, eoff, g["edge_a"], g["edge_b"], len(g["node_pos"]), length=float(opt("--radius-taper-mm", 15.0)) * 1e-3, names=names, fixed=g.get("variant_fixed"))
        print(f"radius taper after the jitter: {len(_trep)} joints", flush=True)
    g["radius"] = r.astype(np.float32)
np.savez_compressed(D / "graph.npz", **g)
(D / "graph.json").write_text(json.dumps(gj))
# network, Murray floor
h, c = P["heart"], P["circulation"]
net = ["--hr", h["hr"], "--sv", h["sv_ml"], "--p-sys", c["p_sys"], "--p-pul", c["p_pul"], "--ctot-scale", c["ctot_scale"], "--terr-sigma", c["terr_sigma"], "--seed", SEED]
VD_ARGS = ["--valve-disease", json.dumps(VDIS), "--p-sys", c["p_sys"], "--p-pul", c["p_pul"]] if VDIS else []
net += VD_ARGS[:2]
run(*PY, HERE / "network_flow.py", D, D / "flow", *net)
for _ in range(2 if BASE is None else 0):
    run(*PY, HERE / "murray_floor.py", D, D / "flow", *MTAPER)
    out = run(*PY, HERE / "network_flow.py", D, D / "flow", *net)
gates = json.loads((D / "flow" / "network_flow.json").read_text())
print(f"network: residual {gates['solve_residual_max']:.1e}, Kirchhoff {gates['kirchhoff_dc_rel_max']:.1e}, root pressures {gates['mean_root_pressure_mmHg']} ({time.time() - t0:.0f}s)")
run(*PY, HERE / "pack_vessels.py", D, D / "flow", D / "vessels.hv")
# motion
b = P["breathing"]
run_parallel((*PY, HERE / "cardiac_motion.py", P0 / "tissue_2mm.hv", TT, D, D / "flow", D / "motion", "--dav", h["dav_mm"], "--twist", h["twist_deg"], "--fe", h["fe"], *VD_ARGS),
             (*PY, HERE / "resp_motion.py", P0 / "tissue_2mm.hv", D / "motion", "--dsi", b["dsi_mm"], "--dap", b["dap_mm"], "--dlr", b["dlr_mm"], "--graph", D / "graph.npz"))
cmj = json.loads((D / "motion" / "cardiac_motion.json").read_text())
print(f"cardiac motion: EF {cmj['ef']}, fit {cmj['volume_fit_max_err_ml']:.1e} mL, myocardium {cmj['myocardium_volume_change_pct']:.2f} % ({time.time() - t0:.0f}s)")
# clearance (research/flow_phantom.md Tier B5): no vessel lumen overlaps another vessel, a heart chamber or the
# myocardium at any instant of this phantom's motion. The solve moves centrelines / shrinks radii in graph.npz; the
# network and the packing follow (positions changed; the Murray floors are not re-applied, they would undo shrinks);
# the audit (every sample, every instant) gates the phantom. Opt-in `--clearance` until validated (then the default);
# `--allow-overlap`: report only.
if "--clearance" in A:
    def _clear(*extra):
        r_ = subprocess.run([str(a) for a in (*PY, HERE / "clearance.py", D, *extra)], capture_output=True, text=True)
        print(f"clearance {' '.join(extra) or 'solve'}: " + " | ".join(x for x in r_.stdout.splitlines() if x.startswith(("before", "solve", "audit", "carved"))) + f" ({time.time() - t0:.0f}s)")
        return r_
    _clear()
    # the chambers yield where a great vessel lies in another chamber (user 2026-11-25): carve, refit the motion to the
    # carved labels (<phantom>/atlas, used by every later step), solve again. The refit moves the heart a little
    # differently (and the solve moves vessels), so the carve is repeated until no tissue contact is left
    # (--carve-rounds, 3)
    # --carve-rounds auto (default, 2026-11-26): up to 6 rounds until no tissue violation remains, stopping after two
    # rounds without a better worst tissue depth (the aneurysm variant's second round got worse, the third better);
    # an integer: that many rounds (as before: 3)
    _cr = opt("--carve-rounds", "auto")
    _nmax = 6 if _cr == "auto" else int(_cr)
    _best, _stall = np.inf, 0
    for _round in range(_nmax):
        _cj = json.loads((D / "clearance.json").read_text())["after"]
        if not _cj["tissue"]:
            break
        if _cr == "auto":
            _w = float(_cj.get("tissue_max_mm", np.inf))
            _stall = 0 if _w < _best - 1e-3 else _stall + 1
            _best = min(_best, _w)
            if _stall >= 2:
                break
        _clear("--audit-only", "--carve")
        P0 = D / "atlas"
        run_parallel((*PY, HERE / "cardiac_motion.py", P0 / "tissue_2mm.hv", TT, D, D / "flow", D / "motion", "--dav", h["dav_mm"], "--twist", h["twist_deg"], "--fe", h["fe"], *VD_ARGS),
                     (*PY, HERE / "resp_motion.py", P0 / "tissue_2mm.hv", D / "motion", "--dsi", b["dsi_mm"], "--dap", b["dap_mm"], "--dlr", b["dlr_mm"], "--graph", D / "graph.npz"))
        _clear()
    run(*PY, HERE / "network_flow.py", D, D / "flow", *net)
    run(*PY, HERE / "pack_vessels.py", D, D / "flow", D / "vessels.hv")
    r_ = _clear("--audit-only")
    if r_.returncode != 0 and "--allow-overlap" not in A:
        print(r_.stdout[-3000:], r_.stderr[-2000:])
        raise SystemExit(f"clearance gate failed for {D} (clearance.json; --allow-overlap to continue)")
# moving truth
TRUTH = {"gated_1mm": ["--labels-src", P0 / "tissue_1mm.hv", "--vox", 1, "--frames", 20, "--viewer-step", 2],
         "realtime_1mm": ["--labels-src", P0 / "tissue_1mm.hv", "--vox", 1, "--frames", 32, "--realtime", "0,8", "--seed", SEED, "--viewer-step", 2],
         "gated_0.5mm_heart": ["--labels-src", P0 / "tissue_0.5mm_heart.hv", "--vox", 0.5, "--frames", 20, "--crop", "-68,108,-118,58,1202,1418"],
         "gated_2mm": ["--labels-src", P0 / "tissue_1mm.hv", "--vox", 2, "--frames", 20]}
for t in (opt("--truth", "gated_1mm").split(",") if opt("--truth", "gated_1mm") else []):
    run(*PY, HERE / "phantom_4d.py", D, D / "motion", D / t, *TRUTH[t])
    gt = json.loads((D / t / "gates.json").read_text())["frames"]
    imb = max(abs(cc["imbalance_ml_s"]) for f in gt for cc in f["components"])
    mv = [abs(f["mitral_flux_ml_s"] - f["Q_MV_ml_s"]) / f["Q_MV_ml_s"] for f in gt if f["Q_MV_ml_s"] > 50]
    print(f"truth {t}: max imbalance {imb:.1f} mL/s, mitral err {max(mv) * 100 if mv else 0:.2f} %, chamber p99.9 {max(f['chamber_speed_m_s']['p999'] for f in gt):.2f} m/s ({time.time() - t0:.0f}s)")
print(f"phantom {D} done ({time.time() - t0:.0f}s)")
