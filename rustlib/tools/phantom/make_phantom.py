"""Phantom factory: one seed → one complete moving flow phantom on the atlas backbone.

    python -I make_phantom.py <seed> [--out data/hasty/phantom/factory] [--truth gated_1mm,realtime_1mm] [--nominal]
        [--diffeo <preset|spec.json>[:seed]]

Seed 9001 is the held-out phantom (never in training / validation); `--nominal` uses the reference parameters (the
P0 / P1 phantom). Sampled per phantom (params.json):
- heart: HR 55–85 bpm, SV 60–85 mL (EDV = the atlas LV, so EF 0.54–0.76), E-wave share 0.55–0.75, AV-plane descent
  11–16 mm, twist 6–12°;
- circulation: mean systemic 85–100 mmHg, mean pulmonary 12–18 mmHg, compliance ×0.75–1.3, territory fractions with
  15 % lognormal jitter;
- vessels: per vessel (name) radius × lognormal (σ 6 %), then the Murray floor and the network;
- breathing: period 3.2–5.5 s, cos^{2n} with n ∈ {1, 2, 3}, period / depth variability 5–15 % / 8–20 %, diaphragm
  10–20 mm, AP 4–9 mm, LR 2–4 mm;
- tissue (`tissue_variation.py`): jittered PD / T1 / T2 / T2* / fat fraction, contrast native / gadolinium /
  ferumoxytol, texture gain 0.5–1.5 and texture seed.
--diffeo: a diffeomorphic variant of the atlas first (`diffeo.py`: habitus / aneurysm / narrow presets or a spec):
the deformed label volumes and vessel graph go to <phantom>/atlas and replace p0 and the Z-Anatomy graph in every
later step (prep_sim, sim_truth and lbm_case pick <phantom>/atlas up by themselves).
Steps: graph copy + radius jitter → network (+ 2 Murray-floor rounds) → pack → cardiac motion → respiratory motion →
tissue parameters → moving truth (`phantom_4d.py`) for each requested `--truth` (gated_1mm, realtime_1mm,
gated_0.5mm_heart, gated_2mm).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

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


def run(*args, quiet=True):
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"step failed: {args[:3]}")
    return r.stdout


# ── parameters ─────────────────────────────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
if "--nominal" in A:
    P = {"heart": {"hr": 70.0, "sv_ml": 75.0, "fe": 0.65, "dav_mm": 14.0, "twist_deg": 9.0},
         "circulation": {"p_sys": 93.0, "p_pul": 15.0, "ctot_scale": 1.0, "terr_sigma": 0.0},
         "vessels": {"radius_sigma": 0.0},
         "breathing": {"period_s": 4.0, "n": 2, "cv_period": 0.1, "cv_depth": 0.12, "dsi_mm": 15.0, "dap_mm": 7.0, "dlr_mm": 3.0}}
else:
    P = {"heart": {"hr": float(rng.uniform(55, 85)), "sv_ml": float(rng.uniform(60, 85)), "fe": float(rng.uniform(0.55, 0.75)),
                   "dav_mm": float(rng.uniform(11, 16)), "twist_deg": float(rng.uniform(6, 12))},
         "circulation": {"p_sys": float(rng.uniform(85, 100)), "p_pul": float(rng.uniform(12, 18)), "ctot_scale": float(rng.uniform(0.75, 1.3)), "terr_sigma": 0.15},
         "vessels": {"radius_sigma": 0.06},
         "breathing": {"period_s": float(rng.uniform(3.2, 5.5)), "n": int(rng.integers(1, 4)), "cv_period": float(rng.uniform(0.05, 0.15)),
                       "cv_depth": float(rng.uniform(0.08, 0.2)), "dsi_mm": float(rng.uniform(10, 20)), "dap_mm": float(rng.uniform(4, 9)), "dlr_mm": float(rng.uniform(2, 4))}}
P["seed"] = SEED
if DIFFEO:
    P["diffeo"] = {"spec": DIFFEO.split(":")[0], "seed": int(DIFFEO.split(":")[1]) if ":" in DIFFEO else SEED}
P["held_out"] = SEED == 9001
P["tissue"] = sample_tissue(rng, json.loads(TT.read_text())) if "--nominal" not in A else {"contrast": "native", "texture_gain": 0.0, "texture_seed": 0, "texture": {},
                                                                                               "table": {r[0]: dict(zip(json.loads(TT.read_text())["columns"], r)) for r in json.loads(TT.read_text())["tissues"]}}
(D / "params.json").write_text(json.dumps(P, indent=1))
(D / "tissue_params.json").write_text(json.dumps(P["tissue"], indent=1))
print(f"phantom {D.name}: HR {P['heart']['hr']:.0f}, SV {P['heart']['sv_ml']:.0f} mL, breathing {P['breathing']['period_s']:.1f} s / {P['breathing']['dsi_mm']:.0f} mm, contrast {P['tissue']['contrast']}")

# ── diffeomorphic atlas variant ──────────────────────────────────────────────────────────────────────────────────────
if DIFFEO:
    if not (D / "atlas" / "diffeo.json").exists():
        print(run(*PY, HERE / "diffeo.py", D / "atlas", "--spec", P["diffeo"]["spec"], "--seed", P["diffeo"]["seed"]).strip().splitlines()[-6:])
    P0 = D / "atlas"
    EXP = D / "atlas"
    print(f"diffeo {DIFFEO}: gates {json.loads((D / 'atlas' / 'diffeo.json').read_text())['gates']} ({time.time() - t0:.0f}s)")

# ── vessels: graph copy with radius jitter ─────────────────────────────────────────────────────────────────────────
g = dict(np.load(EXP / "graph.npz"))
gj = json.loads((EXP / "graph.json").read_text())
g.pop("radius_atlas", None)
if P["vessels"]["radius_sigma"] > 0:
    names = gj["edge_names"]
    base = [n.split(": ", 1)[1] if n.startswith(("connector: ", "join: ", "bridge: ")) else n for n in names]
    fac = {n: float(np.exp(rng.normal(0, P["vessels"]["radius_sigma"]))) for n in sorted(set(base))}
    eoff = g["edge_offsets"]
    r = g["radius"].astype(np.float64)
    for k in range(len(names)):
        r[eoff[k] : eoff[k + 1]] *= fac[base[k]]
    g["radius"] = r.astype(np.float32)
np.savez_compressed(D / "graph.npz", **g)
(D / "graph.json").write_text(json.dumps(gj))
# network, Murray floor
h, c = P["heart"], P["circulation"]
net = ["--hr", h["hr"], "--sv", h["sv_ml"], "--p-sys", c["p_sys"], "--p-pul", c["p_pul"], "--ctot-scale", c["ctot_scale"], "--terr-sigma", c["terr_sigma"], "--seed", SEED]
run(*PY, HERE / "network_flow.py", D, D / "flow", *net)
for _ in range(2):
    run(*PY, HERE / "murray_floor.py", D, D / "flow")
    out = run(*PY, HERE / "network_flow.py", D, D / "flow", *net)
gates = json.loads((D / "flow" / "network_flow.json").read_text())
print(f"network: residual {gates['solve_residual_max']:.1e}, Kirchhoff {gates['kirchhoff_dc_rel_max']:.1e}, root pressures {gates['mean_root_pressure_mmHg']} ({time.time() - t0:.0f}s)")
run(*PY, HERE / "pack_vessels.py", D, D / "flow", D / "vessels.hv")
# motion
run(*PY, HERE / "cardiac_motion.py", P0 / "tissue_2mm.hv", TT, D, D / "flow", D / "motion", "--dav", h["dav_mm"], "--twist", h["twist_deg"], "--fe", h["fe"])
cmj = json.loads((D / "motion" / "cardiac_motion.json").read_text())
print(f"cardiac motion: EF {cmj['ef']}, fit {cmj['volume_fit_max_err_ml']:.1e} mL, myocardium {cmj['myocardium_volume_change_pct']:.2f} % ({time.time() - t0:.0f}s)")
b = P["breathing"]
run(*PY, HERE / "resp_motion.py", P0 / "tissue_2mm.hv", D / "motion", "--dsi", b["dsi_mm"], "--dap", b["dap_mm"], "--dlr", b["dlr_mm"])
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
