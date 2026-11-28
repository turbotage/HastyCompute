"""A valve-disease variant of a finished phantom for LBM checks: same geometry (atlas hard-linked, graph copied),
network + cardiac model rerun with the disease (forward SV capped as make_phantom does). Not a gated phantom: the
clearance audit is not rerun.
    python -I vd_variant.py <src_phantom> <dst_phantom> <spec>"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
REPO = Path.cwd()
sys.path.insert(0, str(REPO / "python/lib")); sys.path.insert(0, str(REPO / "rustlib/tools/phantom"))
from hastycompute import value_format as vf
import valve_disease as VD
S, D, spec = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
D.mkdir(parents=True, exist_ok=True)
if not (D / "atlas").exists():
    subprocess.run(["cp", "-al", str(S / "atlas"), str(D / "atlas")], check=True)
for f in ("graph.npz", "graph.json", "tissue_params.json"):
    shutil.copy(S / f, D / f)
P = json.loads((S / "params.json").read_text())
VDIS = VD.parse(spec)
P["pathology"] = {"valves": VDIS}
ts = vf.read(str(D / "atlas" / "tissue_2mm.hv")); nm = list(ts["tissues"]); L = np.asarray(ts["labels"]); h3 = float(ts["voxel_m"]) ** 3
edv = {c: float((L == nm.index(c)).sum() * h3) for c in ("blood_LV", "blood_RV")}
h, c = P["heart"], P["circulation"]
rv = VD.totals(VDIS, 60 / h["hr"], c["p_sys"], c["p_pul"])
cap = min(0.75 * edv["blood_LV"] - rv.get("mitral", 0) - rv.get("aortic", 0), 0.75 * edv["blood_RV"] - rv.get("tricuspid", 0) - rv.get("pulmonary", 0)) * 1e6
if h["sv_ml"] > cap:
    h["sv_ml_sampled"], h["sv_ml"] = h["sv_ml"], cap
(D / "params.json").write_text(json.dumps(P, indent=1))
print(f"{VD.describe(VDIS, 60 / h['hr'], c['p_sys'], c['p_pul'])}; EDV {edv['blood_LV'] * 1e6:.0f} mL; forward SV {h['sv_ml']:.1f} mL")
PY = [str(REPO / "rustlib/.venv/bin/python"), "-I"]; T = REPO / "rustlib/tools/phantom"
vda = ["--valve-disease", json.dumps(VDIS)]
net = ["--hr", h["hr"], "--sv", h["sv_ml"], "--p-sys", c["p_sys"], "--p-pul", c["p_pul"], "--ctot-scale", c["ctot_scale"], "--terr-sigma", c["terr_sigma"], "--seed", P["seed"], *vda]
r = subprocess.run([*PY, T / "network_flow.py", D, D / "flow", *map(str, net)], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-2000:]
TT = REPO / "rustlib/crates/hasty-phantom/data/atlas_tissues_3t.json"
r = subprocess.run([*PY, T / "cardiac_motion.py", D / "atlas" / "tissue_2mm.hv", TT, D, D / "flow", D / "motion", "--dav", str(h["dav_mm"]), "--twist", str(h["twist_deg"]), "--fe", str(h["fe"]), *vda, "--p-sys", str(c["p_sys"]), "--p-pul", str(c["p_pul"])], capture_output=True, text=True)
print(r.stdout[-1500:] if r.returncode == 0 else r.stderr[-3000:])
r = subprocess.run([*PY, T / "pack_vessels.py", D, D / "flow", D / "vessels.hv"], capture_output=True, text=True)
print("pack:", "ok" if r.returncode == 0 else r.stderr[-2000:])
