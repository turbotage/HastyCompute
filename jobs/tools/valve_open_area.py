"""Open area of a case's valves at their opening angle, with lbm_case.py's own leaflet_polys / open_area (extracted by
ast, so the numbers are the case builder's), against the annulus and healthy references.
   rustlib/.venv/bin/python -I jobs/tools/valve_open_area.py <case.json> [dx_mm=1]"""
import ast, json, sys
from pathlib import Path
import numpy as np

src = (Path(__file__).resolve().parents[2] / "rustlib/tools/phantom/lbm_case.py").read_text()
mod = ast.parse(src)
ns = {"np": np}
for node in mod.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("leaflet_polys", "open_area"):
        exec(compile(ast.Module([node], []), "lbm_case", "exec"), ns)
case = json.loads(Path(sys.argv[1]).read_text())
dx = float(sys.argv[2]) * 1e-3 if len(sys.argv) > 2 else 1e-3
# healthy adult effective / geometric orifice areas (cm²) for reference (echo: AVA 3–4, PV ≈ 2–3 normalised… MVA 4–6, TVA 6–8)
ref = {"aortic": "3–4", "pulmonary": "3–4", "mitral": "4–6", "tricuspid": "6–8"}
for v in case["valves"]:
    R = v["radius"]
    polys = ns["leaflet_polys"](v["leaflets"])
    th_r = (v["thickness"] + dx) / R
    ri = v.get("lumen_radius", R) / R
    a = ns["open_area"](polys, v["theta_open"], th_r, r_in=ri) * np.pi * R**2 * 1e4
    a_thin = ns["open_area"](polys, v["theta_open"], (1e-3 + dx) / R, r_in=ri) * np.pi * R**2 * 1e4
    print(f"{v['name']:10s} R {R*1e3:5.1f} mm  annulus {np.pi*R*R*1e4:5.2f} cm²  open {a:5.2f} cm² at {np.degrees(v['theta_open']):.0f}° "
          f"(thickness {v['thickness']*1e3:.1f} mm + 1 cell; with 1 mm leaflets {a_thin:5.2f})  ring {v['ring']*1e3:.1f} mm  healthy {ref.get(v['name'], '?')} cm²")
