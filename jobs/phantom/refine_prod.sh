#!/usr/bin/env bash
# Grid refinement on the production case (2026-11-27): full18 at 0.5 mm with membranes (lbm_full18_05m, all defaults)
# + level 1 (0.25 mm) around the aortic and pulmonary valves along their path (refine_spheres.py, margin 6 mm, 10 mm
# downstream) — the one changed factor. Same run (6 cycles, warmup 3), report, then the valves of both side by side.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
S=data/hasty/scratch
L=jobs/logs
O=$S/lbm_full18_05r
flock jobs/locks/gpu.lock systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm $O/case $O/run --cycles 6 --warmup 3 --quiet > $L/refine_prod_run.log 2>&1 || { echo "cardiac_lbm failed"; tail -5 $L/refine_prod_run.log; exit 1; }
tail -2 $L/refine_prod_run.log
systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_report.py $O/case $O/run > $L/refine_prod_report.log 2>&1 || { echo "lbm_report failed"; tail -5 $L/refine_prod_report.log; exit 1; }
grep -E "^(valve gate|density gate|tree balance)" $L/refine_prod_report.log | cut -c1-300
rustlib/.venv/bin/python -I - <<'PYEOF'
import json, os
for t in ("lbm_full18_05m", "lbm_full18_05r"):
    f = f"data/hasty/scratch/{t}/run/report/report.json"
    if not os.path.exists(f):
        print(t, "missing"); continue
    r = json.load(open(f))
    v = r["valves"]
    print("==", t)
    for n in ("aortic", "pulmonary", "mitral", "tricuspid"):
        x = v[n]; a = x["at_peak_flow"]; e = a.get("eccentricity", {}).get("1 diameter", {})
        print(f"  {n:10s} jet {x['jet_max_m_s']:.2f} at {x['jet_where']['t_ms']:.0f} ms p95 {x['jet_p95_peak_m_s']:.2f} | disk mean {a['un_mean_m_s']:.2f} max {a['un_max_m_s']:.2f} (max/mean {a['un_max_m_s'] / max(a['un_mean_m_s'], 1e-9):.2f}) 12mm {a['jet_max_m_s']:.2f} | EOA {x.get('eoa_cm2', 0):.2f}/{x.get('goa_cm2', 0):.2f} | disp {e.get('displacement', 0):.2f} | stroke {x['stroke_link_ml']:.1f}/{x['stroke_model_ml']:.1f}")
PYEOF
echo refine prod done
