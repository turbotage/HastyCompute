#!/usr/bin/env bash
# Grid refinement, the WALE filter-width ramp (2026-11-28): refine_prod.sh's case (10 mm downstream reach: the plane one
# diameter downstream at the boundary, pulmonary displacement 0.03 → 0.12) with run.refine_wale_ramp 8 (Δ² from the
# coarse value at the interface to the fine one over 8 fine nodes): is the deflection the 4× jump of ν_t? Against 05r / 05rd.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
S=data/hasty/scratch
L=jobs/logs
O=$S/lbm_full18_05rw
flock jobs/locks/gpu.lock systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm $O/case $O/run --cycles 6 --warmup 3 --quiet > $L/refine_ramp_run.log 2>&1 || { echo "cardiac_lbm failed"; tail -5 $L/refine_ramp_run.log; exit 1; }
tail -2 $L/refine_ramp_run.log
systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_report.py $O/case $O/run > $L/refine_ramp_report.log 2>&1 || { echo "lbm_report failed"; tail -5 $L/refine_ramp_report.log; exit 1; }
grep -E "^(valve gate|density gate|tree balance)" $L/refine_ramp_report.log | cut -c1-300
rustlib/.venv/bin/python -I - <<'PYEOF'
import json, os
for t in ("lbm_full18_05m", "lbm_full18_05r", "lbm_full18_05rd", "lbm_full18_05rw"):
    f = f"data/hasty/scratch/{t}/run/report/report.json"
    if not os.path.exists(f):
        print(t, "missing"); continue
    r = json.load(open(f))
    v = r["valves"]
    print("==", t)
    for n in ("aortic", "pulmonary", "mitral", "tricuspid"):
        x = v[n]; a = x["at_peak_flow"]; e = a.get("eccentricity", {}).get("1 diameter", {})
        ang = " ".join(f"{k_}: disp {y_['displacement']:.3f} angle {y_['jet_angle_deg']:.1f} A50 {y_['half_flux_area_share']:.2f}" for k_, y_ in a.get("eccentricity", {}).items())
        print(f"    {n} {ang}")
        print(f"  {n:10s} jet {x['jet_max_m_s']:.2f} at {x['jet_where']['t_ms']:.0f} ms p95 {x['jet_p95_peak_m_s']:.2f} | disk mean {a['un_mean_m_s']:.2f} max {a['un_max_m_s']:.2f} (max/mean {a['un_max_m_s'] / max(a['un_mean_m_s'], 1e-9):.2f}) 12mm {a['jet_max_m_s']:.2f} | EOA {x.get('eoa_cm2', 0):.2f}/{x.get('goa_cm2', 0):.2f} | disp {e.get('displacement', 0):.2f} | stroke {x['stroke_link_ml']:.1f}/{x['stroke_model_ml']:.1f}")
PYEOF
echo refine prod done
