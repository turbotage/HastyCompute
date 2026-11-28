#!/usr/bin/env bash
# (1) full18 at 0.5 mm with the membrane leaflets (all defaults): the production-resolution numbers;
# (2) the 1 mm membrane case of full17 (hfatcms) rerun with WALE off: does the eddy viscosity peak the semilunar profile?
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
S=data/hasty/scratch
L=jobs/logs
rustlib/tools/phantom/lbm_burst.sh data/hasty/phantom/factory_test18/ph0001 $S/lbm_full18_05m 0.5 6 > $L/lbm_full18_05m.log 2>&1
# WALE off on a copy of the hfatcms case
W=$S/lbm_full17_hfatcmsw
rm -rf "${W:?}"
mkdir -p $W
cp -r $S/lbm_full17_hfatcms/case $W/case
rustlib/.venv/bin/python -I - "$W/case/case.json" <<'PYEOF'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); c["run"]["cwale"] = 0.0; open(p, "w").write(json.dumps(c))
PYEOF
flock jobs/locks/gpu.lock systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm $W/case $W/run --cycles 6 --warmup 3 --quiet > $L/wale_off_run.log 2>&1
systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_report.py $W/case $W/run > $L/wale_off_report.log 2>&1
rustlib/.venv/bin/python -I - <<'PYEOF'
import json, os
for t in ("lbm_full18_05", "lbm_full18_05m", "lbm_full17_hfatcms", "lbm_full17_hfatcmsw"):
    f = f"data/hasty/scratch/{t}/run/report/report.json"
    if not os.path.exists(f):
        print(t, "missing"); continue
    v = json.load(open(f))["valves"]
    print("==", t)
    for n in ("aortic", "pulmonary", "mitral", "tricuspid"):
        x = v[n]; a = x["at_peak_flow"]; e = a.get("eccentricity", {}).get("1 diameter", {})
        print(f"  {n:10s} jet {x['jet_max_m_s']:.2f} at {x['jet_where']['t_ms']:.0f} ms p95 {x['jet_p95_peak_m_s']:.2f} | disk mean {a['un_mean_m_s']:.2f} max {a['un_max_m_s']:.2f} 12mm {a['jet_max_m_s']:.2f} | EOA {x.get('eoa_cm2', 0):.2f}/{x.get('goa_cm2', 0):.2f} | disp {e.get('displacement', 0):.2f} | stroke {x['stroke_link_ml']:.1f}/{x['stroke_model_ml']:.1f}")
PYEOF
grep -E "valve gate|density gate|tree balance" $L/lbm_full18_05m.log $L/wale_off_report.log | sed 's|^[^:]*:||' | cut -c1-200
echo membrane prod done
