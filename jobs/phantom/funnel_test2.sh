#!/usr/bin/env bash
# Outflow-tract funnel (lbm_case --ot-funnel, default 20,1.6) on full17 with the sinus hinge and the physio motion, 1 mm,
# 6 cycles, against lbm_full17_hf (the same without the funnel). Case build, slices of both domains, LBM, report, jets.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
PH=data/hasty/phantom/factory_test/ph0001
S=data/hasty/scratch
L=jobs/logs
TAG=${1:-hff}; shift || true; O=$S/lbm_full17_$TAG
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
LOCK=jobs/locks/gpu.lock
rm -rf "${O:?}"
mkdir -p "$O"
$PY $T/lbm_case.py $PH $O/case --dx-mm 1 --motion $S/cm_physio/cardiac_motion.hv "$@" > $L/funnel_${TAG}_case.log 2>&1 || { echo "lbm_case failed"; tail -5 $L/funnel_${TAG}_case.log; exit 1; }
grep -E "funnel|mouth|sealing|lumen gate|open .* cm² of the" $L/funnel_${TAG}_case.log | cut -c1-220
rustlib/.venv/bin/python -I jobs/tools/valve_slice.py $O/slices_aortic.png $S/lbm_full17_hf/case $O/case --valve aortic --with mitral
rustlib/.venv/bin/python -I jobs/tools/valve_slice.py $O/slices_pulmonary.png $S/lbm_full17_hf/case $O/case --valve pulmonary --with tricuspid
flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm $O/case $O/run --cycles 6 --warmup 3 --quiet > $L/funnel_${TAG}_run.log 2>&1 || { echo "cardiac_lbm failed"; tail -5 $L/funnel_${TAG}_run.log; exit 1; }
tail -1 $L/funnel_${TAG}_run.log
$PY $T/lbm_report.py $O/case $O/run --flow-slices > $L/funnel_${TAG}_report.log 2>&1 || { echo "lbm_report failed"; tail -5 $L/funnel_${TAG}_report.log; exit 1; }
grep -E "^(valve gate|density gate|tree balance)" $L/funnel_${TAG}_report.log | cut -c1-300
rustlib/.venv/bin/python -I - lbm_full17_$TAG <<'EOF'
import json
import sys, os
for t in ["lbm_full17_hf", "lbm_full17_hff"] + [a for a in sys.argv[1:] if a != "lbm_full17_hff"]:
    if not os.path.exists(f"data/hasty/scratch/{t}/run/report/report.json"): continue
    v = json.load(open(f"data/hasty/scratch/{t}/run/report/report.json"))["valves"]
    print("==", t)
    for n in ("aortic", "pulmonary", "mitral", "tricuspid"):
        x = v[n]; a = x["at_peak_flow"]; e = a["eccentricity"]
        print(f"  {n:10s} jet {x['jet_max_m_s']:.2f} at {x['jet_where']['t_ms']:.0f} ms | peak flow: disk mean {a['un_mean_m_s']:.2f}, disk max {a['un_max_m_s']:.2f}, 12 mm {a['jet_max_m_s']:.2f} | "
              + " ; ".join(f"{k} disp {y['displacement']:.2f} angle {y['jet_angle_deg']:.0f}° A50 {y['half_flux_area_share']:.2f}" for k, y in e.items()) + f" | stroke {x['stroke_link_ml']:.1f}")
EOF
echo funnel test done
