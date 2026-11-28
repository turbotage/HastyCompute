#!/usr/bin/env bash
# full18 (2026-11-27): the full17 chain (full15 graph export, test phantom seed 1, clearance with plateau worst, adaptive
# carve rounds, solver sampling ≤ 2 mm) with the realism defaults of that day: SV = EF × the atlas EDV (--sv-mode ef),
# E share 0.65–0.80, the physio E / A filling — then the audit, the flow check and a 1 mm LBM burst with the sinus hinge.
# Into data/hasty/phantom/factory_test18 (full17 stays for comparison). Paths inside the repo only; logs in jobs/logs.
set -euo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
X=data/hasty/phantom/factory_test/export_full15
OUT=data/hasty/phantom/factory_test18
TP=$OUT/ph0001
LOCK=jobs/locks/gpu.lock
rm -rf "${TP:?}"
flock "$LOCK" $PY $T/make_phantom.py 1 --out $OUT --graph "$X" --truth "" --clearance --allow-overlap | grep -E "stroke volume|filling|clearance|cardiac motion|root mobility|done"
flock "$LOCK" $PY $T/clearance.py $TP --audit-only > $TP/clearance_audit.log 2>&1 || true
grep -E "^audit|^   pairs|^   within|^     [0-9]" $TP/clearance_audit.log | head -20
$PY $T/flow_check.py $TP | tail -12 || true
$T/lbm_burst.sh $TP data/hasty/scratch/lbm_full18 1 6 || true
rustlib/.venv/bin/python -I - <<'EOF'
import json
v = json.load(open("data/hasty/scratch/lbm_full18/run/report/report.json"))["valves"]
for n in ("aortic", "pulmonary", "mitral", "tricuspid"):
    x = v[n]
    print(f"{n:10s} jet {x['jet_max_m_s']:.2f} m/s at {x['jet_where']['t_ms']:.0f} ms; at peak flow disk mean {x['at_peak_flow']['un_mean_m_s']:.2f}, disk max {x['at_peak_flow']['un_max_m_s']:.2f} m/s; stroke {x['stroke_link_ml']:.1f} / {x['stroke_model_ml']:.1f} mL")
EOF
echo chain done
