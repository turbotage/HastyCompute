#!/usr/bin/env bash
# Valve / filling realism on the full17 test phantom, one factor at a time against the full17 LBM baseline
# (data/hasty/scratch/lbm_full17: filling sin2 with fe 0.579, semilunar hinge "annulus"):
#   h   semilunar hinge "lumen" (open cusps outside the lumen), the phantom's own (sin2) motion
#   hf  + the physio E/A filling with fe 0.672 (seed 1's draw under the new 0.65–0.80 range)
# 1 mm, 6 cycles (3 warm-up), wall pump + pump gain. Outputs in data/hasty/scratch (program output), logs in jobs/logs.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
PH=data/hasty/phantom/factory_test/ph0001
S=data/hasty/scratch
L=jobs/logs
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
CM=$S/cm_physio
rm -rf "${CM:?}" "${S:?}/lbm_full17_h" "${S:?}/lbm_full17_hf"
$PY $T/cardiac_motion.py $PH/atlas/tissue_2mm.hv rustlib/crates/hasty-phantom/data/atlas_tissues_3t.json $PH $PH/flow $CM \
  --dav 15.743 --twist 7.871 --fe 0.6716 > $L/cm_physio.log 2>&1 || { echo "cardiac_motion failed"; tail -5 $L/cm_physio.log; exit 1; }
grep -E "filling|EF" $L/cm_physio.log | head -4
for t in h hf; do
  ARGS=""
  [[ $t == hf ]] && ARGS="--motion $CM/cardiac_motion.hv"
  CASE_ARGS="$ARGS" $T/lbm_burst.sh $PH $S/lbm_full17_$t 1 6 > $L/lbm_full17_$t.log 2>&1
  echo "== $t"; grep -E "open .* cm² of the|valve gate|tree balance|density gate" $S/lbm_full17_$t/case.log $L/lbm_full17_$t.log | sed 's|^[^:]*:||' | cut -c1-260
done
rustlib/.venv/bin/python -I - <<'EOF'
import json
for t in ["lbm_full17", "lbm_full17_h", "lbm_full17_hf"]:
    try:
        v = json.load(open(f"data/hasty/scratch/{t}/run/report/report.json"))["valves"]
    except Exception as e:
        print(t, "no report", e); continue
    print(f"{t:15s} " + "  ".join(f"{n} jet {v[n]['jet_max_m_s']:.2f} (p95 {v[n]['jet_p95_peak_m_s']:.2f}) m/s, peak q {v[n]['peak_q_model_ml_s']:.0f}" for n in ("aortic", "pulmonary", "mitral", "tricuspid")))
EOF
echo valve realism done
