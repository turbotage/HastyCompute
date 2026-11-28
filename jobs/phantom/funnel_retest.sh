#!/usr/bin/env bash
# After the causal-opening test (hfatc): the outflow-tract funnel again, now that the valve axis follows the channel
# (hfatcf = hfatc + --ot-funnel 20,1.6), 1 mm, full17 + physio motion.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
until grep -qE "funnel test done|failed" jobs/logs/funnel_test_hfatc.log 2>/dev/null; do sleep 30; done
jobs/phantom/funnel_test2.sh hfatcf --ot-funnel 20,1.6 > jobs/logs/funnel_test_hfatcf.log 2>&1
rustlib/.venv/bin/python -I - <<'PYEOF'
import json, os
for t in ("lbm_full17_hfat", "lbm_full17_hfatc", "lbm_full17_hfatcf"):
    f = f"data/hasty/scratch/{t}/run/report/report.json"
    if not os.path.exists(f):
        print(t, "missing"); continue
    v = json.load(open(f))["valves"]
    print("==", t)
    for n in ("aortic", "pulmonary", "mitral", "tricuspid"):
        x = v[n]; a = x["at_peak_flow"]; e = a["eccentricity"]["1 diameter"]
        print(f"  {n:10s} jet {x['jet_max_m_s']:.2f} at {x['jet_where']['t_ms']:.0f} ms (normal {x['jet_where']['normal_share']:+.2f}) p95 {x['jet_p95_peak_m_s']:.2f} | disk mean {a['un_mean_m_s']:.2f} max {a['un_max_m_s']:.2f} | EOA {x.get('eoa_cm2', 0):.2f}/{x.get('goa_cm2', 0):.2f} | disp {e['displacement']:.2f} | stroke {x['stroke_link_ml']:.1f} closed rms {x['closed_rms_q_ml_s']:.2f}")
PYEOF
grep -E "outflow|valve gate|density|tree balance" jobs/logs/funnel_hfatcf_case.log jobs/logs/funnel_hfatcf_report.log | sed 's|^[^:]*:||' | cut -c1-220
echo retest done
