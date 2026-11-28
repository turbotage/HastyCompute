#!/usr/bin/env bash
# After the full18 chain: on full17 with the physio motion, against lbm_full17_hf (stub axis, 85/80°):
#   hfa   semilunar channel axis (lbm_case default --sl-axis channel), opening 85,80
#   hfat  + opening 90,85 (semilunar cusps parallel to the axis, AV leaflets 85°)
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
until grep -q "chain done" jobs/logs/chain_full18.log; do sleep 30; done
jobs/phantom/funnel_test2.sh hfa > jobs/logs/funnel_test_hfa.log 2>&1
jobs/phantom/funnel_test2.sh hfat --theta-open-deg 90,85 > jobs/logs/funnel_test_hfat.log 2>&1
for t in hfa hfat; do echo "== $t"; grep -E "channel axis|valve gate|density gate|tree balance|effective orifice" jobs/logs/funnel_test_$t.log jobs/logs/funnel_${t}_case.log jobs/logs/funnel_${t}_report.log 2>/dev/null | sed 's|^[^:]*:||' | cut -c1-260; done
rustlib/.venv/bin/python -I - <<'PYEOF'
import json, os
for t in ("lbm_full17_hf", "lbm_full17_hfa", "lbm_full17_hfat"):
    f = f"data/hasty/scratch/{t}/run/report/report.json"
    if not os.path.exists(f):
        print(t, "missing"); continue
    v = json.load(open(f))["valves"]
    print("==", t)
    for n in ("aortic", "pulmonary", "mitral", "tricuspid"):
        x = v[n]; a = x["at_peak_flow"]; e = a["eccentricity"]
        print(f"  {n:10s} jet {x['jet_max_m_s']:.2f} | peak flow: disk mean {a['un_mean_m_s']:.2f}, max {a['un_max_m_s']:.2f}, 12 mm {a['jet_max_m_s']:.2f} | EOA {x.get('eoa_cm2', 0):.2f} / GOA {x.get('goa_cm2', 0):.2f} | "
              + " ; ".join(f"{k}: disp {y['displacement']:.2f} angle {y['jet_angle_deg']:.0f}° A50 {y['half_flux_area_share']:.2f}" for k, y in e.items()) + f" | stroke {x['stroke_link_ml']:.1f}")
PYEOF
echo axis theta tests done
