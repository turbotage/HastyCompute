#!/usr/bin/env bash
# Vessel rod in the breathing field (2026-11-28) on factory_test19/ph0001 (CPU only, the LBM owns the GPU).
# Pre-registered question: does the rod (ℓ 30 mm) make the IVC's breathing profile monotone (no still spot between the
# pancreas and the liver; renal level 4–7 mm; intrahepatic within 2 mm of the liver beside it; min stretch ≥ 0.85),
# with min det J ≥ 0.4 and the aorta's resp-only motion changed ≤ 0.5 mm? Yes → default on, phantoms rebuilt with it.
# Self-check first: --vessel-rod-mm 0 must reproduce the phantom's resp_motion.hv bit for bit.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_test19/ph0001
O=data/hasty/scratch/motion_rod
T=rustlib/tools/phantom
PY="systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= rustlib/.venv/bin/python -I"
AMP=$(rustlib/.venv/bin/python -I -c "import json; b=json.load(open('$P/params.json'))['breathing']; print(b['dsi_mm'], b['dap_mm'], b['dlr_mm'])")
read -r DSI DAP DLR <<< "$AMP"
for m in ${MODES:-off rod30}; do
  case $m in off) X="--vessel-rod-mm 0 --tether-below-rmin-mm 3";; rod30) X="--vessel-rod-mm 30";; rodv) X="--vessel-rod-mm 30 --tether-below-rmin-mm 3";; tb5) X="--vessel-rod-mm 0 --tether-below-rmin-mm 5";; rodvt) X="--vessel-rod-mm 30 --tether-below-rmin-mm 5";; fin) X="";; esac
  mkdir -p $O/$m
  $PY $T/resp_motion.py $P/atlas/tissue_2mm.hv $O/$m --dsi $DSI --dap $DAP --dlr $DLR --graph $P/graph.npz $X > jobs/logs/motion_rod_$m.log 2>&1 || { echo "resp $m failed"; tail -5 jobs/logs/motion_rod_$m.log; exit 1; }
  grep -E "^(tether|vessel rod|embedded)" jobs/logs/motion_rod_$m.log | cut -c1-400
  # a phantom view: everything of ph0001, the breathing field replaced
  V=$O/$m/phantom
  rm -rf "${V:?}"
  mkdir -p $V/motion
  for f in $P/*; do [ "$(basename $f)" != motion ] && ln -s "$(realpath $f)" $V/; done
  for f in $P/motion/*; do [ "$(basename $f)" != resp_motion.hv ] && ln -s "$(realpath $f)" $V/motion/; done
  ln -s "$(realpath $O/$m/resp_motion.hv)" $V/motion/resp_motion.hv
  rustlib/.venv/bin/python -I jobs/tools/vessel_motion_check.py $V $O/$m/check 2>&1 | grep -E "^(aorta|IVC|Superior vena|Inferior vena)" | cut -c1-330
  rustlib/.venv/bin/python -I jobs/tools/resp_det.py $O/$m/resp_motion.hv $P/atlas/tissue_2mm.hv | head -5
  rustlib/.venv/bin/python -I jobs/tools/vessel_resp_neighbours.py $V $P/atlas/tissue_2mm.hv "^Inferior vena cava" --step-mm 10 > $O/$m/ivc_neighbours.txt 2>&1
  rm -rf "${V:?}/_tmp_check"
done
[ -f $O/off/resp_motion.hv ] && rustlib/.venv/bin/python -I - <<EOF
import sys
sys.path.insert(0, "python/lib")
import numpy as np
from hastycompute.store import load_any
a = np.asarray(load_any("$P/motion/resp_motion.hv")["B"]); b = np.asarray(load_any("$O/off/resp_motion.hv")["B"])
print("self-check rod off vs phantom field: max |ΔB|", float(np.abs(a - b).max()), "m")
EOF
echo motion rod done
