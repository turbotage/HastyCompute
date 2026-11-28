#!/usr/bin/env bash
# the wall pump with the per-side gain: healthy full15 at 1 mm and 0.5 mm, 6 cycles (3 warm-up), existing cases
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
PH=data/hasty/phantom/factory_test/ph0001
for c in lbm_p1 lbm_p05; do
  R=$PH/$c/run_g; rm -rf $R $PH/$c/run_dbg
  flock jobs/locks/gpu.lock systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm $PH/$c/case $R --cycles 6 --warmup 3 --quiet > $PH/$c/run_g.log 2>&1
  systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_report.py $PH/$c/case $R > $PH/$c/report_g.log 2>&1
  echo "== $c ($(date +%H:%M))"; tail -1 $PH/$c/run_g.log; grep -E "^(valve gate|density gate|tree balance)" $PH/$c/report_g.log | cut -c1-600
  rustlib/.venv/bin/python -c "
import json
L=[json.loads(l) for l in open('$R/log.jsonl') if l.strip()]
for d in L[::max(1,len(L)//8)]: print(round(d['t'],3), {k: round(v,3) for k,v in (d.get('pump_gain') or {}).items()})
"
  rm -f $R/flow_bins.f32 $R/ckpt.bin
done
echo campaign6 done
