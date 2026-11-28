#!/usr/bin/env bash
# LBM cost split at 1 mm (mr case): base, geometry every 16 steps, WALE off, accumulation every 200, all three
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
C=data/hasty/phantom/factory_test/ph0001_mr/lbm_p1/case
B=data/hasty/scratch/lbmprof
rm -rf "${B:?}"
mkdir -p "${B:?}"
for v in base geom16 nowale acc200 all; do
  mkdir -p "${B:?}/${v:?}"
  cp -al "$C" "${B:?}/${v:?}/case"
  rm -f "${B:?}/${v:?}/case/case.json"
  rustlib/.venv/bin/python -c "
import json; c = json.load(open('$C/case.json')); r = c['run']; v = '$v'
if v in ('geom16', 'all'): r['geom_every'] = 16
if v in ('nowale', 'all'): r['cwale'] = 0.0
if v in ('acc200', 'all'): r['acc_every'] = 200
json.dump(c, open('$B/$v/case/case.json', 'w'))"
done
for v in base geom16 nowale acc200 all; do
  flock jobs/locks/gpu.lock systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm "${B:?}/${v:?}/case" "${B:?}/${v:?}/run" --cycles 1 --warmup 0 --quiet 2>&1 | tail -1 | sed "s/^/$v: /" | cut -c1-120
  rm -rf "${B:?}/${v:?}/run"
done
