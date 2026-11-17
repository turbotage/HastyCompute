#!/bin/bash
# levels in order of cost (cheapest first); each level skipped if its last CSV exists
cd /home/turbotage/Documents/GitHub/HastyCompute
O=data/hasty/phantom/factory_arms/ph0001/spline2d
for lv in "$@"; do
  tag=${lv%%:*}; keep=${lv##*:}
  [ -f $O/p12_${tag}_arms_e1.csv ] && continue
  research/experiments/convergence_p12/run_level.sh $tag $keep 41 "${ARMS:-}"
done
