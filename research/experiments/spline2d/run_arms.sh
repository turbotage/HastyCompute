#!/bin/bash
# Respiratory × cardiac basis test (research/temporal_operator.md §16): three arms, the same CGLS (no regulariser),
# the same physics (B0 segments, true maps, 8 coils), cardiac 10 cubic knots; respiratory 8 bins / 6 linear / 4 linear.
cd /home/turbotage/Documents/GitHub/HastyCompute
O=data/hasty/phantom/factory_arms/ph0001/spline2d
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
EV="0.07:$O/idx_r0.07_f32.npy,0.181:$O/idx_r0.181_f32.npy,0.493:$O/idx_r0.493_f32.npy"
for arm in "8 0 bins8" "6 1 lin6" "4 1 lin4"; do
  set -- $arm
  if ls $O/$3_r0.493_it20_lumen.npy >/dev/null 2>&1; then echo "skip $3 (done)"; continue; fi
  /usr/bin/time -v -o $O/$3_time.txt systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 \
    rustlib/target/release/splinerecon $S --rknots $1 --rdeg $2 --cknots 10 --resp truth --coils 8 --no-gnl --seg-basis fixed --nseg 10 --conc-fit quad --iters 20 --save-at 5,10,20 --eval $EV --out $O/$3 > $O/$3.log 2>&1
  echo "$3 exit $?"
done
