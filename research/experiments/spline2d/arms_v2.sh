#!/bin/bash
# Arms v2 (temporal_operator.md §19): preconditioned knot CG (G⁻¹ ⊗ C⁻¹), warm start, held-out 1/20, 40 iterations,
# encodes 0–1; respiratory 8 bins / 6 linear / 4 linear × 10 cubic cardiac knots. Then score (encode 1 − 0).
cd /home/turbotage/Documents/GitHub/HastyCompute
O=data/hasty/phantom/factory_arms/ph0001/spline2d
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
EV="0.07:$O/idx_r0.07_f32.npy,0.181:$O/idx_r0.181_f32.npy,0.493:$O/idx_r0.493_f32.npy"
for arm in "8 0 v2_bins8" "6 1 v2_lin6" "4 1 v2_lin4"; do
  set -- $arm
  { [ -f $O/$3_r0.493_itbest_lumen.npy ] || [ -f $O/$3_r0.493_itbest_lumen.hv ]; } && continue
  systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 rustlib/target/release/splinerecon $S --rknots $1 --rdeg $2 \
    --iters 40 --save-at 20,40 --warm 10 --precond knot --heldout 20 --ho-every 4 --encodes 0,1 --phys b0 --eval $EV --out $O/$3 > $O/$3.log 2>&1
done
rustlib/.venv/bin/python -I research/experiments/spline2d/score_pair.py $O/v2_bins8 $O/v2_lin6 $O/v2_lin4 > $O/v2_scores.txt 2>&1
grep -h "held-out" $O/v2_*.log > $O/v2_heldout.txt
echo "done $(date)" >> $O/v2_scores.txt
