#!/bin/bash
# End-to-end check of --rcompress in the drivers on real data (box1mm_b): the same short run with off / box, outputs
# compared by compare_runs.py. splinerecon: native 1 mm grid, 2 x 4 knots, encode 0, 1/4 of the readouts, held-out 1/6,
# 6 CGLS iterations. recon_p2np: 2 mm grid (--grid-scale 0.5 --kcrop), complex P2nP without prior, 3 iterations.
#   research/experiments/sample_compress/driver_check.sh <gpu.lock> <out dir>
set -u
cd "$(dirname "$0")/../../.."
L=${1:?gpu lock}
O=${2:?out dir}
mkdir -p $O
D1=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min_box1mm_b/scan
EV=data/hasty/phantom/factory_arms/ph0001/box_recon/lumen_idx_f32.npy
# (these drivers have no internal lock: the flock covers their CPU setup too; resumable: finished logs are skipped)
for m in off box; do
  grep -q "^done" $O/spline_$m.log 2>/dev/null && continue
  flock "$L" systemd-run --user --scope -q -p MemoryMax=36G -p MemorySwapMax=0 rustlib/target/rcompress_snap/splinerecon $D1 --rknots 2 --cknots 4 --encodes 0 --keep 4 --heldout 6 --ho-every 1 --iters 6 --save-at 3,6 --eval 0.5:$EV --rcompress $m --out $O/spline_$m > $O/spline_$m.log 2>&1
done
for m in off box; do
  [ -f $O/p2np_$m/run.json ] && continue
  flock "$L" systemd-run --user --scope -q -p MemoryMax=36G -p MemorySwapMax=0 rustlib/target/rcompress_snap/recon_p2np $D1 --grid-scale 0.5 --kcrop --keep 4 --iters 3 --every 1 --kappa 0 --no-full-save --fresh --rcompress $m --out $O/p2np_$m > $O/p2np_$m.log 2>&1
done
echo done
