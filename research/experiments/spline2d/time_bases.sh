#!/bin/bash
# Gradient-pass time (one forward + adjoint, encode 0) of each basis on the same 2-D knot engine, incl. today's
# per-frame layout (8 respiratory × 20 cardiac bins, degree 0 × 0). That layout has no CG state that fits, so it is
# timed on one breathing bin (--resp-keep 8:0, one respiratory knot, 1/8 of the samples): its pass time is 8 × the
# logged value.
# Usage: time_bases.sh [log suffix]   (writes to $O/timing$1.log via the caller's redirection; see §17/§20)
cd /home/turbotage/Documents/GitHub/HastyCompute
O=data/hasty/phantom/factory_arms/ph0001/spline2d
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
for b in "1 0 20 0 frames8x20 --resp-keep 8:0" "8 0 10 3 bins8" "6 1 10 3 lin6" "4 1 10 3 lin4" "4 1 20 0 lin4_card20bins"; do
  set -- $b
  echo "== $5 (resp $1 deg $2, cardiac $3 deg $4) ${@:6}"
  systemd-run --user --scope -q -p MemoryMax=60G -p MemorySwapMax=0 rustlib/target/release/splinerecon $S --rknots $1 --rdeg $2 --cknots $3 --cdeg $4 --resp truth --coils 8 --no-gnl --seg-basis fixed --nseg 10 --conc-fit quad "${@:6}" --time-only --encodes 0 --eval 0.07:$O/idx_r0.07_f32.npy --out $O/time_$5 2>&1 | grep -E "time-only|ledger|GiB|autotune|panicked|does not fit"
done
