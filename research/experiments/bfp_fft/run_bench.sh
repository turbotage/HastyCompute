#!/bin/bash
# BFP step 1 timing (research/bfp_storage.md §8): f32 against bfp22 / bfp20 / bfp24 y <-> z intermediates of the grid
# FFT (in place in the grid slot, default; bfp22 also with a separate dense buffer), one chain (same power limit,
# interleaved). opbench: flow_thorax 2 mm, image 225 x 225 x 112, grid 288 x 288 x 140, 10 virtual coils, L 10, coil
# batch 5 (50 channels); 2-D knots 6:1,16:3 at orders 3 and 2, and the static engine (7 frames: one set).
# bfp_fft_bench: the same grid transform alone, ball k support (columns 0.85, cells 0.64), slabs 1 and 0. ncu: one
# forward + inverse pair of whole-grid passes, per pass time / DRAM bytes / issue. One case per log, .done per case.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/bfp_fft
OB=$SP/bfi/opbench_bfi
FB=$SP/bfi/bfp_fft_bench
NCU=/usr/local/cuda/bin/ncu
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
run() {
  local name=$1; shift
  [ -e $D/STOP ] && { echo "STOP before $name"; exit 0; }
  [ -e $D/$name.done ] && return
  echo "=== $name: $* ($(date '+%F %T'))" > $D/$name.log
  flock $SP/gpu.lock bash -c "echo \"locked \$(date +%T); power limit / draw / sm clock / memory used: \$(nvidia-smi --query-gpu=power.limit,power.draw,clocks.sm,memory.used --format=csv,noheader)\" >> $D/$name.log; /usr/bin/time -v timeout 900 $* >> $D/$name.log 2>&1; echo \"exit \$? \$(date +%T); power limit \$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)\" >> $D/$name.log"
  touch $D/$name.done
}
K2="--models seg-full --nseg 10 --vcoils 10 --sets 1 --seq flow --knots2d 6:1,16:3 --scan-min 0.44 --reps 3"
STC="--models seg-full --nseg 10 --vcoils 10 --order 2 --frames 7 --scan-min 0.44 --sets 1 --reps 3 --seq flow"
G="288 288 140 225 225 112 50 --ball 151,82"
for s in f32 bfp22 bfp20 bfp24; do
  run fb_s1_$s $FB $G --store $s --slab 1 --reps 15
  run fb_s0_$s $FB $G --store $s --slab 0 --reps 15
done
run fb_s1_bfp22_dense $FB $G --store bfp22 --dense --slab 1 --reps 15
run fb_s0_bfp22_dense $FB $G --store bfp22 --dense --slab 0 --reps 15
for s in f32 bfp22 bfp20 bfp24; do run k2o3_$s env HASTY_SEG_FFT_STORE=$s $OB $K2 --order 3; done
run k2o3_bfp22_dense env HASTY_SEG_FFT_STORE=bfp22 HASTY_SEG_FFT_BFP_LAYOUT=dense $OB $K2 --order 3
for s in f32 bfp22; do run k2o2_$s env HASTY_SEG_FFT_STORE=$s $OB $K2 --order 2; done
for s in f32 bfp22 bfp20 bfp24; do run st_$s env HASTY_SEG_FFT_STORE=$s $OB $STC; done
run k2o3_f32_again env HASTY_SEG_FFT_STORE=f32 $OB $K2 --order 3
M="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_issued.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active"
for s in f32 bfp22 bfp24; do
  run ncu_$s $NCU --kernel-name fft_axis --launch-skip 6 --launch-count 6 --metrics $M --csv $FB $G --store $s --slab 0 --once
done
run ncu_bfp22_dense $NCU --kernel-name fft_axis --launch-skip 6 --launch-count 6 --metrics $M --csv $FB $G --store bfp22 --dense --slab 0 --once
touch $D/bench.done
