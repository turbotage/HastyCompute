#!/bin/bash
# §13: targeted Nsight Compute profiles (application replay: the 6 GiB grid breaks kernel replay, §11.x) of the vt /
# vs best shapes. One lock hold per profile; out/ncu/<name>.ncu-rep + .txt; resumable; STOP file.
#   systemd-run --user --unit=proto-vs2-ncu --collect -p MemoryMax=24G -p MemorySwapMax=0 $PWD/research/experiments/sample_kernels/run_vt_ncu.sh [name ...]
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels
O=$D/out/ncu
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
LOCK=$SP/gpu.lock
R=/home/turbotage/Documents/GitHub/HastyCompute/rustlib
BIN=${BIN:-$SP/bin/proto_gather_s13b}
NCU=/usr/local/cuda/bin/ncu
mkdir -p $O
cd $R
SECS="--section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section LaunchStats --section InstructionStats"
prof() {   # prof <name> <proto_gather args...>
  local n=$1; shift
  [ -f $O/$n.ncu-rep ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n $* ($(date))" > $O/$n.log
  flock $LOCK env HASTY_NVRTC_LINEINFO=1 timeout 2400 $NCU --replay-mode application --launch-count 1 -k regex:pg_fwd_vs $SECS --import-source yes \
      -o $O/$n -f $BIN "$@" --check 10 --reps 1 >> $O/$n.log 2>&1 || echo "FAILED rc=$?" >> $O/$n.log
  [ -f $O/$n.ncu-rep ] && $NCU --import $O/$n.ncu-rep --page details > $O/$n.txt 2>&1
}
prof vt_m4M_c10_o2 --mode fwd --m 4000000 --ncb 10 --nseg 10 --order 2 --designs vs --fixed "4,4,8,48,7,1,vt=1"
echo "NCU DONE $(date)" >> $O/ncu_driver.log
