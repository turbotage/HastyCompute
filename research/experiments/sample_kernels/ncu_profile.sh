#!/bin/bash
# Nsight Compute profiles of the sample-side NUFFT kernels, for use BEFORE the reboot that enables non-admin
# profiling (/etc/modprobe.d/nvidia-profiling.conf). Run once by the user:
#     sudo /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels/ncu_profile.sh
# It runs the (already built) proto_gather benchmark as root under ncu, waits for the session GPU lock, writes
# out/ncu/<name>.ncu-rep + a text summary, and gives every output back to the user. Resumable: finished profiles
# are skipped. After the reboot the same script works without sudo.
set -u
U=turbotage
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels
O=$D/out/ncu
R=/home/turbotage/Documents/GitHub/HastyCompute/rustlib
BIN=$R/target/release/proto_gather
NCU=/usr/local/cuda/bin/ncu
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
export CUDA_PATH=/usr/local/cuda LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} HOME=/root
mkdir -p $O
cd $R

prof() {   # prof <name> <kernel regex> <launch count> <env> <proto_gather args...>
  local n=$1 k=$2 c=$3 e=$4; shift 4
  [ -f $O/$n.ncu-rep ] && { echo "skip $n (done)"; return 0; }
  echo "=== $n ($(date))"
  flock $LOCK env $e timeout 1800 $NCU --set full --import-source yes -k "regex:$k" --launch-count $c \
      -o $O/$n -f $BIN "$@" > $O/$n.log 2>&1 || echo "FAILED rc=$? (see $O/$n.log)"
  [ -f $O/$n.ncu-rep ] && $NCU --import $O/$n.ncu-rep --page details > $O/$n.txt 2>&1
  chown -R $U:$U $O
}

# plane-streamed forward (ps) at the best autotuned configs (out/ps_*.log); --fixed tx,ty,tz,ncg,fc,fsb,nt
prof ps_m4000000_c48_l1_o0 pg_fwd_ps 2 "" --mode fwd --m 4000000 --ncb 48 --nseg 1 --order 0 --designs ps --fixed 4,4,8,4,2,2,256 --check 50 --reps 1
prof ps_m4000000_c10_l10_o0 pg_fwd_ps 2 "" --mode fwd --m 4000000 --ncb 10 --nseg 10 --order 0 --designs ps --fixed 4,4,8,2,2,2,128 --check 50 --reps 1
prof ps_m1300000_c10_l10_o0 pg_fwd_ps 2 "" --mode fwd --m 1300000 --ncb 10 --nseg 10 --order 0 --designs ps --fixed 4,4,8,2,2,1,128 --check 50 --reps 1
# production knot-major kernels (variant 1, the default), 4 M 10 x 10, orders 0 and 2: every autotune launch is
# profiled (the report's launch config / registers identify the TileCfg)
prof knot_kv1_m4000000_c10_l10_o0 "seg_(forward|adjoint)_k" 40 "HASTY_SEG_KV=1" --mode base --knot --m 4000000 --ncb 10 --nseg 10 --order 0
prof knot_kv1_m4000000_c10_l10_o2 "seg_(forward|adjoint)_k" 40 "HASTY_SEG_KV=1" --mode base --knot --m 4000000 --ncb 10 --nseg 10 --order 2
chown -R $U:$U $O
echo "ALL DONE $(date)"
