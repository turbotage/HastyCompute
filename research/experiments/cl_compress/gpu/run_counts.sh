#!/bin/bash
# GPU accuracy / count runs of the joint coil × segment basis (research/coil_segment_compression.md, "Implementation
# and measured gains"). Each run takes the GPU lock; heavy CPU parts (Gram on the host, error checks) use 8 threads.
# Usage: run_counts.sh <scratch dir with gpu.lock>
set -u
S=${1:?scratch dir}
R=$(cd "$(dirname "$0")/../../../.." && pwd)
B=$R/rustlib/target/release/segjoint_bench
SCAN=$R/data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
O=$R/research/experiments/cl_compress/gpu/out
mkdir -p $O
# resumable: a run whose log ends with "done <name>" is skipped
run() { local name=$1; shift; if grep -q "^done $name" $O/log_$name.txt 2>/dev/null; then echo "== $name: done, skipped"; return; fi; echo "== $name $(date +%T)"; flock $S/gpu.lock env RAYON_NUM_THREADS=8 "$@" > $O/log_$name.txt 2>&1 && echo "done $name" >> $O/log_$name.txt; echo "   exit $? $(date +%T)"; }
run box_pd_noise   $B counts $SCAN --region box --weights pd --time noise --op --rebuild --out $O/box_pd_noise.json
run box_pd_uniform $B counts $SCAN --region box --weights pd --time uniform --out $O/box_pd_uniform.json
run box_warm_noise $B counts $SCAN --region box --weights warm --time noise --out $O/box_warm_noise.json
run box_pd_noise_clip300 $B counts $SCAN --region box --weights pd --time noise --clip-hz 300 --out $O/box_pd_noise_clip300.json
run box_pd_noise_tsqr $B counts $SCAN --region box --weights pd --time noise --tsqr --tols 1e-3,1e-4 --out $O/box_pd_noise_tsqr.json
run full_pd_noise  $B counts $SCAN --region full --weights pd --time noise --out $O/full_pd_noise.json
run full_pd_noise_clip300 $B counts $SCAN --region full --weights pd --time noise --clip-hz 300 --out $O/full_pd_noise_clip300.json
run head_2mm       $B head --dims 96,112,96 --res 2e-3 --out $O/head_2mm.json
run coilbox_160 $B coilbox $SCAN --presc-box 160 --margin 10 --energy 0.999,0.9995 --out $O/coilbox_160.json
# splinerecon: the new default (svdn, auto:err 1e-4) vs fixed nodes at the same rule, one forward + adjoint of encode 0
EV=$R/data/hasty/phantom/factory_arms/ph0001/box_recon/lumen_idx_f32.npy
SR=$R/rustlib/target/release/splinerecon
run spline_svdn  $SR $SCAN --keep 8 --encodes 0 --time-only --eval 0.5:$EV --out $O/spline_svdn
run spline_fixed $SR $SCAN --keep 8 --encodes 0 --time-only --seg-basis fixed --eval 0.5:$EV --out $O/spline_fixed
run spline_fixed10 $SR $SCAN --keep 8 --encodes 0 --time-only --seg-basis fixed --nseg 10 --eval 0.5:$EV --out $O/spline_fixed10
