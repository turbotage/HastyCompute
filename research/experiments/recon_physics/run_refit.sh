#!/bin/bash
# GIRF refit tests (research/recon_physics_wiring.md item 5), after run_ablation.sh (needs y_exact.npy, ynav_exact.npy).
# Each step skips its output if present (resumable); STOP file in the case directory stops. GPU ≤ ~5 GB per run.
set -u
REPO=/home/turbotage/Documents/GitHub/HastyCompute
BIN=$REPO/rustlib/.agents/seg/target/release/opval_v2
G=$REPO/data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11
C=$G/case_v2
EST=$G/girf_est_resonance_T200.json
export HASTY_REPO=$REPO HASTY_THREADS=6
# fit voxels: every 9th support voxel for the refit / navigator engines (segment basis details do not enter the refit);
# the cost runs build one encode with the production step 3
export HASTY_OPVAL_FIT_STEP=9
B="--fat union --rate fit --conc full --L 15"
SIG=$(python3 -c "import json; print(json.load(open('$C/case.json'))['sigma'])")
go() {  # output file, args...
  [ -f "$C/STOP" ] && { echo STOP; exit 0; }
  local out=$1; shift
  [ -f "$C/$out" ] && { echo "skip $out"; return; }
  echo "== $out: $*"
  systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 $BIN "$@" 2>&1 | grep -v "^warning" | grep -E "jac|iter|refit|forward|navigator|segments:|engine:|panicked|error|\{|\}|_s\"|over" 
}
# 1. Jacobian (k-gradient x sensitivities) vs central differences of the operator, 6 readouts, gains included
go jac_check.json jac $C --model $EST --params delay,b0,res,gain $B --readouts 6 --tag check
# 2. mismatched chain: delays +100 / -100 / +100 ns, B0-term amplitude +20 %, resonance f0 +0.5 / -0.5 / +0.5 %
TH="1e-7;-1e-7;1e-7;0.2;0.005;0;-0.005;0;0.005;0"
go y_mis.npy forward $C --models "mis=$EST#$TH" $B
go refit_recovery.json refit $C --model $EST --data $C/y_mis.npy --noise $SIG --seeds 1,2,3,4,5,6 --iters 3 --tag recovery $B
# 3. the realistic case: exact data (true chain + everything), model = the estimated GIRF
go refit_exact.json refit $C --model $EST --data $C/y_exact.npy --iters 4 --tag exact $B
go refit_exact_enc.json refit $C --model $EST --data $C/y_exact.npy --iters 4 --enc --tag exact_enc $B
go refit_exact_enconly.json refit $C --model $EST --data $C/y_exact.npy --iters 3 --enc-only --tag exact_enconly $B
go refit_exact_gain.json refit $C --model $EST --params delay,b0,res,gain --data $C/y_exact.npy --iters 4 --tag exact_gain $B
# 4. navigator monitor (per TR k offset / phase against the estimated chain)
go nav_est.json nav $C --model $EST --fat union --rate fit --conc full --L 6 --tag est
go nav_ideal.json nav $C --model ideal --fat union --rate fit --conc full --L 6 --tag ideal
# 5. cost on a production-size set (300 readouts x 4745 samples, encode 0)
HASTY_OPVAL_FIT_STEP=3 HASTY_OPVAL_GRID_GB=1.5 go cost_16c_L15.json cost $C --model $EST $B --set-readouts 300 --coils 16 --reps 5
HASTY_OPVAL_FIT_STEP=3 HASTY_OPVAL_GRID_GB=1.5 go cost_24c_L15.json cost $C --model $EST $B --set-readouts 300 --coils 24 --reps 5
