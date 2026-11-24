#!/bin/bash
# Run 2 (2026-11-23): spatial pyramid 64x36x30 (scale 0.25, 20 it, own pooled start) -> 128x72x60 (scale 0.5, 25 it,
# --init-from the coarse level) in the run_guard.sh dev config: measured noisy data, out-half ADC, all readouts with
# --holdout 25, 4 resp x 20 cardiac x 7 encodes, vcoils auto 0.999, ESPIRiT (out-half calibration), full physics +
# GNL + deg4, nseg auto, LLR k1 + db4 mix 0.3 + DFW sure + static lock group, ZMSHR1 + step guard 2.
# Question: does the pyramid reach the single-level 60-it quality (held-out 0.0613, pc RMSE 0.258; g_esp) in <= 25
# fine iterations, and at what total wall time? Pause: touch $ROOT/STOP; rerun this script to resume.
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
ROOT=data/hasty/phantom/factory_arms/ph0001/precond_compare/pyr_esp
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
export HASTY_KSTORE_DIR=$PWD/$S/kstore
flock $LOCK timeout 14400 bash rustlib/tools/recon/pyramid.sh $S $ROOT --levels 0.25,0.5 --iters 20,25 --mem 24 \
  --first-args "--warm-full 60 --pool-metric none --init warm" -- \
  --adc-part out --keep 1 --resp-bins 4 --vcoils auto --coil-energy 0.999 --smaps espirit \
  --phys full --gnl --conc-fit deg4 --nseg auto --prior llr --kappa 1 --wavelet-mix 0.3 --dfw sure \
  --static-lock 20:10:5 --lock-mode group --lam-method lanczos --report-detail --score-pc $PC \
  --holdout 25 --holdout-res --warm 0 --every 5 --precond zmshr1 --step-guard 2
echo "pyramid exit $?" >> $ROOT/driver.log
