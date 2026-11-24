#!/bin/bash
# §13 control: the POOLED per-encode solve on inverse-crime data (x★ = PD + truth blood, noise-free) from zero,
# --keep 4 and --keep 16, 60 pooled PCG iterations, scored at it 0 of a 1-iteration P²nP run. Same rules as run_score.sh.
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/tpyr_score
B=rustlib/target/release/recon_p2np
for K in 4 16; do
  n=pool_crime_k$K
  [ -f $R/$n/DONE ] && continue
  mkdir -p $R/$n
  flock $LOCK timeout 900 systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env NOISE_SCALE=0 $B $S \
    --gnl --phys full --conc-fit deg4 --grid-scale 0.5 --kcrop --keep $K --vcoils 8 --nseg 6 --precond zmshr1m --lam-method lanczos \
    --report-detail --score-pc $R/pc --holdout 8 --synthetic --synth-x pc --plan host --warm 0 --warm-full 60 --init warm --iters 1 --no-full-save \
    --out $R/$n > $R/$n/log.txt 2>&1 && touch $R/$n/DONE
done
