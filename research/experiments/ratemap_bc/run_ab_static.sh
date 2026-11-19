#!/bin/bash
# Pre-registered (2026-11-19): does the per-(b,c) true ratemap (truth-bc) lower the velocity error vs the static true
# ratemap (truth), one factor changed, 8 mm smoke, 8 iterations, at-knots velocity model with trust region + box?
# Decision: truth-bc better or equal at equal cost -> carry the per-frame ratemap into the velocity operator frames.
set -u
cd /home/turbotage/Documents/GitHub/HastyCompute
B=rustlib/target/snap_b0/recon_p2np_rmbc2
COMMON="data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan --vmodel knots --vop knots --vtrust 1 --vbox 3 --resp bellows --gnl --phys full --conc-fit deg4 --vcoils 8 --keep 8 --grid-scale 0.25 --kcrop --warm 4 --iters 8 --vcont 2 --vinit cg:3 --precond zmshr1m --every 1 --lam-iters 2 --static-lock 20:2:1"
for rm in truth-bc:static; do
  [ -f research/experiments/ratemap_bc/ab_${rm/:/_}.done ] && continue
  rm -rf research/experiments/ratemap_bc/ab_${rm/:/_}
  flock /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 -E HASTY_SEG_SHAPE=legacy -E HASTY_SEG_CB_BALANCE=0 $B $COMMON --ratemap $rm --out /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/ratemap_bc/ab_${rm/:/_} > research/experiments/ratemap_bc/ab_${rm/:/_}.log 2>&1 && touch research/experiments/ratemap_bc/ab_${rm/:/_}.done
done
