#!/bin/bash
# λ_max estimator validation (dev scale: ph0001 sim_flow_10min, grid-scale 0.5 = 128×72×60, keep 16, 8 virtual coils)
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
O=data/hasty/phantom/factory_arms/ph0001/lanczos_dev
mkdir -p $O
B=rustlib/target/release/recon_p2np
COM="--gnl --phys full --conc-fit deg4 --grid-scale 0.5 --kcrop --keep 16 --vcoils 8 --nseg 6 --every 1 --no-full-save"
run() { name=$1; shift; flock $LOCK systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 timeout 1500 $B $S $COM "$@" --out $O/$name > $O/$name.log 2>&1; echo "$name exit $?"; }
for a in "$@"; do
case $a in
  F_lz) run F_lz --warm 3 --iters 3 --precond zmshr1m,zmshr1 --lam-compare --lam-method lanczos ;;
  F_pw) run F_pw --warm 3 --iters 3 --precond zmshr1m,zmshr1 --lam-method power ;;
  V_lz) run V_lz --vmodel knots --knots2d 4:1,8:3 --warm 3 --iters 4 --vcont 2 --precond zmshr1m --lam-compare --lam-method lanczos --lam-warm on ;;
  V_pw) run V_pw --vmodel knots --knots2d 4:1,8:3 --warm 3 --iters 4 --vcont 2 --precond zmshr1m --lam-method power ;;
esac
done
