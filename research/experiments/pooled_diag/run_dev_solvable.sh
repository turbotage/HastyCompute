#!/bin/bash
# The solvable dev case (research/recon_diag_2026-11-22.md §5): ph0001 sim_flow_10min, 128 × 72 × 60 grid, k-space
# cropped to the grid, 8 virtual coils, full physics (6 ratemap/B0 segments, concomitant degree 4, GNL), 20 cardiac
# frames × 7 encodes, --keep ${KEEP:-2} (1/KEEP of the readouts). Noise-free inverse crime, x★ = PD + truth blood.
# Pooled start: 60 pooled plain-CG iterations per encode (--pool-metric none). Then P²nP 30 iterations, LLR κ 1, from
# the pooled start and from zero, with the circulant-metric preconditioner (zmshr1m) and without it (zmshr1).
# Score: --score-pc (it 0 = the pooled image; oracle: time-averaged truth median 0.137 m/s, zero image RMSE 0.425).
# Usage: run_dev_solvable.sh [name]. Done marker skips; STOP: touch <run dir>/STOP, rerun resumes.
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/pool_diag
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
B=rustlib/target/release/recon_p2np
K=${KEEP:-2}
n=${1:-dev_k$K}
[ -f $R/$n/DONE ] && { echo "skip $n"; exit 0; }
mkdir -p $R/$n
flock $LOCK timeout 2400 systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env NOISE_SCALE=0 $B $S \
  --gnl --phys full --conc-fit deg4 --nseg 6 --grid-scale 0.5 --kcrop --keep $K --vcoils 8 \
  --precond zmshr1m,zmshr1 --lam-method lanczos --report-detail --score-pc $PC \
  --synthetic --synth-x pc --plan host --warm 0 --warm-full 60 --pool-metric none \
  --init warm,zero --iters 30 --every 3 --no-full-save --no-vel-save 1 --out $R/$n > $R/$n/log.txt 2>&1
rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
