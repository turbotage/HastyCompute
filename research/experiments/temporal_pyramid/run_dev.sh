#!/bin/bash
# Dev check of --init meanres (temporal_pyramid.md §10 eq. (8)) and --anchor mean (Theorem 5 case B), §12.
# Dev scale: ph0001 sim_flow_10min, grid-scale 0.5 (128×72×60), keep 16, 8 virtual coils, 6 segments, 20 frames × 7
# encodes. One setup; runs = inits × anchors (ZMSHR1-M, LLR κ 1, 12 iterations). Stop: touch $O/STOP; rerun resumes.
# Second pass: INITS=warm,zero,meanres:1:circ,meanres:1:pcg3 ANCH=none,mean:100 LOGN=run2 (logs: dev_run{1,2}.log;
# tables: python3 -I parse_dev.py <log> <inits> <anchors> → dev_results.md).
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
O=data/hasty/phantom/factory_arms/ph0001/tpyr_dev
mkdir -p $O
B=rustlib/target/release/recon_p2np
COM="--gnl --phys full --conc-fit deg4 --grid-scale 0.5 --kcrop --keep 16 --vcoils 8 --nseg 6 --every 1 --no-full-save"
INITS=${INITS:-warm,zero,meanres:1:circ,meanres:2:circ,meanres:1:pcg3,meanres:2:pcg3}
ANCH=${ANCH:-none,mean:1,mean:10}
LOGN=${LOGN:-run}
flock $LOCK timeout 2700 $B $S $COM --warm 3 --warm-full 8 --iters 12 --precond zmshr1m --lam-method lanczos \
  --report-detail --init $INITS --anchor $ANCH --out $O/run > $O/$LOGN.log 2>&1
echo "exit $?" >> $O/$LOGN.log
