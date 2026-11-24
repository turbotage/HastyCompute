#!/bin/bash
# 2026-11-23 evening, after chain_ctrl (all steps take the GPU lock):
# 1. GPU unit test of setop::PerSetSegs (per-frame segment tables == resident tables, exact);
# 2. the inverse crime again (first try failed: --synthetic needs staged data, --plan host);
# 3. if 1 passed: the P²nP reference (g_esp_air config) with the TRUE ratemap per frame (--ratemap truth-bc, frame mode).
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
R=data/hasty/phantom/factory_arms/ph0001/precond_compare
E=research/experiments/precond_compare
flock $LOCK timeout 900 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 rustlib/target/admm/snap_rm/per_set_segs --nocapture --test-threads 1 > $R/per_set_segs_test.log 2>&1
rc=$?; echo "exit $rc" >> $R/per_set_segs_test.log; echo "per_set_segs test exit $rc"
bash $E/run_ctrl.sh crime c_crime_air 7200
if [ $rc = 0 ]; then
  RG_BIN=rustlib/target/admm/snap_rm/recon_p2np RG_MEM=40G bash $E/run_guard.sh g_esp_air_rm 9000 espirit --precond zmshr1 --iters 60 --step-guard 2 --air auto --ratemap truth-bc
else
  echo "per_set_segs test failed: g_esp_air_rm not started"
fi
