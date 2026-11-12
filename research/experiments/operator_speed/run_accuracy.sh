#!/bin/bash
# Operator accuracy of the NUFFT (σ, W) choices on the validation case of research/recon_physics_wiring.md
# (case_v2: 48 v2 readouts, 32 whitened coils, the true chain, union svdn L 15, fitted rate, full concomitant):
# opval_v2 forward --nufft σ:W,… writes y_truthchain_s<σ>w<W>.npy (segments built once, one SegNufft per variant), then
# case.py eval against the direct sum of the operator's own model (RP_REF=y_exact_modelfit_truth: R = 0.0375 at 1.25:10)
# and against the exact physics (R = 0.425). Fit voxels every 9th (HASTY_OPVAL_FIT_STEP=9, as truthchain_f9: 0.0377).
# GPU ≈ 4–6 GB (grid budget 2 GiB per variant); hold data/hasty/GPU_BENCH_LOCK.
set -u
REPO=/home/turbotage/Documents/GitHub/HastyCompute
BIN=$REPO/rustlib/.agents/seg/target/release/opval_v2
C=$REPO/data/hasty/studies/operator_speed_2026-11-12/case_v2   # symlinks to recon_physics_2026-11-11/case_v2 inputs; outputs stay here
OUT=$REPO/data/hasty/studies/operator_speed_2026-11-12
PY=$REPO/rustlib/.venv/bin/python
LOCK=$REPO/data/hasty/GPU_BENCH_LOCK
VARS=${VARS:-1.25:10,1.25:8,1.25:7,1.25:6,1.25:5,1.5:6,1.5:5}
export HASTY_REPO=$REPO HASTY_THREADS=6 RAYON_NUM_THREADS=6 HASTY_OPVAL_FIT_STEP=9
# wait for the GPU: another job may hold the lock (never remove a lock that is not ours)
while [ -e "$LOCK" ]; do echo "$(date +%T) waiting for $LOCK: $(head -c 200 "$LOCK")"; sleep 60; done
echo "$(date +%T) opspeed: operator accuracy (opval_v2 --nufft $VARS)" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
mkdir -p $OUT/logs
systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 $BIN forward $C --models "truthchain=resonance" \
  --fat union --rate fit --conc full --L 15 --nufft "$VARS" 2>&1 | grep -v "^warning" | tee $OUT/logs/accuracy_opval.log | grep -E "forward|segments:|engine|panicked|error"
TAGS=$(echo $VARS | tr ',' '\n' | sed -E 's/^([^:]*):(.*)$/truthchain_s\1w\2/' | tr '\n' ' ')
cd $REPO/research/experiments/recon_physics
RP_CASE=$C RP_REF=y_exact_modelfit_truth $PY case.py eval $TAGS 2>&1 | tee $OUT/logs/accuracy_eval_model.log
RP_CASE=$C $PY case.py eval $TAGS 2>&1 | tee $OUT/logs/accuracy_eval_exact.log
