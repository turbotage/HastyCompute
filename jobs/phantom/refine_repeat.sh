#!/usr/bin/env bash
# Run-to-run variability of the refined valve-plane metrics (2026-11-28). Question: 05r (semilunar refined) and 05ra
# (all four refined) refine the aortic / pulmonary valves identically but their valve-plane profiles differ (disk max/mean,
# displacement). Is that difference larger than the beat-to-beat sampling noise of a 3-cycle phase average?
# Run: the 05r case again, averaging cycles 7–9 (warmup 6) instead of 4–6 — a non-overlapping window, same code, same case.
# Decision: |05r(7–9) − 05r(4–6)| ≥ |05ra − 05r| for a metric → that metric's 05r/05ra difference is noise; report it with
# ≥ 6 averaged cycles from now on. Otherwise the difference is the changed inflow.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
S=data/hasty/scratch
L=jobs/logs
O=$S/lbm_full18_05r
flock jobs/locks/gpu.lock systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm $O/case $O/run_late --cycles 9 --warmup 6 --quiet > $L/refine_repeat_run.log 2>&1 || { echo "cardiac_lbm failed"; tail -5 $L/refine_repeat_run.log; exit 1; }
tail -2 $L/refine_repeat_run.log
systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_report.py $O/case $O/run_late > $L/refine_repeat_report.log 2>&1 || { echo "lbm_report failed"; tail -5 $L/refine_repeat_report.log; exit 1; }
grep -E "^(valve gate|density gate|tree balance)" $L/refine_repeat_report.log | cut -c1-300
echo refine repeat done
