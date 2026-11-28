#!/usr/bin/env bash
# full17: the full15 graph export (velocity floor), test phantom seed 1 with clearance (plateau worst, adaptive carve, solver sampling ≤ 2 mm
# rounds), then the audit and the flow check. Paths inside the repo only.
set -euo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
X=data/hasty/phantom/factory_test/export_full15
rm -rf data/hasty/phantom/factory_test/ph0001
LOCK=jobs/locks/gpu.lock
flock "$LOCK" $PY $T/make_phantom.py 1 --out data/hasty/phantom/factory_test --graph "$X" --truth "" --clearance --allow-overlap | grep -E "clearance|cardiac motion|root mobility|done"
TP=data/hasty/phantom/factory_test/ph0001
flock "$LOCK" $PY $T/clearance.py $TP --audit-only > $TP/clearance_audit.log 2>&1 || true
grep -E "^audit|^   pairs|^   within|^     [0-9]" $TP/clearance_audit.log | head -20
$PY $T/flow_check.py $TP | tail -12 || true
echo chain done
