#!/usr/bin/env bash
# Rebuild the vessel graph into a test export, then a test phantom (seed 1, with the clearance solve / carve) and audit it — without touching
# the Z-Anatomy export graph or factory_arms (research/flow_phantom.md Tier B5).
#   rustlib/tools/phantom/graph_test_chain.sh <tag> [vessel_graph options]
set -euo pipefail
cd "$(dirname "$0")/../../.."
TAG=$1; shift
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
X=data/hasty/phantom/factory_test/export_$TAG
E=data/hasty/external/zanatomy/export
mkdir -p "$X"
$PY $T/vessel_graph.py $E --out "$X" "$@" > "$X/vessel_graph.log" 2>&1
grep -E "^separation|^junction relocation|^crossing joins|^self-loops|^bridge" "$X/vessel_graph.log" | cut -c1-400 || true
$PY $T/network_flow.py "$X" "$X/flow" > /dev/null
for _ in 1 2; do $PY $T/murray_floor.py "$X" "$X/flow" > /dev/null; $PY $T/network_flow.py "$X" "$X/flow" > /dev/null; done
rm -rf data/hasty/phantom/factory_test/ph0001
LOCK=${HASTY_GPU_LOCK:-jobs/locks/gpu.lock}
flock "$LOCK" $PY $T/make_phantom.py 1 --out data/hasty/phantom/factory_test --graph "$X" --truth "" --clearance --allow-overlap | grep -E "clearance|cardiac motion|done"
TP=data/hasty/phantom/factory_test/ph0001
flock "$LOCK" $PY $T/clearance.py $TP --audit-only > $TP/clearance_audit.log 2>&1 || true
grep -E "^audit|^   pairs|^   within" $TP/clearance_audit.log
echo chain done
