#!/usr/bin/env bash
# Graph export full20 (2026-11-28, CPU only): full19 + radius taper at degree-2 joints (vessel_graph --joint-taper-mm 15).
# Self-check: --joint-taper-mm 0 must reproduce export_full19's graph.npz exactly (same code path otherwise).
# Then network + Murray floors on full20 and the radius-step survey (degree-2 joints, r ≥ 2 mm).
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
E=data/hasty/external/zanatomy/export
X19=data/hasty/phantom/factory_test/export_full19
X0=data/hasty/scratch/vc_geom/export_taper0
X=data/hasty/phantom/factory_test/export_full20
mkdir -p $X0 $X
$PY $T/vessel_graph.py $E --out $X0 --joint-taper-mm 0 > $X0/vessel_graph.log 2>&1 || { echo "taper0 failed"; tail -5 $X0/vessel_graph.log; exit 1; }
rustlib/.venv/bin/python -I - <<EOF
import numpy as np
a = np.load("$X19/graph.npz"); b = np.load("$X0/graph.npz")
d = {k: (float(np.abs(a["radius_atlas" if k == "radius" else k].astype(np.float64) - b[k].astype(np.float64)).max()) if a[k].shape == b[k].shape else "shape") for k in b.files}
print("self-check taper 0 vs export_full19 (max |Δ| per array; radius vs full19's pre-Murray radius_atlas):", d)
EOF
$PY $T/vessel_graph.py $E --out $X > $X/vessel_graph.log 2>&1 || { echo "full20 graph failed"; tail -5 $X/vessel_graph.log; exit 1; }
grep -E "^joint taper|^de-overlap" $X/vessel_graph.log | cut -c1-600
$PY $T/network_flow.py "$X" "$X/flow" > /dev/null || { echo network failed; exit 1; }
for _ in 1 2; do $PY $T/murray_floor.py "$X" "$X/flow" > /dev/null; $PY $T/network_flow.py "$X" "$X/flow" > $X/network.log 2>&1; done
grep -E "residual|Kirchhoff" $X/network.log | head -3 | cut -c1-300
rm -rf "${X0:?}"
echo export full20 done
