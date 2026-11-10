#!/usr/bin/env bash
# Atlas flow phantom P0: vessel graph → 1D network flow (with Murray radius floors) → GPU truth on the chosen grids.
#   rustlib/tools/phantom/run_p0.sh [--graph] [--labels] [--publish]
# --graph: rebuild the vessel graph from the Blender export; --labels: rebuild the tissue labels (winding numbers);
# --publish: send the 1 mm thorax truth to the flow viewer (hasty server on HASTY_SERVER or https://localhost:8443).
# Grids: 2 mm and 1 mm thorax with the arms (520 × 320 × 360 mm; until 2026-11-07 400 × 300 × 360, which cut the arms
# and shoulders off at x = ±200 mm and the back at y = +130 mm — a scanner excites everything in x, y), 0.5 mm heart
# (200 × 200 × 240 mm). Outputs: data/hasty/phantom/p0.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
PY="rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
E=data/hasty/external/zanatomy/export
P=data/hasty/phantom/p0
TT=rustlib/crates/hasty-phantom/data/atlas_tissues_3t.json
mkdir -p "$P"
has() { [[ " $* " == *" $FLAG "* ]]; }
ARGS=" $* "

if [[ "$ARGS" == *" --graph "* ]]; then
  $PY $T/vessel_graph.py $E | grep -E "connectors|root separation|de-overlap|joins:" | cut -c1-240
fi
$PY $T/network_flow.py $E $P/flow > /dev/null
for _ in 1 2; do
  $PY $T/murray_floor.py $E $P/flow | cut -c1-240
  $PY $T/network_flow.py $E $P/flow | grep -E "residual_max|kirchhoff"
done
$PY $T/pack_vessels.py $E $P/flow $P/vessels.hv

declare -A CENTER=([2mm]="0.0,0.0,1.30" [1mm]="0.0,0.0,1.30" [0.5mm_heart]="0.02,-0.03,1.31")
declare -A SIZE=([2mm]="0.52,0.32,0.36" [1mm]="0.52,0.32,0.36" [0.5mm_heart]="0.20,0.20,0.24")
declare -A VOX=([2mm]=0.002 [1mm]=0.001 [0.5mm_heart]=0.0005)
declare -A SUB=([2mm]=4 [1mm]=4 [0.5mm_heart]=3)
for G in 2mm 1mm 0.5mm_heart; do
  if [[ "$ARGS" == *" --labels "* || ! -f $P/tissue_$G.hv ]]; then
    rustlib/target/release/phantomvox $E/phantom_meshes.hv $P/vox_$G.hv --center ${CENTER[$G]} --size ${SIZE[$G]} --vox ${VOX[$G]} | tail -1
    $PY $T/phantom_labels.py $P/vox_$G.hv $TT $P/tissue_$G.hv $P/figs_$G --at 0.02,-0.03,1.29 | head -1
  fi
  rustlib/target/release/phantomflow $P/vessels.hv $P/flow_$G.hv --center ${CENTER[$G]} --size ${SIZE[$G]} --vox ${VOX[$G]} --sub ${SUB[$G]} | grep -E "contacts|blood"
  EXTRA=""
  if [[ $G == 1mm ]]; then
    EXTRA="--viewer-crop -130,160,-125,95,1150,1470 --viewer-step 2"
    [[ "$ARGS" == *" --publish "* ]] && EXTRA="$EXTRA --publish"
  fi
  echo "== truth $G"
  $PY $T/phantom_truth.py $P/flow_$G.hv $P/tissue_$G.hv $TT $E $P/flow $P/truth_$G $EXTRA | grep -E "flux_rel_rms|rel_median|rel_p95|viewer:" | tr -s ' ' | paste -sd' ' | sed 's/"flux_rel_rms"/\n  flux/g'
done
