#!/bin/bash
# Per-pass diagnostics of the joint operator's inner engine (research/coil_segment_compression.md §10): box 160³ at
# 1 mm, today's 30 coils × 10 segments vs nseg = 1 engines with 96 channels per pass, register blocking FSB / ACB.
set -u
S=${1:?scratch dir}
R=$(cd "$(dirname "$0")/../../../.." && pwd)
B=$R/rustlib/target/release/segjoint_bench
O=$R/research/experiments/cl_compress/gpu/out
mkdir -p $O
[ -f $O/diag_static_o2.done ] || { for rb in "1 1" "4 4" "2 8"; do
  set -- $rb
  echo "== static order 2, HASTY_SEG_FSB=$1 HASTY_SEG_ACB=$2 $(date +%T)"
  flock $S/gpu.lock env HASTY_SEG_FSB=$1 HASTY_SEG_ACB=$2 $B time --dims 160,160,160 --res 1e-3 --m 4000000 --order 2 --today 30:10,96:1 --cb-cap 96 --joint 30:190:96
done > $O/diag_static_o2.txt 2>&1 && touch $O/diag_static_o2.done; }
[ -f $O/diag_knot_o3.done ] || { for rb in "1 1" "4 4" "2 8"; do
  set -- $rb
  echo "== 2-D knots 2x4 order 3, HASTY_SEG_FSB=$1 HASTY_SEG_ACB=$2 $(date +%T)"
  flock $S/gpu.lock env HASTY_SEG_FSB=$1 HASTY_SEG_ACB=$2 $B time --dims 160,160,160 --res 1e-3 --m 2000000 --order 3 --knots 2,1,4,3 --today 30:10,96:1 --cb-cap 96 --joint 30:190:96
done > $O/diag_knot_o3.txt 2>&1 && touch $O/diag_knot_o3.done; }
echo done
