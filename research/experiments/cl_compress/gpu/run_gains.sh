#!/bin/bash
# Operator time per encode pass across grids (research/coil_segment_compression.md §10): today's C × L channels vs
# the joint operator at R, static engine and 2-D knots (2 lin × 4 cubic, s = 8), order 3, ball samples in readouts of
# 3990. 1 mm head / torso: maps + h do not fit next to 30 coils → today with 2 coils (per-coil-pass cost), joint
# timing-only (--no-gen, Φ generation timed on the box and scaled by voxels). Resumable: finished rows are skipped.
set -u
S=${1:?scratch dir}
R=$(cd "$(dirname "$0")/../../../.." && pwd)
B=$R/rustlib/target/release/segjoint_bench
O=$R/research/experiments/cl_compress/gpu/out
mkdir -p $O
row() { local name=$1; shift; if [ -f $O/gain_$name.done ]; then echo "== $name skipped"; return; fi; echo "== $name $(date +%T)"; flock $S/gpu.lock "$@" > $O/gain_$name.txt 2>&1 && touch $O/gain_$name.done; }
T="--today 30:10,30:14,30:20 --joint 30:190:96,30:330:96,30:420:96 --cb-cap 96"
row 2mm_static  $B time --dims 254,144,114 --res 2e-3 --m 4000000 --order 3 $T
row 2mm_knot    $B time --dims 254,144,114 --res 2e-3 --m 2000000 --order 3 --knots 2,1,4,3 $T
row box_static  $B time --dims 160,160,160 --res 1e-3 --m 4000000 --order 3 $T
row box_knot    $B time --dims 160,160,160 --res 1e-3 --m 2000000 --order 3 --knots 2,1,4,3 $T
H="--today 2:10,2:20 --joint 30:64:16,30:128:16 --no-gen --gen-nv 4096000 --cb-cap 16"
row head320_static $B time --dims 320,320,320 --res 1e-3 --m 4000000 --order 3 $H
row head384_static $B time --dims 384,384,384 --res 1e-3 --m 4000000 --order 3 $H
row head320_knot   $B time --dims 320,320,320 --res 1e-3 --m 2000000 --order 3 --knots 2,1,4,3 $H
TO="--today 2:4,2:6 --joint 30:32:8,30:64:8 --no-gen --gen-nv 4096000 --cb-cap 8"
row torso_static $B time --dims 640,384,384 --res 1e-3 --m 4000000 --order 3 --grid-gb 12 $TO
row torso_knot   $B time --dims 640,384,384 --res 1e-3 --m 2000000 --order 3 --knots 2,1,4,3 --grid-gb 12 $TO
echo done
