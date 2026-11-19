#!/bin/bash
# Tile-kernel launch shapes of research/coil_segment_compression.md §12: (1) GPU tests of the new shapes / joint paths
# against the old ones, (2) the per-pass diagnostics of cl_compress/gpu/run_diag.sh, (3) the box rows of run_gains.sh
# (box_static, box_knot, 2mm_static), each with the new defaults and with HASTY_SEG_SHAPE=legacy (the previous launch
# shapes, same binary), plus ablations: forward coil blocking off (HASTY_SEG_FCB=1), forward atomics off
# (HASTY_SEG_DEFS=PROF_NOATOMIC: timing only, wrong results), Φ by the kernel (HASTY_JOINT_PHI=kernel), sample chunks.
# Usage: run.sh <scratch dir holding gpu.lock>. Every GPU step takes flock "$S"/gpu.lock around the GPU command only;
# finished steps leave out/<name>.done and are skipped on a re-run (resumable). Builds run first (CPU, nice -j 4).
set -u
S=${1:?scratch dir (holds gpu.lock)}
R=$(cd "$(dirname "$0")/../../.." && pwd)
O=$R/research/experiments/tile_block/out
B=$R/rustlib/target/release/segjoint_bench
mkdir -p $O "$S"
cd $R/rustlib
if [ ! -f $O/build.done ]; then
  nice cargo build -j 4 --release -p hasty-bench --bin segjoint_bench > $O/build.txt 2>&1 \
    && nice cargo test -j 4 --release -p hasty-nufft --no-run --test seg_tile_block --test seg_joint_chunks --test seg_knot_kv --test seg_knot --test seg_joint >> $O/build.txt 2>&1 \
    && touch $O/build.done || { echo "build failed: $O/build.txt"; exit 1; }
fi
step() { local name=$1; shift; if [ -f $O/$name.done ]; then echo "== $name skipped"; return; fi; echo "== $name $(date +%T)"; flock "$S"/gpu.lock "$@" > $O/$name.txt 2>&1 && touch $O/$name.done; }
# (1) tests: new shapes vs base / one-coil batches, joint CGEMM / chunks / inner batches; old suites as regressions
for t in seg_tile_block seg_joint_chunks seg_knot_kv seg_knot seg_joint; do
  step test_$t nice cargo test -j 4 --release -p hasty-nufft --test $t -- --nocapture --test-threads 1
done
# (2) diagnostics (run_diag.sh rows): today 30 × 10 and 96 × 1, joint R = 190 in 96-channel batches
DS="time --dims 160,160,160 --res 1e-3 --m 4000000 --order 2 --today 30:10,96:1 --cb-cap 96 --joint 30:190:96"
DK="time --dims 160,160,160 --res 1e-3 --m 2000000 --order 3 --knots 2,1,4,3 --today 30:10,96:1 --cb-cap 96 --joint 30:190:96"
for v in "new" "legacy HASTY_SEG_SHAPE=legacy" "fcb1 HASTY_SEG_FCB=1" "noatomic HASTY_SEG_DEFS=PROF_NOATOMIC" "phikernel HASTY_JOINT_PHI=kernel"; do
  set -- $v
  tag=$1; shift
  step diag_static_o2_$tag env "$@" $B $DS
  step diag_knot_o3_$tag env "$@" $B $DK
done
# (3) gain rows (run_gains.sh): today 30 × 10 / 14 / 20 vs joint R = 190 / 330 / 420 in 96-channel batches
T="--today 30:10,30:14,30:20 --joint 30:190:96,30:330:96,30:420:96 --cb-cap 96"
for v in "new" "legacy HASTY_SEG_SHAPE=legacy"; do
  set -- $v
  tag=$1; shift
  step gain_box_static_$tag env "$@" $B time --dims 160,160,160 --res 1e-3 --m 4000000 --order 3 $T
  step gain_box_knot_$tag env "$@" $B time --dims 160,160,160 --res 1e-3 --m 2000000 --order 3 --knots 2,1,4,3 $T
  step gain_2mm_static_$tag env "$@" $B time --dims 254,144,114 --res 2e-3 --m 4000000 --order 3 $T
done
# sample chunks (g buffer of one chunk) next to the whole set, and the Φ generation kernel vs CGEMM alone
step chunks_box_static $B time --dims 160,160,160 --res 1e-3 --m 4000000 --order 3 --joint 30:190:96,30:330:96 --cb-cap 96 --chunks 4
step chunks_box_knot $B time --dims 160,160,160 --res 1e-3 --m 2000000 --order 3 --knots 2,1,4,3 --joint 30:190:96 --cb-cap 96 --chunks 4
step phi_gen $B time --dims 160,160,160 --res 1e-3 --m 400000 --order 3 --joint 30:190:96 --cb-cap 96 --no-gen --gen-nv 4096000
echo done
