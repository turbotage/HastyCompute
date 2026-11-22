#!/bin/bash
# §18 runner: one opbench invocation per case line "name|args" of $CASES_FILE; a .done per case, STOP file stops
# before the next case; every case under the GPU lock with timeout 900; power limit logged per case.
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
BIN=${BIN:?BIN}
TAG=${TAG:?TAG}
CASES_FILE=${CASES_FILE:?CASES_FILE}
COMMON="--models seg-full --sets 1 --seq flow --no-gnl --reps 3"
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
mkdir -p $D/$TAG
while IFS='|' read -r name args; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  [ -e $D/STOP ] && { echo "STOP before $name"; exit 0; }
  [ -e $D/$TAG/$name.done ] && continue
  L=$D/$TAG/$name.log
  echo "=== $name: $BIN $args $COMMON ($(date '+%F %T'))" > $L
  echo "power.limit / draw / sm clock at start: $(nvidia-smi --query-gpu=power.limit,power.draw,clocks.sm,memory.used --format=csv,noheader)" >> $L
  flock $SP/gpu.lock timeout 900 ${PRE} $BIN $args $COMMON --jsonl $D/$TAG/results.jsonl >> $L 2>&1
  echo "exit $? ($(date '+%F %T')); power.limit at end: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)" >> $L
  touch $D/$TAG/$name.done
done < $CASES_FILE
touch $D/$TAG/ALL.done
