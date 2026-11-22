#!/bin/bash
# nsys per-kernel times of exactly one gradient (opbench --profile: cuProfilerStart / Stop around it) per case line
# "name|args" of $CASES_FILE; keeps the kernel summary CSV, deletes the report files. .done per case, STOP file.
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
BIN=${BIN:?BIN}; TAG=${TAG:?TAG}; CASES_FILE=${CASES_FILE:?CASES_FILE}
COMMON="--models seg-full --sets 1 --seq flow --no-gnl --reps 1 --profile"
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
mkdir -p $D/$TAG $SP/gs/nsys
while IFS='|' read -r name args; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  [ -e $D/STOP ] && exit 0
  [ -e $D/$TAG/$name.done ] && continue
  L=$D/$TAG/$name.log
  echo "=== $name: nsys + $BIN $args $COMMON ($(date '+%F %T')); power $(nvidia-smi --query-gpu=power.limit,power.draw,clocks.sm --format=csv,noheader)" > $L
  rm -f $SP/gs/nsys/$name.*
  flock $SP/gpu.lock timeout 900 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda --sample=none --cpuctxsw=none -o $SP/gs/nsys/$name $BIN $args $COMMON --jsonl $D/$TAG/results.jsonl >> $L 2>&1
  echo "exit $? ($(date '+%F %T'))" >> $L
  nsys stats --report cuda_gpu_kern_sum --format csv --output $D/$TAG/$name $SP/gs/nsys/$name.nsys-rep >> $L 2>&1
  nsys stats --report cuda_gpu_mem_time_sum --format csv --output $D/$TAG/$name $SP/gs/nsys/$name.nsys-rep >> $L 2>&1
  nsys stats --report cuda_gpu_trace --format csv --output $D/$TAG/$name $SP/gs/nsys/$name.nsys-rep >> $L 2>&1
  rm -f $SP/gs/nsys/$name.nsys-rep $SP/gs/nsys/$name.sqlite
  touch $D/$TAG/$name.done
done < $CASES_FILE
touch $D/$TAG/ALL.done
