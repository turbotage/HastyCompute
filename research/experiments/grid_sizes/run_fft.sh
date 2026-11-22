#!/bin/bash
# FFT pass bandwidth per grid (fftpass_bench) under nsys, all configs inside one GPU-lock hold; trace CSV kept
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
B=${B:-$SP/bin/fftpass_bench0}
TAG=${TAG:-fft}
mkdir -p $D/$TAG $SP/gs/nsys
run() {
  name=$1; shift
  [ -e $D/$TAG/$name.done ] && return
  echo "== $name: $* ($(date '+%T')) power $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)" >> $D/$TAG/fft.log
  timeout 300 nsys profile --trace=cuda --sample=none --cpuctxsw=none -o $SP/gs/nsys/$name $B "$@" >> $D/$TAG/fft.log 2>&1
  nsys stats --report cuda_gpu_trace --format csv --output $D/$TAG/$name $SP/gs/nsys/$name.nsys-rep > /dev/null 2>&1
  rm -f $SP/gs/nsys/$name.nsys-rep $SP/gs/nsys/$name.sqlite
  touch $D/$TAG/$name.done
}
export -f run
flock $SP/gpu.lock bash -c '
run g240_c100_s0 240 240 240 192 192 192 100 0
run g240_c100_s1 240 240 240 192 192 192 100 1
run g240_c20_s0 240 240 240 192 192 192 20 0
run g240_c20_s1 240 240 240 192 192 192 20 1
run g320_c50_s0 320 320 320 256 256 256 50 0
run g320_c50_s1 320 320 320 256 256 256 50 1
run g320_c20_s1 320 320 320 256 256 256 20 1
run g400_c30_s0 400 400 400 320 320 320 30 0
run g400_c30_s1 400 400 400 320 320 320 30 1
run g400_c10_s1 400 400 400 320 320 320 10 1
run g640_c2_s0 640 640 640 512 512 512 2 0
run g640_c4_s0 640 640 640 512 512 512 4 0
run g640_c4_s2 640 640 640 512 512 512 4 2
run g800_c10_s0 800 480 400 640 384 320 10 0
run g800_c10_s1 800 480 400 640 384 320 10 1
run g800_c5_s1 800 480 400 640 384 320 5 1
'
touch $D/$TAG/ALL.done
