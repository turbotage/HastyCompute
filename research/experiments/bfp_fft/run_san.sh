#!/bin/bash
# compute-sanitizer racecheck / synccheck (and memcheck) on a reduced case set, one (tool, layout case, width) per
# invocation: racecheck's host shadow memory grows with shared-memory traffic × launches (the whole bfp_store test in
# one racecheck invocation exceeded the 24 GB cap; one small layout at one width peaks at 1.3–1.8 GB). Peak RSS per
# run from /usr/bin/time -v. bfp_store cases: 0 block 3 ch (partial tiles), 1 ball 3 ch, 7 in-place y / z block,
# 10 block 8 ch, 11 non-convex 8 ch, 12 in-place y / z ball 4 ch (10–12: whole tiles, dense and in-place layouts).
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
B=$SP/bfi/bin
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/bfp_fft/san
CS=/usr/local/cuda/bin/compute-sanitizer
mkdir -p $D
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
run() {
  local name=$1; shift
  [ -e $D/../STOP ] && { echo "STOP before $name"; exit 0; }
  [ -e $D/$name.done ] && return
  echo "=== $name: $* ($(date '+%F %T'))" > $D/$name.log
  flock $SP/gpu.lock /usr/bin/time -v timeout 900 "$@" >> $D/$name.log 2>&1
  echo "exit $? $(date +%T)" >> $D/$name.log
  touch $D/$name.done
}
for tool in memcheck racecheck synccheck; do
  for cb in 0:16 1:22 7:23 10:22 11:23 12:22 12:16; do
    c=${cb%%:*}; b=${cb##*:}
    run ${tool}_fft_c${c}_b${b} env HASTY_BFP_TEST_CASES=$c HASTY_BFP_TEST_BITS=$b $CS --tool $tool $B/bfp_store --test-threads 1 bfp_store_matches_codec_and_f32
  done
  run ${tool}_fft_short $CS --tool $tool $B/bfp_store --test-threads 1 bfp_store_short_batch
  # engine (in place): static order 0 fused; static order 0 separate pad, whole-grid passes; 2-D knots order 3
  for c in 0 1 6; do
    run ${tool}_seg_c${c} env HASTY_BFP_TEST_CASES=$c $CS --tool $tool $B/seg_fft_store --test-threads 1 bfp22_engine_matches_f32
  done
  run ${tool}_seg_c0_dense env HASTY_BFP_TEST_CASES=0 HASTY_SEG_FFT_BFP_LAYOUT=dense $CS --tool $tool $B/seg_fft_store --test-threads 1 bfp22_engine_matches_f32
done
touch $D/san.done
