#!/bin/bash
# tests, then phantom accuracy, then (only if every test passed) sanitizer and timing (research/experiments/bfp_fft/)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/bfp_fft
$D/run_tests.sh
if grep -l 'test result: FAILED\|^exit [1-9]' $D/t_*.log > $D/failed.txt; then echo "tests failed: stop" >> $D/failed.txt; exit 1; fi
rm -f $D/failed.txt
$D/run_phantom.sh
$D/run_san.sh
$D/run_bench.sh
touch $D/all.done
