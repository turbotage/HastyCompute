#!/bin/bash
# DRAM bytes per kernel of one gradient at both matrices, slab and whole-grid, current defaults (+ f32 store at 256)
cd /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/ncu_cz
export B=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/cz/opbench_dev2
A256="--models seg-full --sets 1 --seq flow --no-gnl --matrix 256,256,256 --order 3 --frames 20 --nseg 10 --vcoils 10 --cb-list 5"
A640="--models seg-full --sets 1 --seq flow --no-gnl --matrix 640,384,320 --order 3 --frames 20 --nseg 10 --vcoils 1 --cb-list 1"
./eng.sh b256_slab $A256
HASTY_SEG_FFT_SLAB=0 ./eng.sh b256_whole $A256
HASTY_SEG_FFT_STORE=f32 ./eng.sh b256_slab_f32 $A256
./eng.sh b640_slab $A640
HASTY_SEG_FFT_SLAB=0 ./eng.sh b640_whole $A640
