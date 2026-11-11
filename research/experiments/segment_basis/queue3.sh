#!/bin/bash
cd /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/seg
until grep -q "QUEUE2 DONE" queue2.log; do sleep 10; done
D=/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001
PY=/home/turbotage/Documents/GitHub/HastyCompute/rustlib/.venv/bin/python
run() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -p CPUQuota=400% nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 SEG_FFT_WORKERS=4 $PY -I sweep3.py $D $PWD $1 $2 $PWD/res3_$1_q3.jsonl "$3" x > log3_q3_$1_$2.log 2>&1; }
run water shimquadn "fixed:10/12/14;svdm_e:8/10/12"
run water shimquadn+abs2 "svdm_e:8/10"
echo QUEUE3 DONE
