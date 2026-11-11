#!/bin/bash
cd /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/seg
# (runs in parallel with queue.sh)
D=/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001
PY=/home/turbotage/Documents/GitHub/HastyCompute/rustlib/.venv/bin/python
run() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -p CPUQuota=400% nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 SEG_FFT_WORKERS=4 $PY -I sweep3.py $D $PWD $1 $2 $PWD/res3_$1_q2.jsonl "$3" x > log3_q2_$1_$2.log 2>&1; }
run water raw "fixed:13/16/17;svdm_e:14;fixed_e:12/14"
run fat raw "fixed:20/22;svdm_e:11/13;fixed_e:14/16"
echo QUEUE2 DONE
