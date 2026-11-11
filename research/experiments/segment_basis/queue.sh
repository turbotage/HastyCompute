#!/bin/bash
# sequential sweep queue under the CPU cap (4 threads)
cd /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/seg
D=/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001
PY=/home/turbotage/Documents/GitHub/HastyCompute/rustlib/.venv/bin/python
run() { systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 -p CPUQuota=400% nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 SEG_FFT_WORKERS=4 $PY -I sweep3.py $D $PWD $1 $2 $PWD/res3_$1_q.jsonl "$3" x > log3_q_$1_$2.log 2>&1; }
run fat raw "fixed:12/14/16/18;cheb:10/12/14/16"
run water raw "cheb:8/10/12/14;svdm:8/9/10;svdm_m:8/9/10"
run water shimquad "fixed:10/12/14;svdm_e:8/10/12"
run water shimquad+abs2t "fixed:10/12/14;svdm_e:8/10/12"
run water shimlin "svdm_e:8/10/12"
run water abs1 "svdm_e:8/9/10"
echo QUEUE DONE
