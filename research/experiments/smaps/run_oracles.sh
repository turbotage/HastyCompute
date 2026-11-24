#!/bin/bash
# CPU oracle controls, one after another (single thread, nice). Results: results/oracle_<tag>.json
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SMAPS_CACHE=${SMAPS_CACHE:-/tmp/claude-1000/smaps_cache}
PY=../../../rustlib/.venv/bin/python
for args in "--tag A0" "--tag D_out --rate" "--tag D_mix --rate --mix-return" "--tag C32 --cal 32"; do
  nice $PY oracle.py $args 2>&1
done
