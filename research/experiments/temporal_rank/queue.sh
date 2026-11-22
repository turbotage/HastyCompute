#!/bin/bash
# the temporal-rank study's remaining runs, one at a time, each skipped when its output exists (touch out/STOP to stop
# between steps). Run detached:
#   systemd-run --user --unit tr-queue -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=800% --working-directory=$PWD bash queue.sh
cd "$(dirname "$0")"
PY=/home/turbotage/Documents/GitHub/HastyCompute/rustlib/.venv/bin/python3
log=out/queue.log
step() { local out=$1; shift; [ -e out/STOP ] && { echo "STOP" >>$log; exit 0; }; [ -e "$out" ] && return; echo "== $(date +%T) $*" >>$log; "$@" >>$log 2>&1 || echo "FAILED $*" >>$log; }
while systemctl --user is-active -q tr-regimes; do sleep 30; done
step out/joint_amp1.0_ven0.0.json $PY joint_rank.py
step out/joint_amp0.5_ven0.0.json $PY joint_rank.py --amp 0.5
step out/joint_amp2.0_ven0.0.json $PY joint_rank.py --amp 2.0
step out/joint_amp1.0_ven0.5.json $PY joint_rank.py --venous 0.5
step out/joint_amp1.0_ven0.0_rdot.json $PY joint_rank.py --rdot
step out/resp_fov_amp1.0.json $PY resp_fov.py
step out/resp_fov_amp2.0.json $PY resp_fov.py --amp 2.0
step out/beats_run_lib.json $PY beats_rank.py run_lib 10
step out/beats_run_cor.json $PY beats_rank.py run_cor 6
step out/estimability.json $PY estimability.py
step out/hrv.json $PY run_hrv.py
$PY plot_all.py >>$log 2>&1
echo "== $(date +%T) queue done" >>$log
