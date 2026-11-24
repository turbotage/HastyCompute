#!/bin/bash
# 2026-11-24 after the disk-full failures (/mnt/hastydata was full; crime store deleted): dry start of --temporal, then
# the three gated reruns (binary snap_t1: support-masked frame self-check, surfaced writer errors).
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
GC_ITERS=2 GC_WARM=2 bash $E/run_t.sh dry_t_cubic_rm 2400 --temporal cubic --ratemap truth-bc
GC_SMAPS=truth bash $E/run_t.sh gc_rm_tmaps 5400 --ratemap truth-bc
bash $E/run_t.sh gc_rm_bg 5400 --ratemap truth-bc --bin-group
bash $E/run_t.sh gc_llr_rep 3600
