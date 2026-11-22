#!/bin/bash
# research/sample_kernels.md §22 campaign driver (unit wsp-campaign): one case per line of cases.txt ("name|env|command"),
# a .done per case (resumable), STOP file checked between cases; every GPU step under the session GPU lock with timeout
# 900 and /usr/bin/time -v (peak RSS); power limit logged per case (never changed). start.sh / stop.sh / resume.sh.
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/warp_spec
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
export B=$SP/wsp/bin SP D
export RAYON_NUM_THREADS=8
mkdir -p $D/logs $SP/wsp/ncu
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
echo "=== run.sh start $(date '+%F %T'), binaries frozen at $(cat $SP/wsp/frozen_at.txt)" >> $D/campaign.log
while IFS='|' read -r name envs cmd; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  [ -e $D/STOP ] && { echo "STOP before $name ($(date '+%F %T'))" >> $D/campaign.log; exit 0; }
  [ -e $D/logs/$name.done ] && continue
  L=$D/logs/$name.log
  envs=$(eval echo "$envs"); cmd=$(eval echo "$cmd")
  echo "=== $name: env $envs $cmd ($(date '+%F %T'))" > $L
  echo "start $name $(date '+%F %T')" >> $D/campaign.log
  flock $SP/gpu.lock bash -c "echo \"locked \$(date +%T); power limit / draw / sm clock / memory used: \$(nvidia-smi --query-gpu=power.limit,power.draw,clocks.sm,memory.used --format=csv,noheader)\" >> $L; /usr/bin/time -v timeout 900 env $envs $cmd >> $L 2>&1; echo \"exit \$? \$(date +%T); power limit at end \$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)\" >> $L"
  rc=$(grep -o '^exit [0-9]*' $L | tail -1)
  rss=$(grep 'Maximum resident set size' $L | tail -1 | awk '{print $NF}')
  echo "done  $name: $rc, peak RSS ${rss} kB ($(date '+%F %T'))" >> $D/campaign.log
  touch $D/logs/$name.done
done < $D/cases.txt
touch $D/ALL.done
echo "=== all cases done $(date '+%F %T')" >> $D/campaign.log
