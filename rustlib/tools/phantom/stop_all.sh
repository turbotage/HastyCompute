#!/usr/bin/env bash
# Stop a realistic_scan.sh pipeline gracefully and wait until no GPU job of it runs:
#   rustlib/tools/phantom/stop_all.sh <phantom_dir> [--at HH:MM]
# --at: wait until that time first (a watchdog for a compute window).
# STOP files: <phantom_dir> (the driver ends before the next step), the LBM runs and the simulator outputs (they write
# their state at the next boundary and exit: resumable); the short python steps (blood frames, prescription, truths)
# get SIGINT and rerun from the start next time. Waits up to 20 min, then reports what is left.
set -uo pipefail
PH=$(realpath "$1"); shift
if [[ ${1:-} == --at ]]; then
  target=$(date -d "$2" +%s); now=$(date +%s)
  (( target <= now )) && target=$((target + 86400))
  echo "watchdog: stopping at $(date -d @$target +%H:%M)"; sleep $((target - now))
fi
echo "stop: $(date +%H:%M:%S)"
touch "$PH/STOP"
for d in "$PH"/lbm/run_* "$PH"/sim_*/scan; do [[ -d $d ]] && touch "$d/STOP"; done
pkill -INT -f "tools/phantom/(phantom_4d|prep_sim|sim_truth|bias_truth|slab|lbm_case|lbm_coronary)\.py" 2>/dev/null
for i in $(seq 1 240); do
  if ! pgrep -x cardiac_lbm >/dev/null && ! pgrep -x simphantom >/dev/null && ! pgrep -f "tools/phantom/.*\.py" >/dev/null; then break; fi
  sleep 5
done
left=$(pgrep -af "cardiac_lbm|simphantom|tools/phantom/.*\.py" | grep -v pgrep)
if [[ -n $left ]]; then echo "still running after 20 min:"; echo "$left"; else echo "all stopped: $(date +%H:%M:%S)"; fi
# the STOP files of the resumable runs stay until the next start (realistic_scan.sh removes them at its start)
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader
