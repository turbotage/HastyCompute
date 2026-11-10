#!/usr/bin/env bash
# Oracle-vs-estimate evaluation of one flow scan: the same reconstruction with true or measured fields / coil maps.
#   rustlib/tools/recon/eval_matrix.sh <flow_sim_dir> <recon_b0_run> <out_root> [--configs a,b,..] [--mem 40] -- [recon_p2np args]
# Configurations (each a recon_p2np run in <out_root>/<name>):
#   oracle     true ω, no R2*, true coil maps (the previous default)
#   est_field  ω and R2* measured by the B0 scan (recon_b0 run), true coil maps
#   est_smaps  true ω, ESPIRiT coil maps
#   est_both   measured ω + R2*, ESPIRiT coil maps          (the realistic case)
#   true_r2s   true ω + true R2* (sim truth_r2s.npy), true coil maps   (isolates the T2* decay model)
# A finished configuration (run.json and a metrics CSV ending at the last iteration) is skipped; an interrupted one
# resumes (recon_p2np checkpoints; STOP file in <out_root> or Ctrl-C ends the driver between and inside runs).
# Afterwards tools/recon/eval_table.py writes <out_root>/eval_table.{md,csv} and figs/eval_rmse.png.
# Large scans: recon_p2np streams the data (--kstore auto); with `-- --kstore disk:zstd1` the first configuration
# builds one store in <sim>/kstore/ and the others reuse it (same data, compression and noise), `--table-store zstd`
# packs the operator's sample tables (≈ ×6).
set -uo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
SIM=$(realpath "$1"); B0=$(realpath "$2"); OUT=$(realpath -m "$3"); shift 3
CONFIGS="oracle,est_field,est_smaps,est_both"; MEM=40
while [[ $# -gt 0 ]]; do
  case $1 in
    --configs) CONFIGS=$2; shift 2 ;;
    --mem) MEM=$2; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1 (recon_p2np arguments go after --)"; exit 2 ;;
  esac
done
EXTRA=("$@")
mkdir -p "$OUT"
rm -f "$OUT/STOP"
B=rustlib/target/release
cfg_args() {
  case $1 in
    oracle) echo "" ;;
    est_field) echo "--field est:$B0 --r2s est:$B0" ;;
    est_smaps) echo "--smaps espirit" ;;
    est_both) echo "--field est:$B0 --r2s est:$B0 --smaps espirit" ;;
    true_r2s) echo "--r2s truth" ;;
    *) echo "unknown configuration $1" >&2; exit 2 ;;
  esac
}
iters=40
for ((i = 0; i < ${#EXTRA[@]}; i++)); do [[ ${EXTRA[$i]} == --iters ]] && iters=${EXTRA[$((i + 1))]}; done
done_run() {
  local d=$1
  [[ -f $d/run.json ]] || return 1
  local f; f=$(ls "$d"/metrics_*.csv 2>/dev/null | head -1)
  [[ -n $f ]] || return 1
  [[ $(tail -1 "$f" | cut -d, -f1) == "$iters" ]]
}
IFS=, read -ra CFG <<< "$CONFIGS"
for c in "${CFG[@]}"; do
  [[ -e $OUT/STOP ]] && { echo "== STOP file: ending"; exit 0; }
  d=$OUT/$c
  if done_run "$d"; then echo "== $c: done"; continue; fi
  echo "== $c: $(date +%H:%M:%S)"
  mkdir -p "$d"
  # shellcheck disable=SC2046
  systemd-run --user --scope -q -p MemoryMax=${MEM}G -p MemorySwapMax=0 \
    $B/recon_p2np "$SIM" "${EXTRA[@]}" $(cfg_args "$c") --out "$d" 2>&1 | tee -a "$d/log.txt" | grep -E "coil|field|R2\*|ESPIRiT|segments|velocity RMSE|done" | tail -12
  done_run "$d" || { echo "== $c: not finished (stopped or failed; rerun to resume) — see $d/log.txt"; exit 1; }
done
rustlib/.venv/bin/python -I rustlib/tools/recon/eval_table.py "$OUT" "${CFG[@]}"
