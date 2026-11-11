#!/usr/bin/env bash
# Coarse-to-fine spatial pyramid for recon_p2np (hasty_recon::pyramid): every level reconstructs the same scan on a
# grid of the same FOV at scale s (--grid-scale s, only the k-space samples inside that grid's band: --kcrop) and
# starts from the previous level's images (--init-from <previous level dir>, band-limited resampling).
#   rustlib/tools/recon/pyramid.sh <sim_dir> <out_root> [--levels 0.25,0.5,1] [--iters 10,10,20] [--kcrop FRAC[:prefix]]
#       [--first-args "ARGS"] [--next-args "ARGS"] [--mem GB] [--bin DIR] [--redo] [--allow-new-store] -- <recon_p2np arguments for every level>
# --levels: grid scales, coarse to fine; 1 = the native grid (no --grid-scale, no crop: every sample is in its band).
# --iters: iterations per level (one number: all levels).
# --kcrop: crop of the levels with s < 1 (default 1 = the grid's whole band; `:prefix` = outgoing readout prefixes
#   only, fewer B0 / concomitant segments; see recon_p2np --kcrop). Coarse levels also get --truth-sampling box.
# --first-args: extra arguments of the first level only (e.g. "--warm-full 10": a physics-model warm start is cheap on
#   the coarsest grid; at a finer level it would replace the per-frame --init-from start by an encode average).
# --next-args: extra arguments of every level after the first; `{prev}` becomes the previous level's directory
#   (e.g. --next-args "--air-init {prev_air}": `{prev_air}` = the previous level's newest air_map_*.npy). --warm n may stay in the common arguments: levels with
#   --init-from skip it.
# Resumable: a finished level has <out_root>/L<i>_s<s>/DONE and is skipped; a level stopped by Ctrl-C, SIGTERM or
# `touch <out_root>/STOP` (forwarded to the running level) keeps its checkpoint and the same command continues it.
# --redo: forget the DONE markers (the levels' own checkpoints / results still decide what is recomputed; delete a
#   level directory to recompute it).
# Each level runs under a memory cap without swap (--mem, default 60 GB). The k-space store is shared with the
# full-resolution runs (its key does not depend on the grid): HASTY_KSTORE_DIR as in eval_10min.sh; a level that
# would have to write a new file store stops instead (HASTY_KSTORE_REUSE_ONLY=1) unless --allow-new-store.
# Example (the 10-min thorax scan, oracle physics, 2 respiratory bins):
#   rustlib/tools/recon/pyramid.sh data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan \
#     data/hasty/phantom/factory_arms/ph0001/pyramid --levels 0.25,0.5,1 --iters 10,10,20 --first-args "--warm-full 10" -- \
#     --gnl --phys full --kstore disk:zstd1 --table-store zstd --resp-bins 2 --resp bellows --fat-fraction truth \
#     --fat-model joint --r2s truth --kappa 1 --precond zmshr1s --warm 20
set -uo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
[[ $# -ge 2 ]] || { sed -n '2,27p' "$0"; exit 2; }
SIM=$(realpath "$1"); ROOT=$(realpath -m "$2"); shift 2
LEVELS=0.25,0.5,1; ITERS=10,10,20; KCROP=1; FIRST=""; NEXT=""; MEM=60; BIN=$R/rustlib/target/release; REDO=0; NEWSTORE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --levels) LEVELS=$2; shift 2 ;;
    --iters) ITERS=$2; shift 2 ;;
    --kcrop) KCROP=$2; shift 2 ;;
    --first-args) FIRST=$2; shift 2 ;;
    --next-args) NEXT=$2; shift 2 ;;
    --mem) MEM=$2; shift 2 ;;
    --bin) BIN=$2; shift 2 ;;
    --redo) REDO=1; shift ;;
    --allow-new-store) NEWSTORE=1; shift ;;
    --) shift; break ;;
    *) echo "unknown option $1 (recon arguments go after --)"; exit 2 ;;
  esac
done
COMMON=("$@")
for a in "${COMMON[@]}"; do
  case $a in
    --fresh) echo "!! --fresh would discard a paused level's checkpoint on resume: delete the level directory instead"; exit 2 ;;
    --out|--grid-scale|--kcrop|--init-from|--iters) echo "!! $a is set per level by the driver"; exit 2 ;;
  esac
done
IFS=, read -r -a LV <<< "$LEVELS"
IFS=, read -r -a IT <<< "$ITERS"
[[ ${#IT[@]} -eq 1 ]] && for ((i = 1; i < ${#LV[@]}; i++)); do IT[i]=${IT[0]}; done
[[ ${#IT[@]} -eq ${#LV[@]} ]] || { echo "!! --iters: one value or one per level (${#LV[@]})"; exit 2; }
export HASTY_KSTORE_DIR=${HASTY_KSTORE_DIR:-$R/rustlib/target/kstore}
export HASTY_REPO=${HASTY_REPO:-$R}
# never write a new k-space file store from a pyramid level (reuse the full-resolution runs' store; --allow-new-store)
export HASTY_KSTORE_REUSE_ONLY=${HASTY_KSTORE_REUSE_ONLY:-$((1 - NEWSTORE))}
mkdir -p "$ROOT"
rm -f "$ROOT/STOP"
echo "== pyramid $SIM → $ROOT: levels ${LV[*]}, iterations ${IT[*]}, crop $KCROP; $(date +%H:%M:%S)"
PREV=""
for ((i = 0; i < ${#LV[@]}; i++)); do
  s=${LV[i]}
  L=$ROOT/L${i}_s$s
  mkdir -p "$L"
  [[ $REDO -eq 1 ]] && rm -f "$L/DONE"
  if [[ -f $L/DONE ]]; then
    echo "== level $i (s = $s): done earlier, skipped"
    PREV=$L
    continue
  fi
  [[ -e $ROOT/STOP ]] && { echo "== STOP file: ending before level $i"; exit 0; }
  ARGS=("$SIM" "${COMMON[@]}" --iters "${IT[i]}" --out "$L")
  if [[ $(awk -v s="$s" 'BEGIN { print (s + 0 == 1) }') -ne 1 ]]; then
    ARGS+=(--grid-scale "$s" --kcrop "$KCROP" --truth-sampling box)
  fi
  if [[ -z $PREV && -n $FIRST ]]; then
    read -r -a FX <<< "$FIRST"
    ARGS+=("${FX[@]}")
  fi
  if [[ -n $PREV ]]; then
    ARGS+=(--init-from "$PREV")
    if [[ -n $NEXT ]]; then
      # {prev_air}: the previous level's air map (its file name carries the level's tag, e.g. a trailing "s" when
      # that level was itself seeded), resolved here instead of guessed
      PA=$(find "$PREV" -maxdepth 1 -type f -name 'air_map_*.npy' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
      NX_S=${NEXT//\{prev_air\}/$PA}
      read -r -a NX <<< "${NX_S//\{prev\}/$PREV}"
      ARGS+=("${NX[@]}")
    fi
  fi
  echo "== level $i (s = $s, ${IT[i]} iterations): $(date +%H:%M:%S)"
  echo "   $BIN/recon_p2np ${ARGS[*]}" | tee -a "$L/pyramid_cmd.txt"
  # forward a STOP of the pyramid to the running level (its run control removes its own STOP on start)
  ( while sleep 5; do [[ -e $ROOT/STOP ]] && { touch "$L/STOP"; break; }; done ) &
  WATCH=$!
  # (the log filter ignores SIGINT: Ctrl-C reaches recon_p2np, which checkpoints while its output still has a reader)
  systemd-run --user --scope -q -p MemoryMax=${MEM}G -p MemorySwapMax=0 "$BIN/recon_p2np" "${ARGS[@]}" 2>&1 | ( trap '' INT; tee -a "$L/log.txt" \
    | grep -E "reconstruction grid|k-space crop|initial images|warm start|segments \(auto\)|SegNufft:|lambda|κ .* it +[0-9]+:|stopped|finished|panicked|error" )
  rc=${PIPESTATUS[0]}
  kill "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null
  [[ $rc -eq 0 ]] || { echo "!! level $i failed (exit $rc): $L/log.txt"; exit 1; }
  # finished: results written and no solver state left (a stopped run exits 0 with its state saved; the warm
  # starts' final states state_warm*_done_* stay by design)
  if compgen -G "$L/recon_*.npy" >/dev/null && [[ -z $(find "$L" -maxdepth 1 -name 'state_*.hv' ! -name 'state_warm*' -print -quit) ]]; then
    touch "$L/DONE"
    echo "== level $i done: $(ls -t "$L"/recon_*.npy | head -1)"
  else
    echo "== level $i paused (checkpoint in $L): rerun the same command to continue"
    exit 0
  fi
  PREV=$L
done
echo "== pyramid done: $PREV ($(date +%H:%M:%S))"
