#!/usr/bin/env bash
# Nsight Compute on simphantom's dsum: one readout launch and one navigator launch of the dev input (beat library,
# resp field, navigator on, 32 coils as in production), full section set. Paths inside the repo only.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_arms/ph0001
SF=$P/sim_flow_dev_respfield
O=data/hasty/scratch/ncu
LOCK=jobs/locks/gpu.lock
mkdir -p "$O"
rm -rf "${O:?}/run"
export HASTY_IO=npy
ARGS=(--seq flow --flow-run data/hasty/seq/flow_thorax --flip 12 --scan-time 0.3 --beats "$P/lbm/run_lib" --beats-ref "$P/lbm/run_cor" --beat-cache 2 --resp-field "$SF/resp_field.hv" --gpus 0)
# readout and navigator launches alternate per bin: skip the first bins' launches, take two consecutive ones
flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
  /usr/local/cuda/bin/ncu --kernel-name dsum --launch-skip 6 --launch-count 2 --set full --import-source yes \
  -o "$O/dsum_dev" -f rustlib/target/release/simphantom "$SF/sim_input.hv" "$O/run" "${ARGS[@]}" > "jobs/logs/ncu_dsum_run.log" 2>&1
/usr/local/cuda/bin/ncu --import "$O/dsum_dev.ncu-rep" --page details --print-units base > "$O/details.txt" 2>&1
/usr/local/cuda/bin/ncu --import "$O/dsum_dev.ncu-rep" --page raw --csv > "$O/raw.csv" 2>&1
rm -rf "${O:?}/run"
echo ncu done
