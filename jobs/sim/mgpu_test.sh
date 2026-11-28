#!/usr/bin/env bash
# simphantom multi-GPU equivalence: the pre-change binary (old), the new one with 1 worker (n1), 2 workers on GPU 0
# (n2), and 2 workers stopped after the first chunk and resumed (n2r). 1 s of the dev input, chunks of 40 bins,
# beat library + coronary library + resp field + navigator ADC on. Paths inside the repo only.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_arms/ph0001
SF=$P/sim_flow_dev_respfield
O=data/hasty/scratch/mgpu
LOCK=jobs/locks/gpu.lock
NEW=rustlib/target/release/simphantom
OLD=rustlib/target/snap/simphantom_old
rm -rf "${O:?}"
mkdir -p "$O"
export HASTY_IO=npy SIMPH_CHUNK_BINS=40
ARGS=(--seq flow --flow-run data/hasty/seq/flow_thorax --flip 12 --scan-time 1.0 --beats "$P/lbm/run_lib" --beats-ref "$P/lbm/run_cor" --beat-cache 2 --resp-field "$SF/resp_field.hv")
RUN="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0"
for t in old n1 n2; do
  case $t in old) B=$OLD; G=();; n1) B=$NEW; G=(--gpus 0);; n2) B=$NEW; G=(--gpus 0,0);; esac
  /usr/bin/time -f "$t wall %e s, max RSS %M kB" flock "$LOCK" $RUN $B "$SF/sim_input.hv" "$O/$t" "${ARGS[@]}" "${G[@]}" > "jobs/logs/mgpu_$t.log" 2>&1
  tail -1 "jobs/logs/mgpu_$t.log"; grep -E "GPU ready|chunks of|^  bins|simulation " "jobs/logs/mgpu_$t.log" | head -12
done
# n2r: stop after the first finished chunk, then resume
( flock "$LOCK" $RUN $NEW "$SF/sim_input.hv" "$O/n2r" "${ARGS[@]}" --gpus 0,0 > "jobs/logs/mgpu_n2r_a.log" 2>&1 ) &
PID=$!
for i in $(seq 1 600); do grep -q "^  bins .* readouts" "jobs/logs/mgpu_n2r_a.log" 2>/dev/null && { touch "$O/n2r/STOP"; break; }; sleep 1; done
wait $PID
grep -E "^  bins|stopped" "jobs/logs/mgpu_n2r_a.log"
flock "$LOCK" $RUN $NEW "$SF/sim_input.hv" "$O/n2r" "${ARGS[@]}" --gpus 0,0 > "jobs/logs/mgpu_n2r_b.log" 2>&1
grep -E "chunks of|^  bins|simulation " "jobs/logs/mgpu_n2r_b.log"
rustlib/.venv/bin/python -I - "$O" <<'EOF'
import sys, glob, os, numpy as np
O = sys.argv[1]
ref = os.path.join(O, "old")
for t in ["n1", "n2", "n2r"]:
    for f in sorted(glob.glob(os.path.join(ref, "data_clean.npy")) + glob.glob(os.path.join(ref, "nav_*.npy"))):
        n = os.path.basename(f)
        if n.endswith("_samples.npy") or "samples_played" in n:
            continue
        a = np.load(f, mmap_mode="r"); g = os.path.join(O, t, n)
        if not os.path.exists(g):
            print(f"{t} {n}: MISSING"); continue
        b = np.load(g, mmap_mode="r")
        if a.shape != b.shape:
            print(f"{t} {n}: shape {a.shape} vs {b.shape}"); continue
        d = np.abs(np.asarray(a) - np.asarray(b)); r = np.sqrt(np.mean(np.abs(np.asarray(a)) ** 2))
        print(f"{t} {n} {a.shape}: max |diff| {d.max():.3e} ({d.max() / r:.2e} of rms), identical {np.array_equal(a, b)}, zero rows {int((np.abs(np.asarray(b)).reshape(b.shape[0], -1).max(1) == 0).sum())}")
EOF
echo test done
