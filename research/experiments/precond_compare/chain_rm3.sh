#!/bin/bash
# 2026-11-24 ~01:30, after the audits (results.md "Audits 2026-11-24"). Fixed frame mode (snap_rm3): one table per
# (frame, encode) with the encode's own w̄_e, auto:noise:0.2 + svdn + energy (as knot mode), startup self-check.
# One factor: the static reference gets the same segment settings (auto:noise, svdn, energy), so the pair differs only
# in the ratemap (static field, no fat / decay  vs  true per-frame field + water/fat decay + fat fraction).
# Both on the full prior of g_esp_air (comparison with g_esp_air), fresh --out. Queued behind chain_comp on the GPU lock.
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
SEG="--nseg auto:noise --seg-basis svdn --seg-weight energy"
RG_BIN=rustlib/target/admm/snap_rm3/recon_p2np RG_MEM=40G bash $E/run_guard.sh s_esp_air_svdn 9000 espirit --precond zmshr1 --iters 60 --step-guard 2 --air auto $SEG
RG_BIN=rustlib/target/admm/snap_rm3/recon_p2np RG_MEM=64G bash $E/run_guard.sh g_esp_air_rm3 12000 espirit --precond zmshr1 --iters 60 --step-guard 2 --air auto --ratemap truth-bc $SEG
