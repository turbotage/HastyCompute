#!/bin/bash
# Operator-validation variants (research/recon_physics_wiring.md item 6): each `opval_v2 forward` writes y_<tag>.npy into
# the case directory; finished tags are skipped (resumable; a file STOP in the case directory stops the loop). One
# process per segment configuration, every trajectory variant of that configuration on the same engine.
# Then: case.py eval. GPU ≤ 4 GB per run (HASTY_OPVAL_GRID_GB 2, one encode's segments at a time); 6 CPU threads.
set -u
REPO=/home/turbotage/Documents/GitHub/HastyCompute
BIN=$REPO/rustlib/.agents/seg/target/release/opval_v2
G=$REPO/data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11
C=$G/case_v2
EST=$G/girf_est_resonance_T200.json
EST80=$G/girf_est_resonance_T80.json
export HASTY_REPO=$REPO HASTY_THREADS=6
run() {  # models, options...
  [ -f "$C/STOP" ] && { echo "STOP"; exit 0; }
  local models=$1; shift
  systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 $BIN forward $C --models "$models" "$@" 2>&1 | grep -v "^warning" | grep -E "forward|segments:|engine:|panicked|error" 
}
B="--fat union --rate fit --conc full --L 15"
run "truthchain=resonance,all=$EST,girf_nominal=ideal,girf_hist1=$EST@1,girf_T80=$EST80,girf_truth_h1=resonance@1" $B
# the term ablations use every 9th support voxel as fit voxel (3x faster segment build on a shared CPU); its effect on
# the segment error: truthchain_f9 vs truthchain
export HASTY_OPVAL_FIT_STEP=9
run "truthchain_f9=resonance,all_f9=$EST" $B
run "fat_mean=$EST,t_fat_mean=resonance" --fat mean --rate fit --conc full --L 15
run "fat_none=$EST,t_fat_none=resonance" --fat none --rate fit --conc full --L 15
run "rate_none=$EST,t_rate_none=resonance" --fat union --rate none --conc full --L 15
run "rate_tab=$EST,t_rate_tab=resonance" --fat union --rate tab --conc full --L 15
run "rate_wt=$EST,t_rate_wt=resonance" --fat union --rate wt --conc full --L 15
run "conc_off=$EST,t_conc_off=resonance" --fat union --rate fit --conc off --L 15
run "conc_mean=$EST,t_conc_mean=resonance" --fat union --rate fit --conc mean --L 15
run "seg_L13=$EST,t_L13=resonance" --fat union --rate fit --conc full --L 13
run "seg_L17=$EST,t_L17=resonance" --fat union --rate fit --conc full --L 17
