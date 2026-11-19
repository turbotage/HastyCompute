#!/bin/bash
# Reduction survey of hasty_recon::rcompress on the real trajectories (CPU only: selection, designs, direct-sum gate).
# Outputs: out/impl/cpu_<case>.json (rcompress_bench --no-gpu). research/sample_compression.md § Implementation.
#   research/experiments/sample_compress/survey_cpu.sh <cpu.lock>
set -u
cd "$(dirname "$0")/../../.."
L=${1:?cpu lock}
B=rustlib/target/release/rcompress_bench
D2=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
D1=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min_box1mm_b/scan
O=research/experiments/sample_compress/out/impl
run() { local name=$1; shift; flock "$L" systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 $B "$@" --no-gpu --gate 4 --json $O/cpu_$name.json 2>&1 | grep -E "^design|^gate" | sed "s/^/[$name] /"; }
# 2 mm protocol (sim_flow_10min, readout FOV 0.60 m)
run full2_box     $D2 --dims 254,144,114 --vox 2 --roi box --stride 16
run full2_ball    $D2 --dims 254,144,114 --vox 2 --roi ball --stride 16
run box2_box      $D2 --dims 80,80,80 --vox 2 --roi box --stride 16
run box2_ball     $D2 --dims 80,80,80 --vox 2 --roi ball --stride 16
run box2_ball80   $D2 --dims 80,80,80 --vox 2 --roi ball:80 --stride 16
# 1 mm proxy of the 2 mm readout (k x 2)
run box1p_box     $D2 --dims 160,160,160 --vox 1 --kscale 2 --roi box --stride 16
run box1p_ball    $D2 --dims 160,160,160 --vox 1 --kscale 2 --roi ball --stride 16
run box1p_ball80  $D2 --dims 160,160,160 --vox 1 --kscale 2 --roi ball:80 --stride 16
run torso1p_box   $D2 --dims 640,384,384 --vox 0.94 --kscale 2 --roi box --stride 32
run head320p_box  $D2 --dims 320,320,320 --vox 1 --kscale 2 --roi box --stride 32
run head320p_b100 $D2 --dims 320,320,320 --vox 1 --kscale 2 --roi ball:100 --stride 32
# 1 mm protocol (box1mm_b, flow_thorax_v2_1mm readout, readout FOV 0.47 m)
run box1_fov_box  $D1 --dims 198,189,228 --vox 1 --roi box --stride 4
run box1_fov_ball $D1 --dims 198,189,228 --vox 1 --roi ball --stride 4
run box1_box      $D1 --dims 160,160,160 --vox 1 --roi box --stride 4
run box1_ball     $D1 --dims 160,160,160 --vox 1 --roi ball --stride 4
run box1_ball80   $D1 --dims 160,160,160 --vox 1 --roi ball:80 --stride 4
run box1at2_box   $D1 --dims 80,80,80 --vox 2 --roi box --stride 4
run torso1_box    $D1 --dims 640,384,384 --vox 0.94 --roi box --stride 8
run torso1_ball   $D1 --dims 640,384,384 --vox 0.94 --roi ball --stride 8
run full1_box     $D1 --dims 508,287,228 --vox 1 --roi box --stride 8
run head320_box   $D1 --dims 320,320,320 --vox 1 --roi box --stride 8
run head320_ball  $D1 --dims 320,320,320 --vox 1 --roi ball --stride 8
run head320_b160  $D1 --dims 320,320,320 --vox 1 --roi ball:160 --stride 8
run head320_b100  $D1 --dims 320,320,320 --vox 1 --roi ball:100 --stride 8
run head384_box   $D1 --dims 384,384,384 --vox 1 --roi box --stride 8
run head384_b100  $D1 --dims 384,384,384 --vox 1 --roi ball:100 --stride 8
# protocol change: dwell / 2 (2x ADC oversampling; cubic interpolation of the stored records)
run full2_box_dw2   $D2 --dims 254,144,114 --vox 2 --roi box --stride 16 --upsample 2
run full2_ball_dw2  $D2 --dims 254,144,114 --vox 2 --roi ball --stride 16 --upsample 2
run full1_box_dw2   $D1 --dims 508,287,228 --vox 1 --roi box --stride 8 --upsample 2
run torso1_box_dw2  $D1 --dims 640,384,384 --vox 0.94 --roi box --stride 8 --upsample 2
run head320_box_dw2 $D1 --dims 320,320,320 --vox 1 --roi box --stride 8 --upsample 2
run head320_b100_dw2 $D1 --dims 320,320,320 --vox 1 --roi ball:100 --stride 8 --upsample 2
run box1_box_dw2    $D1 --dims 160,160,160 --vox 1 --roi box --stride 4 --upsample 2
run box1_ball80_dw2 $D1 --dims 160,160,160 --vox 1 --roi ball:80 --stride 4 --upsample 2
run full2_box_dw3   $D2 --dims 254,144,114 --vox 2 --roi box --stride 16 --upsample 3
run head320_box_dw3 $D1 --dims 320,320,320 --vox 1 --roi box --stride 8 --upsample 3
echo done
