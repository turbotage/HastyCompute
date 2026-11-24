# Preconditioner comparison: status (2026-11-22, campaign 2)

Code fixes (in recon_p2np, behind switches, defaults unchanged): `--holdout-res` (held-out residual inside the solve),
`--pp-metric p2` allowed (Jacobi metric), `--warm-only` (pooled start once, checkpoints kept). Each arm runs as its own
process / unit.

Script: /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/precond_compare/run.sh <name> <timeout s> <extra args>
(GPU lock, 24 G / no-swap scope, /usr/bin/time -v for peak RSS, outputs in
data/hasty/phantom/factory_arms/ph0001/precond_compare/<name>/, DONE marker; an arm copies warm/state_warm_full_done_*.hv
into its directory, so its --warm-full 60 loads the shared pooled start; pause: touch <run dir>/STOP, rerun the same
command to resume; the log is appended).

Started 22:56. Budget extended by the user: GPU until 2026-11-23 06:30, all four arms at 60 iterations, then the best arm (held-out plateau wall time) to 120 iterations if time remains.

- done: warm (60 pooled plain-CG iterations, 18.7 min wall incl. a one-off autotune miss; peak RSS 21.8 GB at setup)
  `systemd-run --user --unit=pc_warm --collect bash /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/precond_compare/run.sh warm 2700 --holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --precond zmshr1 --init warm --iters 60 --warm-only`
- remaining, in this order (A = common arm arguments):
  A = `--holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --init warm --iters 60 --every 5`
  - done 00:10 (55 min wall, peak RSS 21.7 GB, no OOM): zmshr1m: `systemd-run --user --unit=pc_zmshr1m --collect bash .../run.sh zmshr1m 5400 A --precond zmshr1m`
  - zmshr1:  `systemd-run --user --unit=pc_zmshr1 --collect bash .../run.sh zmshr1 5400 A --precond zmshr1`
  - p2:      `systemd-run --user --unit=pc_p2 --collect bash .../run.sh p2 5400 A --precond zmshr1m --pp-metric p2`
  - pe:      `systemd-run --user --unit=pc_pe --collect bash .../run.sh pe 5400 A --precond zmshr1m --pp-metric pe1e-1_0.5`
  - done (pc_chain 23:49–03:03; zmshr1 00:10–01:04, p2 01:04–02:01, pe 02:01–03:03): zmshr1, p2, pe, launched in sequence by chain.sh (each its own unit, started only if expected to end before 06:30):
    `systemd-run --user --unit=pc_chain --collect bash /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/precond_compare/chain.sh` (log: chain unit journal)
  - running since 03:05, expected end ~04:50: zmshr1 to 120 iterations (best stable held-out plateau; zmshr1m's minimum at it 20 is followed by a rise), fresh run = also a replicate of zmshr1's first 60:
    `systemd-run --user --unit=pc_zmshr1_120 --collect bash .../run.sh zmshr1_120 8400 --holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --init warm --iters 120 --every 5 --precond zmshr1`
  - FOUND: setup is not identical between processes (ESPIRiT maps vary slightly → support count, auto L = 5 or 6, held-out at it 0 0.0789..0.0799); see results.md

# Campaign 3 (2026-11-23, from 07:48): step guard, true vs ESPIRiT maps, spatial pyramid

Code (behind switches; defaults unchanged except the calibration, see below), built:
- `--step-guard F` (p2np.rs run_objective_with + hoststack.rs run_objective_host): a ZMSHR1 iterate whose data term
  exceeds F × the previous iterate's is rejected (x, α∇f, f restored from the secant pair, plain scalar step from
  there); one extra gradient at the end checks the last iterate. F = 2 here.
- `--calib-adc out|full`: ESPIRiT calibration and the `--coil-gram data` Gram use only the operator's ADC samples
  (outgoing half); default out when `--adc-part out` (old behaviour: `--calib-adc full`).
- pyramid.sh: `--truth-sampling box,point` per level (default box as before); level DONE also when the recon is .hv.

Script: /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/precond_compare/run_guard.sh <name> <timeout s> <espirit|truth> <extra>
(same config as run.sh; each arm computes its own pooled start in-process: --warm-full 60 plain CG). Pause: touch
data/hasty/phantom/factory_arms/ph0001/precond_compare/<name>/STOP; rerun the same command to resume.

- running (queued on the GPU lock, in order): Run 1 arms, 60 iterations, ZMSHR1 + guard 2
  - `systemd-run --user --unit=pc_g_esp --collect bash /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/precond_compare/run_guard.sh g_esp 6600 espirit --precond zmshr1 --iters 60 --step-guard 2`
  - `systemd-run --user --unit=pc_g_truth --collect bash /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/precond_compare/run_guard.sh g_truth 6600 truth --precond zmshr1 --iters 60 --step-guard 2`
- remaining: Run 2 spatial pyramid 0.25 → 0.5 (command added here when launched)

## 2026-11-23 09:3x (main session)
- done: g_esp (ESPIRiT out-half calibration + step guard 2, 60 it): held-out 0.0613 at it 55, pc RMSE 0.258.
- running: pc_g_truth (truth maps). Queued on the GPU lock: smaps_val (coil-map validation chain), pc_pyr_esp (Run 2).
- Run 2: `systemd-run --user --unit=pc_pyr_esp --collect bash research/experiments/precond_compare/run_pyramid.sh`
  (levels 0.25 → 0.5, 20 + 25 it; pause: touch data/hasty/phantom/factory_arms/ph0001/precond_compare/pyr_esp/STOP).
- queued (main session): g_truth_mask = g_truth + `--coil-mask support` (true maps zeroed outside the data-derived support, as ESPIRiT maps are). One factor: support/mask. Question: is g_truth's worse pc (0.346 vs 0.258) from the maps or from the support (551k vs 428k voxels → prior/LLR/lock act on air)?
  `systemd-run --user --unit=pc_g_truth_mask --collect bash research/experiments/precond_compare/run_guard.sh g_truth_mask 7200 truth --precond zmshr1 --iters 60 --step-guard 2 --coil-mask support`
