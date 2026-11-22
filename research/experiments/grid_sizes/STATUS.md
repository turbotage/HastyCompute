# §18 grid-size gradient study — status at pause (2026-11-20 ~09:15)

Paused on request. STOP file touched; all gs-* units stopped (gs-a, gs-bp, gs-power, gs-fft, gs-t1, gs-san1). No GPU
process of mine is running. Nothing committed.

## Done
- **Baseline campaign `a/`** (binary scratchpad/bin/opbench_gs0, static engine, one (frame, encode) set): 20 of 43
  cases have `.done` and records in `a/results.jsonl` (all 192³ cases; 256³ cb 1/2/5 at F 20 and F 67, orders 0 and 3).
  `a/a256_o0_f7_cb5` was interrupted mid-run (log only, no .done, no record) and reruns on resume.
- **FFT pass bandwidth `fft/`** (fftpass_bench under nsys, 16 configs, all done): whole-grid passes run at 86–91 % of
  the 1008 GB/s DRAM peak at every grid (240³ … 800×480×400, 2–100 channels); L2 slabs save 1–23 % of the transform
  but their passes run L2-limited (≈ 50–70 % of the DRAM-equivalent rate). `trace.py` groups the per-launch trace.
- **Tests (all pass):** `tests/t1_seg_split.log` (split_matches_plain, one_bin_knot_matches_static),
  `tests/t2_sanitize.log` (one_bin_sanitize under compute-sanitizer memcheck: 0 errors). CPU unit test
  `segsplit::tests::plan_batch_rules` passes.

## Code state (uncommitted; the tree builds: opbench, recon_p2np, fftpass_bench, hasty-nufft lib + tests)
- new `rustlib/crates/hasty-nufft/src/segsplit.rs`: `SegSplit` (segment groups inside a coil), `plan_batch`,
  `BatchPlan`, `SegSplit::planned`; exported in hasty-nufft lib.rs.
- `segnufft.rs` (dispatch only): `forward_accumulate` flag (honoured in `init_rows`); one-bin knot engine allowed
  (`knot_major && (frames > 1 || bspline == 0)`).
- new test `rustlib/crates/hasty-nufft/tests/seg_split.rs`.
- `opbench.rs`: `--matrix`, `--profile`, `--cb-list`, `--jsonl`, `--seg-batch`, `--static-knot`, `--plan-budget-gb`.
- `recon_p2np.rs`: opt-in `--grid-budget legacy|auto|SIZE` (default legacy = unchanged behaviour).
- new `rustlib/crates/hasty-bench/src/bin/fftpass_bench.rs`.
- Not yet done: seg_* GPU regression rerun, the §18 write-up in research/sample_kernels.md, planner cases (C).

## Not started / pending
- `b/` (tuned: one-bin knot path, segment split at 512³ / 640×384×320, 16×16) — cases_b.txt, binary opbench_gs1.
- `p/` (nsys per-kernel profiles of one gradient) — cases_p.txt, binary opbench_gs1.

## Resume commands
```
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
rm -f $D/STOP
systemd-run --user --unit gs-power -p MemoryMax=1G $D/power_log.sh
systemd-run --user --unit gs-a -E PATH=$HOME/.cargo/bin:$PATH -E BIN=$SP/bin/opbench_gs0 -E TAG=a -E CASES_FILE=$D/cases_a.txt -p MemoryMax=24G -p MemorySwapMax=0 $D/run_cases.sh
systemd-run --user --unit gs-bp -E PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH -p MemoryMax=24G -p MemorySwapMax=0 $D/chain_bp.sh
```
(gs-bp waits for gs-a, then runs b/ and p/.) Remaining GPU time ≈ 23 baseline cases × ~2.5 min + 40 tuned × ~3 min
+ 7 profiles ≈ 3.5 h at the 300 W limit. If the scratchpad binaries were cleared, rebuild with
`cargo build --release -j 4 -p hasty-bench --bin opbench` and copy to $SP/bin/opbench_gs0 / opbench_gs1 (gs1 = current
tree; gs0 only lacks --static-knot / --seg-batch / --plan-budget-gb, which the a/ cases do not use).

## Resumed 2026-11-20 11:56 (power limit 300 W, logged in power.log and per case)
- gs-power, gs-a (baseline remainder) and gs-bp running; gs-bp = b/ (tuned) → p/ (nsys) → c/ (planner, cases_c.txt)
  → run_reg.sh (seg_* regression on frozen binaries in scratchpad/bin/reg, logs tests/reg_*.log). Binary for b/p/c:
  scratchpad/bin/opbench_gs2.
- No further edits to crates/hasty-fft (coordination rule; I made none at any point).
- segnufft.rs edits, complete list (all dispatch-level):
  1. field `forward_accumulate` (+ initialiser) and its use in `init_rows` (skip zeroing y when set);
  2. `new_impl`: knot-major also for one bin (`frames == 1 && bspline == 0`), and env `HASTY_SEG_STATIC_KNOT=1`
     (default off) builds every static engine that way; doc line on `new_temporal`.
- New test `seg_split::static_knot_env_switch` (in the frozen seg_split binary of the regression run).

## 2026-11-20 15:35
- a/ (43), b/ (40), p/ (7), c/ (6) all done, every case exit 0, power limit 300 W throughout.
- planner rule changed in segsplit.rs (fewest transformed channels, then fewest passes; avoids ragged batches);
  opbench_gs3 has it (used for c/).
- My earlier frozen regression dir scratchpad/bin/reg was overwritten at 13:30 by another agent (binaries of unknown
  provenance; logs moved to tests/old_foreign_reg/). Re-frozen at 15:33 into scratchpad/bin/gs_reg from the current
  tree, which by then also contains other agents' concurrent edits (hasty-bfp, hasty-fft bfp storage, segnufft*.cu /
  segnufft.rs changes not mine). Regression = gs-reg unit, logs tests/reg_*.log.

## 15:40
- a/ (43), b/ (40), p/ (7 profiles), c/ (6 planner cases) all done, all exit 0, power limit 300 W throughout.
- Planner rule changed (segsplit.rs `plan_batch`): fewest transformed channels, then fewest passes (ragged last coil
  batch measured slower at 320³). c/ ran with opbench_gs3 (this rule).
- Another writer shares this session's scratchpad and edited the §18 drafts (scratchpad/gs/s18_*.md), tables_md.py,
  run_reg.sh and bin/reg; see COORD.md. The seg_* regression now runs as unit gs-reg (not started by me) on bin/gs_reg
  built 15:29 from the shared tree (includes other agents' later segnufft.rs / .cu edits).
- gs-s: HASTY_SEG_FFT_SLAB=0 on four b/ cases (s/), queued behind the lock.
- §18: assemble.py → scratchpad/gs/s18_full.md, to be appended once the regression finishes.

## Final (2026-11-20 ~17:20)
- All campaigns done (a 43, b 40, p 7, c 6; every case exit 0; 300 W). Regression done (tests/reg_*.log): all pass
  except the known seg_joint_chunks and seg_ra_vt::ra_static_orders, plus seg_warp_gpu::warp_adjoint_gpu (flaky
  run-to-run bound; passes on rerun, tests/warp3.log; gs-warp may still be finishing its 3rd rerun, waiting on the
  shared GPU lock).
- §18 inserted in research/sample_kernels.md (before the other agent's §19) and the Status line updated.
- segnufft.rs edits (complete): `forward_accumulate` field + `init_rows` test; one-bin knot condition +
  `HASTY_SEG_STATIC_KNOT` env switch in `new_impl`; doc lines. No hasty-fft edits.
- Note: a second, foreign draft (scratchpad/gs/s18_tail_gs.md) and an edit to my s18_results.md appeared at 15:40;
  the unmeasured claim in it ("slabs off 3–7 % slower") was removed before writing §18.
- Nothing committed. No gs-* unit of mine left running except possibly gs-warp (finishes by itself).
