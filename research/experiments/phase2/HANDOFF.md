# Phase 2 (arch_targets.md items 2 and 3): handoff for the integration / GPU-campaign agent

Write-up: research/sample_kernels.md §19. Audit tables: audit_tables.md (this directory). ncu: ncu_summary.md.
Nothing of this session is running. No commits were made.

## What exists in the code (all behind switches; the previous kernels are unchanged and are timed first)

- **ra** (`segnufft_ra.cu`, `segnufft/ra.rs`): a staging variant `RaVar { inplace, stages, lbmin }` next to `RaCfg`.
  `RaVar::BASE` = the old kernel.
  - `inplace`: the item list is read in place instead of being copied to shared memory.
  - `stages 2`: cp.async double buffer (needs `inplace`).
  - `lbmin`: `__launch_bounds__(128, m)`.
- **kl** (`segnufft_kl_adj.cu`, `segnufft/kl.rs`): the same `stages` / `lbmin` (it always reads in place).
- **vs / vt** (`segnufft_vs.cu`, `segnufft/vs.rs`): `sch` < smax stages an item in sub-chunks. The tables are the same,
  and the result is bitwise the same.
- **Spill screen.** Registers, local size and occupancy come from the driver. Spill bytes come from a ptxas `-v`
  report, cached as `<kernel cache>/ptxinfo/*.spill`. Local size alone is not a spill test: the sincosf slow path
  gives spill-free 32 B frames.
- **Autotune phases.**
  1. The old candidates.
  2. Staging variants for the fastest two (nxg, cb): chunks 8, 12, 16 and the device fit ≤ 32, each with the register
     prefetch and with cp.async.
  3. Hints 3 / 4 for the fastest three (kept only without spills and with at least the unhinted warps).
  4. vs sub-chunks per extra block count.
- **Env.**
  - `HASTY_SEG_VSA_STAGE = 0 | inplace | cpasync | auto`, `HASTY_SEG_LBMIN = auto | 0 | m`,
    `HASTY_SEG_SPILLSKIP = 1 | 0`, `HASTY_SEG_VS_SCH = auto | 0 | n`.
  - The `_CFG` envs take the extra fields: `HASTY_SEG_VSA_CFG = nxg,cb,chk,inplace,stages,lbmin`,
    `HASTY_SEG_KL_CFG = nxg,cb,chk,stages,lbmin`, `HASTY_SEG_VS_CFG = nwc,nws,vxpad,sch`.
  - The old behaviour exactly: `HASTY_SEG_VSA_STAGE=0 HASTY_SEG_LBMIN=0 HASTY_SEG_SPILLSKIP=0 HASTY_SEG_VS_SCH=0`.
- **Outside the owned files.**
  - segnufft.rs line 36: `pub use ra::{KernRes, RaCfg, RaVar}`;
  - lib.rs: the re-export;
  - hasty-nufft/Cargo.toml: `hasty-site` dependency (the cache path for the ptxas sidecar files).

## Which shapes / stage counts should be the defaults

Measured on the 4090 at 300 W. Velocity reconstruction: 288×288×140 grid, 10 coils × L 10, coil batch 5 = 50 channels,
order 3, 2-D knots 6:1,16:3.

- **ra and kl: nxg 4, cb 7, chunk 12, cp.async (stages 2), no hint.**
  - This is what the autotune picked at 1.33 M and 13.3 M samples per encode.
  - Resources: 125 / 123 registers, 21.1 KB shared memory, 4 blocks = 16 warps per SM (from 12).
  - ncu: 11.85 → 16.3 achieved warps, kernel −11 % (ra) / −14 % (kl).
  - The hint 4 is a tie on the 4090 but is **required on sm_80** (A100), where cp.async alone keeps 156 registers and
    12 warps. Keep the hint dimension in the autotune; do not hard-code it.
- **Chunk 16 with cp.async gives nothing on 100 KB parts** (28 KB → 3 blocks). Chunk 8 reaches 16 warps but is 3 %
  slower than chunk 12 (barriers). On sm_90 / sm_100 chunks up to 24 keep 16 warps; let `chunk_fit` choose there.
- **In place with the register prefetch alone: do not use.** It costs about 33 registers (128 → 161), stays at 12
  warps and is slower. Keep it only as a candidate.
- **vt: nwc 4, sub-chunk 24–29** (the autotune picked 29).
  - Resources: 42–49 KB, 2 blocks = 8 warps on sm_89 / sm_120 (from 4).
  - ncu at 50 channels: −30 % per launch.
  - The whole forward with the shape fixed on every knot: −9.3 % (1.33 M case, 192×192×96 grid).
- **100-channel shapes (coil batch 10, nxg 4 cb 13).** They are register-bound on every architecture (8–12 warps), and
  the hint 4 spills 228–376 B there (the screen rejects it). No footprint lever helps; leave the autotune's choice.
- **Gradient before → after** (the old candidate set vs the defaults, one opbench run each; ob_*.log):
  - 1.33 M per encode: 9919.0 → 9404.2 ms (−5.2 %);
  - 13.3 M per encode: 26061.5 → 25556.3 ms (−1.9 %).

  At 13.3 M the forward did not gain, because the ra / vs autotune still times knot 0 of the 2-D engine, an edge
  respiratory knot. The fix is a one-line change in the segnufft.rs autotune hook: pass the kl step's representative
  knot `kb` to `autotune_ra` / `autotune_vs`. Do it before any default decision is made on 13.3 M data.
- **Autotune cost.** The first run of a new engine shape now compiles about 2× more kernels, plus one ptxas-report
  compile per kernel that has local memory. Both are disk-cached. The per-GPU tune cache (arch_targets.md item 0) would
  also remove the re-timing.

## Tests

`rustlib/crates/hasty-nufft/tests/seg_phase2.rs`. Every test is filterable:
- `P2_ORDERS = "0,2"`, `P2_MI = 16384 | 64` (unsplit / split tile lists);
- run one part per process so each stays under 900 s (`run_tests_split.sh`).

| test | what |
|---|---|
| `p2_ra_static` | ra variants vs the base ra of the same (nxg, cb), static orders 0–2, ZERO tiles / full, odd channel blocks; bitwise when unsplit (fused crop off: the fused crop adds into the image atomically), within the conditioning floor when split |
| `p2_ra_knot` | the same on 1-D (orders 0–3, 15-channel batches, c0 > 0) and 2-D knots |
| `p2_kl_knot` | kl variants vs the base kl, bitwise in both list modes |
| `p2_vs_subchunks` | vs / vt sub-chunks vs one pass, bitwise; orders 0–3, static / 1-D / 2-D, nwc 5 / 7 with x padding; adjointness of the vt-sub-chunk × ra-cp.async pair |
| `p2_spill_screen` | the 48 × 1 shape nxg 2 cb 3 must report a spill on sm_89 / sm_90, nxg 4 cb 3 none |
| `p2_autotune` | autotune with the new candidates, result equal to the base |
| `p2_sanitizer_case` | small case for compute-sanitizer (env `P2_SAN=1`, `P2_SAN_KIND=0..3`, `P2_SAN_SPLIT=1`, `P2_SAN_FAM=ra\|kl\|vs`, `P2_SAN_PAIRS=n`) |

Status:
- All parts pass. Logs: t4_* (frozen binary v4), t5_* (vs with nwc 5 / x padding, v5). 3260 variant runs in total.
- The last test-file edits (`P2_SAN_FAM`, `P2_SAN_PAIRS`) were not run in a normal test pass. They only add filters to
  the sanitizer case.

## Sanitizer: what is done and the case that ran out of memory

Logs: san/.
- memcheck: 0 errors on all 4 cases × unsplit / split.
- racecheck, filtered with `--kernel-name kns=seg_adjoint_ra --kernel-name kns=seg_adjoint_kl --kernel-name
  kns=seg_forward_vs`: 0 hazards on case 0 (static order 0, 316 s) and case 1 (static order 2, 562 s), unsplit.
- **Out of memory: case 2, 1-D knots order 1, racecheck.** It was killed at the 24 GB `MemoryMax` of the unit twice:
  - once with all families in one process;
  - once with `P2_SAN_FAM=ra` only (san/racecheck_k2_s0_ra.log is incomplete).

  Cause: racecheck's host memory grows with each instrumented module, and every variant is its own module. The 1-D knot
  case launches every variant once per knot (6 knots × coil batches) with the knot-window code.
- **How to size it.**
  1. Measure the peak RSS per invocation first: wrap with `/usr/bin/time -v` (reports "Maximum resident set size"), on
     `P2_SAN_KIND=2 P2_SAN_FAM=ra P2_SAN_PAIRS=1`, i.e. one (nxg, cb) and its 10 variants.
  2. Scale by the number of pairs (3) to decide whether a 24 GB unit fits.
  3. If one pair already exceeds it, split further per variant. That needs a small test change: a `P2_SAN_VARIANT`
     index into `variants()`.
  4. Do not raise `MemoryMax` past what the machine has free while other jobs run (no swap; see the memory ledger
     rule).
- **Not run at all:** racecheck on cases 2 / 3 and on all split-list runs; synccheck on everything. `run_san3.sh` is the
  per-(case, family, list mode) driver; it skips what is done.

## Unfinished

- **The seg_* GPU regression.** It has not run. The frozen binaries (session scratchpad bin/reg/, built from this
  session's source) may be stale against later source; rebuild. Expected pre-existing failures: seg_joint_chunks
  (Φ-CGEMM) and seg_ra_vt::ra_static_orders (marginal ≈ 1.05× bound).
- **Racecheck / synccheck** as above.
- **The review is done.** An independent subagent review of the kernel code found no bug; its four robustness remarks
  were applied (§19.3). Nothing open there.
- **The kernaudit rerun is done** for the new shapes: arch_audit/out_p2/, 2970 kernels, all five architectures, 0
  compile errors. The base kernels were re-audited identical (out_p2base/). Rerun only if the kernels change again:
  `kernaudit --families ra2,kl2,vs2 --out out_p2`, about 25 min at 4 threads.
- **Not built:**
  - spill screening of the tile-kernel candidates (segnufft.rs autotune loops);
  - the representative-knot fix above.
