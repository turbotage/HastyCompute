# Dense-regime knot kernel work: status (paused 2026-11-16 on coordinator request)

## Implemented (all compiles; last `cargo build --release -j 4` of hasty-bench bins succeeded)
- rustlib/crates/hasty-nufft/src/segnufft_knot.cu: original kernels kept under `#if KV == 0`. New `KV >= 1` section:
  - forward v1: dense unrolled loop, float4 tile-value loads (even NSEG), real weights at order 0 (2 FFMA per MAC).
  - forward v2 (opt-in): union-of-support-box loops, lane order (pp, coil, sample), needs box-sorted tables.
  - adjoint v>=1: sub-box warps (2x2x2 pts x 4 coils / 4x2x2 x 2 / 4x4x2 x 1), ballot sample skip, z = y*a*beta.
  - `stage_kv`: fused staging pass (beta folded, adjoint writes z directly); `knot_beta` helpers (1-D and 2-D).
- segnufft.cu: `wt_t` = float when KV>=1 && CORR==0 (WT_REAL; interval-major kernels compiled out in that case).
- segnufft.rs: `TileCfg.kv` field, `default_kv` (HASTY_SEG_KV, default 1 on knot engines with TT 4/acb 1),
  `kv_candidates` (autotune times 0 and 1), smem formula (z array, float weight rows), `#define KV`, module name `_v{kv}`,
  `SegNufft::tile_shape(...)` (pub, for tests), `box_sort` (opt-in HASTY_SEG_BOXSORT=1) applied in points (knot) and points2d.
- tests/seg_knot_kv.rs (new): v1 and v2 vs v0, many shapes, orders 0-2, NSEG even/odd, 1-D/2-D, split lists. PASSES.
- hasty-bench: segnufft_bench gains HASTY_BENCH_2D="kr,rdeg", HASTY_BENCH_W, HASTY_BENCH_RES; splinerecon prints the
  tuned TileCfgs and honours HASTY_SPLINE_NOTUNE=1 (small edits only; other splinerecon/motion.rs diffs are not mine).
- research/temporal_operator.md: §17 inserted before §18, with 2 mm results; placeholder line `1MM_RESULTS` still there.

## Passing
seg_knot, seg_knot2d, seg_knot_kv (after the last kernel change: float weights). NOT yet re-run after the last changes:
segnufft, seg_wf, seg_multi, seg_pruned, seg_fat_union_gpu (static/interval paths are untouched except the WT_REAL guard,
which is inactive for kv 0).

## Measured (real 2 mm lin6 x cubic10, splinerecon --time-only, autotuned)
| state | fwd | adj | pass |
| v0 (§16.2 baseline) | 9.73 | 9.24 | 18.97 s |
| v1 first form | 7.53 | 7.96 | 15.49 s |
| union fwd + box sort (now v2) | 9.61 | 7.88 | 17.48 s |
| v1 final (fused staging + float wt) | 7.23 | 6.44 | 13.66 s |
Knot-0 kernels (one batch): 42.1/43.7 -> 29.5/30.6 ms. Log: scratchpad/kernels/real_kv1f.log.
Dev (uniform random, half grid; unrepresentative): v0 5.35/4.80 s -> v1 4.42/4.18 s.

## Next steps to resume
1. 1 mm benches (killed before printing): `scratchpad/kernels/mm1.sh` (knot cubic-10 unit + static 167k; autotune
   includes kv 0/1). Compare to the prompt's baseline 5.9 + 4.9 s and 287 + 261 ms; if the knot case regresses,
   run `mm1.sh HASTY_SEG_KV=0` for a same-session baseline.
2. Gate tests: `cargo test --release -j 4 -p hasty-nufft --test seg_knot --test seg_knot2d --test segnufft --test seg_wf
   --test seg_multi --test seg_pruned --test seg_fat_union_gpu --test seg_knot_kv` (under systemd-run MemoryMax).
3. Replace `1MM_RESULTS` in research/temporal_operator.md §17 with the 1 mm table; optional 2-D 1 mm pass via
   segnufft_bench HASTY_BENCH_2D=6,1 HASTY_BENCH_W=10.
4. Final report (<= 25 lines).
