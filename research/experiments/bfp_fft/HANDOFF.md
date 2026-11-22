# Handoff: BFP y ↔ z grid-FFT intermediates (step 1), for the combined GPU campaign

Results and design: research/bfp_storage.md §8.1. Nothing here is committed. Nothing is running.

## Switches and defaults

| switch | values (default first) | set for the campaign |
|---|---|---|
| `HASTY_SEG_FFT_STORE` | `f32` · `bfp16` … `bfp24` | `f32` (baseline), then `bfp24`, `bfp22`, `bfp20` |
| `HASTY_SEG_FFT_BFP_LAYOUT` | `inplace` · `dense` | `inplace` (recommended); `dense` only for one comparison row |

- **In place** keeps the mantissas inside the grid slot. Only the exponent bytes are extra (29 MB at the
  288 × 288 × 140 grid with 50 channels).
  - It needs a plane's line count n0 · nch to be a multiple of 16. If not, or with `dense`, the engine allocates a
    separate buffer instead: 0.56 of a slot at bfp22.
  - The engine prints its choice in the FFT description line, e.g. "y ↔ z intermediates bfp24 (in the grid slot,
    0.029 GB extra)".
- **Recommended final setting: bfp24 in place.** It moves the same DRAM bytes as bfp22 in place, writes whole sectors,
  and every measured error is at the f32 one-ulp floor.
- **Per engine in code:** `SegNufft::set_fft_store(Store::Bfp(24))`. The thread default for tests is
  `hasty_nufft::set_thread_fft_store`.
- **Limit:** on 99 KB-per-block cards, y or z lengths above 396 use 8-line tiles. BFP is then refused and the engine
  stays f32 with a printed note (e.g. a 480-point axis at 1 mm).

## Tests (all pass on the final build)

```
cd rustlib
cargo test --release -j 4 -p hasty-fft --test bfp_store -- --nocapture --test-threads 1      # ~2 s cached, 50 s cold
cargo test --release -j 4 -p hasty-fft --test chanfft -- --test-threads 1 pruned_matches_full ksupport_matches_full slab_passes_equal_whole_grid
cargo test --release -j 4 -p hasty-nufft --test seg_fft_store -- --nocapture --test-threads 1
HASTY_SEG_FFT_BFP_LAYOUT=dense cargo test --release -j 4 -p hasty-nufft --test seg_fft_store -- --test-threads 1 bfp22_engine_matches_f32
cargo test --release -j 4 -p hasty-nufft --test seg_fft_traffic -- --test-threads 1
HASTY_SEG_FFT_STORE=bfp22 cargo test --release -j 4 -p hasty-nufft --test seg_fft_traffic -- --test-threads 1
```

Selection variables for short runs:
- `HASTY_BFP_TEST_CASES` (comma-separated case indices) and `HASTY_BFP_TEST_BITS` (widths) for `bfp_store`;
- `HASTY_BFP_TEST_CASES` for `seg_fft_store`'s `bfp22_engine_matches_f32`.

Phantom accuracy, already measured (§8.1), takes about 70 s per case and 12 GB host RAM:
```
target/release/bfp_phantom data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan --order 0|3 [--knots2d 3:1,8:3 --readouts 340] [--stores bfp24,bfp22,...]
```

## Sanitizer: what ran, what was killed, and how to size the rest

- **Done** (`san/`): 0 errors and 0 hazards everywhere.
  - memcheck: 12 runs.
  - racecheck: 10 runs, namely `bfp_store` layouts 0, 1, 7, 10, 11, 12 at one width each, the short-batch test, and
    engine cases 0 and 1.
- **Killed:** racecheck of engine case 6 (`seg_fft_store` `bfp22_engine_matches_f32`, 2-D knots order 3,
  `HASTY_BFP_TEST_CASES=6`). The unit's 24 GB cap stopped it.
  - Racecheck's host shadow memory grows with the shared-memory traffic of every kernel it checks, the engine's
    tile / gather kernels included.
  - Measured peak RSS:

    | run | peak RSS |
    |---|---|
    | one `bfp_store` layout at one width | 1.1–4.4 GB |
    | short batch | 9.1 GB |
    | engine cases 0 and 1 | 22.4 and 23.4 GB |

- **Not run:** synccheck, all cases.
- **Sizing:** check only the FFT kernels, where all the BFP code lives. Add `--kernel-name kns=fft_axis` to racecheck
  and synccheck of the engine tests. This should bring the engine cases down to the `bfp_store` range; measure the peak
  once with `/usr/bin/time -v`.
  - Otherwise run one engine case per invocation under a cap of about 32 GB, with nothing else heavy on the host.
  - Script with the case list: `run_san.sh`. Its `.done` files skip the runs already done. Add the kernel filter to the
    `seg` lines before resuming.

## Timing cases (planned, not run)

All at the 2 mm thorax: image 225 × 225 × 112, grid 288 × 288 × 140, 10 virtual coils, L 10, coil batch 5
(50 channels). The script is `run_bench.sh`, with the binaries frozen in the session scratchpad; rebuild them from the
current tree.

| case | command | stores |
|---|---|---|
| grid transform alone, ball k-support (columns 0.85, cells 0.64) | `cargo run --release -p hasty-fft --example bfp_fft_bench -- 288 288 140 225 225 112 50 --ball 151,82 --slab 1 --reps 15 --store S [--dense]` (also `--slab 0`) | f32, bfp24, bfp22, bfp20; bfp22 `--dense` |
| 2-D knots 6:1,16:3, order 3 | `HASTY_SEG_FFT_STORE=S opbench --models seg-full --nseg 10 --vcoils 10 --sets 1 --seq flow --knots2d 6:1,16:3 --scan-min 0.44 --reps 3 --order 3` | f32, bfp24, bfp22, bfp20; bfp22 dense |
| the same, order 2 | `--order 2` | f32, bfp22 |
| static (one set) | `HASTY_SEG_FFT_STORE=S opbench --models seg-full --nseg 10 --vcoils 10 --order 2 --frames 7 --scan-min 0.44 --sets 1 --reps 3 --seq flow` | f32, bfp24, bfp22, bfp20 |
| drift check | the 2-D knot order-3 f32 case again at the end | f32 |
| per-pass DRAM bandwidth | `ncu --kernel-name fft_axis --launch-skip 6 --launch-count 6 --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__inst_issued.avg.pct_of_peak_sustained_active,launch__registers_per_thread --csv bfp_fft_bench … --slab 0 --once` | f32, bfp24, bfp22, bfp22 dense |

Decision rules, fixed before running:
- **Keep threshold:** keep the switch if the 2-D-knot gradient falls by at least 0.6 × the byte model's saving.
  - In place: model 21.3 → 17.95 GB per load + flush, −0.97 s per gradient, threshold **0.58 s**.
  - Dense bfp22: 17.1 GB, −1.21 s, threshold **0.73 s**.
- **ncu check:** the BFP y and z passes keep issue utilisation ≤ 50 % and DRAM throughput ≥ 85 %.
- **Partial-sector check:** see whether bfp22's partial last sector (88 of 96 bytes) shows extra DRAM reads in the
  writing pass. bfp24 should not.
- **Power limit:** record it per run (300 W when the work was done), and compare only within one campaign.

## Known open items

- **Multi-image budget.** The multi-image passes' extra grid transforms (`multi_plan`, used by interval-major order 3
  and fat) each own an exponent array (in place) or a dense buffer. The budget does not count them.
- **Re-budget after a switch.** `SegNufft::set_fft_store` switches the store after construction without re-budgeting
  the coil batch. With in place the difference is tens of MB; with dense it is 0.56 of a slot.
- **Racecheck and synccheck** of the engine cases: see the sanitizer section above.
- **The gradient comparison with the NUFFT's own error is unmeasured.** The engine amplifies grid-level errors about
  17–22 × into Aᴴ n and Aᴴ A x (§8.1). BFP error ≪ NUFFT error is established for the forward; for the gradient it
  assumes the NUFFT's errors are amplified the same way.
- **Undefined cells now hold mantissa bits.** With k-support pruning in place, cells outside the support hold BFP
  mantissa bits instead of stale floats. The gather never reads them, but any debug dump of a whole slot will show
  NaNs there. Use the block-pruned transform (`set_kprune(false)`) for whole-slot comparisons.
