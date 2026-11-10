# rustlib

Rust re-implementation of the HastyCompute stack. Start with `PLAN.md`; research notes in `../research/`. The old C++/Python stack was removed on 2026-11-10 (last present in commit `232eabf`, see PLAN.md "Legacy features").

```
cargo test --workspace                                  # CPU refs + NVRTC smoke tests (needs a CUDA GPU)
cargo run --release -p hasty-bench --bin cufft_baseline # baseline numbers -> bench/baselines.md
HASTY_DUMP_KERNELS=/tmp/k cargo test ...                # dump generated CUDA sources
```

Toolchain pinned in `rust-toolchain.toml` (installed side by side; the system default is untouched).
