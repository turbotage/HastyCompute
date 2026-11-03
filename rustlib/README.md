# rustlib

Rust re-implementation of the HastyCompute stack. Start with `PLAN.md`; research context in `docs/`.

```
cargo test --workspace                                  # CPU refs + NVRTC smoke tests (needs a CUDA GPU)
cargo run --release -p hasty-bench --bin cufft_baseline # baseline numbers -> bench/baselines.md
HASTY_DUMP_KERNELS=/tmp/k cargo test ...                # dump generated CUDA sources
```

Toolchain pinned in `rust-toolchain.toml` (installed side by side; the system default is untouched).
