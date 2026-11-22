#!/bin/bash
# copy the built binaries (opbench, test executables) to the scratchpad so the campaign runs exactly what was built,
# even if the tree is rebuilt meanwhile. Run after `cargo build --release -j 4 -p hasty-bench --bin opbench` and
# `cargo test --release -j 4 -p hasty-nufft --test seg_phase2 --test seg_vs --no-run`.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
R=/home/turbotage/Documents/GitHub/HastyCompute/rustlib/target/release
mkdir -p $SP/wsp/bin
cp $R/opbench $SP/wsp/bin/
for t in seg_phase2 seg_vs; do
  f=$(ls -t $R/deps/$t-* | grep -v '\.d$' | head -1); cp $f $SP/wsp/bin/$t
done
date '+%F %T' > $SP/wsp/frozen_at.txt
ls -la $SP/wsp/bin
