# Operator benchmark matrix (gradient operator formulations)

`matrix.sh` times forward + adjoint of one gradient unit for each operator formulation and situation. It is the test
suite for "which way of applying the gradient operator is fastest here". Results and their reading are in
research/temporal_operator.md §11.

- **Unit:** one encode in one respiratory bin, i.e. everything a temporal basis can share.
  - At 1 mm it holds 3.34 M samples, scaled ∝ 1/res.
- **Formulations:**
  - `perframe`: 20 static images (today's operator; time × 20);
  - `knot`: knot-major B-spline operator (`SegNufft::new_temporal(.., true)`);
  - `interval`: interval-major B-spline operator (`SegNufft::new` with frames > 1).
- **Axes:**
  - resolution: 1 / 1.5 / 2 mm, box FOV;
  - B-spline degree (1, 3) and knot count M;
  - coils, segments L, ΔC correction order.
  - Launch shapes are autotuned per row.
  - `ACBFSB=1` lets the autotune also try register blocking.
  - `GRID_GB` sets the fine-grid budget (coil batch).
  - `QUICK=1` runs 1 mm only.
- **Run:**
  `OUT=results_<date>.csv systemd-run --user --scope -q -p MemoryMax=50G -p MemorySwapMax=0 ./matrix.sh`
  - Duration: about 50 min on the RTX 4090. It holds the GPU for the whole run.
- **CSV columns:**
  - res_mm, variant, deg, knots, coils, nseg, order;
  - samples_per_image;
  - fwd_ms, adj_ms (one call);
  - unit_s (unit time);
  - coil_batch;
  - tile_fwd / tile_adj (autotuned tile-kernel ms per coil batch).
- **Caveat:** uniform-random k-space coordinates, not radial. The static 167 k case reproduces the real per-set time
  (0.82 vs 0.78 s, §4), but a centre-dense trajectory loads the central tiles more [CHECK with a recorded trajectory].
- `results_2026-11-15.csv`: first full run, after the coalescing fix and the knot-major operator.
