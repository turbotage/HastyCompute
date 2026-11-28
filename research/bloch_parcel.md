# Bloch parcel viewer (2026-11-25)

A parcel of ~200 spins is carried along pathlines of a 4D-flow value while the designed flow sequence is played. The page
(`plotting_website/bloch`) shows every spin as an arrow on the unit sphere with the ensemble mean, the vessel-tree MIP with
the parcel, the played TR (RF, gradients, mean magnetisation) and the echo / M_z over all TRs.

## Pieces
- `hasty_seq::bloch` (`rustlib/crates/hasty-seq/src/bloch.rs`): engine. Exact rotation per 10 µs raster step about the
  effective field (hard pulse with RF-spoiling phase, physical gradients of every encode and rotation, off-resonance,
  concomitant Maxwell terms, B1 scale), T1 / T2 split per step, no diffusion. `FlowDesign::load` reads a `seqdesign` run
  directory (`final.json`, `summary.json`, `tr_grad.npy`, `spoke_grad.npy`); TR i = encode i mod 7, rotation i / 7
  (super-Fibonacci, acquisition order, 6000 rotations), RF phase φ_n = inc·n(n+1)/2 counted from the first pre-roll TR.
- Server commands (`hasty-server/src/bloch.rs`): `bloch_designs`, `bloch_parcel` (options in its doc string; outputs m, pos,
  mean, g). Paths from `Flow::parcel_paths` (RK4 through the cyclic field, `substeps` per frame); without a flow value the
  parcel moves with a constant velocity. Designs live in `<data>/seq` (env HASTY_SEQ_DIR).
- Start modes: `relaxed` (M_z = M0 at the first TR: a parcel entering from the vena cava) and `steady` (spins first rest at
  their start positions for `preroll` TRs, default 600 ≈ 8.5 s, with the same encode / rotation / RF-phase sequence).

## Checks (unit tests in bloch.rs)
- 90° hard pulse about +x turns M_z into +y (left-handed precession), first echo of a relaxed parcel = sin α.
- Toy design with a large spoiler and 117° RF spoiling: steady-state echo = Ernst (T2 = 30 ms, within 5 %; with T2 = 0.1 s the
  known T2-dependent RF-spoiling deficit is ≈ 5 %).
- v2 design, static blood parcel (T1 1.7 s, T2 0.2 s, 15°): steady echo 0.0484 vs Ernst 0.0504 (−4 %; 760 TRs × 200 spins 0.33 s).

## Slab excitation and the actual v2 (2026-11-25)
- Default design is `flow_thorax_v2_1mm` (the one the realistic scans use). Its TrSpec has only a 100 µs hard pulse, so
  `FlowDesign::with_slab` swaps the RF block: 100 µs dead, windowed-sinc (Hamming) B1 over 0.8 ms, TBW 6 (simphantom's
  defaults) with G_ss = BW / (γ̄ · thickness) on z (1.0 mT/m for 175 mm), a rephasing lobe of half the area laid onto the start
  of the unchanged encode block, 30 µs gap. The TR grows by the new RF block − the old one (v2_1mm: 15.17 → 15.87 ms) and TE
  from the new RF centre 1.58 → 1.93 ms: the real sequence needs this until the designer gets a slab (sequence_design.md
  TODO). The rephaser is summed with the encode lobes and is not re-checked against G_max / slew.
- Slab centre ≠ iso: B1 rotates at γ G_ss z_c (transmit frequency offset). Test: centre flip 15.0°, FWHM ≈ 175 mm, < 0.01° at
  ±0.2 m, centre shift follows.
- LBM data: `rustlib/tools/bloch/show_lbm.py` publishes `<scan>/truth_vel_logical` (+ lumen mask) of a simulated scan; the logical
  grid is centred on the isocentre (rot_presc = identity), so voxel − centre = position in the gradient frame. Blood after
  gadolinium: T1 0.388 s, T2 0.111 s (case.json); native 1.7 / 0.2.

## Notes / limits
- `rf = hard` keeps the design's non-selective pulse.
- Precession sense is the physical one (clockwise) by default; the recon's signal model uses the opposite sign (phase
  +γ M0·x) — `left_handed = false` switches.
- Not modelled yet: diffusion, gradient nonlinearity, GIRF-played gradients (nominal waveforms), B0 map, coil sensitivity.
- Viewer: URL presets `?id=<flow id>&auto=1&start=steady&ntr=150&seek=3000`, `?plug=1` for plug flow.

## Slow replay and separating spread from saturation (2026-11-25)

- Rates down to 2 µs of sequence time per second, single-sample step (◀| |▶, arrow keys), "loop this TR", and **fine here**
  (re-simulates the current TR + next with every 10 µs step recorded; same seed, so same spins).
- Fine window default now TRs 0–3 at every raster step.
- A shrinking vector sum |⟨M⊥⟩| has two causes: spin spread (dephasing across the parcel, mostly the spoiler over a 3 mm parcel
  vs ~1 mm voxels) and real saturation of each spin. The per-spin mean ⟨|M⊥|⟩ ignores spread. Shown as: dashed line in the
  timeline lane, green dashed line in the echo history, dashed circle in the scope. Control: radius 0 mm = identical spins.
- Scope inset: transverse plane scatter (white = vector sum, dashed circle = mean length) + Mz of every spin. "show components"
  and "camera" selects give transverse-only / longitudinal-only 3D views, top / side cameras.
- bloch_parcel has a 5th output `mean_inc` and meta `echo_inc` [⟨|M⊥|⟩, ⟨Mz⟩, ⟨|M|⟩] per TR.

## Hover help (all sites)

`plotting_website/common/help.js` (`installHelp(HELP)`) + per-page `src/help.js` ({control id: text}). Hover a label ~0.9 s → tooltip.
Used by bloch, flow, mesh, orthoslicer; `?helpcheck=1` logs controls without text. New controls need an entry in the page's help.js.

## 1 µs simulation step (2026-11-25)

`FlowDesign::refine(k)` puts the design on a k× finer raster (corner gradients linearly interpolated = same piecewise-linear waveform;
slab envelope re-sampled; slab ramps stay 10 µs because refine runs after `with_slab`). `bloch_parcel` option `dt_us` (default 1).
Steady-state echo agrees within 3 % with the 10 µs run (test `refined_raster_agrees`). Cost: 10× — 600 pre-roll + 300 TRs, 200 spins ≈ 5 s.
Detail stride is now entered in µs in the UI; default fine range is TR 0 only (one TR at 1 µs ≈ 16 000 samples/spin ≈ 76 MB at 200 spins).

## Slab edge is soft (2026-11-25)

TBW 6 / 0.8 ms / 175 mm slab, nominal flip 15°: flip at distance from the centre — 50 mm 14.5°, 80 mm 9.5°, 90 mm 6.7°, 100 mm 4.1°,
110 mm 2.1°, 120 mm 0.8°, 130 mm 0.17°, ≥ 140 mm < 0.05°. The dashed box on the MIP is the 50 % level (87.5 mm), not the end of excitation.
`bloch_parcel` returns `slab_profile {z, flip_deg}`; the MIP shades the band by flip angle and the readout gives the flip at the parcel centre.
Test `slab_profile_print` prints the profile (`cargo test -p hasty-seq slab_profile_print -- --nocapture`).
