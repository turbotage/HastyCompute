//! Time-resolved P²nP-ISTA reconstruction of the synthetic spiral 4D-flow data (frames × encodes), no density
//! compensation, whitened + coil-compressed data, LLR prior over all frames × encodes or the learned 7-point prior.
//! Any encoding of the simulator (referenced / Hadamard 4-point, 7-point dual-VENC): velocities by hasty_recon::flow.
//!   recon_p2np <sim_dir> [--model sense|seg] [--temporal off|linear|cubic] [--phys none|b0|full] [--gnl]
//!              [--nseg L|auto|auto:noise[:R]] [--order o] [--precond none,circ,zmshr1,zmshr1m,zmshr1s,zmshr1ms] [--iters n] [--kappa k,k,..] [--block b] [--llr-devices 0[,1,..]|cpu]
//!              [--vcoils auto|n] [--coil-energy 0.999] [--coil-gram calib|data] [--every n] [--warm n] [--eps e] [--conc-dc] [--synthetic] [--out dir]
//!              [--learned <export_dir> [--tile n] [--keep-frames n] [--learned-prec bf16|fp16|fp32] [--learned-cpu]]
//!              [--fresh] [--ckpt-every n] [--fat-fraction <npy|truth> [--fat-model shared|separate|joint]]
//!              [--conc-fit quad|deg4] [--table-prefetch]
//!              [--field truth|est:<recon_b0 run>[:bin]] [--r2s off|truth|est:<run>] [--rate <z.npy>] [--smaps truth|espirit[:eig_min]] [--coil-mask auto|none|support]
//!              [--coil-sampling keep:draw[:hold[:last[:seed]]]] [--lagged TAU[:LS|auto]] [--prior llr|wavelet] [--wavelet THR[:LEVELS]] [--wavelet-mix β] [--dfw sure[:TDF:TN]|fixed:TDF:TN [--dfw-levels L] [--dfw-mix β]] [--static-lock MU[:IT0[:EVERY]] [--lock-mode quad|group] [--lock-blur MM] [--lock-dilate MM] [--lock-score LO:HI] [--lock-vmin M/S] [--lock-detect OK:MAX]] [--air auto[:IT0[:EVERY]]|fixed [--air-init <npy>] [--air-margin MM] [--air-tau T] [--air-release R] [--air-blur MM[,MM..]]] [--state auto|device|host] [--resp-bins NR [--resp bellows|truth]] [--seg-basis fixed|cheb|svd|svdw|svdn] [--seg-weight count|mass[:lo,hi]|energy] [--b0-absorb 0|1|2] [--kstore auto|legacy|<store spec>]
//!              [--table-store auto|pinned|lz4|zstd|zstdN]   (env HASTY_KSTORE_DIR: k-space store directory, default <sim>/kstore)
//!              [--grid-scale s|sx,sy,sz] [--kcrop [FRAC][:prefix]] [--init-from <npy|run dir>] [--truth-sampling point|box] [--init-truth] [--vdecode auto|hier|ml[:MARGIN]] [--seg-fit-stride N] [--seg-table-stride S]
//! Coarse-to-fine pyramid (hasty_recon::pyramid, driver rustlib/tools/recon/pyramid.sh):
//!   --kcrop [FRAC][:prefix]: keep only the samples a (coarse, --grid-scale) grid represents, the ball
//!     Σ_a (k_a Δ_a)² ≤ (π·FRAC)² (default FRAC 1; ∩ the Nyquist box for FRAC > 1), applied to the sample records,
//!     the segment tables and the data (the k-space store is filtered on read, never rewritten; the kept data are
//!     held as a dense RAM copy when they fit, < 30 % of the available RAM; HASTY_KCROP_VIEW=1: always the filtered
//!     view). `prefix`: only each readout's outgoing part up to its first sample
//!     outside the ball (the kept ADC window ends early: fewer B0 / concomitant segments with --nseg auto).
//!   --init-from <path>: start from a previous level's images, resampled band-limited to this grid (Fourier
//!     coefficients kept: the same NUFFT data on the common band): recon_<tag>.npy (X, Y, Z, E, F) or warm.npy /
//!     warm_full.npy (X, Y, Z, E); a run directory picks its newest recon_*.npy, else warm_full.npy, else warm.npy.
//!     Frames: equal, 1 (broadcast) or the cardiac frames (broadcast over respiratory bins). Replaces the --warm
//!     PCGLS start (its per-encode time average is the warm start: support, warm.npy, --vmodel); --warm-full starts
//!     from it, else the solver starts from the per-frame images.
//!   --truth-sampling point|box: oracle maps (field, R2*, fat fraction, lumen fraction) at the reconstruction voxels'
//!     true positions (point, default) or averaged over the reconstruction voxel (box; differs only on grids coarser
//!     than the native one). Native-grid maps given to --fat-fraction are resampled to the reconstruction grid.
//!              [--warm-full n | --tpyramid RxC:n[,RxC:n..]] [--pooled-grad auto|grouped|replicate]
//! --rate <z.npy>: per-voxel complex rate z = R2* + iΔω (complex64 [X, Y, Z], x fastest; s⁻¹ and rad/s) on the native or
//!   the reconstruction grid: R2* folded into the segment functions (as --r2s) and Δω added to the field (--field): the
//!   voxel's signal after the echo e^{−z (τ − τ0)} on top of the map's e^{−iω τ} (research/readout_physics_budget.md T4:
//!   the signal-weighted effective frequency + R2* is the lever; research/recon_physics_wiring.md item 4). Excludes
//!   --r2s. With per-frame fat maps and --seg-basis svdn-union the union basis then carries the decay too
//!   (build_segments_svd_fat_union_g). Run directory suffix `_rate`.
//! --tpyramid: temporal pyramid of pooled PCG warm starts through the full model, coarse to fine (e.g.
//!   "1x1:4,2x1:3,2x5:3"): per level R respiratory-bin groups × C cardiac-phase groups, n iterations; every
//!   (encode, resp group, cardiac group) shares one image; the last level initialises every set of the solver.
//!   Pooled levels go through the grouped operator (`SetOp::forward_group`: one grid FFT per segment and coil batch per
//!   group); with per-frame joint fat maps a group uses the mean of its frames' maps (an approximation on pooled
//!   levels only). --pooled-grad replicate: the per-set path (every set pays its own FFTs).
//! --learned: the exported denoiser (tools/denoiser/export.py) through the libtorch plugin (hasty_recon::learned) in
//!   place of LLR, x ← x + κ (D(x) − x) per P²nP iteration; --tile / --keep-frames: exact overlap tiles in space /
//!   cyclic time windows (bounded GPU memory); bf16 autocast by default (the training regime).
//! Interrupt / resume: Ctrl-C, SIGTERM or a file STOP in the run directory checkpoints (every --ckpt-every iterations
//! too) and exits; rerunning the same command continues (warm start per encode, then each preconditioner / κ run).
//! --fresh discards the saved state.
//! Every run writes `dir` (default <repo>/data/hasty/runs/<sim>_<model>…, stable so that reruns resume): run.json, coil maps, masks, warm
//! start, final complex images and velocities per (preconditioner, κ), metrics CSV, and figures (`figs/`, made by
//! rustlib/tools/figs/run_figs.py at the end).
//! sense: plain NUFFT per (frame, encode) set; `--conc-dc` folds the per-encode mean concomitant phase into the coil maps.
//! seg:   segmented physics NUFFT (`hasty_nufft::SegNufft`): B0 + full concomitant history per sample (segments for the
//!        readout-time mean, per-sample k-shift / phase / ΔC kernel correction), coil maps and fields at the true (GNL)
//!        positions with `--gnl`; per (frame, encode) set, or per encode over all frames with temporal B-spline
//!        interpolation in cardiac time (`--temporal`).
//!        `--conc-fit deg4` (default quad; needs --phys full and order 2): the concomitant monomials at the GNL-warped
//!        true positions are fitted to degree 4 instead of 2; the quadratic part is used as before, the degree-3/4 part
//!        per sample to first order in the kernel (SegNufft correction order 3, no extra FFTs). Removes most of the
//!        full model's floor from the quadratic fit's residual (CPU emulation, flow_thorax, oblique off-centre, GNL,
//!        L = 20: 2.89 T 4.8e-3 / 8.3e-3 → 1.2e-3 / 2.2e-3 forward / adjoint, 1.5 T 8.9e-3 / 1.5e-2 → 3.0e-3 / 6.5e-3;
//!        hasty-bench tests/seg_physics_direct.rs). Useful with --gnl only; the run directory gets a `_deg4` suffix.
//! Memory: `--plan auto|device|host` and `--vram-budget` also place the operator's per-set sample tables (SegPoints,
//!   ≈ 200 B/sample): resident, or pinned host memory uploaded into two device slots per set use (≈ bytes / pinned
//!   bandwidth per gradient). They are built straight into host memory when they would not fit next to what is still
//!   to come (auto) or with --plan host; the velocity-model planner then decides (`plan::choose_tables`) and may
//!   bring them back. With --vmodel the k-space data are likewise kept on the host from the start when the
//!   resident problem would not fit (auto) or with --plan host. --table-prefetch uploads the next set's tables on a
//!   second stream during the current set (default: upload on the compute stream, ~2–3 % of the operator time).
//! K-space data (`--kstore`, hasty_recon kload / kstore / kstage):
//!   legacy: the whole sim in RAM, whitened and compressed there (the original loader; sequential noise generator);
//!   a store spec — ram | lz4 | zstd | zstdN, optional disk: or disk=PATH: prefix, optional +qREL (noise-relative
//!     quantisation, lossy by REL²/12 of the noise variance): the sim is streamed readout by readout from its files,
//!     whitened, coil-compressed and given counter-based noise on load, and kept in that representation; a disk store
//!     (default <sim>/kstore/<content hash>.kst) is reused by later runs with the same data (same compression, noise,
//!     sets, injected phase);
//!   auto (default): legacy when the sim fits comfortably in RAM (3 × raw < ½ available), else streamed into ram,
//!     zstd1 or disk:zstd1, whichever fits.
//!   Streamed data stay on the device when they fit (as before); otherwise they are staged per set from the store
//!   (double-buffered pinned upload, the next set decoded during the current one; image-space P²nP and velocity
//!   model).
//! --table-store: host form of the sample tables when they leave the device: pinned (as before), or packed blocks
//!   (lz4 / zstd = zstd1 / zstdN; trajectory tables compress ≈ ×6 with zstd1) decoded in parallel into a pinned
//!   staging buffer per slot on use; auto = pinned unless all tables pinned would take > 40 % of the available RAM.
//! --fat-fraction: water + fat with a known fat fraction. --fat-model (default shared):
//!   shared:   two species (FatSets), one shared-gridding call (SegNufft multi-image kernels): 2 images × L segments;
//!   separate: two species, two operator calls (FatSets::separate, the first implementation);
//!   joint:    one image whose segments carry each voxel's water / fat mix (--model seg): L FFT channels instead of 2L.
//!             A single map is folded into the segment diagonals (build_segments_complex, plain SegSets); per-frame
//!             maps keep one ff-independent table per encode (build_segments_fat) and the maps separate
//!             (FatJointSets, SegNufft::forward_fat). --nseg auto then counts the phase cycles of the water and the
//!             main fat peak of every fitted voxel (same rule). CPU budget (hasty-nufft tests/seg_b0_direct.rs
//!             fat_species_error_budget_cpu, 8 ms, mixed voxels, 2.89 T): joint L = 13 (the rule) 2.5e-4 with 13
//!             channels vs two species L = 8 6.5e-4 with 16; 1.5 T: rule L = 11 ≈ 2e-4. The run directory suffix is
//!             `_fat` (shared), `_fatjoint` or `_fatseparate`.
//! Reports data fidelity and the per-frame velocity RMSE in the lumen (velocities J⁻¹-corrected and compared at the
//! true positions with `--gnl`).
//! Also reads schedule-based sims (`simphantom --seq flow`, the yarn TR; hasty-recon `SimData`): the trajectory is
//! the stored nominal one, or with `HASTY_TRAJ=played` the one played through the simulated GIRF (incl. its B0-term
//! phase), i.e. a reconstruction that models the chain exactly. `--conc-dc` is spiral-only; velocities come from the
//! hierarchical decoder (coprime 7: unaliased to ±VENC_hi, not the ±(p+1)·VENC_lo of the CRT in hasty-seq `venc`).


#![recursion_limit = "256"]
use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::CudaSlice;
use hasty_cuda::{Device, C32};
use hasty_mrsim::{mat_vec, transpose};
use hasty_nufft::{build_segments, build_segments_complex, build_segments_fat, build_segments_svd_weighted, SegNufft, SegNufftOpts};
use hasty_recon::checkpoint::{value_dev, Outcome, RunCtl};
use hasty_recon::data::{inv3, SampleSet, SimData};
use hasty_recon::kload::LoadOpts;
use hasty_recon::kstore::{KStore, StoreSpec as KStoreSpec};
use hasty_recon::flow::FlowDecoder;
use hasty_recon::gpu::Kernels;
use hasty_recon::llr::Llr;
use hasty_recon::op::SenseOp;
use hasty_recon::p2np::{P2np, Precond, Stack};
use hasty_recon::precond::CirculantPrecond;
use hasty_recon::segphys::{fit_monomials, fit_monomials_hi, hi_table, mean_coefficients, seg_tables, seg_tables_hi, segment_phase};
use hasty_recon::setop::{SegSets, SenseSets, SetOp};
use num_complex::Complex64;

/// bytes of RAM this process can still use (MemAvailable capped by the enclosing memory cgroups, v1 and v2)
fn available_memory() -> usize {
    hasty_site::available_memory()
}

/// FNV-1a 64-bit (content tags)
fn fnv1a(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x0100_0000_01b3))
}

fn apply_cc(x: &[C32], len: usize, nc: usize, t: &[Vec<Complex64>]) -> Vec<C32> {
    let mut out = vec![C32::new(0.0, 0.0); t.len() * len];
    for (v, row) in t.iter().enumerate() {
        for i in 0..len {
            let mut acc = Complex64::new(0.0, 0.0);
            for c in 0..nc {
                let s = x[c * len + i];
                acc += row[c] * Complex64::new(s.re as f64, s.im as f64);
            }
            out[v * len + i] = C32::new(acc.re as f32, acc.im as f32);
        }
    }
    out
}

/// .npy (Fortran order) from byte parts written in sequence, without assembling them in memory
fn save_parts<'a>(path: &std::path::Path, descr: &str, shape: &[usize], parts: impl Iterator<Item = &'a [u8]>) {
    use std::io::Write;
    let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(path).unwrap());
    w.write_all(&hasty_phantom::volume::npy_header(descr, shape, true)).unwrap();
    for p in parts {
        w.write_all(p).unwrap();
    }
    w.flush().unwrap();
}
fn c32_bytes(v: &[C32]) -> &[u8] {
    // SAFETY: C32 is two f32, no padding
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) }
}
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: plain f32 data
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn save(path: &std::path::Path, descr: &str, shape: &[usize], bytes: Vec<u8>) {
    hasty_phantom::volume::write_npy(path, descr, shape, true, &bytes).unwrap();
}
fn c8(v: &[C32]) -> Vec<u8> {
    v.iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat()).collect()
}
fn f4(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| args.iter().position(|a| a == name).map(|i| { args.remove(i); true }).unwrap_or(false);
    let (synthetic, conc_dc, gnl, fresh) = (flag("--synthetic"), flag("--conc-dc"), flag("--gnl"), flag("--fresh"));
    // --init-truth (diagnostic): every frame starts from the warm image with the true velocity phases in the lumen (the
    // --synthetic construction, sampled on any grid): a solution that drifts away from it has a biased fixed point
    // (model or estimator); one that stays shows that the usual start / convergence is the problem
    let init_truth = flag("--init-truth");
    let learned_cpu = flag("--learned-cpu");
    let table_prefetch = flag("--table-prefetch");
    let bin_group = flag("--bin-group");
    // --bin-group-mid: the group's frame from its middle readout (before 2026-11-13: lags by 0.44 frame with the
    // referenced 7-point scheme, encode 0 carries a first moment); default the readout nearest the velocity time
    let bin_group_mid = flag("--bin-group-mid");
    // --save-smaps: the full coil-map stack in the output directory (1 GB at 2 mm); default: one axial slice for the figures
    let save_smaps = flag("--save-smaps");
    // --no-full-save: no final full recon / velocity stacks (8.9 + 1.9 GB at 2 mm), e.g. with --save-lumen diagnostics
    let no_full_save = flag("--no-full-save");
    // --llr-hard: hard singular-value threshold of the LLR prior (keep above, drop below; Llr::hard), run tag suffix _h
    let llr_hard = flag("--llr-hard");
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    // --vdecode auto|hier|ml[:MARGIN]: velocity decoder. hier = hierarchical unwrapping (range of the least sensitive
    // encodes); ml = + search over the first stage's aliases inside the design's unaliased range
    // (meta sequence.encoding.unaliased; hasty_recon::flow::AliasSearch), an alias accepted only with a phase-cost
    // advantage > MARGIN (default 1.0); auto = ml:1.0 when the design declares a range beyond the first stage's
    // (coprime / dual-VENC), else hier. The log also prints the hierarchical decoder's RMSE when ml is active.
    let vdecode: String = take("--vdecode").unwrap_or("auto".into());
    // --seg-fit-stride N (default 3): every N-th fit voxel enters the segment-basis fit; --seg-table-stride S (default 1):
    // one interpolator table entry every S ADC samples (≥ 256 entries). Every 9th voxel and S = 4 gave the same operator
    // error as the defaults at ~10× less setup CPU on ph0001 v2 (research/recon_physics_wiring.md).
    let seg_fit_stride: usize = take("--seg-fit-stride").map(|s| s.parse().expect("--seg-fit-stride N")).unwrap_or(3).max(1);
    let seg_tab_stride: usize = take("--seg-table-stride").map(|s| s.parse().expect("--seg-table-stride S")).unwrap_or(1).max(1);
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(40);
    let kappas: Vec<f64> = take("--kappa").unwrap_or("1.0".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let block: usize = take("--block").map(|s| s.parse().unwrap()).unwrap_or(8);
    // virtual coils: a number, or auto (default) = the fewest holding --coil-energy (default 99.9 %) of the whitened data
    // energy (`hasty_recon::coils`)
    let coil_energy: f64 = take("--coil-energy").map(|s| s.parse().unwrap()).unwrap_or(hasty_recon::coils::CoilRule::DEFAULT_ENERGY);
    let coil_rule = hasty_recon::coils::CoilRule::parse(&take("--vcoils").unwrap_or("auto".into()), coil_energy).unwrap();
    // --coil-gram calib (default): the Gram of the Cartesian calibration region (uniform k-space grid: density-neutral,
    // ≈ Σ_r ρ ρᴴ of the low-resolution coil images); data: the raw samples of set 0 (the previous default; centre-dense
    // trajectories weight it towards k = 0, i.e. towards one vector Σ_r S(r) ρ(r), so it keeps too few coils)
    let coil_gram = take("--coil-gram").unwrap_or("calib".into());
    assert!(coil_gram == "calib" || coil_gram == "data", "--coil-gram calib|data");
    let warm_iters: usize = take("--warm").map(|s| s.parse().unwrap()).unwrap_or(0);
    // --warm-full n: after the plain-model warm start (which ignores B0, fat, R2*, concomitant fields and GNL: the
    // SENSE operator), n more pooled PCG iterations per encode through the full model operator (all frames of an encode
    // share one image; one gradient pass over all sets per iteration; the per-encode circulant as preconditioner when
    // the run builds circulants). The solver starts from that. 0 (default): the plain warm start only.
    let warm_full: usize = take("--warm-full").map(|s| s.parse().unwrap()).unwrap_or(0);
    // --tpyramid SPEC: temporal pyramid of pooled PCG warm starts through the full model operator, coarse to fine:
    // comma-separated levels RxC:N = R respiratory-bin groups × C cardiac-phase groups (contiguous bins / phases, sizes
    // may differ by one; R ≤ --resp-bins, C ≤ cardiac frames), N pooled PCG iterations. Each (encode, resp group,
    // cardiac group) shares one image; level 1 starts from the plain warm start (--warm) or zero, each later level from
    // the previous one (a group from the mean of its sets' previous group images: a copy where the levels nest); the
    // last level initialises every set of the solver. --warm-full n is the level "1x1:n" (kept as its own path).
    let tpyramid: Option<Vec<(usize, usize, usize)>> = take("--tpyramid").map(|s| parse_tpyramid(&s));
    assert!(tpyramid.is_none() || warm_full == 0, "--tpyramid and --warm-full: give the pooled level as \"1x1:n\" in --tpyramid");
    // --pooled-grad auto|grouped|replicate: group gradients of pooled PCG (--warm-full, --tpyramid): auto / grouped =
    // the grouped operator (one grid FFT per segment, coil batch and group), replicate = every set separately
    let pooled_grad = take("--pooled-grad").map_or(hasty_recon::hoststack::PooledGrad::Auto, |s| hasty_recon::hoststack::PooledGrad::parse(&s).expect("--pooled-grad auto|grouped|replicate"));
    let every: usize = take("--every").map(|s| s.parse().unwrap()).unwrap_or(5);
    let model = take("--model").unwrap_or("seg".into());
    let temporal: Option<u32> = match take("--temporal").as_deref() {
        None | Some("off") => None,
        Some("linear") => Some(1),
        Some("cubic") => Some(3),
        Some(o) => panic!("--temporal {o}"),
    };
    let phys = take("--phys").unwrap_or("full".into());
    // time segments: --nseg L, or auto (default): from the phase cycles the field spans over the readout,
    // L = ⌈1.3 · Δf·T + 3.5⌉ (≈ 1e-3 operator error with least-squares interpolators: CPU budgets of
    // tests/seg_b0_direct.rs — 450 Hz over 8 ms → 8, fat included 843 Hz → 12; a fixed 10 was too few with fat)
    // --nseg auto:noise[:R] (research/segment_basis.md): the smallest L whose model error, predicted from the weighted
    // separable error (energy voxel weights, per-ADC-sample weights 1 + (E(n) + E_vol) / (C σ²)), is at most R noise rms
    // in the data (default R = 0.2); the plain `auto` rule above stays the default
    let nseg_spec: Option<String> = take("--nseg");
    let seg_noise_rule: Option<f64> = nseg_spec.as_deref().and_then(|s| s.strip_prefix("auto:noise")).map(|r| r.strip_prefix(':').map_or(0.2, |v| v.parse().expect("--nseg auto:noise[:R]")));
    let nseg_arg: Option<usize> = nseg_spec.and_then(|s| if s == "auto" || s.starts_with("auto:") { None } else { Some(s.parse().unwrap()) });
    let mut nseg: usize = nseg_arg.unwrap_or(if phys == "none" { 1 } else { 0 });
    let order: u32 = take("--order").map(|s| s.parse().unwrap()).unwrap_or(if phys == "full" { 2 } else { 0 });
    // relative floor of the circulant preconditioner's Fourier diagonal (larger = weaker, better-posed metric)
    let eps: f64 = take("--eps").map(|s| s.parse().unwrap()).unwrap_or(1e-3);
    let preconds: Vec<String> = take("--precond").unwrap_or("zmshr1".into()).split(',').map(|s| s.to_string()).collect();
    // --circ-share set|encode: circulant metrics per set, or one per encode shared by its frames (default: encode)
    let circ_share_arg: Option<String> = take("--circ-share");
    let out_arg = take("--out");
    // periodic checkpoints (tens of GB per write at 2 mm) only on request; a STOP / Ctrl-C checkpoint is always written
    let ckpt_every: usize = take("--ckpt-every").map(|s| s.parse().unwrap()).unwrap_or(0);
    // --save-lumen idx.npy: at every report iteration, the reconstruction at these flat (Fortran) native-grid voxels
    // only, it<k>_<tag>_lumen.npy = complex64 (n, ne, nf) C-order (research/experiments/recon_diag/frame_fit.py's
    // *_lumen.npy form): per-frame scoring along the iterations without full stacks on disk
    let save_lumen: Option<Vec<usize>> = take("--save-lumen").map(|p| hasty_recon::data::read_npy(std::path::Path::new(&p)).expect("--save-lumen idx.npy").f32s().iter().map(|&v| v as usize).collect());
    // velocity-parametrised model (research/velocity_model_sr.md): --vmodel KZ,KV [--vdeg DZ,DV] [--vpoly d] [--vcont n]
    let vmodel: Option<(usize, usize)> = take("--vmodel").map(|s| { let v: Vec<usize> = s.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) });
    let vdeg: (u32, u32) = take("--vdeg").map(|s| { let v: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).unwrap_or((1, 1));
    let vpoly: Option<usize> = take("--vpoly").map(|s| s.parse().unwrap());
    let vcont: usize = take("--vcont").map(|s| s.parse().unwrap()).unwrap_or(0);
    // z warm-up: first n iterations update z only (v and the residual phase frozen): makes z consistent with the
    // operator (B0, coils, physics) before velocity moves, so encode-common phase errors cannot leak into v
    let vzwarm: usize = take("--vzwarm").map(|s| s.parse().unwrap()).unwrap_or(0);
    // velocity-model prior strengths (LLR on the z nodes and on p = |z̄|·v; 0 = no prior), one run each
    let vkappas: Vec<f64> = take("--vkappa").unwrap_or("0".into()).split(',').map(|s| s.parse().unwrap()).collect();
    // memory planner (velocity path): device-usage cap (e.g. 8G; default: the whole card) and placement mode
    let vram_budget: Option<usize> = take("--vram-budget").map(|s| hasty_recon::plan::parse_bytes(&s).expect("--vram-budget e.g. 8G"));
    let plan_mode = take("--plan").unwrap_or("auto".into());
    let kstore_arg = take("--kstore").unwrap_or("auto".into());
    // host form of the operator's sample tables when they leave the device: pinned (as before) or packed blocks
    let table_store = take("--table-store").unwrap_or("auto".into());
    let table_codec: Option<Option<hasty_value::Options>> = match table_store.as_str() {
        "auto" => None,
        "pinned" => Some(None),
        "lz4" => Some(Some(hasty_value::Options::default())),
        "zstd" => Some(Some(hasty_value::Options::zstd(1))),
        z if z.starts_with("zstd") => Some(Some(hasty_value::Options::zstd(z[4..].parse().expect("--table-store zstdN")))),
        o => panic!("--table-store {o}: auto | pinned | lz4 | zstd | zstdN"),
    };
    // reconstruction grid relative to the native one: s or sx,sy,sz (research/velocity_model_sr.md §B2)
    // voxel basis function: dirac (default) | rect | bspline3, a per-sample weight Π sinc(k Δ / 2)^p
    // --fat-fraction <npy | truth>: water + fat with a known signal fat fraction per voxel (hasty_recon::setop::FatSets:
    // the fat part carries the six-peak spectrum per sample; the B0 segments then only cover the B0 range); `truth`:
    // truth_fat_fraction.npy of the sim directory (sim_truth.py), else a float32 npy on the reconstruction grid
    // --coil-mask auto|none|support (see the coil-map block): auto = none for --smaps truth, support otherwise
    let coil_mask_arg: Option<String> = take("--coil-mask").filter(|v| v != "auto");
    let fat_fraction: Option<String> = take("--fat-fraction");
    // measured instead of true fields / coil maps (the oracle-vs-estimate evaluation; defaults = the oracle as before):
    // --field truth|est:<recon_b0 run>[:bin]: off-resonance ω; --r2s off|truth|est:<run>[:bin]: T2* decay during the
    // readout folded into the B0 segments; --smaps truth|espirit[:eig_min]: ESPIRiT maps from the compressed data of
    // encode 0 (hasty_recon::espirit); --fat-fraction est:<run>[:bin] takes the fat fraction of the B0 scan
    let field_spec = take("--field").unwrap_or("truth".into());
    let r2s_spec = take("--r2s").unwrap_or("off".into());
    let rate_spec: Option<String> = take("--rate");
    assert!(rate_spec.is_none() || r2s_spec == "off", "--rate carries R2*: use it without --r2s");
    let smaps_spec = take("--smaps").unwrap_or("truth".into());
    // --coil-sampling keep:draw[:hold[:last[:seed]]]: stochastic coil sampling of the data term (hasty_recon::p2np::
    // CoilSampling, random reshuffling; defaults hold 2, last 12, seed 1); off by default
    // --lagged TAU[:LS|auto]: lagged (control-variate) gradients with a reduced-segment surrogate on the same sample
    // tables (P2np::with_lagged, SegSets::with_surrogate): exact gradients only when ‖x − x_a‖ > TAU ‖x‖. LS = the
    // surrogate's segment count; auto (default): the smallest with ‖(N − Ñ)x‖ ≤ 3 % ‖N x‖ on the warm start / a
    // probe (pays only for small mismatch: tests/lagged.rs).
    // --seg-basis fixed|svd: time-segment diagonals at L fixed readout times (default) or the optimal rank-L basis of
    // the voxel × readout-time function (hasty_nufft::build_segments_svd: complex diagonals, same operator cost;
    // ≈ 0.6× the separable-model error at equal L on the sim_point field, about half a segment; tests/seg_basis_cpu.rs).
    // Joint water + fat with per-frame maps keeps the fixed-time diagonals.
    // svdw: the SVD basis weighted per readout time by the measured data energy at that ADC time (centre-out readouts
    // carry most energy early, where the accumulated phase is small): one segment fewer at the same data-weighted
    // error on the factory ph0001 field (tests/seg_basis_cpu.rs)
    // --resp-bins NR [--resp bellows|truth]: respiratory × cardiac frames. Readouts are binned by breathing amplitude
    // (equal counts, bin 0 = the lowest amplitude) and every (bin, cardiac frame) is a frame of its own: frame index
    // r · F + f. The LLR prior spans all of them; the learned prior runs per bin (cyclic cardiac windows). Amplitude:
    // the breathing belt as the scanner records it (physio.npy, default) or the simulator's true state (resp.npy).
    // --state auto|device|host: where the solver's image stacks (iterate, gradient, step, secant pair) live. host: in host
    // memory, the gradient streams set by set through two device slots (hasty_recon::hoststack) — the device never
    // holds a stack; auto: host when three stacks would not fit in half the free device memory
    let state_arg = take("--state").unwrap_or("auto".into());
    // --prior llr|wavelet (complex-image prior without --learned; default llr); --wavelet THR[:LEVELS] (default 1:3):
    // 3D Daubechies-4 shrinkage per image, threshold THR × the MAD noise level of each image (scale-equivariant), cycle
    // spinning; --wavelet-mix β: with --learned, x ← x + κ ((1 − β) D(x) + β W(x) − x) (research/wavelet_prior.md)
    let prior_kind = take("--prior").unwrap_or("llr".into());
    // --llr-devices 0[,1,..] | cpu: the CUDA devices sharing the LLR prior's blocks (default: the solver's device 0),
    // or the CPU operator (same result to float rounding: tests/llr_gpu.rs)
    let llr_devices = take("--llr-devices").unwrap_or_else(|| "0".into());
    // --llr-rank R: rank cap of the LLR prior, each block's threshold raised to its (R+1)-th singular value (Llr::rank;
    // the κσ̂ rule alone keeps ~200 of 280 components); run tag suffix _r{R}
    let llr_rank: Option<usize> = take("--llr-rank").map(|v| v.parse().expect("--llr-rank R"));
    assert!(matches!(prior_kind.as_str(), "llr" | "wavelet" | "tlp"), "--prior llr|wavelet|tlp");
    // --prior tlp [--tlp-k K]: cardiac low-pass, every (respiratory bin, encode, voxel) series over the cardiac frames
    // projected onto the harmonics |k| ≤ K (default 4), x ← x + min(κ, 1) (P x − x) (hasty_recon::tlp; run tag _tlpK)
    let tlp_k: usize = take("--tlp-k").map_or(4, |v| v.parse().expect("--tlp-k K"));
    // --prior-mode pnp|red (default pnp): red puts the denoiser residual into the gradient, g = ∇f + λ (x − D(x)), and
    // drops the denoiser after the step (hasty_recon::red; research/red_prior.md). --red-rho ρ (default 1): λ = ρ / (α μ),
    // μ the metric's mean gain (1 without a metric), so one step moves x by about τρ (D(x) − x) (ρ = 1: P²nP's full
    // denoising step). --red-shifts n (LLR only, default 1): the LLR shift is fixed in RED mode (a random shift per
    // iteration makes the gradient stochastic and breaks the ZMSHR1 secant pairs); n > 1 averages D over n fixed shifts
    // (n LLR calls per iteration, one extra stack). Run tag suffix _red{ρ}[_sh{n}].
    let prior_mode = take("--prior-mode").unwrap_or("pnp".into());
    assert!(matches!(prior_mode.as_str(), "pnp" | "red"), "--prior-mode pnp|red");
    let red = prior_mode == "red";
    let red_rho: f64 = take("--red-rho").map_or(1.0, |v| v.parse().expect("--red-rho ρ"));
    let red_shifts: usize = take("--red-shifts").map_or(1, |v| v.parse().expect("--red-shifts n"));
    assert!(red_shifts >= 1, "--red-shifts n >= 1");
    // --tsub fourier:K | bspline:M[:P]: temporal subspace, x = T c with T the same per voxel and encode (hasty_recon::tsub):
    // the cyclic cardiac Fourier basis |k| ≤ K (orthonormal) or M periodic B-splines of degree P (default 3), each per
    // respiratory bin; the solver iterates on the coefficients (gradient Tᴴ ∇f(T c), the data term unchanged). Host
    // state only. Run tag suffix _ts{basis}.
    let tsub_spec: Option<String> = take("--tsub");
    let red_tag = if red { format!("_red{red_rho}{}", if red_shifts > 1 { format!("_sh{red_shifts}") } else { String::new() }) } else { String::new() };
    if red && prior_kind == "llr" && llr_rank.is_none() {
        eprintln!("warning: --prior-mode red with the κσ̂ LLR rule: σ̂ is re-estimated from every iterate, so the RED objective changes with x (use --llr-rank R [--llr-hard] for a fixed objective)");
    }
    let (wav_thr, wav_levels): (f64, usize) = take("--wavelet").map_or((1.0, 3), |v| {
        let mut it = v.split(':');
        (it.next().unwrap().parse().expect("--wavelet THR[:LEVELS]"), it.next().map_or(3, |l| l.parse().expect("--wavelet THR[:LEVELS]")))
    });
    let wav_mix: f64 = take("--wavelet-mix").map_or(0.0, |v| v.parse().expect("--wavelet-mix β"));
    // --dfw sure[:TDF:TN] | fixed:TDF:TN [--dfw-levels L] [--dfw-mix β]: divergence-free wavelet shrinkage of the
    // velocity after the prior (hasty_recon::dfwavelet, research/static_lock.md): complex model — velocity decoded per
    // frame, shrunk with the magnitude-weighted SURE / fixed thresholds (TDF for the divergence-free, TN for the
    // non-divergence-free coefficients), re-encoded as a phase change of each encode; velocity model — the v nodes.
    // β mixes x ← x + β (DFW(x) − x) (default 1).
    let dfw_rule: Option<hasty_recon::dfwavelet::DfRule> = take("--dfw").map(|v| {
        use hasty_recon::dfwavelet::DfRule;
        let it: Vec<&str> = v.split(':').collect();
        let num = |i: usize, d: f64| it.get(i).map_or(d, |x| x.parse().expect("--dfw sure[:TDF:TN] | fixed:TDF:TN"));
        match it[0] {
            "sure" => DfRule::Sure { tdf: num(1, 1.0), tn: num(2, 1.0) },
            "fixed" => DfRule::Fixed { tdf: num(1, 1.0), tn: num(2, 2.0) },
            _ => panic!("--dfw sure[:TDF:TN] | fixed:TDF:TN"),
        }
    });
    let dfw_levels: usize = take("--dfw-levels").map_or(3, |v| v.parse().expect("--dfw-levels L"));
    let dfw_mix: f64 = take("--dfw-mix").map_or(1.0, |v| v.parse().expect("--dfw-mix β"));
    let dfw_tag = dfw_rule.map_or(String::new(), |r| match r {
        hasty_recon::dfwavelet::DfRule::Sure { tdf, tn } => format!("_dfws{tdf}-{tn}"),
        hasty_recon::dfwavelet::DfRule::Fixed { tdf, tn } => format!("_dfwf{tdf}-{tn}"),
    }) + if dfw_rule.is_some() && dfw_mix != 1.0 { format!("m{dfw_mix}") } else { String::new() }.as_str();
    // --static-lock MU[:IT0[:EVERY]] [--lock-mode quad|group] [--lock-blur MM] [--lock-dilate MM] [--lock-score LO:HI]
    // [--lock-vmin M/S] [--lock-detect OK:MAX]: static / dynamic tissue map from the iterate (hasty_recon::tlock; no labels) from iteration
    // IT0 (default 5), refreshed every EVERY iterations (default 5), and the temporal lock of the static voxels after
    // the prior: quadratic prox weight MU (default mode), or group soft threshold MU × the local noise level.
    // Complex-image models only (the map and the lock act on the frames × encodes stack).
    // --mag-couple GAMMA[:DELTA]: after the prior, pull the magnitudes of the encodes of every frame towards their mean
    // (phases kept): |x_e| ← m̄ + (|x_e| − m̄)/(1 + GAMMA w_e), w_e = |x_e|² / (|x_e|² + (DELTA · m_med)²) with m_med the
    // median of m̄ over the coil-map support (dark encodes are not pulled up), gain clipped to [0.5, 1.5]. The prox of
    // (GAMMA/2) Σ_e w_e (|x_e| − m̄)² for fixed weights. Weak coupling only (research/magnitude_coupling.md: the map is
    // expansive in the phase direction of encodes below the mean; GAMMA ≈ 0.1 helped in the toy, ≥ 1 stalled).
    let mag_couple: Option<(f32, f32)> = take("--mag-couple").map(|v| {
        let (g, d) = v.split_once(':').map_or((v.as_str(), "0.1"), |(a, b)| (a, b));
        (g.parse().expect("--mag-couple GAMMA[:DELTA]"), d.parse().expect("--mag-couple GAMMA[:DELTA]"))
    });
    // --bin-group: every group of n_enc consecutive readouts (one rotation, all encodes) goes into the frame of its middle
    // readout, so all encodes of a frame share the same rotations (the same undersampling and aliasing); costs a
    // temporal footprint of one group (n_enc TRs) on top of the frame width
    let static_lock: Option<(hasty_recon::tlock::LockOpts, usize, usize)> = take("--static-lock").map(|v| {
        let it: Vec<&str> = v.split(':').collect();
        let mut o = hasty_recon::tlock::LockOpts { mu: it[0].parse().expect("--static-lock MU[:IT0[:EVERY]]"), ..Default::default() };
        let it0 = it.get(1).map_or(5, |x| x.parse().expect("--static-lock MU[:IT0[:EVERY]]"));
        let every = it.get(2).map_or(5, |x| x.parse::<usize>().expect("--static-lock MU[:IT0[:EVERY]]")).max(1);
        if let Some(m) = take("--lock-mode") {
            o.group = match m.as_str() { "group" => true, "quad" => false, _ => panic!("--lock-mode quad|group") };
        }
        if let Some(b) = take("--lock-blur") { o.blur_mm = b.parse().expect("--lock-blur MM"); }
        if let Some(b) = take("--lock-dilate") { o.dilate_mm = b.parse().expect("--lock-dilate MM"); }
        if let Some(b) = take("--lock-score") {
            let (lo, hi) = b.split_once(':').expect("--lock-score LO:HI");
            (o.s_lo, o.s_hi) = (lo.parse().unwrap(), hi.parse().unwrap());
        }
        if let Some(b) = take("--lock-vmin") { o.v_min = b.parse().expect("--lock-vmin M/S"); }
        if let Some(b) = take("--lock-detect") {
            let (lo, hi) = b.split_once(':').expect("--lock-detect OK:MAX");
            (o.detect_ok, o.detect_max) = (lo.parse().unwrap(), hi.parse().unwrap());
        }
        (o, it0, every)
    });
    let lock_tag = static_lock.as_ref().map_or(String::new(), |(o, it0, ev)| format!("_lock{}{}i{it0}e{ev}", if o.group { "g" } else { "q" }, o.mu));
    // --air auto[:IT0[:EVERY]] | fixed [--air-init <npy>] [--air-margin MM] [--air-tau T] [--air-release R] [--air-blur MM[,MM..]]:
    // air map from the iterate (hasty_recon::airmap; no labels): voxels reachable from the FOV boundary through
    // low-signal voxels (reference magnitude < T × the body level, default 0.05), with a safety margin of MM (default
    // 4 mm) to anything else, estimated from iteration IT0 (default 0) and refreshed every EVERY (default 2)
    // iterations; monotone (a zeroed voxel is released, and then protected, only where the iterate puts signal above
    // R × the level back, default 0.15); x = 0 there in every frame and encode, before and after the prior.
    // --air-init: a seed on any grid of the same FOV (an air_map_*.npy, an air mask, or a coarse level's images, see
    // airmap::load_seed), applied from the first iteration; `fixed`: the seed only, never re-estimated.
    // Complex-image models only.
    let air: Option<(hasty_recon::airmap::AirOpts, usize, usize)> = take("--air").map(|v| {
        let it: Vec<&str> = v.split(':').collect();
        let mut o = hasty_recon::airmap::AirOpts::default();
        let (it0, every) = match it[0] {
            "auto" => (it.get(1).map_or(0, |x| x.parse().expect("--air auto[:IT0[:EVERY]]")), it.get(2).map_or(2, |x| x.parse::<usize>().expect("--air auto[:IT0[:EVERY]]")).max(1)),
            "fixed" => (usize::MAX, usize::MAX),
            _ => panic!("--air auto[:IT0[:EVERY]] | fixed"),
        };
        if let Some(b) = take("--air-margin") { o.margin_mm = b.parse().expect("--air-margin MM"); }
        if let Some(b) = take("--air-tau") { o.tau = b.parse().expect("--air-tau T"); }
        if let Some(b) = take("--air-release") { o.release = b.parse().expect("--air-release R"); }
        if let Some(b) = take("--air-blur") { o.blur_mm = b.split(',').map(|x| x.parse().expect("--air-blur MM[,MM..]")).collect(); }
        (o, it0, every)
    });
    let air_init: Option<PathBuf> = take("--air-init").map(PathBuf::from);
    assert!(air_init.is_none() || air.is_some(), "--air-init needs --air auto[:IT0[:EVERY]] | fixed");
    assert!(air.as_ref().map_or(true, |a| a.1 != usize::MAX || air_init.is_some()), "--air fixed needs --air-init");
    let air_tag = air.as_ref().map_or(String::new(), |(o, it0, ev)| {
        let s = if air_init.is_some() { "s" } else { "" };
        if *it0 == usize::MAX { format!("_airfix{s}m{}", o.margin_mm) } else { format!("_air{}m{}i{it0}e{ev}{s}", o.tau, o.margin_mm) }
    });
    let resp_bins: usize = take("--resp-bins").map_or(1, |v| v.parse().expect("--resp-bins N"));
    let resp_src = take("--resp").unwrap_or("bellows".into());
    // --resp-accept FRAC: respiratory gating, as navigator / bellows gating in practice: keep the readouts whose
    // breathing amplitude (--resp source) is in the lowest FRAC of the scan (end-expiration, the quietest state), drop the
    // rest (their frame becomes usize::MAX: in no set). --resp-bins then bins the accepted readouts. Encode groups stay
    // whole with --bin-group (a group is kept or dropped by its frame readout). Run tag suffix _ga{FRAC}.
    let resp_accept: f64 = take("--resp-accept").map_or(1.0, |v| v.parse().expect("--resp-accept FRAC"));
    assert!(resp_accept > 0.0 && resp_accept <= 1.0, "--resp-accept FRAC in (0, 1]");
    let ga_tag = if resp_accept < 1.0 { format!("_ga{resp_accept}") } else { String::new() };
    let seg_basis_arg = take("--seg-basis").unwrap_or("fixed".into());
    // --seg-weight count|mass[:lo,hi]: voxel weights of the SVD basis, all data-derived (no labels): count = every voxel
    // of the warm-start support equally (default); mass = the smoothed warm-start magnitude relative to its median over
    // the support, clamped to [lo, hi] (default 0.3, 3): bright regions count more, dark tissue keeps a floor so the
    // basis is not biased away from it, air (outside the support) never counts
    let seg_weight = take("--seg-weight").unwrap_or("count".into());
    // energy (research/segment_basis.md): the smoothed warm-start magnitude squared × the coil energy Σ_c |S_c|² relative
    // to its median over the support: the signal energy each voxel feeds into the data (no clamp)
    let seg_energy = seg_weight == "energy";
    let seg_mass: Option<(f32, f32)> = match seg_weight.split(':').next().unwrap() {
        "count" | "energy" => None,
        "mass" => Some(seg_weight.split(':').nth(1).map_or((0.3, 3.0), |b| {
            let v: Vec<f32> = b.split(',').map(|x| x.parse().expect("--seg-weight mass:lo,hi")).collect();
            (v[0], v[1])
        })),
        o => panic!("--seg-weight count|mass[:lo,hi]|energy, got {o}"),
    };
    // svdn (research/segment_basis.md): the rank-L basis weighted per ADC sample by 1 + (E(n) + E_vol) / (C σ²) (whitened
    // data energy per sample plus its k-space volume average, over the noise energy per sample; hasty_nufft::
    // noise_time_weights) on a time grid that resolves the k-space-centre samples
    // (hasty_nufft::seg_time_grid, build_segments_svd_grid), voxel weights of --seg-weight; cheb: fixed-time
    // diagonals at Chebyshev nodes (clustered at the readout ends) with --seg-weight-weighted interpolators
    // svdn-union (research/readout_physics_budget.md §3.3): svdn, and with per-frame fat maps the union water + fat
    // noise-weighted SVD in cardinal form (hasty_nufft::build_segments_svd_fat_union) instead of the uniform-node fallback
    let seg_union = seg_basis_arg == "svdn-union";
    let seg_svdn = seg_basis_arg == "svdn" || seg_union;
    let seg_cheb = seg_basis_arg == "cheb";
    let (seg_svd, seg_svdw) = match seg_basis_arg.as_str() {
        "fixed" | "cheb" => (false, false),
        "svd" | "svdn" | "svdn-union" => (true, false),
        "svdw" => (true, true),
        o => panic!("--seg-basis fixed|cheb|svd|svdw|svdn|svdn-union, got {o}"),
    };
    // --b0-absorb D (0 default): the degree-D polynomial part of the field (fit over the support, weighted by the
    // expected signal energy) moves from the segments to the per-sample tables (hasty_recon::segphys::FieldPoly):
    // degree 1 = constant phase + k-space shift per sample (exact, free); degree 2 adds the quadratic part relative to
    // the readout centre through the ΔC kernel correction (Taylor: only for a few Hz of curvature; reported)
    let b0_absorb: u32 = take("--b0-absorb").map_or(0, |v| v.parse().expect("--b0-absorb 0|1|2"));
    let lagged: Option<(f64, Option<usize>)> = take("--lagged").map(|v| {
        let mut it = v.split(':');
        let tau: f64 = it.next().unwrap().parse().expect("--lagged TAU[:LS|auto]");
        let ls = it.next().filter(|s| *s != "auto").map(|s| s.parse::<usize>().expect("--lagged TAU[:LS|auto]"));
        (tau, ls)
    });
    let coil_sampling: Option<hasty_recon::p2np::CoilSampling> = take("--coil-sampling").map(|v| {
        let f: Vec<usize> = v.split(':').map(|x| x.parse().expect("--coil-sampling keep:draw[:hold[:last[:seed]]]")).collect();
        assert!(f.len() >= 2, "--coil-sampling keep:draw[:hold[:last[:seed]]]");
        hasty_recon::p2np::CoilSampling { keep: f[0], draw: f[1], hold: *f.get(2).unwrap_or(&2), full_last: *f.get(3).unwrap_or(&12), reshuffle: true, seed: *f.get(4).unwrap_or(&1) as u64 }
    });
    assert!(coil_sampling.is_none() || model == "seg", "--coil-sampling needs --model seg");
    assert!(lagged.is_none() || (model == "seg" && vmodel.is_none()), "--lagged needs --model seg (complex images, no --vmodel)");
    assert!(field_spec == "truth" || field_spec.starts_with("est:"), "--field truth|est:<recon_b0 run>[:bin]");
    assert!(r2s_spec == "off" || r2s_spec == "truth" || r2s_spec.starts_with("est:"), "--r2s off|truth|est:<run>[:bin]");
    assert!(smaps_spec == "truth" || smaps_spec.starts_with("espirit"), "--smaps truth|espirit[:eig_min]");
    let fat_model = take("--fat-model").unwrap_or("shared".into());
    assert!(matches!(fat_model.as_str(), "shared" | "separate" | "joint"), "--fat-model {fat_model}: shared, separate or joint");
    let fat_joint = fat_fraction.is_some() && fat_model == "joint";
    assert!(!fat_joint || model == "seg", "--fat-model joint needs --model seg");
    // --conc-fit quad (default) | deg4: degree of the concomitant-monomial fit at the GNL-warped positions (see the docs)
    let conc_fit = take("--conc-fit").unwrap_or("quad".into());
    assert!(matches!(conc_fit.as_str(), "quad" | "deg4"), "--conc-fit {conc_fit}: quad or deg4");
    let conc_hi = conc_fit == "deg4";
    assert!(!conc_hi || (phys == "full" && order == 2), "--conc-fit deg4 needs --phys full and correction order 2");
    let voxel_basis: Option<i32> = match take("--voxel-basis").as_deref() { None | Some("dirac") => None, Some("rect") => Some(1), Some("bspline3") => Some(4), Some(o) => panic!("--voxel-basis {o}") };
    let grid_scale: Option<[f64; 3]> = take("--grid-scale").map(|s| { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); if v.len() == 1 { [v[0]; 3] } else { [v[0], v[1], v[2]] } });
    // test data: per-encode residual phase c_e + g_e·r (Legendre degree 1, amplitude AMP rad, seeded), injected exactly
    // as a coordinate shift + data phase (E_k(x e^{i g (r − n/2)}) = E_{k−g}(x)); recorded in run.json
    let inject: Option<(u64, f64)> = take("--inject-phase").map(|s| { let v: Vec<&str> = s.split(',').collect(); (v[0].parse().unwrap(), v[1].parse().unwrap()) });
    // static-tissue anchor for --vpoly: relative strength μ; mask from the simulator truth (outside the lumen band)
    // --vanchor μ[,auto|truth]: auto = static mask from the robust residual-phase fit (vinit), truth = simulator
    // auto refinement: pulsatile = temporal std of v over the nodes > mult × median (static mask), dilated by dil voxels
    let vanchor: Option<(f64, String)> = take("--vanchor").map(|s| { let v: Vec<&str> = s.split(',').collect(); (v[0].parse().unwrap(), v.get(1).unwrap_or(&"auto").to_string()) });
    let vanchor_refine: (f64, usize) = take("--vanchor-refine").map(|s| { let v: Vec<&str> = s.split(',').collect(); (v[0].parse().unwrap(), v.get(1).map_or(2, |d| d.parse().unwrap())) }).unwrap_or((1.5, 3));
    // learned prior (7-point dual-VENC data): export directory of tools/denoiser/export.py; κ mixes x + κ (D(x) − x)
    let learned_dir = take("--learned").map(PathBuf::from);
    let tile: Option<usize> = take("--tile").map(|s| s.parse().unwrap()).filter(|&t: &usize| t > 0);
    let keep_frames: Option<usize> = take("--keep-frames").map(|s| s.parse().unwrap()).filter(|&t: &usize| t > 0);
    // learned prior precision on CUDA: fp32 | bf16 (autocast, the training regime) | fp16
    let learned_prec = take("--learned-prec").unwrap_or("bf16".into());
    let init_from: Option<PathBuf> = take("--init-from").map(PathBuf::from);
    let truth_box = match take("--truth-sampling").as_deref() {
        None | Some("point") => false,
        Some("box") => true,
        Some(o) => panic!("--truth-sampling point|box, got {o}"),
    };
    // --kcrop takes an optional value ([FRAC][:prefix]): the next argument when it parses as one
    let kcrop: Option<hasty_recon::pyramid::KCrop> = args.iter().position(|a| a == "--kcrop").map(|i| {
        args.remove(i);
        match args.get(i).map(|v| hasty_recon::pyramid::KCrop::parse(v)) {
            Some(Ok(k)) => {
                args.remove(i);
                k
            }
            _ => hasty_recon::pyramid::KCrop::default(),
        }
    });
    let cmdline = std::env::args().collect::<Vec<_>>().join(" ");
    let dir = PathBuf::from(&args[0]);
    assert!(model == "seg" || temporal.is_none(), "--temporal needs --model seg");
    // run output directory: arrays, metrics and figures (tools/figs/run_figs.py)
    let out = out_arg.map(PathBuf::from).unwrap_or_else(|| {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let tag = format!("{name}_{model}{}_{phys}{}{}", temporal.map_or(String::new(), |b| format!("_t{b}")), if gnl { "_gnl" } else { "" }, if conc_hi { "_deg4" } else { "" }) + if fat_fraction.is_some() { "_fat" } else { "" } + if fat_fraction.is_some() && fat_model != "shared" { fat_model.as_str() } else { "" }
            + if field_spec != "truth" { "_fest" } else { "" } + if r2s_spec != "off" { "_r2s" } else { "" } + if rate_spec.is_some() { "_rate" } else { "" } + if smaps_spec != "truth" { "_espirit" } else { "" }
            + &coil_sampling.as_ref().map_or(String::new(), |c| format!("_cs{}-{}", c.keep, c.draw))
            + &if kcrop.is_some() || init_from.is_some() { format!("_g{}", grid_scale.map_or("1".into(), |g| g.map(|v| v.to_string()).join("-"))) } else { String::new() }
            + &kcrop.map_or(String::new(), |k| format!("_kc{}{}", k.frac, if k.prefix { "p" } else { "" }))
            + if init_from.is_some() { "_init" } else { "" };
        hasty_site::runs().join(tag)
    });
    std::fs::create_dir_all(&out).unwrap();
    if fresh {
        for e in std::fs::read_dir(&out).unwrap().flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if n.starts_with("state_") || n.starts_with("metrics_") || n.starts_with("recon_") || n.starts_with("vel_") {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    let ctl = RunCtl::new(&out, ckpt_every);
    let every_s = if ckpt_every > 0 { format!("checkpoint every {ckpt_every} iterations") } else { "no periodic checkpoints (--ckpt-every N)".to_string() };
    println!("output directory {} ({every_s}; Ctrl-C or touch {}/STOP to pause with a checkpoint)", out.display(), out.display());
    let t0 = Instant::now();
    // k-space data path: legacy in-memory loader, or streamed (kload) — asked for, or the sim would not fit in RAM
    const GIB: f64 = 1073741824.0;
    let mem_avail = available_memory();
    let streamed = {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let raw = ["n_readouts", "n_adc", "n_coils"].iter().map(|k| meta[*k].as_u64().unwrap() as usize).product::<usize>() * 8;
        let s = match kstore_arg.as_str() {
            "legacy" => false,
            // the legacy loader holds the file twice while loading, then the whitened and the compressed copies
            "auto" => 3 * raw > mem_avail / 2,
            _ => true,
        };
        println!("k-space data: {:.2} GiB raw coil data, {:.1} GiB RAM available → {}", raw as f64 / GIB, mem_avail as f64 / GIB, if s { "streamed from the files (kload)" } else { "legacy in-memory loader" });
        s
    };
    let mut sd = if streamed { SimData::load_lazy(&dir) } else { SimData::load(&dir) }.unwrap();
    if let Some(s) = grid_scale {
        sd.set_recon_grid(s);
        println!("reconstruction grid {:?} (native {:?}), voxel {:?} mm", sd.dims, sd.tdims, sd.vox.map(|v| (v * 1e4).round() / 10.0));
    }
    // respiratory bins: frame index r · F + f
    let nf_card = sd.n_frames;
    let mut resp_info = serde_json::Value::Null;
    if resp_bins > 1 || resp_accept < 1.0 {
        assert!(temporal.is_none() && vmodel.is_none(), "--resp-bins / --resp-accept with --temporal / --vmodel: not yet (their time bases are cyclic over the cardiac frames)");
        let npy_f64 = |p: &std::path::Path| -> (Vec<usize>, Vec<f64>) {
            let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
            let h = std::str::from_utf8(&b[10..10 + hl]).unwrap();
            assert!(h.contains("'<f8'") && !h.contains("'fortran_order': True"), "{}: C-order f8 expected", p.display());
            let shape: Vec<usize> = h.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter_map(|x| x.trim().parse().ok()).collect();
            (shape, b[10 + hl..].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
        };
        let n_ro = sd.n_readouts();
        let amp: Vec<f64> = match resp_src.as_str() {
            // physio.npy columns: t, ECG phase, PPG phase, bellows, true r (simphantom)
            "bellows" => { let (sh, v) = npy_f64(&PathBuf::from(&dir).join("physio.npy")); assert_eq!(sh[0], n_ro); (0..n_ro).map(|i| v[i * sh[1] + 3]).collect() }
            "truth" => { let (sh, v) = npy_f64(&PathBuf::from(&dir).join("resp.npy")); assert_eq!(sh[0], n_ro); (0..n_ro).map(|i| v[i * sh[1]]).collect() }
            o => panic!("--resp bellows|truth, got {o}"),
        };
        // gating: the accepted readouts (lowest FRAC of the amplitude), binned among themselves
        let acc: Vec<usize> = if resp_accept < 1.0 {
            let mut o: Vec<usize> = (0..n_ro).collect();
            o.sort_by(|&a, &b| amp[a].partial_cmp(&amp[b]).unwrap());
            let mut keep: Vec<usize> = o[..((resp_accept * n_ro as f64).round() as usize).max(1)].to_vec();
            keep.sort_unstable();
            keep
        } else {
            (0..n_ro).collect()
        };
        let amp_acc: Vec<f64> = acc.iter().map(|&i| amp[i]).collect();
        let bins_acc = hasty_recon::selfgate::amplitude_bins(&amp_acc, resp_bins);
        let mut bins = vec![usize::MAX; n_ro];
        for (j, &i) in acc.iter().enumerate() {
            bins[i] = bins_acc[j];
        }
        let mut mean = vec![(0.0f64, 0usize); resp_bins];
        for i in 0..n_ro {
            if bins[i] == usize::MAX {
                sd.ro.frame[i] = usize::MAX;
                continue;
            }
            sd.ro.frame[i] += bins[i] * nf_card;
            mean[bins[i]].0 += amp[i];
            mean[bins[i]].1 += 1;
        }
        sd.n_frames = resp_bins * nf_card;
        let means: Vec<f64> = mean.iter().map(|(a, n)| a / (*n).max(1) as f64).collect();
        // the simulator's true breathing state of each bin (physio.npy column 4, when present): the truth to score against
        let true_r: Option<Vec<f64>> = {
            let p = PathBuf::from(&dir).join("physio.npy");
            p.exists().then(|| npy_f64(&p)).filter(|(sh, _)| sh.len() == 2 && sh[1] >= 5 && sh[0] == n_ro).map(|(sh, v)| {
                let mut m = vec![(0.0f64, 0usize); resp_bins];
                for i in 0..n_ro {
                    if bins[i] != usize::MAX {
                        m[bins[i]].0 += v[i * sh[1] + 4];
                        m[bins[i]].1 += 1;
                    }
                }
                m.iter().map(|(a, n)| a / (*n).max(1) as f64).collect()
            })
        };
        let kept = acc.len();
        if resp_accept < 1.0 {
            println!("respiratory gating: {kept} of {n_ro} readouts accepted ({:.1} %, {resp_src} amplitude ≤ {:.3})", 100.0 * kept as f64 / n_ro as f64, amp_acc.iter().cloned().fold(f64::MIN, f64::max));
        }
        println!("respiratory bins: {resp_bins} by {resp_src} amplitude (mean {:?}{}) × {nf_card} cardiac frames = {} frames", means.iter().map(|m| (m * 1000.0).round() / 1000.0).collect::<Vec<_>>(), true_r.as_ref().map_or(String::new(), |r| format!(", true breathing state {:?}", r.iter().map(|m| (m * 1000.0).round() / 1000.0).collect::<Vec<_>>())), sd.n_frames);
        resp_info = serde_json::json!({ "bins": resp_bins, "source": resp_src, "mean_amplitude": means, "true_state": true_r, "cardiac_frames": nf_card, "accept": resp_accept, "accepted_readouts": kept });
    }
    if bin_group {
        let ne_g = sd.encode_m1.len();
        let n_ro = sd.n_readouts();
        let mut moved = 0usize;
        for g0 in (0..n_ro).step_by(ne_g) {
            let g1 = (g0 + ne_g).min(n_ro);
            assert!((g0..g1).all(|i| sd.ro.enc[i] == i - g0), "--bin-group: readouts are not grouped as encode 0..{ne_g} per rotation");
            let fm = sd.ro.frame[bin_group_readout(&sd, g0, g1, bin_group_mid)];
            for i in g0..g1 {
                moved += (sd.ro.frame[i] != fm) as usize;
                sd.ro.frame[i] = fm;
            }
        }
        println!("--bin-group: every rotation's {ne_g} encodes in one frame ({}); {moved} of {n_ro} readouts moved ({:.1} %)", if bin_group_mid { "the middle readout's" } else { "the readout nearest the velocity time, hasty_recon::data::velocity_time" }, 100.0 * moved as f64 / n_ro as f64);
    }
    let sd = sd;
    let (nv, nc, nf) = (sd.nv(), sd.nc, sd.n_frames);
    // encodes (4-point referenced / Hadamard, 7-point dual-VENC, ...) and their velocity decoder
    let dec_h = FlowDecoder::new(&sd.encode_m1);
    let vext_design = sd.meta["sequence"]["encoding"]["unaliased"].as_f64();
    let r_first = dec_h.range().iter().cloned().fold(0.0, f64::max);
    let (dec, vdec_info) = {
        let (mode, margin) = match vdecode.split_once(':') {
            Some((m, x)) => (m.to_string(), x.parse::<f64>().expect("--vdecode ml:MARGIN")),
            None => (vdecode.clone(), 1.0),
        };
        let ml = match mode.as_str() {
            "hier" => false,
            "ml" => true,
            "auto" => vext_design.is_some_and(|v| v > 1.001 * r_first),
            _ => panic!("--vdecode auto|hier|ml[:MARGIN]"),
        };
        if ml {
            let vext = vext_design.expect("--vdecode ml needs the design's unaliased range (meta sequence.encoding.unaliased)");
            let d = FlowDecoder::new(&sd.encode_m1).with_alias_search(hasty_recon::flow::AliasSearch { vext, margin });
            let active = d.alias_search().is_some();
            println!("velocity decoder: hierarchical (first stage ±{r_first:.2} m/s){}", if active { format!(" + alias search over ±{vext:.2} m/s (design), margin {margin}") } else { " (no alias inside the design range)".into() });
            (d, serde_json::json!({ "mode": "ml", "active": active, "vext": vext, "margin": margin, "first_stage_range": r_first }))
        } else {
            println!("velocity decoder: hierarchical (±{r_first:.2} m/s){}", vext_design.map_or(String::new(), |v| format!("; design unaliased range ±{v:.2} m/s")));
            (FlowDecoder::new(&sd.encode_m1), serde_json::json!({ "mode": "hier", "first_stage_range": r_first, "design_range": vext_design }))
        }
    };
    let dec_cmp = dec.alias_search().is_some();
    let ne = dec.n_enc();
    let learned = learned_dir.as_ref().map(|d| {
        assert_eq!(ne, 7, "--learned needs 7-point dual-VENC data (sim4d --venc-hi)");
        // the model's slot layout [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]: encode e ≥ 1 along axis (e − 1) / 2
        for (i, k) in dec.k().iter().enumerate() {
            let a = i / 2;
            assert!((0..3).all(|b| b == a || k[b].abs() < 1e-9 * k[a].abs()), "encode {} is not along axis {a}", i + 1);
        }
        let p = hasty_recon::learned::LearnedPrior::load(&hasty_recon::learned::default_plugin(), d, !learned_cpu).unwrap();
        if p.on_cuda {
            p.set_autocast(match learned_prec.as_str() { "fp32" => 0, "fp16" => 2, _ => 1 }).unwrap();
        }
        println!("learned prior {} on {} ({}), tiles {tile:?}, kept frames per window {keep_frames:?}", d.display(), if p.on_cuda { "CUDA" } else { "CPU" }, if p.on_cuda { learned_prec.as_str() } else { "fp32" });
        p
    });
    let kv7: [f32; 7] = if ne == 7 { std::array::from_fn(|e| dec.kv()[e] as f32) } else { [0.0; 7] };
    // --init-from: a previous level's images (checked now, read when needed)
    let init = init_from.as_ref().map(|p| {
        let ini = hasty_recon::pyramid::InitImages::open(p).unwrap_or_else(|e| panic!("--init-from {}: {e}", p.display()));
        assert_eq!(ini.ne, ne, "--init-from {}: {} encodes, this scan has {ne}", ini.path.display(), ini.ne);
        ini.frame_for(0, nf, nf_card).unwrap_or_else(|e| panic!("{e}"));
        println!("initial images: {} ({:?} × {} encodes × {} frames) → grid {:?}, band-limited (Fourier coefficients kept, values × {:.4} voxel-volume ratio)", ini.path.display(), ini.dims, ini.ne, ini.nf, sd.dims, ini.nv() as f64 / nv as f64);
        if warm_iters > 0 {
            println!("  --init-from replaces the --warm {warm_iters} PCGLS start");
        }
        ini
    });
    let mut pyr_info = serde_json::json!({ "grid": sd.dims, "vox": sd.vox, "native": sd.tdims, "truth_sampling": if truth_box { "box" } else { "point" }, "init_from": init.as_ref().map(|i| i.path.clone()) });
    let w = sd.whitener();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    // sets: static t = frame·ne + encode (one frame each); temporal t = encode (all frames)
    let fe: Vec<(Option<usize>, usize)> = if temporal.is_some() { (0..ne).map(|e| (None, e)).collect() } else { (0..nf).flat_map(|f| (0..ne).map(move |e| (Some(f), e))).collect() };
    let enc_of = |t: usize| fe[t].1;
    let t_load = Instant::now();
    let mut host_sets: Vec<SampleSet> = if streamed {
        // coordinates, τ and concomitant coefficients now; the data after the coil compression is known
        let meta_only = LoadOpts { w: &w, cc: None, keep: None, data: false };
        fe.iter().map(|&(f, e)| sd.load_set(&sd.readouts_of(e, f), &meta_only).unwrap()).collect()
    } else {
        std::thread::scope(|sc| {
            let hs: Vec<_> = fe.iter().map(|&(f, e)| { let (sd, w) = (&sd, &w); sc.spawn(move || sd.sample_set(&sd.readouts_of(e, f), w)) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    if streamed {
        println!("sample records of {} sets ({} samples) in {:.1}s", fe.len(), host_sets.iter().map(|s| s.tau.len()).sum::<usize>(), t_load.elapsed().as_secs_f64());
    }
    // per-set phase of the injected residual phase (applied to streamed data on load)
    let mut inj_rot = vec![Complex64::new(1.0, 0.0); fe.len()];
    // injected residual phase: a[e-1] = [a0, a_x, a_y, a_z] (Legendre P0, P1(u_x), P1(u_y), P1(u_z), u = (2i+1)/n − 1)
    let injected: Vec<[f64; 4]> = match inject {
        None => vec![],
        Some((seed, amp)) => {
            let mut sgen = seed.max(1) ^ 0x2545F4914F6CDD1D;
            let mut u = || { sgen ^= sgen << 13; sgen ^= sgen >> 7; sgen ^= sgen << 17; 2.0 * ((sgen >> 11) as f64 / (1u64 << 53) as f64) - 1.0 };
            let a: Vec<[f64; 4]> = (1..ne).map(|_| [amp * u(), amp * u(), amp * u(), amp * u()]).collect();
            for (t, &(_, e)) in fe.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                let c = &a[e - 1];
                // φ = c0 + Σ_a (2 a_a / n_a)(i_a − n_a/2): constant c0 + Σ a_a / n_a, slope g_a = 2 a_a / n_a rad / voxel
                let c0 = c[0] + (0..3).map(|ax| c[1 + ax] / sd.dims[ax] as f64).sum::<f64>();
                let g: [f64; 3] = [0, 1, 2].map(|ax| 2.0 * c[1 + ax] / sd.dims[ax] as f64);
                let s = &mut host_sets[t];
                for ax in 0..3 {
                    for k in s.coords[ax].iter_mut() {
                        *k += g[ax] as f32;
                    }
                }
                let rot = Complex64::from_polar(1.0, c0);
                inj_rot[t] = rot;
                for y in s.y.iter_mut() {
                    let z = Complex64::new(y.re as f64, y.im as f64) * rot;
                    *y = C32::new(z.re as f32, z.im as f32);
                }
            }
            println!("injected residual phase (rad; P0, P1x, P1y, P1z per encode ≥ 1): {:?}", a.iter().map(|c| c.map(|v| (v * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
            a
        }
    };
    // calibration region (whitened, all coils; encode 0, all frames, the samples inside the box) by a low-resolution CG
    // (hasty_recon::espirit::calib_from_noncart): the density-neutral coil Gram and the ESPIRiT input
    let cal_dims = sd.dims.map(|d| 24.min(d) & !1);
    let calib_raw: Option<(Vec<C32>, usize)> = (coil_gram == "calib" || smaps_spec != "truth").then(|| {
        let t_c = std::time::Instant::now();
        let (mut kc, mut ky): ([Vec<f32>; 3], Vec<Vec<C32>>) = ([vec![], vec![], vec![]], vec![vec![]; nc]);
        let in_box = |k: &[f32; 3]| (0..3).all(|a| (k[a] as f64 * sd.dims[a] as f64 / cal_dims[a] as f64).abs() < std::f64::consts::PI * (1.0 - 1e-6));
        for t in (0..fe.len()).filter(|&t| enc_of(t) == 0) {
            if streamed {
                let s = sd.load_set(&host_sets[t].readouts, &LoadOpts { w: &w, cc: None, keep: Some(&in_box), data: true }).unwrap();
                let m = s.tau.len();
                for a in 0..3 {
                    kc[a].extend_from_slice(&s.coords[a]);
                }
                for c in 0..nc {
                    ky[c].extend_from_slice(&s.y[c * m..(c + 1) * m]);
                }
                continue;
            }
            let hsx = &host_sets[t];
            let m = hsx.tau.len();
            let sel: Vec<usize> = (0..m).filter(|&j| (0..3).all(|a| (hsx.coords[a][j] as f64 * sd.dims[a] as f64 / cal_dims[a] as f64).abs() < std::f64::consts::PI * (1.0 - 1e-6))).collect();
            for a in 0..3 {
                kc[a].extend(sel.iter().map(|&j| hsx.coords[a][j]));
            }
            for c in 0..nc {
                ky[c].extend(sel.iter().map(|&j| hsx.y[c * m + j]));
            }
        }
        let yall: Vec<C32> = ky.concat();
        let (cal, used) = hasty_recon::espirit::calib_from_noncart(&dev, &st, &kern, [&kc[0], &kc[1], &kc[2]], &yall, nc, sd.dims, cal_dims, 30, 1e-3).unwrap();
        println!("calibration region {cal_dims:?}: {used} samples of encode 0, {nc} coils ({:.1}s)", t_c.elapsed().as_secs_f64());
        (cal, used)
    });
    let ncal = cal_dims[0] * cal_dims[1] * cal_dims[2];
    let coil_info;
    let cc = {
        let mut g = hasty_recon::coils::CoilGram::new(nc);
        let what = if coil_gram == "calib" {
            g.add(&calib_raw.as_ref().unwrap().0, ncal, 1);
            "whitened calibration region"
        } else {
            // the whitened data of set 0 (≤ 20000 samples, as before)
            let m0 = host_sets[0].tau.len();
            if streamed {
                let s0 = sd.load_set(&host_sets[0].readouts, &LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
                g.add(&s0.y, m0, (m0 / 20000).max(1));
            } else {
                g.add(&host_sets[0].y, m0, (m0 / 20000).max(1));
            }
            "whitened samples of set 0"
        };
        let ccx = hasty_recon::coils::CoilCompression::from_gram(&g, coil_rule);
        println!("coil compression ({what}): {}", ccx.report());
        coil_info = serde_json::json!({"gram": coil_gram, "rule": coil_rule.to_string(), "kept": ccx.nvc(), "of": nc, "retained": ccx.retained, "eig_rel": ccx.eig.iter().map(|v| v / ccx.eig.iter().sum::<f64>().max(1e-300)).collect::<Vec<_>>()});
        ccx.rows
    };
    // k-space store key: the stored data are a function of the scan (canonical path: a relative and an absolute spelling
    // share one store), the sets, the noise, the injected phases and the coil-compression rule. Not of the compression
    // basis itself: it is not reproducible run to run (near-degenerate eigenvalues of the calibration Gram let the
    // eigenvectors rotate inside the kept subspace under f32 rounding, 1–4 % entry changes), which gave every run its
    // own 46 GB store. A run that finds a store adopts the basis stored with it (the reconstruction is invariant to a
    // unitary change of the virtual-coil basis: data, maps and noise are all compressed with the same rows).
    let kstore_prefix = streamed.then(|| {
        let canon = std::fs::canonicalize(&dir).unwrap_or_else(|_| std::path::PathBuf::from(&dir));
        let sigma_c = sd.lazy.as_ref().and_then(|l| l.sigma).map(|s| s / 2f64.sqrt());
        format!("{}|traj={}|sets={:?}|nc={nc}|sigma={:?}|inject={inject:?}|cc={coil_gram},{coil_rule}|", canon.display(), std::env::var("HASTY_TRAJ").unwrap_or_default(), host_sets.iter().map(|s| s.tau.len()).collect::<Vec<_>>(), sigma_c)
    });
    let kstore_default = kstore_prefix.as_ref().map(|pre| {
        let hash = fnv1a(pre.as_bytes());
        // HASTY_KSTORE_DIR: stores (pure caches) in another directory, e.g. rustlib/target/kstore on a disk with room;
        // the key includes the canonical scan path, so stores of different scans never collide
        let f = match std::env::var("HASTY_KSTORE_DIR") {
            Ok(d) if !d.is_empty() => std::path::PathBuf::from(d).join(format!("{hash:016x}.kst")),
            _ => dir.join("kstore").join(format!("{hash:016x}.kst")),
        };
        (hash, f)
    });
    let cc: Vec<Vec<Complex64>> = match (&kstore_prefix, &kstore_default) {
        (Some(pre), Some((_, f))) if f.exists() => match KStore::open(f).ok().and_then(|k| k.tag().strip_prefix(pre.as_str()).map(|rest| (k.nvc(), rest.to_string()))) {
            Some((kv, rest)) => {
                // stored as {:.9e}: 1e-10 relative, far below the f32 data
                let z: Vec<Complex64> = rest.split(';').filter(|p| !p.is_empty()).map(|p| {
                    let (a, b) = p.split_once(',').unwrap();
                    Complex64::new(a.parse().unwrap(), b.parse().unwrap())
                }).collect();
                if z.len() == kv * nc {
                    println!("coil compression: the basis of the k-space store {} ({kv} virtual coils)", f.display());
                    z.chunks(nc).map(|r| r.to_vec()).collect()
                } else {
                    cc
                }
            }
            None => cc,
        },
        _ => cc,
    };
    let nvc = cc.len();
    // oracle-vs-estimate diagnostics for run.json
    let mut diag = serde_json::Map::new();
    let mut grid = sd.physics_grid(gnl, &w);
    // --truth-sampling box: the oracle field averaged over each (coarser) reconstruction voxel
    let sample_truth = |field: &[f32], comp: usize, x: &[f64; 3]| if truth_box { sd.sample_log_box(field, comp, x) } else { sd.sample_log(field, comp, x) };
    if truth_box && !sd.native_grid() {
        use rayon::prelude::*;
        let wfac = 2.0 * std::f64::consts::PI * hasty_mrsim::GAMMA_BAR * sd.b0 * 1e-6;
        let om: Vec<f64> = grid.x_log.par_iter().map(|x| wfac * sd.sample_log_box(&sd.field_ppm, 0, x)).collect();
        grid.omega = om;
    }
    // scanner position of every voxel's true location (for maps measured in another scan)
    let xs: Vec<[f64; 3]> = grid.x_log.iter().map(|xl| [0, 1, 2].map(|a| (0..3).map(|b| sd.rot[a][b] * xl[b]).sum::<f64>() + sd.fov_c[a])).collect();
    let est_maps = |spec: &str| -> hasty_recon::fieldmap::FieldMaps {
        let (run, bin) = match spec.rsplit_once(':') {
            Some((r, b)) if b.parse::<usize>().is_ok() => (r.to_string(), Some(b.parse::<usize>().unwrap())),
            _ => (spec.to_string(), None),
        };
        let mut fm = hasty_recon::fieldmap::FieldMaps::load_recon_b0(std::path::Path::new(&run), bin).unwrap_or_else(|e| panic!("field maps {run}: {e:?}"));
        fm.fill(20);
        fm
    };
    // voxels with coil signal (for the reports below)
    let sig: Vec<bool> = {
        let e: Vec<f32> = (0..nv).map(|q| (0..nc).map(|c| grid.smaps[c * nv + q].re.powi(2) + grid.smaps[c * nv + q].im.powi(2)).sum()).collect();
        let mut v: Vec<f32> = e.iter().copied().filter(|&x| x > 0.0).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = v.get(v.len() / 2).copied().unwrap_or(0.0);
        e.iter().map(|&x| x > 0.05 * med).collect()
    };
    if let Some(spec) = field_spec.strip_prefix("est:") {
        let fm = est_maps(spec);
        let tw = 2.0 * std::f64::consts::PI;
        let d: Vec<f64> = (0..nv).filter(|&q| sig[q] && fm.known_at(&xs[q])).map(|q| fm.sample(&xs[q]).0 - grid.omega[q] / tw).collect();
        for q in 0..nv {
            grid.omega[q] = tw * fm.sample(&xs[q]).0;
        }
        let (mean, rmse) = (d.iter().sum::<f64>() / d.len().max(1) as f64, (d.iter().map(|x| x * x).sum::<f64>() / d.len().max(1) as f64).sqrt());
        println!("field: measured ω from {spec} (fit voxels {} of {}); estimate − truth over {} voxels with a B0 fit: mean {mean:.2} Hz, RMSE {rmse:.2} Hz", fm.known.iter().filter(|&&k| k).count(), fm.known.len(), d.len());
        diag.insert("field_err_hz".into(), serde_json::json!({"n": d.len(), "mean": mean, "rmse": rmse}));
    }
    // T2* decay during the readout (segments g = e^{−R2* (τ − τ0)} e^{−iψ}; none by default, as before)
    let r2v: Option<Vec<f64>> = match r2s_spec.as_str() {
        "off" => None,
        "truth" => {
            let t: Vec<f32> = hasty_recon::data::read_npy(&dir.join("truth_r2s.npy")).expect("--r2s truth needs truth_r2s.npy").f32s().iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect();
            let r: Vec<f64> = (0..nv).map(|q| sample_truth(&t, 0, &grid.x_log[q])).collect();
            println!("R2*: truth (tissue PD-weighted 1/T2*; blood not in the truth map), median over signal voxels {:.1} 1/s", { let mut v: Vec<f64> = (0..nv).filter(|&q| sig[q]).map(|q| r[q]).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v.get(v.len() / 2).copied().unwrap_or(0.0) });
            Some(r)
        }
        e => {
            let fm = est_maps(e.strip_prefix("est:").unwrap());
            let r: Vec<f64> = xs.iter().map(|x| fm.sample(x).1.max(0.0)).collect();
            println!("R2*: measured ({e}), median over voxels with a fit {:.1} 1/s", { let mut v: Vec<f64> = (0..nv).filter(|&q| fm.known_at(&xs[q])).map(|q| r[q]).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v.get(v.len() / 2).copied().unwrap_or(0.0) });
            Some(r)
        }
    };
    // --rate: per-voxel complex rate z = R2* + iΔω (R2* into the segment decay, Δω onto the field)
    let rate_on = rate_spec.is_some();
    // Δω of --rate: added to the field, with the static phase Δω τ0 taken out again in the segment phase (below), so
    // the rate term is e^{−z (τ − τ0)} (τ0 = the first ADC sample) and the image keeps its meaning at the echo
    let mut rate_dw: Option<Vec<f64>> = None;
    let r2v: Option<Vec<f64>> = match &rate_spec {
        None => r2v,
        Some(path) => {
            let z = hasty_recon::data::read_npy(std::path::Path::new(path)).unwrap_or_else(|e| panic!("--rate {path}: {e:?}")).c32s();
            let ntv = sd.tdims.iter().product::<usize>();
            let (re, im): (Vec<f32>, Vec<f32>) = z.iter().map(|c| (if c.re.is_finite() { c.re } else { 0.0 }, if c.im.is_finite() { c.im } else { 0.0 })).unzip();
            let (r, dw): (Vec<f64>, Vec<f64>) = if z.len() == nv && !(nv == ntv && sd.native_grid()) {
                (re.iter().map(|&v| v as f64).collect(), im.iter().map(|&v| v as f64).collect())
            } else {
                assert_eq!(z.len(), ntv, "--rate {path}: {} values, the native grid has {ntv} and the reconstruction grid {nv}", z.len());
                ((0..nv).map(|q| sample_truth(&re, 0, &grid.x_log[q])).collect(), (0..nv).map(|q| sample_truth(&im, 0, &grid.x_log[q])).collect())
            };
            for q in 0..nv {
                grid.omega[q] += dw[q];
            }
            rate_dw = Some(dw.clone());
            let med = |v: Vec<f64>| { let mut v = v; v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v.get(v.len() / 2).copied().unwrap_or(0.0) };
            let (mr, md) = (med((0..nv).filter(|&q| sig[q]).map(|q| r[q]).collect()), med((0..nv).filter(|&q| sig[q]).map(|q| dw[q].abs() / (2.0 * std::f64::consts::PI)).collect()));
            println!("complex rate (--rate {path}): R2* median {mr:.1} 1/s, |Δf| median {md:.1} Hz over the signal voxels (R2* in the segment decay, Δω added to the field)");
            diag.insert("rate".into(), serde_json::json!({"file": path, "r2s_median": mr, "df_abs_median_hz": md}));
            Some(r)
        }
    };
    let ff_est: Option<Vec<f32>> = fat_fraction.as_deref().and_then(|s| s.strip_prefix("est:")).map(|spec| {
        let fm = est_maps(spec);
        xs.iter().map(|x| fm.sample(x).2.clamp(0.0, 1.0) as f32).collect()
    });
    let smaps_true_cc = apply_cc(&grid.smaps, nv, nc, &cc);
    let mut smaps = if let Some(arg) = smaps_spec.strip_prefix("espirit") {
        // the calibration region compressed (it is linear in the data)
        let cd = cal_dims;
        let opts = hasty_recon::espirit::EspiritOpts { eig_min: arg.strip_prefix(':').map_or(0.8, |v| v.parse().unwrap()), ..Default::default() };
        let t_e = std::time::Instant::now();
        let (cal, used) = calib_raw.as_ref().unwrap();
        let used = *used;
        let calib = apply_cc(cal, ncal, nc, &cc);
        let em = hasty_recon::espirit::espirit(&calib, cd, cc.len(), sd.dims, &opts);
        // agreement with the (compressed, whitened) true maps: 1 − |⟨ŝ, s/‖s‖⟩| over voxels with signal
        let (mut err, mut cnt, mut masked) = (0.0f64, 0usize, 0usize);
        for q in (0..nv).filter(|&q| sig[q]) {
            let n: f64 = (0..cc.len()).map(|c| { let z = smaps_true_cc[c * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum::<f64>().sqrt();
            let ip = (0..cc.len()).fold(Complex64::new(0.0, 0.0), |a, c| { let (e, t) = (em.maps[c * nv + q], smaps_true_cc[c * nv + q]); a + Complex64::new(e.re as f64, -e.im as f64) * Complex64::new(t.re as f64, t.im as f64) });
            if (0..cc.len()).all(|c| em.maps[c * nv + q] == C32::new(0.0, 0.0)) {
                masked += 1;
            }
            err += 1.0 - ip.norm() / n.max(1e-30);
            cnt += 1;
        }
        println!("coil maps: ESPIRiT ({used} calibration samples of encode 0, calib {cd:?}, {} kernels, eig_min {}, {:.1}s; stages {:?}); vs the true maps over {cnt} signal voxels: mean 1 − |⟨ŝ, s⟩| {:.2e}, {masked} masked", em.n_kernels, opts.eig_min, t_e.elapsed().as_secs_f64(), em.timings.map(|v| (v * 10.0).round() / 10.0), err / cnt.max(1) as f64);
        diag.insert("espirit".into(), serde_json::json!({"kernels": em.n_kernels, "eig_min": opts.eig_min, "calib": cd, "calib_samples": used, "mean_1_minus_ip": err / cnt.max(1) as f64, "masked": masked, "signal_voxels": cnt}));
        em.maps
    } else {
        smaps_true_cc.clone()
    };
    // memory: with the velocity model the k-space data can stay in pinned host memory from the start (the planner
    // streams it per set); decided here, before anything large is on the device
    let samples_total: usize = host_sets.iter().map(|s| s.tau.len()).sum();
    // streamed data: the store (representation from --kstore, or what fits for auto), filled set by set
    let mut kstore: Option<std::sync::Arc<KStore>> = None;
    let mut kstore_info = serde_json::json!({"path": if streamed { "streamed" } else { "legacy" }});
    if streamed {
        let data_b = samples_total * nvc * 8;
        let sigma_c = sd.lazy.as_ref().and_then(|l| l.sigma).map(|s| s / 2f64.sqrt());
        // store tag: the key (above) and the compression basis the data were stored with (adopted on reuse)
        let mut tag = kstore_prefix.clone().unwrap();
        for r in &cc {
            for z in r {
                tag += &format!("{:.9e},{:.9e};", z.re, z.im);
            }
        }
        let (hash, default_file) = kstore_default.clone().unwrap();
        if let Some(p) = default_file.parent() {
            std::fs::create_dir_all(p).unwrap();
        }
        let spec_s = if kstore_arg == "auto" {
            let avail = available_memory();
            if (data_b as f64) < 0.45 * avail as f64 { "ram" } else if (data_b as f64) < 0.45 * 1.2 * avail as f64 { "zstd1" } else { "disk:zstd1" }.to_string()
        } else {
            kstore_arg.clone()
        };
        let spec = KStoreSpec::parse(&spec_s, &default_file).unwrap();
        let reuse = spec.file.as_ref().filter(|p| p.exists()).and_then(|p| KStore::open(p).ok()).filter(|k| k.tag() == tag && k.nvc() == nvc && k.spec().quant_rel == spec.quant_rel);
        let t_k = Instant::now();
        let ks = match reuse {
            Some(k) => {
                println!("k-space store reused: {}", k.report());
                k
            }
            None => {
                // HASTY_KSTORE_REUSE_ONLY (set by tools/recon/pyramid.sh): never write a new file store (they are tens
                // of GB; the pyramid levels share the full-resolution runs' store, whose key does not depend on the grid)
                if spec.file.is_some() && std::env::var("HASTY_KSTORE_REUSE_ONLY").is_ok_and(|v| !v.is_empty() && v != "0") {
                    panic!("HASTY_KSTORE_REUSE_ONLY: no k-space store {} with key {hash:016x} (other arguments than the run that built the store? the key covers the scan, trajectory, sets incl. --resp-bins, noise, --inject-phase and the coil rule)", spec.file.as_ref().unwrap().display());
                }
                let ks = KStore::create(spec, nvc, host_sets.iter().map(|s| s.tau.len()).collect(), sigma_c, &tag).unwrap();
                let opts = LoadOpts { w: &w, cc: Some(&cc), keep: None, data: true };
                for t in 0..fe.len() {
                    let mut s = sd.load_set(&host_sets[t].readouts, &opts).unwrap();
                    assert_eq!(s.tau.len(), host_sets[t].tau.len());
                    if inj_rot[t] != Complex64::new(1.0, 0.0) {
                        for y in s.y.iter_mut() {
                            let z = Complex64::new(y.re as f64, y.im as f64) * inj_rot[t];
                            *y = C32::new(z.re as f32, z.im as f32);
                        }
                    }
                    ks.put_owned(t, s.y).unwrap();
                    if t % 16 == 15 || t + 1 == fe.len() {
                        println!("  k-space store: {} / {} sets ({:.0}s)", t + 1, fe.len(), t_k.elapsed().as_secs_f64());
                    }
                }
                ks.finish().unwrap();
                println!("k-space store built in {:.1}s: {}", t_k.elapsed().as_secs_f64(), ks.report());
                ks
            }
        };
        let stt = ks.stats();
        kstore_info = serde_json::json!({"path": "streamed", "spec": spec_s, "describe": ks.spec().describe(), "file": ks.spec().file, "raw_bytes": stt.raw_bytes, "stored_bytes": stt.stored_bytes, "ratio": stt.ratio(), "quant_step": ks.step(), "tag_hash": format!("{hash:016x}")});
        kstore = Some(std::sync::Arc::new(ks));
    }
    // --kcrop: the samples inside the grid's band. The sample records are cropped (everything downstream sees the
    // kept samples); the whole records stay for the segment tables (built per readout × ADC sample, then cropped);
    // the store (keyed and filled with the whole sets, shared with the full-resolution runs) is filtered on read.
    let mut full_sets: Vec<SampleSet> = vec![];
    let mut crop_sel: Option<Vec<Vec<u32>>> = None;
    let mut crop_adc: Option<hasty_recon::pyramid::AdcStats> = None;
    if let Some(kc) = kcrop {
        use rayon::prelude::*;
        let t_c = Instant::now();
        let sel: Vec<Vec<u32>> = host_sets.par_iter().map(|s| kc.select(&s.coords, sd.na)).collect();
        let lens_full: Vec<usize> = host_sets.iter().map(|s| s.coords[0].len()).collect();
        let st = hasty_recon::pyramid::AdcStats::new(&sel, &lens_full, sd.na);
        let (w0, w1) = st.window;
        let (t_w0, t_w1) = (host_sets[0].tau[w0] * 1e3, host_sets[0].tau[w1] * 1e3);
        let cropped: Vec<SampleSet> = host_sets.par_iter().zip(sel.par_iter()).map(|(s, i)| hasty_recon::pyramid::crop_set(s, i)).collect();
        full_sets = std::mem::replace(&mut host_sets, cropped);
        for s in full_sets.iter_mut() {
            s.y = Vec::new();
        }
        if model != "seg" {
            full_sets = vec![];
        }
        println!("k-space crop ({}): {} of {} samples kept ({:.2} %), ADC window {w0}..{w1} of {} ({t_w0:.3}..{t_w1:.3} ms), {} .. {} per set ({:.1}s)", kc.describe(), st.kept, st.total, 100.0 * st.fraction(), sd.na, sel.iter().map(|s| s.len()).min().unwrap_or(0), sel.iter().map(|s| s.len()).max().unwrap_or(0), t_c.elapsed().as_secs_f64());
        if let Some(ks) = kstore.take() {
            let ks = std::sync::Arc::try_unwrap(ks).unwrap_or_else(|_| panic!("k-space store shared before the crop"));
            let view = ks.with_selection(sel.clone()).unwrap();
            let bytes = st.kept * nvc * 8;
            // the kept samples as a dense RAM copy when they fit (one pass over the store; no per-gradient decode)
            // (HASTY_KCROP_VIEW=1: always read through the filtered view)
            let ram = !view.spec().is_dense() && (bytes as f64) < 0.3 * available_memory() as f64 && std::env::var("HASTY_KCROP_VIEW").is_err();
            let ks = if ram {
                let t_r = Instant::now();
                let r = KStore::create(KStoreSpec::dense(), nvc, view.lens().to_vec(), None, &format!("{}|kcrop", view.tag())).unwrap();
                for t in 0..view.n_sets() {
                    r.put_owned(t, view.read_vec(t).unwrap()).unwrap();
                }
                println!("k-space crop: kept data {:.2} GiB copied to RAM ({:.1}s; the store stays as it is)", bytes as f64 / GIB, t_r.elapsed().as_secs_f64());
                r
            } else {
                println!("k-space crop: kept data {:.2} GiB read through a filtered view of the store", bytes as f64 / GIB);
                view
            };
            kstore = Some(std::sync::Arc::new(ks));
        }
        pyr_info["kcrop"] = serde_json::json!({ "frac": kc.frac, "prefix": kc.prefix, "kept": st.kept, "total": st.total, "fraction": st.fraction(), "adc_window": [w0, w1], "window_ms": [t_w0, t_w1] });
        crop_sel = Some(sel);
        crop_adc = Some(st);
    }
    // (sizes below: the kept samples)
    let samples_total: usize = host_sets.iter().map(|s| s.coords[0].len()).sum();
    dev_mem(&dev, "after loading and coil compression");
    let (early_host, staged) = {
        let (free, total) = dev.ctx.mem_get_info().unwrap();
        let budget = vram_budget.unwrap_or(total.saturating_sub(512 << 20)).min(free);
        let data_b = samples_total * nvc * 8;
        // auto: only when the data cannot be resident during the setup anyway (data + SENSE coordinates + ~4 GiB for
        // the operator, maps and the warm start exceed the budget); otherwise the planner decides later as before
        let est = data_b + samples_total * 12 + (4usize << 30);
        let off = plan_mode == "host" || (plan_mode == "auto" && est > budget);
        // legacy data that do not fit: pinned host memory (velocity model only, as before)
        let host = !streamed && vmodel.is_some() && off;
        // streamed data that do not fit stay in the store, staged per set (image-space P²nP and velocity model)
        let staged = streamed && off;
        if host {
            println!("memory: k-space data {:.2} GiB kept in pinned host memory from the start (data + coordinates + operator ≈ {:.1} GiB vs budget {:.1} GiB)", data_b as f64 / 1073741824.0, est as f64 / 1073741824.0, budget as f64 / 1073741824.0);
        }
        if staged {
            println!("memory: k-space data {:.2} GiB stay in the store, staged per set (data + coordinates + operator ≈ {:.1} GiB vs budget {:.1} GiB)", data_b as f64 / 1073741824.0, est as f64 / 1073741824.0, budget as f64 / 1073741824.0);
        }
        (host, staged)
    };
    // --synthetic with staged data: model-consistent residuals computed on the fly (hasty_recon::synth, below)
    let set_data = |t: usize| -> Vec<C32> {
        match &kstore {
            Some(k) => k.read_vec(t).unwrap(),
            None => apply_cc(&host_sets[t].y, host_sets[t].tau.len(), nc, &cc),
        }
    };
    let mut ys_pin: Vec<cudarc::driver::PinnedHostSlice<C32>> = vec![];
    let ys: Vec<CudaSlice<C32>> = if early_host {
        // device placeholders (never read): the data live in `ys_pin` until the velocity problem takes them
        ys_pin = (0..host_sets.len()).map(|t| {
            let v = set_data(t);
            let mut h = unsafe { dev.ctx.alloc_pinned_with_flags::<C32>(v.len().max(1), 0).unwrap() };
            h.as_mut_slice().unwrap()[..v.len()].copy_from_slice(&v);
            h
        }).collect();
        (0..host_sets.len()).map(|_| st.alloc_zeros::<C32>(1).unwrap()).collect()
    } else if staged {
        (0..host_sets.len()).map(|_| st.alloc_zeros::<C32>(1).unwrap()).collect()
    } else {
        (0..host_sets.len()).map(|t| st.clone_htod(&set_data(t)).unwrap()).collect()
    };
    if !staged {
        // resident on the device (or pinned for the velocity model): the store is not needed any more
        kstore = None;
    }
    if early_host {
        // the uncompressed per-set data are not needed any more (host memory for the pinned copies and tables)
        for s in host_sets.iter_mut() {
            s.y = Vec::new();
        }
    }
    // the set's data on the host (warm start, support): from the device copy or the pinned one
    let y_host = |ys: &[CudaSlice<C32>], t: usize| -> Vec<C32> {
        if let Some(k) = &kstore {
            return k.read_vec(t).unwrap();
        }
        if early_host { let n = host_sets[t].tau.len() * nvc; ys_pin[t].as_slice().unwrap()[..n].to_vec() } else { st.clone_dtoh(&ys[t]).unwrap() }
    };
    // SENSE coordinates on the device: the SENSE model, the support without a warm start (not needed otherwise when
    // the data are streamed)
    let coords: Vec<[CudaSlice<f32>; 3]> = if (early_host || staged) && model == "seg" && warm_iters > 0 {
        vec![]
    } else {
        host_sets.iter().map(|s| [0, 1, 2].map(|a| st.clone_htod(&s.coords[a]).unwrap())).collect()
    };
    let ms: Vec<usize> = host_sets.iter().map(|s| s.tau.len()).collect();
    println!(
        "{} sets ({}), {}..{} readouts per set, {} virtual coils, model {model}, physics {phys}{}, setup {:.1}s",
        fe.len(),
        if temporal.is_some() { format!("{ne} encodes x {nf} frames, temporal B-spline {}", temporal.unwrap()) } else { format!("{nf} frames x {ne} encodes") },
        host_sets.iter().map(|s| s.readouts.len()).min().unwrap(),
        host_sets.iter().map(|s| s.readouts.len()).max().unwrap(),
        nvc,
        if gnl { " + GNL" } else { "" },
        t0.elapsed().as_secs_f64()
    );
    dev_mem(&dev, "after the data placement");
    // ESPIRiT-style normalisation Σ_c |S_c|² = 1 (receive bias moves into the image intensity; phases and hence
    // velocities are unchanged). Raw Biot–Savart maps blow up next to the loops and dominate λ_max of the normal operator.
    if std::env::var("HASTY_RAW_MAPS").is_err() {
        let dmax = (0..nv).map(|q| (0..nvc).map(|c| smaps[c * nv + q].re.powi(2) + smaps[c * nv + q].im.powi(2)).sum::<f32>()).fold(0f32, f32::max);
        for q in 0..nv {
            let d: f32 = (0..nvc).map(|c| smaps[c * nv + q].re.powi(2) + smaps[c * nv + q].im.powi(2)).sum();
            let sc = 1.0 / (d + 1e-6 * dmax).sqrt();
            for c in 0..nvc {
                smaps[c * nv + q].re *= sc;
                smaps[c * nv + q].im *= sc;
            }
        }
    }
    // plain SENSE operator on the normalised maps (the support estimate without a warm start uses it: raw Biot–Savart
    // maps would weight its adjoint by Σ_c |S_c|², ~90× deeper in the thorax, and cut the heart out of the support)
    let mut op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&host_sets[0].coords[0], &host_sets[0].coords[1], &host_sets[0].coords[2]], &smaps, nvc, vec![], 1e-4).unwrap();
    // per-encode mean concomitant phase folded into coil maps (sense model only)
    let wc = hasty_mrsim::GAMMA / (2.0 * sd.b0);
    let mut maps_host: Vec<Vec<C32>> = if conc_dc && model == "sense" {
        (0..ne)
            .map(|e| {
                let np = sd.spiral().pre[e].len();
                let mut pp = [0.0f64; 6];
                for j in 0..np {
                    let ph = sd.spiral().pre[e][j].rotate(&sd.rot).p;
                    for q in 0..6 {
                        pp[q] += ph[q] / np as f64;
                    }
                }
                let wq = [wc * (pp[0] + pp[1]), wc * pp[2] / 4.0, -wc * pp[4], -wc * pp[5]];
                let mut m = smaps.clone();
                for q in 0..nv {
                    let phi: f64 = (0..4).map(|i| wq[i] * grid.mono[q][i]).sum();
                    let e = Complex64::from_polar(1.0, -phi);
                    for c in 0..nvc {
                        let s = m[c * nv + q];
                        let z = Complex64::new(s.re as f64, s.im as f64) * e;
                        m[c * nv + q] = C32::new(z.re as f32, z.im as f32);
                    }
                }
                m
            })
            .collect()
    } else {
        vec![]
    };
    // warm start: time-averaged plain-model reconstruction per encode (all frames pooled), PCGLS + circulant. The pooled
    // sets stream through the plain operator one at a time with the data residual on the host (pcgls_sets_ctl): pooling
    // them on the device needs 4 data-space vectors of the pooled size (7.5 GB each for a 10-min scan at 2 mm).
    let warm: Vec<Vec<C32>> = if let Some(ini) = &init {
        // --init-from: the per-encode time average of the previous level, resampled (the solver starts per frame)
        let t_i = Instant::now();
        let w: Vec<Vec<C32>> = (0..ne).map(|e| ini.mean_image(e, sd.dims).unwrap()).collect();
        println!("warm start: --init-from time average per encode, resampled {:?} → {:?} ({:.1}s)", ini.dims, sd.dims, t_i.elapsed().as_secs_f64());
        w
    } else if warm_iters > 0 {
        (0..ne)
            .map(|e| {
                let idx: Vec<usize> = (0..fe.len()).filter(|&t| enc_of(t) == e).collect();
                let done = format!("warm_done_{e}");
                if let Some(v) = ctl.load(&done).unwrap() {
                    return Some(st.clone_dtoh(&value_dev(&st, &v, nv).unwrap()).unwrap());
                }
                let cat: [Vec<f32>; 3] = [0, 1, 2].map(|a| idx.iter().flat_map(|&t| host_sets[t].coords[a].iter().copied()).collect());
                let maps_e: &[C32] = if !maps_host.is_empty() { &maps_host[e] } else { &smaps };
                let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), sd.dims, [&cat[0], &cat[1], &cat[2]], maps_e, nvc, 1e-3).unwrap();
                drop(cat);
                let mut swapped = if !maps_host.is_empty() { Some(st.clone_htod(&maps_host[e]).unwrap()) } else { None };
                if let Some(m) = swapped.as_mut() {
                    op.swap_smaps(m);
                }
                let set_coords: Vec<[&[f32]; 3]> = idx.iter().map(|&t| [0, 1, 2].map(|a| &host_sets[t].coords[a][..])).collect();
                let mut xe = st.alloc_zeros::<C32>(nv).unwrap();
                let (_, oc) = hasty_recon::cg::pcgls_sets_ctl(&mut op, Some(&mut pre), &st, &set_coords, &mut |i| y_host(&ys, idx[i]), &mut xe, 0.0, warm_iters, &mut |_, _| {}, Some((&ctl, &format!("warm{e}")))).unwrap();
                if let Some(m) = swapped.as_mut() {
                    op.swap_smaps(m);
                }
                if let Outcome::Stopped(k) = oc {
                    println!("stopped in the warm start (encode {e}, CG iteration {k}): state saved, rerun the same command to resume");
                    return None;
                }
                ctl.save(&done, &hasty_recon::checkpoint::dev_value(&st, &xe).unwrap()).unwrap();
                Some(st.clone_dtoh(&xe).unwrap())
            })
            .collect::<Option<Vec<_>>>()
            .unwrap_or_else(|| std::process::exit(0))
    } else {
        vec![]
    };
    println!("warm start: {} PCGLS iterations per encode ({:.0}s)", if init.is_some() { 0 } else { warm_iters }, t0.elapsed().as_secs_f64());
    dev_mem(&dev, "after the warm start");
    // object support: smoothed magnitude of the warm start (or of the summed adjoint of the reference encode without
    // one), thresholded, enclosed low-signal regions filled per slice (background = what a flood fill from the slice
    // border reaches through sub-threshold voxels)
    // (also keeps the smoothed magnitude: the data-derived signal weight of --seg-weight mass)
    let mut support_mag: Vec<f32> = vec![];
    let support: Vec<bool> = {
        let mut acc = vec![0f32; nv];
        if !warm.is_empty() {
            for w in &warm {
                for q in 0..nv {
                    acc[q] += (w[q].re * w[q].re + w[q].im * w[q].im).sqrt();
                }
            }
        } else {
            let mut tmp = st.alloc_zeros::<C32>(nv).unwrap();
            for t in (0..fe.len()).filter(|&t| enc_of(t) == 0) {
                let c = &coords[t];
                op.set_points(&st, [&c[0], &c[1], &c[2]], ms[t]).unwrap();
                if early_host || staged {
                    let d = st.clone_htod(&y_host(&ys, t)).unwrap();
                    op.adjoint(&st, &d, &mut tmp).unwrap();
                } else {
                    op.adjoint(&st, &ys[t], &mut tmp).unwrap();
                }
                let h = st.clone_dtoh(&tmp).unwrap();
                for q in 0..nv {
                    acc[q] += (h[q].re * h[q].re + h[q].im * h[q].im).sqrt();
                }
            }
        }
        // smooth (3 box passes, radius 4 in-plane / 1 through-plane native voxels, σ ≈ 4.5 / 1.4 voxels: in mm, so
        // the same physical radius on any reconstruction grid) so that dark structures inside the body (vessel lumen
        // with flow dephasing, lungs) do not open holes in the mask
        let d = sd.dims;
        for (axis, r) in [(0usize, 4usize), (1, 4), (2, 1)] {
            let r = (r as f64 * sd.res / sd.vox[axis]).round() as usize;
            let (n, stride) = (d[axis], [1, d[0], d[0] * d[1]][axis]);
            for _ in 0..3 {
                let src = acc.clone();
                for q in 0..nv {
                    let i = (q / stride) % n;
                    let (lo, hi) = (i.saturating_sub(r), (i + r).min(n - 1));
                    let base = q - i * stride;
                    acc[q] = (lo..=hi).map(|k| src[base + k * stride]).sum::<f32>() / (hi - lo + 1) as f32;
                }
            }
        }
        let mut sorted = acc.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let thr = 0.05 * sorted[sorted.len() * 99 / 100];
        support_mag = acc.clone();
        let fg: Vec<bool> = acc.iter().map(|&v| v > thr).collect();
        let mut bg = vec![false; nv];
        for k in 0..d[2] {
            let mut stack: Vec<(usize, usize)> = Vec::new();
            for i in 0..d[0] {
                stack.push((i, 0));
                stack.push((i, d[1] - 1));
            }
            for j in 0..d[1] {
                stack.push((0, j));
                stack.push((d[0] - 1, j));
            }
            while let Some((i, j)) = stack.pop() {
                let q = i + d[0] * (j + d[1] * k);
                if bg[q] || fg[q] {
                    continue;
                }
                bg[q] = true;
                if i > 0 { stack.push((i - 1, j)); }
                if i + 1 < d[0] { stack.push((i + 1, j)); }
                if j > 0 { stack.push((i, j - 1)); }
                if j + 1 < d[1] { stack.push((i, j + 1)); }
            }
        }
        (0..nv).map(|q| !bg[q]).collect()
    };
    println!("support {} voxels", support.iter().filter(|&&b| b).count());
    // coil maps zero outside the (dilated) object support, as ESPIRiT maps are: no unconstrained air modes
    // (dilation by 3 native voxels in mm: the same physical margin on any reconstruction grid)
    let mut dil = support.clone();
    {
        let d = sd.dims;
        let n_dil = ((3.0 * sd.res / (sd.vox.iter().sum::<f64>() / 3.0)).round() as usize).max(1);
        for _ in 0..n_dil {
            let prev = dil.clone();
            for q in 0..nv {
                if prev[q] {
                    continue;
                }
                let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
                let nb = [(i > 0).then(|| q - 1), (i + 1 < d[0]).then(|| q + 1), (j > 0).then(|| q - d[0]), (j + 1 < d[1]).then(|| q + d[0]), (k > 0).then(|| q - d[0] * d[1]), (k + 1 < d[2]).then(|| q + d[0] * d[1])];
                dil[q] = nb.iter().flatten().any(|&n| prev[n]);
            }
        }
    }
    // --coil-mask auto|none|support: zero the coil maps outside `dil` (support) or not (none). auto: none for maps that
    // exist everywhere (--smaps truth: a threshold of the blurred warm start would cut tissue with signal out of the
    // model, e.g. 25k voxels on ph0001; air is handled by --air instead), support for estimated maps (ESPIRiT has no
    // information outside the object). `dil` stays the fit region (concomitant and segment fits) either way.
    let coil_mask_mode = coil_mask_arg.clone().unwrap_or_else(|| if smaps_spec == "truth" { "none".into() } else { "support".into() });
    assert!(coil_mask_mode == "none" || coil_mask_mode == "support", "--coil-mask {coil_mask_mode}: auto, none or support");
    let mask_maps = coil_mask_mode == "support" && std::env::var("HASTY_NO_MASK").is_err();
    let maps_mask: Vec<bool> = if mask_maps { dil.clone() } else { vec![true; nv] };
    if !mask_maps {
        println!("coil maps not masked (--coil-mask {coil_mask_mode}): {} voxels; fit region {} voxels", nv, dil.iter().filter(|&&b| b).count());
    }
    if mask_maps {
        for q in (0..nv).filter(|&q| !dil[q]) {
            for c in 0..nvc {
                smaps[c * nv + q] = C32::new(0.0, 0.0);
                for m in maps_host.iter_mut() {
                    m[c * nv + q] = C32::new(0.0, 0.0);
                }
            }
        }
        op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&host_sets[0].coords[0], &host_sets[0].coords[1], &host_sets[0].coords[2]], &smaps, nvc, vec![], 1e-4).unwrap();
        println!("coil maps masked to {} voxels", dil.iter().filter(|&&b| b).count());
    }
    let maps: Vec<CudaSlice<C32>> = maps_host.iter().map(|m| st.clone_htod(m).unwrap()).collect();
    // lumen and truth at the true positions of the reconstruction voxels
    let lumen: Vec<usize> = (0..nv).filter(|&q| sample_truth(&sd.truth_frac, 0, &grid.x_log[q]) > 0.9).collect();
    println!("lumen {} voxels: {} in the support, {} in the masked maps", lumen.len(), lumen.iter().filter(|&&q| support[q]).count(), lumen.iter().filter(|&&q| maps_mask[q]).count());
    let dims4 = |n: usize| [sd.dims[0], sd.dims[1], sd.dims[2], n];
    if save_smaps {
        save(&out.join("smaps.npy"), "<c8", &dims4(nvc), c8(&smaps));
    }
    {
        // the axial slice with the most lumen voxels (the figures' slice), all virtual coils: [x, y, coil]
        let nxy = sd.dims[0] * sd.dims[1];
        let mut cnt = vec![0usize; sd.dims[2]];
        for &q in &lumen {
            cnt[q / nxy] += 1;
        }
        let zs = (0..sd.dims[2]).rev().max_by_key(|&z| cnt[z]).unwrap_or(sd.dims[2] / 2);
        let sl: Vec<C32> = (0..nvc).flat_map(|c| smaps[c * nv + zs * nxy..c * nv + (zs + 1) * nxy].iter().copied()).collect();
        save(&out.join(format!("smaps_axial_z{zs}.npy")), "<c8", &[sd.dims[0], sd.dims[1], nvc], c8(&sl));
    }
    save(&out.join("mask_support.npy"), "<f4", &sd.dims, f4(support.iter().map(|&b| b as u8 as f32)));
    save(&out.join("mask_maps.npy"), "<f4", &sd.dims, f4(maps_mask.iter().map(|&b| b as u8 as f32)));
    save(&out.join("mask_fit.npy"), "<f4", &sd.dims, f4(dil.iter().map(|&b| b as u8 as f32)));
    if !warm.is_empty() {
        save(&out.join("warm.npy"), "<c8", &dims4(ne), c8(&warm.concat()));
    }
    // fixed circulant preconditioners per frame (block-diagonal over the frames of a time-resolved set)
    // (the velocity model uses its own Gauss–Newton metric: no circulant kernels, they would cost ~8 GB)
    let mut lag_tau: Option<f64> = None;
    let mut lag_info = serde_json::Value::Null;
    let need_circ = vmodel.is_none() && preconds.iter().any(|p| Precond::parse(p).is_some_and(|q| q.metric()));
    // --circ-share encode (default for frame-resolved sets): one circulant per encode from the pooled samples of all its
    // frames (the frames of an encode share the trajectory's density; per-set circulants would need 133 MB each at 2 mm,
    // ≈ 37 GB for 280 sets); set: one per set (the original, needed for --temporal block-diagonal metrics)
    let circ_share = circ_share_arg.clone().unwrap_or_else(|| if temporal.is_some() { "set".into() } else { "encode".into() });
    assert!(circ_share == "set" || circ_share == "encode", "--circ-share {circ_share}: set or encode");
    assert!(circ_share == "set" || temporal.is_none(), "--circ-share encode needs one frame per set (no --temporal)");
    let circ_map: Vec<usize> = if need_circ && circ_share == "encode" { (0..fe.len()).map(enc_of).collect() } else { vec![] };
    // shared per-encode circulants live in host memory (uploaded per application into one shared device workspace):
    // the joint water + fat operator with per-frame maps leaves no room for 7 device-resident N-grid preconditioners
    let circ_ws = if need_circ && circ_share == "encode" { Some(hasty_recon::precond::CircShared::new(&dev, &st, sd.dims).unwrap()) } else { None };
    let circ: Vec<Vec<CirculantPrecond>> = if need_circ && circ_share == "encode" {
        (0..ne)
            .map(|e| {
                let idx: Vec<usize> = (0..fe.len()).filter(|&t| enc_of(t) == e).collect();
                let c: [Vec<f32>; 3] = [0, 1, 2].map(|a| idx.iter().flat_map(|&t| host_sets[t].coords[a].iter().copied()).collect());
                let pc = CirculantPrecond::new(&dev, &st, kern.clone(), sd.dims, [&c[0], &c[1], &c[2]], &smaps, nvc, eps).unwrap();
                println!("circulant metric for encode {e}: {} sets pooled, dynamic range {:.2e} (host-resident)", idx.len(), pc.dyn_range);
                vec![pc.to_host(&st, circ_ws.as_ref().unwrap()).unwrap()]
            })
            .collect()
    } else if need_circ {
        host_sets
            .iter()
            .enumerate()
            .map(|(t, s)| {
                let frames: Vec<Option<usize>> = if temporal.is_some() { (0..nf).map(Some).collect() } else { vec![None] };
                frames
                    .iter()
                    .map(|&f| {
                        let c: [Vec<f32>; 3] = match &crop_sel {
                            // --kcrop: kept sample i comes from readout sel[i] / na of the whole set
                            Some(cs) => [0, 1, 2].map(|a| (0..s.coords[a].len()).filter(|&i| f.is_none_or(|f| sd.ro.frame[s.readouts[cs[t][i] as usize / sd.na]] == f)).map(|i| s.coords[a][i]).collect()),
                            None => {
                                let sel: Vec<usize> = (0..s.readouts.len()).filter(|&r| f.is_none_or(|f| sd.ro.frame[s.readouts[r]] == f)).collect();
                                [0, 1, 2].map(|a| sel.iter().flat_map(|&r| s.coords[a][r * sd.na..(r + 1) * sd.na].iter().copied()).collect())
                            }
                        };
                        CirculantPrecond::new(&dev, &st, kern.clone(), sd.dims, [&c[0], &c[1], &c[2]], &smaps, nvc, eps).unwrap()
                    })
                    .collect()
            })
            .collect()
    } else {
        vec![]
    };
    if !warm.is_empty() {
        // diagnostic: warm-start velocity vs the time-averaged truth over the lumen
        let (mut mr, mut mt, mut e2, mut n) = ([0.0f64; 3], [0.0f64; 3], 0.0f64, 0usize);
        for &q in &lumen {
            let z: Vec<Complex64> = (0..ne).map(|e| Complex64::new(warm[e][q].re as f64, warm[e][q].im as f64)).collect();
            let vq = dec.velocity(&z);
            for a in 0..3 {
                let v = vq[a];
                let t = (0..nf_card).map(|f| sd.sample_log(&sd.truth_vel, a + 3 * f, &grid.x_log[q])).sum::<f64>() / nf_card as f64;
                mr[a] += v;
                mt[a] += t;
                e2 += (v - t).powi(2);
            }
            n += 1;
        }
        let mut wv = vec![0f32; 3 * nv];
        for q in 0..nv {
            let z: Vec<Complex64> = (0..ne).map(|e| Complex64::new(warm[e][q].re as f64, warm[e][q].im as f64)).collect();
            let v = dec.velocity(&z);
            for a in 0..3 {
                wv[q + nv * a] = v[a] as f32;
            }
        }
        save(&out.join("warm_vel.npy"), "<f4", &dims4(3), f4(wv));
        println!("warm-start velocity in the lumen: mean {:?} m/s, truth (time average) {:?}, RMSE {:.3} m/s", mr.map(|v| format!("{:.3}", v / n as f64)), mt.map(|v| format!("{:.3}", v / n as f64)), (e2 / n as f64).sqrt());
    }
    // the operator stack
    let mut sense_sets;
    let mut seg_sets;
    let mut fat_joint_sets;
    let sop: &mut dyn SetOp = if model == "sense" {
        sense_sets = SenseSets { op: &mut op, coords, ms: ms.clone(), maps, map_of_set: if conc_dc { (0..fe.len()).map(enc_of).collect() } else { vec![] } };
        &mut sense_sets
    } else {
        // the plain SENSE operator (device coil maps + NUFFT plan) served the warm start and the support only: free it
        // before the segment operator is built (the joint water + fat operator needs the room)
        drop(op);
        let dev_free = |what: &str| {
            if let Ok((free, total)) = dev.ctx.mem_get_info() {
                println!("device memory {what}: {:.2} GiB free of {:.2} GiB", free as f64 / GIB, total as f64 / GIB);
            }
        };
        dev_free("before the segment operator");
        let ts = Instant::now();
        let (with_b0, with_conc) = (phys != "none", phys == "full");
        let fit_vox: Vec<usize> = (0..nv).filter(|&q| dil[q]).collect();
        let fit = fit_monomials(&sd, &grid, &fit_vox);
        println!("monomial quadratic fit over {} voxels: rms {:.1e} / max {:.1e} m² (z², x²+y², xz, yz: {:?})", fit_vox.len(), fit.rms.iter().cloned().fold(0.0, f64::max), fit.max.iter().cloned().fold(0.0, f64::max), fit.max.map(|v| format!("{v:.1e}")));
        let fit4 = conc_hi.then(|| fit_monomials_hi(&sd, &grid, &fit_vox));
        if let Some(f4) = &fit4 {
            println!("monomial degree-4 fit (--conc-fit deg4): rms {:.1e} / max {:.1e} m² ({:?}); degree 3/4 part to first order in the kernel (order 3)", f4.quad.rms.iter().cloned().fold(0.0, f64::max), f4.quad.max.iter().cloned().fold(0.0, f64::max), f4.quad.max.map(|v| format!("{v:.1e}")));
        }
        let htab = fit4.as_ref().map(hi_table);
        // --fat-model joint: the fat fraction (one map or one per frame) and the six-peak spectrum c(τ), for the segments
        let fs_joint = hasty_qfit::waterfat::FatSpectrum::six_peak();
        let ff_joint: Option<Vec<f32>> = fat_joint.then(|| {
            let src = fat_fraction.as_ref().unwrap();
            let path = if src == "truth" { PathBuf::from(&dir).join("truth_fat_fraction.npy") } else { PathBuf::from(src) };
            let ff = match &ff_est { Some(v) => v.clone(), None => maps_on_grid(&sd, &path, &grid.x_log, truth_box) };
            let ff = expand_cardiac_maps(ff, nv, nf, nf_card);
            assert!(ff.len() == nv || ff.len() == nv * nf, "--fat-fraction: {} values for {nv} voxels (or {nv} × {nf} frames, or {nv} × {nf_card} cardiac frames)", ff.len());
            println!("water + fat (--fat-model joint): known fat fraction ({}{}), {} voxels with ff > 5 %, six-peak spectrum at {:.2} T in the segments", path.display(), if ff.len() > nv { ", per frame" } else { "" }, ff[..nv].iter().filter(|&&f| f > 0.05).count(), sd.b0);
            ff.iter().map(|f| f.clamp(0.0, 1.0)).collect()
        });
        let cfat = |t: f64| -> Complex64 {
            let w = hasty_recon::setop::fat_spectrum_weights(&fs_joint.ppm, &fs_joint.amp, sd.b0, &[t])[0];
            Complex64::new(w.re as f64, w.im as f64)
        };
        // the main fat peak (rad/s), for the segment count of the joint model
        let om_fat = {
            let i = (0..fs_joint.amp.len()).max_by(|&a, &b| fs_joint.amp[a].partial_cmp(&fs_joint.amp[b]).unwrap()).unwrap();
            2.0 * std::f64::consts::PI * hasty_mrsim::GAMMA_BAR * sd.b0 * fs_joint.ppm[i] * 1e-6
        };
        // the encode-wide mean concomitant coefficients (all frames): the segments and the auto segment count
        // the whole sample records (readout × ADC sample structure): the uncropped sets with --kcrop
        let rec_sets: &[SampleSet] = if full_sets.is_empty() { &host_sets } else { &full_sets };
        // segment time window: the ADC samples kept by --kcrop (all of them otherwise)
        let (wa, wb) = crop_adc.as_ref().map_or((0, sd.na - 1), |s| s.window);
        let wbar: Vec<Vec<[f64; 4]>> = (0..ne).map(|e| mean_coefficients(&(0..fe.len()).filter(|&t| enc_of(t) == e).map(|t| &rec_sets[t]).collect::<Vec<_>>(), sd.na)).collect();
        let wbar: Vec<Vec<[f64; 4]>> = if with_conc { wbar } else { vec![vec![[0.0; 4]; sd.na]; ne] };
        // --b0-absorb: the smooth field part handled per sample (weights: smoothed warm-start magnitude² × coil energy)
        let coil_energy: Vec<f64> = (0..nv).map(|q| (0..nvc).map(|c| { let z = smaps[c * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect();
        let field_poly = if b0_absorb > 0 && with_b0 {
            let tau_a: Vec<f64> = (0..sd.na).map(|n| rec_sets[0].tau[n]).collect();
            let wv: Vec<f64> = fit_vox.iter().map(|&q| (support_mag.get(q).copied().unwrap_or(1.0) as f64).powi(2) * coil_energy[q]).collect();
            let fp = hasty_recon::segphys::fit_field_poly(&sd, &grid, &fit_vox, Some(&wv), b0_absorb, 0.5 * (tau_a[wa] + tau_a[wb]));
            let hz = |c: f64| c / (2.0 * std::f64::consts::PI);
            println!("--b0-absorb {b0_absorb}: field part {:.1} Hz + [{:.1}, {:.1}, {:.1}] Hz/m · ρ{} in the sample tables", hz(fp.c[0]), hz(fp.c[1]), hz(fp.c[2]), hz(fp.c[3]), if b0_absorb >= 2 { format!(" + quadratic [{}] Hz/m² (ΔC, relative to the readout centre)", fp.c[4..].iter().map(|c| format!("{:.0}", hz(*c))).collect::<Vec<_>>().join(", ")) } else { String::new() });
            assert!(b0_absorb < 2 || order >= 1, "--b0-absorb 2 needs the ΔC kernel correction (--order 1|2)");
            fp
        } else {
            hasty_recon::segphys::FieldPoly::none()
        };
        // voxel weights of the segment basis over the fit voxels (seg_fit below): energy (--seg-weight energy)
        let seg_fit_e: Vec<usize> = fit_vox.iter().copied().step_by(seg_fit_stride).collect();
        let seg_ew: Option<Vec<f64>> = seg_energy.then(|| {
            let e: Vec<f64> = seg_fit_e.iter().map(|&q| (support_mag.get(q).copied().unwrap_or(1.0) as f64).powi(2) * coil_energy[q]).collect();
            let mut srt = e.clone();
            srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = srt[srt.len() / 2].max(1e-300);
            println!("segment basis voxel weights: smoothed warm-start magnitude² × coil energy / median ({} fit voxels)", e.len());
            e.into_iter().map(|v| v / med).collect()
        });
        // per-ADC-sample weights 1 + (E(n) + E_vol) / (C σ²) for --seg-basis svdn and --nseg auto:noise: whitened data
        // energy per sample (every 7th readout, at most 512) plus its k-space volume average, over the noise energy per
        // sample (σ: HASTY_NOISE_SIGMA, the sim's noise_sigma, else 1 = data whitened with the measured noise covariance)
        let noise_sigma: f64 = std::env::var("HASTY_NOISE_SIGMA").ok().and_then(|v| v.parse::<f64>().ok()).filter(|v| *v > 0.0).or_else(|| sd.meta["noise_sigma"].as_f64().filter(|v| *v > 0.0)).unwrap_or(1.0);
        let snr_w: Option<Vec<f64>> = (seg_svdn || seg_noise_rule.is_some()).then(|| {
            let ros: Vec<usize> = (0..sd.n_readouts()).step_by(7).take(512).collect();
            let e = sd.adc_energy_whitened(&ros, &sd.whitener()).unwrap();
            // |k| of every ADC sample (first readout of set 0; the shells of the white floor)
            let s0 = &rec_sets[0];
            let kap: Vec<f64> = (wa..=wb).map(|n| (0..3).map(|a| (s0.coords[a][n] as f64 / sd.vox[a]).powi(2)).sum::<f64>().sqrt()).collect();
            let noise = nc as f64 * noise_sigma * noise_sigma;
            let w = hasty_nufft::noise_time_weights(&e[wa..=wb], &kap, noise);
            println!("per-sample weights 1 + (E(n) + E_vol) / (C σ²), σ = {noise_sigma:.3}: first {:.2e}, middle {:.2e}, last {:.2e}, mean {:.2e}; white floor E_vol / (C σ²) = {:.1}", w[0] - 1.0, w[w.len() / 2] - 1.0, w[w.len() - 1] - 1.0, w.iter().sum::<f64>() / w.len() as f64 - 1.0, hasty_nufft::vol_energy(&e[wa..=wb], &kap) / noise);
            w
        });
        let snr_grid: Option<(Vec<f64>, Vec<f64>)> = snr_w.as_ref().map(|w| {
            let tau_a: Vec<f64> = (wa..=wb).map(|n| rec_sets[0].tau[n]).collect();
            hasty_nufft::seg_time_grid(&tau_a, w, 96)
        });
        if let (Some(r_noise), Some((ts_n, tw_n)), Some(w)) = (seg_noise_rule, snr_grid.as_ref(), snr_w.as_ref()) {
            // relative weighted error ε(L) → data error in noise rms ≈ ε(L) √(mean w) (w = 1 + (E(n) + E_vol) / (C σ²):
            // proportional to the sample's signal plus the white floor; research/segment_basis.md)
            // margin 1.5: the separable prediction was up to 1.3–2× optimistic against the exact data-space error at
            // L ≥ 9 on ph0001 (research/segment_basis.md §2–3); HASTY_SEG_MARGIN overrides
            let margin: f64 = std::env::var("HASTY_SEG_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
            let gain = (w.iter().sum::<f64>() / w.len() as f64).sqrt();
            let tol = r_noise / (gain * margin);
            let tau_a: Vec<f64> = (0..sd.na).map(|n| rec_sets[0].tau[n]).collect();
            let (t0, t1) = (tau_a[wa], tau_a[wb]);
            let ffm: Option<Vec<f64>> = ff_joint.as_ref().map(|ff| ff[..nv].iter().map(|&f| f as f64).collect());
            let mut lsel = 2usize;
            // error weights of the prediction: the energy weights (always); fit weights of cheb: --seg-weight
            let seg_ew_rule: Option<Vec<f64>> = seg_ew.clone().or_else(|| {
                let e: Vec<f64> = seg_fit_e.iter().map(|&q| (support_mag.get(q).copied().unwrap_or(1.0) as f64).powi(2) * coil_energy[q]).collect();
                let mut srt = e.clone();
                srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = srt[srt.len() / 2].max(1e-300);
                Some(e.into_iter().map(|v| v / med).collect())
            });
            let seg_vw_rule: Option<Vec<f64>> = seg_ew.clone();
            for e in 0..ne {
                let psi_raw0 = segment_phase(&grid, &wbar[e], &tau_a, with_b0);
                let psi_raw = |v: usize, t: f64| psi_raw0(v, t) - rate_dw.as_ref().map_or(0.0, |d| d[v] * tau_a[0]);
                let psi = field_poly.residual(&sd, &psi_raw);
                let dec = |v: usize, t: f64| -> f64 { r2v.as_ref().map_or(1.0, |r| (-r[v] * (t - tau_a[0])).exp()) };
                let g = |v: usize, t: f64| -> Complex64 {
                    let z = Complex64::from_polar(dec(v, t), -psi(v, t));
                    match &ffm { Some(f) => z * (Complex64::new(1.0 - f[v], 0.0) + f[v] * cfat(t)), None => z }
                };
                // per-frame fat maps: the segments are fixed-time (uniform nodes for svdn) whatever the basis, except
                // svdn-union (the union water + fat Gram, its eigenvalue tail is the rule)
                let per_frame_fat = ff_joint.as_ref().is_some_and(|ff| ff.len() > nv);
                let l_e = if seg_union && per_frame_fat {
                    let (ww, wf) = union_weights(&seg_fit_e, seg_ew_rule.as_deref(), ff_joint.as_ref().unwrap(), nv);
                    let gw = |v: usize, t: f64| Complex64::from_polar(dec(v, t), -psi(v, t));
                    let ev = if rate_on { hasty_nufft::fat_union_spectrum_g(&seg_fit_e, &ww, &wf, ts_n, tw_n, &gw, &cfat) } else { hasty_nufft::fat_union_spectrum(&seg_fit_e, &ww, &wf, ts_n, tw_n, &psi, &cfat) };
                    hasty_nufft::seg_count_for(&ev, tol, 2, 64)
                } else if seg_svd && !per_frame_fat {
                    // the optimal rank-L error in the energy-weighted metric (exact for --seg-weight energy, a lower
                    // bound for other voxel weights)
                    let ev = hasty_nufft::weighted_spectrum(&seg_fit_e, seg_ew_rule.as_deref(), ts_n, tw_n, &g);
                    hasty_nufft::seg_count_for(&ev, tol, 2, 64)
                } else {
                    // fixed-time diagonals: scan L with the interpolators solved on the time grid
                    (2..=64).find(|&l| {
                        let cheb = seg_cheb;
                        let tl: Vec<f64> = if cheb { hasty_nufft::cheb_nodes(l, t0, t1) } else { (0..l).map(|i| t0 + (t1 - t0) * i as f64 / (l - 1) as f64).collect() };
                        let weighted = cheb || seg_energy || (seg_svdn && per_frame_fat);
                        hasty_nufft::nodes_weighted_error(&seg_fit_e, if weighted { seg_vw_rule.as_deref() } else { None }, seg_ew_rule.as_deref(), ts_n, tw_n, &tl, &g) <= tol
                    }).unwrap_or(64)
                };
                lsel = lsel.max(l_e);
            }
            println!("segments (auto:noise): predicted data error × {margin} ≤ {r_noise} noise rms (relative weighted error ≤ {tol:.2e}, ×{gain:.1}) → L = {lsel} ({} basis)", seg_basis_arg);
            nseg = lsel;
        }
        if nseg == 0 {
            // phase cycles spanned over the ADC window by the fitted voxels (B0 and the mean concomitant phase), worst encode
            let tau_adc: Vec<f64> = (0..sd.na).map(|n| rec_sets[0].tau[n]).collect();
            let (t0, t1) = (tau_adc[wa], tau_adc[wb]);
            let mut cyc = 0.0f64;
            for e in 0..ne {
                let psi_raw = segment_phase(&grid, &wbar[e], &tau_adc, with_b0);
                let psi = field_poly.residual(&sd, &psi_raw);
                let mut d: Vec<f64> = fit_vox.iter().step_by(7).map(|&q| psi(q, t1) - psi(q, t0)).collect();
                if fat_joint {
                    // joint water + fat: the main fat peak of every voxel too
                    let n = d.len();
                    for i in 0..n {
                        d.push(d[i] + om_fat * (t1 - t0));
                    }
                }
                let (lo, hi) = d.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                cyc = cyc.max((hi - lo) / (2.0 * std::f64::consts::PI));
            }
            nseg = ((1.3 * cyc + 3.5).ceil() as usize).max(2);
            println!("segments (auto): the field{} spans {cyc:.2} phase cycles over the {:.2} ms readout → L = {nseg}", if fat_joint { " with the main fat peak (joint water + fat)" } else { "" }, (t1 - t0) * 1e3);
        }
        let seg_opts = |nseg: usize| SegNufftOpts { sigma: 1.25, width: 10, nseg, order: if conc_hi { 3 } else { order }, frames: if temporal.is_some() { nf } else { 1 }, bspline: temporal.unwrap_or(1), max_grid_bytes: 6 << 30 };
        let mut eng = SegNufft::new(&dev, sd.dims, sd.vox, nvc, seg_opts(nseg)).unwrap();
        eng.set_maps(&st, &smaps).unwrap();
        // segments per encode from the encode-wide mean coefficients (all frames)
        let tau_adc: Vec<f64> = (0..sd.na).map(|n| rec_sets[0].tau[n]).collect();
        let (tw0, tw1, ntab) = (tau_adc[wa], tau_adc[wb], ((wb - wa) / seg_tab_stride + 1).max(256));
        // svdw with --kcrop: the time weights count the kept samples only
        let keep_w: Option<Vec<f64>> = crop_adc.as_ref().map(|s| s.keep_fraction());
        let seg_fit: Vec<usize> = fit_vox.iter().copied().step_by(seg_fit_stride).collect();
        // data energy per ADC time for --seg-basis svdw (every 7th readout, at most 512)
        let adc_w: Option<Vec<f64>> = seg_svdw.then(|| {
            let ros: Vec<usize> = (0..sd.n_readouts()).step_by(7).take(512).collect();
            let e = sd.adc_energy(&ros).unwrap();
            let n = e.len();
            println!("data energy per readout quarter: {:?} (--seg-basis svdw)", (0..4).map(|q| (e[q * n / 4..(q + 1) * n / 4].iter().sum::<f64>() / e.iter().sum::<f64>() * 1000.0).round() / 1000.0).collect::<Vec<_>>());
            e
        });
        let tw_fn = |t: f64| -> f64 {
            let e = adc_w.as_ref().unwrap();
            let dt = (tau_adc[sd.na - 1] - tau_adc[0]) / (sd.na - 1).max(1) as f64;
            let i = (((t - tau_adc[0]) / dt).round().max(0.0) as usize).min(e.len() - 1);
            e[i] * keep_w.as_ref().map_or(1.0, |k| k[i.min(k.len() - 1)])
        };
        let tw: Option<&(dyn Fn(f64) -> f64 + Sync)> = if seg_svdw { Some(&tw_fn) } else { None };
        let seg_vw: Option<Vec<f64>> = seg_mass.map(|(lo, hi)| {
            let mut m: Vec<f32> = seg_fit.iter().map(|&q| support_mag[q]).collect();
            let mut srt = m.clone();
            srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = srt[srt.len() / 2].max(1e-30);
            for v in m.iter_mut() {
                *v = (*v / med).clamp(lo, hi);
            }
            println!("segment basis voxel weights: warm-start magnitude / median, clamped to [{lo}, {hi}] ({} fit voxels)", m.len());
            m.into_iter().map(|v| v as f64).collect()
        });
        assert!(seg_mass.is_none() || seg_svd, "--seg-weight mass needs --seg-basis svd|svdw|svdn");
        let seg_vw: Option<Vec<f64>> = seg_vw.or_else(|| seg_ew.clone());
        let svd_build = |nseg: usize, g: &(dyn Fn(usize, f64) -> Complex64 + Sync)| match (&snr_grid, seg_svdn) {
            (Some((ts_n, tw_n)), true) => hasty_nufft::build_segments_svd_grid(nv, nseg, tw0, tw1, ntab, &seg_fit, seg_vw.as_deref(), ts_n, tw_n, g),
            _ => build_segments_svd_weighted(nv, nseg, tw0, tw1, ntab, &seg_fit, seg_vw.as_deref(), tw, 96, g),
        };
        // --seg-basis cheb: Chebyshev node times, interpolators weighted by --seg-weight; --seg-basis fixed with
        // --seg-weight energy: the uniform nodes with energy-weighted interpolators (the same diagonals as `fixed`)
        let nodes_of = |nseg: usize| -> Vec<f64> { if seg_cheb { hasty_nufft::cheb_nodes(nseg, tw0, tw1) } else { (0..nseg).map(|l| if nseg == 1 { 0.5 * (tw0 + tw1) } else { tw0 + (tw1 - tw0) * l as f64 / (nseg - 1) as f64 }).collect() } };
        let seg_nodes_w = seg_cheb || seg_energy;
        let cheb_build = |nseg: usize, g: &(dyn Fn(usize, f64) -> Complex64 + Sync)| hasty_nufft::build_segments_nodes(nv, &nodes_of(nseg), tw0, tw1, ntab, &seg_fit, seg_vw.as_deref(), g);
        let make_segs = |eng: &SegNufft, nseg: usize| -> Vec<hasty_nufft::SegSegments> {
        (0..ne)
            .map(|e| {
                let psi_raw0 = segment_phase(&grid, &wbar[e], &tau_adc, with_b0);
                let psi_raw = |v: usize, t: f64| psi_raw0(v, t) - rate_dw.as_ref().map_or(0.0, |d| d[v] * tau_adc[0]);
                let psi = field_poly.residual(&sd, &psi_raw);
                // one interpolator table entry per ADC sample (256 left a 3–7e-4 floor with fat at 8–12 ms readouts)
                // T2* decay relative to the readout start (--r2s; 1 without)
                let dec = |v: usize, t: f64| -> f64 { r2v.as_ref().map_or(1.0, |r| (-r[v] * (t - tau_adc[0])).exp()) };
                if let Some(ff) = &ff_joint {
                    if ff.len() == nv {
                        // joint, one map: folded into the diagonals, table fitted to g_v(τ) = [(1 − ff) + ff c(τ)] e^{−iψ}
                        let g = |v: usize, t: f64| Complex64::from_polar(dec(v, t), -psi(v, t)) * (Complex64::new(1.0 - ff[v] as f64, 0.0) + ff[v] as f64 * cfat(t));
                        let (sp, at, ta, dt) = if seg_svd { svd_build(nseg, &g) } else if seg_nodes_w { cheb_build(nseg, &g) } else { build_segments_complex(nv, nseg, tw0, tw1, ntab, &seg_fit, &g) };
                        return eng.segments(&st, &sp, &at, ta, dt).unwrap();
                    }
                    // joint, per-frame maps: water diagonals + an ff-independent table, maps applied by the operator
                    // (--seg-basis cheb|svdn or --seg-weight energy: node times / weights of hasty_nufft::
                    // build_segments_fat_nodes — the rank-L SVD basis does not fit the per-node fat factor c_l of the
                    // operator, so svdn falls back to uniform nodes with its voxel weights here; Chebyshev nodes lose
                    // in the noise-referenced error, research/segment_basis.md)
                    if let (true, Some((ts_n, tw_n))) = (seg_union, snr_grid.as_ref()) {
                        let (ww, wf) = union_weights(&seg_fit, seg_vw.as_deref(), ff, nv);
                        // --rate: the union basis carries the decay as well (the plain call keeps the phase-only basis)
                        let gw = |v: usize, t: f64| Complex64::from_polar(dec(v, t), -psi(v, t));
                        let (sp, at, cl, ta, dt, nodes, cond) = if rate_on {
                            hasty_nufft::build_segments_svd_fat_union_g(nv, nseg, tw0, tw1, ntab, &seg_fit, &ww, &wf, ts_n, tw_n, &gw, &cfat)
                        } else {
                            hasty_nufft::build_segments_svd_fat_union(nv, nseg, tw0, tw1, ntab, &seg_fit, &ww, &wf, ts_n, tw_n, &psi, &cfat)
                        };
                        if e == 0 {
                            println!("segments (svdn-union, per-frame fat maps): L = {nseg}, cardinal nodes (ms) {:?}, condition {cond:.1}", nodes.iter().map(|&q| ((ta + q as f64 * dt - tw0) * 1e4).round() / 10.0).collect::<Vec<_>>());
                        }
                        return eng.segments_fat(&st, &sp, &at, &cl, ta, dt).unwrap();
                    }
                    let (sp, at, cl, ta, dt) = if seg_cheb || seg_svdn || seg_energy {
                        let tl = if seg_cheb { hasty_nufft::cheb_nodes(nseg, tw0, tw1) } else { (0..nseg).map(|l| if nseg == 1 { 0.5 * (tw0 + tw1) } else { tw0 + (tw1 - tw0) * l as f64 / (nseg - 1) as f64 }).collect() };
                        hasty_nufft::build_segments_fat_nodes(nv, &tl, tw0, tw1, ntab, &seg_fit, seg_vw.as_deref(), &psi, &cfat)
                    } else {
                        build_segments_fat(nv, nseg, tw0, tw1, ntab, &seg_fit, &psi, &cfat)
                    };
                    return eng.segments_fat(&st, &sp, &at, &cl, ta, dt).unwrap();
                }
                if r2v.is_some() {
                    let g = |v: usize, t: f64| Complex64::from_polar(dec(v, t), -psi(v, t));
                    let (sp, at, ta, dt) = if seg_svd { svd_build(nseg, &g) } else if seg_nodes_w { cheb_build(nseg, &g) } else { build_segments_complex(nv, nseg, tw0, tw1, ntab, &seg_fit, &g) };
                    return eng.segments(&st, &sp, &at, ta, dt).unwrap();
                }
                let (sp, at, ta, dt) = if seg_svd {
                    svd_build(nseg, &|v, t| Complex64::from_polar(1.0, -psi(v, t)))
                } else if seg_nodes_w {
                    cheb_build(nseg, &|v, t| Complex64::from_polar(1.0, -psi(v, t)))
                } else {
                    build_segments(nv, nseg, tw0, tw1, ntab, &seg_fit, &psi)
                };
                eng.segments(&st, &sp, &at, ta, dt).unwrap()
            })
            .collect()
        };
        let segs = make_segs(&eng, nseg);
        let mut resid = 0.0f64;
        // --b0-absorb 2: largest per-sample quadratic B0 phase handed to the ΔC Taylor correction
        let mut absorb_q = 0.0f64;
        // sample tables per set: resident, or (they would crowd the device / --plan host) pinned host memory streamed
        // into two device slots; decided from set 0's measured size
        let mut tables_host: Option<bool> = None;
        let mut table_pack: Option<Option<hasty_value::Options>> = None;
        let samples_all: usize = host_sets.iter().map(|s| s.tau.len()).sum();
        let pts: Vec<_> = (0..fe.len())
            .map(|t| {
                let mut p = (|| {
                if let (Some(f4), Some(h)) = (&fit4, &htab) {
                    let (mut tab, dw) = seg_tables_hi(&sd, &rec_sets[t], f4, with_conc, with_conc.then(|| &wbar[enc_of(t)][..]));
                    absorb_q = absorb_q.max(hasty_recon::segphys::absorb_field_poly(&mut tab, &sd, &field_poly, &fit_vox));
                    let (tab, dw) = match &crop_sel { Some(sel) => { let m = tab.tau.len(); (tab.select(&sel[t]), hasty_recon::pyramid::gather_rows(&dw, m, &sel[t])) } None => (tab, dw) };
                    resid = resid.max(tab.max_resid_phase);
                    return eng.points_hi(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, &dw, h, temporal.map(|_| &tab.t[..])).unwrap();
                }
                let mut tab = seg_tables(&sd, &rec_sets[t], &fit, with_conc, with_conc.then(|| &wbar[enc_of(t)][..]));
                absorb_q = absorb_q.max(hasty_recon::segphys::absorb_field_poly(&mut tab, &sd, &field_poly, &fit_vox));
                let tab = match &crop_sel { Some(sel) => tab.select(&sel[t]), None => tab };
                resid = resid.max(tab.max_resid_phase);
                eng.points(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, temporal.map(|_| &tab.t[..])).unwrap()
                })();
                let on_host = *tables_host.get_or_insert_with(|| {
                    let bps = p.table_bytes() as f64 / p.m.max(1) as f64;
                    let est = bps * samples_all as f64;
                    // what is still to come on the device: the complex P²nP keeps residuals (data size) and up to
                    // six image stacks; the velocity model's state is small and its planner revisits the tables
                    let (free, total) = dev.ctx.mem_get_info().unwrap();
                    let avail = vram_budget.map_or(free, |b| b.saturating_sub(total - free).min(free)) as f64;
                    let stack = (nf * ne * nv * 8) as f64;
                    // (residual buffers of the data size, unless the data are staged from the store)
                    let data_b = if staged { 0.0 } else { (samples_all * nvc * 8) as f64 };
                    let reserve = if vmodel.is_some() { 3.0 * 1073741824.0 } else { data_b + (6.0 + if lagged.is_some() { 2.0 } else { 0.0 }) * stack + 1073741824.0 };
                    let host = plan_mode == "host" || (plan_mode == "auto" && est > avail - reserve);
                    println!("sample tables: {:.0} B/sample, {:.2} GiB for {} sets → {}", bps, est / 1073741824.0, fe.len(), if host { "pinned host memory, streamed into two device slots" } else { "device" });
                    host
                });
                if on_host {
                    p.to_host(&dev.ctx, &st).unwrap();
                    let codec = *table_pack.get_or_insert_with(|| match table_codec {
                        Some(c) => c,
                        None => {
                            let all = p.table_bytes() as f64 / p.m.max(1) as f64 * samples_all as f64;
                            let pack = all > 0.4 * available_memory() as f64;
                            println!("sample tables on the host: {:.2} GiB pinned would be {} of the available RAM → {}", all / 1073741824.0, if pack { "> 40 %" } else { "≤ 40 %" }, if pack { "packed (zstd1)" } else { "pinned" });
                            pack.then(|| hasty_value::Options::zstd(1))
                        }
                    });
                    if let Some(o) = codec {
                        p.pack(&o).unwrap();
                    }
                }
                if t == 0 && std::env::var("HASTY_TABLE_PROBE").is_ok() && !p.is_packed() {
                    // compressibility of the sample tables (set 0): blocks of 1 MiB, 4-byte shuffle
                    let host_now = p.is_host();
                    p.to_host(&dev.ctx, &st).unwrap();
                    for (name, b) in p.host_parts().unwrap() {
                        let mut line = format!("table probe set 0 {name}: {:.1} MB ({:.1} B/sample)", b.len() as f64 / 1e6, b.len() as f64 / p.m.max(1) as f64);
                        let dz = hasty_value::Options { delta: true, ..hasty_value::Options::zstd(1) };
                        for (cn, o) in [("lz4", hasty_value::Options::default()), ("zstd1", hasty_value::Options::zstd(1)), ("zstd3", hasty_value::Options::zstd(3)), ("zstd1+delta", dz)] {
                            let t0 = Instant::now();
                            let c: usize = b.chunks(1 << 20).map(|blk| hasty_value::codec::compress_block(blk, 4, cn.ends_with("delta") && name != "rec" && !name.starts_with("hi"), &o).len()).sum();
                            line += &format!(", {cn} ×{:.2} ({:.0} MB/s)", b.len() as f64 / c.max(1) as f64, b.len() as f64 / 1e6 / t0.elapsed().as_secs_f64());
                        }
                        println!("{line}");
                    }
                    if !host_now {
                        p.to_device(&st).unwrap();
                    }
                }
                p
            })
            .collect();
        if tables_host == Some(true) {
            eng.stream_reserve(&st, &pts.iter().collect::<Vec<_>>()).unwrap();
            eng.stream_prefetch = table_prefetch;
            let (logical, stored) = (pts.iter().map(|p| p.table_bytes()).sum::<usize>(), pts.iter().map(|p| p.host_stored_bytes()).sum::<usize>());
            println!("sample tables streamed: {:.2} GiB on the host ({}, {:.2} GiB of tables), {:.2} GiB device slots, prefetch {}", stored as f64 / 1073741824.0, if pts[0].is_packed() { format!("packed ×{:.2}", logical as f64 / stored.max(1) as f64) } else { "pinned".into() }, logical as f64 / 1073741824.0, eng.stream_slot_bytes() as f64 / 1073741824.0, if table_prefetch { "on (second stream)" } else { "off" });
        }
        if b0_absorb >= 2 {
            println!("--b0-absorb 2: largest per-sample quadratic B0 phase in the ΔC Taylor correction {absorb_q:.3} rad (Taylor-2 truncation ≈ {:.1e})", absorb_q.powi(3) / 6.0);
        }
        let (kf, ka) = eng.autotune(&st, &segs[0], &pts[0]).unwrap();
        println!(
            "SegNufft: L = {nseg}, order {}, frames/set {}, coil batch {}, kernels {kf:.1}/{ka:.1} ms (set 0), max phase left out by the {} fit {resid:.2e} rad, built in {:.1}s",
            eng.order,
            eng.frames,
            eng.cb,
            if conc_hi { "degree-4" } else { "quadratic" },
            ts.elapsed().as_secs_f64()
        );
        let joint_maps = ff_joint.as_ref().filter(|ff| ff.len() > nv).map(|ff| eng.fat_maps(&st, ff).unwrap());
        if joint_maps.is_some() {
            // joint water + fat with per-frame maps: the single-image kernels of the multi path (forward_fat)
            let (mf, ma) = eng.autotune_multi(&st, &segs[0], &pts[0], 1).unwrap();
            println!("SegNufft joint water + fat (per-frame maps): kernels {mf:.1}/{ma:.1} ms (set 0)");
        }
        if fat_fraction.is_some() && fat_model == "shared" {
            // water + fat: both species in one multi-image pass (FatSets); tune those kernels too
            let (g, cbm, extra) = eng.multi_info(&st, 2).unwrap();
            let (mf, ma) = eng.autotune_multi(&st, &segs[0], &pts[0], 2).unwrap();
            println!("SegNufft water + fat: {g} images per gridding pass, coil batch {cbm}, extra grid memory {:.2} GiB, kernels {mf:.1}/{ma:.1} ms (set 0)", extra as f64 / (1u64 << 30) as f64);
        }
        seg_sets = SegSets { eng, pts, segs, seg_of_set: (0..fe.len()).map(enc_of).collect(), kern: kern.clone(), surrogate: None };
        if let Some((tau, ls_arg)) = lagged {
            assert!(joint_maps.is_none(), "--lagged: not with joint water + fat per-frame maps");
            // probe: the warm start of encode 0 (every frame of a time-resolved set), else a smooth blob
            let lx = seg_sets.len_x(0);
            let probe: Vec<C32> = if let Some(w0) = warm.first() {
                (0..lx).map(|i| w0[i % nv]).collect()
            } else {
                (0..lx).map(|i| {
                    let q = i % nv;
                    let u = [0, 1, 2].map(|a| [q % sd.dims[0], (q / sd.dims[0]) % sd.dims[1], q / (sd.dims[0] * sd.dims[1])][a] as f64 / sd.dims[a] as f64 - 0.5);
                    C32::new((-(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]) / 0.05).exp() as f32, 0.0)
                }).collect()
            };
            let dprobe = st.clone_htod(&probe).unwrap();
            let mut y_ex = st.alloc_zeros::<C32>(seg_sets.len_y(0)).unwrap();
            seg_sets.forward(&st, 0, &dprobe, &mut y_ex).unwrap();
            let ny = kern.dot(&st, &y_ex, &y_ex).unwrap().re.sqrt();
            let cands: Vec<usize> = match ls_arg { Some(l) => vec![l], None => (1..nseg).collect() };
            let mut chosen = None;
            for l in cands {
                let mut se = SegNufft::new(&dev, sd.dims, sd.vox, nvc, seg_opts(l)).unwrap();
                se.set_maps(&st, &smaps).unwrap();
                let ss = make_segs(&se, l);
                let mut ys_ = st.alloc_zeros::<C32>(seg_sets.len_y(0)).unwrap();
                se.forward(&st, &ss[seg_sets.seg_of_set[0]], &seg_sets.pts[0], &dprobe, &mut ys_).unwrap();
                kern.axpby(&st, Complex64::new(-1.0, 0.0), &y_ex, Complex64::new(1.0, 0.0), &mut ys_).unwrap();
                let mis = kern.dot(&st, &ys_, &ys_).unwrap().re.sqrt() / ny.max(1e-30);
                println!("lagged gradient: surrogate L_s = {l} of {nseg}: ‖(E − Ẽ)x‖ / ‖E x‖ = {mis:.4} (set 0, {})", if warm.is_empty() { "probe blob" } else { "warm start" });
                if ls_arg.is_some() || mis <= 0.03 {
                    chosen = Some((se, ss, l, mis));
                    break;
                }
            }
            match chosen {
                Some((mut se, ss, l, mis)) => {
                    if seg_sets.tables_on_host() {
                        se.stream_reserve(&st, &seg_sets.pts.iter().collect::<Vec<_>>()).unwrap();
                    }
                    let (kf, ka) = se.autotune(&st, &ss[0], &seg_sets.pts[0]).unwrap();
                    println!("lagged gradient: τ = {tau}, surrogate L_s = {l} (kernels {kf:.1}/{ka:.1} ms), exact gradients only on anchor refreshes");
                    lag_info = serde_json::json!({ "tau": tau, "ls": l, "mismatch": mis });
                    lag_tau = Some(tau);
                    seg_sets = seg_sets.with_surrogate(se, ss);
                }
                None => println!("lagged gradient: no surrogate with fewer than {nseg} segments is within 3 % → exact gradients (see tests/lagged.rs)"),
            }
        }
        if let Some(maps) = joint_maps {
            // time-resolved set per encode: map f for frame f; static set t = frame f · ne + encode e: map f
            let map_of_set = (0..fe.len()).map(|t| if temporal.is_some() { None } else { Some(t / ne) }).collect();
            fat_joint_sets = hasty_recon::setop::FatJointSets { sets: seg_sets, maps, map_of_set };
            &mut fat_joint_sets as &mut dyn SetOp
        } else {
            &mut seg_sets
        }
    };
    // per-sample host records the operator has consumed: the concomitant coefficients always (32 B / sample), the
    // coordinates and τ unless a step below needs them (voxel-basis weights, fat spectrum weights) — 11 GB of the
    // 10-minute scan otherwise held for nothing
    {
        let (keep_coords, keep_tau) = (voxel_basis.is_some(), fat_fraction.is_some() && !fat_joint);
        let mut freed = 0usize;
        // --kcrop: the whole records the segment tables were built from
        for s in std::mem::take(&mut full_sets) {
            freed += s.wq.len() * 32 + s.coords.iter().map(|c| c.len() * 4).sum::<usize>() + s.tau.len() * 8;
        }
        for s in host_sets.iter_mut() {
            freed += s.wq.len() * 32;
            s.wq = Vec::new();
            if !keep_coords && model == "seg" {
                freed += s.coords.iter().map(|c| c.len() * 4).sum::<usize>();
                s.coords = [vec![], vec![], vec![]];
            }
            if !keep_tau && model == "seg" {
                freed += s.tau.len() * 8;
                s.tau = Vec::new();
            }
        }
        if freed > (256 << 20) {
            println!("host sample records released: {:.2} GiB", freed as f64 / 1073741824.0);
        }
        // freed mid-sized buffers stay resident in the heap arenas until trimmed
        let (b, a) = hasty_recon::sysmem::trim();
        println!("memory after setup: RSS {:.1} GiB → {:.1} GiB (malloc_trim)", b as f64 / 1073741824.0, a as f64 / 1073741824.0);
    }
    let mut wsets;
    let sop: &mut dyn SetOp = match voxel_basis {
        None => sop,
        Some(pw) => {
            let w: Vec<Vec<C32>> = host_sets.iter().map(|s| hasty_recon::setop::WeightedSets::basis_weights([&s.coords[0], &s.coords[1], &s.coords[2]], pw)).collect();
            let wmin = w.iter().flatten().map(|c| c.re).fold(f32::INFINITY, f32::min);
            println!("voxel basis {}: per-sample weights, min {wmin:.3}", if pw == 1 { "rect" } else { "cubic B-spline" });
            wsets = hasty_recon::setop::WeightedSets::new(&st, sop, w).unwrap();
            &mut wsets
        }
    };
    let mut fsets;
    let sop: &mut dyn SetOp = match &fat_fraction {
        None => sop,
        // joint: the fat is in the segments already
        Some(_) if fat_joint => sop,
        Some(src) => {
            let path = if src == "truth" { PathBuf::from(&dir).join("truth_fat_fraction.npy") } else { PathBuf::from(src) };
            let ff = match &ff_est { Some(v) => v.clone(), None => maps_on_grid(&sd, &path, &grid.x_log, truth_box) };
            let ff = expand_cardiac_maps(ff, nv, nf, nf_card);
            assert!(ff.len() == nv || ff.len() == nv * nf, "--fat-fraction: {} values for {nv} voxels (or {nv} × {nf} frames, or {nv} × {nf_card} cardiac frames)", ff.len());
            let fs = hasty_qfit::waterfat::FatSpectrum::six_peak();
            let c: Vec<Vec<C32>> = host_sets.iter().map(|s| hasty_recon::setop::fat_spectrum_weights(&fs.ppm, &fs.amp, sd.b0, &s.tau)).collect();
            let nfat = ff[..nv].iter().filter(|&&f| f > 0.05).count();
            let per_frame = ff.len() == nv * nf;
            println!("water + fat: known fat fraction ({}{}), {nfat} voxels with ff > 5 %, six-peak spectrum at {:.2} T", path.display(), if per_frame { ", per frame" } else { "" }, sd.b0);
            fsets = if !per_frame {
                hasty_recon::setop::FatSets::new(&st, sop, &ff, c).unwrap()
            } else if temporal.is_some() {
                // one time-resolved set per encode: all frames concatenated (frame-major, as the stack)
                let ns_ = sop.nsets();
                hasty_recon::setop::FatSets::with_maps(&st, sop, &[ff.clone()], vec![0; ns_], c).unwrap()
            } else {
                // set t = frame f · ne + encode e: the frame's map
                let maps: Vec<Vec<f32>> = (0..nf).map(|f| ff[f * nv..(f + 1) * nv].to_vec()).collect();
                let ns_ = sop.nsets();
                hasty_recon::setop::FatSets::with_maps(&st, sop, &maps, (0..ns_).map(|t| t / ne).collect(), c).unwrap()
            };
            if fat_model == "separate" {
                fsets.shared_gridding = false;
            }
            &mut fsets
        }
    };
    let ns = sop.nsets();
    let lens: Vec<usize> = (0..ns).map(|t| sop.len_x(t)).collect();
    // image (frame f, encode e) inside the stack: (set, offset)
    let at = |f: usize, e: usize| -> (usize, usize) { if temporal.is_some() { (e, f * nv) } else { (f * ne + e, 0) } };
    // --synthetic with data staged from the store (they do not fit on the device): the residuals of model-consistent
    // data y_t = E_t x★_t − n_t, formed on the fly as E_t (x − x★_t) + n_t (hasty_recon::synth; no second store; the
    // staged measured data are not read). x★ is the --init-truth image (warm images per encode, the true velocity
    // phases on warm[0] in voxels of lumen fraction > 0.5), so --synthetic --init-truth starts at the exact solution of
    // the noise-free data term: a drift away from it is noise + prior (exact model), against model error on the
    // measured data. Noise: complex Gaussian, σ / √2 per component (σ: HASTY_NOISE_SIGMA, else the sim's noise_sigma;
    // × NOISE_SCALE), counter-based per (seed, set, coil, sample): the same in every pass and coil subset.
    let mut synth_sets;
    let sop: &mut dyn SetOp = if synthetic && staged {
        assert!(vmodel.is_none(), "--synthetic with staged data: image-space P²nP only (not --vmodel)");
        assert!(warm_full == 0, "--synthetic with staged data: no --warm-full (the reference uses the warm images)");
        let t_s = Instant::now();
        let sigma = std::env::var("HASTY_NOISE_SIGMA").ok().and_then(|v| v.parse::<f64>().ok()).filter(|v| *v > 0.0).or_else(|| sd.meta["noise_sigma"].as_f64().filter(|v| *v > 0.0)).unwrap_or(1.0)
            / 2f64.sqrt()
            * std::env::var("NOISE_SCALE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
        let seed = std::env::var("HASTY_SYNTH_SEED").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0x51);
        let mut lumen: Vec<(usize, C32)> = vec![];
        for q in 0..nv {
            if sample_truth(&sd.truth_frac, 0, &grid.x_log[q]) > 0.5 {
                lumen.push((q, warm[0][q]));
            }
        }
        let nl = lumen.len();
        let mut lvel: Vec<[f64; 3]> = Vec::with_capacity(nl * nf_card);
        for c in 0..nf_card {
            for &(q, _) in &lumen {
                lvel.push([0, 1, 2].map(|a| sample_truth(&sd.truth_vel, a + 3 * c, &grid.x_log[q])));
            }
        }
        let mut members: Vec<Vec<(usize, usize, usize)>> = vec![vec![]; ns];
        for fr in 0..nf {
            for e in 0..ne {
                let (t, o) = at(fr, e);
                members[t].push((fr, e, o));
            }
        }
        for t in 0..ns {
            assert_eq!(members[t].len() * nv, lens[t], "--synthetic: set {t} is not a whole number of (frame, encode) images");
        }
        let floor: f64 = sigma * sigma * (0..ns).map(|t| sop.len_y(t) as f64).sum::<f64>();
        // own copy of the warm images (the solver start consumes `warm` later)
        let (warm_r, dec_r) = (warm.clone(), &dec);
        let truth: hasty_recon::synth::TruthFn = Box::new(move |t: usize, out: &mut [C32]| {
            for &(fr, e, o) in &members[t] {
                out[o..o + nv].copy_from_slice(&warm_r[e]);
                let c = fr % nf_card;
                for (l, &(q, r0)) in lumen.iter().enumerate() {
                    let v = lvel[c * nl + l];
                    let z = Complex64::new(r0.re as f64, r0.im as f64) * Complex64::from_polar(1.0, dec_r.phase(e, &v) - dec_r.phase(0, &v));
                    out[o + q] = C32::new(z.re as f32, z.im as f32);
                }
            }
        });
        synth_sets = hasty_recon::synth::SynthSets::new(&dev, sop, nvc, sigma, seed, truth).unwrap();
        println!(
            "synthetic model-consistent residuals E(x − x★) + n (staged data unused): x★ = warm images with the true velocity phases in {nl} lumen voxels (lumen fraction > 0.5); noise σ {sigma:.3} per component, seed {seed}; data-term floor ½ E‖n‖² = {floor:.4e} ({:.1}s)",
            t_s.elapsed().as_secs_f64()
        );
        &mut synth_sets
    } else {
        sop
    };
    // model-consistent test data: y_t = E_t x_true,t + noise (solver + prior test without simulator physics)
    let mut ys = ys;
    if synthetic && !staged {
        assert!(!warm.is_empty(), "--synthetic needs --warm");
        assert!(sd.native_grid(), "--synthetic indexes the truth by voxel: native grid only");
        let sigma = sd.meta["noise_sigma"].as_f64().unwrap() / 2f64.sqrt() * std::env::var("NOISE_SCALE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
        let mut seed = 0x51u64;
        let mut gauss = || {
            let mut u = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
            (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
        };
        let mut xt = vec![vec![C32::new(0.0, 0.0); 0]; ns];
        for t in 0..ns {
            xt[t] = vec![C32::new(0.0, 0.0); lens[t]];
        }
        for fr in 0..nf {
            for e in 0..ne {
                let (t, o) = at(fr, e);
                for q in 0..nv {
                    let r = warm[0][q];
                    let mut z = Complex64::new(r.re as f64, r.im as f64);
                    if sd.truth_frac[q] > 0.0 {
                        let v = [0, 1, 2].map(|a| sd.truth_vel[q + nv * (a + 3 * (fr % nf_card))] as f64);
                        z *= Complex64::from_polar(1.0, dec.phase(e, &v) - dec.phase(0, &v));
                    }
                    xt[t][o + q] = C32::new(z.re as f32, z.im as f32);
                }
            }
        }
        for t in 0..ns {
            let dx = st.clone_htod(&xt[t]).unwrap();
            let mut y = st.alloc_zeros::<C32>(sop.len_y(t)).unwrap();
            sop.forward(&st, t, &dx, &mut y).unwrap();
            let mut h = st.clone_dtoh(&y).unwrap();
            for v in h.iter_mut() {
                v.re += (sigma * gauss()) as f32;
                v.im += (sigma * gauss()) as f32;
            }
            if early_host {
                ys_pin[t].as_mut_slice().unwrap()[..h.len()].copy_from_slice(&h);
            } else {
                ys[t] = st.clone_htod(&h).unwrap();
            }
        }
        println!("synthetic model-consistent data (noise sigma {:.2} per component)", sigma);
    }
    if let Some((kz, kv)) = vmodel {
        assert!(!warm.is_empty(), "--vmodel needs --warm (initialisation from the time-averaged reconstruction)");
        use hasty_recon::vmodel::{cpu::Theta, VelModel, VelSpec};
        use hasty_recon::vobj::VelProblem;
        let m1 = sd.encode_m1.clone();
        // centred encodings k̃_e = γ M1_e − mean_e: z becomes the mean-encode reference, which makes the Gauss–Newton
        // z–v cross term Σ_e k̃_e |x_e|² vanish exactly, so encode-common phase errors (B0, coil, warm-start
        // mismatch) no longer drive v. Velocity is unchanged (only encode differences carry it).
        let kraw: Vec<[f64; 3]> = m1.iter().map(|m| m.map(|x| hasty_mrsim::GAMMA * x)).collect();
        let kmean: [f64; 3] = [0, 1, 2].map(|a| kraw.iter().map(|k| k[a]).sum::<f64>() / kraw.len() as f64);
        // (opt-in: centring makes z carry the velocity-dependent mean-encode phase, which breaks its temporal
        // smoothness: 7-point 0.138 → 0.174 m/s; the z warm-up below fixes the coupling instead)
        let kenc: Vec<[f64; 3]> = if std::env::var("HASTY_VM_CENTRED").is_err() { kraw.clone() } else { kraw.iter().map(|k| [0, 1, 2].map(|a| k[a] - kmean[a])).collect() };
        let spec = VelSpec { dims: sd.dims, kenc: kenc.clone(), nf, kz, deg_z: vdeg.0, kv, deg_v: vdeg.1, poly: vpoly };
        let vm = VelModel::new(&dev, &st, spec.clone()).unwrap();
        // initialisation: time-averaged warm start, z from the reference encode, v by the hierarchical decoder (inside
        // the support), no residual phase
        let mut th = Theta::zeros(&spec);
        let warm64: Vec<Vec<Complex64>> = warm.iter().map(|w| w.iter().map(|c| Complex64::new(c.re as f64, c.im as f64)).collect()).collect();
        // residual phase: robust static-tissue fit to the warm-start phase differences; it also yields the static mask
        let sfit = vpoly.map(|d| {
            let f = hasty_recon::vinit::fit_static_phase(&warm64, &support, sd.dims, d, 0.05);
            println!("residual-phase init (robust static fit, degree {d}): σ per encode {:?} rad, static mask {} voxels", f.sigma.iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>(), f.mask.iter().filter(|&&m| m > 0.0).count());
            if !injected.is_empty() && d == 1 {
                let np = 4;
                let err: f64 = (1..ne).map(|e| (0..np).map(|p| (f.a[(e - 1) * np + p] - injected[e - 1][p]).powi(2)).sum::<f64>()).sum::<f64>().sqrt() / ((ne - 1) as f64 * 4.0).sqrt();
                println!("  init residual-phase RMS error vs injected: {err:.4} rad");
            }
            th.a.copy_from_slice(&f.a);
            f
        });
        let warm_c = match &sfit { Some(f) => hasty_recon::vinit::remove_phase(&warm64, &f.a, sd.dims, vpoly.unwrap()), None => warm64.clone() };
        for q in 0..nv {
            let z: Vec<Complex64> = (0..ne).map(|e| warm_c[e][q]).collect();
            let v = if support[q] { dec.velocity(&z) } else { [0.0; 3] };
            let z0 = z[0] * Complex64::from_polar(1.0, kenc[0][0] * v[0] + kenc[0][1] * v[1] + kenc[0][2] * v[2]);
            for k in 0..kz {
                th.z[k][q] = z0;
            }
            for k in 0..kv {
                th.v[k][q] = v;
            }
        }
        let coil: Vec<f32> = (0..nv).map(|q| (0..nvc).map(|c| { let s = smaps[c * nv + q]; s.re * s.re + s.im * s.im }).sum()).collect();
        let lens = spec.lens();
        let mut pr = VelProblem::new(&st, kern.clone(), sop, ys, vm, Some(coil)).unwrap();
        if staged {
            let sg = hasty_recon::kstage::Stager::new(&dev.ctx, kstore.clone().unwrap()).unwrap();
            println!("data staged from the store: 2 pinned staging buffers of {:.2} GiB", sg.pinned_bytes() as f64 / 2.0 / GIB);
            pr.data_from_store(sg);
        }
        if early_host {
            pr.ys = std::mem::take(&mut ys_pin).into_iter().map(hasty_recon::vobj::YData::Host).collect();
        }
        if let Some((mu, mode)) = &vanchor {
            let w: Vec<f32> = match (mode.as_str(), &sfit) {
                // oracle: inside the coil-map mask, no blood at the voxel's true position
                ("truth", _) => (0..nv).map(|q| (dil[q] && sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) < 1e-3) as u8 as f32).collect(),
                (_, Some(f)) => f.mask.clone(),
                _ => panic!("--vanchor auto needs --vpoly (the static mask comes from the residual-phase fit)"),
            };
            let wrong = (0..nv).filter(|&q| w[q] > 0.0 && sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) > 0.1).count();
            println!("static-tissue anchor μ {mu} ({mode}) on {} voxels ({wrong} of them blood by the simulator truth)", w.iter().filter(|&&x| x > 0.0).count());
            pr.set_anchor(&st, &w, *mu).unwrap();
        }
        // memory plan: what is resident now, what the run needs, where the placeable buffers go
        use hasty_recon::plan::{choose, choose_tables, probe, Need, TableNeed};
        let pb = probe(&dev.ctx, &st).unwrap();
        let (data_b, set_max) = pr.data_bytes();
        // data bytes on the device now (all, or none when kept on the host from the start)
        let data_dev: usize = pr.ys.iter().map(|y| if let hasty_recon::vobj::YData::Dev(d) = y { d.len() * 8 } else { 0 }).sum();
        let budget = vram_budget.unwrap_or(pb.total.saturating_sub(512 << 20)).min(pb.total);
        let quasi = preconds.iter().any(|p| Precond::parse(p).is_some_and(|q| q.quasi()));
        let mut need = Need { fixed: (pb.total - pb.free).saturating_sub(data_dev), stack: spec.stack_bytes(), chunk_max: lens.iter().max().unwrap() * 8, data: data_b, set_max, quasi, gradient_s: 0.0 };
        // the operator's sample tables (SegPoints): placeable too
        let (tab_b, tab_max) = pr.op.table_bytes();
        let tab = (tab_b > 0).then(|| TableNeed { bytes: tab_b, set_max: tab_max, device_now: if pr.op.tables_on_host() { 2 * tab_max } else { tab_b } });
        let data_fixed = (early_host || staged).then_some(true);
        let plan_with = |need: &Need, budget: usize, data: Option<bool>, tables: Option<bool>| -> Result<hasty_recon::plan::Plan, hasty_recon::Error> {
            match &tab {
                Some(tb) => choose_tables(need, tb, &pb, budget, data, tables),
                None => choose(need, &pb, budget, data),
            }
        };
        let forced = |data: bool, prev: bool, w: hasty_recon::p2np::WMode| -> hasty_recon::plan::Plan {
            let mut p = plan_with(&need, usize::MAX, Some(data), Some(data)).unwrap();
            p.prev_on_host = prev;
            p.w_mode = w;
            // forced placement: report the real budget, not the unlimited one used for the choice
            p.budget = budget;
            p
        };
        let mut plan = match plan_mode.as_str() {
            "device" => forced(false, false, hasty_recon::p2np::WMode::Device),
            "host" => forced(true, true, hasty_recon::p2np::WMode::Host),
            _ => plan_with(&need, budget, data_fixed, None).unwrap_or_else(|e| panic!("{e}")),
        };
        if tab.is_some() && plan.tables_on_host != pr.op.tables_on_host() {
            let freed = pr.op.place_tables(&dev.ctx, &st, plan.tables_on_host).unwrap();
            println!("sample tables moved to {} ({:+.2} GiB device)", if plan.tables_on_host { "pinned host memory (streamed)" } else { "the device" }, -(freed as f64) / 1073741824.0);
        }
        println!("device: {:.2} GiB total, {:.2} GiB free; pinned bandwidth h2d {:.1} / d2h {:.1} GB/s; budget {:.2} GiB", pb.total as f64 / 1073741824.0, pb.free as f64 / 1073741824.0, pb.h2d / 1e9, pb.d2h / 1e9, budget as f64 / 1073741824.0);
        if plan.data_on_host {
            let freed = pr.data_to_host(&dev.ctx, &st).unwrap();
            println!("k-space data moved to pinned host memory ({:.2} GiB freed)", freed as f64 / 1073741824.0);
        }
        print!("{}", plan.describe());
        // VENC continuation: the low-VENC encodes (all but the least sensitive group) are left out of the first stage
        let kn: Vec<f64> = (0..ne).map(|e| (0..3).map(|a| (kenc[e][a] - kenc[0][a]).powi(2)).sum::<f64>().sqrt()).collect();
        let kmin = kn.iter().copied().filter(|&x| x > 0.0).fold(f64::INFINITY, f64::min);
        let low: Vec<bool> = kn.iter().map(|&x| x > kmin * (1.0 + 1e-6)).collect();
        let vtag = |pc: &str, kap: f64| format!("{model}_{pc}_vm{kz}x{kv}_d{}{}{}{}{}{}{dfw_tag}", vdeg.0, vdeg.1, vpoly.map_or(String::new(), |d| format!("_p{d}")), if vcont > 0 { format!("_c{vcont}") } else { String::new() }, if vzwarm > 0 { format!("_z{vzwarm}") } else { String::new() }, if kap > 0.0 { format!("_llr{kap}") } else { String::new() });
        let runs: Vec<String> = preconds.iter().flat_map(|p| vkappas.iter().map(move |&k| (p, k))).map(|(p, k)| vtag(p, k)).collect();
        let mut info = serde_json::json!({
            "cmdline": cmdline, "sim_dir": dir, "model": model, "phys": phys, "gnl": gnl, "iters": iters, "warm_iters": warm_iters, "vcoils": nvc, "dims": sd.dims, "vox": sd.vox, "native_dims": sd.tdims,
            "n_frames": nf, "n_enc": ne, "venc": sd.venc, "venc_range": dec.range(), "encoding": sd.meta["encoding"], "runs": runs,
            "prior": if vkappas.iter().any(|&k| k > 0.0) { "llr(z, p)" } else { "none" }, "vkappas": vkappas, "vmodel": {"kz": kz, "kv": kv, "deg": [vdeg.0, vdeg.1], "poly": vpoly, "vcont": vcont, "low_venc_encodes": low, "anchor": vanchor.as_ref().map(|(m, s)| format!("{m},{s}"))}, "injected_phase": injected, "setup_s": t0.elapsed().as_secs_f64(),
        });
        info["pyramid"] = pyr_info.clone();
        info["segments"] = serde_json::json!({ "basis": seg_basis_arg, "weight": seg_weight }); info["respiration"] = resp_info.clone(); info["prior_detail"] = serde_json::json!({ "prior": prior_kind, "wavelet": { "thr": wav_thr, "levels": wav_levels, "mix": wav_mix } });
        info["prior_detail"]["dfw"] = serde_json::json!(dfw_rule.map(|r| serde_json::json!({ "rule": format!("{r:?}"), "levels": dfw_levels, "mix": dfw_mix })));
        assert!(static_lock.is_none(), "--static-lock: complex-image models only (not --vmodel)");
        assert!(air.is_none(), "--air: complex-image models only (not --vmodel)");
    std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
        let rt = transpose(&sd.rot);
        for (pc, kap) in preconds.iter().flat_map(|p| vkappas.iter().map(move |&k| (p, k))) {
            let tag = vtag(pc, kap);
            let precond = Precond::parse(pc).unwrap_or(Precond::Zmshr1M);
            let with_metric = precond.metric();
            let resuming = out.join(format!("state_{tag}_s0.hv")).exists() || out.join(format!("state_{tag}_s1.hv")).exists() || out.join(format!("vdone_{tag}_s0")).exists();
            if !resuming && out.join(format!("recon_{tag}.npy")).exists() {
                println!("  {tag}: finished in an earlier run, skipped (--fresh to redo)");
                continue;
            }
            let mut x = Stack { v: th.pack(&spec).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
            pr.reset();
            let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(out.join(format!("metrics_{tag}.csv"))).unwrap();
            if !resuming {
                use std::io::Write;
                csv.set_len(0).unwrap();
                writeln!(csv, "iter,f,rmse,rel,wrap_frac,seconds,{}", (0..nf).map(|f| format!("frame{f}")).collect::<Vec<_>>().join(",")).unwrap();
            }
            let tr = Instant::now();
            let wv = pr.model.wv.clone();
            // velocity of frame fr at voxel q from host copies of the v chunks
            let vel_of = |vh: &[Vec<f32>], fr: usize, q: usize| -> [f64; 3] {
                let mut v = [0.0; 3];
                for k in 0..kv {
                    let w = wv[fr * kv + k];
                    if w != 0.0 {
                        for a in 0..3 {
                            v[a] += w * vh[k][a * nv + q] as f64;
                        }
                    }
                }
                if gnl {
                    v = mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &mat_vec(&sd.rot, &v)));
                }
                v
            };
            let mut stages: Vec<(usize, bool)> = if vcont > 0 && low.iter().any(|&b| b) { vec![(vcont.min(iters), true), (iters.saturating_sub(vcont), false)] } else { vec![(iters, false)] };
            // z warm-up as an extra first stage (all encodes, v frozen); stage indices shift by one
            let zstage = vzwarm > 0;
            if zstage {
                stages.insert(0, (vzwarm, false));
            }
            let mut k_off = 0;
            // the automatic anchor waits for pulsatility: stage 0 without it, refined mask from stage 1 on
            let anchor_auto = matches!(&vanchor, Some((_, m)) if m == "auto") && stages.len() > 1;
            let fit_mask: Option<Vec<f32>> = sfit.as_ref().map(|f| f.mask.clone());
            for (si, &(n_it, only_hi)) in stages.iter().enumerate() {
                let stag = format!("{tag}_s{si}");
                pr.freeze_v = zstage && si == 0;
                let si = if zstage { si.saturating_sub(1) } else { si }; // continuation / anchor logic sees the original stages
                if anchor_auto && si == 0 {
                    pr.clear_anchor();
                }
                if anchor_auto && si == 1 {
                    // static = accepted by the robust fit, low temporal variation of v over the nodes, and not
                    // within 2 voxels of a pulsatile voxel
                    let vh: Vec<Vec<f32>> = (0..kv).map(|kk| st.clone_dtoh(&x.v[spec.chunk_v(kk)]).unwrap().iter().flat_map(|c| [c.re, c.im]).collect()).collect();
                    let std: Vec<f64> = (0..nv).map(|q| {
                        let mean: [f64; 3] = [0, 1, 2].map(|a| (0..kv).map(|kk| vh[kk][a * nv + q] as f64).sum::<f64>() / kv as f64);
                        ((0..kv).map(|kk| (0..3).map(|a| (vh[kk][a * nv + q] as f64 - mean[a]).powi(2)).sum::<f64>()).sum::<f64>() / kv as f64).sqrt()
                    }).collect();
                    let fm = fit_mask.as_ref().unwrap();
                    let mut sv: Vec<f64> = (0..nv).filter(|&q| fm[q] > 0.0).map(|q| std[q]).collect();
                    sv.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let tau = (vanchor_refine.0 * sv.get(sv.len() / 2).copied().unwrap_or(0.0)).max(0.02);
                    let mut puls: Vec<bool> = (0..nv).map(|q| support[q] && std[q] > tau).collect();
                    let d = sd.dims;
                    for _ in 0..vanchor_refine.1 {
                        let prev = puls.clone();
                        for q in 0..nv {
                            if prev[q] {
                                continue;
                            }
                            let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
                            let nb = [(i > 0).then(|| q - 1), (i + 1 < d[0]).then(|| q + 1), (j > 0).then(|| q - d[0]), (j + 1 < d[1]).then(|| q + d[0]), (k > 0).then(|| q - d[0] * d[1]), (k + 1 < d[2]).then(|| q + d[0] * d[1])];
                            puls[q] = nb.iter().flatten().any(|&n| prev[n]);
                        }
                    }
                    let w: Vec<f32> = (0..nv).map(|q| (fm[q] > 0.0 && !puls[q]) as u8 as f32).collect();
                    let wrong = (0..nv).filter(|&q| w[q] > 0.0 && sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) > 0.1).count();
                    println!("  anchor refined by pulsatility (std of v over nodes > {tau:.3} m/s, dilated {}):", vanchor_refine.1);
                    println!("    {} voxels ({wrong} blood by the simulator truth)", w.iter().filter(|&&x| x > 0.0).count());
                    pr.set_anchor(&st, &w, vanchor.as_ref().unwrap().0).unwrap();
                }
                let done = out.join(format!("vdone_{stag}"));
                // set t holds encode t % E (per-frame sets t = f·E + e; temporal sets t = e)
                for t in 0..pr.weights.len() {
                    pr.weights[t] = if only_hi && low[t % ne] { 0.0 } else { 1.0 };
                }
                if done.exists() {
                    if let Some(v) = ctl.load(&format!("{stag}_final")).unwrap() {
                        x = hasty_recon::checkpoint::value_stack(&st, &v, &lens).unwrap();
                    }
                    k_off += n_it;
                    continue;
                }
                let tl = Instant::now();
                let lam = pr.lambda_max(&st, &x, with_metric, 12).unwrap();
                st.synchronize().unwrap();
                println!("  {tag} stage {si} ({}): {n_it} iterations, λ_max({}) = {lam:.3e}", if only_hi { "reference + high-VENC encodes" } else { "all encodes" }, if with_metric { "M H" } else { "H" });
                if vanchor.is_some() {
                    println!("    anchor λ {:.3e} (n̂ {:.3e})", pr.anchor_lam, pr.nhat);
                }
                // refine the state placement with the measured cost of one gradient (≈ one Hessian-vector product)
                if plan_mode == "auto" && need.gradient_s == 0.0 {
                    need.gradient_s = tl.elapsed().as_secs_f64() / 12.0;
                    plan = plan_with(&need, budget, Some(plan.data_on_host), Some(plan.tables_on_host)).unwrap_or_else(|e| panic!("{e}"));
                    print!("  measured {:.3} s per gradient; {}", need.gradient_s, plan.describe());
                }
                let sopts = plan.state_opts(&dev.ctx);
                let mut report = |k: usize, xs: &Stack, f: f64| {
                    let k = if k == usize::MAX { k_off } else { k + 1 + k_off };
                    if k % every != 0 && k != iters && k != 0 {
                        return;
                    }
                    let vh: Vec<Vec<f32>> = (0..kv).map(|kk| st.clone_dtoh(&xs.v[spec.chunk_v(kk)]).unwrap().iter().flat_map(|c| [c.re, c.im]).collect()).collect();
                    let (mut e2, mut t2, mut n, mut wrap) = (0.0f64, 0.0f64, 0usize, 0usize);
                    let mut per_frame = Vec::with_capacity(nf);
                    for fr in 0..nf {
                        let (mut fe2, mut fn_) = (0.0, 0usize);
                        for &q in &lumen {
                            let v = vel_of(&vh, fr, q);
                            for a in 0..3 {
                                let t = sd.sample_log(&sd.truth_vel, a + 3 * (fr % nf_card), &grid.x_log[q]);
                                let d = v[a] - t;
                                e2 += d * d;
                                t2 += t * t;
                                fe2 += d * d;
                                wrap += (d.abs() > 0.8 * sd.venc.min(dec.range()[a])) as usize;
                            }
                            n += 1;
                            fn_ += 1;
                        }
                        per_frame.push((fe2 / fn_.max(1) as f64).sqrt());
                    }
                    let wf = wrap as f64 / (3 * n).max(1) as f64;
                    {
                        use std::io::Write;
                        writeln!(csv, "{k},{f:.6e},{:.5},{:.5},{wf:.5},{:.1},{}", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), tr.elapsed().as_secs_f64(), per_frame.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",")).unwrap();
                    }
                    println!("  {pc:<7} vm it {k:>3}: f = {f:.4e}, velocity RMSE {:.3} m/s (rel {:.3}), wrapped {:.2} %, frames {}, {:.0}s", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), 100.0 * wf, per_frame.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "), tr.elapsed().as_secs_f64());
                };
                // prior: LLR on the z nodes and, separately, on p = |z̄|·v (3 K_v real images), then v = p / |z̄|
                let (mut llr_z, mut llr_p) = (Llr::new(sd.dims, block, kap, support.clone()), Llr::new(sd.dims, block, kap, support.clone()));
                let voff: Vec<usize> = (0..kv).map(|k| kz * nv + k * (3 * nv).div_ceil(2)).collect();
                let mut vden = |k: usize, h: &mut Vec<C32>| {
                    if kap <= 0.0 {
                        return;
                    }
                    let it = (k + k_off) as u64;
                    let nonfinite = |h: &[C32]| h.iter().filter(|c| !c.re.is_finite() || !c.im.is_finite()).count();
                    let n0 = nonfinite(h);
                    llr_z.set_iteration(it);
                    llr_z.apply(&mut h[..kz * nv], kz);
                    let n1 = nonfinite(&h[..kz * nv]);
                    let zbar: Vec<f32> = (0..nv).map(|q| (0..kz).map(|kk| { let c = h[kk * nv + q]; (c.re * c.re + c.im * c.im).sqrt() }).sum::<f32>() / kz as f32).collect();
                    let zmax = zbar.iter().cloned().fold(0f32, f32::max);
                    let mut pimg = vec![C32::new(0.0, 0.0); 3 * kv * nv];
                    for kk in 0..kv {
                        let f: Vec<f32> = h[voff[kk]..voff[kk] + (3 * nv).div_ceil(2)].iter().flat_map(|c| [c.re, c.im]).collect();
                        for c in 0..3 {
                            for q in 0..nv {
                                pimg[(3 * kk + c) * nv + q] = C32::new(zbar[q] * f[c * nv + q], 0.0);
                            }
                        }
                    }
                    let n2 = nonfinite(&pimg);
                    llr_p.set_iteration(it);
                    llr_p.apply(&mut pimg, 3 * kv);
                    let n3 = nonfinite(&pimg);
                    if n0 + n1 + n2 + n3 > 0 {
                        eprintln!("    vden it {k}: non-finite before {n0}, after LLR(z) {n1}, p {n2}, after LLR(p) {n3}");
                    }
                    for kk in 0..kv {
                        let mut f: Vec<f32> = h[voff[kk]..voff[kk] + (3 * nv).div_ceil(2)].iter().flat_map(|c| [c.re, c.im]).collect();
                        for c in 0..3 {
                            for q in 0..nv {
                                if zbar[q] > 1e-3 * zmax {
                                    f[c * nv + q] = pimg[(3 * kk + c) * nv + q].re / zbar[q];
                                }
                            }
                        }
                        for (i, pair) in f.chunks(2).enumerate() {
                            h[voff[kk] + i] = C32::new(pair[0], *pair.get(1).unwrap_or(&0.0));
                        }
                    }
                };
                // divergence-free wavelets on every v node after the prior, weighted by |z̄| (velocity noise ∝ 1 / |m|)
                let mut dfw_v = dfw_rule.map(|r| hasty_recon::dfwavelet::DfWavelet::new(sd.dims, dfw_levels, sd.vox, r));
                let mut vden_all = |k: usize, h: &mut Vec<C32>| {
                    vden(k, h);
                    let Some(dfw) = dfw_v.as_mut() else { return };
                    let t = Instant::now();
                    dfw.set_iteration((k + k_off) as u64);
                    let zbar: Vec<f32> = (0..nv).map(|q| (0..kz).map(|kk| { let c = h[kk * nv + q]; (c.re * c.re + c.im * c.im).sqrt() }).sum::<f32>() / kz as f32).collect();
                    let b = dfw_mix as f32;
                    let mut msg = String::new();
                    for kk in 0..kv {
                        let mut f: Vec<f32> = h[voff[kk]..voff[kk] + (3 * nv).div_ceil(2)].iter().flat_map(|c| [c.re, c.im]).collect();
                        let f0 = f.clone();
                        let st = dfw.apply(&mut f[..3 * nv], Some(&zbar));
                        msg += &format!(" node {kk}: λ df {:.2} n {:.2}, kept df {:.1} % n {:.1} %;", st.lam_df, st.lam_n, 100.0 * st.kept_df, 100.0 * st.kept_n);
                        for (i, pair) in f.chunks(2).enumerate() {
                            let (a, a0) = (pair[0], f0[2 * i]);
                            let (c, c0) = (*pair.get(1).unwrap_or(&0.0), *f0.get(2 * i + 1).unwrap_or(&0.0));
                            h[voff[kk] + i] = C32::new(a0 + b * (a - a0), c0 + b * (c - c0));
                        }
                    }
                    if k % 10 == 0 {
                        println!("    DFW on v ({:.2}s):{msg}", t.elapsed().as_secs_f64());
                    }
                };
                if let Outcome::Stopped(k) = hasty_recon::p2np::run_objective_with(&mut pr, &st, &mut x, precond, 1.0 / lam, n_it, &mut vden_all, &mut report, Some((&ctl, &stag)), &sopts).unwrap() {
                    println!("stopped {stag} at iteration {k} / {n_it}: state saved in {}, rerun the same command to resume", out.display());
                    std::process::exit(0);
                }
                ctl.save(&format!("{stag}_final"), &hasty_recon::checkpoint::stack_value(&st, &x).unwrap()).unwrap();
                std::fs::write(&done, b"").unwrap();
                k_off += n_it;
            }
            // outputs: velocities and model images per frame, parameters
            let vh: Vec<Vec<f32>> = (0..kv).map(|kk| st.clone_dtoh(&x.v[spec.chunk_v(kk)]).unwrap().iter().flat_map(|c| [c.re, c.im]).collect()).collect();
            let mut vout = vec![0f32; nv * 3 * nf];
            let mut xout = vec![C32::new(0.0, 0.0); nv * ne * nf];
            let mut xi = st.alloc_zeros::<C32>(nv).unwrap();
            for fr in 0..nf {
                for q in 0..nv {
                    let v = vel_of(&vh, fr, q);
                    for a in 0..3 {
                        vout[q + nv * (a + 3 * fr)] = v[a] as f32;
                    }
                }
                for e in 0..ne {
                    pr.model.eval(&st, &x, e, fr, &mut xi).unwrap();
                    xout[nv * (e + ne * fr)..nv * (e + ne * fr + 1)].copy_from_slice(&st.clone_dtoh(&xi).unwrap());
                }
            }
            save(&out.join(format!("vel_{tag}.npy")), "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 3, nf], f4(vout));
            save(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], ne, nf], c8(&xout));
            if spec.np() > 0 {
                let af: Vec<f32> = st.clone_dtoh(&x.v[spec.chunk_a()]).unwrap().iter().flat_map(|c| [c.re, c.im]).collect();
                let np = spec.np();
                let rec: Vec<Vec<f64>> = (1..ne).map(|e| (0..np).map(|p| (af[(e - 1) * np + p] as f64 * 1000.0).round() / 1000.0).collect()).collect();
                println!("  recovered residual phase (rad, Legendre order of vmodel::poly_exps): {rec:?}");
                if !injected.is_empty() && vpoly == Some(1) {
                    let err: f64 = (1..ne).map(|e| (0..4).map(|p| (af[(e - 1) * np + p] as f64 - injected[e - 1][p]).powi(2)).sum::<f64>()).sum::<f64>().sqrt() / ((ne - 1) as f64 * 4.0).sqrt();
                    println!("  residual-phase RMS error vs injected: {err:.4} rad");
                }
            }
            let params = hasty_recon::checkpoint::stack_value(&st, &x).unwrap();
            hasty_value::write(&params, &hasty_value::Options::default(), &mut std::io::BufWriter::new(std::fs::File::create(out.join(format!("vparams_{tag}.hv"))).unwrap())).unwrap();
            for si in 0..stages.len() {
                ctl.clear(&format!("{tag}_s{si}_final"));
                let _ = std::fs::remove_file(out.join(format!("vdone_{tag}_s{si}")));
            }
        }
        let tools = hasty_site::tool("figs/run_figs.py");
        let py = hasty_site::python();
        match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
            Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
            r => println!("figure script failed ({r:?}); run: {} -I {} {}", py.display(), tools.display(), out.display()),
        }
        return;
    }
    if let Ok((free, total)) = dev.ctx.mem_get_info() {
        println!("device memory before the solver: {:.2} GiB free of {:.2} GiB", free as f64 / GIB, total as f64 / GIB);
    }
    let mut p = if staged {
        let sg = hasty_recon::kstage::Stager::new(&dev.ctx, kstore.clone().unwrap()).unwrap();
        println!("data staged from the store: {} pinned staging buffers of {:.2} GiB", 2, sg.pinned_bytes() as f64 / 2.0 / GIB);
        P2np::from_store(sop, sg, circ, &st).unwrap().with_circ_map_opt(circ_map.clone())
    } else {
        P2np::new(sop, ys.iter().collect(), circ, &st).unwrap().with_circ_map_opt(circ_map.clone())
    };
    let mut lmax = p.step_norm(&st, false, 12).unwrap();
    let mut lmax_c = if need_circ { p.step_norm(&st, true, 12).unwrap() } else { 1.0 };
    if let Some(cs) = coil_sampling.clone() {
        // the step must be stable for every subset: α from the largest λ_max over a few draws
        p = p.with_sampling(cs.clone());
        let (ls, lsc) = (p.step_norm_sampled(&st, false, 12, 4).unwrap(), if need_circ { p.step_norm_sampled(&st, true, 12, 4).unwrap() } else { 1.0 });
        println!("coil sampling: keep {} + draw {} of {} (hold {}, last {} full): λ_max(sampled) / λ_max(full) = {:.3} (M-weighted {:.3})", cs.keep, cs.draw, nvc - cs.keep, cs.hold, cs.full_last, ls / lmax, lsc / lmax_c);
        lmax = lmax.max(ls);
        lmax_c = lmax_c.max(lsc);
    }
    if let Some(tau) = lag_tau {
        p = p.with_lagged(tau);
    }
    println!("lambda_max(N) = {lmax:.3e}, lambda_max(M N) = {lmax_c:.3e}, setup {:.0}s", t0.elapsed().as_secs_f64());
    // physics-model warm start (--warm-full): pooled PCG per encode through the full objective
    let warm: Vec<Vec<C32>> = if warm_full > 0 {
        assert!(temporal.is_none(), "--warm-full needs one frame per set (no --temporal)");
        let group_of: Vec<usize> = (0..fe.len()).map(enc_of).collect();
        let done: Vec<String> = (0..ne).map(|e| format!("warm_full_done_{e}")).collect();
        let loaded: Vec<Option<Vec<C32>>> = done.iter().map(|d| ctl.load(d).unwrap().map(|v| st.clone_dtoh(&value_dev(&st, &v, nv).unwrap()).unwrap())).collect();
        let xw: Vec<Vec<C32>> = if loaded.iter().all(|v| v.is_some()) {
            println!("physics-model warm start: loaded ({} encodes)", ne);
            loaded.into_iter().map(|v| v.unwrap()).collect()
        } else {
            let mut xw: Vec<Vec<C32>> = if warm.is_empty() { vec![vec![C32::new(0.0, 0.0); nv]; ne] } else { warm.clone() };
            let tw = Instant::now();
            let lens_w = p.lens();
            assert!(lens_w.iter().all(|&l| l == nv), "--warm-full: one image per set");
            let oc = hasty_recon::hoststack::pooled_pcg_host_with(&mut p, &st, &group_of, &mut xw, warm_full, need_circ, Some((&ctl, "warm_full")), 2, pooled_grad).unwrap();
            if let Outcome::Stopped(k) = oc {
                println!("stopped in the physics-model warm start (PCG iteration {k}): state saved, rerun the same command to resume");
                std::process::exit(0);
            }
            for (e, d) in done.iter().enumerate() {
                ctl.save(d, &hasty_recon::checkpoint::dev_value(&st, &st.clone_htod(&xw[e]).unwrap()).unwrap()).unwrap();
            }
            println!("physics-model warm start: {warm_full} pooled PCG iterations per encode{} ({:.0}s)", if need_circ { " (per-encode circulant)" } else { "" }, tw.elapsed().as_secs_f64());
            xw
        };
        save(&out.join("warm_full.npy"), "<c8", &dims4(ne), c8(&xw.concat()));
        // diagnostic: time-averaged velocity of the physics warm start vs the time-averaged truth over the lumen
        let (mut e2, mut n) = (0.0f64, 0usize);
        for &q in &lumen {
            let z: Vec<Complex64> = (0..ne).map(|e| Complex64::new(xw[e][q].re as f64, xw[e][q].im as f64)).collect();
            let vq = dec.velocity(&z);
            for a in 0..3 {
                let tm: f64 = (0..nf_card).map(|f| sd.sample_log(&sd.truth_vel, a + 3 * f, &grid.x_log[q])).sum::<f64>() / nf_card as f64;
                e2 += (vq[a] - tm).powi(2);
            }
            n += 1;
        }
        if n > 0 {
            println!("physics-model warm-start velocity in the lumen: RMSE {:.3} m/s vs the time-averaged truth", (e2 / n as f64).sqrt());
        }
        xw
    } else {
        warm
    };
    // temporal pyramid (--tpyramid): pooled PCG levels, coarse to fine; the last level's group images initialise the
    // solver per set (group_of of the last level, its images)
    let tpyr_init: Option<(Vec<usize>, Vec<Vec<C32>>)> = tpyramid.as_ref().map(|levels| {
        assert!(temporal.is_none(), "--tpyramid needs one frame per set (no --temporal)");
        assert!(p.lens().iter().all(|&l| l == nv), "--tpyramid: one image per set");
        let frame_enc: Vec<(usize, usize)> = fe.iter().map(|&(f, e)| (f.unwrap(), e)).collect();
        let level_groups = |rg: usize, cg: usize| tpyr_groups(&frame_enc, ne, nf_card, resp_bins, rg, cg);
        let done = |li: usize, rg: usize, cg: usize| format!("tpyr_done_{li}_{rg}x{cg}");
        // resume: the finest level already finished
        let found = (0..levels.len()).rev().find_map(|li| { let (rg, cg, _) = levels[li]; ctl.load(&done(li, rg, cg)).unwrap().map(|v| (li, v)) });
        let last_done = found.as_ref().map(|f| f.0);
        let mut prev: Option<(Vec<usize>, Vec<Vec<C32>>)> = None;
        if let Some((li, v)) = found {
            let (rg, cg, _) = levels[li];
            let (ng, gof) = level_groups(rg, cg);
            let mut h = hasty_recon::hoststack::HostStack::zeros(&vec![nv; ng]);
            h.load_value(&v).unwrap();
            println!("temporal pyramid: level {} ({rg}x{cg}, {ng} groups) loaded", li + 1);
            prev = Some((gof, (0..ng).map(|g| h.chunk(g).to_vec()).collect()));
        }
        let tw = Instant::now();
        for (li, &(rg, cg, it)) in levels.iter().enumerate().skip(last_done.map_or(0, |l| l + 1)) {
            let (ng, gof) = level_groups(rg, cg);
            // initial images: the previous level (mean over each group's sets of their previous group images), else
            // the plain warm start per encode, else zero
            let mut xg: Vec<Vec<C32>> = match &prev {
                Some((pg, px)) => tpyr_prolong(pg, px, &gof, ng),
                None => (0..ng).map(|g| match warm.is_empty() {
                    true => vec![C32::new(0.0, 0.0); nv],
                    false => warm[enc_of((0..fe.len()).find(|&t| gof[t] == g).unwrap())].clone(),
                }).collect(),
            };
            let sizes: Vec<usize> = (0..ng).map(|g| gof.iter().filter(|&&h| h == g).count()).collect();
            println!("temporal pyramid level {} / {}: {rg}x{cg} → {ng} groups of {}..{} sets, {it} pooled PCG iterations", li + 1, levels.len(), sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
            if it > 0 {
                let tl = Instant::now();
                let oc = hasty_recon::hoststack::pooled_pcg_host_with(&mut p, &st, &gof, &mut xg, it, need_circ, Some((&ctl, &format!("tpyr_{li}"))), 2, pooled_grad).unwrap();
                if let Outcome::Stopped(k) = oc {
                    println!("stopped in temporal pyramid level {} (PCG iteration {k}): state saved, rerun the same command to resume", li + 1);
                    std::process::exit(0);
                }
                println!("temporal pyramid level {}: {:.0}s", li + 1, tl.elapsed().as_secs_f64());
            }
            // this level's result for a resume; the previous level's is no longer needed
            let mut h = hasty_recon::hoststack::HostStack::zeros(&vec![nv; ng]);
            for g in 0..ng {
                h.chunk_mut(g).copy_from_slice(&xg[g]);
            }
            ctl.save(&done(li, rg, cg), &h.to_value().unwrap()).unwrap();
            if li > 0 {
                let (prg, pcg, _) = levels[li - 1];
                ctl.clear(&done(li - 1, prg, pcg));
            }
            prev = Some((gof, xg));
        }
        let (gof, xg) = prev.unwrap();
        // the last level per encode (time average over its groups, weighted by sets): array + lumen diagnostic
        let enc: Vec<usize> = (0..fe.len()).map(enc_of).collect();
        let mean_e = tpyr_prolong(&gof, &xg, &enc, ne);
        save(&out.join("tpyramid_mean.npy"), "<c8", &dims4(ne), c8(&mean_e.concat()));
        let (mut e2, mut n) = (0.0f64, 0usize);
        for &q in &lumen {
            let z: Vec<Complex64> = (0..ne).map(|e| Complex64::new(mean_e[e][q].re as f64, mean_e[e][q].im as f64)).collect();
            let vq = dec.velocity(&z);
            for a in 0..3 {
                let tm: f64 = (0..nf_card).map(|f| sd.sample_log(&sd.truth_vel, a + 3 * f, &grid.x_log[q])).sum::<f64>() / nf_card as f64;
                e2 += (vq[a] - tm).powi(2);
            }
            n += 1;
        }
        println!("temporal pyramid: {} levels ({:.0}s); time-averaged velocity in the lumen: RMSE {:.3} m/s vs the time-averaged truth", levels.len(), tw.elapsed().as_secs_f64(), (e2 / n.max(1) as f64).sqrt());
        (gof, xg)
    });
    let ttag = temporal.map_or(String::new(), |b| format!("_t{b}"));
    let runs: Vec<String> = preconds.iter().flat_map(|p| kappas.iter().map(|k| (p.clone(), *k)).collect::<Vec<_>>()).map(|(p, k)| format!("{model}{ttag}_{p}{}_k{k}{}", if learned.is_some() { "_learned" } else { "" }, llr_rank.map_or(String::new(), |r| format!("_r{r}")) + if llr_hard { "_h" } else { "" } + &red_tag + &ga_tag)).collect();
    let mut info = serde_json::json!({
        "cmdline": cmdline, "sim_dir": dir, "model": model, "temporal": temporal, "phys": phys, "gnl": gnl, "nseg": nseg,
        "order": order, "conc_fit": conc_fit, "fat_model": fat_fraction.as_ref().map(|_| fat_model.clone()), "eps": eps, "iters": iters, "warm_iters": warm_iters, "block": block, "vcoils": nvc, "dims": sd.dims, "vox": sd.vox, "native_dims": sd.tdims,
        "n_frames": nf, "n_enc": ne, "venc": sd.venc, "venc_range": dec.range(), "encoding": sd.meta["encoding"], "lambda_max": lmax, "lambda_max_mn": lmax_c, "setup_s": t0.elapsed().as_secs_f64(),
        "noise_sigma_env": std::env::var("HASTY_NOISE_SIGMA").ok(), "runs": runs,
        "prior": if learned.is_some() { "learned" } else { "llr" }, "prior_mode": prior_mode, "red_rho": red.then_some(red_rho), "red_shifts": red.then_some(red_shifts), "tsub": tsub_spec, "learned_dir": learned_dir, "tile": tile, "keep_frames": keep_frames, "learned_prec": learned_prec,
        "field": field_spec, "r2s": r2s_spec, "smaps": smaps_spec, "coil_mask": coil_mask_mode, "coils": coil_info, "estimates": diag, "kdata": kstore_info, "lagged": lag_info, "vdecode": vdec_info,
    });
    info["pyramid"] = pyr_info.clone();
    info["segments"] = serde_json::json!({ "basis": seg_basis_arg, "weight": seg_weight }); info["respiration"] = resp_info.clone(); info["prior_detail"] = serde_json::json!({ "prior": prior_kind, "wavelet": { "thr": wav_thr, "levels": wav_levels, "mix": wav_mix } });
    info["prior_detail"]["dfw"] = serde_json::json!(dfw_rule.map(|r| serde_json::json!({ "rule": format!("{r:?}"), "levels": dfw_levels, "mix": dfw_mix })));
    info["prior_detail"]["static_lock"] = serde_json::json!(static_lock.as_ref().map(|(o, it0, ev)| serde_json::json!({ "opts": format!("{o:?}"), "it0": it0, "every": ev })));
    info["prior_detail"]["air"] = serde_json::json!(air.as_ref().map(|(o, it0, ev)| serde_json::json!({ "opts": format!("{o:?}"), "it0": if *it0 == usize::MAX { None } else { Some(it0) }, "every": if *ev == usize::MAX { None } else { Some(ev) }, "init": air_init })));
    std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
    let rt = transpose(&sd.rot);
    for (pc, kappa) in preconds.iter().flat_map(|p| kappas.iter().map(move |&k| (p, k))) {
        let precond = Precond::parse(pc).unwrap_or(Precond::Zmshr1);
        let alpha = if precond.metric() { 1.0 / lmax_c } else { 1.0 / lmax };
        let stack_bytes = lens.iter().sum::<usize>() * 8;
        let host_state = match state_arg.as_str() {
            "host" => true,
            "device" => false,
            _ => {
                let (free, _) = dev.ctx.mem_get_info().unwrap();
                3 * stack_bytes > free / 2
            }
        };
        if host_state && (lag_tau.is_some() || coil_sampling.is_some()) {
            panic!("--state host with --lagged / --coil-sampling: not yet (they need device stacks)");
        }
        let mut x = if host_state { Stack { v: vec![] } } else { Stack::zeros_lens(&st, &lens).unwrap() };
        let mut xh = if host_state { hasty_recon::hoststack::HostStack::zeros(&lens) } else { hasty_recon::hoststack::HostStack::default() };
        if pc == &preconds[0] && kappa == kappas[0] {
            println!("solver state: {} ({:.2} GiB per stack, {} sets)", if host_state { "host memory, sets streamed through two device slots" } else { "device" }, stack_bytes as f64 / GIB, lens.len());
        }
        if let Some((gof, xg)) = &tpyr_init {
            // the temporal pyramid's last level: each set from its group's image
            for fr in 0..nf {
                for e in 0..ne {
                    let (t, o) = at(fr, e);
                    if host_state {
                        xh.chunk_mut(t)[o..o + nv].copy_from_slice(&xg[gof[t]]);
                    } else {
                        st.memcpy_htod(&xg[gof[t]], &mut x.v[t].slice_mut(o..o + nv)).unwrap();
                    }
                }
            }
        } else if !warm.is_empty() {
            for fr in 0..nf {
                for e in 0..ne {
                    let (t, o) = at(fr, e);
                    if host_state {
                        xh.chunk_mut(t)[o..o + nv].copy_from_slice(&warm[e]);
                    } else {
                        st.memcpy_htod(&warm[e], &mut x.v[t].slice_mut(o..o + nv)).unwrap();
                    }
                }
            }
        }
        let mut llr = Llr::new(sd.dims, block, kappa, support.clone());
        llr.rank = llr_rank;
        llr.hard = llr_hard;
        let mut llr_gpu = (llr_devices != "cpu" && prior_kind == "llr" && kappa > 0.0 && learned.is_none()).then(|| {
            let ords: Vec<usize> = llr_devices.split(',').map(|v| v.trim().parse().expect("--llr-devices 0[,1,..] | cpu")).collect();
            let mut l = Llr::new(sd.dims, block, kappa, support.clone());
            l.rank = llr_rank;
            l.hard = llr_hard;
            hasty_recon::llr_gpu::LlrGpu::new(l, &ords).unwrap()
        });
        let mut wav = hasty_recon::wavelet::Wavelet::new(sd.dims, wav_levels, wav_thr);
        let nimg = nf * ne;
        let tr = Instant::now();
        let (mut td, mut tn) = (0.0f64, 0usize);
        let mut denoise_prior = |k: usize, h: &mut Vec<C32>| {
            if kappa <= 0.0 {
                return;
            }
            let Some(prior) = &learned else {
                if prior_kind == "tlp" {
                    assert!(temporal.is_none(), "--prior tlp: one image per frame and encode (no --temporal)");
                    let t = Instant::now();
                    hasty_recon::tlp::cardiac_lowpass(h, nv, ne, nf_card, resp_bins, tlp_k, kappa.min(1.0) as f32);
                    if k < 2 || k % 10 == 0 {
                        eprintln!("    cardiac low-pass |k| ≤ {tlp_k}: {:.1}s", t.elapsed().as_secs_f64());
                    }
                    return;
                }
                if prior_kind == "wavelet" {
                    // x ← x + κ (W(x) − x)
                    wav.set_iteration(k as u64);
                    let mut w = h.clone();
                    wav.apply(&mut w, nimg);
                    let kf = kappa.min(1.0) as f32;
                    { use rayon::prelude::*; h.par_iter_mut().zip(w.par_iter()).for_each(|(x, d)| *x = C32::new(x.re + (d.re - x.re) * kf, x.im + (d.im - x.im) * kf)); }
                } else {
                    // RED: fixed shifts (seeds 0..n, averaged), else the cycle-spinning shift of iteration k
                    let seeds: Vec<u64> = if red { (0..red_shifts as u64).collect() } else { vec![k as u64] };
                    let mut one = |seed: u64, v: &mut Vec<C32>| {
                        if let Some(g) = llr_gpu.as_mut() {
                            g.set_iteration(seed);
                            g.apply(v, nimg).unwrap();
                            if k < 2 || k % 10 == 0 {
                                eprintln!("    LLR on GPU {:?}: {} blocks ({} left unchanged), k ≤ {}, pass 1 {:.1}s, pass 2 {:.1}s", g.devices(), g.last.blocks, g.last.skipped, g.last.k_max, g.last.secs[0], g.last.secs[1]);
                            }
                        } else {
                            llr.set_iteration(seed);
                            llr.apply(v, nimg);
                        }
                    };
                    if seeds.len() == 1 {
                        one(seeds[0], h);
                    } else {
                        use rayon::prelude::*;
                        let wn = 1.0 / seeds.len() as f32;
                        let mut acc = vec![C32::new(0.0, 0.0); h.len()];
                        let mut v = h.clone();
                        for (j, &sd) in seeds.iter().enumerate() {
                            if j > 0 {
                                v.par_iter_mut().zip(h.par_iter()).for_each(|(a, b)| *a = *b);
                            }
                            one(sd, &mut v);
                            acc.par_iter_mut().zip(v.par_iter()).for_each(|(a, b)| *a = C32::new(a.re + wn * b.re, a.im + wn * b.im));
                        }
                        *h = acc;
                    }
                }
                return;
            };
            // learned + wavelet: W(x) in the stack's own order (one image per frame × encode either way)
            let wv: Option<Vec<C32>> = (wav_mix > 0.0).then(|| {
                wav.set_iteration(k as u64);
                let mut w = h.clone();
                wav.apply(&mut w, nimg);
                w
            });
            // stack -> images[frame * 7 + e] (the temporal model stores encode-major sets of all frames)
            let mut img: Vec<C32> = if temporal.is_some() { (0..nf * ne).flat_map(|i| { let (f, e) = (i / ne, i % ne); h[(e * nf + f) * nv..(e * nf + f + 1) * nv].iter().copied() }).collect() } else { h.clone() };
            let t = Instant::now();
            // per respiratory bin: the model's time windows are cyclic over the cardiac frames
            for chunk in img.chunks_mut(nf_card * ne * nv) {
                prior.denoise(chunk, sd.dims, nf_card, kv7, sd.vox[0] as f32, tile, keep_frames).unwrap();
            }
            td += t.elapsed().as_secs_f64();
            tn += 1;
            let kf = kappa as f32;
            let (bl, bw) = ((1.0 - wav_mix) as f32, wav_mix as f32);
            for i in 0..nf * ne {
                let (f, e) = (i / ne, i % ne);
                let o = if temporal.is_some() { (e * nf + f) * nv } else { i * nv };
                for q in 0..nv {
                    let mut d = img[i * nv + q];
                    if let Some(w) = &wv {
                        let wq = w[o + q];
                        d = C32::new(bl * d.re + bw * wq.re, bl * d.im + bw * wq.im);
                    }
                    let x = &mut h[o + q];
                    *x = C32::new(x.re + (d.re - x.re) * kf, x.im + (d.im - x.im) * kf);
                }
            }
            if tn == 1 || tn % 10 == 0 {
                println!("    learned prior: {:.2}s per call ({tn} calls)", td / tn as f64);
            }
        };
        let ptag = match (learned.is_some(), prior_kind.as_str(), wav_mix > 0.0) {
            (true, _, true) => format!("_learned_wav{wav_mix}"),
            (true, _, false) => "_learned".to_string(),
            (false, "wavelet", _) => format!("_wav{wav_thr}"),
            (false, "tlp", _) => format!("_tlp{tlp_k}"),
            _ => String::new(),
        };
        let tsub_basis = tsub_spec.as_ref().map(|sp| {
            assert!(temporal.is_none() && nf == nf_card * resp_bins, "--tsub: one frame per set (no --temporal)");
            let parts: Vec<&str> = sp.split(':').collect();
            match parts[0] {
                "fourier" => hasty_recon::tsub::TemporalBasis::cardiac_fourier(nf_card, resp_bins, parts.get(1).expect("--tsub fourier:K").parse().expect("--tsub fourier:K")),
                "bspline" => hasty_recon::tsub::TemporalBasis::periodic_bspline(nf_card, resp_bins, parts.get(1).expect("--tsub bspline:M[:P]").parse().expect("--tsub bspline:M[:P]"), parts.get(2).map_or(3, |v| v.parse().expect("--tsub bspline:M[:P]"))),
                o => panic!("--tsub {o}: fourier:K | bspline:M[:P]"),
            }
        });
        let ts_tag = tsub_basis.as_ref().map_or(String::new(), |b| format!("_ts{}", b.name));
        let tag = format!("{model}{}_{pc}{ptag}_k{kappa}{}{dfw_tag}{lock_tag}{air_tag}{red_tag}{ts_tag}{ga_tag}", temporal.map_or(String::new(), |b| format!("_t{b}")), llr_rank.map_or(String::new(), |r| format!("_r{r}")) + if llr_hard { "_h" } else { "" });
        let resuming = out.join(format!("state_{tag}.hv")).exists();
        if !resuming && out.join(format!("recon_{tag}.npy")).exists() {
            println!("  {tag}: finished in an earlier run (recon_{tag}.npy), skipped (--fresh to redo)");
            continue;
        }
        if let (Some(ini), false, 0) = (&init, resuming, warm_full) {
            // --init-from: the previous level's images frame by frame (each source frame resampled once); one source
            // frame is the warm start already
            if ini.nf > 1 {
                let t_i = Instant::now();
                for fs in 0..ini.nf {
                    let imgs: Vec<Vec<C32>> = (0..ne).map(|e| ini.image(fs, e, sd.dims).unwrap()).collect();
                    for fr in (0..nf).filter(|&fr| ini.frame_for(fr, nf, nf_card).unwrap() == fs) {
                        for e in 0..ne {
                            let (t, o) = at(fr, e);
                            if host_state {
                                xh.chunk_mut(t)[o..o + nv].copy_from_slice(&imgs[e]);
                            } else {
                                st.memcpy_htod(&imgs[e], &mut x.v[t].slice_mut(o..o + nv)).unwrap();
                            }
                        }
                    }
                }
                println!("  {tag}: started from --init-from, {} source frames → {nf} frames, resampled {:?} → {:?} ({:.1}s)", ini.nf, ini.dims, sd.dims, t_i.elapsed().as_secs_f64());
            }
        }
        if init_truth && !resuming {
            let t_i = Instant::now();
            let mut nl = 0usize;
            for fr in 0..nf {
                for e in 0..ne {
                    let mut img = vec![C32::new(0.0, 0.0); nv];
                    for q in 0..nv {
                        let r = warm[e][q];
                        let mut z = Complex64::new(r.re as f64, r.im as f64);
                        if sample_truth(&sd.truth_frac, 0, &grid.x_log[q]) > 0.5 {
                            let v = [0, 1, 2].map(|a| sample_truth(&sd.truth_vel, a + 3 * (fr % nf_card), &grid.x_log[q]));
                            let r0 = warm[0][q];
                            z = Complex64::new(r0.re as f64, r0.im as f64) * Complex64::from_polar(1.0, dec.phase(e, &v) - dec.phase(0, &v));
                            if fr == 0 && e == 0 {
                                nl += 1;
                            }
                        }
                        img[q] = C32::new(z.re as f32, z.im as f32);
                    }
                    let (t, o) = at(fr, e);
                    if host_state {
                        xh.chunk_mut(t)[o..o + nv].copy_from_slice(&img);
                    } else {
                        st.memcpy_htod(&img, &mut x.v[t].slice_mut(o..o + nv)).unwrap();
                    }
                }
            }
            println!("  {tag}: --init-truth: warm images with the true velocity phases in {nl} lumen voxels (lumen fraction > 0.5), {nf} frames ({:.1}s)", t_i.elapsed().as_secs_f64());
        }
        let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(out.join(format!("metrics_{tag}.csv"))).unwrap();
        if !resuming {
            use std::io::Write;
            csv.set_len(0).unwrap();
            writeln!(csv, "iter,f,rmse,rel,seconds,{}", (0..nf).map(|f| format!("frame{f}")).collect::<Vec<_>>().join(",")).unwrap();
        }
        // after the prior: divergence-free wavelets on the velocity of every frame, then the static-tissue lock
        let img_of = |f: usize, e: usize| if temporal.is_some() { e * nf + f } else { f * ne + e };
        let mut dfw_op = dfw_rule.map(|r| hasty_recon::dfwavelet::DfWavelet::new(sd.dims, dfw_levels, sd.vox, r));
        let lay = hasty_recon::tlock::FrameLayout::new(sd.dims, sd.vox, nf_card, resp_bins, ne, temporal.is_some());
        let (map_npy, map_json) = (out.join(format!("static_map_{tag}.npy")), out.join(format!("static_map_{tag}.json")));
        let mut lmap: Option<hasty_recon::tlock::StaticMap> = None;
        if resuming && static_lock.is_some() && map_npy.exists() && map_json.exists() {
            let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&map_json).unwrap()).unwrap();
            let v = hasty_recon::data::read_npy(&map_npy).unwrap().f32s();
            let m = hasty_recon::tlock::StaticMap::from_stacked(&v, nv, meta["iteration"].as_u64().unwrap() as usize);
            println!("  static map of iteration {} loaded (static: cardiac {:.1} %, respiratory {:.1} %)", m.iteration, 100.0 * m.static_card, 100.0 * m.static_resp);
            lmap = Some(m);
        }
        // air map: reloaded on resume, else the seed (--air-init); next estimate at air_next
        let (air_npy, air_json) = (out.join(format!("air_map_{tag}.npy")), out.join(format!("air_map_{tag}.json")));
        let vox_mm = sd.vox.map(|v| v * 1e3);
        let mut amap: Option<hasty_recon::airmap::AirMap> = None;
        let mut air_next = usize::MAX;
        // a seed has not been applied to the iterate yet: applied before its first refresh, so that the release test
        // only sees what the gradient steps put back into the zeroed region, not the haze the seed is meant to remove
        let mut air_seed_pending = false;
        let n_dil = dil.iter().filter(|&&b| b).count().max(1);
        if let Some((ao, it0, ev)) = &air {
            air_next = *it0;
            if resuming && air_npy.exists() && air_json.exists() {
                let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&air_json).unwrap()).unwrap();
                let v = hasty_recon::data::read_npy(&air_npy).unwrap().f32s();
                let m = hasty_recon::airmap::AirMap::from_stacked(&v, nv, meta["iteration"].as_u64().unwrap() as usize, meta["level"].as_f64().unwrap_or(f64::NAN));
                if !meta["seed"].as_bool().unwrap_or(false) && *it0 != usize::MAX {
                    air_next = air_next.max(m.iteration.saturating_add(*ev));
                }
                println!("  air map of iteration {} loaded{}: {} voxels zeroed ({:.1} % of the coil-map support)", m.iteration, if meta["seed"].as_bool().unwrap_or(false) { " (seed)" } else { "" }, m.stats.marked, 100.0 * (0..nv).filter(|&q| dil[q] && m.w[q] >= 0.5).count() as f64 / n_dil as f64);
                amap = Some(m);
            } else if let Some(p) = &air_init {
                let (m, desc) = hasty_recon::airmap::load_seed(p, sd.dims, vox_mm, ao).unwrap();
                println!("  air seed {}: {desc} ({:.1} % of the coil-map support)", p.display(), 100.0 * (0..nv).filter(|&q| dil[q] && m.w[q] >= 0.5).count() as f64 / n_dil as f64);
                save(&air_npy, "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 4], f4(m.stacked()));
                let mut j = hasty_recon::airmap::sidecar(&m, ao, sd.dims, vox_mm);
                j["seed"] = serde_json::json!(true);
                j["init"] = serde_json::json!(p);
                std::fs::write(&air_json, serde_json::to_string_pretty(&j).unwrap()).unwrap();
                amap = Some(m);
                air_seed_pending = true;
            }
        }
        let (mut tdf, mut ndf, mut tlk, mut nlk) = (0.0f64, 0usize, 0.0f64, 0usize);
        let mut denoise = |k: usize, h: &mut Vec<C32>| {
            // air map: estimated on the iterate after the gradient step (zeroed voxels show what the step put back),
            // projected before the prior (it sees no haze in air) and again after it
            if let Some((ao, _, ev)) = &air {
                if k >= air_next {
                    let t = Instant::now();
                    if std::mem::take(&mut air_seed_pending) {
                        hasty_recon::airmap::apply(h, &lay, amap.as_ref().unwrap());
                    }
                    let m = hasty_recon::airmap::estimate(h, &lay, ao, amap.as_ref(), k);
                    let in_dil = (0..nv).filter(|&q| dil[q] && m.w[q] >= 0.5).count();
                    let lz = lumen.iter().filter(|&&q| m.w[q] > 0.0).count();
                    save(&air_npy, "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 4], f4(m.stacked()));
                    let mut j = hasty_recon::airmap::sidecar(&m, ao, sd.dims, vox_mm);
                    j["seed"] = serde_json::json!(false);
                    j["support_zeroed"] = serde_json::json!(in_dil);
                    j["support_fraction"] = serde_json::json!(in_dil as f64 / n_dil as f64);
                    j["truth_lumen_zeroed"] = serde_json::json!(lz);
                    std::fs::write(&air_json, serde_json::to_string_pretty(&j).unwrap()).unwrap();
                    let st = &m.stats;
                    // oracle diagnostic: truth-lumen voxels with w > 0 (should be 0)
                    println!("    air map at it {k}: {} voxels zeroed ({:.1} % of the coil-map support, {in_dil} voxels), {} added, {} released ({} protected), level {:.3e}; truth lumen zeroed {lz} ({:.2}s)", st.marked, 100.0 * in_dil as f64 / n_dil as f64, st.added, st.released, st.protected, m.level, t.elapsed().as_secs_f64());
                    amap = Some(m);
                    air_next = k.saturating_add(*ev);
                }
                if let Some(m) = &amap {
                    hasty_recon::airmap::apply(h, &lay, m);
                }
            }
            denoise_prior(k, h);
            if let Some(dfw) = dfw_op.as_mut() {
                let t = Instant::now();
                dfw.set_iteration(k as u64);
                let mut fr = vec![C32::new(0.0, 0.0); ne * nv];
                let (mut lam, mut kept) = ([0f64; 2], [0f64; 2]);
                let b = dfw_mix as f32;
                for f in 0..nf {
                    for e in 0..ne {
                        fr[e * nv..(e + 1) * nv].copy_from_slice(&h[img_of(f, e) * nv..][..nv]);
                    }
                    let st = hasty_recon::dfwavelet::shrink_encodes(dfw, &mut fr, ne, &dec);
                    (lam[0], lam[1], kept[0], kept[1]) = (lam[0] + st.lam_df, lam[1] + st.lam_n, kept[0] + st.kept_df, kept[1] + st.kept_n);
                    for e in 0..ne {
                        let dst = &mut h[img_of(f, e) * nv..][..nv];
                        for (x, d) in dst.iter_mut().zip(&fr[e * nv..(e + 1) * nv]) {
                            *x = C32::new(x.re + b * (d.re - x.re), x.im + b * (d.im - x.im));
                        }
                    }
                }
                tdf += t.elapsed().as_secs_f64();
                ndf += 1;
                if ndf == 1 || ndf % 10 == 0 {
                    let n = nf as f64;
                    println!("    DFW: {:.2}s per call; mean λ df {:.2} n {:.2}, kept df {:.1} % n {:.1} %", tdf / ndf as f64, lam[0] / n, lam[1] / n, 100.0 * kept[0] / n, 100.0 * kept[1] / n);
                }
            }
            if let Some((lo, it0, ev)) = &static_lock {
                if k >= *it0 && nf_card >= 2 {
                    let t = Instant::now();
                    if lmap.as_ref().map_or(true, |m| k >= m.iteration + ev) {
                        let m = hasty_recon::tlock::estimate(h, &lay, lo, Some(&dec), k);
                        save(&map_npy, "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 6], f4(m.stacked()));
                        std::fs::write(&map_json, serde_json::to_string_pretty(&serde_json::json!({
                            "iteration": m.iteration, "sigma2_blur": m.sigma2_blur, "sigma2_raw": m.sigma2_raw, "static_card": m.static_card, "static_resp": m.static_resp, "undetermined": m.undetermined,
                            "truth_lumen_locked": [lumen.iter().filter(|&&q| m.w_card[q] > 0.5).count() as f64 / lumen.len().max(1) as f64, lumen.iter().filter(|&&q| m.w_resp[q] > 0.5).count() as f64 / lumen.len().max(1) as f64],
                            "maps": ["s_card", "s_resp", "s_flow", "w_card", "w_resp", "sigma_raw"], "opts": format!("{lo:?}"),
                        })).unwrap()).unwrap();
                        // oracle diagnostic: truth-lumen voxels the map would lock (should be ≈ 0)
                        let ll = |w: &[f32]| if lumen.is_empty() { f64::NAN } else { lumen.iter().filter(|&&q| w[q] > 0.5).count() as f64 / lumen.len() as f64 };
                        println!("    static map at it {k}: static over the cardiac cycle {:.1} %, over breathing {:.1} % of the voxels, undetermined (motion not detectable) {:.1} %; truth lumen locked {:.2} % / {:.2} % ({:.2}s)", 100.0 * m.static_card, 100.0 * m.static_resp, 100.0 * m.undetermined, 100.0 * ll(&m.w_card), 100.0 * ll(&m.w_resp), t.elapsed().as_secs_f64());
                        lmap = Some(m);
                    }
                    let t = Instant::now();
                    hasty_recon::tlock::apply(h, &lay, lmap.as_ref().unwrap(), lo);
                    tlk += t.elapsed().as_secs_f64();
                    nlk += 1;
                    if nlk == 1 || nlk % 10 == 0 {
                        println!("    static lock: {:.2}s per call", tlk / nlk as f64);
                    }
                }
            }
            if let Some((gm, dl)) = mag_couple {
                use rayon::prelude::*;
                let t = Instant::now();
                // reference magnitude level: median of the mean magnitude over the coil-map support (frame 0)
                let mut mm: Vec<f32> = (0..nv).filter(|&q| dil[q]).map(|q| (0..ne).map(|e| { let z = h[img_of(0, e) * nv + q]; (z.re * z.re + z.im * z.im).sqrt() }).sum::<f32>() / ne as f32).collect();
                let m_med = if mm.is_empty() { 0.0 } else { let mid = mm.len() / 2; *mm.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1 };
                let d2 = (dl * m_med) * (dl * m_med);
                let mut fr = vec![C32::new(0.0, 0.0); ne * nv];
                let mut changed = 0f64;
                for f in 0..nf {
                    for e in 0..ne {
                        let src = &h[img_of(f, e) * nv..][..nv];
                        for q in 0..nv {
                            fr[q * ne + e] = src[q];
                        }
                    }
                    changed += fr.par_chunks_mut(ne).map(|z| {
                        let m: Vec<f32> = z.iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect();
                        let mb = m.iter().sum::<f32>() / ne as f32;
                        let mut dc = 0f64;
                        for (c, &me) in z.iter_mut().zip(&m) {
                            if me > 0.0 {
                                let w = me * me / (me * me + d2 + 1e-30);
                                let mn = mb + (me - mb) / (1.0 + gm * w);
                                let g = (mn / me).clamp(0.5, 1.5);
                                dc += ((g - 1.0) * me) as f64 * ((g - 1.0) * me) as f64;
                                *c = C32::new(c.re * g, c.im * g);
                            }
                        }
                        dc
                    }).sum::<f64>();
                    for e in 0..ne {
                        let dst = &mut h[img_of(f, e) * nv..][..nv];
                        for q in 0..nv {
                            dst[q] = fr[q * ne + e];
                        }
                    }
                }
                let tot: f64 = h.par_iter().map(|c| (c.re as f64).powi(2) + (c.im as f64).powi(2)).sum();
                if k < 3 || k % 10 == 0 {
                    println!("    magnitude coupling (γ {gm}, δ {dl} × {m_med:.3e}): relative change {:.2e} ({:.2}s)", (changed / tot.max(1e-300)).sqrt(), t.elapsed().as_secs_f64());
                }
            }
            if let Some(m) = &amap {
                hasty_recon::airmap::apply(h, &lay, m);
            }
        };
        // metrics / outputs of iteration k from per-set host images (device stacks are downloaded by the wrapper below)
        let report_due = |k: usize| { let k = k.wrapping_add(1); !(k % every != 0 && k != iters && k != 0) };
        let mut report_imgs = |k: usize, imgs: &[&[C32]], f: f64| {
            let k = k.wrapping_add(1);
            if k % every != 0 && k != iters && k != 0 {
                return;
            }
            if let Some(li) = &save_lumen {
                assert!(sd.native_grid(), "--save-lumen indexes native-grid voxels");
                let mut v = Vec::with_capacity(li.len() * ne * nf);
                for &q in li {
                    for e in 0..ne {
                        for fr in 0..nf {
                            let (t, o) = at(fr, e);
                            v.push(imgs[t][o + q]);
                        }
                    }
                }
                hasty_phantom::volume::write_npy(&out.join(format!("it{k}_{tag}_lumen.npy")), "<c8", &[li.len(), ne, nf], false, &c8(&v)).unwrap();
            }
            let img = |fr: usize, e: usize, q: usize| -> Complex64 {
                let (t, o) = at(fr, e);
                let c = imgs[t][o + q];
                Complex64::new(c.re as f64, c.im as f64)
            };
            // velocity at voxel q, frame fr (logical components; GNL: apparent → true via J⁻¹)
            let vel = |fr: usize, q: usize| -> [f64; 3] {
                let z: Vec<Complex64> = (0..ne).map(|e| img(fr, e, q)).collect();
                let mut v = dec.velocity(&z);
                if gnl {
                    v = mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &mat_vec(&sd.rot, &v)));
                }
                v
            };
            let (mut e2, mut t2, mut n) = (0.0f64, 0.0f64, 0usize);
            let mut e2h = 0.0f64;
            let mut per_frame = Vec::with_capacity(nf);
            for fr in 0..nf {
                let (mut fe2, mut fn_) = (0.0, 0usize);
                for &q in &lumen {
                    let v = vel(fr, q);
                    // the hierarchical decoder's answer, for comparability with runs before --vdecode
                    let vh = if dec_cmp {
                        let z: Vec<Complex64> = (0..ne).map(|e| img(fr, e, q)).collect();
                        let mut v = dec_h.velocity(&z);
                        if gnl {
                            v = mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &mat_vec(&sd.rot, &v)));
                        }
                        v
                    } else {
                        v
                    };
                    for a in 0..3 {
                        let t = sd.sample_log(&sd.truth_vel, a + 3 * (fr % nf_card), &grid.x_log[q]);
                        e2 += (v[a] - t).powi(2);
                        e2h += (vh[a] - t).powi(2);
                        t2 += t * t;
                        fe2 += (v[a] - t).powi(2);
                    }
                    n += 1;
                    fn_ += 1;
                }
                per_frame.push((fe2 / fn_.max(1) as f64).sqrt());
            }
            if dec_cmp {
                println!("    hierarchical decoder (±{r_first:.2} m/s): velocity RMSE {:.3} m/s (rel {:.3})", (e2h / n as f64).sqrt(), (e2h / t2).sqrt());
            }
            {
                use std::io::Write;
                writeln!(csv, "{k},{f:.6e},{:.5},{:.5},{:.1},{}", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), tr.elapsed().as_secs_f64(), per_frame.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",")).unwrap();
            }
            if k == iters && !no_full_save {
                let mut vout = vec![0f32; nv * 3 * nf];
                for fr in 0..nf {
                    for q in 0..nv {
                        let v = vel(fr, q);
                        for a in 0..3 {
                            vout[q + nv * (a + 3 * fr)] = v[a] as f32;
                        }
                    }
                }
                // streamed from the buffers: no reordered stack copy, no byte copy (≈ 2 stacks at the end otherwise)
                save_parts(&out.join(format!("vel_{tag}.npy")), "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 3, nf], std::iter::once(f32_bytes(&vout)));
                save_parts(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], ne, nf], (0..nf).flat_map(|fr| (0..ne).map(move |e| (fr, e))).map(|(fr, e)| {
                    let (t, o) = at(fr, e);
                    c32_bytes(&imgs[t][o..o + nv])
                }));
            }
            println!(
                "  {:<7} κ {kappa:<4} it {:>3}: f = {:.4e}, velocity RMSE {:.3} m/s (rel {:.3}), frames {}, {:.0}s",
                pc,
                k,
                f,
                (e2 / n as f64).sqrt(),
                (e2 / t2).sqrt(),
                per_frame.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "),
                tr.elapsed().as_secs_f64()
            );
        };
        let lag0 = p.lag_stats;
        // RED: λ = ρ / (α μ), μ the metric's mean gain on a fixed ±1 vector over four sets (deterministic: a resumed
        // run gets the same λ)
        let red_lambda = red.then(|| {
            let mu = if precond.metric() {
                let nt = lens.len();
                let mut sets = vec![0, nt / 3, 2 * nt / 3, nt - 1];
                sets.dedup();
                hasty_recon::red::metric_gain(&mut p, &st, &sets).unwrap()
            } else {
                1.0
            };
            let l = red_rho / (alpha * mu);
            println!("  {tag}: RED, g = ∇f + λ (x − D(x)), λ = ρ / (α μ) = {l:.4e} (ρ {red_rho}, α {alpha:.4e}, metric gain μ {mu:.4e}{})", if red_shifts > 1 { format!(", D averaged over {red_shifts} LLR shifts") } else { String::new() });
            l
        });
        // RED: the per-set preconditioners need the sets the denoiser couples in one secant group (Objective::secant_groups)
        let red_groups: Option<Vec<usize>> = red.then(|| {
            let nt = lens.len();
            let per_frame = temporal.is_none();
            // set → (frame, encode); with --temporal a set holds all frames of one encode
            let (fr, en) = (|t: usize| if per_frame { t / ne } else { usize::MAX }, |t: usize| if per_frame { t % ne } else { t });
            let mut keys: Vec<Vec<usize>> = vec![];
            if kappa > 0.0 {
                if learned.is_some() {
                    keys.push((0..nt).map(|t| if per_frame { fr(t) / nf_card } else { 0 }).collect());
                } else if prior_kind == "tlp" {
                    keys.push((0..nt).map(|t| (fr(t) / nf_card) * ne + en(t)).collect());
                } else if prior_kind == "llr" {
                    keys.push(vec![0; nt]);
                }
            }
            if dfw_rule.is_some() || mag_couple.is_some() {
                keys.push((0..nt).map(|t| if per_frame { fr(t) } else { 0 }).collect());
            }
            if static_lock.is_some() {
                keys.push((0..nt).map(en).collect());
            }
            let g = union_groups(nt, &keys);
            println!("  {tag}: RED secant groups: {} of {nt} sets", g.iter().copied().max().map_or(0, |m| m + 1));
            g
        });
        let outcome = if host_state {
            let mut noop = |_: usize, _: &mut Vec<C32>| {};
            let mut ro_store;
            let (obj, den): (&mut dyn hasty_recon::p2np::Objective, &mut dyn FnMut(usize, &mut Vec<C32>)) = match red_lambda {
                Some(l) => {
                    ro_store = hasty_recon::red::Red::new(&mut p, &mut denoise, l);
                    ro_store.groups = red_groups.clone();
                    (&mut ro_store, &mut noop)
                }
                None => (&mut p, &mut denoise),
            };
            if let Some(basis) = &tsub_basis {
                use hasty_recon::tsub::apply_sets;
                let (bb, pinv, nfb, nkb) = (basis.b.clone(), basis.pinv(), basis.nf, basis.nk);
                let gmax = basis.gram_max_eig();
                let alpha_c = alpha / gmax;
                let nvs = lens[0];
                let mut sub = hasty_recon::tsub::Subspace::new(obj, basis.clone(), ne);
                let mut ch = hasty_recon::hoststack::HostStack::zeros(&sub.coeff_lens());
                // start: the least-squares coefficients of the start images (a resumed run loads its own)
                apply_sets(&pinv, nkb, nfb, ne, nvs, &xh.data, &mut ch.data);
                let ng = hasty_recon::p2np::Objective::secant_groups(&sub).map_or(0, |g| g.iter().copied().max().map_or(0, |m| m + 1));
                println!("  {tag}: temporal subspace {}: {nkb} coefficients for {nfb} frames per encode ({:.2} GiB per coefficient stack), λ_max(TᵀT) {gmax:.3}, α {alpha_c:.4e}, {ng} secant groups", basis.name, ch.bytes() as f64 / GIB);
                // the frame stack: report buffer, and the P²nP denoiser's (T c → D → coefficients)
                let xbuf = std::cell::RefCell::new(std::mem::take(&mut xh));
                let prior_on = red_lambda.is_none();
                let mut den_c = |k: usize, c: &mut Vec<C32>| {
                    if !prior_on {
                        return;
                    }
                    let mut xb = xbuf.borrow_mut();
                    apply_sets(&bb, nfb, nkb, ne, nvs, c, &mut xb.data);
                    den(k, &mut xb.data);
                    apply_sets(&pinv, nkb, nfb, ne, nvs, &xb.data, c);
                };
                let mut rep_c = |k: usize, cs: &hasty_recon::hoststack::HostStack, f: f64| {
                    if !report_due(k) {
                        return;
                    }
                    let mut xb = xbuf.borrow_mut();
                    apply_sets(&bb, nfb, nkb, ne, nvs, &cs.data, &mut xb.data);
                    let r: Vec<&[C32]> = (0..xb.nsets()).map(|t| xb.chunk(t)).collect();
                    report_imgs(k, &r, f)
                };
                hasty_recon::hoststack::run_objective_host(&mut sub, &st, &mut ch, precond, alpha_c, iters, &mut den_c, &mut rep_c, Some((&ctl, &tag))).unwrap()
            } else {
                let mut report = |k: usize, xs: &hasty_recon::hoststack::HostStack, f: f64| {
                    let r: Vec<&[C32]> = (0..xs.nsets()).map(|t| xs.chunk(t)).collect();
                    report_imgs(k, &r, f)
                };
                hasty_recon::hoststack::run_objective_host(obj, &st, &mut xh, precond, alpha, iters, den, &mut report, Some((&ctl, &tag))).unwrap()
            }
        } else {
            assert!(tsub_basis.is_none(), "--tsub needs --state host");
            let mut report = |k: usize, xs: &Stack, f: f64| {
                if !report_due(k) {
                    return;
                }
                let h: Vec<Vec<C32>> = xs.v.iter().map(|v| st.clone_dtoh(v).unwrap()).collect();
                let r: Vec<&[C32]> = h.iter().map(|v| v.as_slice()).collect();
                report_imgs(k, &r, f)
            };
            if let Some(l) = red_lambda {
                assert!(lag_tau.is_none(), "--prior-mode red with --lagged: not yet");
                let mut ro = hasty_recon::red::Red::new(&mut p, &mut denoise, l);
                ro.groups = red_groups.clone();
                hasty_recon::p2np::run_objective(&mut ro, &st, &mut x, precond, alpha, iters, &mut |_, _| {}, &mut report, Some((&ctl, &tag))).unwrap()
            } else {
                p.run_from(&st, &mut x, precond, alpha, iters, &mut denoise, &mut report, Some((&ctl, &tag))).unwrap()
            }
        };
        if let Some(sr) = p.streamer_stats() {
            println!("  set streaming: {} sets, {:.1} GiB up / {:.1} GiB down", sr.0, sr.1 as f64 / GIB, sr.2 as f64 / GIB);
        }
        if let Outcome::Stopped(k) = outcome {
            println!("stopped {tag} at iteration {k} / {iters}: state saved in {}, rerun the same command to resume", out.display());
            std::process::exit(0);
        }
        if lag_tau.is_some() {
            println!("  lagged gradient: {} exact, {} surrogate (f is the surrogate estimate on surrogate iterations)", p.lag_stats.0 - lag0.0, p.lag_stats.1 - lag0.1);
        }
    }
    // all runs finished: the warm-start checkpoints only served resuming
    for e in 0..ne {
        ctl.clear(&format!("warm_done_{e}"));
        ctl.clear(&format!("warm_full_done_{e}"));
    }
    let tools = hasty_site::tool("figs/run_figs.py");
    let py = hasty_site::python();
    match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
        Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
        r => println!("figure script failed ({r:?}); run: {} -I {} {}", py.display(), tools.display(), out.display()),
    }
}

/// Groups of a temporal-pyramid level with `rg` respiratory × `cg` cardiac groups (clamped to the bins / phases):
/// set t of (frame f = r · nf_card + c, encode e) → (e · rg + ⌊r · rg / resp_bins⌋) · cg + ⌊c · cg / nf_card⌋
/// (contiguous bins and phases, group sizes differing by at most one). Returns (groups, group of every set).
/// Union of set partitions: sets sharing a key in any of `keys` (one key per set each) end in one group; group ids
/// 0.. in order of first appearance
fn union_groups(nt: usize, keys: &[Vec<usize>]) -> Vec<usize> {
    let mut par: Vec<usize> = (0..nt).collect();
    fn find(p: &mut [usize], mut a: usize) -> usize {
        while p[a] != a {
            p[a] = p[p[a]];
            a = p[a];
        }
        a
    }
    for key in keys {
        let mut first = std::collections::HashMap::new();
        for t in 0..nt {
            let r = *first.entry(key[t]).or_insert(t);
            let (a, b) = (find(&mut par, r), find(&mut par, t));
            if a != b {
                par[a.max(b)] = a.min(b);
            }
        }
    }
    let mut id = std::collections::HashMap::new();
    (0..nt).map(|t| { let r = find(&mut par, t); let n = id.len(); *id.entry(r).or_insert(n) }).collect()
}

fn tpyr_groups(frame_enc: &[(usize, usize)], ne: usize, nf_card: usize, resp_bins: usize, rg: usize, cg: usize) -> (usize, Vec<usize>) {
    let (rg, cg) = (rg.clamp(1, resp_bins), cg.clamp(1, nf_card));
    let g = frame_enc.iter().map(|&(f, e)| {
        let (r, c) = (f / nf_card, f % nf_card);
        (e * rg + r * rg / resp_bins) * cg + c * cg / nf_card
    }).collect();
    (ne * rg * cg, g)
}

/// Prolongation between pyramid levels: group g of the new level (`gof`, `ng` groups) gets the mean over its sets of
/// their previous group images (`pg`, `px`); a copy of the parent image where the levels nest. Also restriction to
/// any coarser grouping (e.g. per encode: the set-weighted time average).
fn tpyr_prolong(pg: &[usize], px: &[Vec<C32>], gof: &[usize], ng: usize) -> Vec<Vec<C32>> {
    let nv = px[0].len();
    (0..ng).map(|g| {
        let ts: Vec<usize> = (0..gof.len()).filter(|&t| gof[t] == g).collect();
        let mut acc = vec![[0.0f64; 2]; nv];
        for &t in &ts {
            for (a, v) in acc.iter_mut().zip(&px[pg[t]]) {
                a[0] += v.re as f64;
                a[1] += v.im as f64;
            }
        }
        let w = 1.0 / ts.len().max(1) as f64;
        acc.iter().map(|a| C32::new((a[0] * w) as f32, (a[1] * w) as f32)).collect()
    }).collect()
}

#[cfg(test)]
mod tpyramid_tests {
    use super::*;

    #[test]
    fn levels_parse_group_and_prolong() {
        assert_eq!(parse_tpyramid("1x1:4,2x1:3,2x5:3"), vec![(1, 1, 4), (2, 1, 3), (2, 5, 3)]);
        assert_eq!(parse_tpyramid("1X20:0"), vec![(1, 20, 0)]);
        assert!(std::panic::catch_unwind(|| parse_tpyramid("2x:3")).is_err());
        assert!(std::panic::catch_unwind(|| parse_tpyramid("0x1:3")).is_err());
        // 2 resp bins × 20 cardiac phases × 7 encodes, t = f · ne + e
        let (nr, nfc, ne) = (2usize, 20usize, 7usize);
        let fe: Vec<(usize, usize)> = (0..nr * nfc).flat_map(|f| (0..ne).map(move |e| (f, e))).collect();
        // 1x1: one group per encode
        let (ng, g) = tpyr_groups(&fe, ne, nfc, nr, 1, 1);
        assert_eq!(ng, ne);
        assert!(fe.iter().zip(&g).all(|(&(_, e), &gi)| gi == e));
        // R x F: one group per set
        let (ng, g) = tpyr_groups(&fe, ne, nfc, nr, nr, nfc);
        assert_eq!(ng, fe.len());
        let mut s = g.clone();
        s.sort_unstable();
        s.dedup();
        assert_eq!(s.len(), fe.len());
        // 2x3 (20 phases in 3 groups: 7, 7, 6) and clamping (3 resp groups of 2 bins → 2)
        let (ng, g) = tpyr_groups(&fe, ne, nfc, nr, 2, 3);
        assert_eq!(ng, ne * 6);
        let sizes: Vec<usize> = (0..ng).map(|k| g.iter().filter(|&&h| h == k).count()).collect();
        assert!(sizes.iter().all(|&n| n == 6 || n == 7), "{sizes:?}");
        assert!(fe.iter().zip(&g).all(|(&(_, e), &gi)| gi / 6 == e), "groups never mix encodes");
        assert_eq!(tpyr_groups(&fe, ne, nfc, nr, 3, 1).0, ne * 2);
        // nested levels: prolongation copies the parent image; restriction per encode averages
        let (_, g1) = tpyr_groups(&fe, ne, nfc, nr, 1, 1);
        let x1: Vec<Vec<C32>> = (0..ne).map(|e| vec![C32::new(e as f32, 1.0); 5]).collect();
        let (ng2, g2) = tpyr_groups(&fe, ne, nfc, nr, 2, 5);
        let x2 = tpyr_prolong(&g1, &x1, &g2, ng2);
        assert!((0..fe.len()).all(|t| x2[g2[t]] == x1[g1[t]]));
        let back = tpyr_prolong(&g2, &x2, &g1, ne);
        assert_eq!(back, x1);
        // non-nested (1x3 → 1x2): the mean of the parents, weighted by sets
        let (n3, g3) = tpyr_groups(&fe, ne, nfc, nr, 1, 3);
        let x3: Vec<Vec<C32>> = (0..n3).map(|k| vec![C32::new(k as f32, 0.0); 2]).collect();
        let (n4, g4) = tpyr_groups(&fe, ne, nfc, nr, 1, 2);
        let x4 = tpyr_prolong(&g3, &x3, &g4, n4);
        // encode 0, cardiac half 0 = phases 0..10: groups 0 (phases 0..6, 7 phases) and 1 (phases 7..9, 3 phases)
        assert!((x4[0][0].re - (7.0 * 0.0 + 3.0 * 1.0) / 10.0).abs() < 1e-6, "{:?}", x4[0][0]);
    }
}

/// `--tpyramid` levels "RxC:N,…" → [(resp groups, cardiac groups, pooled PCG iterations)]
fn parse_tpyramid(s: &str) -> Vec<(usize, usize, usize)> {
    let bad = |l: &str| -> ! { panic!("--tpyramid: level {l:?} is not RxC:N (e.g. 1x1:4,2x1:3,2x5:3)") };
    let v: Vec<(usize, usize, usize)> = s
        .split(',')
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (g, n) = l.split_once(':').unwrap_or_else(|| bad(l));
            let (r, c) = g.split_once(['x', 'X']).unwrap_or_else(|| bad(l));
            match (r.trim().parse::<usize>(), c.trim().parse::<usize>(), n.trim().parse::<usize>()) {
                (Ok(r), Ok(c), Ok(n)) if r >= 1 && c >= 1 => (r, c, n),
                _ => bad(l),
            }
        })
        .collect();
    assert!(!v.is_empty(), "--tpyramid: no levels");
    v
}

/// Per-cardiac-frame maps (nv × nf_card, e.g. the sim's truth fat fraction) for a run with respiratory bins
/// (nf = bins × nf_card frames, frame f = bin · nf_card + cardiac phase): frame f takes cardiac map f % nf_card.
/// One map, or maps already per frame, are returned unchanged.
fn expand_cardiac_maps(ff: Vec<f32>, nv: usize, nf: usize, nf_card: usize) -> Vec<f32> {
    if nf == nf_card || ff.len() != nv * nf_card {
        return ff;
    }
    (0..nf).flat_map(|f| ff[(f % nf_card) * nv..(f % nf_card + 1) * nv].iter().copied()).collect()
}

/// A float32 map file (the fat fraction) on the reconstruction grid: as stored on it, or resampled at the voxels'
/// true positions from the native grid (hasty_recon::pyramid::maps_on_recon_grid)
fn maps_on_grid(sd: &SimData, path: &std::path::Path, x_log: &[[f64; 3]], boxed: bool) -> Vec<f32> {
    let (v, resampled) = hasty_recon::pyramid::maps_on_recon_grid(sd, path, x_log, boxed).unwrap_or_else(|e| panic!("--fat-fraction {}: {e:?}", path.display()));
    if resampled {
        println!("  {}: {} native-grid map(s) {:?} resampled to the reconstruction grid {:?} ({})", path.display(), v.len() / sd.nv(), sd.tdims, sd.dims, if boxed { "voxel averages" } else { "at the true voxel positions" });
    }
    v
}

/// free device memory at a setup milestone (diagnostic for the device budget)
fn dev_mem(dev: &Device, what: &str) {
    if let Ok((free, total)) = dev.ctx.mem_get_info() {
        println!("device memory {what}: {:.2} GiB free of {:.2} GiB", free as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64);
    }
}


/// union weights of `--seg-basis svdn-union` per fit voxel: water w (1 − f_min)², fat w f_max² over the frames' fat
/// maps (ff: nv or nv × frames, frame-major), w = the fit voxel weights (1 without)
fn union_weights(fit: &[usize], w: Option<&[f64]>, ff: &[f32], nv: usize) -> (Vec<f64>, Vec<f64>) {
    let nfr = (ff.len() / nv).max(1);
    let mut ww = Vec::with_capacity(fit.len());
    let mut wf = Vec::with_capacity(fit.len());
    for (r, &q) in fit.iter().enumerate() {
        let (mut lo, mut hi) = (1.0f64, 0.0f64);
        for f in 0..nfr {
            let v = ff[f * nv + q] as f64;
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let b = w.map_or(1.0, |w| w[r]);
        ww.push(b * (1.0 - lo).max(0.0).powi(2) + 1e-12);
        wf.push(b * hi * hi + 1e-12);
    }
    (ww, wf)
}

/// `--bin-group`: the readout whose frame a group of encodes (readouts g0..g1, encode 0..) gets: the middle one
/// (`mid`), or the one nearest the time the group's velocity estimate refers to ([`hasty_recon::data::velocity_time`])
fn bin_group_readout(sd: &hasty_recon::data::SimData, g0: usize, g1: usize, mid: bool) -> usize {
    if mid || g1 - g0 < 2 {
        return g0 + (g1 - g0) / 2;
    }
    let t: Vec<f64> = (g0..g1).map(|i| sd.ro.t_exc[i] - sd.ro.t_exc[g0]).collect();
    let m1: Vec<_> = (g0..g1).map(|i| sd.encode_m1[sd.ro.enc[i]]).collect();
    let tv = hasty_recon::data::velocity_time(&m1, &t);
    (g0..g1).min_by(|&a, &b| (t[a - g0] - tv).abs().partial_cmp(&(t[b - g0] - tv).abs()).unwrap()).unwrap()
}
