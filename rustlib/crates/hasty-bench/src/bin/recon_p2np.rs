//! Time-resolved P²nP-ISTA reconstruction of the synthetic spiral 4D-flow data (frames × encodes), no density
//! compensation, whitened + coil-compressed data, LLR prior over all frames × encodes or the learned 7-point prior.
//! Any encoding of the simulator (referenced / Hadamard 4-point, 7-point dual-VENC): velocities by hasty_recon::flow.
//!   recon_p2np <sim_dir> [--model sense|seg] [--temporal off|linear|cubic] [--phys none|b0|full] [--gnl]
//!              [--nseg L|auto|auto:noise[:R]] [--order o] [--precond none,circ,zmshr1,zmshr1m,zmshr1s,zmshr1ms] [--step-guard F] [--iters n] [--kappa k,k,..] [--block b] [--llr-devices 0[,1,..]|cpu]
//!              [--vcoils auto|n] [--coil-energy 0.999] [--coil-gram calib|data] [--every n] [--warm n] [--eps e] [--conc-dc] [--synthetic] [--out dir]
//!              [--learned <export_dir> [--tile n] [--keep-frames n] [--learned-prec bf16|fp16|fp32] [--learned-cpu]]
//!              [--fresh] [--ckpt-every n] [--fat-fraction <npy|truth> [--fat-model shared|separate|joint]]
//!              [--conc-fit quad|deg4] [--table-prefetch]
//!              [--field truth|est:<recon_b0 run>[:bin]] [--r2s off|truth|est:<run>] [--rate <z.npy>] [--smaps truth|espirit[:eig_min]] [--coil-mask auto|none|support]
//!              [--coil-sampling keep:draw[:hold[:last[:seed]]]] [--lagged TAU[:LS|auto]] [--prior llr|wavelet] [--wavelet THR[:LEVELS]] [--wavelet-mix β] [--dfw sure[:TDF:TN]|fixed:TDF:TN [--dfw-levels L] [--dfw-mix β]] [--static-lock MU[:IT0[:EVERY]] [--lock-mode quad|group] [--lock-blur MM] [--lock-dilate MM] [--lock-score LO:HI] [--lock-vmin M/S] [--lock-detect OK:MAX]] [--air auto[:IT0[:EVERY]]|fixed [--air-init <npy>] [--air-margin MM] [--air-tau T] [--air-release R] [--air-blur MM[,MM..]]] [--state auto|device|host] [--state-store f32|bfpN|G:P] [--kstage f32|bfp[+qREL]|bfpN[:resident]] [--resp-bins NR [--resp bellows|truth]] [--seg-basis fixed|cheb|svd|svdw|svdn] [--seg-weight count|mass[:lo,hi]|energy] [--b0-absorb 0|1|2] [--kstore auto|legacy|<store spec>]
//!              [--table-store auto|pinned|lz4|zstd|zstdN]   (env HASTY_KSTORE_DIR: k-space store directory, default <sim>/kstore)
//!              [--grid-scale s|sx,sy,sz] [--kcrop [FRAC][:prefix]] [--adc-part out|full] [--init-from <npy|run dir>] [--truth-sampling point|box] [--init-truth] [--vdecode auto|hier|ml[:MARGIN]] [--seg-fit-stride N] [--seg-table-stride S]
//!              [--vmodel knots [--knots2d RK:RDEG,CK:CDEG] [--resp bellows|truth] [--eval LABEL[@AMP]:idx.npy,..] [--eval-every N]] [--keep N]
//! --vmodel knots: the velocity-parametrised model at the knots of the 2-D respiratory × cardiac knot operator
//!   (hasty_recon::vmodel::VelSpec::at_knots, vobj at-knots mode, vknot): per knot k = r · CK + c a complex reference z_k
//!   and a velocity v_k (+ the --vpoly residual phase), x_{e,k} = z_k e^{i(φ_e − k_e·v_k)}; one set per encode holding all
//!   K = RK · CK knot images, interpolated per sample by SegNufft::new_temporal2d (respiratory degree RDEG 0 bins | 1
//!   linear hats at the breathing-amplitude quantiles, cardiac periodic B-splines of degree CDEG; default 6:1,16:3 = the
//!   target config, research/temporal_operator.md §21). Breathing coordinate --resp bellows (default; physio.npy column
//!   3) | truth (oracle). The Gauss–Newton metric carries each knot's temporal-Gram weight. Memory: the knot ledger
//!   (hasty_recon::vknot::plan) places state (device, or host with θ streamed knot by knot), data and tables before the
//!   solver, honouring --state / --plan / --vram-budget, and refuses a plan that fits nowhere. Priors as --vmodel
//!   (--vkappa LLR on z and |z̄|·v, --dfw on v), plus the per-axis knot lock --static-lock MU[:IT0[:EVERY]]
//!   [--lock-coupling additive|nested] (hasty_recon::knotlock). Outputs: vel_<tag> / recon_<tag> at the 20 cardiac frame
//!   centres and the median breathing amplitude, vparams_<tag>.hv + vknots_<tag>.json (the knot state; --init-from of a
//!   finer pyramid level resamples it), --eval lumen files knots_<tag>_r<LABEL>_it<k>_lumen (frame_fit / score_pair form;
//!   AMP: the breathing amplitude to evaluate, default LABEL). --keep N: 1/N of the readouts (as splinerecon --keep N).
//!   That is the at-knots model, now `--vop knots`. Default `--vop frames:F[:DEG]` (F 16, DEG 3): the parameter-spline
//!   model (VelSpec::splines, research/velocity_model_sr.md §F): m, ψ, v per knot, the operator sees F cardiac frames per
//!   respiratory knot (cubic-spline interpolated), sets = encodes × respiratory groups; start --vinit cg:N[:λ] (default
//!   15: complex knot CGLS + per-knot dual-VENC decode + spline fit; needs the device memory of a coarse level, then
//!   --init-from) | warm; priors on by default: --vkappa 1, --dfw sure, --vmwav 1:3 (magnitude db4), --vpoly 1 with
//!   --vanchor 1,auto, --static-lock 20:10:5, --vcont 10 (each: 0 | off to switch off).
//! Coarse-to-fine pyramid (hasty_recon::pyramid, driver rustlib/tools/recon/pyramid.sh):
//!   --kcrop [FRAC][:prefix]: keep only the samples a (coarse, --grid-scale) grid represents, the ball
//!     Σ_a (k_a Δ_a)² ≤ (π·FRAC)² (default FRAC 1; ∩ the Nyquist box for FRAC > 1), applied to the sample records,
//!     the segment tables and the data (the k-space store is filtered on read, never rewritten; the kept data are
//!     held as a dense RAM copy when they fit, < 30 % of the available RAM; HASTY_KCROP_VIEW=1: always the filtered
//!     view). `prefix`: only each readout's outgoing part up to its first sample
//!     outside the ball (the kept ADC window ends early: fewer B0 / concomitant segments with --nseg auto).
//! --adc-part out|full: the ADC samples of every readout the operator uses (hasty_recon::pyramid::AdcPart). out
//!   (default): the outgoing half of the out-and-back readout, from the first k-centre pass to the first |k| maximum,
//!   detected on the per-sample coordinates (one ADC range [i0, i1) for all readouts: rotations of one base readout);
//!   the returning half (and samples during spoilers / encoders) stays in the files and the k-space store for
//!   navigators, dropped by the same sample selection as --kcrop (with --kcrop: the intersection); the segment
//!   time window (--nseg auto, the segment basis) is the kept ADC window. full: every ADC sample (the behaviour
//!   before --adc-part). Default output directories of out runs end in _aout.
//! --calib-adc out|full: the ADC samples of the coil calibration (the ESPIRiT calibration region and the
//!   --coil-gram data Gram). out (default with --adc-part out): the operator's outgoing half [i0, i1) only; full (the
//!   behaviour before 2026-11-23; the default with --adc-part full): every ADC sample. Noise whitening uses the noise
//!   covariance ψ of the scan metadata (noise statistics, no signal samples) either way.
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
//!              [--init warm|zero|meanres[:STEPS][:circ|pcgN][,..]] [--resid-steps n] [--anchor none|mean:μ|<dir>:μ[,..]]
//!              [--report-detail]
//! --init / --anchor (research/temporal_pyramid.md §10 eq. (8), Theorem 5 case B, §12): comma lists, every combination
//!   one more run after one setup (run tags _mr<steps><c|pN>, _zero, _anc_mean<μ>, _anc<K>f<μ>). meanres = pooled
//!   --warm-full solution + (N_tot/N)⁻¹ A_kᴴ(y_k − A_k x̄) per frame (hasty_recon::meanres); the anchor = ½ μ d̄ Σ_a
//!   n_a ‖mean_a(x) − c_a‖² in the gradient (hasty_recon::tanchor), μ in units of the mean diagonal d̄ of one frame's
//!   normal operator, included in λ_max. Frame mode only.
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
//! Knot grid slot: `--grid-budget auto|legacy|free|SIZE` (default auto: sized by the knot memory ledger before the
//!   engine is built: the largest coil batch whose slot fits next to the rest of the plan bounds it, the engine planner's
//!   rule picks (fewest transformed channels, then passes: no ragged last batch), 5 % of the
//!   device kept free for the estimate, research/sample_kernels.md §20; legacy: (free − work) / 3, at most 6 GiB, the
//!   previous default; free: (free − work − 2 GiB) / 1.5, no cap; §18: 50 → 100 channels per slot = −16 % gradient).
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
use hasty_recon::lanczos::{LamEst, LamMethod};
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

fn dtype_of(descr: &str) -> hasty_value::DType {
    match descr {
        "<c8" => hasty_value::DType::C64,
        "<f4" => hasty_value::DType::F32,
        "<f8" => hasty_value::DType::F64,
        d => panic!("dtype {d}"),
    }
}

/// an output array (Fortran order) from byte parts written in sequence, without assembling them in memory:
/// `hasty_value::store` image policy (1e-4 relative rms, research/io_policy.md) → `<name>.hv` (`HASTY_IO=npy`: .npy)
fn save_parts<'a>(path: &std::path::Path, descr: &str, shape: &[usize], parts: impl Iterator<Item = &'a [u8]>) {
    let parts: Vec<&[u8]> = parts.collect();
    hasty_value::store::save_parts(path, dtype_of(descr), shape, hasty_value::Order::F, &parts, &hasty_value::Save::image()).unwrap();
}
fn c32_bytes(v: &[C32]) -> &[u8] {
    // SAFETY: C32 is two f32, no padding
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) }
}
fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: plain f32 data
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
/// an output array (Fortran order): image policy, `<name>.hv` (see `save_parts`)
fn save(path: &std::path::Path, descr: &str, shape: &[usize], bytes: Vec<u8>) {
    hasty_value::store::save_bytes(path, dtype_of(descr), shape, hasty_value::Order::F, &bytes, &hasty_value::Save::image()).unwrap();
}

/// state reloaded on resume (static / air maps): lossless, so a resumed run continues exactly
fn save_exact(path: &std::path::Path, descr: &str, shape: &[usize], bytes: Vec<u8>) {
    hasty_value::store::save_bytes(path, dtype_of(descr), shape, hasty_value::Order::F, &bytes, &hasty_value::Save::lossless()).unwrap();
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
    // --lam-compare (diagnostic): λ_max by power 12 / 100 and Lanczos 4 / 6 / 8 before the solver (frame mode: N and
    // M N; knot / velocity mode: each stage's M H or H)
    let lam_compare = flag("--lam-compare");
    let learned_cpu = flag("--learned-cpu");
    let table_prefetch = flag("--table-prefetch");
    let bin_group = flag("--bin-group");
    // --prox-metric: see --llr-thr below (metric-consistent prox for global ZMSHR1)
    let prox_metric = flag("--prox-metric");
    // --bin-group-mid: the group's frame from its middle readout (before 2026-11-13: lags by 0.44 frame with the
    // referenced 7-point scheme, encode 0 carries a first moment); default the readout nearest the velocity time
    let bin_group_mid = flag("--bin-group-mid");
    // --save-smaps: the full coil-map stack in the output directory (1 GB at 2 mm); default: one axial slice for the figures
    let save_smaps = flag("--save-smaps");
    // --no-full-save: no final full recon / velocity stacks (8.9 + 1.9 GB at 2 mm), e.g. with --save-lumen diagnostics
    let no_full_save = flag("--no-full-save");
    // --adj-test: print the adjoint identity of the solver's operator and grouped vs member-wise application (setup)
    let adj_test = flag("--adj-test");
    // --report-detail: also print the lumen velocity error split into the time mean and the temporal detail (v_f − v̄)
    let report_detail = flag("--report-detail");
    // --llr-hard: hard singular-value threshold of the LLR prior (keep above, drop below; Llr::hard), run tag suffix _h
    let llr_hard = flag("--llr-hard");
    // --pp-band (with --pp-study / --pp-pool): band-limit the pooled estimator to the sampled ball (see --pp-study)
    let pp_band = flag("--pp-band");
    // --holdout-res (with --holdout N): the held-out readouts go through the solve's own operator pieces (the same
    // engine options, coil maps, coil basis, field and segments; their own sample tables) and the relative held-out
    // residual ‖E_h x − y_h‖ / ‖y_h‖ (whitened, compressed data, all held-out sets) is printed at every report
    // iteration (it 0 and every --every); CSV holdout_<tag>.csv. Default off: identical runs.
    let holdout_res = flag("--holdout-res");
    // --warm-only (with --warm-full N): exit after the pooled warm start, keeping its checkpoints
    // (state_warm_full_done_<e>: copied into other run directories, their --warm-full N loads them). Default off.
    let warm_only = flag("--warm-only");
    // --smaps-resp: coil maps per respiratory bin (see --smaps-fit below; research/smaps_estimation.md)
    let smaps_resp = flag("--smaps-resp");
    // --admm-rho-adapt (with --prior-mode admm): residual balancing of ρ (see --prior-mode below)
    let admm_adapt = flag("--admm-rho-adapt");
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
    // --coil-gram box[:MARGIN_MM]: the calibration-region coil images inside the prescription box (+ margin, default
    // 10 mm), so the virtual coils are energy-ordered for the box (research/coil_cost_reduction.md §5, §7.1); the box
    // from --presc-box S|SX,SY,SZ[@CX,CY,CZ] (mm, logical frame, centre relative to the FOV centre) or the scan
    // metadata `presc_box`. The full-FOV operator and data stay: only the compression rows change.
    let coil_box_margin: Option<f64> = coil_gram.strip_prefix("box").map(|r| r.strip_prefix(':').map_or(10.0, |v| v.parse().expect("--coil-gram box[:MARGIN_MM]")) * 1e-3);
    assert!(coil_gram == "calib" || coil_gram == "data" || coil_box_margin.is_some(), "--coil-gram calib|data|box[:MARGIN_MM]");
    let presc_box_arg: Option<String> = take("--presc-box");
    let warm_iters: usize = take("--warm").map(|s| s.parse().unwrap()).unwrap_or(0);
    // --warm-full n: after the plain-model warm start (which ignores B0, fat, R2*, concomitant fields and GNL: the
    // SENSE operator), n more pooled PCG iterations per encode through the full model operator (all frames of an encode
    // share one image; one gradient pass over all sets per iteration; the per-encode circulant as preconditioner when
    // the run builds circulants). The solver starts from that. 0 (default): the plain warm start only.
    // --pool-metric circ|none: the preconditioner of the pooled PCG (--warm-full, --tpyramid). circ (default, the
    // per-encode circulant whenever the solver's preconditioner uses the metric) or none (plain CG). At 1/16 .. 1/2 of
    // the readouts with the full physics model the circulant makes the pooled image error GROW (2026-11-22,
    // research/recon_diag_2026-11-22.md); plain CG converges.
    let pool_metric_none = match take("--pool-metric").as_deref() {
        None | Some("circ") => false,
        Some("none") => true,
        Some(o) => panic!("--pool-metric circ|none, got {o}"),
    };
    let warm_full: usize = take("--warm-full").map(|s| s.parse().unwrap()).unwrap_or(0);
    // Estimator-preserving pooled preconditioners (research/precond_estimator.md, hasty_recon::ppre; all off by default):
    // --pp-study "name=SPEC:ITERS[:TOL];...": after setup, pooled PCG of the encodes --pp-groups e,.. (default 0) with
    //   each preconditioner, scored per iteration (pp_mu*_name.csv, pp_summary.txt: residual, distance to the reference
    //   x̂ = the final image of a run whose name starts with "ref", error vs x★ with --synth-x), Ritz report of P Q
    //   (κ_eff, where the outliers sit in k and image space); then exit. SPEC = none | circ (the existing metric) | p0
    //   (nominal Chan symbol, matched floor) | p1 | p1n (probed symbol; n = ratio to the nominal) | p12 | p12n (p1 with
    //   D from the probed image diagonal) | p2 (that diagonal alone), + "+dT[,B]" = deflate the T largest (B smallest)
    //   Ritz pairs harvested from the same base's earlier run.
    // --pp-mu R[,R..]: estimator ridge μ = R × λ̄ (body mean of the probed image diagonal of the pooled normal
    //   operator), several = the study per μ; --pp-band: + band-limit to the sampled ball; --pp-dfloor F (0.1): D floored
    //   at F × its body median; --pp-probes J (8 per symbol / diagonal: one pooled pass each); --pp-smooth R (2): box
    //   radius of the symbol / diagonal smoothing; --pp-eps E (1e-3): symbol floor ε λ_max when μ = 0; --pp-harvest K
    //   (40): Lanczos vectors kept per group for the Ritz pairs; --pp-tol T (0): pooled stop at ‖r‖/‖r₀‖ ≤ T.
    // §7 (G1–G4): the study prints the Ritz energy against the shift map s = Δt |∇f| / k_max (Δt between the two
    //   k-centre passes, from the sample times; f the operator's ratemap field), s > 0.3. SPEC also: "+u" = G2
    //   recentred metric U P Uᴴ (U from the STORED segment maps at --pp-tref MS, default the sample-weighted mid time of
    //   the two centre passes); sur<L>k<K>[n] = G3 coil + segment circulant surrogate (L_s = L compressed stored maps)
    //   inverted by K inner PCG steps (inner metric: the existing circulant; n: none), outer FLEXIBLE PCG;
    //   cheb<L>k<K>[n] = the same with K-step Chebyshev (fixed SPD; --pp-cheb-ratio R, default 30: lo = hi / R; hi by
    //   Lanczos), deflatable (+dT). --pp-sur-coils N: the surrogate's virtual coils (default all).
    // --pp-pool SPEC (with --warm-full N): the pooled start with SPEC instead of --pool-metric (writes the warm_full_done
    //   checkpoints the --warm-full block then loads); --pp-metric SPEC: the zmshr1m metric per encode from SPEC's probed
    //   symbol (CirculantPrecond::from_factors; +dT[,B]: deflated with the --pp-pool harvest of the same base).
    let pp_args = PpArgs {
        study: take("--pp-study"),
        pool: take("--pp-pool"),
        metric: take("--pp-metric"),
        groups: take("--pp-groups").map(|s| s.split(',').map(|v| v.parse().expect("--pp-groups e,e,..")).collect()),
        mu: take("--pp-mu").map_or(vec![], |s| s.split(',').map(|v| v.parse().expect("--pp-mu R[,R..]")).collect()),
        band: pp_band,
        dfloor: take("--pp-dfloor").map_or(0.1, |v| v.parse().expect("--pp-dfloor F")),
        probes: take("--pp-probes").map_or(8, |v| v.parse().expect("--pp-probes J")),
        smooth: take("--pp-smooth").map_or(2, |v| v.parse().expect("--pp-smooth R")),
        eps: take("--pp-eps").map_or(1e-3, |v| v.parse().expect("--pp-eps E")),
        harvest: take("--pp-harvest").map_or(40, |v| v.parse().expect("--pp-harvest K")),
        tol: take("--pp-tol").map_or(0.0, |v| v.parse().expect("--pp-tol T")),
        tref: take("--pp-tref").map(|v| v.parse::<f64>().expect("--pp-tref MS") * 1e-3),
        sur_coils: take("--pp-sur-coils").map(|v| v.parse().expect("--pp-sur-coils N")),
        cheb_ratio: take("--pp-cheb-ratio").map_or(30.0, |v| v.parse().expect("--pp-cheb-ratio R")),
    };
    // --field-scale S (diagnostic, default 1): the operator's off-resonance ω (B0 field incl. --rate Δω) × S before the
    // segment model (concomitant fields and GNL unchanged); with --synthetic the data follow (inverse crime)
    let field_scale: f64 = take("--field-scale").map_or(1.0, |v| v.parse().expect("--field-scale S"));
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
    // --init SPEC[,SPEC..] (frame mode; research/temporal_pyramid.md §10): the solver's start per run, each spec one
    // more run (runs = preconditioners × κ × inits × anchors, one setup): warm (default: the existing start, --warm /
    // --warm-full / --tpyramid), zero, or meanres[:STEPS][:circ|pcgN] = the pooled --warm-full solution x̄ plus the
    // mean-operator deconvolution of each frame's residual back-projection, x_k = x̄ + (N_tot/N)⁻¹ A_kᴴ(y_k − A_k x̄),
    // STEPS Richardson steps (default --resid-steps, 1), the inverse by the per-encode circulant with a line-searched
    // scale (circ, default) or N PCG iterations on the mean normal operator (pcgN). Needs --warm-full n.
    let init_specs: Vec<String> = take("--init").map_or(vec!["warm".to_string()], |s| s.split(',').map(|v| v.trim().to_string()).collect());
    let resid_steps: usize = take("--resid-steps").map_or(1, |s| s.parse().expect("--resid-steps n"));
    for s in &init_specs {
        assert!(s == "warm" || s == "zero" || hasty_recon::meanres::MeanRes::parse(s, resid_steps).is_some(), "--init {s}: warm | zero | meanres[:STEPS][:circ|pcgN]");
    }
    // --anchor SPEC[,SPEC..] (research/temporal_pyramid.md Theorem 5 case B): ½ μ d̄ Σ_a n_a ‖R_a x − c_a‖² in the data
    // term, R_a = the plain (data-independent) mean over the fine frames of encode e inside coarse frame a. SPEC =
    // none | mean:μ (c = the pooled --warm-full solution, all frames of an encode) | <coarse run dir>:μ (its newest
    // recon_*.npy, nested frames: integer ratio). μ in units of the mean diagonal d̄ of one frame's normal operator
    // (anchor curvature on the group-constant mode = μ d̄); the step's λ_max includes it (Weyl bound).
    let anchor_specs: Vec<Option<(String, f64)>> = take("--anchor").map_or(vec![None], |s| {
        s.split(',')
            .map(|v| {
                let v = v.trim();
                if v == "none" {
                    return None;
                }
                let (src, mu) = v.rsplit_once(':').unwrap_or_else(|| panic!("--anchor {v}: none | mean:MU | <run dir>:MU"));
                Some((src.to_string(), mu.parse::<f64>().unwrap_or_else(|_| panic!("--anchor {v}: MU is not a number"))))
            })
            .collect()
    });
    let every: usize = take("--every").map(|s| s.parse().unwrap()).unwrap_or(5);
    let model = take("--model").unwrap_or("seg".into());
    let temporal: Option<u32> = match take("--temporal").as_deref() {
        None | Some("off") => None,
        Some("linear") => Some(1),
        Some("cubic") => Some(3),
        Some(o) => panic!("--temporal {o}"),
    };
    // --temporal-engine knot|interval (with --temporal): the knot-major operator (segnufft_knot.cu: one grid slot per
    // knot; the only one with per-knot ratemaps, --ratemap truth-bc) or the interval-major one (frame slots per
    // interval; the multi-image / group paths). Same operator up to f32 summation order. Default knot.
    let temporal_engine: Option<String> = take("--temporal-engine");
    assert!(temporal_engine.is_none() || temporal.is_some(), "--temporal-engine needs --temporal");
    let temporal_knot = match temporal_engine.as_deref() {
        None | Some("knot") => true,
        Some("interval") => false,
        Some(o) => panic!("--temporal-engine knot|interval, got {o}"),
    };
    let phys = take("--phys").unwrap_or("full".into());
    // time segments: --nseg L, or auto (default): from the phase cycles the field spans over the readout,
    // L = ⌈1.3 · Δf·T + 3.5⌉ (≈ 1e-3 operator error with least-squares interpolators: CPU budgets of
    // tests/seg_b0_direct.rs — 450 Hz over 8 ms → 8, fat included 843 Hz → 12; a fixed 10 was too few with fat)
    // --nseg auto:noise[:R] (research/segment_basis.md): the smallest L whose model error, predicted from the weighted
    // separable error (energy voxel weights, per-ADC-sample weights 1 + (E(n) + E_vol) / (C σ²)), is at most R noise rms
    // in the data (default R = 0.2); the plain `auto` rule above stays the default
    let nseg_spec: Option<String> = take("--nseg");
    let nseg_given = nseg_spec.is_some();
    let seg_noise_rule: Option<f64> = nseg_spec.as_deref().and_then(|s| s.strip_prefix("auto:noise")).map(|r| r.strip_prefix(':').map_or(0.2, |v| v.parse().expect("--nseg auto:noise[:R]")));
    // --nseg auto:err[:TOL]: the smallest L whose relative signal-weighted model error of the ratemap (the per-voxel
    // phase / decay / fat model of the segments, after the interpolator fit) is ≤ TOL (default 1e-4)
    let seg_err_rule: Option<f64> = nseg_spec.as_deref().and_then(|s| s.strip_prefix("auto:err")).map(|r| r.strip_prefix(':').map_or(1e-4, |v| v.parse().expect("--nseg auto:err[:TOL]")));
    let nseg_arg: Option<usize> = nseg_spec.and_then(|s| if s == "auto" || s.starts_with("auto:") { None } else { Some(s.parse().unwrap()) });
    let mut nseg: usize = nseg_arg.unwrap_or(if phys == "none" { 1 } else { 0 });
    let order: u32 = take("--order").map(|s| s.parse().unwrap()).unwrap_or(if phys == "full" { 2 } else { 0 });
    // relative floor of the circulant preconditioner's Fourier diagonal (larger = weaker, better-posed metric)
    let eps: f64 = take("--eps").map(|s| s.parse().unwrap()).unwrap_or(1e-3);
    let preconds: Vec<String> = take("--precond").unwrap_or("zmshr1".into()).split(',').map(|s| s.to_string()).collect();
    // --step-guard F: reject a ZMSHR1 iterate whose data term exceeds F × the previous one (back to the previous iterate,
    // plain scalar step; no extra gradient except one final check). Default off.
    let step_guard: Option<f64> = take("--step-guard").map(|s| s.parse().expect("--step-guard F (factor > 1)"));
    if let Some(fa) = step_guard {
        assert!(fa > 1.0, "--step-guard F: needs F > 1");
        println!("step guard: a ZMSHR1 iterate is rejected when the data term rises by more than × {fa}");
    }
    hasty_recon::p2np::set_step_guard(step_guard);
    // --circ-share set|encode: circulant metrics per set, or one per encode shared by its frames (default: encode)
    let circ_share_arg: Option<String> = take("--circ-share");
    let out_arg = take("--out");
    // periodic checkpoints (tens of GB per write at 2 mm) only on request; a STOP / Ctrl-C checkpoint is always written
    let ckpt_every: usize = take("--ckpt-every").map(|s| s.parse().unwrap()).unwrap_or(0);
    // --save-lumen idx.npy: at every report iteration, the reconstruction at these flat (Fortran) native-grid voxels
    // only, it<k>_<tag>_lumen.npy = complex64 (n, ne, nf) C-order (research/experiments/recon_diag/frame_fit.py's
    // *_lumen.npy form): per-frame scoring along the iterations without full stacks on disk
    // --score-pc DIR[:COVMIN] (scoring only, temporal_pyramid.md §13): per-frame velocity score against the sim's PC
    // signal truth resampled to this grid (hasty_recon::pcscore; DIR from research/experiments/temporal_pyramid/
    // pc_native.py), on the voxels that are blood (coverage ≥ COVMIN, default 0.99) throughout the cycle. Printed next
    // to the legacy all-lumen yardstick, with oracle checks at setup; CSV pcscore_<tag>.csv
    let score_pc: Option<(PathBuf, f64)> = take("--score-pc").map(|v| match v.rsplit_once(':') {
        Some((d, c)) if c.parse::<f64>().is_ok() => (PathBuf::from(d), c.parse().unwrap()),
        _ => (PathBuf::from(v), 0.99),
    });
    // --score-pc-bins D0,D1,..: one truth directory per respiratory bin (pc_native.py on bias_truth.py --resp R of the
    // bin's TRUE breathing state, printed under "respiratory bins"), stacked in the operator's frame order on the voxels
    // that are blood in every bin. Without it every bin is scored against the one --score-pc truth (one breathing state:
    // a bin at another state counts its correct displacement as error). Needs --score-pc (COVMIN, --synth-x).
    let score_pc_bins: Option<Vec<PathBuf>> = take("--score-pc-bins").map(|v| v.split(',').map(PathBuf::from).collect());
    assert!(score_pc_bins.is_none() || score_pc.is_some(), "--score-pc-bins needs --score-pc");
    // --synth-x pc (with --synthetic, staged data, --score-pc): the inverse-crime object x★ = PD_water + PD_fat + the
    // truth's blood signal (box means on this grid, synth_pd.py resampled) instead of the warm images; the velocity
    // score then decodes without the GNL J⁻¹ (x★ carries the nominal encoding phases)
    // --synth-x pd: the static part only (PD_water + PD_fat, no blood signal): the static-object control of the pooled
    // solve (research/recon_diag_2026-11-22.md)
    let (synth_pc, synth_static) = match take("--synth-x").as_deref() {
        None | Some("warm") => (false, false),
        Some("pc") => (true, false),
        Some("pd") => (true, true),
        Some(o) => panic!("--synth-x warm|pc|pd, got {o}"),
    };
    assert!(score_pc_bins.is_none() || !synth_pc, "--score-pc-bins with --synth-x pc: x★ is built from the --score-pc truth (one state), the per-bin score would not be its own truth");
    // --holdout N[:SEED]: a fixed 1/N of the (kept) readouts (hash, salt "HOLD" + SEED) is left out of every set — the
    // held-out data for the data-space score; --holdout-eval N[:SEED]: ONLY those readouts (evaluate a recon with
    // --init-from <recon npy> --init warm,zero --iters 1: f at it 1 = ½‖E x − y_h‖², the zero start gives ½‖y_h‖²).
    // --cc-load FILE: the coil-compression basis of the solve run (it saves <out>/coil_basis.txt with --holdout), so
    // the held-out residual lives in the same virtual coils. Default off: identical runs.
    let holdout: Option<(usize, u64, bool)> = take("--holdout").map(|v| (v, false)).or_else(|| take("--holdout-eval").map(|v| (v, true))).map(|(v, ev)| {
        let mut it = v.split(':');
        let n: usize = it.next().unwrap().parse().expect("--holdout N[:SEED]");
        let s: u64 = it.next().map_or(0, |x| x.parse().expect("--holdout N[:SEED]"));
        assert!(n >= 2, "--holdout N: N ≥ 2");
        (n, s, ev)
    });
    let cc_load: Option<PathBuf> = take("--cc-load").map(PathBuf::from);
    // --no-vel-save: the final recon stack only (e.g. for a --holdout-eval run), no velocity stack
    let no_vel_save = take("--no-vel-save").is_some_and(|v| v != "0");
    let card_frames: Option<usize> = take("--card-frames").map(|v| v.parse().expect("--card-frames K"));
    let save_lumen: Option<Vec<usize>> = take("--save-lumen").map(|p| hasty_recon::data::read_npy(std::path::Path::new(&p)).expect("--save-lumen idx.npy").f32s().iter().map(|&v| v as usize).collect());
    // velocity-parametrised model (research/velocity_model_sr.md): --vmodel KZ,KV [--vdeg DZ,DV] [--vpoly d] [--vcont n]
    // --vmodel knots [--knots2d RK:RDEG,CK:CDEG]: the velocity model at the knots of the 2-D respiratory × cardiac knot
    // operator (VelSpec::at_knots, research/velocity_model_sr.md, temporal_operator.md §16–§21): one set per encode
    // with all K = RK · CK knot images x_{e,k} = z_k e^{i(φ_e − k_e·v_k)}, the operator interpolating them to every
    // sample's (breathing, cardiac) time; respiratory knots RK of degree RDEG (0 bins, 1 linear hats at the --resp
    // amplitude quantiles; bellows by default, truth = oracle), cardiac CK periodic B-spline knots of degree CDEG
    // (default 6:1,16:3, the target config of temporal_operator.md §21)
    let vmodel_arg = take("--vmodel");
    let knots_mode = vmodel_arg.as_deref() == Some("knots");
    let knots2d_arg = take("--knots2d");
    assert!(knots2d_arg.is_none() || knots_mode, "--knots2d needs --vmodel knots (complex knot images: splinerecon)");
    let knots2d: Option<(usize, u32, usize, u32)> = knots_mode.then(|| hasty_recon::vknot::KnotBasis::parse_spec(knots2d_arg.as_deref().unwrap_or("6:1,16:3")).unwrap_or_else(|e| panic!("{e}")));
    // --vop frames:F[:DEG] (default frames:16:3) | knots (--vmodel knots): the parameter-spline model (research/
    // velocity_model_sr.md §F: m, ψ, v in the knot basis, the operator sees F cardiac frames per respiratory knot, cubic-
    // spline interpolated (DEG 3) or linear (1); one set per encode × respiratory group) or the at-knots model (§E:
    // complex interpolation of the model's knot values, kept selectable)
    let vop_arg = take("--vop");
    assert!(vop_arg.is_none() || knots_mode, "--vop needs --vmodel knots");
    let op_frames: Option<hasty_recon::vknot::OpFrames> = match (knots2d, vop_arg.as_deref()) {
        (None, _) | (_, Some("knots")) => None,
        (Some((kr, rdeg, mc, cdeg)), v) => {
            let (f, d) = v.map_or(Ok((16, 3)), hasty_recon::vknot::OpFrames::parse_spec).unwrap_or_else(|e| panic!("{e}"));
            assert!(rdeg == 1 && cdeg != 0, "--vop frames: linear respiratory knots and cardiac degree 1 or 3 (--knots2d RK:1,CK:1|3), or --vop knots");
            Some(hasty_recon::vknot::OpFrames::new(kr, mc, cdeg, f, d))
        }
    };
    // spline mode defaults: priors on (LLR κ 1, divergence-free wavelets, magnitude wavelets, residual phase degree 1 with
    // the static anchor, per-axis knot lock), VENC continuation 10; every one can be switched off (… 0 | off)
    let spline_defaults = op_frames.is_some();
    // --vinit cg:N[:LAMBDA] | warm: spline mode's start when there is no --init-from: N CGLS iterations of complex
    // parameter-basis knot images per encode (through the same operator), the dual-VENC decoder per knot, then m, ψ, v
    // spline coefficients (default cg:15); warm = the time-averaged warm start in every knot
    let vinit: Option<(usize, f64)> = match take("--vinit").as_deref() {
        Some("warm") => None,
        Some(v) => {
            let it: Vec<&str> = v.split(':').collect();
            assert!(it[0] == "cg" && it.len() >= 2, "--vinit cg:N[:LAMBDA] | warm");
            Some((it[1].parse().expect("--vinit cg:N"), it.get(2).map_or(0.0, |x| x.parse().expect("--vinit cg:N:LAMBDA"))))
        }
        None => spline_defaults.then_some((15, 0.0)),
    };
    // spline-mode safeguards (research/velocity_model_sr.md §F.2):
    // --vinit-snr S | design | off: the robust start decodes a knot's velocity only where its SNR (mean encode magnitude
    //   over the per-voxel noise level from the encode-magnitude dispersion) reaches S, else the voxel's time-mean decode
    //   (if that reaches S), else 0; design (default) = the sequence's SNR of a 1e-3 unwrap failure rate
    //   (meta sequence.encoding.snr_unwrap_1e-3, 3 if absent); off = decode every knot (the old start)
    // --vbox V | design | off: v coefficients projected onto |v_a| ≤ V after the prior (the indicator of the decoder's
    //   unaliased range, part of the prior; the B-spline convex hull keeps the curve inside); default design
    //   (meta sequence.encoding.unaliased, else the decoder range)
    // --vtrust RHO | off: per (voxel, knot) trust region of the gradient step, max_e |k'_e·Δv| ≤ RHO rad (the region where
    //   the phase linearisation of the Gauss–Newton metric holds); default 1
    let vinit_snr_arg: Option<String> = take("--vinit-snr");
    let vbox_arg: Option<String> = take("--vbox");
    let vtrust: Option<f64> = match take("--vtrust").as_deref() {
        Some("off") | Some("0") => None,
        Some(v) => Some(v.parse().expect("--vtrust RHO | off")),
        None => spline_defaults.then_some(1.0),
    };
    // --vmwav THR[:LEVELS] | off: spatial db4 wavelet shrinkage of the magnitude knots m_k (spline mode; τ = THR × the
    // per-image MAD noise level; default 1.0:3)
    let vmwav: Option<(f64, usize)> = match take("--vmwav").as_deref() {
        Some("off") | Some("0") => None,
        Some(v) => { let (a, b) = v.split_once(':').unwrap_or((v, "3")); Some((a.parse().expect("--vmwav THR[:LEVELS]"), b.parse().expect("--vmwav THR[:LEVELS]"))) }
        None => spline_defaults.then_some((1.0, 3)),
    };
    // knot mode without --nseg: the noise-referenced error budget (auto:noise:0.2: the ratemap segment model's predicted
    // data error ≤ 0.2 noise rms), on the noise-weighted SVD basis (svdn) with energy voxel weights unless --seg-basis /
    // --seg-weight say otherwise. 1e-4 is the NUFFT's rule, not the segments' (research/pipeline_target.md §2);
    // --nseg auto:err:TOL keeps a fixed relative tolerance when wanted
    let seg_noise_rule = if knots_mode && !nseg_given { Some(0.2) } else { seg_noise_rule };
    let vmodel: Option<(usize, usize)> = vmodel_arg.filter(|s| s != "knots").map(|s| { let v: Vec<usize> = s.split(',').map(|x| x.parse().expect("--vmodel KZ,KV | knots")).collect(); (v[0], v[1]) });
    // --keep N: a deterministic 1/N of the readouts (hash-spread over all strata, the same subset as splinerecon
    // --keep N: a shorter scan for dev runs); the others are in no set
    let keep_n: usize = take("--keep").map_or(1, |v| v.parse().expect("--keep N"));
    // knot mode: --eval LABEL[@AMP]:idx.npy,... evaluates the knot model at breathing amplitude AMP (default LABEL) and the
    // cardiac frame centres at the native-grid voxels idx (frame_fit / score_pair lumen form, files
    // <out>/knots_<tag>_r<LABEL>_it<k>_lumen), every --eval-every iterations and at the end
    let eval_spec: Option<String> = take("--eval");
    let eval_every: usize = take("--eval-every").map_or(10, |v| v.parse().expect("--eval-every N"));
    // knot mode: power iterations of each stage's λ_max (default 12, as --vmodel)
    let lam_iters: usize = take("--lam-iters").map_or(12, |v| v.parse().expect("--lam-iters N"));
    // --lam-method lanczos[:K]|power: the step size's λ_max estimator (research/temporal_operator.md, "Step size by
    // Lanczos"). lanczos (default, K = 8 steps): α = 1 / (θ + residual bound); power: the original power iteration
    // (frame mode 12 iterations, knot / velocity mode --lam-iters), α = 1 / λ exactly as before
    let lam_method = take("--lam-method").map_or(hasty_recon::lanczos::LamMethod::Lanczos(8), |v| hasty_recon::lanczos::LamMethod::parse(&v).expect("--lam-method lanczos[:K]|power"));
    // --lam-warm auto|on|off (Lanczos, velocity stages): start each stage's λ_max from the previous stage's top Ritz
    // vector; keeps the K Lanczos vectors in host memory, auto = when (K + 1) stacks fit min(HASTY_LAM_WARM_GB
    // (default 4), a quarter of the available host memory)
    let lam_warm_arg = take("--lam-warm").unwrap_or_else(|| "auto".into());
    let lam_warm_on = |stack_bytes: usize, k: usize| -> bool {
        match lam_warm_arg.as_str() {
            "on" => true,
            "off" => false,
            _ => {
                let cap = std::env::var("HASTY_LAM_WARM_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(4.0) * (1u64 << 30) as f64;
                let avail = std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).map_or(0.0, |kb| kb * 1024.0);
                ((k + 1) * stack_bytes) as f64 <= cap.min(avail / 4.0)
            }
        }
    };
    let vdeg: (u32, u32) = take("--vdeg").map(|s| { let v: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) }).unwrap_or((1, 1));
    let vpoly: Option<usize> = match take("--vpoly").as_deref() { Some("off") => None, Some(s) => Some(s.parse().unwrap()), None => spline_defaults.then_some(1) };
    let vcont: usize = take("--vcont").map(|s| s.parse().unwrap()).unwrap_or(if spline_defaults { 10 } else { 0 });
    // z warm-up: first n iterations update z only (v and the residual phase frozen): makes z consistent with the
    // operator (B0, coils, physics) before velocity moves, so encode-common phase errors cannot leak into v
    let vzwarm: usize = take("--vzwarm").map(|s| s.parse().unwrap()).unwrap_or(0);
    // velocity-model prior strengths (LLR on the z nodes and on p = |z̄|·v; 0 = no prior), one run each
    let vkappas: Vec<f64> = take("--vkappa").unwrap_or(if spline_defaults { "1".into() } else { "0".into() }).split(',').map(|s| s.parse().unwrap()).collect();
    // memory planner (velocity path): device-usage cap (e.g. 8G; default: the whole card) and placement mode
    let vram_budget: Option<usize> = take("--vram-budget").map(|s| hasty_recon::plan::parse_bytes(&s).expect("--vram-budget e.g. 8G"));
    let plan_mode = take("--plan").unwrap_or("auto".into());
    // knot-engine grid slot budget (research/sample_kernels.md §18, §20): auto (default) = sized by the knot ledger
    // (hasty_recon::vknot::slot_budget: the planner's batch (no ragged last batch) under the largest slot that fits next to the planned state,
    // data, tables and scratch; falls back to legacy if the estimate refuses); legacy = (free − work) / 3 clamped to
    // 6 GiB (the previous default); free = (free − work − 2 GiB) / 1.5, no clamp; or an explicit size (e.g. 10G). The
    // knot ledger still checks the whole plan before the solver.
    let grid_budget_arg = take("--grid-budget").unwrap_or("auto".into());
    let kstore_arg = take("--kstore").unwrap_or("auto".into());
    // --rcompress off|auto[:OS]|ball[:OS[:R_MM]]|box[:OS]: exact per-readout sample compression of the data term
    // (hasty_recon::rcompress, research/sample_compression.md; ROI = the grid box, one design per readout, or the ball
    // enclosing it, one shared design; band = the operator's ratemap). auto: on when the estimated reduction ≥ 1.2×
    let rc_mode = hasty_recon::rcompress::RcMode::parse(&take("--rcompress").unwrap_or_else(|| "off".into())).unwrap_or_else(|e| panic!("{e}"));
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
    // --ratemap est:<recon_b0 run>[:bin]|truth: the whole ratemap in one flag — off-resonance (--field), decay (--r2s) and
    // the fat fraction folded into the segment diagonals (--fat-fraction with --fat-model joint), all from the ratemap
    // scan's maps (est, real-data path) or the phantom truth (oracle control). Explicit --field / --r2s /
    // --fat-fraction / --fat-model override the corresponding part. research/pipeline_target.md §3 rows 1–2.
    // --ratemap truth-bc: one TRUE ratemap per operator frame (oracle, research/mathematics_of_paper/
    // forward_chain_section.tex eq. seg): the simulator's per-(breathing, cardiac)-state maps (scan/truth_ratemap_bc/,
    // research/resp_field_sim.md) at each knot's state, each knot's image padded with its own segments (2-D knot
    // engines, SegNufft::segments_frames). The static flags below stay as they are.
    let ratemap_spec: Option<String> = take("--ratemap");
    if let Some(r) = &ratemap_spec {
        assert!(r == "truth" || r == "truth-bc" || r == "truth-bc:static" || r.starts_with("est:"), "--ratemap est:<recon_b0 run>[:bin]|truth|truth-bc[:static], got {r}");
    }
    // truth-bc:static: the same per-knot path and segment build, every knot given the data-weighted mean of the
    // knots' true ratemaps (the best static ratemap): the one-factor control of truth-bc
    let ratemap_static = ratemap_spec.as_deref() == Some("truth-bc:static");
    let ratemap_frames = ratemap_spec.as_deref() == Some("truth-bc") || ratemap_static;
    let ratemap_spec: Option<String> = ratemap_spec.filter(|r| !r.starts_with("truth-bc"));
    // frame mode of --ratemap truth-bc (no knots, operator frames or temporal model): one segment table per (frame,
    // encode). Its defaults follow knot mode, the trusted path of the same option: plain `auto` (or no --nseg) becomes the
    // noise-referenced rule auto:noise:0.2 on every table's own model (the static rule sizes L from the static field
    // without fat), the noise-weighted SVD basis svdn with energy voxel weights (research/segment_basis.md)
    let frame_rm_mode = ratemap_frames && !knots_mode && op_frames.is_none() && temporal.is_none();
    assert!(!(temporal.is_some() && !temporal_knot && ratemap_frames), "--ratemap truth-bc with --temporal needs the knot-major engine (per-knot tables; not --temporal-engine interval)");
    let seg_noise_rule = if frame_rm_mode && nseg_arg.is_none() && seg_noise_rule.is_none() && seg_err_rule.is_none() {
        println!("--ratemap truth-bc (frame mode): segment count by auto:noise:0.2 on every (frame, encode) model (as knot mode)");
        Some(0.2)
    } else {
        seg_noise_rule
    };
    let fat_fraction: Option<String> = take("--fat-fraction").or_else(|| ratemap_spec.clone());
    // measured instead of true fields / coil maps (the oracle-vs-estimate evaluation; defaults = the oracle as before):
    // --field truth|est:<recon_b0 run>[:bin]: off-resonance ω; --r2s off|truth|est:<run>[:bin]: T2* decay during the
    // readout folded into the B0 segments; --smaps truth|espirit[:eig_min]: ESPIRiT maps from the compressed data of
    // encode 0 (hasty_recon::espirit); --fat-fraction est:<run>[:bin] takes the fat fraction of the B0 scan
    let field_spec = take("--field").or_else(|| ratemap_spec.clone()).unwrap_or("truth".into());
    let r2s_spec = take("--r2s").or_else(|| ratemap_spec.clone()).unwrap_or("off".into());
    let rate_spec: Option<String> = take("--rate");
    assert!(rate_spec.is_none() || r2s_spec == "off", "--rate carries R2*: use it without --r2s");
    let smaps_spec = take("--smaps").unwrap_or("truth".into());
    // research/smaps_estimation.md (both off by default; --smaps espirit only):
    // --smaps-fit bspline:H_MM[:LAMBDA[:E0]]: replace the ESPIRiT maps by a weighted cubic B-spline fit (knot spacing
    //   H_MM mm, second-difference penalty LAMBDA, default 1e-3; weights ((λ − E0)/(1 − E0))² of the ESPIRiT
    //   eigenvalue λ inside the dilated support, E0 default 0.5, which is then also ESPIRiT's crop threshold): fills the
    //   low-signal holes (lungs) that the eig_min crop zeroes, removes structure finer than ≈ H (image leakage),
    //   renormalised Σ|s|² = 1 (hasty_recon::smapfit)
    // --smaps-resp: one ESPIRiT map set per respiratory bin (calibration from the encode-0 sets of that bin only), phase
    //   gauge aligned to the pooled maps; the operator uses bin r's maps for every set of bin r (setop::PerSetMaps,
    //   also in --holdout-res). The warm start, preconditioners and support keep the pooled maps.
    let smaps_fit: Option<(f64, f64, f64)> = take("--smaps-fit").map(|s| {
        let p: Vec<&str> = s.split(':').collect();
        assert!(p[0] == "bspline" && p.len() >= 2, "--smaps-fit bspline:H_MM[:LAMBDA[:E0]], got {s}");
        let h = p[1].parse::<f64>().expect("--smaps-fit bspline:H_MM") * 1e-3;
        let lam = p.get(2).map_or(1e-3, |v| v.parse().expect("--smaps-fit LAMBDA"));
        let e0 = p.get(3).map_or(0.5, |v| v.parse().expect("--smaps-fit E0"));
        assert!(h > 0.0 && lam >= 0.0 && (0.0..1.0).contains(&e0), "--smaps-fit: H > 0, LAMBDA ≥ 0, 0 ≤ E0 < 1");
        (h, lam, e0)
    });
    assert!(smaps_fit.is_none() || smaps_spec.starts_with("espirit"), "--smaps-fit: with --smaps espirit");
    assert!(!smaps_resp || smaps_spec.starts_with("espirit"), "--smaps-resp: with --smaps espirit");
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
    // --state-store f32|bfpN|G:P (= HASTY_STATE_STORE): host solver state with g, g_prev and the secant step as BFP
    // stacks, x f32 (hasty_recon::bfpstate); --kstage f32|bfp[+qREL]|bfpN[:resident] (= HASTY_KSTAGE): k-space
    // staged as BFP (σ rule) and decoded on the device (hasty_recon::kstage). Default f32 for both (research/bfp_storage.md §9)
    for (flag, var) in [("--state-store", "HASTY_STATE_STORE"), ("--kstage", "HASTY_KSTAGE")] {
        if let Some(v) = take(flag) {
            std::env::set_var(var, &v);
        }
    }
    let state_store = hasty_recon::bfpstate::StateStore::from_env().unwrap_or_else(|e| panic!("--state-store: {e}"));
    let kstage = hasty_recon::kstage::stage_from_env().unwrap_or_else(|e| panic!("--kstage: {e}"));
    if !state_store.is_f32() || kstage.is_some() {
        println!("BFP storage: host state {}; k-space staging {}", state_store.describe(), kstage.map_or("f32".to_string(), |(r, res)| format!("{}{}", r.describe(), if res { ", resident on the device" } else { "" })));
    }
    // --prior llr|wavelet (complex-image prior without --learned; default llr); --wavelet THR[:LEVELS] (default 1:3):
    // 3D Daubechies-4 shrinkage per image, threshold THR × the MAD noise level of each image (scale-equivariant), cycle
    // spinning; --wavelet-mix β: with --learned, x ← x + κ ((1 − β) D(x) + β W(x) − x) (research/wavelet_prior.md);
    // with --prior llr (no --learned) the LLR output is mixed the same way: x ← (1 − β) LLR(x) + β W(x) (run tag _llrwav{β})
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
    // --prior-mode admm (hasty_recon::admm; research/precond_sampling_theory.md §2–§5): ADMM with the data step solved
    // per set, x_t ← argmin ½‖E_t x − y_t‖² + (ρ/2)‖x − (z_t − u_t)‖² by --admm-inner K (default 3) PCG iterations on
    // N_t + ρI (warm start x_t, on the device per set), z ← D(x + u) with the same prior pipeline as P²nP (LLR / wavelet
    // mix, DFW, static lock, air), u ← u + x − z. Its fixed point does not depend on the inner metric. Output and scores
    // are z (recon_<tag>); x is scored too (held-out residual, pc score). --admm-rho R: ρ = R ρ₁ (default 1), ρ₁ = one
    // sample's curvature in one Nyquist cell, coil energy summed (measured at setup: Hutchinson trace of N_t, checked on
    // plane waves). --admm-rho-adapt: residual balancing (×2 / ÷2 by the 10× rule, dual residual in units of ρ₁, u
    // rescaled). --admm-metric none|clip (default none): clip = D^{-1/2} F⁻¹ diag(1/(max(μ̃_t, 1) + ρ/ρ₁)) F D^{-1/2},
    // μ̃_t the set's expected samples per cell (per-encode coil-smoothed Chan symbol × the set's sample share). The κσ̂
    // thresholds are re-estimated from x + u = z − ∇f(z)/ρ, so they carry the 1/ρ scale of prox_{(λ/ρ)R} themselves (no κ
    // rescaling; hasty_recon::admm). Host state x, z, u (+ one work stack). Run tag suffix _admm{metric}{R}_K{K}[_ad].
    let prior_mode = take("--prior-mode").unwrap_or("pnp".into());
    assert!(matches!(prior_mode.as_str(), "pnp" | "red" | "admm"), "--prior-mode pnp|red|admm");
    let red = prior_mode == "red";
    let admm = prior_mode == "admm";
    let admm_rho: f64 = take("--admm-rho").map_or(1.0, |v| v.parse().expect("--admm-rho R (units of ρ₁)"));
    let admm_inner: usize = take("--admm-inner").map_or(3, |v| v.parse().expect("--admm-inner K"));
    let admm_metric = take("--admm-metric").map_or(hasty_recon::admm::AdmmMetric::None, |v| hasty_recon::admm::AdmmMetric::parse(&v).expect("--admm-metric none|clip"));
    assert!(admm_rho > 0.0 && admm_inner >= 1, "--admm-rho R > 0, --admm-inner K ≥ 1");
    assert!(admm || !admm_adapt, "--admm-rho-adapt needs --prior-mode admm");
    let admm_tag = if admm { format!("_admm{}{admm_rho}_K{admm_inner}{}", admm_metric.name(), if admm_adapt { "_ad" } else { "" }) } else { String::new() };
    let red_rho: f64 = take("--red-rho").map_or(1.0, |v| v.parse().expect("--red-rho ρ"));
    let red_shifts: usize = take("--red-shifts").map_or(1, |v| v.parse().expect("--red-shifts n"));
    // A FIXED objective f + λR for the LLR prior (research/solver_choice.md §2):
    // --llr-shifts S: the LLR prox averaged over the fixed block shifts of seeds 0..S−1 in every call (the proximal
    //   average of the shifted nuclear norms: itself a prox), instead of a new shift per iteration (cycle spinning);
    // --llr-thr freeze: the penalty λ = κσ̂ / s from the first call (s = the prox step, hasty_recon::p2np::prox_step),
    //   then every call thresholds at λ·s (P²nP: s = α; ADMM: 1/ρ; --prox-metric: ατ_k), instead of κσ̂ per call;
    // --prox-metric: global ZMSHR1 with the prox in the step's metric (Becker–Fadili–Ochs; hasty_recon::p2np::
    //   set_prox_in_metric): the fixed point no longer moves with the metric. Needs --llr-thr freeze and LLR only.
    let llr_shifts: Option<usize> = take("--llr-shifts").map(|v| v.parse().expect("--llr-shifts S"));
    assert!(llr_shifts.is_none_or(|s| s >= 1), "--llr-shifts S >= 1");
    let llr_freeze = match take("--llr-thr").as_deref() {
        None | Some("adaptive") => false,
        Some("freeze") => true,
        Some(o) => panic!("--llr-thr adaptive|freeze, got {o}"),
    };
    assert!(!prox_metric || llr_freeze, "--prox-metric needs --llr-thr freeze (a fixed prox scaled by the step)");
    hasty_recon::p2np::set_prox_in_metric(prox_metric);
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
    let dfw_rule: Option<hasty_recon::dfwavelet::DfRule> = take("--dfw").or_else(|| spline_defaults.then(|| "sure".to_string())).filter(|v| v != "off").map(|v| {
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
    let static_lock: Option<(hasty_recon::tlock::LockOpts, usize, usize)> = take("--static-lock").or_else(|| spline_defaults.then(|| "20:10:5".to_string())).filter(|v| v != "off").map(|v| {
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
    // --lock-coupling additive|nested (--vmodel knots with --static-lock; hasty_recon::knotlock): additive (default) also
    // locks the breathing-dependence of the cardiac curve; nested = tlock's decomposition
    let lock_coupling = match take("--lock-coupling").as_deref() {
        None | Some("additive") => hasty_recon::knotlock::Coupling::Additive,
        Some("nested") => hasty_recon::knotlock::Coupling::Nested,
        Some(o) => panic!("--lock-coupling additive|nested, got {o}"),
    };
    // --air auto[:IT0[:EVERY]] | fixed [--air-init <npy>] [--air-margin MM] [--air-tau T] [--air-release R] [--air-blur MM[,MM..]]:
    // air map from the iterate (hasty_recon::airmap; no labels): voxels reachable from the FOV boundary through
    // low-signal voxels (reference magnitude < T × the body level, default 0.05), with a safety margin of MM (default
    // 4 mm) to anything else, estimated from iteration IT0 (default 0) and refreshed every EVERY (default 2)
    // iterations; monotone (a zeroed voxel is released, and then protected, only where the iterate puts signal above
    // R × the level back, default 0.15); x = 0 there in every frame and encode, before and after the prior.
    // --air-init: a seed on any grid of the same FOV (an air_map_*.npy, an air mask, or a coarse level's images, see
    // airmap::load_seed), applied from the first iteration; `fixed`: the seed only, never re-estimated.
    // Complex-image models only.
    let air_spec: Option<String> = take("--air");
    let air_start = air_spec.as_deref() == Some("start");
    let air: Option<(hasty_recon::airmap::AirOpts, usize, usize)> = air_spec.map(|v| {
        let it: Vec<&str> = v.split(':').collect();
        let mut o = hasty_recon::airmap::AirOpts::default();
        let (it0, every) = match it[0] {
            "auto" => (it.get(1).map_or(0, |x| x.parse().expect("--air auto[:IT0[:EVERY]]")), it.get(2).map_or(2, |x| x.parse::<usize>().expect("--air auto[:IT0[:EVERY]]")).max(1)),
            // start: estimated once, on the start stack x0 (the warm images in every frame; the same for every solver),
            // never refreshed, binary (no ramp: a projection, so the denoiser stays the prox of a fixed function)
            "start" => {
                o.ramp_mm = 0.0;
                (0, usize::MAX)
            }
            "fixed" => (usize::MAX, usize::MAX),
            _ => panic!("--air auto[:IT0[:EVERY]] | start | fixed"),
        };
        if let Some(b) = take("--air-ramp") { o.ramp_mm = b.parse().expect("--air-ramp MM"); }
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
    let seg_basis_arg = take("--seg-basis").unwrap_or(if knots_mode || frame_rm_mode { "svdn".into() } else { "fixed".into() });
    // --seg-weight count|mass[:lo,hi]: voxel weights of the SVD basis, all data-derived (no labels): count = every voxel
    // of the warm-start support equally (default); mass = the smoothed warm-start magnitude relative to its median over
    // the support, clamped to [lo, hi] (default 0.3, 3): bright regions count more, dark tissue keeps a floor so the
    // basis is not biased away from it, air (outside the support) never counts
    let seg_weight = take("--seg-weight").unwrap_or(if knots_mode || frame_rm_mode { "energy".into() } else { "count".into() });
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
    let fat_model = take("--fat-model").unwrap_or(if ratemap_spec.is_some() { "joint".into() } else { "shared".into() });
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
    let vanchor: Option<(f64, String)> = take("--vanchor").or_else(|| (spline_defaults && vpoly.is_some()).then(|| "1,auto".to_string())).filter(|v| v != "off").map(|s| { let v: Vec<&str> = s.split(',').collect(); (v[0].parse().unwrap(), v.get(1).unwrap_or(&"auto").to_string()) });
    let vanchor_refine: (f64, usize) = take("--vanchor-refine").map(|s| { let v: Vec<&str> = s.split(',').collect(); (v[0].parse().unwrap(), v.get(1).map_or(2, |d| d.parse().unwrap())) }).unwrap_or((1.5, 3));
    // learned prior (7-point dual-VENC data): export directory of tools/denoiser/export.py; κ mixes x + κ (D(x) − x)
    let learned_dir = take("--learned").map(PathBuf::from);
    let tile: Option<usize> = take("--tile").map(|s| s.parse().unwrap()).filter(|&t: &usize| t > 0);
    let keep_frames: Option<usize> = take("--keep-frames").map(|s| s.parse().unwrap()).filter(|&t: &usize| t > 0);
    // learned prior precision on CUDA: fp32 | bf16 (autocast, the training regime) | fp16
    let learned_prec = take("--learned-prec").unwrap_or("bf16".into());
    let init_from: Option<PathBuf> = take("--init-from").map(PathBuf::from);
    // --adc-part out (default) | full: the operator's ADC samples per readout (hasty_recon::pyramid::AdcPart)
    let adc_part = hasty_recon::pyramid::AdcPart::parse(&take("--adc-part").unwrap_or("out".into())).unwrap_or_else(|e| panic!("{e:?}"));
    // --calib-adc out|full: the ADC samples of the coil calibration (ESPIRiT region, --coil-gram data); default = --adc-part
    let calib_adc_out = match take("--calib-adc").as_deref() {
        None => adc_part == hasty_recon::pyramid::AdcPart::Out,
        Some("out") => true,
        Some("full") => false,
        Some(o) => panic!("--calib-adc out|full, got {o}"),
    };
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
    // every argument must have been consumed: a misspelled or repeated option would otherwise be ignored silently
    // (take() reads the first occurrence only)
    assert!(args.len() == 1, "recon_p2np: unrecognised or repeated arguments {:?} (expected only the scan directory)", &args[1.min(args.len())..]);
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
            + if init_from.is_some() { "_init" } else { "" }
            + if adc_part == hasty_recon::pyramid::AdcPart::Out { "_aout" } else { "" };
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
    // config fingerprint (audit 2026-11-24): saved states resume only under the same configuration. The arguments
    // (without --out / --fresh) and the HASTY_* / NOISE_SCALE environment are recorded in config_fingerprint.txt; a
    // state_* file with a missing or different fingerprint aborts (use --fresh or a new --out). The binary may differ
    // (a fixed build resumes a stopped run).
    {
        let mut a: Vec<String> = Vec::new();
        let mut it = std::env::args().skip(1);
        // run-length / checkpoint-cadence options may change on a resume (extend --iters, report more often)
        while let Some(x) = it.next() {
            if ["--out", "--iters", "--every", "--ckpt-every"].contains(&x.as_str()) {
                it.next();
            } else if x != "--fresh" {
                a.push(x);
            }
        }
        let mut env: Vec<String> = std::env::vars().filter(|(k, _)| (k.starts_with("HASTY_") && k != "HASTY_TUNE") || k == "NOISE_SCALE").map(|(k, v)| format!("{k}={v}")).collect();
        env.sort();
        let fp = format!("args: {}\nenv: {}\n", a.join(" "), env.join(" "));
        let fpath = out.join("config_fingerprint.txt");
        // the pooled warm starts (state_warm_full_done_*) are shared between arms on purpose (precond_compare/run.sh
        // copies them, --warm-only): not checked here; every other state_* is this run's own iterate
        let names: Vec<String> = std::fs::read_dir(&out).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        let has_state = names.iter().any(|n| n.starts_with("state_") && !n.starts_with("state_warm_full_done_"));
        if names.iter().any(|n| n.starts_with("state_warm_full_done_")) {
            println!("resume: pooled warm-start states state_warm_full_done_* present (shared between arms, not fingerprint-checked)");
        }
        if has_state {
            match std::fs::read_to_string(&fpath) {
                Ok(old) if old == fp => println!("resume: saved state with the same configuration fingerprint"),
                Ok(old) => panic!("{}: saved state from a different configuration (use --fresh or a new --out)\n  saved:   {}\n  current: {}", out.display(), old.trim(), fp.trim()),
                Err(_) => panic!("{}: saved state without a configuration fingerprint (written before 2026-11-24); use --fresh or a new --out", out.display()),
            }
        }
        std::fs::write(&fpath, &fp).unwrap();
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
    // --card-frames K: K nested cardiac frames (K divides the sim's frames; readout frame f → f / (F / K)), e.g. the
    // coarse level of a temporal anchor test (temporal_pyramid.md §13). The legacy lumen yardstick indexes the truth
    // by frame and is not meaningful then; --score-pc averages its truth over the nested frames.
    if let Some(k) = card_frames {
        let f0 = sd.n_frames;
        assert!(k > 0 && f0 % k == 0, "--card-frames {k}: must divide the {f0} cardiac frames");
        let m = f0 / k;
        for f in sd.ro.frame.iter_mut() {
            if *f != usize::MAX {
                *f /= m;
            }
        }
        sd.n_frames = k;
        println!("--card-frames {k}: {f0} cardiac frames pooled {m} by {m} (nested)");
    }
    // respiratory bins: frame index r · F + f
    let nf_card = sd.n_frames;
    let mut resp_info = serde_json::Value::Null;
    if resp_bins > 1 || resp_accept < 1.0 {
        assert!(vmodel.is_none() && (temporal.is_none() || resp_bins == 1), "--resp-bins / --resp-accept with --vmodel, or --resp-bins > 1 with --temporal: not yet (their time bases are cyclic over the cardiac frames; --temporal with gating to one bin works: the gated readouts keep their cardiac times)");
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
    if keep_n > 1 {
        // the same subset as splinerecon --keep N (knotsolve::subset_mask, salt "KEEP")
        let mask = hasty_recon::knotsolve::subset_mask(sd.n_readouts(), keep_n, 0x4B45_4550);
        let mut kept = 0usize;
        for (i, &k) in mask.iter().enumerate() {
            if k {
                kept += (sd.ro.frame[i] != usize::MAX) as usize;
            } else {
                sd.ro.frame[i] = usize::MAX;
            }
        }
        println!("--keep {keep_n}: {kept} of {} readouts kept (a shorter scan; hash-spread over all strata)", sd.n_readouts());
    }
    // --holdout-res: the frame of every held-out readout (usize::MAX: not held out)
    let mut held_frame: Vec<usize> = vec![usize::MAX; if holdout_res { sd.n_readouts() } else { 0 }];
    assert!(!holdout_res || holdout.is_some_and(|h| !h.2), "--holdout-res needs --holdout N (not --holdout-eval)");
    if let Some((hn, hs, ev)) = holdout {
        // stratified: every N-th readout of each (encode, frame) set from a hashed offset, so that every set keeps
        // held-out readouts (≥ 1 when it has any) and the eval run has no empty set
        let mut mask = vec![false; sd.n_readouts()];
        let ne_h = 1 + sd.ro.enc.iter().copied().max().unwrap_or(0);
        let off_h = hasty_recon::knotsolve::subset_mask(ne_h * sd.n_frames * hn, hn, 0x484F_4C44 ^ hs.wrapping_mul(0x9E37_79B9));
        for e in 0..ne_h {
            for f in 0..sd.n_frames {
                let r = sd.readouts_of(e, Some(f));
                let base = (e * sd.n_frames + f) * hn;
                let off = (0..hn).find(|&j| off_h[base + j]).unwrap_or(0) % hn.min(r.len().max(1));
                for (j, &i) in r.iter().enumerate() {
                    mask[i] = j % hn == off;
                }
            }
        }
        let (mut held, mut left) = (0usize, 0usize);
        for (i, &h) in mask.iter().enumerate() {
            if sd.ro.frame[i] == usize::MAX {
                continue;
            }
            if h == ev {
                left += 1;
            } else {
                held += 1;
                if holdout_res {
                    held_frame[i] = sd.ro.frame[i];
                }
                sd.ro.frame[i] = usize::MAX;
            }
        }
        if ev {
            println!("--holdout-eval {hn}:{hs}: only the {left} held-out readouts ({held} solve readouts dropped): data-space score");
        } else {
            println!("--holdout {hn}:{hs}: {held} readouts held out of the solve, {left} used");
        }
    }
    // knot mode: the 2-D basis over the breathing amplitude of the readouts used (bellows by default, truth = oracle)
    let knot_basis: Option<(hasty_recon::vknot::KnotBasis, Vec<(u32, f32)>, Vec<f64>)> = knots2d.map(|(kr, rdeg, mc, cdeg)| {
        assert!(model == "seg" && temporal.is_none() && resp_bins == 1 && resp_accept == 1.0 && !bin_group, "--vmodel knots: --model seg, without --temporal / --resp-bins / --resp-accept / --bin-group (the knots are the time basis)");
        let (file, col) = match resp_src.as_str() { "bellows" => ("physio.npy", 3), "truth" => ("resp.npy", 0), o => panic!("--resp bellows|truth, got {o}") };
        let a = hasty_recon::data::read_npy(&dir.join(file)).unwrap_or_else(|e| panic!("{file}: {e:?}"));
        let (w, v) = (a.shape[1], a.f64s());
        let nro = sd.n_readouts();
        assert_eq!(a.shape[0], nro, "{file}: one row per readout");
        let amp: Vec<f64> = (0..nro).map(|i| v[i * w + col]).collect();
        let used: Vec<usize> = (0..nro).filter(|&i| sd.ro.frame[i] != usize::MAX).collect();
        let amp_used: Vec<f64> = used.iter().map(|&i| amp[i]).collect();
        let b = hasty_recon::vknot::KnotBasis::from_amplitude(&amp_used, kr, rdeg, mc, cdeg);
        // per readout (interval, fraction); bins by rank among the used readouts
        let mut per = vec![(0u32, 0f32); nro];
        for (j, x) in b.assign(&amp_used).into_iter().enumerate() {
            per[used[j]] = x;
        }
        println!("knot basis ({resp_src}{}): {}", if resp_src == "truth" { ", ORACLE breathing coordinate" } else { "" }, b.describe());
        (b, per, amp)
    });
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
    // spline mode (--vop frames): t = e · ng + g, (Some(g), e) = encode e, respiratory group g (vknot::OpFrames)
    let fe: Vec<(Option<usize>, usize)> = if let Some(of) = &op_frames {
        (0..ne).flat_map(|e| (0..of.ng()).map(move |g| (Some(g), e))).collect()
    } else if temporal.is_some() || knots2d.is_some() { (0..ne).map(|e| (None, e)).collect() } else { (0..nf).flat_map(|f| (0..ne).map(move |e| (Some(f), e))).collect() };
    // the readouts of a set (readouts in no set, frame usize::MAX: gated out or not kept by --keep, never enter);
    // spline mode: the readouts of encode e whose respiratory interval lies in group g
    let set_readouts = |f: Option<usize>, e: usize| -> Vec<usize> {
        if let (Some(of), Some((_, per, _))) = (&op_frames, &knot_basis) {
            let mut r = sd.readouts_of(e, None);
            r.retain(|&i| sd.ro.frame[i] != usize::MAX && of.group_of(per[i].0).0 == f.unwrap());
            return r;
        }
        let mut r = sd.readouts_of(e, f); r.retain(|&i| sd.ro.frame[i] != usize::MAX); r
    };
    let enc_of = |t: usize| fe[t].1;
    let t_load = Instant::now();
    let mut host_sets: Vec<SampleSet> = if streamed {
        // coordinates, τ and concomitant coefficients now; the data after the coil compression is known
        let meta_only = LoadOpts { w: &w, cc: None, keep: None, data: false };
        fe.iter().map(|&(f, e)| sd.load_set(&set_readouts(f, e), &meta_only).unwrap()).collect()
    } else {
        std::thread::scope(|sc| {
            let hs: Vec<_> = fe.iter().map(|&(f, e)| { let (sd, w, sr) = (&sd, &w, &set_readouts); sc.spawn(move || sd.sample_set(&sr(f, e), w)) }).collect();
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
    // --calib-adc out: the calibration sees the operator's ADC samples only (the outgoing half [i0, i1) of every readout,
    // detected here as for the solve below)
    let calib_win: Option<(usize, usize)> = calib_adc_out.then(|| {
        let cs: Vec<&[Vec<f32>; 3]> = host_sets.iter().map(|s| &s.coords).collect();
        let oh = hasty_recon::pyramid::OutHalf::detect(&cs, sd.na, sd.vox);
        (oh.i0, oh.i1)
    });
    let calib_raw: Option<(Vec<C32>, usize)> = (coil_gram == "calib" || coil_box_margin.is_some() || smaps_spec != "truth").then(|| {
        let t_c = std::time::Instant::now();
        let (mut kc, mut ky): ([Vec<f32>; 3], Vec<Vec<C32>>) = ([vec![], vec![], vec![]], vec![vec![]; nc]);
        // samples in the calibration box over the whole ADC (counted for the log), and the kept ones
        let n_full = std::sync::atomic::AtomicUsize::new(0);
        let in_box = |k: &[f32; 3]| {
            let b = (0..3).all(|a| (k[a] as f64 * sd.dims[a] as f64 / cal_dims[a] as f64).abs() < std::f64::consts::PI * (1.0 - 1e-6));
            if b {
                n_full.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            b
        };
        let na_ = sd.na;
        let in_win = |j: usize| calib_win.map_or(true, |(w0, w1)| (w0..w1).contains(&(j % na_)));
        for t in (0..fe.len()).filter(|&t| enc_of(t) == 0) {
            if streamed {
                let s = sd.load_set_window(&host_sets[t].readouts, &LoadOpts { w: &w, cc: None, keep: Some(&in_box), data: true }, calib_win).unwrap();
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
            let sel: Vec<usize> = (0..m).filter(|&j| in_box(&[hsx.coords[0][j], hsx.coords[1][j], hsx.coords[2][j]]) && in_win(j)).collect();
            for a in 0..3 {
                kc[a].extend(sel.iter().map(|&j| hsx.coords[a][j]));
            }
            for c in 0..nc {
                ky[c].extend(sel.iter().map(|&j| hsx.y[c * m + j]));
            }
        }
        let yall: Vec<C32> = ky.concat();
        let (cal, used) = hasty_recon::espirit::calib_from_noncart(&dev, &st, &kern, [&kc[0], &kc[1], &kc[2]], &yall, nc, sd.dims, cal_dims, 30, 1e-3).unwrap();
        let given = kc[0].len();
        match calib_win {
            Some((w0, w1)) => println!("calibration region {cal_dims:?}: {used} samples of encode 0, {nc} coils ({:.1}s); --calib-adc out: ADC samples {w0}..{w1} of {na_} per readout, {given} of the {} samples inside the region over the whole ADC", t_c.elapsed().as_secs_f64(), n_full.load(std::sync::atomic::Ordering::Relaxed)),
            None => println!("calibration region {cal_dims:?}: {used} samples of encode 0, {nc} coils ({:.1}s); --calib-adc full: every ADC sample ({given} inside the region)", t_c.elapsed().as_secs_f64()),
        }
        (cal, used)
    });
    // --smaps-resp: the same calibration per respiratory bin (the encode-0 sets of frames r · nf_card + c, c any)
    let calib_resp: Vec<(Vec<C32>, usize)> = if smaps_resp {
        assert!(resp_bins > 1, "--smaps-resp: needs --resp-bins N > 1");
        assert!(fe.iter().all(|f| f.0.is_some()), "--smaps-resp: one frame per set (no --temporal / knot sets)");
        (0..resp_bins)
            .map(|b| {
                let t_c = std::time::Instant::now();
                let (mut kc, mut ky): ([Vec<f32>; 3], Vec<Vec<C32>>) = ([vec![], vec![], vec![]], vec![vec![]; nc]);
                let in_box = |k: &[f32; 3]| (0..3).all(|a| (k[a] as f64 * sd.dims[a] as f64 / cal_dims[a] as f64).abs() < std::f64::consts::PI * (1.0 - 1e-6));
                let na_ = sd.na;
                let in_win = |j: usize| calib_win.map_or(true, |(w0, w1)| (w0..w1).contains(&(j % na_)));
                for t in (0..fe.len()).filter(|&t| enc_of(t) == 0 && fe[t].0.is_some_and(|f| f / nf_card == b)) {
                    if streamed {
                        let s = sd.load_set_window(&host_sets[t].readouts, &LoadOpts { w: &w, cc: None, keep: Some(&in_box), data: true }, calib_win).unwrap();
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
                    let sel: Vec<usize> = (0..m).filter(|&j| in_box(&[hsx.coords[0][j], hsx.coords[1][j], hsx.coords[2][j]]) && in_win(j)).collect();
                    for a in 0..3 {
                        kc[a].extend(sel.iter().map(|&j| hsx.coords[a][j]));
                    }
                    for c in 0..nc {
                        ky[c].extend(sel.iter().map(|&j| hsx.y[c * m + j]));
                    }
                }
                let yall: Vec<C32> = ky.concat();
                let (cal, used) = hasty_recon::espirit::calib_from_noncart(&dev, &st, &kern, [&kc[0], &kc[1], &kc[2]], &yall, nc, sd.dims, cal_dims, 30, 1e-3).unwrap();
                println!("calibration region {cal_dims:?}, respiratory bin {b}: {used} samples of encode 0 ({:.1}s)", t_c.elapsed().as_secs_f64());
                (cal, used)
            })
            .collect()
    } else {
        vec![]
    };
    let ncal = cal_dims[0] * cal_dims[1] * cal_dims[2];
    let mut coil_info;
    let cc = {
        let mut g = hasty_recon::coils::CoilGram::new(nc);
        let mut box_mask: Option<Vec<bool>> = None;
        let what = if coil_gram == "calib" {
            g.add(&calib_raw.as_ref().unwrap().0, ncal, 1);
            "whitened calibration region"
        } else if let Some(margin) = coil_box_margin {
            let pb = presc_box_arg.as_deref().map(|s| hasty_recon::coils::PrescBox::parse(s).unwrap()).or_else(|| hasty_recon::coils::PrescBox::from_meta(&sd.meta)).expect("--coil-gram box: needs --presc-box S|SX,SY,SZ[@CX,CY,CZ] (mm) or presc_box in the scan metadata");
            let fov = [0, 1, 2].map(|a| sd.dims[a] as f64 * sd.vox[a]);
            let mk = pb.mask(cal_dims, fov, margin);
            println!("coil Gram: prescription box {:?} mm at {:?} mm + {:.0} mm margin: {} of {ncal} calibration voxels", pb.size.map(|v| v * 1e3), pb.center.map(|v| v * 1e3), margin * 1e3, mk.iter().filter(|&&v| v).count());
            g = hasty_recon::coils::masked_gram(&calib_raw.as_ref().unwrap().0, ncal, nc, &mk);
            box_mask = Some(mk);
            "whitened calibration region inside the prescription box"
        } else {
            // the whitened data of set 0 (≤ 20000 samples, as before); --calib-adc out: its ADC samples [i0, i1) only
            let m0 = host_sets[0].tau.len();
            let y0: Vec<C32> = if streamed {
                sd.load_set_window(&host_sets[0].readouts, &LoadOpts { w: &w, cc: None, keep: None, data: true }, calib_win).unwrap().y
            } else if let Some((w0, w1)) = calib_win {
                let sel: Vec<usize> = (0..m0).filter(|&j| (w0..w1).contains(&(j % sd.na))).collect();
                let hy = &host_sets[0].y;
                (0..nc).flat_map(|c| sel.iter().map(move |&j| hy[c * m0 + j])).collect()
            } else {
                host_sets[0].y.clone()
            };
            let m = y0.len() / nc;
            println!("coil Gram from set 0: {m} of {m0} samples (--calib-adc {})", if calib_win.is_some() { "out" } else { "full" });
            g.add(&y0, m, (m / 20000).max(1));
            "whitened samples of set 0"
        };
        let ccx = hasty_recon::coils::CoilCompression::from_gram(&g, coil_rule);
        println!("coil compression ({what}): {}", ccx.report());
        coil_info = serde_json::json!({"gram": coil_gram, "rule": coil_rule.to_string(), "kept": ccx.nvc(), "of": nc, "retained": ccx.retained, "eig_rel": ccx.eig.iter().map(|v| v / ccx.eig.iter().sum::<f64>().max(1e-300)).collect::<Vec<_>>()});
        if let (Some(mk), Some((cal, _))) = (&box_mask, &calib_raw) {
            // what the box-ordered coils keep inside the box (per-voxel loss) and of the whole calibration region
            let (kin, p50, p99, mx) = hasty_recon::coils::masked_retention(&ccx, cal, ncal, Some(mk));
            let (kall, ..) = hasty_recon::coils::masked_retention(&ccx, cal, ncal, None);
            println!("  box-ordered coils: {:.3} % of the box energy (per-voxel loss p50 {:.1e}, p99 {:.1e}, max {:.1e}), {:.2} % of the whole calibration region", 100.0 * kin, p50, p99, mx, 100.0 * kall);
            coil_info["box"] = serde_json::json!({"kept_box": kin, "loss_p50": p50, "loss_p99": p99, "loss_max": mx, "kept_all": kall});
        }
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
    // --cc-load: the solve run's basis (held-out residual in the same virtual coils); --holdout saves it
    let cc: Vec<Vec<Complex64>> = match &cc_load {
        Some(p) => {
            let s = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("--cc-load {}: {e}", p.display()));
            let z: Vec<Vec<Complex64>> = s.lines().filter(|l| !l.trim().is_empty()).map(|l| l.split(';').filter(|p| !p.is_empty()).map(|p| {
                let (a, b) = p.split_once(',').unwrap();
                Complex64::new(a.parse().unwrap(), b.parse().unwrap())
            }).collect()).collect();
            assert!(z.iter().all(|r| r.len() == nc), "--cc-load {}: rows of {nc} coils expected", p.display());
            println!("coil compression: basis loaded from {} ({} virtual coils)", p.display(), z.len());
            z
        }
        None => cc,
    };
    if holdout.is_some_and(|h| !h.2) {
        let s: String = cc.iter().map(|r| r.iter().map(|z| format!("{:.9e},{:.9e};", z.re, z.im)).collect::<String>() + "\n").collect();
        std::fs::write(out.join("coil_basis.txt"), s).unwrap();
    }
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
    if field_scale != 1.0 {
        grid.omega.iter_mut().for_each(|w| *w *= field_scale);
        println!("--field-scale {field_scale}: the operator's off-resonance field scaled (concomitant fields and GNL unchanged)");
    }
    let ff_est: Option<Vec<f32>> = fat_fraction.as_deref().and_then(|s| s.strip_prefix("est:")).map(|spec| {
        let fm = est_maps(spec);
        xs.iter().map(|x| fm.sample(x).2.clamp(0.0, 1.0) as f32).collect()
    });
    let smaps_true_cc = apply_cc(&grid.smaps, nv, nc, &cc);
    // ESPIRiT eigenvalue maps (pooled; per respiratory bin with --smaps-resp: maps, eigenvalues), for --smaps-fit
    let mut espirit_eig: Option<Vec<f32>> = None;
    let mut maps_resp: Vec<(Vec<C32>, Vec<f32>)> = Vec::new();
    let mut smaps = if let Some(arg) = smaps_spec.strip_prefix("espirit") {
        // the calibration region compressed (it is linear in the data)
        let cd = cal_dims;
        let mut opts = hasty_recon::espirit::EspiritOpts { eig_min: arg.strip_prefix(':').map_or(0.8, |v| v.parse().unwrap()), ..Default::default() };
        if let Some((_, _, e0)) = smaps_fit {
            // the fit weights the maps by their eigenvalue down to E0: keep them down to there
            opts.eig_min = e0;
        }
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
        if smaps_resp {
            // one map set per respiratory bin; the subspace iteration starts at the pooled kernel count
            let ob = hasty_recon::espirit::EspiritOpts { block0: Some(em.n_kernels * 5 / 4 + 16), ..opts.clone() };
            let mut rows = Vec::new();
            for (b, (cal_b, used_b)) in calib_resp.iter().enumerate() {
                let t_b = std::time::Instant::now();
                let eb = hasty_recon::espirit::espirit(&apply_cc(cal_b, ncal, nc, &cc), cd, cc.len(), sd.dims, &ob);
                let mut mb = eb.maps;
                // the image phase convention of the pooled maps (no gauge phase jumps between bins)
                hasty_recon::smapfit::align_phase(&mut mb, &em.maps, cc.len());
                let (e_all, e_nz, zero) = smaps_agreement(&mb, &smaps_true_cc, cc.len(), &sig);
                let d_pool = hasty_recon::smapfit::map_disagreement(&mb, &em.maps, cc.len());
                let both: Vec<usize> = (0..nv).filter(|&q| sig[q] && eb.eig[q] as f64 > opts.eig_min && em.eig[q] as f64 > opts.eig_min).collect();
                let dp = both.iter().map(|&q| d_pool[q]).sum::<f64>() / both.len().max(1) as f64;
                println!("coil maps, respiratory bin {b}: ESPIRiT ({used_b} samples, {} kernels, {:.1}s); vs the true maps: mean 1 − |⟨ŝ, s⟩| {e_all:.2e} (zero maps as 1), {e_nz:.2e} over the {} nonzero, {zero} zero; vs the pooled maps {dp:.2e} over {} voxels where both are kept", eb.n_kernels, t_b.elapsed().as_secs_f64(), cnt - zero, both.len());
                rows.push(serde_json::json!({"bin": b, "kernels": eb.n_kernels, "calib_samples": used_b, "mean_1_minus_ip": e_all, "mean_1_minus_ip_nonzero": e_nz, "zero": zero, "vs_pooled": dp}));
                maps_resp.push((mb, eb.eig));
            }
            diag.insert("espirit_resp".into(), serde_json::json!(rows));
        }
        espirit_eig = Some(em.eig);
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
    // --kcrop: the samples inside the grid's band; --adc-part out: the outgoing half of every readout (the returning
    // half stays in the files and the store for navigators); both: their intersection. The sample records are cropped
    // (everything downstream sees the kept samples); the whole records stay for the segment tables (built per
    // readout × ADC sample, then cropped) and the segment time window is the kept ADC window; the store (keyed and
    // filled with the whole sets, shared with the full-resolution runs) is filtered on read.
    let mut full_sets: Vec<SampleSet> = vec![];
    let mut crop_sel: Option<Vec<Vec<u32>>> = None;
    let mut crop_adc: Option<hasty_recon::pyramid::AdcStats> = None;
    let out_half: Option<hasty_recon::pyramid::OutHalf> = (adc_part == hasty_recon::pyramid::AdcPart::Out).then(|| {
        let t_o = Instant::now();
        let cs: Vec<&[Vec<f32>; 3]> = host_sets.iter().map(|s| &s.coords).collect();
        let oh = hasty_recon::pyramid::OutHalf::detect(&cs, sd.na, sd.vox);
        let tau = &host_sets[0].tau;
        let ms = |n: usize| tau[n] * 1e3;
        let ros: Vec<usize> = host_sets.iter().map(|s| s.readouts.len()).collect();
        println!("ADC part out (outgoing half, |k| increasing): samples {}..{} of {} kept per readout ({:.3}..{:.3} ms after excitation; ADC {:.3}..{:.3} ms), {:.2} % of the samples; per-readout first centre pass {}..{}, first |k| maximum {}..{} over {} readouts; {} decreasing steps inside the readouts' ranges; {} .. {} readouts and {} .. {} samples per set ({:.1}s)",
            oh.i0, oh.i1, sd.na, ms(oh.i0), ms(oh.i1 - 1), ms(0), ms(sd.na - 1), 100.0 * (oh.i1 - oh.i0) as f64 / sd.na as f64, oh.i0_span.0, oh.i0_span.1, oh.i1_span.0 - 1, oh.i1_span.1 - 1, oh.readouts, oh.decreasing,
            ros.iter().min().unwrap_or(&0), ros.iter().max().unwrap_or(&0), ros.iter().min().unwrap_or(&0) * (oh.i1 - oh.i0), ros.iter().max().unwrap_or(&0) * (oh.i1 - oh.i0), t_o.elapsed().as_secs_f64());
        pyr_info["adc_part"] = serde_json::json!({ "part": "out", "i0": oh.i0, "i1": oh.i1, "na": sd.na, "t_ms": [ms(oh.i0), ms(oh.i1 - 1)], "i0_span": oh.i0_span, "i1_span": oh.i1_span, "readouts": oh.readouts, "decreasing_steps": oh.decreasing, "fraction": (oh.i1 - oh.i0) as f64 / sd.na as f64 });
        oh
    });
    if out_half.is_none() {
        pyr_info["adc_part"] = serde_json::json!({ "part": "full", "na": sd.na });
    }
    if kcrop.is_some() || out_half.is_some() {
        use rayon::prelude::*;
        let t_c = Instant::now();
        let sel: Vec<Vec<u32>> = host_sets.par_iter().map(|s| match (kcrop, &out_half) {
            (Some(kc), oh) => {
                let mut v = kc.select(&s.coords, sd.na);
                if let Some(oh) = oh {
                    oh.restrict(&mut v, sd.na);
                }
                v
            }
            (None, Some(oh)) => oh.select(s.coords[0].len(), sd.na),
            (None, None) => unreachable!(),
        }).collect();
        let lens_full: Vec<usize> = host_sets.iter().map(|s| s.coords[0].len()).collect();
        let st = hasty_recon::pyramid::AdcStats::new(&sel, &lens_full, sd.na);
        let (w0, w1) = st.window;
        let (t_w0, t_w1) = (host_sets[0].tau[w0] * 1e3, host_sets[0].tau[w1] * 1e3);
        let what = match (kcrop, &out_half) {
            (Some(kc), Some(_)) => format!("{} ∩ outgoing half", kc.describe()),
            (Some(kc), None) => kc.describe(),
            _ => "outgoing half".to_string(),
        };
        let cropped: Vec<SampleSet> = host_sets.par_iter().zip(sel.par_iter()).map(|(s, i)| hasty_recon::pyramid::crop_set(s, i)).collect();
        full_sets = std::mem::replace(&mut host_sets, cropped);
        for s in full_sets.iter_mut() {
            s.y = Vec::new();
        }
        if model != "seg" {
            full_sets = vec![];
        }
        println!("k-space crop ({what}): {} of {} samples kept ({:.2} %), ADC window {w0}..{w1} of {} ({t_w0:.3}..{t_w1:.3} ms), {} .. {} per set ({:.1}s)", st.kept, st.total, 100.0 * st.fraction(), sd.na, sel.iter().map(|s| s.len()).min().unwrap_or(0), sel.iter().map(|s| s.len()).max().unwrap_or(0), t_c.elapsed().as_secs_f64());
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
        if let Some(kc) = kcrop {
            pyr_info["kcrop"] = serde_json::json!({ "frac": kc.frac, "prefix": kc.prefix, "kept": st.kept, "total": st.total, "fraction": st.fraction(), "adc_window": [w0, w1], "window_ms": [t_w0, t_w1] });
        }
        pyr_info["adc_part"]["kept"] = serde_json::json!({ "samples": st.kept, "total": st.total, "fraction": st.fraction(), "adc_window": [w0, w1], "window_ms": [t_w0, t_w1], "per_set": [sel.iter().map(|s| s.len()).min().unwrap_or(0), sel.iter().map(|s| s.len()).max().unwrap_or(0)] });
        crop_sel = Some(sel);
        crop_adc = Some(st);
    }
    // (sizes below: the kept samples)
    let samples_total: usize = host_sets.iter().map(|s| s.coords[0].len()).sum();
    // --holdout-res: the held-out readouts of every solve set (same frame and encode): whole sample records (for the
    // segment tables, built with the solve's rules below), the kept-sample selection of the same --kcrop / --adc-part
    // rule, and the kept data (whitened, the solve's coil basis) on the host
    let mut held_host: Option<HeldHost> = holdout_res.then(|| {
        assert!(model == "seg" && knots2d.is_none() && op_frames.is_none() && vmodel.is_none() && inject.is_none() && !synthetic && rc_mode.is_off() && fat_fraction.is_none(),
            "--holdout-res: the frame-mode or --temporal segment model (no knots / --vmodel / --inject-phase / --synthetic / --rcompress / --fat-fraction)");
        let t_h = Instant::now();
        let opts = LoadOpts { w: &w, cc: Some(&cc), keep: None, data: true };
        let mut hh = HeldHost { recs: Vec::with_capacity(fe.len()), sels: Vec::with_capacity(fe.len()), ys: Vec::with_capacity(fe.len()), nro: 0 };
        for &(f, e) in &fe {
            // frame mode: the set's frame; --temporal (one set per encode over all frames): every held-out readout of e
            let ros: Vec<usize> = (0..sd.n_readouts()).filter(|&i| sd.ro.enc[i] == e && match f { Some(f) => held_frame[i] == f, None => held_frame[i] != usize::MAX }).collect();
            hh.nro += ros.len();
            let mut s = if streamed {
                sd.load_set(&ros, &opts).unwrap()
            } else {
                let mut s = sd.sample_set(&ros, &w);
                s.y = apply_cc(&s.y, s.tau.len(), nc, &cc);
                s
            };
            let sel: Option<Vec<u32>> = match (kcrop, &out_half) {
                (Some(kc), oh) => {
                    let mut v = kc.select(&s.coords, sd.na);
                    if let Some(oh) = oh {
                        oh.restrict(&mut v, sd.na);
                    }
                    Some(v)
                }
                (None, Some(oh)) => Some(oh.select(s.coords[0].len(), sd.na)),
                (None, None) => None,
            };
            let m = s.coords[0].len();
            let y = std::mem::take(&mut s.y);
            hh.ys.push(match &sel { Some(v) => hasty_recon::pyramid::gather_rows(&y, m, v), None => y });
            hh.recs.push(s);
            hh.sels.push(sel);
        }
        let kept: usize = hh.ys.iter().map(|y| y.len() / nvc).sum();
        println!("--holdout-res: {} held-out readouts in {} sets, {kept} kept samples, {:.2} GiB data on the host ({:.1}s)", hh.nro, fe.len(), (kept * nvc * 8) as f64 / GIB, t_h.elapsed().as_secs_f64());
        hh
    });
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
        if let Some((kr, rdeg, mc, cdeg)) = knots2d { format!("{ne} encodes x {kr}·{mc} knots (respiratory degree {rdeg}, cardiac degree {cdeg})") } else if temporal.is_some() { format!("{ne} encodes x {nf} frames, temporal B-spline {}", temporal.unwrap()) } else { format!("{nf} frames x {ne} encodes") },
        host_sets.iter().map(|s| s.readouts.len()).min().unwrap(),
        host_sets.iter().map(|s| s.readouts.len()).max().unwrap(),
        nvc,
        if gnl { " + GNL" } else { "" },
        t0.elapsed().as_secs_f64()
    );
    // the data term's noise floor (measured data with the sim's noise; whitening with the normalised Ψ keeps σ):
    // ½ E‖n‖² = (σ² / 2) · samples · virtual coils. A model without mismatch ends near it (the inverse crime does).
    if let Some(sig) = sd.meta["noise_sigma"].as_f64().filter(|v| *v > 0.0) {
        let m: usize = ms.iter().sum();
        println!("data-term noise floor ½E‖n‖² ≈ {:.4e} (sim noise σ {sig:.2} per complex sample, {m} samples × {nvc} virtual coils in the solve)", 0.5 * sig * sig * (m * nvc) as f64);
    }
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
    // estimated maps against the truth over the object support (research/smaps_estimation.md: the whole-FOV number
    // above counts every air voxel with a zero map as error 1); --smaps-fit / --smaps-resp maps
    if smaps_spec != "truth" {
        let (e_all, e_nz, zero) = smaps_agreement(&smaps, &smaps_true_cc, nvc, &support);
        let ns = support.iter().filter(|&&b| b).count();
        println!("coil maps vs the true maps over the {ns} support voxels: mean 1 − |⟨ŝ, s⟩| {e_all:.2e} (zero maps as 1), {e_nz:.2e} over the {} nonzero, {zero} zero", ns - zero);
        diag.insert("smaps_support".into(), serde_json::json!({"voxels": ns, "mean_1_minus_ip": e_all, "mean_1_minus_ip_nonzero": e_nz, "zero": zero}));
    }
    if let (Some((h, lam, e0)), Some(eig)) = (smaps_fit, espirit_eig.as_ref()) {
        let t_f = Instant::now();
        let basis = hasty_recon::smapfit::Bspline3::new(sd.dims, sd.vox, h);
        let fo = hasty_recon::smapfit::FitOpts { h, lambda: lam, ..Default::default() };
        let fit_one = |m: &[C32], e: &[f32]| -> (Vec<C32>, hasty_recon::smapfit::FitReport) {
            let wgt = hasty_recon::smapfit::eig_weights(e, &dil, e0);
            let (mut f, rep) = hasty_recon::smapfit::fit_maps(&basis, m, nvc, &wgt, &fo);
            hasty_recon::smapfit::normalise_rss(&mut f, nvc, &maps_mask);
            (f, rep)
        };
        let (fit, rep) = fit_one(&smaps[..], &eig[..]);
        let (e_all, e_nz, zero) = smaps_agreement(&fit, &smaps_true_cc, nvc, &support);
        println!("--smaps-fit bspline {:.0} mm (λ {lam:.1e}, E0 {e0}): {} coefficients {:?} per coil, CG {} its (rel. res. {:.1e}), weighted misfit {:.2e} ({:.1}s); vs the true maps over the support: mean 1 − |⟨ŝ, s⟩| {e_all:.2e}, {e_nz:.2e} over the nonzero, {zero} zero", h * 1e3, rep.ncoef, rep.kdims, rep.iters, rep.rel_res, rep.misfit, t_f.elapsed().as_secs_f64());
        diag.insert("smaps_fit".into(), serde_json::json!({"h_mm": h * 1e3, "lambda": lam, "e0": e0, "coefficients": rep.ncoef, "kdims": rep.kdims, "cg_iters": rep.iters, "misfit": rep.misfit, "mean_1_minus_ip": e_all, "mean_1_minus_ip_nonzero": e_nz, "zero": zero}));
        smaps = fit;
        for (b, (m, e)) in maps_resp.iter_mut().enumerate() {
            let (mut f, rep) = fit_one(&m[..], &e[..]);
            hasty_recon::smapfit::align_phase(&mut f, &smaps, nvc);
            let (e_all, _, _) = smaps_agreement(&f, &smaps_true_cc, nvc, &support);
            let dp = hasty_recon::smapfit::map_disagreement(&f, &smaps, nvc);
            let dpm = (0..nv).filter(|&q| support[q]).map(|q| dp[q]).sum::<f64>() / (0..nv).filter(|&q| support[q]).count().max(1) as f64;
            println!("--smaps-fit, respiratory bin {b}: misfit {:.2e}; vs the true maps over the support {e_all:.2e}; vs the pooled fit {dpm:.2e}", rep.misfit);
            *m = f;
        }
        op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&host_sets[0].coords[0], &host_sets[0].coords[1], &host_sets[0].coords[2]], &smaps, nvc, vec![], 1e-4).unwrap();
    } else if mask_maps {
        for (m, _) in maps_resp.iter_mut() {
            for q in (0..nv).filter(|&q| !dil[q]) {
                for c in 0..nvc {
                    m[c * nv + q] = C32::new(0.0, 0.0);
                }
            }
        }
    }
    let maps: Vec<CudaSlice<C32>> = maps_host.iter().map(|m| st.clone_htod(m).unwrap()).collect();
    // lumen and truth at the true positions of the reconstruction voxels
    let lumen: Vec<usize> = (0..nv).filter(|&q| sample_truth(&sd.truth_frac, 0, &grid.x_log[q]) > 0.9).collect();
    println!("lumen {} voxels: {} in the support, {} in the masked maps", lumen.len(), lumen.iter().filter(|&&q| support[q]).count(), lumen.iter().filter(|&&q| maps_mask[q]).count());
    // --score-pc: the PC signal truth on this grid + the oracle checks of the score (scoring only)
    let pc_nat = score_pc.as_ref().map(|(d, _)| hasty_recon::pcscore::PcNative::open(d, &sd).unwrap_or_else(|e| panic!("--score-pc {}: {e}", d.display())));
    let pc_truth = score_pc.as_ref().map(|(d, covmin)| {
        let t_p = Instant::now();
        let nat = pc_nat.as_ref().unwrap();
        let t = match &score_pc_bins {
            None => hasty_recon::pcscore::PcTruth::build(nat, &sd, &grid.x_log, &dec, *covmin).coarsen(nf_card, &dec),
            Some(dirs) => {
                assert_eq!(dirs.len() * nf_card, nf, "--score-pc-bins: {} directories for {} respiratory bins", dirs.len(), nf / nf_card);
                let mut per_state: Vec<f64> = Vec::new();
                let per: Vec<hasty_recon::pcscore::PcTruth> = dirs
                    .iter()
                    .map(|d| {
                        let nb = hasty_recon::pcscore::PcNative::open(d, &sd).unwrap_or_else(|e| panic!("--score-pc-bins {}: {e}", d.display()));
                        // the directory's breathing state (pc_resp_state.txt, pc_native.py) must be its bin's true state
                        let b = per_state.len();
                        let st_d: f64 = std::fs::read_to_string(d.join("pc_resp_state.txt")).unwrap_or_else(|_| panic!("--score-pc-bins {}: no pc_resp_state.txt (rerun pc_native.py)", d.display())).trim().parse().expect("pc_resp_state.txt");
                        let st_b = resp_info["true_state"][b].as_f64().unwrap_or_else(|| panic!("--score-pc-bins: no true breathing state of bin {b} (physio.npy)"));
                        assert!((st_d - st_b).abs() < 0.02, "--score-pc-bins {}: truth at breathing state {st_d:.3}, bin {b} is at {st_b:.3}", d.display());
                        per_state.push(st_d);
                        let tb = hasty_recon::pcscore::PcTruth::build(&nb, &sd, &grid.x_log, &dec, *covmin).coarsen(nf_card, &dec);
                        println!("pc score bin {b} truth {} (breathing state {st_d:.3}, bin true state {st_b:.3}): {} voxels blood throughout the cycle", d.display(), tb.n());
                        tb
                    })
                    .collect();
                hasty_recon::pcscore::PcTruth::stack_bins(per)
            }
        };
        let in_lumen = { let l: std::collections::HashSet<usize> = lumen.iter().copied().collect(); t.vox.iter().filter(|q| l.contains(q)).count() };
        println!("pc score: {} voxels blood throughout the cycle (coverage ≥ {covmin} at every native voxel of the box mean; {} truth frames) from {}: {} in the legacy lumen set, {} in the support ({:.1}s)", t.n(), t.nf, d.display(), in_lumen, t.vox.iter().filter(|&&q| support[q]).count(), t_p.elapsed().as_secs_f64());
        assert_eq!(t.nf, if score_pc_bins.is_some() { nf } else { nf_card }, "--score-pc: truth frames vs cardiac frames");
        for (name, s) in t.oracles(&dec, nf) {
            println!("  pc score oracle {name:<20}: {}", s.line());
        }
        if score_pc_bins.is_some() {
            // the --score-pc truth (one breathing state) scored as if it were a reconstruction against the per-bin
            // truths: the bias of scoring every bin against one state; voxels not blood at that state count as exact
            let tm = hasty_recon::pcscore::PcTruth::build(nat, &sd, &grid.x_log, &dec, *covmin).coarsen(nf_card, &dec);
            let at_m: std::collections::HashMap<usize, usize> = tm.vox.iter().enumerate().map(|(j, &q)| (q, j)).collect();
            let jm: Vec<Option<usize>> = t.vox.iter().map(|q| at_m.get(q).copied()).collect();
            let miss = jm.iter().filter(|j| j.is_none()).count();
            let s = t.score(
                nf,
                &|fr, i| jm[i].map_or(t.v[i * t.nf + fr % t.nf], |j| tm.v[j * tm.nf + fr % tm.nf]),
                &|fr, e, i| jm[i].map_or(t.c_at(i, e, fr % t.nf), |j| tm.c_at(j, e, fr % tm.nf)),
            );
            println!("  pc score oracle {:<20}: {} ({miss} of {} voxels not blood there: scored as exact)", "--score-pc truth", s.line(), t.n());
        }
        // the legacy yardstick's truth (truth_vel, point / box at the voxel) scored as if it were a reconstruction
        let tv = |fr: usize, i: usize| -> [f64; 3] { [0, 1, 2].map(|a| sample_truth(&sd.truth_vel, a + 3 * (fr % nf_card), &grid.x_log[t.vox[i]])) };
        let s = t.score(nf, &tv, &|fr, e, i| t.c_at(i, e, fr % t.nf));
        println!("  pc score oracle {:<20}: {} (velocity part only: the gap between the two yardsticks)", "truth_vel sampled", s.line());
        t
    });
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
        // --smaps-resp: the same slice of every respiratory bin's maps [x, y, coil, bin]; the ESPIRiT eigenvalue slice
        if !maps_resp.is_empty() {
            let sr: Vec<C32> = maps_resp.iter().flat_map(|(m, _)| (0..nvc).flat_map(move |c| m[c * nv + zs * nxy..c * nv + (zs + 1) * nxy].iter().copied())).collect();
            save(&out.join(format!("smaps_resp_axial_z{zs}.npy")), "<c8", &[sd.dims[0], sd.dims[1], nvc, maps_resp.len()], c8(&sr));
        }
        if let Some(e) = &espirit_eig {
            save(&out.join(format!("espirit_eig_axial_z{zs}.npy")), "<f4", &[sd.dims[0], sd.dims[1]], f4(e[zs * nxy..(zs + 1) * nxy].iter().copied()));
        }
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
    let need_circ = vmodel.is_none() && knots2d.is_none() && preconds.iter().any(|p| Precond::parse(p).is_some_and(|q| q.metric()));
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
    // --prior-mode admm: the clip metric's samples-per-cell symbols (one per encode, all its sets pooled: the
    // coil-smoothed Chan symbol normalised to samples per Nyquist cell; set share = its samples / the encode's) and the
    // data of the ρ₁ check (one set's own symbol, an isolated one-sample cell at high k and the k = 0 cell), while the
    // sample coordinates are still on the host
    let admm_sym: Option<AdmmSym> = admm.then(|| {
        use hasty_recon::precond::{chan_spectrum, coil_convolved};
        let ta = Instant::now();
        let dsum: Vec<f32> = (0..nv).map(|q| (0..nvc).map(|c| { let s = smaps[c * nv + q]; s.re * s.re + s.im * s.im }).sum()).collect();
        let dbar = dsum.iter().map(|&v| v as f64).sum::<f64>() / nv as f64;
        let dims = sd.dims;
        // |k| / (ball radius) of FFT index q (the --kcrop ball |k·Δ| ≤ π is 1)
        let kball = |q: usize| -> f64 {
            let i = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
            (0..3).map(|a| { let n = dims[a] as f64; let v = i[a] as f64; let f = if v <= n / 2.0 { v } else { v - n }; (f / (n / 2.0)).powi(2) }).sum::<f64>().sqrt()
        };
        let m_of = |t: usize| host_sets[t].coords[0].len();
        let sym = |idx: &[usize]| -> (Vec<f32>, usize) {
            let c: [Vec<f32>; 3] = [0, 1, 2].map(|a| idx.iter().flat_map(|&t| host_sets[t].coords[a].iter().copied()).collect());
            let m = c[0].len();
            let lc = chan_spectrum(&dev, &st, dims, [&c[0], &c[1], &c[2]]).unwrap();
            let sum: f64 = lc.iter().map(|&v| v as f64).sum();
            let conv = coil_convolved(&dev, &st, dims, &lc, &smaps, nvc).unwrap();
            (hasty_recon::admm::samples_per_cell(&conv, sum, m, dbar), m)
        };
        // radial profile of a per-set symbol: the first |k| band (0.025 wide) whose mean count drops below 1
        let r_dense = |mu: &[f32], sh: f64| -> f64 {
            let nb = 40;
            let (mut s, mut n) = (vec![0.0f64; nb], vec![0usize; nb]);
            for q in 0..nv {
                let b = (kball(q) / 0.025) as usize;
                if b < nb {
                    s[b] += sh * mu[q] as f64;
                    n[b] += 1;
                }
            }
            (0..nb).find(|&b| n[b] > 0 && s[b] / (n[b] as f64) < 1.0).map_or(1.0, |b| b as f64 * 0.025)
        };
        let ball = (0..nv).filter(|&q| kball(q) <= 1.0).count().max(1);
        let (mut mu, mut of_set) = (vec![], vec![]);
        if admm_metric == hasty_recon::admm::AdmmMetric::Clip {
            of_set = vec![(0usize, 0.0f64); fe.len()];
            for e in 0..ne {
                let idx: Vec<usize> = (0..fe.len()).filter(|&t| enc_of(t) == e).collect();
                let (me, m_e) = sym(&idx);
                for &t in &idx {
                    of_set[t] = (e, m_of(t) as f64 / m_e as f64);
                }
                let sh = idx.iter().map(|&t| of_set[t].1).sum::<f64>() / idx.len() as f64;
                let dense = (0..nv).filter(|&q| sh * me[q] as f64 >= 1.0).count();
                let dense_ball = (0..nv).filter(|&q| kball(q) <= 1.0 && sh * me[q] as f64 >= 1.0).count();
                let mx = me.iter().cloned().fold(0f32, f32::max) as f64 * sh;
                println!("admm clip metric, encode {e}: {} sets pooled ({m_e} samples, mean set share {sh:.4}); per set μ̃ ≥ 1 in {:.2} % of all cells, {:.2} % of the ball (mean μ̃ ≥ 1 out to |k| ≈ {:.3} of the ball radius), max μ̃ {mx:.1}", idx.len(), 100.0 * dense as f64 / nv as f64, 100.0 * dense_ball as f64 / ball as f64, r_dense(&me, sh));
                mu.push(me);
            }
        }
        // ρ₁ check set: the middle set of encode 0; its cells by snapping the samples (the sign convention that matches
        // its own Chan symbol), an isolated one-sample cell at 0.5–0.9 of the ball radius, and k = 0
        let e0: Vec<usize> = (0..fe.len()).filter(|&t| enc_of(t) == 0).collect();
        let tc = e0[e0.len() / 2];
        let (mu_t, m_t) = sym(&[tc]);
        let hs = &host_sets[tc];
        let counts = |sg: f64| -> Vec<u32> {
            let mut n = vec![0u32; nv];
            for j in 0..m_t {
                let i: Vec<usize> = (0..3).map(|a| ((sg * hs.coords[a][j] as f64 * dims[a] as f64 / std::f64::consts::TAU).round() as i64).rem_euclid(dims[a] as i64) as usize).collect();
                n[i[0] + dims[0] * (i[1] + dims[1] * i[2])] += 1;
            }
            n
        };
        let corr = |n: &[u32]| -> f64 { n.iter().zip(&mu_t).map(|(&a, &b)| a as f64 * b as f64).sum() };
        let (np, nm) = (counts(1.0), counts(-1.0));
        let (sign, n) = if corr(&np) >= corr(&nm) { (1, np) } else { (-1, nm) };
        let nb = |q: usize| -> bool {
            let i = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
            for d in 0..27usize {
                let o = [d % 3, (d / 3) % 3, d / 9];
                if o == [1, 1, 1] {
                    continue;
                }
                let j: Vec<usize> = (0..3).map(|a| (i[a] + dims[a] + o[a] - 1) % dims[a]).collect();
                if n[j[0] + dims[0] * (j[1] + dims[1] * j[2])] > 0 {
                    return false;
                }
            }
            true
        };
        let iso = (0..nv).filter(|&q| n[q] == 1 && kball(q) >= 0.5 && kball(q) <= 0.9 && nb(q)).min_by(|&a, &b| (kball(a) - 0.7).abs().partial_cmp(&(kball(b) - 0.7).abs()).unwrap().then(a.cmp(&b)));
        let mut cells: Vec<(usize, u32, f64)> = vec![(0, n[0], 0.0)];
        if let Some(q) = iso {
            cells.push((q, n[q], kball(q)));
        }
        println!("admm ρ₁ check data: set {tc} ({m_t} samples; snapped with sign {sign:+}), cells {} ({:.1}s for the symbols)", cells.iter().map(|c| format!("{} (|k| {:.2} of the ball, {} samples, own μ̃ {:.3})", c.0, c.2, c.1, mu_t[c.0])).collect::<Vec<_>>().join(", "), ta.elapsed().as_secs_f64());
        AdmmSym { mu, of_set, dsum, dbar, tc, mu_t, cells }
    });
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
    // --rcompress: after the warm start and the circulant metric (both on the original samples: the metric needs the
    // original sample density), before the segment operator. Per set: the kept samples S of every readout and the
    // virtual data ỹ = R⁻ᴴ Eᴴ y (store / device / pinned copies replaced; per readout and coil ‖y‖² − ‖ỹ‖² kept), the
    // sample records cropped to S (the whole records stay for the segment tables, as with --kcrop: crop_sel composed),
    // the banded R between operator and data (RcSets, the outermost operator wrapper below)
    let mut ys = ys;
    let rc_state: Option<(hasty_recon::rcompress::Compressor, Vec<hasty_recon::rcompress::SetComp>)> = if rc_mode.is_off() {
        None
    } else {
        assert!(model == "seg" && streamed && !synthetic, "--rcompress needs --model seg, the streamed data path and no --synthetic");
        let t_rc = Instant::now();
        let na = sd.na;
        // readout layout per set: samples per readout (the kept ones with --kcrop; readouts without any dropped)
        let lens: Vec<Vec<usize>> = (0..host_sets.len()).map(|t| match &crop_sel {
            Some(cs) => {
                let mut l = vec![0usize; host_sets[t].readouts.len()];
                for &j in &cs[t] {
                    l[j as usize / na] += 1;
                }
                l.into_iter().filter(|&x| x > 0).collect()
            }
            None => vec![na; host_sets[t].readouts.len()],
        }).collect();
        // band: the operator's ratemap (field incl. --rate, R2*; fat peaks with a fat model), coil-energy weighted
        let ce: Vec<f64> = (0..nv).map(|q| (0..nvc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
        let mut band = hasty_recon::rcompress::Band::from_ratemap(&grid.omega, r2v.as_deref(), &ce, 1e-4, 1.0, 200.0);
        band.r_max = band.r_max.max(300.0);
        if fat_fraction.is_some() {
            let fs = hasty_qfit::waterfat::FatSpectrum::six_peak();
            let wf = 2.0 * std::f64::consts::PI * hasty_mrsim::GAMMA_BAR * sd.b0 * 1e-6;
            let (lo, hi) = fs.ppm.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &p| (a.min(p * wf), b.max(p * wf)));
            band = band.union(band.w_lo + lo.min(0.0), band.w_hi + hi.max(0.0));
        }
        let (opts, line) = hasty_recon::rcompress::decide(&rc_mode, sd.dims, sd.vox, band, &host_sets[0], &lens[0]);
        println!("rcompress: band {}; {line}", band.describe());
        opts.map(|o| {
            // gate: direct sums on two readouts of set 0 (sources in the ROI × band, half on its edge; the readouts'
            // concomitant coefficients)
            {
                let (s0, l0) = (&host_sets[0], &lens[0]);
                let mut gs = hasty_recon::rcompress::GateStats::default();
                for r in [0, l0.len() / 2] {
                    let a0: usize = l0[..r].iter().sum();
                    let rng = a0..a0 + l0[r];
                    let k: Vec<[f64; 3]> = rng.clone().map(|j| [0, 1, 2].map(|a| s0.coords[a][j] as f64 / sd.vox[a])).collect();
                    let d = hasty_recon::rcompress::Design::build(&k, &s0.tau[rng.clone()], &o);
                    gs = gs.max(hasty_recon::rcompress::gate(&d, &k, &s0.tau[rng.clone()], Some(&s0.wq[rng]), None, &o, 200, 5 + r as u64));
                }
                println!("rcompress gate (2 readouts of set 0, 200 sources, half on the band edge): worst source {:.2e}, normal operator on the sources {:.2e}", gs.worst, gs.gram);
                diag.insert("rcompress_gate".into(), serde_json::json!({"worst": gs.worst, "frob": gs.frob, "gram": gs.gram}));
            }
            let mut comp = hasty_recon::rcompress::Compressor::new(o);
            let mut scs: Vec<hasty_recon::rcompress::SetComp> = Vec::with_capacity(host_sets.len());
            let mut yc_all: Vec<Vec<C32>> = Vec::with_capacity(host_sets.len());
            for t in 0..host_sets.len() {
                let m_t = host_sets[t].tau.len();
                let y: Vec<C32> = match &kstore {
                    Some(k) => k.read_vec(t).unwrap(),
                    None if early_host => ys_pin[t].as_slice().unwrap()[..m_t * nvc].to_vec(),
                    None => st.clone_dtoh(&ys[t]).unwrap(),
                };
                let (sc, yc) = comp.compress_set(&host_sets[t], &lens[t], sd.vox, &y, nvc);
                if staged {
                    yc_all.push(yc);
                } else if early_host {
                    let mut h = unsafe { dev.ctx.alloc_pinned_with_flags::<C32>(yc.len().max(1), 0).unwrap() };
                    h.as_mut_slice().unwrap()[..yc.len()].copy_from_slice(&yc);
                    ys_pin[t] = h;
                } else {
                    ys[t] = st.clone_htod(&yc).unwrap();
                }
                scs.push(sc);
            }
            if staged {
                let ks = KStore::create(KStoreSpec::dense(), nvc, scs.iter().map(|c| c.m()).collect(), None, &format!("{}|rcompress", kstore.as_ref().unwrap().tag())).unwrap();
                for (t, yc) in yc_all.into_iter().enumerate() {
                    ks.put_owned(t, yc).unwrap();
                }
                ks.finish().unwrap();
                kstore = Some(std::sync::Arc::new(ks));
            }
            // records: cropped to S; the whole records (for the segment tables) and the composed selection
            let pre: Vec<SampleSet> = host_sets.iter().zip(&scs).map(|(s, c)| hasty_recon::pyramid::crop_set(s, &c.sel)).collect();
            let old = std::mem::replace(&mut host_sets, pre);
            crop_sel = Some(match crop_sel.take() {
                Some(cs) => cs.iter().zip(&scs).map(|(c, sc)| sc.sel.iter().map(|&i| c[i as usize]).collect()).collect(),
                None => scs.iter().map(|c| c.sel.clone()).collect(),
            });
            if full_sets.is_empty() {
                full_sets = old;
                for s in full_sets.iter_mut() {
                    s.y = Vec::new();
                }
            }
            let (n0, n1) = (scs.iter().map(|c| c.n_orig()).sum::<usize>(), scs.iter().map(|c| c.m()).sum::<usize>());
            let cn: f64 = scs.iter().map(|c| c.cnorm.iter().sum::<f64>()).sum();
            let yn: f64 = scs.iter().map(|c| c.ynorm.iter().sum::<f64>()).sum();
            // all signal inside ROI × band: the dropped energy is noise, E = (n − |S|) · coils · σ² (σ² per complex
            // whitened sample); a ratio ≫ 1 flags signal outside the ROI or band (data-derived check)
            let noise_exp = sd.lazy.as_ref().and_then(|l| l.sigma).map(|sg| (n0 - n1) as f64 * nvc as f64 * sg * sg);
            println!("rcompress: {} ({:.1}s): {n0} → {n1} samples ({:.3}×), R tables {:.3} GiB; data constant Σ ‖y‖² − ‖ỹ‖² = {:.3e} of ‖y‖² {:.3e}{}", comp.describe(), t_rc.elapsed().as_secs_f64(), n0 as f64 / n1.max(1) as f64, comp.table_bytes() as f64 / GIB, cn, yn, noise_exp.map_or(String::new(), |e| format!(" ({:.3}× the noise expectation)", cn / e.max(1e-300))));
            diag.insert("rcompress".into(), serde_json::json!({"mode": format!("{rc_mode:?}"), "roi": comp.opts.roi.describe(), "band": comp.opts.band.describe(), "os": comp.opts.os, "q": comp.opts.q, "designs": comp.designs.len(), "samples": n0, "kept": n1, "reduction": n0 as f64 / n1.max(1) as f64, "table_gib": comp.table_bytes() as f64 / GIB, "data_constant": cn, "data_energy": yn}));
            (comp, scs)
        })
    };
    // knot mode: the temporal Gram of the basis under the sample times (accumulated while the tables are built)
    let nk_all = knots2d.map_or(0, |(kr, _, mc, _)| kr * mc);
    let mut knot_gram_acc = vec![0.0f64; nk_all * nk_all];
    let mut knot_gram_n = 0usize;
    // the operator stack
    let mut sense_sets;
    let mut seg_sets;
    // --ratemap truth-bc in frame mode: one host segment table per operator frame (setop::PerSetSegs)
    let mut frame_tabs: Option<std::sync::Arc<Vec<hasty_recon::setop::HostSegTab>>> = None;
    // --holdout-res: the held-out operator and data (built with the segment operator below)
    let mut held_eval: Option<HeldOut> = None;
    let mut fat_joint_sets;
    // joint water + fat with per-frame fat maps (FatJointSets); a static map is folded into the segment diagonals
    let mut ff_per_frame = false;
    let mut pp_seg_host: Option<Vec<hasty_nufft::SegHostTables>> = None;
    // --pp-*: the operator's per-set sample coordinates (the segment tables', cropped) for the G3 surrogate
    let mut pp_coords: Option<Vec<[Vec<f32>; 3]>> = pp_args.active().then(|| vec![[vec![], vec![], vec![]]; fe.len()]);
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
        println!("  [build timer] monomial fit done: {:.1}s since the build start", ts.elapsed().as_secs_f64());
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
        // --ratemap truth-bc: the true ratemap of every knot's (breathing, cardiac) state, on the reconstruction grid
        // (built before the segment rule: with per-knot ratemaps the rule runs on every knot's own model)
        // frame mode (one image per (frame, encode) set): the per-frame ratemaps below (rm_frames), not the knot path
        let ratemap_frame_mode = ratemap_frames && knots2d.is_none() && op_frames.is_none() && temporal.is_none();
        let rm_knots: Option<Vec<RateFrame>> = (ratemap_frames && !ratemap_frame_mode).then(|| {
            assert!((knots2d.is_some() || temporal.is_some()) && op_frames.is_none(), "--ratemap truth-bc: complex 2-D knots (--knots2d), --temporal or the frame mode; the velocity model's operator frames come next");
            let t_ro: Vec<f32> = (0..sd.n_readouts()).map(|i| { let c = sd.ro.card[i]; (c.phase * nf as f64 - 0.5).rem_euclid(nf as f64) as f32 }).collect();
            let used: Vec<usize> = (0..sd.n_readouts()).filter(|&i| sd.ro.frame[i] != usize::MAX).collect();
            let tf: Vec<f32> = used.iter().map(|&i| t_ro[i]).collect();
            let (w, nk): (Vec<Vec<(usize, f64)>>, usize) = match (&knot_basis, temporal) {
                (Some((b, per, _)), _) => {
                    let (kt, ri, rv) = b.sample_coords(&tf, nf, per, &|j| used[j]);
                    ((0..used.len()).map(|j| b.sample_weights(kt[j] as f64, ri[j], rv[j])).collect(), b.nk())
                }
                // --temporal: the periodic B-spline weights of the engine (knot k centred at t = k, hasty_recon::vmodel::
                // bspline) at the readout's cardiac time; knot k's ratemap = the weighted mean state of its readouts
                (None, Some(p)) => {
                    let w = tf
                        .iter()
                        .map(|&t| {
                            let t = t as f64;
                            let i0 = t.floor() as i64;
                            (i0 - 2..=i0 + 2)
                                .filter_map(|k| {
                                    let b = hasty_recon::vmodel::bspline(p, t - k as f64);
                                    (b > 0.0).then(|| (k.rem_euclid(nf as i64) as usize, b))
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    (w, nf)
                }
                (None, None) => unreachable!(),
            };
            let mut rm = truth_ratemap_knots(std::path::Path::new(&dir), &sd, &grid.x_log, truth_box, nk, &used, &w);
            if ratemap_static {
                // knot weights: the samples' basis weights summed per knot
                let mut kw = vec![0.0f64; rm.len()];
                for ws in &w {
                    for &(b_, wb) in ws {
                        kw[b_] += wb.abs();
                    }
                }
                let nvr = rm[0].om.len();
                let avg64 = |get: &dyn Fn(&RateFrame, usize) -> f64| -> Vec<f64> {
                    (0..nvr).map(|v| {
                        let (mut s_, mut ws_) = (0.0f64, 0.0f64);
                        for (f, &k) in rm.iter().zip(&kw) {
                            let x = get(f, v);
                            if x.is_finite() && k > 0.0 { s_ += k * x; ws_ += k; }
                        }
                        if ws_ > 0.0 { s_ / ws_ } else { f64::NAN }
                    }).collect()
                };
                let avg = |get: &dyn Fn(&RateFrame, usize) -> f64| -> Vec<f32> { avg64(get).into_iter().map(|x| x as f32).collect() };
                let mean = RateFrame { om: avg64(&|f, v| f.om[v]), rw: avg(&|f, v| f.rw[v] as f64), rf: avg(&|f, v| f.rf[v] as f64), ff: avg(&|f, v| f.ff[v] as f64) };
                println!("ratemap truth-bc:static: every knot gets the data-weighted mean of the {} knot ratemaps (control)", rm.len());
                for f in rm.iter_mut() {
                    *f = RateFrame { om: mean.om.clone(), rw: mean.rw.clone(), rf: mean.rf.clone(), ff: mean.ff.clone() };
                }
            }
            // voxels without a source in a state: the static field (they carry no signal there)
            for f in rm.iter_mut() {
                for (v, x) in f.om.iter_mut().enumerate() {
                    if !x.is_finite() {
                        *x = grid.omega[v];
                    }
                }
            }
            // sign / unit check against the static field the rest of the operator uses (rad/s)
            let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
            for &v in fit_vox.iter().step_by(13) {
                let m: f64 = rm.iter().map(|f| f.om[v] as f64).sum::<f64>() / rm.len() as f64;
                sxy += m * grid.omega[v];
                sxx += m * m;
                syy += grid.omega[v] * grid.omega[v];
            }
            println!("ratemap per frame (truth-bc, ORACLE): {} knots; knot-mean ω vs the static field: correlation {:.3}, rms {:.1} / {:.1} Hz", rm.len(), sxy / (sxx * syy).sqrt().max(1e-30), (sxx / fit_vox.len().div_ceil(13) as f64).sqrt() / std::f64::consts::TAU, (syy / fit_vox.len().div_ceil(13) as f64).sqrt() / std::f64::consts::TAU);
            rm
        });
        // --ratemap truth-bc in frame mode: one TRUE ratemap per operator frame f (oracle), the state of the frame's
        // readouts (mean breathing amplitude, circular-mean cardiac time) from scan/truth_ratemap_bc, the same maps as
        // the knot path (field, water / fat decay, fat fraction). The operator gets one segment table per frame
        // (setop::PerSetSegs, below); the segment rule runs on every frame's own model.
        let rm_frames: Option<Vec<RateFrame>> = ratemap_frame_mode.then(|| {
            assert!(!ratemap_static, "--ratemap truth-bc:static: knot mode only (in frame mode the per-frame ratemap is the model)");
            let used: Vec<usize> = (0..sd.n_readouts()).filter(|&i| sd.ro.frame[i] != usize::MAX).collect();
            assert!(used.iter().all(|&i| sd.ro.frame[i] < nf), "--ratemap truth-bc: readout frame index ≥ the {nf} operator frames");
            let w: Vec<Vec<(usize, f64)>> = used.iter().map(|&i| vec![(sd.ro.frame[i], 1.0)]).collect();
            let mut rm = truth_ratemap_knots(std::path::Path::new(&dir), &sd, &grid.x_log, truth_box, nf, &used, &w);
            // voxels without a source in a state: the static field (they carry no signal there)
            for f in rm.iter_mut() {
                for (v, x) in f.om.iter_mut().enumerate() {
                    if !x.is_finite() {
                        *x = grid.omega[v];
                    }
                }
            }
            // per-frame field vs the static field the warm start and the preconditioners use (rad/s → Hz), fit voxels
            let hz = 1.0 / std::f64::consts::TAU;
            let d: Vec<f64> = rm.iter().map(|f| {
                let s: f64 = fit_vox.iter().step_by(13).map(|&v| (f.om[v] as f64 - grid.omega[v]).powi(2)).sum();
                (s / fit_vox.len().div_ceil(13) as f64).sqrt() * hz
            }).collect();
            let (dmin, dmax) = d.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
            println!("ratemap per frame (truth-bc, ORACLE, frame mode): {} frames; rms ω_f − static field over the fit voxels {dmin:.1}..{dmax:.1} Hz", rm.len());
            rm
        });
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
        let snr_w: Option<Vec<f64>> = (seg_svdn || seg_noise_rule.is_some() || seg_err_rule.is_some()).then(|| {
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
        if let (Some(r_noise), Some((ts_n, tw_n)), Some(w)) = (seg_noise_rule.or(seg_err_rule), snr_grid.as_ref(), snr_w.as_ref()) {
            // relative weighted error ε(L) → data error in noise rms ≈ ε(L) √(mean w) (w = 1 + (E(n) + E_vol) / (C σ²):
            // proportional to the sample's signal plus the white floor; research/segment_basis.md)
            // margin 1.5: the separable prediction was up to 1.3–2× optimistic against the exact data-space error at
            // L ≥ 9 on ph0001 (research/segment_basis.md §2–3); HASTY_SEG_MARGIN overrides
            let margin: f64 = std::env::var("HASTY_SEG_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
            let gain = (w.iter().sum::<f64>() / w.len() as f64).sqrt();
            // auto:err: the tolerance is the relative weighted error itself
            let tol = if seg_noise_rule.is_some() { r_noise / (gain * margin) } else { r_noise };
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
            // per-knot ratemaps (--ratemap truth-bc): the rule on every knot's own model g_b (its field, water and fat
            // decay and fat fraction, as `make_segs` builds them), the maximum over knots and encodes
            // (research/field_absorb.md: the static field with phase only gave L = 8–9 where the knot models need 11).
            // Env HASTY_SEG_RULE_KNOTS=static: the previous rule (static field) also here.
            // (frame mode with --ratemap truth-bc: the same rule on every operator frame's model)
            let rm_rule = rm_knots.as_ref().or(rm_frames.as_ref());
            let rule_knots = rm_rule.is_some() && !std::env::var("HASTY_SEG_RULE_KNOTS").is_ok_and(|v| v == "static");
            if let (true, Some(rm)) = (rule_knots, rm_rule) {
                let t_rule = Instant::now();
                let l_static = lsel;
                let ta = tau_a[0];
                // knot models of encode e; per (knot, encode): (L, error at L, spectrum or None for the node scan)
                let tl_of = |l: usize| -> Vec<f64> { if seg_cheb { hasty_nufft::cheb_nodes(l, t0, t1) } else { (0..l).map(|i| t0 + (t1 - t0) * i as f64 / (l - 1).max(1) as f64).collect() } };
                let weighted = seg_cheb || seg_energy;
                let mut per: Vec<Vec<(usize, Option<Vec<f64>>)>> = vec![Vec::with_capacity(ne); rm.len()];
                for e in 0..ne {
                    let conc0 = segment_phase(&grid, &wbar[e], &tau_a, false);
                    for (kb, f) in rm.iter().enumerate() {
                        let psi_raw = |v: usize, t: f64| conc0(v, t) + f.om[v] as f64 * t;
                        let psi = field_poly.residual(&sd, &psi_raw);
                        let g = |v: usize, t: f64| -> Complex64 {
                            let ph = Complex64::from_polar(1.0, -psi(v, t));
                            let ffv = f.ff[v] as f64;
                            let wat = (1.0 - ffv) * (-(f.rw[v] as f64) * (t - ta)).exp();
                            let fat = ffv * (-(f.rf[v] as f64) * (t - ta)).exp();
                            ph * (Complex64::new(wat, 0.0) + fat * cfat(t))
                        };
                        if seg_svd {
                            let ev = hasty_nufft::weighted_spectrum(&seg_fit_e, seg_ew_rule.as_deref(), ts_n, tw_n, &g);
                            per[kb].push((hasty_nufft::seg_count_for(&ev, tol, 2, 64), Some(ev)));
                        } else {
                            let l = (2..=64).find(|&l| hasty_nufft::nodes_weighted_error(&seg_fit_e, if weighted { seg_vw_rule.as_deref() } else { None }, seg_ew_rule.as_deref(), ts_n, tw_n, &tl_of(l), &g) <= tol).unwrap_or(64);
                            per[kb].push((l, None));
                        }
                    }
                }
                lsel = per.iter().flatten().map(|p| p.0).max().unwrap_or(l_static).max(2);
                // predicted error of every knot at the chosen L (worst encode)
                let err_at = |kb: usize, e: usize| -> f64 {
                    match &per[kb][e].1 {
                        Some(ev) => hasty_nufft::tail_errors(ev)[lsel.min(ev.len())],
                        None => {
                            let conc0 = segment_phase(&grid, &wbar[e], &tau_a, false);
                            let f = &rm[kb];
                            let psi_raw = |v: usize, t: f64| conc0(v, t) + f.om[v] as f64 * t;
                            let psi = field_poly.residual(&sd, &psi_raw);
                            let g = |v: usize, t: f64| -> Complex64 {
                                let ffv = f.ff[v] as f64;
                                Complex64::from_polar(1.0, -psi(v, t)) * (Complex64::new((1.0 - ffv) * (-(f.rw[v] as f64) * (t - ta)).exp(), 0.0) + ffv * (-(f.rf[v] as f64) * (t - ta)).exp() * cfat(t))
                            };
                            hasty_nufft::nodes_weighted_error(&seg_fit_e, if weighted { seg_vw_rule.as_deref() } else { None }, seg_ew_rule.as_deref(), ts_n, tw_n, &tl_of(lsel), &g)
                        }
                    }
                };
                let errs: Vec<f64> = (0..rm.len()).map(|kb| (0..ne).map(|e| err_at(kb, e)).fold(0.0, f64::max)).collect();
                let lk: Vec<usize> = per.iter().map(|v| v.iter().map(|p| p.0).max().unwrap_or(0)).collect();
                let worst = (0..rm.len()).max_by(|&a, &b| errs[a].partial_cmp(&errs[b]).unwrap()).unwrap_or(0);
                println!("segments (per-knot ratemaps): the rule on each of the {} knots' own models (field, water / fat decay, fat fraction), {ne} encodes → L = {lsel} (static-field rule: {l_static}); per-knot L {}..{}; predicted relative weighted error at L = {lsel}: max {:.2e} (knot {worst}), target {tol:.2e}; {:.1}s", rm.len(), lk.iter().min().unwrap(), lk.iter().max().unwrap(), errs[worst], t_rule.elapsed().as_secs_f64());
                for (r, chunk) in errs.chunks(16).enumerate() {
                    println!("    knots {:>3}..{:>3}: L {:?}, error {}", r * 16, r * 16 + chunk.len() - 1, &lk[r * 16..r * 16 + chunk.len()], chunk.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(" "));
                }
            }
            if seg_noise_rule.is_some() {
                println!("segments (auto:noise): predicted data error × {margin} ≤ {r_noise} noise rms (relative weighted error ≤ {tol:.2e}, ×{gain:.1}) → L = {lsel} ({} basis)", seg_basis_arg);
            } else {
                println!("segments (auto:err): relative signal-weighted model error of the ratemap ≤ {tol:.1e} (predicted on the time grid) → L = {lsel} ({} basis, {seg_weight} voxel weights)", seg_basis_arg);
            }
            println!("  [build timer] auto:noise segment rule done: {:.1}s since the build start", ts.elapsed().as_secs_f64());
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
        // knot mode: the 2-D knot engine (cardiac knots × degree in frames / bspline), its fine-grid slot budgeted next
        // to one encode's two knot stacks and data (the ledger checks the whole plan before the solver)
        // spline mode: the engine's knots are (kre respiratory knots, linear) × (fop operator frames, degree odeg)
        let eng_knots: Option<(usize, u32, usize, u32)> = match &op_frames { Some(of) => Some((of.kre, 1, of.fop, of.odeg)), None => knots2d };
        let knot_grid_budget: usize = eng_knots.map_or(6 << 30, |(kr, _, mc, _)| {
            let (free, total) = dev.ctx.mem_get_info().unwrap();
            let m_max = host_sets.iter().map(|s| s.tau.len()).max().unwrap_or(0);
            let work = kr * mc * nv * 8 + 2 * nvc * m_max * 8 + (3usize << 30);
            let legacy = (free.saturating_sub(work) / 3).clamp(256 << 20, 6 << 30);
            // the knot ledger's slot entry: everything else of the plan estimated now (the real ledger runs before the
            // solver on measured sizes): operator without the slot = what is resident now + coil maps + segment
            // diagonals (per-knot ratemaps: every knot's) + 0.5 GiB of FFT plans / kernels; data and tables from the
            // sample counts (≈ 220 B of tables per sample); the velocity model's parameter stack
            let ledger_slot = || -> Result<usize, String> {
                let nk = kr * mc;
                let kenc_e: Vec<[f64; 3]> = sd.encode_m1.iter().map(|m| m.map(|x| hasty_mrsim::GAMMA * x)).collect();
                let spec_e = match &op_frames {
                    Some(of) => hasty_recon::vmodel::VelSpec::splines(sd.dims, kenc_e, of.clone(), vpoly),
                    None => hasty_recon::vmodel::VelSpec::at_knots(sd.dims, kenc_e, kr, mc, vpoly),
                };
                let ms: Vec<usize> = host_sets.iter().map(|s| s.tau.len()).collect();
                let samples: usize = ms.iter().sum();
                let segs = ne * nseg * nv * 8 * if ratemap_frames { nk } else { 1 };
                let fixed = (total - free) + nvc * nv * 8 + segs + (512usize << 20);
                let pf = 0.4;
                let quasi = preconds.iter().any(|p| Precond::parse(p).is_some_and(|q| q.quasi()));
                let lin = hasty_recon::vknot::LedgerIn {
                    nv,
                    nk,
                    img_frames: op_frames.as_ref().map_or(0, |o| o.set_frames()),
                    slots: 0,
                    stack: spec_e.stack_bytes(),
                    chunk_max: spec_e.lens().iter().max().copied().unwrap_or(0) * 8,
                    nsets: ms.len(),
                    data_all: samples * nvc * 8,
                    data_max: m_max * nvc * 8,
                    data_movable: true,
                    tab_all: samples * 220,
                    tab_max: m_max * 220,
                    fixed,
                    dev_budget: vram_budget.unwrap_or(total.saturating_sub(512 << 20)).min(total),
                    host_avail: available_memory(),
                    quasi,
                    prior_dev: 0,
                    h2d: 20e9,
                    pageable: pf * 20e9,
                };
                let force_state = match state_arg.as_str() { "host" => Some(true), "device" => Some(false), _ => None };
                let force_mode = match plan_mode.as_str() { "host" => Some(true), "device" => Some(false), _ => None };
                let (slot, cb, pl) = hasty_recon::vknot::slot_budget(&lin, op_frames.is_none(), force_state, force_mode, force_mode, nvc, 16, 0.05, &|cb| hasty_nufft::seg_slot_bytes(sd.dims, 1.25, 10, nseg, cb, 1))?;
                println!("knot grid slot (auto, sized by the knot ledger): coil batch {cb} of {nvc} ({} passes), {:.2} GiB; estimated plan: state {}, data {}, tables {}, device peak {:.2} GiB", nvc.div_ceil(cb), slot as f64 / GIB, if pl.state_on_host { "host" } else { "device" }, if pl.data_on_host { "host" } else { "device" }, if pl.tables_on_host { "host" } else { "device" }, pl.device_peak as f64 / GIB);
                Ok(slot + (1 << 20))
            };
            let b = match grid_budget_arg.as_str() {
                "legacy" => legacy,
                "free" => ((free.saturating_sub(work + (2usize << 30))) as f64 / 1.5) as usize,
                "auto" => ledger_slot().unwrap_or_else(|e| {
                    println!("knot grid slot (auto): the ledger estimate fits no slot ({}); legacy budget", e.lines().next().unwrap_or(""));
                    legacy
                }),
                g => hasty_recon::plan::parse_bytes(g).expect("--grid-budget auto | legacy | free | size (e.g. 10G)"),
            }
            .max(256 << 20);
            println!("knot grid slot budget ({grid_budget_arg}): {:.2} GiB of {:.2} GiB free (work {:.2} GiB)", b as f64 / GIB, free as f64 / GIB, work as f64 / GIB);
            b
        });
        let seg_opts = |nseg: usize| match eng_knots {
            Some((_, _, mc, cdeg)) => SegNufftOpts { sigma: 1.25, width: 10, nseg, order: if conc_hi { 3 } else { order }, frames: mc, bspline: cdeg, max_grid_bytes: knot_grid_budget },
            None => SegNufftOpts { sigma: 1.25, width: 10, nseg, order: if conc_hi { 3 } else { order }, frames: if temporal.is_some() { nf } else { 1 }, bspline: temporal.unwrap_or(1), max_grid_bytes: 6 << 30 },
        };
        // --temporal: the knot-major engine unless --temporal-engine interval (the per-knot ratemaps need it)
        let new_engine = |nseg: usize| -> SegNufft {
            match eng_knots {
                Some((kr, rdeg, _, _)) => SegNufft::new_temporal2d(&dev, sd.dims, sd.vox, nvc, seg_opts(nseg), kr, rdeg).unwrap(),
                None if temporal.is_some() => SegNufft::new_temporal(&dev, sd.dims, sd.vox, nvc, seg_opts(nseg), temporal_knot).unwrap(),
                None => SegNufft::new(&dev, sd.dims, sd.vox, nvc, seg_opts(nseg)).unwrap(),
            }
        };
        let mut eng = new_engine(nseg);
        if temporal.is_some() {
            println!("--temporal: {} B-spline over {nf} periodic knots (cardiac time; each sample at its own time), {} engine", if temporal == Some(3) { "cubic" } else { "linear" }, if temporal_knot { "knot-major" } else { "interval-major" });
        }
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
        // knot mode: the achieved segment-model error of the ratemap per encode (relative, voxel weights = smoothed warm
        // magnitude² × coil energy over a fit-voxel subset, ADC-time weights = the per-sample SNR weights when computed, else uniform)
        // (≤ ~20 000 voxels × every 16th ADC sample: seconds per encode)
        let seg_err_vox: Vec<usize> = fit_vox.iter().copied().step_by((fit_vox.len() / 20000).max(9)).collect();
        let seg_err_w: Vec<f64> = seg_err_vox.iter().map(|&q| (support_mag.get(q).copied().unwrap_or(1.0) as f64).powi(2) * coil_energy[q]).collect();
        let seg_err = |e: usize, sp: &[C32], at: &[C32], ta: f64, dt: f64, nseg: usize, g: &(dyn Fn(usize, f64) -> Complex64 + Sync)| {
            if knots2d.is_none() && rm_frames.is_none() {
                return;
            }
            let taus: Vec<f64> = (wa..=wb).step_by(16).map(|n| rec_sets[0].tau[n]).collect();
            let tws: Vec<f64> = (wa..=wb).step_by(16).map(|n| snr_w.as_ref().map_or(1.0, |w| w[n - wa])).collect();
            let err = hasty_nufft::segment_model_error(sp, at, nseg, nv, ta, dt, &seg_err_vox, Some(&seg_err_w), &taus, &tws, g);
            println!("  segment model error of the ratemap, encode {e}: L = {nseg}, relative {err:.2e} ({} over ADC time, signal energy over {} fit voxels)", if snr_w.is_some() { "SNR-weighted" } else { "uniform" }, seg_err_vox.len());
        };
        let make_segs = |eng: &SegNufft, nseg: usize| -> Vec<hasty_nufft::SegSegments> {
        (0..ne)
            .map(|e| {
                if let Some(rm) = &rm_knots {
                    // one ratemap per knot: g_b(v, τ) = [(1 − φ_b) e^{−R^w_b τ'} + φ_b e^{−R^f_b τ'} c(τ)] e^{−iψ_b(v, τ)},
                    // ψ_b = ω_b τ + encode-mean concomitant phase (minus the absorbed static field polynomial)
                    let conc0 = segment_phase(&grid, &wbar[e], &tau_adc, false);
                    let t_seg = Instant::now();
                    let mut sp_all: Vec<C32> = Vec::with_capacity(rm.len() * nseg * nv);
                    let mut at_all: Vec<C32> = Vec::new();
                    let (mut ta0, mut dt0) = (0.0, 0.0);
                    for (kb, f) in rm.iter().enumerate() {
                        let psi_raw = |v: usize, t: f64| conc0(v, t) + f.om[v] as f64 * t;
                        let psi = field_poly.residual(&sd, &psi_raw);
                        let ta = tau_adc[0];
                        let g = |v: usize, t: f64| -> Complex64 {
                            let ph = Complex64::from_polar(1.0, -psi(v, t));
                            let ffv = f.ff[v] as f64;
                            let wat = (1.0 - ffv) * (-(f.rw[v] as f64) * (t - ta)).exp();
                            let fat = ffv * (-(f.rf[v] as f64) * (t - ta)).exp();
                            ph * (Complex64::new(wat, 0.0) + fat * cfat(t))
                        };
                        let (sp, at, ta_, dt_) = if seg_svd { svd_build(nseg, &g) } else { build_segments_complex(nv, nseg, tw0, tw1, ntab, &seg_fit, &g) };
                        if e == 0 && (kb == 0 || kb == rm.len() / 2) {
                            seg_err(e, &sp, &at, ta_, dt_, nseg, &g);
                        }
                        ta0 = ta_;
                        dt0 = dt_;
                        sp_all.extend_from_slice(&sp);
                        at_all.extend_from_slice(&at);
                    }
                    if e == 0 {
                        println!("  per-frame ratemap segments: {} knots × L = {nseg}, encode 0 built in {:.1}s", rm.len(), t_seg.elapsed().as_secs_f64());
                    }
                    return eng.segments_frames(&st, &sp_all, &at_all, ta0, dt0).unwrap();
                }
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
                        seg_err(e, &sp, &at, ta, dt, nseg, &g);
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
                    seg_err(e, &sp, &at, ta, dt, nseg, &g);
                    return eng.segments(&st, &sp, &at, ta, dt).unwrap();
                }
                let (sp, at, ta, dt) = if seg_svd {
                    svd_build(nseg, &|v, t| Complex64::from_polar(1.0, -psi(v, t)))
                } else if seg_nodes_w {
                    cheb_build(nseg, &|v, t| Complex64::from_polar(1.0, -psi(v, t)))
                } else {
                    build_segments(nv, nseg, tw0, tw1, ntab, &seg_fit, &psi)
                };
                seg_err(e, &sp, &at, ta, dt, nseg, &|v, t| Complex64::from_polar(1.0, -psi(v, t)));
                eng.segments(&st, &sp, &at, ta, dt).unwrap()
            })
            .collect()
        };
        if let (Some(rm), Some(_)) = (&rm_knots, temporal) {
            // --temporal with per-knot ratemaps: every encode's per-knot diagonals resident (solve, and again for the
            // --holdout-res engine)
            let per = rm.len() * nseg * nv * 8;
            let need = ne * per * if holdout_res { 2 } else { 1 };
            let (free, _) = dev.ctx.mem_get_info().unwrap();
            println!("--temporal per-knot segment diagonals: {} encodes × {} knots × L = {nseg}: {:.2} GiB on the device{} ({:.2} GiB free)", ne, rm.len(), need as f64 / GIB, if holdout_res { " (solve + held-out engine)" } else { "" }, free as f64 / GIB);
            assert!(need * 10 < free * 6, "--temporal per-knot diagonals need {:.2} GiB, more than 60 % of the {:.2} GiB free", need as f64 / GIB, free as f64 / GIB);
        }
        let segs = make_segs(&eng, nseg);
        // --ratemap truth-bc, frame mode: one segment table per (operator frame, encode) on the host, index f·ne + e
        // (setop::PerSetSegs uploads the set's table). Each table is built exactly as the knot path builds a knot's
        // table: the encode's own mean concomitant coefficients w̄_e (the per-sample tables take their deviations from
        // the same w̄_e, as in the static path; the encodes' means differ by up to ~10 %, flow-encoding lobes), the
        // frame's field, water / fat decay and fat fraction.
        // g_{f,e}(v, τ) of frame model f and encode e, handed to `use_g` (one definition for the build, the
        // self-check and the error report)
        let frame_model = |f: &RateFrame, e: usize, use_g: &mut dyn FnMut(&(dyn Fn(usize, f64) -> Complex64 + Sync))| {
            let conc0 = segment_phase(&grid, &wbar[e], &tau_adc, false);
            let ta = tau_adc[0];
            let psi_raw = |v: usize, t: f64| conc0(v, t) + f.om[v] as f64 * t;
            let psi = field_poly.residual(&sd, &psi_raw);
            let g = |v: usize, t: f64| -> Complex64 {
                let ph = Complex64::from_polar(1.0, -psi(v, t));
                let ffv = f.ff[v] as f64;
                let wat = (1.0 - ffv) * (-(f.rw[v] as f64) * (t - ta)).exp();
                let fat = ffv * (-(f.rf[v] as f64) * (t - ta)).exp();
                ph * (Complex64::new(wat, 0.0) + fat * cfat(t))
            };
            use_g(&g);
        };
        let frame_tab = |f: &RateFrame, e: usize| -> hasty_recon::setop::HostSegTab {
            let mut out = None;
            frame_model(f, e, &mut |g| {
                let (sp, at, ta_, dt_) = if seg_svd { svd_build(nseg, g) } else { build_segments_complex(nv, nseg, tw0, tw1, ntab, &seg_fit, g) };
                out = Some(hasty_recon::setop::HostSegTab { segph: sp, atab: at, t0: ta_, dt: dt_ });
            });
            out.unwrap()
        };
        frame_tabs = rm_frames.as_ref().map(|rm| {
            // host memory before the build (no-OOM ledger): nf · ne tables of L diagonals (+ interpolators, ~ntab · L)
            let need = rm.len() * ne * (nseg * nv + ntab * nseg) * 8;
            let avail = available_memory();
            println!("per-frame ratemap segments (frame mode): {} tables × L = {nseg} need {:.2} GiB on the host, {:.2} GiB available", rm.len() * ne, need as f64 / GIB, avail as f64 / GIB);
            assert!(need * 10 < avail * 7, "--ratemap truth-bc (frame mode): the per-(frame, encode) tables need {:.2} GiB, more than 70 % of the {:.2} GiB available", need as f64 / GIB, avail as f64 / GIB);
            let t_seg = Instant::now();
            let mut tabs: Vec<hasty_recon::setop::HostSegTab> = Vec::with_capacity(rm.len() * ne);
            for (fi, f) in rm.iter().enumerate() {
                for e in 0..ne {
                    let tab = frame_tab(f, e);
                    if (fi == 0 && e == 0) || (fi == rm.len() / 2 && e == ne - 1) {
                        print!("  frame {fi}: ");
                        frame_model(f, e, &mut |g| seg_err(e, &tab.segph, &tab.atab, tab.t0, tab.dt, nseg, g));
                    }
                    tabs.push(tab);
                }
            }
            let bytes: usize = tabs.iter().map(|t| (t.segph.len() + t.atab.len()) * 8).sum();
            println!(
                "per-frame ratemap segments (frame mode): {} frames × {ne} encodes × L = {nseg}, {:.2} GiB on the host, built in {:.1}s",
                rm.len(),
                bytes as f64 / GIB,
                t_seg.elapsed().as_secs_f64()
            );
            std::sync::Arc::new(tabs)
        });
        if pp_args.active() {
            // host copies of the stored segment tables for --pp-* (G2 recentring, G3 surrogate); None if unsupported
            pp_seg_host = segs.iter().map(|s| s.host_tables(&st)).collect::<Result<Vec<_>, _>>().map_err(|e| println!("pp: no host segment tables ({e:?})")).ok();
        }
        println!("  [build timer] segment bases of all encodes done: {:.1}s since the build start", ts.elapsed().as_secs_f64());
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
                // knot mode: per-sample knot coordinates (cardiac t in knot units from the tables' frame coordinate,
                // respiratory interval / fraction of the sample's readout), on the whole records, then cropped
                let mut knot_coords = |tf: &[f32]| -> Option<(Vec<f32>, Vec<u32>, Vec<f32>)> {
                    let (b, per, _) = knot_basis.as_ref()?;
                    let ros = &rec_sets[t].readouts;
                    let na = sd.na;
                    let (mut kt, mut ri, rv) = b.sample_coords(tf, nf, per, &|j| ros[j / na]);
                    // temporal Gram of the basis under the sample times (every 7th sample; knot metric weights)
                    for j in (0..kt.len()).step_by(7) {
                        let w = b.sample_weights(kt[j] as f64, ri[j], rv[j]);
                        for &(a, wa) in &w {
                            for &(c, wc) in &w {
                                knot_gram_acc[a * b.nk() + c] += wa * wc;
                            }
                        }
                        knot_gram_n += 1;
                    }
                    // spline mode: the operator's coordinates (cardiac in frame units of fop, the interval local to
                    // the set's respiratory group); the Gram above is the parameter basis'
                    if let Some(of) = &op_frames {
                        let g = fe[t].0.unwrap();
                        let (mc, fop) = (b.mc as f64, of.fop as f64);
                        for j in 0..kt.len() {
                            kt[j] = (((kt[j] as f64 + 0.5) * fop / mc - 0.5).rem_euclid(fop)) as f32;
                            let (gg, li) = of.group_of(ri[j]);
                            assert_eq!(gg, g, "sample of set {t} in respiratory group {gg}");
                            ri[j] = li;
                        }
                    }
                    Some(match &crop_sel {
                        Some(sel) => (hasty_recon::pyramid::gather(&kt, &sel[t]), hasty_recon::pyramid::gather(&ri, &sel[t]), hasty_recon::pyramid::gather(&rv, &sel[t])),
                        None => (kt, ri, rv),
                    })
                };
                let mut p = (|| {
                if let (Some(f4), Some(h)) = (&fit4, &htab) {
                    let (mut tab, dw) = seg_tables_hi(&sd, &rec_sets[t], f4, with_conc, with_conc.then(|| &wbar[enc_of(t)][..]));
                    absorb_q = absorb_q.max(hasty_recon::segphys::absorb_field_poly(&mut tab, &sd, &field_poly, &fit_vox));
                    let kc = knot_coords(&tab.t);
                    let (tab, dw) = match &crop_sel { Some(sel) => { let m = tab.tau.len(); (tab.select(&sel[t]), hasty_recon::pyramid::gather_rows(&dw, m, &sel[t])) } None => (tab, dw) };
                    if let Some(pc) = pp_coords.as_mut() {
                        pc[t] = tab.coords.clone();
                    }
                    resid = resid.max(tab.max_resid_phase);
                    if let Some((kt, ri, rv)) = kc {
                        return eng.points2d_hi(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, &kt, &ri, &rv, &dw, h).unwrap();
                    }
                    return eng.points_hi(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, &dw, h, temporal.map(|_| &tab.t[..])).unwrap();
                }
                let mut tab = seg_tables(&sd, &rec_sets[t], &fit, with_conc, with_conc.then(|| &wbar[enc_of(t)][..]));
                absorb_q = absorb_q.max(hasty_recon::segphys::absorb_field_poly(&mut tab, &sd, &field_poly, &fit_vox));
                let kc = knot_coords(&tab.t);
                let tab = match &crop_sel { Some(sel) => tab.select(&sel[t]), None => tab };
                if let Some(pc) = pp_coords.as_mut() {
                    pc[t] = tab.coords.clone();
                }
                resid = resid.max(tab.max_resid_phase);
                if let Some((kt, ri, rv)) = kc {
                    return eng.points2d(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, &kt, &ri, &rv).unwrap();
                }
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
                    let reserve = if let Some((kr, _, mc, _)) = eng_knots {
                        // knot mode: one encode's set image + adjoint, the residual (and the set's data when staged) of
                        // the largest set, the resident data otherwise; the ledger revisits the placement
                        let m_max = host_sets.iter().map(|s| s.tau.len()).max().unwrap_or(0) as f64;
                        (kr * mc * nv * 8) as f64 + 2.0 * nvc as f64 * m_max * 8.0 + data_b + 2.0 * 1073741824.0
                    } else if vmodel.is_some() { 3.0 * 1073741824.0 } else { data_b + (6.0 + if lagged.is_some() { 2.0 } else { 0.0 }) * stack + 1073741824.0 };
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
        println!("  [build timer] sample tables (280 sets) done: {:.1}s since the build start", ts.elapsed().as_secs_f64());
        if tables_host == Some(true) {
            eng.stream_reserve(&st, &pts.iter().collect::<Vec<_>>()).unwrap();
            eng.stream_prefetch = table_prefetch;
            let (logical, stored) = (pts.iter().map(|p| p.table_bytes()).sum::<usize>(), pts.iter().map(|p| p.host_stored_bytes()).sum::<usize>());
            println!("sample tables streamed: {:.2} GiB on the host ({}, {:.2} GiB of tables), {:.2} GiB device slots, prefetch {}", stored as f64 / 1073741824.0, if pts[0].is_packed() { format!("packed ×{:.2}", logical as f64 / stored.max(1) as f64) } else { "pinned".into() }, logical as f64 / 1073741824.0, eng.stream_slot_bytes() as f64 / 1073741824.0, if table_prefetch { "on (second stream)" } else { "off" });
        }
        if b0_absorb >= 2 {
            println!("--b0-absorb 2: largest per-sample quadratic B0 phase in the ΔC Taylor correction {absorb_q:.3} rad (Taylor-2 truncation ≈ {:.1e})", absorb_q.powi(3) / 6.0);
        }
        let (tune_t0, tune_cs0, tune_tm0) = (Instant::now(), hasty_cuda::compile_stats(), hasty_nufft::tune_timing_stats());
        let (kf, ka) = eng.autotune(&st, &segs[0], &pts[0]).unwrap();
        println!("  [build timer] autotune done: {:.1}s since the build start", ts.elapsed().as_secs_f64());
        {
            // where the autotune's time went (HASTY_TUNE=reuse: a cache hit times nothing; research/arch_targets.md item 0)
            let (cs, tm) = (hasty_cuda::compile_stats().since(&tune_cs0), hasty_nufft::tune_timing_stats());
            let (all, timing) = (tune_t0.elapsed().as_secs_f64(), tm.1 - tune_tm0.1);
            println!(
                "  autotune {all:.1} s: timing launches {timing:.1} s ({} candidates), NVRTC compiles {:.1} s ({}), cubin cache reads {:.1} s ({}), module loads {:.1} s ({}), rest {:.1} s (HASTY_TUNE={:?})",
                tm.0 - tune_tm0.0, cs.miss_s, cs.misses, cs.hit_s, cs.hits, cs.load_s, cs.loads, all - timing - cs.miss_s - cs.hit_s - cs.load_s, hasty_nufft::tune_mode()
            );
        }
        println!(
            "SegNufft: L = {nseg}, order {}, frames/set {}, coil batch {}, kernels {kf:.1}/{ka:.1} ms (set 0), max phase left out by the {} fit {resid:.2e} rad, built in {:.1}s",
            eng.order,
            eng.frames,
            eng.cb,
            if conc_hi { "degree-4" } else { "quadratic" },
            ts.elapsed().as_secs_f64()
        );
        let joint_maps = ff_joint.as_ref().filter(|ff| ff.len() > nv).map(|ff| eng.fat_maps(&st, ff).unwrap());
        ff_per_frame = joint_maps.is_some();
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
        if let Some(hh) = held_host.take() {
            // --holdout-res: a second engine with the same options, coil maps and segment bases; the held-out sets'
            // tables by the same builder as the solve's (degree-4 or quadratic fit, field absorption, kept samples);
            // tables built before the engine's tuning, as above; the launch shapes from the solve's tune record (set 0)
            let th = Instant::now();
            println!("--holdout-res: held-out operator (second engine, same options; segment bases rebuilt):");
            let mut eng_h = new_engine(nseg);
            eng_h.set_maps(&st, &smaps).unwrap();
            let segs_h = make_segs(&eng_h, nseg);
            let build_h = |eng_h: &mut SegNufft, rec: &SampleSet, sel: Option<&[u32]>, t: usize| {
                    let wb = with_conc.then(|| &wbar[enc_of(t)][..]);
                    if let (Some(f4), Some(h)) = (&fit4, &htab) {
                        let (mut tab, dw) = seg_tables_hi(&sd, rec, f4, with_conc, wb);
                        hasty_recon::segphys::absorb_field_poly(&mut tab, &sd, &field_poly, &fit_vox);
                        let (tab, dw) = match sel { Some(v) => { let m = tab.tau.len(); (tab.select(v), hasty_recon::pyramid::gather_rows(&dw, m, v)) } None => (tab, dw) };
                        return eng_h.points_hi(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, &dw, h, temporal.map(|_| &tab.t[..])).unwrap();
                    }
                    let mut tab = seg_tables(&sd, rec, &fit, with_conc, wb);
                    hasty_recon::segphys::absorb_field_poly(&mut tab, &sd, &field_poly, &fit_vox);
                    let tab = match sel { Some(v) => tab.select(v), None => tab };
                    eng_h.points(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, temporal.map(|_| &tab.t[..])).unwrap()
            };
            let pts_h: Vec<_> = (0..fe.len()).map(|t| build_h(&mut eng_h, &hh.recs[t], hh.sels[t].as_deref(), t)).collect();
            // check: solve set 0's own records through this builder and engine = the solve operator on set 0
            let chk0 = build_h(&mut eng_h, &rec_sets[0], crop_sel.as_ref().map(|v| &v[0][..]), 0);
            for (t, p) in pts_h.iter().enumerate() {
                assert_eq!(p.m * nvc, hh.ys[t].len(), "--holdout-res: held-out set {t}: tables and data disagree");
            }
            if tables_host == Some(true) {
                eng_h.stream_reserve(&st, &pts.iter().collect::<Vec<_>>()).unwrap();
            }
            let (kf, ka) = eng_h.autotune(&st, &segs_h[0], &pts[0]).unwrap();
            {
                let probe1: Vec<C32> = match warm.first() {
                    Some(w0) => w0.clone(),
                    None => (0..nv).map(|q| C32::new(((q * 7919) % 1000) as f32 / 1000.0 - 0.5, ((q * 104729) % 1000) as f32 / 1000.0 - 0.5)).collect(),
                };
                // --temporal: set 0 holds all knot images (the probe in every knot)
                let probe: Vec<C32> = if temporal.is_some() { probe1.iter().cycle().take(nf * nv).copied().collect() } else { probe1 };
                let dx = st.clone_htod(&probe).unwrap();
                let mut ya = st.alloc_zeros::<C32>(nvc * pts[0].m).unwrap();
                let mut yb = st.alloc_zeros::<C32>(nvc * chk0.m).unwrap();
                eng.forward(&st, &segs[enc_of(0)], &pts[0], &dx, &mut ya).unwrap();
                eng_h.forward(&st, &segs_h[enc_of(0)], &chk0, &dx, &mut yb).unwrap();
                let na = kern.dot(&st, &ya, &ya).unwrap().re.sqrt();
                kern.axpby(&st, Complex64::new(-1.0, 0.0), &ya, Complex64::new(1.0, 0.0), &mut yb).unwrap();
                let d = kern.dot(&st, &yb, &yb).unwrap().re.sqrt() / na.max(1e-30);
                println!("--holdout-res check: solve set 0 through the held-out builder + engine vs the solve operator: relative difference {d:.2e} ({})", if warm.is_empty() { "probe" } else { "warm image" });
                assert!(d < 1e-5, "--holdout-res check failed: the held-out builder + engine differ from the solve operator by {d:.2e}");
            }
            drop(chk0);
            let ny2: f64 = hh.ys.iter().flat_map(|y| y.iter()).map(|c| (c.re as f64).powi(2) + (c.im as f64).powi(2)).sum();
            let tb: usize = pts_h.iter().map(|p| p.table_bytes()).sum();
            println!("--holdout-res: {} sets, {} readouts, tables {:.2} GiB on the device, kernels {kf:.1}/{ka:.1} ms, ‖y_h‖² {ny2:.4e} ({:.1}s)", pts_h.len(), hh.nro, tb as f64 / GIB, th.elapsed().as_secs_f64());
            let slot_h = |t: usize| if frame_tabs.is_some() { 0 } else { enc_of(t) };
            held_eval = Some(HeldOut {
                sets: SegSets { eng: eng_h, pts: pts_h, segs: segs_h, seg_of_set: (0..fe.len()).map(slot_h).collect(), kern: kern.clone(), surrogate: None },
                ys: hh.ys,
                ny2,
                nro: hh.nro,
                maps: Vec::new(),
                map_of_set: Vec::new(),
                seg_tabs: frame_tabs.clone(),
                seg_tab_of_set: (0..fe.len()).map(|t| fe[t].0.unwrap_or(0) * ne + enc_of(t)).collect(),
                seg_cur: None,
            });
            dev_free("after the held-out operator");
        }
        // per-frame tables (frame_tabs): every set uses slot 0, loaded with its frame's table (setop::PerSetSegs below)
        let slot_of = |t: usize| if frame_tabs.is_some() { 0 } else { enc_of(t) };
        seg_sets = SegSets { eng, pts, segs, seg_of_set: (0..fe.len()).map(slot_of).collect(), kern: kern.clone(), surrogate: None };
        // frame mode self-check on the real sets (startup; aborts on failure): the frame-table builder in its degenerate
        // case (the static field the static tables use, no fat, no decay) must reproduce the static table of the set's
        // encode to the fp32 floor; then the size of the true per-frame model change on the same sets.
        if let (Some(tabs), Some(rm)) = (&frame_tabs, &rm_frames) {
            let t_chk = Instant::now();
            let om_static: Vec<f64> = (0..nv).map(|v| if with_b0 { grid.omega[v] } else { 0.0 }).collect();
            let deg = RateFrame { om: om_static, rw: vec![0.0; nv], rf: vec![0.0; nv], ff: vec![0.0; nv] };
            let t_a = (0..fe.len()).find(|&t| enc_of(t) == 0).unwrap();
            let t_b = (0..fe.len()).rev().find(|&t| enc_of(t) == ne - 1 && fe[t].0 == Some(rm.len() / 2)).unwrap_or(fe.len() - 1);
            let mut worst = 0.0f64;
            for &t in &[t_a, t_b] {
                let (e, f) = (enc_of(t), fe[t].0.unwrap());
                let lx = seg_sets.len_x(t);
                let mut z = 0x9e3779b97f4a7c15u64 ^ t as u64;
                // random image on the support (2026-11-24: with maps that are nonzero in air, the air voxels' large
                // fields made the f32-stored field of the frame model differ by 7e-5 from the static f64 one there)
                let x: Vec<C32> = (0..lx).map(|i| { let mut u = || { z ^= z << 13; z ^= z >> 7; z ^= z << 17; (z >> 11) as f32 / (1u64 << 53) as f32 - 0.5 }; let c = C32::new(u(), u()); if support[i % nv] { c } else { C32::new(0.0, 0.0) } }).collect();
                let dx = st.clone_htod(&x).unwrap();
                let ny = seg_sets.len_y(t);
                let fw = |sets: &mut SegSets, slot: usize| -> Vec<C32> {
                    sets.seg_of_set[t] = slot;
                    let mut y = st.alloc_zeros::<C32>(ny).unwrap();
                    sets.forward(&st, t, &dx, &mut y).unwrap();
                    st.clone_dtoh(&y).unwrap()
                };
                let rel = |a: &[C32], b: &[C32]| -> f64 {
                    let (mut d, mut n) = (0.0f64, 0.0f64);
                    for (p, q) in a.iter().zip(b) { d += ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2); n += (q.re as f64).powi(2) + (q.im as f64).powi(2); }
                    assert!(n > 0.0, "frame-mode self-check: set {t} gives a zero reference (no samples?)");
                    (d / n).sqrt()
                };
                assert!(ny > 0, "frame-mode self-check: set {t} has no samples");
                // the table index f·ne + e: the stored table is the one the builder gives for (frame f, encode e)
                let rebuilt = frame_tab(&rm[f], e);
                let stored = &tabs[f * ne + e];
                let tdiff = |a: &[C32], b: &[C32]| -> f64 {
                    let ab = |re: f32, im: f32| ((re as f64).powi(2) + (im as f64).powi(2)).sqrt();
                    let m = b.iter().map(|z| ab(z.re, z.im)).fold(0.0, f64::max).max(1e-30);
                    a.iter().zip(b).map(|(p, q)| ab(p.re - q.re, p.im - q.im)).fold(0.0, f64::max) / m
                };
                let (d_sp, d_at) = (tdiff(&rebuilt.segph, &stored.segph), tdiff(&rebuilt.atab, &stored.atab));
                assert!(rebuilt.segph.len() == stored.segph.len() && d_sp < 1e-5 && d_at < 1e-5, "frame-mode self-check: table {} is not (frame {f}, encode {e}): diagonals {d_sp:.2e}, interpolators {d_at:.2e}", f * ne + e);
                let y_static = fw(&mut seg_sets, e);
                let tab_d = frame_tab(&deg, e);
                let slot = seg_sets.segs.len();
                seg_sets.segs.push(seg_sets.eng.segments(&st, &tab_d.segph, &tab_d.atab, tab_d.t0, tab_d.dt).unwrap());
                let y_deg = fw(&mut seg_sets, slot);
                let tab_t = &tabs[f * ne + e];
                seg_sets.segs[slot] = seg_sets.eng.segments(&st, &tab_t.segph, &tab_t.atab, tab_t.t0, tab_t.dt).unwrap();
                let y_true = fw(&mut seg_sets, slot);
                seg_sets.segs.pop();
                seg_sets.seg_of_set[t] = slot_of(t);
                let (e_deg, e_true) = (rel(&y_deg, &y_static), rel(&y_true, &y_static));
                println!("frame-mode self-check, set {t} (frame {f}, encode {e}): degenerate frame table vs the static table {e_deg:.2e}; the frame's true ratemap vs the static table {e_true:.2e} (random image on the support)");
                worst = worst.max(e_deg);
            }
            assert!(worst < 1e-5, "frame-mode self-check failed: the degenerate frame table differs from the static table by {worst:.2e} (builder or slot bug)");
            println!("frame-mode self-check passed ({:.1}s)", t_chk.elapsed().as_secs_f64());
        }
        // --temporal self-check on a real set (startup; aborts on failure): set 0 (encode 0, every knot) through the
        // temporal engine against an independent reference, the trusted static engine (one image) applied per knot k
        // with knot k's table, each sample weighted on the host by B_p(t_s − k) (hasty_recon::vmodel::bspline, the
        // engine's convention and the scores' box weights): y = Σ_k diag(B_p(t − k)) E[table_k] c_k. Then the adjoint
        // (dot test) of the temporal operator.
        // the static-ratemap reference (degenerate frame model) equals make_segs' static table only for phase-only
        // static tables (no --rate / --r2s / joint fat / node-weighted builder): otherwise the check is skipped
        let temporal_static_ok = rate_dw.is_none() && ff_joint.is_none() && r2v.is_none() && !seg_nodes_w;
        if let (Some(pdeg), false) = (temporal, rm_knots.is_none() && !temporal_static_ok) {
            let t_chk = Instant::now();
            let om_static: Vec<f64> = (0..nv).map(|v| if with_b0 { grid.omega[v] } else { 0.0 }).collect();
            let deg = RateFrame { om: om_static, rw: vec![0.0; nv], rf: vec![0.0; nv], ff: vec![0.0; nv] };
            let mut z = 0x2545f4914f6cdd1du64;
            let mut u = || { z ^= z << 13; z ^= z >> 7; z ^= z << 17; (z >> 11) as f32 / (1u64 << 53) as f32 - 0.5 };
            // sets 0 and ne − 1: the first and the last encode
            for t in [0usize, fe.len() - 1] {
                let e = enc_of(t);
                // the set's host sample tables, as the solve built them (field-poly absorption, k-space crop)
                let wb = with_conc.then(|| &wbar[e][..]);
                let (tab, dw) = match (&fit4, &htab) {
                    (Some(f4), Some(_)) => {
                        let (mut tab, dw) = seg_tables_hi(&sd, &rec_sets[t], f4, with_conc, wb);
                        hasty_recon::segphys::absorb_field_poly(&mut tab, &sd, &field_poly, &fit_vox);
                        match &crop_sel { Some(sel) => { let m = tab.tau.len(); (tab.select(&sel[t]), Some(hasty_recon::pyramid::gather_rows(&dw, m, &sel[t]))) } None => (tab, Some(dw)) }
                    }
                    _ => {
                        let mut tab = seg_tables(&sd, &rec_sets[t], &fit, with_conc, wb);
                        hasty_recon::segphys::absorb_field_poly(&mut tab, &sd, &field_poly, &fit_vox);
                        (match &crop_sel { Some(sel) => tab.select(&sel[t]), None => tab }, None)
                    }
                };
                let m = tab.tau.len();
                assert!(m > 0, "--temporal self-check: set {t} has no samples");
                assert_eq!(m * nvc, seg_sets.len_y(t), "--temporal self-check: the rebuilt tables of set {t} differ from the solve's ({m} samples)");
                let mut o_ref = seg_opts(nseg);
                o_ref.frames = 1;
                o_ref.max_grid_bytes = 1 << 30;
                let mut eng_r = SegNufft::new(&dev, sd.dims, sd.vox, nvc, o_ref).unwrap();
                eng_r.set_maps(&st, &smaps).unwrap();
                let pts_r = match (&dw, &htab) {
                    (Some(dw), Some(h)) => eng_r.points_hi(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, dw, h, None).unwrap(),
                    _ => eng_r.points(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, None).unwrap(),
                };
                // random knot images on the support
                let c: Vec<C32> = (0..nf * nv).map(|i| { let c = C32::new(u(), u()); if support[i % nv] { c } else { C32::new(0.0, 0.0) } }).collect();
                let dc_ = st.clone_htod(&c).unwrap();
                let mut y_t = st.alloc_zeros::<C32>(nvc * m).unwrap();
                seg_sets.forward(&st, t, &dc_, &mut y_t).unwrap();
                let y_t = st.clone_dtoh(&y_t).unwrap();
                // knot k's static-engine forward E[table_j] c_k for table j = k (reference) and j = k + 1 (control: a
                // one-knot table shift must show, or the check cannot see per-knot table mistakes)
                let tabs_k: Vec<hasty_recon::setop::HostSegTab> = (0..nf).map(|k| frame_tab(rm_knots.as_ref().map_or(&deg, |rm| &rm[k]), e)).collect();
                let shift_ctl = rm_knots.is_some();
                let mut y_ref = vec![Complex64::new(0.0, 0.0); nvc * m];
                let mut y_sh = vec![Complex64::new(0.0, 0.0); nvc * m];
                for k in 0..nf {
                    let wk: Vec<f64> = tab.t.iter().map(|&ts| { let mut d = ts as f64 - k as f64; d -= (d / nf as f64).round() * nf as f64; hasty_recon::vmodel::bspline(pdeg, d) }).collect();
                    if wk.iter().all(|&x| x == 0.0) {
                        continue;
                    }
                    let xk = st.clone_htod(&c[k * nv..(k + 1) * nv]).unwrap();
                    for (j, acc, is_ctl) in [(k, &mut y_ref, false), ((k + 1) % nf, &mut y_sh, true)] {
                        if is_ctl && !shift_ctl {
                            continue;
                        }
                        let tk = &tabs_k[j];
                        let sk = eng_r.segments(&st, &tk.segph, &tk.atab, tk.t0, tk.dt).unwrap();
                        let mut yk = st.alloc_zeros::<C32>(nvc * m).unwrap();
                        eng_r.forward(&st, &sk, &pts_r, &xk, &mut yk).unwrap();
                        let yk = st.clone_dtoh(&yk).unwrap();
                        for ch in 0..nvc {
                            for (s_, &w) in wk.iter().enumerate() {
                                if w != 0.0 {
                                    let v = yk[ch * m + s_];
                                    acc[ch * m + s_] += Complex64::new(v.re as f64, v.im as f64) * w;
                                }
                            }
                        }
                    }
                }
                let rel_to = |r: &[Complex64]| -> f64 {
                    let (mut d2, mut n2) = (0.0f64, 0.0f64);
                    for (a, b) in y_t.iter().zip(r) {
                        d2 += (a.re as f64 - b.re).powi(2) + (a.im as f64 - b.im).powi(2);
                        n2 += b.norm_sqr();
                    }
                    assert!(n2 > 0.0, "--temporal self-check: zero reference");
                    (d2 / n2).sqrt()
                };
                let e_fwd = rel_to(&y_ref);
                let e_sh = if shift_ctl { rel_to(&y_sh) } else { f64::NAN };
                drop(pts_r);
                drop(eng_r);
                // adjoint: ⟨A c, y⟩ = ⟨c, Aᴴ y⟩
                let yr: Vec<C32> = (0..nvc * m).map(|_| C32::new(u(), u())).collect();
                let dyr = st.clone_htod(&yr).unwrap();
                let mut ah = st.alloc_zeros::<C32>(nf * nv).unwrap();
                seg_sets.adjoint(&st, t, &dyr, &mut ah).unwrap();
                let ah = st.clone_dtoh(&ah).unwrap();
                let lhs: Complex64 = y_t.iter().zip(&yr).map(|(a, b)| Complex64::new(b.re as f64, -b.im as f64) * Complex64::new(a.re as f64, a.im as f64)).sum();
                let rhs: Complex64 = c.iter().zip(&ah).map(|(a, b)| Complex64::new(b.re as f64, -b.im as f64) * Complex64::new(a.re as f64, a.im as f64)).sum();
                let e_adj = (lhs - rhs).norm() / lhs.norm().max(1e-300);
                println!("--temporal self-check, set {t} (encode {e}, {m} samples, {} knot ratemaps): forward vs the per-knot static reference {e_fwd:.2e}; one-knot table shift (control) {e_sh:.2e}; adjoint dot test {e_adj:.2e}", if rm_knots.is_some() { "true" } else { "static" });
                assert!(e_fwd < 1e-4 && e_adj < 1e-4, "--temporal self-check failed: forward {e_fwd:.2e}, adjoint {e_adj:.2e} (engine, time weights or per-knot tables)");
                assert!(!shift_ctl || e_sh > 10.0 * e_fwd.max(1e-7), "--temporal self-check cannot discriminate per-knot tables: the one-knot shift gives {e_sh:.2e} vs {e_fwd:.2e}");
            }
            println!("--temporal self-check passed ({:.1}s)", t_chk.elapsed().as_secs_f64());
        } else if temporal.is_some() {
            println!("--temporal self-check skipped: the static reference covers phase-only static tables only (--rate / --r2s / joint fat / node-weighted segments)");
        }
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
        // (--pp-*: the surrogate symbols and the centre-pass times need the kept samples' coordinates and times)
        let (keep_coords, keep_tau) = (voxel_basis.is_some() || pp_args.active(), (fat_fraction.is_some() && !fat_joint) || pp_args.active());
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
    // --ratemap truth-bc in frame mode: set t = (frame f, encode e) uses frame f's segment table (slot 0 of the
    // segment operator), in the warm start, the solve and the held-out residual
    let mut frame_seg_sets;
    let sop: &mut dyn SetOp = match frame_tabs.clone() {
        None => sop,
        Some(tabs) => {
            assert!(model == "seg" && lag_tau.is_none() && coil_sampling.is_none() && !pp_args.active(), "--ratemap truth-bc (frame mode): --model seg, without --lagged / --coil-sampling / --pp-*");
            let tab_of_set: Vec<usize> = (0..fe.len()).map(|t| fe[t].0.expect("--ratemap truth-bc: frame mode sets carry a frame") * ne + enc_of(t)).collect();
            assert!(tab_of_set.iter().all(|&k| k < tabs.len()), "--ratemap truth-bc (frame mode): table index ≥ {} tables", tabs.len());
            frame_seg_sets = hasty_recon::setop::PerSetSegs::new(&st, sop, tabs.clone(), tab_of_set, vec![0; fe.len()]).unwrap();
            println!("--ratemap truth-bc (frame mode): {} per-frame segment tables, one uploaded per frame change into the operator's slot 0", tabs.len());
            &mut frame_seg_sets
        }
    };
    // --smaps-resp: respiratory bin r's maps for every set of bin r (frame f = r · nf_card + c), in the solve and in the
    // held-out residual
    let mut resp_sets;
    let sop: &mut dyn SetOp = if maps_resp.is_empty() {
        sop
    } else {
        assert!(model == "seg" && lag_tau.is_none() && coil_sampling.is_none(), "--smaps-resp: --model seg, without --lagged / --coil-sampling");
        let map_of_set: Vec<Option<usize>> = (0..fe.len()).map(|t| fe[t].0.map(|f| f / nf_card)).collect();
        let mh: Vec<Vec<C32>> = maps_resp.iter().map(|(m, _)| m.clone()).collect();
        if let Some(h) = held_eval.as_mut() {
            h.maps = mh.iter().map(|m| st.clone_htod(m).unwrap()).collect();
            h.map_of_set = map_of_set.clone();
        }
        resp_sets = hasty_recon::setop::PerSetMaps::new(&st, sop, &mh, map_of_set).unwrap();
        println!("--smaps-resp: {} coil-map sets ({:.2} GiB on the device), one per respiratory bin", mh.len(), (mh.len() * nvc * nv * 8) as f64 / GIB);
        &mut resp_sets
    };
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
    // --temporal: the scores read frame f as the knot model's box average over the frame, x̄_f = Σ_k W_fk c_k with
    // W_fk = ∫_{f−½}^{f+½} B_p(t − k) dt (periodic): the observable of the binned images and of the truth (frame means);
    // knot coefficients are not frame values (cubic: c_k ≠ x(k))
    let tbox: Option<Vec<f64>> = temporal.map(|p| temporal_box_weights(nf, p));
    if let Some(w) = &tbox {
        let row: Vec<String> = (0..nf.min(5)).map(|k| format!("{:.4}", w[k])).collect();
        println!("--temporal scoring: frame box averages of the knot model, W[0][0..] = {} … (rows sum to 1)", row.join(" "));
    }
    // --synthetic with data staged from the store (they do not fit on the device): the residuals of model-consistent
    // data y_t = E_t x★_t − n_t, formed on the fly as E_t (x − x★_t) + n_t (hasty_recon::synth; no second store; the
    // staged measured data are not read). x★ is the --init-truth image (warm images per encode, the true velocity
    // phases on warm[0] in voxels of lumen fraction > 0.5), so --synthetic --init-truth starts at the exact solution of
    // the noise-free data term: a drift away from it is noise + prior (exact model), against model error on the
    // measured data. Noise: complex Gaussian, σ / √2 per component (σ: HASTY_NOISE_SIGMA, else the sim's noise_sigma;
    // × NOISE_SCALE), counter-based per (seed, set, coil, sample): the same in every pass and coil subset.
    let mut synth_sets;
    // --synth-x pc|pd: the per-encode frame mean of x★ (what a pooled solve should reach) and the blood-voxel mask, for
    // the pooled-solve image error printed by --warm-full (scoring only)
    let mut synth_ref: Option<(Vec<Vec<C32>>, Vec<f32>, Vec<bool>)> = None;
    let sop: &mut dyn SetOp = if synthetic && staged {
        assert!(vmodel.is_none(), "--synthetic with staged data: image-space P²nP only (not --vmodel)");
        assert!(warm_full == 0 || synth_pc, "--synthetic with staged data: no --warm-full (the reference uses the warm images; --synth-x pc does not)");
        let t_s = Instant::now();
        let sigma = std::env::var("HASTY_NOISE_SIGMA").ok().and_then(|v| v.parse::<f64>().ok()).filter(|v| *v > 0.0).or_else(|| sd.meta["noise_sigma"].as_f64().filter(|v| *v > 0.0)).unwrap_or(1.0)
            / 2f64.sqrt()
            * std::env::var("NOISE_SCALE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
        let seed = std::env::var("HASTY_SYNTH_SEED").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0x51);
        let mut lumen: Vec<(usize, C32)> = vec![];
        for q in 0..nv {
            if sample_truth(&sd.truth_frac, 0, &grid.x_log[q]) > 0.5 {
                lumen.push((q, warm.first().map_or(C32::new(0.0, 0.0), |w| w[q])));
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
        // --synth-x pc: x★ = PD + the truth's blood signal on this grid (hasty_recon::pcscore::synth_object)
        let pc_obj = synth_pc.then(|| {
            let nat = pc_nat.as_ref().expect("--synth-x pc needs --score-pc");
            let (stat, mut blood) = hasty_recon::pcscore::synth_object(nat, &sd, &grid.x_log, &dir).unwrap();
            if synth_static {
                blood.clear();
                println!("--synth-x pd: static object only (no blood signal)");
            }
            println!("synthetic x★ (--synth-x pc): static PD_water + PD_fat box means on this grid, {} voxels with blood signal (truth box means, encode 0 real); median |blood| / median PD in the body: {:.3}", blood.len(), {
                let mut b: Vec<f64> = blood.iter().map(|(_, c)| c[0].norm()).collect();
                let mut s: Vec<f64> = stat.iter().filter(|&&v| v > 0.0).map(|&v| v as f64).collect();
                b.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                b.get(b.len() / 2).copied().unwrap_or(0.0) / s.get(s.len() / 2).copied().unwrap_or(1.0)
            });
            (stat, blood, nat.nf)
        });
        if let Some((stat, blood, nfc)) = &pc_obj {
            let mut xm: Vec<Vec<C32>> = (0..ne).map(|_| stat.iter().map(|&v| C32::new(v, 0.0)).collect()).collect();
            let mut bm = vec![false; nv];
            for (q, b) in blood {
                bm[*q] = true;
                for (e, x) in xm.iter_mut().enumerate() {
                    let z: Complex64 = (0..nf).map(|fr| b[e * nfc + fr % nfc]).sum::<Complex64>() / nf as f64;
                    x[*q] = C32::new(x[*q].re + z.re as f32, x[*q].im + z.im as f32);
                }
            }
            // the static object (2 MB at this grid): the error spectra of research/experiments/pooled_diag/err_spec.py
            save(&out.join("synth_pd.npy"), "<f4", &sd.dims, f4(stat.iter().copied()));
            synth_ref = Some((xm, stat.clone(), bm));
        }
        let truth: hasty_recon::synth::TruthFn = if let Some((stat, blood, nfc)) = pc_obj {
            Box::new(move |t: usize, out: &mut [C32]| {
                for &(fr, e, o) in &members[t] {
                    for (d, &s) in out[o..o + nv].iter_mut().zip(&stat) {
                        *d = C32::new(s, 0.0);
                    }
                    let c = fr % nfc;
                    for (q, b) in &blood {
                        let z = b[e * nfc + c];
                        { let a = out[o + q]; out[o + q] = C32::new(a.re + z.re as f32, a.im + z.im as f32); }
                    }
                }
            })
        } else { Box::new(move |t: usize, out: &mut [C32]| {
            for &(fr, e, o) in &members[t] {
                out[o..o + nv].copy_from_slice(&warm_r[e]);
                let c = fr % nf_card;
                for (l, &(q, r0)) in lumen.iter().enumerate() {
                    let v = lvel[c * nl + l];
                    let z = Complex64::new(r0.re as f64, r0.im as f64) * Complex64::from_polar(1.0, dec_r.phase(e, &v) - dec_r.phase(0, &v));
                    out[o + q] = C32::new(z.re as f32, z.im as f32);
                }
            }
        }) };
        synth_sets = hasty_recon::synth::SynthSets::new(&dev, sop, nvc, sigma, seed, truth).unwrap();
        if synth_pc {
            println!("synthetic model-consistent residuals E(x − x★) + n (staged data unused): x★ = --synth-x pc (above); noise σ {sigma:.3} per component, seed {seed}; data-term floor ½ E‖n‖² = {floor:.4e}");
        } else { println!(
            "synthetic model-consistent residuals E(x − x★) + n (staged data unused): x★ = warm images with the true velocity phases in {nl} lumen voxels (lumen fraction > 0.5); noise σ {sigma:.3} per component, seed {seed}; data-term floor ½ E‖n‖² = {floor:.4e} ({:.1}s)",
            t_s.elapsed().as_secs_f64()
        ); }
        &mut synth_sets
    } else {
        sop
    };
    // --rcompress: the banded R between the operator (all wrappers above act on the kept samples) and the data
    let mut rc_sets;
    let sop: &mut dyn SetOp = match rc_state {
        None => sop,
        Some((comp, scs)) => {
            // a ball ROI smaller than the grid: the unknowns outside it are excluded (mask before / after the operator)
            let mask = hasty_recon::rcompress::roi_mask(&comp.opts.roi, sd.dims, sd.vox);
            let rc = hasty_recon::rcompress::RcDev::new(&dev, &st, &comp, &scs).unwrap().with_mask(&st, mask.as_deref()).unwrap();
            println!("rcompress: R tables on the device ({:.3} GiB, q {}){}", comp.table_bytes() as f64 / GIB, rc.q, mask.as_ref().map_or(String::new(), |m| format!("; unknowns restricted to the ROI ({} of {} voxels)", m.iter().filter(|&&v| v > 0.0).count(), m.len())));
            rc_sets = hasty_recon::rcompress::RcSets::new(sop, rc);
            &mut rc_sets
        }
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
    // ───────────── velocity model at the knots of the 2-D respiratory × cardiac knot operator (--vmodel knots) ─────────────
    // θ = (z_k, v_k per knot, residual phase), operator frames = the K knots (identity weights), one set per encode;
    // state on the device or in host memory (θ streamed knot by knot), chosen by the knot ledger (vknot::plan)
    if let Some((kb, _, amp_all)) = knot_basis.as_ref() {
        use hasty_recon::hoststack::HostStack;
        use hasty_recon::vmodel::{VelModel, VelSpec};
        use hasty_recon::vobj::VelProblem;
        use rayon::prelude::*;
        assert!(!warm.is_empty(), "--vmodel knots needs --warm or --init-from (initialisation from the time-averaged reconstruction)");
        assert!(!synthetic, "--vmodel knots: --synthetic not supported (model-consistent knot data: hasty-recon tests/vknot.rs)");
        // fat: a static fat-fraction map with --fat-model joint is folded into the ratemap segment diagonals
        // (g_v(τ) = [(1 − ff_v) + ff_v c(τ)] e^{−R_v(τ−τ0)} e^{−iψ_v(τ)}), the operator is unchanged; per-frame maps and the
        // separate-fat-image model are not supported with knots
        assert!(fat_fraction.is_none() || (fat_joint && !ff_per_frame), "--vmodel knots: --fat-fraction needs --fat-model joint with one static map (per-frame fat maps vs knots: not supported)");
        assert!(air.is_none(), "--air: complex-image models only (not --vmodel)");
        assert!(lagged.is_none(), "--vmodel knots: no --lagged");
        assert!(learned.is_none(), "--vmodel knots: no --learned (7-point complex prior)");
        let (kr, mc, nk) = (kb.kr, kb.mc, kb.nk());
        let kenc: Vec<[f64; 3]> = sd.encode_m1.iter().map(|m| m.map(|x| hasty_mrsim::GAMMA * x)).collect();
        let spec = match &op_frames {
            Some(of) => VelSpec::splines(sd.dims, kenc.clone(), of.clone(), vpoly),
            None => VelSpec::at_knots(sd.dims, kenc.clone(), kr, mc, vpoly),
        };
        let spline = spec.spline();
        // safeguards of §F.2 (flags above): the velocity box and the robust start's SNR gate
        let vbox: Option<f64> = if !spline { None } else {
            match vbox_arg.as_deref() {
                Some("off") | Some("0") => None,
                None | Some("design") => Some(vext_design.unwrap_or(r_first)),
                Some(v) => Some(v.parse().expect("--vbox V | design | off")),
            }
        };
        let vinit_snr: Option<f64> = match vinit_snr_arg.as_deref() {
            Some("off") | Some("0") => None,
            None | Some("design") => Some(sd.meta["sequence"]["encoding"]["snr_unwrap_1e-3"].as_f64().unwrap_or(3.0)),
            Some(v) => Some(v.parse().expect("--vinit-snr S | design | off")),
        };
        if spline {
            println!("velocity safeguards: box {}, step trust region {}, start SNR gate {}", vbox.map_or("off".into(), |b| format!("|v_a| ≤ {b:.2} m/s")), vtrust.map_or("off".into(), |r| format!("max_e |k'_e·Δv| ≤ {r} rad")), vinit_snr.map_or("off".into(), |s| format!("{s:.2}")));
        }
        if let Some(of) = &op_frames {
            println!("spline velocity model (research/velocity_model_sr.md §F): m, ψ, v in the {kr} × {mc} basis; {}; encodes centred (k̄ = {:?} rad/(m/s))", of.describe(), spec.kbar().map(|v| (v * 1000.0).round() / 1000.0));
        }
        let vm = VelModel::new(&dev, &st, spec.clone()).unwrap();
        let lens = spec.lens();
        let stack_b = spec.stack_bytes();
        // the breathing amplitude of the outputs / in-driver scores: the median of the readouts used
        let amp_med = {
            let mut v: Vec<f64> = (0..sd.n_readouts()).filter(|&i| sd.ro.frame[i] != usize::MAX).map(|i| amp_all[i]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        // knot weights of the cardiac frame centres at that amplitude (the operator's interpolation)
        let wfr: Vec<Vec<f64>> = (0..nf_card).map(|f| kb.weights(amp_med, (f as f64 + 0.5) / nf_card as f64)).collect();
        let gn = knot_gram_n.max(1) as f64;
        let gram: Vec<f64> = knot_gram_acc.iter().map(|v| v / gn).collect();
        let gdiag: Vec<f64> = (0..nk).map(|k| gram[k * nk + k]).collect();
        let gcond = hasty_recon::knotsolve::gram_condition(&gram, nk);
        let gmean = gdiag.iter().sum::<f64>() / nk as f64;
        println!("knot model: {kr} × {mc} = {nk} knots ({}), {} parameters per voxel, {:.2} GiB per parameter stack; temporal Gram of the samples: condition {gcond:.1}, diagonal {:.3}..{:.3} × mean (per-knot metric weights)", kb.describe(), 5 * nk, stack_b as f64 / GIB, gdiag.iter().cloned().fold(f64::INFINITY, f64::min) / gmean, gdiag.iter().cloned().fold(0.0, f64::max) / gmean);
        // deviations from the target config (research/temporal_operator.md §21)
        let mut dev_target: Vec<String> = vec![];
        if (kr, kb.rdeg, mc, kb.cdeg) != (6, 1, 16, 3) {
            dev_target.push(format!("knots {kr}:{},{mc}:{} (target 6:1,16:3)", kb.rdeg, kb.cdeg));
        }
        if resp_src != "bellows" {
            dev_target.push(format!("breathing coordinate {resp_src} (target bellows; truth = oracle)"));
        }
        if !matches!(coil_rule.to_string().as_str(), s if s.starts_with("energy")) || coil_energy < 0.999 {
            dev_target.push(format!("coils {coil_rule} (target SV compression ≥ 99.9 %)"));
        }
        if phys != "full" {
            dev_target.push(format!("phys {phys} (target full)"));
        }
        if !gnl {
            dev_target.push("no GNL (target --gnl)".into());
        }
        if !conc_hi {
            dev_target.push("concomitant fit quad (target --conc-fit deg4)".into());
        }
        if keep_n > 1 {
            dev_target.push(format!("--keep {keep_n}: 1/{keep_n} of the readouts"));
        }
        if adc_part != hasty_recon::pyramid::AdcPart::Out {
            dev_target.push(format!("--adc-part {} (target out: the outgoing half of every readout)", adc_part.as_str()));
        }
        if grid_scale.is_some() || kcrop.is_some() {
            dev_target.push(format!("pyramid level (grid {:?}, kcrop {})", sd.dims, kcrop.map_or("no".into(), |k| k.describe())));
        }
        let config_line = if dev_target.is_empty() { "config: target".to_string() } else { format!("config: differs from target: {}", dev_target.join("; ")) };
        println!("{config_line}");
        // --eval LABEL[@AMP]:idx.npy,...: (label, amplitude, native-grid voxels)
        let evals: Vec<(String, f64, Vec<usize>)> = eval_spec.as_deref().map_or(vec![], |s| {
            s.split(',').map(|p| {
                let (lab, path) = p.split_once(':').expect("--eval LABEL[@AMP]:idx.npy");
                let (lab, amp) = match lab.split_once('@') { Some((l, a)) => (l.to_string(), a.parse::<f64>().expect("--eval LABEL@AMP")), None => (lab.to_string(), lab.parse::<f64>().expect("--eval LABEL (a breathing amplitude)")) };
                let idx: Vec<usize> = hasty_recon::data::read_npy(std::path::Path::new(path)).unwrap_or_else(|e| panic!("--eval {path}: {e:?}")).f32s().iter().map(|&v| v as usize).collect();
                (lab, amp, idx)
            }).collect()
        });
        assert!(evals.is_empty() || sd.native_grid(), "--eval: native-grid voxel indices (no --grid-scale)");
        for (lab, amp, idx) in &evals {
            println!("  eval {lab}: breathing amplitude {amp}, {} voxels, respiratory weights {:?}", idx.len(), kb.resp_weights(*amp).iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        }
        // initial state: the time-averaged warm start (z from the reference encode, v by the decoder inside the support,
        // the robust static-tissue residual phase) in every knot, or a previous knot run (--init-from) resampled
        let warm64: Vec<Vec<Complex64>> = warm.iter().map(|w| w.iter().map(|c| Complex64::new(c.re as f64, c.im as f64)).collect()).collect();
        let sfit = vpoly.map(|d| {
            let f = hasty_recon::vinit::fit_static_phase(&warm64, &support, sd.dims, d, 0.05);
            println!("residual-phase init (robust static fit, degree {d}): σ per encode {:?} rad, static mask {} voxels", f.sigma.iter().map(|s| (s * 1000.0).round() / 1000.0).collect::<Vec<_>>(), f.mask.iter().filter(|&&m| m > 0.0).count());
            f
        });
        let mut h0 = HostStack::zeros(&lens);
        let from_knots = init_from.as_ref().and_then(|p| knot_init(p, &spec, kb));
        let have_init = from_knots.is_some();
        match from_knots {
            Some((src, chunks)) => {
                println!("knot init: {} resampled {:?} → {:?}", src.display(), chunks.0, sd.dims);
                for t in 0..lens.len() {
                    h0.chunk_mut(t).copy_from_slice(&chunks.1[t]);
                }
            }
            None => {
                let warm_c = match &sfit { Some(f) => hasty_recon::vinit::remove_phase(&warm64, &f.a, sd.dims, vpoly.unwrap()), None => warm64.clone() };
                let mut z0 = vec![C32::new(0.0, 0.0); nv];
                let mut v0 = vec![0f32; 3 * nv];
                for q in 0..nv {
                    let z: Vec<Complex64> = (0..ne).map(|e| warm_c[e][q]).collect();
                    let v = if support[q] { dec.velocity(&z) } else { [0.0; 3] };
                    let zq = z[0] * Complex64::from_polar(1.0, kenc[0][0] * v[0] + kenc[0][1] * v[1] + kenc[0][2] * v[2]);
                    z0[q] = C32::new(zq.re as f32, zq.im as f32);
                    for a in 0..3 {
                        v0[a * nv + q] = v[a] as f32;
                    }
                }
                let vp = pack_f32(&v0, lens[spec.chunk_v(0)]);
                if spline {
                    // (m, ψ') = (|z0|, arg z0 − k̄·v): the zero-moment phase decoupled from v (VelSpec::splines)
                    let kb_ = spec.kbar();
                    for q in 0..nv {
                        let z = z0[q];
                        let ps = (z.im as f64).atan2(z.re as f64) - (0..3).map(|a| kb_[a] * v0[a * nv + q] as f64).sum::<f64>();
                        z0[q] = C32::new((z.re * z.re + z.im * z.im).sqrt(), ps as f32);
                    }
                }
                for k in 0..nk {
                    h0.chunk_mut(spec.chunk_z(k)).copy_from_slice(&z0);
                    h0.chunk_mut(spec.chunk_v(k)).copy_from_slice(&vp);
                }
                if let Some(f) = &sfit {
                    let a: Vec<f32> = f.a.iter().map(|&x| x as f32).collect();
                    let ca = spec.chunk_a();
                    let l = lens[ca];
                    h0.chunk_mut(ca).copy_from_slice(&pack_f32(&a, l));
                }
            }
        }
        drop(warm64);
        let coil: Vec<f32> = (0..nv).map(|q| (0..nvc).map(|c| { let s = smaps[c * nv + q]; s.re * s.re + s.im * s.im }).sum()).collect();
        let mut pr = VelProblem::new(&st, kern.clone(), sop, ys, vm, Some(coil)).unwrap();
        if staged {
            let sg = hasty_recon::kstage::Stager::new(&dev.ctx, kstore.clone().unwrap()).unwrap();
            println!("data staged from the store: 2 pinned staging buffers of {:.2} GiB", sg.pinned_bytes() as f64 / 2.0 / GIB);
            pr.data_from_store(sg);
        }
        if early_host {
            pr.ys = std::mem::take(&mut ys_pin).into_iter().map(hasty_recon::vobj::YData::Host).collect();
        }
        pr.set_knot_weights(&gdiag);
        pr.vtrust = vtrust;
        // spline mode, robust start (--vinit cg:N, research/velocity_model_sr.md §F): complex parameter-basis knot images
        // per encode (CGLS through the same operator, from the warm image), their values at the knot centres decoded per
        // knot by the dual-VENC decoder (the robust residual phase removed first), then m, ψ, v as cardiac spline
        // coefficients (the values' circulant inverse)
        let mut init_desc = if have_init { "--init-from".to_string() } else { "warm (time-averaged)".to_string() };
        if let (true, false, Some((cg_it, cg_lam))) = (spline, have_init, vinit) {
            let need = (0..ne).map(|e| pr.complex_knots_bytes(e)).max().unwrap_or(0);
            let (free, _) = dev.ctx.mem_get_info().unwrap();
            if need + (1usize << 30) > free {
                eprintln!("--vinit cg: the complex knot CGLS needs {:.2} GiB on the device, {:.2} GiB free: start from a coarser level (--init-from <run dir>) or --vinit warm", need as f64 / GIB, free as f64 / GIB);
                std::process::exit(2);
            }
            let ti = Instant::now();
            let (kre_, mcc) = (spec.kbar(), mc);
            let mut cv: Vec<Vec<C32>> = Vec::with_capacity(ne);
            for e in 0..ne {
                let (c, hist) = pr.complex_knots(&st, e, &warm[e], cg_it, cg_lam).unwrap();
                println!("  init encode {e}: {cg_it} CGLS iterations on {nk} complex knots, ‖s‖/‖s₀‖ → {:.3e} ({:.0}s)", hist.last().copied().unwrap_or(1.0), ti.elapsed().as_secs_f64());
                cv.push(c);
            }
            // values at the knot centres (complex, per encode), the robust residual phase removed
            let pexi = vpoly.map(hasty_recon::vmodel::poly_exps).unwrap_or_default();
            let a_init: Vec<f64> = sfit.as_ref().map_or(vec![], |f| f.a.clone());
            let npi = pexi.len();
            let cdeg = kb.cdeg;
            // per voxel (hasty_recon::vinit::SplineStart, tested on model-consistent data in vinit::tests): knot-centre
            // values, the robust residual phase removed, SNR-gated dual-VENC decode, zero-moment phase, spline fit
            let sstart = hasty_recon::vinit::SplineStart { dec: &dec, kenc: &kenc, kbar: kre_, kr, mc: mcc, cdeg, snr_gate: vinit_snr, vbox };
            let per_voxel: Vec<(Vec<(f32, f32)>, Vec<[f32; 3]>, [usize; 3])> = (0..nv).into_par_iter().map(|q| {
                let phi: Vec<f64> = (0..ne).map(|e| if e == 0 || npi == 0 { 0.0 } else { (0..npi).map(|p| a_init[(e - 1) * npi + p] * hasty_recon::vmodel::poly_at(&pexi[p], q, sd.dims)).sum() }).collect();
                sstart.voxel(&|e, k| { let x = cv[e][k * nv + q]; Complex64::new(x.re as f64, x.im as f64) }, &phi, support[q])
            }).collect();
            drop(cv);
            if let Some(smin) = vinit_snr {
                let c = per_voxel.iter().fold([0usize; 3], |a, p| [a[0] + p.2[0], a[1] + p.2[1], a[2] + p.2[2]]);
                let t = (c[0] + c[1] + c[2]).max(1) as f64;
                println!("  start SNR gate {smin:.2}: support voxel-knots decoded per knot {:.1} %, from the voxel's time mean {:.1} %, v = 0 {:.1} %", 100.0 * c[0] as f64 / t, 100.0 * c[1] as f64 / t, 100.0 * c[2] as f64 / t);
            }
            let lv = lens[spec.chunk_v(0)];
            for k in 0..nk {
                let zc: Vec<C32> = per_voxel.iter().map(|p| C32::new(p.0[k].0, p.0[k].1)).collect();
                h0.chunk_mut(spec.chunk_z(k)).copy_from_slice(&zc);
                let mut vf = vec![0f32; 3 * nv];
                for (q, p) in per_voxel.iter().enumerate() {
                    for a in 0..3 {
                        vf[a * nv + q] = p.1[k][a];
                    }
                }
                h0.chunk_mut(spec.chunk_v(k)).copy_from_slice(&pack_f32(&vf, lv));
            }
            init_desc = format!("complex knot CGLS {cg_it} (λ {cg_lam}) + dual-VENC decode per knot + spline fit");
            println!("spline init: {init_desc} ({:.0}s)", ti.elapsed().as_secs_f64());
        }
        // stochastic coil sampling (p2np::CoilSampling): the iteration's coil groups through VelProblem::begin_iteration
        if let Some(cs) = &coil_sampling {
            let ncs = pr.op.n_coils();
            assert!(ncs > cs.keep, "--coil-sampling keep {} of {ncs} coils", cs.keep);
            println!("coil sampling: keep {} + draw {} of the {} tail coils (hold {}, last {} full)", cs.keep, cs.draw, ncs - cs.keep, cs.hold, cs.full_last);
            pr.sampling = Some(cs.clone());
        }
        if let Some((mu, mode)) = &vanchor {
            let w: Vec<f32> = match (mode.as_str(), &sfit) {
                ("truth", _) => (0..nv).map(|q| (dil[q] && sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) < 1e-3) as u8 as f32).collect(),
                (_, Some(f)) => f.mask.clone(),
                _ => panic!("--vanchor auto needs --vpoly (the static mask comes from the residual-phase fit)"),
            };
            println!("static-tissue anchor μ {mu} ({mode}{}) on {} voxels", if mode == "truth" { ", ORACLE mask" } else { "" }, w.iter().filter(|&&x| x > 0.0).count());
            pr.set_anchor(&st, &w, *mu).unwrap();
        }
        // memory ledger: placement of state, data and tables before anything large is allocated (research/memory_ledger.md)
        let pb = hasty_recon::plan::probe(&dev.ctx, &st).unwrap();
        let ns = pr.op.nsets();
        let data_sets: Vec<usize> = pr.ys.iter().map(|y| y.len() * 8).collect();
        let data_dev_now: usize = pr.ys.iter().map(|y| if let hasty_recon::vobj::YData::Dev(d) = y { d.len() * 8 } else { 0 }).sum();
        let (tab_all, tab_max) = pr.op.table_bytes();
        let tab_dev_now = if pr.op.tables_on_host() { 2 * tab_max } else { tab_all };
        let quasi = preconds.iter().any(|p| Precond::parse(p).is_some_and(|q| q.quasi()));
        let lin = hasty_recon::vknot::LedgerIn {
            nv,
            nk,
            img_frames: op_frames.as_ref().map_or(0, |o| o.set_frames()),
            slots: if spline { pr.row_slot_bytes() } else { 0 },
            stack: stack_b,
            chunk_max: lens.iter().max().unwrap() * 8,
            nsets: ns,
            data_all: data_sets.iter().sum(),
            data_max: data_sets.iter().copied().max().unwrap_or(0),
            data_movable: !(staged || early_host),
            tab_all,
            tab_max,
            fixed: (pb.total - pb.free).saturating_sub(data_dev_now + tab_dev_now),
            dev_budget: vram_budget.unwrap_or(pb.total.saturating_sub(512 << 20)).min(pb.total),
            host_avail: available_memory(),
            quasi,
            prior_dev: 0,
            h2d: pb.h2d,
            pageable: 0.4 * pb.h2d,
        };
        let force_state = match state_arg.as_str() { "host" => Some(true), "device" => Some(false), _ => None };
        let force_data = if staged || early_host { Some(true) } else { match plan_mode.as_str() { "host" => Some(true), "device" => Some(false), _ => None } };
        let force_tab = if tab_all == 0 { None } else { match plan_mode.as_str() { "host" => Some(true), "device" => Some(false), _ => None } };
        // the velocity objective's gradient sink (at-knots mode): a BFP host state keeps g, g_prev at their width
        let has_sink = { use hasty_recon::p2np::Objective; pr.has_gradient_sink() };
        let lp = hasty_recon::vknot::plan_sink(&lin, has_sink, force_state, force_data, force_tab).unwrap_or_else(|e| {
            eprintln!("{e}");
            eprintln!("refused before the solver (no OOM): fewer knots / coils, --keep N, a smaller grid (--grid-scale) or a box, or a larger card");
            std::process::exit(2)
        });
        print!("{}", lp.describe());
        if tab_all > 0 && lp.tables_on_host != pr.op.tables_on_host() {
            let freed = pr.op.place_tables(&dev.ctx, &st, lp.tables_on_host).unwrap();
            println!("sample tables moved to {} ({:+.2} GiB device)", if lp.tables_on_host { "pinned host memory (streamed)" } else { "the device" }, -(freed as f64) / GIB);
        }
        if lp.data_on_host && data_dev_now > 0 {
            let freed = pr.data_to_host(&dev.ctx, &st).unwrap();
            println!("k-space data moved to pinned host memory ({:.2} GiB freed)", freed as f64 / GIB);
        }
        let state_host = lp.state_on_host;
        let sopts = hasty_recon::p2np::StateOpts { ctx: Some(dev.ctx.clone()), prev_on_host: lp.prev_on_host, w_mode: if lp.prev_on_host { hasty_recon::p2np::WMode::Host } else { hasty_recon::p2np::WMode::Device } };
        // VENC continuation: the low-VENC encodes (all but the least sensitive group) are left out of the first stage
        let kn: Vec<f64> = (0..ne).map(|e| (0..3).map(|a| (kenc[e][a] - kenc[0][a]).powi(2)).sum::<f64>().sqrt()).collect();
        let kmin = kn.iter().copied().filter(|&x| x > 0.0).fold(f64::INFINITY, f64::min);
        let low: Vec<bool> = kn.iter().map(|&x| x > kmin * (1.0 + 1e-6)).collect();
        let vtag = |pc: &str, kap: f64| format!("{model}_{pc}_vk{kr}x{mc}_r{}c{}{}{}{}{}{dfw_tag}{lock_tag}{}", kb.rdeg, kb.cdeg, vpoly.map_or(String::new(), |d| format!("_p{d}")), if vcont > 0 { format!("_c{vcont}") } else { String::new() }, if vzwarm > 0 { format!("_z{vzwarm}") } else { String::new() }, if kap > 0.0 { format!("_llr{kap}") } else { String::new() }, if static_lock.is_some() && lock_coupling == hasty_recon::knotlock::Coupling::Nested { "n" } else { "" });
        let runs: Vec<String> = preconds.iter().flat_map(|p| vkappas.iter().map(move |&k| (p, k))).map(|(p, k)| vtag(p, k)).collect();
        // per-axis static lock on (z, v) (hasty_recon::knotlock): basis from the temporal Gram of the samples
        let lock_basis = static_lock.as_ref().map(|_| hasty_recon::knotlock::KnotBasis::from_gram(&gram, kr, mc, true, hasty_recon::knotlock::NoiseModel::Auto));
        let lock_opts = static_lock.as_ref().map(|(o, _, _)| hasty_recon::knotlock::KnotLockOpts { base: o.clone(), coupling: lock_coupling });
        if let Some(b) = &lock_basis {
            println!("knot lock (--static-lock {:?}, {lock_coupling:?}): from IT0 {} every {}; separable-metric mismatch {:.3}, G-norm bound {:.3}", lock_opts.as_ref().unwrap().base.mu, static_lock.as_ref().unwrap().1, static_lock.as_ref().unwrap().2, b.nonsep, b.metric_bound);
        }
        let mut info = serde_json::json!({
            "cmdline": cmdline, "sim_dir": dir, "model": model, "phys": phys, "gnl": gnl, "conc_fit": conc_fit, "iters": iters, "warm_iters": warm_iters, "vcoils": nvc, "dims": sd.dims, "vox": sd.vox, "native_dims": sd.tdims,
            "n_frames": nf, "n_enc": ne, "venc": sd.venc, "venc_range": dec.range(), "encoding": sd.meta["encoding"], "runs": runs, "keep": keep_n, "config": config_line,
            "prior": if vkappas.iter().any(|&k| k > 0.0) { "llr(z, p)" } else { "none" }, "vkappas": vkappas,
            "vknots": { "kr": kr, "rdeg": kb.rdeg, "mc": mc, "cdeg": kb.cdeg, "resp": resp_src, "rknots": kb.rknots, "amp_median": amp_med, "gram_condition": gcond, "gram_diag_rel": gdiag.iter().map(|v| v / gmean).collect::<Vec<_>>(), "poly": vpoly, "vcont": vcont, "low_venc_encodes": low, "anchor": vanchor.as_ref().map(|(m, s)| format!("{m},{s}")) },
            "memory": { "state_on_host": lp.state_on_host, "prev_on_host": lp.prev_on_host, "data_on_host": lp.data_on_host, "tables_on_host": lp.tables_on_host, "device_peak": lp.device_peak, "host_peak": lp.host_peak, "stack_bytes": stack_b },
            "coils": coil_info, "kstore": kstore_info, "setup_s": t0.elapsed().as_secs_f64(),
        });
        info["pyramid"] = pyr_info.clone();
        info["adc_part"] = serde_json::json!(adc_part.as_str());
        info["segments"] = serde_json::json!({ "basis": seg_basis_arg, "weight": seg_weight, "nseg": nseg });
        info["vspline"] = match &op_frames {
            Some(of) => serde_json::json!({ "param": "polar (m, ψ, v)", "fop": of.fop, "odeg": of.odeg, "kre": of.kre, "groups": of.r0, "images_per_encode": of.images_per_encode(), "kbar": spec.kbar(), "init": init_desc, "vmwav": vmwav.map(|(t, l)| format!("{t}:{l}")) }),
            None => serde_json::json!({ "param": "complex z at the knots (--vop knots)" }),
        };
        info["prior_detail"] = serde_json::json!({ "dfw": dfw_rule.map(|r| serde_json::json!({ "rule": format!("{r:?}"), "levels": dfw_levels, "mix": dfw_mix })), "static_lock": static_lock.as_ref().map(|(o, it0, ev)| serde_json::json!({ "opts": format!("{o:?}"), "it0": it0, "every": ev, "coupling": format!("{lock_coupling:?}") })) });
        std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
        let rt = transpose(&sd.rot);
        // velocity (logical → physical, J⁻¹ with GNL) of knot values at voxel q
        let vphys = |v: [f64; 3], q: usize| -> [f64; 3] { if gnl { mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &mat_vec(&sd.rot, &v))) } else { v } };
        // the truth of the in-driver score: the 20 cardiac frames (scan-mean breathing state) at the lumen voxels
        let truth_l: Vec<[f64; 3]> = (0..nf_card).flat_map(|f| lumen.iter().map(move |&q| (f, q))).map(|(f, q)| [0, 1, 2].map(|a| sd.sample_log(&sd.truth_vel, a + 3 * f, &grid.x_log[q]))).collect();
        let nl = lumen.len();
        let np = spec.np();
        let pex = vpoly.map(hasty_recon::vmodel::poly_exps).unwrap_or_default();
        // the initial state: kept for the next run only when there is one (host memory: one parameter stack)
        let multi = runs.len() > 1;
        let mut h0 = Some(h0);
        for (pc, kap) in preconds.iter().flat_map(|p| vkappas.iter().map(move |&k| (p, k))) {
            let tag = vtag(pc, kap);
            let precond = Precond::parse(pc).unwrap_or(Precond::Zmshr1M);
            let with_metric = precond.metric();
            let resuming = std::fs::read_dir(&out).unwrap().flatten().any(|e| { let n = e.file_name().to_string_lossy().into_owned(); n.starts_with(&format!("state_{tag}_s")) || n.starts_with(&format!("vdone_{tag}_s")) });
            if !resuming && hasty_value::store::exists_any(&out.join(format!("vel_{tag}.npy"))) {
                println!("  {tag}: finished in an earlier run, skipped (--fresh to redo)");
                continue;
            }
            // the state of this run (device stack or host stack)
            let init_x = if multi { h0.as_ref().unwrap().clone() } else { h0.take().expect("initial state") };
            let (mut xd, mut xh) = if state_host {
                (Stack { v: vec![] }, init_x)
            } else {
                let mut d = Stack::zeros_lens(&st, &lens).unwrap();
                for t in 0..lens.len() {
                    st.memcpy_htod(init_x.chunk(t), &mut d.v[t]).unwrap();
                }
                (d, HostStack::default())
            };
            pr.reset();
            let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(out.join(format!("metrics_{tag}.csv"))).unwrap();
            if !resuming {
                use std::io::Write;
                csv.set_len(0).unwrap();
                writeln!(csv, "iter,f,rmse,rel,wrap_frac,seconds,{},rmse_dec,rel_dec", (0..nf_card).map(|f| format!("frame{f}")).collect::<Vec<_>>().join(",")).unwrap();
            }
            let tr = Instant::now();
            let mut stages: Vec<(usize, bool)> = if vcont > 0 && low.iter().any(|&b| b) { vec![(vcont.min(iters), true), (iters.saturating_sub(vcont), false)] } else { vec![(iters, false)] };
            let zstage = vzwarm > 0;
            if zstage {
                stages.insert(0, (vzwarm, false));
            }
            let mut k_off = 0;
            // the top Ritz vector of the last stage's λ_max (Lanczos warm start of the next stage)
            let mut lam_ritz: Option<Vec<C32>> = None;
            let anchor_auto = matches!(&vanchor, Some((_, m)) if m == "auto") && stages.len() > 1;
            let fit_mask: Option<Vec<f32>> = sfit.as_ref().map(|f| f.mask.clone());
            // the knot lock's map (saved exactly with every refresh: a resumed run continues with it)
            let (lmap_f, lmap_j) = (out.join(format!("knotlock_map_{tag}.npy")), out.join(format!("knotlock_map_{tag}.json")));
            let mut lock_map: Option<hasty_recon::knotlock::KnotMap> = None;
            if resuming && static_lock.is_some() && hasty_value::store::exists_any(&lmap_f) && lmap_j.exists() {
                let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&lmap_j).unwrap()).unwrap();
                let v = hasty_recon::data::read_npy(&lmap_f).unwrap().f32s();
                let nm = [0, 1, 2].map(|i| j["noise_mix"][i].as_f64().unwrap_or(f64::INFINITY));
                lock_map = Some(hasty_recon::knotlock::KnotMap::from_stacked(&v, nv, j["iteration"].as_u64().unwrap() as usize, nm));
                println!("  knot lock map of iteration {} reloaded", j["iteration"]);
            }
            for (si, &(n_it, only_hi)) in stages.iter().enumerate() {
                let stag = format!("{tag}_s{si}");
                pr.freeze_v = zstage && si == 0;
                let si = if zstage { si.saturating_sub(1) } else { si };
                // host copies of the chunks of the current state
                let chunk_of = |xd: &Stack, xh: &HostStack, t: usize| -> Vec<C32> { if state_host { xh.chunk(t).to_vec() } else { st.clone_dtoh(&xd.v[t]).unwrap() } };
                if anchor_auto && si == 0 {
                    pr.clear_anchor();
                }
                if anchor_auto && si == 1 {
                    // static = accepted by the robust fit, low variation of v over the knots, not near a pulsatile voxel
                    let vh: Vec<Vec<f32>> = (0..nk).map(|k| unpack_f32(&chunk_of(&xd, &xh, spec.chunk_v(k)), 3 * nv)).collect();
                    let sd_v: Vec<f64> = (0..nv).map(|q| {
                        let mean: [f64; 3] = [0, 1, 2].map(|a| (0..nk).map(|k| vh[k][a * nv + q] as f64).sum::<f64>() / nk as f64);
                        ((0..nk).map(|k| (0..3).map(|a| (vh[k][a * nv + q] as f64 - mean[a]).powi(2)).sum::<f64>()).sum::<f64>() / nk as f64).sqrt()
                    }).collect();
                    let fm = fit_mask.as_ref().unwrap();
                    let mut sv: Vec<f64> = (0..nv).filter(|&q| fm[q] > 0.0).map(|q| sd_v[q]).collect();
                    sv.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let tau = (vanchor_refine.0 * sv.get(sv.len() / 2).copied().unwrap_or(0.0)).max(0.02);
                    let mut puls: Vec<bool> = (0..nv).map(|q| support[q] && sd_v[q] > tau).collect();
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
                    println!("  anchor refined by pulsatility (std of v over the knots > {tau:.3} m/s, dilated {}): {} voxels", vanchor_refine.1, w.iter().filter(|&&x| x > 0.0).count());
                    pr.set_anchor(&st, &w, vanchor.as_ref().unwrap().0).unwrap();
                }
                let done = out.join(format!("vdone_{stag}"));
                for t in 0..pr.weights.len() {
                    pr.weights[t] = if only_hi && low[if spline { t / op_frames.as_ref().unwrap().ng() } else { t % ne }] { 0.0 } else { 1.0 };
                }
                if done.exists() {
                    if let Some(v) = ctl.load(&format!("{stag}_final")).unwrap() {
                        if state_host { xh.load_value(&v).unwrap() } else { xd = hasty_recon::checkpoint::value_stack(&st, &v, &lens).unwrap() }
                    }
                    k_off += n_it;
                    continue;
                }
                let tl = Instant::now();
                if lam_compare {
                    // diagnostic: the stage's λ_max by every estimator (cold starts), relative to power 100
                    let r = if state_host { pr.lambda_max_host(&st, &xh, with_metric, 100).unwrap() } else { pr.lambda_max(&st, &xd, with_metric, 100).unwrap() };
                    let mut line = format!("  λ compare ({}): power 100 = {r:.5e}", if with_metric { "M H" } else { "H" });
                    let e = if state_host { pr.lambda_max_host(&st, &xh, with_metric, lam_iters).unwrap() } else { pr.lambda_max(&st, &xd, with_metric, lam_iters).unwrap() };
                    line += &format!("; power {lam_iters} {:+.2e}", e / r - 1.0);
                    for k in [4usize, 6, 8] {
                        let e = if state_host { pr.lambda_lanczos_host(&st, &xh, with_metric, k, None).unwrap() } else { pr.lambda_lanczos(&st, &xd, with_metric, k, None).unwrap() };
                        line += &format!("; lanczos {k}: θ {:+.2e}, θ+b {:+.2e}", e.theta / r - 1.0, e.hat() / r - 1.0);
                    }
                    if let (Some(wv), LamMethod::Lanczos(k)) = (lam_ritz.clone(), lam_method) {
                        let mut wv = Some(wv);
                        let e = if state_host { pr.lambda_lanczos_host(&st, &xh, with_metric, k.min(4), Some(&mut wv)).unwrap() } else { pr.lambda_lanczos(&st, &xd, with_metric, k.min(4), Some(&mut wv)).unwrap() };
                        line += &format!("; warm lanczos {}: θ {:+.2e}, θ+b {:+.2e}", k.min(4), e.theta / r - 1.0, e.hat() / r - 1.0);
                    }
                    println!("{line} ({:.0}s)", tl.elapsed().as_secs_f64());
                }
                let tl = Instant::now();
                let (mut lam, mut applies) = match lam_method {
                    LamMethod::Power => (if state_host { pr.lambda_max_host(&st, &xh, with_metric, lam_iters).unwrap() } else { pr.lambda_max(&st, &xd, with_metric, lam_iters).unwrap() }, lam_iters),
                    LamMethod::Lanczos(k) => {
                        let warm = lam_warm_on(lens.iter().sum::<usize>() * 8, k);
                        let e = if state_host { pr.lambda_lanczos_host(&st, &xh, with_metric, k, warm.then_some(&mut lam_ritz)).unwrap() } else { pr.lambda_lanczos(&st, &xd, with_metric, k, warm.then_some(&mut lam_ritz)).unwrap() };
                        println!("  λ_max by Lanczos ({k} steps{}): θ {:.4e} + residual bound {:.2e}", if warm { ", Ritz vector kept for the next stage" } else { "" }, e.theta, e.bound);
                        (e.hat(), e.applies)
                    }
                };
                if let Some(cs) = coil_sampling.clone() {
                    // the step must be stable for every coil subset: α from the largest λ_max over a few draws
                    let ncs = pr.op.n_coils();
                    let mut ls = 0.0f64;
                    for d in 0..3 {
                        pr.cur_groups = cs.groups(ncs, d * cs.hold.max(1), usize::MAX);
                        let l = match lam_method {
                            LamMethod::Power => if state_host { pr.lambda_max_host(&st, &xh, with_metric, 8).unwrap() } else { pr.lambda_max(&st, &xd, with_metric, 8).unwrap() },
                            // each draw starts from the full operator's Ritz vector (when kept; the draws do not replace it)
                            LamMethod::Lanczos(k) => {
                                let mut wv = lam_ritz.clone();
                                let wr = wv.is_some().then_some(&mut wv);
                                (if state_host { pr.lambda_lanczos_host(&st, &xh, with_metric, k, wr).unwrap() } else { pr.lambda_lanczos(&st, &xd, with_metric, k, wr).unwrap() }).hat()
                            }
                        };
                        applies += match lam_method { LamMethod::Power => 8, LamMethod::Lanczos(k) => k };
                        ls = ls.max(l);
                    }
                    pr.cur_groups = None;
                    println!("  coil sampling: λ_max(sampled) / λ_max(full) = {:.3} (3 draws)", ls / lam);
                    lam = lam.max(ls);
                }
                st.synchronize().unwrap();
                println!("  {tag} stage {si} ({}): {n_it} iterations, λ_max({}) = {lam:.3e} ({:.1}s, {applies} Hessian-vector products, {:.1}s each)", if only_hi { "reference + high-VENC encodes" } else { "all encodes" }, if with_metric { "M H" } else { "H" }, tl.elapsed().as_secs_f64(), tl.elapsed().as_secs_f64() / applies as f64);
                if vanchor.is_some() {
                    if anchor_auto && si == 0 {
                        // --vanchor …,auto: no anchor in the first stage (its mask is refined by pulsatility after it)
                        println!("    anchor off in this stage (auto: the static mask is refined by pulsatility after it; n̂ {:.3e})", pr.nhat);
                    } else {
                        println!("    anchor λ {:.3e} (n̂ {:.3e})", pr.anchor_lam, pr.nhat);
                    }
                }
                // report: velocity of the knot interpolant at the 20 cardiac frames (median breathing amplitude) in the
                // lumen against the frame truth; lumen evaluations (--eval) every --eval-every iterations
                let mut report_core = |k: usize, chunk: &dyn Fn(usize) -> Vec<C32>, f: f64| {
                    let k = if k == usize::MAX { k_off } else { k + 1 + k_off };
                    let last = k == iters;
                    if k % every != 0 && !last && k != 0 {
                        return;
                    }
                    let vl: Vec<Vec<f32>> = (0..nk).map(|kk| { let v = unpack_f32(&chunk(spec.chunk_v(kk)), 3 * nv); lumen.iter().flat_map(|&q| [v[q], v[nv + q], v[2 * nv + q]]).collect() }).collect();
                    let (mut e2, mut t2, mut n, mut wrap) = (0.0f64, 0.0f64, 0usize, 0usize);
                    // split of the score (research/velocity_model_sr.md §F.2): the upper half of the lumen by the model
                    // magnitude (inside the support), the components that are not wrapped, the voxels outside the support
                    let mbar: Vec<f64> = {
                        let mut s = vec![0.0f64; nl];
                        for kk in 0..nk {
                            let c = chunk(spec.chunk_z(kk));
                            for (l, &q) in lumen.iter().enumerate() {
                                s[l] += if spline { c[q].re.abs() as f64 } else { ((c[q].re * c[q].re + c[q].im * c[q].im) as f64).sqrt() };
                            }
                        }
                        s
                    };
                    let mmed = { let mut v = mbar.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v.get(nl / 2).copied().unwrap_or(0.0) };
                    let (mut sp_et, mut sp_tt, mut sp_vt, mut sp_t2a) = (0.0f64, 0.0f64, [0.0f64; 3], [0.0f64; 3]);
                    let (mut sp_en, mut sp_tn, mut sp_eo, mut sp_to) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                    let mut per_frame = Vec::with_capacity(nf_card);
                    for fr in 0..nf_card {
                        let (mut fe2, mut fn_) = (0.0, 0usize);
                        for (l, &q) in lumen.iter().enumerate() {
                            let mut v = [0.0f64; 3];
                            for kk in 0..nk {
                                let w = wfr[fr][kk];
                                if w != 0.0 {
                                    for a in 0..3 {
                                        v[a] += w * vl[kk][3 * l + a] as f64;
                                    }
                                }
                            }
                            let v = vphys(v, q);
                            for a in 0..3 {
                                let t = truth_l[fr * nl + l][a];
                                let dd = v[a] - t;
                                e2 += dd * dd;
                                t2 += t * t;
                                fe2 += dd * dd;
                                let wr = dd.abs() > 0.8 * sd.venc.min(dec.range()[a]);
                                wrap += wr as usize;
                                if !wr {
                                    sp_en += dd * dd;
                                    sp_tn += t * t;
                                }
                                if !support[q] {
                                    sp_eo += dd * dd;
                                    sp_to += t * t;
                                } else if mbar[l] >= mmed {
                                    sp_et += dd * dd;
                                    sp_tt += t * t;
                                    sp_vt[a] += v[a] * t;
                                    sp_t2a[a] += t * t;
                                }
                            }
                            n += 1;
                            fn_ += 1;
                        }
                        per_frame.push((fe2 / fn_.max(1) as f64).sqrt());
                    }
                    let wf = wrap as f64 / (3 * n).max(1) as f64;
                    // the same frames decoded from the interpolated complex model (what the operator represents between
                    // knots; the knot velocities' interpolant above can differ, research/velocity_model_sr.md §E)
                    let (ed2, td2) = {
                        let a = unpack_f32(&chunk(spec.chunk_a()), spec.na());
                        let xs = if spline { spline_lumen(chunk, &spec, kb, &a, &pex, &lumen, amp_med, nf_card) } else { knot_lumen(chunk, &spec, &a, &pex, &lumen, &wfr) };
                        let (mut e, mut t) = (0.0f64, 0.0f64);
                        for (l, &q) in lumen.iter().enumerate() {
                            for fr in 0..nf_card {
                                let z: Vec<Complex64> = (0..ne).map(|e| { let c = xs[(l * ne + e) * nf_card + fr]; Complex64::new(c.re as f64, c.im as f64) }).collect();
                                let v = vphys(dec.velocity(&z), q);
                                for a in 0..3 {
                                    let tt = truth_l[fr * nl + l][a];
                                    e += (v[a] - tt).powi(2);
                                    t += tt * tt;
                                }
                            }
                        }
                        (e, t)
                    };
                    {
                        use std::io::Write;
                        writeln!(csv, "{k},{f:.6e},{:.5},{:.5},{wf:.5},{:.1},{},{:.5},{:.5}", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), tr.elapsed().as_secs_f64(), per_frame.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(","), (ed2 / n as f64).sqrt(), (ed2 / td2.max(1e-30)).sqrt()).unwrap();
                    }
                    // f is the data term + anchor at the iterate before this one (the gradient's value); the stage's
                    // initial state has none (p2np reports NaN there: not evaluated, research/velocity_model_sr.md §F.2)
                    let fs = if f.is_nan() { "— (initial state, not evaluated)".to_string() } else { format!("{f:.4e}") };
                    println!("  {pc:<7} vk it {k:>3}: f(previous iterate) = {fs}, velocity RMSE {:.3} m/s (rel {:.3}; decoded from the interpolated model {:.3} / rel {:.3}), wrapped {:.2} %, frames {}, {:.0}s", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), (ed2 / n as f64).sqrt(), (ed2 / td2.max(1e-30)).sqrt(), 100.0 * wf, per_frame.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "), tr.elapsed().as_secs_f64());
                    println!("    score split: upper-half magnitude (support) rel {:.3}, slope {:?}; unwrapped components rel {:.3}; outside the support rel {:.3}", (sp_et / sp_tt.max(1e-30)).sqrt(), [0, 1, 2].map(|a| (sp_vt[a] / sp_t2a[a].max(1e-30) * 1000.0).round() / 1000.0), (sp_en / sp_tn.max(1e-30)).sqrt(), (sp_eo / sp_to.max(1e-30)).sqrt());
                    if spline && (k == k_off || last) {
                        // the operator's view of the current model (frames + cubic interpolation) against the model itself
                        let sub: Vec<usize> = lumen.iter().copied().step_by((lumen.len() / 2000).max(1)).collect();
                        let a = unpack_f32(&chunk(spec.chunk_a()), spec.na());
                        let (rel, p999) = op_frame_error(chunk, &spec, &a, &pex, &sub);
                        println!("    operator-frame approximation error of the model ({} lumen voxels, all rows and encodes): relative {rel:.2e}, p99.9 per sample {p999:.2e}", sub.len());
                    }
                    if !evals.is_empty() && (k % eval_every == 0 || last) {
                        let a = unpack_f32(&chunk(spec.chunk_a()), spec.na());
                        for (lab, amp, idx) in &evals {
                            let w: Vec<Vec<f64>> = (0..nf_card).map(|f| kb.weights(*amp, (f as f64 + 0.5) / nf_card as f64)).collect();
                            let vals = if spline { spline_lumen(chunk, &spec, kb, &a, &pex, idx, *amp, nf_card) } else { knot_lumen(chunk, &spec, &a, &pex, idx, &w) };
                            let p = out.join(format!("knots_{tag}_r{lab}_it{k}_lumen.npy"));
                            hasty_value::store::save_bytes(&p, hasty_value::DType::C64, &[idx.len(), ne, nf_card], hasty_value::Order::C, &c8(&vals), &hasty_value::Save::image()).unwrap();
                        }
                    }
                };
                // prior: LLR on the knot z images and on p = |z̄|·v (3 K real images), then v = p / |z̄|; DFW on every v
                // knot; then the per-axis static lock hook (below)
                let (mut llr_z, mut llr_p) = (Llr::new(sd.dims, block, kap, support.clone()), Llr::new(sd.dims, block, kap, support.clone()));
                let voff: Vec<usize> = (0..nk).map(|k| nk * nv + k * (3 * nv).div_ceil(2)).collect();
                let mut dfw_v = dfw_rule.map(|r| hasty_recon::dfwavelet::DfWavelet::new(sd.dims, dfw_levels, sd.vox, r));
                // spline mode: the priors act on the physical parameters (m, ψ0 = ψ' + k̄·v, v); converted at entry / exit
                let kbar_d = spec.kbar();
                let mut mwav = vmwav.map(|(t, l)| hasty_recon::wavelet::Wavelet::new(sd.dims, l, t));
                let mut lock_sig: Option<(Vec<f32>, f64, Vec<f32>, f64)> = None;
                // step diagnostics (spline mode, research/velocity_model_sr.md §F.2) on a support subsample
                let dsub: Vec<usize> = { let s: Vec<usize> = (0..nv).filter(|&q| support[q]).collect(); let st_ = (s.len() / 20000).max(1); s.into_iter().step_by(st_).collect() };
                let kmod = spec.kenc_model();
                let dbox = vbox.unwrap_or(vext_design.unwrap_or(r_first));
                let fpk = op_frames.as_ref().map_or(1.0, |o| o.fop as f64 / mc as f64);
                let mut diag_prev: Option<Vec<(f32, [f32; 3])>> = None;
                let mut den = |k: usize, h: &mut Vec<C32>| {
                    let it = (k + k_off) as u64;
                    let s_in = if spline { Some(vk_sample(h, &dsub, nk, nv, &voff)) } else { None };
                    let lv = (3 * nv).div_ceil(2);
                    let psi_shift = |h: &mut Vec<C32>, sign: f64| {
                        for kk in 0..nk {
                            let f = unpack_f32(&h[voff[kk]..voff[kk] + lv], 3 * nv);
                            h[kk * nv..(kk + 1) * nv].par_iter_mut().enumerate().for_each(|(q, z)| {
                                let d = kbar_d[0] * f[q] as f64 + kbar_d[1] * f[nv + q] as f64 + kbar_d[2] * f[2 * nv + q] as f64;
                                z.im = (z.im as f64 + sign * d) as f32;
                            });
                        }
                    };
                    if spline {
                        psi_shift(h, 1.0);
                    }
                    let zbar = |h: &[C32]| -> Vec<f32> { (0..nv).into_par_iter().map(|q| (0..nk).map(|kk| { let c = h[kk * nv + q]; if spline { c.re.abs() } else { (c.re * c.re + c.im * c.im).sqrt() } }).sum::<f32>() / nk as f32).collect() };
                    // complex pseudo-images z = m e^{iψ0} of the spline parameters, and back by continuity of ψ0
                    let to_z = |h: &[C32]| -> Vec<C32> { h[..nk * nv].par_iter().map(|c| { let (sn, cs) = (c.im).sin_cos(); C32::new(c.re * cs, c.re * sn) }).collect() };
                    let from_z = |h: &mut [C32], z: &[C32]| {
                        h[..nk * nv].par_iter_mut().zip(z.par_iter()).for_each(|(c, zn)| {
                            let zo = Complex64::from_polar(c.re as f64, c.im as f64);
                            let zn = Complex64::new(zn.re as f64, zn.im as f64);
                            let ps = c.im as f64 + if zo.norm() > 0.0 { (zn * zo.conj()).arg() } else { 0.0 };
                            let m = (zn * Complex64::from_polar(1.0, -ps)).re;
                            *c = C32::new(m as f32, ps as f32);
                        });
                    };
                    if kap > 0.0 {
                        llr_z.set_iteration(it);
                        if spline {
                            let mut zc = to_z(h);
                            llr_z.apply(&mut zc, nk);
                            from_z(h, &zc);
                        } else {
                            llr_z.apply(&mut h[..nk * nv], nk);
                        }
                        let zb = zbar(h);
                        let zmax = zb.iter().cloned().fold(0f32, f32::max);
                        let mut pimg = vec![C32::new(0.0, 0.0); 3 * nk * nv];
                        for kk in 0..nk {
                            let f = unpack_f32(&h[voff[kk]..voff[kk] + (3 * nv).div_ceil(2)], 3 * nv);
                            for c in 0..3 {
                                for q in 0..nv {
                                    pimg[(3 * kk + c) * nv + q] = C32::new(zb[q] * f[c * nv + q], 0.0);
                                }
                            }
                        }
                        llr_p.set_iteration(it);
                        llr_p.apply(&mut pimg, 3 * nk);
                        for kk in 0..nk {
                            let mut f = unpack_f32(&h[voff[kk]..voff[kk] + (3 * nv).div_ceil(2)], 3 * nv);
                            for c in 0..3 {
                                for q in 0..nv {
                                    if zb[q] > 1e-3 * zmax {
                                        f[c * nv + q] = pimg[(3 * kk + c) * nv + q].re / zb[q];
                                    }
                                }
                            }
                            let l = (3 * nv).div_ceil(2);
                            h[voff[kk]..voff[kk] + l].copy_from_slice(&pack_f32(&f, l));
                        }
                    }
                    if let Some(dfw) = dfw_v.as_mut() {
                        dfw.set_iteration(it);
                        let zb = zbar(h);
                        let b = dfw_mix as f32;
                        for kk in 0..nk {
                            let l = (3 * nv).div_ceil(2);
                            let mut f = unpack_f32(&h[voff[kk]..voff[kk] + l], 3 * nv);
                            let f0 = f.clone();
                            dfw.apply(&mut f[..3 * nv], Some(&zb));
                            for (a, a0) in f.iter_mut().zip(&f0) {
                                *a = a0 + b * (*a - a0);
                            }
                            h[voff[kk]..voff[kk] + l].copy_from_slice(&pack_f32(&f, l));
                        }
                    }
                    // spline mode: spatial db4 shrinkage of every magnitude knot (--vmwav)
                    if let (true, Some(w)) = (spline, mwav.as_mut()) {
                        w.set_iteration(it);
                        let mut mi: Vec<C32> = h[..nk * nv].par_iter().map(|c| C32::new(c.re, 0.0)).collect();
                        w.apply(&mut mi, nk);
                        h[..nk * nv].par_iter_mut().zip(mi.par_iter()).for_each(|(c, m)| c.re = m.re);
                    }
                    // per-axis static lock on (z, v) (hasty_recon::knotlock, --static-lock MU[:IT0[:EVERY]]): the map from
                    // the iterate from IT0, refreshed every EVERY iterations, the lock after the prior every iteration;
                    // spline mode: the map from the pseudo-images m e^{iψ0}, the lock on m, ψ0 (own noise levels) and v
                    if let (Some((_, it0, ev)), Some(lb), Some(lo)) = (&static_lock, &lock_basis, &lock_opts) {
                        let kit = k + k_off;
                        let lay = KnotLayout { dims: sd.dims, vox: sd.vox, kr, mc, nv, ne };
                        let refresh = kit >= *it0 && (lock_map.is_none() || (kit - it0) % ev == 0);
                        if refresh {
                            let t = Instant::now();
                            let zc = if spline { Some(to_z(h)) } else { None };
                            let (zs, vs) = lay.split(h);
                            let zr: Vec<&[C32]> = match &zc { Some(z) => z.chunks(nv).collect(), None => zs.iter().map(|s| &**s).collect() };
                            let vr: Vec<&[f32]> = vs.iter().map(|s| &**s).collect();
                            let m = hasty_recon::knotlock::estimate_vel(&zr, &vr, sd.dims, sd.vox, lb, lo, &dec, kit);
                            save_exact(&lmap_f, "<f4", &[m.stacked().len()], f4(m.stacked()));
                            std::fs::write(&lmap_j, serde_json::json!({ "iteration": m.iteration, "noise_mix": m.noise_mix }).to_string()).unwrap();
                            println!("    knot lock map (it {kit}, {:.1}s): static over the cycle {:.3}, over breathing {:.3}, fully {:.3}, undetermined {:.3}", t.elapsed().as_secs_f64(), m.static_card, m.static_resp, m.static_full, m.undetermined);
                            lock_map = Some(m);
                        }
                        if let Some(m) = &lock_map {
                            if spline {
                                let mut ms: Vec<Vec<f32>> = (0..nk).map(|kk| h[kk * nv..(kk + 1) * nv].iter().map(|c| c.re).collect()).collect();
                                let mut ps: Vec<Vec<f32>> = (0..nk).map(|kk| h[kk * nv..(kk + 1) * nv].iter().map(|c| c.im).collect()).collect();
                                if refresh || lock_sig.is_none() {
                                    let mr: Vec<&[f32]> = ms.iter().map(|v| &v[..]).collect();
                                    let pr_: Vec<&[f32]> = ps.iter().map(|v| &v[..]).collect();
                                    let (sm, tm) = hasty_recon::knotlock::real_noise(&mr, sd.dims, sd.vox, lb, lo);
                                    let (sp_, tp) = hasty_recon::knotlock::real_noise(&pr_, sd.dims, sd.vox, lb, lo);
                                    lock_sig = Some((sm, tm, sp_, tp));
                                }
                                let (sm, tm, sp_, tp) = lock_sig.as_ref().unwrap();
                                {
                                    let mut mm: Vec<&mut [f32]> = ms.iter_mut().map(|v| &mut v[..]).collect();
                                    hasty_recon::knotlock::apply_real(&mut mm, lb, m, lo, sm, *tm);
                                    let mut pp: Vec<&mut [f32]> = ps.iter_mut().map(|v| &mut v[..]).collect();
                                    hasty_recon::knotlock::apply_real(&mut pp, lb, m, lo, sp_, *tp);
                                }
                                for kk in 0..nk {
                                    for q in 0..nv {
                                        h[kk * nv + q] = C32::new(ms[kk][q], ps[kk][q]);
                                    }
                                }
                                let (_, mut vs) = lay.split(h);
                                hasty_recon::knotlock::apply_v(&mut vs, lb, m, lo);
                            } else {
                                let (mut zs, mut vs) = lay.split(h);
                                hasty_recon::knotlock::apply_z(&mut zs, lb, m, lo);
                                hasty_recon::knotlock::apply_v(&mut vs, lb, m, lo);
                            }
                        }
                    }
                    // --vbox: projection of the v coefficients onto the decoder's unaliased range (ψ0 kept)
                    if let Some(b) = vbox {
                        let b = b as f32;
                        for kk in 0..nk {
                            h[voff[kk]..voff[kk] + lv].par_iter_mut().for_each(|c| {
                                c.re = c.re.clamp(-b, b);
                                c.im = c.im.clamp(-b, b);
                            });
                        }
                    }
                    if spline {
                        psi_shift(h, -1.0);
                    }
                    if let Some(si) = s_in {
                        let so = vk_sample(h, &dsub, nk, nv, &voff);
                        println!("    {}", vk_diag(diag_prev.as_deref(), &si, &so, &kmod, dbox, fpk, dsub.len(), mc, k + k_off));
                        diag_prev = Some(so);
                    }
                };
                let oc = if state_host {
                    hasty_recon::hoststack::run_objective_host(&mut pr, &st, &mut xh, precond, 1.0 / lam, n_it, &mut den, &mut |k, xs: &HostStack, f| report_core(k, &|t| xs.chunk(t).to_vec(), f), Some((&ctl, &stag))).unwrap()
                } else {
                    hasty_recon::p2np::run_objective_with(&mut pr, &st, &mut xd, precond, 1.0 / lam, n_it, &mut den, &mut |k, xs: &Stack, f| report_core(k, &|t| st.clone_dtoh(&xs.v[t]).unwrap(), f), Some((&ctl, &stag)), &sopts).unwrap()
                };
                if let Outcome::Stopped(k) = oc {
                    println!("stopped {stag} at iteration {k} / {n_it}: state saved in {}, rerun the same command to resume", out.display());
                    std::process::exit(0);
                }
                let fin = if state_host { xh.to_value().unwrap() } else { hasty_recon::checkpoint::stack_value(&st, &xd).unwrap() };
                ctl.save(&format!("{stag}_final"), &fin).unwrap();
                std::fs::write(&done, b"").unwrap();
                k_off += n_it;
            }
            // outputs: the knot parameters (vparams_<tag>.hv + vknots_<tag>.json: --init-from of a finer level), the
            // velocity of the knot interpolant and (unless --no-full-save) the model images at the 20 cardiac frames
            // at the median breathing amplitude (recon_<tag>.npy: X, Y, Z, E, F, the other modes' form)
            let to = Instant::now();
            let chunks: Vec<Vec<C32>> = (0..lens.len()).map(|t| if state_host { xh.chunk(t).to_vec() } else { st.clone_dtoh(&xd.v[t]).unwrap() }).collect();
            drop(xd);
            drop(xh);
            {
                let tens: Vec<hasty_value::Value> = chunks.iter().map(|h| hasty_value::Value::Tensor(hasty_value::Tensor::from_bytes(hasty_value::DType::C64, vec![h.len()], c32_bytes(h)).unwrap())).collect();
                hasty_value::write(&hasty_value::Value::List(tens), &hasty_value::Options::default(), &mut std::io::BufWriter::new(std::fs::File::create(out.join(format!("vparams_{tag}.hv"))).unwrap())).unwrap();
                let meta = serde_json::json!({ "tag": tag, "dims": sd.dims, "kr": kr, "rdeg": kb.rdeg, "mc": mc, "cdeg": kb.cdeg, "rknots": kb.rknots, "resp": resp_src, "poly": vpoly, "lens": lens, "params": format!("vparams_{tag}.hv"), "param": if spline { "polar" } else { "complex" }, "kbar": spec.kbar(), "fop": op_frames.as_ref().map(|o| o.fop) });
                std::fs::write(out.join(format!("vknots_{tag}.json")), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
            }
            let vk: Vec<Vec<f32>> = (0..nk).map(|k| unpack_f32(&chunks[spec.chunk_v(k)], 3 * nv)).collect();
            let mut vout = vec![0f32; nv * 3 * nf_card];
            for fr in 0..nf_card {
                for q in 0..nv {
                    let mut v = [0.0f64; 3];
                    for k in 0..nk {
                        let w = wfr[fr][k];
                        if w != 0.0 {
                            for a in 0..3 {
                                v[a] += w * vk[k][a * nv + q] as f64;
                            }
                        }
                    }
                    let v = vphys(v, q);
                    for a in 0..3 {
                        vout[q + nv * (a + 3 * fr)] = v[a] as f32;
                    }
                }
            }
            drop(vk);
            save(&out.join(format!("vel_{tag}.npy")), "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 3, nf_card], f4(vout));
            if !no_full_save && spline {
                // the spline model's own signal at the 20 cardiac frames (median breathing amplitude), in voxel blocks
                let a = unpack_f32(&chunks[spec.chunk_a()], spec.na());
                let mut xout = vec![C32::new(0.0, 0.0); nv * ne * nf_card];
                let blk = 1 << 18;
                for q0 in (0..nv).step_by(blk) {
                    let idx: Vec<usize> = (q0..(q0 + blk).min(nv)).collect();
                    let vals = spline_lumen(&|t| chunks[t].clone(), &spec, kb, &a, &pex, &idx, amp_med, nf_card);
                    for (i, &q) in idx.iter().enumerate() {
                        for e in 0..ne {
                            for fr in 0..nf_card {
                                xout[q + nv * (e + ne * fr)] = vals[(i * ne + e) * nf_card + fr];
                            }
                        }
                    }
                }
                save(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], ne, nf_card], c8(&xout));
            } else if !no_full_save {
                // x_e(f) = Σ_k w_k(f) x_{e,k}: knot images on the device, accumulated per frame
                let mut xout = vec![C32::new(0.0, 0.0); nv * ne * nf_card];
                let (mut zs, mut vs) = (st.alloc_zeros::<C32>(lens[spec.chunk_z(0)]).unwrap(), st.alloc_zeros::<C32>(lens[spec.chunk_v(0)]).unwrap());
                let ad = st.clone_htod(&chunks[spec.chunk_a()]).unwrap();
                let xk = st.alloc_zeros::<C32>(nv).unwrap();
                let mut acc: Vec<CudaSlice<C32>> = (0..nf_card).map(|_| st.alloc_zeros::<C32>(nv).unwrap()).collect();
                for e in 0..ne {
                    for a in acc.iter_mut() {
                        st.memset_zeros(a).unwrap();
                    }
                    for k in 0..nk {
                        st.memcpy_htod(&chunks[spec.chunk_z(k)], &mut zs).unwrap();
                        st.memcpy_htod(&chunks[spec.chunk_v(k)], &mut vs).unwrap();
                        use cudarc::driver::DevicePtr;
                        let (zp, vp, ap, xp) = (zs.device_ptr(&st).0, vs.device_ptr(&st).0, ad.device_ptr(&st).0, xk.device_ptr(&st).0);
                        pr.model.eval_knot(&st, zp, vp, ap, e, xp).unwrap();
                        for fr in 0..nf_card {
                            let w = wfr[fr][k];
                            if w != 0.0 {
                                kern.axpby(&st, Complex64::new(w, 0.0), &xk, Complex64::new(1.0, 0.0), &mut acc[fr]).unwrap();
                            }
                        }
                    }
                    for fr in 0..nf_card {
                        st.memcpy_dtoh(&acc[fr], &mut xout[nv * (e + ne * fr)..nv * (e + ne * fr + 1)]).unwrap();
                    }
                }
                save(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], ne, nf_card], c8(&xout));
            }
            if np > 0 {
                let af = unpack_f32(&chunks[spec.chunk_a()], spec.na());
                let rec: Vec<Vec<f64>> = (1..ne).map(|e| (0..np).map(|p| (af[(e - 1) * np + p] as f64 * 1000.0).round() / 1000.0).collect()).collect();
                println!("  recovered residual phase (rad, Legendre order of vmodel::poly_exps): {rec:?}");
                if !injected.is_empty() && vpoly == Some(1) {
                    let err: f64 = (1..ne).map(|e| (0..4).map(|p| (af[(e - 1) * np + p] as f64 - injected[e - 1][p]).powi(2)).sum::<f64>()).sum::<f64>().sqrt() / ((ne - 1) as f64 * 4.0).sqrt();
                    println!("  residual-phase RMS error vs injected: {err:.4} rad");
                }
            }
            println!("  {tag}: outputs written ({:.0}s): vel_{tag}, {}vparams_{tag}.hv + vknots_{tag}.json{}", to.elapsed().as_secs_f64(), if no_full_save { "" } else { "recon_<tag> (20 cardiac frames at the median breathing amplitude), " }, if evals.is_empty() { String::new() } else { format!(", knots_{tag}_r*_it*_lumen") });
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
        let spec = VelSpec { dims: sd.dims, kenc: kenc.clone(), nf, kz, deg_z: vdeg.0, kv, deg_v: vdeg.1, poly: vpoly, knots: None, op: None };
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
        info["adc_part"] = serde_json::json!(adc_part.as_str());
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
            if !resuming && hasty_value::store::exists_any(&out.join(format!("recon_{tag}.npy"))) {
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
            let mut lam_ritz: Option<Vec<C32>> = None;
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
                let lam = match lam_method {
                    LamMethod::Power => pr.lambda_max(&st, &x, with_metric, 12).unwrap(),
                    LamMethod::Lanczos(k) => {
                        let warm = lam_warm_on(lens.iter().sum::<usize>() * 8, k);
                        let e = pr.lambda_lanczos(&st, &x, with_metric, k, warm.then_some(&mut lam_ritz)).unwrap();
                        println!("  λ_max by Lanczos ({k} steps): θ {:.4e} + residual bound {:.2e}", e.theta, e.bound);
                        e.hat()
                    }
                };
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
    // --adj-test: ⟨E x, y⟩ = ⟨x, Eᴴ y⟩ of the exact operator the solver uses (random x, y; a few sets), and the grouped
    // forward / adjoint against the member-wise ones (the pooled gradient path)
    if adj_test {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = |n: usize| -> Vec<C32> {
            (0..n)
                .map(|_| {
                    let mut u = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 };
                    C32::new(u(), u())
                })
                .collect()
        };
        let dotc = |a: &[C32], b: &[C32]| -> Complex64 { a.iter().zip(b).map(|(p, q)| Complex64::new(p.re as f64, -p.im as f64) * Complex64::new(q.re as f64, q.im as f64)).sum() };
        let nrm = |a: &[C32]| -> f64 { a.iter().map(|v| (v.re as f64).powi(2) + (v.im as f64).powi(2)).sum::<f64>().sqrt() };
        let ns_all = sop.nsets();
        for &t in &[0, ns_all / 2, ns_all - 1] {
            let (nx, ny) = (sop.len_x(t), sop.len_y(t));
            let (xh, yh) = (rnd(nx), rnd(ny));
            let (xd, yd) = (st.clone_htod(&xh).unwrap(), st.clone_htod(&yh).unwrap());
            let mut ax = st.alloc_zeros::<C32>(ny).unwrap();
            let mut ahy = st.alloc_zeros::<C32>(nx).unwrap();
            sop.forward(&st, t, &xd, &mut ax).unwrap();
            sop.adjoint(&st, t, &yd, &mut ahy).unwrap();
            let (axh, ahyh) = (st.clone_dtoh(&ax).unwrap(), st.clone_dtoh(&ahy).unwrap());
            let (l, r) = (dotc(&axh, &yh), dotc(&xh, &ahyh));
            println!("adjoint test set {t}: ⟨Ex, y⟩ = {l:.6e}, ⟨x, Eᴴy⟩ = {r:.6e}, |diff| / (‖Ex‖‖y‖) = {:.2e}", (l - r).norm() / (nrm(&axh) * nrm(&yh)));
        }
        // grouped vs member-wise: the first 4 sets of encode 0
        let mem: Vec<usize> = (0..ns_all).filter(|&t| enc_of(t) == 0).take(4).collect();
        let nx = sop.len_x(mem[0]);
        let xh = rnd(nx);
        let xd = st.clone_htod(&xh).unwrap();
        let mut ys_g: Vec<CudaSlice<C32>> = mem.iter().map(|&t| st.alloc_zeros::<C32>(sop.len_y(t)).unwrap()).collect();
        {
            let mut refs: Vec<&mut CudaSlice<C32>> = ys_g.iter_mut().collect();
            sop.forward_group(&st, &mem, &xd, &mut refs).unwrap();
        }
        let (mut ef, mut nf2) = (0.0f64, 0.0f64);
        let mut sum_adj = vec![Complex64::new(0.0, 0.0); nx];
        for (j, &t) in mem.iter().enumerate() {
            let mut y1 = st.alloc_zeros::<C32>(sop.len_y(t)).unwrap();
            sop.forward(&st, t, &xd, &mut y1).unwrap();
            let (a, b) = (st.clone_dtoh(&y1).unwrap(), st.clone_dtoh(&ys_g[j]).unwrap());
            ef += a.iter().zip(&b).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum::<f64>();
            nf2 += nrm(&a).powi(2);
            let mut g1 = st.alloc_zeros::<C32>(nx).unwrap();
            sop.adjoint(&st, t, &y1, &mut g1).unwrap();
            for (s, v) in sum_adj.iter_mut().zip(st.clone_dtoh(&g1).unwrap()) {
                *s += Complex64::new(v.re as f64, v.im as f64);
            }
        }
        let mut gg = st.alloc_zeros::<C32>(nx).unwrap();
        {
            let refs: Vec<&CudaSlice<C32>> = ys_g.iter().collect();
            sop.adjoint_group(&st, &mem, &refs, &mut gg).unwrap();
        }
        let ggh = st.clone_dtoh(&gg).unwrap();
        let ea: f64 = ggh.iter().zip(&sum_adj).map(|(p, q)| (Complex64::new(p.re as f64, p.im as f64) - q).norm_sqr()).sum::<f64>();
        let na: f64 = sum_adj.iter().map(|q| q.norm_sqr()).sum();
        println!("grouped vs member-wise ({} sets of encode 0): forward rel {:.2e}, adjoint rel {:.2e}", mem.len(), (ef / nf2).sqrt(), (ea / na).sqrt());
    }
    let mut p = if staged {
        let sg = hasty_recon::kstage::Stager::new(&dev.ctx, kstore.clone().unwrap()).unwrap();
        println!("data staged from the store: {} pinned staging buffers of {:.2} GiB", 2, sg.pinned_bytes() as f64 / 2.0 / GIB);
        P2np::from_store(sop, sg, circ, &st).unwrap().with_circ_map_opt(circ_map.clone())
    } else {
        P2np::new(sop, ys.iter().collect(), circ, &st).unwrap().with_circ_map_opt(circ_map.clone())
    };
    if lam_compare {
        // diagnostic: every estimator on the same operators (set 0 and the middle set), relative to power 100
        let tc = Instant::now();
        for circ in [false, true] {
            if circ && !need_circ {
                continue;
            }
            let r = p.step_norm_with(&st, circ, LamMethod::Power, 100).unwrap();
            let mut line = format!("λ compare ({}): power 100 = {:.5e} ({} applies)", if circ { "M N" } else { "N" }, r.theta, r.applies);
            let e = p.step_norm_with(&st, circ, LamMethod::Power, 12).unwrap();
            line += &format!("; power 12 {:+.2e}", e.theta / r.theta - 1.0);
            for k in [4usize, 6, 8] {
                let e = p.step_norm_with(&st, circ, LamMethod::Lanczos(k), 12).unwrap();
                line += &format!("; lanczos {k}: θ {:+.2e}, θ+b {:+.2e}", e.theta / r.theta - 1.0, e.hat() / r.theta - 1.0);
            }
            println!("{line}");
        }
        println!("λ compare: {:.0}s", tc.elapsed().as_secs_f64());
    }
    // --pp-study / --pp-pool / --pp-metric (research/precond_estimator.md): before the step sizes, so λ_max(M N) is the
    // new metric's
    if pp_args.active() {
        assert!(temporal.is_none() && vmodel.is_none() && knots2d.is_none(), "--pp-*: frame mode (one image per set)");
        let enc: Vec<usize> = (0..fe.len()).map(enc_of).collect();
        pp_run(&pp_args, &mut p, &st, &dev, &kern, sd.dims, &enc, ne, &support, &smaps, nvc, &grid.omega, synth_ref.as_ref(), &out, &ctl, circ_ws.as_ref(), warm_full, &host_sets, pp_seg_host.as_deref(), pp_coords.as_deref());
    }
    // λ_max(N) only when a precondition without the metric uses it, λ_max(M N) only when one with it does
    let need_plain = preconds.iter().any(|q| !Precond::parse(q).unwrap_or(Precond::Zmshr1).metric());
    let tl = Instant::now();
    let mut est_n = if need_plain { Some(p.step_norm_with(&st, false, lam_method, 12).unwrap()) } else { None };
    let mut est_c = if need_circ { Some(p.step_norm_with(&st, true, lam_method, 12).unwrap()) } else { None };
    if let Some(cs) = coil_sampling.clone() {
        // the step must be stable for every subset: α from the largest λ_max over a few draws
        p = p.with_sampling(cs.clone());
        let ls = est_n.map(|_| p.step_norm_sampled_with(&st, false, lam_method, 12, 4).unwrap());
        let lsc = est_c.map(|_| p.step_norm_sampled_with(&st, true, lam_method, 12, 4).unwrap());
        let ratio = |a: Option<LamEst>, b: Option<LamEst>| a.zip(b).map_or("-".to_string(), |(a, b)| format!("{:.3}", a.hat() / b.hat()));
        println!("coil sampling: keep {} + draw {} of {} (hold {}, last {} full): λ_max(sampled) / λ_max(full) = {} (M-weighted {})", cs.keep, cs.draw, nvc - cs.keep, cs.hold, cs.full_last, ratio(ls, est_n), ratio(lsc, est_c));
        est_n = est_n.zip(ls).map(|(a, b)| a.max(b));
        est_c = est_c.zip(lsc).map(|(a, b)| a.max(b));
    }
    let lmax = est_n.map_or(f64::NAN, |e| e.hat());
    let lmax_c = est_c.map_or(1.0, |e| e.hat());
    let fmt_est = |e: Option<LamEst>| e.map_or("not needed".to_string(), |e| format!("{:.3e} (θ {:.4e} + bound {:.2e})", e.hat(), e.theta, e.bound));
    println!("step size by {lam_method}: λ_max(N) = {}, λ_max(M N) = {}; {} operator applications, {:.1}s", fmt_est(est_n), fmt_est(est_c), est_n.map_or(0, |e| e.applies) + est_c.map_or(0, |e| e.applies), tl.elapsed().as_secs_f64());
    if let Some(tau) = lag_tau {
        p = p.with_lagged(tau);
    }
    println!("lambda_max(N) = {lmax:.3e}, lambda_max(M N) = {lmax_c:.3e}, setup {:.0}s", t0.elapsed().as_secs_f64());
    // physics-model warm start (--warm-full): pooled PCG per encode through the full objective
    let warm: Vec<Vec<C32>> = if warm_full > 0 {
        // --temporal: one set per encode holding all knot images; the pooled start ties them (ConstOverFrames: every knot
        // the same image, the time-static problem through the full operator since Σ_k B_k = 1), one group per set
        assert!(temporal.is_none() || !(need_circ && !pool_metric_none), "--warm-full with --temporal: the plain pooled start only (--pool-metric none)");
        let group_of: Vec<usize> = (0..fe.len()).map(enc_of).collect();
        let done: Vec<String> = (0..ne).map(|e| format!("warm_full_done_{e}")).collect();
        let loaded: Vec<Option<Vec<C32>>> = done.iter().map(|d| ctl.load(d).unwrap().map(|v| st.clone_dtoh(&value_dev(&st, &v, nv).unwrap()).unwrap())).collect();
        let xw: Vec<Vec<C32>> = if loaded.iter().all(|v| v.is_some()) {
            println!("physics-model warm start: loaded ({} encodes)", ne);
            loaded.into_iter().map(|v| v.unwrap()).collect()
        } else {
            let mut xw: Vec<Vec<C32>> = if warm.is_empty() { vec![vec![C32::new(0.0, 0.0); nv]; ne] } else { warm.clone() };
            let tw = Instant::now();
            let mut p_const = temporal.map(|_| hasty_recon::p2np::ConstOverFrames::new(&mut p, nf));
            let obj_w: &mut dyn hasty_recon::p2np::Objective = match p_const.as_mut() { Some(c) => c, None => &mut p };
            let lens_w = obj_w.lens();
            assert!(lens_w.iter().all(|&l| l == nv), "--warm-full: one image per set");
            let mut probe = synth_ref.as_ref().map(|(xm, stat, bm)| {
                move |k: usize, xs: &hasty_recon::hoststack::HostStack| {
                    if k <= 3 || k % 5 == 0 {
                        println!("  pooled image vs x★ frame mean, it {k}: {}", pooled_image_error(xs, xm, stat, bm));
                    }
                }
            });
            let oc = hasty_recon::hoststack::pooled_pcg_host_probe(obj_w, &st, &group_of, &mut xw, warm_full, need_circ && !pool_metric_none, Some((&ctl, "warm_full")), 2, pooled_grad, probe.as_mut().map(|f| f as &mut dyn FnMut(usize, &hasty_recon::hoststack::HostStack))).unwrap();
            if let Outcome::Stopped(k) = oc {
                println!("stopped in the physics-model warm start (PCG iteration {k}): state saved, rerun the same command to resume");
                std::process::exit(0);
            }
            for (e, d) in done.iter().enumerate() {
                ctl.save(d, &hasty_recon::checkpoint::dev_value(&st, &st.clone_htod(&xw[e]).unwrap()).unwrap()).unwrap();
            }
            println!("physics-model warm start: {warm_full} pooled PCG iterations per encode{} ({:.0}s)", if need_circ && !pool_metric_none { " (per-encode circulant)" } else { "" }, tw.elapsed().as_secs_f64());
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
        if warm_only {
            println!("--warm-only: pooled warm start done, checkpoints state_warm_full_done_<e> kept in {}", out.display());
            std::process::exit(0);
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
                let oc = hasty_recon::hoststack::pooled_pcg_host_with(&mut p, &st, &gof, &mut xg, it, need_circ && !pool_metric_none, Some((&ctl, &format!("tpyr_{li}"))), 2, pooled_grad).unwrap();
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
    // --init meanres / --anchor: frame mode, the pooled solution, the data scale d̄ and λ_max(M) for the step bound
    let need_meanres = init_specs.iter().any(|s| s.starts_with("meanres"));
    let any_anchor = anchor_specs.iter().any(|a| a.is_some());
    if need_meanres || any_anchor || init_specs.iter().any(|s| s == "zero") {
        assert!(temporal.is_none() && p.lens().iter().all(|&l| l == nv), "--init / --anchor: frame mode (one image per set, no --temporal)");
        assert!(init_from.is_none() && !init_truth && tpyramid.is_none() || !need_meanres, "--init meanres: no --init-from / --init-truth / --tpyramid");
    }
    if need_meanres || anchor_specs.iter().any(|a| a.as_ref().is_some_and(|(s, _)| s == "mean")) {
        assert!(warm_full > 0, "--init meanres / --anchor mean: need the pooled solution (--warm-full n)");
    }
    let anc_scale: Option<(f64, f64)> = any_anchor.then(|| {
        assert!(lag_tau.is_none() && coil_sampling.is_none(), "--anchor with --lagged / --coil-sampling: not yet");
        let ta = Instant::now();
        let dbar = p.mean_diag(&st).unwrap();
        // one set per distinct circulant (per encode when shared), +5 % for the power iteration's approach from below
        let lam_m = if need_circ {
            let reps: Vec<usize> = (0..ne).map(|e| (0..fe.len()).find(|&t| enc_of(t) == e).unwrap()).collect();
            1.05 * p.metric_lmax(&st, &reps, 30).unwrap()
        } else {
            f64::NAN
        };
        println!("temporal anchor: mean diagonal of one frame's normal operator d̄ = {dbar:.4e} (λ_max(N)/d̄ = {:.2}), λ_max(M) = {lam_m:.4e} (λ_max(M N)/λ_max(M) = {:.3e}) ({:.1}s)", lmax / dbar, lmax_c / lam_m, ta.elapsed().as_secs_f64());
        (dbar, lam_m)
    });
    // anchor groups and targets per spec (set → group, group images), built once
    let anchor_data: Vec<Option<(Vec<usize>, Vec<Vec<C32>>, f64, String)>> = anchor_specs
        .iter()
        .map(|a| {
            let (src, mu) = a.as_ref()?;
            let (dbar, _) = anc_scale.unwrap();
            if src == "mean" {
                let gof: Vec<usize> = (0..fe.len()).map(enc_of).collect();
                return Some((gof, warm.clone(), mu * dbar, format!("_anc_mean{mu}")));
            }
            let ini = hasty_recon::pyramid::InitImages::open(std::path::Path::new(src)).unwrap_or_else(|e| panic!("--anchor {src}: {e}"));
            assert_eq!(ini.ne, ne, "--anchor {src}: {} encodes, this scan has {ne}", ini.ne);
            let ta = Instant::now();
            let gof: Vec<usize> = (0..fe.len()).map(|t| hasty_recon::tanchor::nested_frame(fe[t].0.unwrap(), nf_card, resp_bins, ini.nf) * ne + enc_of(t)).collect();
            let tg: Vec<Vec<C32>> = (0..ini.nf * ne).map(|a| ini.image(a / ne, a % ne, sd.dims).unwrap()).collect();
            println!("temporal anchor {src}: {} coarse frames → {nf} frames, resampled {:?} → {:?} ({:.1}s)", ini.nf, ini.dims, sd.dims, ta.elapsed().as_secs_f64());
            Some((gof, tg, mu * dbar, format!("_anc{}f{mu}", ini.nf)))
        })
        .collect();
    // --prior-mode admm: ρ₁ = c · D_body from the operator (c by a Hutchinson trace of N_t: tr N_t = c m_t D̄ in the
    // per-cell model N_t ≈ D^{1/2} F⁻¹ diag(c n_t) F D^{1/2}; an unnormalised NUFFT gives c = nv × the samples' mean
    // physics attenuation), a plane-wave check of N_t at two cells of one set, and the clip metric on the device
    let mut admm_info = serde_json::Value::Null;
    let mut admm_rho1 = f64::NAN;
    let mut admm_clip: Option<hasty_recon::admm::ClipMetric> = None;
    if let Some(a) = admm_sym {
        assert!(temporal.is_none() && vmodel.is_none() && knots2d.is_none() && lag_tau.is_none() && coil_sampling.is_none(), "--prior-mode admm: frame mode with exact gradients (no --temporal, velocity model, --lagged, --coil-sampling)");
        let ta = Instant::now();
        let nt = p.op.nsets();
        let dims = sd.dims;
        let ly: Vec<usize> = (0..nt).map(|t| p.op.len_y(t)).collect();
        // ‖E_t v‖² = vᴴ N_t v (one forward)
        let mut quad = |t: usize, v: &[C32]| -> f64 {
            let vd = st.clone_htod(v).unwrap();
            let mut e = st.alloc_zeros::<C32>(ly[t]).unwrap();
            p.op.forward(&st, t, &vd, &mut e).unwrap();
            kern.dot(&st, &e, &e).unwrap().re
        };
        // Hutchinson: complex ±1 ± i probes (|z_q|² = 2), 4 per set, 4 sets
        let mut sets = vec![0, nt / 3, 2 * nt / 3, nt - 1];
        sets.dedup();
        let mut cs: Vec<f64> = vec![];
        for &t in &sets {
            let mut acc = 0.0;
            for pr in 0..4u64 {
                let mut sd_ = 0x9E37_79B9_7F4A_7C15u64 ^ ((t as u64) << 8 | pr).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let z: Vec<C32> = (0..nv)
                    .map(|_| {
                        sd_ ^= sd_ << 13;
                        sd_ ^= sd_ >> 7;
                        sd_ ^= sd_ << 17;
                        C32::new(if sd_ & 1 == 0 { 1.0 } else { -1.0 }, if sd_ & 2 == 0 { 1.0 } else { -1.0 })
                    })
                    .collect();
                acc += quad(t, &z);
            }
            let tr = acc / 4.0 / 2.0;
            let m = ly[t] / nvc;
            cs.push(tr / (m as f64 * a.dbar));
        }
        let c_op = cs.iter().sum::<f64>() / cs.len() as f64;
        let mut db: Vec<f32> = (0..nv).filter(|&q| support[q]).map(|q| a.dsum[q]).collect();
        db.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let d_body = db.get(db.len() / 2).copied().unwrap_or(1.0) as f64;
        let rho1 = c_op * d_body;
        println!("admm ρ₁: c = tr(N_t) / (m_t D̄) = {c_op:.5e} = {:.4} nv (sets {sets:?}: {}; an unnormalised NUFFT gives nv × the mean physics attenuation), D = Σ_c |S_c|²: mean {:.4e}, body median {d_body:.4e} → ρ₁ = c · D_body = {rho1:.5e}; λ_max(N) / ρ₁ = {:.1}", c_op / nv as f64, cs.iter().map(|c| format!("{:.4}", c / nv as f64)).collect::<Vec<_>>().join(" "), a.dbar, lmax / rho1);
        // plane waves e_k (unit norm, FFT index k: F e_k ∝ δ_k) at the check set's cells; the per-cell model predicts
        // ⟨e_k, N_t e_k⟩ = c D̄ μ̃_t(k) with the set's own coil-smoothed count
        let mut checks = vec![];
        for &(q, cnt, kb) in &a.cells {
            let kq = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
            let pw = |sg: f64| -> Vec<C32> {
                let s0 = 1.0 / (nv as f64).sqrt();
                (0..nv)
                    .map(|n| {
                        let i = [n % dims[0], (n / dims[0]) % dims[1], n / (dims[0] * dims[1])];
                        let ph = sg * std::f64::consts::TAU * (0..3).map(|ax| (kq[ax] * i[ax]) as f64 / dims[ax] as f64).sum::<f64>();
                        C32::new((s0 * ph.cos()) as f32, (s0 * ph.sin()) as f32)
                    })
                    .collect()
            };
            let (rp, rm) = (quad(a.tc, &pw(1.0)), quad(a.tc, &pw(-1.0)));
            let model = c_op * a.dbar * a.mu_t[q] as f64;
            println!("  plane wave at cell {q} (|k| {kb:.2} of the ball radius, {cnt} sample(s) of set {} in the cell): ⟨e_k, N e_k⟩ = {rp:.4e} = {:.4} ρ₁ (mirror −k: {rm:.4e}); per-cell model c D̄ μ̃_t(k) = {model:.4e} (μ̃_t = {:.3}), measured / model {:.3}", a.tc, rp / rho1, a.mu_t[q], rp / model.max(1e-300));
            checks.push(serde_json::json!({ "cell": q, "k_ball": kb, "samples": cnt, "rayleigh": rp, "mirror": rm, "rho1_units": rp / rho1, "model": model, "mu_t": a.mu_t[q] }));
        }
        admm_rho1 = rho1;
        if admm_metric == hasty_recon::admm::AdmmMetric::Clip {
            let dfl = (0.1 * d_body) as f32;
            let dinv: Vec<f32> = a.dsum.iter().map(|&d| 1.0 / d.max(dfl).sqrt()).collect();
            let cm = hasty_recon::admm::ClipMetric::new(&dev, &st, kern.clone(), sd.dims, a.mu, a.of_set, rho1, &dinv).unwrap();
            let fr: Vec<String> = (0..ne)
                .map(|e| {
                    let (fa, fo) = cm.dense_fraction(e);
                    format!("{:.2} % ({:.2} % of the occupied)", 100.0 * fa, 100.0 * fo)
                })
                .collect();
            println!("admm clip metric: D = max(Σ_c |S_c|², 0.1 × body median {d_body:.3e}); cells with μ̃ ≥ 1 per encode {}", fr.join(", "));
            admm_clip = Some(cm);
        }
        println!("admm setup: ρ = {admm_rho} ρ₁ = {:.5e}{}, K = {admm_inner} inner PCG iterations per set, metric {} ({:.1}s)", admm_rho * rho1, if admm_adapt { " (residual balancing)" } else { "" }, admm_metric.name(), ta.elapsed().as_secs_f64());
        admm_info = serde_json::json!({ "rho_rel": admm_rho, "rho1": rho1, "c": c_op, "c_over_nv": c_op / nv as f64, "c_sets": cs, "d_mean": a.dbar, "d_body": d_body, "inner": admm_inner, "metric": admm_metric.name(), "adapt": admm_adapt, "lambda_max_over_rho1": lmax / rho1, "plane_waves": checks });
    }
    let ttag = temporal.map_or(String::new(), |b| format!("_t{b}"));
    let runs: Vec<String> = preconds.iter().flat_map(|p| kappas.iter().map(|k| (p.clone(), *k)).collect::<Vec<_>>()).map(|(p, k)| format!("{model}{ttag}_{p}{}_k{k}{}", if learned.is_some() { "_learned" } else { "" }, llr_rank.map_or(String::new(), |r| format!("_r{r}")) + if llr_hard { "_h" } else { "" } + &red_tag + &admm_tag + &ga_tag)).collect();
    let mut info = serde_json::json!({
        "cmdline": cmdline, "sim_dir": dir, "model": model, "temporal": temporal, "phys": phys, "gnl": gnl, "nseg": nseg,
        "order": order, "conc_fit": conc_fit, "fat_model": fat_fraction.as_ref().map(|_| fat_model.clone()), "eps": eps, "iters": iters, "warm_iters": warm_iters, "block": block, "vcoils": nvc, "dims": sd.dims, "vox": sd.vox, "native_dims": sd.tdims,
        "n_frames": nf, "n_enc": ne, "venc": sd.venc, "venc_range": dec.range(), "encoding": sd.meta["encoding"], "lambda_max": est_n.map(|e| e.hat()), "lambda_max_mn": est_c.map(|e| e.hat()), "lambda_method": lam_method.to_string(), "lambda_applies": est_n.map_or(0, |e| e.applies) + est_c.map_or(0, |e| e.applies), "setup_s": t0.elapsed().as_secs_f64(),
        "noise_sigma_env": std::env::var("HASTY_NOISE_SIGMA").ok(), "runs": runs,
        "prior": if learned.is_some() { "learned" } else { "llr" }, "prior_mode": prior_mode, "red_rho": red.then_some(red_rho), "red_shifts": red.then_some(red_shifts), "tsub": tsub_spec, "learned_dir": learned_dir, "tile": tile, "keep_frames": keep_frames, "learned_prec": learned_prec,
        "field": field_spec, "r2s": r2s_spec, "smaps": smaps_spec, "coil_mask": coil_mask_mode, "coils": coil_info, "estimates": diag, "kdata": kstore_info, "lagged": lag_info, "vdecode": vdec_info,
    });
    info["pyramid"] = pyr_info.clone();
    info["adc_part"] = serde_json::json!(adc_part.as_str());
    info["admm"] = admm_info.clone();
    info["segments"] = serde_json::json!({ "basis": seg_basis_arg, "weight": seg_weight }); info["respiration"] = resp_info.clone(); info["prior_detail"] = serde_json::json!({ "prior": prior_kind, "wavelet": { "thr": wav_thr, "levels": wav_levels, "mix": wav_mix } });
    info["prior_detail"]["dfw"] = serde_json::json!(dfw_rule.map(|r| serde_json::json!({ "rule": format!("{r:?}"), "levels": dfw_levels, "mix": dfw_mix })));
    info["prior_detail"]["static_lock"] = serde_json::json!(static_lock.as_ref().map(|(o, it0, ev)| serde_json::json!({ "opts": format!("{o:?}"), "it0": it0, "every": ev })));
    info["prior_detail"]["air"] = serde_json::json!(air.as_ref().map(|(o, it0, ev)| serde_json::json!({ "opts": format!("{o:?}"), "it0": if *it0 == usize::MAX { None } else { Some(it0) }, "every": if *ev == usize::MAX { None } else { Some(ev) }, "init": air_init })));
    std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
    let rt = transpose(&sd.rot);
    let variants: Vec<(&String, f64, &String, usize)> = preconds.iter().flat_map(|p| kappas.iter().map(move |&k| (p, k))).flat_map(|(p, k)| { let na = anchor_specs.len(); init_specs.iter().flat_map(move |i| (0..na).map(move |a| (p, k, i, a))) }).collect();
    let mut meanres_last: Option<(String, hasty_recon::hoststack::HostStack)> = None;
    for (pc, kappa, init_spec, anc_i) in variants {
        let precond = Precond::parse(pc).unwrap_or(Precond::Zmshr1);
        // the anchor's curvature μ_abs on the group-constant mode adds to λ_max (Weyl): μ_abs, or μ_abs λ_max(M)
        let anc = anchor_data[anc_i].as_ref();
        let lam_anchor = anc.map_or(0.0, |a| if precond.metric() { a.2 * anc_scale.unwrap().1 } else { a.2 });
        let alpha = if precond.metric() { 1.0 / (lmax_c + lam_anchor) } else { 1.0 / (lmax + lam_anchor) };
        if let Some(a) = anc {
            println!("  temporal anchor {}: μ_abs = {:.4e}, λ_max {:.4e} → {:.4e} ({})", &a.3[1..], a.2, if precond.metric() { lmax_c } else { lmax }, 1.0 / alpha, if precond.metric() { "M N" } else { "N" });
        }
        let stack_bytes = lens.iter().sum::<usize>() * 8;
        // --prior-mode admm: host stacks always (x, z, u and a work stack; the data step streams set by set)
        let host_state = admm || match state_arg.as_str() {
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
        // --llr-thr freeze: λ of the fixed objective, set by the first LLR call of this arm
        let mut llr_lambda: Option<f64> = None;
        let mut llr_gpu = (llr_devices != "cpu" && prior_kind == "llr" && kappa > 0.0 && learned.is_none()).then(|| {
            let ords: Vec<usize> = llr_devices.split(',').map(|v| v.trim().parse().expect("--llr-devices 0[,1,..] | cpu")).collect();
            let mut l = Llr::new(sd.dims, block, kappa, support.clone());
            l.rank = llr_rank;
            l.hard = llr_hard;
            hasty_recon::llr_gpu::LlrGpu::new(l, &ords).unwrap()
        });
        let mut wav = hasty_recon::wavelet::Wavelet::new(sd.dims, wav_levels, wav_thr);
        let nimg = nf * ne;
        // the start stack (the warm images in every frame, the solver's x0), host layout of the denoiser: shared by
        // every solver arm, so the frozen LLR penalty and the start-image air map are the same objective for all
        let lens_h = lens.clone();
        let start_stack = |warm: &Vec<Vec<C32>>| -> Vec<C32> {
            let off: Vec<usize> = lens_h.iter().scan(0, |a, &n| { let o = *a; *a += n; Some(o) }).collect();
            let mut h0 = vec![C32::new(0.0, 0.0); lens_h.iter().sum()];
            for fr in 0..nf {
                for e in 0..ne {
                    let (t, o) = at(fr, e);
                    h0[off[t] + o..off[t] + o + nv].copy_from_slice(&warm[e]);
                }
            }
            h0
        };
        let h0: Option<Vec<C32>> = (llr_freeze || air_start).then(|| {
            assert!(!warm.is_empty(), "--llr-thr freeze / --air start: need a warm start (--warm-full N)");
            start_stack(&warm)
        });
        // --llr-thr freeze: λ of the fixed objective f + λR in objective units, from the start stack: λ = κσ̂(x0)/α with
        // α = 1/λ_max(N) (the PnP-ISTA step); every prox call then thresholds at λ·s, s its own step (P²nP α, the metric
        // prox ατ_k, ADMM 1/ρ): all solvers minimise the same objective, and a resume recomputes the same λ
        if llr_freeze {
            assert!(llr_shifts.unwrap_or(1) == 1 || !prox_metric, "--prox-metric with --llr-thr freeze needs --llr-shifts 1 (a proximal average over S > 1 shifts is not homogeneous in the step)");
            if llr_shifts.unwrap_or(1) > 1 {
                eprintln!("warning: --llr-shifts > 1 with --llr-thr freeze: the proximal average over S shifts is not homogeneous in the prox step, so solvers with different steps (P²nP vs ADMM) minimise slightly different objectives");
            }
            // σ̂ on PnP-ISTA's first prox input z0 = x0 − α∇f(x0), α = 1/λ_max(N) (the start stack itself is rank ≤ ne
            // per block — every frame the same image — so its lower singular values are 0): deterministic given x0
            let mut v = {
                let x0 = h0.as_ref().unwrap();
                let mut xs = hasty_recon::hoststack::HostStack::zeros(&lens);
                xs.data.copy_from_slice(x0);
                let mut gs = hasty_recon::hoststack::HostStack::zeros(&lens);
                let f0 = hasty_recon::p2np::Objective::gradient_host(&mut p, &st, &xs, &mut gs).unwrap();
                let a0 = 1.0 / lmax;
                println!("  start stack: data term f(x0) = {f0:.4e}; σ̂ measured on z0 = x0 − ∇f(x0)/λ_max");
                x0.iter().zip(&gs.data).map(|(x, g)| C32::new(x.re - (a0 * g.re as f64) as f32, x.im - (a0 * g.im as f64) as f32)).collect::<Vec<C32>>()
            };
            let sig = match llr_gpu.as_mut() {
                Some(g) => { g.set_iteration(0); g.llr.thr_abs = None; g.apply(&mut v, nimg).unwrap(); g.llr.last.sigma }
                None => { llr.set_iteration(0); llr.thr_abs = None; llr.apply(&mut v, nimg); llr.last.sigma }
            };
            let lam = kappa * sig * lmax;
            assert!(lam.is_finite() && lam > 0.0, "--llr-thr freeze: no σ̂ on z0 (σ̂ {sig:.3e})");
            llr_lambda = Some(lam);
            println!("  LLR penalty frozen (objective units): λ = κσ̂(z0) λ_max(N) = {kappa} × {sig:.4e} × {lmax:.4e} = {lam:.4e}; every prox thresholds at λ·s");
        }
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
                    let seeds: Vec<u64> = if red { (0..red_shifts as u64).collect() } else if let Some(sn) = llr_shifts { (0..sn as u64).collect() } else { vec![k as u64] };
                    let step_now = hasty_recon::p2np::prox_step().unwrap_or(1.0);
                    assert!(!llr_freeze || llr_lambda.is_some(), "--llr-thr freeze: λ not set");
                    let thr_now: Option<f64> = llr_lambda.map(|l| l * step_now);
                    let mut one = |seed: u64, v: &mut Vec<C32>| {
                        if let Some(g) = llr_gpu.as_mut() {
                            g.set_iteration(seed);
                            g.llr.thr_abs = thr_now;
                            g.apply(v, nimg).unwrap();
                            if k < 2 || k % 10 == 0 {
                                eprintln!("    LLR on GPU {:?}: {} blocks ({} left unchanged), k ≤ {}, pass 1 {:.1}s, pass 2 {:.1}s, threshold {:.4e}", g.devices(), g.last.blocks, g.last.skipped, g.last.k_max, g.last.secs[0], g.last.secs[1], g.llr.last.thr);
                            }
                        } else {
                            llr.set_iteration(seed);
                            llr.thr_abs = thr_now;
                            llr.apply(v, nimg);
                        }
                    };
                    // --wavelet-mix β without --learned: D(x) = (1 − β) LLR(x) + β W(x) (both on the same x)
                    let wv0: Option<Vec<C32>> = (wav_mix > 0.0).then(|| {
                        wav.set_iteration(k as u64);
                        let mut w = h.clone();
                        wav.apply(&mut w, nimg);
                        w
                    });
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
                    if let Some(w) = &wv0 {
                        use rayon::prelude::*;
                        let (bl, bw) = ((1.0 - wav_mix) as f32, wav_mix as f32);
                        h.par_iter_mut().zip(w.par_iter()).for_each(|(x, d)| *x = C32::new(bl * x.re + bw * d.re, bl * x.im + bw * d.im));
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
            (false, "llr", true) => format!("_llrwav{wav_mix}"),
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
        let tag = format!("{model}{}_{pc}{ptag}_k{kappa}{}{dfw_tag}{lock_tag}{air_tag}{red_tag}{admm_tag}{ts_tag}{ga_tag}", temporal.map_or(String::new(), |b| format!("_t{b}")), llr_rank.map_or(String::new(), |r| format!("_r{r}")) + if llr_hard { "_h" } else { "" });
        // --init / --anchor variants (default tags unchanged)
        let meanres = hasty_recon::meanres::MeanRes::parse(init_spec, resid_steps);
        let tag = tag + &match (init_spec.as_str(), &meanres) {
            ("warm", _) => String::new(),
            (_, Some(m)) => format!("_{}", m.tag()),
            (s, None) => format!("_{s}"),
        } + anc.map_or("", |a| a.3.as_str());
        // (host-resident state: raw checkpoints state_<tag>.hsk)
        let resuming = out.join(format!("state_{tag}.hv")).exists() || out.join(format!("state_{tag}.hsk")).exists();
        if !resuming && hasty_value::store::exists_any(&out.join(format!("recon_{tag}.npy"))) {
            println!("  {tag}: finished in an earlier run (recon_{tag}), skipped (--fresh to redo)");
            continue;
        }
        if !resuming && init_spec == "zero" {
            if host_state {
                xh.data.iter_mut().for_each(|v| *v = C32::new(0.0, 0.0));
            } else {
                for v in x.v.iter_mut() {
                    st.memset_zeros(v).unwrap();
                }
            }
            println!("  {tag}: started from zero");
        }
        if let (Some(m), false) = (&meanres, resuming) {
            let ti = Instant::now();
            let gof: Vec<usize> = (0..fe.len()).map(enc_of).collect();
            // the previous run's start is reused when only the anchor differs (variants are init-major: one stack kept)
            if meanres_last.as_ref().is_none_or(|(s, _)| s != &m.tag()) {
                meanres_last = None;
                let (xs, oc) = hasty_recon::meanres::meanres_init(&mut p, &st, &gof, &warm, *m, need_circ, Some((&ctl, &format!("meanres_{}", m.tag())))).unwrap();
                if let Outcome::Stopped(k) = oc {
                    println!("stopped in the meanres initialiser of {tag} (step {k}): state saved, rerun the same command to resume");
                    std::process::exit(0);
                }
                meanres_last = Some((m.tag(), xs));
            } else {
                println!("  {tag}: meanres start {} reused from the previous run", m.tag());
            }
            let xs = &meanres_last.as_ref().unwrap().1;
            for t in 0..xs.nsets() {
                if host_state {
                    xh.chunk_mut(t).copy_from_slice(xs.chunk(t));
                } else {
                    st.memcpy_htod(xs.chunk(t), &mut x.v[t]).unwrap();
                }
            }
            println!("  {tag}: started from mean + residual deconvolution ({} step(s), {:?}) ({:.1}s)", m.steps, m.inner, ti.elapsed().as_secs_f64());
        }
        if let (Some(ini), false, 0, false) = (&init, resuming, warm_full, init_spec == "zero") {
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
        if resuming && static_lock.is_some() && hasty_value::store::exists_any(&map_npy) && map_json.exists() {
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
            if resuming && hasty_value::store::exists_any(&air_npy) && air_json.exists() {
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
                save_exact(&air_npy, "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 4], f4(m.stacked()));
                let mut j = hasty_recon::airmap::sidecar(&m, ao, sd.dims, vox_mm);
                j["seed"] = serde_json::json!(true);
                j["init"] = serde_json::json!(p);
                std::fs::write(&air_json, serde_json::to_string_pretty(&j).unwrap()).unwrap();
                amap = Some(m);
                air_seed_pending = true;
            }
        }
        // oracle diagnostic of the air map (scoring only, never used by the method): the true PD (water + fat, box mean
        // at the voxel's true position, reference pose) relative to its median over the body; a zeroed voxel above 5 %
        // of that median holds tissue
        let truth_pd_rel: Option<Vec<f32>> = air.as_ref().and_then(|_| {
            let pw = hasty_recon::data::read_npy(&std::path::Path::new(&dir).join("truth_pd_water.npy")).ok()?.f32s();
            let pf = hasty_recon::data::read_npy(&std::path::Path::new(&dir).join("truth_pd_fat.npy")).ok()?.f32s();
            let pd: Vec<f32> = pw.iter().zip(&pf).map(|(a, b)| if (a + b).is_finite() { a + b } else { 0.0 }).collect();
            use rayon::prelude::*;
            let g: Vec<f32> = (0..nv).into_par_iter().map(|q| sd.sample_log_box(&pd, 0, &grid.x_log[q]) as f32).collect();
            let mut body: Vec<f32> = g.iter().copied().filter(|&v| v > 0.0).collect();
            body.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = body.get(body.len() / 2).copied().unwrap_or(1.0).max(1e-30);
            Some(g.iter().map(|&v| v / med).collect())
        });
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
                    // --air start: on the start stack x0 (shared by every solver), not on this call's input
                    let m = match (air_start, h0.as_ref()) {
                        (true, Some(x0)) => hasty_recon::airmap::estimate(x0, &lay, ao, amap.as_ref(), k),
                        _ => hasty_recon::airmap::estimate(h, &lay, ao, amap.as_ref(), k),
                    };
                    let in_dil = (0..nv).filter(|&q| dil[q] && m.w[q] >= 0.5).count();
                    let lz = lumen.iter().filter(|&&q| m.w[q] > 0.0).count();
                    if let Some(tp) = &truth_pd_rel {
                        let zeroed: Vec<usize> = (0..nv).filter(|&q| m.w[q] >= 0.5).collect();
                        let tissue = zeroed.iter().filter(|&&q| tp[q] > 0.05).count();
                        let mx = zeroed.iter().map(|&q| tp[q]).fold(0.0f32, f32::max);
                        println!("    air map oracle (scoring only): {tissue} of {} zeroed voxels hold true tissue (PD > 5 % of the body median; max {mx:.3} of the median)", zeroed.len());
                    }
                    save_exact(&air_npy, "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 4], f4(m.stacked()));
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
                        save_exact(&map_npy, "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 6], f4(m.stacked()));
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
        // `ximgs` (--prior-mode admm): the data-step iterate x, scored next to the reported z (held-out residual, pc score)
        let mut report_imgs = |k: usize, imgs: &[&[C32]], f: f64, ximgs: Option<&[&[C32]]>| {
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
                hasty_value::store::save_bytes(&out.join(format!("it{k}_{tag}_lumen.npy")), hasty_value::DType::C64, &[li.len(), ne, nf], hasty_value::Order::C, c32_bytes(&v), &hasty_value::Save::image()).unwrap();
            }
            let img = |fr: usize, e: usize, q: usize| -> Complex64 {
                if let Some(w) = &tbox {
                    let (t, _) = at(fr, e);
                    return (0..nf).filter(|&k| w[fr * nf + k] != 0.0).map(|k| { let c = imgs[t][k * nv + q]; Complex64::new(c.re as f64, c.im as f64) * w[fr * nf + k] }).sum();
                }
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
            if report_detail && nf > 1 {
                // split of the lumen velocity error: time mean v̄ − t̄ and temporal detail (v_f − v̄) − (t_f − t̄)
                let (mut em, mut ed, mut td2, mut nm) = (0.0f64, 0.0f64, 0.0f64, 0usize);
                let mut ev: Vec<f64> = Vec::with_capacity(lumen.len() * nf);
                for &q in &lumen {
                    let vs: Vec<[f64; 3]> = (0..nf).map(|fr| vel(fr, q)).collect();
                    let ts: Vec<[f64; 3]> = (0..nf).map(|fr| [0, 1, 2].map(|a| sd.sample_log(&sd.truth_vel, a + 3 * (fr % nf_card), &grid.x_log[q]))).collect();
                    ev.extend((0..nf).map(|fr| (0..3).map(|a| (vs[fr][a] - ts[fr][a]).powi(2)).sum::<f64>().sqrt()));
                    for a in 0..3 {
                        let vm = vs.iter().map(|v| v[a]).sum::<f64>() / nf as f64;
                        let tm = ts.iter().map(|v| v[a]).sum::<f64>() / nf as f64;
                        em += (vm - tm).powi(2);
                        for fr in 0..nf {
                            ed += ((vs[fr][a] - vm) - (ts[fr][a] - tm)).powi(2);
                            td2 += (ts[fr][a] - tm).powi(2);
                        }
                    }
                    nm += 1;
                }
                let nm = nm.max(1) as f64;
                ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let pct = |p: f64| ev.get(((ev.len() as f64 - 1.0) * p).round() as usize).copied().unwrap_or(f64::NAN);
                println!("    velocity error split: time-mean RMSE {:.3} m/s, temporal-detail RMSE {:.3} m/s (truth detail RMS {:.3} m/s, relative {:.3}); |v − t| median {:.3}, p90 {:.3} m/s", (em / nm).sqrt(), (ed / (nm * nf as f64)).sqrt(), (td2 / (nm * nf as f64)).sqrt(), (ed / td2.max(1e-300)).sqrt(), pct(0.5), pct(0.9));
            }
            if dec_cmp {
                println!("    hierarchical decoder (±{r_first:.2} m/s): velocity RMSE {:.3} m/s (rel {:.3})", (e2h / n as f64).sqrt(), (e2h / t2).sqrt());
            }
            if let Some(pt) = &pc_truth {
                // the validated per-frame score (hasty_recon::pcscore): PC signal truth on this grid, blood voxels only
                let vg = gnl && !synth_pc;
                let vpc = |d: &FlowDecoder, fr: usize, i: usize| -> [f64; 3] {
                    let q = pt.vox[i];
                    let z: Vec<Complex64> = (0..ne).map(|e| img(fr, e, q)).collect();
                    let mut v = d.velocity(&z);
                    if vg {
                        v = mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &mat_vec(&sd.rot, &v)));
                    }
                    v
                };
                let s = pt.score(nf, &|fr, i| vpc(&dec, fr, i), &|fr, e, i| img(fr, e, pt.vox[i]));
                println!("    pc score ({} voxels): {}; frames {}", pt.n(), s.line(), s.per_frame.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" "));
                let sh = dec_cmp.then(|| pt.score(nf, &|fr, i| vpc(&dec_h, fr, i), &|fr, e, i| img(fr, e, pt.vox[i])));
                if let Some(sh) = &sh {
                    println!("    pc score, hierarchical decoder: {}", sh.line());
                }
                use std::io::Write;
                let p = out.join(format!("pcscore_{tag}.csv"));
                let new = !p.exists();
                let mut fcsv = std::fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
                if new {
                    writeln!(fcsv, "it,f,rmse,rel,mean,detail,median,p90,wrap,phase,rmse_hier,median_hier,p90_hier").unwrap();
                }
                if k == iters {
                    // per voxel-frame diagnostics for offline analysis (small): |Δv| (decoder), |Δv| (hierarchical),
                    // |x_0| of the recon, |C_0| of the truth; pc_vox.npy = the scored voxels (flat Fortran, this grid)
                    let n = pt.n();
                    let mut d = vec![0f32; n * nf * 4];
                    for i in 0..n {
                        for fr in 0..nf {
                            let t = pt.v[i * pt.nf + fr % pt.nf];
                            let a = vpc(&dec, fr, i);
                            let b = vpc(&dec_h, fr, i);
                            let o = (i * nf + fr) * 4;
                            d[o] = (0..3).map(|c| (a[c] - t[c]).powi(2)).sum::<f64>().sqrt() as f32;
                            d[o + 1] = (0..3).map(|c| (b[c] - t[c]).powi(2)).sum::<f64>().sqrt() as f32;
                            d[o + 2] = img(fr, 0, pt.vox[i]).norm() as f32;
                            d[o + 3] = pt.c_at(i, 0, fr % pt.nf).norm() as f32;
                        }
                    }
                    save(&out.join(format!("pcerr_{tag}.npy")), "<f4", &[4, nf, n], f4(d.iter().copied())); // numpy (4, nf, n) F = this (n, nf, 4) C layout
                    save(&out.join("pc_vox.npy"), "<f4", &[n], f4(pt.vox.iter().map(|&q| q as f32)));
                }
                let shv = sh.unwrap_or_default();
                writeln!(fcsv, "{k},{f:.6e},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}", s.rmse, s.rel, s.mean_rmse, s.detail_rmse, s.median, s.p90, s.wrap, s.phase_rms, shv.rmse, shv.median, shv.p90).unwrap();
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
                if !no_vel_save {
                    save_parts(&out.join(format!("vel_{tag}.npy")), "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 3, nf], std::iter::once(f32_bytes(&vout)));
                }
                if let Some(w) = &tbox {
                    // --temporal: the frame images are the knot model's box averages (as the scores read them), not
                    // the knot coefficients (those stay in the solver state / checkpoint)
                    let mut fx = vec![C32::new(0.0, 0.0); nv * ne * nf];
                    for fr in 0..nf {
                        for e in 0..ne {
                            let (t, _) = at(fr, e);
                            let dst = &mut fx[nv * (e + ne * fr)..nv * (e + ne * fr + 1)];
                            for k in (0..nf).filter(|&k| w[fr * nf + k] != 0.0) {
                                let wk = w[fr * nf + k] as f32;
                                for (d, c) in dst.iter_mut().zip(&imgs[t][k * nv..(k + 1) * nv]) {
                                    d.re += wk * c.re;
                                    d.im += wk * c.im;
                                }
                            }
                        }
                    }
                    save_parts(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], ne, nf], std::iter::once(c32_bytes(&fx)));
                } else {
                    save_parts(&out.join(format!("recon_{tag}.npy")), "<c8", &[sd.dims[0], sd.dims[1], sd.dims[2], ne, nf], (0..nf).flat_map(|fr| (0..ne).map(move |e| (fr, e))).map(|(fr, e)| {
                        let (t, o) = at(fr, e);
                        c32_bytes(&imgs[t][o..o + nv])
                    }));
                }
            }
            if let Some(h) = held_eval.as_mut() {
                // --holdout-res: wall time of the iterate (before this evaluation), then the held-out residual
                let (secs, th) = (tr.elapsed().as_secs_f64(), Instant::now());
                let rel = h.rel_residual(&st, imgs);
                println!("    held-out residual ({} readouts, {} sets): ‖E_h x − y_h‖ / ‖y_h‖ = {rel:.6} at it {k}, {secs:.1}s ({:.1}s)", h.nro, h.ys.len(), th.elapsed().as_secs_f64());
                use std::io::Write;
                let p = out.join(format!("holdout_{tag}.csv"));
                let new = !p.exists();
                let mut hcsv = std::fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
                if new {
                    writeln!(hcsv, "it,rel,seconds").unwrap();
                }
                writeln!(hcsv, "{k},{rel:.8},{secs:.2}").unwrap();
            }
            // --prior-mode admm: the data-step iterate x (not before the first iteration: x = z = the start)
            if let (Some(xi), true) = (ximgs, k > 0) {
                if let Some(h) = held_eval.as_mut() {
                    let th = Instant::now();
                    let rel = h.rel_residual(&st, xi);
                    println!("    held-out residual of x (data step): ‖E_h x − y_h‖ / ‖y_h‖ = {rel:.6} at it {k} ({:.1}s)", th.elapsed().as_secs_f64());
                    use std::io::Write;
                    let p = out.join(format!("holdout_x_{tag}.csv"));
                    let new = !p.exists();
                    let mut hcsv = std::fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
                    if new {
                        writeln!(hcsv, "it,rel").unwrap();
                    }
                    writeln!(hcsv, "{k},{rel:.8}").unwrap();
                }
                if let Some(pt) = &pc_truth {
                    let vg = gnl && !synth_pc;
                    let imgx = |fr: usize, e: usize, q: usize| -> Complex64 {
                        if let Some(w) = &tbox {
                            let (t, _) = at(fr, e);
                            return (0..nf).filter(|&k| w[fr * nf + k] != 0.0).map(|k| { let c = xi[t][k * nv + q]; Complex64::new(c.re as f64, c.im as f64) * w[fr * nf + k] }).sum();
                        }
                        let (t, o) = at(fr, e);
                        let c = xi[t][o + q];
                        Complex64::new(c.re as f64, c.im as f64)
                    };
                    let vpcx = |fr: usize, i: usize| -> [f64; 3] {
                        let q = pt.vox[i];
                        let z: Vec<Complex64> = (0..ne).map(|e| imgx(fr, e, q)).collect();
                        let mut v = dec.velocity(&z);
                        if vg {
                            v = mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &mat_vec(&sd.rot, &v)));
                        }
                        v
                    };
                    let s = pt.score(nf, &vpcx, &|fr, e, i| imgx(fr, e, pt.vox[i]));
                    println!("    pc score of x ({} voxels): {}", pt.n(), s.line());
                    use std::io::Write;
                    let p = out.join(format!("pcscore_x_{tag}.csv"));
                    let new = !p.exists();
                    let mut fcsv = std::fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
                    if new {
                        writeln!(fcsv, "it,rmse,rel,mean,detail,median,p90,wrap,phase").unwrap();
                    }
                    writeln!(fcsv, "{k},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}", s.rmse, s.rel, s.mean_rmse, s.detail_rmse, s.median, s.p90, s.wrap, s.phase_rms).unwrap();
                }
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
        let outcome = if admm {
            // --prior-mode admm (hasty_recon::admm): x-step per set on the device, z = D(x + u) with this run's denoiser
            assert!(red_lambda.is_none() && anc.is_none() && tsub_basis.is_none(), "--prior-mode admm: no --anchor / --tsub");
            use hasty_recon::admm::{run_admm, AdmmCfg, AdmmIter, AdmmP2np};
            use hasty_recon::hoststack::HostStack;
            // start: z = x (the warm start), u = 0 (a resumed run loads x, z, u, ρ)
            let mut zh = xh.clone();
            let mut uh = HostStack::zeros(&lens);
            let mut cfg = AdmmCfg::new(iters, admm_rho, admm_rho1, admm_inner);
            if admm_adapt {
                cfg.adapt = Some((2.0, 10.0));
            }
            println!("  {tag}: ADMM, ρ = {admm_rho} ρ₁ = {:.5e}{}, {admm_inner} PCG iterations per set on N_t + ρI (inner metric {}), z = D(x + u) with the P²nP prior pipeline; host stacks x, z, u + one work stack ({:.2} GiB)", cfg.rho, if admm_adapt { " (residual balancing ×2 by the 10× rule)" } else { "" }, admm_metric.name(), 4.0 * stack_bytes as f64 / GIB);
            let mut data = AdmmP2np { p: &mut p, st: st.clone(), metric: admm_clip.take() };
            let mut rep = |k: usize, zs: &HostStack, xs: &HostStack, info: Option<&AdmmIter>| {
                let f = info.map_or(f64::NAN, |i| i.x.f);
                if let Some(i) = info {
                    println!(
                        "    admm it {k}: time inner PCG {:.1}s, prior {:.1}s, other {:.1}s; inner ‖r_K‖/‖r₀‖ {:.3e} (per set median {:.3e}, max {:.3e}{}){}; primal ‖x − z‖ {:.4e} (rel {:.3e}), dual ρ‖z − z_prev‖ {:.4e} (rel {:.3e}; ρ₁ units {:.4e}); ρ {:.4} ρ₁{}; f(x) {:.5e} (step start {:.5e})",
                        i.t_inner,
                        i.t_prior,
                        i.t_other,
                        i.x.rk / i.x.r0.max(1e-300),
                        i.x.median,
                        i.x.worst,
                        if i.x.early > 0 { format!(", {} sets stopped early", i.x.early) } else { String::new() },
                        if i.x.metric_updates > 0 { format!(", clip symbol {} uploads {:.1}s", i.x.metric_updates, i.x.metric_secs) } else { String::new() },
                        i.primal,
                        i.primal_rel,
                        i.dual,
                        i.dual_rel,
                        i.dual_nat,
                        i.rho / admm_rho1,
                        if i.rho_next != i.rho { format!(" → {:.4} ρ₁", i.rho_next / admm_rho1) } else { String::new() },
                        i.x.f,
                        i.x.f0
                    );
                    // host memory per outer iteration (memory ledger): RSS now, peak so far, RSS after returning free
                    // heap to the OS (the difference is glibc arena fragmentation, not live data)
                    let gib = |b: usize| b as f64 / 1073741824.0;
                    let peak = hasty_recon::sysmem::peak_rss_bytes();
                    let (b, a) = hasty_recon::sysmem::trim();
                    println!("    admm it {k}: host RSS {:.2} GiB (peak {:.2} GiB), {:.2} GiB after malloc_trim", gib(b), gib(peak), gib(a));
                }
                if !report_due(k) {
                    return;
                }
                let t_r = Instant::now();
                let zi: Vec<&[C32]> = (0..zs.nsets()).map(|t| zs.chunk(t)).collect();
                let xi: Vec<&[C32]> = (0..xs.nsets()).map(|t| xs.chunk(t)).collect();
                report_imgs(k, &zi, f, Some(&xi));
                if info.is_some() {
                    println!("    admm it {k}: report {:.1}s", t_r.elapsed().as_secs_f64());
                }
            };
            let (oc, rho_end) = run_admm(&mut data, &mut xh, &mut zh, &mut uh, &cfg, &mut denoise, &mut rep, Some((&ctl, &tag))).unwrap();
            admm_clip = data.metric.take();
            if admm_adapt {
                println!("  {tag}: ρ at the end {:.4} ρ₁", rho_end / admm_rho1);
            }
            oc
        } else if host_state {
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
            // --anchor: the temporal anchor around the data term (and RED)
            let mut anc_store;
            let obj: &mut dyn hasty_recon::p2np::Objective = match anc {
                Some((gof, tg, mu, _)) => {
                    assert!(tsub_basis.is_none(), "--anchor with --tsub: not yet");
                    anc_store = hasty_recon::tanchor::TAnchor::new(obj, gof.clone(), tg.clone(), *mu);
                    &mut anc_store
                }
                None => obj,
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
                    report_imgs(k, &r, f, None)
                };
                hasty_recon::hoststack::run_objective_host(&mut sub, &st, &mut ch, precond, alpha_c, iters, &mut den_c, &mut rep_c, Some((&ctl, &tag))).unwrap()
            } else {
                let mut report = |k: usize, xs: &hasty_recon::hoststack::HostStack, f: f64| {
                    let r: Vec<&[C32]> = (0..xs.nsets()).map(|t| xs.chunk(t)).collect();
                    report_imgs(k, &r, f, None)
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
                report_imgs(k, &r, f, None)
            };
            if let (Some(l), Some((gof, tg, mu, _))) = (red_lambda, anc) {
                // RED inside the anchor
                let mut ro = hasty_recon::red::Red::new(&mut p, &mut denoise, l);
                ro.groups = red_groups.clone();
                let mut ao = hasty_recon::tanchor::TAnchor::new(&mut ro, gof.clone(), tg.clone(), *mu);
                hasty_recon::p2np::run_objective(&mut ao, &st, &mut x, precond, alpha, iters, &mut |_, _| {}, &mut report, Some((&ctl, &tag))).unwrap()
            } else if let Some((gof, tg, mu, _)) = anc {
                // the anchor around the data term (no lagged gradients: asserted at the anchor setup)
                let mut ao = hasty_recon::tanchor::TAnchor::new(&mut p, gof.clone(), tg.clone(), *mu);
                hasty_recon::p2np::run_objective(&mut ao, &st, &mut x, precond, alpha, iters, &mut denoise, &mut report, Some((&ctl, &tag))).unwrap()
            } else if let Some(l) = red_lambda {
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

/// --prior-mode admm: the clip metric's per-encode samples-per-cell symbols, set → (encode, share), the coil energy
/// D = Σ_c |S_c|² and its mean, and the ρ₁ check set with its own symbol and cells (index, samples, |k| / ball radius)
struct AdmmSym {
    mu: Vec<Vec<f32>>,
    of_set: Vec<(usize, f64)>,
    dsum: Vec<f32>,
    dbar: f64,
    tc: usize,
    mu_t: Vec<f32>,
    cells: Vec<(usize, u32, f64)>,
}

/// Groups of a temporal-pyramid level with `rg` respiratory × `cg` cardiac groups (clamped to the bins / phases):
/// set t of (frame f = r · nf_card + c, encode e) → (e · rg + ⌊r · rg / resp_bins⌋) · cg + ⌊c · cg / nf_card⌋
/// (contiguous bins and phases, group sizes differing by at most one). Returns (groups, group of every set).
/// Union of set partitions: sets sharing a key in any of `keys` (one key per set each) end in one group; group ids
/// 0.. in order of first appearance
/// --holdout-res: the held-out sets on the host before their tables are built (whole records, kept-sample selection,
/// kept data: whitened, coil-compressed, coil-major)
struct HeldHost {
    recs: Vec<SampleSet>,
    sels: Vec<Option<Vec<u32>>>,
    ys: Vec<Vec<C32>>,
    nro: usize,
}

/// --holdout-res: the held-out operator (second engine, the solve's options, maps, segments) and data (host, uploaded
/// set by set: one set's image, data and residual on the device at a time)
struct HeldOut {
    sets: SegSets,
    ys: Vec<Vec<C32>>,
    ny2: f64,
    nro: usize,
    /// --smaps-resp: per-set coil maps (device; empty = the engine's own maps for every set)
    maps: Vec<cudarc::driver::CudaSlice<C32>>,
    map_of_set: Vec<Option<usize>>,
    /// --ratemap truth-bc in frame mode: the solve's per-frame segment tables, loaded into slot 0 per set
    seg_tabs: Option<std::sync::Arc<Vec<hasty_recon::setop::HostSegTab>>>,
    seg_tab_of_set: Vec<usize>,
    seg_cur: Option<usize>,
}

impl HeldOut {
    /// ‖E_h x − y_h‖ / ‖y_h‖ over all held-out sets, x = the per-set images of the solve (frame mode: set t = its image)
    fn rel_residual(&mut self, st: &std::sync::Arc<cudarc::driver::CudaStream>, imgs: &[&[C32]]) -> f64 {
        assert_eq!(imgs.len(), self.ys.len(), "--holdout-res: one image per held-out set");
        let mut r2 = 0.0f64;
        for t in 0..self.ys.len() {
            if self.ys[t].is_empty() {
                continue;
            }
            let x = st.clone_htod(imgs[t]).unwrap();
            let y = st.clone_htod(&self.ys[t]).unwrap();
            let mut r = st.alloc_zeros::<C32>(self.ys[t].len()).unwrap();
            // the set's own maps swapped in for this set only (a pointer exchange)
            let mi = self.map_of_set.get(t).copied().flatten().filter(|_| !self.maps.is_empty());
            if let Some(i) = mi {
                self.sets.eng.swap_smaps(&mut self.maps[i]);
            }
            // the set's frame table (per-frame ratemap), uploaded when the frame changes
            if let Some(tabs) = &self.seg_tabs {
                let k = self.seg_tab_of_set[t];
                if self.seg_cur != Some(k) {
                    assert!(self.sets.load_segs(st, 0, &tabs[k]).unwrap(), "--holdout-res: per-frame segment table load refused");
                    self.seg_cur = Some(k);
                }
            }
            self.sets.forward_residual(st, t, &x, &y, &mut r).unwrap();
            if let Some(i) = mi {
                self.sets.eng.swap_smaps(&mut self.maps[i]);
            }
            r2 += self.sets.kern.dot(st, &r, &r).unwrap().re;
        }
        (r2 / self.ny2.max(1e-300)).sqrt()
    }
}

/// coil-map agreement with reference maps over `region` (coil-major nc × nv both): (mean of 1 − |⟨ŝ, s⟩| / (‖ŝ‖ ‖s‖)
/// with zero estimates counted as 1, the mean over the nonzero estimates, the number of zero estimates)
fn smaps_agreement(est: &[C32], tru: &[C32], nc: usize, region: &[bool]) -> (f64, f64, usize) {
    let nv = region.len();
    let d = hasty_recon::smapfit::map_disagreement(est, tru, nc);
    let (mut all, mut nz, mut n, mut zero) = (0.0f64, 0.0f64, 0usize, 0usize);
    for q in (0..nv).filter(|&q| region[q]) {
        n += 1;
        all += d[q];
        if (0..nc).all(|c| est[c * nv + q] == C32::new(0.0, 0.0)) {
            zero += 1;
        } else {
            nz += d[q];
        }
    }
    (all / n.max(1) as f64, nz / (n - zero).max(1) as f64, zero)
}

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

/// f32 values packed in pairs into a complex chunk of `len` elements (the velocity-model chunk layout)
fn pack_f32(v: &[f32], len: usize) -> Vec<C32> {
    (0..len).map(|i| C32::new(*v.get(2 * i).unwrap_or(&0.0), *v.get(2 * i + 1).unwrap_or(&0.0))).collect()
}

/// (m, v) of every (knot, voxel) of the subsample `sub` (knot-major) from the host form of a spline-mode state
fn vk_sample(h: &[C32], sub: &[usize], nk: usize, nv: usize, voff: &[usize]) -> Vec<(f32, [f32; 3])> {
    let pf = |off: usize, i: usize| { let c = h[off + i / 2]; if i % 2 == 0 { c.re } else { c.im } };
    (0..nk).flat_map(|kk| sub.iter().map(move |&q| (h[kk * nv + q].re, [0, 1, 2].map(|a| pf(voff[kk], a * nv + q))))).collect()
}

/// largest encode phase |k'_e · d| (rad) of a velocity change d
fn kphase(km: &[[f64; 3]], d: [f32; 3]) -> f64 {
    km.iter().map(|k| (k[0] * d[0] as f64 + k[1] * d[1] as f64 + k[2] * d[2] as f64).abs()).fold(0.0, f64::max)
}

/// one line of step diagnostics (research/velocity_model_sr.md §F.2): the gradient step's and the prior's encode phase
/// per (voxel, knot), the velocity box of the decoder design, the phase per operator frame between cardiac knots
#[allow(clippy::too_many_arguments)]
fn vk_diag(prev: Option<&[(f32, [f32; 3])]>, si: &[(f32, [f32; 3])], so: &[(f32, [f32; 3])], km: &[[f64; 3]], vbox: f64, fpk: f64, ns: usize, mc: usize, it: usize) -> String {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let mut mm: Vec<f32> = so.iter().map(|p| p.0.abs()).collect();
    mm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mmed = mm[mm.len() / 2].max(1e-30) as f64;
    let n = so.len() as f64;
    let pct = |v: &mut Vec<f64>, p: f64| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v.get(((v.len() as f64 - 1.0) * p) as usize).copied().unwrap_or(0.0) };
    let mut out = format!("step diag (it {it}):");
    if let Some(pv) = prev {
        let ph: Vec<f64> = si.iter().zip(pv).map(|(a, b)| kphase(km, sub(a.1, b.1))).collect();
        let big: Vec<f64> = ph.iter().zip(si).filter(|p| *p.0 > std::f64::consts::PI).map(|p| p.1 .0.abs() as f64 / mmed).collect();
        let mut bm = big.clone();
        let mut phs = ph.clone();
        out += &format!(" gradient step |k'Δv| p99 {:.2} max {:.2} rad, > 1 rad {:.2} %, > π {:.2} % (median m there {:.2} × median)", pct(&mut phs.clone(), 0.99), pct(&mut phs, 1.0), 100.0 * ph.iter().filter(|&&x| x > 1.0).count() as f64 / n, 100.0 * big.len() as f64 / n, pct(&mut bm, 0.5));
    }
    let mut pp: Vec<f64> = so.iter().zip(si).map(|(a, b)| kphase(km, sub(a.1, b.1))).collect();
    let pbig = pp.iter().filter(|&&x| x > std::f64::consts::PI).count() as f64 / n;
    out += &format!("; prior/lock |k'Δv| p99 {:.2} max {:.2}, > π {:.2} %", pct(&mut pp.clone(), 0.99), pct(&mut pp, 1.0), 100.0 * pbig);
    let ob = so.iter().filter(|p| p.1.iter().any(|x| (x.abs() as f64) > vbox)).count() as f64 / n;
    let nkr = so.len() / ns / mc;
    let mut jp: Vec<f64> = Vec::with_capacity(so.len());
    for r in 0..nkr {
        for c in 0..mc {
            let (k0, k1) = (r * mc + c, r * mc + (c + 1) % mc);
            for i in 0..ns {
                jp.push(kphase(km, sub(so[k1 * ns + i].1, so[k0 * ns + i].1)) / fpk);
            }
        }
    }
    out += &format!("; |v| > {vbox:.1} m/s {:.2} %; phase per operator frame p99.9 {:.2} max {:.2} rad", 100.0 * ob, pct(&mut jp.clone(), 0.999), pct(&mut jp, 1.0));
    out
}

/// the first `n` f32 values of a packed chunk
fn unpack_f32(c: &[C32], n: usize) -> Vec<f32> {
    c.iter().flat_map(|q| [q.re, q.im]).take(n).collect()
}

/// Layout of the knot-mode state for the static-lock hook: h = [z_0 .. z_{K−1} (nv complex each), v_0 .. v_{K−1}
/// ([3][nv] f32 packed in pairs), residual phase], knot k = r · mc + c (respiratory r, cardiac c)
#[allow(dead_code)]
struct KnotLayout {
    dims: [usize; 3],
    vox: [f64; 3],
    kr: usize,
    mc: usize,
    nv: usize,
    ne: usize,
}

#[allow(dead_code)]
impl KnotLayout {
    fn nk(&self) -> usize {
        self.kr * self.mc
    }
    /// offset of z_k in h
    fn z_off(&self, k: usize) -> usize {
        k * self.nv
    }
    /// offset of v_k in h (packed f32 pairs, (3 nv).div_ceil(2) complex)
    fn v_off(&self, k: usize) -> usize {
        self.nk() * self.nv + k * (3 * self.nv).div_ceil(2)
    }
    /// h as the knots' z (nv complex each) and v (3 nv f32 each, component-major), in place
    fn split<'a>(&self, h: &'a mut [C32]) -> (Vec<&'a mut [C32]>, Vec<&'a mut [f32]>) {
        let (nk, nv) = (self.nk(), self.nv);
        let lv = (3 * nv).div_ceil(2);
        let (zh, rest) = h.split_at_mut(nk * nv);
        let zs: Vec<&mut [C32]> = zh.chunks_mut(nv).collect();
        let vs: Vec<&mut [f32]> = rest[..nk * lv]
            .chunks_mut(lv)
            // SAFETY: C32 is two f32 (repr C, no padding); the chunk's first 3 nv floats are v (component-major)
            .map(|c| unsafe { std::slice::from_raw_parts_mut(c.as_mut_ptr() as *mut f32, 3 * nv) })
            .collect();
        (zs, vs)
    }
}


/// The knot model at voxels `idx` for the frame weights `w` (per frame, K knot weights):
/// x_e(q, f) = Σ_k w_k(f) z_k(q) e^{i(φ_e(q) − k_e·v_k(q))}, as complex64 (n, E, F) C order (frame_fit lumen form)
fn knot_lumen(chunk: &dyn Fn(usize) -> Vec<C32>, spec: &hasty_recon::vmodel::VelSpec, a: &[f32], pex: &[[i32; 3]], idx: &[usize], w: &[Vec<f64>]) -> Vec<C32> {
    let (nv, ne, nk, nf) = (spec.nv(), spec.ne(), spec.kz, w.len());
    let n = idx.len();
    let np = pex.len();
    let zl: Vec<Vec<C32>> = (0..nk).map(|k| { let c = chunk(spec.chunk_z(k)); idx.iter().map(|&q| c[q]).collect() }).collect();
    let vl: Vec<Vec<[f32; 3]>> = (0..nk).map(|k| { let v = unpack_f32(&chunk(spec.chunk_v(k)), 3 * nv); idx.iter().map(|&q| [v[q], v[nv + q], v[2 * nv + q]]).collect() }).collect();
    let mut out = vec![C32::new(0.0, 0.0); n * ne * nf];
    for (i, &q) in idx.iter().enumerate() {
        for e in 0..ne {
            let phi: f64 = if e == 0 || np == 0 { 0.0 } else { (0..np).map(|p| a[(e - 1) * np + p] as f64 * hasty_recon::vmodel::poly_at(&pex[p], q, spec.dims)).sum() };
            let ke = spec.kenc[e];
            let xk: Vec<Complex64> = (0..nk).map(|k| { let z = zl[k][i]; let v = vl[k][i]; Complex64::new(z.re as f64, z.im as f64) * Complex64::from_polar(1.0, phi - (ke[0] * v[0] as f64 + ke[1] * v[1] as f64 + ke[2] * v[2] as f64)) }).collect();
            for f in 0..nf {
                let x: Complex64 = (0..nk).filter(|&k| w[f][k] != 0.0).map(|k| xk[k] * w[f][k]).sum();
                out[(i * ne + e) * nf + f] = C32::new(x.re as f32, x.im as f32);
            }
        }
    }
    out
}

/// --init-from a knot run (vknots_<tag>.json + vparams_<tag>.hv, newest; same knot counts and residual-phase degree):
/// its parameters resampled band-limited to this grid (z keeps its Fourier coefficients — the values scale with the
/// voxel volume as the other initial images; v is a physical velocity, rescaled back). None: no such run.
fn knot_init(p: &std::path::Path, spec: &hasty_recon::vmodel::VelSpec, kb: &hasty_recon::vknot::KnotBasis) -> Option<(PathBuf, ([usize; 3], Vec<Vec<C32>>))> {
    let dir = if p.is_dir() { p.to_path_buf() } else { p.parent()?.to_path_buf() };
    let mut metas: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&dir).ok()?.flatten().filter(|e| { let n = e.file_name().to_string_lossy().into_owned(); n.starts_with("vknots_") && n.ends_with(".json") }).filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path()))).collect();
    metas.sort();
    let (_, mp) = metas.pop()?;
    let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&mp).ok()?).ok()?;
    let [kr, mc] = spec.knots?;
    if m["kr"].as_u64()? as usize != kr || m["mc"].as_u64()? as usize != mc || m["rdeg"].as_u64()? as u32 != kb.rdeg || m["cdeg"].as_u64()? as u32 != kb.cdeg {
        println!("knot init: {} has other knots, not used", mp.display());
        return None;
    }
    let src: [usize; 3] = [0, 1, 2].map(|a| m["dims"][a].as_u64().unwrap() as usize);
    let pf = dir.join(m["params"].as_str()?);
    let v = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(&pf).ok()?)).ok()?;
    let hasty_value::Value::List(items) = v else { return None };
    let nvs: usize = src.iter().product();
    let nv = spec.nv();
    let ratio = nvs as f64 / nv as f64;
    let chunk = |i: usize| -> Vec<C32> {
        let t = items[i].as_tensor().unwrap();
        // SAFETY: complex64 tensor bytes
        unsafe { std::slice::from_raw_parts(t.bytes().as_ptr() as *const C32, t.numel()) }.to_vec()
    };
    let lens = spec.lens();
    let mut out: Vec<Vec<C32>> = lens.iter().map(|&l| vec![C32::new(0.0, 0.0); l]).collect();
    let nk = kr * mc;
    // spline runs store (m, ψ' = ψ0 − k̄·v): resampled as the physical complex z0 = m e^{iψ0} (band-limited like the
    // complex models' z) and v; a spline target takes (m, ψ0) back from z0 with ψ0 unwrapped along the knots
    let src_polar = m["param"].as_str() == Some("polar");
    let kbs: [f64; 3] = [0, 1, 2].map(|a| m["kbar"][a].as_f64().unwrap_or(0.0));
    for k in 0..nk {
        let zc = chunk(k);
        let z0: Vec<C32> = if src_polar {
            let vs = unpack_f32(&chunk(nk + k), 3 * nvs);
            zc.iter().enumerate().map(|(q, c)| { let ps = c.im as f64 + kbs[0] * vs[q] as f64 + kbs[1] * vs[nvs + q] as f64 + kbs[2] * vs[2 * nvs + q] as f64; let z = Complex64::from_polar(c.re as f64, ps); C32::new(z.re as f32, z.im as f32) }).collect()
        } else {
            zc
        };
        out[spec.chunk_z(k)] = hasty_recon::pyramid::resample_fourier(&z0, src, spec.dims);
        let vs = unpack_f32(&chunk(nk + k), 3 * nvs);
        let mut vd = vec![0f32; 3 * nv];
        for a in 0..3 {
            let c: Vec<C32> = vs[a * nvs..(a + 1) * nvs].iter().map(|&x| C32::new(x, 0.0)).collect();
            let r = hasty_recon::pyramid::resample_fourier(&c, src, spec.dims);
            for q in 0..nv {
                vd[a * nv + q] = (r[q].re as f64 / ratio) as f32;
            }
        }
        out[spec.chunk_v(k)] = pack_f32(&vd, lens[spec.chunk_v(k)]);
    }
    if spec.spline() {
        let kb_t = spec.kbar();
        let vk: Vec<Vec<f32>> = (0..nk).map(|k| unpack_f32(&out[spec.chunk_v(k)], 3 * nv)).collect();
        let zk: Vec<Vec<C32>> = (0..nk).map(|k| out[spec.chunk_z(k)].clone()).collect();
        for q in 0..nv {
            let z: Vec<Complex64> = (0..nk).map(|k| Complex64::new(zk[k][q].re as f64, zk[k][q].im as f64)).collect();
            for (k, (mm, ps)) in polar_of(&z).into_iter().enumerate() {
                let d = kb_t[0] * vk[k][q] as f64 + kb_t[1] * vk[k][nv + q] as f64 + kb_t[2] * vk[k][2 * nv + q] as f64;
                out[spec.chunk_z(k)][q] = C32::new(mm as f32, (ps - d) as f32);
            }
        }
    }
    if m["poly"] == serde_json::json!(spec.poly) && items.len() > 2 * nk {
        let a = chunk(2 * nk);
        let l = lens[spec.chunk_a()];
        out[spec.chunk_a()] = a.into_iter().chain(std::iter::repeat(C32::new(0.0, 0.0))).take(l).collect();
    }
    Some((pf, (src, out)))
}

// ─── parameter-spline velocity model helpers (research/velocity_model_sr.md §F) ─────────────────────────────────────

/// The spline model at voxels `idx`, breathing amplitude `amp` and the `nfr` cardiac frame centres (the model's own
/// signal: the respiratory rows mixed complex-linearly, each row m e^{i(ψ + φ_e − k'_e·v)} with (m, ψ, v) the cardiac
/// splines of the row), complex64 (n, E, F) C order (frame_fit lumen form)
#[allow(clippy::too_many_arguments)]
fn spline_lumen(chunk: &dyn Fn(usize) -> Vec<C32>, spec: &hasty_recon::vmodel::VelSpec, kb: &hasty_recon::vknot::KnotBasis, a: &[f32], pex: &[[i32; 3]], idx: &[usize], amp: f64, nfr: usize) -> Vec<C32> {
    use rayon::prelude::*;
    let (nv, ne, nk) = (spec.nv(), spec.ne(), spec.kz);
    let (kr, mc) = (kb.kr, kb.mc);
    let n = idx.len();
    let np = pex.len();
    let wr = kb.resp_weights(amp);
    let wc: Vec<Vec<f64>> = (0..nfr).map(|f| kb.cardiac_weights(kb.cardiac_t((f as f64 + 0.5) / nfr as f64))).collect();
    let rows: Vec<usize> = (0..kr).filter(|&r| wr[r] != 0.0).collect();
    let zl: Vec<Vec<C32>> = (0..nk).map(|k| if rows.contains(&(k / mc)) { let c = chunk(spec.chunk_z(k)); idx.iter().map(|&q| c[q]).collect() } else { vec![] }).collect();
    let vl: Vec<Vec<[f32; 3]>> = (0..nk).map(|k| if rows.contains(&(k / mc)) { let v = unpack_f32(&chunk(spec.chunk_v(k)), 3 * nv); idx.iter().map(|&q| [v[q], v[nv + q], v[2 * nv + q]]).collect() } else { vec![] }).collect();
    let kem = spec.kenc_model();
    let mut out = vec![C32::new(0.0, 0.0); n * ne * nfr];
    out.par_chunks_mut(ne * nfr).enumerate().for_each(|(i, o)| {
        let q = idx[i];
        let phi: Vec<f64> = (0..ne).map(|e| if e == 0 || np == 0 { 0.0 } else { (0..np).map(|p| a[(e - 1) * np + p] as f64 * hasty_recon::vmodel::poly_at(&pex[p], q, spec.dims)).sum() }).collect();
        for &r in &rows {
            for f in 0..nfr {
                let (mut m, mut ps, mut v) = (0.0f64, 0.0f64, [0.0f64; 3]);
                for c in 0..mc {
                    let w = wc[f][c];
                    if w != 0.0 {
                        let z = zl[r * mc + c][i];
                        let vv = vl[r * mc + c][i];
                        m += w * z.re as f64;
                        ps += w * z.im as f64;
                        for d in 0..3 {
                            v[d] += w * vv[d] as f64;
                        }
                    }
                }
                for e in 0..ne {
                    let k = kem[e];
                    let x = Complex64::from_polar(wr[r] * m, ps + phi[e] - (k[0] * v[0] + k[1] * v[1] + k[2] * v[2]));
                    let s = &mut o[e * nfr + f];
                    *s = C32::new(s.re + x.re as f32, s.im + x.im as f32);
                }
            }
        }
    });
    out
}

/// (m, ψ) knot values of complex zero-moment values (hasty_recon::vinit::polar_of)
fn polar_of(z: &[Complex64]) -> Vec<(f64, f64)> {
    hasty_recon::vinit::polar_of(z)
}

/// operator-frame approximation error of the spline model (research/velocity_model_sr.md §F) at the voxels `idx`: per
/// respiratory row and encode, the cubic (or linear) interpolation through the fop frame values against the model at 8
/// cardiac times per frame interval; (relative RMS over the signal, the 99.9th percentile of the per-sample relative error)
fn op_frame_error(chunk: &dyn Fn(usize) -> Vec<C32>, spec: &hasty_recon::vmodel::VelSpec, a: &[f32], pex: &[[i32; 3]], idx: &[usize]) -> (f64, f64) {
    use rayon::prelude::*;
    let of = spec.op.as_ref().unwrap();
    let (nv, ne, mc, fop) = (spec.nv(), spec.ne(), of.mc, of.fop);
    let np = pex.len();
    let kem = spec.kenc_model();
    let pre = of.prefilter();
    let nfine = 8 * fop;
    let fw: Vec<Vec<(usize, f64)>> = (0..fop).map(|f| of.frame_weights(f)).collect();
    let card = |t: f64| -> Vec<(usize, f64)> { (0..mc).map(|c| (c, hasty_recon::vknot::cardiac_weight(t, c, mc, of.cdeg))).filter(|p| p.1 != 0.0).collect() };
    let fine_w: Vec<Vec<(usize, f64)>> = (0..nfine).map(|j| card(((j as f64 + 0.5) / nfine as f64 * mc as f64 - 0.5).rem_euclid(mc as f64))).collect();
    let op_w: Vec<Vec<(usize, f64)>> = (0..nfine).map(|j| { let t = ((j as f64 + 0.5) / nfine as f64 * fop as f64 - 0.5).rem_euclid(fop as f64); (0..fop).map(|b| (b, hasty_recon::vknot::cardiac_weight(t, b, fop, of.odeg))).filter(|p| p.1 != 0.0).collect() }).collect();
    let zs: Vec<Vec<C32>> = (0..spec.kz).map(|k| { let c = chunk(spec.chunk_z(k)); idx.iter().map(|&q| c[q]).collect() }).collect();
    let vs: Vec<Vec<[f32; 3]>> = (0..spec.kz).map(|k| { let v = unpack_f32(&chunk(spec.chunk_v(k)), 3 * nv); idx.iter().map(|&q| [v[q], v[nv + q], v[2 * nv + q]]).collect() }).collect();
    let res: Vec<(f64, f64, Vec<f64>)> = (0..idx.len()).into_par_iter().map(|i| {
        let q = idx[i];
        let (mut num, mut den, mut rel) = (0.0, 0.0, vec![]);
        let eval = |r: usize, w: &[(usize, f64)], e: usize| -> Complex64 {
            let (mut m, mut ps, mut v) = (0.0, 0.0, [0.0f64; 3]);
            for &(c, x) in w {
                let z = zs[r * mc + c][i];
                let vv = vs[r * mc + c][i];
                m += x * z.re as f64;
                ps += x * z.im as f64;
                for d in 0..3 {
                    v[d] += x * vv[d] as f64;
                }
            }
            let phi = if e == 0 || np == 0 { 0.0 } else { (0..np).map(|p| a[(e - 1) * np + p] as f64 * hasty_recon::vmodel::poly_at(&pex[p], q, spec.dims)).sum::<f64>() };
            let k = kem[e];
            Complex64::from_polar(m, ps + phi - (k[0] * v[0] + k[1] * v[1] + k[2] * v[2]))
        };
        for r in 0..of.kr {
            for e in 0..ne {
                let xf: Vec<Complex64> = (0..fop).map(|f| eval(r, &fw[f], e)).collect();
                let cf: Vec<Complex64> = (0..fop).map(|f| (0..fop).map(|n| xf[(f + fop - n) % fop] * pre[n]).sum()).collect();
                let scale = xf.iter().map(|x| x.norm()).fold(0.0, f64::max).max(1e-30);
                for j in 0..nfine {
                    let xm = eval(r, &fine_w[j], e);
                    let xo: Complex64 = op_w[j].iter().map(|&(b, w)| cf[b] * w).sum();
                    num += (xo - xm).norm_sqr();
                    den += xm.norm_sqr();
                    rel.push((xo - xm).norm() / scale);
                }
            }
        }
        (num, den, rel)
    }).collect();
    let (num, den) = res.iter().fold((0.0, 0.0), |a, r| (a.0 + r.0, a.1 + r.1));
    let mut rel: Vec<f64> = res.into_iter().flat_map(|r| r.2).collect();
    rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ((num / den.max(1e-300)).sqrt(), rel.get((rel.len() as f64 * 0.999) as usize).copied().unwrap_or(0.0))
}

/// The true ratemap of one operator frame on the reconstruction grid (oracle): off-resonance ω (rad/s, the field the
/// signal sees, no chemical shift), water / fat decay rates (1/s) and the fat fraction.
struct RateFrame {
    /// off-resonance (rad/s), f64: the degenerate frame model (static field) is then the static path's field exactly
    om: Vec<f64>,
    rw: Vec<f32>,
    rf: Vec<f32>,
    ff: Vec<f32>,
}

/// `--ratemap truth-bc`: the simulator's per-state truth ratemap (scan/truth_ratemap_bc/, states s = q K + k on the
/// grid r_k × c_q, research/resp_field_sim.md) at each knot's state. A knot's state is the basis-weighted mean of the
/// true breathing r (resp.npy) and the true cardiac time (physio.npy ECG phase × period) over the used readouts (ORACLE
/// coordinates: evaluation / development of the flow sequence only). Bilinear in (r, c), periodic in c; NaN corners
/// (no source of that kind) are skipped, and voxels with none fall back to ω = the static field, rates 0, φ = 0.
fn truth_ratemap_knots(dir: &std::path::Path, sd: &SimData, x_log: &[[f64; 3]], boxed: bool, nk: usize, used: &[usize], w: &[Vec<(usize, f64)>]) -> Vec<RateFrame> {
    let td = dir.join("truth_ratemap_bc");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(td.join("meta.json")).unwrap_or_else(|e| panic!("--ratemap truth-bc: {}: {e}", td.display()))).unwrap();
    let stv = &meta["states"];
    let rk: Vec<f64> = stv["r"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let cq: Vec<f64> = stv["c_s"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let period = stv["period_s"].as_f64().unwrap();
    let (kk, qq) = (rk.len(), cq.len());
    let mask_a = hasty_recon::data::read_npy(&td.join("mask.npy")).unwrap();
    // resp_field.py saves the mask as bool, F order (x fastest): file bytes are already in our linear order
    assert!(mask_a.fortran || mask_a.shape.iter().filter(|&&n| n > 1).count() <= 1, "truth_ratemap_bc mask.npy: expected fortran_order (x fastest)");
    let mask: Vec<bool> = mask_a.bools();
    let mdims = [mask_a.shape[0], mask_a.shape[1], mask_a.shape[2]];
    let idx: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
    let load = |name: &str| -> Vec<f32> { hasty_recon::data::read_npy(&td.join(name)).unwrap_or_else(|e| panic!("{name}: {e:?}")).f32s() };
    let (om_s, rw_s, rf_s, ff_s) = (load("omega_hz.npy"), load("r_water.npy"), load("r_fat.npy"), load("ff.npy"));
    let nm = idx.len();
    assert_eq!(om_s.len(), kk * qq * nm, "truth_ratemap_bc: {} values, {} states × {nm} mask voxels", om_s.len(), kk * qq);
    // knot states (oracle): weighted mean r, circular mean cardiac time
    let ra = hasty_recon::data::read_npy(&dir.join("resp.npy")).unwrap();
    let (rw_, rv_) = (ra.shape[1], ra.f64s());
    let pa = hasty_recon::data::read_npy(&dir.join("physio.npy")).unwrap();
    let (pw_, pv_) = (pa.shape[1], pa.f64s());
    let mut acc = vec![(0.0f64, 0.0f64, Complex64::new(0.0, 0.0)); nk];
    for (j, &i) in used.iter().enumerate() {
        let (r, ph) = (rv_[i * rw_], pv_[i * pw_ + 1]);
        let z = Complex64::from_polar(1.0, std::f64::consts::TAU * ph);
        for &(b, wb) in &w[j] {
            acc[b].0 += wb;
            acc[b].1 += wb * r;
            acc[b].2 += wb * z;
        }
    }
    let to_grid = |full: Vec<f32>| -> Vec<f32> {
        if mdims == sd.dims || sd.native_grid() {
            full
        } else {
            assert_eq!(mdims, sd.tdims, "truth_ratemap_bc grid {mdims:?} is neither the reconstruction grid {:?} nor the native {:?}", sd.dims, sd.tdims);
            hasty_recon::pyramid::native_maps_to_grid(sd, &full, x_log, boxed)
        }
    };
    let ntot = mask.len();
    let mut out = Vec::with_capacity(nk);
    for (b, &(wsum, wr, wz)) in acc.iter().enumerate() {
        let r = if wsum > 0.0 { wr / wsum } else { 0.0 };
        let c = (wz.arg() / std::f64::consts::TAU).rem_euclid(1.0) * period;
        // bilinear corners: r on the (uniform) r_k grid, clamped; c periodic on the c_q grid
        let ur = if kk > 1 { ((r - rk[0]) / (rk[kk - 1] - rk[0]) * (kk - 1) as f64).clamp(0.0, (kk - 1) as f64) } else { 0.0 };
        let (k0, fr) = ((ur.floor() as usize).min(kk.saturating_sub(2)), 0.0f64);
        let fr = if kk > 1 { ur - k0 as f64 } else { fr };
        let uc = (c / period * qq as f64).rem_euclid(qq as f64);
        let (q0, fc) = (uc.floor() as usize % qq, uc - uc.floor());
        let q1 = (q0 + 1) % qq;
        let k1 = (k0 + 1).min(kk - 1);
        let corners = [(q0 * kk + k0, (1.0 - fr) * (1.0 - fc)), (q0 * kk + k1, fr * (1.0 - fc)), (q1 * kk + k0, (1.0 - fr) * fc), (q1 * kk + k1, fr * fc)];
        let interp = |a: &[f32], fill: f32| -> Vec<f32> {
            let mut full = vec![fill; ntot];
            for (n, &vi) in idx.iter().enumerate() {
                let (mut s, mut ws) = (0.0f64, 0.0f64);
                for &(st, cw) in &corners {
                    let x = a[st * nm + n];
                    if cw > 0.0 && x.is_finite() {
                        s += cw * x as f64;
                        ws += cw;
                    }
                }
                if ws > 0.0 {
                    full[vi] = (s / ws) as f32;
                }
            }
            full
        };
        let om_full = interp(&om_s, f32::NAN);
        let om_g = to_grid(om_full);
        let om: Vec<f64> = om_g.iter().map(|&x| if x.is_finite() { x as f64 * std::f64::consts::TAU } else { f64::NAN }).collect();
        let rw = to_grid(interp(&rw_s, 0.0)).into_iter().map(|x| if x.is_finite() { x } else { 0.0 }).collect();
        let rf = to_grid(interp(&rf_s, 0.0)).into_iter().map(|x| if x.is_finite() { x } else { 0.0 }).collect();
        let ff = to_grid(interp(&ff_s, 0.0)).into_iter().map(|x| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }).collect();
        if b == 0 || b == nk - 1 {
            println!("  knot {b}: true state r = {r:.3}, cardiac time {c:.3} s (of {period:.3})");
        }
        out.push(RateFrame { om, rw, rf, ff });
    }
    out
}

/// Pooled image x_e against the inverse-crime frame mean x̄★_e (scoring only): relative error over all voxels, the
/// body (PD > 0) without blood, the blood voxels and the air; the complex LS scale ⟨x̄★, x⟩ / ‖x̄★‖² (all encodes); the
/// Pearson correlation of |x_0| with the PD over the body and over all voxels.
fn pooled_image_error(xs: &hasty_recon::hoststack::HostStack, xm: &[Vec<C32>], stat: &[f32], bm: &[bool]) -> String {
    let nv = stat.len();
    let (mut num, mut den) = ([0.0f64; 4], [0.0f64; 4]);
    let mut sc = (Complex64::new(0.0, 0.0), 0.0f64);
    for (e, x) in xm.iter().enumerate() {
        let v = xs.chunk(e);
        for q in 0..nv {
            let (a, b) = (Complex64::new(v[q].re as f64, v[q].im as f64), Complex64::new(x[q].re as f64, x[q].im as f64));
            let cls = if bm[q] { 2 } else if stat[q] > 0.0 { 1 } else { 3 };
            for c in [0, cls] {
                num[c] += (a - b).norm_sqr();
                den[c] += b.norm_sqr();
            }
            sc.0 += b.conj() * a;
            sc.1 += b.norm_sqr();
        }
    }
    let corr = |mask: &dyn Fn(usize) -> bool| -> f64 {
        let v = xs.chunk(0);
        let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
        for q in (0..nv).filter(|&q| mask(q)) {
            let (a, b) = ((v[q].re as f64).hypot(v[q].im as f64), stat[q] as f64);
            n += 1.0;
            sa += a;
            sb += b;
            saa += a * a;
            sbb += b * b;
            sab += a * b;
        }
        let cov = sab / n - sa / n * sb / n;
        cov / ((saa / n - (sa / n).powi(2)).max(0.0).sqrt() * (sbb / n - (sb / n).powi(2)).max(0.0).sqrt()).max(1e-300)
    };
    let r = |c: usize| if den[c] > 0.0 { (num[c] / den[c]).sqrt() } else { f64::NAN };
    let s = sc.0 / sc.1.max(1e-300);
    // x★ = 0 in the air: its error relative to the whole object's norm
    let air = (num[3] / den[0].max(1e-300)).sqrt();
    format!(
        "rel error all {:.4}, static body {:.4}, blood {:.4}, air (/‖x★‖) {air:.4}; LS scale {:.4}{:+.4}i; corr |x|~PD body {:.4}, all {:.4}",
        r(0), r(1), r(2), s.re, s.im, corr(&|q| stat[q] > 0.0), corr(&|_| true)
    )
}

// ---------------------------------------------------------------------------------------------------------------------
// Estimator-preserving pooled preconditioners (research/precond_estimator.md, hasty_recon::ppre): --pp-study (pooled
// solves of a few encodes with several preconditioners, scored, then exit), --pp-pool (the --warm-full pooled start
// with one of them), --pp-metric (the P²nP circulant metric of zmshr1m from the probed symbol, + deflation).
// ---------------------------------------------------------------------------------------------------------------------

/// the --pp-* switches (all off by default: no change to existing runs)
struct PpArgs {
    /// `name=SPEC:ITERS[:TOL]` runs separated by ';' (a name starting with "ref" sets the reference x̂ of its μ)
    study: Option<String>,
    pool: Option<String>,
    metric: Option<String>,
    /// encodes of the study (default 0)
    groups: Option<Vec<usize>>,
    /// ridge μ relative to the body mean of the probed image diagonal of the pooled N (0: no ridge); several for the study
    mu: Vec<f64>,
    band: bool,
    dfloor: f64,
    probes: usize,
    smooth: usize,
    eps: f64,
    harvest: usize,
    tol: f64,
    /// G2: reference time of the recentring (s; None: the sample-weighted mid time of the two k-centre passes)
    tref: Option<f64>,
    /// G3: virtual coils of the surrogate (None: all)
    sur_coils: Option<usize>,
    /// G3 Chebyshev: lo = hi / ratio
    cheb_ratio: f64,
}

impl PpArgs {
    fn active(&self) -> bool {
        self.study.is_some() || self.pool.is_some() || self.metric.is_some()
    }
}

enum PpBase<'r> {
    Id,
    /// the existing per-encode circulant (Objective::metric_host of the group's first set)
    Circ(Vec<usize>),
    Sym(&'r [hasty_recon::ppre::SymPre]),
    /// the coil + segment surrogate (G3) inverted by its inner solver on the device; `ci`: the group's circulant (inner
    /// metric when `metric`)
    Sur { devs: &'r mut [hasty_recon::surrogate::SurDev], ci: Vec<usize>, inner: hasty_recon::surrogate::Inner, metric: bool },
}

/// PCG operators of the pooled problem B (N + μ) B on the host (B = band-limit, optional)
struct PpOps<'r, 'a> {
    p: &'r mut P2np<'a>,
    st: std::sync::Arc<cudarc::driver::CudaStream>,
    groups: &'r [Vec<usize>],
    g0: &'r hasty_recon::hoststack::HostStack,
    mu: Vec<f64>,
    band: Option<(std::sync::Arc<hasty_recon::ppre::Fft3>, std::sync::Arc<Vec<bool>>)>,
    base: PpBase<'r>,
    defl: Vec<Option<hasty_recon::ppre::Deflation>>,
    pre_s: f64,
    pre_n: usize,
    q_s: f64,
    q_n: usize,
    /// G2 recentring per group: U (unit modulus); the metric is U P Uᴴ
    rec: Option<Vec<Vec<C32>>>,
}

impl hasty_recon::ppre::PcgOps for PpOps<'_, '_> {
    fn q(&mut self, v: &hasty_recon::hoststack::HostStack, out: &mut hasty_recon::hoststack::HostStack) -> Result<(), hasty_recon::Error> {
        let t = Instant::now();
        // N v = Σ ∇f(v) − Σ ∇f(0)
        self.p.gradient_groups_host(&self.st, self.groups, v, out)?;
        hasty_recon::hoststack::HostStack::axpby(-1.0, self.g0, 1.0, out);
        for g in 0..v.nsets() {
            if self.mu[g] > 0.0 {
                hasty_recon::ppre::caxpy(Complex64::new(self.mu[g], 0.0), v.chunk(g), out.chunk_mut(g));
            }
            if let Some((f, m)) = &self.band {
                hasty_recon::ppre::project_band(f, m, out.chunk_mut(g));
            }
        }
        self.q_s += t.elapsed().as_secs_f64();
        self.q_n += 1;
        Ok(())
    }
    fn p(&mut self, g: usize, r: &[C32], z: &mut [C32]) -> Result<(), hasty_recon::Error> {
        let t = Instant::now();
        if let PpBase::Sur { devs, ci, inner, metric } = &mut self.base {
            let st = &self.st;
            let dr = st.clone_htod(r)?;
            let mut dz = st.alloc_zeros::<C32>(r.len())?;
            let m = if *metric { Some(&mut self.p.circ[ci[g]][0]) } else { None };
            devs[g].solve(st, m, *inner, &dr, &mut dz)?;
            st.memcpy_dtoh(&dz, z)?;
            if let Some(d) = self.defl[g].as_ref() {
                d.add(r, z);
            }
            self.pre_s += t.elapsed().as_secs_f64();
            self.pre_n += 1;
            return Ok(());
        }
        // G2: z = U P (Uᴴ r)
        let rr: Option<Vec<C32>> = self.rec.as_ref().map(|u| {
            let mut v = r.to_vec();
            hasty_recon::ppre::mul_cx(&mut v, &u[g], true);
            v
        });
        let r = rr.as_deref().unwrap_or(r);
        let band = self.band.as_ref().map(|(f, m)| (&**f, &m[..]));
        let defl = self.defl[g].as_ref();
        let (pp, st, base) = (&mut *self.p, &self.st, &self.base);
        hasty_recon::ppre::apply_composite(band, defl, r, z, &mut |a, b| match base {
            PpBase::Id => {
                b.copy_from_slice(a);
                Ok(())
            }
            PpBase::Circ(reps) => hasty_recon::p2np::Objective::metric_host(&mut *pp, st, reps[g], a, b, false),
            PpBase::Sym(s) => {
                s[g].apply(a, b);
                Ok(())
            }
            PpBase::Sur { .. } => unreachable!(),
        })?;
        if let Some(u) = &self.rec {
            hasty_recon::ppre::mul_cx(z, &u[g], false);
        }
        self.pre_s += t.elapsed().as_secs_f64();
        self.pre_n += 1;
        Ok(())
    }
}

/// image error of one encode's pooled image vs the frame mean of x★: [all, static body, blood, air (/‖x★‖), corr(|x|, PD) body]
fn pp_score(x: &[C32], xm: &[C32], stat: &[f32], bm: &[bool]) -> [f64; 5] {
    let (mut num, mut den) = ([0.0f64; 4], [0.0f64; 4]);
    for q in 0..x.len() {
        let (a, b) = (Complex64::new(x[q].re as f64, x[q].im as f64), Complex64::new(xm[q].re as f64, xm[q].im as f64));
        let cls = if bm[q] { 2 } else if stat[q] > 0.0 { 1 } else { 3 };
        for c in [0, cls] {
            num[c] += (a - b).norm_sqr();
            den[c] += b.norm_sqr();
        }
    }
    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    for q in (0..x.len()).filter(|&q| stat[q] > 0.0) {
        let (a, b) = ((x[q].re as f64).hypot(x[q].im as f64), stat[q] as f64);
        n += 1.0;
        sa += a;
        sb += b;
        saa += a * a;
        sbb += b * b;
        sab += a * b;
    }
    let corr = (sab / n - sa / n * sb / n) / ((saa / n - (sa / n).powi(2)).max(0.0).sqrt() * (sbb / n - (sb / n).powi(2)).max(0.0).sqrt()).max(1e-300);
    let r = |c: usize| if den[c] > 0.0 { (num[c] / den[c]).sqrt() } else { f64::NAN };
    [r(0), r(1), r(2), (num[3] / den[0].max(1e-300)).sqrt(), corr]
}

/// the probed quantities shared by all runs of one pp invocation (per group)
struct PpProbes {
    /// D^{-1/2} of the floored coil power Σ_c |S_c|² (P0, P1)
    dinv_c: Vec<f32>,
    dmed_c: f32,
    /// body mean of the probed image diagonal of N per group (the unit of μ), and the smoothed, floored diagonal (P2)
    lam_bar: Vec<f64>,
    diag: Vec<Option<Vec<f32>>>,
    /// probed Fourier symbols (two halves) with the coil D and with the diagonal D
    sym_c: Option<Vec<[Vec<f32>; 2]>>,
    sym_d: Option<Vec<[Vec<f32>; 2]>>,
    /// nominal (no-physics) symbol per group: the existing circulant's coil-convolved spectrum, if built
    nominal: Vec<Option<Vec<f32>>>,
}

/// the symbol (λ_eff > 0) and D^{-1/2} of a probed base for group g with ridge μ_g (absolute)
fn pp_factors(base: &str, g: usize, pr: &PpProbes, mu: f64, a: &PpArgs, dims: [usize; 3], support: &[bool]) -> (Vec<f32>, Vec<f32>) {
    let nv = dims.iter().product::<usize>();
    let finish = |raw: Vec<f32>, dmed: f32| -> Vec<f32> {
        let lmax = raw.iter().cloned().fold(0f32, f32::max);
        // the floor IS the ridge's symbol (μ / D in the body) when there is a ridge, else ε λ_max (as the old metric)
        let fl = if mu > 0.0 { (mu / dmed.max(f32::MIN_POSITIVE) as f64) as f32 } else { (a.eps * lmax as f64) as f32 };
        let lam: Vec<f32> = if mu > 0.0 { raw.iter().map(|&l| l.max(0.0) + fl).collect() } else { raw.iter().map(|&l| l.max(fl)).collect() };
        let at_floor = raw.iter().filter(|&&l| l < fl).count();
        println!("    {base} symbol, group {g}: λ_max {lmax:.3e}, floor {fl:.3e} ({}), {:.1} % of the modes below it", if mu > 0.0 { "μ / D_body, added" } else { "ε λ_max" }, 100.0 * at_floor as f64 / nv as f64);
        lam
    };
    match base {
        "p0" => {
            let nom = pr.nominal[g].as_ref().expect("--pp p0: needs the nominal circulant (add zmshr1m to --precond)");
            let h = &pr.sym_c.as_ref().unwrap()[g];
            let pm: f64 = h[0].iter().chain(&h[1]).map(|&v| v as f64).sum::<f64>() / (2 * nv) as f64;
            let nm: f64 = nom.iter().map(|&v| v as f64).sum::<f64>() / nv as f64;
            let c = (pm / nm.max(1e-300)) as f32;
            (finish(nom.iter().map(|&v| c * v).collect(), pr.dmed_c), pr.dinv_c.clone())
        }
        "p1" | "p1n" => {
            let h = &pr.sym_c.as_ref().unwrap()[g];
            let nom = if base == "p1n" { Some(&pr.nominal[g].as_ref().expect("--pp p1n: needs the nominal circulant (add zmshr1m to --precond)")[..]) } else { None };
            let (lam, noise) = hasty_recon::ppre::smooth_symbol(h, nom, dims, a.smooth);
            println!("    {base} probe noise after smoothing (median split-half relative difference): {noise:.3}");
            (finish(lam, pr.dmed_c), pr.dinv_c.clone())
        }
        "p12" | "p12n" => {
            let h = &pr.sym_d.as_ref().unwrap()[g];
            let d = pr.diag[g].as_ref().unwrap();
            let dmed = hasty_recon::ppre::median_on(d, support);
            let nom = if base == "p12n" { Some(&pr.nominal[g].as_ref().expect("--pp p12n: needs the nominal circulant")[..]) } else { None };
            let (lam, noise) = hasty_recon::ppre::smooth_symbol(h, nom, dims, a.smooth);
            println!("    {base} probe noise after smoothing: {noise:.3}");
            (finish(lam, dmed), d.iter().map(|&v| 1.0 / v.sqrt()).collect())
        }
        "p2" => {
            // Jacobi: P = (diag N + μ)⁻¹ (constant symbol 1)
            let d = pr.diag[g].as_ref().unwrap();
            let lam = vec![1f32; nv];
            (lam, d.iter().map(|&v| 1.0 / (v as f64 + mu).sqrt() as f32).collect())
        }
        b if b.starts_with("pe") => {
            // bounded dynamic range: (λ_nom + ε λ_max)^p (P applies its inverse), nominal symbol, coil D (floored)
            let (eps, pw) = hasty_recon::ppre::PreSpec::parse(b).and_then(|s| s.power()).expect("pe<ε>_<p>");
            let nom = pr.nominal[g].as_ref().expect("--pp pe: needs the nominal circulant (add zmshr1m to --precond)");
            let lmax = nom.iter().cloned().fold(0f32, f32::max) as f64;
            let lam: Vec<f32> = nom.iter().map(|&v| ((v.max(0.0) as f64 + eps * lmax) / lmax).powf(pw) as f32).collect();
            let lmin = lam.iter().cloned().fold(f32::MAX, f32::min);
            println!("    {b} symbol, group {g}: (λ/λ_max + {eps:e})^{pw}: dynamic range {:.2e}", 1.0 / lmin as f64 * lam.iter().cloned().fold(0f32, f32::max) as f64);
            (lam, pr.dinv_c.clone())
        }
        b => panic!("pp_factors: {b} is not a probed base"),
    }
}

/// G3 setup: the coil + segment surrogate of every group (encode sel[g]) from its pooled kept samples (coordinates,
/// times) and the STORED segment tables, rank-`ls` compressed; then the operator check that fixes Λ's conventions
/// (k mirror, pair transpose) and scale: for a random x per group, min over the four conventions of
/// ‖c Ñx − Nx‖ / ‖Nx‖ with the best real c (data-free: the recon's own N). μ is set by the caller.
#[allow(clippy::too_many_arguments)]
fn pp_build_sur(p: &mut P2np<'_>, st: &std::sync::Arc<cudarc::driver::CudaStream>, dev: &Device, kern: &std::sync::Arc<Kernels>, dims: [usize; 3], groups: &[Vec<usize>], sel: &[usize], sets: &[SampleSet], omega: &[f64], op_coords: Option<&[[Vec<f32>; 3]]>, seg_host: &[hasty_nufft::SegHostTables], dcoil: &[f32], smaps: &[C32], ncoil: usize, g0: &hasty_recon::hoststack::HostStack, ls: usize) -> Vec<hasty_recon::surrogate::SurDev> {
    use hasty_recon::hoststack::HostStack;
    use hasty_recon::ppre::{phases, rdot};
    use hasty_recon::surrogate::{build_symbols, SurConv, SurDev};
    let nv: usize = dims.iter().product();
    let tb = Instant::now();
    let mut devs = Vec::with_capacity(sel.len());
    for (g, &e) in sel.iter().enumerate() {
        let mem = &groups[g];
        // the OPERATOR's sample coordinates (seg_tables: + the concomitant linear terms and the absorbed linear field
        // c₁ τ, cropped), else the nominal ones (2026-11-22: with the nominal coordinates the full-physics surrogate was
        // uncorrelated with N, error 0.99; the stored maps hold only the residual field)
        let c: [Vec<f32>; 3] = [0, 1, 2].map(|a| mem.iter().flat_map(|&t| match op_coords { Some(oc) if !oc[t][a].is_empty() => oc[t][a].clone(), _ => sets[t].coords[a].clone() }).collect());
        let taus: Vec<f64> = mem.iter().flat_map(|&t| sets[t].tau.iter().copied()).collect();
        assert!(!taus.is_empty() && taus.len() == c[0].len(), "--pp sur: the kept samples' coordinates / times are missing ({} / {})", c[0].len(), taus.len());
        let tb_ = &seg_host[e];
        {
            // decomposition check (diagnostic): |Σ_l B_l(v) a_l(τ)| and its distance from e^{−iωτ} (exact for --phys b0
            // without decay; with concomitant fields it includes their phase), magnitudes of B and a
            let mut a = vec![num_complex::Complex64::new(0.0, 0.0); tb_.nseg];
            let (mut m1, mut d1, mut n, mut amax) = (0.0f64, 0.0f64, 0usize, 0.0f64);
            for j in 0..20 {
                let tau = tb_.t0 + (tb_.ntab - 1) as f64 * tb_.dt * j as f64 / 19.0;
                tb_.interp(tau, &mut a);
                amax = a.iter().map(|z| z.norm()).fold(amax, f64::max);
                for v in (0..nv).step_by(997) {
                    let g: num_complex::Complex64 = (0..tb_.nseg).map(|l| { let b = tb_.segph[l * nv + v]; num_complex::Complex64::new(b.re as f64, b.im as f64) * a[l] }).sum();
                    m1 += g.norm();
                    d1 += (g - num_complex::Complex64::from_polar(1.0, -omega[v] * tau)).norm();
                    n += 1;
                }
            }
            let bmax = tb_.segph.iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).fold(0f32, f32::max);
            println!("  segment decomposition check encode {e}: mean |Σ B a| {:.3}, mean |Σ B a − e^(−iωτ)| {:.3}, max |B| {bmax:.3}, max |a| {amax:.3} ({} nodes, nseg {})", m1 / n as f64, d1 / n as f64, tb_.ntab, tb_.nseg);
        }
        let (tmin, tmax) = taus.iter().fold((f64::MAX, f64::MIN), |(a, b), &t| (a.min(t), b.max(t)));
        println!("  surrogate encode {e}: sample τ {:.4}..{:.4} ms, table {:.4}..{:.4} ms ({} nodes); coordinates: {}", 1e3 * tmin, 1e3 * tmax, 1e3 * tb_.t0, 1e3 * (tb_.t0 + (tb_.ntab - 1) as f64 * tb_.dt), tb_.ntab, if op_coords.is_some() { "the operator's (shifted)" } else { "nominal" });
        let (maps, lam, loss) = build_symbols(dev, st, dims, &seg_host[e], [&c[0], &c[1], &c[2]], &taus, dcoil, ls).unwrap();
        let sm: Vec<&[C32]> = (0..ncoil).map(|cc| &smaps[cc * nv..(cc + 1) * nv]).collect();
        let sd = SurDev::new(dev, st, kern.clone(), dims, &maps, lam, &sm, 0.0).unwrap();
        println!("  surrogate encode {e}: L_s {} of {} stored maps (coil-power-weighted energy not kept {loss:.2e}), {ncoil} coils, {} pooled samples, {:.0} MB device", sd.ls, seg_host[e].nseg, taus.len(), sd.device_bytes() as f64 / 1e6);
        devs.push(sd);
    }
    let glens = vec![nv; groups.len()];
    let mut xs = HostStack::zeros(&glens);
    for g in 0..groups.len() {
        xs.chunk_mut(g).copy_from_slice(&phases(nv, 0xCA11 + g as u64));
    }
    let mut nx = HostStack::zeros(&glens);
    p.gradient_groups_host(st, groups, &xs, &mut nx).unwrap();
    HostStack::axpby(-1.0, g0, 1.0, &mut nx);
    for g in 0..groups.len() {
        let dx = st.clone_htod(xs.chunk(g)).unwrap();
        let mut dy = st.alloc_zeros::<C32>(nv).unwrap();
        let target = nx.chunk(g);
        let tn = rdot(target, target).sqrt();
        let mut best = (f64::MAX, SurConv::default());
        let mut line = String::new();
        for (mirror, transpose, conj_maps) in [(false, false, false), (true, false, false), (false, true, false), (true, true, false), (false, false, true), (true, false, true), (false, true, true), (true, true, true)] {
            devs[g].set_conv(st, SurConv { mirror, transpose, conj_maps, scale: 1.0 }).unwrap();
            devs[g].apply(st, &dx, &mut dy).unwrap();
            let y = st.clone_dtoh(&dy).unwrap();
            let c = rdot(&y, target) / rdot(&y, &y).max(1e-300);
            let d: Vec<C32> = y.iter().zip(target).map(|(a, b)| C32::new(c as f32 * a.re - b.re, c as f32 * a.im - b.im)).collect();
            let err = rdot(&d, &d).sqrt() / tn.max(1e-300);
            line += &format!(" m{}t{}c{}: {err:.3} ({c:.2e});", mirror as u8, transpose as u8, conj_maps as u8);
            if err < best.0 && c > 0.0 {
                best = (err, SurConv { mirror, transpose, conj_maps, scale: c });
            }
        }
        devs[g].set_conv(st, best.1).unwrap();
        devs[g].applies = 0;
        println!("  surrogate encode {}: ‖(N − c Ñ) x‖ / ‖N x‖ by convention:{line} chosen {:?} (error {:.3})", sel[g], best.1, best.0);
    }
    println!("  surrogate setup (L_s {ls}): {:.1}s", tb.elapsed().as_secs_f64());
    devs
}

/// all --pp-* work. Returns after a --pp-pool / --pp-metric setup; --pp-study exits the process.
#[allow(clippy::too_many_arguments)]
fn pp_run(
    a: &PpArgs,
    p: &mut P2np<'_>,
    st: &std::sync::Arc<cudarc::driver::CudaStream>,
    dev: &Device,
    kern: &std::sync::Arc<Kernels>,
    dims: [usize; 3],
    enc: &[usize],
    ne: usize,
    support: &[bool],
    smaps: &[C32],
    nvc: usize,
    omega: &[f64],
    synth: Option<&(Vec<Vec<C32>>, Vec<f32>, Vec<bool>)>,
    out: &std::path::Path,
    ctl: &RunCtl,
    circ_ws: Option<&std::sync::Arc<std::sync::Mutex<hasty_recon::precond::CircShared>>>,
    warm_full: usize,
    sets: &[SampleSet],
    seg_host: Option<&[hasty_nufft::SegHostTables]>,
    op_coords: Option<&[[Vec<f32>; 3]]>,
) {
    use hasty_recon::hoststack::HostStack;
    use hasty_recon::ppre::{self, Fft3, PcgCfg, PreSpec, SymPre};
    use std::io::Write;
    use std::sync::Arc;
    let tp = Instant::now();
    let nv: usize = dims.iter().product();
    let fft = Arc::new(Fft3::new(dims));
    let ball = Arc::new(ppre::ball_mask(dims));
    let study = a.study.is_some();
    // the runs: (name, spec, iterations, tol)
    let runs: Vec<(String, PreSpec, usize, f64)> = match &a.study {
        Some(s) => s
            .split(';')
            .filter(|r| !r.trim().is_empty())
            .map(|r| {
                let (name, rest) = r.trim().split_once('=').unwrap_or_else(|| panic!("--pp-study run {r}: name=SPEC:ITERS[:TOL]"));
                let f: Vec<&str> = rest.split(':').collect();
                let spec = PreSpec::parse(f[0]).unwrap_or_else(|| panic!("--pp-study: bad preconditioner {}", f[0]));
                let it: usize = f.get(1).map_or(100, |v| v.parse().expect("--pp-study ITERS"));
                let tol: f64 = f.get(2).map_or(0.0, |v| v.parse().expect("--pp-study TOL"));
                (name.to_string(), spec, it, tol)
            })
            .collect(),
        None => vec![],
    };
    let pool = a.pool.as_ref().map(|s| PreSpec::parse(s).unwrap_or_else(|| panic!("--pp-pool {s}: none|circ|p0|p1|p1n|p12|p12n|p2[+dT[,B]]")));
    let metric = a.metric.as_ref().map(|s| PreSpec::parse(s).unwrap_or_else(|| panic!("--pp-metric {s}")));
    if let Some(m) = &metric {
        // p2 = the Jacobi metric (diag N + μ)⁻¹: from_factors with the constant symbol 1 and D^{-1/2} = (diag + μ)^{-1/2}
        assert!(m.base != "none" && m.base != "circ" && m.sur().is_none() && !m.recentre, "--pp-metric: a circulant base (p0, p1, p1n, p12, p12n, pe<ε>_<p>) or the Jacobi metric p2");
        assert!(!m.deflates() || pool.as_ref().is_some_and(|q| q.base == m.base), "--pp-metric with deflation: --pp-pool must use the same base (its harvest gives the Ritz vectors)");
        assert!(p.circ.len() == ne && !p.circ_map.is_empty(), "--pp-metric: needs the per-encode circulants (--precond zmshr1m, --circ-share encode)");
    }
    assert!(pool.is_none() || warm_full > 0, "--pp-pool: give the pooled iteration count with --warm-full N");
    let sel: Vec<usize> = if study { a.groups.clone().unwrap_or_else(|| vec![0]) } else { (0..ne).collect() };
    assert!(sel.iter().all(|&e| e < ne), "--pp-groups: encodes < {ne}");
    let groups: Vec<Vec<usize>> = sel.iter().map(|&e| (0..enc.len()).filter(|&t| enc[t] == e).collect()).collect();
    let ng = groups.len();
    let glens = vec![nv; ng];
    let reps: Vec<usize> = groups.iter().map(|g| g[0]).collect();
    println!("pp: {} group(s) (encodes {:?}, {} sets each), grid {:?}; estimator: ridge μ = {:?} × λ̄_body{}, D floor {} × body median; {} probes per symbol, smoothing radius {}",
        ng, sel, groups[0].len(), dims, a.mu, if a.band { ", band-limited to the sampled ball" } else { "" }, a.dfloor, a.probes, a.smooth);
    // b = −Σ ∇f(0)
    let mut g0 = HostStack::zeros(&glens);
    p.gradient_groups_host(st, &groups, &HostStack::zeros(&glens), &mut g0).unwrap();
    // ---- probes (one pooled pass each) --------------------------------------------------------------------------
    let all_specs: Vec<&PreSpec> = runs.iter().map(|r| &r.1).chain(pool.iter()).chain(metric.iter()).collect();
    let need_c = all_specs.iter().any(|s| matches!(s.base.as_str(), "p0" | "p1" | "p1n"));
    // the symbol with the diagonal D only for p12 / p12n (p2 needs the diagonal alone)
    let need_d = all_specs.iter().any(|s| s.base.starts_with("p12"));
    let need_diag = all_specs.iter().any(|s| s.diag()) || a.mu.iter().any(|&m| m > 0.0);
    let dcoil: Vec<f32> = (0..nv).map(|q| (0..nvc).map(|c| { let s = smaps[c * nv + q]; s.re * s.re + s.im * s.im }).sum()).collect();
    let dfl = ppre::floor_at_median(&dcoil, support, a.dfloor);
    let dmed_c = ppre::median_on(&dfl, support);
    let dinv_c: Vec<f32> = dfl.iter().map(|&v| 1.0 / v.sqrt()).collect();
    let nominal: Vec<Option<Vec<f32>>> = reps.iter().map(|&t| {
        if p.circ.is_empty() {
            return None;
        }
        let ci = if p.circ_map.is_empty() { t } else { p.circ_map[t] };
        Some(p.circ[ci][0].lambda.clone())
    }).collect();
    let tpr = Instant::now();
    let (lam_bar, diag, sym_c, sym_d) = {
        let mut napply = |v: &HostStack, o: &mut HostStack| -> Result<(), hasty_recon::Error> {
            p.gradient_groups_host(st, &groups, v, o)?;
            HostStack::axpby(-1.0, &g0, 1.0, o);
            Ok(())
        };
        let (lam_bar, diag) = if need_diag {
            let h = ppre::probe_diag(ng, nv, a.probes, 0xD1A6, &mut napply).unwrap();
            let mut lb = vec![];
            let mut dg = vec![];
            for (g, [h0, h1]) in h.into_iter().enumerate() {
                let raw: Vec<f32> = h0.iter().zip(&h1).map(|(x, y)| 0.5 * (x + y)).collect();
                let nb = support.iter().filter(|&&b| b).count().max(1);
                let mean_body = raw.iter().zip(support).filter(|(_, &b)| b).map(|(&v, _)| v as f64).sum::<f64>() / nb as f64;
                let mut sm = raw.clone();
                ppre::smooth_box(&mut sm, dims, a.smooth.max(1), 2, false);
                let (mut s0, mut s1) = (h0.clone(), h1.clone());
                ppre::smooth_box(&mut s0, dims, a.smooth.max(1), 2, false);
                ppre::smooth_box(&mut s1, dims, a.smooth.max(1), 2, false);
                let mut rd: Vec<f64> = (0..nv).filter(|&q| support[q] && sm[q] > 0.0).map(|q| ((s0[q] - s1[q]) as f64 / sm[q] as f64).abs()).collect();
                rd.sort_by(|x, y| x.total_cmp(y));
                let dmean = dcoil.iter().zip(support).filter(|(_, &b)| b).map(|(&v, _)| v as f64).sum::<f64>() / nb as f64;
                println!("  probed image diagonal of N, group {g}: body mean λ̄ = {mean_body:.4e} (coil power body mean {dmean:.3e}); probe noise after smoothing (median split-half) {:.3}", rd.get(rd.len() / 2).copied().unwrap_or(f64::NAN));
                lb.push(mean_body);
                dg.push(Some(ppre::floor_at_median(&sm, support, a.dfloor)));
            }
            (lb, dg)
        } else {
            (vec![f64::NAN; ng], vec![None; ng])
        };
        let sym_c = need_c.then(|| ppre::probe_symbol(&fft, &vec![Some(&dinv_c[..]); ng], a.probes, 0x5E1C, &mut napply).unwrap());
        let sym_d = need_d.then(|| {
            let di: Vec<Vec<f32>> = diag.iter().map(|d| d.as_ref().unwrap().iter().map(|&v| 1.0 / v.sqrt()).collect()).collect();
            let r: Vec<Option<&[f32]>> = di.iter().map(|v| Some(&v[..])).collect();
            ppre::probe_symbol(&fft, &r, a.probes, 0x5E1D, &mut napply).unwrap()
        });
        (lam_bar, diag, sym_c, sym_d)
    };
    let pr = PpProbes { dinv_c, dmed_c, lam_bar, diag, sym_c, sym_d, nominal };
    println!("  pp probes: {:.1}s", tpr.elapsed().as_secs_f64());
    // where the physics moves the symbol (E1): probed vs scaled nominal per k shell, group 0
    if let (Some(sc), Some(nom)) = (&pr.sym_c, &pr.nominal[0]) {
        let h = &sc[0];
        let pm: Vec<f64> = h[0].iter().zip(&h[1]).map(|(x, y)| 0.5 * (*x as f64 + *y as f64)).collect();
        let c = pm.iter().sum::<f64>() / nom.iter().map(|&v| v as f64).sum::<f64>().max(1e-300);
        let edges = [0.125, 0.25, 0.5, 0.75, 0.9, 1.0, 1.2, f64::INFINITY];
        let (mut sp, mut sn, mut cnt) = ([0.0f64; 8], [0.0f64; 8], [0usize; 8]);
        for q in 0..nv {
            let i = edges.iter().position(|&e| ppre::kradius(dims, q) <= e).unwrap_or(7);
            sp[i] += pm[q];
            sn[i] += c * nom[q] as f64;
            cnt[i] += 1;
        }
        println!("  probed / nominal symbol (coil D, nominal scaled to the probed mean ×{c:.3e}) per k shell r ≤ {:?}: {}", edges, (0..8).map(|i| format!("{:.3}", sp[i] / sn[i].max(1e-300))).collect::<Vec<_>>().join(" "));
        println!("    mean probed symbol per shell: {}", (0..8).map(|i| format!("{:.2e}", sp[i] / cnt[i].max(1) as f64)).collect::<Vec<_>>().join(" "));
    }
    // image masks for the Ritz reports: air, body, the 10 % highest |Δf| in the body
    let air: Vec<bool> = support.iter().map(|&b| !b).collect();
    let thr = {
        let mut f: Vec<f64> = (0..nv).filter(|&q| support[q]).map(|q| omega[q].abs()).collect();
        f.sort_by(|x, y| x.total_cmp(y));
        f.get(f.len() * 9 / 10).copied().unwrap_or(0.0)
    };
    let hi: Vec<bool> = (0..nv).map(|q| support[q] && omega[q].abs() > thr).collect();
    println!("  Ritz masks: air {} voxels, body {}, high field (|Δf| > {:.0} Hz, top 10 % of the body) {}", air.iter().filter(|&&b| b).count(), support.iter().filter(|&&b| b).count(), thr / std::f64::consts::TAU, hi.iter().filter(|&&b| b).count());
    // G1 / G2: the two k-centre passes of the readout from the kept samples of encode 0 (lowest 0.5 % of |k|, split at
    // the middle of the ADC time range): Δt between them and t_ref, their sample-weighted mid time
    let (dt_c, t_ref) = {
        let mem: Vec<usize> = (0..enc.len()).filter(|&t| enc[t] == 0).collect();
        let mut v: Vec<(f32, f64)> = mem.iter().flat_map(|&t| { let s = &sets[t]; (0..s.tau.len().min(s.coords[0].len())).map(move |i| (s.coords[0][i].powi(2) + s.coords[1][i].powi(2) + s.coords[2][i].powi(2), s.tau[i])) }).collect();
        if v.is_empty() {
            println!("  pp: no sample times kept, Δt / t_ref unknown");
            (f64::NAN, f64::NAN)
        } else {
            let (tmin, tmax) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(x.1), b.max(x.1)));
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            let nlow = (v.len() / 200).max(10).min(v.len());
            let mid = 0.5 * (tmin + tmax);
            let (mut s1, mut n1, mut s2, mut n2) = (0.0, 0usize, 0.0, 0usize);
            for &(_, t) in &v[..nlow] {
                if t < mid { s1 += t; n1 += 1 } else { s2 += t; n2 += 1 }
            }
            let (t1, t2) = (s1 / n1.max(1) as f64, s2 / n2.max(1) as f64);
            let tr = (s1 + s2) / (n1 + n2).max(1) as f64;
            println!("  k-centre passes (lowest {nlow} |k| samples of encode 0): {n1} at {:.3} ms, {n2} at {:.3} ms → Δt {:.3} ms, t_ref {:.3} ms (sample-weighted)", 1e3 * t1, 1e3 * t2, 1e3 * (t2 - t1), 1e3 * tr);
            (if n1 > 0 && n2 > 0 { t2 - t1 } else { f64::NAN }, tr)
        }
    };
    let t_ref = a.tref.unwrap_or(t_ref);
    // G1: shift of the second centre pass s(x) = Δt |∇ω(x)| / π (rad per voxel over k_max = π per voxel), ω the
    // operator's ratemap field, central differences in voxel units
    let shift: Vec<f32> = {
        let [nx, ny, _] = dims;
        let strides = [1usize, nx, nx * ny];
        (0..nv)
            .map(|q| {
                let i = [q % nx, (q / nx) % ny, q / (nx * ny)];
                let mut g2 = 0.0f64;
                for ax in 0..3 {
                    if dims[ax] < 2 {
                        continue;
                    }
                    let sd = strides[ax];
                    let lo = if i[ax] > 0 { q - sd } else { q };
                    let hi = if i[ax] + 1 < dims[ax] { q + sd } else { q };
                    let d = (omega[hi] - omega[lo]) / ((hi - lo) / sd) as f64;
                    g2 += d * d;
                }
                (dt_c * g2.sqrt() / std::f64::consts::PI) as f32
            })
            .collect()
    };
    let shifted: Vec<bool> = (0..nv).map(|q| support[q] && shift[q] > 0.3).collect();
    let vol_shift = shifted.iter().filter(|&&b| b).count() as f64 / nv as f64;
    {
        let mut b: Vec<f32> = (0..nv).filter(|&q| support[q]).map(|q| shift[q]).collect();
        b.sort_by(|x, y| x.total_cmp(y));
        let pc = |f: f64| b.get(((b.len() as f64 - 1.0) * f) as usize).copied().unwrap_or(f32::NAN);
        println!("  G1 shift map s = Δt |∇f| / k_max in the body: median {:.2}, 75 % {:.2}, 90 % {:.2}, 95 % {:.2}; s > 0.3 in {:.1} % of the grid", pc(0.5), pc(0.75), pc(0.9), pc(0.95), 100.0 * vol_shift);
    }
    let ritz_report = |tag: &str, hv: &ppre::Harvest| -> ppre::RitzSet {
        let rs = ppre::ritz(hv);
        let n = rs.theta.len();
        if n == 0 {
            return rs;
        }
        let conv = rs.res.iter().filter(|&&r| r <= 1e-2).count();
        let show = |i: usize| format!("{:.3e} (res {:.1e})", rs.theta[i], rs.res[i]);
        println!("  Ritz {tag}: m {} basis, {n} pairs, {conv} with residual ≤ 1e-2; κ_eff (converged) {:.3e}; top {}; bottom {}", rs.m, rs.kappa(1e-2),
            (0..n.min(5)).map(|j| show(n - 1 - j)).collect::<Vec<_>>().join(", "), (0..n.min(3)).map(show).collect::<Vec<_>>().join(", "));
        let masks: [(&str, &[bool]); 4] = [("air", &air), ("body", support), ("high-field", &hi), ("s>0.3", &shifted)];
        for (lab, i) in [("top 1", n - 1), ("top 2", n.saturating_sub(2)), ("top 3", n.saturating_sub(3)), ("bottom 1", 0), ("bottom 2", 1usize.min(n - 1))] {
            let u = rs.vector(hv, i);
            let e_all: f64 = u.iter().map(|v| v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64).sum();
            let e_sh: f64 = u.iter().zip(&shifted).filter(|(_, &b)| b).map(|(v, _)| v.re as f64 * v.re as f64 + v.im as f64 * v.im as f64).sum();
            let share = e_sh / e_all.max(1e-300);
            println!("    {lab} θ {:.3e}: {} | G1: energy share in s > 0.3 {share:.3} vs volume share {vol_shift:.3} → ratio {:.2}", rs.theta[i], ppre::energy_where(&fft, &u, &masks), share / vol_shift.max(1e-300));
        }
        rs
    };
    // ---- the study -------------------------------------------------------------------------------------------------
    if study {
        let mut summary = std::fs::File::create(out.join("pp_summary.txt")).unwrap();
        let need_harvest: Vec<String> = runs.iter().filter(|r| r.1.deflates()).map(|r| r.1.base.clone()).collect();
        let mus: Vec<f64> = if a.mu.is_empty() { vec![0.0] } else { a.mu.clone() };
        let mut sur_cache: std::collections::HashMap<usize, Vec<hasty_recon::surrogate::SurDev>> = std::collections::HashMap::new();
        let ncoil = a.sur_coils.unwrap_or(nvc).min(nvc).max(1);
        for &mu_rel in &mus {
            let mu_g: Vec<f64> = pr.lam_bar.iter().map(|&l| if mu_rel > 0.0 { mu_rel * l } else { 0.0 }).collect();
            let mut syms: std::collections::HashMap<String, Vec<SymPre>> = std::collections::HashMap::new();
            let mut harvests: std::collections::HashMap<String, (Vec<ppre::Harvest>, Vec<ppre::RitzSet>)> = std::collections::HashMap::new();
            let mut xref: Option<HostStack> = None;
            let mut b = g0.clone();
            b.data.iter_mut().for_each(|v| *v = C32::new(-v.re, -v.im));
            if a.band {
                for g in 0..ng {
                    ppre::project_band(&fft, &ball, b.chunk_mut(g));
                }
            }
            for (name, spec, iters, tol) in &runs {
                let tr = Instant::now();
                println!("== pp run {name}: {} {}{} μ_rel {mu_rel:e}, {iters} iterations{}", spec.base, if spec.deflates() { format!("+deflation top {} bottom {}", spec.top, spec.bot) } else { String::new() }, if a.band { ", band" } else { "" }, if *tol > 0.0 { format!(", tol {tol:e}") } else { String::new() });
                if spec.sym() {
                    if !syms.contains_key(&spec.base) {
                        let v: Vec<SymPre> = (0..ng).map(|g| { let (l, d) = pp_factors(&spec.base, g, &pr, mu_g[g], a, dims, support); SymPre::new(fft.clone(), l, Some(d)) }).collect();
                        syms.insert(spec.base.clone(), v);
                    }
                }
                let defl: Vec<Option<ppre::Deflation>> = if spec.deflates() {
                    let (hv, rs) = harvests.get(&spec.base).unwrap_or_else(|| panic!("pp run {name}: run the base {} (no deflation) earlier in --pp-study", spec.base));
                    (0..ng).map(|g| {
                        let d = rs[g].deflation(&hv[g], spec.top, spec.bot, 1e-2);
                        if let Some(d) = &d {
                            println!("    deflation group {g}: rank {}, θ {:?} → {:?}", d.rank(), d.theta.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>(), d.target.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>());
                        } else {
                            println!("    deflation group {g}: no converged Ritz pair, none");
                        }
                        d
                    }).collect()
                } else {
                    vec![None; ng]
                };
                assert!(!(spec.recentre && spec.deflates()), "pp run {name}: +u with +d is not supported");
                let rec: Option<Vec<Vec<C32>>> = spec.recentre.then(|| {
                    let sh = seg_host.expect("+u: needs the stored segment tables (--model seg)");
                    sel.iter().map(|&e| hasty_recon::surrogate::recentre_diag(&sh[e], nv, t_ref)).collect()
                });
                let ci: Vec<usize> = reps.iter().map(|&t| if p.circ_map.is_empty() { 0 } else { p.circ_map[t] }).collect();
                let base = if let Some((cheb, ls, kk, met)) = spec.sur() {
                    assert!(!met || !p.circ.is_empty(), "pp run {name}: the inner metric needs the circulants (--precond zmshr1m), or use the n suffix");
                    if !sur_cache.contains_key(&ls) {
                        let sh = seg_host.expect("sur / cheb: needs the stored segment tables (--model seg)");
                        let v = pp_build_sur(&mut *p, st, dev, kern, dims, &groups, &sel, sets, omega, op_coords, sh, &dcoil, smaps, ncoil, &g0, ls);
                        sur_cache.insert(ls, v);
                    }
                    let devs = sur_cache.get_mut(&ls).unwrap();
                    for (g, d) in devs.iter_mut().enumerate() {
                        d.mu = mu_g[g];
                        if cheb {
                            let m = if met { Some(&mut p.circ[ci[g]][0]) } else { None };
                            let (lo, hi) = d.bounds(st, m, 12, a.cheb_ratio).unwrap();
                            println!("    Chebyshev interval of M Q̃, group {g}: [{lo:.3e}, {hi:.3e}] (Lanczos 12 + 5 %, ratio {})", a.cheb_ratio);
                        }
                    }
                    let inner = if cheb { hasty_recon::surrogate::Inner::Cheb(kk) } else { hasty_recon::surrogate::Inner::Pcg(kk) };
                    PpBase::Sur { devs: &mut devs[..], ci, inner, metric: met }
                } else {
                    match spec.base.as_str() {
                        "none" => PpBase::Id,
                        "circ" => {
                            assert!(!p.circ.is_empty(), "pp run {name}: circ needs the circulants (--precond zmshr1m)");
                            PpBase::Circ(reps.clone())
                        }
                        s => PpBase::Sym(&syms[s][..]),
                    }
                };
                // every run harvests its Lanczos data (the Ritz report: κ_eff, where the outliers sit); only a base run
                // that a later run deflates keeps it; none for a variable (flexible) preconditioner
                let harvest = if spec.flexible() { 0 } else { a.harvest };
                let t_setup = tr.elapsed().as_secs_f64();
                let t_solve = Instant::now();
                let mut tol_hit: [Option<(usize, f64)>; 3] = [None; 3];
                let mut csv = std::fs::File::create(out.join(format!("pp_mu{mu_rel:e}_{name}.csv"))).unwrap();
                writeln!(csv, "it,rel_res,err_ref,err_all,err_body,err_blood,air,corr_pd").unwrap();
                let (mut it1, mut it01, mut best) = (None::<usize>, None::<usize>, (f64::MAX, 0usize));
                let mut last = [f64::NAN; 5];
                let mut last_ref = f64::NAN;
                let mut x = HostStack::zeros(&glens);
                let o = {
                    let mut ops = PpOps { p: &mut *p, st: st.clone(), groups: &groups, g0: &g0, mu: mu_g.clone(), band: a.band.then(|| (fft.clone(), ball.clone())), base, defl, pre_s: 0.0, pre_n: 0, q_s: 0.0, q_n: 0, rec: rec.clone() };
                    let xr = xref.as_ref();
                    let mut cb = |k: usize, xs: &HostStack, rel: &[f64]| -> Result<bool, hasty_recon::Error> {
                        let e_ref = xr.map_or(f64::NAN, |r| {
                            let (mut n, mut d) = (0.0f64, 0.0f64);
                            for g in 0..ng {
                                let (u, v) = (xs.chunk(g), r.chunk(g));
                                n += u.iter().zip(v).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum::<f64>();
                                d += v.iter().map(|q| (q.re as f64).powi(2) + (q.im as f64).powi(2)).sum::<f64>();
                            }
                            (n / d.max(1e-300)).sqrt()
                        });
                        let sc = synth.map_or([f64::NAN; 5], |(xm, stat, bm)| pp_score(xs.chunk(0), &xm[sel[0]], stat, bm));
                        let rmax = rel.iter().cloned().fold(0.0, f64::max);
                        writeln!(csv, "{k},{rmax:.6e},{e_ref:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}", sc[0], sc[1], sc[2], sc[3], sc[4]).unwrap();
                        for (j, th) in [1e-2, 1e-3, 1e-4].iter().enumerate() {
                            if e_ref <= *th && tol_hit[j].is_none() {
                                tol_hit[j] = Some((k, t_solve.elapsed().as_secs_f64()));
                            }
                        }
                        if e_ref <= 1e-2 && it1.is_none() {
                            it1 = Some(k);
                        }
                        if e_ref <= 1e-3 && it01.is_none() {
                            it01 = Some(k);
                        }
                        if sc[0] < best.0 {
                            best = (sc[0], k);
                        }
                        last = sc;
                        last_ref = e_ref;
                        if k <= 3 || k % 10 == 0 {
                            println!("  it {k}: ‖r‖/‖r₀‖ {rmax:.2e}, vs x̂ {e_ref:.3e}, vs x★ all {:.4} body {:.4} blood {:.4} air {:.4} corr {:.3}", sc[0], sc[1], sc[2], sc[3], sc[4]);
                        }
                        Ok(!ctl.stop_requested())
                    };
                    let o = ppre::pcg_groups(&mut ops, &b, &mut x, &PcgCfg { iters: *iters, tol: *tol, harvest, flexible: spec.flexible() }, &mut cb).unwrap();
                    let ms_p = 1e3 * ops.pre_s / ops.pre_n.max(1) as f64;
                    let ms_q = 1e3 * ops.q_s / ops.q_n.max(1) as f64;
                    (o, ms_p, ms_q)
                };
                let (o, ms_p, ms_q) = o;
                let mut ext = (f64::NAN, f64::NAN);
                let kap = if harvest > 0 {
                    let rs: Vec<ppre::RitzSet> = (0..ng).map(|g| ritz_report(&format!("{name} group {g}"), &o.harvest[g])).collect();
                    let k0 = rs[0].kappa(1e-2);
                    ext = (rs[0].theta.last().copied().unwrap_or(f64::NAN), rs[0].theta.first().copied().unwrap_or(f64::NAN));
                    if !spec.deflates() && need_harvest.contains(&spec.base) {
                        harvests.insert(spec.base.clone(), (o.harvest.clone(), rs));
                    }
                    k0
                } else {
                    f64::NAN
                };
                let line = format!("pp {name} spec {}{} mu_rel {mu_rel:e} band {}: iters {} final rel {:.2e} conv {:?}; it to 1% / 0.1% of x̂ {:?} / {:?}; vs x̂ end {last_ref:.3e}; vs x★ end all {:.4} body {:.4} blood {:.4} air {:.4} corr {:.3}; best all {:.4} at it {}; kappa_ritz {kap:.3e}; ritz extremes {:.3e} / {:.3e} (ratio {:.3e}); x̂-tol it/s 1e-2 {} 1e-3 {} 1e-4 {}; setup {t_setup:.1}s solve {:.1}s; ms per P {ms_p:.1}, per Q {ms_q:.0}; {:.0}s",
                    format!("{}{}", spec.base, if spec.recentre { "+u" } else { "" }), if spec.deflates() { format!("+d{},{}", spec.top, spec.bot) } else { String::new() }, a.band, o.iters, o.rel.iter().cloned().fold(0.0, f64::max), o.converged, it1, it01, last[0], last[1], last[2], last[3], last[4], best.0, best.1, ext.0, ext.1, ext.0 / ext.1,
                    tol_hit.map(|h| h.map_or("–".to_string(), |(k, t)| format!("{k}/{t:.1}")))[0], tol_hit.map(|h| h.map_or("–".to_string(), |(k, t)| format!("{k}/{t:.1}")))[1], tol_hit.map(|h| h.map_or("–".to_string(), |(k, t)| format!("{k}/{t:.1}")))[2], t_solve.elapsed().as_secs_f64(), tr.elapsed().as_secs_f64());
                println!("{line}");
                writeln!(summary, "{line}").unwrap();
                if name.starts_with("ref") {
                    xref = Some(x.clone());
                    println!("  reference x̂ (μ_rel {mu_rel:e}) set from run {name}");
                }
                if ctl.stop_requested() {
                    println!("stopped in --pp-study (run {name}); rerun repeats the study");
                    std::process::exit(0);
                }
            }
        }
        println!("pp study done, {:.0}s", tp.elapsed().as_secs_f64());
        std::process::exit(0);
    }
    // ---- --pp-pool: the pooled start of --warm-full with this preconditioner -------------------------------------
    let mu_rel = a.mu.first().copied().unwrap_or(0.0);
    let mu_g: Vec<f64> = pr.lam_bar.iter().map(|&l| if mu_rel > 0.0 { mu_rel * l } else { 0.0 }).collect();
    let mut pool_harvest: Option<(Vec<ppre::Harvest>, Vec<ppre::RitzSet>)> = None;
    if let Some(spec) = &pool {
        assert!(!spec.deflates(), "--pp-pool: give the base only (the deflation is harvested from it)");
        let done: Vec<String> = (0..ne).map(|e| format!("warm_full_done_{e}")).collect();
        let have = done.iter().all(|d| ctl.load(d).unwrap().is_some());
        let need_h = metric.as_ref().is_some_and(|m| m.deflates());
        if have && !need_h {
            println!("pp pool: pooled start already done (warm_full_done_*), not repeated");
        } else {
            let syms: Option<Vec<SymPre>> = spec.sym().then(|| (0..ng).map(|g| { let (l, d) = pp_factors(&spec.base, g, &pr, mu_g[g], a, dims, support); SymPre::new(fft.clone(), l, Some(d)) }).collect());
            let rec: Option<Vec<Vec<C32>>> = spec.recentre.then(|| {
                let sh = seg_host.expect("+u: needs the stored segment tables (--model seg)");
                sel.iter().map(|&e| hasty_recon::surrogate::recentre_diag(&sh[e], nv, t_ref)).collect()
            });
            let ci: Vec<usize> = reps.iter().map(|&t| if p.circ_map.is_empty() { 0 } else { p.circ_map[t] }).collect();
            let mut sur_devs: Vec<hasty_recon::surrogate::SurDev> = vec![];
            if let Some((cheb, ls, _, met)) = spec.sur() {
                let sh = seg_host.expect("sur / cheb: needs the stored segment tables (--model seg)");
                let ncoil = a.sur_coils.unwrap_or(nvc).min(nvc).max(1);
                sur_devs = pp_build_sur(&mut *p, st, dev, kern, dims, &groups, &sel, sets, omega, op_coords, sh, &dcoil, smaps, ncoil, &g0, ls);
                for (g, d) in sur_devs.iter_mut().enumerate() {
                    d.mu = mu_g[g];
                    if cheb {
                        let m = if met { Some(&mut p.circ[ci[g]][0]) } else { None };
                        d.bounds(st, m, 12, a.cheb_ratio).unwrap();
                    }
                }
            }
            let base = match spec.sur() {
                Some((cheb, _, kk, met)) => PpBase::Sur { devs: &mut sur_devs[..], ci: ci.clone(), inner: if cheb { hasty_recon::surrogate::Inner::Cheb(kk) } else { hasty_recon::surrogate::Inner::Pcg(kk) }, metric: met },
                None => match spec.base.as_str() {
                    "none" => PpBase::Id,
                    "circ" => PpBase::Circ(reps.clone()),
                    _ => PpBase::Sym(&syms.as_ref().unwrap()[..]),
                },
            };
            let mut b = g0.clone();
            b.data.iter_mut().for_each(|v| *v = C32::new(-v.re, -v.im));
            if a.band {
                for g in 0..ng {
                    ppre::project_band(&fft, &ball, b.chunk_mut(g));
                }
            }
            let mut x = HostStack::zeros(&glens);
            let tw = Instant::now();
            let harvest = if need_h { a.harvest.max(8) } else { 0 };
            let (o, ms_p) = {
                let mut ops = PpOps { p: &mut *p, st: st.clone(), groups: &groups, g0: &g0, mu: mu_g.clone(), band: a.band.then(|| (fft.clone(), ball.clone())), base, defl: vec![None; ng], pre_s: 0.0, pre_n: 0, q_s: 0.0, q_n: 0, rec: rec.clone() };
                let mut cb = |k: usize, xs: &HostStack, rel: &[f64]| -> Result<bool, hasty_recon::Error> {
                    if k <= 3 || k % 5 == 0 {
                        let rmax = rel.iter().cloned().fold(0.0, f64::max);
                        match synth {
                            Some((xm, stat, bm)) => println!("  pp pooled it {k}: ‖r‖/‖r₀‖ max {rmax:.2e}; pooled image vs x★ frame mean: {}", pooled_image_error(xs, xm, stat, bm)),
                            None => println!("  pp pooled it {k}: ‖r‖/‖r₀‖ max {rmax:.2e}"),
                        }
                    }
                    Ok(true)
                };
                let o = ppre::pcg_groups(&mut ops, &b, &mut x, &PcgCfg { iters: warm_full, tol: a.tol, harvest, flexible: spec.flexible() }, &mut cb).unwrap();
                (o, 1e3 * ops.pre_s / ops.pre_n.max(1) as f64)
            };
            println!("pp pooled start ({}, μ_rel {mu_rel:e}{}): {} iterations, final ‖r‖/‖r₀‖ max {:.2e}, converged {:?}; {ms_p:.1} ms per preconditioner application; {:.0}s",
                spec.base, if a.band { ", band" } else { "" }, o.iters, o.rel.iter().cloned().fold(0.0, f64::max), o.converged, tw.elapsed().as_secs_f64());
            for (e, d) in done.iter().enumerate() {
                ctl.save(d, &hasty_recon::checkpoint::dev_value(st, &st.clone_htod(x.chunk(e)).unwrap()).unwrap()).unwrap();
            }
            if need_h {
                let rs: Vec<ppre::RitzSet> = (0..ng).map(|g| ritz_report(&format!("pool encode {g}"), &o.harvest[g])).collect();
                pool_harvest = Some((o.harvest, rs));
            }
        }
    }
    // ---- --pp-metric: the P²nP circulant metric from the probed symbol (+ deflation) ------------------------------
    if let Some(spec) = &metric {
        let tm = Instant::now();
        for g in 0..ng {
            let (mut lam, dinv) = pp_factors(&spec.base, g, &pr, mu_g[g], a, dims, support);
            let ci = p.circ_map[reps[g]];
            // device vs host on a random vector: the FFT conventions must agree (else the symbol is mirrored)
            let host = SymPre::new(fft.clone(), lam.clone(), Some(dinv.clone()));
            let xr = ppre::phases(nv, 77 + g as u64);
            let mut want = vec![C32::new(0.0, 0.0); nv];
            host.apply(&xr, &mut want);
            let check = |pc: &mut CirculantPrecond| -> f64 {
                let dx = st.clone_htod(&xr).unwrap();
                let mut z = st.alloc_zeros::<C32>(nv).unwrap();
                pc.apply(st, &dx, &mut z).unwrap();
                let got = st.clone_dtoh(&z).unwrap();
                let n: f64 = got.iter().zip(&want).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum();
                let d: f64 = want.iter().map(|q| (q.re as f64).powi(2) + (q.im as f64).powi(2)).sum();
                (n / d).sqrt()
            };
            let mut pc = CirculantPrecond::from_factors(dev, st, kern.clone(), dims, &lam, &dinv).unwrap();
            let err = check(&mut pc);
            if err > 1e-4 {
                let lm = ppre::mirror_symbol(&lam, dims);
                let mut pm = CirculantPrecond::from_factors(dev, st, kern.clone(), dims, &lm, &dinv).unwrap();
                let em = check(&mut pm);
                println!("  pp metric encode {}: device vs host {err:.2e}, mirrored symbol {em:.2e}: {}", sel[g], if em < err { "the device FFT has the other sign: symbol mirrored" } else { "MISMATCH (kept as is)" });
                if em < err {
                    pc = pm;
                    lam = lm;
                }
            }
            let _ = &lam;
            if spec.deflates() {
                let (hv, rs) = pool_harvest.as_ref().expect("--pp-metric deflation: no pool harvest");
                match rs[g].deflation(&hv[g], spec.top, spec.bot, 1e-2) {
                    Some(d) => {
                        println!("  pp metric encode {}: deflation rank {}, θ {:?} → {:?}", sel[g], d.rank(), d.theta.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>(), d.target.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>());
                        pc = pc.with_deflation(st, &d.u, &d.theta, &d.target).unwrap();
                    }
                    None => println!("  pp metric encode {}: no converged Ritz pair, no deflation", sel[g]),
                }
            }
            if let Some(ws) = circ_ws {
                pc = pc.to_host(st, ws).unwrap();
            }
            // ms per application (device)
            let dx = st.clone_htod(&xr).unwrap();
            let mut z = st.alloc_zeros::<C32>(nv).unwrap();
            st.synchronize().unwrap();
            let t5 = Instant::now();
            for _ in 0..5 {
                pc.apply(st, &dx, &mut z).unwrap();
            }
            st.synchronize().unwrap();
            println!("  pp metric encode {} ({}{}): {:.2} ms per device application, dynamic range {:.2e}", sel[g], spec.base, if pc.deflation_rank() > 0 { format!("+d{}", pc.deflation_rank()) } else { String::new() }, 1e3 * t5.elapsed().as_secs_f64() / 5.0, pc.dyn_range);
            p.circ[ci] = vec![pc];
        }
        println!("pp metric: per-encode circulants replaced ({:.0}s)", tm.elapsed().as_secs_f64());
    }
    println!("pp setup total {:.0}s", tp.elapsed().as_secs_f64());
}

/// `--temporal` scoring weights (nf × nf, row-major): W[f][k] = ∫_{f−½}^{f+½} B_p(t − k) dt, the box average of the
/// periodic B-spline model over frame f (knot k centred at t = k, `hasty_recon::vmodel::bspline`, as the engine's
/// kernels), by the midpoint rule on 2000 points per frame (exact to ~1e-7 for these piecewise polynomials). Rows sum
/// to 1 (partition of unity).
fn temporal_box_weights(nf: usize, p: u32) -> Vec<f64> {
    let nq = 2000;
    let mut w = vec![0.0f64; nf * nf];
    for f in 0..nf {
        for q in 0..nq {
            let t = f as f64 - 0.5 + (q as f64 + 0.5) / nq as f64;
            for k in 0..nf {
                // periodic: the nearest image of knot k
                let mut d = t - k as f64;
                d -= (d / nf as f64).round() * nf as f64;
                w[f * nf + k] += hasty_recon::vmodel::bspline(p, d) / nq as f64;
            }
        }
        let sum: f64 = w[f * nf..(f + 1) * nf].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "temporal_box_weights: row {f} sums to {sum}");
    }
    w
}
