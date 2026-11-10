//! Time-resolved P²nP-ISTA reconstruction of the synthetic spiral 4D-flow data (frames × encodes), no density
//! compensation, whitened + coil-compressed data, LLR prior over all frames × encodes or the learned 7-point prior.
//! Any encoding of the simulator (referenced / Hadamard 4-point, 7-point dual-VENC): velocities by hasty_recon::flow.
//!   recon_p2np <sim_dir> [--model sense|seg] [--temporal off|linear|cubic] [--phys none|b0|full] [--gnl]
//!              [--nseg L|auto] [--order o] [--precond none,circ,zmshr1,zmshr1m] [--iters n] [--kappa k,k,..] [--block b]
//!              [--vcoils auto|n] [--coil-energy 0.999] [--coil-gram calib|data] [--every n] [--warm n] [--eps e] [--conc-dc] [--synthetic] [--out dir]
//!              [--learned <export_dir> [--tile n] [--keep-frames n] [--learned-prec bf16|fp16|fp32] [--learned-cpu]]
//!              [--fresh] [--ckpt-every n] [--fat-fraction <npy|truth> [--fat-model shared|separate|joint]]
//!              [--conc-fit quad|deg4] [--table-prefetch]
//!              [--field truth|est:<recon_b0 run>[:bin]] [--r2s off|truth|est:<run>] [--smaps truth|espirit[:eig_min]]
//!              [--coil-sampling keep:draw[:hold[:last[:seed]]]] [--lagged TAU[:LS|auto]] [--prior llr|wavelet] [--wavelet THR[:LEVELS]] [--wavelet-mix β] [--dfw sure[:TDF:TN]|fixed:TDF:TN [--dfw-levels L] [--dfw-mix β]] [--static-lock MU[:IT0[:EVERY]] [--lock-mode quad|group] [--lock-blur MM] [--lock-dilate MM] [--lock-score LO:HI] [--lock-vmin M/S] [--lock-detect OK:MAX]] [--state auto|device|host] [--resp-bins NR [--resp bellows|truth]] [--seg-basis fixed|svd|svdw] [--seg-weight count|mass[:lo,hi]] [--kstore auto|legacy|<store spec>]
//!              [--table-store auto|pinned|lz4|zstd|zstdN]   (env HASTY_KSTORE_DIR: k-space store directory, default <sim>/kstore)
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

/// bytes of RAM this process can still use: MemAvailable, capped by the cgroup's memory.max (systemd-run scopes,
/// batch schedulers) minus its current use
fn available_memory() -> usize {
    let mut avail = std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<usize>().ok())).map_or(usize::MAX, |kb| kb * 1024);
    if let Some(cg) = std::fs::read_to_string("/proc/self/cgroup").ok().and_then(|s| s.lines().find_map(|l| l.strip_prefix("0::").map(|p| p.trim().to_string()))) {
        let base = std::path::Path::new("/sys/fs/cgroup").join(cg.trim_start_matches('/'));
        let rd = |f: &str| std::fs::read_to_string(base.join(f)).ok().and_then(|v| v.trim().parse::<usize>().ok());
        if let (Some(max), Some(cur)) = (rd("memory.max"), rd("memory.current")) {
            avail = avail.min(max.saturating_sub(cur));
        }
    }
    avail
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
    let learned_cpu = flag("--learned-cpu");
    let table_prefetch = flag("--table-prefetch");
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
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
    let nseg_arg: Option<usize> = take("--nseg").and_then(|s| if s == "auto" { None } else { Some(s.parse().unwrap()) });
    let mut nseg: usize = nseg_arg.unwrap_or(if phys == "none" { 1 } else { 0 });
    let order: u32 = take("--order").map(|s| s.parse().unwrap()).unwrap_or(if phys == "full" { 2 } else { 0 });
    // relative floor of the circulant preconditioner's Fourier diagonal (larger = weaker, better-posed metric)
    let eps: f64 = take("--eps").map(|s| s.parse().unwrap()).unwrap_or(1e-3);
    let preconds: Vec<String> = take("--precond").unwrap_or("zmshr1".into()).split(',').map(|s| s.to_string()).collect();
    let out_arg = take("--out");
    let ckpt_every: usize = take("--ckpt-every").map(|s| s.parse().unwrap()).unwrap_or(10);
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
    let fat_fraction: Option<String> = take("--fat-fraction");
    // measured instead of true fields / coil maps (the oracle-vs-estimate evaluation; defaults = the oracle as before):
    // --field truth|est:<recon_b0 run>[:bin]: off-resonance ω; --r2s off|truth|est:<run>[:bin]: T2* decay during the
    // readout folded into the B0 segments; --smaps truth|espirit[:eig_min]: ESPIRiT maps from the compressed data of
    // encode 0 (hasty_recon::espirit); --fat-fraction est:<run>[:bin] takes the fat fraction of the B0 scan
    let field_spec = take("--field").unwrap_or("truth".into());
    let r2s_spec = take("--r2s").unwrap_or("off".into());
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
    assert!(matches!(prior_kind.as_str(), "llr" | "wavelet"), "--prior llr|wavelet");
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
    let resp_bins: usize = take("--resp-bins").map_or(1, |v| v.parse().expect("--resp-bins N"));
    let resp_src = take("--resp").unwrap_or("bellows".into());
    let seg_basis_arg = take("--seg-basis").unwrap_or("fixed".into());
    // --seg-weight count|mass[:lo,hi]: voxel weights of the SVD basis, all data-derived (no labels): count = every voxel
    // of the warm-start support equally (default); mass = the smoothed warm-start magnitude relative to its median over
    // the support, clamped to [lo, hi] (default 0.3, 3): bright regions count more, dark tissue keeps a floor so the
    // basis is not biased away from it, air (outside the support) never counts
    let seg_weight = take("--seg-weight").unwrap_or("count".into());
    let seg_mass: Option<(f32, f32)> = match seg_weight.split(':').next().unwrap() {
        "count" => None,
        "mass" => Some(seg_weight.split(':').nth(1).map_or((0.3, 3.0), |b| {
            let v: Vec<f32> = b.split(',').map(|x| x.parse().expect("--seg-weight mass:lo,hi")).collect();
            (v[0], v[1])
        })),
        o => panic!("--seg-weight count|mass[:lo,hi], got {o}"),
    };
    let (seg_svd, seg_svdw) = match seg_basis_arg.as_str() {
        "fixed" => (false, false),
        "svd" => (true, false),
        "svdw" => (true, true),
        o => panic!("--seg-basis fixed|svd|svdw, got {o}"),
    };
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
    let cmdline = std::env::args().collect::<Vec<_>>().join(" ");
    let dir = PathBuf::from(&args[0]);
    assert!(model == "seg" || temporal.is_none(), "--temporal needs --model seg");
    // run output directory: arrays, metrics and figures (tools/figs/run_figs.py)
    let out = out_arg.map(PathBuf::from).unwrap_or_else(|| {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let tag = format!("{name}_{model}{}_{phys}{}{}", temporal.map_or(String::new(), |b| format!("_t{b}")), if gnl { "_gnl" } else { "" }, if conc_hi { "_deg4" } else { "" }) + if fat_fraction.is_some() { "_fat" } else { "" } + if fat_fraction.is_some() && fat_model != "shared" { fat_model.as_str() } else { "" }
            + if field_spec != "truth" { "_fest" } else { "" } + if r2s_spec != "off" { "_r2s" } else { "" } + if smaps_spec != "truth" { "_espirit" } else { "" }
            + &coil_sampling.as_ref().map_or(String::new(), |c| format!("_cs{}-{}", c.keep, c.draw));
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/runs").join(tag)
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
    println!("output directory {} (checkpoint every {ckpt_every} iterations; Ctrl-C or touch {}/STOP to pause)", out.display(), out.display());
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
    if resp_bins > 1 {
        assert!(temporal.is_none() && vmodel.is_none(), "--resp-bins with --temporal / --vmodel: not yet (their time bases are cyclic over the cardiac frames)");
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
        let bins = hasty_recon::selfgate::amplitude_bins(&amp, resp_bins);
        let mut mean = vec![(0.0f64, 0usize); resp_bins];
        for i in 0..n_ro {
            sd.ro.frame[i] += bins[i] * nf_card;
            mean[bins[i]].0 += amp[i];
            mean[bins[i]].1 += 1;
        }
        sd.n_frames = resp_bins * nf_card;
        let means: Vec<f64> = mean.iter().map(|(a, n)| a / (*n).max(1) as f64).collect();
        println!("respiratory bins: {resp_bins} by {resp_src} amplitude (mean {:?}) × {nf_card} cardiac frames = {} frames", means.iter().map(|m| (m * 1000.0).round() / 1000.0).collect::<Vec<_>>(), sd.n_frames);
        resp_info = serde_json::json!({ "bins": resp_bins, "source": resp_src, "mean_amplitude": means, "cardiac_frames": nf_card });
    }
    let sd = sd;
    let (nv, nc, nf) = (sd.nv(), sd.nc, sd.n_frames);
    // encodes (4-point referenced / Hadamard, 7-point dual-VENC, ...) and their velocity decoder
    let dec = FlowDecoder::new(&sd.encode_m1);
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
            let r: Vec<f64> = (0..nv).map(|q| sd.sample_log(&t, 0, &grid.x_log[q])).collect();
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
    assert!(!(staged && synthetic), "--synthetic writes the data on the device: not with staged data");
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
    let warm: Vec<Vec<C32>> = if warm_iters > 0 {
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
    println!("warm start: {} PCGLS iterations per encode ({:.0}s)", warm_iters, t0.elapsed().as_secs_f64());
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
        // smooth (3 box passes, radius 4 in-plane / 1 through-plane, σ ≈ 4.5 / 1.4 voxels) so that dark structures
        // inside the body (vessel lumen with flow dephasing, lungs) do not open holes in the mask
        let d = sd.dims;
        for (axis, r) in [(0usize, 4usize), (1, 4), (2, 1)] {
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
    let mut dil = support.clone();
    {
        let d = sd.dims;
        for _ in 0..3 {
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
    if std::env::var("HASTY_NO_MASK").is_err() {
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
    let lumen: Vec<usize> = (0..nv).filter(|&q| sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) > 0.9).collect();
    println!("lumen {} voxels: {} in the support, {} in the masked maps", lumen.len(), lumen.iter().filter(|&&q| support[q]).count(), lumen.iter().filter(|&&q| dil[q]).count());
    let dims4 = |n: usize| [sd.dims[0], sd.dims[1], sd.dims[2], n];
    save(&out.join("smaps.npy"), "<c8", &dims4(nvc), c8(&smaps));
    save(&out.join("mask_support.npy"), "<f4", &sd.dims, f4(support.iter().map(|&b| b as u8 as f32)));
    save(&out.join("mask_maps.npy"), "<f4", &sd.dims, f4(dil.iter().map(|&b| b as u8 as f32)));
    if !warm.is_empty() {
        save(&out.join("warm.npy"), "<c8", &dims4(ne), c8(&warm.concat()));
    }
    // fixed circulant preconditioners per frame (block-diagonal over the frames of a time-resolved set)
    // (the velocity model uses its own Gauss–Newton metric: no circulant kernels, they would cost ~8 GB)
    let mut lag_tau: Option<f64> = None;
    let mut lag_info = serde_json::Value::Null;
    let need_circ = vmodel.is_none() && preconds.iter().any(|p| p == "circ" || p == "zmshr1m");
    let circ: Vec<Vec<CirculantPrecond>> = if need_circ {
        host_sets
            .iter()
            .map(|s| {
                let frames: Vec<Option<usize>> = if temporal.is_some() { (0..nf).map(Some).collect() } else { vec![None] };
                frames
                    .iter()
                    .map(|&f| {
                        let sel: Vec<usize> = (0..s.readouts.len()).filter(|&r| f.is_none_or(|f| sd.ro.frame[s.readouts[r]] == f)).collect();
                        let c: [Vec<f32>; 3] = [0, 1, 2].map(|a| sel.iter().flat_map(|&r| s.coords[a][r * sd.na..(r + 1) * sd.na].iter().copied()).collect());
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
            let ff = match &ff_est { Some(v) => v.clone(), None => hasty_recon::data::read_npy(&path).unwrap_or_else(|e| panic!("--fat-fraction {}: {e:?}", path.display())).f32s() };
            assert!(ff.len() == nv || ff.len() == nv * nf, "--fat-fraction: {} values for {nv} voxels (or {nv} × {nf} frames)", ff.len());
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
        let wbar: Vec<Vec<[f64; 4]>> = (0..ne).map(|e| mean_coefficients(&(0..fe.len()).filter(|&t| enc_of(t) == e).map(|t| &host_sets[t]).collect::<Vec<_>>(), sd.na)).collect();
        let wbar: Vec<Vec<[f64; 4]>> = if with_conc { wbar } else { vec![vec![[0.0; 4]; sd.na]; ne] };
        if nseg == 0 {
            // phase cycles spanned over the ADC window by the fitted voxels (B0 and the mean concomitant phase), worst encode
            let tau_adc: Vec<f64> = (0..sd.na).map(|n| host_sets[0].tau[n]).collect();
            let (t0, t1) = (tau_adc[0], tau_adc[sd.na - 1]);
            let mut cyc = 0.0f64;
            for e in 0..ne {
                let psi = segment_phase(&grid, &wbar[e], &tau_adc, with_b0);
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
        let tau_adc: Vec<f64> = (0..sd.na).map(|n| host_sets[0].tau[n]).collect();
        let seg_fit: Vec<usize> = fit_vox.iter().copied().step_by(3).collect();
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
            e[(((t - tau_adc[0]) / dt).round().max(0.0) as usize).min(e.len() - 1)]
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
        assert!(seg_mass.is_none() || seg_svd, "--seg-weight mass needs --seg-basis svd|svdw");
        let svd_build = |nseg: usize, g: &(dyn Fn(usize, f64) -> Complex64 + Sync)| build_segments_svd_weighted(nv, nseg, tau_adc[0], tau_adc[sd.na - 1], sd.na.max(256), &seg_fit, seg_vw.as_deref(), tw, 96, g);
        let make_segs = |eng: &SegNufft, nseg: usize| -> Vec<hasty_nufft::SegSegments> {
        (0..ne)
            .map(|e| {
                let psi = segment_phase(&grid, &wbar[e], &tau_adc, with_b0);
                // one interpolator table entry per ADC sample (256 left a 3–7e-4 floor with fat at 8–12 ms readouts)
                // T2* decay relative to the readout start (--r2s; 1 without)
                let dec = |v: usize, t: f64| -> f64 { r2v.as_ref().map_or(1.0, |r| (-r[v] * (t - tau_adc[0])).exp()) };
                if let Some(ff) = &ff_joint {
                    if ff.len() == nv {
                        // joint, one map: folded into the diagonals, table fitted to g_v(τ) = [(1 − ff) + ff c(τ)] e^{−iψ}
                        let g = |v: usize, t: f64| Complex64::from_polar(dec(v, t), -psi(v, t)) * (Complex64::new(1.0 - ff[v] as f64, 0.0) + ff[v] as f64 * cfat(t));
                        let (sp, at, ta, dt) = if seg_svd { svd_build(nseg, &g) } else { build_segments_complex(nv, nseg, tau_adc[0], tau_adc[sd.na - 1], sd.na.max(256), &seg_fit, &g) };
                        return eng.segments(&st, &sp, &at, ta, dt).unwrap();
                    }
                    // joint, per-frame maps: water diagonals + an ff-independent table, maps applied by the operator
                    let (sp, at, cl, ta, dt) = build_segments_fat(nv, nseg, tau_adc[0], tau_adc[sd.na - 1], sd.na.max(256), &seg_fit, &psi, &cfat);
                    return eng.segments_fat(&st, &sp, &at, &cl, ta, dt).unwrap();
                }
                if r2v.is_some() {
                    let g = |v: usize, t: f64| Complex64::from_polar(dec(v, t), -psi(v, t));
                    let (sp, at, ta, dt) = if seg_svd { svd_build(nseg, &g) } else { build_segments_complex(nv, nseg, tau_adc[0], tau_adc[sd.na - 1], sd.na.max(256), &seg_fit, &g) };
                    return eng.segments(&st, &sp, &at, ta, dt).unwrap();
                }
                let (sp, at, ta, dt) = if seg_svd {
                    svd_build(nseg, &|v, t| Complex64::from_polar(1.0, -psi(v, t)))
                } else {
                    build_segments(nv, nseg, tau_adc[0], tau_adc[sd.na - 1], sd.na.max(256), &seg_fit, &psi)
                };
                eng.segments(&st, &sp, &at, ta, dt).unwrap()
            })
            .collect()
        };
        let segs = make_segs(&eng, nseg);
        let mut resid = 0.0f64;
        // sample tables per set: resident, or (they would crowd the device / --plan host) pinned host memory streamed
        // into two device slots; decided from set 0's measured size
        let mut tables_host: Option<bool> = None;
        let mut table_pack: Option<Option<hasty_value::Options>> = None;
        let samples_all: usize = host_sets.iter().map(|s| s.tau.len()).sum();
        let pts: Vec<_> = (0..fe.len())
            .map(|t| {
                let mut p = (|| {
                if let (Some(f4), Some(h)) = (&fit4, &htab) {
                    let (tab, dw) = seg_tables_hi(&sd, &host_sets[t], f4, with_conc, with_conc.then(|| &wbar[enc_of(t)][..]));
                    resid = resid.max(tab.max_resid_phase);
                    return eng.points_hi(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, &dw, h, temporal.map(|_| &tab.t[..])).unwrap();
                }
                let tab = seg_tables(&sd, &host_sets[t], &fit, with_conc, with_conc.then(|| &wbar[enc_of(t)][..]));
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
            let ff = match &ff_est { Some(v) => v.clone(), None => hasty_recon::data::read_npy(&path).unwrap_or_else(|e| panic!("--fat-fraction {}: {e:?}", path.display())).f32s() };
            assert!(ff.len() == nv || ff.len() == nv * nf, "--fat-fraction: {} values for {nv} voxels (or {nv} × {nf} frames)", ff.len());
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
    // model-consistent test data: y_t = E_t x_true,t + noise (solver + prior test without simulator physics)
    let mut ys = ys;
    if synthetic {
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
        let quasi = preconds.iter().any(|p| p == "zmshr1" || p == "zmshr1m");
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
        info["segments"] = serde_json::json!({ "basis": seg_basis_arg, "weight": seg_weight }); info["respiration"] = resp_info.clone(); info["prior_detail"] = serde_json::json!({ "prior": prior_kind, "wavelet": { "thr": wav_thr, "levels": wav_levels, "mix": wav_mix } });
        info["prior_detail"]["dfw"] = serde_json::json!(dfw_rule.map(|r| serde_json::json!({ "rule": format!("{r:?}"), "levels": dfw_levels, "mix": dfw_mix })));
        assert!(static_lock.is_none(), "--static-lock: complex-image models only (not --vmodel)");
    std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
        let rt = transpose(&sd.rot);
        for (pc, kap) in preconds.iter().flat_map(|p| vkappas.iter().map(move |&k| (p, k))) {
            let tag = vtag(pc, kap);
            let precond = match pc.as_str() { "none" => Precond::None, "circ" => Precond::Circulant, "zmshr1" => Precond::Zmshr1, _ => Precond::Zmshr1M };
            let with_metric = matches!(precond, Precond::Circulant | Precond::Zmshr1M);
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
        let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/figs/run_figs.py");
        let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
        match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
            Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
            r => println!("figure script failed ({r:?}); run: {} -I {} {}", py.display(), tools.display(), out.display()),
        }
        return;
    }
    let mut p = if staged {
        let sg = hasty_recon::kstage::Stager::new(&dev.ctx, kstore.clone().unwrap()).unwrap();
        println!("data staged from the store: {} pinned staging buffers of {:.2} GiB", 2, sg.pinned_bytes() as f64 / 2.0 / GIB);
        P2np::from_store(sop, sg, circ, &st).unwrap()
    } else {
        P2np::new(sop, ys.iter().collect(), circ, &st).unwrap()
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
    let ttag = temporal.map_or(String::new(), |b| format!("_t{b}"));
    let runs: Vec<String> = preconds.iter().flat_map(|p| kappas.iter().map(|k| (p.clone(), *k)).collect::<Vec<_>>()).map(|(p, k)| format!("{model}{ttag}_{p}{}_k{k}", if learned.is_some() { "_learned" } else { "" })).collect();
    let mut info = serde_json::json!({
        "cmdline": cmdline, "sim_dir": dir, "model": model, "temporal": temporal, "phys": phys, "gnl": gnl, "nseg": nseg,
        "order": order, "conc_fit": conc_fit, "fat_model": fat_fraction.as_ref().map(|_| fat_model.clone()), "eps": eps, "iters": iters, "warm_iters": warm_iters, "block": block, "vcoils": nvc, "dims": sd.dims, "vox": sd.vox, "native_dims": sd.tdims,
        "n_frames": nf, "n_enc": ne, "venc": sd.venc, "venc_range": dec.range(), "encoding": sd.meta["encoding"], "lambda_max": lmax, "lambda_max_mn": lmax_c, "setup_s": t0.elapsed().as_secs_f64(),
        "noise_sigma_env": std::env::var("HASTY_NOISE_SIGMA").ok(), "runs": runs,
        "prior": if learned.is_some() { "learned" } else { "llr" }, "learned_dir": learned_dir, "tile": tile, "keep_frames": keep_frames, "learned_prec": learned_prec,
        "field": field_spec, "r2s": r2s_spec, "smaps": smaps_spec, "coils": coil_info, "estimates": diag, "kdata": kstore_info, "lagged": lag_info,
    });
    info["segments"] = serde_json::json!({ "basis": seg_basis_arg, "weight": seg_weight }); info["respiration"] = resp_info.clone(); info["prior_detail"] = serde_json::json!({ "prior": prior_kind, "wavelet": { "thr": wav_thr, "levels": wav_levels, "mix": wav_mix } });
    info["prior_detail"]["dfw"] = serde_json::json!(dfw_rule.map(|r| serde_json::json!({ "rule": format!("{r:?}"), "levels": dfw_levels, "mix": dfw_mix })));
    info["prior_detail"]["static_lock"] = serde_json::json!(static_lock.as_ref().map(|(o, it0, ev)| serde_json::json!({ "opts": format!("{o:?}"), "it0": it0, "every": ev })));
    std::fs::write(out.join("run.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();
    let rt = transpose(&sd.rot);
    for (pc, kappa) in preconds.iter().flat_map(|p| kappas.iter().map(move |&k| (p, k))) {
        let (precond, alpha) = match pc.as_str() {
            "none" => (Precond::None, 1.0 / lmax),
            "circ" => (Precond::Circulant, 1.0 / lmax_c),
            "zmshr1m" => (Precond::Zmshr1M, 1.0 / lmax_c),
            _ => (Precond::Zmshr1, 1.0 / lmax),
        };
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
        if !warm.is_empty() {
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
        let mut wav = hasty_recon::wavelet::Wavelet::new(sd.dims, wav_levels, wav_thr);
        let nimg = nf * ne;
        let tr = Instant::now();
        let (mut td, mut tn) = (0.0f64, 0usize);
        let mut denoise_prior = |k: usize, h: &mut Vec<C32>| {
            if kappa <= 0.0 {
                return;
            }
            let Some(prior) = &learned else {
                if prior_kind == "wavelet" {
                    // x ← x + κ (W(x) − x)
                    wav.set_iteration(k as u64);
                    let mut w = h.clone();
                    wav.apply(&mut w, nimg);
                    let kf = kappa.min(1.0) as f32;
                    { use rayon::prelude::*; h.par_iter_mut().zip(w.par_iter()).for_each(|(x, d)| *x = C32::new(x.re + (d.re - x.re) * kf, x.im + (d.im - x.im) * kf)); }
                } else {
                    llr.set_iteration(k as u64);
                    llr.apply(h, nimg);
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
            _ => String::new(),
        };
        let tag = format!("{model}{}_{pc}{ptag}_k{kappa}{dfw_tag}{lock_tag}", temporal.map_or(String::new(), |b| format!("_t{b}")));
        let resuming = out.join(format!("state_{tag}.hv")).exists();
        if !resuming && out.join(format!("recon_{tag}.npy")).exists() {
            println!("  {tag}: finished in an earlier run (recon_{tag}.npy), skipped (--fresh to redo)");
            continue;
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
        let (mut tdf, mut ndf, mut tlk, mut nlk) = (0.0f64, 0usize, 0.0f64, 0usize);
        let mut denoise = |k: usize, h: &mut Vec<C32>| {
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
        };
        // metrics / outputs of iteration k from per-set host images (device stacks are downloaded by the wrapper below)
        let report_due = |k: usize| { let k = k.wrapping_add(1); !(k % every != 0 && k != iters && k != 0) };
        let mut report_imgs = |k: usize, imgs: &[&[C32]], f: f64| {
            let k = k.wrapping_add(1);
            if k % every != 0 && k != iters && k != 0 {
                return;
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
            let mut per_frame = Vec::with_capacity(nf);
            for fr in 0..nf {
                let (mut fe2, mut fn_) = (0.0, 0usize);
                for &q in &lumen {
                    let v = vel(fr, q);
                    for a in 0..3 {
                        let t = sd.sample_log(&sd.truth_vel, a + 3 * (fr % nf_card), &grid.x_log[q]);
                        e2 += (v[a] - t).powi(2);
                        t2 += t * t;
                        fe2 += (v[a] - t).powi(2);
                    }
                    n += 1;
                    fn_ += 1;
                }
                per_frame.push((fe2 / fn_.max(1) as f64).sqrt());
            }
            {
                use std::io::Write;
                writeln!(csv, "{k},{f:.6e},{:.5},{:.5},{:.1},{}", (e2 / n as f64).sqrt(), (e2 / t2).sqrt(), tr.elapsed().as_secs_f64(), per_frame.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(",")).unwrap();
            }
            if k == iters {
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
        let outcome = if host_state {
            let mut report = |k: usize, xs: &hasty_recon::hoststack::HostStack, f: f64| {
                let r: Vec<&[C32]> = (0..xs.nsets()).map(|t| xs.chunk(t)).collect();
                report_imgs(k, &r, f)
            };
            hasty_recon::hoststack::run_objective_host(&mut p, &st, &mut xh, precond, alpha, iters, &mut denoise, &mut report, Some((&ctl, &tag))).unwrap()
        } else {
            let mut report = |k: usize, xs: &Stack, f: f64| {
                if !report_due(k) {
                    return;
                }
                let h: Vec<Vec<C32>> = xs.v.iter().map(|v| st.clone_dtoh(v).unwrap()).collect();
                let r: Vec<&[C32]> = h.iter().map(|v| v.as_slice()).collect();
                report_imgs(k, &r, f)
            };
            p.run_from(&st, &mut x, precond, alpha, iters, &mut denoise, &mut report, Some((&ctl, &tag))).unwrap()
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
    let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/figs/run_figs.py");
    let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
        Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
        r => println!("figure script failed ({r:?}); run: {} -I {} {}", py.display(), tools.display(), out.display()),
    }
}
