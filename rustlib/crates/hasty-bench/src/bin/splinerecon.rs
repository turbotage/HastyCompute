//! Respiratory × cardiac knot reconstruction (research/temporal_operator.md §16 ff.): per encode, the knot images of a
//! 2-D basis (`SegNufft::new_temporal2d`): respiratory bins (degree 0) or linear B-splines (degree 1) in the breathing
//! coordinate × periodic B-splines in cardiac time, sample-level weights, B0 time segments + concomitant + GNL physics,
//! SV-compressed coils; CGLS (optionally knot-preconditioned, warm-started) and the solver arms (--arms, smoothed
//! wavelet + LLR priors). The knot images are evaluated at given breathing states r* and the 20 cardiac frame centres
//! and written at the truth voxels of each state (frame_fit.py lumen form, FRAME_FIT_TRUTH=<truth npz>).
//!
//!   splinerecon <scan dir> [--rknots 6] [--rdeg 0|1 (1)] [--cknots 16] [--cdeg 3] [--iters 20] [--save-at 5,10,20]
//!       [--vcoils auto|n] [--coil-energy 0.999] [--coils N (legacy: N strongest physical coils)] [--nseg 10]
//!       [--width 8] [--resp bellows|truth] [--phys conc|b0] [--no-gnl] [--conc-fit deg4|quad]
//!       --eval r1:idx1.npy,r2:idx2.npy,... --out <prefix>
//!
//! Defaults = the target config (const TARGET + SV coils, conc, GNL, deg4); the header prints every deviation.
//! Breathing coordinate: the bellows amplitude (physio.npy column 3, default) or the sim's true state (resp.npy, oracle).
//! Bins: equal-count amplitude bins (as recon_p2np --resp-bins); r* evaluates the bin whose amplitude range holds it.
//! Linear: K knots at the amplitude quantiles 0, 1/(K−1), …, 1; r* evaluates the hat weights (clamped at the ends).
//! Note: with --resp bellows, --eval r* values are in bellows-amplitude units (the knots sit at its quantiles).
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts, SegPoints, SegSegments};
use hasty_recon::cg::pcgls_cb;
use hasty_recon::data::{read_npy, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::setop::LinOp;
use hasty_recon::Error;

struct Op2d<'a> {
    eng: &'a std::cell::RefCell<SegNufft>,
    /// (forward, adjoint) passes so far: the gradient count of the solver arms
    cnt: &'a std::cell::Cell<(usize, usize)>,
    seg: &'a SegSegments,
    pts: &'a SegPoints,
    kern: Arc<Kernels>,
    nx: usize,
    ny: usize,
}

impl LinOp for Op2d<'_> {
    fn len_x(&self) -> usize {
        self.nx
    }
    fn len_y(&self) -> usize {
        self.ny
    }
    fn kernels(&self) -> &Kernels {
        &self.kern
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let c = self.cnt.get();
        self.cnt.set((c.0 + 1, c.1));
        Ok(self.eng.borrow_mut().forward(st, self.seg, self.pts, x, y)?)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let c = self.cnt.get();
        self.cnt.set((c.0, c.1 + 1));
        Ok(self.eng.borrow_mut().adjoint(st, self.seg, self.pts, y, x)?)
    }
}

/// The static image s through the 2-D operator: A(1 ⊗ s) = A_static s (the basis weights sum to 1 per sample), adjoint
/// = the sum of the knot images of Aᴴ r. `big` is a knot-stack scratch buffer.
struct WarmOp<'a> {
    inner: &'a mut dyn LinOp,
    big: &'a mut CudaSlice<C32>,
    tmp: CudaSlice<C32>,
    nv: usize,
}

impl LinOp for WarmOp<'_> {
    fn len_x(&self) -> usize {
        self.nv
    }
    fn len_y(&self) -> usize {
        self.inner.len_y()
    }
    fn kernels(&self) -> &Kernels {
        self.inner.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let nk = self.inner.len_x() / self.nv;
        for k in 0..nk {
            st.memcpy_dtod(x, &mut self.big.slice_mut(k * self.nv..(k + 1) * self.nv))?;
        }
        self.inner.forward(st, self.big, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let nk = self.inner.len_x() / self.nv;
        self.inner.adjoint(st, y, self.big)?;
        st.memset_zeros(x)?;
        let one = num_complex::Complex64::new(1.0, 0.0);
        for k in 0..nk {
            st.memcpy_dtod(&self.big.slice(k * self.nv..(k + 1) * self.nv), &mut self.tmp)?;
            self.inner.kernels().axpby(st, one, &self.tmp, one, x)?;
        }
        Ok(())
    }
}

/// complex64 C-order output (lumen values): `hasty_value::store` image policy (1e-4 relative rms,
/// research/io_policy.md) → `<name>.hv` (`HASTY_IO=npy`: .npy)
fn write_c8_c(path: &std::path::Path, shape: &[usize], data: &[C32]) {
    // SAFETY: C32 is two f32, repr(C)
    let b = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    hasty_value::store::save_bytes(path, hasty_value::DType::C64, shape, hasty_value::Order::C, b, &hasty_value::Save::image()).unwrap();
}

/// centred periodic B-spline weight of knot b at time t (knot units, period m), degree 1 or 3
fn bweight(t: f64, b: usize, m: usize, deg: u32) -> f64 {
    let n = m as f64;
    let mut d = (t - b as f64).rem_euclid(n);
    if d > 0.5 * n {
        d -= n;
    }
    let a = d.abs();
    match deg {
        // degree 0 as the kernels: sample time t in [b, b + 1) belongs to knot b
        0 => ((t.rem_euclid(n)).floor() as usize % m == b) as u8 as f64,
        1 => (1.0 - a).max(0.0),
        _ if a <= 1.0 => 2.0 / 3.0 - a * a + 0.5 * a * a * a,
        _ if a <= 2.0 => (2.0 - a).powi(3) / 6.0,
        _ => 0.0,
    }
}

/// The target reconstruction config (research/temporal_operator.md § Target config); the drivers' defaults
struct Target {
    rknots: usize,
    rdeg: u32,
    cknots: usize,
    cdeg: u32,
    resp: &'static str,
}
const TARGET: Target = Target { rknots: 6, rdeg: 1, cknots: 16, cdeg: 3, resp: "bellows" };

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // flags without a value (before `take` borrows args)
    let no_gnl = args.iter().position(|a| a == "--no-gnl").map(|i| { args.remove(i); true }).unwrap_or(false);
    if let Some(i) = args.iter().position(|a| a == "--gnl") {
        args.remove(i);
    }
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    // defaults = the target reconstruction config (research/temporal_operator.md § Target config): 6 linear
    // respiratory knots × 16 cubic cardiac knots, bellows breathing coordinate, SV coil compression (≥ 99.9 % energy),
    // B0 + concomitant (degree-4 fit at the GNL positions) + GNL. Every deviation is printed in the header line.
    let kr: usize = take("--rknots").map_or(TARGET.rknots, |v| v.parse().unwrap());
    let rdeg: u32 = take("--rdeg").map_or(TARGET.rdeg, |v| v.parse().unwrap());
    let mc: usize = take("--cknots").map_or(TARGET.cknots, |v| v.parse().unwrap());
    let cdeg: u32 = take("--cdeg").map_or(TARGET.cdeg, |v| v.parse().unwrap());
    let iters: usize = take("--iters").map_or(20, |v| v.parse().unwrap());
    let save_at: Vec<usize> = take("--save-at").map_or(vec![iters], |s| s.split(',').map(|v| v.parse().unwrap()).collect());
    // coils: --coils N = the N strongest physical coils (legacy test bench, no compression); default: all coils,
    // whitened and SV-compressed (--vcoils auto|n, --coil-energy, hasty_recon::coils, Gram of the whitened data)
    let ncoil_sel: Option<usize> = take("--coils").map(|v| v.parse().unwrap());
    let coil_energy: f64 = take("--coil-energy").map_or(hasty_recon::coils::CoilRule::DEFAULT_ENERGY, |v| v.parse().unwrap());
    let coil_rule = hasty_recon::coils::CoilRule::parse(&take("--vcoils").unwrap_or("auto".into()), coil_energy).unwrap();
    // --coil-gram data (default: the whitened encode-0 samples, ≤ 2·10⁵) | calib (the low-resolution calibration-region
    // coil images of encode 0, density-neutral; as recon_p2np) | box[:MARGIN_MM] (those images inside the prescription
    // box + margin, default 10 mm: virtual coils energy-ordered for the box, research/coil_cost_reduction.md §7.1); the
    // box from --presc-box S|SX,SY,SZ[@CX,CY,CZ] (mm) or the scan metadata `presc_box`. Only the compression rows change.
    let coil_gram = take("--coil-gram").unwrap_or_else(|| "data".into());
    let coil_box_margin: Option<f64> = coil_gram.strip_prefix("box").map(|r| r.strip_prefix(':').map_or(10.0, |v| v.parse().expect("--coil-gram box[:MARGIN_MM]")) * 1e-3);
    assert!(coil_gram == "data" || coil_gram == "calib" || coil_box_margin.is_some(), "--coil-gram data|calib|box[:MARGIN_MM]");
    let presc_box_arg: Option<String> = take("--presc-box");
    // time segments (research/coil_segment_compression.md, "Implementation and measured gains"): --seg-basis svdn
    // (default: the rank-L basis of the signal-weighted time functions, noise-referenced time weights, as recon_p2np's
    // knot modes) | svd (uniform time weights) | fixed (L uniform node times, equal-weight interpolators, 128-entry
    // table: the previous default, unchanged); --nseg L | auto:err[:TOL] (default auto:err:1e-4: the smallest L whose
    // relative signal-weighted model error of the operator's time function ≤ TOL, as recon_p2np --nseg auto:err)
    let seg_basis = take("--seg-basis").unwrap_or_else(|| "svdn".into());
    assert!(matches!(seg_basis.as_str(), "svdn" | "svd" | "fixed"), "--seg-basis svdn|svd|fixed");
    // --nseg auto:noise[:R] (default, R = 0.2; research/segment_basis.md, research/pipeline_target.md error budget): the
    // smallest L whose predicted data-space model error, × margin HASTY_SEG_MARGIN (1.5), is ≤ R noise rms, as
    // recon_p2np --nseg auto:noise; auto:err[:TOL] keeps the relative-error rule (the earlier default, TOL 1e-4)
    let nseg_spec = take("--nseg").unwrap_or_else(|| "auto:noise".into());
    let seg_noise: Option<f64> = nseg_spec.strip_prefix("auto:noise").map(|r| r.strip_prefix(':').map_or(0.2, |v| v.parse().expect("--nseg auto:noise[:R]")));
    let seg_tol: Option<f64> = nseg_spec.strip_prefix("auto:err").map(|r| r.strip_prefix(':').map_or(1e-4, |v| v.parse().expect("--nseg auto:err[:TOL]")));
    let mut nseg: usize = if seg_tol.is_some() || seg_noise.is_some() { 0 } else { nseg_spec.parse().expect("--nseg L|auto:noise[:R]|auto:err[:TOL]") };
    // fit voxels of the SVD basis: every N-th voxel of the coil-energy support (the fixed nodes keep every 13th)
    let seg_fit_stride: usize = take("--seg-fit-stride").map_or(3, |v| v.parse().unwrap()).max(1);
    let width: usize = take("--width").map_or(8, |v| v.parse().unwrap());
    let sigma: f64 = take("--sigma").map_or(1.25, |v| v.parse().unwrap());
    // --resp bellows (default, real data) | truth (the sim's true state: oracle / test bench only)
    let resp_src = take("--resp").unwrap_or_else(|| TARGET.resp.into());
    // --phys b0 | conc (default): conc = recon_p2np's physics: per-encode mean concomitant phase in the segments and
    // the per-sample residual (φ0, k-shift, ΔC; hasty_recon::segphys).
    // GNL (default on, --no-gnl off; --gnl accepted): coil maps, B0 and the concomitant monomials at the true (warped)
    // positions. --conc-fit deg4 (default with GNL and conc) | quad: the concomitant monomials at the GNL positions
    // fitted to degree 4; the degree-3/4 part per sample to first order in the kernel (correction order 3, as
    // recon_p2np --conc-fit deg4); quad = correction order 2 (leaves the quadratic fit's floor under GNL).
    let phys = take("--phys").unwrap_or_else(|| "conc".into());
    assert!(phys == "b0" || phys == "conc", "--phys b0|conc");
    let conc = phys == "conc";
    let gnl = !no_gnl;
    let conc_fit = take("--conc-fit").unwrap_or_else(|| if gnl && conc { "deg4".into() } else { "quad".into() });
    assert!(conc_fit == "quad" || conc_fit == "deg4", "--conc-fit quad|deg4");
    let conc_hi = conc_fit == "deg4";
    assert!(!conc_hi || conc, "--conc-fit deg4 needs --phys conc");
    // --resp-keep NR:B: only the readouts of breathing bin B of NR equal-count bins (e.g. timing one bin of a layout
    // whose key count exceeds the tables' 127)
    let resp_keep: Option<(usize, usize)> = take("--resp-keep").map(|v| { let (a, b) = v.split_once(':').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) });
    // --warm N: start every knot from the encode's static image (N preconditioned CGLS iterations through the 2-D
    // operator; circulant preconditioner of the plain SENSE normal operator)
    let warm: usize = take("--warm").map_or(0, |v| v.parse().unwrap());
    // --heldout N: keep 1/N of the readouts out of the fit (hash-spread, all strata) and track their residual every
    // --ho-every iterations; the iterate with the lowest held-out residual is also written (_itbest). 0: off
    let ho_n: usize = take("--heldout").map_or(0, |v| v.parse().unwrap());
    let ho_every: usize = take("--ho-every").map_or(2, |v| v.parse().unwrap());
    // --rcompress off|auto[:OS]|ball[:OS[:R_MM]]|box[:OS]: exact per-readout sample compression of the data term
    // (hasty_recon::rcompress, research/sample_compression.md): ROI = the grid box (one design per readout) or the
    // ball enclosing it (one shared design); band = the operator's field map. auto: on when the estimated reduction
    // is ≥ 1.2×. Held-out residuals stay exact (whole readouts; per-readout ‖y‖² kept)
    let rc_mode = hasty_recon::rcompress::RcMode::parse(&take("--rcompress").unwrap_or_else(|| "off".into())).unwrap_or_else(|e| panic!("{e}"));
    // --precond knot | none: Kronecker preconditioner G⁻¹ ⊗ C⁻¹ (hasty_recon::knotsolve) for the knot CG
    let precond_knot = take("--precond").map_or(false, |v| v == "knot");
    // --keep N: keep a deterministic 1/N of the readouts (hash-spread over all strata: a shorter scan), before the
    // held-out split
    let keep_n: usize = take("--keep").map_or(1, |v| v.parse().unwrap());
    // --arms pcg,ncg,ncg0,zmshr1,fista,admm0.3,admm1,admm3: solver comparison (research/convergence_strategy.md
    // §8 P1–P2): every arm from the same warm start; the prior arms continue from the PCG iterate after --prefix
    // gradients, where the prior weights and the prior-aware metric are estimated (data-derived); each arm stops at
    // --budget gradients (A + Aᴴ pairs); outputs <out>_<arm>_r*_it<gradients>_lumen and a per-iteration CSV
    let arms: Vec<String> = take("--arms").map_or(vec![], |v| v.split(',').map(|s| s.to_string()).collect());
    let prefix: usize = take("--prefix").map_or(5, |v| v.parse().unwrap());
    let budget: f64 = take("--budget").map_or(41.0, |v| v.parse().unwrap());
    // priors: γ = gamma_rel / bulk; wavelet ℓ1 weight λ_w = bulk · wthr · σ̂_w (σ̂_w: MAD of the finest diagonal
    // subband of the prefix iterate); LLR λ_l = bulk · kappa · σ̂_l (σ̂_l: the LLR σ̂ rule on the prefix iterate)
    let gamma_rel: f64 = take("--gamma-rel").map_or(0.1, |v| v.parse().unwrap());
    let wthr: f64 = take("--wthr").map_or(1.0, |v| v.parse().unwrap());
    let kappa: f64 = take("--kappa").map_or(1.0, |v| v.parse().unwrap());
    let block: usize = take("--block").map_or(8, |v| v.parse().unwrap());
    let wlevels: usize = take("--wlevels").map_or(3, |v| v.parse().unwrap());
    let admm_inner: usize = take("--admm-inner").map_or(3, |v| v.parse().unwrap());
    let power_its: usize = take("--power").map_or(8, |v| v.parse().unwrap());
    let llr_budget: f64 = take("--llr-budget-gb").map_or(1.0, |v| v.parse().unwrap());
    let priors_sel = take("--priors").unwrap_or_else(|| "wav,llr".into());
    // --prior-dev gpu|cpu: where the smoothed priors of the arms run (gpu: transforms, line search and LLR on the
    // device stacks, knotncg device mode; cpu: the host path); --prior-chunk: knot images per wavelet scratch (gpu)
    let prior_dev: hasty_recon::knotncg::PriorDev = take("--prior-dev").map_or(hasty_recon::knotncg::PriorDev::Gpu, |v| v.parse().unwrap());
    let prior_chunk: usize = take("--prior-chunk").map_or(8, |v| v.parse().unwrap());
    let encodes: Option<Vec<usize>> = take("--encodes").map(|s| s.split(',').map(|v| v.parse().unwrap()).collect());
    let evals: Vec<(f64, PathBuf)> = take("--eval").expect("--eval r:idx.npy,...").split(',').map(|p| { let (r, f) = p.split_once(':').unwrap(); (r.parse().unwrap(), PathBuf::from(f)) }).collect();
    let out = take("--out").expect("--out prefix");
    // --estimate-motion NR:SCALE [--roi x0:x1,y0:y1,z0:z1] [--truth-resp resp_motion.hv]: data-driven affine
    // respiratory motion (no labels), written to <out>_affine.json, then exit (see estimate_motion)
    let est: Option<(usize, f64)> = take("--estimate-motion").map(|v| { let (a, b) = v.split_once(':').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) });
    let roi: [(f64, f64); 3] = take("--roi").map_or([(0.0, 1.0); 3], |v| { let p: Vec<(f64, f64)> = v.split(',').map(|r| { let (a, b) = r.split_once(':').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) }).collect(); [p[0], p[1], p[2]] });
    let truth_resp: Option<PathBuf> = take("--truth-resp").map(PathBuf::from);
    if let Some((nr, scale)) = est {
        estimate_motion(&PathBuf::from(&args[0]), nr, scale, roi, truth_resp.as_deref(), &out);
        return;
    }
    // --time-only: one forward + adjoint of encode 0 (no solver state; for bases whose CG state does not fit)
    let time_only = args.iter().position(|a| a == "--time-only").map(|i| { args.remove(i); true }).unwrap_or(false);
    let dir = PathBuf::from(&args[0]);
    let t0 = Instant::now();
    let sd = SimData::load_lazy(&dir).unwrap();
    let (dims, nv, nc_all) = (sd.dims, sd.nv(), sd.nc);
    let ne = sd.encode_m1.len();
    let nro = sd.n_readouts();
    // breathing coordinate per readout and the respiratory basis
    let amp: Vec<f64> = {
        let (file, col) = match resp_src.as_str() { "truth" => ("resp.npy", 0), "bellows" => ("physio.npy", 3), o => panic!("--resp truth|bellows, got {o}") };
        let a = read_npy(&dir.join(file)).unwrap();
        let (w, v) = (a.shape[1], a.f64s());
        assert_eq!(a.shape[0], nro);
        (0..nro).map(|i| v[i * w + col]).collect()
    };
    let nri = if rdeg == 0 { kr } else { kr - 1 };
    let (ri_ro, rv_ro, knots_desc): (Vec<u32>, Vec<f32>, String);
    let resp_weights: Box<dyn Fn(f64) -> Vec<f64>>;
    if rdeg == 0 {
        let bins = hasty_recon::selfgate::amplitude_bins(&amp, kr);
        let mut lo = vec![f64::INFINITY; kr];
        let mut hi = vec![f64::NEG_INFINITY; kr];
        for i in 0..nro {
            lo[bins[i]] = lo[bins[i]].min(amp[i]);
            hi[bins[i]] = hi[bins[i]].max(amp[i]);
        }
        knots_desc = format!("bins {:?}", (0..kr).map(|b| format!("[{:.3}, {:.3}]", lo[b], hi[b])).collect::<Vec<_>>());
        ri_ro = bins.iter().map(|&b| b as u32).collect();
        rv_ro = vec![0.0; nro];
        let (lo2, hi2) = (lo.clone(), hi.clone());
        resp_weights = Box::new(move |r: f64| {
            let b = (0..kr).find(|&b| r >= lo2[b] && r <= hi2[b]).unwrap_or_else(|| (0..kr).min_by(|&a, &b| (0.5 * (lo2[a] + hi2[a]) - r).abs().partial_cmp(&(0.5 * (lo2[b] + hi2[b]) - r).abs()).unwrap()).unwrap());
            (0..kr).map(|k| (k == b) as u8 as f64).collect()
        });
    } else {
        let mut s = amp.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q: Vec<f64> = (0..kr).map(|k| s[((k as f64 / (kr - 1) as f64) * (nro - 1) as f64).round() as usize]).collect();
        knots_desc = format!("linear knots at {:?}", q.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        let locate = {
            let q = q.clone();
            move |a: f64| -> (usize, f64) {
                let i = (0..kr - 1).rev().find(|&i| a >= q[i]).unwrap_or(0);
                let v = ((a - q[i]) / (q[i + 1] - q[i]).max(1e-12)).clamp(0.0, 1.0);
                (i, v)
            }
        };
        let iv: Vec<(usize, f64)> = amp.iter().map(|&a| locate(a)).collect();
        ri_ro = iv.iter().map(|p| p.0 as u32).collect();
        rv_ro = iv.iter().map(|p| p.1 as f32).collect();
        resp_weights = Box::new(move |r: f64| {
            let (i, v) = locate(r);
            (0..kr).map(|k| if k == i { 1.0 - v } else if k == i + 1 { v } else { 0.0 }).collect()
        });
    }
    let w = sd.whitener();
    let grid = sd.physics_grid(gnl, &w);
    // coils. Default: SV compression of all whitened coils; Gram of the whitened data of encode 0 (every readout
    // stride so that ≤ ~2·10⁵ samples enter; no labels). --coils N (legacy bench): the N strongest physical coils by
    // sensitivity energy in the truth lumen (oracle selection), uncompressed.
    let (nc, keep, cc_rows, coil_desc): (usize, Vec<usize>, Option<Vec<Vec<num_complex::Complex64>>>, String) = match ncoil_sel {
        Some(n) => {
            let lum = read_npy(&dir.join("truth_lumen_fraction.npy")).unwrap().f32s();
            let lq: Vec<usize> = (0..nv).filter(|&q| lum[q] > 0.5).collect();
            let en: Vec<f64> = (0..nc_all).map(|c| lq.iter().map(|&q| { let z = grid.smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
            let mut order: Vec<usize> = (0..nc_all).collect();
            order.sort_by(|&a, &b| en[b].partial_cmp(&en[a]).unwrap());
            let nc = n.min(nc_all);
            (nc, order[..nc].to_vec(), None, format!("{nc} strongest of {nc_all} (truth-lumen selection, uncompressed)"))
        }
        None if coil_gram != "data" => {
            // calibration region of encode 0 (all readouts, the samples inside the low-resolution k box), low-res CG
            let cal_dims = dims.map(|d| 24.min(d) & !1);
            let in_box = |k: &[f32; 3]| (0..3).all(|a| (k[a] as f64 * dims[a] as f64 / cal_dims[a] as f64).abs() < std::f64::consts::PI * (1.0 - 1e-6));
            let s0 = sd.load_set(&sd.readouts_of(0, None), &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: Some(&in_box), data: true }).unwrap();
            let dev0 = Device::new(0).unwrap();
            let st0 = dev0.default_stream();
            let kern0 = Kernels::new(&dev0).unwrap();
            let (cal, used) = hasty_recon::espirit::calib_from_noncart(&dev0, &st0, &kern0, [&s0.coords[0], &s0.coords[1], &s0.coords[2]], &s0.y, nc_all, dims, cal_dims, 30, 1e-3).unwrap();
            drop(s0);
            let ncal: usize = cal_dims.iter().product();
            let (g, what, mask) = match coil_box_margin {
                None => {
                    let mut g = hasty_recon::coils::CoilGram::new(nc_all);
                    g.add(&cal, ncal, 1);
                    (g, "calibration region".to_string(), None)
                }
                Some(margin) => {
                    let pb = presc_box_arg.as_deref().map(|s| hasty_recon::coils::PrescBox::parse(s).unwrap()).or_else(|| hasty_recon::coils::PrescBox::from_meta(&sd.meta)).expect("--coil-gram box: needs --presc-box S|SX,SY,SZ[@CX,CY,CZ] (mm) or presc_box in the scan metadata");
                    let fov = [0, 1, 2].map(|a| dims[a] as f64 * sd.vox[a]);
                    let mk = pb.mask(cal_dims, fov, margin);
                    let what = format!("calibration region inside the prescription box {:?} mm at {:?} mm + {:.0} mm ({} of {ncal} voxels)", pb.size.map(|v| v * 1e3), pb.center.map(|v| v * 1e3), margin * 1e3, mk.iter().filter(|&&v| v).count());
                    (hasty_recon::coils::masked_gram(&cal, ncal, nc_all, &mk), what, Some(mk))
                }
            };
            let ccx = hasty_recon::coils::CoilCompression::from_gram(&g, coil_rule);
            let mut desc = format!("SV compression ({what}, {used} calibration samples): {}", ccx.report());
            if let Some(mk) = &mask {
                let (kin, p50, p99, mx) = hasty_recon::coils::masked_retention(&ccx, &cal, ncal, Some(mk));
                let (kall, ..) = hasty_recon::coils::masked_retention(&ccx, &cal, ncal, None);
                desc += &format!("; box energy kept {:.3} % (per-voxel loss p50 {p50:.1e}, p99 {p99:.1e}, max {mx:.1e}), whole calibration region {:.2} %", 100.0 * kin, 100.0 * kall);
            }
            (ccx.nvc(), vec![], Some(ccx.rows), desc)
        }
        None => {
            let ros0 = sd.readouts_of(0, None);
            let stride = (ros0.len() * sd.na / 200_000).max(1);
            let sub: Vec<usize> = ros0.iter().step_by(stride).copied().collect();
            let s0 = sd.load_set(&sub, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
            let m0 = s0.tau.len();
            let mut g = hasty_recon::coils::CoilGram::new(nc_all);
            g.add(&s0.y, m0, 1);
            let ccx = hasty_recon::coils::CoilCompression::from_gram(&g, coil_rule);
            let desc = format!("SV compression: {}", ccx.report());
            (ccx.nvc(), vec![], Some(ccx.rows), desc)
        }
    };
    let smaps: Vec<C32> = match &cc_rows {
        None => keep.iter().flat_map(|&c| grid.smaps[c * nv..(c + 1) * nv].iter().cloned()).collect(),
        Some(rows) => hasty_recon::coils::CoilCompression { rows: rows.clone(), eig: vec![], retained: 1.0, rule: coil_rule }.apply(&grid.smaps, nv),
    };
    // deviations from the target config (research/temporal_operator.md § Target config), printed with the header
    let mut dev_target: Vec<String> = vec![];
    if kr != TARGET.rknots || rdeg != TARGET.rdeg {
        dev_target.push(format!("respiratory {kr} knots degree {rdeg} (target {} degree {})", TARGET.rknots, TARGET.rdeg));
    }
    if mc != TARGET.cknots || cdeg != TARGET.cdeg {
        dev_target.push(format!("cardiac {mc} knots degree {cdeg} (target {} degree {})", TARGET.cknots, TARGET.cdeg));
    }
    if resp_src != TARGET.resp {
        dev_target.push(format!("breathing coordinate {resp_src} (target {}; truth = oracle)", TARGET.resp));
    }
    if ncoil_sel.is_some() {
        dev_target.push("coil subset (target: SV compression of all coils)".into());
    }
    if ncoil_sel.is_none() && coil_gram != "data" {
        dev_target.push(format!("coil Gram {coil_gram} (target: the whitened encode-0 data)"));
    }
    if !conc {
        dev_target.push("phys b0 (target conc)".into());
    }
    if !gnl {
        dev_target.push("no GNL (target GNL)".into());
    }
    if conc && gnl && !conc_hi {
        dev_target.push("concomitant fit quad (target deg4 under GNL)".into());
    }
    if keep_n > 1 {
        dev_target.push(format!("1/{keep_n} of the readouts (a shorter scan)"));
    }
    if seg_noise.is_none_or(|r| r > 0.2) || seg_basis != "svdn" {
        dev_target.push(format!("time segments {nseg_spec}, {seg_basis} basis (target: auto:noise:0.2 on the svdn basis)"));
    }
    // evaluation states: the truth voxels of each, the respiratory weights, the cardiac weights of the 20 frame centres
    let nfo = sd.n_frames;
    let ev: Vec<(f64, Vec<usize>, Vec<f64>)> = evals.iter().map(|(r, p)| (*r, read_npy(p).unwrap().f32s().iter().map(|&v| v as usize).collect(), resp_weights(*r))).collect();
    let cw: Vec<Vec<f64>> = (0..nfo).map(|f| { let t = ((f as f64 + 0.5) / nfo as f64) * mc as f64 - 0.5; (0..mc).map(|b| bweight(t, b, mc, cdeg)).collect() }).collect();
    println!(
        "{}: grid {dims:?}, {ne} encodes, coils {nc} of {nc_all}, respiratory {} ({resp_src}): {knots_desc}; cardiac {mc} knots degree {cdeg}; time segments {nseg_spec} ({seg_basis} basis), W {width}, {iters} CGLS iterations ({:.1}s)",
        dir.display(),
        if rdeg == 0 { format!("{kr} bins") } else { format!("{kr} linear knots") },
        t0.elapsed().as_secs_f64()
    );
    println!("  coils: {coil_desc}");
    println!("  physics: {}{}{}", if conc { "B0 + concomitant" } else { "B0" }, if gnl { " + GNL" } else { "" }, if conc { format!(" (concomitant fit {conc_fit}, correction order {})", if conc_hi { 3 } else { 2 }) } else { String::new() });
    if dev_target.is_empty() {
        println!("  config: target");
    } else {
        println!("  config: differs from target: {}", dev_target.join("; "));
    }
    for (r, idx, g) in &ev {
        println!("  eval r* = {r}: {} voxels, respiratory weights {:?}", idx.len(), g.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    }
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    // segment basis inputs (all data-derived), before the engine (whose nseg the rule sets):
    // * voxel weights: signal energy ≈ |x̂|² / Σ_c|S_c|² with x̂ the smoothed magnitude of the SENSE adjoint of a readout
    //   subset of the first encode (≈ ρ² Σ_c|S_c|²: the weight of the coil-common error, as Σ_c |S_c Δg|²);
    // * time weights: 1 + (E(n) + E_vol) / (C σ²) per ADC sample (hasty_nufft::noise_time_weights) on seg_time_grid;
    // * the count: --nseg auto:err from the weighted spectrum (svd) or by scanning L (fixed), on the first encode (the
    //   encodes' mean concomitant phases differ by a constant: research/coil_segment_compression.md §2.6).
    let seg_t = Instant::now();
    let e_first = encodes.as_ref().map_or(0, |v| v[0]);
    let cen: Vec<f64> = (0..nv).map(|q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect();
    let cmax = cen.iter().cloned().fold(0.0, f64::max);
    let support: Vec<usize> = (0..nv).filter(|&q| cen[q] > 1e-3 * cmax).collect();
    let fit_svd: Vec<usize> = support.iter().copied().step_by(seg_fit_stride).collect();
    let (seg_w, seg_grid, tau_seg, tw_adc, tw_noise): (Vec<f64>, (Vec<f64>, Vec<f64>), Vec<f64>, Vec<f64>, Vec<f64>) = {
        let ros_all = sd.readouts_of(e_first, None);
        let ros_w: Vec<usize> = ros_all.iter().copied().step_by((ros_all.len() / 1500).max(1)).collect();
        let sw = sd.load_set(&ros_w, &hasty_recon::kload::LoadOpts { w: &w, cc: cc_rows.as_deref(), keep: None, data: true }).unwrap();
        let mw = sw.tau.len();
        let naw = mw / ros_w.len();
        let yw: Vec<C32> = match &cc_rows { Some(_) => sw.y.clone(), None => keep.iter().flat_map(|&c| sw.y[c * mw..(c + 1) * mw].iter().cloned()).collect() };
        let mag = {
            let mut op = hasty_recon::op::SenseOp::new(&dev, &st, kern.clone(), dims, [&sw.coords[0], &sw.coords[1], &sw.coords[2]], &smaps, nc, vec![], 1e-4).unwrap();
            let yd = st.clone_htod(&yw).unwrap();
            let mut xd = st.alloc_zeros::<C32>(nv).unwrap();
            op.adjoint(&st, &yd, &mut xd).unwrap();
            hasty_recon::segphys::smooth_magnitude(&st.clone_dtoh(&xd).unwrap(), dims, sd.vox, [8e-3, 8e-3, 2e-3])
        };
        let mut wv: Vec<f64> = fit_svd.iter().map(|&q| (mag[q] as f64).powi(2) / cen[q].max(1e-30)).collect();
        let mut srt = wv.clone();
        srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = srt[srt.len() / 2].max(1e-300);
        wv.iter_mut().for_each(|v| *v /= med);
        let tau_a: Vec<f64> = sw.tau[..naw].to_vec();
        let ros_e: Vec<usize> = (0..nro).step_by(7).take(512).collect();
        let e = sd.adc_energy_whitened(&ros_e, &w).unwrap();
        let kap: Vec<f64> = (0..naw).map(|n| (0..3).map(|a| (sw.coords[a][n] as f64 / sd.vox[a]).powi(2)).sum::<f64>().sqrt()).collect();
        let sigma: f64 = std::env::var("HASTY_NOISE_SIGMA").ok().and_then(|v| v.parse::<f64>().ok()).filter(|v| *v > 0.0).or_else(|| sd.meta["noise_sigma"].as_f64().filter(|v| *v > 0.0)).unwrap_or(1.0);
        // noise-referenced per-ADC weights 1 + (E(n) + E_vol) / (C σ²): the svdn basis weights, and always the metric of
        // the auto:noise rule and of the achieved-error print (so every basis is compared in the same metric)
        let twn: Vec<f64> = hasty_nufft::noise_time_weights(&e[..naw], &kap, nc_all as f64 * sigma * sigma);
        let tw: Vec<f64> = if seg_basis == "svdn" { twn.clone() } else { vec![1.0; naw] };
        let grid_t = hasty_nufft::seg_time_grid(&tau_a, &tw, 96);
        (wv, grid_t, tau_a, tw, twn)
    };
    // error voxels: ≤ 20 000 of the support with the same weights (as recon_p2np's achieved-error print)
    let err_idx: Vec<usize> = (0..fit_svd.len()).step_by((fit_svd.len() / 20000).max(1)).collect();
    let (err_vox, err_w): (Vec<usize>, Vec<f64>) = (err_idx.iter().map(|&i| fit_svd[i]).collect(), err_idx.iter().map(|&i| seg_w[i]).collect());
    let fit13: Vec<usize> = support.iter().copied().step_by(13).collect();
    let w13: Vec<f64> = { let pos: std::collections::HashMap<usize, usize> = fit_svd.iter().enumerate().map(|(i, &q)| (q, i)).collect(); fit13.iter().map(|&q| pos.get(&q).map_or_else(|| (cen[q] / cmax).max(1e-12), |&i| seg_w[i])).collect() };
    // noise rule: relative weighted error ε(L) in the noise-weighted metric → data error ≈ ε(L) √(mean w) noise rms
    let seg_margin: f64 = std::env::var("HASTY_SEG_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
    let noise_gain = (tw_noise.iter().sum::<f64>() / tw_noise.len().max(1) as f64).sqrt();
    let noise_grid = hasty_nufft::seg_time_grid(&tau_seg, &tw_noise, 96);
    let seg_rule: Option<(f64, bool)> = seg_noise.map(|r| (r / (noise_gain * seg_margin), true)).or(seg_tol.map(|t| (t, false)));
    if let Some((tol, noise_rule)) = seg_rule {
        let ros = sd.readouts_of(e_first, None);
        let sub: Vec<usize> = ros.iter().copied().step_by((ros.len() / 400).max(1)).collect();
        let s1 = sd.load_set(&sub, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: false }).unwrap();
        let na1 = s1.tau.len() / sub.len();
        let wb = if conc { hasty_recon::segphys::mean_coefficients(&[&s1], na1) } else { vec![[0.0; 4]; na1] };
        let psi = hasty_recon::segphys::segment_phase(&grid, &wb, &tau_seg, true);
        let g = |v: usize, t: f64| num_complex::Complex64::from_polar(1.0, -psi(v, t));
        // the noise rule predicts in the noise-weighted metric whatever the basis (for svd / fixed a lower bound /
        // their own error in that metric); auto:err in the basis' own metric, as before
        let (ts, tws) = if noise_rule { (&noise_grid.0, &noise_grid.1) } else { (&seg_grid.0, &seg_grid.1) };
        nseg = if seg_basis == "fixed" {
            let (t0, t1) = (tau_seg[0], tau_seg[tau_seg.len() - 1]);
            (2..=64).find(|&l| { let tl: Vec<f64> = (0..l).map(|i| t0 + (t1 - t0) * i as f64 / (l - 1) as f64).collect(); hasty_nufft::nodes_weighted_error(&fit13, None, Some(&w13), ts, tws, &tl, &g) <= tol }).unwrap_or(64)
        } else {
            let ev = hasty_nufft::weighted_spectrum(&fit_svd, Some(&seg_w), ts, tws, &g);
            // the whole predicted error curve, so a segment count can be chosen against the error budget afterwards
            let te = hasty_nufft::tail_errors(&ev);
            println!("  predicted {seg_basis} segment error by L: {}", (4..=20.min(te.len() - 1)).map(|l| format!("{l}: {:.1e}", te[l])).collect::<Vec<_>>().join(", "));
            hasty_nufft::seg_count_for(&ev, tol, 2, 64)
        };
        if noise_rule {
            println!("  segments (auto:noise:{}): predicted data error ≤ {} noise rms (margin {seg_margin}, √mean w = {noise_gain:.1}: relative tolerance {tol:.2e}) → L = {nseg}", seg_noise.unwrap(), seg_noise.unwrap());
        }
        println!("  segments ({nseg_spec}): relative signal-weighted model error ≤ {tol:.1e} (predicted on the {}-point time grid, {} time weights) → L = {nseg} ({seg_basis} basis, {:.1}s)", ts.len(), if seg_basis == "svdn" { "noise-referenced" } else { "uniform" }, seg_t.elapsed().as_secs_f64());
    }
    // device memory ledger (research/memory_ledger.md), planned before any heavy work: CGLS image vectors (x, s, p +
    // one spare), data vectors (y, r, q), sample tables (records 48 B + ≈ 1.25 × W³/64 tile entries of 4 B per sample),
    // coil maps and segment diagonals, the grid slot; the grid budget shrinks to what is left, else stop here
    let gib = (1u64 << 30) as f64;
    let keep_mask = hasty_recon::knotsolve::subset_mask(nro, keep_n, 0x4B45_4550);
    let m_max = (0..ne).map(|e| sd.readouts_of(e, None).iter().filter(|&&r| keep_mask[r]).count()).max().unwrap() * sd.na / resp_keep.map_or(1, |k| k.0) * if ho_n > 0 { ho_n - 1 } else { 1 } / ho_n.max(1);
    // solver image stacks: CGLS x, s, p (+ z with a preconditioner); arms: the largest arm (ZMSHR1 6, NCG / FISTA /
    // ADMM 5, PCG 4)
    let arm_stacks = arms.iter().map(|a| if a == "zmshr1" { 6.0 } else if a == "pcg" { 4.0 } else { 5.0 }).fold(0.0f64, f64::max);
    let has_prior_arms = arms.iter().any(|a| a != "pcg");
    // device mode of the priors: the largest prior footprint over the power iteration and the prior arms
    // (knotncg::Priors::dev_bytes: wavelet chunk + engine, LLR scratch stack + block table, the solver's prior stacks)
    let prior_dev_bytes = if has_prior_arms && prior_dev == hasty_recon::knotncg::PriorDev::Gpu {
        use hasty_recon::knotncg::{DevSolver, Priors};
        let npc: usize = hasty_recon::knotncg::WavHuber::new(dims, wlevels, 0.0, 1.0).p.iter().product();
        let (hw, hl) = (priors_sel.contains("wav"), priors_sel.contains("llr"));
        let mut sol: Vec<DevSolver> = arms.iter().filter(|a| *a != "pcg").map(|a| if a.starts_with("ncg") { DevSolver::Ncg } else if a == "fista" { DevSolver::Fista } else if a == "zmshr1" { DevSolver::Zmshr1 } else { DevSolver::Admm }).collect();
        if arms.iter().any(|a| a == "fista" || a == "zmshr1") {
            sol.push(DevSolver::Power);
        }
        sol.into_iter().map(|s| Priors::dev_bytes(s, kr * mc, nv, npc, hw, hl, prior_chunk)).max().unwrap_or(0) as f64
    } else {
        0.0
    };
    let tiles_per_sample = 1.25 * ((width as f64 + 3.0) / 4.0).powi(3);
    let led: Vec<(&str, f64)> = vec![
        ("solver image vectors (CGLS x, s, p; arms: the largest arm; time-only 2)", if time_only { 2.0 } else if arms.is_empty() { 3.0 } else { arm_stacks - 1.0 } * (kr * mc * nv) as f64 * 8.0),
        ("CGLS data vectors (r, q: y becomes r; time-only 2)", 2.0 * (nc * m_max) as f64 * 8.0),
        ("sample tables (+ 16 B Δw per sample at order 3)", m_max as f64 * (48.0 + 4.0 * tiles_per_sample + if conc_hi { 16.0 } else { 0.0 })),
        ("coil maps + segment diagonals", ((nc + nseg) * nv) as f64 * 8.0),
        ("knot preconditioner (CG z vector + 2 images + circulant)", if precond_knot { ((kr * mc + 2) * nv * 8) as f64 + 0.5 * gib } else { 0.0 }),
        ("prior-aware metric (3 nv f32 + plans) + LLR on GPU (batch budget + cuBLAS / cuSOLVER handles)", if has_prior_arms { hasty_recon::knotncg::PriorMetric::bytes(kr * mc, nv) as f64 + llr_budget * gib + 0.5 * gib } else { 0.0 }),
        ("priors on the device (--prior-dev gpu: coefficient stacks, LLR stacks, wavelet scratch)", prior_dev_bytes),
        ("FFT / kernel workspace, margin", 1.5 * gib),
    ];
    let (free, total) = cudarc::driver::result::mem_get_info().unwrap();
    let fixed: f64 = led.iter().map(|l| l.1).sum();
    let grid_cap = std::env::var("SPLINE_GRID_GB").ok().map_or(3.0, |v| v.parse::<f64>().unwrap()) * gib;
    let grid_budget = (free as f64 * 0.97 - fixed).min(grid_cap);
    println!("  device ledger ({:.1} of {:.1} GiB free):", free as f64 / gib, total as f64 / gib);
    for (k, v) in &led {
        println!("    {k}: {:.2} GiB", v / gib);
    }
    println!("    grid slot budget: {:.2} GiB", grid_budget / gib);
    if grid_budget < 0.4 * gib {
        eprintln!("splinerecon: the plan does not fit the device ({:.2} GiB fixed, {:.2} GiB free): fewer coils / knots, --prior-dev cpu ({:.2} GiB of device priors) or a host-resident solver", fixed / gib, free as f64 / gib, prior_dev_bytes / gib);
        std::process::exit(2);
    }
    let opts = SegNufftOpts { sigma, width, nseg, order: if conc_hi { 3 } else if conc { 2 } else { 0 }, frames: mc, bspline: cdeg, max_grid_bytes: grid_budget as usize };
    let mut eng = SegNufft::new_temporal2d(&dev, dims, sd.vox, nc, opts, kr, rdeg).unwrap();
    eng.set_maps(&st, &smaps).unwrap();
    let eng = std::cell::RefCell::new(eng);
    let nx = kr * mc * nv;
    let ho_mask: Vec<bool> = if ho_n > 0 { hasty_recon::knotsolve::heldout_mask(nro, ho_n) } else { vec![false; nro] };
    // --rcompress: the rate band of the operator's field map (coil-energy weighted 1e-4 quantiles, 2× margin, at least
    // ±200 Hz; decay ≤ 300 1/s for the data's T2*), the mode decided at the first encode
    let rc_band = {
        let ce: Vec<f64> = (0..nv).map(|q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
        hasty_recon::rcompress::Band { r_max: 300.0, ..hasty_recon::rcompress::Band::from_ratemap(&grid.omega, None, &ce, 1e-4, 1.0, 200.0) }
    };
    let mut rc_comp: Option<hasty_recon::rcompress::Compressor> = None;
    let mut rc_decided = rc_mode.is_off();
    assert!(rc_mode.is_off() || arms.is_empty(), "--rcompress: not with --arms (the arms drive the operator directly)");
    let mut vals: Vec<Vec<Vec<C32>>> = (0..=save_at.len()).map(|_| ev.iter().map(|(_, idx, _)| vec![C32::new(0.0, 0.0); idx.len() * ne * nfo]).collect()).collect();
    let mut seg: Option<SegSegments> = None;
    let mut wbar_e: Option<Vec<[f64; 4]>> = None;
    let mut monofit: Option<hasty_recon::segphys::MonoFit> = None;
    let mut monofit_hi: Option<hasty_recon::segphys::MonoFitHi> = None;
    let mut htab: Option<Vec<f64>> = None;
    let enc_list: Vec<usize> = encodes.unwrap_or_else(|| (0..ne).collect());
    let cnt = std::cell::Cell::new((0usize, 0usize));
    // (arm, label) → per encode the compact lumen values per evaluation state
    let mut arm_vals: std::collections::BTreeMap<(String, String), Vec<(usize, Vec<Vec<C32>>)>> = Default::default();
    let mut timing = vec![];
    for &e in &enc_list {
        let te = Instant::now();
        let mut ros = sd.readouts_of(e, None);
        ros.retain(|&r| keep_mask[r]);
        if let Some((nr, b)) = resp_keep {
            let bins = hasty_recon::selfgate::amplitude_bins(&amp, nr);
            ros.retain(|&r| bins[r] == b);
        }
        let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: cc_rows.as_deref(), keep: None, data: true }).unwrap();
        let m = set.tau.len();
        let na = m / ros.len();
        assert_eq!(na * ros.len(), m, "fixed ADC count per readout expected");
        if conc || seg.is_none() {
            // segments over the readout window: ψ(v, τ) = ω_v τ (+ the encode's mean concomitant phase), fitted on object voxels
            let (tmin, tmax) = set.tau.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| (a.min(t), b.max(t)));
            let emax = (0..nv).map(|q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum::<f64>()).fold(0.0, f64::max);
            let fit: Vec<usize> = (0..nv).filter(|&q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum::<f64>() > 1e-3 * emax).step_by(13).collect();
            let tau_adc: Vec<f64> = set.tau[..na].to_vec();
            let wb = if conc { hasty_recon::segphys::mean_coefficients(&[&set], na) } else { vec![[0.0; 4]; na] };
            let (sp, at, s0, ds) = {
                let psi = hasty_recon::segphys::segment_phase(&grid, &wb, &tau_adc, true);
                let g = |v: usize, t: f64| num_complex::Complex64::from_polar(1.0, -psi(v, t));
                let r = if seg_basis == "fixed" {
                    // 128-entry interpolator table for an explicit --nseg L (the previous default, unchanged); one entry
                    // per ADC sample when L comes from a rule: the 128-entry table has a 3–7e-4 floor of its own, which
                    // the rule's prediction does not see (auto:err 1e-4 picked L = 18 and achieved 8.4e-4)
                    build_segments(nv, nseg, tmin, tmax, if seg_rule.is_some() { na } else { 128 }, &fit, &psi)
                } else {
                    // one interpolator entry per ADC sample (as recon_p2np: a 128/256-entry table leaves a 3–7e-4 floor)
                    hasty_nufft::build_segments_svd_grid(nv, nseg, tmin, tmax, na, &fit_svd, Some(&seg_w), &seg_grid.0, &seg_grid.1, &g)
                };
                // achieved error at the ADC times (every 16th, with the basis' time weights) over the error voxels
                let tix: Vec<usize> = (0..na).step_by(16).collect();
                let taus: Vec<f64> = tix.iter().map(|&n| tau_adc[n]).collect();
                let tws: Vec<f64> = tix.iter().map(|&n| tw_adc[n.min(tw_adc.len() - 1)]).collect();
                let err = hasty_nufft::segment_model_error(&r.0, &r.1, nseg, nv, r.2, r.3, &err_vox, Some(&err_w), &taus, &tws, &g);
                // the same error in the noise-referenced metric for every basis, and as data error in noise rms
                let twn: Vec<f64> = tix.iter().map(|&n| tw_noise[n.min(tw_noise.len() - 1)]).collect();
                let err_n = hasty_nufft::segment_model_error(&r.0, &r.1, nseg, nv, r.2, r.3, &err_vox, Some(&err_w), &taus, &twn, &g);
                println!("  segment model error of encode {e}, noise-referenced metric: relative {err_n:.2e} ≈ {:.3} noise rms", err_n * noise_gain);
                println!("  segment model error of encode {e}: L = {nseg} ({seg_basis}), relative {err:.2e} (signal-weighted, {} time weights, {} voxels)", if seg_basis == "svdn" { "noise-referenced" } else { "uniform" }, err_vox.len());
                r
            };
            wbar_e = Some(wb);
            if conc && monofit.is_none() {
                if conc_hi {
                    let f4 = hasty_recon::segphys::fit_monomials_hi(&sd, &grid, &fit);
                    println!("  concomitant monomials at the {} positions: degree-4 fit rms {:.1e} / max {:.1e} m² (quadratic part), degree 3/4 to first order (order 3)", if gnl { "GNL" } else { "nominal" }, f4.quad.rms.iter().cloned().fold(0.0, f64::max), f4.quad.max.iter().cloned().fold(0.0, f64::max));
                    htab = Some(hasty_recon::segphys::hi_table(&f4));
                    monofit_hi = Some(f4);
                }
                monofit = Some(hasty_recon::segphys::fit_monomials(&sd, &grid, &fit));
            }
            seg = Some(eng.borrow().segments(&st, &sp, &at, s0, ds).unwrap());
            println!("  segments: {nseg} over τ ∈ [{:.2}, {:.2}] ms, {} fit voxels", tmin * 1e3, tmax * 1e3, fit.len());
        }
        let seg = seg.as_ref().unwrap();
        let mut t = vec![0f32; m];
        let mut ri = vec![0u32; m];
        let mut rv = vec![0f32; m];
        for (r, &ro) in ros.iter().enumerate() {
            let card = &sd.ro.card[ro];
            for n in 0..na {
                let j = r * na + n;
                let phase = card.phase + set.tau[j] / card.rr;
                t[j] = (phase * mc as f64 - 0.5).rem_euclid(mc as f64) as f32;
                ri[j] = ri_ro[ro];
                rv[j] = rv_ro[ro];
            }
        }
        assert!(ri.iter().all(|&i| (i as usize) < nri));
        // per-sample physics: the residual concomitant terms against the encode's mean (as recon_p2np), or none
        // dwv: with --conc-fit deg4 the per-sample concomitant deviation Δw (4 × m SoA) for the degree-3/4 phase
        let (coords, tau, phi0, dcv, dwv) = if conc {
            match &monofit_hi {
                Some(f4) => {
                    let (tab, dw) = hasty_recon::segphys::seg_tables_hi(&sd, &set, f4, true, wbar_e.as_deref());
                    (tab.coords, tab.tau, tab.phi0, tab.dc, dw)
                }
                None => {
                    let tab = hasty_recon::segphys::seg_tables(&sd, &set, monofit.as_ref().unwrap(), true, wbar_e.as_deref());
                    (tab.coords, tab.tau, tab.phi0, tab.dc, vec![])
                }
            }
        } else {
            (set.coords.clone(), set.tau.iter().map(|&v| v as f32).collect(), vec![0f32; m], vec![], vec![])
        };
        let yh = if cc_rows.is_some() {
            // already compressed by load_set: nvc × m
            set.y.clone()
        } else {
            let mut yh = vec![C32::new(0.0, 0.0); nc * m];
            for (ci, &c) in keep.iter().enumerate() {
                yh[ci * m..(ci + 1) * m].copy_from_slice(&set.y[c * m..(c + 1) * m]);
            }
            yh
        };
        assert_eq!(yh.len(), nc * m);
        drop(set);
        // held-out split (whole readouts): fit = the rest
        let (coords, tau, phi0, dcv, t, ri, rv, yh, dwv, ho) = {
            let hold: Vec<bool> = (0..m).map(|j| ho_mask[ros[j / na]]).collect();
            let (fi, hi): (Vec<usize>, Vec<usize>) = (0..m).partition(|&j| !hold[j]);
            let g32 = |v: &[f32], ix: &[usize]| -> Vec<f32> { ix.iter().map(|&j| v[j]).collect() };
            let gdc = |ix: &[usize]| -> Vec<f32> { if dcv.is_empty() { vec![] } else { (0..6).flat_map(|q| ix.iter().map(move |&j| q * m + j)).map(|k| dcv[k]).collect() } };
            let gdw = |ix: &[usize]| -> Vec<f32> { if dwv.is_empty() { vec![] } else { (0..4).flat_map(|q| ix.iter().map(move |&j| q * m + j)).map(|k| dwv[k]).collect() } };
            let gy = |ix: &[usize]| -> Vec<C32> { (0..nc).flat_map(|c| ix.iter().map(move |&j| c * m + j)).map(|k| yh[k]).collect() };
            let part = |ix: &[usize]| ([g32(&coords[0], ix), g32(&coords[1], ix), g32(&coords[2], ix)], g32(&tau, ix), g32(&phi0, ix), gdc(ix), g32(&t, ix), ix.iter().map(|&j| ri[j]).collect::<Vec<u32>>(), g32(&rv, ix), gy(ix), gdw(ix));
            if hi.is_empty() {
                (coords, tau, phi0, dcv, t, ri, rv, yh, dwv, None)
            } else {
                let f = part(&fi);
                let h = part(&hi);
                (f.0, f.1, f.2, f.3, f.4, f.5, f.6, f.7, f.8, Some(h))
            }
        };
        let m = tau.len();
        // --rcompress: the fit and held-out parts compressed per readout (na samples each); the full fit samples stay
        // for the preconditioners below (circulant metric, knot Gram: the original sample density)
        if !rc_decided {
            let ss = hasty_recon::data::SampleSet { readouts: vec![], coords: coords.clone(), y: vec![], tau: tau.iter().map(|&v| v as f64).collect(), wq: vec![] };
            let (o, line) = hasty_recon::rcompress::decide(&rc_mode, dims, sd.vox, rc_band, &ss, &vec![na; m / na]);
            println!("  rcompress: band {}; {line}", rc_band.describe());
            rc_comp = o.map(hasty_recon::rcompress::Compressor::new);
            rc_decided = true;
        }
        let rcz = rc_comp.as_mut().map(|comp| {
            let tr = Instant::now();
            let mut part = |c: &[Vec<f32>; 3], tau: &[f32], phi0: &[f32], dc: &[f32], t: &[f32], ri: &[u32], rv: &[f32], y: &[C32], dw: &[f32]| {
                let mm = tau.len();
                let ss = hasty_recon::data::SampleSet { readouts: vec![], coords: c.clone(), y: vec![], tau: tau.iter().map(|&v| v as f64).collect(), wq: vec![] };
                let (sc, yc) = comp.compress_set(&ss, &vec![na; mm / na], sd.vox, y, nc);
                let g = |v: &[f32]| hasty_recon::pyramid::gather(v, &sc.sel);
                let out = ([g(&c[0]), g(&c[1]), g(&c[2])], g(tau), g(phi0), hasty_recon::rcompress::select_rows(dc, mm, &sc.sel), g(t), hasty_recon::pyramid::gather(ri, &sc.sel), g(rv), yc, hasty_recon::rcompress::select_rows(dw, mm, &sc.sel));
                (out, sc)
            };
            let f = part(&coords, &tau, &phi0, &dcv, &t, &ri, &rv, &yh, &dwv);
            let h = ho.as_ref().map(|h| part(&h.0, &h.1, &h.2, &h.3, &h.4, &h.5, &h.6, &h.7, &h.8));
            if e == enc_list[0] {
                // gate on two fit readouts (direct sums; cardiac knot envelope, no concomitant phase)
                let mut gs = hasty_recon::rcompress::GateStats::default();
                for r in [0, (m / na) / 2] {
                    let k: Vec<[f64; 3]> = (r * na..(r + 1) * na).map(|j| [0, 1, 2].map(|a| coords[a][j] as f64 / sd.vox[a])).collect();
                    let tt: Vec<f64> = (r * na..(r + 1) * na).map(|j| tau[j] as f64).collect();
                    let kt: Vec<f64> = (r * na..(r + 1) * na).map(|j| t[j] as f64).collect();
                    let d = hasty_recon::rcompress::Design::build(&k, &tt, &comp.opts);
                    gs = gs.max(hasty_recon::rcompress::gate(&d, &k, &tt, None, Some(&kt), &comp.opts, 200, 3 + r as u64));
                }
                println!("  rcompress gate (2 readouts, 200 sources, half on the band edge): worst source {:.2e}, normal operator on the sources {:.2e}", gs.worst, gs.gram);
            }
            // dropped energy against its noise expectation (n − |S|) · coils · σ² (≫ 1: signal outside ROI × band)
            let noise_exp = sd.lazy.as_ref().and_then(|l| l.sigma).map(|sg| (m - f.1.m()) as f64 * nc as f64 * sg * sg);
            let cn: f64 = f.1.cnorm.iter().sum();
            println!("  rcompress encode {e}: {} ({:.1}s): fit {} → {} samples ({:.3}×){}; dropped energy {:.3e}{}", comp.describe(), tr.elapsed().as_secs_f64(), m, f.1.m(), m as f64 / f.1.m().max(1) as f64, h.as_ref().map_or(String::new(), |h| format!(", held-out {} → {}", h.1.n_orig(), h.1.m())), cn, noise_exp.map_or(String::new(), |x| format!(" ({:.3}× the noise expectation)", cn / x.max(1e-300))));
            (f, h)
        });
        let mut pre = if warm > 0 { Some(hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, 1e-3).unwrap()) } else { None };
        // sample tables: correction order 3 attaches the degree-3/4 coefficients (points2d_hi)
        let mk_pts = |e: &mut SegNufft, c: &[Vec<f32>; 3], tau: &[f32], phi0: &[f32], dc: &[f32], t: &[f32], ri: &[u32], rv: &[f32], dw: &[f32]| -> SegPoints {
            match &htab {
                Some(h) => e.points2d_hi(&st, [&c[0], &c[1], &c[2]], tau, phi0, dc, t, ri, rv, dw, h).unwrap(),
                None => e.points2d(&st, [&c[0], &c[1], &c[2]], tau, phi0, dc, t, ri, rv).unwrap(),
            }
        };
        let pts = match &rcz {
            Some((f, _)) => { let p = &f.0; mk_pts(&mut eng.borrow_mut(), &p.0, &p.1, &p.2, &p.3, &p.4, &p.5, &p.6, &p.8) }
            None => mk_pts(&mut eng.borrow_mut(), &coords, &tau, &phi0, &dcv, &t, &ri, &rv, &dwv),
        };
        let ho_src = match &rcz { Some((_, h)) => h.as_ref().map(|h| &h.0), None => ho.as_ref() };
        let pts_ho = ho_src.map(|h| mk_pts(&mut eng.borrow_mut(), &h.0, &h.1, &h.2, &h.3, &h.4, &h.5, &h.6, &h.8));
        let y_ho = ho_src.map(|h| st.clone_htod(&h.7).unwrap());
        let m_ho = ho_src.map_or(0, |h| h.1.len());
        // --rcompress: the R tables of the fit (set 0) and held-out (set 1) parts; the held-out constant Σ ‖y‖² − ‖ỹ‖²
        // and the original held-out energy; the operator's data length; the fit data become the compressed ones
        let rcd = rcz.as_ref().map(|(f, h)| {
            let c = rc_comp.as_ref().unwrap();
            let mask = hasty_recon::rcompress::roi_mask(&c.opts.roi, dims, sd.vox);
            hasty_recon::rcompress::RcDev::new(&dev, &st, c, &[f.1.clone(), h.as_ref().map_or_else(Default::default, |h| h.1.clone())]).unwrap().with_mask(&st, mask.as_deref()).unwrap()
        });
        let (ho_const, ho_y2) = match &rcz { Some((_, Some(h))) => (h.1.cnorm.iter().sum::<f64>(), Some(h.1.ynorm.iter().sum::<f64>().max(1e-300))), _ => (0.0, None) };
        let m_op = rcz.as_ref().map_or(m, |(f, _)| f.1.m());
        let yh = match rcz { Some((f, _)) => f.0 .7, None => yh };
        drop(ho);
        // knot preconditioner: temporal Gram of the basis under the fit samples' times (every 7th sample)
        let mut kgram: Option<(Vec<f64>, Vec<f32>, Vec<f32>)> = None;
        let mut kpre = if precond_knot {
            let kk = kr * mc;
            let wts = (0..m).step_by(7).map(|j| {
                let rw: Vec<(usize, f64)> = if rdeg == 0 { vec![(ri[j] as usize, 1.0)] } else { vec![(ri[j] as usize, 1.0 - rv[j] as f64), (ri[j] as usize + 1, rv[j] as f64)] };
                let i0 = (t[j] as f64).floor() as i64;
                let cwj: Vec<(usize, f64)> = (i0 - 2..=i0 + 2).map(|b| b.rem_euclid(mc as i64) as usize).map(|b| (b, bweight(t[j] as f64, b, mc, cdeg))).filter(|p| p.1 != 0.0).collect();
                rw.iter().flat_map(|&(k, a)| cwj.iter().map(move |&(b, c)| (k * mc + b, a * c))).collect::<Vec<_>>()
            });
            let g = hasty_recon::knotsolve::temporal_gram(kk, wts);
            println!("  knot preconditioner: temporal Gram {kk}×{kk}, condition {:.1}", hasty_recon::knotsolve::gram_condition(&g, kk));
            let gi = hasty_recon::knotsolve::gram_inverse(&g, kk, 1e-3);
            let circ = hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, 1e-3).unwrap();
            if !arms.is_empty() {
                let (lam, dinv) = circ.factors(&st).unwrap();
                kgram = Some((g.clone(), lam, dinv));
            }
            Some(hasty_recon::knotsolve::KnotPrecond::new(&dev, &st, kk, nv, &gi, Some(circ)).unwrap())
        } else {
            None
        };
        drop((coords, tau, phi0, dcv, t, ri, rv, dwv));
        if timing.is_empty() && std::env::var("HASTY_SPLINE_NOTUNE").map_or(true, |v| v == "0") {
            let tt = Instant::now();
            let (kf, ka) = eng.borrow_mut().autotune(&st, seg, &pts).unwrap();
            println!("  autotune {:.1}s: tile kernels {kf:.1} / {ka:.1} ms (knot 0, one coil batch of {}; fwd {:?}, adj {:?})", tt.elapsed().as_secs_f64(), eng.borrow().cb, eng.borrow().tiles_fwd, eng.borrow().tiles_adj);
        }
        let y = st.clone_htod(&yh).unwrap();
        let yh_keep = if arms.is_empty() { drop(yh); None } else { Some(yh) };
        if time_only {
            let xr: Vec<C32> = (0..nx).map(|i| C32::new(((i * 7919) % 1000) as f32 * 1e-3, ((i * 104729) % 997) as f32 * 1e-3)).collect();
            let xd = st.clone_htod(&xr).unwrap();
            drop(xr);
            let mut yy = st.alloc_zeros::<C32>(nc * m_op).unwrap();
            let mut xa = st.alloc_zeros::<C32>(nx).unwrap();
            let mut tf = vec![];
            for rep in 0..3 {
                st.synchronize().unwrap();
                let a = Instant::now();
                eng.borrow_mut().forward(&st, seg, &pts, &xd, &mut yy).unwrap();
                st.synchronize().unwrap();
                let b = Instant::now();
                eng.borrow_mut().adjoint(&st, seg, &pts, &y, &mut xa).unwrap();
                st.synchronize().unwrap();
                if rep > 0 {
                    tf.push((b.duration_since(a).as_secs_f64(), b.elapsed().as_secs_f64()));
                }
            }
            let (f, a) = tf.iter().fold((0.0, 0.0), |s, t| (s.0 + t.0 / tf.len() as f64, s.1 + t.1 / tf.len() as f64));
            println!("  time-only encode {e}: forward {f:.2}s + adjoint {a:.2}s = {:.2}s per gradient pass ({} knots, {m} samples, coil batch {})", f + a, kr * mc, eng.borrow().cb);
            return;
        }
        let mut x = st.alloc_zeros::<C32>(nx).unwrap();
        if let Some(pr) = pre.as_mut() {
            let tw = Instant::now();
            let mut sx = st.alloc_zeros::<C32>(nv).unwrap();
            {
                let mut op = Op2d { eng: &eng, cnt: &cnt, seg, pts: &pts, kern: kern.clone(), nx, ny: nc * m_op };
                let mut rop = rcd.as_ref().map(|d| hasty_recon::rcompress::RcLin::new(Op2d { eng: &eng, cnt: &cnt, seg, pts: &pts, kern: kern.clone(), nx, ny: nc * m_op }, d, 0));
                let inner: &mut dyn LinOp = match rop.as_mut() { Some(r) => r, None => &mut op };
                let mut wop = WarmOp { inner, big: &mut x, tmp: st.alloc_zeros::<C32>(nv).unwrap(), nv };
                pcgls_cb(&mut wop, Some(pr), &st, &y, &mut sx, 0.0, warm, &mut |_, _| {}).unwrap();
            }
            for k in 0..kr * mc {
                st.memcpy_dtod(&sx, &mut x.slice_mut(k * nv..(k + 1) * nv)).unwrap();
            }
            st.synchronize().unwrap();
            println!("  encode {e}: warm start, {warm} preconditioned CGLS iterations on the static image ({:.0}s)", tw.elapsed().as_secs_f64());
        }
        drop(pre);
        if !arms.is_empty() {
            assert!(precond_knot, "--arms needs --precond knot");
            let (gram, lam_eff, dinv) = kgram.take().unwrap();
            let evf = |h: &[C32]| lumen_compact(&ev, &cw, kr, mc, nv, nfo, h);
            let ac = ArmsCtx { eval: &evf, dev: &dev, st: &st, kern: &kern, eng: &eng, cnt: &cnt, seg, pts: &pts, pts_ho: pts_ho.as_ref(), y_ho: y_ho.as_ref(), m_ho, nc, m, nx, nv, kk: kr * mc, dims, out: &out, e };
            let opts = ArmsOpts { arms: arms.clone(), prefix, budget, gamma_rel, wthr, kappa, block, wlevels, admm_inner, power_its, llr_budget, priors: priors_sel.clone(), save_at: save_at.clone(), prior_dev, prior_chunk };
            let res = run_arms(&ac, &opts, kpre.take().unwrap(), gram, lam_eff, dinv, x, y, yh_keep.unwrap());
            for (name, snaps) in res {
                for (lab, h) in snaps {
                    arm_vals.entry((name.clone(), lab)).or_default().push((e, h));
                }
            }
            println!("  encode {e}: arms done ({:.0}s)", te.elapsed().as_secs_f64());
            timing.push(0.0);
            continue;
        }
        let tl = Instant::now();
        let mut snaps: Vec<(usize, Vec<C32>)> = vec![];
        let mut op0 = Op2d { eng: &eng, cnt: &cnt, seg, pts: &pts, kern: kern.clone(), nx, ny: nc * m_op };
        let mut rop = rcd.as_ref().map(|d| hasty_recon::rcompress::RcLin::new(Op2d { eng: &eng, cnt: &cnt, seg, pts: &pts, kern: kern.clone(), nx, ny: nc * m_op }, d, 0));
        let mut op = hasty_recon::rcompress::DynLin(match rop.as_mut() { Some(r) => r, None => &mut op0 });
        // held-out residual relative to the original held-out energy (compressed: ‖R A_S x − ỹ‖² + Σ (‖y‖² − ‖ỹ‖²))
        let ho_norm = ho_y2.unwrap_or_else(|| y_ho.as_ref().map_or(1.0, |v| kern.dot(&st, v, v).unwrap().re.max(1e-300)));
        let mut ho_pred = (m_ho > 0).then(|| st.alloc_zeros::<C32>(nc * m_ho).unwrap());
        let mut ho_hist: Vec<(usize, f64)> = vec![];
        let mut best: Option<(usize, f64, Vec<C32>)> = None;
        let hist = hasty_recon::cg::pcgls_dyn(&mut op, kpre.as_mut().map(|p| p as &mut dyn hasty_recon::cg::Precond), &st, hasty_recon::cg::Data::Owned(y), &mut x, 0.0, iters, &mut |k, xd| {
            if save_at.contains(&k) {
                snaps.push((k, st.clone_dtoh(xd).unwrap()));
            }
            if let (Some(ph), Some(yv), Some(pp)) = (pts_ho.as_ref(), y_ho.as_ref(), ho_pred.as_mut()) {
                if k % ho_every == 0 || k == iters {
                    eng.borrow_mut().forward(&st, seg, ph, xd, pp).unwrap();
                    if let Some(d) = rcd.as_ref() {
                        d.apply(&st, 1, pp, false).unwrap();
                    }
                    kern.axpby(&st, num_complex::Complex64::new(-1.0, 0.0), yv, num_complex::Complex64::new(1.0, 0.0), pp).unwrap();
                    let r = (kern.dot(&st, pp, pp).unwrap().re + ho_const) / ho_norm;
                    ho_hist.push((k, r));
                    if best.as_ref().is_none_or(|b| r < b.1) {
                        best = Some((k, r, st.clone_dtoh(xd).unwrap()));
                    }
                }
            }
        }, None)
        .unwrap()
        .0;
        if let Some((kb, rb, h)) = best.take() {
            println!("  encode {e}: held-out residual (rel. energy) {}; best at iteration {kb} ({rb:.4})", ho_hist.iter().map(|(k, r)| format!("{k}:{r:.4}")).collect::<Vec<_>>().join(" "));
            snaps.push((usize::MAX, h));
        }
        st.synchronize().unwrap();
        let tsolve = tl.elapsed().as_secs_f64();
        for (k, h) in snaps {
            let si = if k == usize::MAX { save_at.len() } else { save_at.iter().position(|&a| a == k).unwrap() };
            fill_vals(&ev, &cw, kr, mc, nv, ne, nfo, e, &h, &mut vals[si]);
        }
        let h0 = hist.first().copied().unwrap_or(1.0).max(1e-300);
        println!(
            "  encode {e}: {} readouts, {m} samples; {iters} CGLS iterations {tsolve:.1}s ({:.2}s / it); residual rel {:.3e} → {:.3e} ({:.0}s)",
            ros.len(),
            tsolve / iters as f64,
            hist.first().copied().unwrap_or(0.0) / h0,
            hist.last().copied().unwrap_or(0.0) / h0,
            te.elapsed().as_secs_f64()
        );
        timing.push(tsolve / iters as f64);
    }
    for (si, k) in save_at.iter().map(|k| k.to_string()).chain((ho_n > 0).then(|| "best".to_string())).enumerate() {
        for (s, (r, idx, _)) in ev.iter().enumerate() {
            let p = PathBuf::from(format!("{out}_r{r}_it{k}_lumen.npy"));
            write_c8_c(&p, &[idx.len(), ne, nfo], &vals[si][s]);
        }
    }
    for ((name, k), per_e) in &arm_vals {
        for (s, (r, idx, _)) in ev.iter().enumerate() {
            let mut full = vec![C32::new(0.0, 0.0); idx.len() * ne * nfo];
            for (e, slot) in per_e {
                for i in 0..idx.len() {
                    full[(i * ne + e) * nfo..(i * ne + e + 1) * nfo].copy_from_slice(&slot[s][i * nfo..(i + 1) * nfo]);
                }
            }
            let p = PathBuf::from(format!("{out}_{name}_r{r}_it{k}_lumen.npy"));
            write_c8_c(&p, &[idx.len(), ne, nfo], &full);
        }
    }
    if !arms.is_empty() {
        println!("done ({:.0}s): arms {:?}, outputs {out}_<arm>_r*_it*_lumen.hv, per-iteration {out}_arms_e*.csv", t0.elapsed().as_secs_f64(), arms);
        return;
    }
    let mean_it: f64 = timing.iter().sum::<f64>() / timing.len().max(1) as f64;
    println!("done ({:.0}s): mean {:.2}s per CGLS iteration per encode ({} knots per encode); outputs {out}_r*_it*_lumen.hv", t0.elapsed().as_secs_f64(), mean_it, kr * mc);
}


/// Data-driven affine respiratory motion, no labels or segmentation:
/// 1. encode 0, equal-count bins of the bellows amplitude (the real-scan surrogate), plain-SENSE preconditioned CGLS
///    static images on a coarser grid (all cardiac phases pooled);
/// 2. one smooth Eulerian field S shared by all bins, I_b(x) ≈ I_0(x − r_b S(x)) (`motion::fit_shared`; the support
///    mask = signal above 10 % of the 99th percentile of bin 0);
/// 3. a robust affine fit S(q) ≈ B (q − c) + t over the prescribed box (`--roi`, fractions of the FOV: the scanner's
///    prescription, not anatomy) ∩ support, weighted by magnitude and refitted with Cauchy weights on the residual
///    (IRLS, 5 rounds), so the dominant coherent motion in the box wins;
/// 4. writes A₁ (per unit bellows, physical), t₁ (m per unit bellows) relative to bin 0's surrogate, the field's affine
///    residual, and (optional `--truth-resp`, oracle check only) the phantom's displacement at the box centre.
fn estimate_motion(dir: &std::path::Path, nr: usize, scale: f64, roi: [(f64, f64); 3], truth_resp: Option<&std::path::Path>, out: &str) {
    use hasty_recon::op::SenseOp;
    use hasty_recon::precond::CirculantPrecond;
    let t0 = Instant::now();
    let mut sd = SimData::load_lazy(dir).unwrap();
    sd.set_recon_grid([scale; 3]);
    let (dims, nv, nc_all, vox) = (sd.dims, sd.nv(), sd.nc, sd.vox);
    let nro = sd.n_readouts();
    let bel: Vec<f64> = { let a = read_npy(&dir.join("physio.npy")).unwrap(); let (w, v) = (a.shape[1], a.f64s()); (0..nro).map(|i| v[i * w + 3]).collect() };
    let bins = hasty_recon::selfgate::amplitude_bins(&bel, nr);
    let w = sd.whitener();
    let grid = sd.physics_grid(false, &w);
    let en: Vec<f64> = (0..nc_all).map(|c| (0..nv).map(|q| { let z = grid.smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
    let mut order: Vec<usize> = (0..nc_all).collect();
    order.sort_by(|&a, &b| en[b].partial_cmp(&en[a]).unwrap());
    let keep: Vec<usize> = order[..12.min(nc_all)].to_vec();
    let nc = keep.len();
    let smaps: Vec<C32> = keep.iter().flat_map(|&c| grid.smaps[c * nv..(c + 1) * nv].iter().cloned()).collect();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let mut ims: Vec<Vec<f32>> = vec![];
    let mut rb = vec![];
    for b in 0..nr {
        let ros: Vec<usize> = sd.readouts_of(0, None).into_iter().filter(|&r| bins[r] == b).collect();
        rb.push(ros.iter().map(|&r| bel[r]).sum::<f64>() / ros.len() as f64);
        let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        let m = set.tau.len();
        let mut yh = vec![C32::new(0.0, 0.0); nc * m];
        for (ci, &c) in keep.iter().enumerate() {
            yh[ci * m..(ci + 1) * m].copy_from_slice(&set.y[c * m..(c + 1) * m]);
        }
        let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nc, vec![], 1e-4).unwrap();
        let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nc, 1e-3).unwrap();
        drop(set);
        let y = st.clone_htod(&yh).unwrap();
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        pcgls_cb(&mut op, Some(&mut pre), &st, &y, &mut x, 0.0, 10, &mut |_, _| {}).unwrap();
        let h = st.clone_dtoh(&x).unwrap();
        let mag: Vec<f32> = h.iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).collect();
        let mean = mag.iter().map(|&v| v as f64).sum::<f64>() / nv as f64;
        ims.push(mag.iter().map(|&v| (v as f64 / mean) as f32).collect());
        println!("  bin {b}: {} readouts, mean bellows {:.3} ({:.0}s)", ros.len(), rb[b], t0.elapsed().as_secs_f64());
    }
    let r_rel: Vec<f64> = rb.iter().map(|r| r - rb[0]).collect();
    let mut srt: Vec<f32> = ims[0].clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = srt[(0.99 * (nv - 1) as f64) as usize];
    let mask: Vec<bool> = ims[0].iter().map(|&v| v > 0.1 * p99).collect();
    let fit = hasty_recon::motion::fit_shared(&ims, &r_rel, dims, &mask, 8.0, 2, 1e-3, 60);
    let sfield = fit.field.dense();
    println!("  shared field: relative residual {:.3} (1 = no motion model), min Jacobian {:.3}", fit.rel_resid, fit.min_jac);
    // robust affine fit of S over the box ∩ support (voxel units, positions relative to the grid centre)
    let inbox = |q: usize| -> bool {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        let f = [i as f64 / dims[0] as f64, j as f64 / dims[1] as f64, k as f64 / dims[2] as f64];
        (0..3).all(|a| f[a] >= roi[a].0 && f[a] <= roi[a].1)
    };
    let qs: Vec<usize> = (0..nv).filter(|&q| mask[q] && inbox(q)).collect();
    let pos = |q: usize| -> [f64; 3] { [(q % dims[0]) as f64 - (dims[0] / 2) as f64, ((q / dims[0]) % dims[1]) as f64 - (dims[1] / 2) as f64, (q / (dims[0] * dims[1])) as f64 - (dims[2] / 2) as f64] };
    let mut wts: Vec<f64> = qs.iter().map(|&q| ims[0][q] as f64).collect();
    let mut coef = [[0.0f64; 4]; 3];
    let mut scale_r = 1.0;
    for round in 0..5 {
        let mut ata = nalgebra::DMatrix::<f64>::zeros(4, 4);
        let mut atb = nalgebra::DMatrix::<f64>::zeros(4, 3);
        for (i, &q) in qs.iter().enumerate() {
            let p = pos(q);
            let row = [p[0], p[1], p[2], 1.0];
            for a in 0..4 {
                for b in 0..4 {
                    ata[(a, b)] += wts[i] * row[a] * row[b];
                }
                for c in 0..3 {
                    atb[(a, c)] += wts[i] * row[a] * sfield[c * nv + q];
                }
            }
        }
        let sol = ata.lu().solve(&atb).unwrap();
        for c in 0..3 {
            for a in 0..4 {
                coef[c][a] = sol[(a, c)];
            }
        }
        let res: Vec<f64> = qs.iter().map(|&q| { let p = pos(q); (0..3).map(|c| (sfield[c * nv + q] - (coef[c][0] * p[0] + coef[c][1] * p[1] + coef[c][2] * p[2] + coef[c][3])).powi(2)).sum::<f64>().sqrt() }).collect();
        let mut rs = res.clone();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scale_r = (1.4826 * rs[rs.len() / 2]).max(1e-6);
        let rms = (res.iter().zip(&wts).map(|(r, w)| w * r * r).sum::<f64>() / wts.iter().sum::<f64>()).sqrt();
        println!("  affine IRLS round {round}: weighted residual rms {:.3} vox (robust scale {:.3} vox)", rms, scale_r);
        wts = qs.iter().zip(&res).map(|(&q, &r)| ims[0][q] as f64 / (1.0 + (r / (2.0 * scale_r)).powi(2))).collect();
    }
    // physical units: S in voxels per unit bellows (relative to bin 0); A₁[a][b] = B[a][b] Δ_a / Δ_b, t₁ = t Δ
    let a1: [[f64; 3]; 3] = [0, 1, 2].map(|a| [0, 1, 2].map(|b| coef[a][b] * vox[a] / vox[b]));
    let t1: [f64; 3] = [0, 1, 2].map(|a| coef[a][3] * vox[a]);
    let rmax = r_rel.iter().cloned().fold(0.0f64, f64::max);
    let smag: f64 = qs.iter().map(|&q| (0..3).map(|c| (sfield[c * nv + q] * vox[c] * 1e3).powi(2)).sum::<f64>().sqrt()).sum::<f64>() / qs.len() as f64 * rmax;
    println!("  box ∩ support {} voxels: mean |S| × r_max = {:.2} mm; affine t₁ × r_max = {:?} mm", qs.len(), smag, t1.map(|v| (v * rmax * 1e4).round() / 10.0));
    let mut js = serde_json::json!({"surrogate": "bellows", "r_ref": rb[0], "bins": nr, "bin_bellows": rb, "scale": scale, "roi": roi, "A1": a1, "t1_m": t1,
        "field_rel_resid": fit.rel_resid, "affine_robust_scale_vox": scale_r, "voxels": qs.len(), "mean_disp_mm_at_rmax": smag});
    if let Some(tp) = truth_resp {
        // oracle check only: the phantom's Lagrangian field at the box centre for the bins' true-state difference
        let ph = read_npy(&dir.join("physio.npy")).unwrap();
        let (w, v) = (ph.shape[1], ph.f64s());
        let rt: Vec<f64> = (0..nr).map(|b| { let r: Vec<f64> = (0..nro).filter(|&i| bins[i] == b).map(|i| v[i * w + 4]).collect(); r.iter().sum::<f64>() / r.len() as f64 }).collect();
        js["oracle_true_r_per_bin"] = serde_json::json!(rt);
        js["oracle_note"] = serde_json::json!(format!("compare the predicted displacement at the box centre with the phantom field {} at Δr = {:.3}", tp.display(), rt[nr - 1] - rt[0]));
    }
    let p = PathBuf::from(format!("{out}_affine.json"));
    std::fs::write(&p, serde_json::to_string_pretty(&js).unwrap()).unwrap();
    println!("wrote {} ({:.0}s)", p.display(), t0.elapsed().as_secs_f64());
}

/// the knot images `h` of encode `e` evaluated at every evaluation state and cardiac frame centre, into `slot[state]`
#[allow(clippy::too_many_arguments)]
fn fill_vals(ev: &[(f64, Vec<usize>, Vec<f64>)], cw: &[Vec<f64>], kr: usize, mc: usize, nv: usize, ne: usize, nfo: usize, e: usize, h: &[C32], slot: &mut [Vec<C32>]) {
    for (s, (_, idx, g)) in ev.iter().enumerate() {
        for (i, &q) in idx.iter().enumerate() {
            for f in 0..nfo {
                let mut z = (0.0f64, 0.0f64);
                for kk in 0..kr {
                    if g[kk] == 0.0 {
                        continue;
                    }
                    for b in 0..mc {
                        let wgt = g[kk] * cw[f][b];
                        if wgt == 0.0 {
                            continue;
                        }
                        let v = h[(kk * mc + b) * nv + q];
                        z.0 += wgt * v.re as f64;
                        z.1 += wgt * v.im as f64;
                    }
                }
                slot[s][(i * ne + e) * nfo + f] = C32::new(z.0 as f32, z.1 as f32);
            }
        }
    }
}

/// compact lumen values of one encode: per evaluation state, [voxel][frame] (`fill_vals` for a single encode)
fn lumen_compact(ev: &[(f64, Vec<usize>, Vec<f64>)], cw: &[Vec<f64>], kr: usize, mc: usize, nv: usize, nfo: usize, h: &[C32]) -> Vec<Vec<C32>> {
    let mut slot: Vec<Vec<C32>> = ev.iter().map(|(_, idx, _)| vec![C32::new(0.0, 0.0); idx.len() * nfo]).collect();
    fill_vals(ev, cw, kr, mc, nv, 1, nfo, 0, h, &mut slot);
    slot
}

/// resident host memory of this process (GiB), for the logs
fn rss_gib() -> f64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|t| t.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).unwrap_or(0.0) / (1u64 << 20) as f64
}

/// what the solver arms of one encode share
struct ArmsCtx<'a> {
    dev: &'a Device,
    st: &'a Arc<CudaStream>,
    kern: &'a Arc<Kernels>,
    eng: &'a std::cell::RefCell<SegNufft>,
    cnt: &'a std::cell::Cell<(usize, usize)>,
    seg: &'a SegSegments,
    pts: &'a SegPoints,
    pts_ho: Option<&'a SegPoints>,
    y_ho: Option<&'a CudaSlice<C32>>,
    m_ho: usize,
    nc: usize,
    m: usize,
    nx: usize,
    nv: usize,
    kk: usize,
    dims: [usize; 3],
    out: &'a str,
    e: usize,
    /// knot stack → compact lumen values (see [`lumen_compact`])
    eval: &'a dyn Fn(&[C32]) -> Vec<Vec<C32>>,
}

struct ArmsOpts {
    arms: Vec<String>,
    prefix: usize,
    budget: f64,
    gamma_rel: f64,
    wthr: f64,
    kappa: f64,
    block: usize,
    wlevels: usize,
    admm_inner: usize,
    power_its: usize,
    llr_budget: f64,
    priors: String,
    save_at: Vec<usize>,
    prior_dev: hasty_recon::knotncg::PriorDev,
    prior_chunk: usize,
}

/// held-out residual, snapshots at gradient counts and the per-iteration log of one arm
struct Tracker<'a, 'b> {
    ac: &'b ArmsCtx<'a>,
    name: String,
    offset: f64,
    budget: f64,
    save_at: Vec<usize>,
    next_save: usize,
    pred: Option<CudaSlice<C32>>,
    ho_norm: f64,
    snaps: Vec<(String, Vec<Vec<C32>>)>,
    best: Option<(f64, f64, Vec<Vec<C32>>)>,
    ho: Vec<(f64, f64)>,
    csv: String,
    t0: Instant,
}

impl<'a, 'b> Tracker<'a, 'b> {
    fn new(ac: &'b ArmsCtx<'a>, name: &str, o: &ArmsOpts, offset: f64) -> Self {
        let ho_norm = ac.y_ho.map_or(1.0, |v| ac.kern.dot(ac.st, v, v).unwrap().re.max(1e-300));
        Self { ac, name: name.into(), offset, budget: o.budget, save_at: o.save_at.clone(), next_save: 0, pred: (ac.m_ho > 0).then(|| ac.st.alloc_zeros::<C32>(ac.nc * ac.m_ho).unwrap()), ho_norm, snaps: vec![], best: None, ho: vec![], csv: String::new(), t0: Instant::now() }
    }

    fn grads(&self) -> f64 {
        let c = self.ac.cnt.get();
        (c.0 + c.1) as f64 / 2.0 + self.offset
    }

    /// after an iteration: held-out residual, snapshots, log; false = the gradient budget is spent
    fn on(&mut self, it: usize, info: Option<&hasty_recon::knotncg::Info>, x: &CudaSlice<C32>) -> bool {
        let ac = self.ac;
        let g = self.grads();
        let mut ho = f64::NAN;
        if let (Some(ph), Some(yv), Some(pp)) = (ac.pts_ho, ac.y_ho, self.pred.as_mut()) {
            ac.eng.borrow_mut().forward(ac.st, ac.seg, ph, x, pp).unwrap();
            ac.kern.axpby(ac.st, num_complex::Complex64::new(-1.0, 0.0), yv, num_complex::Complex64::new(1.0, 0.0), pp).unwrap();
            ho = ac.kern.dot(ac.st, pp, pp).unwrap().re / self.ho_norm;
            self.ho.push((g, ho));
            if self.best.as_ref().is_none_or(|b| ho < b.1) {
                self.best = Some((g, ho, (ac.eval)(&ac.st.clone_dtoh(x).unwrap())));
            }
        }
        let mut h: Option<Vec<Vec<C32>>> = None;
        while self.next_save < self.save_at.len() && g + 1e-9 >= self.save_at[self.next_save] as f64 {
            let v = h.get_or_insert_with(|| (ac.eval)(&ac.st.clone_dtoh(x).unwrap())).clone();
            self.snaps.push((self.save_at[self.next_save].to_string(), v));
            self.next_save += 1;
        }
        let nan = f64::NAN;
        let (f, pw, pl, step, beta, rs, nw, nl) = info.map_or((nan, nan, nan, nan, nan, false, 0, 0), |i| (i.f, i.pw, i.pl, i.step, i.beta, i.restart, i.n_wt, i.n_llr));
        let fp = |v: f64| if v.is_nan() { 0.0 } else { v };
        let ftot = f + fp(pw) + fp(pl);
        let secs = self.t0.elapsed().as_secs_f64();
        self.csv += &format!("{},{it},{g:.2},{f:.8e},{pw:.8e},{pl:.8e},{ftot:.8e},{ho:.6e},{step:.4e},{beta:.4e},{},{nw},{nl},{secs:.1}\n", self.name, rs as u8);
        println!("    [{} e{}] it {it:3} grads {g:5.1}: f {f:.5e} Pw {pw:.4e} Pl {pl:.4e} F {ftot:.6e} held-out {ho:.5} step {step:.3e} β {beta:.3}{} (wt {nw}, llr {nl}, {secs:.0}s)", self.name, ac.e, if rs { " R" } else { "" });
        g + 1e-9 < self.budget
    }

    /// snapshots incl. the best held-out iterate ("best")
    fn finish(mut self) -> (Vec<(String, Vec<Vec<C32>>)>, String) {
        if let Some((g, r, h)) = self.best.take() {
            println!("  [{} e{}] held-out: {}; best at {g:.1} gradients ({r:.5})", self.name, self.ac.e, self.ho.iter().map(|(g, r)| format!("{g:.0}:{r:.5}")).collect::<Vec<_>>().join(" "), );
            self.snaps.push(("best".into(), h));
        }
        (self.snaps, self.csv)
    }
}

/// The solver comparison of one encode (research/convergence_strategy.md §8 P1–P2; see `--arms`). `x` = the warm
/// start, `y` the fit data (device; consumed by the PCG arm), `yh` the same on the host (re-uploaded per arm).
#[allow(clippy::too_many_arguments)]
fn run_arms(ac: &ArmsCtx, o: &ArmsOpts, mut kpre: hasty_recon::knotsolve::KnotPrecond, gram: Vec<f64>, lam_eff: Vec<f32>, dinv: Vec<f32>, mut x: CudaSlice<C32>, y: CudaSlice<C32>, yh: Vec<C32>) -> Vec<(String, Vec<(String, Vec<Vec<C32>>)>)> {
    use hasty_recon::knotncg::{self as kn, Info, LlrBackend, LlrHuber, PriorMetric, Priors, WavHuber};
    let (st, kern, e, k, nv) = (ac.st, ac.kern, ac.e, ac.kk, ac.nv);
    let gib = (1u64 << 30) as f64;
    let mut results = vec![];
    let mut csv = String::from("arm,it,grads,f,pw,pl,F,heldout,step,beta,restart,n_wt,n_llr,secs\n");
    let mut pm = PriorMetric::new(ac.dev, st, k, ac.dims, &gram, 1e-3, lam_eff, dinv).unwrap();
    let bulk = pm.bulk();
    let (lmin, lmax) = pm.lam_g.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
    println!(
        "  arms e{e}: {} unknowns, {} fit samples → R = unknowns / samples = {:.1} (per coil); bulk curvature {bulk:.4e} (D_s {:.3e}, mean λ_G {:.3e}, λ_G range {:.1}); budget {} gradients; arms {:?}",
        ac.nx, ac.m, ac.nx as f64 / ac.m as f64, pm.d_s, pm.lam_g.iter().sum::<f64>() / k as f64, lmax / lmin, o.budget, o.arms
    );
    {
        // the data symbol's floor: the share of k-space at the metric's floor (prior-only band at high R)
        let mut c = pm.c.clone();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (cmin, cmed, cmax) = (c[0] as f64, c[c.len() / 2] as f64, c[c.len() - 1] as f64);
        let at_floor = c.iter().filter(|&&v| (v as f64) <= 1.01 * cmin).count() as f64 / c.len() as f64;
        println!("  arms e{e}: data symbol c(k): min {cmin:.3e}, median {cmed:.3e}, max {cmax:.3e} (range {:.0}); {:.1} % of k at the floor", cmax / cmin, 100.0 * at_floor);
    }
    let mk_op = || Op2d { eng: ac.eng, cnt: ac.cnt, seg: ac.seg, pts: ac.pts, kern: kern.clone(), nx: ac.nx, ny: ac.nc * ac.m };
    // ---- PCG (the arms v2 path: CGLS with G⁻¹ ⊗ C⁻¹) and the prefix iterate of the prior arms
    let run_pcg = o.arms.iter().any(|a| a == "pcg");
    let mut x5: Option<(Vec<C32>, (usize, usize))> = None;
    {
        ac.cnt.set((0, 0));
        let mut tr = Tracker::new(ac, "pcg", o, 0.0);
        tr.on(0, None, &x);
        let its = if run_pcg { (o.budget - 1.0).ceil().max(o.prefix as f64) as usize } else { o.prefix };
        let mut op = mk_op();
        let prefix = o.prefix as f64;
        hasty_recon::cg::pcgls_dyn(&mut op, Some(&mut kpre as &mut dyn hasty_recon::cg::Precond), st, hasty_recon::cg::Data::Owned(y), &mut x, 0.0, its, &mut |k, xd| {
            let g = tr.grads();
            if x5.is_none() && g + 1e-9 >= prefix {
                x5 = Some((st.clone_dtoh(xd).unwrap(), ac.cnt.get()));
            }
            if run_pcg {
                tr.on(k, None, xd);
            }
        }, None)
        .unwrap();
        if run_pcg {
            let (snaps, c) = tr.finish();
            csv += &c;
            results.push(("pcg".to_string(), snaps));
        }
    }
    drop(kpre);
    let prior_arms: Vec<&String> = o.arms.iter().filter(|a| *a != "pcg").collect();
    if !prior_arms.is_empty() {
        let (x5h, c5) = x5.take().expect("prefix iterate");
        let gamma = o.gamma_rel / bulk;
        // ---- prior weights and the prior-aware metric from the prefix iterate (data-derived, no truth)
        let tp = Instant::now();
        let mut wav = WavHuber::new(ac.dims, o.wlevels, 0.0, gamma);
        let mut cw = vec![];
        wav.coef(&x5h, k, &mut cw);
        let sig_w = wav.sigma_finest(&cw, k);
        let lam_w = bulk * o.wthr * sig_w;
        wav.theta = lam_w * gamma;
        let fr = wav.curvature_fractions(&cw, k);
        let sw = wav.symbol(&fr);
        drop(cw);
        let mut l = hasty_recon::llr::Llr::new(ac.dims, o.block, 1.0, vec![true; nv]);
        l.zeta_basis = Some(pm.u.clone());
        let mut lg = hasty_recon::llr_gpu::LlrGpu::new(l, &[0]).unwrap();
        lg.budget = (o.llr_budget * gib) as usize;
        let mut lh = LlrHuber { be: LlrBackend::Gpu(lg), theta: 0.0, gamma, shift: 1 };
        let mut tmp = x5h.clone();
        {
            let l = lh.be.llr();
            l.thr_abs = None;
            l.set_iteration(1);
        }
        lh.be.apply(&mut tmp, k).unwrap();
        let sig_l = lh.be.llr().last.sigma;
        let lam_l = bulk * o.kappa * sig_l;
        lh.theta = lam_l * gamma;
        tmp.copy_from_slice(&x5h);
        let zs = lh.svt(&mut tmp, k, lh.theta).unwrap();
        lh.be.llr().zeta_basis = None;
        drop(tmp);
        let (use_w, use_l) = (o.priors.contains("wav"), o.priors.contains("llr"));
        let sl: Vec<f64> = if use_l { zs.zeta.iter().map(|z| z.max(0.0) / gamma).collect() } else { vec![0.0; k] };
        let swo = use_w.then_some(sw);
        println!(
            "  arms e{e}: priors {}: γ = {:.2}/bulk = {gamma:.4e}; wavelet db4 L{} σ̂_w {sig_w:.4e}, λ_w = bulk·{}·σ̂_w = {lam_w:.4e}, θ_w {:.4e}; LLR B{} σ̂_l {sig_l:.4e}, λ_l = bulk·{}·σ̂_l = {lam_l:.4e}, θ_l {:.4e} (kept {} of {} singular values at θ_l) ({:.0}s)",
            o.priors, o.gamma_rel, o.wlevels, o.wthr, wav.theta, o.block, o.kappa, lh.theta, zs.n_above, zs.n_sv, tp.elapsed().as_secs_f64()
        );
        println!("    wavelet curvature fractions (level, ε, frac): {}", fr.iter().map(|(l, e, f)| format!("{l}{}{}{}:{f:.3}", e[0], e[1], e[2])).collect::<Vec<_>>().join(" "));
        println!("    LLR discarded fraction ζ(ω) (ω in G-eigenvector order): {}", zs.zeta.iter().map(|z| format!("{z:.3}")).collect::<Vec<_>>().join(" "));
        let mut pri = Priors::new(k, nv, use_w.then_some(wav), use_l.then_some(lh));
        if o.prior_dev == kn::PriorDev::Gpu {
            pri.to_gpu(ac.dev, st, o.prior_chunk).unwrap();
        }
        println!("  arms e{e}: priors on the {:?} (wavelet chunk {} knots)", pri.dev(), o.prior_chunk);
        let set_metric = |pm: &mut PriorMetric, arm: &str| {
            pm.ident = (arm == "ncgI").then_some(bulk);
            if arm == "ncg0" || arm == "ncgI" {
                pm.set_prior(st, &vec![0.0; k], None).unwrap();
            } else if arm == "ncgcap" {
                // the data metric capped by the prior curvature: G ⊗ C + 1/γ
                pm.set_prior(st, &vec![1.0 / gamma; k], None).unwrap();
            } else if let Some(v) = arm.strip_prefix("admm") {
                let rho: f64 = v.parse::<f64>().unwrap() * bulk;
                let ns = use_w as usize + use_l as usize;
                pm.set_prior(st, &vec![ns as f64 * rho; k], None).unwrap();
            } else {
                pm.set_prior(st, &sl, swo.as_deref()).unwrap();
            }
        };
        // ---- λ_max(P∇²F_γ) for FISTA / ZMSHR1 (charged to both)
        let (mut lip, mut pow_cost) = (0.0, 0.0);
        if prior_arms.iter().any(|a| *a == "fista" || *a == "zmshr1") {
            set_metric(&mut pm, "ncg");
            st.memcpy_htod(&x5h, &mut x).unwrap();
            let c0 = ac.cnt.get();
            let mut op = mk_op();
            let (l, h) = kn::power_lmax(&mut op, kern, st, &pm, &mut pri, &x, ac.nc * ac.m, o.power_its).unwrap();
            let c1 = ac.cnt.get();
            pow_cost = ((c1.0 - c0.0) + (c1.1 - c0.1)) as f64 / 2.0;
            lip = 1.1 * l;
            println!("  arms e{e}: power iteration λ_max(P∇²F_γ) {:?} → L = 1.1 × {l:.4} ({pow_cost} gradients, charged to fista / zmshr1)", h.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
            // diagnostic (not charged): the data term alone in the same metric, λ_max(P AᴴA)
            let mut none = Priors::new(k, nv, None, None);
            let (ld, hd) = kn::power_lmax(&mut op, kern, st, &pm, &mut none, &x, ac.nc * ac.m, o.power_its).unwrap();
            println!("  arms e{e}: diagnostic λ_max(P AᴴA) (data term only, not charged) {:?} → {ld:.4}", hd.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
        }
        for arm in prior_arms {
            let arm = arm.as_str();
            set_metric(&mut pm, arm);
            st.memcpy_htod(&x5h, &mut x).unwrap();
            pri.n_wt = 0;
            pri.n_llr = 0;
            pri.secs = 0.0;
            let offset = if arm == "fista" || arm == "zmshr1" { pow_cost } else { 0.0 };
            let mut tr = Tracker::new(ac, arm, o, offset);
            let mut op = mk_op();
            let ydev = st.clone_htod(&yh).unwrap();
            // the residual at the prefix iterate (carried over from PCG in a single run: not charged; the counter is
            // reset to the prefix's count after each solver's initialisation)
            let resid = |op: &mut Op2d, mut yv: CudaSlice<C32>| -> CudaSlice<C32> {
                let mut q = st.alloc_zeros::<C32>(yv.len()).unwrap();
                op.forward(st, &x, &mut q).unwrap();
                kern.axpby(st, num_complex::Complex64::new(-1.0, 0.0), &q, num_complex::Complex64::new(1.0, 0.0), &mut yv).unwrap();
                yv
            };
            let ta = Instant::now();
            // continuation stages (ncgc) stop at stage_end; the counter is reset to the prefix count only at the first
            // initialisation
            let stage_end = std::cell::Cell::new(f64::INFINITY);
            let first = std::cell::Cell::new(true);
            let mut cb = |i: &Info, xd: &CudaSlice<C32>| -> bool {
                if i.iter == 0 && first.get() {
                    ac.cnt.set(c5);
                    first.set(false);
                }
                let go = tr.on(i.iter, Some(i), xd);
                let c = ac.cnt.get();
                go && ((c.0 + c.1) as f64 / 2.0) + 1e-9 < stage_end.get()
            };
            match arm {
                "ncgc" => {
                    // γ continuation 1 → 0.1 → 0.01 per bulk, equal gradient shares, NCG restarted per stage (the
                    // metric's prior symbol is kept: outside the quadratic zone the curvature λ/|c| does not depend on γ)
                    let mut r = resid(&mut op, ydev);
                    let span = (o.budget - o.prefix as f64) / 3.0;
                    for (si, gr) in [1.0f64, 0.1, 0.01].into_iter().enumerate() {
                        let g = gr / bulk;
                        if let Some(w) = pri.wav.as_mut() {
                            w.gamma = g;
                            w.theta = lam_w * g;
                        }
                        if let Some(l) = pri.llr.as_mut() {
                            l.gamma = g;
                            l.theta = lam_l * g;
                        }
                        stage_end.set(o.prefix as f64 + span * (si + 1) as f64);
                        println!("    [ncgc e{e}] stage {si}: γ = {gr}/bulk until {:.1} gradients", stage_end.get());
                        r = kn::ncg(&mut op, kern, st, &pm, &mut pri, &mut x, r, &mut cb).unwrap();
                        let c = ac.cnt.get();
                        if (c.0 + c.1) as f64 / 2.0 + 1e-9 >= o.budget {
                            break;
                        }
                    }
                    if let Some(w) = pri.wav.as_mut() {
                        w.gamma = gamma;
                        w.theta = lam_w * gamma;
                    }
                    if let Some(l) = pri.llr.as_mut() {
                        l.gamma = gamma;
                        l.theta = lam_l * gamma;
                    }
                }
                "ncg" | "ncg0" | "ncgcap" | "ncgI" => {
                    let r = resid(&mut op, ydev);
                    kn::ncg(&mut op, kern, st, &pm, &mut pri, &mut x, r, &mut cb).unwrap();
                }
                "zmshr1" => {
                    let r = resid(&mut op, ydev);
                    kn::zmshr1(&mut op, kern, st, &pm, &mut pri, &mut x, r, 1.0 / lip, &mut cb).unwrap();
                }
                "fista" => kn::fista(&mut op, kern, st, &pm, &mut pri, &mut x, ydev, lip, &mut cb).unwrap(),
                a if a.starts_with("admm") => {
                    let rho = a[4..].parse::<f64>().unwrap() * bulk;
                    let r = resid(&mut op, ydev);
                    kn::admm(&mut op, kern, st, &pm, &mut pri, &mut x, r, rho, o.admm_inner, &mut cb).unwrap();
                }
                o => panic!("unknown arm {o} (pcg, ncg, ncg0, ncgcap, ncgI, ncgc, zmshr1, fista, admm<ρ/bulk>)"),
            }
            println!("  arms e{e}: {arm} done ({:.0}s; priors {:.0}s ({:?}): {} wavelet stack transforms, {} LLR sweeps; host RSS {:.1} GiB)", ta.elapsed().as_secs_f64(), pri.secs, pri.dev(), pri.n_wt, pri.n_llr, rss_gib());
            let (snaps, c) = tr.finish();
            csv += &c;
            results.push((arm.to_string(), snaps));
        }
    }
    std::fs::write(format!("{}_arms_e{e}.csv", ac.out), csv).unwrap();
    results
}
