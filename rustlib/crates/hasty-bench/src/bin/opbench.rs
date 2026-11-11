//! Encoding-operator benchmark and reconstruction-time projection for realistic scans (GPU).
//!
//!   opbench [--seq flow,b0] [--flow-run data/hasty/seq/flow_thorax] [--scan-min 10] [--b0-scan-min 2.5]
//!           [--fov 0.45,0.45,0.22] [--res 2e-3] [--frames 20] [--rr 1.0] [--vcoils 12] [--b0 2.89]
//!           [--models sense,seg-none,seg-b0,seg-full] [--nseg 10] [--order 2] [--b0-nseg 6] [--ntab 256]
//!           [--temporal off|linear|cubic] [--iters 40] [--sets 2] [--reps 5] [--adc-step 1]
//!           [--fov-centre 0,0,0.05] [--no-gnl] [--no-tune] [--max-grid-gb 6] [--quick] [--json file] [--plan]
//!
//! Defaults: the user's 10-minute free-breathing 4D-flow scan (flow_thorax: TR 11.29 ms, 7 encodes, 3990 ADC samples
//! at 2 µs per readout → ≈ 53 000 readouts, 2.1e8 samples), 12 virtual coils (32 → 12 compression happens before
//! the operator), 20 cardiac frames, 2 mm isotropic over a 450 × 450 × 220 mm FOV (grid rounded up to 2·3·5·7-smooth
//! sizes), and the 2.5-minute B0 / R2* / fat multi-echo scan (`simphantom --seq b0`: 24-lobe train, TE shifts 0 and
//! ½ lobe, 3 µs dwell; one set per (TE shift, k = 0 crossing) as `recon_b0`).
//!
//! Measured per model on `--sets` representative sets (smallest / largest (frame, encode) set; `--sets 1`: the median)
//! with the real sample positions, readout times and concomitant coefficients of the design (ideal GIRF) and the
//! reconstruction's own table builders (`segphys`); coil maps and field map are synthetic (smooth maps around the body,
//! smooth field + a −3.4 ppm shell, clinical GNL model) — values do not change the cost:
//!   * forward, adjoint and normal (forward + adjoint, i.e. the operator part of one P²nP gradient) per set, CUDA events,
//!     median of `--reps` after a warm-up; `sense` includes the per-call `set_points` of `SenseSets` (reported too);
//!   * breakdown: SegNufft tile kernels (gridding + segment / ΔC physics, from `autotune`), the partial-axis FFTs
//!     (timed separately on the same channel-interleaved grid) and the rest (pad / crop / memsets); SENSE: set_points,
//!     FFT, rest (spreading / interpolation, deconvolution, coil expand / reduce). The cost of the physics itself is the
//!     difference between `seg-none` (L = 1), `seg-b0` (L segments) and `seg-full` (L segments + ΔC correction);
//!   * device memory: operator engine (fine-grid slots, FFT plans, maps), segment tables per encode, per-sample bytes of
//!     the sample tables, measured with cuMemGetInfo.
//! Projection: per-set times fitted as t(M) = a + b·M over the measured sets, summed over all sets of the scan; plus
//! k-space data streaming when the resident problem does not fit the card (pinned host → device at the measured
//! bandwidth), the host round trip of the image stack for the prior (as `p2np::run_objective_with` does every
//! iteration), the ZMSHR1 vector passes (≈ 10 passes over the stack at the measured device bandwidth); setup (tables,
//! tile lists, segments, step-size power iteration) separately. Projected total for `--iters` iterations.
//! Prior cost (LLR / learned denoiser) is not included.
//!
//! `--plan`: CPU-only dry run (schedules, set statistics, the median set's host tables, formula memory estimates).
//! `--quick`: scaled-down smoke run (4 mm, 4 virtual coils, every 2nd ADC sample, 1 set per model, 3 reps): the
//! projection is then for that scaled problem.
//! Expected run time on an RTX 4090 (defaults): ~5–10 min (physics grid with GNL ≈ 30 s CPU, segments ≈ 5 s per encode,
//! autotune ≈ 10–30 s per seg model the first time, then seconds per measurement).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::CudaStream;
use hasty_bench::seqsched::{EchoSchedule, FlowSchedule};
use hasty_cuda::{Device, C32};
use hasty_fft::{Direction, FftPlan, Norm};
use hasty_mrsim::gnl::GradientCoils;
use hasty_mrsim::physio::{CardiacTime, Ecg};
use hasty_mrsim::sequence::{Schedule, SAMPLE_FLOATS};
use hasty_mrsim::{Vec3, GAMMA_BAR, IDENTITY};
use hasty_nufft::{build_segments, next_smooth_even, EsKernel, SegNufft, SegNufftOpts};
use hasty_recon::data::{smooth7, PhysicsGrid, ReadoutInfo, SampleSet, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_recon::segphys::{fit_monomials, mean_coefficients, seg_tables, segment_phase, MonoFit};
use num_complex::Complex64;
use rayon::prelude::*;

const GB: f64 = (1u64 << 30) as f64;

#[derive(Clone)]
struct Args {
    seqs: Vec<String>,
    flow_run: PathBuf,
    scan_min: f64,
    b0_scan_min: f64,
    fov: Vec3,
    res: f64,
    frames: usize,
    rr: f64,
    nvc: usize,
    b0: f64,
    models: Vec<String>,
    nseg: usize,
    order: u32,
    b0_nseg: usize,
    ntab: usize,
    temporal: Option<u32>,
    iters: usize,
    nsets: usize,
    reps: usize,
    adc_step: usize,
    fov_c: Vec3,
    gnl: bool,
    tune: bool,
    max_grid: usize,
    json: Option<PathBuf>,
    /// CPU-only dry run: schedules, sets, one set's host tables, memory estimates (no GPU)
    plan: bool,
}

fn parse() -> Args {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |n: &str| a.iter().position(|x| x == n).map(|i| { a.remove(i); true }).unwrap_or(false);
    let (quick, no_gnl, no_tune, plan) = (flag("--quick"), flag("--no-gnl"), flag("--no-tune"), flag("--plan"));
    let mut take = |n: &str| -> Option<String> { a.iter().position(|x| x == n).map(|i| { a.remove(i); a.remove(i) }) };
    let v3 = |s: String| -> Vec3 { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); [v[0], v[1], v[2]] };
    let args = Args {
        seqs: take("--seq").unwrap_or("flow,b0".into()).split(',').map(String::from).collect(),
        flow_run: take("--flow-run").map(PathBuf::from).unwrap_or_else(|| hasty_site::data().join("seq/flow_thorax")),
        scan_min: take("--scan-min").map(|s| s.parse().unwrap()).unwrap_or(10.0),
        b0_scan_min: take("--b0-scan-min").map(|s| s.parse().unwrap()).unwrap_or(2.5),
        fov: take("--fov").map(v3).unwrap_or([0.45, 0.45, 0.22]),
        res: take("--res").map(|s| s.parse().unwrap()).unwrap_or(if quick { 4e-3 } else { 2e-3 }),
        frames: take("--frames").map(|s| s.parse().unwrap()).unwrap_or(20),
        rr: take("--rr").map(|s| s.parse().unwrap()).unwrap_or(1.0),
        nvc: take("--vcoils").map(|s| s.parse().unwrap()).unwrap_or(if quick { 4 } else { 12 }),
        b0: take("--b0").map(|s| s.parse().unwrap()).unwrap_or(2.89),
        models: take("--models").unwrap_or(if quick { "sense,seg-full".into() } else { "sense,seg-none,seg-b0,seg-full".into() }).split(',').map(String::from).collect(),
        nseg: take("--nseg").map(|s| s.parse().unwrap()).unwrap_or(10),
        order: take("--order").map(|s| s.parse().unwrap()).unwrap_or(2),
        b0_nseg: take("--b0-nseg").map(|s| s.parse().unwrap()).unwrap_or(6),
        ntab: take("--ntab").map(|s| s.parse().unwrap()).unwrap_or(256),
        temporal: match take("--temporal").as_deref() {
            None | Some("off") => None,
            Some("linear") => Some(1),
            Some("cubic") => Some(3),
            Some(o) => panic!("--temporal {o}: off | linear | cubic"),
        },
        iters: take("--iters").map(|s| s.parse().unwrap()).unwrap_or(40),
        nsets: take("--sets").map(|s| s.parse().unwrap()).unwrap_or(if quick { 1 } else { 2 }),
        reps: take("--reps").map(|s| s.parse().unwrap()).unwrap_or(if quick { 3 } else { 5 }),
        adc_step: take("--adc-step").map(|s| s.parse().unwrap()).unwrap_or(if quick { 2 } else { 1 }),
        fov_c: take("--fov-centre").map(v3).unwrap_or([0.0, 0.0, 0.05]),
        gnl: !no_gnl,
        tune: !no_tune,
        max_grid: (take("--max-grid-gb").map(|s| s.parse::<f64>().unwrap()).unwrap_or(6.0) * GB) as usize,
        json: take("--json").map(PathBuf::from),
        plan,
    };
    assert!(a.is_empty(), "unknown arguments {a:?}");
    args
}

// ───────────────────────────────────────────────────────────── scans and sample sets

#[derive(Clone, Debug)]
struct SetDef {
    label: String,
    enc: usize,
    /// scan readouts of the set
    ros: Vec<usize>,
    /// ADC sample range of every readout (the whole readout, or an echo window)
    adc: (usize, usize),
}

impl SetDef {
    fn na(&self, step: usize) -> usize {
        (self.adc.1 - self.adc.0).div_ceil(step)
    }
    fn m(&self, step: usize) -> usize {
        self.ros.len() * self.na(step)
    }
}

struct Scan {
    name: String,
    describe: String,
    sch: Box<dyn Schedule>,
    /// sets of the static reconstruction (frame × encode, or TE shift × crossing)
    sets: Vec<SetDef>,
    /// sets of the temporal reconstruction (one per encode, all frames); empty if not applicable
    tsets: Vec<SetDef>,
    n_enc: usize,
    frames: usize,
    nseg: usize,
    scan_s: f64,
}

fn flow_scan(a: &Args) -> Scan {
    let scan_s = a.scan_min * 60.0;
    let ecg = Ecg::new(scan_s + 2.0, a.rr, 0.05, 1);
    let fs = FlowSchedule::load(&a.flow_run, scan_s, 2e-6, a.b0, ecg, a.frames, hasty_seq::girf::Girf::ideal(), "ideal");
    let (n_ro, na, n_enc) = (fs.n_readouts(), fs.n_adc(), fs.n_enc);
    let mut by: Vec<Vec<usize>> = vec![vec![]; a.frames * n_enc];
    let mut by_enc: Vec<Vec<usize>> = vec![vec![]; n_enc];
    for i in 0..n_ro {
        let row = fs.readout_row(i);
        let (e, f) = (row[1] as usize, row[3] as usize);
        by[f * n_enc + e].push(i);
        by_enc[e].push(i);
    }
    let sets = by.into_iter().enumerate().map(|(t, ros)| SetDef { label: format!("frame {} encode {}", t / n_enc, t % n_enc), enc: t % n_enc, ros, adc: (0, na) }).collect();
    let tsets = by_enc.into_iter().enumerate().map(|(e, ros)| SetDef { label: format!("encode {e} (all frames)"), enc: e, ros, adc: (0, na) }).collect();
    let describe = fs.describe();
    Scan { name: format!("4D flow {:.1} min", a.scan_min), describe, sch: Box::new(fs), sets, tsets, n_enc, frames: a.frames, nseg: a.nseg, scan_s }
}

fn b0_scan(a: &Args) -> Scan {
    use hasty_seq::multiecho::{fit_slew_to_pns, EchoSpec, EchoTrain};
    let scan_s = a.b0_scan_min * 60.0;
    // as `simphantom --seq b0` (defaults: 2 mm, 24 lobes, PNS 80 %, 3 µs dwell)
    let sys = hasty_seq::system::System::prisma_xr();
    let hw = hasty_seq::pns::SafeHw::example();
    let mut spec = EchoSpec::new(2e-3, 24);
    let t0 = fit_slew_to_pns(&spec, &sys, &hw, 80.0, 6);
    spec.slew_frac = t0.spec.slew_frac;
    spec.te_shifts = vec![0.0, sys.ceil_raster(0.5 * t0.t_lobe)];
    let train = EchoTrain::design(&spec, &sys);
    let te: Vec<Vec<f64>> = spec.te_shifts.iter().map(|&s| train.te(s)).collect();
    let t_lobe = train.t_lobe;
    let sch = EchoSchedule::new(train, scan_s, 3e-6, a.b0, Ecg::new(scan_s + 2.0, a.rr, 0.05, 1), 1);
    let (n_ro, nsh, n_echo) = (sch.n_readouts(), spec.te_shifts.len(), te[0].len());
    // every sample to its nearest k = 0 crossing (recon_b0 / b0data)
    let mut sets = Vec::new();
    for s in 0..nsh {
        let times = sch.adc_times(s);
        let win: Vec<usize> = times.iter().map(|&t| (((t - te[s][0]) / t_lobe).round().max(0.0) as usize).min(n_echo - 1)).collect();
        let ros: Vec<usize> = (0..n_ro).filter(|&i| i % nsh == s).collect();
        for e in 0..n_echo {
            let (Some(n0), Some(n1)) = (win.iter().position(|&w| w == e), win.iter().rposition(|&w| w == e)) else { continue };
            sets.push(SetDef { label: format!("shift {s} crossing {e}"), enc: s * n_echo + e, ros: ros.clone(), adc: (n0, n1 + 1) });
        }
    }
    let describe = sch.describe();
    let n_enc = sets.len();
    Scan { name: format!("B0/R2*/fat {:.1} min", a.b0_scan_min), describe, sch: Box::new(sch), sets, tsets: vec![], n_enc, frames: 1, nseg: a.b0_nseg, scan_s }
}

/// a set's sample set (logical = physical axes, coordinates k·Δ) and the cardiac time of its readouts
fn build_set(sch: &dyn Schedule, set: &SetDef, vox: f64, step: usize, rr: f64) -> (SampleSet, Vec<CardiacTime>) {
    let adc: Vec<usize> = (set.adc.0..set.adc.1).step_by(step).collect();
    let per: Vec<(Vec<[f32; 3]>, Vec<f64>, Vec<[f64; 4]>)> = set
        .ros
        .par_iter()
        .map(|&i| {
            let p = sch.sample_params(i..i + 1);
            let mut k = Vec::with_capacity(adc.len());
            let mut tau = Vec::with_capacity(adc.len());
            let mut w = Vec::with_capacity(adc.len());
            for &n in &adc {
                let r = &p[n * SAMPLE_FLOATS..(n + 1) * SAMPLE_FLOATS];
                k.push([0, 1, 2].map(|a| (r[a] as f64 * vox) as f32));
                w.push([r[9] as f64, r[10] as f64, r[11] as f64, r[12] as f64]);
                tau.push(r[13] as f64);
            }
            (k, tau, w)
        })
        .collect();
    let m = set.ros.len() * adc.len();
    let mut coords = [Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m)];
    let (mut tau, mut wq) = (Vec::with_capacity(m), Vec::with_capacity(m));
    for (k, t, w) in per {
        for v in &k {
            for a in 0..3 {
                coords[a].push(v[a]);
            }
        }
        tau.extend(t);
        wq.extend(w);
    }
    let card = set.ros.iter().map(|&i| { let ph = sch.readout_row(i)[4]; CardiacTime { beat: 0, since_r: ph * rr, rr, phase: ph } }).collect();
    (SampleSet { readouts: (0..set.ros.len()).collect(), coords, y: vec![], tau, wq }, card)
}

/// representative sets: smallest and largest (2), median (1), or spread by size (n)
fn pick(sets: &[SetDef], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..sets.len()).filter(|&i| !sets[i].ros.is_empty()).collect();
    idx.sort_by_key(|&i| sets[i].m(1));
    match n {
        0 | 1 => vec![idx[idx.len() / 2]],
        n => {
            let mut v: Vec<usize> = (0..n).map(|k| idx[k * (idx.len() - 1) / (n - 1)]).collect();
            v.dedup();
            v
        }
    }
}

// ───────────────────────────────────────────────────────────── physics (synthetic maps, real sequence)

fn physics(a: &Args, dims: [usize; 3], nc: usize, frames: usize) -> (SimData, PhysicsGrid, Vec<usize>, MonoFit) {
    let nv = dims.iter().product::<usize>();
    let res = a.res;
    let vlog = |q: usize| -> Vec3 {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [(i as f64 - (dims[0] / 2) as f64) * res, (j as f64 - (dims[1] / 2) as f64) * res, (k as f64 - (dims[2] / 2) as f64) * res]
    };
    let ext = dims.map(|d| d as f64 * res);
    // coils on an elliptic ring around the body (two rows), 1/r²-like falloff, smooth phase
    let coil_pos: Vec<Vec3> = (0..nc).map(|c| { let ang = 2.0 * std::f64::consts::PI * c as f64 / nc as f64; [0.24 * ang.cos(), 0.17 * ang.sin(), if c % 2 == 0 { 0.05 } else { -0.05 }] }).collect();
    let smaps_true: Vec<C32> = (0..nc * nv).into_par_iter().map(|i| {
        let (c, q) = (i / nv, i % nv);
        let p = vlog(q);
        let d2 = (0..3).map(|k| (p[k] - coil_pos[c][k]).powi(2)).sum::<f64>();
        let z = Complex64::from_polar(0.01 / (d2 + 0.004), 4.0 * p[0] - 3.0 * p[1] + 2.0 * (c as f64) * p[2]);
        C32::new(z.re as f32, z.im as f32)
    }).collect();
    let inside = |p: &Vec3, s: f64| (0..3).map(|k| (p[k] / (s * ext[k])).powi(2)).sum::<f64>();
    let field_ppm: Vec<f32> = (0..nv).into_par_iter().map(|q| {
        let p = vlog(q);
        let hz = 150.0 * (2.0 * std::f64::consts::PI * p[0] / 0.4).sin() * (2.0 * std::f64::consts::PI * p[1] / 0.3).cos() + 90.0 * p[2] / 0.1;
        let r2 = inside(&p, 0.45);
        (hz / (GAMMA_BAR * a.b0) * 1e6 + if (0.8..1.0).contains(&r2) { -3.4 } else { 0.0 }) as f32
    }).collect();
    let ident: Vec<Vec<f64>> = (0..nc).map(|i| (0..nc).map(|j| (i == j) as u8 as f64).collect()).collect();
    let sd = SimData {
        dir: PathBuf::new(),
        meta: serde_json::json!({}),
        acq: None,
        ro: ReadoutInfo { enc: vec![], frame: vec![], card: vec![], t_exc: vec![] },
        sched: None,
        encode_m1: vec![],
        data: vec![],
        nc,
        na: 0,
        dims,
        vox: [res; 3],
        tdims: dims,
        res,
        psi: ident.clone(),
        gnl: if a.gnl { GradientCoils::clinical_synthetic() } else { GradientCoils::ideal() },
        rot: IDENTITY,
        fov_c: a.fov_c,
        b0: a.b0,
        venc: 1.0,
        smaps_true,
        field_ppm,
        truth_vel: vec![],
        truth_frac: vec![],
        n_frames: frames,
        lazy: None,
    };
    let t = Instant::now();
    let mut grid = sd.physics_grid(a.gnl, &ident);
    // body support (maps masked outside, as recon_p2np), fit voxels = support
    let support: Vec<usize> = (0..nv).filter(|&q| inside(&vlog(q), 0.47) <= 1.0).collect();
    let mut sup = vec![false; nv];
    support.iter().for_each(|&q| sup[q] = true);
    for c in 0..nc {
        for q in 0..nv {
            if !sup[q] {
                grid.smaps[c * nv + q] = C32::new(0.0, 0.0);
            }
        }
    }
    let fit = fit_monomials(&sd, &grid, &support);
    println!("  physics grid ({}GNL) + quadratic monomial fit over {} support voxels: {:.1} s", if a.gnl { "" } else { "no " }, support.len(), t.elapsed().as_secs_f64());
    (sd, grid, support, fit)
}

// ───────────────────────────────────────────────────────────── timing helpers

struct Timer {
    ev: (cudarc::driver::CudaEvent, cudarc::driver::CudaEvent),
    reps: usize,
}

impl Timer {
    fn new(dev: &Device, reps: usize) -> Self {
        let f = Some(CUevent_flags::CU_EVENT_DEFAULT);
        Self { ev: (dev.ctx.new_event(f).unwrap(), dev.ctx.new_event(f).unwrap()), reps: reps.max(1) }
    }
    /// median ms over `reps` runs after one warm-up
    fn ms(&self, st: &Arc<CudaStream>, f: &mut dyn FnMut()) -> f64 {
        f();
        let mut t = Vec::new();
        for _ in 0..self.reps {
            self.ev.0.record(st).unwrap();
            f();
            self.ev.1.record(st).unwrap();
            self.ev.1.synchronize().unwrap();
            t.push(self.ev.0.elapsed_ms(&self.ev.1).unwrap() as f64);
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t[t.len() / 2]
    }
}

fn free_mem(dev: &Device, st: &Arc<CudaStream>) -> usize {
    st.synchronize().unwrap();
    dev.ctx.mem_get_info().unwrap().0
}

/// forward + inverse partial-axis FFT time (ms) of `nch` interleaved channels on a fine grid nf (x fastest)
fn fft_ms(dev: &Device, st: &Arc<CudaStream>, tm: &Timer, nf: [usize; 3], nch: usize) -> (f64, f64) {
    let fd = [nf[2], nf[1], nf[0], nch];
    let fwd = FftPlan::<f32>::new_partial(dev, &fd, &[true, true, true, false], Direction::Forward, Norm::None).unwrap();
    let inv = FftPlan::<f32>::new_partial(dev, &fd, &[true, true, true, false], Direction::Inverse, Norm::None).unwrap();
    let mut buf = st.alloc_zeros::<C32>(fd.iter().product()).unwrap();
    let f = tm.ms(st, &mut || fwd.execute_inplace(st, &mut buf).unwrap());
    let i = tm.ms(st, &mut || inv.execute_inplace(st, &mut buf).unwrap());
    (f, i)
}

// ───────────────────────────────────────────────────────────── measurements

#[derive(Clone, Debug, Default)]
struct Meas {
    label: String,
    m: usize,
    fwd: f64,
    adj: f64,
    normal: f64,
    /// SENSE: set_points per call (ms)
    points: f64,
    /// device bytes of the set's sample tables (SegPoints, or coordinates for SENSE)
    pts_bytes: usize,
    /// host time to build the set's tables / tile lists (s)
    setup_s: f64,
    /// breakdown per call (ms): kernels (gridding + physics) / FFT / rest, forward and adjoint
    kern: (f64, f64),
    fft: (f64, f64),
}

#[derive(Clone, Debug, Default)]
struct ModelRes {
    model: String,
    nseg: usize,
    order: u32,
    frames: usize,
    meas: Vec<Meas>,
    /// engine bytes (grids, FFT plans, maps, scratch) and segment tables per encode
    engine_bytes: usize,
    seg_bytes: usize,
    seg_s: f64,
    tune_s: f64,
    coil_batch: usize,
}

/// one model on the measured sets of a scan
#[allow(clippy::too_many_arguments)]
fn run_model(dev: &Device, st: &Arc<CudaStream>, kern: &Arc<Kernels>, a: &Args, scan: &Scan, model: &str, sd: &mut SimData, grid: &PhysicsGrid, support: &[usize], fit: &MonoFit, sets: &[SetDef], tm: &Timer) -> ModelRes {
    let (dims, nc, nv) = (sd.dims, sd.nc, sd.nv());
    let temporal = a.temporal.is_some() && !scan.tsets.is_empty() && model.starts_with("seg");
    let frames = if temporal { scan.frames } else { 1 };
    let (nseg, order, with_b0, with_conc) = match model {
        "sense" => (1, 0, false, false),
        "seg-none" => (1, 0, false, false),
        "seg-b0" => (scan.nseg, 0, true, false),
        "seg-full" => (scan.nseg, a.order, true, true),
        m => panic!("model {m}: sense | seg-none | seg-b0 | seg-full"),
    };
    let mut res = ModelRes { model: model.into(), nseg, order, frames, ..Default::default() };
    let host_x: Vec<C32> = (0..frames * nv).map(|i| C32::new(1.0 + 0.1 * (i % 7) as f32, 0.05 * (i % 5) as f32)).collect();
    let x = st.clone_htod(&host_x).unwrap();
    drop(host_x);
    let mut xo = st.alloc_zeros::<C32>(frames * nv).unwrap();
    if model == "sense" {
        let w = EsKernel::width_for(1e-4, 2.0);
        let nf = dims.map(|n| next_smooth_even(((2.0 * n as f64).ceil() as usize).max(2 * w)));
        let (ff, fi) = fft_ms(dev, st, tm, nf, 1);
        let f0 = free_mem(dev, st);
        let (s0, _) = build_set(scan.sch.as_ref(), &sets[0], sd.res, a.adc_step, a.rr);
        let mut op = SenseOp::new(dev, st, kern.clone(), dims, [&s0.coords[0], &s0.coords[1], &s0.coords[2]], &grid.smaps, nc, vec![], 1e-4).unwrap();
        drop(s0);
        res.engine_bytes = f0.saturating_sub(free_mem(dev, st));
        res.coil_batch = nc;
        for set in sets {
            let t = Instant::now();
            let (ss, _) = build_set(scan.sch.as_ref(), set, sd.res, a.adc_step, a.rr);
            let m = ss.tau.len();
            let co = [0, 1, 2].map(|k| st.clone_htod(&ss.coords[k]).unwrap());
            let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
            op.set_points(st, [&co[0], &co[1], &co[2]], m).unwrap();
            let setup_s = t.elapsed().as_secs_f64();
            // SenseSets keeps every set's coordinates on the device (3 × f32); the sorted points are per call
            let pts_bytes = 12 * m;
            let points = tm.ms(st, &mut || op.set_points(st, [&co[0], &co[1], &co[2]], m).unwrap());
            let fwd = tm.ms(st, &mut || { op.set_points(st, [&co[0], &co[1], &co[2]], m).unwrap(); op.forward(st, &x, &mut y).unwrap() });
            let adj = tm.ms(st, &mut || { op.set_points(st, [&co[0], &co[1], &co[2]], m).unwrap(); op.adjoint(st, &y, &mut xo).unwrap() });
            let normal = tm.ms(st, &mut || {
                op.set_points(st, [&co[0], &co[1], &co[2]], m).unwrap();
                op.forward(st, &x, &mut y).unwrap();
                op.set_points(st, [&co[0], &co[1], &co[2]], m).unwrap();
                op.adjoint(st, &y, &mut xo).unwrap();
            });
            let fft = (ff * nc as f64, fi * nc as f64);
            println!("    {:<28} M {:>9}: forward {fwd:>8.1} ms, adjoint {adj:>8.1} ms, normal {normal:>8.1} ms | set_points {points:.1} ms, FFT {:.1} / {:.1} ms, rest {:.1} / {:.1} ms", set.label, m, fft.0, fft.1, fwd - points - fft.0, adj - points - fft.1);
            res.meas.push(Meas { label: set.label.clone(), m, fwd, adj, normal, points, pts_bytes, setup_s, kern: (f64::NAN, f64::NAN), fft });
        }
        return res;
    }
    // segmented physics NUFFT: one engine, per-set points, segments per encode
    let f0 = free_mem(dev, st);
    let mut eng = SegNufft::new(dev, dims, sd.vox, nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames, bspline: a.temporal.unwrap_or(1), max_grid_bytes: a.max_grid }).unwrap();
    eng.set_maps(st, &grid.smaps).unwrap();
    res.engine_bytes = f0.saturating_sub(free_mem(dev, st));
    res.coil_batch = eng.cb;
    let nf = eng.plan.nf;
    let (ff, fi) = fft_ms(dev, st, tm, nf, nseg);
    let seg_fit: Vec<usize> = support.iter().copied().step_by((support.len() / 20000).max(1)).collect();
    let mut segs_by_enc: Vec<(usize, hasty_nufft::SegSegments)> = Vec::new();
    for (si, set) in sets.iter().enumerate() {
        let t = Instant::now();
        let (ss, card) = build_set(scan.sch.as_ref(), set, sd.res, a.adc_step, a.rr);
        let m = ss.tau.len();
        let na = set.na(a.adc_step);
        sd.na = na;
        sd.ro = ReadoutInfo { enc: vec![set.enc; card.len()], frame: vec![0; card.len()], t_exc: vec![0.0; card.len()], card };
        let wbar = if with_conc { mean_coefficients(&[&ss], na) } else { vec![[0.0; 4]; na] };
        let tab = seg_tables(sd, &ss, fit, with_conc, Some(&wbar));
        drop(ss);
        // segments for this set's encode (built once per encode, like recon_p2np)
        let mut seg_here = 0.0;
        if !segs_by_enc.iter().any(|(e, _)| *e == set.enc) {
            let ts = Instant::now();
            let psi = segment_phase(grid, &wbar, &tab.tau_adc, with_b0);
            let fb = free_mem(dev, st);
            let (sp, at, t0, dt) = build_segments(nv, nseg, tab.tau_adc[0], tab.tau_adc[na - 1], a.ntab, &seg_fit, &psi);
            let seg = eng.segments(st, &sp, &at, t0, dt).unwrap();
            res.seg_bytes = res.seg_bytes.max(fb.saturating_sub(free_mem(dev, st)));
            res.seg_s = ts.elapsed().as_secs_f64();
            seg_here = res.seg_s;
            segs_by_enc.push((set.enc, seg));
        }
        let seg = &segs_by_enc.iter().find(|(e, _)| *e == set.enc).unwrap().1;
        let fb = free_mem(dev, st);
        let pts = eng.points(st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, temporal.then_some(&tab.t[..])).unwrap();
        let pts_bytes = fb.saturating_sub(free_mem(dev, st));
        let setup_s = t.elapsed().as_secs_f64() - seg_here;
        drop(tab);
        let mut kern_t = (f64::NAN, f64::NAN);
        if si == 0 && a.tune {
            let tt = Instant::now();
            let (kf, ka) = eng.autotune(st, seg, &pts).unwrap();
            res.tune_s = tt.elapsed().as_secs_f64();
            println!("    autotune {:.1} s: fwd {:?}, adj {:?}", res.tune_s, eng.tiles_fwd, eng.tiles_adj);
            // kernels of one coil batch (static sets: the whole set) → all batches
            let nb = nc.div_ceil(eng.cb) as f64;
            if !temporal {
                kern_t = (kf as f64 * nb, ka as f64 * nb);
            }
        }
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let fwd = tm.ms(st, &mut || eng.forward(st, seg, &pts, &x, &mut y).unwrap());
        let adj = tm.ms(st, &mut || eng.adjoint(st, seg, &pts, &y, &mut xo).unwrap());
        let normal = tm.ms(st, &mut || { eng.forward(st, seg, &pts, &x, &mut y).unwrap(); eng.adjoint(st, seg, &pts, &y, &mut xo).unwrap() });
        // FFT per call: every frame once per coil batch (temporal: + the wrap-around slots)
        let loads = if temporal { (frames + eng.nf - 1) as f64 } else { 1.0 };
        let fft = (ff * nc as f64 * loads, fi * nc as f64 * loads);
        let rest = (fwd - fft.0 - kern_t.0, adj - fft.1 - kern_t.1);
        println!(
            "    {:<28} M {:>9}: forward {fwd:>8.1} ms, adjoint {adj:>8.1} ms, normal {normal:>8.1} ms | kernels {} / {} ms, FFT {:.1} / {:.1} ms, pad/crop/rest {} / {} ms | tables {:.1} s, {:.0} B/sample on the GPU",
            set.label, m, fmt(kern_t.0), fmt(kern_t.1), fft.0, fft.1, fmt(rest.0), fmt(rest.1), setup_s, pts_bytes as f64 / m as f64
        );
        res.meas.push(Meas { label: set.label.clone(), m, fwd, adj, normal, points: 0.0, pts_bytes, setup_s, kern: kern_t, fft });
        // propagate the tuned kernel times to the other sets (scaled by samples: the kernels are ∝ samples)
        if si > 0 && res.meas[0].kern.0.is_finite() {
            let r = m as f64 / res.meas[0].m as f64;
            let k0 = res.meas[0].kern;
            res.meas[si].kern = (k0.0 * r, k0.1 * r);
        }
        drop(pts);
    }
    res
}

fn fmt(v: f64) -> String {
    if v.is_finite() { format!("{v:.1}") } else { "–".into() }
}

// ───────────────────────────────────────────────────────────── projection

/// least-squares t(M) = a + b·M over the measured sets (one set: proportional with no fixed part beyond that set's)
fn affine(ms: &[(f64, f64)]) -> (f64, f64) {
    if ms.len() == 1 || ms.iter().all(|p| (p.0 - ms[0].0).abs() < 1e-9 * ms[0].0) {
        let (m, t) = ms[0];
        return (0.0, t / m);
    }
    let n = ms.len() as f64;
    let (sx, sy) = (ms.iter().map(|p| p.0).sum::<f64>(), ms.iter().map(|p| p.1).sum::<f64>());
    let sxx = ms.iter().map(|p| p.0 * p.0).sum::<f64>();
    let sxy = ms.iter().map(|p| p.0 * p.1).sum::<f64>();
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let a = (sy - b * sx) / n;
    if a < 0.0 {
        return (0.0, sy / sx);
    }
    (a, b)
}

#[derive(Debug, Default)]
struct Projection {
    n_sets: usize,
    samples: usize,
    op_s: f64,
    normal_s: f64,
    stream_s: f64,
    roundtrip_s: f64,
    vec_s: f64,
    iter_s: f64,
    setup_s: f64,
    total_s: f64,
    resident_gb: f64,
    device_gb: f64,
    fits: bool,
    kern_frac: f64,
    fft_frac: f64,
    /// SENSE: per-call set_points (bin sort) share
    pts_frac: f64,
    /// the memory planner's choice (`hasty_recon::plan::choose_tables`, as recon_p2np): description, sample tables
    /// streamed, data streamed; None if nothing fits the card
    plan: Option<String>,
    tables_host: bool,
    data_host: bool,
}

struct Machine {
    total: usize,
    h2d: f64,
    d2h: f64,
    dev_bw: f64,
}

fn project(a: &Args, scan: &Scan, r: &ModelRes, sets: &[SetDef], nv: usize, nc: usize, mach: &Machine) -> Projection {
    let mt: Vec<f64> = sets.iter().map(|s| s.m(a.adc_step) as f64).collect();
    let fit = |sel: &dyn Fn(&Meas) -> f64| -> f64 {
        let (ca, cb) = affine(&r.meas.iter().map(|m| (m.m as f64, sel(m))).collect::<Vec<_>>());
        mt.iter().map(|&m| ca + cb * m).sum::<f64>() / 1e3
    };
    let op_s = fit(&|m| m.fwd) + fit(&|m| m.adj);
    let normal_s = fit(&|m| m.normal);
    let kern = fit(&|m| if m.kern.0.is_finite() { m.kern.0 + m.kern.1 } else { 0.0 });
    let fft = fit(&|m| m.fft.0 + m.fft.1);
    let pts = fit(&|m| 2.0 * m.points);
    let samples: usize = sets.iter().map(|s| s.m(a.adc_step)).sum();
    // memory if everything is resident: operator, segment tables (one per encode), sample tables, data + residual,
    // P²nP stacks (x, ∇f, step on the device; ZMSHR1 secant pair + w placeable)
    let per_sample = r.meas.iter().map(|m| m.pts_bytes as f64 / m.m as f64).sum::<f64>() / r.meas.len() as f64;
    let len_x = nv * r.frames;
    let stack = (sets.len() * len_x * 8) as f64;
    let data = (samples * nc * 8) as f64;
    let n_seg_tabs = if r.model == "sense" { 0 } else { scan.n_enc };
    let fixed = r.engine_bytes as f64 + r.seg_bytes as f64 * n_seg_tabs as f64;
    let resident = fixed + per_sample * samples as f64 + 2.0 * data + 6.0 * stack;
    let mut fits = resident < 0.95 * mach.total as f64;
    // not resident: data (and, beyond what fits, the sample tables) streamed from pinned host memory once per gradient
    let mut device_only = fixed + 3.0 * stack + per_sample * samples as f64;
    let mut stream_s = if fits { 0.0 } else { data / mach.h2d + if device_only > 0.95 * mach.total as f64 { per_sample * samples as f64 / mach.h2d } else { 0.0 } };
    // the planner's choice (recon_p2np's memory plan): data, ZMSHR1 state and the sample tables placed by the budget,
    // transfers = bytes / measured pinned bandwidth (the residual is formed per set, as the planned objective does)
    let m_max = sets.iter().map(|s| s.m(a.adc_step)).max().unwrap_or(0);
    let need = hasty_recon::plan::Need { fixed: fixed as usize, stack: stack as usize, chunk_max: len_x * 8, data: data as usize, set_max: m_max * nc * 8, quasi: true, gradient_s: op_s };
    let tab = hasty_recon::plan::TableNeed { bytes: (per_sample * samples as f64) as usize, set_max: (per_sample * m_max as f64) as usize, device_now: 0 };
    let probe = hasty_recon::plan::Probe { free: mach.total, total: mach.total, h2d: mach.h2d, d2h: mach.d2h };
    let (mut tables_host, mut data_host) = (false, false);
    let plan_s = match hasty_recon::plan::choose_tables(&need, &tab, &probe, (0.95 * mach.total as f64) as usize, None, None) {
        Ok(p) => {
            stream_s = p.transfer_s;
            device_only = p.device_bytes as f64;
            fits = !p.data_on_host && !p.tables_on_host && !p.prev_on_host;
            tables_host = p.tables_on_host;
            data_host = p.data_on_host;
            Some(p.describe())
        }
        Err(e) => Some(format!("{e} (projection: everything streamed)\n")),
    };
    let roundtrip_s = stack / mach.d2h + stack / mach.h2d;
    let vec_s = 10.0 * stack / mach.dev_bw;
    let iter_s = op_s + stream_s + roundtrip_s + vec_s;
    // setup: sample tables / tile lists per set, segments per encode, autotune, step size (15 power iterations on 2 sets)
    let setup_tab = r.meas.iter().map(|m| m.setup_s / m.m as f64).sum::<f64>() / r.meas.len() as f64 * samples as f64;
    let setup_s = setup_tab + r.seg_s * n_seg_tabs as f64 + r.tune_s + 2.0 * 15.0 * normal_s / sets.len() as f64;
    Projection {
        n_sets: sets.len(),
        samples,
        op_s,
        normal_s,
        stream_s,
        roundtrip_s,
        vec_s,
        iter_s,
        setup_s,
        total_s: setup_s + a.iters as f64 * iter_s,
        resident_gb: resident / GB,
        device_gb: device_only / GB,
        fits,
        kern_frac: if kern > 0.0 { kern / op_s } else { f64::NAN },
        fft_frac: fft / op_s,
        pts_frac: pts / op_s,
        plan: plan_s,
        tables_host,
        data_host,
    }
}

// ───────────────────────────────────────────────────────────── main

/// `--plan`: the scans' set statistics, the median set's host-side tables (timed) and formula memory estimates of the
/// resident problem — CPU only, no CUDA call
fn plan(a: &Args) {
    let dims = a.fov.map(|f| ((f / a.res).ceil() as usize..).find(|&n| smooth7(n)).unwrap());
    let nv = dims.iter().product::<usize>();
    let ge: usize = dims.map(|n| next_smooth_even(((1.25 * n as f64).ceil() as usize).max(20))).iter().product();
    println!("plan (CPU only): grid {dims:?} ({nv} voxels), SegNufft fine grid {ge} points ({:.2} GiB per coil per segment)", (ge * 8) as f64 / GB);
    for seq in &a.seqs {
        let t = Instant::now();
        let scan = if seq == "flow" { flow_scan(a) } else { b0_scan(a) };
        let sets: Vec<&SetDef> = scan.sets.iter().filter(|s| !s.ros.is_empty()).collect();
        let mut sizes: Vec<usize> = sets.iter().map(|s| s.m(a.adc_step)).collect();
        sizes.sort();
        let samples: usize = sizes.iter().sum();
        println!("\n== {} ({:.0} s): {}\n   {} readouts × {} ADC, {} samples, {} sets ({} / {} / {}), schedule {:.1} s", scan.name, scan.scan_s, scan.describe, scan.sch.n_readouts(), scan.sch.n_adc(), samples, sets.len(), sizes[0], sizes[sizes.len() / 2], sizes[sizes.len() - 1], t.elapsed().as_secs_f64());
        let med = pick(&scan.sets, 1)[0];
        let t = Instant::now();
        let (ss, _) = build_set(scan.sch.as_ref(), &scan.sets[med], a.res, a.adc_step, a.rr);
        let tb = t.elapsed().as_secs_f64();
        let kmax = (0..ss.tau.len()).map(|j| (0..3).map(|k| ss.coords[k][j].abs()).fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
        let wmax = ss.wq.iter().map(|w| w.iter().map(|v| v.abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
        println!("   median set {}: {} samples built in {:.2} s; τ {:.3} … {:.3} ms, max |k·Δ| {:.2} rad, max |w| {:.1} rad/m²", scan.sets[med].label, ss.tau.len(), tb, ss.tau.iter().cloned().fold(f64::MAX, f64::min) * 1e3, ss.tau.iter().cloned().fold(f64::MIN, f64::max) * 1e3, kmax, wmax);
        let data = (samples * a.nvc * 8) as f64 / GB;
        let stack = |frames: usize| (sets.len() * nv * frames * 8) as f64 / GB;
        println!("   resident estimate: data {:.1} GiB (+ residual {:.1}), SENSE coordinates {:.1} GiB, SegPoints ≈ {:.1} GiB (≈ 185 B/sample), segment tables {:.2} GiB per encode × {}, image stack {:.2} GiB (× 6 for P²nP-ZMSHR1)",
            data, data, (samples * 12) as f64 / GB, (samples * 185) as f64 / GB, (scan.nseg * nv * 8) as f64 / GB, scan.n_enc, stack(1));
    }
}

fn main() {
    let a = parse();
    if a.plan {
        return plan(&a);
    }
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let tm = Timer::new(&dev, a.reps);
    let pr = hasty_recon::plan::probe(&dev.ctx, &st).unwrap();
    // device bandwidth (axpby on 512 MiB: 3 × 8 bytes per element)
    let dev_bw = {
        let n = 64usize << 20;
        let xa = st.alloc_zeros::<C32>(n).unwrap();
        let mut ya = st.alloc_zeros::<C32>(n).unwrap();
        let t = tm.ms(&st, &mut || kern.axpby(&st, Complex64::new(1.0, 0.0), &xa, Complex64::new(0.5, 0.0), &mut ya).unwrap());
        (3 * n * 8) as f64 / (t * 1e-3)
    };
    let mach = Machine { total: pr.total, h2d: pr.h2d, d2h: pr.d2h, dev_bw };
    let dims = a.fov.map(|f| ((f / a.res).ceil() as usize..).find(|&n| smooth7(n)).unwrap());
    let nv = dims.iter().product::<usize>();
    println!(
        "{} ({:.1} GiB, {:.1} GiB free), PCIe pinned {:.1} / {:.1} GB/s (h2d / d2h), device {:.0} GB/s",
        dev.ctx.name().unwrap_or_default(),
        pr.total as f64 / GB,
        pr.free as f64 / GB,
        pr.h2d / 1e9,
        pr.d2h / 1e9,
        dev_bw / 1e9
    );
    println!(
        "grid {dims:?} @ {:.1} mm ({:.0} × {:.0} × {:.0} mm, {nv} voxels), {} virtual coils, GNL {}, FOV centre {:?} mm, {} iterations",
        a.res * 1e3,
        dims[0] as f64 * a.res * 1e3,
        dims[1] as f64 * a.res * 1e3,
        dims[2] as f64 * a.res * 1e3,
        a.nvc,
        if a.gnl { "on" } else { "off" },
        a.fov_c.map(|v| v * 1e3),
        a.iters
    );
    let mut out = serde_json::Map::new();
    for seq in &a.seqs {
        let t = Instant::now();
        let scan = match seq.as_str() {
            "flow" => flow_scan(&a),
            "b0" => b0_scan(&a),
            s => panic!("--seq {s}: flow | b0"),
        };
        let nonempty: Vec<&SetDef> = scan.sets.iter().filter(|s| !s.ros.is_empty()).collect();
        let sizes: Vec<usize> = { let mut v: Vec<usize> = nonempty.iter().map(|s| s.m(a.adc_step)).collect(); v.sort(); v };
        let total: usize = sizes.iter().sum();
        println!(
            "\n== {} ({:.0} s): {}\n   {} readouts × {} ADC samples (step {}), {} samples; {} static sets (samples per set {} / {} / {} min / median / max){}; schedule {:.1} s",
            scan.name,
            scan.scan_s,
            scan.describe,
            scan.sch.n_readouts(),
            scan.sch.n_adc(),
            a.adc_step,
            total,
            nonempty.len(),
            sizes[0],
            sizes[sizes.len() / 2],
            sizes[sizes.len() - 1],
            if scan.tsets.is_empty() { String::new() } else { format!(", {} temporal sets (one per encode)", scan.tsets.len()) },
            t.elapsed().as_secs_f64()
        );
        let (mut sd, grid, support, fit) = physics(&a, dims, a.nvc, scan.frames);
        let mut scan_json = serde_json::Map::new();
        let mut summary = Vec::new();
        for model in &a.models {
            let temporal = a.temporal.is_some() && !scan.tsets.is_empty() && model.starts_with("seg");
            let all: Vec<SetDef> = if temporal { scan.tsets.clone() } else { nonempty.iter().map(|s| (*s).clone()).collect() };
            let sel: Vec<SetDef> = pick(&all, a.nsets).into_iter().map(|i| all[i].clone()).collect();
            println!("  model {model}{}:", if temporal { format!(" (temporal B-spline {})", a.temporal.unwrap()) } else { String::new() });
            let r = run_model(&dev, &st, &kern, &a, &scan, model, &mut sd, &grid, &support, &fit, &sel, &tm);
            let p = project(&a, &scan, &r, &all, nv, a.nvc, &mach);
            println!(
                "    engine {:.2} GiB (coil batch {}), segment tables {:.2} GiB per encode ({:.1} s), resident problem {:.1} GiB ({}), device part without data {:.1} GiB",
                r.engine_bytes as f64 / GB,
                r.coil_batch,
                r.seg_bytes as f64 / GB,
                r.seg_s,
                p.resident_gb,
                if p.fits { "fits" } else { "does not fit: data streamed from host" },
                p.device_gb
            );
            println!(
                "    per iteration: operator {:.2} s (forward + adjoint over {} sets, {:.3e} samples{}) + data streaming {:.2} s + prior round trip {:.2} s + ZMSHR1 vectors {:.2} s = {:.2} s; setup {:.0} s; {} iterations ≈ {:.1} min",
                p.op_s,
                p.n_sets,
                p.samples as f64,
                if p.kern_frac.is_finite() {
                    format!("; tile kernels (gridding + physics) {:.0} %, FFT {:.0} %, pad/crop/rest {:.0} %", 100.0 * p.kern_frac, 100.0 * p.fft_frac, 100.0 * (1.0 - p.kern_frac - p.fft_frac))
                } else if r.model == "sense" {
                    format!("; set_points {:.0} %, FFT {:.0} %, spread/interp/deconv/coil {:.0} %", 100.0 * p.pts_frac, 100.0 * p.fft_frac, 100.0 * (1.0 - p.pts_frac - p.fft_frac))
                } else {
                    format!("; FFT {:.0} %", 100.0 * p.fft_frac)
                },
                p.stream_s,
                p.roundtrip_s,
                p.vec_s,
                p.iter_s,
                p.setup_s,
                a.iters,
                p.total_s / 60.0
            );
            if let Some(pl) = &p.plan {
                // the planner's placement (sample tables, data, ZMSHR1 state), as recon_p2np would choose on this card
                for line in pl.lines().take(1) {
                    println!("    {line}");
                }
                if p.tables_host {
                    println!("    sample tables streamed from pinned host memory into two device slots per set use ({:.2} s of uploads per gradient at {:.1} GB/s)", p.samples as f64 * (r.meas.iter().map(|m| m.pts_bytes as f64 / m.m as f64).sum::<f64>() / r.meas.len() as f64) / pr.h2d, pr.h2d / 1e9);
                }
                if p.data_host {
                    println!("    k-space data streamed per set");
                }
            }
            summary.push((model.clone(), r.nseg, r.order, p.iter_s, p.total_s, p.resident_gb, p.fits));
            scan_json.insert(model.clone(), serde_json::json!({
                "nseg": r.nseg, "order": r.order, "frames_per_set": r.frames, "coil_batch": r.coil_batch,
                "engine_gib": r.engine_bytes as f64 / GB, "segment_tables_gib": r.seg_bytes as f64 / GB, "segments_s": r.seg_s, "autotune_s": r.tune_s,
                "measured": r.meas.iter().map(|m| serde_json::json!({"set": m.label, "samples": m.m, "forward_ms": m.fwd, "adjoint_ms": m.adj, "normal_ms": m.normal,
                    "set_points_ms": m.points, "kernels_ms": [m.kern.0, m.kern.1], "fft_ms": [m.fft.0, m.fft.1], "tables_s": m.setup_s, "sample_table_bytes": m.pts_bytes})).collect::<Vec<_>>(),
                "projection": {"sets": p.n_sets, "samples": p.samples, "operator_s": p.op_s, "normal_s": p.normal_s, "stream_s": p.stream_s, "prior_roundtrip_s": p.roundtrip_s,
                    "vector_s": p.vec_s, "iteration_s": p.iter_s, "setup_s": p.setup_s, "iterations": a.iters, "total_s": p.total_s, "resident_gib": p.resident_gb,
                    "fits": p.fits, "kernel_fraction": p.kern_frac, "fft_fraction": p.fft_frac,
                    "plan": p.plan, "tables_streamed": p.tables_host, "data_streamed": p.data_host, "device_gib": p.device_gb},
            }));
        }
        println!("\n  summary {} ({} iterations):", scan.name, a.iters);
        println!("    model      L  o | s / iteration | total min | resident GiB");
        for (m, l, o, it, tot, gb, fits) in &summary {
            println!("    {m:<9} {l:>2}  {o} | {it:>13.2} | {:>9.1} | {gb:>8.1}{}", tot / 60.0, if *fits { "" } else { " (streamed)" });
        }
        out.insert(scan.name.clone(), serde_json::Value::Object(scan_json));
        drop(sd);
    }
    if let Some(path) = &a.json {
        let doc = serde_json::json!({
            "device": dev.ctx.name().unwrap_or_default(), "total_gib": pr.total as f64 / GB, "h2d_gbs": pr.h2d / 1e9, "d2h_gbs": pr.d2h / 1e9, "device_gbs": dev_bw / 1e9,
            "dims": dims, "res": a.res, "vcoils": a.nvc, "gnl": a.gnl, "fov_centre": a.fov_c, "iters": a.iters, "adc_step": a.adc_step,
            "cmdline": std::env::args().collect::<Vec<_>>().join(" "), "scans": out,
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        println!("wrote {}", path.display());
    }
}
