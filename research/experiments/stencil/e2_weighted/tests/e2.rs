//! The reconstruction's physics operator (`recon_p2np --model seg`) against the simulator's physics, f64 direct sums.
//!
//! Sample parameters come from the real 4D-flow sequence (`data/hasty/seq/flow_thorax`, `FlowSchedule` with the ideal
//! GIRF, exactly what `simphantom --seq flow` simulates; the spiral `Acquisition` if that directory is missing), the
//! operator is built by the reconstruction's own code path — `SimData::sample_set`, `physics_grid` (GNL inverse warp,
//! B0, coil maps and concomitant monomials at the true positions), `fit_monomials`, `mean_coefficients`,
//! `seg_tables`, `segment_phase`, `build_segments`, `SegNufft` — and compared with
//!
//!   y_c(j) = Σ_f b_f(t_j) Σ_q S_c(x_q) x_f(q) exp(−i[k_j·u_q + w_j·m(x_q) + ω(x_q) τ_j])
//!
//! (the simulator's phase model, `hasty_mrsim::direct`: u = apparent position, x = true position from an independent
//! Newton inverse of the GNL field, m = [z², x²+y², xz, yz] at the true scanner position, w = (γ/2B0)·∫GGᵀ from the
//! actual waveform since excitation, ω from the field map at the true position). The adjoint is compared with the
//! exact transpose of the same sum.
//!
//! Test grid: 24 × 24 × 16 voxels of 12 mm (288 × 288 × 192 mm, body-sized, so the quadratic concomitant phase over the
//! FOV is realistic), object and coil maps inside an ellipsoid (maps masked to the dilated support as in recon_p2np),
//! 4 coils, 2 encodes × 6 readouts, every 6th ADC sample (≈ 4000 samples per set over the whole 8 ms readout). k
//! beyond this grid's band wraps; both the NUFFT and the direct sum are 2π-periodic in k·Δ for sources on the grid,
//! so every part of the readout is exercised.
//!
//! Every GPU case is also emulated on the CPU (f64): the same diagonals P_l, table interpolators a_l(τ), per-sample
//! φ0 / k-shift and ΔC Taylor factor, so the printout splits the error into "model" (the approximation itself) and
//! "impl" (GPU vs emulation: NUFFT interpolation, derivative kernels, f32).
//! Expected run time on an RTX 4090: ~1 min first run (kernel compiles), ~20–40 s afterwards (CPU direct sums).

use std::path::PathBuf;

use hasty_bench::seqsched::FlowSchedule;
use hasty_cuda::{Device, C32};
use hasty_mrsim::direct::{direct_sum_cpu, monomials, Source};
use hasty_mrsim::gnl::GradientCoils;
use hasty_mrsim::physio::{CardiacTime, Ecg};
use hasty_mrsim::sequence::{Acquisition, FlowSpiralParams, Schedule, SAMPLE_FLOATS};
use hasty_mrsim::{mat_vec, rotation, transpose, Mat3, Vec3, GAMMA_BAR, IDENTITY};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::data::{inv3, PhysicsGrid, ReadoutInfo, SampleSet, SchedSamples, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_recon::segphys::{fit_monomials, fit_monomials_hi, hi_table, mean_coefficients, seg_tables, seg_tables_hi, segment_phase, MonoFitHi, SegTables};
use hasty_testkit::physops::{bspline_weight, frame_indices, interp_table, rel_on, taylor_expm, DirectSum};
use hasty_testkit::{Rng, C64};

const DIMS: [usize; 3] = [24, 24, 16];
const VOX: f64 = 0.012;
const NC: usize = 4;
const FAT_PPM: f64 = -3.4;
const PI: f64 = std::f64::consts::PI;

// ───────────────────────────────────────────────────────────── problem

#[derive(Clone)]
struct Spec {
    b0: f64,
    fov_c: Vec3,
    rot: Mat3,
    gnl: bool,
    /// off-resonance map (smooth ±150 Hz + a −3.4 ppm shell) on / off
    field: bool,
    conc: bool,
    frames: usize,
    n_ro: usize,
    adc_step: usize,
}

impl Spec {
    fn conc_only(b0: f64, fov_c: Vec3, rot: Mat3) -> Self {
        Self { b0, fov_c, rot, gnl: false, field: false, conc: true, frames: 1, n_ro: 6, adc_step: 6 }
    }
    fn full(frames: usize) -> Self {
        let rot = rotation([1.0, 0.6, 0.2], 0.45);
        Self { b0: 2.89, fov_c: [0.03, -0.04, 0.10], rot, gnl: true, field: true, conc: true, frames, n_ro: if frames > 1 { 10 } else { 6 }, adc_step: if frames > 1 { 10 } else { 6 } }
    }
    /// the full geometry with one physics part switched off (decomposition of the full model's error)
    fn without(mut self, part: &str) -> Self {
        match part {
            "conc" => self.conc = false,
            "gnl" => self.gnl = false,
            "field" => self.field = false,
            p => panic!("{p}"),
        }
        self
    }
    fn label(&self) -> String {
        format!("B0 {:.2} T, FOV centre ({:.0}, {:.0}, {:.0}) mm{}{}{}{}{}", self.b0, self.fov_c[0] * 1e3, self.fov_c[1] * 1e3, self.fov_c[2] * 1e3, if self.rot == IDENTITY { "" } else { ", oblique" }, if self.conc { ", concomitant" } else { "" }, if self.gnl { ", GNL" } else { "" }, if self.field { ", B0 map + fat" } else { "" }, if self.frames > 1 { format!(", {} frames", self.frames) } else { String::new() })
    }
}

struct Prob {
    spec: Spec,
    seq: String,
    sd: SimData,
    grid: PhysicsGrid,
    sets: Vec<SampleSet>,
    /// raw f32 sample records per set (physical k, moments, w, τ): the simulator's inputs
    recs: Vec<Vec<f32>>,
    support: Vec<usize>,
    /// dilated support: coil maps nonzero, quadratic-fit voxels
    fitv: Vec<usize>,
    /// reference physics per voxel (independent of physics_grid): apparent physical position − FOV centre, true
    /// position (scanner), monomials, ω, maps (coil-major, masked)
    u_phys: Vec<Vec3>,
    x_true: Vec<Vec3>,
    mono: Vec<[f64; 4]>,
    omega: Vec<f64>,
    smaps: Vec<C64>,
    /// operator maps: physics_grid's, masked identically
    smaps_op: Vec<C32>,
    /// centred index and logical position (m) per voxel
    idx: Vec<Vec3>,
    rho: Vec<Vec3>,
    /// object, frames × nv
    x: Vec<C64>,
}

fn flow_run() -> PathBuf {
    PathBuf::from("/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/seq/flow_thorax")
}

/// the scan's sequence at field strength b0: the flow_thorax design if present, else the spiral 4D-flow sequence
fn schedule(b0: f64) -> (Box<dyn Schedule>, String) {
    let run = flow_run();
    if run.join("summary.json").exists() && run.join("spoke_grad.npy").exists() {
        let fs = FlowSchedule::load(&run, 1.5, 2e-6, b0, Ecg::new(4.0, 1.0, 0.0, 1), 20, hasty_seq::girf::Girf::ideal(), "ideal");
        let d = fs.describe();
        (Box::new(fs), d)
    } else {
        let mut p = FlowSpiralParams::aorta_3t_default();
        p.b0 = b0;
        p.scan_time = 1.0;
        let a = Acquisition::new(p);
        let d = Schedule::describe(&a);
        (Box::new(a), format!("{d} (flow_thorax not found)"))
    }
}

fn invert_gnl(g: &GradientCoils, target: &Vec3) -> Vec3 {
    let mut x = *target;
    for _ in 0..60 {
        let f = g.field(&x);
        let r = [f[0] - target[0], f[1] - target[1], f[2] - target[2]];
        if r.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-15 {
            break;
        }
        let dx = mat_vec(&inv3(&g.jacobian(&x)), &r);
        for a in 0..3 {
            x[a] -= dx[a];
        }
    }
    let f = g.field(&x);
    let res = (0..3).map(|a| (f[a] - target[a]).abs()).fold(0.0, f64::max);
    assert!(res < 1e-12, "GNL inverse residual {res}");
    x
}

fn problem(spec: &Spec) -> Prob {
    let (sch, seq) = schedule(spec.b0);
    let na_full = sch.n_adc();
    let encs: Vec<usize> = (0..sch.n_readouts().min(64)).map(|i| sch.readout_row(i)[1] as usize).collect();
    let n_enc = encs.iter().max().unwrap() + 1;
    let set_enc = [0usize, n_enc - 2];
    let adc: Vec<usize> = (0..na_full).step_by(spec.adc_step).collect();
    let na = adc.len();
    // local readouts: set s = readouts of encode set_enc[s] (the first n_ro of the scan)
    let mut ros: Vec<(usize, usize)> = Vec::new(); // (scan readout, set)
    for (s, &e) in set_enc.iter().enumerate() {
        let mut found = 0;
        for i in 0..sch.n_readouts() {
            if sch.readout_row(i)[1] as usize == e {
                ros.push((i, s));
                found += 1;
                if found == spec.n_ro {
                    break;
                }
            }
        }
        assert_eq!(found, spec.n_ro, "not enough readouts of encode {e}");
    }
    let mut rec = Vec::with_capacity(ros.len() * na * SAMPLE_FLOATS);
    for &(i, _) in &ros {
        let full = sch.sample_params(i..i + 1);
        for &n in &adc {
            rec.extend_from_slice(&full[n * SAMPLE_FLOATS..(n + 1) * SAMPLE_FLOATS]);
        }
    }
    if !spec.conc {
        for r in rec.chunks_mut(SAMPLE_FLOATS) {
            r[9..13].fill(0.0);
        }
    }
    let n_lr = ros.len();
    // cardiac phase of each local readout: spread over the cycle per set (temporal test), RR 1 s
    let card: Vec<CardiacTime> = (0..n_lr).map(|r| {
        let k = r % spec.n_ro;
        let phase = ((k as f64 + 0.31) / spec.n_ro as f64 + 0.013 * (k % 3) as f64).rem_euclid(1.0);
        CardiacTime { beat: 0, since_r: phase, rr: 1.0, phase }
    }).collect();
    let nv = DIMS.iter().product::<usize>();
    let ro = ReadoutInfo { enc: ros.iter().map(|&(_, s)| set_enc[s]).collect(), frame: vec![0; n_lr], card, t_exc: ros.iter().map(|&(i, _)| sch.t_exc(i)).collect() };
    // native-grid truth fields: smooth coil maps (Biot–Savart-like falloff + phase) and the off-resonance map (ppm)
    let vlog = |q: usize| -> Vec3 {
        let (i, j, k) = (q % DIMS[0], (q / DIMS[0]) % DIMS[1], q / (DIMS[0] * DIMS[1]));
        [(i as f64 - (DIMS[0] / 2) as f64) * VOX, (j as f64 - (DIMS[1] / 2) as f64) * VOX, (k as f64 - (DIMS[2] / 2) as f64) * VOX]
    };
    let ext = DIMS.map(|d| d as f64 * VOX);
    let mut smaps_true = vec![C32::new(0.0, 0.0); NC * nv];
    for c in 0..NC {
        let ang = 2.0 * PI * c as f64 / NC as f64;
        let ctr = [0.2 * ang.cos(), 0.2 * ang.sin(), 0.03 * c as f64];
        for q in 0..nv {
            let p = vlog(q);
            let d2 = (0..3).map(|a| (p[a] - ctr[a]).powi(2)).sum::<f64>();
            let z = C64::from_polar(0.02 / (d2 + 0.01), 6.0 * (c as f64 + 1.0) * p[0] - 4.0 * p[1] + 3.0 * p[2]);
            smaps_true[c * nv + q] = C32::new(z.re as f32, z.im as f32);
        }
    }
    let shell = |p: &Vec3| -> bool {
        let r2 = (0..3).map(|a| (p[a] / (0.45 * ext[a])).powi(2)).sum::<f64>();
        (0.72..1.0).contains(&r2)
    };
    let field_ppm: Vec<f32> = (0..nv).map(|q| {
        if !spec.field {
            return 0.0;
        }
        let p = vlog(q);
        let hz = 150.0 * (2.0 * PI * p[0] / 0.4).sin() * (2.0 * PI * p[1] / 0.3).cos() + 90.0 * p[2] / 0.1;
        (hz / (GAMMA_BAR * spec.b0) * 1e6 + if shell(&p) { FAT_PPM } else { 0.0 }) as f32
    }).collect();
    let ident: Vec<Vec<f64>> = (0..NC).map(|i| (0..NC).map(|j| (i == j) as u8 as f64).collect()).collect();
    let sd = SimData {
        dir: PathBuf::new(),
        meta: serde_json::json!({}),
        acq: None,
        ro,
        sched: Some(SchedSamples { rec, traj: "nominal".into() }),
        encode_m1: vec![[0.0; 3]; n_enc],
        data: vec![C32::new(0.0, 0.0); n_lr * na * NC],
        nc: NC,
        na,
        dims: DIMS,
        vox: [VOX; 3],
        tdims: DIMS,
        res: VOX,
        psi: ident.clone(),
        gnl: if spec.gnl { GradientCoils::clinical_synthetic() } else { GradientCoils::ideal() },
        rot: spec.rot,
        fov_c: spec.fov_c,
        b0: spec.b0,
        venc: 1.0,
        smaps_true,
        field_ppm,
        truth_vel: vec![],
        truth_frac: vec![],
        n_frames: spec.frames,
        lazy: None,
        na_file: na,
        adc_keep: None,
        model_traj: None,
    };
    let sets: Vec<SampleSet> = (0..set_enc.len()).map(|s| {
        let lr: Vec<usize> = (0..n_lr).filter(|&r| ros[r].1 == s).collect();
        sd.sample_set(&lr, &ident)
    }).collect();
    let recs: Vec<Vec<f32>> = (0..set_enc.len()).map(|s| {
        let rec = &sd.sched.as_ref().unwrap().rec;
        (0..n_lr).filter(|&r| ros[r].1 == s).flat_map(|r| rec[r * na * SAMPLE_FLOATS..(r + 1) * na * SAMPLE_FLOATS].iter().copied()).collect()
    }).collect();
    let mut grid = sd.physics_grid(spec.gnl, &ident);
    // support: ellipsoid (semi-axes 0.42 of the FOV), dilated by one voxel for the maps / fit (as recon_p2np)
    let inside = |q: usize, s: f64| { let p = vlog(q); (0..3).map(|a| (p[a] / (s * ext[a])).powi(2)).sum::<f64>() <= 1.0 };
    let support: Vec<usize> = (0..nv).filter(|&q| inside(q, 0.42)).collect();
    let fitv: Vec<usize> = (0..nv).filter(|&q| inside(q, 0.42 + 1.2 * VOX / ext[2])).collect();
    let infit: Vec<bool> = { let mut b = vec![false; nv]; fitv.iter().for_each(|&q| b[q] = true); b };
    // reference physics at the true positions
    let rt = transpose(&spec.rot);
    let (mut u_phys, mut x_true, mut mono, mut omega) = (Vec::with_capacity(nv), Vec::with_capacity(nv), Vec::with_capacity(nv), Vec::with_capacity(nv));
    let mut smaps = vec![C64::new(0.0, 0.0); NC * nv];
    let (re_t, im_t): (Vec<f32>, Vec<f32>) = sd.smaps_true.iter().map(|c| (c.re, c.im)).unzip();
    let wfac = 2.0 * PI * GAMMA_BAR * spec.b0 * 1e-6;
    for q in 0..nv {
        let u = mat_vec(&spec.rot, &sd.voxel_log(q));
        let x = invert_gnl(&sd.gnl, &[u[0] + spec.fov_c[0], u[1] + spec.fov_c[1], u[2] + spec.fov_c[2]]);
        let xl = mat_vec(&rt, &[x[0] - spec.fov_c[0], x[1] - spec.fov_c[1], x[2] - spec.fov_c[2]]);
        omega.push(wfac * sd.sample_log(&sd.field_ppm, 0, &xl));
        if infit[q] {
            for c in 0..NC {
                smaps[c * nv + q] = C64::new(sd.sample_log(&re_t, c, &xl), sd.sample_log(&im_t, c, &xl));
            }
        }
        mono.push(monomials(&x));
        u_phys.push(u);
        x_true.push(x);
    }
    for q in 0..nv {
        if !infit[q] {
            for c in 0..NC {
                grid.smaps[c * nv + q] = C32::new(0.0, 0.0);
            }
        }
    }
    let smaps_op = grid.smaps.clone();
    let idx: Vec<Vec3> = (0..nv).map(|q| sd.voxel_log(q).map(|v| v / VOX)).collect();
    let rho: Vec<Vec3> = (0..nv).map(|q| sd.voxel_log(q)).collect();
    let mut r = Rng::new(12345);
    let base = r.complex_vec(nv);
    let mut x = vec![C64::new(0.0, 0.0); spec.frames * nv];
    for f in 0..spec.frames {
        let d = r.complex_vec(nv);
        for &q in &support {
            x[f * nv + q] = base[q] + d[q] * 0.5;
        }
    }
    Prob { spec: spec.clone(), seq, sd, grid, sets, recs, support, fitv, u_phys, x_true, mono, omega, smaps, smaps_op, idx, rho, x }
}

// ───────────────────────────────────────────────────────────── tables, references, emulation, GPU

struct Tables {
    tab: SegTables,
    segph: Vec<C32>,
    atab: Vec<C32>,
    t0: f64,
    dt: f64,
    amax: f64,
}

fn tables(p: &Prob, s: usize, nseg: usize, ntab: usize) -> Tables {
    let (sd, set) = (&p.sd, &p.sets[s]);
    let na = sd.na;
    let (with_b0, with_conc) = (p.spec.field, p.spec.conc);
    let fit = fit_monomials(sd, &p.grid, &p.fitv);
    let wbar = if with_conc { mean_coefficients(&[set], na) } else { vec![[0.0; 4]; na] };
    let tab = seg_tables(sd, set, &fit, with_conc, Some(&wbar));
    let tau_adc: Vec<f64> = (0..na).map(|n| set.tau[n]).collect();
    let psi = segment_phase(&p.grid, &wbar, &tau_adc, with_b0);
    let seg_fit: Vec<usize> = p.fitv.iter().copied().step_by(3).collect();
    let ntab = if ntab == 0 { na } else { ntab };
    let (segph, atab, t0, dt) = build_segments(sd.nv(), nseg, tau_adc[0], tau_adc[na - 1], ntab, &seg_fit, &psi);
    let amax = atab.chunks(nseg).map(|r| r.iter().map(|z| (z.re as f64).hypot(z.im as f64)).sum::<f64>()).fold(0.0, f64::max);
    Tables { tab, segph, atab, t0, dt, amax }
}

/// per-sample frame weights from the tables' acquisition times (checked against an independent computation)
fn frame_weights(p: &Prob, s: usize, tab: &SegTables, degree: u32) -> Option<Vec<f64>> {
    let nf = p.spec.frames;
    if nf == 1 {
        return None;
    }
    let set = &p.sets[s];
    let na = p.sd.na;
    let mut bw = Vec::with_capacity(set.tau.len() * nf);
    for j in 0..set.tau.len() {
        let card = p.sd.ro.card[set.readouts[j / na]];
        let t = ((card.phase + set.tau[j] / card.rr) * nf as f64 - 0.5).rem_euclid(nf as f64);
        let d = (t - tab.t[j] as f64).abs();
        assert!(d.min(nf as f64 - d) < 1e-5, "sample time {t} vs table {}", tab.t[j]);
        let tj = tab.t[j] as f64;
        bw.extend((0..nf).map(|f| bspline_weight(nf, degree, f, tj)));
    }
    Some(bw)
}

/// exact physics (the simulator's model), forward and adjoint
fn reference(p: &Prob, s: usize, bw: Option<&[f64]>, y: &[C64]) -> (Vec<C64>, Vec<C64>) {
    let rec = &p.recs[s];
    let m = rec.len() / SAMPLE_FLOATS;
    let smp: Vec<[f64; 8]> = rec.chunks(SAMPLE_FLOATS).map(|r| [r[0], r[1], r[2], r[9], r[10], r[11], r[12], r[13]].map(|v| v as f64)).collect();
    let f = |j: usize, q: usize| {
        let (k, u, mq) = (&smp[j], &p.u_phys[q], &p.mono[q]);
        let ph = k[0] * u[0] + k[1] * u[1] + k[2] * u[2] + k[3] * mq[0] + k[4] * mq[1] + k[5] * mq[2] + k[6] * mq[3] + p.omega[q] * k[7];
        C64::from_polar(1.0, -ph)
    };
    let ds = DirectSum { nv: p.sd.nv(), nc: NC, m, frames: p.spec.frames, smaps: &p.smaps, voxels: &p.fitv, bw };
    ds.apply(&f, &p.x, y)
}

/// CPU emulation of the operator's approximation (segments, table interpolation, φ0, k-shift, ΔC Taylor of `order`)
fn emulate(p: &Prob, t: &Tables, nseg: usize, order: u32, bw: Option<&[f64]>, y: &[C64]) -> (Vec<C64>, Vec<C64>) {
    let nv = p.sd.nv();
    let tab = &t.tab;
    let m = tab.tau.len();
    let p64: Vec<C64> = t.segph.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
    let a64: Vec<C64> = t.atab.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
    let at: Vec<C64> = tab.tau.iter().flat_map(|&tau| interp_table(&a64, nseg, t.t0, t.dt, tau as f64)).collect();
    let smaps64: Vec<C64> = p.smaps_op.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
    let f = |j: usize, q: usize| {
        let (ix, r) = (&p.idx[q], &p.rho[q]);
        let lin = tab.coords[0][j] as f64 * ix[0] + tab.coords[1][j] as f64 * ix[1] + tab.coords[2][j] as f64 * ix[2] + tab.phi0[j] as f64;
        let dc = |i: usize| tab.dc[i * m + j] as f64;
        let psi = if order > 0 { dc(0) * r[0] * r[0] + dc(1) * r[1] * r[1] + dc(2) * r[2] * r[2] + 2.0 * (dc(3) * r[0] * r[1] + dc(4) * r[0] * r[2] + dc(5) * r[1] * r[2]) } else { 0.0 };
        let s: C64 = (0..nseg).map(|l| at[j * nseg + l] * p64[l * nv + q]).sum();
        C64::from_polar(1.0, -lin) * s * taylor_expm(psi, order)
    };
    let ds = DirectSum { nv, nc: NC, m, frames: p.spec.frames, smaps: &smaps64, voxels: &p.fitv, bw };
    ds.apply(&f, &p.x, y)
}

/// the reconstruction's operator on the GPU: (A x, Aᴴ y, ⟨y, A x⟩ vs ⟨Aᴴ y, x⟩ relative mismatch)
fn gpu(dev: &Device, p: &Prob, t: &Tables, nseg: usize, order: u32, bspline: u32, y: &[C64]) -> (Vec<C64>, Vec<C64>, f64) {
    let st = dev.default_stream();
    let nv = p.sd.nv();
    let frames = p.spec.frames;
    let mut eng = SegNufft::new(dev, p.sd.dims, p.sd.vox, NC, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames, bspline, max_grid_bytes: 1 << 30 }).unwrap();
    eng.set_maps(&st, &p.smaps_op).unwrap();
    let seg = eng.segments(&st, &t.segph, &t.atab, t.t0, t.dt).unwrap();
    let tab = &t.tab;
    let pts = eng.points(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, (frames > 1).then_some(&tab.t[..])).unwrap();
    let m = tab.tau.len();
    let dx = st.clone_htod(&c32(&p.x)).unwrap();
    let dy = st.clone_htod(&c32(y)).unwrap();
    let mut ay = st.alloc_zeros::<C32>(NC * m).unwrap();
    let mut ahx = st.alloc_zeros::<C32>(frames * nv).unwrap();
    eng.forward(&st, &seg, &pts, &dx, &mut ay).unwrap();
    eng.adjoint(&st, &seg, &pts, &dy, &mut ahx).unwrap();
    let (ay, ahx) = (c64(&st.clone_dtoh(&ay).unwrap()), c64(&st.clone_dtoh(&ahx).unwrap()));
    let l: C64 = y.iter().zip(&ay).map(|(a, b)| a.conj() * b).sum();
    let r: C64 = ahx.iter().zip(&p.x).map(|(a, b)| a.conj() * b).sum();
    (ay, ahx, (l - r).norm() / l.norm())
}

fn c32(v: &[C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}
fn c64(v: &[C32]) -> Vec<C64> {
    v.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect()
}
fn rel3(a: &[C64], b: &[C64], r: &[C64]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(p, q)| (p - q).norm_sqr()).sum();
    let d: f64 = r.iter().map(|p| p.norm_sqr()).sum();
    (n / d).sqrt()
}

/// size of the physics in the set: concomitant phase range over the support (max over samples of max − min of w·m),
/// RMS / max of the per-sample ΔC phase ρᵀΔCρ over the support, off-resonance range (Hz)
fn physics_size(p: &Prob, s: usize, tab: &SegTables) -> (f64, f64, f64, f64) {
    let rec = &p.recs[s];
    let m = tab.tau.len();
    let sub: Vec<usize> = p.support.iter().copied().step_by(5).collect();
    let mut conc = 0.0f64;
    for r in rec.chunks(SAMPLE_FLOATS).step_by(37) {
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for &q in &sub {
            let v: f64 = (0..4).map(|i| r[9 + i] as f64 * p.mono[q][i]).sum();
            lo = lo.min(v);
            hi = hi.max(v);
        }
        conc = conc.max(hi - lo);
    }
    let (mut s2, mut n, mut mx) = (0.0, 0usize, 0.0f64);
    for j in (0..m).step_by(7) {
        let dc = |i: usize| tab.dc[i * m + j] as f64;
        for &q in &sub {
            let r = &p.rho[q];
            let psi = dc(0) * r[0] * r[0] + dc(1) * r[1] * r[1] + dc(2) * r[2] * r[2] + 2.0 * (dc(3) * r[0] * r[1] + dc(4) * r[0] * r[2] + dc(5) * r[1] * r[2]);
            s2 += psi * psi;
            n += 1;
            mx = mx.max(psi.abs());
        }
    }
    let (lo, hi) = p.support.iter().fold((f64::MAX, f64::MIN), |(a, b), &q| (a.min(p.omega[q]), b.max(p.omega[q])));
    (conc, (s2 / n.max(1) as f64).sqrt(), mx, (hi - lo) / (2.0 * PI))
}

#[derive(Debug, Clone, Copy, Default)]
struct Row {
    set: usize,
    nseg: usize,
    order: u32,
    ntab: usize,
    fwd: f64,
    adj: f64,
    mfwd: f64,
    madj: f64,
    ifwd: f64,
    iadj: f64,
    dot: f64,
    amax: f64,
    psi_rms: f64,
}

/// every (set, L, order, table) of one problem; `dev` None: CPU emulation only
fn sweep(p: &Prob, cfgs: &[(usize, u32, usize)], bspline: u32, dev: Option<&Device>) -> Vec<Row> {
    let mut rows = Vec::new();
    for s in 0..p.sets.len() {
        let m = p.sets[s].tau.len();
        let y = Rng::new(99 + s as u64).complex_vec(NC * m);
        let t_ref = tables(p, s, 1, 2);
        let bw = frame_weights(p, s, &t_ref.tab, bspline);
        let (yref, xref) = reference(p, s, bw.as_deref(), &y);
        let (conc, psi_rms, psi_max, df) = physics_size(p, s, &t_ref.tab);
        eprintln!("  set {s} (encode {}, {m} samples): concomitant phase range over the support {conc:.2} rad, per-sample ΔC phase rms {psi_rms:.3} / max {psi_max:.3} rad, off-resonance range {df:.0} Hz, max phase left out by the quadratic fit {:.1e} rad",
            p.sd.ro.enc[p.sets[s].readouts[0]], t_ref.tab.max_resid_phase);
        let fi = frame_indices(&p.fitv, p.sd.nv(), p.spec.frames);
        for &(nseg, order, ntab) in cfgs {
            let t = tables(p, s, nseg, ntab);
            let (yem, xem) = emulate(p, &t, nseg, order, bw.as_deref(), &y);
            let mut row = Row { set: s, nseg, order, ntab: if ntab == 0 { p.sd.na } else { ntab }, mfwd: rel_on(&yem, &yref, None), madj: rel_on(&xem, &xref, Some(&fi)), amax: t.amax, psi_rms, ..Default::default() };
            if let Some(dev) = dev {
                let (ay, ahx, dot) = gpu(dev, p, &t, nseg, order, bspline, &y);
                row.fwd = rel_on(&ay, &yref, None);
                row.adj = rel_on(&ahx, &xref, Some(&fi));
                row.ifwd = rel3(&ay, &yem, &yref);
                let xi: Vec<C64> = fi.iter().map(|&i| ahx[i] - xem[i]).collect();
                row.iadj = (xi.iter().map(|z| z.norm_sqr()).sum::<f64>() / fi.iter().map(|&i| xref[i].norm_sqr()).sum::<f64>()).sqrt();
                row.dot = dot;
            } else {
                row.fwd = f64::NAN;
                row.adj = f64::NAN;
                row.ifwd = f64::NAN;
                row.iadj = f64::NAN;
                row.dot = f64::NAN;
            }
            rows.push(row);
        }
    }
    eprintln!("     set  L  o  ntab |   fwd      adj    | model fwd  model adj |  impl fwd  impl adj | adjoint id | Σ|a_l|");
    for r in &rows {
        eprintln!("      {}  {:>2}  {}  {:>4} | {:.2e} {:.2e} | {:.2e}  {:.2e}  | {:.2e}  {:.2e} |  {:.1e}   | {:.1}", r.set, r.nseg, r.order, r.ntab, r.fwd, r.adj, r.mfwd, r.madj, r.ifwd, r.iadj, r.dot, r.amax);
    }
    rows
}

// ───────────────────────────────────────────────────────────── configurations

/// (L, correction order, table entries; 0 = one per (decimated) ADC sample)
const CONC_CFGS: [(usize, u32, usize); 8] = [(1, 2, 256), (4, 2, 256), (6, 2, 256), (8, 2, 256), (10, 0, 256), (10, 1, 256), (10, 2, 256), (16, 2, 256)];
const FULL_CFGS: [(usize, u32, usize); 9] = [(8, 2, 256), (10, 2, 256), (12, 2, 256), (14, 2, 256), (16, 2, 256), (20, 2, 256), (10, 2, 0), (14, 2, 0), (20, 2, 0)];

/// the full model and its parts in the same geometry: B0 map + GNL (no concomitant), B0 map + concomitant (no GNL),
/// all three
fn full_specs() -> Vec<Spec> {
    vec![Spec::full(1).without("conc"), Spec::full(1).without("gnl"), Spec::full(1)]
}

fn conc_specs() -> Vec<Spec> {
    let obl = rotation([0.3, 1.0, 0.1], 0.6);
    let mut v = Vec::new();
    for b0 in [0.55, 1.5, 2.89] {
        v.push(Spec::conc_only(b0, [0.0, 0.0, 0.0], IDENTITY));
        v.push(Spec::conc_only(b0, [0.0, 0.0, 0.12], IDENTITY));
        v.push(Spec::conc_only(b0, [0.08, -0.06, 0.15], obl));
    }
    v
}

/// tolerance of the implementation (GPU vs emulation): NUFFT accuracy (σ 1.25, W 10, f32, ~5e-6) times the
/// interpolators' Σ|a_l|, plus the derivative-kernel interpolation of the ΔC terms (2nd derivatives ~1e-4, 4th
/// ~1e-3 relative per the SegNufft notes) times the ΔC phase
fn impl_tol(r: &Row) -> f64 {
    3e-5 * r.amax.max(1.0) + if r.order > 0 { 1e-3 * r.psi_rms + 3e-3 * r.psi_rms * r.psi_rms } else { 0.0 }
}

// ───────────────────────────────────────────────────────────── tests

/// The reference model is the simulator's: the exact factor equals `hasty_mrsim::direct::direct_sum_cpu` (the f64
/// reference of the GPU truth simulator) on a sample subset, the independent GNL inverse agrees with `physics_grid`,
/// and the B0 / coil maps at the true positions agree. CPU only.
#[test]
fn reference_physics_is_the_simulators() {
    let p = problem(&Spec::full(1));
    eprintln!("sequence: {}", p.seq);
    let nv = p.sd.nv();
    // physics_grid vs the independent evaluation
    let rt = transpose(&p.spec.rot);
    let (mut dx, mut dw, mut ds) = (0.0f64, 0.0f64, 0.0f64);
    for q in 0..nv {
        let x = &p.x_true[q];
        let xl = mat_vec(&rt, &[x[0] - p.spec.fov_c[0], x[1] - p.spec.fov_c[1], x[2] - p.spec.fov_c[2]]);
        dx = dx.max((0..3).map(|a| (xl[a] - p.grid.x_log[q][a]).abs()).fold(0.0, f64::max));
        dw = dw.max((p.omega[q] - p.grid.omega[q]).abs());
        for c in 0..NC {
            let (a, b) = (p.smaps[c * nv + q], p.smaps_op[c * nv + q]);
            ds = ds.max((a - C64::new(b.re as f64, b.im as f64)).norm() / a.norm().max(1e-3));
        }
    }
    let disp = (0..nv).map(|q| (0..3).map(|a| (p.x_true[q][a] - p.u_phys[q][a] - p.spec.fov_c[a]).powi(2)).sum::<f64>().sqrt()).fold(0.0, f64::max);
    eprintln!("GNL displacement up to {:.2} mm; physics_grid vs independent: positions {dx:.1e} m, ω {dw:.1e} rad/s, maps {ds:.1e} (rel)", disp * 1e3);
    assert!(disp > 1e-3, "GNL should displace the voxels in this geometry");
    assert!(dx < 1e-9 && dw < 1e-6 && ds < 1e-5);
    // the exact factor vs the simulator's CPU direct sum (sample-major output), 200 samples
    let s = 0;
    let rec = &p.recs[s];
    let m = rec.len() / SAMPLE_FLOATS;
    let sel: Vec<usize> = (0..m).step_by(m / 200).collect();
    let sub: Vec<f32> = sel.iter().flat_map(|&j| rec[j * SAMPLE_FLOATS..(j + 1) * SAMPLE_FLOATS].iter().copied()).collect();
    let src: Vec<Source> = p.support.iter().map(|&q| Source { u: p.u_phys[q], x: p.x_true[q], omega: p.omega[q], ..Default::default() }).collect();
    let (px, psm) = (&p.x, &p.smaps);
    let amp: Vec<C64> = p.support.iter().flat_map(|&q| (0..NC).map(move |c| px[q] * psm[c * nv + q])).collect();
    let sim = direct_sum_cpu(&sub, &src, &amp, NC);
    let (yref, _) = reference(&p, s, None, &vec![C64::new(0.0, 0.0); NC * m]);
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &j) in sel.iter().enumerate() {
        for c in 0..NC {
            num += (sim[i * NC + c] - yref[c * m + j]).norm_sqr();
            den += yref[c * m + j].norm_sqr();
        }
    }
    let e = (num / den).sqrt();
    eprintln!("reference vs hasty_mrsim::direct::direct_sum_cpu on {} samples: {e:.1e}", sel.len());
    assert!(e < 1e-10, "reference direct sum differs from the simulator's model: {e}");
}

/// CPU-only error budget of the concomitant configurations (the "model" columns; no GPU).
#[test]
#[ignore = "CPU-only error budget (~30 s on 32 cores); run with --ignored"]
fn seg_physics_model_error_budget_cpu_conc() {
    for spec in conc_specs() {
        let p = problem(&spec);
        eprintln!("\nconcomitant only: {} — {}", spec.label(), p.seq);
        sweep(&p, &CONC_CFGS, 1, None);
    }
}

/// CPU-only error budget of the full model, its decomposition and the temporal variant (no GPU).
#[test]
#[ignore = "CPU-only error budget (~30 s on 32 cores); run with --ignored"]
fn seg_physics_model_error_budget_cpu_full() {
    for spec in full_specs() {
        let p = problem(&spec);
        eprintln!("\nfull-model geometry: {} — {}", spec.label(), p.seq);
        sweep(&p, &FULL_CFGS, 1, None);
    }
    for (frames, deg) in [(5usize, 1u32), (6, 3)] {
        let p = problem(&Spec::full(frames));
        eprintln!("\ntemporal (B-spline {deg}): {}", p.spec.label());
        sweep(&p, &[(14, 2, 0)], deg, None);
    }
}

/// (c) concomitant fields only (Maxwell terms of the actual waveform, segments for the readout-time mean, per-sample
/// φ0 / k-shift / ΔC correction), B0 = 0.55 / 1.5 / 2.89 T, at the isocentre and off-centre (12 cm along z; 8, −6,
/// 15 cm with an oblique prescription). Asserts, for every case: implementation within `impl_tol`, adjoint identity
/// ≤ 1e-5 (f32); recon default L = 10, order 2: ≤ 1e-3 at 2.89 T and ≤ 6e-3 at 1.5 T in both directions.
/// CPU budget (model part, `seg_physics_model_error_budget_cpu_conc`), L = 10 o2, forward / adjoint:
///   2.89 T: 4.9e-5 / 1.1e-4 (isocentre), 1.2e-4 / 1.8e-4 (z 12 cm), 2.6e-4 / 5.4e-4 (oblique 8, −6, 15 cm);
///   1.5 T: 3.3e-4 / 7.8e-4, 8.0e-4 / 1.3e-3, 2.3e-3 / 4.4e-3; 0.55 T: 6.7e-3 / 1.6e-2 … 2.9e-2 / 6.5e-2 (reported
///   only: the per-sample ΔC phase reaches 1–1.4 rad, beyond a second-order Taylor correction).
/// Order 0 / 1 at L = 10 (2.89 T): 3.2e-2 / 1.3e-3 — the per-readout ΔC correction is needed.
#[test]
fn seg_concomitant_matches_direct_sum() {
    let dev = Device::new(0).unwrap();
    for spec in conc_specs() {
        let p = problem(&spec);
        eprintln!("\nconcomitant only: {} — {}", spec.label(), p.seq);
        let rows = sweep(&p, &CONC_CFGS, 1, Some(&dev));
        for r in &rows {
            assert!(r.ifwd < impl_tol(r) && r.iadj < impl_tol(r), "{}: set {} L {} o{}: implementation error {:.2e} / {:.2e} (tol {:.1e})", spec.label(), r.set, r.nseg, r.order, r.ifwd, r.iadj, impl_tol(r));
            assert!(r.dot < 1e-5, "adjoint identity {:.1e}", r.dot);
            if r.nseg == 10 && r.order == 2 {
                let tol = if spec.b0 > 2.0 { 1e-3 } else if spec.b0 > 1.0 { 6e-3 } else { f64::INFINITY };
                assert!(r.fwd < tol && r.adj < tol, "{}: set {}: L 10 order 2 error {:.2e} / {:.2e} (tol {tol:.0e})", spec.label(), r.set, r.fwd, r.adj);
            }
        }
    }
}

/// (d) + (e) B0 map (smooth ±150 Hz + a −3.4 ppm shell) + concomitant + GNL (true positions for B0, concomitant
/// monomials and coil maps; image in apparent coordinates), 2.89 T, oblique off-centre prescription, 8 ms readouts,
/// and the same geometry with the concomitant or the GNL part switched off (decomposition): forward and adjoint vs the
/// direct sum for L = 8 … 20 with the reconstruction's 256-entry table and with an ADC-aligned table; adjoint identity
/// ⟨y, A x⟩ = ⟨Aᴴ y, x⟩ for every configuration (f32: ≤ 1e-5).
/// Asserts: implementation within `impl_tol` everywhere; aligned table, L = 20: B0 + GNL ≤ 1e-3, B0 + concomitant
/// ≤ 2e-3, full ≤ 1.5e-2.
/// CPU budget (`seg_physics_model_error_budget_cpu_full`), forward / adjoint:
///   B0 + GNL: L 10 4.9e-3, L 14 5.7e-5 (aligned) / 3.3e-4 (256 table), L 20 7.8e-6 (aligned);
///   B0 + concomitant (joint segments): L 10 8.6e-3 / 1.0e-2, L 14 2.0e-3 / 2.4e-3, L 20 7.0e-4 / 9.7e-4;
///   full (+ GNL): L 10 9.4e-3 / 1.2e-2, L ≥ 14 floor 4.8e-3 / 8.3e-3 — the quadratic fit of the concomitant
///   monomials at the GNL-warped true positions leaves up to 0.17 rad of the per-sample deviation out
///   (`SegTables::max_resid_phase`); neither L nor the table removes it.
#[test]
fn seg_full_physics_matches_direct_sum_and_is_adjoint() {
    let dev = Device::new(0).unwrap();
    for spec in full_specs() {
        let p = problem(&spec);
        eprintln!("\nfull-model geometry: {} — {}", spec.label(), p.seq);
        let rows = sweep(&p, &FULL_CFGS, 1, Some(&dev));
        let tol20 = match (spec.conc, spec.gnl) {
            (false, _) => 1e-3,
            (true, false) => 2e-3,
            (true, true) => 1.5e-2,
        };
        for r in &rows {
            assert!(r.ifwd < impl_tol(r) && r.iadj < impl_tol(r), "{}: set {} L {} ntab {}: implementation error {:.2e} / {:.2e} (tol {:.1e})", spec.label(), r.set, r.nseg, r.ntab, r.ifwd, r.iadj, impl_tol(r));
            assert!(r.dot < 1e-5, "adjoint identity {:.1e}", r.dot);
            if r.nseg == 20 && r.ntab != 256 {
                assert!(r.fwd < tol20 && r.adj < tol20, "{}: set {}: L 20 aligned table error {:.2e} / {:.2e} (tol {tol20:.0e})", spec.label(), r.set, r.fwd, r.adj);
            }
        }
    }
}

/// (f) temporal B-spline interpolation over cardiac frames (`--temporal linear|cubic`) with the full physics:
/// samples at their own cardiac time (from the readout's cardiac phase and τ, as `seg_tables`), 5 frames linear and
/// 6 frames cubic, L = 14, order 2, aligned table. Asserts: implementation within `impl_tol`, adjoint identity ≤ 1e-5,
/// error ≤ 1.5e-2 (CPU budget: 5.4e-3 / 8.9e-3, the same GNL fit floor as the static full model).
#[test]
fn seg_temporal_full_physics_matches_direct_sum() {
    let dev = Device::new(0).unwrap();
    for (frames, deg) in [(5usize, 1u32), (6, 3)] {
        let p = problem(&Spec::full(frames));
        eprintln!("\ntemporal (B-spline {deg}): {}", p.spec.label());
        let rows = sweep(&p, &[(14, 2, 0)], deg, Some(&dev));
        for r in &rows {
            assert!(r.ifwd < impl_tol(r) && r.iadj < impl_tol(r), "frames {frames}: implementation error {:.2e} / {:.2e} (tol {:.1e})", r.ifwd, r.iadj, impl_tol(r));
            assert!(r.dot < 1e-5, "adjoint identity {:.1e}", r.dot);
            assert!(r.fwd < 1.5e-2 && r.adj < 1.5e-2, "frames {frames}: error {:.2e} / {:.2e}", r.fwd, r.adj);
        }
    }
}

/// (a) `--model sense` (hasty-recon SenseOp: σ = 2, tol 1e-4 as recon_p2np) and `--model seg --phys none` (SegNufft
/// L = 1) on the real trajectory with coil maps, no physics, vs the direct sum; forward and adjoint. Tolerances: the
/// NUFFT accuracy (8 × tol in f32 as in hasty-nufft `tests/accuracy.rs`: 8e-4 for tol 1e-4; 3e-5 for W = 10 at
/// σ = 1.25, FINUFFT-rule tolerance 3.3e-6).
#[test]
fn sense_and_plain_seg_match_direct_sum() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let mut spec = Spec::conc_only(2.89, [0.0; 3], IDENTITY);
    spec.conc = false;
    let p = problem(&spec);
    let nv = p.sd.nv();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    for s in 0..p.sets.len() {
        let set = &p.sets[s];
        let m = set.tau.len();
        let y = Rng::new(7 + s as u64).complex_vec(NC * m);
        let (yref, xref) = reference(&p, s, None, &y);
        let mut op = SenseOp::new(&dev, &st, kern.clone(), p.sd.dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &p.smaps_op, NC, vec![], 1e-4).unwrap();
        let (dx, dy) = (st.clone_htod(&c32(&p.x)).unwrap(), st.clone_htod(&c32(&y)).unwrap());
        let mut ay = st.alloc_zeros::<C32>(NC * m).unwrap();
        let mut ahx = st.alloc_zeros::<C32>(nv).unwrap();
        op.forward(&st, &dx, &mut ay).unwrap();
        op.adjoint(&st, &dy, &mut ahx).unwrap();
        let (ef, ea) = (rel_on(&c64(&st.clone_dtoh(&ay).unwrap()), &yref, None), rel_on(&c64(&st.clone_dtoh(&ahx).unwrap()), &xref, Some(&p.fitv)));
        let t = tables(&p, s, 1, 2);
        let (sy, sx, dot) = gpu(&dev, &p, &t, 1, 0, 1, &y);
        let (gf, ga) = (rel_on(&sy, &yref, None), rel_on(&sx, &xref, Some(&p.fitv)));
        eprintln!("set {s} ({m} samples): SenseOp σ 2 tol 1e-4: forward {ef:.2e}, adjoint {ea:.2e}; SegNufft L 1: forward {gf:.2e}, adjoint {ga:.2e}, adjoint identity {dot:.1e}");
        assert!(ef < 8e-4 && ea < 8e-4, "SenseOp error {ef:.2e} / {ea:.2e}");
        assert!(gf < 3e-5 && ga < 3e-5, "SegNufft L 1 error {gf:.2e} / {ga:.2e}");
    }
}

// ───────────────────────────────────────────────────────────── where the full model's floor comes from (CPU)

/// Split of the full model's (B0 map + concomitant + GNL) error into its sources, CPU emulation only, and the candidate
/// fixes. Per sample j and voxel ρ the operator replaces the deviation phase Δw_j·m(x(ρ)) (x = true, GNL-warped
/// position; Δw_j = w_j − w̄(τ_j), w̄ handled exactly by the segments) by φ0 + b·ρ + ρᵀΔC_jρ from the quadratic fit of
/// m_q over the support, and e^{−iρᵀΔCρ} by its second-order Taylor factor T2. Variants (factor per (j, ρ)):
///   segs        e^{−ik·ρ} S_j(ρ) e^{−iΔw·m}           segments / table only (everything else exact)
///   quad exact  e^{−i lin} S e^{−iψ2}                 + the fit residual R = Δw·m − (φ0 + b·ρ + ψ2) left out
///   current     e^{−i lin} S T2(ψ2)                   the operator (`--order 2`)
///   cur + R     e^{−i lin} S T2(ψ2) e^{−iR}           Taylor truncation only (residual exact)
///   cur + 1−iR  e^{−i lin} S T2(ψ2) (1 − iR)          the residual to first order (≈ 4 extra images, multi path)
///   deg4 exact  e^{−i lin'} S e^{−i(ψ2' + c34)}       degree-4 fit, polynomial exact: its residual R' left out
///   order 3     e^{−i lin'} S [T2(ψ2') − i c34]       `SegNufft` order 3: degree-4 fit, cubic / quartic first order
/// Also: RMS / max of R and R' over (samples, support) and the singular values of R (rank ≤ 4: Δw_j × r_q(ρ)).
#[test]
#[ignore = "CPU-only analysis (~2 min on 32 cores); run with --ignored"]
fn seg_full_conc_fit_analysis_cpu() {
    for b0 in [2.89, 1.5, 0.55] {
        let mut spec = Spec::full(1);
        spec.b0 = b0;
        let p = problem(&spec);
        eprintln!("\nfull model: {} — {}", spec.label(), p.seq);
        analyse_fit(&p, &[(10, 0), (20, 0), (20, 256)]);
    }
}

fn analyse_fit(p: &Prob, cfgs: &[(usize, usize)]) {
    let nv = p.sd.nv();
    let na = p.sd.na;
    let fit = fit_monomials(&p.sd, &p.grid, &p.fitv);
    let fit4: MonoFitHi = fit_monomials_hi(&p.sd, &p.grid, &p.fitv);
    eprintln!("  quadratic fit residual (m²) max {:?}, rms {:?}", fit.max.map(|v| format!("{v:.1e}")), fit.rms.map(|v| format!("{v:.1e}")));
    eprintln!("  degree-4 fit residual  (m²) max {:?}, rms {:?}", fit4.quad.max.map(|v| format!("{v:.1e}")), fit4.quad.rms.map(|v| format!("{v:.1e}")));
    let smaps64: Vec<C64> = p.smaps_op.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
    for s in 0..p.sets.len() {
        let set = &p.sets[s];
        let m = set.tau.len();
        let y = Rng::new(99 + s as u64).complex_vec(NC * m);
        let (yref, xref) = reference(p, s, None, &y);
        let wbar = mean_coefficients(&[set], na);
        let tab = seg_tables(&p.sd, set, &fit, true, Some(&wbar));
        let (tab4, dw4) = seg_tables_hi(&p.sd, set, &fit4, true, Some(&wbar));
        let dw: Vec<[f64; 4]> = (0..m).map(|j| [0, 1, 2, 3].map(|q| set.wq[j][q] - wbar[j % na][q])).collect();
        for j in (0..m).step_by(97) {
            for q in 0..4 {
                assert!((dw4[q * m + j] as f64 - dw[j][q]).abs() <= 1e-6 * dw[j][q].abs().max(1e-3));
            }
        }
        // per-sample phases on the fit voxels: exact deviation, linear part, ψ2, residuals R (quadratic) and R' (deg 4)
        let lin = |t: &SegTables, j: usize, q: usize| (0..3).map(|a| t.coords[a][j] as f64 * p.idx[q][a]).sum::<f64>() + t.phi0[j] as f64;
        let psi2 = |t: &SegTables, j: usize, q: usize| {
            let (r, dc) = (&p.rho[q], |i: usize| t.dc[i * m + j] as f64);
            dc(0) * r[0] * r[0] + dc(1) * r[1] * r[1] + dc(2) * r[2] * r[2] + 2.0 * (dc(3) * r[0] * r[1] + dc(4) * r[0] * r[2] + dc(5) * r[1] * r[2])
        };
        let k0 = |j: usize, q: usize| (0..3).map(|a| set.coords[a][j] as f64 * p.idx[q][a]).sum::<f64>();
        let dev = |j: usize, q: usize| (0..4).map(|i| dw[j][i] * p.grid.mono[q][i]).sum::<f64>();
        let c34 = |j: usize, q: usize| fit4.hi_phase(&dw[j], &p.rho[q]);
        let resid = |j: usize, q: usize| dev(j, q) - (lin(&tab, j, q) - k0(j, q)) - psi2(&tab, j, q);
        let resid4 = |j: usize, q: usize| dev(j, q) - (lin(&tab4, j, q) - k0(j, q)) - psi2(&tab4, j, q) - c34(j, q);
        let stat = |f: &dyn Fn(usize, usize) -> f64| {
            let (mut s2, mut n, mut mx) = (0.0, 0usize, 0.0f64);
            for j in (0..m).step_by(3) {
                for &q in &p.support {
                    let v = f(j, q);
                    s2 += v * v;
                    n += 1;
                    mx = mx.max(v.abs());
                }
            }
            ((s2 / n as f64).sqrt(), mx)
        };
        let (r_rms, r_max) = stat(&resid);
        let (r4_rms, r4_max) = stat(&resid4);
        let (p2_rms, p2_max) = stat(&|j, q| psi2(&tab, j, q));
        let (c_rms, c_max) = stat(&c34);
        // singular values of R(j, ρ) = Σ_q Δw_jq r_q(ρ): eigenvalues of (DWᵀDW)(RvRvᵀ)
        let rq: Vec<[f64; 4]> = p.support.iter().map(|&q| {
            let b = [1.0, p.rho[q][0], p.rho[q][1], p.rho[q][2], p.rho[q][0].powi(2), p.rho[q][1].powi(2), p.rho[q][2].powi(2), 2.0 * p.rho[q][0] * p.rho[q][1], 2.0 * p.rho[q][0] * p.rho[q][2], 2.0 * p.rho[q][1] * p.rho[q][2]];
            [0, 1, 2, 3].map(|i| p.grid.mono[q][i] - (0..10).map(|c| fit.theta[i][c] * b[c]).sum::<f64>())
        }).collect();
        let dd = nalgebra::Matrix4::<f64>::from_fn(|a, b| dw.iter().map(|w| w[a] * w[b]).sum());
        let rr = nalgebra::Matrix4::<f64>::from_fn(|a, b| rq.iter().map(|r| r[a] * r[b]).sum());
        let mut ev: Vec<f64> = (dd * rr).complex_eigenvalues().iter().map(|z| z.re.max(0.0).sqrt()).collect();
        ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let tot: f64 = ev.iter().map(|v| v * v).sum();
        eprintln!("  set {s} ({m} samples): residual R rms {r_rms:.2e} / max {r_max:.2e} rad (deg-4 R' {r4_rms:.2e} / {r4_max:.2e}); ψ2 rms {p2_rms:.2e} / max {p2_max:.2e}; c34 rms {c_rms:.2e} / max {c_max:.2e}; σ(R) energy fractions {:?}",
                  ev.iter().map(|v| format!("{:.3}", v * v / tot)).collect::<Vec<_>>());
        eprintln!("      L  ntab | segs            | quad exact      | current         | cur + R         | cur + 1−iR      | deg4 exact      | order 3         | o3 + ψ2·c34     | T2(ψ2 + c34)");
        for &(nseg, ntab) in cfgs {
            let t = tables(p, s, nseg, ntab);
            let p64: Vec<C64> = t.segph.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
            let a64: Vec<C64> = t.atab.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
            let at: Vec<C64> = tab.tau.iter().flat_map(|&tau| interp_table(&a64, nseg, t.t0, t.dt, tau as f64)).collect();
            let sj = |j: usize, q: usize| -> C64 { (0..nseg).map(|l| at[j * nseg + l] * p64[l * nv + q]).sum() };
            let ds = DirectSum { nv, nc: NC, m, frames: 1, smaps: &smaps64, voxels: &p.fitv, bw: None };
            let ex = |ph: f64| C64::from_polar(1.0, -ph);
            let t2 = |v: f64| taylor_expm(v, 2);
            let variants: [Box<dyn Fn(usize, usize) -> C64 + Sync>; 9] = [
                Box::new(|j, q| ex(k0(j, q)) * sj(j, q) * ex(dev(j, q))),
                Box::new(|j, q| ex(lin(&tab, j, q)) * sj(j, q) * ex(psi2(&tab, j, q))),
                Box::new(|j, q| ex(lin(&tab, j, q)) * sj(j, q) * t2(psi2(&tab, j, q))),
                Box::new(|j, q| ex(lin(&tab, j, q)) * sj(j, q) * t2(psi2(&tab, j, q)) * ex(resid(j, q))),
                Box::new(|j, q| ex(lin(&tab, j, q)) * sj(j, q) * t2(psi2(&tab, j, q)) * C64::new(1.0, -resid(j, q))),
                Box::new(|j, q| ex(lin(&tab4, j, q)) * sj(j, q) * ex(psi2(&tab4, j, q) + c34(j, q))),
                Box::new(|j, q| ex(lin(&tab4, j, q)) * sj(j, q) * (t2(psi2(&tab4, j, q)) - C64::new(0.0, c34(j, q)))),
                Box::new(|j, q| ex(lin(&tab4, j, q)) * sj(j, q) * (t2(psi2(&tab4, j, q)) - C64::new(psi2(&tab4, j, q) * c34(j, q), c34(j, q)))),
                Box::new(|j, q| ex(lin(&tab4, j, q)) * sj(j, q) * t2(psi2(&tab4, j, q) + c34(j, q))),
            ];
            let mut line = format!("     {nseg:>2}  {:>4} |", if ntab == 0 { na } else { ntab });
            for v in &variants {
                let (ye, xe) = ds.apply(v.as_ref(), &p.x, &y);
                line += &format!(" {:.2e} {:.2e} |", rel_on(&ye, &yref, None), rel_on(&xe, &xref, Some(&p.fitv)));
            }
            eprintln!("{line}");
        }
    }
}

// ───────────────────────────────────────────────────────────── `--conc-fit deg4`: SegNufft correction order 3

/// tables of the order-3 operator: the degree-4 fit, `seg_tables_hi` (quadratic part, Δw), segments as `tables`
fn tables3(p: &Prob, s: usize, nseg: usize, ntab: usize) -> (Tables, Vec<f32>, MonoFitHi) {
    let (sd, set) = (&p.sd, &p.sets[s]);
    let na = sd.na;
    let fit4 = fit_monomials_hi(sd, &p.grid, &p.fitv);
    let wbar = mean_coefficients(&[set], na);
    let (tab, dw) = seg_tables_hi(sd, set, &fit4, true, Some(&wbar));
    let t = tables(p, s, nseg, ntab);
    (Tables { tab, ..t }, dw, fit4)
}

/// CPU emulation of order 3: e^{−i lin} S [T2(ψ2) − i c34], c34 = Σ_α (Σ_q Δw_q h_qα) ρ^α
fn emulate3(p: &Prob, t: &Tables, dw: &[f32], fit4: &MonoFitHi, nseg: usize, y: &[C64]) -> (Vec<C64>, Vec<C64>) {
    let nv = p.sd.nv();
    let tab = &t.tab;
    let m = tab.tau.len();
    let p64: Vec<C64> = t.segph.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
    let a64: Vec<C64> = t.atab.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
    let at: Vec<C64> = tab.tau.iter().flat_map(|&tau| interp_table(&a64, nseg, t.t0, t.dt, tau as f64)).collect();
    let smaps64: Vec<C64> = p.smaps_op.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
    let f = |j: usize, q: usize| {
        let (ix, r) = (&p.idx[q], &p.rho[q]);
        let lin = tab.coords[0][j] as f64 * ix[0] + tab.coords[1][j] as f64 * ix[1] + tab.coords[2][j] as f64 * ix[2] + tab.phi0[j] as f64;
        let dc = |i: usize| tab.dc[i * m + j] as f64;
        let psi = dc(0) * r[0] * r[0] + dc(1) * r[1] * r[1] + dc(2) * r[2] * r[2] + 2.0 * (dc(3) * r[0] * r[1] + dc(4) * r[0] * r[2] + dc(5) * r[1] * r[2]);
        let w = [0, 1, 2, 3].map(|i| dw[i * m + j] as f64);
        let s: C64 = (0..nseg).map(|l| at[j * nseg + l] * p64[l * nv + q]).sum();
        C64::from_polar(1.0, -lin) * s * (taylor_expm(psi, 2) - C64::new(0.0, fit4.hi_phase(&w, r)))
    };
    let ds = DirectSum { nv, nc: NC, m, frames: 1, smaps: &smaps64, voxels: &p.fitv, bw: None };
    ds.apply(&f, &p.x, y)
}

/// the order-3 operator on the GPU (static image): (A x, Aᴴ y, adjoint-identity mismatch)
fn gpu3(dev: &Device, p: &Prob, t: &Tables, dw: &[f32], h: &[f64], nseg: usize, y: &[C64]) -> (Vec<C64>, Vec<C64>, f64) {
    let st = dev.default_stream();
    let nv = p.sd.nv();
    let mut eng = SegNufft::new(dev, p.sd.dims, p.sd.vox, NC, SegNufftOpts { sigma: 1.25, width: 10, nseg, order: 3, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
    eng.set_maps(&st, &p.smaps_op).unwrap();
    let seg = eng.segments(&st, &t.segph, &t.atab, t.t0, t.dt).unwrap();
    let tab = &t.tab;
    let pts = eng.points_hi(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, dw, h, None).unwrap();
    let m = tab.tau.len();
    let dx = st.clone_htod(&c32(&p.x)).unwrap();
    let dy = st.clone_htod(&c32(y)).unwrap();
    let mut ay = st.alloc_zeros::<C32>(NC * m).unwrap();
    let mut ahx = st.alloc_zeros::<C32>(nv).unwrap();
    eng.forward(&st, &seg, &pts, &dx, &mut ay).unwrap();
    eng.adjoint(&st, &seg, &pts, &dy, &mut ahx).unwrap();
    let (ay, ahx) = (c64(&st.clone_dtoh(&ay).unwrap()), c64(&st.clone_dtoh(&ahx).unwrap()));
    let l: C64 = y.iter().zip(&ay).map(|(a, b)| a.conj() * b).sum();
    let r: C64 = ahx.iter().zip(&p.x).map(|(a, b)| a.conj() * b).sum();
    (ay, ahx, (l - r).norm() / l.norm())
}

/// (g) `recon_p2np --conc-fit deg4`: full model (B0 map + concomitant + GNL, oblique off-centre, 8 ms readouts) at
/// 2.89 T and 1.5 T, L = 10 and 20 with the ADC-aligned table: the order-3 operator (degree-4 fit of the monomials at
/// the GNL-warped positions, degree-3/4 part to first order in the kernel) and the order-2 operator vs the direct sum.
/// Asserts: implementation (GPU vs order-3 emulation) within `impl_tol` + 2e-3 × the degree-3/4 phase rms, adjoint
/// identity ≤ 1e-5; at L = 20: order 3 ≤ 2e-3 / 3.5e-3 (2.89 T) and ≤ 5e-3 / 1e-2 (1.5 T) forward / adjoint and at least
/// 2× below order 2 in both directions.
/// CPU budget (`seg_full_conc_fit_analysis_cpu`, columns "current" / "order 3"), forward / adjoint:
///   2.89 T: L 10 9.4e-3 / 1.2e-2 → 8.3e-3 / 9.5e-3 (segments dominate), L 20 4.8e-3 / 8.3e-3 → 1.15e-3 / 2.2e-3;
///   1.5 T:  L 10 1.0e-2 / 1.7e-2 → 6.1e-3 / 9.2e-3, L 20 8.9e-3 / 1.5e-2 → 2.95e-3 / 6.5e-3;
///   0.55 T (reported): L 20 3.1e-2 / 7.0e-2 → 2.9e-2 / 7.3e-2 (the second-order Taylor of ψ2 dominates).
#[test]
fn seg_full_physics_order3_matches_direct_sum() {
    let dev = Device::new(0).unwrap();
    for b0 in [2.89, 1.5] {
        let mut spec = Spec::full(1);
        spec.b0 = b0;
        let p = problem(&spec);
        eprintln!("\norder 3 (--conc-fit deg4): {} — {}", spec.label(), p.seq);
        for s in 0..p.sets.len() {
            let m = p.sets[s].tau.len();
            let y = Rng::new(99 + s as u64).complex_vec(NC * m);
            let (yref, xref) = reference(&p, s, None, &y);
            for nseg in [10usize, 20] {
                let t2 = tables(&p, s, nseg, 0);
                let (ay2, ahx2, _) = gpu(&dev, &p, &t2, nseg, 2, 1, &y);
                let (e2f, e2a) = (rel_on(&ay2, &yref, None), rel_on(&ahx2, &xref, Some(&p.fitv)));
                let (t3, dw, fit4) = tables3(&p, s, nseg, 0);
                let h = hi_table(&fit4);
                let (ay, ahx, dot) = gpu3(&dev, &p, &t3, &dw, &h, nseg, &y);
                let (yem, xem) = emulate3(&p, &t3, &dw, &fit4, nseg, &y);
                let (ef, ea) = (rel_on(&ay, &yref, None), rel_on(&ahx, &xref, Some(&p.fitv)));
                let (mf, ma) = (rel_on(&yem, &yref, None), rel_on(&xem, &xref, Some(&p.fitv)));
                let ifwd = rel3(&ay, &yem, &yref);
                let xi: Vec<C64> = p.fitv.iter().map(|&i| ahx[i] - xem[i]).collect();
                let iadj = (xi.iter().map(|z| z.norm_sqr()).sum::<f64>() / p.fitv.iter().map(|&i| xref[i].norm_sqr()).sum::<f64>()).sqrt();
                let (_, psi_rms, _, _) = physics_size(&p, s, &t3.tab);
                let (mut c2, mut n) = (0.0, 0usize);
                for j in (0..m).step_by(7) {
                    let w = [0, 1, 2, 3].map(|i| dw[i * m + j] as f64);
                    for &q in p.support.iter().step_by(5) {
                        c2 += fit4.hi_phase(&w, &p.rho[q]).powi(2);
                        n += 1;
                    }
                }
                let c_rms = (c2 / n as f64).sqrt();
                let row = Row { order: 3, amax: t3.amax, psi_rms, ..Default::default() };
                let tol = impl_tol(&row) + 2e-3 * c_rms;
                eprintln!("  set {s} L {nseg:>2}: order 2 {e2f:.2e} / {e2a:.2e} | order 3 {ef:.2e} / {ea:.2e} (model {mf:.2e} / {ma:.2e}, impl {ifwd:.2e} / {iadj:.2e}, tol {tol:.1e}; degree-3/4 phase rms {c_rms:.1e} rad) | adjoint id {dot:.1e}");
                assert!(ifwd < tol && iadj < tol, "{}: set {s} L {nseg}: order-3 implementation error {ifwd:.2e} / {iadj:.2e} (tol {tol:.1e})", spec.label());
                assert!(dot < 1e-5, "adjoint identity {dot:.1e}");
                if nseg == 20 {
                    let (tf, ta) = if b0 > 2.0 { (2e-3, 3.5e-3) } else { (5e-3, 1e-2) };
                    assert!(ef < tf && ea < ta, "{}: set {s}: order 3 L 20 error {ef:.2e} / {ea:.2e} (tol {tf:.0e} / {ta:.0e})", spec.label());
                    assert!(ef < 0.5 * e2f && ea < 0.5 * e2a, "order 3 {ef:.2e} / {ea:.2e} not 2× below order 2 {e2f:.2e} / {e2a:.2e}");
                }
            }
        }
    }
}

/// CPU-only error budget of `--conc-fit deg4` (order 3) against the order-2 operator, full model, ADC-aligned table and
/// the 256 table, L = 10 / 20, 2.89 / 1.5 / 0.55 T (the emulations of `seg_full_physics_order3_matches_direct_sum`).
#[test]
#[ignore = "CPU-only error budget (~1 min on 32 cores); run with --ignored"]
fn seg_physics_model_error_budget_cpu_order3() {
    for b0 in [2.89, 1.5, 0.55] {
        let mut spec = Spec::full(1);
        spec.b0 = b0;
        let p = problem(&spec);
        eprintln!("\n{}", spec.label());
        eprintln!("     set  L  ntab | order 2 fwd  adj      | order 3 fwd  adj      | max phase left out: quadratic / degree-4 fit");
        for s in 0..p.sets.len() {
            let m = p.sets[s].tau.len();
            let y = Rng::new(99 + s as u64).complex_vec(NC * m);
            let (yref, xref) = reference(&p, s, None, &y);
            for (nseg, ntab) in [(10usize, 0usize), (20, 0), (20, 256)] {
                let t2 = tables(&p, s, nseg, ntab);
                let (y2, x2) = emulate(&p, &t2, nseg, 2, None, &y);
                let (t3, dw, fit4) = tables3(&p, s, nseg, ntab);
                let (y3, x3) = emulate3(&p, &t3, &dw, &fit4, nseg, &y);
                eprintln!("      {s}  {nseg:>2}  {:>4} | {:.2e}  {:.2e}  | {:.2e}  {:.2e}  | {:.2e} / {:.2e} rad", if ntab == 0 { p.sd.na } else { ntab },
                    rel_on(&y2, &yref, None), rel_on(&x2, &xref, Some(&p.fitv)), rel_on(&y3, &yref, None), rel_on(&x3, &xref, Some(&p.fitv)), t2.tab.max_resid_phase, t3.tab.max_resid_phase);
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════ E2 (research/stencil_weighted_update.md)
// Per-sample polynomial p_j applied by the order-2 / order-3 kernel, chosen by Taylor, centred Taylor, Chebyshev
// (near-minimax on the per-sample range) or weighted least squares; CPU emulation as `emulate` / `emulate3` (exact
// derivatives, i.e. the kernel's window error is not included), L = 20, ADC-aligned table.
use hasty_recon::segphys::MonoFit;
use rayon::prelude::*;

fn box3(v: &[f64]) -> Vec<f64> {
    let (n0, n1, n2) = (DIMS[0] as i64, DIMS[1] as i64, DIMS[2] as i64);
    let mut o = vec![0.0; v.len()];
    for k in 0..n2 { for j in 0..n1 { for i in 0..n0 {
        let (mut s, mut c) = (0.0, 0.0);
        for dk in -1..=1i64 { for dj in -1..=1i64 { for di in -1..=1i64 {
            let (a, b, cc) = (i + di, j + dj, k + dk);
            if a < 0 || b < 0 || cc < 0 || a >= n0 || b >= n1 || cc >= n2 { continue; }
            s += v[((cc * n1 + b) * n0 + a) as usize]; c += 1.0;
        }}}
        o[((k * n1 + j) * n0 + i) as usize] = s / c;
    }}}
    o
}

/// real-basis weighted LS with complex target: min Σ_i w_i |Σ_b c_b B_b(i) − t_i|²
/// weighted LS shrunk toward c0: min Σ_i w_i |Σ_b c_b B_b(i) − t_i|² + eps Σw ‖c − c0‖², basis in rad units (no
/// per-sample rescaling, so a basis function that is ≈ 0 on the support keeps its Taylor coefficient)
fn wls_toward(nb: usize, bas: &[f64], t: &[C64], w: &[f64], c0: &[C64], eps: f64) -> Vec<C64> {
    let n = t.len();
    let sw: f64 = w.iter().sum::<f64>().max(1e-300);
    let mut g = nalgebra::DMatrix::<f64>::zeros(nb, nb);
    let mut r = nalgebra::DMatrix::<f64>::zeros(nb, 2);
    for i in 0..n {
        if w[i] == 0.0 { continue; }
        let pred: C64 = (0..nb).map(|b| c0[b] * bas[i * nb + b]).sum();
        let res = t[i] - pred;
        for a in 0..nb {
            let ba = bas[i * nb + a];
            for b in 0..nb { g[(a, b)] += w[i] * ba * bas[i * nb + b]; }
            r[(a, 0)] += w[i] * ba * res.re;
            r[(a, 1)] += w[i] * ba * res.im;
        }
    }
    for a in 0..nb { g[(a, a)] += eps * sw; }
    let s = g.cholesky().expect("wls_toward").solve(&r);
    (0..nb).map(|a| c0[a] + C64::new(s[(a, 0)], s[(a, 1)])).collect()
}

fn wls(nb: usize, bas: &[f64], t: &[C64], w: &[f64]) -> Vec<C64> {
    let n = t.len();
    let mut sc = vec![0.0f64; nb];
    let sw: f64 = w.iter().sum::<f64>().max(1e-300);
    for b in 0..nb { sc[b] = ((0..n).map(|i| w[i] * bas[i * nb + b].powi(2)).sum::<f64>() / sw).sqrt().max(1e-300); }
    let mut g = nalgebra::DMatrix::<f64>::zeros(nb, nb);
    let mut r = nalgebra::DMatrix::<f64>::zeros(nb, 2);
    for i in 0..n {
        if w[i] == 0.0 { continue; }
        for a in 0..nb {
            let ba = bas[i * nb + a] / sc[a];
            for b in 0..nb { g[(a, b)] += w[i] * ba * bas[i * nb + b] / sc[b]; }
            r[(a, 0)] += w[i] * ba * t[i].re;
            r[(a, 1)] += w[i] * ba * t[i].im;
        }
    }
    let tr = (0..nb).map(|a| g[(a, a)]).sum::<f64>() / nb as f64;
    for a in 0..nb { g[(a, a)] += 1e-13 * tr; }
    let s = g.cholesky().expect("wls").solve(&r);
    (0..nb).map(|a| C64::new(s[(a, 0)], s[(a, 1)]) / sc[a]).collect()
}

/// energy-weighted quadratic / degree-4 monomial fit (weights per voxel of `vox`)
fn fit_w(p: &Prob, vox: &[usize], w: &[f64], deg4: bool) -> (MonoFit, [[f64; 25]; 4]) {
    let ex = hasty_recon::segphys::hi_exponents();
    let s = vox.iter().map(|&v| p.rho[v].iter().fold(0.0f64, |a, &c| a.max(c.abs()))).fold(0.0f64, f64::max);
    let nb = if deg4 { 35 } else { 10 };
    let deg = |i: usize| -> i32 { if i == 0 { 0 } else if i < 4 { 1 } else if i < 10 { 2 } else if i < 20 { 3 } else { 4 } };
    let bas = |r: &Vec3| -> Vec<f64> {
        let u = [r[0] / s, r[1] / s, r[2] / s];
        let mut b = vec![1.0, u[0], u[1], u[2], u[0] * u[0], u[1] * u[1], u[2] * u[2], 2.0 * u[0] * u[1], 2.0 * u[0] * u[2], 2.0 * u[1] * u[2]];
        if deg4 { for i in 0..25 { b.push(u[0].powi(ex[i][0] as i32) * u[1].powi(ex[i][1] as i32) * u[2].powi(ex[i][2] as i32)); } }
        b
    };
    let mut a = nalgebra::DMatrix::<f64>::zeros(nb, nb);
    let mut rhs = nalgebra::DMatrix::<f64>::zeros(nb, 4);
    for (k, &v) in vox.iter().enumerate() {
        let b = bas(&p.rho[v]);
        for i in 0..nb { for l in 0..nb { a[(i, l)] += w[k] * b[i] * b[l]; } for q in 0..4 { rhs[(i, q)] += w[k] * b[i] * p.grid.mono[v][q]; } }
    }
    let tr = (0..nb).map(|i| a[(i, i)]).sum::<f64>() / nb as f64;
    for i in 0..nb { a[(i, i)] += 1e-12 * tr; }
    let sol = a.cholesky().expect("weighted fit").solve(&rhs);
    let mut theta = [[0.0; 10]; 4];
    let mut hi = [[0.0; 25]; 4];
    for q in 0..4 { for i in 0..nb { let c = sol[(i, q)] / s.powi(deg(i)); if i < 10 { theta[q][i] = c } else { hi[q][i - 10] = c } } }
    (MonoFit { theta, rms: [0.0; 4], max: [1.0; 4] }, hi)
}

/// the per-sample coefficients [1, ψ, ψ², c] through the proposed 16-byte side record (8 × f16, deviations from Taylor)
fn quant16(cf: &[C64], nb: usize) -> Vec<C64> {
    let i = C64::new(0.0, 1.0);
    let h = |z: C64| C64::new(half(z.re), half(z.im));
    cf.chunks(5).flat_map(|c| {
        let mut o = [C64::new(0.0, 0.0); 5];
        o[0] = 1.0 + h(c[0] - 1.0);
        o[1] = -(i + h(-c[1] - i));
        if nb > 2 { o[2] = -0.5 * (1.0 + h(-2.0 * c[2] - 1.0)); }
        if nb > 3 { o[3] = -i * (1.0 + h(i * c[3] - 1.0)); }
        o
    }).collect()
}

/// band-limited copies of an image on the DIMS grid: x_b = IDFT(1_{|κ| ∈ band b} DFT x), |κ| in rad/voxel (wrapped)
fn bands(x: &[C64], edges: &[f64]) -> Vec<Vec<C64>> {
    let n = DIMS;
    let dft = |v: &mut Vec<C64>, axis: usize, sign: f64| {
        let len = n[axis];
        let stride = [1, n[0], n[0] * n[1]][axis];
        let tw: Vec<C64> = (0..len * len).map(|i| C64::from_polar(1.0, sign * 2.0 * PI * ((i / len) * (i % len)) as f64 / len as f64)).collect();
        let total = v.len();
        let mut out = vec![C64::new(0.0, 0.0); total];
        for base in 0..total {
            if (base / stride) % len != 0 { continue; }
            for kk in 0..len { let mut acc = C64::new(0.0, 0.0); for t in 0..len { acc += v[base + t * stride] * tw[kk * len + t]; } out[base + kk * stride] = acc; }
        }
        *v = out;
    };
    let mut xf = x.to_vec();
    for a in 0..3 { dft(&mut xf, a, -1.0); }
    let kap = |i: usize, len: usize| -> f64 { let f = if i <= len / 2 { i as f64 } else { i as f64 - len as f64 }; 2.0 * PI * f / len as f64 };
    let nv = x.len();
    (0..edges.len() - 1).map(|b| {
        let mut v: Vec<C64> = (0..nv).map(|q| {
            let (i, j, k) = (q % n[0], (q / n[0]) % n[1], q / (n[0] * n[1]));
            let r = (kap(i, n[0]).powi(2) + kap(j, n[1]).powi(2) + kap(k, n[2]).powi(2)).sqrt();
            if r >= edges[b] && r < edges[b + 1] { xf[q] } else { C64::new(0.0, 0.0) }
        }).collect();
        for a in 0..3 { dft(&mut v, a, 1.0); }
        v.iter_mut().for_each(|z| *z /= nv as f64);
        v
    }).collect()
}

fn half(v: f64) -> f64 {
    // round to IEEE half precision (10 mantissa bits), normal range only
    if v == 0.0 { return 0.0; }
    let e = v.abs().log2().floor();
    let q = 2f64.powf(e - 10.0);
    (v / q).round() * q
}

#[test]
#[ignore = "E2, CPU only"]
fn e2_weighted_chirp_polynomial_cpu() {
    let only: Option<f64> = std::env::var("E2_B0").ok().and_then(|v| v.parse().ok());
    let qonly = std::env::var("E2_QONLY").is_ok();
    let eps: f64 = std::env::var("E2_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-6);
    eprintln!("LS ridge toward Taylor: eps = {eps:.0e} (rad^2 units)");
    for b0 in [2.89, 1.5, 0.55] {
        if only.is_some_and(|o| (o - b0).abs() > 1e-6) { continue; }
        let mut spec = Spec::full(1);
        spec.b0 = b0;
        let p = problem(&spec);
        let nv = p.sd.nv();
        let na = p.sd.na;
        eprintln!("\n=== E2 {}", spec.label());
        let vox = &p.fitv;
        let nvx = vox.len();
        let insup: Vec<bool> = { let mut b = vec![false; nv]; p.support.iter().for_each(|&q| b[q] = true); b };
        let ce: Vec<f64> = (0..nv).map(|q| (0..NC).map(|c| { let z = p.smaps_op[c * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect();
        // images: white (the paper's), smooth object (dim body + bright off-centre blob, smooth phase)
        let x_w = p.x.clone();
        let blob = |q: usize, dx: f64| -> f64 { let r = &p.rho[q]; let d2 = (r[0] - 0.04 - dx).powi(2) + (r[1] + 0.02).powi(2) + (r[2] - 0.01).powi(2); (-d2 / (2.0 * 0.03f64.powi(2))).exp() };
        let x_o: Vec<C64> = (0..nv).map(|q| if insup[q] { let r = &p.rho[q]; C64::from_polar(0.15 + blob(q, 0.0), 3.0 * r[0] / 0.1 + 2.0 * r[2] / 0.1) } else { C64::new(0.0, 0.0) }).collect();
        let en = |x: &dyn Fn(usize) -> f64| -> Vec<f64> { let a: Vec<f64> = (0..nv).map(|q| x(q)).collect(); let b = box3(&a); let v: Vec<f64> = vox.iter().map(|&q| b[q] * ce[q]).collect(); let m = v.iter().sum::<f64>() / v.len() as f64; v.into_iter().map(|e| e / m).collect() };
        let w_white = en(&|q| x_w[q].norm_sqr());
        let w_unif: Vec<f64> = vec![1.0; nvx];
        let w_obj = en(&|q| x_o[q].norm_sqr());
        let w_box = en(&|q| if insup[q] { 1.0 } else { 0.0 });
        let w_shift = en(&|q| if insup[q] { (0.15 + blob(q, 0.024)).powi(2) } else { 0.0 });
        let w_blend: Vec<f64> = w_shift.iter().zip(&w_box).map(|(a, b)| 0.7 * a + 0.3 * b).collect();
        let smaps64: Vec<C64> = p.smaps_op.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
        // weighted (energy) monomial fits, white image weight
        let (wfit2, _) = fit_w(&p, vox, &w_white, false);
        let (wfit4q, wfit4h) = fit_w(&p, vox, &w_white, true);
        let wfit4 = MonoFitHi { quad: MonoFit { theta: wfit4q.theta, rms: [0.0; 4], max: [1.0; 4] }, hi: wfit4h };
        for s in 0..p.sets.len() {
            let set = &p.sets[s];
            let m = set.tau.len();
            let y = Rng::new(99 + s as u64).complex_vec(NC * m);
            let nseg = 20;
            let t = tables(&p, s, nseg, 0);
            let (t3, dw32, fit4) = tables3(&p, s, nseg, 0);
            let wbar = mean_coefficients(&[set], na);
            let tab_w2 = seg_tables(&p.sd, set, &wfit2, true, Some(&wbar));
            let (tab_w4, _) = seg_tables_hi(&p.sd, set, &wfit4, true, Some(&wbar));
            let p64: Vec<C64> = t.segph.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
            let a64: Vec<C64> = t.atab.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
            let at: Vec<C64> = t.tab.tau.iter().flat_map(|&tau| interp_table(&a64, nseg, t.t0, t.dt, tau as f64)).collect();
            let sj = |j: usize, q: usize| -> C64 { (0..nseg).map(|l| at[j * nseg + l] * p64[l * nv + q]).sum() };
            let dw: Vec<[f64; 4]> = (0..m).map(|j| [0, 1, 2, 3].map(|q| set.wq[j][q] - wbar[j % na][q])).collect();
            for j in (0..m).step_by(97) { for q in 0..4 { assert!((dw32[q * m + j] as f64 - dw[j][q]).abs() <= 1e-6 * dw[j][q].abs().max(1e-3)); } }
            let lin = |tb: &SegTables, j: usize, q: usize| (0..3).map(|a| tb.coords[a][j] as f64 * p.idx[q][a]).sum::<f64>() + tb.phi0[j] as f64;
            let psi2 = |tb: &SegTables, j: usize, q: usize| { let (r, dc) = (&p.rho[q], |i: usize| tb.dc[i * m + j] as f64); dc(0) * r[0] * r[0] + dc(1) * r[1] * r[1] + dc(2) * r[2] * r[2] + 2.0 * (dc(3) * r[0] * r[1] + dc(4) * r[0] * r[2] + dc(5) * r[1] * r[2]) };
            let k0 = |j: usize, q: usize| (0..3).map(|a| set.coords[a][j] as f64 * p.idx[q][a]).sum::<f64>();
            let dev = |j: usize, q: usize| (0..4).map(|i| dw[j][i] * p.grid.mono[q][i]).sum::<f64>();
            let ds = DirectSum { nv, nc: NC, m, frames: 1, smaps: &smaps64, voxels: &p.fitv, bw: None };
            // references (white and object image)
            let rec = &p.recs[s];
            let smp: Vec<[f64; 8]> = rec.chunks(SAMPLE_FLOATS).map(|r| [r[0], r[1], r[2], r[9], r[10], r[11], r[12], r[13]].map(|v| v as f64)).collect();
            let fref = |j: usize, q: usize| { let (k, u, mq) = (&smp[j], &p.u_phys[q], &p.mono[q]); C64::from_polar(1.0, -(k[0] * u[0] + k[1] * u[1] + k[2] * u[2] + k[3] * mq[0] + k[4] * mq[1] + k[5] * mq[2] + k[6] * mq[3] + p.omega[q] * k[7])) };
            let dsr = DirectSum { nv, nc: NC, m, frames: 1, smaps: &p.smaps, voxels: &p.fitv, bw: None };
            let (yref_w, xref) = dsr.apply(&fref, &x_w, &y);
            let (yref_o, _) = dsr.apply(&fref, &x_o, &y);
            // family: (tables, number of basis functions, basis(j, q) -> values, with c34?)
            // fam 2: [1, ψ, ψ²] of the quadratic fit; fam 4: [1, ψ', ψ'², c34, ψ'·c34] of the degree-4 fit
            struct Fam<'a> { tab: &'a SegTables, c34: Option<&'a MonoFitHi> }
            let basis = |f: &Fam, j: usize, q: usize| -> [f64; 5] {
                let ps = psi2(f.tab, j, q);
                let c = f.c34.map_or(0.0, |h| h.hi_phase(&dw[j], &p.rho[q]));
                [1.0, ps, ps * ps, c, ps * c]
            };
            let target = |f: &Fam, j: usize, q: usize| -> C64 { C64::from_polar(1.0, -(k0(j, q) + dev(j, q) - lin(f.tab, j, q))) };
            // coefficients per sample for a method
            #[derive(Clone, Copy, PartialEq)]
            enum Mth { Taylor, CTaylor, Cheb, LsChi, LsFull, LsFixC }
            let coeffs = |f: &Fam, nb: usize, mth: Mth, w: &[f64]| -> Vec<C64> {
                let mut out = vec![C64::new(0.0, 0.0); m * 5];
                out.par_chunks_mut(5).enumerate().for_each(|(j, o)| {
                    let i = C64::new(0.0, 1.0);
                    let one = C64::new(1.0, 0.0);
                    match mth {
                        Mth::Taylor => {
                            let tc = [one, -i, C64::new(-0.5, 0.0), -i, -one];
                            for b in 0..nb { o[b] = tc[b]; }
                            if nb == 2 { o[1] = -i; }
                            if nb == 4 { o[3] = -i; }
                        }
                        Mth::CTaylor | Mth::Cheb => {
                            // quadratic families only (nb ≤ 3): polynomial in ψ
                            let (mut sw, mut s1, mut lo, mut hi) = (0.0, 0.0, f64::MAX, f64::MIN);
                            for (k, &q) in vox.iter().enumerate() { let v = psi2(f.tab, j, q); sw += w[k]; s1 += w[k] * v; lo = lo.min(v); hi = hi.max(v); }
                            let pp = nb - 1;
                            if mth == Mth::CTaylor {
                                let c = s1 / sw;
                                let e = C64::from_polar(1.0, -c);
                                // e^{-ic} T_P(s - c) expanded in s
                                if pp == 1 { o[0] = e * (one + i * c); o[1] = e * (-i); } else { o[0] = e * (one + i * c - 0.5 * c * c); o[1] = e * (-i + c); o[2] = e * (-0.5); }
                            } else {
                                let (cc, h) = (0.5 * (lo + hi), (0.5 * (hi - lo)).max(1e-9));
                                let nodes: Vec<f64> = (0..=pp).map(|k| cc + h * ((2 * k + 1) as f64 * PI / (2 * (pp + 1)) as f64).cos()).collect();
                                let bas: Vec<f64> = nodes.iter().flat_map(|&s| (0..=pp).map(move |n| s.powi(n as i32))).collect();
                                let tg: Vec<C64> = nodes.iter().map(|&s| C64::from_polar(1.0, -s)).collect();
                                let c = wls(pp + 1, &bas, &tg, &vec![1.0; pp + 1]);
                                for b in 0..=pp { o[b] = c[b]; }
                            }
                        }
                        Mth::LsChi | Mth::LsFull | Mth::LsFixC => {
                            let mut bas = Vec::with_capacity(nvx * nb);
                            let mut tg = Vec::with_capacity(nvx);
                            for &q in vox.iter() {
                                let bv = basis(f, j, q);
                                let t = if mth == Mth::LsChi { C64::from_polar(1.0, -(bv[1] + bv[3])) } else { target(f, j, q) };
                                if mth == Mth::LsFixC { bas.extend_from_slice(&bv[..3]); tg.push(t + i * bv[3]); } else { bas.extend_from_slice(&bv[..nb]); tg.push(t); }
                            }
                            let tay = [one, -i, C64::new(-0.5, 0.0), -i, -one];
                            let mut c0: Vec<C64> = tay[..nb].to_vec();
                            if nb == 2 { c0[1] = -i; }
                            if mth == Mth::LsFixC { let c = wls_toward(3, &bas, &tg, w, &tay[..3], eps); o[..3].copy_from_slice(&c); o[3] = -i; } else { let c = wls_toward(nb, &bas, &tg, w, &c0, eps); o[..nb].copy_from_slice(&c); }
                        }
                    }
                });
                out
            };
            let run = |f: &Fam, nb: usize, cf: &[C64], x: &[C64], want_adj: bool| -> (Vec<C64>, Vec<C64>) {
                let fac = |j: usize, q: usize| -> C64 {
                    let bv = basis(f, j, q);
                    let pv: C64 = (0..nb).map(|b| cf[j * 5 + b] * bv[b]).sum();
                    C64::from_polar(1.0, -lin(f.tab, j, q)) * sj(j, q) * pv
                };
                let (ye, xe) = ds.apply(&fac, x, if want_adj { &y } else { &y });
                (ye, xe)
            };
            let f2 = Fam { tab: &t.tab, c34: None };
            let f2w = Fam { tab: &tab_w2, c34: None };
            let f4 = Fam { tab: &t3.tab, c34: Some(&fit4) };
            let f4w = Fam { tab: &tab_w4, c34: Some(&wfit4) };
            eprintln!("  set {s} ({m} samples)  rows: method | white fwd  adj | object fwd with weight: oracle  box  shifted-24mm  blend(0.7 shift + 0.3 box)");
            let row = |name: &str, f: &Fam, nb: usize, mth: Mth, ww: &[f64], obj_ws: &[&[f64]]| {
                let keep = name == "P2 LS energy (full target)" || name == "O3 LS energy [1,psi,psi^2,c]";
                if qonly && !keep { return Vec::new(); }
                let obj_ws: &[&[f64]] = if qonly { &[] } else { obj_ws };
                let cf = coeffs(f, nb, mth, ww);
                let (yw, xw) = run(f, nb, &cf, &x_w, true);
                let mut line = format!("    {name:<34} | {:.2e} {:.2e} |", rel_on(&yw, &yref_w, None), rel_on(&xw, &xref, Some(&p.fitv)));
                for wo in obj_ws {
                    let cfo = if mth == Mth::Taylor { cf.clone() } else { coeffs(f, nb, mth, wo) };
                    let (yo, _) = run(f, nb, &cfo, &x_o, false);
                    line += &format!(" {:.2e}", rel_on(&yo, &yref_o, None));
                }
                eprintln!("{line}");
                cf
            };
            let objw: [&[f64]; 4] = [&w_obj, &w_box, &w_shift, &w_blend];
            let obj1: [&[f64]; 1] = [&w_obj];
            eprintln!("   degree-2 fit (order 1 / 2 kernel):");
            row("P1 Taylor", &f2, 2, Mth::Taylor, &w_white, &obj1);
            row("P1 LS energy (full target)", &f2, 2, Mth::LsFull, &w_white, &objw);
            row("P2 Taylor (production)", &f2, 3, Mth::Taylor, &w_white, &obj1);
            row("P2 centred Taylor (energy mean)", &f2, 3, Mth::CTaylor, &w_white, &obj1);
            row("P2 Chebyshev (range over fitv)", &f2, 3, Mth::Cheb, &w_white, &obj1);
            row("P2 LS uniform fitv (chirp target)", &f2, 3, Mth::LsChi, &w_unif, &obj1);
            row("P2 LS energy (chirp target)", &f2, 3, Mth::LsChi, &w_white, &obj1);
            let cf_main = row("P2 LS uniform fitv (full target)", &f2, 3, Mth::LsFull, &w_unif, &obj1);
            let _ = cf_main;
            let cf_e = row("P2 LS energy (full target)", &f2, 3, Mth::LsFull, &w_white, &objw);
            row("P2 LS energy + weighted fit", &f2w, 3, Mth::LsFull, &w_white, &obj1);
            // coefficient statistics in kernel form: p = p0 (1 + q·(−χ)... ) → w = p0[φ + q Dφ + r D²φ], q = −p1/p0, r = p2/p0
            {
                let (mut a0, mut ph, mut qr, mut qi, mut rr, mut ri) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for j in 0..m {
                    let c = &cf_e[j * 5..j * 5 + 3];
                    let q = -c[1] / c[0];
                    let r = c[2] / c[0];
                    a0 = a0.max((c[0].norm() - 1.0).abs()); ph = ph.max(c[0].arg().abs());
                    qr = qr.max(q.re.abs()); qi = qi.max((q.im - 1.0).abs()); rr = rr.max((r.re + 0.5).abs()); ri = ri.max(r.im.abs());
                }
                let (mut d0, mut d1, mut d2) = (0.0f64, 0.0f64, 0.0f64);
                for j in 0..m { let c = &cf_e[j * 5..j * 5 + 3]; d0 = d0.max((c[0] - 1.0).norm()); d1 = d1.max((c[1] - C64::new(0.0, -1.0)).norm()); d2 = d2.max((c[2] + 0.5).norm()); }
                eprintln!("    P2 LS energy: max |p_n - Taylor_n| n = 0, 1, 2: {d0:.2e} {d1:.2e} {d2:.2e}");
                eprintln!("    P2 LS energy coefficients (max over samples): ||p0|-1| {a0:.2e}, |arg p0| {ph:.2e} rad, |Re q| {qr:.2e}, |Im q - 1| {qi:.2e}, |Re r + 1/2| {rr:.2e}, |Im r| {ri:.2e}  (Taylor: q = i, r = -1/2)");
                // kernel format: 8 halves per sample, deviations from Taylor: δp0 = c0 − 1, δq = −c1 − i, δρ2 = −2c2 − 1
                let cq = quant16(&cf_e, 3);
                let (yw, xw) = run(&f2, 3, &cq, &x_w, true);
                eprintln!("    {:<34} | {:.2e} {:.2e} |", "P2 LS energy, f16 kernel params", rel_on(&yw, &yref_w, None), rel_on(&xw, &xref, Some(&p.fitv)));
                // coefficients computed every 8th ADC sample of each readout, linear in between
                let cfr = &cf_e;
                let ci: Vec<C64> = (0..m).flat_map(|j| {
                    let (r0, n) = (j / na * na, j % na);
                    let (a, b) = ((n / 8) * 8, ((n / 8) * 8 + 8).min(na - 1));
                    let fr = if b > a { (n - a) as f64 / (b - a) as f64 } else { 0.0 };
                    (0..5).map(move |k| (1.0 - fr) * cfr[(r0 + a) * 5 + k] + fr * cfr[(r0 + b) * 5 + k]).collect::<Vec<_>>()
                }).collect();
                let (yw, xw) = run(&f2, 3, &ci, &x_w, true);
                eprintln!("    {:<34} | {:.2e} {:.2e} |", "P2 LS energy, coeffs every 8th ADC", rel_on(&yw, &yref_w, None), rel_on(&xw, &xref, Some(&p.fitv)));
            }
            eprintln!("   degree-4 fit (order 3 kernel):");
            row("O3 Taylor (production)", &f4, 4, Mth::Taylor, &w_white, &obj1);
            row("O3 Taylor + psi*c34 product", &f4, 5, Mth::Taylor, &w_white, &obj1);
            row("O3 LS energy, c-term fixed -i", &f4, 4, Mth::LsFixC, &w_white, &obj1);
            // spectral-local energy weight (§4 of the note): sample j uses the smoothed energy of the image band around |k_j|
            {
                let edges = [0.0, 0.35, 0.8, 1.5, 10.0];
                let xb = bands(&x_o, &edges);
                let wb: Vec<Vec<f64>> = xb.iter().map(|v| en(&|q| v[q].norm_sqr())).collect();
                let band_of: Vec<usize> = (0..m).map(|j| {
                    let r = (0..3).map(|a| { let k = (set.coords[a][j] as f64 + PI).rem_euclid(2.0 * PI) - PI; k * k }).sum::<f64>().sqrt();
                    (0..edges.len() - 1).find(|&b| r < edges[b + 1]).unwrap_or(edges.len() - 2)
                }).collect();
                let cnt: Vec<usize> = (0..edges.len() - 1).map(|b| band_of.iter().filter(|&&x| x == b).count()).collect();
                for (name, f, nb) in [("P2 LS band-energy (full target)", &f2, 3usize), ("O3 LS band-energy [1,psi,psi^2,c]", &f4, 4)] {
                    let mut cf = vec![C64::new(0.0, 0.0); m * 5];
                    for b in 0..edges.len() - 1 {
                        let c = coeffs(f, nb, Mth::LsFull, &wb[b]);
                        for j in 0..m { if band_of[j] == b { cf[j * 5..j * 5 + 5].copy_from_slice(&c[j * 5..j * 5 + 5]); } }
                    }
                    let (yo, _) = run(f, nb, &cf, &x_o, false);
                    let co = coeffs(f, nb, Mth::LsFull, &w_obj);
                    let (yo2, _) = run(f, nb, &co, &x_o, false);
                    let ct = coeffs(f, nb, Mth::Taylor, &w_obj);
                    let (yo3, _) = run(f, nb, &ct, &x_o, false);
                    eprintln!("    {name:<34} | object fwd: band {:.2e} vs |x|^2 {:.2e} vs Taylor {:.2e}  (samples per band {cnt:?})", rel_on(&yo, &yref_o, None), rel_on(&yo2, &yref_o, None), rel_on(&yo3, &yref_o, None));
                }
            }
            // Proposition 2 check: does the incoherent proxy ε(θ; w_obj) = Σ_j Σ_q w_obj |E_j|² rank the coefficient sets
            // like the measured object forward error?
            if std::env::var("E2_PROXY").is_ok() {
                for (fname, f, nb) in [("P2", &f2, 3usize), ("O3", &f4, 4)] {
                    let mut out = format!("    proxy check {fname}: (proxy rel, measured rel) vs Taylor:");
                    let sets: Vec<(&str, Vec<C64>)> = vec![
                        ("Taylor", coeffs(f, nb, Mth::Taylor, &w_obj)),
                        ("LS oracle", coeffs(f, nb, Mth::LsFull, &w_obj)),
                        ("LS box", coeffs(f, nb, Mth::LsFull, &w_box)),
                        ("LS shift", coeffs(f, nb, Mth::LsFull, &w_shift)),
                    ];
                    let mut base = (0.0, 0.0);
                    for (k, (nm, cf)) in sets.iter().enumerate() {
                        let pr: f64 = (0..m).into_par_iter().map(|j| vox.iter().enumerate().map(|(ii, &q)| {
                            let bv = basis(f, j, q);
                            let pv: C64 = (0..nb).map(|b| cf[j * 5 + b] * bv[b]).sum();
                            w_obj[ii] * (pv - target(f, j, q)).norm_sqr()
                        }).sum::<f64>()).sum();
                        let pr = pr.sqrt();
                        let (yo, _) = run(f, nb, cf, &x_o, false);
                        let me = rel_on(&yo, &yref_o, None);
                        if k == 0 { base = (pr, me); }
                        out += &format!(" {nm} {:.3} / {:.3};", pr / base.0, me / base.1);
                    }
                    eprintln!("{out}");
                }
            }
            let cf4 = row("O3 LS energy [1,psi,psi^2,c]", &f4, 4, Mth::LsFull, &w_white, &objw);
            {
                let cq = quant16(&cf4, 4);
                let (yw, xw) = run(&f4, 4, &cq, &x_w, true);
                eprintln!("    {:<34} | {:.2e} {:.2e} |", "O3 LS energy, f16 side record", rel_on(&yw, &yref_w, None), rel_on(&xw, &xref, Some(&p.fitv)));
            }
            if qonly { continue; }
            row("O3 LS energy [..,psi*c]", &f4, 5, Mth::LsFull, &w_white, &obj1);
            row("O3 LS energy [1,psi,psi^2,c] + wfit", &f4w, 4, Mth::LsFull, &w_white, &obj1);
            row("O3 LS uniform [1,psi,psi^2,c]", &f4, 4, Mth::LsFull, &w_unif, &obj1);
            eprintln!("   floors (no stencil truncation):");
            for (name, f) in [("deg-2 fit, exact e^{-i psi}", &f2), ("deg-4 fit, exact e^{-i(psi+c)}", &f4), ("exact deviation (segment floor)", &f2)] {
                let exact_dev = name.starts_with("exact dev");
                let fac = |j: usize, q: usize| -> C64 {
                    let bv = basis(f, j, q);
                    let pv = if exact_dev { target(f, j, q) } else { C64::from_polar(1.0, -(bv[1] + bv[3])) };
                    C64::from_polar(1.0, -lin(f.tab, j, q)) * sj(j, q) * pv
                };
                let (yw, xw) = ds.apply(&fac, &x_w, &y);
                let (yo, _) = ds.apply(&fac, &x_o, &y);
                eprintln!("    {name:<34} | {:.2e} {:.2e} | {:.2e}", rel_on(&yw, &yref_w, None), rel_on(&xw, &xref, Some(&p.fitv)), rel_on(&yo, &yref_o, None));
            }
        }
    }
}
