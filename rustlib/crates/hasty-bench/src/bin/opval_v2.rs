//! Operator validation on the v2 readout and the reconstruction-side GIRF refit (research/recon_physics_wiring.md
//! items 5–6). Works on a *case directory* written by research/experiments/recon_physics/case.py: the budget's truth
//! image (2 mm, ph0001), whitened true coil maps, the map field, per-frame fat fractions, complex-rate maps, and the
//! budget's 48 readouts (encode, frame, the rotations / encodes of their 10 previous TRs, the selected ADC samples of
//! the reconstruction window and of the rewinder navigator).
//!
//!   opval_v2 records <case> --chain SPEC --tag T [--hist 10]
//!       played records (SAMPLE_FLOATS) of every readout at its samples through chain SPEC (a Girf::variants() name,
//!       clinical, ideal, or a hasty-girf-1 .json): rec_T.npy [Σ samples, 16], recnav_T.npy (navigator samples).
//!   opval_v2 forward <case> --tag T [--model SPEC] [--theta v,..] [--params delay,b0,res,gain] [--hist 10]
//!            [--fat union|mean|none] [--rate none|fit|wt|tab] [--conc full|mean|off] [--L n] [--nav] [--coils n]
//!       the full operator (SegNufft, svdn / union segments per encode, per-sample concomitant ΔC, B0-term phase)
//!       applied to the case image per readout: y_T.npy [Σ samples, nc] complex64 (sample-major).
//!   opval_v2 refit <case> --data y.npy --tag T [operator options] [--iters 4] [--noise σ] [--seed s] [--lambda 1e-3]
//!            [--readouts n]
//!       Gauss–Newton refit of θ (image fixed = the case image) against data y.npy (+ white noise σ per whitened
//!       coil sample): per iteration θ, the data misfit R, CRB, timings → refit_T.json.
//!   opval_v2 jac <case> [operator options] [--readouts 6]
//!       Jacobian check: J e_p (k-gradient + sensitivities) vs central differences of the operator in θ_p.
//!   opval_v2 cost <case> [operator options] [--set-readouts 300] [--coils 16] [--reps 5]
//!       production-size timing of one (frame, encode) set: gradient pass (forward + adjoint) vs refit pass
//!       (residual + three k-gradient forwards + coil reduction + accumulation) vs the CPU sensitivity tables.
//! GPU memory: the engine's grid budget is 2 GiB (HASTY_OPVAL_GRID_GB), maps and tables ≈ 1–2 GiB more.

use std::path::{Path, PathBuf};
use std::time::Instant;

use cudarc::driver::CudaSlice;
use hasty_cuda::{Device, C32};
use hasty_nufft::{SegNufft, SegNufftOpts, SegSegments};
#[allow(unused_imports)]
use rayon::prelude::*;
use hasty_recon::data::PhysicsGrid;
use hasty_recon::girf_refit::{sensitivities, to_operator_units, ChainModel, Normal, Param, Reducer, RefitOpts};
use hasty_recon::played::{FlowDesign, ModelTraj};
use hasty_recon::segphys::{absorb_field_poly_at, fit_field_poly_at, segment_phase, FieldPoly, SegTables};
use hasty_seq::girf::GirfMeasurement;
use hasty_seq::Mat3;
use num_complex::Complex64;
use serde_json::{json, Value};

const SF: usize = hasty_mrsim::sequence::SAMPLE_FLOATS;

fn npy_write(path: &Path, descr: &str, shape: &[usize], bytes: &[u8]) {
    hasty_phantom::volume::write_npy(path, descr, shape, false, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn c32b(v: &[C32]) -> Vec<u8> {
    v.iter().flat_map(|z| [z.re.to_le_bytes(), z.im.to_le_bytes()].concat()).collect()
}

struct Ro {
    enc: usize,
    frame: usize,
    seq: Vec<(usize, Mat3)>,
    /// ADC indices (from the echo) of the reconstruction samples and of the navigator samples
    n: Vec<usize>,
    nav: Vec<usize>,
}

struct Case {
    dir: PathBuf,
    dims: [usize; 3],
    vox: f64,
    nc: usize,
    sigma: f64,
    b0: f64,
    dwell: f64,
    design: FlowDesign,
    ros: Vec<Ro>,
    /// ADC samples of the reconstruction window
    na_win: usize,
    /// the sim's scan directory (coil maps, Ψ, field, fat fractions)
    scan: PathBuf,
}

impl Case {
    fn load(dir: &Path) -> Self {
        let j: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("case.json")).expect("case.json")).unwrap();
        let u = |v: &Value| v.as_u64().unwrap() as usize;
        let ros = j["readouts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let he: Vec<usize> = r["hist_enc"].as_array().unwrap().iter().map(u).collect();
                let hr: Vec<Mat3> = r["hist_rot"].as_array().unwrap().iter().map(|m| [0, 1, 2].map(|a| [0, 1, 2].map(|b| m[a][b].as_f64().unwrap()))).collect();
                Ro { enc: u(&r["enc"]), frame: u(&r["frame"]), seq: he.into_iter().zip(hr).collect(), n: r["n"].as_array().unwrap().iter().map(u).collect(), nav: r["nav"].as_array().unwrap().iter().map(u).collect() }
            })
            .collect();
        let design = FlowDesign::load(Path::new(j["design"].as_str().unwrap())).unwrap();
        Self {
            dir: dir.to_path_buf(),
            dims: [0, 1, 2].map(|a| u(&j["dims"][a])),
            vox: j["vox"].as_f64().unwrap(),
            nc: u(&j["nc"]),
            sigma: j["sigma"].as_f64().unwrap(),
            b0: j["b0"].as_f64().unwrap(),
            dwell: j["dwell"].as_f64().unwrap(),
            design,
            ros,
            na_win: u(&j["na_win"]),
            scan: PathBuf::from(j["scan"].as_str().unwrap()),
        }
    }
    fn nv(&self) -> usize {
        self.dims.iter().product()
    }
    fn times(&self, n: &[usize]) -> Vec<f64> {
        n.iter().map(|&i| self.design.i_ro as f64 * self.design.dt + i as f64 * self.dwell).collect()
    }
    fn c32(&self, f: &str) -> Vec<C32> {
        hasty_recon::data::read_npy(&self.dir.join(f)).unwrap_or_else(|e| panic!("{f}: {e:?}")).c32s()
    }
    fn f32(&self, f: &str) -> Vec<f32> {
        hasty_recon::data::read_npy(&self.dir.join(f)).unwrap_or_else(|e| panic!("{f}: {e:?}")).f32s()
    }
    /// whitened true coil maps (coil-major, Ψ^{-1/2} from the sim's meta, as research/experiments/readout_physics/pb.py)
    fn smaps(&self, nc_keep: usize) -> Vec<C32> {
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(self.scan.join("meta.json")).unwrap()).unwrap();
        let psi: Vec<Vec<f64>> = meta["psi"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect();
        let n = psi.len();
        let e = nalgebra::SymmetricEigen::new(nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| psi[i][j]));
        let w: Vec<f64> = (0..n * n).map(|q| (0..n).map(|k| e.eigenvectors[(q / n, k)] * e.eigenvectors[(q % n, k)] / e.eigenvalues[k].max(1e-12).sqrt()).sum()).collect();
        let raw = hasty_recon::data::read_npy(&self.scan.join("smaps_true.npy")).unwrap().c32s();
        let nv = self.nv();
        assert_eq!(raw.len(), n * nv);
        use rayon::prelude::*;
        let mut out = vec![C32::new(0.0, 0.0); nc_keep * nv];
        out.par_chunks_mut(nv).enumerate().for_each(|(c, row)| {
            for q in 0..nv {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for c2 in 0..n {
                    let s = raw[c2 * nv + q];
                    re += w[c * n + c2] * s.re as f64;
                    im += w[c * n + c2] * s.im as f64;
                }
                row[q] = C32::new(re as f32, im as f32);
            }
        });
        out
    }
    /// the map field ω̄ (rad/s) of the sim
    fn field(&self) -> Vec<f32> {
        let ppm = hasty_recon::data::read_npy(&self.scan.join("field_ppm_recon.npy")).unwrap().f32s();
        let wf = 2.0 * std::f64::consts::PI * 42.577478e6 * self.b0 * 1e-6;
        ppm.iter().map(|&v| if v.is_finite() { (v as f64 * wf) as f32 } else { 0.0 }).collect()
    }
    /// truth fat fraction per cardiac frame (frames × nv, NaN → 0, clipped to [0, 1])
    fn ff(&self) -> Vec<f32> {
        hasty_recon::data::read_npy(&self.scan.join("truth_fat_fraction.npy")).unwrap().f32s().iter().map(|&v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }).collect()
    }
}

/// command-line options of the operator
#[derive(Clone)]
struct OpOpts {
    model: String,
    theta: Vec<f64>,
    params: RefitOpts,
    hist: usize,
    fat: String,
    rate: String,
    conc: String,
    nseg: usize,
    nav: bool,
    coils: usize,
}

fn op_opts(args: &mut Vec<String>) -> OpOpts {
    let mut take = |k: &str| args.iter().position(|a| a == k).map(|i| { args.remove(i); args.remove(i) });
    let params_s = take("--params").unwrap_or("delay,b0,res".into());
    let params = RefitOpts { delay: params_s.contains("delay"), gain: params_s.contains("gain"), b0amp: params_s.contains("b0"), res: params_s.contains("res") };
    let o = OpOpts {
        model: take("--model").unwrap_or("ideal".into()),
        theta: take("--theta").map(|s| s.split(',').map(|v| v.parse().unwrap()).collect()).unwrap_or_default(),
        params,
        hist: take("--hist").map(|s| s.parse().unwrap()).unwrap_or(10),
        fat: take("--fat").unwrap_or("union".into()),
        rate: take("--rate").unwrap_or("fit".into()),
        conc: take("--conc").unwrap_or("full".into()),
        nseg: take("--L").map(|s| s.parse().unwrap()).unwrap_or(15),
        nav: false,
        coils: take("--coils").map(|s| s.parse().unwrap()).unwrap_or(0),
    };
    let nav = args.iter().position(|a| a == "--nav").map(|i| { args.remove(i); true }).unwrap_or(false);
    OpOpts { nav, ..o }
}

/// the model trajectory: a chain (variant / ideal / json without θ) or a ChainModel at θ (json + --theta / refit)
enum Traj {
    Chain(hasty_seq::girf::PlayedStream),
    Model(ChainModel),
}

fn model_traj(case: &Case, o: &OpOpts) -> Traj {
    if o.model.ends_with(".json") && (!o.theta.is_empty() || o.params.delay || o.params.res) {
        let meas = GirfMeasurement::load(Path::new(&o.model)).unwrap();
        Traj::Model(ChainModel::new(meas, o.params))
    } else {
        let g = ModelTraj::girf_of(&o.model).unwrap();
        Traj::Chain(case.design.stream(&g, o.hist))
    }
}

/// per readout: records at (recon or navigator) samples and, for a ChainModel, the sensitivities
fn records(case: &Case, tr: &Traj, theta: &[f64], hist: usize, nav: bool, with_sens: bool) -> Vec<(Vec<f32>, Option<Vec<Vec<[f64; 4]>>>)> {
    let seqs: Vec<Vec<(usize, Mat3)>> = case.ros.iter().map(|r| r.seq[r.seq.len().saturating_sub(hist + 1)..].to_vec()).collect();
    let times: Vec<Vec<f64>> = case.ros.iter().map(|r| case.times(if nav { &r.nav } else { &r.n })).collect();
    match tr {
        Traj::Chain(ps) => hasty_seq::par_map(&(0..case.ros.len()).collect::<Vec<_>>(), |&i| {
            let pl = ps.moments(&seqs[i], case.design.t_rf, &times[i]);
            let mut rec = Vec::with_capacity(pl.len() * SF);
            for (m, &t) in pl.iter().zip(&times[i]) {
                let mut r = hasty_mrsim::sequence::sample_record(&m.m0, &m.m1, &m.m2, &m.p, case.b0, t - case.design.t_rf);
                r[15] = m.phi_b0 as f32;
                rec.extend_from_slice(&r);
            }
            (rec, None)
        }),
        Traj::Model(cm) => {
            let th = if theta.is_empty() { vec![0.0; cm.n()] } else { theta.to_vec() };
            sensitivities(cm, &th, &case.design, hist, case.b0, &seqs, &times).into_iter().map(|s| (s.rec, with_sens.then_some(s.d))).collect()
        }
    }
}

/// the engine and everything that does not depend on the trajectory
struct Engine {
    eng: SegNufft,
    /// segments per encode, host copies (P, a, fat c_l, t0, dt): one encode on the device at a time (≤ 6 GB)
    segs: Vec<(Vec<C32>, Vec<C32>, Option<Vec<C32>>, f64, f64)>,
    cur: Option<(usize, SegSegments)>,
    maps: Option<hasty_nufft::SegFatMaps>,
    x: CudaSlice<C32>,
    grid: PhysicsGrid,
    fit: [[f64; 10]; 4],
    poly: FieldPoly,
    wbar: Vec<Vec<[f64; 4]>>,
    tau_win: Vec<f64>,
    nc: usize,
    union: bool,
    with_conc: bool,
    order: u32,
}

fn voxel_log(dims: [usize; 3], vox: f64, q: usize) -> [f64; 3] {
    let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
    [(i as f64 - (dims[0] / 2) as f64) * vox, (j as f64 - (dims[1] / 2) as f64) * vox, (k as f64 - (dims[2] / 2) as f64) * vox]
}

fn fat_c(t: f64, b0: f64) -> Complex64 {
    let a = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048];
    let s: f64 = a.iter().sum();
    let ppm = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60];
    (0..6).map(|p| Complex64::from_polar(a[p] / s, -2.0 * std::f64::consts::PI * hasty_mrsim::GAMMA_BAR * b0 * ppm[p] * 1e-6 * t)).sum()
}

/// build the engine: segments per encode from the encode-mean concomitant coefficients of `full_recs` (records of
/// every readout at every window sample), noise time weights from the exact data energy (energy.npy), fit voxels =
/// every 3rd support voxel with energy weights |x|² Σ|S|²
fn engine(dev: &Device, case: &Case, o: &OpOpts, full_recs: &[Vec<f32>]) -> Engine {
    engine_opt(Some(dev), case, o, full_recs, None).unwrap()
}

/// `dev` None: the CPU part only (segment build, timed; returns None); `only_enc`: build one encode's segments
fn engine_opt(dev: Option<&Device>, case: &Case, o: &OpOpts, full_recs: &[Vec<f32>], only_enc: Option<usize>) -> Option<Engine> {
    let (dims, nv) = (case.dims, case.nv());
    let nc_all = case.nc;
    let nc = if o.coils > 0 { o.coils.min(nc_all) } else { nc_all };
    let smaps: Vec<C32> = case.smaps(nc);
    let x = case.c32("x.npy");
    let field = case.field();
    let support: Vec<usize> = case.f32("support.npy").iter().enumerate().filter(|(_, v)| **v > 0.5).map(|(q, _)| q).collect();
    let rate: Option<Vec<C32>> = match o.rate.as_str() {
        "none" => None,
        r => Some(case.c32(&format!("z_{r}.npy"))),
    };
    let mut grid = PhysicsGrid {
        x_log: (0..nv).map(|q| voxel_log(dims, case.vox, q)).collect(),
        mono: vec![],
        omega: field.iter().map(|&v| v as f64).collect(),
        jac: vec![],
        smaps: smaps.clone(),
    };
    grid.mono = grid.x_log.iter().map(|p| [p[2] * p[2], p[0] * p[0] + p[1] * p[1], p[0] * p[2], p[1] * p[2]]).collect();
    if let Some(z) = &rate {
        for q in 0..nv {
            grid.omega[q] += z[q].im as f64;
        }
    }
    let r2: Vec<f64> = rate.as_ref().map_or(vec![0.0; nv], |z| z.iter().map(|c| c.re as f64).collect());
    let dwr: Vec<f64> = rate.as_ref().map_or(vec![0.0; nv], |z| z.iter().map(|c| c.im as f64).collect());
    let na = case.na_win;
    let tau_win: Vec<f64> = (0..na).map(|n| case.design.i_ro as f64 * case.design.dt + n as f64 * case.dwell - case.design.t_rf).collect();
    let (tw0, tw1) = if o.nav {
        let t: Vec<f64> = case.times(&case.ros[0].nav).iter().map(|t| t - case.design.t_rf).collect();
        (t[0], t[t.len() - 1])
    } else {
        (tau_win[0], tau_win[na - 1])
    };
    // interpolator table: every HASTY_OPVAL_NTAB_STEP-th window sample (default 4 = 8 µs at 2 µs dwell; the linear
    // interpolation error of the B0 phase over 8 µs is ≈ (2π·300 Hz·8 µs)²/8 ≈ 3e-5; the LS interpolator fit is the
    // dominant CPU cost of the union build and scales with the table length)
    let nstep: usize = std::env::var("HASTY_OPVAL_NTAB_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    let ntab = if o.nav { 1024 } else { ((na - 1) / nstep + 1).max(256) };
    // encode-mean concomitant coefficients per window sample
    let ne = case.design.n_enc();
    let with_conc = o.conc != "off";
    let mut wbar = vec![vec![[0.0f64; 4]; na]; ne];
    let mut cnt = vec![0usize; ne];
    if with_conc {
        for (r, rec) in case.ros.iter().zip(full_recs) {
            cnt[r.enc] += 1;
            for n in 0..na {
                for q in 0..4 {
                    wbar[r.enc][n][q] += rec[n * SF + 9 + q] as f64;
                }
            }
        }
        for e in 0..ne {
            for n in 0..na {
                for q in 0..4 {
                    wbar[e][n][q] /= cnt[e].max(1) as f64;
                }
            }
        }
    }
    // quadratic fit of the monomials over the support (segphys::fit_monomials on logical positions)
    let basis = |p: &[f64; 3]| [1.0, p[0], p[1], p[2], p[0] * p[0], p[1] * p[1], p[2] * p[2], 2.0 * p[0] * p[1], 2.0 * p[0] * p[2], 2.0 * p[1] * p[2]];
    let mut ata = nalgebra::DMatrix::<f64>::zeros(10, 10);
    let mut atb = nalgebra::DMatrix::<f64>::zeros(10, 4);
    for &v in support.iter().step_by(5) {
        let b = basis(&grid.x_log[v]);
        for i in 0..10 {
            for k in 0..10 {
                ata[(i, k)] += b[i] * b[k];
            }
            for q in 0..4 {
                atb[(i, q)] += b[i] * grid.mono[v][q];
            }
        }
    }
    let sol = ata.cholesky().unwrap().solve(&atb);
    let fit: [[f64; 10]; 4] = std::array::from_fn(|q| std::array::from_fn(|i| sol[(i, q)]));
    // energy weights and the degree-1 field part (--b0-absorb 1)
    let coil_e: Vec<f64> = (0..nv).map(|q| (0..nc).map(|c| (smaps[c * nv + q].re as f64).powi(2) + (smaps[c * nv + q].im as f64).powi(2)).sum()).collect();
    // fit voxels: every HASTY_OPVAL_FIT_STEP-th support voxel (default 3, as recon_p2np's seg_fit)
    let fstep: usize = std::env::var("HASTY_OPVAL_FIT_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(3).max(1);
    let fit_vox: Vec<usize> = support.iter().copied().step_by(fstep).collect();
    let mut wv: Vec<f64> = fit_vox.iter().map(|&q| ((x[q].re as f64).powi(2) + (x[q].im as f64).powi(2)) * coil_e[q]).collect();
    let mut srt = wv.clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = srt[srt.len() / 2].max(1e-300);
    wv.iter_mut().for_each(|v| *v /= med);
    let poly = fit_field_poly_at(&|v| grid.x_log[v], &grid.omega, &fit_vox, Some(&wv), 1, 0.5 * (tw0 + tw1));
    // noise time weights 1 + (E(n) + E_vol) / (C σ²) from the exact data energy on the selected samples
    let tw_n: Vec<f64> = match hasty_recon::data::read_npy(&case.dir.join("energy.npy")) {
        Ok(e) if !o.nav => {
            let e = e.f64s();
            let k0: Vec<f64> = (0..na).map(|n| (0..3).map(|a| (full_recs[0][n * SF + a] as f64).powi(2)).sum::<f64>().sqrt()).collect();
            hasty_nufft::noise_time_weights(&e, &k0, nc_all as f64 * case.sigma * case.sigma)
        }
        _ => vec![1.0; na],
    };
    let tgrid: Vec<f64> = if o.nav { (0..256).map(|i| tw0 + (tw1 - tw0) * i as f64 / 255.0).collect() } else { tau_win.clone() };
    let twg: Vec<f64> = if o.nav { vec![1.0; 256] } else { tw_n };
    let (ts, tw) = hasty_nufft::seg_time_grid(&tgrid, &twg, 96);
    let ff_all = case.ff();
    let nfr = ff_all.len() / nv;
    let ffm: Vec<f64> = (0..nv).map(|q| (0..nfr).map(|f| ff_all[f * nv + q] as f64).sum::<f64>() / nfr as f64).collect();
    let order = if o.conc == "full" { 2 } else { 0 };
    let union = o.fat == "union";
    let max_grid = (std::env::var("HASTY_OPVAL_GRID_GB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(2.0) * (1u64 << 30) as f64) as usize;
    let b0 = case.b0;
    let cf = move |t: f64| fat_c(t, b0);
    let t0 = tau_win[0];
    let t_seg = Instant::now();
    let segs: Vec<(Vec<C32>, Vec<C32>, Option<Vec<C32>>, f64, f64)> = (0..ne)
        .map(|e| {
            if only_enc.is_some_and(|k| k != e) {
                return (vec![], vec![], None, 0.0, 0.0);
            }
            let wb: Vec<[f64; 4]> = if with_conc { wbar[e].clone() } else { vec![[0.0; 4]; na] };
            let psi_raw = segment_phase(&grid, &wb, &tau_win, true);
            // the rate's Δω relative to the echo: e^{−z (τ − TE)} (the static Δω TE taken out of ω τ)
            let psi = |v: usize, t: f64| poly.residual_at(&grid.x_log[v], psi_raw(v, t) - dwr[v] * tau_win[0], t);
            let dec = |v: usize, t: f64| (-r2[v] * (t - t0)).exp();
            match o.fat.as_str() {
                "union" => {
                    let fmin: Vec<f64> = fit_vox.iter().map(|&q| (0..nfr).map(|f| ff_all[f * nv + q] as f64).fold(1.0, f64::min)).collect();
                    let fmax: Vec<f64> = fit_vox.iter().map(|&q| (0..nfr).map(|f| ff_all[f * nv + q] as f64).fold(0.0, f64::max)).collect();
                    let ww: Vec<f64> = wv.iter().zip(&fmin).map(|(w, f)| w * (1.0 - f).powi(2) + 1e-12).collect();
                    let wf: Vec<f64> = wv.iter().zip(&fmax).map(|(w, f)| w * f * f + 1e-12).collect();
                    let g = |v: usize, t: f64| Complex64::from_polar(dec(v, t), -psi(v, t));
                    let (sp, at, cl, ta, dt, _nodes, cond) = hasty_nufft::build_segments_svd_fat_union_g(nv, o.nseg, tw0, tw1, ntab, &fit_vox, &ww, &wf, &ts, &tw, &g, &cf);
                    if e == 0 {
                        let ev = hasty_nufft::fat_union_spectrum_g(&fit_vox, &ww, &wf, &ts, &tw, &g, &cf);
                        let gain = (twg.iter().sum::<f64>() / twg.len() as f64).sqrt();
                        let tail = (ev[o.nseg.min(ev.len())..].iter().sum::<f64>() / ev.iter().sum::<f64>()).sqrt();
                        let lrule = (2..ev.len()).find(|&l| 1.5 * gain * (ev[l..].iter().sum::<f64>() / ev.iter().sum::<f64>()).sqrt() <= 0.2).unwrap_or(ev.len());
                        eprintln!("segments: union svdn L {} (condition {cond:.1}; union Gram tail {tail:.2e} × √mean w {gain:.1} = predicted R {:.3}; the rule (×1.5 ≤ 0.2) picks L {lrule})", o.nseg, tail * gain, );
                    }
                    (sp, at, Some(cl), ta, dt)
                }
                f => {
                    let g = |v: usize, t: f64| {
                        let z = Complex64::from_polar(dec(v, t), -psi(v, t));
                        if f == "mean" { z * (Complex64::new(1.0 - ffm[v], 0.0) + ffm[v] * cf(t)) } else { z }
                    };
                    let (sp, at, ta, dt) = hasty_nufft::build_segments_svd_grid(nv, o.nseg, tw0, tw1, ntab, &fit_vox, Some(&wv), &ts, &tw, &g);
                    (sp, at, None, ta, dt)
                }
            }
        })
        .collect();
    eprintln!("engine: {} encodes' segments built ({:.1}s, {} fit voxels, {} table entries)", ne, t_seg.elapsed().as_secs_f64(), fit_vox.len(), ntab);
    let dev = dev?;
    let st = dev.default_stream();
    let mut eng = SegNufft::new(dev, dims, [case.vox; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: o.nseg, order, frames: 1, bspline: 1, max_grid_bytes: max_grid }).unwrap();
    eng.multi_extra_bytes = 0;
    eng.set_maps(&st, &smaps).unwrap();
    let maps = union.then(|| eng.fat_maps(&st, &ff_all).unwrap());
    let xd = st.clone_htod(&x).unwrap();
    Some(Engine { eng, segs, cur: None, maps, x: xd, grid, fit, poly, wbar, tau_win, nc, union, with_conc, order })
}

impl Engine {
    /// the device segments of encode `enc` (uploaded on first use; the previous encode's are released)
    fn seg(&mut self, st: &std::sync::Arc<cudarc::driver::CudaStream>, enc: usize) {
        if self.cur.as_ref().is_some_and(|c| c.0 == enc) {
            return;
        }
        self.cur = None;
        let (sp, at, cl, ta, dt) = &self.segs[enc];
        let s = match cl {
            Some(cl) => self.eng.segments_fat(st, sp, at, cl, *ta, *dt).unwrap(),
            None => self.eng.segments(st, sp, at, *ta, *dt).unwrap(),
        };
        self.cur = Some((enc, s));
    }
}

/// SegNufft tables of one readout from its records (its samples `n`): coordinates κ = k Δ (logical = physical
/// axes here), τ, and the model trajectory's B0-term phase as φ0 (the operator applies e^{−iφ0}); ΔC zero
fn tables(case: &Case, rec: &[f32], n: &[usize]) -> SegTables {
    let m = n.len();
    let mut coords: [Vec<f32>; 3] = [vec![0.0; m], vec![0.0; m], vec![0.0; m]];
    let mut phi0 = vec![0f32; m];
    let mut tau = vec![0f32; m];
    for j in 0..m {
        let s = &rec[j * SF..(j + 1) * SF];
        for a in 0..3 {
            coords[a][j] = (s[a] as f64 * case.vox) as f32;
        }
        tau[j] = s[13];
        phi0[j] = s[15];
    }
    SegTables { coords, tau, phi0, dc: vec![0f32; 6 * m], t: vec![0.0; m], wbar: vec![], tau_adc: vec![], max_resid_phase: 0.0 }
}

/// per-sample concomitant deviation from the encode mean (φ0, k shift, ΔC) and the absorbed field part, in place
fn add_conc(e: &Engine, case: &Case, tab: &mut SegTables, rec: &[f32], n: &[usize], enc: usize, nav: bool) {
    let m = n.len();
    if e.with_conc {
        let wb = &e.wbar[enc];
        let na = wb.len();
        for j in 0..m {
            let s = &rec[j * SF..(j + 1) * SF];
            // the navigator lies beyond the window: its deviation is taken from the last window sample's mean
            let ni = if nav { na - 1 } else { n[j].min(na - 1) };
            let dw: [f64; 4] = [0, 1, 2, 3].map(|q| s[9 + q] as f64 - wb[ni][q]);
            let th = |i: usize| (0..4).map(|q| dw[q] * e.fit[q][i]).sum::<f64>();
            if e.order == 0 && !nav {
                // --conc mean: the segments' encode mean only (per-sample deviation omitted)
                continue;
            }
            tab.phi0[j] += th(0) as f32;
            for a in 0..3 {
                tab.coords[a][j] += (th(1 + a) * case.vox) as f32;
            }
            if e.order > 0 {
                for i in 0..6 {
                    tab.dc[i * m + j] = th(4 + i) as f32;
                }
            }
        }
    }
    absorb_field_poly_at(tab, [case.vox; 3], &e.poly, &mut std::iter::empty());
}

fn forward_one(e: &mut Engine, st: &std::sync::Arc<cudarc::driver::CudaStream>, r: &Ro, pts: &hasty_nufft::SegPoints, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) {
    e.seg(st, r.enc);
    let seg = &e.cur.as_ref().unwrap().1;
    if e.union {
        let maps = e.maps.as_ref().unwrap();
        e.eng.forward_fat(st, seg, pts, maps, Some(r.frame), x, y).unwrap();
    } else {
        e.eng.forward(st, seg, pts, x, y).unwrap();
    }
}

fn points_of(e: &mut Engine, case: &Case, st: &std::sync::Arc<cudarc::driver::CudaStream>, r: &Ro, rec: &[f32], nav: bool) -> hasty_nufft::SegPoints {
    points_shift(e, case, st, r, rec, nav, [0.0; 4])
}

/// [`points_of`] with a constant offset of the coordinates (operator units) and phase: the per-encode coherent
/// k offset / phase a per-encode image absorbs (refit option `enc`)
fn points_shift(e: &mut Engine, case: &Case, st: &std::sync::Arc<cudarc::driver::CudaStream>, r: &Ro, rec: &[f32], nav: bool, sh: [f64; 4]) -> hasty_nufft::SegPoints {
    let n = if nav { &r.nav } else { &r.n };
    let mut tab = tables(case, rec, n);
    add_conc(e, case, &mut tab, rec, n, r.enc, nav);
    if sh != [0.0; 4] {
        for j in 0..n.len() {
            for a in 0..3 {
                tab.coords[a][j] += sh[a] as f32;
            }
            tab.phi0[j] += sh[3] as f32;
        }
    }
    e.eng.points(st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, None).unwrap()
}

/// sample-major [m][nc] from the operator's coil-major [nc][m]
fn to_sample_major(y: &[C32], nc: usize, m: usize) -> Vec<C32> {
    let mut o = vec![C32::new(0.0, 0.0); nc * m];
    for c in 0..nc {
        for j in 0..m {
            o[j * nc + c] = y[c * m + j];
        }
    }
    o
}

fn full_window_records(case: &Case, tr: &Traj, theta: &[f64], hist: usize) -> Vec<Vec<f32>> {
    let na = case.na_win;
    let all: Vec<usize> = (0..na).collect();
    let seqs: Vec<Vec<(usize, Mat3)>> = case.ros.iter().map(|r| r.seq[r.seq.len().saturating_sub(hist + 1)..].to_vec()).collect();
    let times: Vec<Vec<f64>> = case.ros.iter().map(|_| case.times(&all)).collect();
    match tr {
        Traj::Chain(ps) => hasty_seq::par_map(&(0..case.ros.len()).collect::<Vec<_>>(), |&i| {
            let pl = ps.moments(&seqs[i], case.design.t_rf, &times[i]);
            pl.iter().zip(&times[i]).flat_map(|(m, &t)| {
                let mut r = hasty_mrsim::sequence::sample_record(&m.m0, &m.m1, &m.m2, &m.p, case.b0, t - case.design.t_rf);
                r[15] = m.phi_b0 as f32;
                r
            }).collect()
        }),
        Traj::Model(cm) => {
            let th = if theta.is_empty() { vec![0.0; cm.n()] } else { theta.to_vec() };
            sensitivities(cm, &th, &case.design, hist, case.b0, &seqs, &times).into_iter().map(|s| s.rec).collect()
        }
    }
}

fn main() {
    // thread pools ≤ 6 by default (shared workstation): rayon and hasty_seq::par_map
    let nt: usize = std::env::var("HASTY_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(nt).build_global();
    hasty_seq::set_threads(nt);
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.remove(0);
    let dir = PathBuf::from(args.remove(0));
    let case = Case::load(&dir);
    let take = |args: &mut Vec<String>, k: &str| args.iter().position(|a| a == k).map(|i| { args.remove(i); args.remove(i) });
    match mode.as_str() {
        "records" => {
            let chain = take(&mut args, "--chain").expect("--chain");
            let tag = take(&mut args, "--tag").expect("--tag");
            let hist: usize = take(&mut args, "--hist").map(|s| s.parse().unwrap()).unwrap_or(10);
            let t0 = Instant::now();
            let ps = case.design.stream(&ModelTraj::girf_of(&chain).unwrap(), hist);
            let tr = Traj::Chain(ps);
            for (nav, name) in [(false, "rec"), (true, "recnav")] {
                let r = records(&case, &tr, &[], hist, nav, false);
                let all: Vec<f32> = r.into_iter().flat_map(|(v, _)| v).collect();
                npy_write(&dir.join(format!("{name}_{tag}.npy")), "<f4", &[all.len() / SF, SF], &f32b(&all));
            }
            println!("records {tag}: chain {chain}, {hist} previous TRs, {} readouts ({:.1}s)", case.ros.len(), t0.elapsed().as_secs_f64());
        }
        "forward" => {
            // --models tag=spec[@hist][#θ0;θ1;…],…: several trajectories on one engine (segments built once, from the
            // first model's encode-mean concomitant coefficients; the per-sample deviations use the same mean, so
            // every variant is self-consistent); or --tag T with --model / --hist / --theta
            let models_s = take(&mut args, "--models");
            let tag1 = take(&mut args, "--tag");
            let o = op_opts(&mut args);
            let models: Vec<(String, String, usize, Vec<f64>)> = match &models_s {
                Some(ms) => ms.split(',').map(|m| {
                    let (tag, rest) = m.split_once('=').expect("--models tag=spec[@hist][#θ;…]");
                    let (rest, th) = rest.split_once('#').map_or((rest, vec![]), |(a, b)| (a, b.split(';').map(|v| v.parse().unwrap()).collect()));
                    let (spec, hist) = rest.split_once('@').map_or((rest.to_string(), o.hist), |(a, b)| (a.to_string(), b.parse().unwrap()));
                    (tag.to_string(), spec, hist, th)
                }).collect(),
                None => vec![(tag1.expect("--tag or --models"), o.model.clone(), o.hist, o.theta.clone())],
            };
            let todo: Vec<&(String, String, usize, Vec<f64>)> = models.iter().filter(|m| !dir.join(format!("y_{}.npy", m.0)).exists()).collect();
            if todo.is_empty() {
                println!("forward: every tag done");
                return;
            }
            let dev = Device::new(0).unwrap();
            let st = dev.default_stream();
            let t0 = Instant::now();
            let om = |m: &(String, String, usize, Vec<f64>)| OpOpts { model: m.1.clone(), hist: m.2, theta: m.3.clone(), ..o.clone() };
            let o0 = om(todo[0]);
            let tr0 = model_traj(&case, &o0);
            let full = full_window_records(&case, &tr0, &o0.theta, o0.hist);
            let mut e = engine(&dev, &case, &o0, &full);
            let t_eng = t0.elapsed().as_secs_f64();
            let nc = e.nc;
            for m in todo {
                let t1 = Instant::now();
                let om_ = om(m);
                let tr = model_traj(&case, &om_);
                let recs = records(&case, &tr, &om_.theta, om_.hist, o.nav, false);
                // readouts in encode order (one encode's segments on the device at a time), output in readout order
                let mut order: Vec<usize> = (0..case.ros.len()).collect();
                order.sort_by_key(|&i| case.ros[i].enc);
                let mut parts: Vec<Vec<C32>> = vec![vec![]; case.ros.len()];
                let x = e.x.clone();
                for &i in &order {
                    let (r, rec) = (&case.ros[i], &recs[i].0);
                    let pts = points_of(&mut e, &case, &st, r, rec, o.nav);
                    let mm = pts.m;
                    let mut y = st.alloc_zeros::<C32>(nc * mm).unwrap();
                    forward_one(&mut e, &st, r, &pts, &x, &mut y);
                    parts[i] = to_sample_major(&st.clone_dtoh(&y).unwrap(), nc, mm);
                }
                let out: Vec<C32> = parts.concat();
                let tag = &m.0;
                npy_write(&dir.join(format!("y_{tag}.npy")), "<c8", &[out.len() / nc, nc], &c32b(&out));
                let info = json!({"tag": tag, "model": om_.model, "theta": om_.theta, "hist": om_.hist, "fat": o.fat, "rate": o.rate, "conc": o.conc, "L": o.nseg, "nav": o.nav, "coils": nc,
                                  "seconds": {"engine": t_eng, "variant": t1.elapsed().as_secs_f64()}});
                std::fs::write(dir.join(format!("y_{tag}.json")), serde_json::to_string_pretty(&info).unwrap()).unwrap();
                println!("forward {tag}: {} samples × {nc} coils (engine {t_eng:.0}s, variant {:.1}s)", out.len() / nc, t1.elapsed().as_secs_f64());
            }
        }
        "refit" | "jac" => refit(&dir, &case, &mode, &mut args),
        "nav" => nav_monitor(&dir, &case, &mut args),
        "segbuild" => {
            // CPU only: the segment build of one encode (what the reconstruction does per encode)
            let o = op_opts(&mut args);
            let tr = model_traj(&case, &o);
            let full = full_window_records(&case, &tr, &o.theta, o.hist);
            let _ = engine_opt(None, &case, &o, &full, Some(0));
        }
        "sens" => {
            // CPU only: the refit's per-sample tables (model trajectory + ∂k/∂θ, ∂φ0/∂θ) of one production-size set
            let o = op_opts(&mut args);
            let nset: usize = take(&mut args, "--set-readouts").map(|s| s.parse().unwrap()).unwrap_or(300);
            let t0 = Instant::now();
            let cm = ChainModel::new(GirfMeasurement::load(Path::new(&o.model)).unwrap(), o.params);
            let t_model = t0.elapsed().as_secs_f64();
            let all: Vec<usize> = (0..case.na_win).collect();
            let rots: Vec<Mat3> = case.ros.iter().flat_map(|r| r.seq.iter().map(|s| s.1)).collect();
            let seqs: Vec<Vec<(usize, Mat3)>> = (0..nset).map(|i| (0..=o.hist).map(|j| (j % 7, rots[(i * 11 + j) % rots.len()])).collect()).collect();
            let times: Vec<Vec<f64>> = (0..nset).map(|_| case.times(&all)).collect();
            let th = vec![0.0; cm.n()];
            let t1 = Instant::now();
            let s0 = case.design.stream(&cm.chain(&th), o.hist);
            let t_stream = t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let sens = sensitivities(&cm, &th, &case.design, o.hist, case.b0, &seqs, &times);
            let t_sens = t2.elapsed().as_secs_f64();
            let _ = (s0, sens.len());
            let j = json!({"set_readouts": nset, "samples": nset * case.na_win, "params": cm.params.iter().map(|p| p.name()).collect::<Vec<_>>(), "threads": hasty_seq::threads(),
                           "chain_model_s": t_model, "one_stream_s": t_stream, "sensitivities_s (all streams + per-readout tables)": t_sens});
            println!("{}", serde_json::to_string_pretty(&j).unwrap());
            std::fs::write(dir.join(format!("senscost_{nset}.json")), serde_json::to_string_pretty(&j).unwrap()).unwrap();
        }
        "cost" => cost(&case, &mut args),
        m => panic!("mode {m}: records | forward | refit | jac | cost"),
    }
}

/// k-gradient images −i r_a ⊙ x (a = 0, 1, 2)
fn ramps(red: &Reducer, st: &std::sync::Arc<cudarc::driver::CudaStream>, x: &CudaSlice<C32>, dims: [usize; 3]) -> [CudaSlice<C32>; 3] {
    let nv = dims.iter().product::<usize>();
    std::array::from_fn(|a| {
        let mut o = st.alloc_zeros::<C32>(nv).unwrap();
        red.ramp(st, x, a, dims, &mut o);
        o
    })
}

fn refit(dir: &Path, case: &Case, mode: &str, args: &mut Vec<String>) {
    let mut take = |k: &str| args.iter().position(|a| a == k).map(|i| { args.remove(i); args.remove(i) });
    let data_f = take("--data");
    let tag = take("--tag").unwrap_or("refit".into());
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(4);
    let noise: f64 = take("--noise").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    // --seeds a,b,…: one refit per noise realisation on the same engine (default --seed 1)
    let seeds: Vec<u64> = take("--seeds").or_else(|| take("--seed")).map(|s| s.split(',').map(|v| v.parse().unwrap()).collect()).unwrap_or(vec![1]);
    let lambda: f64 = take("--lambda").map(|s| s.parse().unwrap()).unwrap_or(1e-3);
    let nread: usize = take("--readouts").map(|s| s.parse().unwrap()).unwrap_or(if mode == "jac" { 6 } else { case.ros.len() });
    // --enc: also fit a per-encode coherent k offset (3) and phase (1): what per-encode images absorb (the GIRF study's
    // `enc` level); --enc-only: those alone (the chain at θ = 0)
    let enc_only = args.iter().position(|a| a == "--enc-only").map(|i| { args.remove(i); true }).unwrap_or(false);
    let enc_on = enc_only || args.iter().position(|a| a == "--enc").map(|i| { args.remove(i); true }).unwrap_or(false);
    let mut o = op_opts(args);
    assert!(o.model.ends_with(".json"), "refit needs --model <hasty-girf-1 .json> (the estimated chain)");
    o.theta.clear();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let red = Reducer::new(&dev);
    let meas = GirfMeasurement::load(Path::new(&o.model)).unwrap();
    let cm = ChainModel::new(meas, if enc_only { RefitOpts { delay: false, gain: false, b0amp: false, res: false } } else { o.params });
    let ncp = cm.n();
    let ne = case.design.n_enc();
    let np_ = ncp + if enc_on { 4 * ne } else { 0 };
    let tr = Traj::Model(cm.clone());
    let mut theta = vec![0.0; ncp];
    let mut theta_enc = vec![[0.0f64; 4]; ne];
    let full = full_window_records(case, &tr, &theta, o.hist);
    let mut e = engine(&dev, case, &o, &full);
    let nc = e.nc;
    let mut ros: Vec<usize> = (0..case.ros.len()).step_by((case.ros.len() / nread.max(1)).max(1)).take(nread).collect();
    ros.sort_by_key(|&i| case.ros[i].enc);
    // data (sample-major [m][nc_all] → coil-major per readout), + white noise of σ per complex sample
    let offs: Vec<usize> = case.ros.iter().scan(0, |a, r| { let s = *a; *a += if o.nav { r.nav.len() } else { r.n.len() }; Some(s) }).collect();
    let data: Option<Vec<C32>> = data_f.map(|f| hasty_recon::data::read_npy(Path::new(&f)).unwrap().c32s());
    let x = e.x.clone();
    let dx = ramps(&red, &st, &x, case.dims);
    let mut runs = vec![];
    let (mut t_sens, mut t_gpu, mut t_acc) = (0.0, 0.0, 0.0);
    for &seed in &seeds {
    theta = vec![0.0; ncp];
    theta_enc = vec![[0.0f64; 4]; ne];
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut gauss = move || {
        let mut u = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
        (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
    };
    let ydat: Vec<Option<CudaSlice<C32>>> = ros.iter().map(|&i| {
        let r = &case.ros[i];
        let m = if o.nav { r.nav.len() } else { r.n.len() };
        data.as_ref().map(|d| {
            let total: usize = case.ros.iter().map(|r| if o.nav { r.nav.len() } else { r.n.len() }).sum();
            let nc_all = d.len() / total;
            assert_eq!(nc_all * total, d.len(), "data: {} values for {total} samples", d.len());
            let mut cm_ = vec![C32::new(0.0, 0.0); nc * m];
            for j in 0..m {
                for c in 0..nc {
                    let v = d[(offs[i] + j) * nc_all + c];
                    let (a, b) = if noise > 0.0 { (gauss() * noise / 2f64.sqrt(), gauss() * noise / 2f64.sqrt()) } else { (0.0, 0.0) };
                    cm_[c * m + j] = C32::new(v.re + a as f32, v.im + b as f32);
                }
            }
            st.clone_htod(&cm_).unwrap()
        })
    }).collect();
    let mut hist_out = vec![];
    for it in 0..=iters {
        let t0 = Instant::now();
        let recs = records(case, &tr, &theta, o.hist, o.nav, true);
        t_sens += t0.elapsed().as_secs_f64();
        let mut nrm = Normal::new(np_);
        let (mut rr, mut nsamp) = (0.0f64, 0usize);
        let mut jac_rows = vec![];
        for (q, &i) in ros.iter().enumerate() {
            let r = &case.ros[i];
            let (rec, d) = &recs[i];
            let t1 = Instant::now();
            let pts = points_shift(&mut e, case, &st, r, rec, o.nav, theta_enc[r.enc]);
            let m = pts.m;
            let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
            forward_one(&mut e, &st, r, &pts, &x, &mut y);
            let mut g: [CudaSlice<C32>; 3] = std::array::from_fn(|_| st.alloc_zeros::<C32>(nc * m).unwrap());
            for a in 0..3 {
                forward_one(&mut e, &st, r, &pts, &dx[a], &mut g[a]);
            }
            let yh = st.clone_dtoh(&y).unwrap();
            let res: Vec<C32> = match &ydat[q] {
                Some(dd) => st.clone_dtoh(dd).unwrap().iter().zip(&yh).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect(),
                None => vec![C32::new(0.0, 0.0); nc * m],
            };
            rr += res.iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum::<f64>();
            nsamp += m;
            let rd = st.clone_htod(&res).unwrap();
            let mut outd = st.alloc_zeros::<f32>(14 * m).unwrap();
            red.reduce(&st, nc, m, [&g[0], &g[1], &g[2]], &y, &rd, &mut outd);
            let o14 = st.clone_dtoh(&outd).unwrap();
            t_gpu += t1.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let red14: Vec<[f32; 14]> = o14.chunks(14).map(|c| std::array::from_fn(|k| c[k])).collect();
            let rot = hasty_seq::IDENTITY;
            let s: Vec<Vec<[f64; 4]>> = (0..m).map(|j| (0..np_).map(|p| if p < ncp { to_operator_units(&d.as_ref().unwrap()[p][j], &rot, [case.vox; 3]) } else if (p - ncp) / 4 == r.enc { let mut v = [0.0; 4]; v[(p - ncp) % 4] = 1.0; v } else { [0.0; 4] }).collect()).collect();
            nrm.add(&red14, &s);
            t_acc += t2.elapsed().as_secs_f64();
            if mode == "jac" {
                // J e_p from the k-gradient (CPU combination of the downloaded gradients) vs central differences in θ_p
                let gh: Vec<Vec<C32>> = g.iter().map(|v| st.clone_dtoh(v).unwrap()).collect();
                for p in 0..ncp {
                    let jp: Vec<Complex64> = (0..nc * m).map(|k| {
                        let (c, j) = (k / m, k % m);
                        let _ = c;
                        let sp = s[j][p];
                        let mut v = Complex64::new(0.0, 0.0);
                        for a in 0..3 {
                            v += Complex64::new(gh[a][k].re as f64, gh[a][k].im as f64) * sp[a];
                        }
                        v + Complex64::new(yh[k].im as f64, -yh[k].re as f64) * sp[3]
                    }).collect();
                    let h = match cm.params[p] { Param::Delay(_) => 2e-9, Param::B0Amp => 1e-2, _ => 1e-3 };
                    let fd: Vec<Complex64> = {
                        let mut ys = vec![];
                        for sgn in [1.0, -1.0] {
                            let mut th = theta.clone();
                            th[p] += sgn * h;
                            let rr = records(case, &tr, &th, o.hist, o.nav, false);
                            let pts2 = points_of(&mut e, case, &st, r, &rr[i].0, o.nav);
                            let mut y2 = st.alloc_zeros::<C32>(nc * m).unwrap();
                            forward_one(&mut e, &st, r, &pts2, &x, &mut y2);
                            ys.push(st.clone_dtoh(&y2).unwrap());
                        }
                        ys[0].iter().zip(&ys[1]).map(|(a, b)| Complex64::new((a.re - b.re) as f64, (a.im - b.im) as f64) / (2.0 * h)).collect()
                    };
                    let num: f64 = jp.iter().zip(&fd).map(|(a, b)| (a - b).norm_sqr()).sum();
                    let den: f64 = fd.iter().map(|b| b.norm_sqr()).sum();
                    let rel = (num / den.max(1e-300)).sqrt();
                    println!("jac readout {i} {}: |J e_p − FD| / |FD| = {rel:.2e} (|FD| rms {:.3e} per sample and coil)", cm.params[p].name(), (den / (nc * m) as f64).sqrt());
                    jac_rows.push(json!({"readout": i, "param": cm.params[p].name(), "rel": rel, "fd_rms": (den / (nc * m) as f64).sqrt()}));
                }
            }
        }
        if mode == "jac" {
            std::fs::write(dir.join(format!("jac_{tag}.json")), serde_json::to_string_pretty(&json!({"rows": jac_rows, "hist": o.hist, "L": o.nseg, "coils": nc})).unwrap()).unwrap();
            return;
        }
        let rmisfit = (rr / (nsamp as f64 * nc as f64 * case.sigma * case.sigma)).sqrt();
        let crb = nrm.crb(case.sigma);
        let corr = nrm.correlation();
        let row = json!({"iter": it, "theta": theta.clone(), "theta_natural": cm.params.iter().zip(&theta).map(|(p, v)| (p.name(), p.natural(*v))).collect::<Vec<_>>(),
                          "theta_enc": theta_enc.iter().map(|v| [v[0] / case.vox / (2.0 * std::f64::consts::PI), v[1] / case.vox / (2.0 * std::f64::consts::PI), v[2] / case.vox / (2.0 * std::f64::consts::PI), v[3]]).collect::<Vec<_>>(),
                          "R_misfit": rmisfit, "crb": crb[..ncp].to_vec(), "crb_natural": cm.params.iter().zip(&crb).map(|(p, v)| (p.name(), p.natural(*v))).collect::<Vec<_>>(),
                          "corr": corr, "samples": nsamp});
        println!("iter {it}: R(misfit, noise-referenced, incl. added noise) {rmisfit:.4}; θ = {}", cm.params.iter().zip(&theta).map(|(p, v)| format!("{} {:.3}", p.name(), p.natural(*v))).collect::<Vec<_>>().join(", "));
        hist_out.push(row);
        if it == iters {
            break;
        }
        let d = nrm.step(lambda);
        for p in 0..ncp {
            theta[p] += d[p];
        }
        for p in ncp..np_ {
            theta_enc[(p - ncp) / 4][(p - ncp) % 4] += d[p];
        }
    }
    runs.push(json!({"seed": seed, "iterations": hist_out}));
    }
    let params: Vec<String> = cm.params.iter().map(|p| p.name()).collect();
    let outj = json!({"tag": tag, "model": o.model, "params": params, "enc": enc_on, "hist": o.hist, "L": o.nseg, "coils": nc, "readouts": ros, "noise": noise, "seeds": seeds, "lambda": lambda,
                      "runs": runs, "seconds": {"sensitivities_cpu": t_sens, "gpu_passes": t_gpu, "accumulate_cpu": t_acc}});
    std::fs::write(dir.join(format!("refit_{tag}.json")), serde_json::to_string_pretty(&outj).unwrap()).unwrap();
    println!("refit {tag}: sensitivities {t_sens:.1}s CPU, GPU passes {t_gpu:.1}s, accumulation {t_acc:.1}s");
}

/// production-size timing of one (frame, encode) set
fn cost(case: &Case, args: &mut Vec<String>) {
    let mut take = |k: &str| args.iter().position(|a| a == k).map(|i| { args.remove(i); args.remove(i) });
    let nset: usize = take("--set-readouts").map(|s| s.parse().unwrap()).unwrap_or(300);
    let reps: usize = take("--reps").map(|s| s.parse().unwrap()).unwrap_or(5);
    let o = op_opts(args);
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let red = Reducer::new(&dev);
    let meas = GirfMeasurement::load(Path::new(&o.model)).unwrap();
    let cm = ChainModel::new(meas, o.params);
    let tr = Traj::Model(cm.clone());
    let theta = vec![0.0; cm.n()];
    let full = full_window_records(case, &tr, &theta, o.hist);
    let mut e = engine_opt(Some(&dev), case, &o, &full, Some(0)).unwrap();
    let nc = e.nc;
    // one set: `nset` readouts of encode 0, frame of readout 0, the full window, rotations cycled from the case's
    let na = case.na_win;
    let all: Vec<usize> = (0..na).collect();
    let rots: Vec<Mat3> = case.ros.iter().flat_map(|r| r.seq.iter().map(|s| s.1)).collect();
    let seqs: Vec<Vec<(usize, Mat3)>> = (0..nset).map(|i| (0..=o.hist).map(|j| ((j + 7 - o.hist % 7) % 7, rots[(i * 11 + j) % rots.len()])).collect()).collect();
    let times: Vec<Vec<f64>> = (0..nset).map(|_| case.times(&all)).collect();
    let t0 = Instant::now();
    let sens = sensitivities(&cm, &theta, &case.design, o.hist, case.b0, &seqs, &times);
    let t_sens = t0.elapsed().as_secs_f64();
    let r0 = Ro { enc: 0, frame: case.ros[0].frame, seq: vec![], n: all.clone(), nav: vec![] };
    let mut tab_all = SegTables { coords: [vec![], vec![], vec![]], tau: vec![], phi0: vec![], dc: vec![], t: vec![], wbar: vec![], tau_adc: vec![], max_resid_phase: 0.0 };
    let mut dc_parts = vec![];
    for s in &sens {
        let mut tb = tables(case, &s.rec, &all);
        add_conc(&e, case, &mut tb, &s.rec, &all, 0, false);
        for a in 0..3 {
            tab_all.coords[a].extend_from_slice(&tb.coords[a]);
        }
        tab_all.tau.extend_from_slice(&tb.tau);
        tab_all.phi0.extend_from_slice(&tb.phi0);
        dc_parts.push(tb.dc);
    }
    let m = tab_all.tau.len();
    let mut dc = vec![0f32; 6 * m];
    for (r, part) in dc_parts.iter().enumerate() {
        for i in 0..6 {
            dc[i * m + r * na..i * m + (r + 1) * na].copy_from_slice(&part[i * na..(i + 1) * na]);
        }
    }
    let pts = e.eng.points(&st, [&tab_all.coords[0], &tab_all.coords[1], &tab_all.coords[2]], &tab_all.tau, &tab_all.phi0, &dc, None).unwrap();
    let x = e.x.clone();
    let dx = ramps(&red, &st, &x, case.dims);
    let ydat = st.alloc_zeros::<C32>(nc * m).unwrap();
    let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
    let mut xa = st.alloc_zeros::<C32>(case.nv()).unwrap();
    let mut g: [CudaSlice<C32>; 3] = std::array::from_fn(|_| st.alloc_zeros::<C32>(nc * m).unwrap());
    let mut outd = st.alloc_zeros::<f32>(14 * m).unwrap();
    let mut rdev = st.alloc_zeros::<C32>(nc * m).unwrap();
    // warm-up (kernel compiles, autotune state)
    forward_one(&mut e, &st, &r0, &pts, &x, &mut y);
    let s_of = |j: usize, p: usize| to_operator_units(&sens[j / na].d[p][j % na], &hasty_seq::IDENTITY, [case.vox; 3]);
    let (mut tg, mut tf, mut tr_, mut ta) = (vec![], vec![], vec![], vec![]);
    for _ in 0..reps {
        // gradient pass: forward residual + adjoint (what every P²nP iteration does per set)
        st.synchronize().unwrap();
        let t = Instant::now();
        forward_one(&mut e, &st, &r0, &pts, &x, &mut y);
        e.seg(&st, 0);
        let seg = &e.cur.as_ref().unwrap().1;
        if e.union {
            let maps = e.maps.as_ref().unwrap();
            e.eng.adjoint_fat(&st, seg, &pts, maps, Some(r0.frame), &ydat, &mut xa).unwrap();
        } else {
            e.eng.adjoint(&st, seg, &pts, &ydat, &mut xa).unwrap();
        }
        st.synchronize().unwrap();
        tg.push(t.elapsed().as_secs_f64());
        // one forward alone
        let t = Instant::now();
        forward_one(&mut e, &st, &r0, &pts, &x, &mut y);
        st.synchronize().unwrap();
        tf.push(t.elapsed().as_secs_f64());
        // refit pass: three k-gradient forwards + reduction + download (the residual comes from the gradient pass)
        let t = Instant::now();
        for a in 0..3 {
            forward_one(&mut e, &st, &r0, &pts, &dx[a], &mut g[a]);
        }
        red.reduce(&st, nc, m, [&g[0], &g[1], &g[2]], &y, &rdev, &mut outd);
        let o14 = st.clone_dtoh(&outd).unwrap();
        st.synchronize().unwrap();
        tr_.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let red14: Vec<[f32; 14]> = o14.chunks(14).map(|c| std::array::from_fn(|k| c[k])).collect();
        let s: Vec<Vec<[f64; 4]>> = (0..m).map(|j| (0..cm.n()).map(|p| s_of(j, p)).collect()).collect();
        let mut nrm = Normal::new(cm.n());
        nrm.add(&red14, &s);
        ta.push(t.elapsed().as_secs_f64());
    }
    let _ = &mut rdev;
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[0], v[v.len() / 2]) };
    let (g_min, g_med) = med(&mut tg);
    let (f_min, f_med) = med(&mut tf);
    let (r_min, r_med) = med(&mut tr_);
    let (a_min, a_med) = med(&mut ta);
    let j = json!({"set_readouts": nset, "samples": m, "coils": nc, "L": o.nseg, "fat": o.fat, "params": cm.n(), "reps": reps,
                   "gradient_pass_s": [g_min, g_med], "forward_s": [f_min, f_med], "refit_gpu_s": [r_min, r_med], "refit_accumulate_cpu_s": [a_min, a_med],
                   "sensitivities_cpu_s": t_sens, "refit_over_gradient": r_med / g_med, "note": "min, median of reps; GPU shared with other jobs"});
    println!("{}", serde_json::to_string_pretty(&j).unwrap());
    std::fs::write(case.dir.join(format!("cost_{}c_L{}.json", nc, o.nseg)), serde_json::to_string_pretty(&j).unwrap()).unwrap();
}

/// Navigator monitor: per TR a constant k offset and global phase from the rewinder navigator samples against the model
/// trajectory (image = the case image), with Cramér–Rao standard deviations; compared with the per-TR mean of the true
/// − model trajectory over the navigator (records rec*_truth / the model's).
fn nav_monitor(dir: &Path, case: &Case, args: &mut Vec<String>) {
    let mut take = |k: &str| args.iter().position(|a| a == k).map(|i| { args.remove(i); args.remove(i) });
    let tag = take("--tag").unwrap_or("nav".into());
    let data_f = take("--data").unwrap_or_else(|| dir.join("ynav_exact.npy").display().to_string());
    let truth_f = take("--truth-rec").unwrap_or_else(|| dir.join("recnav_truth.npy").display().to_string());
    let mut o = op_opts(args);
    o.nav = true;
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let red = Reducer::new(&dev);
    let tr = model_traj(case, &o);
    let full = full_window_records(case, &tr, &o.theta, o.hist);
    let recs = records(case, &tr, &o.theta, o.hist, true, false);
    let mut e = engine(&dev, case, &o, &full);
    let nc = e.nc;
    let d = hasty_recon::data::read_npy(Path::new(&data_f)).unwrap().c32s();
    let truth = hasty_recon::data::read_npy(Path::new(&truth_f)).unwrap().f32s();
    let total: usize = case.ros.iter().map(|r| r.nav.len()).sum();
    let nc_all = d.len() / total;
    let x = e.x.clone();
    let red_ = ramps(&red, &st, &x, case.dims);
    let mut rows = vec![];
    let two_pi_vox = 2.0 * std::f64::consts::PI * case.vox;
    let mut order: Vec<usize> = (0..case.ros.len()).collect();
    order.sort_by_key(|&i| case.ros[i].enc);
    let offs: Vec<usize> = case.ros.iter().scan(0, |a, r| { let s = *a; *a += r.nav.len(); Some(s) }).collect();
    for &i in &order {
        let (r, rec) = (&case.ros[i], &recs[i].0);
        let off = offs[i];
        let m = r.nav.len();
        let pts = points_of(&mut e, case, &st, r, rec, true);
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        forward_one(&mut e, &st, r, &pts, &x, &mut y);
        let mut g: [CudaSlice<C32>; 3] = std::array::from_fn(|_| st.alloc_zeros::<C32>(nc * m).unwrap());
        for a in 0..3 {
            forward_one(&mut e, &st, r, &pts, &red_[a], &mut g[a]);
        }
        let yh = st.clone_dtoh(&y).unwrap();
        let mut res = vec![C32::new(0.0, 0.0); nc * m];
        for j in 0..m {
            for c in 0..nc {
                let v = d[(off + j) * nc_all + c];
                res[c * m + j] = C32::new(v.re - yh[c * m + j].re, v.im - yh[c * m + j].im);
            }
        }
        let rd = st.clone_htod(&res).unwrap();
        let mut outd = st.alloc_zeros::<f32>(14 * m).unwrap();
        red.reduce(&st, nc, m, [&g[0], &g[1], &g[2]], &y, &rd, &mut outd);
        let o14 = st.clone_dtoh(&outd).unwrap();
        let red14: Vec<[f32; 14]> = o14.chunks(14).map(|c| std::array::from_fn(|k| c[k])).collect();
        let (est, sd) = hasty_recon::girf_refit::navigator_fit(&red14, m, case.sigma)[0];
        // the true − model trajectory over the navigator: mean (k) and the B0-term phase difference
        let mut dk = [0.0f64; 3];
        let mut dph = 0.0;
        for j in 0..m {
            for a in 0..3 {
                dk[a] += (truth[(off + j) * SF + a] - rec[j * SF + a]) as f64 / m as f64;
            }
            dph += (truth[(off + j) * SF + 15] - rec[j * SF + 15]) as f64 / m as f64;
        }
        let cyc = |v: f64| v / two_pi_vox;
        rows.push(json!({"readout": i, "enc": r.enc, "est_dk_cyc_m": [cyc(est[0]), cyc(est[1]), cyc(est[2])], "est_dphi": est[3],
                         "sd_dk_cyc_m": [cyc(sd[0]), cyc(sd[1]), cyc(sd[2])], "sd_dphi": sd[3],
                         "true_mean_dk_cyc_m": dk.map(|v| v / (2.0 * std::f64::consts::PI)), "true_mean_dphi": dph}));
    }
    std::fs::write(dir.join(format!("nav_{tag}.json")), serde_json::to_string_pretty(&json!({"tag": tag, "model": o.model, "hist": o.hist, "L": o.nseg, "coils": nc, "rows": rows})).unwrap()).unwrap();
    println!("navigator monitor {tag}: {} TRs → nav_{tag}.json", rows.len());
}
