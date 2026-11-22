//! Segment count L under field absorption, per knot and per encode (CPU only, no device).
//!
//!   fieldabsorb <sim_dir> --warm <run dir> [--grid-scale s] [--knots2d 6:1,16:3] [--resp bellows|truth] [--vcoils N]
//!               [--seg-fit-stride 3] [--knot-stride 1] [--enc 0,1,..|all] [--variants a,b,c,d,e,f] [--nofat-stride K]
//!               [--control 1] [--json out.json]
//!
//! Question (research/field_absorb.md): how many time segments does the knot engine's segment model need when the
//! constant and linear spatial parts of the field are moved into the per-sample tables (constant → φ0, linear →
//! coordinate shift; `segphys::absorb_field_poly`) before the segment SVD?
//!
//! The segment model of one knot b and encode e is exactly recon_p2np's `--ratemap truth-bc` build:
//!   g_b(v, τ) = [(1 − φ_b) e^{−R^w_b τ'} + φ_b e^{−R^f_b τ'} c(τ)] e^{−iψ_b(v, τ)},
//!   ψ_b = conc0_e(v, τ) + ω_b(v) τ   (conc0_e: the encode-mean concomitant phase, `segment_phase` without B0)
//! with the TRUE per-knot ratemap (scan/truth_ratemap_bc/ at the knot's basis-weighted state, as recon_p2np's
//! truth_ratemap_knots), and L from the noise rule of `--nseg auto:noise:0.2` (svdn basis, energy voxel weights =
//! smoothed warm-start magnitude² × coil energy over the fit region / median, per-ADC-sample weights 1 + (E(n) +
//! E_vol) / (C σ²) on `seg_time_grid`, margin 1.5): the smallest L whose weighted tail ≤ 0.2 / (gain · 1.5). Also L at
//! relative tolerances 1e-3 and 1e-4 of the same weighted tail.
//!
//! The warm start, the support and the fit region come from an existing recon_p2np run on the same grid (`--warm`:
//! warm.npy, mask_fit.npy); the support magnitude is recomputed exactly as recon_p2np does. The coil energy uses the
//! whitened true maps compressed by their own Gram over the fit region (`--vcoils`, default all coils): the run's
//! calibration-region compression needs the device.
//!
//! Variants of the phase the segments carry:
//!   a: today (no absorption);
//!   b: `--b0-absorb 1` as implemented: one global constant + linear fit of the static field `grid.omega`;
//!   c: per knot constant + linear fit of ω_b (weighted LSQ, the same weights);
//!   d: per knot constant + linear + quadratic (quadratic to ΔC relative to the readout centre, as absorb_field_poly);
//!   e: c + the constant + linear part of conc0_e (per encode, weighted LSQ of the monomials, any time function:
//!      absorbable per sample);
//!   f: d + the constant + linear + quadratic part of conc0_e (quadratic to ΔC, the time-mean part left static).
//! `--nofat-stride K`: every K-th knot also without fat (water only) for a, c, e (what the fat term alone costs).
//! `--control 1`: a synthetic ratemap ω = 2π(40 Hz + 1500 Hz/m · y + 800 Hz/m · z) with knot 0's decay and fat: after
//! per-knot absorption (c) it must give the same spectrum as ω = 0.

use std::path::{Path, PathBuf};
use std::time::Instant;

use hasty_cuda::C32;
use hasty_nufft::{noise_time_weights, seg_count_for, seg_time_grid, tail_errors, vol_energy, weighted_spectrum};
use hasty_recon::coils::{CoilCompression, CoilGram, CoilRule};
use hasty_recon::data::{read_npy, SimData};
use hasty_recon::kload::LoadOpts;
use hasty_recon::segphys::{fit_field_poly_at, segment_phase, FieldPoly};
use hasty_recon::vknot::KnotBasis;
use num_complex::Complex64;
use rayon::prelude::*;
use serde_json::json;

const TAU: f64 = std::f64::consts::TAU;

struct RateFrame {
    om: Vec<f64>,
    rw: Vec<f32>,
    rf: Vec<f32>,
    ff: Vec<f32>,
}

/// scan/truth_ratemap_bc/ states (as recon_p2np's truth_ratemap_knots), knot frames built on demand
struct TruthStates {
    idx: Vec<usize>,
    ntot: usize,
    mdims: [usize; 3],
    om: Vec<f32>,
    rw: Vec<f32>,
    rf: Vec<f32>,
    ff: Vec<f32>,
    rk: Vec<f64>,
    cq: Vec<f64>,
    period: f64,
}

impl TruthStates {
    fn load(dir: &Path) -> Self {
        let td = dir.join("truth_ratemap_bc");
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(td.join("meta.json")).unwrap()).unwrap();
        let stv = &meta["states"];
        let rk: Vec<f64> = stv["r"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let cq: Vec<f64> = stv["c_s"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let period = stv["period_s"].as_f64().unwrap();
        let mask_a = read_npy(&td.join("mask.npy")).unwrap();
        assert!(mask_a.fortran || mask_a.shape.iter().filter(|&&n| n > 1).count() <= 1);
        let mask = mask_a.bools();
        let mdims = [mask_a.shape[0], mask_a.shape[1], mask_a.shape[2]];
        let idx: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
        let load = |n: &str| read_npy(&td.join(n)).unwrap().f32s();
        let t = Instant::now();
        let (om, rw, rf, ff) = (load("omega_hz.npy"), load("r_water.npy"), load("r_fat.npy"), load("ff.npy"));
        assert_eq!(om.len(), rk.len() * cq.len() * idx.len());
        println!("truth ratemap states: {} breathing × {} cardiac, {} mask voxels of {:?} ({:.0}s)", rk.len(), cq.len(), idx.len(), mdims, t.elapsed().as_secs_f64());
        TruthStates { idx, ntot: mask.len(), mdims, om, rw, rf, ff, rk, cq, period }
    }

    /// bilinear corners (state index, weight) of a (breathing r, cardiac time c) state
    fn corners(&self, r: f64, c: f64) -> [(usize, f64); 4] {
        let (kk, qq) = (self.rk.len(), self.cq.len());
        let rk = &self.rk;
        let ur = if kk > 1 { ((r - rk[0]) / (rk[kk - 1] - rk[0]) * (kk - 1) as f64).clamp(0.0, (kk - 1) as f64) } else { 0.0 };
        let k0 = (ur.floor() as usize).min(kk.saturating_sub(2));
        let fr = if kk > 1 { ur - k0 as f64 } else { 0.0 };
        let uc = (c / self.period * qq as f64).rem_euclid(qq as f64);
        let (q0, fc) = (uc.floor() as usize % qq, uc - uc.floor());
        let q1 = (q0 + 1) % qq;
        let k1 = (k0 + 1).min(kk - 1);
        [(q0 * kk + k0, (1.0 - fr) * (1.0 - fc)), (q0 * kk + k1, fr * (1.0 - fc)), (q1 * kk + k0, (1.0 - fr) * fc), (q1 * kk + k1, fr * fc)]
    }

    fn frame(&self, corners: &[(usize, f64); 4], sd: &SimData, x_log: &[[f64; 3]], static_om: &[f64]) -> RateFrame {
        let nm = self.idx.len();
        let interp = |a: &[f32], fill: f32| -> Vec<f32> {
            let mut full = vec![fill; self.ntot];
            let vals: Vec<(usize, f32)> = self
                .idx
                .par_iter()
                .enumerate()
                .filter_map(|(n, &vi)| {
                    let (mut s, mut ws) = (0.0f64, 0.0f64);
                    for &(st, cw) in corners {
                        let x = a[st * nm + n];
                        if cw > 0.0 && x.is_finite() {
                            s += cw * x as f64;
                            ws += cw;
                        }
                    }
                    (ws > 0.0).then(|| (vi, (s / ws) as f32))
                })
                .collect();
            for (vi, x) in vals {
                full[vi] = x;
            }
            full
        };
        let to_grid = |full: Vec<f32>| -> Vec<f32> {
            if self.mdims == sd.dims || sd.native_grid() {
                full
            } else {
                assert_eq!(self.mdims, sd.tdims);
                hasty_recon::pyramid::native_maps_to_grid(sd, &full, x_log, false)
            }
        };
        let om_g = to_grid(interp(&self.om, f32::NAN));
        let om: Vec<f64> = om_g.iter().enumerate().map(|(v, &x)| if x.is_finite() { x as f64 * TAU } else { static_om[v] }).collect();
        let fin = |x: f32| if x.is_finite() { x } else { 0.0 };
        let rw = to_grid(interp(&self.rw, 0.0)).into_iter().map(fin).collect();
        let rf = to_grid(interp(&self.rf, 0.0)).into_iter().map(fin).collect();
        let ff = to_grid(interp(&self.ff, 0.0)).into_iter().map(|x| fin(x).clamp(0.0, 1.0)).collect();
        RateFrame { om, rw, rf, ff }
    }
}

/// weighted quantile of (value, weight) pairs
fn wquant(vals: &mut [(f64, f64)], q: f64) -> f64 {
    vals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let tot: f64 = vals.iter().map(|p| p.1).sum();
    let mut acc = 0.0;
    for p in vals.iter() {
        acc += p.1;
        if acc >= q * tot {
            return p.0;
        }
    }
    vals.last().map_or(0.0, |p| p.0)
}

/// signal-weighted summary of a field (Hz): [p1, p99, p0.1, p99.9, weighted std, min, max]
fn field_stats(f: &dyn Fn(usize) -> f64, vox: &[usize], w: &[f64]) -> [f64; 7] {
    let mut p: Vec<(f64, f64)> = vox.iter().zip(w).map(|(&v, &wt)| (f(v) / TAU, wt)).collect();
    let tw: f64 = p.iter().map(|x| x.1).sum();
    let mean = p.iter().map(|x| x.0 * x.1).sum::<f64>() / tw;
    let sd = (p.iter().map(|x| (x.0 - mean).powi(2) * x.1).sum::<f64>() / tw).sqrt();
    let (mn, mx) = p.iter().fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(x.0), b.max(x.0)));
    [wquant(&mut p, 0.01), wquant(&mut p, 0.99), wquant(&mut p, 0.001), wquant(&mut p, 0.999), sd, mn, mx]
}

fn basis10(p: &[f64; 3]) -> [f64; 10] {
    [1.0, p[0], p[1], p[2], p[0] * p[0], p[1] * p[1], p[2] * p[2], 2.0 * p[0] * p[1], 2.0 * p[0] * p[2], 2.0 * p[1] * p[2]]
}

/// weighted LSQ of the four concomitant monomials over `vox` on the first `nb` terms of basis10 (4: lin, 10: quad)
fn fit_mono_w(sd: &SimData, mono: &[[f64; 4]], vox: &[usize], w: &[f64], nb: usize) -> [[f64; 10]; 4] {
    let mut ata = nalgebra::DMatrix::<f64>::zeros(nb, nb);
    let mut atb = nalgebra::DMatrix::<f64>::zeros(nb, 4);
    for (r, &v) in vox.iter().enumerate() {
        let b = basis10(&sd.voxel_log(v));
        for i in 0..nb {
            for k in 0..nb {
                ata[(i, k)] += w[r] * b[i] * b[k];
            }
            for q in 0..4 {
                atb[(i, q)] += w[r] * b[i] * mono[v][q];
            }
        }
    }
    let tr = (0..nb).map(|i| ata[(i, i)]).sum::<f64>() / nb as f64;
    for i in 0..nb {
        ata[(i, i)] += 1e-12 * tr;
    }
    let sol = ata.cholesky().expect("monomial fit").solve(&atb);
    let mut th = [[0.0; 10]; 4];
    for q in 0..4 {
        for i in 0..nb {
            th[q][i] = sol[(i, q)];
        }
    }
    th
}

/// w̄(τ) linearly interpolated as segment_phase does
fn wbar_at(wbar: &[[f64; 4]], tau_adc: &[f64], tau: f64) -> [f64; 4] {
    let n = tau_adc.len();
    let u = ((tau - tau_adc[0]) / (tau_adc[n - 1] - tau_adc[0]) * (n - 1) as f64).clamp(0.0, (n - 1) as f64 - 1e-9);
    let (i, fr) = (u.floor() as usize, u - u.floor());
    let j = (i + 1).min(n - 1);
    [0, 1, 2, 3].map(|q| wbar[i][q] + fr * (wbar[j][q] - wbar[i][q]))
}

/// weighted Lp fit (IRLS) of degree 1: minimises Σ_r w_r |ω − c₀ − c₁·ρ|^p (p large: towards the minimax fit,
/// the field range the segments see)
fn lp_fit(pos: &dyn Fn(usize) -> [f64; 3], om: &[f64], vox: &[usize], w: &[f64], p: f64, tau_ref: f64) -> FieldPoly {
    let mut poly = fit_field_poly_at(pos, om, vox, Some(w), 1, tau_ref);
    for _ in 0..40 {
        let res: Vec<f64> = vox.iter().map(|&v| (om[v] - poly.eval_lin(&pos(v))).abs()).collect();
        let scale = res.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
        let wi: Vec<f64> = res.iter().zip(w).map(|(&r, &wt)| wt * (r / scale).max(1e-3).powf(p - 2.0)).collect();
        poly = fit_field_poly_at(pos, om, vox, Some(&wi), 1, tau_ref);
    }
    poly
}

/// Nelder–Mead minimisation of f over R^n from x0 with initial step `step`, `iters` iterations
fn nelder_mead(f: &mut dyn FnMut(&[f64]) -> f64, x0: &[f64], step: f64, iters: usize) -> (Vec<f64>, f64, usize) {
    let n = x0.len();
    let mut pts: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut evals = 0usize;
    let mut ev = |x: &[f64], evals: &mut usize| { *evals += 1; f(x) };
    let f0 = ev(x0, &mut evals);
    pts.push((x0.to_vec(), f0));
    for i in 0..n {
        let mut x = x0.to_vec();
        x[i] += step;
        let fx = ev(&x, &mut evals);
        pts.push((x, fx));
    }
    for _ in 0..iters {
        pts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let c: Vec<f64> = (0..n).map(|i| pts[..n].iter().map(|p| p.0[i]).sum::<f64>() / n as f64).collect();
        let w = pts[n].clone();
        let at = |t: f64| -> Vec<f64> { (0..n).map(|i| c[i] + t * (w.0[i] - c[i])).collect() };
        let xr = at(-1.0);
        let fr = ev(&xr, &mut evals);
        if fr < pts[0].1 {
            let xe = at(-2.0);
            let fe = ev(&xe, &mut evals);
            pts[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < pts[n - 1].1 {
            pts[n] = (xr, fr);
        } else {
            let xc = at(if fr < w.1 { -0.5 } else { 0.5 });
            let fc = ev(&xc, &mut evals);
            if fc < w.1.min(fr) {
                pts[n] = (xc, fc);
            } else {
                let best = pts[0].0.clone();
                for k in 1..=n {
                    let x: Vec<f64> = (0..n).map(|i| best[i] + 0.5 * (pts[k].0[i] - best[i])).collect();
                    let fx = ev(&x, &mut evals);
                    pts[k] = (x, fx);
                }
            }
        }
    }
    pts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    (pts[0].0.clone(), pts[0].1, evals)
}

#[derive(Clone, Copy)]
struct Ls {
    noise: usize,
    e3: usize,
    e4: usize,
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let warm_dir = PathBuf::from(take("--warm").expect("--warm <recon_p2np run dir with warm.npy, mask_fit.npy>"));
    let grid_scale: Option<[f64; 3]> = take("--grid-scale").map(|s| { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); if v.len() == 1 { [v[0]; 3] } else { [v[0], v[1], v[2]] } });
    let knots_spec = take("--knots2d").unwrap_or("6:1,16:3".into());
    let resp_src = take("--resp").unwrap_or("bellows".into());
    let vcoils: Option<usize> = take("--vcoils").map(|s| s.parse().unwrap());
    let fit_stride: usize = take("--seg-fit-stride").map_or(3, |s| s.parse().unwrap());
    let knot_stride: usize = take("--knot-stride").map_or(1, |s| s.parse().unwrap());
    let enc_spec = take("--enc").unwrap_or("all".into());
    let variants: Vec<char> = take("--variants").unwrap_or("a,b,c,d,e,f,g".into()).split(',').map(|s| s.chars().next().unwrap()).collect();
    let opt_stride: usize = take("--opt-stride").map_or(0, |s| s.parse().unwrap());
    let lp_p: f64 = take("--lp").map_or(8.0, |s| s.parse().unwrap());
    let nofat_stride: usize = take("--nofat-stride").map_or(0, |s| s.parse().unwrap());
    let control = take("--control").map_or(true, |s| s != "0");
    let json_out = take("--json").map(PathBuf::from);
    let dir = PathBuf::from(&args[0]);
    let t_all = Instant::now();

    let mut sd = SimData::load_lazy(&dir).unwrap();
    if let Some(s) = grid_scale {
        sd.set_recon_grid(s);
    }
    let (nv, nc, na, dims) = (sd.nv(), sd.nc, sd.na, sd.dims);
    println!("{}: grid {:?} (native {:?}), voxel {:?} mm, {nc} coils, {na} ADC samples", dir.display(), dims, sd.tdims, sd.vox.map(|v| (v * 1e4).round() / 10.0));
    let w = sd.whitener();
    let t0 = Instant::now();
    let grid = sd.physics_grid(true, &w);
    println!("physics grid (GNL-aware) {:.0}s", t0.elapsed().as_secs_f64());

    // ── warm start → support magnitude, support, fit region (recon_p2np's code)
    let warm = read_npy(&warm_dir.join("warm.npy")).unwrap().c32s();
    assert_eq!(warm.len() % nv, 0, "warm.npy of a run on grid {dims:?}");
    let ne = warm.len() / nv;
    let mut acc = vec![0f32; nv];
    for e in 0..ne {
        for q in 0..nv {
            let z = warm[e * nv + q];
            acc[q] += (z.re * z.re + z.im * z.im).sqrt();
        }
    }
    drop(warm);
    for (axis, r) in [(0usize, 4usize), (1, 4), (2, 1)] {
        let r = (r as f64 * sd.res / sd.vox[axis]).round() as usize;
        let (n, stride) = (dims[axis], [1, dims[0], dims[0] * dims[1]][axis]);
        for _ in 0..3 {
            let src = acc.clone();
            acc.par_iter_mut().enumerate().for_each(|(q, a)| {
                let i = (q / stride) % n;
                let (lo, hi) = (i.saturating_sub(r), (i + r).min(n - 1));
                let base = q - i * stride;
                *a = (lo..=hi).map(|k| src[base + k * stride]).sum::<f32>() / (hi - lo + 1) as f32;
            });
        }
    }
    let support_mag = acc.clone();
    let support: Vec<bool> = {
        let mut sorted = acc.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let thr = 0.05 * sorted[sorted.len() * 99 / 100];
        let fg: Vec<bool> = acc.iter().map(|&v| v > thr).collect();
        let d = dims;
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
    let mut dil = support.clone();
    {
        let d = dims;
        let n_dil = ((3.0 * sd.res / (sd.vox.iter().sum::<f64>() / 3.0)).round() as usize).max(1);
        for _ in 0..n_dil {
            let prev = dil.clone();
            dil.par_iter_mut().enumerate().for_each(|(q, x)| {
                if prev[q] {
                    return;
                }
                let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
                let nb = [(i > 0).then(|| q - 1), (i + 1 < d[0]).then(|| q + 1), (j > 0).then(|| q - d[0]), (j + 1 < d[1]).then(|| q + d[0]), (k > 0).then(|| q - d[0] * d[1]), (k + 1 < d[2]).then(|| q + d[0] * d[1])];
                *x = nb.iter().flatten().any(|&n| prev[n]);
            });
        }
    }
    let fit_vox: Vec<usize> = (0..nv).filter(|&q| dil[q]).collect();
    let run_mask = read_npy(&warm_dir.join("mask_fit.npy")).unwrap().f32s();
    let agree = (0..nv).filter(|&q| (run_mask[q] > 0.5) == dil[q]).count();
    println!("support {} voxels, fit region {} voxels; the run's mask_fit: {} voxels, agreement {:.5}", support.iter().filter(|&&b| b).count(), fit_vox.len(), run_mask.iter().filter(|&&v| v > 0.5).count(), agree as f64 / nv as f64);

    // ── coil energy
    let nvc = vcoils.unwrap_or(nc).min(nc);
    let coil_energy: Vec<f64> = if nvc < nc {
        let mut g = CoilGram::new(nc);
        let nf = fit_vox.len();
        let mut blk = vec![C32::new(0.0, 0.0); nc * nf];
        for c in 0..nc {
            for (r, &q) in fit_vox.iter().enumerate() {
                blk[c * nf + r] = grid.smaps[c * nv + q];
            }
        }
        g.add(&blk, nf, 1);
        let cc = CoilCompression::from_gram(&g, CoilRule::Fixed(nvc));
        println!("coil energy: whitened true maps compressed by their fit-region Gram: {}", cc.report());
        let sm = cc.apply(&grid.smaps, nv);
        (0..nv).map(|q| (0..nvc).map(|c| { let z = sm[c * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect()
    } else {
        println!("coil energy: all {nc} whitened true maps");
        (0..nv).map(|q| (0..nc).map(|c| { let z = grid.smaps[c * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect()
    };
    // fit weights (recon_p2np's --b0-absorb weights) over fit_vox; segment-rule voxel weights over the strided subset
    let wv: Vec<f64> = fit_vox.iter().map(|&q| (support_mag[q] as f64).powi(2) * coil_energy[q]).collect();
    let seg_fit: Vec<usize> = fit_vox.iter().copied().step_by(fit_stride).collect();
    let seg_ew: Vec<f64> = {
        let e: Vec<f64> = seg_fit.iter().map(|&q| (support_mag[q] as f64).powi(2) * coil_energy[q]).collect();
        let mut s = e.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = s[s.len() / 2].max(1e-300);
        e.into_iter().map(|v| v / med).collect()
    };
    println!("segment-rule voxels: {} (fit region stride {fit_stride})", seg_fit.len());

    // ── per-ADC-sample weights, time grid, tolerance (recon_p2np's noise rule)
    let ros0 = sd.readouts_of(0, None);
    let s0 = sd.load_set(&ros0[..1], &LoadOpts { w: &w, cc: None, keep: None, data: false }).unwrap();
    let tau_adc: Vec<f64> = s0.tau[..na].to_vec();
    let kap: Vec<f64> = (0..na).map(|n| (0..3).map(|a| (s0.coords[a][n] as f64 / sd.vox[a]).powi(2)).sum::<f64>().sqrt()).collect();
    let noise_sigma: f64 = std::env::var("HASTY_NOISE_SIGMA").ok().and_then(|v| v.parse::<f64>().ok()).filter(|v| *v > 0.0).or_else(|| sd.meta["noise_sigma"].as_f64().filter(|v| *v > 0.0)).unwrap_or(1.0);
    let t1 = Instant::now();
    let ros_e: Vec<usize> = (0..sd.n_readouts()).step_by(7).take(512).collect();
    let e_adc = sd.adc_energy_whitened(&ros_e, &w).unwrap();
    let noise = nc as f64 * noise_sigma * noise_sigma;
    let snr_w = noise_time_weights(&e_adc, &kap, noise);
    let (ts_n, tw_n) = seg_time_grid(&tau_adc, &snr_w, 96);
    let gain = (snr_w.iter().sum::<f64>() / na as f64).sqrt();
    let margin: f64 = std::env::var("HASTY_SEG_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
    let r_noise = 0.2;
    let tol = r_noise / (gain * margin);
    println!("per-sample weights (σ = {noise_sigma:.3}): first {:.2e}, middle {:.2e}, last {:.2e}; white floor {:.1}; time grid {} points; noise rule: relative weighted error ≤ {tol:.3e} (gain {gain:.1}, margin {margin}) ({:.0}s)", snr_w[0] - 1.0, snr_w[na / 2] - 1.0, snr_w[na - 1] - 1.0, vol_energy(&e_adc, &kap) / noise, ts_n.len(), t1.elapsed().as_secs_f64());
    let (tw0, tw1) = (tau_adc[0], tau_adc[na - 1]);
    let tau_ref = 0.5 * (tw0 + tw1);
    let dtmax = (tw1 - tau_ref).max(tau_ref - tw0);

    // ── encode-mean concomitant coefficients (all readouts of the encode, chunked)
    let ne_scan = sd.encode_m1.len();
    let encs: Vec<usize> = if enc_spec == "all" { (0..ne_scan).collect() } else { enc_spec.split(',').map(|s| s.parse().unwrap()).collect() };
    let t2 = Instant::now();
    let wbar: Vec<Vec<[f64; 4]>> = (0..ne_scan)
        .map(|e| {
            if !encs.contains(&e) {
                return vec![[0.0; 4]; na];
            }
            let ros = sd.readouts_of(e, None);
            let mut acc = vec![[0.0f64; 4]; na];
            for ch in ros.chunks(256) {
                let s = sd.load_set(ch, &LoadOpts { w: &w, cc: None, keep: None, data: false }).unwrap();
                for (j, wq) in s.wq.iter().enumerate() {
                    for q in 0..4 {
                        acc[j % na][q] += wq[q];
                    }
                }
            }
            acc.iter().map(|a| a.map(|x| x / ros.len() as f64)).collect()
        })
        .collect();
    println!("encode-mean concomitant coefficients of {} encodes ({:.0}s)", encs.len(), t2.elapsed().as_secs_f64());

    // ── knot basis and knot states (recon_p2np: bellows amplitude of the used readouts; truth state = oracle)
    let (kr, rdeg, mc, cdeg) = KnotBasis::parse_spec(&knots_spec).unwrap();
    let nf = sd.n_frames;
    let nro = sd.n_readouts();
    let (file, col) = match resp_src.as_str() { "bellows" => ("physio.npy", 3), "truth" => ("resp.npy", 0), o => panic!("--resp {o}") };
    let a = read_npy(&dir.join(file)).unwrap();
    let (aw, av) = (a.shape[1], a.f64s());
    let amp: Vec<f64> = (0..nro).map(|i| av[i * aw + col]).collect();
    let used: Vec<usize> = (0..nro).filter(|&i| sd.ro.frame[i] != usize::MAX).collect();
    let amp_used: Vec<f64> = used.iter().map(|&i| amp[i]).collect();
    let kb = KnotBasis::from_amplitude(&amp_used, kr, rdeg, mc, cdeg);
    let mut per = vec![(0u32, 0f32); nro];
    for (j, x) in kb.assign(&amp_used).into_iter().enumerate() {
        per[used[j]] = x;
    }
    println!("knot basis ({resp_src}): {}", kb.describe());
    let t_ro: Vec<f32> = (0..nro).map(|i| { let c = sd.ro.card[i]; (c.phase * nf as f64 - 0.5).rem_euclid(nf as f64) as f32 }).collect();
    let tf: Vec<f32> = used.iter().map(|&i| t_ro[i]).collect();
    let (kt, ri, rv) = kb.sample_coords(&tf, nf, &per, &|j| used[j]);
    let nk = kb.nk();
    let ts = TruthStates::load(&dir);
    let ra = read_npy(&dir.join("resp.npy")).unwrap();
    let (rw_, rv_) = (ra.shape[1], ra.f64s());
    let pa = read_npy(&dir.join("physio.npy")).unwrap();
    let (pw_, pv_) = (pa.shape[1], pa.f64s());
    let mut kacc = vec![(0.0f64, 0.0f64, Complex64::new(0.0, 0.0)); nk];
    for (j, &i) in used.iter().enumerate() {
        let (r, ph) = (rv_[i * rw_], pv_[i * pw_ + 1]);
        let z = Complex64::from_polar(1.0, TAU * ph);
        for (b, wb) in kb.sample_weights(kt[j] as f64, ri[j], rv[j]) {
            kacc[b].0 += wb;
            kacc[b].1 += wb * r;
            kacc[b].2 += wb * z;
        }
    }
    let kstate: Vec<(f64, f64, f64)> = kacc.iter().map(|&(ws, wr, wz)| (if ws > 0.0 { wr / ws } else { 0.0 }, (wz.arg() / TAU).rem_euclid(1.0) * ts.period, ws)).collect();

    // ── fat spectrum on the time grid
    let fs = hasty_qfit::waterfat::FatSpectrum::six_peak();
    let cf: Vec<Complex64> = hasty_recon::setop::fat_spectrum_weights(&fs.ppm, &fs.amp, sd.b0, &ts_n).iter().map(|w| Complex64::new(w.re as f64, w.im as f64)).collect();
    let cfat = |t: f64| -> Complex64 {
        let i = ts_n.partition_point(|&x| x < t - 1e-12);
        if i < ts_n.len() && (ts_n[i] - t).abs() < 1e-12 {
            cf[i]
        } else {
            let w = hasty_recon::setop::fat_spectrum_weights(&fs.ppm, &fs.amp, sd.b0, &[t])[0];
            Complex64::new(w.re as f64, w.im as f64)
        }
    };
    let ta = tau_adc[0];
    let spectrum_l = |g: &(dyn Fn(usize, f64) -> Complex64 + Sync)| -> (Ls, Vec<f64>) {
        let ev = weighted_spectrum(&seg_fit, Some(&seg_ew), &ts_n, &tw_n, g);
        let tails = tail_errors(&ev);
        (Ls { noise: seg_count_for(&ev, tol, 2, 64), e3: seg_count_for(&ev, 1e-3, 2, 64), e4: seg_count_for(&ev, 1e-4, 2, 64) }, tails[..=30.min(tails.len() - 1)].to_vec())
    };

    // ── geometry of the shifts: grid cell of the σ = 1.25 grid in k·Δ = 2π / (σ N)
    let sigma_os = 1.25;
    let cells = |c1: &[f64], t: f64| -> [f64; 3] { [0, 1, 2].map(|a| (c1[a] * sd.vox[a] * t).abs() * sigma_os * dims[a] as f64 / TAU) };
    let pos = |v: usize| sd.voxel_log(v);

    // ── global fit (variant b) and the static rule of today (recon_p2np truth-bc picks L from the static field)
    let poly_glob = fit_field_poly_at(&pos, &grid.omega, &fit_vox, Some(&wv), 1, tau_ref);
    let hz = |c: f64| c / TAU;
    println!("global fit (--b0-absorb 1): {:.1} Hz + [{:.1}, {:.1}, {:.1}] Hz/m; max shift [{:.2}, {:.2}, {:.2}] cells at τ = {:.2} ms", hz(poly_glob.c[0]), hz(poly_glob.c[1]), hz(poly_glob.c[2]), hz(poly_glob.c[3]), cells(&poly_glob.c[1..4], tw1)[0], cells(&poly_glob.c[1..4], tw1)[1], cells(&poly_glob.c[1..4], tw1)[2], tw1 * 1e3);
    let static_stats = field_stats(&|v| grid.omega[v], &fit_vox, &wv);
    let mut l_static = Vec::new();
    for &e in &encs {
        let conc0 = segment_phase(&grid, &wbar[e], &tau_adc, false);
        let g = |v: usize, t: f64| Complex64::from_polar(1.0, -(conc0(v, t) + grid.omega[v] * t));
        let (l, _) = spectrum_l(&g);
        l_static.push(l.noise);
    }
    println!("today's rule (static field, phase only, as recon_p2np --ratemap truth-bc): L per encode {:?} → L = {}", l_static, l_static.iter().max().unwrap());

    // ── concomitant mean phase: constant / linear / quadratic parts per encode (weighted LSQ, fit weights)
    let th_lin = fit_mono_w(&sd, &grid.mono, &fit_vox, &wv, 4);
    let th_quad = fit_mono_w(&sd, &grid.mono, &fit_vox, &wv, 10);
    let mut conc_info = Vec::new();
    for &e in &encs {
        // per time grid point: constant (rad), linear (rad/m) of conc0, shift cells, conc0 range before / after
        let (mut c0max, mut shmax, mut rng_b, mut rng_a, mut rng_q, mut qtay) = (0.0f64, [0.0f64; 3], 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let wref = wbar_at(&wbar[e], &tau_adc, tau_ref);
        for &t in ts_n.iter().step_by(4).chain(std::iter::once(&tw1)) {
            let wb = wbar_at(&wbar[e], &tau_adc, t);
            let c0: f64 = (0..4).map(|q| wb[q] * th_lin[q][0]).sum();
            let c1: Vec<f64> = (1..4).map(|i| (0..4).map(|q| wb[q] * th_lin[q][i]).sum()).collect();
            c0max = c0max.max(c0.abs());
            let s = [0, 1, 2].map(|a| (c1[a] * sd.vox[a]).abs() * sigma_os * dims[a] as f64 / TAU);
            for a in 0..3 {
                shmax[a] = shmax[a].max(s[a]);
            }
            let mut before: Vec<(f64, f64)> = Vec::new();
            let mut after: Vec<(f64, f64)> = Vec::new();
            let mut afterq: Vec<(f64, f64)> = Vec::new();
            for (r, &v) in fit_vox.iter().enumerate().step_by(5) {
                let p = sd.voxel_log(v);
                let b = basis10(&p);
                let ph: f64 = (0..4).map(|q| wb[q] * grid.mono[v][q]).sum();
                let lin: f64 = (0..4).map(|q| wb[q] * (0..4).map(|i| th_lin[q][i] * b[i]).sum::<f64>()).sum();
                let quad: f64 = (0..4).map(|q| wb[q] * (0..10).map(|i| th_quad[q][i] * b[i]).sum::<f64>()).sum();
                let qt: f64 = (0..4).map(|q| (wb[q] - wref[q]) * (4..10).map(|i| th_quad[q][i] * b[i]).sum::<f64>()).sum();
                qtay = qtay.max(qt.abs());
                before.push((ph, wv[r]));
                after.push((ph - lin, wv[r]));
                afterq.push((ph - quad, wv[r]));
            }
            let span = |x: &mut Vec<(f64, f64)>| wquant(x, 0.999) - wquant(x, 0.001);
            rng_b = rng_b.max(span(&mut before));
            rng_a = rng_a.max(span(&mut after));
            rng_q = rng_q.max(span(&mut afterq));
        }
        println!("conc0 encode {e}: constant part ≤ {c0max:.3} rad, linear part ≤ [{:.3}, {:.3}, {:.3}] cells (shift); signal-weighted 0.1–99.9 % phase span over the readout: {rng_b:.3} rad → {rng_a:.3} rad without const + lin → {rng_q:.4} rad without quad; quad Taylor |ΔQ| ≤ {qtay:.3} rad", shmax[0], shmax[1], shmax[2]);
        conc_info.push(json!({ "enc": e, "c0_max_rad": c0max, "lin_shift_cells": shmax, "span_rad": rng_b, "span_after_lin_rad": rng_a, "span_after_quad_rad": rng_q, "quad_taylor_rad": qtay }));
    }

    // ── control: synthetic exactly constant + linear ratemap vs zero field (knot 0's decay and fat)
    let mut control_info = json!(null);
    if control {
        let f0 = ts.frame(&ts.corners(kstate[0].0, kstate[0].1), &sd, &grid.x_log, &grid.omega);
        let om_syn: Vec<f64> = (0..nv).map(|v| { let p = sd.voxel_log(v); TAU * (40.0 + 1500.0 * p[1] + 800.0 * p[2]) }).collect();
        let p_syn = fit_field_poly_at(&pos, &om_syn, &fit_vox, Some(&wv), 1, tau_ref);
        let mut rows = Vec::new();
        for &e in &encs {
            let conc0 = segment_phase(&grid, &wbar[e], &tau_adc, false);
            let om_zero = vec![0.0f64; nv];
            let none = FieldPoly::none();
            let gctl = |om: &[f64], poly: &FieldPoly, v: usize, t: f64| -> Complex64 {
                let p = sd.voxel_log(v);
                let psi = poly.residual_at(&p, conc0(v, t) + om[v] * t, t);
                let ffv = f0.ff[v] as f64;
                let wat = (1.0 - ffv) * (-(f0.rw[v] as f64) * (t - ta)).exp();
                let fat = ffv * (-(f0.rf[v] as f64) * (t - ta)).exp();
                Complex64::from_polar(1.0, -psi) * (Complex64::new(wat, 0.0) + fat * cfat(t))
            };
            let (l_zero, t_zero) = spectrum_l(&|v, t| gctl(&om_zero, &none, v, t));
            let (l_syn_a, _) = spectrum_l(&|v, t| gctl(&om_syn, &none, v, t));
            let (l_syn_c, t_syn_c) = spectrum_l(&|v, t| gctl(&om_syn, &p_syn, v, t));
            let dmax = t_zero.iter().zip(&t_syn_c).filter(|(a, _)| **a > 1e-6).map(|(a, b)| ((a - b) / a).abs()).fold(0.0, f64::max);
            println!("control encode {e}: zero field L {}/{}/{} (noise / 1e-3 / 1e-4); synthetic const + lin field, no absorption {}/{}/{}; after per-knot absorption {}/{}/{}; weighted tails above 1e-6 differ by ≤ {dmax:.1e} relative", l_zero.noise, l_zero.e3, l_zero.e4, l_syn_a.noise, l_syn_a.e3, l_syn_a.e4, l_syn_c.noise, l_syn_c.e3, l_syn_c.e4);
            rows.push(json!({ "enc": e, "zero": [l_zero.noise, l_zero.e3, l_zero.e4], "syn_none": [l_syn_a.noise, l_syn_a.e3, l_syn_a.e4], "syn_abs": [l_syn_c.noise, l_syn_c.e3, l_syn_c.e4], "tail_rel_diff": dmax }));
        }
        control_info = json!({ "field": "2π(40 Hz + 1500 Hz/m y + 800 Hz/m z)", "fit_c_hz": p_syn.c[..4].iter().map(|c| c / TAU).collect::<Vec<_>>(), "rows": rows });
    }

    // ── per knot
    let mut knot_rows = Vec::new();
    let tk = Instant::now();
    for b in (0..nk).step_by(knot_stride) {
        let tb = Instant::now();
        let (r_st, c_st, kw) = kstate[b];
        let f = ts.frame(&ts.corners(r_st, c_st), &sd, &grid.x_log, &grid.omega);
        let p1 = fit_field_poly_at(&pos, &f.om, &fit_vox, Some(&wv), 1, tau_ref);
        let p2 = fit_field_poly_at(&pos, &f.om, &fit_vox, Some(&wv), 2, tau_ref);
        let pg = lp_fit(&pos, &f.om, &fit_vox, &wv, lp_p, tau_ref);
        let st_g = field_stats(&|v| f.om[v] - pg.eval_lin(&sd.voxel_log(v)), &fit_vox, &wv);
        let st_a = field_stats(&|v| f.om[v], &fit_vox, &wv);
        let st_b = field_stats(&|v| f.om[v] - poly_glob.eval_lin(&sd.voxel_log(v)), &fit_vox, &wv);
        let st_c = field_stats(&|v| f.om[v] - p1.eval_lin(&sd.voxel_log(v)), &fit_vox, &wv);
        let st_d = field_stats(&|v| f.om[v] - p2.eval(&sd.voxel_log(v)), &fit_vox, &wv);
        let qmax = fit_vox.iter().map(|&v| { let p = sd.voxel_log(v); (p2.eval(&p) - p2.eval_lin(&p)).abs() }).fold(0.0f64, f64::max);
        // signal-weighted 99.9 % of |Q| (the bulk of the Taylor argument)
        let mut qv: Vec<(f64, f64)> = fit_vox.iter().zip(&wv).map(|(&v, &w)| { let p = sd.voxel_log(v); ((p2.eval(&p) - p2.eval_lin(&p)).abs(), w) }).collect();
        let q999 = wquant(&mut qv, 0.999);
        let taylor = qmax * dtmax;
        let mut per_enc = Vec::new();
        let ff_any = f.ff.iter().any(|&x| x > 0.0);
        for &e in &encs {
            let conc0 = segment_phase(&grid, &wbar[e], &tau_adc, false);
            let wb_e = &wbar[e];
            let conc_lin = |v: usize, t: f64| -> f64 {
                let wb = wbar_at(wb_e, &tau_adc, t);
                let bb = basis10(&sd.voxel_log(v));
                (0..4).map(|q| wb[q] * (0..4).map(|i| th_lin[q][i] * bb[i]).sum::<f64>()).sum()
            };
            let conc_quad = |v: usize, t: f64| -> f64 {
                let wb = wbar_at(wb_e, &tau_adc, t);
                let bb = basis10(&sd.voxel_log(v));
                (0..4).map(|q| wb[q] * (0..10).map(|i| th_quad[q][i] * bb[i]).sum::<f64>()).sum()
            };
            // variant h: the linear term (Hz across the FOV per axis) that minimises the weighted tail at L_a − 1
            let gpoly = |poly: &FieldPoly, v: usize, t: f64| -> Complex64 {
                let p = sd.voxel_log(v);
                let psi = poly.residual_at(&p, conc0(v, t) + f.om[v] * t, t);
                let ffv = f.ff[v] as f64;
                let wat = (1.0 - ffv) * (-(f.rw[v] as f64) * (t - ta)).exp();
                let ft = ffv * (-(f.rf[v] as f64) * (t - ta)).exp();
                Complex64::from_polar(1.0, -psi) * (Complex64::new(wat, 0.0) + ft * cfat(t))
            };
            let mut ph = FieldPoly::none();
            let mut opt_info = json!(null);
            if opt_stride > 0 && (b / knot_stride) % opt_stride == 0 && e == encs[0] {
                let to_poly = |x: &[f64]| -> FieldPoly {
                    let mut c = [0.0; 10];
                    for a in 0..3 {
                        c[1 + a] = TAU * x[a] / (dims[a] as f64 * sd.vox[a]);
                    }
                    FieldPoly { deg: 1, c, tau_ref }
                };
                let none = FieldPoly::none();
                let ev_a = weighted_spectrum(&seg_fit, Some(&seg_ew), &ts_n, &tw_n, &|v, t| gpoly(&none, v, t));
                let l_a = seg_count_for(&ev_a, tol, 2, 64);
                let l0 = l_a.saturating_sub(1).max(2);
                let obj0 = tail_errors(&ev_a)[l0];
                let mut obj = |x: &[f64]| -> f64 {
                    let pl = to_poly(x);
                    let ev = weighted_spectrum(&seg_fit, Some(&seg_ew), &ts_n, &tw_n, &|v, t| gpoly(&pl, v, t));
                    tail_errors(&ev)[l0].max(1e-300).ln()
                };
                let x_lsq: Vec<f64> = (0..3).map(|a| p1.c[1 + a] * dims[a] as f64 * sd.vox[a] / TAU).collect();
                let (f_lsq, f_zero) = (obj(&x_lsq), obj(&[0.0, 0.0, 0.0]));
                let x0 = if f_lsq < f_zero { x_lsq.clone() } else { vec![0.0; 3] };
                let (xb, fb, nev) = nelder_mead(&mut obj, &x0, 60.0, 45);
                ph = to_poly(&xb);
                opt_info = json!({ "L0": l0, "tail_a": obj0, "tail_lsq": f_lsq.exp(), "tail_opt": fb.exp(), "x_hz_fov": xb, "evals": nev });
            }
            let psi_v = |var: char, v: usize, t: f64| -> f64 {
                let p = sd.voxel_log(v);
                let raw = conc0(v, t) + f.om[v] * t;
                match var {
                    'a' => raw,
                    'b' => poly_glob.residual_at(&p, raw, t),
                    'c' => p1.residual_at(&p, raw, t),
                    'd' => p2.residual_at(&p, raw, t),
                    'e' => p1.residual_at(&p, raw, t) - conc_lin(v, t),
                    'f' => p2.residual_at(&p, raw, t) - conc_quad(v, t),
                    'g' => pg.residual_at(&p, raw, t),
                    'h' => ph.residual_at(&p, raw, t),
                    o => panic!("variant {o}"),
                }
            };
            let mut res = serde_json::Map::new();
            let mut line = format!("  knot {b:>3} enc {e}:");
            let mut vars = variants.clone();
            if !opt_info.is_null() {
                vars.push('h');
            }
            for &var in &vars {
                for fat in [true, false] {
                    if !fat && !(nofat_stride > 0 && (b / knot_stride) % nofat_stride == 0 && "ace".contains(var) && ff_any) {
                        continue;
                    }
                    let g = |v: usize, t: f64| -> Complex64 {
                        let ffv = if fat { f.ff[v] as f64 } else { 0.0 };
                        let wat = (1.0 - ffv) * (-(f.rw[v] as f64) * (t - ta)).exp();
                        let ft = ffv * (-(f.rf[v] as f64) * (t - ta)).exp();
                        Complex64::from_polar(1.0, -psi_v(var, v, t)) * (Complex64::new(wat, 0.0) + ft * cfat(t))
                    };
                    let (l, tails) = spectrum_l(&g);
                    let key = if fat { var.to_string() } else { format!("{var}_nofat") };
                    line += &format!(" {key} {}/{}/{}", l.noise, l.e3, l.e4);
                    res.insert(key, json!({ "L": [l.noise, l.e3, l.e4], "tails": tails }));
                }
            }
            println!("{line}");
            if !opt_info.is_null() {
                println!("    h (tail-optimal linear term at L = {}): tail a {:.3e}, LSQ {:.3e}, optimum {:.3e} at {:?} Hz across the FOV ({} spectra)", opt_info["L0"], opt_info["tail_a"].as_f64().unwrap(), opt_info["tail_lsq"].as_f64().unwrap(), opt_info["tail_opt"].as_f64().unwrap(), opt_info["x_hz_fov"].as_array().unwrap().iter().map(|x| (x.as_f64().unwrap() * 10.0).round() / 10.0).collect::<Vec<_>>(), opt_info["evals"]);
            }
            per_enc.push(json!({ "enc": e, "res": res, "opt": opt_info }));
        }
        let sh1 = cells(&p1.c[1..4], tw1);
        let sh1r = cells(&p1.c[1..4], dtmax);
        println!(
            "knot {b:>3} (r {r_st:.3}, c {:.3} s, weight {kw:.0}): fit {:.1} Hz + [{:.1}, {:.1}, {:.1}] Hz/m, shift [{:.2}, {:.2}, {:.2}] cells (τ from excitation; [{:.2}, {:.2}, {:.2}] from the readout centre); field 0.1–99.9 % {:.0}..{:.0} Hz → b {:.0}..{:.0}, c {:.0}..{:.0}, d {:.0}..{:.0}, g {:.0}..{:.0}; quad Taylor {taylor:.3} rad (99.9 % {:.3}) ({:.1}s)",
            c_st, hz(p1.c[0]), hz(p1.c[1]), hz(p1.c[2]), hz(p1.c[3]), sh1[0], sh1[1], sh1[2], sh1r[0], sh1r[1], sh1r[2], st_a[2], st_a[3], st_b[2], st_b[3], st_c[2], st_c[3], st_d[2], st_d[3], st_g[2], st_g[3], q999 * dtmax, tb.elapsed().as_secs_f64()
        );
        knot_rows.push(json!({
            "knot": b, "r_index": b / mc, "c_index": b % mc, "state_r": r_st, "state_c_s": c_st, "basis_weight": kw,
            "p1_hz": p1.c[..4].iter().map(|c| c / TAU).collect::<Vec<_>>(), "p2_hz": p2.c.iter().map(|c| c / TAU).collect::<Vec<_>>(),
            "shift_cells_exc": sh1, "shift_cells_ref": sh1r, "shift2_cells_exc": cells(&p2.c[1..4], tw1),
            "hz_per_voxel": ([0, 1, 2].map(|a| p1.c[1 + a] * sd.vox[a] / TAU)), "hz_across_fov": ([0, 1, 2].map(|a| p1.c[1 + a] * sd.vox[a] * dims[a] as f64 / TAU)),
            "field_stats_hz": { "a": st_a, "b": st_b, "c": st_c, "d": st_d, "g": st_g }, "pg_hz": pg.c[..4].iter().map(|c| c / TAU).collect::<Vec<_>>(), "quad_taylor_rad": taylor, "quad_taylor_999_rad": q999 * dtmax,
            "encodes": per_enc,
        }));
        if let Some(p) = &json_out {
            let doc = json!({
                "sim": dir.display().to_string(), "warm": warm_dir.display().to_string(), "grid": dims, "vox": sd.vox, "knots2d": knots_spec, "resp": resp_src,
                "vcoils": nvc, "seg_fit_stride": fit_stride, "seg_fit_voxels": seg_fit.len(), "fit_voxels": fit_vox.len(), "tol_noise": tol, "gain": gain, "margin": margin,
                "tau_adc_ms": [tw0 * 1e3, tw1 * 1e3], "tau_ref_ms": tau_ref * 1e3, "sigma_os": sigma_os, "time_grid": ts_n.len(), "encodes": encs,
                "static_rule_L": l_static, "static_field_stats_hz": static_stats,
                "global_fit_hz": poly_glob.c[..4].iter().map(|c| c / TAU).collect::<Vec<_>>(), "global_shift_cells": cells(&poly_glob.c[1..4], tw1),
                "conc": conc_info, "control": control_info, "knots": knot_rows, "elapsed_s": t_all.elapsed().as_secs_f64(),
            });
            std::fs::write(p, serde_json::to_string(&doc).unwrap()).unwrap();
        }
    }
    println!("{} knots in {:.0}s; done in {:.0}s", knot_rows.len(), tk.elapsed().as_secs_f64(), t_all.elapsed().as_secs_f64());
}
