//! Time-segment budget on the real operator (GPU): seconds per set and data-space model error for each
//! (B0 absorption degree, segment basis, L), on one set of a schedule-based sim (real trajectory, real field, fat).
//!
//!   segbudget <sim_dir> --mask <npy> --x <npy> [--enc 0] [--frame 0] [--set-readouts 190] [--vcoils 31]
//!             [--cfgs ABS:BASIS:L[;..]] [--ref ABS:BASIS:L] [--fat none|truth] [--r2s off|truth] [--shim none|lin|quad]
//!             [--reps 3] [--direct-vox 3000] [--direct-ros 4] [--tune 0|1] [--json out.json] [--rule-only]
//!
//! BASIS: fixed (uniform nodes, LS interpolators: today's default) | fixed_e (the same nodes, energy-weighted LS) |
//! cheb (Chebyshev nodes) | svd | svdw
//! (`build_segments_svd_weighted` as `--seg-basis svd|svdw`, count weights) | svdn | svdn_e | svdn_m (noise-referenced:
//! `build_segments_svd_grid` on `seg_time_grid` with per-ADC weights 1 + SNR(n); voxel weights count / smoothed
//! energy |x̄|²·Σ|S|² / mass). ABS: 0 none, 1 = linear B0 part as k-shift + phase (exact), 2 = + quadratic part via ΔC
//! (Taylor). `L` may be `auto` for the svdn bases: the noise rule of `--nseg auto:noise[:R]` (smallest L whose
//! weighted tail predicts ≤ R / 1.5 noise rms, R = HASTY_SEG_R, default 0.2, margin HASTY_SEG_MARGIN) on the same
//! weights; `--rule-only` prints the unscaled prediction.
//!
//! Measured per configuration: forward + adjoint per set (CUDA events, median of --reps after a warm-up; kernel tiles
//! autotuned unless --tune 0), y = A x for the warm-start image x (scaled to the data: ‖A x‖ matched to the set's
//! measured k-space centre), the data-space error against the reference configuration R = ‖y − y_ref‖ / (σ √(M C))
//! (σ = the sim's noise per whitened coil: error rms in units of the noise rms), the relative error ‖y − y_ref‖/‖y_ref‖,
//! and the error against an exact f64 direct sum (true per-sample concomitant coefficients, fat, R2*) for x restricted
//! to `--direct-vox` energy-sampled voxels on the samples of `--direct-ros` readouts.
//! --shim lin|quad adds a synthetic shim residual (±150 Hz along y and ±100 Hz along z / a Z2-like ±150 Hz quadratic
//! over the tissue) to the field, for the absorption study.

use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments_complex, build_segments_nodes, build_segments_svd_grid, build_segments_svd_weighted, cheb_nodes, nodes_weighted_error, noise_time_weights, seg_count_for, seg_time_grid, snr_time_weights, tail_errors, vol_energy, weighted_spectrum, SegNufft, SegNufftOpts};
use hasty_recon::coils::{CoilCompression, CoilGram, CoilRule};
use hasty_recon::data::{read_npy, SimData};
use hasty_recon::kload::LoadOpts;
use hasty_recon::segphys::{absorb_field_poly, fit_field_poly, fit_monomials, mean_coefficients, seg_tables, segment_phase, FieldPoly};
use num_complex::Complex64;
use rayon::prelude::*;

#[derive(Clone, Debug)]
struct Cfg {
    abs: u32,
    basis: String,
    l: Option<usize>,
}

fn parse_cfg(s: &str) -> Cfg {
    let f: Vec<&str> = s.split(':').collect();
    assert!(f.len() == 3, "config ABS:BASIS:L, got {s}");
    Cfg { abs: f[0].parse().unwrap(), basis: f[1].to_string(), l: if f[2] == "auto" { None } else { Some(f[2].parse().unwrap()) } }
}

fn box_smooth(acc: &mut Vec<f32>, d: [usize; 3]) {
    let nv = acc.len();
    for (axis, r) in [(0usize, 4usize), (1, 4), (2, 1)] {
        let (n, stride) = (d[axis], [1, d[0], d[0] * d[1]][axis]);
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
    let _ = nv;
}

fn norm2(v: &[C32]) -> f64 {
    v.par_iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum()
}
fn diff2(a: &[C32], b: &[C32]) -> f64 {
    a.par_iter().zip(b.par_iter()).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // --rule-only: CPU only, the predicted R(L) of the noise rule for every configuration (no GPU)
    let rule_only = args.iter().position(|a| a == "--rule-only").map(|i| { args.remove(i); }).is_some();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let mask_p = PathBuf::from(take("--mask").expect("--mask <npy>"));
    let x_p = PathBuf::from(take("--x").expect("--x <npy>"));
    let enc: usize = take("--enc").map_or(0, |s| s.parse().unwrap());
    let frame: usize = take("--frame").map_or(0, |s| s.parse().unwrap());
    let set_ro: usize = take("--set-readouts").map_or(190, |s| s.parse().unwrap());
    let nvc_arg: usize = take("--vcoils").map_or(31, |s| s.parse().unwrap());
    let cfgs: Vec<Cfg> = take("--cfgs").unwrap_or("0:fixed:17;0:svdn_e:10".into()).split(';').filter(|s| !s.is_empty()).map(parse_cfg).collect();
    let refc = parse_cfg(&take("--ref").unwrap_or("0:fixed:28".into()));
    let fat = take("--fat").unwrap_or("none".into());
    let r2s = take("--r2s").unwrap_or("off".into());
    let shim = take("--shim").unwrap_or("none".into());
    let reps: usize = take("--reps").map_or(3, |s| s.parse().unwrap());
    let ndv: usize = take("--direct-vox").map_or(3000, |s| s.parse().unwrap());
    let ndr: usize = take("--direct-ros").map_or(4, |s| s.parse().unwrap());
    let tune = take("--tune").map_or(true, |s| s != "0");
    let json_out = take("--json").map(PathBuf::from);
    // the noise rule of `--nseg auto:noise[:R]`: predicted data error ≤ R noise rms
    let r_target: f64 = std::env::var("HASTY_SEG_R").ok().and_then(|v| v.parse().ok()).unwrap_or(0.2);
    let dir = PathBuf::from(&args[0]);
    let t_all = Instant::now();

    let sd = SimData::load_lazy(&dir).unwrap();
    let (nv, nc, na, dims) = (sd.nv(), sd.nc, sd.na, sd.dims);
    let sigma = sd.meta["noise_sigma"].as_f64().unwrap();
    let w = sd.whitener();
    let t0 = Instant::now();
    let mut grid = sd.physics_grid(true, &w);
    println!("{}: {:?}, {} coils, {} ADC samples; physics grid (GNL-aware) {:.0}s", dir.display(), dims, nc, na, t0.elapsed().as_secs_f64());
    let mask: Vec<f32> = read_npy(&mask_p).unwrap().f32s();
    assert_eq!(mask.len(), nv, "--mask on the reconstruction grid");
    let fit_vox: Vec<usize> = (0..nv).filter(|&q| mask[q] > 0.5).collect();
    let seg_fit: Vec<usize> = fit_vox.iter().copied().step_by(3).collect();
    // virtual coils from the whitened maps over the mask
    let cc = {
        let mut g = CoilGram::new(nc);
        let nf = fit_vox.len();
        let mut blk = vec![C32::new(0.0, 0.0); nc * nf];
        for c in 0..nc {
            for (r, &q) in fit_vox.iter().enumerate() {
                blk[c * nf + r] = grid.smaps[c * nv + q];
            }
        }
        g.add(&blk, nf, 1);
        CoilCompression::from_gram(&g, CoilRule::Fixed(nvc_arg))
    };
    let nvc = cc.nvc();
    let smaps = cc.apply(&grid.smaps, nv);
    println!("coil maps: {}", cc.report());
    let coil_e: Vec<f64> = (0..nv).map(|q| (0..nvc).map(|c| { let z = smaps[c * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect();
    // image: warm start (encode enc) and the smoothed magnitude over all encodes
    let xa = read_npy(&x_p).unwrap().c32s();
    assert!(xa.len() >= (enc + 1) * nv, "--x: complex64 (dims…, encodes) in Fortran order");
    let ne_x = xa.len() / nv;
    let x_img: Vec<C32> = xa[enc * nv..(enc + 1) * nv].to_vec();
    let mut mag = vec![0f32; nv];
    for e in 0..ne_x {
        for q in 0..nv {
            mag[q] += xa[e * nv + q].re.hypot(xa[e * nv + q].im) / ne_x as f32;
        }
    }
    box_smooth(&mut mag, dims);
    // the set: first set_ro readouts of (enc, frame)
    let ros_all = sd.readouts_of(enc, Some(frame));
    let ros: Vec<usize> = ros_all.iter().copied().take(set_ro).collect();
    let set = sd.load_set(&ros, &LoadOpts { w: &w, cc: Some(&cc.rows), keep: None, data: true }).unwrap();
    let m = set.tau.len();
    let tau_adc: Vec<f64> = set.tau[..na].to_vec();
    let (tw0, tw1, ntab) = (tau_adc[0], tau_adc[na - 1], na.max(256));
    println!("set: encode {enc}, frame {frame}: {} readouts ({} of the frame), {m} samples, {nvc} virtual coils; noise σ {sigma:.2} per whitened coil", ros.len(), ros_all.len());
    // synthetic shim residual
    if shim != "none" {
        let sig: Vec<usize> = fit_vox.iter().copied().filter(|&q| mag[q] > 0.3 * mag.iter().cloned().fold(0.0f32, f32::max) * 0.1).collect();
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for &q in &sig {
            let p = sd.voxel_log(q);
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let tp = std::f64::consts::TAU;
        for q in 0..nv {
            let p = sd.voxel_log(q);
            let u: [f64; 3] = std::array::from_fn(|a| (p[a] - 0.5 * (lo[a] + hi[a])) / (0.5 * (hi[a] - lo[a])).max(1e-3));
            grid.omega[q] += tp * match shim.as_str() {
                "lin" => 150.0 * u[1] + 100.0 * u[2],
                "quad" => 150.0 * (2.0 * u[2] * u[2] - u[0] * u[0] - u[1] * u[1]) / 2.0,
                o => panic!("--shim none|lin|quad, got {o}"),
            };
        }
        println!("synthetic shim residual {shim} added over the tissue box {:?} – {:?} m", lo.map(|v| (v * 1e3).round() / 1e3), hi.map(|v| (v * 1e3).round() / 1e3));
    }
    let fit = fit_monomials(&sd, &grid, &fit_vox);
    let wbar = mean_coefficients(&[&set], na);
    // R2*, fat
    let r2v: Option<Vec<f64>> = (r2s == "truth").then(|| read_npy(&dir.join("truth_r2s.npy")).unwrap().f32s()[..nv].iter().map(|v| if v.is_finite() { *v as f64 } else { 0.0 }).collect());
    let ff: Option<Vec<f64>> = (fat == "truth").then(|| read_npy(&dir.join("truth_fat_fraction.npy")).unwrap().f32s()[..nv].iter().map(|v| (*v as f64).clamp(0.0, 1.0)).collect());
    let fs = hasty_qfit::waterfat::FatSpectrum::six_peak();
    let ctab: Vec<Complex64> = tau_adc.iter().map(|&t| fs.basis(sd.b0, t)).collect();
    let cfat = |t: f64| -> Complex64 { fs.basis(sd.b0, t) };
    let _ = &ctab;
    // weights: smoothed energy (|x̄|² Σ|S|²), mass (|x̄| / median clamped [0.3, 3]) over the fit voxels; SNR time weights
    let e_vox: Vec<f64> = seg_fit.iter().map(|&q| (mag[q] as f64).powi(2) * coil_e[q]).collect();
    let med = |v: &[f64]| { let mut s = v.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[s.len() / 2].max(1e-30) };
    let em = med(&e_vox);
    let w_energy: Vec<f64> = e_vox.iter().map(|v| v / em).collect();
    let mm: Vec<f64> = seg_fit.iter().map(|&q| mag[q] as f64).collect();
    let mmed = med(&mm);
    let w_mass: Vec<f64> = mm.iter().map(|v| (v / mmed).clamp(0.3, 3.0)).collect();
    let e_adc = sd.adc_energy_whitened(&ros_all.iter().copied().step_by(3).take(128).collect::<Vec<_>>(), &w).unwrap();
    let kap: Vec<f64> = (0..na).map(|n| (0..3).map(|a| (set.coords[a][n] as f64 / sd.vox[a]).powi(2)).sum::<f64>().sqrt()).collect();
    let snr_w = if std::env::var("HASTY_SEG_WEIGHTS").as_deref() == Ok("snr") { snr_time_weights(&e_adc, nc as f64 * sigma * sigma) } else { noise_time_weights(&e_adc, &kap, nc as f64 * sigma * sigma) };
    println!("white floor E_vol / (C σ²) = {:.1}", vol_energy(&e_adc, &kap) / (nc as f64 * sigma * sigma));
    let (ts_n, tw_n) = seg_time_grid(&tau_adc, &snr_w, 96);
    let margin: f64 = std::env::var("HASTY_SEG_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.5);
    let tol = r_target / ((snr_w.iter().sum::<f64>() / na as f64).sqrt() * margin);
    let emax = e_adc.iter().cloned().fold(0.0f64, f64::max);
    let data_w = |t: f64| -> f64 { let i = (((t - tau_adc[0]) / (tau_adc[1] - tau_adc[0])).round().max(0.0) as usize).min(na - 1); e_adc[i] / emax };
    println!("SNR per ADC sample: first {:.3e}, mid {:.3e}, last {:.3e}; time grid {} samples", snr_w[0] - 1.0, snr_w[na / 2] - 1.0, snr_w[na - 1] - 1.0, ts_n.len());
    let psi0 = segment_phase(&grid, &wbar, &tau_adc, true);
    if rule_only {
        // CPU only: the predicted data error R(L) = ε_E(L) √(mean w) of each configuration (no operator)
        let gain = (snr_w.iter().sum::<f64>() / na as f64).sqrt();
        let mut out = Vec::new();
        for cfg in std::iter::once(&refc).chain(cfgs.iter()) {
            let tc = Instant::now();
            let poly = if cfg.abs > 0 {
                let wv: Vec<f64> = fit_vox.iter().map(|&q| (mag[q] as f64).powi(2) * coil_e[q]).collect();
                fit_field_poly(&sd, &grid, &fit_vox, Some(&wv), cfg.abs, 0.5 * (tw0 + tw1))
            } else {
                FieldPoly::none()
            };
            let psi = poly.residual(&sd, &psi0);
            let g = |v: usize, t: f64| -> Complex64 {
                let mut z = Complex64::from_polar(1.0, -psi(v, t));
                if let Some(r) = &r2v {
                    z *= (-r[v] * (t - tau_adc[0])).exp();
                }
                if let Some(f) = &ff {
                    z *= Complex64::new(1.0 - f[v], 0.0) + f[v] * cfat(t);
                }
                z
            };
            let ls: Vec<usize> = (2..=26).collect();
            let curve: Vec<f64> = match cfg.basis.as_str() {
                "svdn_e" => { let ev = weighted_spectrum(&seg_fit, Some(&w_energy), &ts_n, &tw_n, &g); let t = tail_errors(&ev); ls.iter().map(|&l| t[l] * gain).collect() }
                "fixed" | "fixed_e" | "cheb" => ls.iter().map(|&l| {
                    let tl: Vec<f64> = if cfg.basis == "cheb" { cheb_nodes(l, tw0, tw1) } else { (0..l).map(|i| tw0 + (tw1 - tw0) * i as f64 / (l - 1) as f64).collect() };
                    let fw: Option<&[f64]> = if cfg.basis == "fixed_e" { Some(&w_energy) } else { None };
                    nodes_weighted_error(&seg_fit, fw, Some(&w_energy), &ts_n, &tw_n, &tl, &g) * gain
                }).collect(),
                o => panic!("--rule-only: bases svdn_e | fixed | fixed_e | cheb, got {o}"),
            };
            let pick = |r: f64| ls.iter().zip(&curve).find(|(_, &e)| e <= r).map(|(&l, _)| l);
            println!("abs {} {:>7}: predicted R(L) {} | L for R ≤ 0.1 / 0.2 / 0.3: {:?} / {:?} / {:?} ({:.0}s)", cfg.abs, cfg.basis,
                ls.iter().zip(&curve).filter(|(l, _)| **l % 2 == 0 || **l < 12).map(|(l, e)| format!("{l}:{e:.3}")).collect::<Vec<_>>().join(" "), pick(0.1), pick(0.2), pick(0.3), tc.elapsed().as_secs_f64());
            out.push(serde_json::json!({ "abs": cfg.abs, "basis": cfg.basis, "L": ls, "R_pred": curve, "poly": poly.c.to_vec() }));
        }
        if let Some(p) = &json_out {
            std::fs::write(p, serde_json::to_string_pretty(&serde_json::json!({ "rule_only": true, "gain": gain, "fat": fat, "r2s": r2s, "shim": shim, "results": out })).unwrap()).unwrap();
        }
        return;
    }
    // energy-sampled voxel subset for the direct sum
    let dvox: Vec<usize> = {
        let tot: f64 = fit_vox.iter().map(|&q| (mag[q] as f64).powi(2) * coil_e[q]).sum();
        let mut acc = 0.0;
        let mut out = Vec::new();
        let step = tot / ndv as f64;
        let mut next = 0.5 * step;
        for &q in &fit_vox {
            acc += (mag[q] as f64).powi(2) * coil_e[q];
            while acc >= next && out.len() < ndv {
                if out.last() != Some(&q) {
                    out.push(q);
                }
                next += step;
            }
        }
        out
    };
    let dros: Vec<usize> = ros.iter().copied().step_by((ros.len() / ndr.max(1)).max(1)).take(ndr).collect();
    let dset = sd.load_set(&dros, &LoadOpts { w: &w, cc: Some(&cc.rows), keep: None, data: false }).unwrap();
    let md = dset.tau.len();
    // exact direct sum on (dvox, dset)
    let half: [f64; 3] = std::array::from_fn(|a| (dims[a] / 2) as f64);
    let direct: Vec<C32> = {
        let t = Instant::now();
        let fac = |v: usize, t: f64| -> Complex64 {
            let mut z = Complex64::new(1.0, 0.0);
            if let Some(r) = &r2v {
                z *= (-r[v] * (t - tau_adc[0])).exp();
            }
            if let Some(f) = &ff {
                z *= Complex64::new(1.0 - f[v], 0.0) + f[v] * cfat(t);
            }
            z
        };
        let rows: Vec<Vec<Complex64>> = (0..md)
            .into_par_iter()
            .map(|j| {
                let mut acc = vec![Complex64::new(0.0, 0.0); nvc];
                for &v in &dvox {
                    let (i, jj, k) = (v % dims[0], (v / dims[0]) % dims[1], v / (dims[0] * dims[1]));
                    let idx = [i as f64 - half[0], jj as f64 - half[1], k as f64 - half[2]];
                    let mut ph = (0..3).map(|a| dset.coords[a][j] as f64 * idx[a]).sum::<f64>() + grid.omega[v] * dset.tau[j];
                    ph += (0..4).map(|q| dset.wq[j][q] * grid.mono[v][q]).sum::<f64>();
                    let e = Complex64::from_polar(1.0, -ph) * fac(v, dset.tau[j]) * Complex64::new(x_img[v].re as f64, x_img[v].im as f64);
                    for c in 0..nvc {
                        let s = smaps[c * nv + v];
                        acc[c] += e * Complex64::new(s.re as f64, s.im as f64);
                    }
                }
                acc
            })
            .collect();
        let mut y = vec![C32::new(0.0, 0.0); nvc * md];
        for (j, r) in rows.iter().enumerate() {
            for c in 0..nvc {
                y[c * md + j] = C32::new(r[c].re as f32, r[c].im as f32);
            }
        }
        println!("direct sum: {} voxels × {md} samples × {nvc} coils in {:.1}s", dvox.len(), t.elapsed().as_secs_f64());
        y
    };
    let mut x_sub = vec![C32::new(0.0, 0.0); nv];
    for &v in &dvox {
        x_sub[v] = x_img[v];
    }

    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let ev = (dev.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap(), dev.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap());
    let dx = st.clone_htod(&x_img).unwrap();
    let dxs = st.clone_htod(&x_sub).unwrap();
    let mut y_ref: Option<Vec<C32>> = None;
    let mut xscale = 1.0f64;
    let mut results = Vec::new();
    let all: Vec<Cfg> = std::iter::once(refc.clone()).chain(cfgs.iter().cloned()).collect();
    for (ci, cfg) in all.iter().enumerate() {
        let tc = Instant::now();
        let poly = if cfg.abs > 0 {
            let wv: Vec<f64> = fit_vox.iter().map(|&q| (mag[q] as f64).powi(2) * coil_e[q]).collect();
            fit_field_poly(&sd, &grid, &fit_vox, Some(&wv), cfg.abs, 0.5 * (tw0 + tw1))
        } else {
            FieldPoly::none()
        };
        let psi = poly.residual(&sd, &psi0);
        let g = |v: usize, t: f64| -> Complex64 {
            let mut z = Complex64::from_polar(1.0, -psi(v, t));
            if let Some(r) = &r2v {
                z *= (-r[v] * (t - tau_adc[0])).exp();
            }
            if let Some(f) = &ff {
                z *= Complex64::new(1.0 - f[v], 0.0) + f[v] * cfat(t);
            }
            z
        };
        let vw: Option<&[f64]> = match cfg.basis.as_str() {
            "svdn_e" => Some(&w_energy),
            "svdn_m" => Some(&w_mass),
            _ => None,
        };
        let mut l = cfg.l.unwrap_or(0);
        let mut tail = f64::NAN;
        if cfg.basis.starts_with("svdn") {
            let evs = weighted_spectrum(&seg_fit, vw, &ts_n, &tw_n, &g);
            if cfg.l.is_none() {
                l = seg_count_for(&evs, tol, 2, 40);
            }
            tail = tail_errors(&evs)[l];
        }
        let (sp, at, ta, dt) = match cfg.basis.as_str() {
            "fixed" => build_segments_complex(nv, l, tw0, tw1, ntab, &seg_fit, &g),
            "cheb" => build_segments_nodes(nv, &cheb_nodes(l, tw0, tw1), tw0, tw1, ntab, &seg_fit, None, &g),
            "fixed_e" => build_segments_nodes(nv, &(0..l).map(|i| if l == 1 { 0.5 * (tw0 + tw1) } else { tw0 + (tw1 - tw0) * i as f64 / (l - 1) as f64 }).collect::<Vec<_>>(), tw0, tw1, ntab, &seg_fit, Some(&w_energy), &g),
            "svd" => build_segments_svd_weighted(nv, l, tw0, tw1, ntab, &seg_fit, None, None, 96, &g),
            "svdw" => build_segments_svd_weighted(nv, l, tw0, tw1, ntab, &seg_fit, None, Some(&data_w), 96, &g),
            "svdn" | "svdn_e" | "svdn_m" => build_segments_svd_grid(nv, l, tw0, tw1, ntab, &seg_fit, vw, &ts_n, &tw_n, &g),
            o => panic!("basis {o}"),
        };
        let t_build = tc.elapsed().as_secs_f64();
        let mut eng = SegNufft::new(&dev, dims, sd.vox, nvc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 2, frames: 1, bspline: 1, max_grid_bytes: 6 << 30 }).unwrap();
        eng.set_maps(&st, &smaps).unwrap();
        let seg = eng.segments(&st, &sp, &at, ta, dt).unwrap();
        let mut tab = seg_tables(&sd, &set, &fit, true, Some(&wbar));
        let qphase = absorb_field_poly(&mut tab, &sd, &poly, &fit_vox);
        let pts = eng.points(&st, [&tab.coords[0], &tab.coords[1], &tab.coords[2]], &tab.tau, &tab.phi0, &tab.dc, None).unwrap();
        let mut dtab = seg_tables(&sd, &dset, &fit, true, Some(&wbar));
        absorb_field_poly(&mut dtab, &sd, &poly, &fit_vox);
        let dpts = eng.points(&st, [&dtab.coords[0], &dtab.coords[1], &dtab.coords[2]], &dtab.tau, &dtab.phi0, &dtab.dc, None).unwrap();
        let (kf, ka) = if tune { eng.autotune(&st, &seg, &pts).unwrap() } else { (f32::NAN, f32::NAN) };
        let mut y = st.alloc_zeros::<C32>(nvc * m).unwrap();
        let mut xo = st.alloc_zeros::<C32>(nv).unwrap();
        let time = |f: &mut dyn FnMut()| -> f64 {
            let mut t = Vec::new();
            for i in 0..=reps {
                ev.0.record(&st).unwrap();
                f();
                ev.1.record(&st).unwrap();
                ev.1.synchronize().unwrap();
                if i >= 1 {
                    t.push(ev.0.elapsed_ms(&ev.1).unwrap() as f64 * 1e-3);
                }
            }
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            t[t.len() / 2]
        };
        let ffts0 = eng.fft_calls();
        let t_fwd = time(&mut || eng.forward(&st, &seg, &pts, &dx, &mut y).unwrap());
        let ffts_f = (eng.fft_calls() - ffts0) / (reps + 1);
        let t_adj = time(&mut || eng.adjoint(&st, &seg, &pts, &y, &mut xo).unwrap());
        eng.forward(&st, &seg, &pts, &dx, &mut y).unwrap();
        let yh = st.clone_dtoh(&y).unwrap();
        let mut ys = st.alloc_zeros::<C32>(nvc * md).unwrap();
        eng.forward(&st, &seg, &dpts, &dxs, &mut ys).unwrap();
        let ysh = st.clone_dtoh(&ys).unwrap();
        let direct_rel = (diff2(&ysh, &direct) / norm2(&direct)).sqrt();
        drop(eng);
        if ci == 0 {
            // scale of the warm start to the data: ‖A x‖ against the measured data over the k-space centre samples
            let ctr: Vec<usize> = (0..m).filter(|&j| { let n = j % na; n < 30 || n + 30 >= na }).collect();
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for c in 0..nvc {
                for &j in &ctr {
                    let (a, b) = (set.y[c * m + j], yh[c * m + j]);
                    num += (a.re as f64).powi(2) + (a.im as f64).powi(2);
                    den += (b.re as f64).powi(2) + (b.im as f64).powi(2);
                }
            }
            xscale = (num / den).sqrt();
            println!("warm start scale to the data: ×{xscale:.4} (k-space centre energy match)");
        }
        let yh: Vec<C32> = yh.iter().map(|z| C32::new((z.re as f64 * xscale) as f32, (z.im as f64 * xscale) as f32)).collect();
        let (r_noise, rel) = match &y_ref {
            None => (0.0, 0.0),
            Some(r) => {
                let d = diff2(&yh, r);
                ((d / (m as f64 * nvc as f64 * sigma * sigma)).sqrt(), (d / norm2(r)).sqrt())
            }
        };
        if y_ref.is_none() {
            y_ref = Some(yh.clone());
        }
        let line = format!(
            "abs {} {:>7} L={l:>2}: fwd {t_fwd:.3} s + adj {t_adj:.3} s = {:.3} s/set ({ffts_f} FFT calls fwd, kernels {kf:.0}/{ka:.0} ms) | R = {r_noise:.4} σ, rel {rel:.2e} | direct-sum rel {direct_rel:.2e} | tail {tail:.2e} | quad Taylor phase {qphase:.2} rad | build {t_build:.0}s",
            cfg.abs,
            cfg.basis,
            t_fwd + t_adj
        );
        println!("{}{line}", if ci == 0 { "[ref] " } else { "" });
        results.push(serde_json::json!({ "ref": ci == 0, "abs": cfg.abs, "basis": cfg.basis, "L": l, "fwd_s": t_fwd, "adj_s": t_adj, "set_s": t_fwd + t_adj, "fft_calls_fwd": ffts_f,
            "kern_fwd_ms": kf, "kern_adj_ms": ka, "R_noise": r_noise, "rel": rel, "direct_rel": direct_rel, "tail": tail, "quad_taylor_rad": qphase, "build_s": t_build, "poly": poly.c.to_vec() }));
        if let Some(p) = &json_out {
            let doc = serde_json::json!({ "sim": dir.display().to_string(), "enc": enc, "frame": frame, "readouts": ros.len(), "samples": m, "vcoils": nvc, "fat": fat, "r2s": r2s, "shim": shim, "sigma": sigma, "xscale": xscale, "tol": tol, "r_target": r_target, "results": results });
            std::fs::write(p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        }
    }
    println!("done in {:.0}s", t_all.elapsed().as_secs_f64());
}
