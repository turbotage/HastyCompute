//! First end-to-end reconstruction of the synthetic spiral 4D-flow data: time-averaged CG-SENSE per encode (no
//! density compensation), physics model ablation, velocity error against the truth.
//!   recon_m0 <sim_dir> [--iters n] [--vcoils n] [--variants plain,b0,full] [--gnl both|on|off] [--tol t]
//! Variants: plain = no phase model; b0 = off-resonance; full = off-resonance + concomitant fields. GNL on = image in
//! apparent coordinates + inverse warp + J⁻¹ velocity correction; off = GNL ignored.

use std::path::PathBuf;
use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_mrsim::{mat_vec, transpose};
use hasty_recon::cg::pcgls;
use hasty_recon::precond::CirculantPrecond;
use hasty_recon::data::{inv3, PhysicsGrid, SampleSet, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_recon::skeleton;
use nalgebra::DMatrix;
use num_complex::Complex64;

/// coil compression: unitary rows (nvc × nc) from the SVD of a subsample of the whitened data
fn compression(set: &SampleSet, nc: usize, nvc: usize) -> Vec<Vec<Complex64>> {
    let m = set.y.len() / nc;
    let step = (m / 20000).max(1);
    let cols: Vec<usize> = (0..m).step_by(step).collect();
    let a = DMatrix::<Complex64>::from_fn(nc, cols.len(), |c, j| {
        let v = set.y[c * m + cols[j]];
        Complex64::new(v.re as f64, v.im as f64)
    });
    let svd = (&a * a.adjoint()).svd(true, false);
    let u = svd.u.unwrap();
    let tot: f64 = svd.singular_values.iter().sum();
    let kept: f64 = svd.singular_values.iter().take(nvc).sum();
    eprintln!("  coil compression {nc} -> {nvc}: {:.4} of the energy", kept / tot);
    (0..nvc).map(|v| (0..nc).map(|c| u[(c, v)].conj()).collect()).collect()
}

fn apply_cc(x: &[C32], len: usize, nc: usize, t: &[Vec<Complex64>]) -> Vec<C32> {
    let nvc = t.len();
    let mut out = vec![C32::new(0.0, 0.0); nvc * len];
    for v in 0..nvc {
        for i in 0..len {
            let mut acc = Complex64::new(0.0, 0.0);
            for c in 0..nc {
                let s = x[c * len + i];
                acc += t[v][c] * Complex64::new(s.re as f64, s.im as f64);
            }
            out[v * len + i] = C32::new(acc.re as f32, acc.im as f32);
        }
    }
    out
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(15);
    let nvc: usize = take("--vcoils").map(|s| s.parse().unwrap()).unwrap_or(12);
    let tol: f64 = take("--tol").map(|s| s.parse().unwrap()).unwrap_or(3e-3);
    let variants: Vec<String> = take("--variants").unwrap_or("plain,b0,full".into()).split(',').map(|s| s.to_string()).collect();
    let gnl_modes: Vec<bool> = match take("--gnl").as_deref() {
        Some("on") => vec![true],
        Some("off") => vec![false],
        _ => vec![false, true],
    };
    let dir = PathBuf::from(&args[0]);
    let t0 = Instant::now();
    let sd = SimData::load(&dir).unwrap();
    let nv = sd.nv();
    let nc = sd.nc;
    let w = sd.whitener();
    println!("loaded {} readouts x {} ADC x {} coils, recon grid {:?} ({:.1}s)", sd.acq.readouts.len(), sd.na, nc, sd.dims, t0.elapsed().as_secs_f64());
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let sets: Vec<SampleSet> = (0..4).map(|e| sd.sample_set(&sd.readouts_of(e, None), &w)).collect();
    let cc = compression(&sets[0], nc, nvc.min(nc));
    let nvc = cc.len();
    // time-averaged truth (logical components), lumen fraction
    let nf = sd.n_frames;
    // matched to what a time-averaged phase-contrast image measures: VENC/π · arg Σ_f exp(iπ v_f / VENC)
    let vavg: Vec<f32> = (0..3 * nv).map(|i| {
        let (a, q) = (i / nv, i % nv);
        let z: Complex64 = (0..nf).map(|f| Complex64::from_polar(1.0, std::f64::consts::PI * sd.truth_vel[q + nv * (a + 3 * f)] as f64 / sd.venc)).sum();
        (z.arg() * sd.venc / std::f64::consts::PI) as f32
    }).collect();
    // support from a plain adjoint image of the reference encode
    let support: Vec<usize> = {
        let grid = sd.physics_grid(false, &w);
        let sm = apply_cc(&grid.smaps, nv, nc, &cc);
        let s0 = &sets[0];
        let m = s0.tau.len();
        let mut op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&s0.coords[0], &s0.coords[1], &s0.coords[2]], &sm, nvc, vec![], 1e-4).unwrap();
        let y = st.clone_htod(&apply_cc(&s0.y, m, nc, &cc)).unwrap();
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        op.adjoint(&st, &y, &mut x).unwrap();
        let h = st.clone_dtoh(&x).unwrap();
        let mut mags: Vec<f32> = h.iter().map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect();
        let mut sorted = mags.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let thr = 0.05 * sorted[sorted.len() * 99 / 100];
        mags.iter_mut().for_each(|v| *v = if *v > thr { 1.0 } else { 0.0 });
        (0..nv).filter(|&q| mags[q] > 0.0).collect()
    };
    println!("support: {} of {} voxels", support.len(), nv);

    for &gnl in &gnl_modes {
        let grid: PhysicsGrid = sd.physics_grid(gnl, &w);
        let smaps = apply_cc(&grid.smaps, nv, nc, &cc);
        for var in &variants {
            let tv = Instant::now();
            let mut imgs: Vec<Vec<C32>> = Vec::new();
            let mut info = Vec::new();
            for (e, set) in sets.iter().enumerate() {
                let m = set.tau.len();
                let use_conc = var == "full";
                let phase = |j: usize, u: usize| -> f64 {
                    let mut p = grid.omega[u] * set.tau[j];
                    if use_conc {
                        for q in 0..4 {
                            p += set.wq[j][q] * grid.mono[u][q];
                        }
                    }
                    p
                };
                let seg: Vec<(Vec<C32>, Vec<C32>)> = if var == "plain" {
                    vec![]
                } else {
                    let sk = skeleton::build(&phase, m, &support, 1200, tol, 60);
                    let l = sk.rows.len();
                    info.push(format!("L={l} (held-out {:.1e})", sk.heldout_err));
                    let ev = |j: usize, u: usize| Complex64::from_polar(1.0, -phase(j, u));
                    // a(j, l) = Σ_l' Φ(j, cols_l') G⁺(l', l)
                    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
                    let chunk = m.div_ceil(nthreads);
                    let parts: Vec<Vec<Vec<C32>>> = std::thread::scope(|sc| {
                        let hs: Vec<_> = (0..nthreads).map(|t| {
                            let (sk, ev) = (&sk, &ev);
                            sc.spawn(move || {
                                let mut a = vec![Vec::new(); l];
                                let mut row = vec![Complex64::new(0.0, 0.0); l];
                                for j in t * chunk..((t + 1) * chunk).min(m) {
                                    for (i, &c) in sk.cols.iter().enumerate() {
                                        row[i] = ev(j, c);
                                    }
                                    for ll in 0..l {
                                        let s: Complex64 = (0..l).map(|i| row[i] * sk.core_pinv[(i, ll)]).sum();
                                        a[ll].push(C32::new(s.re as f32, s.im as f32));
                                    }
                                }
                                a
                            })
                        }).collect();
                        hs.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    (0..l).map(|ll| {
                        let a: Vec<C32> = parts.iter().flat_map(|p| p[ll].iter().copied()).collect();
                        let b: Vec<C32> = (0..nv).map(|u| { let z = ev(sk.rows[ll], u); C32::new(z.re as f32, z.im as f32) }).collect();
                        (a, b)
                    }).collect()
                };
                let mut op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nvc, seg, 1e-4).unwrap();
                let y = st.clone_htod(&apply_cc(&set.y, m, nc, &cc)).unwrap();
                // unbiased preconditioning (no density compensation): circulant inverse of the sampling density
                let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), sd.dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nvc, 1e-3).unwrap();
                let mut x = st.alloc_zeros::<C32>(nv).unwrap();
                let hist = pcgls(&mut op, Some(&mut pre), &st, &y, &mut x, 0.0, iters).unwrap();
                info.push(format!("enc{e}: |grad| {:.1e}", hist.last().unwrap()));
                imgs.push(st.clone_dtoh(&x).unwrap());
            }
            // velocity (apparent, logical components) → true (J⁻¹) → compare with the truth at the true positions
            let rt = transpose(&sd.rot);
            let mut err = [0.0f64; 3];
            let mut bias = [0.0f64; 3];
            let mut tnorm = 0.0;
            let mut n = 0usize;
            let mut vmap = vec![0f32; 3 * nv];
            for q in 0..nv {
                let r0 = Complex64::new(imgs[0][q].re as f64, imgs[0][q].im as f64);
                let mut v = [0.0; 3];
                for a in 0..3 {
                    let ra = Complex64::new(imgs[a + 1][q].re as f64, imgs[a + 1][q].im as f64);
                    // signal ∝ exp(−iγM1·v): the image phase difference is −π v / VENC
                    v[a] = -(ra * r0.conj()).arg() * sd.venc / std::f64::consts::PI;
                }
                if gnl {
                    let vp = mat_vec(&sd.rot, &v);
                    v = mat_vec(&rt, &mat_vec(&inv3(&grid.jac[q]), &vp));
                }
                for a in 0..3 {
                    vmap[a * nv + q] = v[a] as f32;
                }
                let xl = grid.x_log[q];
                if sd.sample_log(&sd.truth_frac, 0, &xl) < 0.9 {
                    continue;
                }
                n += 1;
                for a in 0..3 {
                    let t = sd.sample_log(&vavg, a, &xl);
                    err[a] += (v[a] - t).powi(2);
                    bias[a] += v[a] - t;
                    tnorm += t * t;
                }
            }
            let rmse = ((err[0] + err[1] + err[2]) / n as f64).sqrt();
            let rel = ((err[0] + err[1] + err[2]) / tnorm).sqrt();
            println!(
                "GNL {:<3} {:<5}: velocity RMSE {:.3} m/s (rel {:.3}) over {} lumen voxels; bias [{:.3}, {:.3}, {:.3}] m/s; {:.0}s | {}",
                if gnl { "on" } else { "off" }, var, rmse, rel, n, bias[0] / n as f64, bias[1] / n as f64, bias[2] / n as f64, tv.elapsed().as_secs_f64(), info.join(", ")
            );
            let bytes: Vec<u8> = vmap.iter().flat_map(|v| v.to_le_bytes()).collect();
            let name = format!("recon_vel_{}_{}.npy", var, if gnl { "gnl" } else { "nognl" });
            hasty_phantom::volume::write_npy(&dir.join(name), "<f4", &[sd.dims[0], sd.dims[1], sd.dims[2], 3], true, &bytes).unwrap();
            let mag: Vec<u8> = imgs[0].iter().flat_map(|c| ((c.re * c.re + c.im * c.im).sqrt()).to_le_bytes()).collect();
            hasty_phantom::volume::write_npy(&dir.join(format!("recon_mag_{}_{}.npy", var, if gnl { "gnl" } else { "nognl" })), "<f4", &sd.dims, true, &mag).unwrap();
        }
    }
}
