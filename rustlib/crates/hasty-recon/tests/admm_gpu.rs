//! GPU: the ADMM data step (`hasty_recon::admm`, `P2np::admm_x_step`) on a small `SegSets` case (two encodes, six
//! sets, B0 segments, data staged from a k-space store) against the host reference `GradXStep` (per-set PCG through the
//! objective's streamed gradient only), with and without the clip metric; one full ADMM outer iteration (run_admm with
//! a soft-threshold prox) device vs reference; and the per-sample curvature c = tr(N_t) / (m_t D̄) of the unnormalised
//! NUFFT (= nv for a phase-only physics model).
//!   cargo test --release -p hasty-recon --test admm_gpu -- --nocapture --test-threads 1

use std::f64::consts::PI;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::admm::{clip_symbol, run_admm, samples_per_cell, AdmmCfg, AdmmData, AdmmP2np, ClipMetric, GradXStep};
use hasty_recon::gpu::Kernels;
use hasty_recon::hoststack::HostStack;
use hasty_recon::kstore::{KStore, StoreSpec};
use hasty_recon::p2np::P2np;
use hasty_recon::setop::{SegSets, SetOp};
use hasty_testkit::Rng;
use num_complex::Complex64 as C64;

const DIMS: [usize; 3] = [20, 18, 12];
const NC: usize = 5;
const L: usize = 6;

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

fn smaps(nc: usize, dims: [usize; 3]) -> Vec<C32> {
    let nv: usize = dims.iter().product();
    (0..nc)
        .flat_map(|c| {
            let th = 2.0 * PI * c as f64 / nc as f64;
            (0..nv).map(move |q| {
                let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
                let u = [i as f64 / dims[0] as f64 - 0.5, j as f64 / dims[1] as f64 - 0.5, k as f64 / dims[2] as f64 - 0.5];
                let r2 = (u[0] - 0.5 * th.cos()).powi(2) + (u[1] - 0.5 * th.sin()).powi(2) + u[2].powi(2);
                let z = C64::from_polar((-r2 / 0.3).exp(), 1.2 * (u[0] * th.sin() - u[1] * th.cos()));
                C32::new(z.re as f32, z.im as f32)
            })
        })
        .collect()
}

fn u01(r: &mut Rng) -> f64 {
    0.5 * (1.0 + r.uniform())
}

/// stack-of-spirals readouts: (k·Δ coordinates, τ)
fn readouts(r: &mut Rng, nro: usize, na: usize) -> ([Vec<f32>; 3], Vec<f32>) {
    let mut k: [Vec<f32>; 3] = Default::default();
    let mut tau = vec![];
    for _ in 0..nro {
        let (kz, th) = (0.97 * PI * (2.0 * u01(r) - 1.0), PI * u01(r));
        for i in 0..na {
            let q = i as f64 / (na - 1) as f64;
            let (rad, ang) = (0.97 * PI * q.sqrt(), th + 2.0 * PI * 7.0 * q.sqrt());
            k[0].push((rad * ang.cos()) as f32);
            k[1].push((rad * ang.sin()) as f32);
            k[2].push(kz as f32);
            tau.push((1.6e-3 + 8e-3 * q) as f32);
        }
    }
    (k, tau)
}

/// six sets (frame · 2 + encode) of two encodes with their own B0 segments; the sample coordinates per set
fn sets(dev: &Device, kern: &Arc<Kernels>, r: &mut Rng) -> (SegSets, Vec<[Vec<f32>; 3]>) {
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    let o = SegNufftOpts { sigma: 1.25, width: 10, nseg: L, order: 2, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 };
    let mut eng = SegNufft::new(dev, DIMS, [3e-3; 3], NC, o).unwrap();
    eng.set_maps(&st, &smaps(NC, DIMS)).unwrap();
    let om: Vec<f64> = (0..nv).map(|q| 2.0 * PI * 90.0 * ((q % DIMS[0]) as f64 / DIMS[0] as f64 - 0.5 + 0.3 * ((q / DIMS[0]) % DIMS[1]) as f64 / DIMS[1] as f64)).collect();
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let segs = (0..2)
        .map(|e| {
            let psi = |v: usize, t: f64| om[v] * t * (1.0 + 0.3 * e as f64);
            let (sp, at, t0, dt) = build_segments(nv, L, 1.6e-3, 9.6e-3, 300, &fitv, &psi);
            eng.segments(&st, &sp, &at, t0, dt).unwrap()
        })
        .collect();
    let mut pts = vec![];
    let mut coords = vec![];
    for _ in 0..6 {
        let (k, tau) = readouts(r, 5, 300);
        let m = tau.len();
        let phi0: Vec<f32> = (0..m).map(|_| r.uniform() as f32).collect();
        let dc: Vec<f32> = (0..6 * m).map(|_| (150.0 * (2.0 * r.uniform() - 1.0)) as f32).collect();
        pts.push(eng.points(&st, [&k[0], &k[1], &k[2]], &tau, &phi0, &dc, None).unwrap());
        coords.push(k);
    }
    (SegSets { eng, pts, segs, seg_of_set: (0..6).map(|t| t % 2).collect(), kern: kern.clone(), surrogate: None }, coords)
}

fn object(e: usize) -> Vec<C32> {
    let nv: usize = DIMS.iter().product();
    (0..nv)
        .map(|q| {
            let (i, j, k) = (q % DIMS[0], (q / DIMS[0]) % DIMS[1], q / (DIMS[0] * DIMS[1]));
            let u = [i as f64 / DIMS[0] as f64 - 0.5, j as f64 / DIMS[1] as f64 - 0.5, k as f64 / DIMS[2] as f64 - 0.5];
            let a = if u[0] * u[0] / 0.16 + u[1] * u[1] / 0.12 + u[2] * u[2] / 0.1 < 1.0 { 1.0 + 0.3 * (u[0] * 9.0).sin() } else { 0.0 };
            let z = C64::from_polar(a, 0.8 * e as f64 * u[1]);
            C32::new(z.re as f32, z.im as f32)
        })
        .collect()
}

#[test]
fn admm_data_step_device_matches_reference() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let nv: usize = DIMS.iter().product();
    let mut r = Rng::new(77);
    let (mut sop, coords) = sets(&dev, &kern, &mut r);
    // data = E_t x_e + noise, staged from a RAM store
    let ys: Vec<Vec<C32>> = (0..6)
        .map(|t| {
            let mut y = st.alloc_zeros::<C32>(sop.len_y(t)).unwrap();
            sop.forward(&st, t, &st.clone_htod(&object(t % 2)).unwrap(), &mut y).unwrap();
            let mut h = st.clone_dtoh(&y).unwrap();
            for z in h.iter_mut() {
                z.re += 0.05 * (r.uniform() as f32);
                z.im += 0.05 * (r.uniform() as f32);
            }
            h
        })
        .collect();
    let ks = KStore::create(StoreSpec::parse("ram", std::path::Path::new("/unused")).unwrap(), NC, (0..6).map(|t| sop.pts[t].m).collect(), None, "").unwrap();
    for (t, y) in ys.iter().enumerate() {
        ks.put(t, y).unwrap();
    }
    let ks = Arc::new(ks);
    let lens = vec![nv; 6];
    // c = tr(N_t) / (m_t D̄): Hutchinson with 32 complex ±1 ± i probes per set (|z|² = 2 per voxel)
    let maps = smaps(NC, DIMS);
    let dsum: Vec<f32> = (0..nv).map(|q| (0..NC).map(|c| maps[c * nv + q].re.powi(2) + maps[c * nv + q].im.powi(2)).sum()).collect();
    let dbar = dsum.iter().map(|&v| v as f64).sum::<f64>() / nv as f64;
    let mut cs = vec![];
    for t in [0usize, 3] {
        let mut acc = 0.0;
        for _ in 0..32 {
            let z: Vec<C32> = (0..nv).map(|_| C32::new(if r.uniform() > 0.0 { 1.0 } else { -1.0 }, if r.uniform() > 0.0 { 1.0 } else { -1.0 })).collect();
            let mut e = st.alloc_zeros::<C32>(sop.len_y(t)).unwrap();
            sop.forward(&st, t, &st.clone_htod(&z).unwrap(), &mut e).unwrap();
            acc += kern.dot(&st, &e, &e).unwrap().re;
        }
        cs.push(acc / 32.0 / 2.0 / (sop.pts[t].m as f64 * dbar) / nv as f64);
    }
    // report only (the derivation's check; the runs measure c from their own operator, so a different NUFFT
    // normalisation changes ρ₁'s formula, not the solver)
    let c_ok = cs.iter().all(|&c| (c - 1.0).abs() < 0.1);
    eprintln!("per-sample curvature c / nv = {:.4} {:.4} (unnormalised NUFFT, phase-only physics: 1) — {}", cs[0], cs[1], if c_ok { "CHECK OK" } else { "CHECK FAILED: the NUFFT is not unnormalised as derived" });
    let rho1 = nv as f64 * {
        let mut v: Vec<f32> = dsum.iter().copied().filter(|&d| d > 0.05).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2] as f64
    };
    // the clip metric: per-set samples-per-cell symbols (one group per set, share 1)
    let mus: Vec<Vec<f32>> = coords
        .iter()
        .map(|c| {
            let lc = hasty_recon::precond::chan_spectrum(&dev, &st, DIMS, [&c[0], &c[1], &c[2]]).unwrap();
            let sum: f64 = lc.iter().map(|&v| v as f64).sum();
            let conv = hasty_recon::precond::coil_convolved(&dev, &st, DIMS, &lc, &maps, NC).unwrap();
            samples_per_cell(&conv, sum, c[0].len(), dbar)
        })
        .collect();
    let mx: f32 = mus[0].iter().cloned().fold(0.0, f32::max);
    let tot: f64 = mus[0].iter().map(|&v| v as f64).sum();
    eprintln!("set 0: {} samples, Σ μ̃ = {tot:.1}, max μ̃ {mx:.1}", coords[0][0].len());
    assert!((tot / coords[0][0].len() as f64 - 1.0).abs() < 1e-3, "Σ_k μ̃ must be the sample count");
    let dmed = (rho1 / nv as f64) as f32;
    let dinv: Vec<f32> = dsum.iter().map(|&d| 1.0 / d.max(0.1 * dmed).sqrt()).collect();
    let rho = 0.7 * rho1;
    // a start x, z, u (x = z = the object blurred by a phase, u small)
    let mut x0 = HostStack::zeros(&lens);
    let mut z0 = HostStack::zeros(&lens);
    let mut u0 = HostStack::zeros(&lens);
    for t in 0..6 {
        let o = object(t % 2);
        for q in 0..nv {
            let w = C64::from_polar(0.8, 0.3 * (q % 7) as f64) * C64::new(o[q].re as f64, o[q].im as f64);
            x0.chunk_mut(t)[q] = C32::new(w.re as f32, w.im as f32);
            z0.chunk_mut(t)[q] = C32::new(o[q].re * 0.9, o[q].im * 0.9);
            u0.chunk_mut(t)[q] = C32::new(0.01 * ((q % 5) as f32 - 2.0), 0.0);
        }
    }
    for clip in [false, true] {
        let mk = || ClipMetric::new(&dev, &st, kern.clone(), DIMS, mus.clone(), (0..6).map(|t| (t, 1.0)).collect(), rho1, &dinv).unwrap();
        let mut p = P2np::from_store(&mut sop, hasty_recon::kstage::Stager::new(&dev.ctx, ks.clone()).unwrap(), vec![], &st).unwrap();
        // device per-set data step
        let mut xd = x0.clone();
        let mut dd = AdmmP2np { p: &mut p, st: st.clone(), metric: clip.then(mk) };
        let sd = dd.x_step(&mut xd, &z0, &u0, rho, 3).unwrap();
        // reference: per-set PCG through the streamed gradient, the same metric on host slices
        let mut cm = clip.then(mk);
        let mut xr = x0.clone();
        let srf = {
            let mut g = GradXStep::new(&mut p, st.clone());
            if let Some(cm) = cm.as_mut() {
                let st2 = st.clone();
                g.pre = Some(Box::new(move |t: usize, rv: &[C32], zv: &mut [C32]| {
                    let rd = st2.clone_htod(rv)?;
                    let mut zd: CudaSlice<C32> = st2.alloc_zeros(rv.len())?;
                    cm.prepare(&st2, t, rho)?.apply(&st2, &rd, &mut zd)?;
                    st2.memcpy_dtoh(&zd, zv)?;
                    Ok(())
                }));
            }
            g.x_step(&mut xr, &z0, &u0, rho, 3).unwrap()
        };
        let e = rel(&xd.data, &xr.data);
        let moved = rel(&xd.data, &x0.data);
        eprintln!(
            "data step (metric {}): device vs reference {e:.2e} (step moved x by {moved:.2e}); f {:.6e} → {:.6e} (reference {:.6e} → {:.6e}); ‖r_K‖/‖r₀‖ device {:.3e} reference {:.3e}",
            if clip { "clip" } else { "none" },
            sd.f0,
            sd.f,
            srf.f0,
            srf.f,
            sd.rk / sd.r0,
            srf.rk / srf.r0
        );
        assert!(e < 1e-4, "device data step differs from the reference: {e:.2e}");
        assert!(moved > 1e-3, "the data step did not move x");
        assert!((sd.f - srf.f).abs() <= 1e-4 * srf.f.abs() && (sd.f0 - srf.f0).abs() <= 1e-5 * srf.f0.abs());
        assert!(sd.f < sd.f0, "the data step should lower the data term from this start");
        // the tracked f(x) against a direct evaluation
        let mut gtmp = HostStack::zeros(&lens);
        let f_direct = hasty_recon::p2np::Objective::gradient_host(&mut p, &st, &xd, &mut gtmp).unwrap();
        assert!((sd.f - f_direct).abs() <= 1e-4 * f_direct, "tracked f {} vs direct {f_direct}", sd.f);
        // one full outer iteration (prox: soft threshold at 0.02): device vs reference
        let outer = |data: &mut dyn AdmmData| -> (HostStack, HostStack, HostStack) {
            let (mut x, mut z, mut u) = (x0.clone(), z0.clone(), u0.clone());
            let cfg = AdmmCfg::new(1, rho / rho1, rho1, 3);
            let mut den = |_: usize, h: &mut Vec<C32>| {
                for v in h.iter_mut() {
                    let m = (v.re * v.re + v.im * v.im).sqrt();
                    let s = if m > 0.02 { 1.0 - 0.02 / m } else { 0.0 };
                    *v = C32::new(v.re * s, v.im * s);
                }
            };
            let mut rep = |_: usize, _: &HostStack, _: &HostStack, _: Option<&hasty_recon::admm::AdmmIter>| {};
            run_admm(data, &mut x, &mut z, &mut u, &cfg, &mut den, &mut rep, None).unwrap();
            (x, z, u)
        };
        let (xa, za, ua) = outer(&mut AdmmP2np { p: &mut p, st: st.clone(), metric: clip.then(mk) });
        let (xb, zb, ub) = {
            let mut cm2 = clip.then(mk);
            let mut g = GradXStep::new(&mut p, st.clone());
            if let Some(cm) = cm2.as_mut() {
                let st2 = st.clone();
                g.pre = Some(Box::new(move |t: usize, rv: &[C32], zv: &mut [C32]| {
                    let rd = st2.clone_htod(rv)?;
                    let mut zd: CudaSlice<C32> = st2.alloc_zeros(rv.len())?;
                    cm.prepare(&st2, t, rho)?.apply(&st2, &rd, &mut zd)?;
                    st2.memcpy_dtoh(&zd, zv)?;
                    Ok(())
                }));
            }
            outer(&mut g)
        };
        let (ex, ez, eu) = (rel(&xa.data, &xb.data), rel(&za.data, &zb.data), rel(&ua.data, &ub.data));
        eprintln!("one ADMM outer iteration (metric {}): device vs reference x {ex:.2e}, z {ez:.2e}, u {eu:.2e}", if clip { "clip" } else { "none" });
        // u = u₀ + x − z is a difference of nearby images: its relative error is x's times ‖x‖ / ‖u‖
        assert!(ex < 1e-4 && ez < 1e-4 && eu < 1e-2, "outer iteration differs: {ex:.2e} {ez:.2e} {eu:.2e}");
        drop(p);
    }
    // the clip symbol of a set: its whitened spectrum 1/(max(μ̃, 1) + ρ/ρ₁)
    let lam = clip_symbol(&mus[0], 1.0, rho / rho1);
    assert!(lam.iter().all(|&l| l >= 1.0 + (rho / rho1) as f32 - 1e-5));
}
