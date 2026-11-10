//! GPU: stochastic coil sampling in P²nP (`p2np::CoilSampling`) on the segmented physics NUFFT with energy-ordered
//! virtual coils (SVD-compressed smooth maps):
//! - one epoch of random-reshuffling gradients averages exactly to the full gradient (unbiased);
//! - the step-size inflation λ_max(Σ_g w_g N_G) / λ_max(N): drawing among all coils vs keeping the leading ones;
//! - P²nP-ZMSHR1 with sampling and a few final full-coil iterations reaches the full-gradient solution, at a lower
//!   coil-application cost.
//!   cargo test --release -p hasty-recon --test coil_sampling -- --nocapture

use std::f64::consts::PI;

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::coils::{CoilCompression, CoilGram, CoilRule};
use hasty_recon::gpu::Kernels;
use hasty_recon::p2np::{rdot, run_objective, CoilSampling, Objective, P2np, Precond, Stack};
use hasty_recon::setop::{SegSets, SetOp};
use hasty_testkit::Rng;
use num_complex::Complex64 as C64;

struct Setup {
    sets: SegSets,
    ys: Vec<cudarc::driver::CudaSlice<C32>>,
    nv: usize,
    nc: usize,
    st: std::sync::Arc<cudarc::driver::CudaStream>,
    kern: std::sync::Arc<Kernels>,
}

fn setup(nsets: usize) -> Setup {
    let dims = [24usize, 20, 14];
    let res = 3e-3;
    let nv: usize = dims.iter().product();
    let ncoil = 12;
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [i as f64 - (dims[0] / 2) as f64, j as f64 - (dims[1] / 2) as f64, k as f64 - (dims[2] / 2) as f64]
    }).collect();
    // smooth coil maps around the FOV, then an energy-ordered compression (all virtual coils kept)
    let raw: Vec<C32> = (0..ncoil).flat_map(|c| {
        let th = 2.0 * PI * c as f64 / ncoil as f64;
        idx.iter().map(move |p| {
            let u = [p[0] / 24.0, p[1] / 20.0, p[2] / 14.0];
            let r2 = (u[0] - 0.5 * th.cos()).powi(2) + (u[1] - 0.5 * th.sin()).powi(2) + (u[2] - if c % 2 == 0 { 0.2 } else { -0.2 }).powi(2);
            let z = C64::from_polar((-r2 / 0.2).exp(), 1.2 * (u[0] * th.sin() - u[1] * th.cos()));
            C32::new(z.re as f32, z.im as f32)
        }).collect::<Vec<_>>()
    }).collect();
    let mut g = CoilGram::new(ncoil);
    g.add(&raw, nv, 1);
    let cc = CoilCompression::from_gram(&g, CoilRule::Fixed(8));
    let nc = cc.nvc();
    let smaps = cc.apply(&raw, nv);
    eprintln!("virtual coils: {}", cc.report());
    let mut r = Rng::new(5);
    let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * 80.0 * (p[0] / 12.0 * 0.7 - 0.3 * (p[1] / 10.0).powi(2))).collect();
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let l = 8;
    let mut eng = SegNufft::new(&dev, dims, [res; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
    eng.set_maps(&st, &smaps).unwrap();
    let (nro, na, te, t_ro) = (10usize, 500usize, 1.5e-3, 6e-3);
    let mut pts = vec![];
    let mut segs = vec![];
    let mut taus = vec![];
    for _ in 0..nsets {
        let mut k: [Vec<f32>; 3] = Default::default();
        let mut tau = vec![];
        for _ in 0..nro {
            let (kz, th) = (0.97 * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
            for n in 0..na {
                let s = n as f64 / (na - 1) as f64;
                let (rad, ang) = (0.97 * PI * s.sqrt(), th + 2.0 * PI * 6.0 * s.sqrt());
                k[0].push((rad * ang.cos()) as f32);
                k[1].push((rad * ang.sin()) as f32);
                k[2].push(kz as f32);
                tau.push(te + t_ro * s);
            }
        }
        let m = tau.len();
        let tau32: Vec<f32> = tau.iter().map(|&t| t as f32).collect();
        pts.push(eng.points(&st, [&k[0], &k[1], &k[2]], &tau32, &vec![0f32; m], &[], None).unwrap());
        let fitv: Vec<usize> = (0..nv).step_by(2).collect();
        let psi = |v: usize, t: f64| om[v] * t;
        let (segph, atab, t0, dt) = build_segments(nv, l, tau[0], tau[m - 1], na, &fitv, &psi);
        segs.push(eng.segments(&st, &segph, &atab, t0, dt).unwrap());
        taus.push(m);
    }
    let mut sets = SegSets { eng, pts, segs, seg_of_set: (0..nsets).collect(), kern: kern.clone(), surrogate: None };
    // data from a smooth complex object + small noise
    let xt: Vec<C32> = idx.iter().map(|p| { let rr = (p[0] / 10.0).powi(2) + (p[1] / 8.0).powi(2) + (p[2] / 6.0).powi(2); let a = if rr < 1.0 { 1.0 + 0.3 * (p[0] * 0.5).sin() } else { 0.0 }; C32::new(a as f32, (0.2 * a * (p[1] * 0.3).cos()) as f32) }).collect();
    let dx = st.clone_htod(&xt).unwrap();
    let ys = (0..nsets).map(|t| {
        let mut y = st.alloc_zeros::<C32>(nc * taus[t]).unwrap();
        sets.forward(&st, t, &dx, &mut y).unwrap();
        let mut h = st.clone_dtoh(&y).unwrap();
        for z in h.iter_mut() {
            z.re += 1e-3 * (r.uniform() as f32 - 0.5);
            z.im += 1e-3 * (r.uniform() as f32 - 0.5);
        }
        st.clone_htod(&h).unwrap()
    }).collect();
    Setup { sets, ys, nv, nc, st, kern }
}

fn host(st: &std::sync::Arc<cudarc::driver::CudaStream>, x: &Stack) -> Vec<C32> {
    x.v.iter().flat_map(|v| st.clone_dtoh(v).unwrap()).collect()
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n).sqrt()
}

#[test]
fn reshuffled_epoch_is_the_full_gradient_and_sampling_converges() {
    let s = setup(2);
    let (st, nv, nc) = (s.st.clone(), s.nv, s.nc);
    let mut sets = s.sets;
    let ys: Vec<&cudarc::driver::CudaSlice<C32>> = s.ys.iter().collect();
    let mut x = Stack::zeros(&st, 2, nv).unwrap();
    {
        let h: Vec<C32> = (0..nv).map(|i| C32::new(((i as f32) * 0.37).sin(), ((i as f32) * 0.11).cos())).collect();
        for v in x.v.iter_mut() {
            st.memcpy_htod(&h, v).unwrap();
        }
    }
    // full gradient
    let mut gf = Stack::zeros(&st, 2, nv).unwrap();
    let ff = {
        let mut p = P2np::new(&mut sets, ys.clone(), vec![], &st).unwrap();
        p.gradient(&st, &x, &mut gf).unwrap()
    };
    let gfull = host(&st, &gf);
    // one epoch of random reshuffling: keep 2, draw 2 of the other 6 → 3 groups of weight 3
    let samp = CoilSampling { keep: 2, draw: 2, hold: 1, full_last: 0, reshuffle: true, seed: 42 };
    let groups0 = samp.groups(nc, 0, 1000).unwrap();
    assert_eq!(groups0.len(), 2);
    assert_eq!((groups0[1].0.len(), groups0[1].1), (2, 3.0));
    let mut acc = vec![C32::new(0.0, 0.0); gfull.len()];
    let mut fsum = 0.0;
    {
        let mut p = P2np::new(&mut sets, ys.clone(), vec![], &st).unwrap().with_sampling(samp.clone());
        let mut g = Stack::zeros(&st, 2, nv).unwrap();
        let mut seen = vec![0usize; nc];
        for k in 0..3 {
            Objective::begin_iteration(&mut p, k, 1000);
            for c in &samp.groups(nc, k, 1000).unwrap()[1].0 {
                seen[*c] += 1;
            }
            fsum += p.gradient(&st, &x, &mut g).unwrap();
            for (a, b) in acc.iter_mut().zip(host(&st, &g)) {
                a.re += b.re / 3.0;
                a.im += b.im / 3.0;
            }
        }
        assert_eq!(&seen[2..], &[1; 6], "every sampled coil once per epoch");
    }
    let e = rel(&acc, &gfull);
    eprintln!("epoch mean vs full gradient: rel {e:.2e}; f {:.6e} vs {:.6e}", fsum / 3.0, ff);
    assert!(e < 1e-5, "epoch mean of the sampled gradients differs from the full gradient by {e:.2e}");
    assert!((fsum / 3.0 - ff).abs() < 1e-5 * ff);
    // step-size inflation: λ_max of the sampled normal operator relative to the full one
    let lam_full = { let mut p = P2np::new(&mut sets, ys.clone(), vec![], &st).unwrap(); p.step_norm(&st, false, 30).unwrap() };
    let mut infl = vec![];
    for (keep, draw) in [(0usize, 4usize), (2, 3), (3, 2)] {
        let sm = CoilSampling { keep, draw, hold: 1, full_last: 0, reshuffle: true, seed: 7 };
        let mut p = P2np::new(&mut sets, ys.clone(), vec![], &st).unwrap().with_sampling(sm);
        let l = p.step_norm_sampled(&st, false, 30, 4).unwrap();
        infl.push(l / lam_full);
        eprintln!("keep {keep}, draw {draw} of {}: λ_max(sampled) / λ_max(full) = {:.3}, cost per gradient {:.2}", nc - keep, l / lam_full, (keep + draw) as f64 / nc as f64);
    }
    assert!(infl[1] < infl[0], "keeping the leading virtual coils should inflate the step less than drawing among all");
    // convergence at a fixed cost budget (gradient-equivalents of coil applications), against a long full run (the
    // solution): full gradients vs sampling schedules, PnP-ISTA (no preconditioner) and ZMSHR1
    let x0 = Stack::zeros(&st, 2, nv).unwrap();
    let run = |sets: &mut SegSets, samp: Option<CoilSampling>, iters: usize, pc: Precond| -> (Vec<C32>, f64) {
        let mut p = P2np::new(sets, ys.clone(), vec![], &st).unwrap();
        let mut lam = lam_full;
        if let Some(sm) = samp.clone() {
            p = p.with_sampling(sm);
            lam = lam.max(p.step_norm_sampled(&st, false, 30, 6).unwrap());
        }
        let mut x = Stack::zeros(&st, 2, nv).unwrap();
        x.copy_from(&st, &x0).unwrap();
        let cost: f64 = (0..iters).map(|k| samp.as_ref().map_or(1.0, |s| s.cost(nc, k, iters))).sum();
        run_objective(&mut p, &st, &mut x, pc, 1.0 / lam, iters, &mut |_, _| {}, &mut |_, _, _| {}, None).unwrap();
        (host(&st, &x), cost)
    };
    let (xref, _) = run(&mut sets, None, 400, Precond::Zmshr1);
    let budget = 60.0;
    // iterations of a schedule that fit the budget
    let fit = |sm: &CoilSampling| -> usize { (1..2000).take_while(|&n| (0..n).map(|k| sm.cost(nc, k, n)).sum::<f64>() <= budget).last().unwrap_or(1) };
    let mut best_sampled = f64::INFINITY;
    for pc in [Precond::None, Precond::Zmshr1] {
        let (xf, _) = run(&mut sets, None, budget as usize, pc);
        eprintln!("{pc:?}: full gradients, {budget} iterations: error vs the solution {:.2e}", rel(&xf, &xref));
        for (keep, draw, hold, last) in [(2usize, 3usize, 2usize, 12usize), (2, 3, 4, 12), (2, 3, 8, 12), (2, 3, 4, 25), (3, 2, 4, 12), (0, 4, 4, 12)] {
            let sm = CoilSampling { keep, draw, hold, full_last: last, reshuffle: true, seed: 3 };
            let n = fit(&sm);
            let (xs, cs) = run(&mut sets, Some(sm), n, pc);
            let es = rel(&xs, &xref);
            best_sampled = best_sampled.min(es);
            eprintln!("{pc:?}:   keep {keep} draw {draw} hold {hold} last {last}: {n} iterations ({cs:.1} gradient-equivalents), error {es:.2e}");
        }
    }
    assert!(best_sampled < 0.15, "every sampled schedule failed to converge: best {best_sampled:.2e}");
    let _ = rdot;
    let _ = s.kern;
}
