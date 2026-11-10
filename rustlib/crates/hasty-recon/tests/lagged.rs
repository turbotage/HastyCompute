//! GPU: lagged (control-variate) gradients in P²nP (`P2np::with_lagged`) with a reduced-segment surrogate of the
//! segmented physics NUFFT (`SegSets::with_surrogate`, same sample tables):
//! - the surrogate switch leaves the main operator untouched; Ẽ differs from E by the B0 time evolution only;
//! - at the anchor the lagged gradient is the exact one; next to it the error is (Ñ − N)(x − x_a);
//! - P²nP-ZMSHR1 with lagged gradients reaches the exact-gradient solution with a fraction of the exact gradients,
//!   and its cheap gradients are cheaper by about L_s / L.
//!   cargo test --release -p hasty-recon --test lagged -- --nocapture

use std::f64::consts::PI;
use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::p2np::{P2np, Precond, Stack};
use hasty_recon::setop::{SegSets, SetOp};
use hasty_testkit::Rng;
use num_complex::Complex64 as C64;

const L: usize = 8;

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
fn lagged_gradient_with_reduced_segment_surrogate() {
    let dims = [24usize, 20, 14];
    let res = 3e-3;
    let nv: usize = dims.iter().product();
    let nc = 6;
    let nsets = 2;
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [i as f64 - (dims[0] / 2) as f64, j as f64 - (dims[1] / 2) as f64, k as f64 - (dims[2] / 2) as f64]
    }).collect();
    let smaps: Vec<C32> = (0..nc).flat_map(|c| {
        let th = 2.0 * PI * c as f64 / nc as f64;
        idx.iter().map(move |p| {
            let u = [p[0] / 24.0, p[1] / 20.0, p[2] / 14.0];
            let r2 = (u[0] - 0.5 * th.cos()).powi(2) + (u[1] - 0.5 * th.sin()).powi(2) + u[2].powi(2);
            let z = C64::from_polar((-r2 / 0.3).exp(), 1.2 * (u[0] * th.sin() - u[1] * th.cos()));
            C32::new(z.re as f32, z.im as f32)
        }).collect::<Vec<_>>()
    }).collect();
    // off-resonance (HASTY_LAG_HZ, default 30: ≈ 0.3 phase cycles over the readout). Measured: the lagged gradient
    // pays while the surrogate mismatch is small (30 Hz, L_s = 1: 7 exact of 60 at the exact run's error); at 80 Hz
    // (0.8 cycles) and 250 Hz (2.5 cycles) it needs 33–51 exact gradients of 60 and ends 2–3× further from the solution
    let hz: f64 = std::env::var("HASTY_LAG_HZ").ok().and_then(|v| v.parse().ok()).unwrap_or(30.0);
    let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * hz * (p[0] / 12.0 * 0.7 - 0.3 * (p[1] / 10.0).powi(2))).collect();
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(hasty_recon::gpu::Kernels::new(&dev).unwrap());
    let opts = |l: usize| SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 };
    let mut eng = SegNufft::new(&dev, dims, [res; 3], nc, opts(L)).unwrap();
    eng.set_maps(&st, &smaps).unwrap();
    let (nro, na, te, t_ro) = (14usize, 500usize, 1.5e-3, 6e-3);
    let mut r = Rng::new(9);
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    let mut pts = vec![];
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
        taus.push(tau);
    }
    let segs_for = |e: &SegNufft, l: usize| -> Vec<_> {
        taus.iter().map(|tau| {
            let m = tau.len();
            let (segph, atab, t0, dt) = build_segments(nv, l, tau[0], tau[m - 1], na, &fitv, &psi);
            e.segments(&st, &segph, &atab, t0, dt).unwrap()
        }).collect()
    };
    let segs = segs_for(&eng, L);
    let mut sets = SegSets { eng, pts, segs, seg_of_set: (0..nsets).collect(), kern: kern.clone(), surrogate: None };
    // object, data
    let xt: Vec<C32> = idx.iter().map(|p| {
        let rr = (p[0] / 10.0).powi(2) + (p[1] / 8.0).powi(2) + (p[2] / 6.0).powi(2);
        let a = if rr < 1.0 { 1.0 + 0.3 * (p[0] * 0.5).sin() } else { 0.0 };
        C32::new(a as f32, (0.2 * a * (p[1] * 0.3).cos()) as f32)
    }).collect();
    let dx = st.clone_htod(&xt).unwrap();
    let ys: Vec<_> = (0..nsets).map(|t| {
        let mut y = st.alloc_zeros::<C32>(sets.len_y(t)).unwrap();
        sets.forward(&st, t, &dx, &mut y).unwrap();
        let mut h = st.clone_dtoh(&y).unwrap();
        for z in h.iter_mut() {
            z.re += 1e-3 * (r.uniform() as f32 - 0.5);
            z.im += 1e-3 * (r.uniform() as f32 - 0.5);
        }
        st.clone_htod(&h).unwrap()
    }).collect();
    let ysr: Vec<&cudarc::driver::CudaSlice<C32>> = ys.iter().collect();
    let ex = {
        let mut p = P2np::new(&mut sets, ysr.clone(), vec![], &st).unwrap();
        (p.step_norm(&st, false, 30).unwrap(),)
    };
    let lam = ex.0;
    // reference: exact P²nP-ZMSHR1, long run
    let x0 = Stack::zeros(&st, nsets, nv).unwrap();
    let run = |sets: &mut SegSets, tau: Option<f64>, iters: usize| -> (Vec<C32>, (usize, usize), f64) {
        let mut p = P2np::new(sets, ysr.clone(), vec![], &st).unwrap();
        if let Some(t) = tau {
            p = p.with_lagged(t);
        }
        let mut x = Stack::zeros(&st, nsets, nv).unwrap();
        x.copy_from(&st, &x0).unwrap();
        let mut flast = 0.0;
        p.run_from(&st, &mut x, Precond::Zmshr1, 1.0 / lam, iters, &mut |_, _| {}, &mut |_, _, f| flast = f, None).unwrap();
        (host(&st, &x), p.lag_stats, flast)
    };
    let (xref, _, fref) = run(&mut sets, None, 300);
    // attach the surrogate: L_s segments on the same tables
    for ls in [1usize, 2, 3] {
        let mut se = SegNufft::new(&dev, dims, [res; 3], nc, opts(ls)).unwrap();
        se.set_maps(&st, &smaps).unwrap();
        let ss = segs_for(&se, ls);
        let mut sets2 = SegSets { eng: std::mem::replace(&mut sets.eng, SegNufft::new(&dev, dims, [res; 3], nc, opts(1)).unwrap()), pts: std::mem::take(&mut sets.pts), segs: std::mem::take(&mut sets.segs), seg_of_set: sets.seg_of_set.clone(), kern: kern.clone(), surrogate: None }.with_surrogate(se, ss);
        // 1. the switch: main operator unchanged, surrogate mismatch ‖(E − Ẽ) x‖ / ‖E x‖
        let mut y0 = st.alloc_zeros::<C32>(sets2.len_y(0)).unwrap();
        let mut y1 = st.alloc_zeros::<C32>(sets2.len_y(0)).unwrap();
        sets2.forward(&st, 0, &dx, &mut y0).unwrap();
        assert!(sets2.use_surrogate(true));
        sets2.forward(&st, 0, &dx, &mut y1).unwrap();
        assert!(sets2.use_surrogate(false));
        let mut y2 = st.alloc_zeros::<C32>(sets2.len_y(0)).unwrap();
        sets2.forward(&st, 0, &dx, &mut y2).unwrap();
        let (h0, h1, h2) = (st.clone_dtoh(&y0).unwrap(), st.clone_dtoh(&y1).unwrap(), st.clone_dtoh(&y2).unwrap());
        // the gather accumulates tiles with atomics: equal up to the f32 summation order
        assert!(rel(&h2, &h0) < 1e-6, "the main operator changed after the surrogate was used: {:.2e}", rel(&h2, &h0));
        let mis = rel(&h1, &h0);
        // 2. at the anchor exact; next to it (x_b = 1.05 x_a, inside τ = 0.1) the error is (Ñ − N)(x_b − x_a)
        let mk = |scale: f32| {
            let mut s = Stack::zeros(&st, nsets, nv).unwrap();
            let h: Vec<C32> = xt.iter().map(|z| C32::new(z.re * scale, z.im * scale)).collect();
            for v in s.v.iter_mut() {
                st.memcpy_htod(&h, v).unwrap();
            }
            s
        };
        let (xa, xb) = (mk(0.9), mk(0.9 * 1.05));
        let mut g = Stack::zeros(&st, nsets, nv).unwrap();
        let (ga, gb, fa, fb) = {
            let mut p = P2np::new(&mut sets2, ysr.clone(), vec![], &st).unwrap();
            let fa = p.gradient(&st, &xa, &mut g).unwrap();
            let ga = host(&st, &g);
            let fb = p.gradient(&st, &xb, &mut g).unwrap();
            (ga, host(&st, &g), fa, fb)
        };
        let (la, lb, fla, flb, tex, tch) = {
            let mut p = P2np::new(&mut sets2, ysr.clone(), vec![], &st).unwrap().with_lagged(0.1);
            let fla = p.gradient(&st, &xa, &mut g).unwrap();
            let la = host(&st, &g);
            let flb = p.gradient(&st, &xb, &mut g).unwrap();
            let lb = host(&st, &g);
            assert_eq!(p.lag_stats, (1, 1), "first gradient exact (anchor), second cheap");
            // time per gradient: exact (refresh) vs cheap
            st.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..5 {
                p.gradient(&st, &xb, &mut g).unwrap();
            }
            st.synchronize().unwrap();
            let tch = t0.elapsed().as_secs_f64() / 5.0;
            let mut pe = P2np::new(p.op, ysr.clone(), vec![], &st).unwrap();
            let t0 = Instant::now();
            for _ in 0..5 {
                pe.gradient(&st, &xb, &mut g).unwrap();
            }
            st.synchronize().unwrap();
            (la, lb, fla, flb, t0.elapsed().as_secs_f64() / 5.0, tch)
        };
        assert!(rel(&la, &ga) < 1e-4, "lagged gradient at the anchor (f32 atomics; g cancels ≈ 10×) is not the exact one: {:.2e}", rel(&la, &ga));
        assert!((fla - fa).abs() <= 1e-5 * fa.abs());
        // ‖(Ñ − N)d‖ relative to ‖∇f(x_b)‖ (d = 0.05 x_a)
        let eb = rel(&lb, &gb);
        eprintln!("L_s = {ls}: gradient error next to the anchor {eb:.2e} (f {flb:.5e} vs {fb:.5e}); time per gradient exact {:.1} ms, cheap {:.1} ms", tex * 1e3, tch * 1e3);
        assert!(eb < 3.0 * mis + 1e-3, "lagged gradient error {eb:.2e} next to the anchor (mismatch {mis:.2e})");
        let (xl, stats, fl) = run(&mut sets2, Some(0.05), 60);
        let (xl2, stats2, fl2) = run(&mut sets2, Some(0.1), 60);
        let (xe, _, fe) = run(&mut sets2, None, 60);
        eprintln!(
            "L_s = {ls}: ‖(E − Ẽ)x‖/‖Ex‖ {mis:.3}; 60 iterations, error vs the 300-iteration solution: exact {:.2e} (f {fe:.4e}), lagged τ 0.05 {:.2e} ({} exact + {} cheap, f {fl:.4e}), τ 0.1 {:.2e} ({} + {}, f {fl2:.4e}); f(300) {fref:.4e}",
            rel(&xe, &xref),
            rel(&xl, &xref),
            stats.0,
            stats.1,
            rel(&xl2, &xref),
            stats2.0,
            stats2.1
        );
        if ls == 1 && (hz - 30.0).abs() < 1e-9 {
            assert!(rel(&xl2, &xref) < 1.5 * rel(&xe, &xref) && stats2.0 <= 30, "lagged τ 0.1 at small off-resonance: {} exact gradients, error {:.2e} vs {:.2e}", stats2.0, rel(&xl2, &xref), rel(&xe, &xref));
        }
        sets.eng = sets2.eng;
        sets.pts = sets2.pts;
        sets.segs = sets2.segs;
    }
}

