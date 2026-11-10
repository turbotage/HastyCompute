//! GPU: the host-resident P²nP solver (`hoststack::run_objective_host`: stacks in host memory, sets streamed through
//! two device slots, fused plain-ZMSHR1 step, raw streaming checkpoints) against the device solver, and a stop /
//! resume through the raw checkpoint against an uninterrupted run.
//!   cargo test --release -p hasty-recon --test host_state -- --nocapture

use std::f64::consts::PI;

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::checkpoint::{Outcome, RunCtl};
use hasty_recon::hoststack::{run_objective_host, HostStack};
use hasty_recon::p2np::{P2np, Precond, Stack};
use hasty_recon::setop::{SegSets, SetOp};
use hasty_testkit::Rng;
use num_complex::Complex64 as C64;

const L: usize = 8;

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n).sqrt()
}

#[test]
fn host_state_matches_device_and_resumes_exactly() {
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
    let lam = P2np::new(&mut sets, ysr.clone(), vec![], &st).unwrap().step_norm(&st, false, 30).unwrap();
    let iters = 14;
    for pc in [Precond::Zmshr1, Precond::None] {
        let xd = {
            let mut p = P2np::new(&mut sets, ysr.clone(), vec![], &st).unwrap();
            let mut x = Stack::zeros(&st, nsets, nv).unwrap();
            p.run_from(&st, &mut x, pc, 1.0 / lam, iters, &mut |_, _| {}, &mut |_, _, _| {}, None).unwrap();
            x.v.iter().flat_map(|v| st.clone_dtoh(v).unwrap()).collect::<Vec<C32>>()
        };
        let run_host = |sets: &mut SegSets, ctl: Option<(&RunCtl, &str)>, stop_at: Option<usize>| -> (Vec<C32>, Outcome) {
            let mut p = P2np::new(sets, ysr.clone(), vec![], &st).unwrap();
            let mut xh = HostStack::zeros(&vec![nv; nsets]);
            let dir = ctl.map(|(c, _)| c.dir().to_path_buf());
            let mut report = |k: usize, _x: &HostStack, _f: f64| {
                if Some(k) == stop_at {
                    std::fs::write(dir.as_ref().unwrap().join("STOP"), b"").unwrap();
                }
            };
            let o = run_objective_host(&mut p, &st, &mut xh, pc, 1.0 / lam, iters, &mut |_, _| {}, &mut report, ctl).unwrap();
            (xh.data.clone(), o)
        };
        let (xh, _) = run_host(&mut sets, None, None);
        let e = rel(&xh, &xd);
        println!("{pc:?}: host vs device after {iters} iterations: {e:.2e}");
        assert!(e < 1e-4, "{pc:?}: host-resident solver differs from the device solver: {e:e}");
        // stop after iteration 5 (raw checkpoint), resume in a fresh run
        let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("host_state_{pc:?}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (_, o) = run_host(&mut sets, Some((&RunCtl::new(&dir, 0), "t")), Some(5));
        assert_eq!(o, Outcome::Stopped(6), "{pc:?}: expected a stop before iteration 6");
        assert!(dir.join("state_t.hsk").exists(), "raw checkpoint missing");
        let (xr, o) = run_host(&mut sets, Some((&RunCtl::new(&dir, 0), "t")), None);
        assert_eq!(o, Outcome::Finished);
        assert!(!dir.join("state_t.hsk").exists(), "checkpoint not cleared at the end");
        let e = rel(&xr, &xh);
        println!("{pc:?}: stopped at 6 and resumed vs uninterrupted: {e:.2e}");
        assert!(e < 1e-4, "{pc:?}: resumed run differs: {e:e}");
    }
}
