//! GPU: CGLS streamed over sample sets (`cg::pcgls_sets_ctl`: one set on the device at a time, host residual) against
//! CGLS on the pooled sets (`cg::pcgls_ctl`), with and without the circulant preconditioner, and a stop / resume of the
//! streamed solver (residual recomputed from x) against an uninterrupted run.
//!   cargo test --release -p hasty-recon --test pcgls_sets -- --nocapture

use std::f64::consts::PI;

use hasty_cuda::{Device, C32};
use hasty_recon::cg::{pcgls_ctl, pcgls_sets_ctl};
use hasty_recon::checkpoint::{Outcome, RunCtl};
use hasty_recon::op::SenseOp;
use hasty_recon::precond::CirculantPrecond;
use hasty_testkit::Rng;

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n).sqrt()
}

#[test]
fn streamed_cgls_matches_pooled_and_resumes() {
    let dims = [24usize, 20, 16];
    let nv: usize = dims.iter().product();
    let nc = 4;
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(hasty_recon::gpu::Kernels::new(&dev).unwrap());
    let mut r = Rng::new(11);
    let smaps: Vec<C32> = (0..nc * nv).map(|_| C32::new(0.3 + r.uniform() as f32, r.uniform() as f32 - 0.5)).collect();
    // three sets of different sizes, dense near k = 0 like a spiral without density compensation
    let sets: Vec<[Vec<f32>; 3]> = [1700usize, 2300, 1900]
        .iter()
        .map(|&m| {
            let rad: Vec<f64> = (0..m).map(|_| PI * r.uniform().powi(2)).collect();
            let dir: Vec<[f64; 3]> = (0..m)
                .map(|_| {
                    let (u, v) = (2.0 * r.uniform() - 1.0, 2.0 * PI * r.uniform());
                    let s = (1.0 - u * u).sqrt();
                    [s * v.cos(), s * v.sin(), u]
                })
                .collect();
            [0, 1, 2].map(|a| (0..m).map(|j| (rad[j] * dir[j][a]).clamp(-PI + 1e-6, PI - 1e-6) as f32).collect())
        })
        .collect();
    let ys: Vec<Vec<C32>> = sets.iter().map(|s| (0..nc * s[0].len()).map(|_| C32::new(r.uniform() as f32 - 0.5, r.uniform() as f32 - 0.5)).collect()).collect();
    // pooled: coil-major data of the concatenation
    let m_all: usize = sets.iter().map(|s| s[0].len()).sum();
    let cat: [Vec<f32>; 3] = [0, 1, 2].map(|a| sets.iter().flat_map(|s| s[a].iter().copied()).collect());
    let mut y_all = vec![C32::new(0.0, 0.0); nc * m_all];
    let mut off = 0;
    for (s, y) in sets.iter().zip(&ys) {
        let m = s[0].len();
        for c in 0..nc {
            y_all[c * m_all + off..c * m_all + off + m].copy_from_slice(&y[c * m..(c + 1) * m]);
        }
        off += m;
    }
    // 4 iterations: the test problem (random data, density ∝ k^-2.5) is so ill-conditioned that f32 rounding (including
    // the non-deterministic atomics of the pooled spreading itself) grows ~7× per CG iteration (1e-7 after the first)
    let iters = 4;
    let tmp = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("pcgls_sets");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    for precond in [false, true] {
        let mk_pre = || CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&cat[0], &cat[1], &cat[2]], &smaps, nc, 1e-3).unwrap();
        // pooled reference
        let mut op_all = SenseOp::new(&dev, &st, kern.clone(), dims, [&cat[0], &cat[1], &cat[2]], &smaps, nc, vec![], 1e-5).unwrap();
        let dy = st.clone_htod(&y_all).unwrap();
        let mut x_ref = st.alloc_zeros::<C32>(nv).unwrap();
        let mut pre = precond.then(mk_pre);
        let mut xs_ref: Vec<Vec<C32>> = vec![];
        let (h_ref, _) = pcgls_ctl(&mut op_all, pre.as_mut(), &st, &dy, &mut x_ref, 0.0, iters, &mut |_, x| xs_ref.push(st.clone_dtoh(x).unwrap()), None).unwrap();
        let x_ref = st.clone_dtoh(&x_ref).unwrap();
        // streamed
        let set_coords: Vec<[&[f32]; 3]> = sets.iter().map(|s| [&s[0][..], &s[1][..], &s[2][..]]).collect();
        let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&sets[0][0], &sets[0][1], &sets[0][2]], &smaps, nc, vec![], 1e-5).unwrap();
        let run = |op: &mut SenseOp, ctl: Option<(&RunCtl, &str)>| {
            let mut pre = precond.then(mk_pre);
            let mut x = st.alloc_zeros::<C32>(nv).unwrap();
            let (h, oc) = pcgls_sets_ctl(op, pre.as_mut(), &st, &set_coords, &mut |i| ys[i].clone(), &mut x, 0.0, iters, &mut |_, _| {}, ctl).unwrap();
            (st.clone_dtoh(&x).unwrap(), h, oc)
        };
        let mut xs_s: Vec<Vec<C32>> = vec![];
        {
            let mut pre = precond.then(mk_pre);
            let mut x = st.alloc_zeros::<C32>(nv).unwrap();
            pcgls_sets_ctl(&mut op, pre.as_mut(), &st, &set_coords, &mut |i| ys[i].clone(), &mut x, 0.0, iters, &mut |_, x| xs_s.push(st.clone_dtoh(x).unwrap()), None).unwrap();
        }
        println!("precond {precond}: per-iteration streamed vs pooled {:?}", xs_s.iter().zip(&xs_ref).map(|(a, b)| format!("{:.1e}", rel(a, b))).collect::<Vec<_>>());
        let (x_s, h_s, _) = run(&mut op, None);
        let e = rel(&x_s, &x_ref);
        println!("precond {precond}: streamed vs pooled {e:.2e}, residual history pooled {:.3e} → {:.3e}, streamed {:.3e}", h_ref[1], h_ref[iters], h_s[iters]);
        assert!(e < 1e-4, "streamed CGLS differs from pooled: {e:e}");
        assert!(h_s[iters] < 0.5 * h_s[0], "no progress");
        // stop after iteration 2 (STOP file), resume
        let ctl = RunCtl::new(&tmp, 0);
        let name = format!("t{}", precond as u8);
        let mut k = 0;
        let mut stop_cb = |it: usize, _: &cudarc::driver::CudaSlice<C32>| {
            k = it;
            if it == 2 {
                std::fs::write(tmp.join("STOP"), b"").unwrap();
            }
        };
        let mut pre1 = precond.then(mk_pre);
        let mut x1 = st.alloc_zeros::<C32>(nv).unwrap();
        let (_, oc) = pcgls_sets_ctl(&mut op, pre1.as_mut(), &st, &set_coords, &mut |i| ys[i].clone(), &mut x1, 0.0, iters, &mut stop_cb, Some((&ctl, &name))).unwrap();
        assert!(matches!(oc, Outcome::Stopped(2)), "expected a stop after iteration 2, got {oc:?} (k {k})");
        std::fs::remove_file(tmp.join("STOP")).unwrap();
        let (x_r, _, oc) = run(&mut op, Some((&ctl, &name)));
        assert!(matches!(oc, Outcome::Finished));
        let e = rel(&x_r, &x_s);
        println!("precond {precond}: resumed vs uninterrupted {e:.2e}");
        assert!(e < 1e-4, "resumed streamed CGLS differs: {e:e}");
    }
}
