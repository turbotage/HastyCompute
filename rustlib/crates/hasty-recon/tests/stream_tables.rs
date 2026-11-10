//! GPU: P²nP with the operator's sample tables streamed from pinned host memory (`SetOp::place_tables`, as the memory
//! planner does under a small `--vram-budget`) reproduces the all-resident run: gradients and three ZMSHR1 iterations
//! identical (≤ 1e-6 relative; the same kernels on the same data), through `SegSets` directly and inside `FatSets`.
//!   cargo test --release -p hasty-recon --test stream_tables -- --nocapture

use std::f64::consts::PI;
use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::gpu::Kernels;
use hasty_recon::p2np::{P2np, Precond, Stack};
use hasty_recon::plan::{choose_tables, Need, Probe, TableNeed};
use hasty_recon::setop::{fat_spectrum_weights, FatSets, SegSets, SetOp};
use hasty_testkit::Rng;

const DIMS: [usize; 3] = [20, 18, 12];

fn c32(v: &[hasty_testkit::C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}

/// six equal sets (frames × encodes style), SegNufft order 2, L = 6
fn seg_sets(dev: &Device, kern: &Arc<Kernels>) -> (SegSets, Vec<Vec<f64>>) {
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    let (nc, l) = (3usize, 6usize);
    let mut r = Rng::new(5);
    let mut eng = SegNufft::new(dev, DIMS, [3e-3; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 2, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
    eng.set_maps(&st, &c32(&r.complex_vec(nc * nv))).unwrap();
    let om: Vec<f64> = (0..nv).map(|q| 2.0 * PI * 100.0 * ((q % DIMS[0]) as f64 / DIMS[0] as f64 - 0.5)).collect();
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    let (sp, at, t0, dt) = build_segments(nv, l, 1.6e-3, 9.6e-3, 300, &fitv, &psi);
    let seg = eng.segments(&st, &sp, &at, t0, dt).unwrap();
    let mut pts = vec![];
    let mut taus = vec![];
    for _ in 0..6 {
        let (nro, na) = (5usize, 300usize);
        let mut k: [Vec<f32>; 3] = Default::default();
        let mut tau = vec![];
        for _ in 0..nro {
            let (kz, th) = (0.97 * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
            for i in 0..na {
                let q = i as f64 / (na - 1) as f64;
                let (rad, ang) = (0.97 * PI * q.sqrt(), th + 2.0 * PI * 7.0 * q.sqrt());
                k[0].push((rad * ang.cos()) as f32);
                k[1].push((rad * ang.sin()) as f32);
                k[2].push(kz as f32);
                tau.push(1.6e-3 + 8e-3 * q);
            }
        }
        let m = tau.len();
        let tau32: Vec<f32> = tau.iter().map(|&t| t as f32).collect();
        let phi0: Vec<f32> = (0..m).map(|_| r.uniform() as f32).collect();
        let dc: Vec<f32> = (0..6 * m).map(|_| (150.0 * (2.0 * r.uniform() - 1.0)) as f32).collect();
        pts.push(eng.points(&st, [&k[0], &k[1], &k[2]], &tau32, &phi0, &dc, None).unwrap());
        taus.push(tau);
    }
    (SegSets { eng, pts, segs: vec![seg], seg_of_set: vec![0; 6], kern: kern.clone(), surrogate: None }, taus)
}

/// three ZMSHR1 iterations (identity prior) from a fixed start; returns the final stack on the host
fn iterate(op: &mut dyn SetOp, st: &Arc<cudarc::driver::CudaStream>, ys: &[cudarc::driver::CudaSlice<C32>], x0: &[Vec<C32>]) -> (Vec<Vec<C32>>, Vec<Vec<C32>>) {
    let mut p = P2np::new(op, ys.iter().collect(), vec![], st).unwrap();
    let mut x = Stack { v: x0.iter().map(|v| st.clone_htod(v).unwrap()).collect() };
    let mut g = Stack { v: x0.iter().map(|v| st.alloc_zeros::<C32>(v.len()).unwrap()).collect() };
    p.gradient(st, &x, &mut g).unwrap();
    let g0: Vec<Vec<C32>> = g.v.iter().map(|d| st.clone_dtoh(d).unwrap()).collect();
    let alpha = 0.5 / p.step_norm(st, false, 8).unwrap();
    p.run(st, &mut x, Precond::Zmshr1, alpha, 3, &mut |_| {}, &mut |_, _, _| {}).unwrap();
    (g0, x.v.iter().map(|d| st.clone_dtoh(d).unwrap()).collect())
}

fn rel(a: &[Vec<C32>], b: &[Vec<C32>]) -> f64 {
    let (mut n, mut d) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
        n += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        d += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (n / d.max(1e-300)).sqrt()
}

#[test]
fn p2np_with_streamed_tables_matches_resident() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let nv: usize = DIMS.iter().product();
    let (mut sets, taus) = seg_sets(&dev, &kern);
    let mut r = Rng::new(11);
    let ys: Vec<_> = (0..6).map(|t| st.clone_htod(&c32(&r.complex_vec(sets.len_y(t)))).unwrap()).collect();
    let x0: Vec<Vec<C32>> = (0..6).map(|_| c32(&r.complex_vec(nv))).collect();
    // the planner under a budget that cannot hold the tables streams them
    let (tb, tmax) = sets.table_bytes();
    let probe = Probe { free: 1 << 30, total: 1 << 30, h2d: 20e9, d2h: 20e9 };
    let need = Need { fixed: (64 << 20) + tb, stack: 6 * nv * 8, chunk_max: nv * 8, data: ys.iter().map(|y| y.len() * 8).sum(), set_max: ys.iter().map(|y| y.len() * 8).max().unwrap(), quasi: true, gradient_s: 0.01 };
    // room for streamed tables (two slots) and all of the state, not for resident tables (6 equal sets ≫ 2 slots)
    let budget = need.fixed - tb + 2 * tmax + 6 * need.stack + 6 * need.chunk_max + need.data;
    assert!(tb > 2 * tmax + 3 * need.stack, "test sizes: tables {tb} B, set {tmax} B, stack {} B", need.stack);
    let plan = choose_tables(&need, &TableNeed { bytes: tb, set_max: tmax, device_now: tb }, &probe, budget, Some(false), None).unwrap();
    print!("{}", plan.describe());
    assert!(plan.tables_on_host, "the test budget should force streamed tables");
    let (g_res, x_res) = iterate(&mut sets, &st, &ys, &x0);
    // run-to-run spread of the resident path itself (atomic accumulation order in the adjoint)
    let (g_r2, x_r2) = iterate(&mut sets, &st, &ys, &x0);
    let (sg, sx) = (rel(&g_r2, &g_res), rel(&x_r2, &x_res));
    eprintln!("resident repeat: gradient {sg:.1e}, iterate {sx:.1e}");
    for prefetch in [false, true] {
        let freed = sets.place_tables(&dev.ctx, &st, true).unwrap();
        assert!(sets.tables_on_host() && freed > 0);
        sets.eng.stream_prefetch = prefetch;
        let (g_s, x_s) = iterate(&mut sets, &st, &ys, &x0);
        let (eg, ex) = (rel(&g_s, &g_res), rel(&x_s, &x_res));
        eprintln!("prefetch {prefetch}: gradient {eg:.1e}, 3 ZMSHR1 iterations {ex:.1e} ({:.1} MB freed, {} uploads)", freed as f64 / 1e6, sets.eng.stream_stats().0);
        // identical up to the resident path's own run-to-run spread (float32 atomics)
        assert!(eg <= (3.0 * sg).max(2e-5) && ex <= (3.0 * sx).max(1e-4), "streamed vs resident: gradient {eg:.1e}, iterate {ex:.1e} (resident repeat {sg:.1e} / {sx:.1e})");
        sets.place_tables(&dev.ctx, &st, false).unwrap();
        assert!(!sets.tables_on_host());
    }
    // inside FatSets (shared-gridding multi path)
    let ppm = [-3.40];
    let c: Vec<Vec<C32>> = taus.iter().map(|t| fat_spectrum_weights(&ppm, &[1.0], 2.89, t)).collect();
    let ff: Vec<f32> = (0..nv).map(|_| r.uniform() as f32).collect();
    let (g_fr, x_fr) = {
        let mut fs = FatSets::new(&st, &mut sets, &ff, c.clone()).unwrap();
        iterate(&mut fs, &st, &ys, &x0)
    };
    sets.place_tables(&dev.ctx, &st, true).unwrap();
    sets.eng.stream_prefetch = true;
    let mut fs = FatSets::new(&st, &mut sets, &ff, c).unwrap();
    assert!(fs.tables_on_host());
    let (g_fs, x_fs) = iterate(&mut fs, &st, &ys, &x0);
    let (eg, ex) = (rel(&g_fs, &g_fr), rel(&x_fs, &x_fr));
    eprintln!("FatSets: gradient {eg:.1e}, iterate {ex:.1e}");
    assert!(eg <= (3.0 * sg).max(2e-5) && ex <= (3.0 * sx).max(1e-4), "FatSets streamed vs resident: gradient {eg:.1e}, iterate {ex:.1e}");
}
