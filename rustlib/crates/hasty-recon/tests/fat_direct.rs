//! GPU: `FatSets` (water + fat with a known fat fraction) around the segmented physics NUFFT against an f64 direct sum
//! of the exact signal: E x = Σ_r S_c(r) x(r) [(1 − ff(r)) + ff(r) c(τ)] e^{−i(k·r + ω(r) τ)}, six-peak fat at 2.89 T,
//! a smooth B0 (±150 Hz), an 8 ms readout, mixed voxels at the edges of a fat shell; and the adjoint identity.
//! Also: the shared-gridding path (default, `SegNufft::forward_multi`) against the two separate inner applications,
//! directly and through `WeightedSets`.
//!   cargo test --release -p hasty-recon --test fat_direct -- --nocapture

use std::f64::consts::PI;

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::gpu::Kernels;
use hasty_recon::setop::{fat_spectrum_weights, FatSets, SegSets, SetOp, WeightedSets};
use hasty_testkit::physops::{rel_on, DirectSum};
use hasty_testkit::Rng;
use num_complex::Complex64 as C64;

#[test]
fn fat_sets_match_direct_sum_and_are_adjoint() {
    let dims = [20usize, 18, 12];
    let res = 3e-3;
    let nv: usize = dims.iter().product();
    let nc = 3;
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [i as f64 - (dims[0] / 2) as f64, j as f64 - (dims[1] / 2) as f64, k as f64 - (dims[2] / 2) as f64]
    }).collect();
    let mut r = Rng::new(17);
    let x = r.complex_vec(nv);
    let smaps: Vec<C64> = (0..nc).flat_map(|c| idx.iter().map(move |p| C64::from_polar(1.0 + 0.3 * (p[0] * (c as f64 + 1.0) / 10.0).cos(), 0.1 * c as f64 * p[1]))).collect();
    let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * 150.0 * ((p[0] / 10.0) * 0.8 + (p[1] / 9.0).powi(2) * 0.4 - 0.2 * (p[2] / 6.0))).collect();
    let ff: Vec<f32> = idx.iter().map(|p| {
        let rr = ((p[0] / 10.0).powi(2) + (p[1] / 9.0).powi(2) + (p[2] / 6.0).powi(2)).sqrt();
        (1.0 - ((rr - 0.7).abs() * 10.0 / 1.5 - 0.5).clamp(0.0, 1.0)) as f32
    }).collect();
    // spiral-like readouts, 8 ms
    let (nro, na, te, t_ro) = (8usize, 400usize, 1.6e-3, 8e-3);
    let mut k: [Vec<f32>; 3] = Default::default();
    let mut tau = vec![];
    for _ in 0..nro {
        let (kz, th) = (0.97 * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
        for n in 0..na {
            let s = n as f64 / (na - 1) as f64;
            let (rad, ang) = (0.97 * PI * s.sqrt(), th + 2.0 * PI * 7.0 * s.sqrt());
            k[0].push((rad * ang.cos()) as f32);
            k[1].push((rad * ang.sin()) as f32);
            k[2].push(kz as f32);
            tau.push(te + t_ro * s);
        }
    }
    let m = tau.len();
    let ppm = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60];
    let amp: Vec<f64> = { let a = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]; let s: f64 = a.iter().sum(); a.iter().map(|v| v / s).collect() };
    let c = fat_spectrum_weights(&ppm, &amp, 2.89, &tau);
    // exact reference
    let vox: Vec<usize> = (0..nv).collect();
    let yr = Rng::new(3).complex_vec(nc * m);
    let ds = DirectSum { nv, nc, m, frames: 1, smaps: &smaps, voxels: &vox, bw: None };
    let exact = |j: usize, q: usize| {
        let p = &idx[q];
        let cj = C64::new(c[j].re as f64, c[j].im as f64);
        C64::from_polar(1.0, -(k[0][j] as f64 * p[0] + k[1][j] as f64 * p[1] + k[2][j] as f64 * p[2] + om[q] * tau[j])) * (C64::new(1.0 - ff[q] as f64, 0.0) + ff[q] as f64 * cj)
    };
    let (yref, xref) = ds.apply(&exact, &x, &yr);
    // GPU: SegNufft (B0 segments only) inside FatSets
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let l = 10;
    let mut eng = SegNufft::new(&dev, dims, [res; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
    let c32 = |v: &[C64]| -> Vec<C32> { v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect() };
    eng.set_maps(&st, &c32(&smaps)).unwrap();
    let tau32: Vec<f32> = tau.iter().map(|&t| t as f32).collect();
    let pts = eng.points(&st, [&k[0], &k[1], &k[2]], &tau32, &vec![0f32; m], &[], None).unwrap();
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    let (segph, atab, t0, dt) = build_segments(nv, l, tau[0], tau[m - 1], na, &fitv, &psi);
    let seg = eng.segments(&st, &segph, &atab, t0, dt).unwrap();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let mut inner = SegSets { eng, pts: vec![pts], segs: vec![seg], seg_of_set: vec![0], kern, surrogate: None };
    let mut op = FatSets::new(&st, &mut inner, &ff, vec![c.clone()]).unwrap();
    let dx = st.clone_htod(&c32(&x)).unwrap();
    let dyr = st.clone_htod(&c32(&yr)).unwrap();
    let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
    let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
    op.forward(&st, 0, &dx, &mut y).unwrap();
    op.adjoint(&st, 0, &dyr, &mut xa).unwrap();
    let c64v = |v: &[C32]| -> Vec<C64> { v.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect() };
    let (y, xa) = (c64v(&st.clone_dtoh(&y).unwrap()), c64v(&st.clone_dtoh(&xa).unwrap()));
    let (ef, ea) = (rel_on(&y, &yref, None), rel_on(&xa, &xref, None));
    // ⟨y_r, A x⟩ = ⟨Aᴴ y_r, x⟩
    let lhs: C64 = yr.iter().zip(&y).map(|(a, b)| a.conj() * b).sum();
    let rhs: C64 = xa.iter().zip(&x).map(|(a, b)| a.conj() * b).sum();
    let adj = (lhs - rhs).norm() / lhs.norm();
    let nmix = ff.iter().filter(|&&f| f > 0.02 && f < 0.98).count();
    eprintln!("FatSets (L = {l}, {nmix} mixed voxels): forward {ef:.2e}, adjoint {ea:.2e}, adjoint identity {adj:.2e}");
    assert!(ef < 2e-3 && ea < 2e-3, "FatSets vs direct sum: {ef} / {ea}");
    assert!(adj < 1e-3, "adjoint identity {adj}"); // float32 (measured 1.2e-5)
}

/// FatSets' shared gridding (one multi-image call, the default) equals the two separate inner applications (`separate`)
/// to f32 rounding (≤ 1e-5 relative), forward and adjoint, on a SegNufft with order 2 and coil batches of 2, directly
/// and with a voxel-basis `WeightedSets` in between (its multi path passes through).
#[test]
fn fat_sets_shared_gridding_matches_separate() {
    let dims = [20usize, 18, 12];
    let res = 3e-3;
    let nv: usize = dims.iter().product();
    let nc = 5;
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [i as f64 - (dims[0] / 2) as f64, j as f64 - (dims[1] / 2) as f64, k as f64 - (dims[2] / 2) as f64]
    }).collect();
    let mut r = Rng::new(23);
    let smaps: Vec<C64> = (0..nc).flat_map(|c| idx.iter().map(move |p| C64::from_polar(1.0 + 0.3 * (p[0] * (c as f64 + 1.0) / 10.0).cos(), 0.1 * c as f64 * p[1]))).collect();
    let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * 150.0 * ((p[0] / 10.0) * 0.8 + (p[1] / 9.0).powi(2) * 0.4)).collect();
    let ff: Vec<f32> = (0..nv).map(|_| r.uniform() as f32).collect();
    let (nro, na) = (6usize, 400usize);
    let mut k: [Vec<f32>; 3] = Default::default();
    let mut tau = vec![];
    for _ in 0..nro {
        let (kz, th) = (0.97 * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
        for n in 0..na {
            let s = n as f64 / (na - 1) as f64;
            let (rad, ang) = (0.97 * PI * s.sqrt(), th + 2.0 * PI * 7.0 * s.sqrt());
            k[0].push((rad * ang.cos()) as f32);
            k[1].push((rad * ang.sin()) as f32);
            k[2].push(kz as f32);
            tau.push(1.6e-3 + 8e-3 * s);
        }
    }
    let m = tau.len();
    let ppm = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60];
    let amp = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048];
    let c = fat_spectrum_weights(&ppm, &amp, 2.89, &tau);
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let c32 = |v: &[C64]| -> Vec<C32> { v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect() };
    let c64v = |v: &[C32]| -> Vec<C64> { v.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect() };
    let l = 8;
    // coil batch 2 of 5: the multi path plans its own batch inside the engine's grid memory
    let probe = SegNufft::new(&dev, dims, [res; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 2, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
    let pc = probe.plan.grid_elems() * l * 8;
    drop(probe);
    let mut eng = SegNufft::new(&dev, dims, [res; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 2, frames: 1, bspline: 1, max_grid_bytes: 2 * pc + 1 }).unwrap();
    assert_eq!(eng.cb, 2);
    eng.set_maps(&st, &c32(&smaps)).unwrap();
    let tau32: Vec<f32> = tau.iter().map(|&t| t as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| r.uniform() as f32).collect();
    let dc: Vec<f32> = (0..6 * m).map(|_| (200.0 * (2.0 * r.uniform() - 1.0)) as f32).collect();
    let pts = eng.points(&st, [&k[0], &k[1], &k[2]], &tau32, &phi0, &dc, None).unwrap();
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    let (segph, atab, t0, dt) = build_segments(nv, l, tau[0], tau[m - 1], na, &fitv, &psi);
    let seg = eng.segments(&st, &segph, &atab, t0, dt).unwrap();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let mut inner = SegSets { eng, pts: vec![pts], segs: vec![seg], seg_of_set: vec![0], kern, surrogate: None };
    let x = r.complex_vec(nv);
    let yr = r.complex_vec(nc * m);
    let (dx, dyr) = (st.clone_htod(&c32(&x)).unwrap(), st.clone_htod(&c32(&yr)).unwrap());
    let run = |op: &mut dyn SetOp| -> (Vec<C64>, Vec<C64>) {
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
        op.forward(&st, 0, &dx, &mut y).unwrap();
        op.adjoint(&st, 0, &dyr, &mut xa).unwrap();
        (c64v(&st.clone_dtoh(&y).unwrap()), c64v(&st.clone_dtoh(&xa).unwrap()))
    };
    for weighted in [false, true] {
        let mut wsets;
        let base: &mut dyn SetOp = if weighted {
            let w = WeightedSets::basis_weights([&k[0], &k[1], &k[2]], 4);
            wsets = WeightedSets::new(&st, &mut inner, vec![w]).unwrap();
            &mut wsets
        } else {
            &mut inner
        };
        let mut op = FatSets::new(&st, base, &ff, vec![c.clone()]).unwrap();
        assert!(op.shared_gridding);
        let (ys, xs) = run(&mut op);
        op.shared_gridding = false;
        let (yo, xo) = run(&mut op);
        let (ef, ea) = (rel_on(&ys, &yo, None), rel_on(&xs, &xo, None));
        eprintln!("FatSets shared vs separate{}: forward {ef:.2e}, adjoint {ea:.2e}", if weighted { " (through WeightedSets)" } else { "" });
        assert!(ef < 5e-5 && ea < 5e-5, "shared vs separate gridding {ef:.2e} / {ea:.2e}"); // float32
    }
    let (g, cbm, extra) = inner.eng.multi_info(&st, 2).unwrap();
    eprintln!("multi plan for two images: {g} per pass, coil batch {cbm} (engine 2), extra {extra} B");
    assert_eq!((g, extra), (2, 0), "two images inside the engine's grid memory");
}

/// geometry of the joint tests (as `fat_sets_match_direct_sum_and_are_adjoint`): grid indices, coil maps, field (rad/s),
/// fat fraction (a shell with mixed edges), spiral-like readouts over 8 ms, six-peak spectrum at 2.89 T
struct Jg {
    dims: [usize; 3],
    nv: usize,
    nc: usize,
    idx: Vec<[f64; 3]>,
    smaps: Vec<C64>,
    om: Vec<f64>,
    ff: Vec<f32>,
    k: [Vec<f32>; 3],
    tau: Vec<f64>,
    na: usize,
    ppm: [f64; 6],
    amp: Vec<f64>,
}

fn joint_geometry() -> Jg {
    let dims = [20usize, 18, 12];
    let nv: usize = dims.iter().product();
    let nc = 3;
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [i as f64 - (dims[0] / 2) as f64, j as f64 - (dims[1] / 2) as f64, k as f64 - (dims[2] / 2) as f64]
    }).collect();
    let smaps: Vec<C64> = (0..nc).flat_map(|c| idx.iter().map(move |p| C64::from_polar(1.0 + 0.3 * (p[0] * (c as f64 + 1.0) / 10.0).cos(), 0.1 * c as f64 * p[1]))).collect();
    let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * 150.0 * ((p[0] / 10.0) * 0.8 + (p[1] / 9.0).powi(2) * 0.4 - 0.2 * (p[2] / 6.0))).collect();
    let ff: Vec<f32> = idx.iter().map(|p| {
        let rr = ((p[0] / 10.0).powi(2) + (p[1] / 9.0).powi(2) + (p[2] / 6.0).powi(2)).sqrt();
        (1.0 - ((rr - 0.7).abs() * 10.0 / 1.5 - 0.5).clamp(0.0, 1.0)) as f32
    }).collect();
    let mut r = Rng::new(17);
    let (nro, na, te, t_ro) = (8usize, 400usize, 1.6e-3, 8e-3);
    let mut k: [Vec<f32>; 3] = Default::default();
    let mut tau = vec![];
    for _ in 0..nro {
        let (kz, th) = (0.97 * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
        for n in 0..na {
            let s = n as f64 / (na - 1) as f64;
            let (rad, ang) = (0.97 * PI * s.sqrt(), th + 2.0 * PI * 7.0 * s.sqrt());
            k[0].push((rad * ang.cos()) as f32);
            k[1].push((rad * ang.sin()) as f32);
            k[2].push(kz as f32);
            tau.push((te + t_ro * s) as f32 as f64);
        }
    }
    let ppm = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60];
    let amp: Vec<f64> = { let a = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]; let s: f64 = a.iter().sum(); a.iter().map(|v| v / s).collect() };
    Jg { dims, nv, nc, idx, smaps, om, ff, k, tau, na, ppm, amp }
}

/// `--fat-model joint` (one image, the water / fat mix in the segment diagonals) on the GPU:
///   (a) one map folded into the diagonals (`build_segments_complex`, plain SegSets) vs the exact direct sum and vs
///       FatSets (two species, L = 10);
///   (b) the maps kept separate (`build_segments_fat`, `SegNufft::segments_fat`, FatJointSets with one map) vs the
///       same table with the map folded in (`fold_fat_fraction`, plain SegSets): equal to f32 rounding — the fused
///       pad / crop kernels — and vs the direct sum;
///   (c) a time-resolved set (3 frames, linear B-spline) with a different fat map per frame (FatJointSets, map f for
///       frame f) vs the exact per-frame direct sum.
/// Asserts: model + implementation ≤ 2e-3 against the direct sum (CPU budget at L = 12 for this field: ~5e-4), joint
/// vs FatSets ≤ 3e-3, (b) folded vs separate maps ≤ 1e-5, adjoint identity ≤ 1e-5 everywhere.
#[test]
fn fat_joint_matches_direct_sum_and_fat_sets() {
    use hasty_nufft::{build_segments_complex, build_segments_fat, fold_fat_fraction};
    use hasty_recon::setop::FatJointSets;
    use hasty_testkit::physops::bspline_weight;
    let g = joint_geometry();
    let (nv, nc, m) = (g.nv, g.nc, g.tau.len());
    let res = 3e-3;
    let cj: Vec<C64> = fat_spectrum_weights(&g.ppm, &g.amp, 2.89, &g.tau).iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect();
    let cfat = |t: f64| -> C64 { let w = fat_spectrum_weights(&g.ppm, &g.amp, 2.89, &[t])[0]; C64::new(w.re as f64, w.im as f64) };
    let kq = |j: usize, q: usize| g.k[0][j] as f64 * g.idx[q][0] + g.k[1][j] as f64 * g.idx[q][1] + g.k[2][j] as f64 * g.idx[q][2];
    let mix = |ff: &[f32], j: usize, q: usize| C64::new(1.0 - ff[q] as f64, 0.0) + ff[q] as f64 * cj[j];
    let vox: Vec<usize> = (0..nv).collect();
    let mut r = Rng::new(3);
    let x = r.complex_vec(3 * nv);
    let yr = r.complex_vec(nc * m);
    let ds = DirectSum { nv, nc, m, frames: 1, smaps: &g.smaps, voxels: &vox, bw: None };
    let (yref, xref) = ds.apply(&|j, q| C64::from_polar(1.0, -(kq(j, q) + g.om[q] * g.tau[j])) * mix(&g.ff, j, q), &x[..nv], &yr);
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let c32 = |v: &[C64]| -> Vec<C32> { v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect() };
    let c64v = |v: &[C32]| -> Vec<C64> { v.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect() };
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let tau32: Vec<f32> = g.tau.iter().map(|&t| t as f32).collect();
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let psi = |v: usize, t: f64| g.om[v] * t;
    let (t0, t1) = (g.tau[0], g.tau[m - 1]);
    let sets = |l: usize, frames: usize, segph: &[C32], atab: &[C32], cl: Option<&[C32]>, ta: f64, dt: f64, t: Option<&[f32]>| -> SegSets {
        let mut eng = SegNufft::new(&dev, g.dims, [res; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 0, frames, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
        eng.set_maps(&st, &c32(&g.smaps)).unwrap();
        let pts = eng.points(&st, [&g.k[0], &g.k[1], &g.k[2]], &tau32, &vec![0f32; m], &[], t).unwrap();
        let seg = match cl { Some(cl) => eng.segments_fat(&st, segph, atab, cl, ta, dt).unwrap(), None => eng.segments(&st, segph, atab, ta, dt).unwrap() };
        SegSets { eng, pts: vec![pts], segs: vec![seg], seg_of_set: vec![0], kern: kern.clone(), surrogate: None }
    };
    let run = |op: &mut dyn SetOp, xin: &[C64], len: usize| -> (Vec<C64>, Vec<C64>, f64) {
        let (dx, dy) = (st.clone_htod(&c32(xin)).unwrap(), st.clone_htod(&c32(&yr)).unwrap());
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xa = st.alloc_zeros::<C32>(len).unwrap();
        op.forward(&st, 0, &dx, &mut y).unwrap();
        op.adjoint(&st, 0, &dy, &mut xa).unwrap();
        let (y, xa) = (c64v(&st.clone_dtoh(&y).unwrap()), c64v(&st.clone_dtoh(&xa).unwrap()));
        let l: C64 = yr.iter().zip(&y).map(|(a, b)| a.conj() * b).sum();
        let rr: C64 = xa.iter().zip(xin).map(|(a, b)| a.conj() * b).sum();
        (y, xa, (l - rr).norm() / l.norm())
    };
    let l = 12;
    // (a) one map folded, table fitted to g
    let gv = |v: usize, t: f64| C64::from_polar(1.0, -psi(v, t)) * (C64::new(1.0 - g.ff[v] as f64, 0.0) + g.ff[v] as f64 * cfat(t));
    let (sa, aa, ta, dta) = build_segments_complex(nv, l, t0, t1, g.na, &fitv, &gv);
    let mut ja = sets(l, 1, &sa, &aa, None, ta, dta, None);
    let (ya, xa, da) = run(&mut ja, &x[..nv], nv);
    // FatSets reference (two species, B0 segments, L = 10)
    let (sb, ab, tb, dtb) = hasty_nufft::build_segments(nv, 10, t0, t1, g.na, &fitv, &psi);
    let mut inner = sets(10, 1, &sb, &ab, None, tb, dtb, None);
    let mut fsets = FatSets::new(&st, &mut inner, &g.ff, vec![c32(&cj)]).unwrap();
    let (yf, xf, _) = run(&mut fsets, &x[..nv], nv);
    let (ef, ea, gf, ga) = (rel_on(&ya, &yref, None), rel_on(&xa, &xref, None), rel_on(&ya, &yf, None), rel_on(&xa, &xf, None));
    eprintln!("(a) joint L = {l}, folded map: vs direct sum {ef:.2e} / {ea:.2e}, vs FatSets (L 10, 20 channels) {gf:.2e} / {ga:.2e}, adjoint identity {da:.1e}");
    assert!(ef < 2e-3 && ea < 2e-3 && gf < 3e-3 && ga < 3e-3 && da < 1e-3, "(a) {ef:.2e} {ea:.2e} {gf:.2e} {ga:.2e} identity {da:.1e}"); // float32 identity (measured 1.5e-4)
    // (b) ff-independent table: maps separate (fused kernels) vs folded
    let (sw, aw, cl, tw, dtw) = build_segments_fat(nv, l, t0, t1, g.na, &fitv, &psi, &cfat);
    let mut jf = sets(l, 1, &fold_fat_fraction(&sw, &cl, &g.ff), &aw, None, tw, dtw, None);
    let (yb1, xb1, _) = run(&mut jf, &x[..nv], nv);
    let js = sets(l, 1, &sw, &aw, Some(&cl), tw, dtw, None);
    let maps = js.eng.fat_maps(&st, &g.ff).unwrap();
    let mut jm = FatJointSets { sets: js, maps, map_of_set: vec![Some(0)] };
    let (yb2, xb2, db) = run(&mut jm, &x[..nv], nv);
    let (sf, sa_) = (rel_on(&yb2, &yb1, None), rel_on(&xb2, &xb1, None));
    let (ebf, eba) = (rel_on(&yb2, &yref, None), rel_on(&xb2, &xref, None));
    eprintln!("(b) ff-independent table, L = {l}: separate maps vs folded {sf:.2e} / {sa_:.2e}; vs direct sum {ebf:.2e} / {eba:.2e}, adjoint identity {db:.1e}");
    assert!(sf < 5e-5 && sa_ < 5e-5 && ebf < 2e-3 && eba < 2e-3 && db < 1e-3, "(b) {sf:.2e} {sa_:.2e} {ebf:.2e} {eba:.2e} identity {db:.1e}"); // float32 (measured 4.6e-6 / 1.7e-5, identity 2.8e-4)
    // (c) 3 frames, linear B-spline, a fat map per frame (the shell moved by one voxel per frame along x)
    let frames = 3;
    let ffs: Vec<f32> = (0..frames).flat_map(|f| (0..nv).map(move |q| (q, f))).map(|(q, f)| {
        let p = g.idx[q];
        let rr = (((p[0] - f as f64) / 10.0).powi(2) + (p[1] / 9.0).powi(2) + (p[2] / 6.0).powi(2)).sqrt();
        (1.0 - ((rr - 0.7).abs() * 10.0 / 1.5 - 0.5).clamp(0.0, 1.0)) as f32
    }).collect();
    let tf: Vec<f32> = (0..m).map(|_| (frames as f64 * r.uniform()) as f32).collect();
    let bw: Vec<f64> = (0..m).flat_map(|j| (0..frames).map(move |f| (j, f))).map(|(j, f)| bspline_weight(frames, 1, f, tf[j] as f64)).collect();
    let (mut ytr, mut xtr) = (vec![C64::new(0.0, 0.0); nc * m], vec![C64::new(0.0, 0.0); frames * nv]);
    for f in 0..frames {
        let fff = &ffs[f * nv..(f + 1) * nv];
        let yw: Vec<C64> = (0..nc * m).map(|n| yr[n] * bw[(n % m) * frames + f]).collect();
        let (yf_, xf_) = ds.apply(&|j, q| C64::from_polar(1.0, -(kq(j, q) + g.om[q] * g.tau[j])) * mix(fff, j, q), &x[f * nv..(f + 1) * nv], &yw);
        for n in 0..nc * m {
            ytr[n] += yf_[n] * bw[(n % m) * frames + f];
        }
        xtr[f * nv..(f + 1) * nv].copy_from_slice(&xf_);
    }
    let jt = sets(l, frames, &sw, &aw, Some(&cl), tw, dtw, Some(&tf));
    let maps = jt.eng.fat_maps(&st, &ffs).unwrap();
    let mut jtm = FatJointSets { sets: jt, maps, map_of_set: vec![None] };
    let (yc, xc, dc) = run(&mut jtm, &x, frames * nv);
    let (ecf, eca) = (rel_on(&yc, &ytr, None), rel_on(&xc, &xtr, None));
    eprintln!("(c) 3 frames, per-frame maps: vs direct sum {ecf:.2e} / {eca:.2e}, adjoint identity {dc:.1e}");
    assert!(ecf < 2e-3 && eca < 2e-3 && dc < 1e-3, "(c) {ecf:.2e} {eca:.2e} identity {dc:.1e}"); // float32
}
