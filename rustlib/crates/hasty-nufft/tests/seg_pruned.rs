//! GPU: the pruned fine-grid transforms (`hasty_fft::PrunedChanFft3`, default) give the same SegNufft forward /
//! adjoint as the full partial-axis transforms (`HASTY_SEG_NOPRUNE=1`): static and time-resolved, orders 0, 2 and 3,
//! the multi-image path; the fused residual `forward_residual` equals forward − y; timings of forward + adjoint pairs
//! (pruned vs full) at a NUFFT-like size.
//!   cargo test --release -p hasty-nufft --test seg_pruned -- --nocapture --test-threads 1

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaStream;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, hi_exponents, SegNufft, SegNufftOpts, NHI};
use hasty_testkit::Rng;

const PI: f64 = std::f64::consts::PI;

fn c32(v: &[hasty_testkit::C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut n, mut d) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        n += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        d += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (n / d.max(1e-300)).sqrt()
}

struct Case {
    k: [Vec<f32>; 3],
    tau: Vec<f32>,
    phi0: Vec<f32>,
    dc: Vec<f32>,
    t: Vec<f32>,
    u: Vec<f32>,
}

fn case(nro: usize, na: usize, frames: usize, seed: u64) -> Case {
    case_r(nro, na, frames, seed, 0.97, 0.97)
}

/// stack of spirals: in-plane radius `rad` · π, kz in ±`rz` · π
fn case_r(nro: usize, na: usize, frames: usize, seed: u64, rad: f64, rz: f64) -> Case {
    let mut r = Rng::new(seed);
    let mut k: [Vec<f32>; 3] = Default::default();
    let mut tau = vec![];
    for _ in 0..nro {
        let (kz, th) = (rz * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
        for i in 0..na {
            let q = i as f64 / (na - 1) as f64;
            let (rad, ang) = (rad * PI * q.sqrt(), th + 2.0 * PI * 7.0 * q.sqrt());
            k[0].push((rad * ang.cos()) as f32);
            k[1].push((rad * ang.sin()) as f32);
            k[2].push(kz as f32);
            tau.push((1.6e-3 + 6e-3 * q) as f32);
        }
    }
    let m = tau.len();
    Case {
        k,
        phi0: (0..m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect(),
        dc: (0..6 * m).map(|_| (100.0 * (2.0 * r.uniform() - 1.0)) as f32).collect(),
        t: (0..m).map(|_| (frames as f64 * r.uniform()) as f32).collect(),
        u: (0..4 * m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect(),
        tau,
    }
}

/// 3-D radial (kooshball): `nro` spokes of `na` samples in random directions, radius up to `rad` · π (a ball: the kz
/// range of the support differs per column)
fn case_ball(nro: usize, na: usize, seed: u64, rad: f64) -> Case {
    let mut cs = case_r(nro, na, 1, seed, rad, rad);
    let mut r = Rng::new(seed ^ 0x5eed);
    for s in 0..nro {
        let (cz, ph) = (r.uniform(), PI * r.uniform());   // uniform: [−1, 1)
        let sz = (1.0 - cz * cz).sqrt();
        for i in 0..na {
            let q = rad * PI * (2.0 * i as f64 / (na - 1) as f64 - 1.0);
            let j = s * na + i;
            cs.k[0][j] = (q * sz * ph.cos()) as f32;
            cs.k[1][j] = (q * sz * ph.sin()) as f32;
            cs.k[2][j] = (q * cz) as f32;
        }
    }
    cs
}

/// (forward, adjoint, residual) of one engine on a case
#[allow(clippy::too_many_arguments)]
fn apply(dev: &Device, st: &Arc<CudaStream>, dims: [usize; 3], nc: usize, l: usize, order: u32, frames: usize, cs: &Case, timing: bool) -> (Vec<C32>, Vec<C32>, Vec<C32>, bool, f64, f64) {
    let nv: usize = dims.iter().product();
    let mut eng = SegNufft::new(dev, dims, [3e-3; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order, frames, bspline: 1, max_grid_bytes: 2 << 30 }).unwrap();
    let mut r = Rng::new(3);
    eng.set_maps(st, &c32(&r.complex_vec(nc * nv))).unwrap();
    let om: Vec<f64> = (0..nv).map(|q| 2.0 * PI * 150.0 * ((q % dims[0]) as f64 / dims[0] as f64 - 0.5)).collect();
    let fitv: Vec<usize> = (0..nv).step_by(3).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    let (sp, at, t0, dt) = build_segments(nv, l, 1.6e-3, 7.6e-3, 400, &fitv, &psi);
    let seg = eng.segments(st, &sp, &at, t0, dt).unwrap();
    let tt = (frames > 1).then_some(&cs.t[..]);
    let pts = if order >= 3 {
        let ex = hi_exponents();
        let h: Vec<f64> = (0..4 * NHI).map(|i| 0.01 * (((i * 37) % 11) as f64 / 11.0 - 0.5) / 0.03f64.powi(ex[i % NHI].iter().sum::<u32>() as i32)).collect();
        eng.points_hi(st, [&cs.k[0], &cs.k[1], &cs.k[2]], &cs.tau, &cs.phi0, &cs.dc, &cs.u, &h, tt).unwrap()
    } else {
        let dc: &[f32] = if order > 0 { &cs.dc } else { &[] };
        eng.points(st, [&cs.k[0], &cs.k[1], &cs.k[2]], &cs.tau, &cs.phi0, dc, tt).unwrap()
    };
    let m = cs.tau.len();
    let x = st.clone_htod(&c32(&r.complex_vec(frames * nv))).unwrap();
    let yd = st.clone_htod(&c32(&r.complex_vec(nc * m))).unwrap();
    let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
    let mut xa = st.alloc_zeros::<C32>(frames * nv).unwrap();
    let mut res = st.alloc_zeros::<C32>(nc * m).unwrap();
    eng.forward(st, &seg, &pts, &x, &mut y).unwrap();
    eng.adjoint(st, &seg, &pts, &yd, &mut xa).unwrap();
    eng.forward_residual(st, &seg, &pts, &x, &yd, &mut res).unwrap();
    let mut ms = 0.0;
    if timing {
        let _ = eng.autotune(st, &seg, &pts).unwrap();
        st.synchronize().unwrap();
        let mut best = f64::INFINITY;
        for _ in 0..4 {
            let t0 = Instant::now();
            for _ in 0..3 {
                eng.forward(st, &seg, &pts, &x, &mut y).unwrap();
                eng.adjoint(st, &seg, &pts, &yd, &mut xa).unwrap();
            }
            st.synchronize().unwrap();
            best = best.min(t0.elapsed().as_secs_f64() / 3.0 * 1e3);
        }
        ms = best;
    }
    let n = [eng.plan.nf[0], eng.plan.nf[1], eng.plan.nf[2]];
    let cells = eng.grid_fft_ksupport().map_or(1.0, |k| k.cell_fraction(n));
    (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap(), st.clone_dtoh(&res).unwrap(), eng.grid_fft_pruned(), ms, cells)
}

#[test]
fn pruned_grid_fft_and_fused_residual() {
    // an f32-exactness / equivalence check: pin the f32 FFT store (the BFP store's own error is tested in
    // seg_fft_store and bfp_store; its rounding differs between slab / pruned layouts by design)
    hasty_nufft::set_thread_fft_store(Some(hasty_fft::Store::F32));
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // the pruned transform against the full one with the same (separate) pad / crop kernels: the fused pad / crop
    // (default, pruned plans only) rounds differently and has its own test (seg_fft_traffic)
    unsafe { std::env::set_var("HASTY_SEG_FUSE_PADCROP", "0") };
    for (order, frames, l) in [(0u32, 1usize, 4usize), (2, 1, 6), (2, 4, 4), (3, 1, 5)] {
        let dims = [22usize, 18, 13];
        let nc = 3;
        let cs = case(6, 300, frames, 11 + order as u64);
        // single-threaded test binary: the env var is read when the engine is built
        unsafe { std::env::remove_var("HASTY_SEG_NOPRUNE") };
        let (yp, xp, rp, pruned, _, _) = apply(&dev, &st, dims, nc, l, order, frames, &cs, false);
        unsafe { std::env::set_var("HASTY_SEG_NOPRUNE", "1") };
        let (yf, xf, _, full_pruned, _, _) = apply(&dev, &st, dims, nc, l, order, frames, &cs, false);
        // run-to-run spread of the full path itself (float32 atomics in the adjoint scatter)
        let (yf2, xf2, _, _, _, _) = apply(&dev, &st, dims, nc, l, order, frames, &cs, false);
        unsafe { std::env::remove_var("HASTY_SEG_NOPRUNE") };
        let spread = rel(&xf2, &xf).max(rel(&yf2, &yf));
        assert!(pruned && !full_pruned, "pruning switch had no effect");
        // fused residual vs forward − y (same data draw: seed 3 → x then y)
        let mut r = Rng::new(3);
        let nv: usize = dims.iter().product();
        let _maps = r.complex_vec(nc * nv);
        let _x = r.complex_vec(frames * nv);
        let yd = c32(&r.complex_vec(nc * cs.tau.len()));
        let diff: Vec<C32> = yp.iter().zip(&yd).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect();
        let (ef, ea, er) = (rel(&yp, &yf), rel(&xp, &xf), rel(&rp, &diff));
        eprintln!("order {order}, frames {frames}, L {l}: pruned vs full forward {ef:.2e}, adjoint {ea:.2e} (full vs full {spread:.2e}); fused residual vs forward − y {er:.2e}");
        assert!(ef < 1e-5 && ea < (5.0 * spread).max(1e-5), "pruned transforms differ: forward {ef:.2e}, adjoint {ea:.2e} (spread {spread:.2e})");
        assert!(er < 1e-5, "fused residual differs from forward − y: {er:.2e}");
    }
    // timing at a NUFFT-like size: forward + adjoint pair, pruned vs full
    let dims = [144usize, 144, 40];
    let cs = case(120, 1000, 1, 5);
    unsafe { std::env::remove_var("HASTY_SEG_NOPRUNE") };
    let (_, _, _, _, tp, _) = apply(&dev, &st, dims, 8, 8, 2, 1, &cs, true);
    unsafe { std::env::set_var("HASTY_SEG_NOPRUNE", "1") };
    let (_, _, _, _, tf, _) = apply(&dev, &st, dims, 8, 8, 2, 1, &cs, true);
    unsafe { std::env::remove_var("HASTY_SEG_NOPRUNE") };
    eprintln!("{dims:?}, 8 coils, L 8, order 2, {} samples: forward + adjoint {tf:.1} ms full → {tp:.1} ms pruned ({:.2}×)", cs.tau.len(), tf / tp);
    unsafe { std::env::remove_var("HASTY_SEG_FUSE_PADCROP") };
}

/// k-support pruning of the grid FFTs (HASTY_SEG_KPRUNE, default on) against the block-pruned transforms
/// (HASTY_SEG_KPRUNE=0): equal forward / adjoint / residual for full and partial k coverage, static and time-resolved,
/// orders 0, 2 and 3 (multi-image path); and the support growing when a set with a larger k footprint is used on an
/// engine planned for a smaller one.
#[test]
fn kpruned_grid_fft_matches_block_pruned() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    unsafe { std::env::remove_var("HASTY_SEG_NOPRUNE") };
    // (order, frames, L, in-plane radius, kz radius; negative radius: kooshball of that radius)
    for (order, frames, l, rad, rz) in [(0u32, 1usize, 4usize, 0.97, 0.97), (2, 1, 6, 0.4, 0.9), (2, 4, 4, 0.6, 0.3), (3, 1, 5, 0.5, 0.5), (2, 1, 4, -0.97, 0.0), (0, 1, 3, -0.5, 0.0)] {
        let dims = [30usize, 26, 20];
        let nc = 3;
        let cs = if rad < 0.0 { case_ball(60, 120, 51 + order as u64, -rad) } else { case_r(6, 300, frames, 21 + order as u64, rad, rz) };
        unsafe { std::env::set_var("HASTY_SEG_KPRUNE", "0") };
        let (yb, xb, rb, _, _, _) = apply(&dev, &st, dims, nc, l, order, frames, &cs, false);
        let (yb2, xb2, rb2, _, _, _) = apply(&dev, &st, dims, nc, l, order, frames, &cs, false);
        unsafe { std::env::remove_var("HASTY_SEG_KPRUNE") };
        let (yk, xk, rk, _, _, cells) = apply(&dev, &st, dims, nc, l, order, frames, &cs, false);
        // run-to-run spread of the block-pruned path itself (f32 atomics: split work items, adjoint scatter)
        let (sf, sa) = (rel(&yb2, &yb).max(rel(&rb2, &rb)), rel(&xb2, &xb));
        let (ef, ea, er) = (rel(&yk, &yb), rel(&xk, &xb), rel(&rk, &rb));
        eprintln!("order {order}, frames {frames}, L {l}, k radius {rad} / kz {rz}: support cells {cells:.2}; k-pruned vs block-pruned forward {ef:.2e}, adjoint {ea:.2e}, residual {er:.2e} (spread forward {sf:.2e}, adjoint {sa:.2e})");
        assert!(ef < (5.0 * sf).max(1e-6) && er < (5.0 * sf).max(1e-6) && ea < (5.0 * sa).max(1e-6), "k-pruned transforms differ: forward {ef:.2e}, adjoint {ea:.2e}, residual {er:.2e}");
    }
    // growth: one engine, a small-k set first (support planned for it), then a full-k set
    let dims = [22usize, 18, 13];
    let (nc, l) = (3usize, 4usize);
    let nv: usize = dims.iter().product();
    let run = |kprune: bool| -> (Vec<C32>, Vec<C32>, [f64; 2]) {
        let mut eng = SegNufft::new(&dev, dims, [3e-3; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 2, frames: 1, bspline: 1, max_grid_bytes: 2 << 30 }).unwrap();
        eng.set_kprune(kprune).unwrap();
        let mut r = Rng::new(3);
        eng.set_maps(&st, &c32(&r.complex_vec(nc * nv))).unwrap();
        let fitv: Vec<usize> = (0..nv).step_by(3).collect();
        let psi = |v: usize, t: f64| 2.0 * PI * 100.0 * ((v % dims[0]) as f64 / dims[0] as f64 - 0.5) * t;
        let (sp, at, t0, dt) = build_segments(nv, l, 1.6e-3, 7.6e-3, 400, &fitv, &psi);
        let seg = eng.segments(&st, &sp, &at, t0, dt).unwrap();
        let small = case_r(6, 300, 1, 41, 0.3, 0.3);
        let big = case_r(6, 300, 1, 42, 0.97, 0.97);
        let ps = eng.points(&st, [&small.k[0], &small.k[1], &small.k[2]], &small.tau, &small.phi0, &small.dc, None).unwrap();
        let x = st.clone_htod(&c32(&r.complex_vec(nv))).unwrap();
        let mut ys = st.alloc_zeros::<C32>(nc * small.tau.len()).unwrap();
        eng.forward(&st, &seg, &ps, &x, &mut ys).unwrap();
        let fr_small = eng.grid_fft_ksupport().map_or(1.0, |k| k.fractions([eng.plan.nf[0], eng.plan.nf[1], eng.plan.nf[2]]).0);
        let pb = eng.points(&st, [&big.k[0], &big.k[1], &big.k[2]], &big.tau, &big.phi0, &big.dc, None).unwrap();
        let m = big.tau.len();
        let yd = st.clone_htod(&c32(&r.complex_vec(nc * m))).unwrap();
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
        eng.forward(&st, &seg, &pb, &x, &mut y).unwrap();
        eng.adjoint(&st, &seg, &pb, &yd, &mut xa).unwrap();
        let fr_big = eng.grid_fft_ksupport().map_or(1.0, |k| k.fractions([eng.plan.nf[0], eng.plan.nf[1], eng.plan.nf[2]]).0);
        (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap(), [fr_small, fr_big])
    };
    let (yb, xb, _) = run(false);
    let (yb2, xb2, _) = run(false);
    let (yk, xk, fr) = run(true);
    let (sf, sa) = (rel(&yb2, &yb), rel(&xb2, &xb));
    let (ef, ea) = (rel(&yk, &yb), rel(&xk, &xb));
    eprintln!("support growth: z-pass columns {:.2} → {:.2}; forward {ef:.2e}, adjoint {ea:.2e} (spread {sf:.2e} / {sa:.2e})", fr[0], fr[1]);
    assert!(fr[0] < fr[1], "the support did not grow");
    assert!(ef < (5.0 * sf).max(1e-6) && ea < (5.0 * sa).max(1e-6), "k-pruned transforms after growth differ: forward {ef:.2e}, adjoint {ea:.2e}");
}
