//! GPU: grouped point sets on one image (temporal pooling) — `SegNufft::forward_group` / `adjoint_group` (one grid FFT
//! per segment and coil batch for the whole group) against the member-wise operators, through the engine (resident and
//! streamed tables, prefetch on / off, time-resolved members, correction order 3, joint water + fat with a shared and
//! with a mean map), the `SetOp` layer (`SegSets` over two segment tables, `WeightedSets`, `FatSets`, `FatJointSets`),
//! the adjoint identity of the grouped operator, and pooled PCG through the grouped gradient against the replicate
//! path (resident data, data staged from a k-space store, sub-group splitting). `grouped_timing` reports FFT counts
//! and wall times of grouped vs member-wise application.
//!   cargo test --release -p hasty-recon --test grouped_sets -- --nocapture --test-threads 1

use std::f64::consts::PI;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, build_segments_fat, SegNufft, SegNufftOpts, SegPoints, NHI};
use hasty_recon::gpu::Kernels;
use hasty_recon::hoststack::{pooled_pcg_host_with, PooledGrad};
use hasty_recon::p2np::P2np;
use hasty_recon::setop::{fat_spectrum_weights, FatJointSets, FatSets, SegSets, SetOp, WeightedSets};
use hasty_testkit::Rng;
use num_complex::Complex64 as C64;

const DIMS: [usize; 3] = [20, 18, 12];
const NC: usize = 5;
const L: usize = 6;

fn c32(v: &[C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}

/// ‖a − b‖ / ‖b‖ (f64)
fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

/// Σ conj(a) b (f64)
fn dotc(a: &[C32], b: &[C32]) -> C64 {
    a.iter().zip(b).map(|(x, y)| C64::new(x.re as f64, -x.im as f64) * C64::new(y.re as f64, y.im as f64)).sum()
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

/// uniform in [0, 1) (`Rng::uniform` is in [−1, 1))
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

/// grid bytes of one coil of an engine (all segments and frame slots)
fn per_coil(dev: &Device, dims: [usize; 3], o: &SegNufftOpts) -> usize {
    let e = SegNufft::new(dev, dims, [3e-3; 3], 1, SegNufftOpts { max_grid_bytes: 1 << 30, ..*o }).unwrap();
    e.nf * o.nseg * e.plan.grid_elems() * 8
}

/// `n` point sets (order 2 or 3; frames > 1 with random sample times)
fn member_points(eng: &mut SegNufft, st: &Arc<CudaStream>, r: &mut Rng, n: usize, nro: usize, na: usize) -> Vec<SegPoints> {
    (0..n)
        .map(|_| {
            let (k, tau) = readouts(r, nro, na);
            let m = tau.len();
            let phi0: Vec<f32> = (0..m).map(|_| r.uniform() as f32).collect();
            let dc: Vec<f32> = (0..6 * m).map(|_| (150.0 * (2.0 * r.uniform() - 1.0)) as f32).collect();
            let t: Option<Vec<f32>> = (eng.frames > 1).then(|| (0..m).map(|_| (eng.frames as f64 * r.uniform()) as f32).collect());
            if eng.order >= 3 {
                let u: Vec<f32> = (0..4 * m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect();
                let h: Vec<f64> = (0..4 * NHI).map(|i| if i % NHI < 10 { 2e3 } else { 5e4 } * (2.0 * r.uniform() - 1.0)).collect();
                eng.points_hi(st, [&k[0], &k[1], &k[2]], &tau, &phi0, &dc, &u, &h, t.as_deref()).unwrap()
            } else {
                eng.points(st, [&k[0], &k[1], &k[2]], &tau, &phi0, &dc, t.as_deref()).unwrap()
            }
        })
        .collect()
}

fn b0_field(nv: usize) -> Vec<f64> {
    (0..nv).map(|q| 2.0 * PI * 90.0 * ((q % DIMS[0]) as f64 / DIMS[0] as f64 - 0.5 + 0.3 * ((q / DIMS[0]) % DIMS[1]) as f64 / DIMS[1] as f64)).collect()
}

/// the grouped operator of one engine against the member-wise one: (forward, residual, adjoint) relative differences,
/// the grouped adjoint identity and the grid transforms of (member-wise, grouped) forward + adjoint
struct Cmp {
    fwd: f64,
    res: f64,
    adj: f64,
    ident: f64,
    ffts: (usize, usize),
}

/// apply the operator `fwd(e, member, x, y)` / `adj(e, member, y, x)` member-wise and grouped; the grouped calls are
/// `gfwd(e, pts, x, data, ys)` / `gadj(e, pts, ys, x)` (data Some: residual)
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn compare(
    eng: &mut SegNufft,
    st: &Arc<CudaStream>,
    pts: &[&SegPoints],
    x: &[C32],
    ys: &[Vec<C32>],
    fwd: &mut dyn FnMut(&mut SegNufft, &SegPoints, &CudaSlice<C32>, &mut CudaSlice<C32>),
    adj: &mut dyn FnMut(&mut SegNufft, &SegPoints, &CudaSlice<C32>, &mut CudaSlice<C32>),
    gfwd: &mut dyn FnMut(&mut SegNufft, &[&SegPoints], &CudaSlice<C32>, Option<&[&CudaSlice<C32>]>, &mut [&mut CudaSlice<C32>]),
    gadj: &mut dyn FnMut(&mut SegNufft, &[&SegPoints], &[&CudaSlice<C32>], &mut CudaSlice<C32>),
) -> Cmp {
    let n = pts.len();
    let dx = st.clone_htod(x).unwrap();
    let dys: Vec<CudaSlice<C32>> = ys.iter().map(|y| st.clone_htod(y).unwrap()).collect();
    let f0 = eng.fft_calls();
    // member-wise
    let mut yref: Vec<Vec<C32>> = vec![];
    let mut xref = vec![C64::new(0.0, 0.0); x.len()];
    for j in 0..n {
        let mut y = st.alloc_zeros::<C32>(ys[j].len()).unwrap();
        fwd(eng, pts[j], &dx, &mut y);
        yref.push(st.clone_dtoh(&y).unwrap());
        let mut xa = st.alloc_zeros::<C32>(x.len()).unwrap();
        adj(eng, pts[j], &dys[j], &mut xa);
        for (a, v) in xref.iter_mut().zip(st.clone_dtoh(&xa).unwrap()) {
            *a += C64::new(v.re as f64, v.im as f64);
        }
    }
    let f1 = eng.fft_calls();
    // grouped
    let mut gy: Vec<CudaSlice<C32>> = ys.iter().map(|y| st.alloc_zeros::<C32>(y.len()).unwrap()).collect();
    {
        let mut refs: Vec<&mut CudaSlice<C32>> = gy.iter_mut().collect();
        gfwd(eng, pts, &dx, None, &mut refs);
    }
    let mut gx = st.alloc_zeros::<C32>(x.len()).unwrap();
    {
        let refs: Vec<&CudaSlice<C32>> = dys.iter().collect();
        gadj(eng, pts, &refs, &mut gx);
    }
    let f2 = eng.fft_calls();
    let gyh: Vec<Vec<C32>> = gy.iter().map(|y| st.clone_dtoh(y).unwrap()).collect();
    let gxh = st.clone_dtoh(&gx).unwrap();
    let fwd_e = (0..n).map(|j| rel(&gyh[j], &yref[j])).fold(0.0, f64::max);
    let adj_e = rel(&gxh, &c32(&xref));
    // residual r_m = A_m x − y_m
    let mut gr: Vec<CudaSlice<C32>> = ys.iter().map(|y| st.alloc_zeros::<C32>(y.len()).unwrap()).collect();
    {
        let d: Vec<&CudaSlice<C32>> = dys.iter().collect();
        let mut refs: Vec<&mut CudaSlice<C32>> = gr.iter_mut().collect();
        gfwd(eng, pts, &dx, Some(&d), &mut refs);
    }
    let res_e = (0..n)
        .map(|j| {
            let want: Vec<C32> = yref[j].iter().zip(&ys[j]).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect();
            rel(&st.clone_dtoh(&gr[j]).unwrap(), &want)
        })
        .fold(0.0, f64::max);
    // ⟨A_G x, y⟩ = Σ_m ⟨A_m x, y_m⟩ against ⟨x, A_Gᴴ y⟩
    let lhs: C64 = (0..n).map(|j| dotc(&ys[j], &gyh[j])).sum();
    let rhs = dotc(&gxh, x);
    Cmp { fwd: fwd_e, res: res_e, adj: adj_e, ident: (lhs - rhs).norm() / lhs.norm(), ffts: (f1 - f0, f2 - f1) }
}

fn plain_cmp(eng: &mut SegNufft, st: &Arc<CudaStream>, seg: &hasty_nufft::SegSegments, pts: &[&SegPoints], x: &[C32], ys: &[Vec<C32>]) -> Cmp {
    compare(
        eng,
        st,
        pts,
        x,
        ys,
        &mut |e, p, x, y| e.forward(st, seg, p, x, y).unwrap(),
        &mut |e, p, y, x| e.adjoint(st, seg, p, y, x).unwrap(),
        &mut |e, p, x, d, ys| match d {
            None => e.forward_group(st, seg, p, x, ys).unwrap(),
            Some(d) => e.forward_residual_group(st, seg, p, x, d, ys).unwrap(),
        },
        &mut |e, p, ys, x| e.adjoint_group(st, seg, p, ys, x).unwrap(),
    )
}

fn check(name: &str, c: &Cmp) {
    check_n(name, c, true)
}

/// `saving`: the group must need fewer grid transforms than its members (false for a one-member group)
fn check_n(name: &str, c: &Cmp, saving: bool) {
    eprintln!("{name}: grouped vs member-wise forward {:.1e}, residual {:.1e}, adjoint {:.1e}; adjoint identity {:.1e}; grid FFTs member-wise {} → grouped {}", c.fwd, c.res, c.adj, c.ident, c.ffts.0, c.ffts.1);
    assert!(c.fwd <= 1e-4 && c.res <= 1e-4 && c.adj <= 1e-4, "{name}: grouped differs from member-wise: forward {:.1e}, residual {:.1e}, adjoint {:.1e}", c.fwd, c.res, c.adj);
    assert!(c.ident <= 1e-4, "{name}: grouped adjoint identity {:.1e}", c.ident);
    assert!(if saving { c.ffts.1 < c.ffts.0 } else { c.ffts.1 == c.ffts.0 }, "{name}: grid transforms grouped {} vs member-wise {}", c.ffts.1, c.ffts.0);
}

#[test]
fn grouped_engine_matches_memberwise() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    let n = 5;
    let mut r = Rng::new(21);
    let sm = smaps(NC, DIMS);
    let om = b0_field(nv);
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    // a coil batch of 2 (batches 2, 2, 1): the group loop over partial batches
    let base = SegNufftOpts { sigma: 1.25, width: 10, nseg: L, order: 2, frames: 1, bspline: 1, max_grid_bytes: 0 };
    let budget = 2 * per_coil(&dev, DIMS, &base);
    let mut eng = SegNufft::new(&dev, DIMS, [3e-3; 3], NC, SegNufftOpts { max_grid_bytes: budget, ..base }).unwrap();
    assert_eq!(eng.cb, 2, "test setup: coil batch");
    eng.set_maps(&st, &sm).unwrap();
    let (sp, at, t0, dt) = build_segments(nv, L, 1.6e-3, 9.6e-3, 300, &fitv, &psi);
    let seg = eng.segments(&st, &sp, &at, t0, dt).unwrap();
    let mut pts = member_points(&mut eng, &st, &mut r, n, 6, 300);
    let x = c32(&r.complex_vec(nv));
    let ys: Vec<Vec<C32>> = pts.iter().map(|p| c32(&r.complex_vec(NC * p.m))).collect();
    {
        let refs: Vec<&SegPoints> = pts.iter().collect();
        check("resident, order 2, coil batch 2 of 5", &plain_cmp(&mut eng, &st, &seg, &refs, &x, &ys));
        // one member: the group path is the plain operator
        check_n("one member", &plain_cmp(&mut eng, &st, &seg, &refs[..1], &x, &ys[..1]), false);
    }
    // streamed (host-resident) tables, prefetch off / on
    for p in pts.iter_mut() {
        p.to_host(&dev.ctx, &st).unwrap();
    }
    for prefetch in [false, true] {
        let refs: Vec<&SegPoints> = pts.iter().collect();
        eng.stream_reserve(&st, &refs).unwrap();
        eng.stream_prefetch = prefetch;
        let up0 = eng.stream_stats().0;
        let c = plain_cmp(&mut eng, &st, &seg, &refs, &x, &ys);
        check(&format!("streamed tables, prefetch {prefetch}"), &c);
        eprintln!("  table uploads: {}", eng.stream_stats().0 - up0);
        eng.stream_release(&st).unwrap();
    }
    // mixed: two members resident, three streamed
    for p in pts.iter_mut().take(2) {
        p.to_device(&st).unwrap();
    }
    {
        let refs: Vec<&SegPoints> = pts.iter().collect();
        eng.stream_reserve(&st, &refs).unwrap();
        check("mixed resident / streamed members", &plain_cmp(&mut eng, &st, &seg, &refs, &x, &ys));
        eng.stream_release(&st).unwrap();
    }
    for p in pts.iter_mut() {
        p.to_device(&st).unwrap();
    }
    drop(pts);
    drop(eng);

    // time-resolved members (3 frames, linear B-spline): slot eviction / flush inside the group loop
    {
        let o = SegNufftOpts { frames: 3, ..base };
        let mut e3 = SegNufft::new(&dev, DIMS, [3e-3; 3], NC, SegNufftOpts { max_grid_bytes: 2 * per_coil(&dev, DIMS, &o), ..o }).unwrap();
        e3.set_maps(&st, &sm).unwrap();
        let seg3 = e3.segments(&st, &sp, &at, t0, dt).unwrap();
        let p3 = member_points(&mut e3, &st, &mut r, 3, 6, 300);
        let x3 = c32(&r.complex_vec(3 * nv));
        let y3: Vec<Vec<C32>> = p3.iter().map(|p| c32(&r.complex_vec(NC * p.m))).collect();
        let refs: Vec<&SegPoints> = p3.iter().collect();
        check("3 frames, linear B-spline", &plain_cmp(&mut e3, &st, &seg3, &refs, &x3, &y3));
    }
    // correction order 3 (points_hi: the multi-image kernels, K = 1)
    {
        let o = SegNufftOpts { order: 3, ..base };
        let mut e4 = SegNufft::new(&dev, DIMS, [3e-3; 3], NC, SegNufftOpts { max_grid_bytes: 2 * per_coil(&dev, DIMS, &o), ..o }).unwrap();
        e4.set_maps(&st, &sm).unwrap();
        let seg4 = e4.segments(&st, &sp, &at, t0, dt).unwrap();
        let mut p4 = member_points(&mut e4, &st, &mut r, 4, 6, 300);
        let y4: Vec<Vec<C32>> = p4.iter().map(|p| c32(&r.complex_vec(NC * p.m))).collect();
        {
            let refs: Vec<&SegPoints> = p4.iter().collect();
            check("order 3", &plain_cmp(&mut e4, &st, &seg4, &refs, &x, &y4));
        }
        for p in p4.iter_mut() {
            p.to_host(&dev.ctx, &st).unwrap();
        }
        let refs: Vec<&SegPoints> = p4.iter().collect();
        e4.stream_reserve(&st, &refs).unwrap();
        e4.stream_prefetch = true;
        check("order 3, streamed tables", &plain_cmp(&mut e4, &st, &seg4, &refs, &x, &y4));
    }
    // joint water + fat (segments_fat, fat maps applied at pad / crop)
    {
        let (ppm, amp) = ([-3.40, -2.60, 0.60], [0.75, 0.18, 0.07]);
        let cfat = |t: f64| -> C64 { let w = fat_spectrum_weights(&ppm, &amp, 2.89, &[t])[0]; C64::new(w.re as f64, w.im as f64) };
        let lf = L + 2;
        let o = SegNufftOpts { nseg: lf, ..base };
        let mut ef = SegNufft::new(&dev, DIMS, [3e-3; 3], NC, SegNufftOpts { max_grid_bytes: 2 * per_coil(&dev, DIMS, &o), ..o }).unwrap();
        ef.set_maps(&st, &sm).unwrap();
        let (sw, aw, cl, tw, dtw) = build_segments_fat(nv, lf, 1.6e-3, 9.6e-3, 300, &fitv, &psi, &cfat);
        let segf = ef.segments_fat(&st, &sw, &aw, &cl, tw, dtw).unwrap();
        let mut pf = member_points(&mut ef, &st, &mut r, 4, 6, 300);
        let ff: Vec<f32> = (0..4 * nv).map(|_| u01(&mut r) as f32).collect();
        let mut maps = ef.fat_maps(&st, &ff).unwrap();
        let yf: Vec<Vec<C32>> = pf.iter().map(|p| c32(&r.complex_vec(NC * p.m))).collect();
        for host in [false, true] {
            if host {
                for p in pf.iter_mut() {
                    p.to_host(&dev.ctx, &st).unwrap();
                }
                ef.stream_reserve(&st, &pf.iter().collect::<Vec<_>>()).unwrap();
                ef.stream_prefetch = true;
            }
            let refs: Vec<&SegPoints> = pf.iter().collect();
            // every member with map 2
            let c = compare(
                &mut ef,
                &st,
                &refs,
                &x,
                &yf,
                &mut |e, p, x, y| e.forward_fat(&st, &segf, p, &maps, Some(2), x, y).unwrap(),
                &mut |e, p, y, x| e.adjoint_fat(&st, &segf, p, &maps, Some(2), y, x).unwrap(),
                &mut |e, p, x, d, ys| e.forward_fat_group(&st, &segf, p, &maps, Some(2), x, d, ys).unwrap(),
                &mut |e, p, ys, x| e.adjoint_fat_group(&st, &segf, p, &maps, Some(2), ys, x).unwrap(),
            );
            check(&format!("joint water + fat, shared map{}", if host { ", streamed tables" } else { "" }), &c);
        }
        // members with maps 0..3: the group uses their mean (SegFatMaps::mean) — equal to the member-wise operator
        // with the mean map
        let mean_h: Vec<f32> = (0..nv).map(|q| (0..4).map(|i| ff[i * nv + q] as f64).sum::<f64>() as f32 / 4.0).collect();
        let mean_m = ef.fat_maps(&st, &mean_h).unwrap();
        let refs: Vec<&SegPoints> = pf.iter().collect();
        let gm = maps.mean(&st, &[0, 1, 2, 3]).unwrap();
        let c = compare(
            &mut ef,
            &st,
            &refs,
            &x,
            &yf,
            &mut |e, p, x, y| e.forward_fat(&st, &segf, p, &mean_m, Some(0), x, y).unwrap(),
            &mut |e, p, y, x| e.adjoint_fat(&st, &segf, p, &mean_m, Some(0), y, x).unwrap(),
            &mut |e, p, x, d, ys| e.forward_fat_group(&st, &segf, p, gm, Some(0), x, d, ys).unwrap(),
            &mut |e, p, ys, x| e.adjoint_fat_group(&st, &segf, p, gm, Some(0), ys, x).unwrap(),
        );
        check("joint water + fat, mean of 4 maps vs member-wise with the mean map", &c);
    }
}

/// SegSets with two segment tables (encodes) × 3 frames, order 2
fn two_encode_sets(dev: &Device, kern: &Arc<Kernels>, r: &mut Rng, budget_coils: usize) -> (SegSets, Vec<Vec<f32>>) {
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    let base = SegNufftOpts { sigma: 1.25, width: 10, nseg: L, order: 2, frames: 1, bspline: 1, max_grid_bytes: 0 };
    let mut eng = SegNufft::new(dev, DIMS, [3e-3; 3], NC, SegNufftOpts { max_grid_bytes: budget_coils * per_coil(dev, DIMS, &base), ..base }).unwrap();
    eng.set_maps(&st, &smaps(NC, DIMS)).unwrap();
    let om = b0_field(nv);
    let fitv: Vec<usize> = (0..nv).step_by(2).collect();
    let segs = (0..2)
        .map(|e| {
            let psi = |v: usize, t: f64| om[v] * t * (1.0 + 0.3 * e as f64);
            let (sp, at, t0, dt) = build_segments(nv, L, 1.6e-3, 9.6e-3, 300, &fitv, &psi);
            eng.segments(&st, &sp, &at, t0, dt).unwrap()
        })
        .collect();
    // sets t = frame · 2 + encode
    let mut pts = vec![];
    let mut taus = vec![];
    for _ in 0..6 {
        let (k, tau) = readouts(r, 5, 300);
        let m = tau.len();
        let phi0: Vec<f32> = (0..m).map(|_| r.uniform() as f32).collect();
        let dc: Vec<f32> = (0..6 * m).map(|_| (150.0 * (2.0 * r.uniform() - 1.0)) as f32).collect();
        pts.push(eng.points(&st, [&k[0], &k[1], &k[2]], &tau, &phi0, &dc, None).unwrap());
        taus.push(tau);
    }
    (SegSets { eng, pts, segs, seg_of_set: (0..6).map(|t| t % 2).collect(), kern: kern.clone(), surrogate: None }, taus)
}

/// SetOp group calls against per-set forward / adjoint: (forward, residual, adjoint) relative differences, identity
fn setop_cmp(op: &mut dyn SetOp, st: &Arc<CudaStream>, members: &[usize], x: &[C32], ys: &[Vec<C32>]) -> (f64, f64, f64, f64) {
    let dx = st.clone_htod(x).unwrap();
    let dys: Vec<CudaSlice<C32>> = ys.iter().map(|y| st.clone_htod(y).unwrap()).collect();
    let mut yref = vec![];
    let mut xref = vec![C64::new(0.0, 0.0); x.len()];
    for (j, &t) in members.iter().enumerate() {
        let mut y = st.alloc_zeros::<C32>(op.len_y(t)).unwrap();
        op.forward(st, t, &dx, &mut y).unwrap();
        yref.push(st.clone_dtoh(&y).unwrap());
        let mut xa = st.alloc_zeros::<C32>(x.len()).unwrap();
        op.adjoint(st, t, &dys[j], &mut xa).unwrap();
        for (a, v) in xref.iter_mut().zip(st.clone_dtoh(&xa).unwrap()) {
            *a += C64::new(v.re as f64, v.im as f64);
        }
    }
    let mut gy: Vec<CudaSlice<C32>> = members.iter().map(|&t| st.alloc_zeros::<C32>(op.len_y(t)).unwrap()).collect();
    op.forward_group(st, members, &dx, &mut gy.iter_mut().collect::<Vec<_>>()).unwrap();
    let mut gr: Vec<CudaSlice<C32>> = members.iter().map(|&t| st.alloc_zeros::<C32>(op.len_y(t)).unwrap()).collect();
    op.forward_residual_group(st, members, &dx, &dys.iter().collect::<Vec<_>>(), &mut gr.iter_mut().collect::<Vec<_>>()).unwrap();
    let mut gx = st.alloc_zeros::<C32>(x.len()).unwrap();
    op.adjoint_group(st, members, &dys.iter().collect::<Vec<_>>(), &mut gx).unwrap();
    let gyh: Vec<Vec<C32>> = gy.iter().map(|y| st.clone_dtoh(y).unwrap()).collect();
    let gxh = st.clone_dtoh(&gx).unwrap();
    let ef = (0..members.len()).map(|j| rel(&gyh[j], &yref[j])).fold(0.0, f64::max);
    let er = (0..members.len())
        .map(|j| {
            let want: Vec<C32> = yref[j].iter().zip(&ys[j]).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect();
            rel(&st.clone_dtoh(&gr[j]).unwrap(), &want)
        })
        .fold(0.0, f64::max);
    let ea = rel(&gxh, &c32(&xref));
    let lhs: C64 = (0..members.len()).map(|j| dotc(&ys[j], &gyh[j])).sum();
    (ef, er, ea, (lhs - dotc(&gxh, x)).norm() / lhs.norm())
}

fn check_setop(name: &str, e: (f64, f64, f64, f64)) {
    eprintln!("{name}: group vs per-set forward {:.1e}, residual {:.1e}, adjoint {:.1e}; adjoint identity {:.1e}", e.0, e.1, e.2, e.3);
    assert!(e.0 <= 1e-4 && e.1 <= 1e-4 && e.2 <= 1e-4 && e.3 <= 1e-4, "{name}: {e:?}");
}

#[test]
fn grouped_setops_match_per_set() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let nv: usize = DIMS.iter().product();
    let mut r = Rng::new(33);
    let (mut sets, taus) = two_encode_sets(&dev, &kern, &mut r, 2);
    let x = c32(&r.complex_vec(nv));
    let ys: Vec<Vec<C32>> = (0..6).map(|t| c32(&r.complex_vec(sets.len_y(t)))).collect();
    // one encode (one table), and a group across both tables (two sub-groups)
    let enc0 = [0usize, 2, 4];
    let f0 = sets.eng.fft_calls();
    check_setop("SegSets, one encode", setop_cmp(&mut sets, &st, &enc0, &x, &enc0.map(|t| ys[t].clone())));
    eprintln!("  grid FFTs: {}", sets.eng.fft_calls() - f0);
    let all: Vec<usize> = (0..6).collect();
    check_setop("SegSets, both encodes (two sub-groups)", setop_cmp(&mut sets, &st, &all, &x, &ys));
    // voxel-basis weights around it
    {
        let w: Vec<Vec<C32>> = (0..6).map(|t| (0..sets.pts[t].m).map(|_| C32::new(0.5 + r.uniform() as f32, 0.3 * r.uniform() as f32)).collect()).collect();
        let mut ws = WeightedSets::new(&st, &mut sets, w).unwrap();
        check_setop("WeightedSets(SegSets)", setop_cmp(&mut ws, &st, &enc0, &x, &enc0.map(|t| ys[t].clone())));
    }
    // two species (FatSets): one map for every set (exact), then one map per frame (the group uses the members' mean)
    let ppm = [-3.40];
    let c: Vec<Vec<C32>> = taus.iter().map(|t| fat_spectrum_weights(&ppm, &[1.0], 2.89, &t.iter().map(|&v| v as f64).collect::<Vec<_>>())).collect();
    let ffm: Vec<Vec<f32>> = (0..3).map(|_| (0..nv).map(|_| u01(&mut r) as f32).collect()).collect();
    {
        let mut fs = FatSets::new(&st, &mut sets, &ffm[0], c.clone()).unwrap();
        check_setop("FatSets(SegSets), one map", setop_cmp(&mut fs, &st, &enc0, &x, &enc0.map(|t| ys[t].clone())));
    }
    let (gy_mean, gx_mean) = {
        let mut fs = FatSets::with_maps(&st, &mut sets, &ffm, (0..6).map(|t| t / 2).collect(), c.clone()).unwrap();
        let dx = st.clone_htod(&x).unwrap();
        let mut gy: Vec<CudaSlice<C32>> = enc0.iter().map(|&t| st.alloc_zeros::<C32>(fs.len_y(t)).unwrap()).collect();
        fs.forward_group(&st, &enc0, &dx, &mut gy.iter_mut().collect::<Vec<_>>()).unwrap();
        let dys: Vec<CudaSlice<C32>> = enc0.iter().map(|&t| st.clone_htod(&ys[t]).unwrap()).collect();
        let mut gx = st.alloc_zeros::<C32>(nv).unwrap();
        fs.adjoint_group(&st, &enc0, &dys.iter().collect::<Vec<_>>(), &mut gx).unwrap();
        (gy.iter().map(|y| st.clone_dtoh(y).unwrap()).collect::<Vec<_>>(), st.clone_dtoh(&gx).unwrap())
    };
    {
        let mean: Vec<f32> = (0..nv).map(|q| (ffm[0][q] + ffm[1][q] + ffm[2][q]) / 3.0).collect();
        let mut fs = FatSets::new(&st, &mut sets, &mean, c.clone()).unwrap();
        let dx = st.clone_htod(&x).unwrap();
        let mut ey = 0.0f64;
        let mut xs = vec![C64::new(0.0, 0.0); nv];
        for (j, &t) in enc0.iter().enumerate() {
            let mut y = st.alloc_zeros::<C32>(fs.len_y(t)).unwrap();
            fs.forward(&st, t, &dx, &mut y).unwrap();
            ey = ey.max(rel(&gy_mean[j], &st.clone_dtoh(&y).unwrap()));
            let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
            fs.adjoint(&st, t, &st.clone_htod(&ys[t]).unwrap(), &mut xa).unwrap();
            for (a, v) in xs.iter_mut().zip(st.clone_dtoh(&xa).unwrap()) {
                *a += C64::new(v.re as f64, v.im as f64);
            }
        }
        let ex = rel(&gx_mean, &c32(&xs));
        eprintln!("FatSets(SegSets), per-frame maps: group (mean map) vs per-set with the mean map: forward {ey:.1e}, adjoint {ex:.1e}");
        assert!(ey <= 1e-4 && ex <= 1e-4);
    }
    // joint water + fat (FatJointSets) with streamed tables: one map for all, then per-frame maps (mean)
    {
        let cfat = |t: f64| -> C64 { let w = fat_spectrum_weights(&ppm, &[1.0], 2.89, &[t])[0]; C64::new(w.re as f64, w.im as f64) };
        let om = b0_field(nv);
        let fitv: Vec<usize> = (0..nv).step_by(2).collect();
        let segs: Vec<_> = (0..2)
            .map(|e| {
                let psi = |v: usize, t: f64| om[v] * t * (1.0 + 0.3 * e as f64);
                let (sw, aw, cl, tw, dtw) = build_segments_fat(nv, L, 1.6e-3, 9.6e-3, 300, &fitv, &psi, &cfat);
                sets.eng.segments_fat(&st, &sw, &aw, &cl, tw, dtw).unwrap()
            })
            .collect();
        let ffl: Vec<f32> = ffm.concat();
        let maps = sets.eng.fat_maps(&st, &ffl).unwrap();
        let SegSets { eng, pts, seg_of_set, kern: k2, .. } = sets;
        let mut js = FatJointSets { sets: SegSets { eng, pts, segs, seg_of_set, kern: k2, surrogate: None }, maps, map_of_set: vec![Some(1); 6] };
        js.place_tables(&dev.ctx, &st, true).unwrap();
        js.sets.eng.stream_prefetch = true;
        check_setop("FatJointSets, one map, streamed tables", setop_cmp(&mut js, &st, &enc0, &x, &enc0.map(|t| ys[t].clone())));
        check_setop("FatJointSets, one map, both encodes", setop_cmp(&mut js, &st, &all, &x, &ys));
        // per-frame maps: grouped (mean of maps 0..2) vs per set with the mean as every set's map
        js.map_of_set = (0..6).map(|t| Some(t / 2)).collect();
        let dx = st.clone_htod(&x).unwrap();
        let mut gy: Vec<CudaSlice<C32>> = enc0.iter().map(|&t| st.alloc_zeros::<C32>(js.len_y(t)).unwrap()).collect();
        js.forward_group(&st, &enc0, &dx, &mut gy.iter_mut().collect::<Vec<_>>()).unwrap();
        let gyh: Vec<Vec<C32>> = gy.iter().map(|y| st.clone_dtoh(y).unwrap()).collect();
        let mean: Vec<f32> = (0..nv).map(|q| (ffm[0][q] as f64 + ffm[1][q] as f64 + ffm[2][q] as f64) as f32 / 3.0).collect();
        js.maps = js.sets.eng.fat_maps(&st, &mean).unwrap();
        js.map_of_set = vec![Some(0); 6];
        let mut ey = 0.0f64;
        for (j, &t) in enc0.iter().enumerate() {
            let mut y = st.alloc_zeros::<C32>(js.len_y(t)).unwrap();
            js.forward(&st, t, &dx, &mut y).unwrap();
            ey = ey.max(rel(&gyh[j], &st.clone_dtoh(&y).unwrap()));
        }
        eprintln!("FatJointSets, per-frame maps: group (mean map) vs per-set with the mean map: forward {ey:.1e}");
        assert!(ey <= 1e-4);
    }
}

#[test]
fn pooled_pcg_grouped_matches_replicate() {
    use hasty_recon::kstore::{KStore, StoreSpec};
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let nv: usize = DIMS.iter().product();
    let mut r = Rng::new(45);
    let (mut sets, _) = two_encode_sets(&dev, &kern, &mut r, 2);
    // object per encode (one phase ramp per encode), data = E_t x_e + noise
    let xt: Vec<Vec<C32>> = (0..2)
        .map(|e| {
            (0..nv)
                .map(|q| {
                    let (i, j, k) = (q % DIMS[0], (q / DIMS[0]) % DIMS[1], q / (DIMS[0] * DIMS[1]));
                    let u = [i as f64 / DIMS[0] as f64 - 0.5, j as f64 / DIMS[1] as f64 - 0.5, k as f64 / DIMS[2] as f64 - 0.5];
                    let a = if u[0] * u[0] / 0.16 + u[1] * u[1] / 0.12 + u[2] * u[2] / 0.1 < 1.0 { 1.0 + 0.3 * (u[0] * 9.0).sin() } else { 0.0 };
                    let z = C64::from_polar(a, 0.8 * e as f64 * u[1]);
                    C32::new(z.re as f32, z.im as f32)
                })
                .collect()
        })
        .collect();
    let ys: Vec<CudaSlice<C32>> = (0..6)
        .map(|t| {
            let mut y = st.alloc_zeros::<C32>(sets.len_y(t)).unwrap();
            sets.forward(&st, t, &st.clone_htod(&xt[t % 2]).unwrap(), &mut y).unwrap();
            let mut h = st.clone_dtoh(&y).unwrap();
            for z in h.iter_mut() {
                z.re += 2e-3 * (r.uniform() as f32 - 0.5);
                z.im += 2e-3 * (r.uniform() as f32 - 0.5);
            }
            st.clone_htod(&h).unwrap()
        })
        .collect();
    let group_of: Vec<usize> = (0..6).map(|t| t % 2).collect();
    let iters = 6;
    let run = |sets: &mut SegSets, mode: PooledGrad, staged: bool, group_bytes: usize| -> (Vec<C32>, usize, f64) {
        let mut p = if staged {
            let ks = KStore::create(StoreSpec::parse("ram", std::path::Path::new("/unused")).unwrap(), NC, (0..6).map(|t| sets.pts[t].m).collect(), None, "").unwrap();
            for (t, y) in ys.iter().enumerate() {
                ks.put(t, &st.clone_dtoh(y).unwrap()).unwrap();
            }
            P2np::from_store(sets, hasty_recon::kstage::Stager::new(&dev.ctx, Arc::new(ks)).unwrap(), vec![], &st).unwrap()
        } else {
            P2np::new(sets, ys.iter().collect(), vec![], &st).unwrap()
        };
        p.group_bytes = group_bytes;
        let mut xg = vec![vec![C32::new(0.0, 0.0); nv]; 2];
        let t0 = std::time::Instant::now();
        pooled_pcg_host_with(&mut p, &st, &group_of, &mut xg, iters, false, None, 0, mode).unwrap();
        st.synchronize().unwrap();
        let dt = t0.elapsed().as_secs_f64();
        drop(p);
        (xg.concat(), sets.eng.fft_calls(), dt)
    };
    let f0 = sets.eng.fft_calls();
    let (x_rep, f1, t_rep) = run(&mut sets, PooledGrad::Replicate, false, 0);
    let (x_grp, f2, t_grp) = run(&mut sets, PooledGrad::Grouped, false, 0);
    let (x_stg, _, _) = run(&mut sets, PooledGrad::Grouped, true, 0);
    // a residual budget of one member: every member its own sub-group (the splitting path)
    let (x_sub, _, _) = run(&mut sets, PooledGrad::Grouped, true, 1);
    let (e_grp, e_stg, e_sub) = (rel(&x_grp, &x_rep), rel(&x_stg, &x_rep), rel(&x_sub, &x_rep));
    let e_obj = rel(&x_rep, &xt.concat());
    eprintln!(
        "pooled PCG, {iters} iterations, 2 groups of 3 sets: grouped vs replicate {e_grp:.1e} (store-staged {e_stg:.1e}, one member per sub-group {e_sub:.1e}); replicate vs object {e_obj:.2e}; grid FFTs replicate {} / grouped {}; wall {t_rep:.2}s / {t_grp:.2}s",
        f1 - f0,
        f2 - f1
    );
    assert!(e_grp <= 1e-4 && e_stg <= 1e-4 && e_sub <= 1e-4, "pooled PCG grouped vs replicate: {e_grp:.1e} {e_stg:.1e} {e_sub:.1e}");
    assert!(e_obj < 0.5, "pooled PCG did not approach the object: {e_obj:.2e}");
    assert!(f2 - f1 < f1 - f0, "grouped pooled PCG did not save transforms");
}

/// FFT counts and wall time of grouped vs member-wise forward + adjoint: (member-wise ms, grouped ms, FFTs, FFTs)
fn time_case(dev: &Device, dims: [usize; 3], nc: usize, l: usize, n: usize, nro: usize, na: usize, max_grid: usize) -> (f64, f64, usize, usize) {
    let st = dev.default_stream();
    let nv: usize = dims.iter().product();
    let mut r = Rng::new(7);
    let o = SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 2, frames: 1, bspline: 1, max_grid_bytes: max_grid };
    let mut eng = SegNufft::new(dev, dims, [2e-3; 3], nc, o).unwrap();
    eng.set_maps(&st, &smaps(nc, dims)).unwrap();
    let om: Vec<f64> = (0..nv).map(|q| 2.0 * PI * 150.0 * ((q % dims[0]) as f64 / dims[0] as f64 - 0.5)).collect();
    let fitv: Vec<usize> = (0..nv).step_by(3).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    let (sp, at, t0, dt) = build_segments(nv, l, 1.6e-3, 9.6e-3, 300, &fitv, &psi);
    let seg = eng.segments(&st, &sp, &at, t0, dt).unwrap();
    let pts = member_points(&mut eng, &st, &mut r, n, nro, na);
    let refs: Vec<&SegPoints> = pts.iter().collect();
    let x = st.clone_htod(&c32(&r.complex_vec(nv))).unwrap();
    let mut ys: Vec<CudaSlice<C32>> = pts.iter().map(|p| st.alloc_zeros::<C32>(nc * p.m).unwrap()).collect();
    let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
    let mut xt = st.alloc_zeros::<C32>(nv).unwrap();
    let kern = Kernels::new(dev).unwrap();
    let one = C64::new(1.0, 0.0);
    let mut member = |eng: &mut SegNufft, ys: &mut [CudaSlice<C32>]| {
        for j in 0..n {
            eng.forward(&st, &seg, refs[j], &x, &mut ys[j]).unwrap();
        }
        for j in 0..n {
            eng.adjoint(&st, &seg, refs[j], &ys[j], &mut xt).unwrap();
            kern.axpby(&st, one, &xt, if j == 0 { C64::new(0.0, 0.0) } else { one }, &mut xa).unwrap();
        }
    };
    let grouped = |eng: &mut SegNufft, ys: &mut [CudaSlice<C32>], xa: &mut CudaSlice<C32>| {
        eng.forward_group(&st, &seg, &refs, &x, &mut ys.iter_mut().collect::<Vec<_>>()).unwrap();
        eng.adjoint_group(&st, &seg, &refs, &ys.iter().collect::<Vec<_>>(), xa).unwrap();
    };
    let mut time = |f: &mut dyn FnMut(&mut SegNufft)| -> (f64, usize) {
        f(&mut eng);
        st.synchronize().unwrap();
        let f0 = eng.fft_calls();
        let t = std::time::Instant::now();
        for _ in 0..3 {
            f(&mut eng);
        }
        st.synchronize().unwrap();
        (t.elapsed().as_secs_f64() / 3.0, (eng.fft_calls() - f0) / 3)
    };
    let (tm, fm) = time(&mut |e| member(e, &mut ys));
    let mut xg = st.alloc_zeros::<C32>(nv).unwrap();
    let (tg, fg) = time(&mut |e| grouped(e, &mut ys, &mut xg));
    eprintln!(
        "grouped timing ({dims:?}, {nc} coils, coil batch {}, L = {l}, {n} members of {} samples): forward + adjoint member-wise {:.1} ms ({fm} grid FFTs of {} channels), grouped {:.1} ms ({fg} FFTs): ×{:.2}",
        eng.cb,
        pts[0].m,
        tm * 1e3,
        eng.cb * l,
        tg * 1e3,
        tm / tg
    );
    (tm, tg, fm, fg)
}

/// FFT counts and wall time of grouped vs member-wise application: a gridding-bound case (48 × 40 × 32, 8 coils,
/// L = 16, 8 members of 12 000 samples) and an FFT-bound one (96 × 80 × 64, coil batch 2 of 8: 4 batches)
#[test]
fn grouped_timing() {
    let dev = Device::new(0).expect("CUDA device");
    for (dims, max_grid) in [([48usize, 40, 32], 512usize << 20), ([96, 80, 64], 256 << 20)] {
        let n = 8;
        let (_, _, fm, fg) = time_case(&dev, dims, 8, 16, n, 20, 600, max_grid);
        assert_eq!(fm, n * fg, "grouped: one transform set per group instead of per member");
    }
}
