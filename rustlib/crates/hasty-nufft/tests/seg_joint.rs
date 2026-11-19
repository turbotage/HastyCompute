//! Joint coil × segment operator (hasty_nufft::segjoint, research/coil_segment_compression.md §8) on a small grid:
//!   * forward / adjoint against the f64 direct sum of the exact physics (off-resonance with a sharp step, decay),
//!     static inner engine at order 0, at a rank R with a predicted error far below the NUFFT error;
//!   * the dot-product test (adjointness across the channel batches and the accumulating inner adjoint);
//!   * the predicted (spectrum tail), the host-evaluated and the operator's error agree vs R;
//!   * order-3 per-sample corrections (inner multi kernels): joint vs today's operator with a high-L SVD basis, same
//!     sample tables;
//!   * knot-major 2-D inner engine: joint vs today's 2-D engine at high L.
//!   * fp32 TSQR build vs the fp64 Gram: same spectrum to the TSQR accuracy.

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_joint_basis, build_segments_svd_grid, seg_time_grid, JointBuildOpts, JointSegNufft, SegNufft, SegNufftOpts, NHI};
use hasty_testkit::physops::{rel_on, DirectSum};
use hasty_testkit::{Rng, C64};
use num_complex::Complex64;

const PI: f64 = std::f64::consts::PI;
const TE: f64 = 1.6e-3;
const TRO: f64 = 8e-3;

struct Geo {
    dims: [usize; 3],
    nc: usize,
    nv: usize,
    smaps: Vec<C64>,
    x: Vec<C64>,
    idx: Vec<[f64; 3]>,
    hz: Vec<f64>,
    r2: Vec<f64>,
}

fn geo() -> Geo {
    let dims = [24usize, 20, 16];
    let res = 3e-3;
    let nv = dims.iter().product::<usize>();
    let nc = 6;
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [i as f64 - (dims[0] / 2) as f64, j as f64 - (dims[1] / 2) as f64, k as f64 - (dims[2] / 2) as f64]
    }).collect();
    let pos: Vec<[f64; 3]> = idx.iter().map(|p| p.map(|v| v * res)).collect();
    let mut r = Rng::new(31);
    let x = r.complex_vec(nv);
    let mut smaps = Vec::with_capacity(nc * nv);
    for c in 0..nc {
        let ang = 2.0 * PI * c as f64 / nc as f64;
        let ctr = [0.05 * ang.cos(), 0.05 * ang.sin(), 0.01 * (c % 2) as f64];
        for p in &pos {
            let d2 = (p[0] - ctr[0]).powi(2) + (p[1] - ctr[1]).powi(2) + 0.5 * (p[2] - ctr[2]).powi(2);
            smaps.push(C64::from_polar((-d2 / 0.003).exp() + 0.15, 30.0 * (c as f64 + 1.0) * p[0] - 20.0 * p[1] + 8.0 * c as f64 * p[2]));
        }
    }
    let ext = [0, 1, 2].map(|a| dims[a] as f64 * res);
    // smooth field + an interface step (≈ 300 Hz over 4 mm) + decay
    let hz: Vec<f64> = pos.iter().map(|p| 120.0 * (2.0 * PI * p[0] / ext[0]).sin() * (2.0 * PI * p[1] / ext[1]).cos() + 150.0 * ((p[0] - 0.006) / 0.004).tanh() + 40.0 * 2.0 * p[2] / ext[2]).collect();
    let r2: Vec<f64> = pos.iter().map(|p| 20.0 + 30.0 * (p[1] / ext[1] + 0.5)).collect();
    Geo { dims, nc, nv, smaps, x, idx, hz, r2 }
}

struct Traj {
    k: [Vec<f64>; 3],
    tau: Vec<f64>,
    tau_adc: Vec<f64>,
    na: usize,
}

fn traj() -> Traj {
    let (nro, na) = (12usize, 256usize);
    let mut r = Rng::new(77);
    let mut k: [Vec<f64>; 3] = Default::default();
    let mut tau = Vec::with_capacity(nro * na);
    let tau_adc: Vec<f64> = (0..na).map(|n| (TE + TRO * n as f64 / (na - 1) as f64) as f32 as f64).collect();
    for _ in 0..nro {
        let kz = 0.97 * PI * r.uniform();
        let th = PI * r.uniform();
        for n in 0..na {
            let s = n as f64 / (na - 1) as f64;
            let (rad, ang) = (0.97 * PI * s.sqrt(), th + 2.0 * PI * 7.0 * s.sqrt());
            k[0].push((rad * ang.cos()) as f32 as f64);
            k[1].push((rad * ang.sin()) as f32 as f64);
            k[2].push(kz as f32 as f64);
            tau.push(tau_adc[n]);
        }
    }
    Traj { k, tau, tau_adc, na }
}

fn c32(v: &[C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}
fn c64v(v: &[C32]) -> Vec<C64> {
    v.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect()
}

fn gfun(g: &Geo) -> impl Fn(usize, f64) -> Complex64 + Sync + '_ {
    move |q: usize, t: f64| Complex64::from_polar((-g.r2[q] * (t - TE)).exp(), -2.0 * PI * g.hz[q] * t)
}

#[test]
fn joint_static_vs_direct_sum() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let g = geo();
    let tr = traj();
    let m = tr.tau.len();
    let nv = g.nv;
    let gf = gfun(&g);
    let fit: Vec<usize> = (0..nv).collect();
    let w = vec![1.0f64; nv];
    let (ts, tw) = seg_time_grid(&tr.tau_adc, &vec![1.0; tr.na], 96);
    let smaps32 = c32(&g.smaps);
    let mut tsq: Option<Vec<f64>> = None;
    for tsqr in [false, true] {
        let opts = JointBuildOpts { verbose: true, fp32_tsqr: tsqr, ..Default::default() };
        let basis = build_joint_basis(&dev, &st, nv, g.nc, &smaps32, &fit, &w, &ts, &tw, &tr.tau_adc, &gf, &opts).unwrap();
        println!("{}: K = {}, C·K = {}, tail errors R = 20/40/60/80: {:.2e} {:.2e} {:.2e} {:.2e}", if tsqr { "fp32 TSQR" } else { "fp64 Gram" }, basis.k, basis.lam.len(), basis.tail_error(20), basis.tail_error(40), basis.tail_error(60), basis.tail_error(80));
        if tsqr {
            // the TSQR spectrum against the fp64 Gram: captured-energy loss of the same R
            let l64 = tsq.as_ref().unwrap();
            for r in [20usize, 40, 60] {
                let tot: f64 = l64.iter().sum();
                let t64 = (l64[r..].iter().sum::<f64>() / tot).sqrt();
                println!("  R = {r}: tail fp64 {t64:.3e}, fp32 TSQR {:.3e}", basis.tail_error(r));
                assert!((basis.tail_error(r) - t64).abs() <= 0.05 * t64 + 1e-6, "TSQR tail differs");
            }
            continue;
        }
        tsq = Some(basis.lam.clone());
        // R for a predicted error of 1e-6 (far below the NUFFT error)
        let r = basis.count_for(1e-6);
        let tidx: Vec<usize> = (0..tr.na).step_by(4).collect();
        let te = vec![1.0; tidx.len()];
        let eval: Vec<usize> = (0..nv).step_by(7).collect();
        let ew = vec![1.0; eval.len()];
        for rr in [r / 4, r / 2, r] {
            let e = basis.error(rr, &smaps32, &eval, &ew, &tidx, &te, &gf);
            println!("  R = {rr}: predicted {:.2e}, host-evaluated at the ADC times {:.2e}", basis.tail_error(rr), e.global);
        }
        let cbj = JointSegNufft::batch_for(r, 16);
        let so = SegNufftOpts { sigma: 1.25, width: 10, nseg: 1, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 };
        let inner = SegNufft::new(&dev, g.dims, [3e-3; 3], cbj, so).unwrap();
        let mut op = JointSegNufft::new(&dev, &st, inner, basis, &smaps32, r).unwrap();
        println!("joint operator: R = {r}, batches {} × {cbj}", op.nb);
        let co: [Vec<f32>; 3] = [0, 1, 2].map(|a| tr.k[a].iter().map(|&v| v as f32).collect());
        let tau32: Vec<f32> = tr.tau.iter().map(|&v| v as f32).collect();
        let pts = op.inner.points(&st, [&co[0], &co[1], &co[2]], &tau32, &vec![0f32; m], &vec![0f32; 6 * m], None).unwrap();
        let seg1 = op.segment1(&st, None).unwrap();
        let xd = st.clone_htod(&c32(&g.x)).unwrap();
        let mut yd = st.alloc_zeros::<C32>(g.nc * m).unwrap();
        op.forward(&st, &seg1, &pts, &xd, &mut yd).unwrap();
        let y = c64v(&st.clone_dtoh(&yd).unwrap());
        let mut rr = Rng::new(5);
        let yr = rr.complex_vec(g.nc * m);
        let yrd = st.clone_htod(&c32(&yr)).unwrap();
        let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
        op.adjoint(&st, &seg1, &pts, &yrd, &mut xa).unwrap();
        let xh = c64v(&st.clone_dtoh(&xa).unwrap());
        let vox: Vec<usize> = (0..nv).collect();
        let ds = DirectSum { nv, nc: g.nc, m, frames: 1, smaps: &g.smaps, voxels: &vox, bw: None };
        let fac = |j: usize, q: usize| -> C64 {
            let ph = tr.k[0][j] * g.idx[q][0] + tr.k[1][j] * g.idx[q][1] + tr.k[2][j] * g.idx[q][2];
            C64::from_polar(1.0, -ph) * gf(q, tr.tau[j])
        };
        let (yx, xx) = ds.apply(&fac, &g.x, &yr);
        let (ef, ea) = (rel_on(&y, &yx, None), rel_on(&xh, &xx, None));
        // dot test
        let lhs: C64 = y.iter().zip(&yr).map(|(a, b)| a.conj() * b).sum();
        let rhs: C64 = g.x.iter().zip(&xh).map(|(a, b)| a.conj() * b).sum();
        // normalised by ‖y‖‖Ax‖ (as seg_knot.rs: the dot of random vectors cancels ~√N-fold)
        let nrm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let dot = (lhs - rhs).norm() / (nrm(&y) * nrm(&yr));
        println!("joint vs direct sum: forward {ef:.2e}, adjoint {ea:.2e}; dot test {dot:.2e}");
        // control: today's operator (rank-24 SVD segments, the same engine settings) against the same direct sum
        {
            let (sp, at, t0, dt) = build_segments_svd_grid(nv, 24, tr.tau_adc[0], tr.tau_adc[tr.na - 1], tr.na, &fit, None, &ts, &tw, &gf);
            let mut e = SegNufft::new(&dev, g.dims, [3e-3; 3], g.nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: 24, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
            e.set_maps(&st, &smaps32).unwrap();
            let sg = e.segments(&st, &sp, &at, t0, dt).unwrap();
            let p2 = e.points(&st, [&co[0], &co[1], &co[2]], &tau32, &vec![0f32; m], &vec![0f32; 6 * m], None).unwrap();
            let mut y2 = st.alloc_zeros::<C32>(g.nc * m).unwrap();
            let mut x2 = st.alloc_zeros::<C32>(nv).unwrap();
            e.forward(&st, &sg, &p2, &xd, &mut y2).unwrap();
            e.adjoint(&st, &sg, &p2, &yrd, &mut x2).unwrap();
            let (y2, x2) = (c64v(&st.clone_dtoh(&y2).unwrap()), c64v(&st.clone_dtoh(&x2).unwrap()));
            let (cf, ca) = (rel_on(&y2, &yx, None), rel_on(&x2, &xx, None));
            println!("control, rank-24 SVD segments vs direct sum: forward {cf:.2e}, adjoint {ca:.2e}; joint vs control: forward {:.2e}, adjoint {:.2e}", rel_on(&y, &y2, None), rel_on(&xh, &x2, None));
            // the joint operator is as close to the exact physics as today's operator (both at the NUFFT floor of
            // this small grid: W 10, σ 1.25)
            assert!(ef < 1.5 * cf + 1e-5 && ea < 1.5 * ca + 1e-5, "joint operator vs direct sum: {ef:.2e} / {ea:.2e} (control {cf:.2e} / {ca:.2e})");
        }
        assert!(dot < 5e-6);
        // operator error vs R (direct sum = exact): follows the predicted tail above the NUFFT floor
        for rr in [r / 4, r / 2] {
            op.set_rank(&st, rr).unwrap();
            op.forward(&st, &seg1, &pts, &xd, &mut yd).unwrap();
            let y = c64v(&st.clone_dtoh(&yd).unwrap());
            println!("  R = {rr}: operator forward error {:.2e} (predicted model error {:.2e}, white image)", rel_on(&y, &yx, None), op.basis().tail_error(rr));
        }
    }
}

/// joint vs today's operator (rank-L SVD segments, L high) with the same order-3 sample tables; static and 2-D knots
#[test]
fn joint_order3_and_knots_vs_segments() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let g = geo();
    let tr = traj();
    let m = tr.tau.len();
    let nv = g.nv;
    let gf = gfun(&g);
    let fit: Vec<usize> = (0..nv).collect();
    let w = vec![1.0f64; nv];
    let (ts, tw) = seg_time_grid(&tr.tau_adc, &vec![1.0; tr.na], 96);
    let smaps32 = c32(&g.smaps);
    let mut rng = Rng::new(9);
    let co: [Vec<f32>; 3] = [0, 1, 2].map(|a| tr.k[a].iter().map(|&v| v as f32).collect());
    let tau32: Vec<f32> = tr.tau.iter().map(|&v| v as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (0.3 * rng.uniform()) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (rng.uniform() * 5.0) as f32).collect();
    let uh: Vec<f32> = (0..4 * m).map(|_| (rng.uniform() * 1.0) as f32).collect();
    let hh: Vec<f64> = (0..4 * NHI).map(|_| rng.uniform() * 20.0).collect();
    // reference: rank-24 SVD segments (uniform weights) on today's engine
    let lref = 24;
    let (sp, at, t0, dt) = build_segments_svd_grid(nv, lref, tr.tau_adc[0], tr.tau_adc[tr.na - 1], tr.na, &fit, None, &ts, &tw, &gf);
    for (label, kn) in [("static", None), ("2-D knots 2 lin × 4 cubic", Some((2usize, 1u32, 4usize, 3u32)))] {
        let tcard: Vec<f32> = (0..m).map(|j| ((j as f64 * 0.37) % 4.0) as f32).collect();
        let (ri, rv): (Vec<u32>, Vec<f32>) = (0..m).map(|j| (0u32, ((j as f64 * 0.13) % 1.0) as f32)).unzip();
        let nimg = kn.map_or(1, |(kr, _, mc, _)| kr * mc);
        let mk = |nc: usize, nseg: usize| -> SegNufft {
            match kn {
                None => SegNufft::new(&dev, g.dims, [3e-3; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order: 3, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap(),
                Some((kr, rdeg, mc, cdeg)) => SegNufft::new_temporal2d(&dev, g.dims, [3e-3; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order: 3, frames: mc, bspline: cdeg, max_grid_bytes: 1 << 30 }, kr, rdeg).unwrap(),
            }
        };
        let mkpts = |e: &mut SegNufft| match kn {
            None => e.points_hi(&st, [&co[0], &co[1], &co[2]], &tau32, &phi0, &dcv, &uh, &hh, None).unwrap(),
            Some(_) => e.points2d_hi(&st, [&co[0], &co[1], &co[2]], &tau32, &phi0, &dcv, &tcard, &ri, &rv, &uh, &hh).unwrap(),
        };
        let mut eref = mk(g.nc, lref);
        eref.set_maps(&st, &smaps32).unwrap();
        let sref = eref.segments(&st, &sp, &at, t0, dt).unwrap();
        let pref = mkpts(&mut eref);
        let mut rr = Rng::new(3);
        let xs = rr.complex_vec(nv * nimg);
        let xd = st.clone_htod(&c32(&xs)).unwrap();
        let yr = rr.complex_vec(g.nc * m);
        let yrd = st.clone_htod(&c32(&yr)).unwrap();
        let mut y0 = st.alloc_zeros::<C32>(g.nc * m).unwrap();
        let mut x0 = st.alloc_zeros::<C32>(nv * nimg).unwrap();
        eref.forward(&st, &sref, &pref, &xd, &mut y0).unwrap();
        eref.adjoint(&st, &sref, &pref, &yrd, &mut x0).unwrap();
        let (y0, x0) = (c64v(&st.clone_dtoh(&y0).unwrap()), c64v(&st.clone_dtoh(&x0).unwrap()));
        drop(eref);
        let basis = build_joint_basis(&dev, &st, nv, g.nc, &smaps32, &fit, &w, &ts, &tw, &tr.tau_adc, &gf, &JointBuildOpts::default()).unwrap();
        let r = basis.count_for(1e-6);
        let cbj = JointSegNufft::batch_for(r, 16);
        let mut op = JointSegNufft::new(&dev, &st, mk(cbj, 1), basis, &smaps32, r).unwrap();
        let pj = mkpts(&mut op.inner);
        let s1 = op.segment1(&st, None).unwrap();
        let mut y1 = st.alloc_zeros::<C32>(g.nc * m).unwrap();
        let mut x1 = st.alloc_zeros::<C32>(nv * nimg).unwrap();
        op.forward(&st, &s1, &pj, &xd, &mut y1).unwrap();
        op.adjoint(&st, &s1, &pj, &yrd, &mut x1).unwrap();
        let (y1, x1) = (c64v(&st.clone_dtoh(&y1).unwrap()), c64v(&st.clone_dtoh(&x1).unwrap()));
        let lhs: C64 = y1.iter().zip(&yr).map(|(a, b)| a.conj() * b).sum();
        let rhs: C64 = xs.iter().zip(&x1).map(|(a, b)| a.conj() * b).sum();
        let nrm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let (ef, ea, dot) = (rel_on(&y1, &y0, None), rel_on(&x1, &x0, None), (lhs - rhs).norm() / (nrm(&y1) * nrm(&yr)));
        println!("{label}, order 3: joint (R = {r}, {} × {cbj}) vs rank-{lref} SVD segments: forward {ef:.2e}, adjoint {ea:.2e}; dot test {dot:.2e}", op.nb);
        assert!(ef < 2e-4 && ea < 4e-4 && dot < 5e-6);
    }
}
