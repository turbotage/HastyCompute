//! research/sample_kernels.md §20: the integration defaults against the paths they replace.
//! * `static_o3_one_bin_default`: a static order-3 engine is a one-bin knot engine by default (`static_knot_default`),
//!   `StaticKnot::Never` gives the old static engine; forward, adjoint and the fused residual agree, before and after
//!   the autotune (vt / ra / kl), and order ≤ 2 stays static.
//! * `one_bin_fallback_calls`: the calls the knot kernels do not cover run on the one-bin engine through the
//!   multi-image kernels (`as_static`): multi-image forward / adjoint (K = 1 with weights, K = 2), the grouped forward /
//!   residual / adjoint, the water + fat pair (`forward_fat` / `adjoint_fat`, grouped too); at order 3 (default) and
//!   order 2 (`StaticKnot::Always`), against the old static engine on the same samples; the K = 1 shapes tuned at the
//!   first fallback call after `autotune`.
//! * `fft_store_default_and_rebudget`: the default y ↔ z storage is f32 (research/bfp_storage.md §10); bfp24 on
//!   request is in place where a plane's lines are whole 16-line tiles; `set_fft_store` re-budgets the coil batch (dense fallback that does not fit the
//!   grid budget: fewer coils per batch, back on f32) and the operator stays within the BFP error.
//! Bounds as tests/seg_split.rs: max(3e-5, 6 × the one-ulp floor) for values, BFP rows max(2e-6, 6 × floor).
//!   cargo test --release -p hasty-nufft --test seg_defaults -- --nocapture --test-threads=1
use cudarc::driver::CudaStream;
use hasty_cuda::{Device, C32};
use hasty_fft::Store;
use hasty_nufft::{build_segments, hi_exponents, set_thread_cb_cap, set_thread_fft_store, set_thread_static_knot, SegNufft, SegNufftOpts, SegPoints, SegSegments, StaticKnot, NHI};
use std::sync::Arc;

struct R(u64);
impl R {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2)).sum();
    let d: f64 = b.iter().map(|y| (y.re as f64).powi(2) + (y.im as f64).powi(2)).sum();
    (n / d.max(1e-30)).sqrt()
}

const DIMS: [usize; 3] = [24, 22, 24];
const RES: f64 = 0.004;

struct Prob {
    m: usize,
    nc: usize,
    nseg: usize,
    order: u32,
    coords: [Vec<f32>; 3],
    tau: Vec<f32>,
    phi0: Vec<f32>,
    dc: Vec<f32>,
    u: Vec<f32>,
    hraw: Vec<f64>,
    smaps: Vec<C32>,
    sp: Vec<C32>,
    at: Vec<C32>,
    t0: f64,
    dt: f64,
}

fn prob(nc: usize, nseg: usize, order: u32, m: usize, seed: u64) -> Prob {
    let nv: usize = DIMS.iter().product();
    let mut r = R(7717 + seed * 13 + nseg as u64 * 31 + order as u64 * 7 + nc as u64 * 3 + m as u64);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dc: Vec<f32> = (0..6 * m).map(|_| (r.u() * 1.2e4) as f32).collect();
    let u: Vec<f32> = (0..4 * m).map(|_| r.u() as f32).collect();
    let hraw: Vec<f64> = (0..4 * NHI).map(|_| 0.2 * r.u()).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    Prob { m, nc, nseg, order, coords, tau, phi0, dc, u, hraw, smaps, sp, at, t0, dt }
}

fn vecs(n: usize, seed: u64) -> Vec<C32> {
    let mut r = R(seed);
    (0..n).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect()
}

/// a static engine (frames 1) under the given static-knot rule and store, coil batch cap `cb`
fn engine(dev: &Device, p: &Prob, sk: StaticKnot, store: Option<Store>, cb: usize, budget: usize) -> SegNufft {
    let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg: p.nseg, order: p.order, frames: 1, bspline: 1, max_grid_bytes: budget };
    set_thread_cb_cap(Some(cb));
    set_thread_static_knot(Some(sk));
    set_thread_fft_store(store);
    let mut e = SegNufft::new(dev, DIMS, [RES; 3], p.nc, opts).unwrap();
    set_thread_cb_cap(None);
    set_thread_static_knot(None);
    set_thread_fft_store(None);
    e.set_maps(&dev.default_stream(), &p.smaps).unwrap();
    e
}

fn points(st: &Arc<CudaStream>, e: &mut SegNufft, p: &Prob) -> SegPoints {
    let c = [&p.coords[0][..], &p.coords[1][..], &p.coords[2][..]];
    if p.order >= 3 {
        let g: [f64; 3] = [0, 1, 2].map(|a| RES * e.plan.nf[a] as f64 / (std::f64::consts::PI * 8.0));
        let ex = hi_exponents();
        let h: Vec<f64> = (0..4 * NHI).map(|i| p.hraw[i] / (0..3).map(|a| g[a].powi(ex[i % NHI][a] as i32)).product::<f64>()).collect();
        e.points_hi(st, c, &p.tau, &p.phi0, &p.dc, &p.u, &h, None).unwrap()
    } else {
        e.points(st, c, &p.tau, &p.phi0, &p.dc, None).unwrap()
    }
}

/// a second sample set (the group calls): the same coordinates shifted and other readout times
fn points2(st: &Arc<CudaStream>, e: &mut SegNufft, p: &Prob) -> SegPoints {
    let c: [Vec<f32>; 3] = [0, 1, 2].map(|a| p.coords[a].iter().map(|&v| (v * 0.9 + 0.11 * (a as f32 + 1.0)).clamp(-3.1, 3.1)).collect());
    let tau: Vec<f32> = p.tau.iter().map(|&t| 1.6e-2 - t).collect();
    let cr = [&c[0][..], &c[1][..], &c[2][..]];
    if p.order >= 3 {
        let g: [f64; 3] = [0, 1, 2].map(|a| RES * e.plan.nf[a] as f64 / (std::f64::consts::PI * 8.0));
        let ex = hi_exponents();
        let h: Vec<f64> = (0..4 * NHI).map(|i| p.hraw[i] / (0..3).map(|a| g[a].powi(ex[i % NHI][a] as i32)).product::<f64>()).collect();
        e.points_hi(st, cr, &tau, &p.phi0, &p.dc, &p.u, &h, None).unwrap()
    } else {
        e.points(st, cr, &tau, &p.phi0, &p.dc, None).unwrap()
    }
}

/// forward, adjoint, residual of one engine; plus the one-ulp floor of the forward and adjoint
fn fwd_adj(st: &Arc<CudaStream>, e: &mut SegNufft, seg: &SegSegments, pts: &SegPoints, xh: &[C32], yh: &[C32]) -> [Vec<C32>; 3] {
    let (x, y) = (st.clone_htod(xh).unwrap(), st.clone_htod(yh).unwrap());
    let mut f = st.alloc_zeros::<C32>(yh.len()).unwrap();
    e.forward(st, seg, pts, &x, &mut f).unwrap();
    let mut a = st.alloc_zeros::<C32>(xh.len()).unwrap();
    e.adjoint(st, seg, pts, &y, &mut a).unwrap();
    let mut r = st.alloc_zeros::<C32>(yh.len()).unwrap();
    e.forward_residual(st, seg, pts, &x, &y, &mut r).unwrap();
    [st.clone_dtoh(&f).unwrap(), st.clone_dtoh(&a).unwrap(), st.clone_dtoh(&r).unwrap()]
}

fn ulp(v: &[C32], seed: u64) -> Vec<C32> {
    let mut r = R(seed);
    v.iter().map(|z| { let f = 1.0 + if r.u() < 0.0 { -1.0 } else { 1.0 } * f32::EPSILON * 0.5; C32::new(z.re * f, z.im * f) }).collect()
}

/// the one-ulp conditioning floor of the static engine's forward / adjoint
fn floors(st: &Arc<CudaStream>, e: &mut SegNufft, seg: &SegSegments, pts: &SegPoints, xh: &[C32], yh: &[C32]) -> (f64, f64) {
    let [f0, a0, _] = fwd_adj(st, e, seg, pts, xh, yh);
    let [f1, _, _] = fwd_adj(st, e, seg, pts, &ulp(xh, 3), yh);
    let [_, a1, _] = fwd_adj(st, e, seg, pts, xh, &ulp(yh, 5));
    (rel(&f1, &f0), rel(&a1, &a0))
}

#[test]
fn static_o3_one_bin_default() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    for order in [0u32, 2, 3] {
        let p = prob(4, 3, order, 6000, 1);
        let (xh, yh) = (vecs(nv, 11), vecs(p.nc * p.m, 12));
        // the default rule (order 3: one bin) and the old static engine
        set_thread_static_knot(None);
        let mut d = engine(&dev, &p, hasty_nufft::static_knot_default(), Some(Store::F32), 2, 1 << 30);
        let mut s = engine(&dev, &p, StaticKnot::Never, Some(Store::F32), 2, 1 << 30);
        assert_eq!(d.is_one_bin(), order >= 3, "order {order}: one-bin knot engine by default iff order 3");
        assert!(!s.is_one_bin(), "StaticKnot::Never: the static engine");
        let (sd, ss) = (d.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap(), s.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap());
        let (pd, ps) = (points(&st, &mut d, &p), points(&st, &mut s, &p));
        let (ff, fa) = floors(&st, &mut s, &ss, &ps, &xh, &yh);
        let bf = 3e-5f64.max(6.0 * ff);
        let ba = 3e-5f64.max(6.0 * fa);
        let r0 = fwd_adj(&st, &mut s, &ss, &ps, &xh, &yh);
        for tuned in [false, true] {
            if tuned {
                d.autotune(&st, &sd, &pd).unwrap();
            }
            let r = fwd_adj(&st, &mut d, &sd, &pd, &xh, &yh);
            let (ef, ea, er) = (rel(&r[0], &r0[0]), rel(&r[1], &r0[1]), rel(&r[2], &r0[2]));
            eprintln!("order {order} default ({}) vs static{}: forward {ef:.1e} (bound {bf:.1e}), adjoint {ea:.1e} (bound {ba:.1e}), residual {er:.1e}", if d.is_one_bin() { "one bin" } else { "static" }, if tuned { ", autotuned" } else { "" });
            assert!(ef < bf && ea < ba && er < bf, "order {order} tuned {tuned}");
        }
    }
}

#[test]
fn one_bin_fallback_calls() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    for (order, sk) in [(3u32, StaticKnot::Order3), (2, StaticKnot::Always)] {
        let p = prob(4, 3, order, 5000, 2);
        let m = p.m;
        let (x1, x2, yh, y2h) = (vecs(nv, 21), vecs(nv, 22), vecs(p.nc * m, 23), vecs(p.nc * m, 24));
        let wh: Vec<C32> = vecs(m, 25);
        let ff: Vec<f32> = (0..nv).map(|v| ((v % 9) as f32 / 9.0) * 0.6).collect();
        let cn: Vec<C32> = (0..p.nseg).map(|l| C32::new(0.8 * (0.7 * l as f32).cos(), 0.8 * (0.7 * l as f32).sin())).collect();
        let mut res: Vec<Vec<Vec<C32>>> = vec![];
        let mut floor = 0.0f64;
        for (i, rule) in [StaticKnot::Never, sk].into_iter().enumerate() {
            let mut e = engine(&dev, &p, rule, Some(Store::F32), 2, 1 << 30);
            assert_eq!(e.is_one_bin(), i == 1);
            let seg = e.segments_fat(&st, &p.sp, &p.at, &cn, p.t0, p.dt).unwrap();
            let (pa, pb) = (points(&st, &mut e, &p), points2(&st, &mut e, &p));
            if i == 1 {
                // the K = 1 shapes are tuned at the first fallback call after the autotune
                e.autotune(&st, &seg, &pa).unwrap();
            }
            let maps = e.fat_maps(&st, &ff).unwrap();
            let (dx1, dx2, dy, dy2, dw) = (st.clone_htod(&x1).unwrap(), st.clone_htod(&x2).unwrap(), st.clone_htod(&yh).unwrap(), st.clone_htod(&y2h).unwrap(), st.clone_htod(&wh).unwrap());
            let mut out: Vec<Vec<C32>> = vec![];
            let yb = || st.alloc_zeros::<C32>(p.nc * m).unwrap();
            let xb = || st.alloc_zeros::<C32>(nv).unwrap();
            // multi-image forward: K = 1 with weights, K = 2
            let mut y = yb();
            e.forward_multi(&st, &seg, &pa, &[&dx1], &[Some(&dw)], &mut y).unwrap();
            out.push(st.clone_dtoh(&y).unwrap());
            let mut y = yb();
            e.forward_multi(&st, &seg, &pa, &[&dx1, &dx2], &[None, Some(&dw)], &mut y).unwrap();
            out.push(st.clone_dtoh(&y).unwrap());
            // multi-image adjoint, K = 2
            let (mut a1, mut a2) = (xb(), xb());
            e.adjoint_multi(&st, &seg, &pa, &dy, &[None, Some(&dw)], &mut [&mut a1, &mut a2]).unwrap();
            out.push(st.clone_dtoh(&a1).unwrap());
            out.push(st.clone_dtoh(&a2).unwrap());
            // group forward / residual / adjoint of two sets
            let (mut g1, mut g2) = (yb(), yb());
            e.forward_group(&st, &seg, &[&pa, &pb], &dx1, &mut [&mut g1, &mut g2]).unwrap();
            out.push(st.clone_dtoh(&g1).unwrap());
            out.push(st.clone_dtoh(&g2).unwrap());
            let (mut r1, mut r2) = (yb(), yb());
            e.forward_residual_group(&st, &seg, &[&pa, &pb], &dx1, &[&dy, &dy2], &mut [&mut r1, &mut r2]).unwrap();
            out.push(st.clone_dtoh(&r2).unwrap());
            let mut ag = xb();
            e.adjoint_group(&st, &seg, &[&pa, &pb], &[&dy, &dy2], &mut ag).unwrap();
            out.push(st.clone_dtoh(&ag).unwrap());
            // water + fat pair, single and grouped
            let mut yf = yb();
            e.forward_fat(&st, &seg, &pa, &maps, Some(0), &dx1, &mut yf).unwrap();
            out.push(st.clone_dtoh(&yf).unwrap());
            let mut af = xb();
            e.adjoint_fat(&st, &seg, &pa, &maps, Some(0), &dy, &mut af).unwrap();
            out.push(st.clone_dtoh(&af).unwrap());
            let (mut h1, mut h2) = (yb(), yb());
            e.forward_fat_group(&st, &seg, &[&pa, &pb], &maps, Some(0), &dx1, None, &mut [&mut h1, &mut h2]).unwrap();
            out.push(st.clone_dtoh(&h2).unwrap());
            let mut afg = xb();
            e.adjoint_fat_group(&st, &seg, &[&pa, &pb], &maps, Some(0), &[&dy, &dy2], &mut afg).unwrap();
            out.push(st.clone_dtoh(&afg).unwrap());
            // the plain calls still run the knot kernels on the one-bin engine
            let mut yp = yb();
            e.forward(&st, &seg, &pa, &dx1, &mut yp).unwrap();
            out.push(st.clone_dtoh(&yp).unwrap());
            if i == 0 {
                let (f, a) = floors(&st, &mut e, &seg, &pa, &x1, &yh);
                floor = f.max(a);
            }
            res.push(out);
        }
        let names = ["multi fwd K1 w", "multi fwd K2", "multi adj K2 img0", "multi adj K2 img1", "group fwd 0", "group fwd 1", "group residual 1", "group adj", "fat fwd", "fat adj", "fat group fwd 1", "fat group adj", "plain fwd"];
        let b = 3e-5f64.max(6.0 * floor);
        for (k, n) in names.iter().enumerate() {
            let e = rel(&res[1][k], &res[0][k]);
            eprintln!("order {order} {sk:?}: {n}: one bin vs static {e:.1e} (bound {b:.1e})");
            assert!(e < b, "order {order}: {n} {e:e}");
        }
    }
}

#[test]
fn fft_store_default_and_rebudget() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    // default store (no thread override): f32 since research/bfp_storage.md §10 (bfp24 / auto on request)
    assert!(std::env::var("HASTY_SEG_FFT_STORE").is_err(), "run without HASTY_SEG_FFT_STORE");
    assert_eq!(hasty_nufft::fft_store_default(), Store::F32);
    assert!(!hasty_nufft::fft_store_auto());
    let p = prob(6, 3, 2, 6000, 3);
    let (xh, yh) = (vecs(nv, 31), vecs(p.nc * p.m, 32));
    let mut f = engine(&dev, &p, StaticKnot::Never, Some(Store::F32), 3, 1 << 30);
    let n0 = f.plan.nf[0];
    let ge: usize = f.plan.nf.iter().product();
    let per_coil = p.nseg * ge * 8;
    assert_eq!(f.fft_store(), Store::F32);
    let sf = f.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
    let pf = points(&st, &mut f, &p);
    let r0 = fwd_adj(&st, &mut f, &sf, &pf, &xh, &yh);
    let (ff, fa) = floors(&st, &mut f, &sf, &pf, &xh, &yh);
    // the default engine: f32 intermediates
    let d0 = engine(&dev, &p, StaticKnot::Never, None, 3, 1 << 30);
    assert_eq!((d0.fft_store(), d0.fft_store_bytes()), (Store::F32, 0), "default store");
    drop(d0);
    // bfp24 on request: in place when n0 · nch is a multiple of 16, under slabs too (an explicit width)
    let mut d = engine(&dev, &p, StaticKnot::Never, Some(Store::Bfp(24)), 3, 1 << 30);
    // in place when n0 · nch is a multiple of 16; otherwise the default stays f32 (no dense fallback: its buffer would
    // shrink the coil batch)
    let tiles = (n0 * d.cb * p.nseg) % 16 == 0;
    eprintln!("grid {:?}, coil batch {}, requested bfp24 store {} ({}, {} B extra)", d.plan.nf, d.cb, d.fft_store(), if tiles { "in place" } else { "f32: lines not whole 16-line tiles" }, d.fft_store_bytes());
    assert_eq!(d.fft_store(), if tiles { Store::Bfp(24) } else { Store::F32 });
    let sd = d.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
    let pd = points(&st, &mut d, &p);
    let r = fwd_adj(&st, &mut d, &sd, &pd, &xh, &yh);
    let (ef, ea) = (rel(&r[0], &r0[0]), rel(&r[1], &r0[1]));
    let (bf, ba) = (2e-6f64.max(6.0 * ff), 2e-6f64.max(6.0 * fa));
    eprintln!("bfp24 vs f32: forward {ef:.1e} (bound {bf:.1e}), adjoint {ea:.1e} (bound {ba:.1e})");
    assert!(ef < bf && ea < ba, "bfp24 store");
    // re-budget: a budget of exactly 3 coils at f32; with n0 · 3 · nseg not a multiple of 16 the BFP layout is dense
    // (a separate buffer), which no longer fits: set_fft_store must lower the batch, and f32 must restore it
    if (n0 * p.nseg) % 16 == 0 || (n0 * 2 * p.nseg) % 16 == 0 || (n0 * 3 * p.nseg) % 16 == 0 {
        eprintln!("grid x length {n0}: an in-place batch exists, re-budget case not applicable");
        return;
    }
    // the dense layout explicitly (the default would keep f32 on this grid)
    unsafe { std::env::set_var("HASTY_SEG_FFT_BFP_LAYOUT", "dense") };
    let mut t = engine(&dev, &p, StaticKnot::Never, Some(Store::F32), 3, 3 * per_coil + 4096);
    assert_eq!(t.cb, 3, "f32: 3 coils fit");
    let segt = t.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
    let ptt = points(&st, &mut t, &p);
    t.set_fft_store(Store::Bfp(22)).unwrap();
    eprintln!("set_fft_store(bfp22) on a 3-coil budget: coil batch {} ({} B of BFP buffer)", t.cb, t.fft_store_bytes());
    assert!(t.cb < 3 && t.fft_store() == Store::Bfp(22), "the dense bfp22 buffer must lower the batch");
    let r = fwd_adj(&st, &mut t, &segt, &ptt, &xh, &yh);
    let (ef, ea) = (rel(&r[0], &r0[0]), rel(&r[1], &r0[1]));
    let (bf, ba) = (2e-5f64.max(6.0 * ff), 2e-5f64.max(6.0 * fa));
    eprintln!("  bfp22 after re-budget vs f32: forward {ef:.1e} (bound {bf:.1e}), adjoint {ea:.1e} (bound {ba:.1e})");
    assert!(ef < bf && ea < ba, "bfp22 after re-budget");
    t.set_fft_store(Store::F32).unwrap();
    unsafe { std::env::remove_var("HASTY_SEG_FFT_BFP_LAYOUT") };
    assert_eq!(t.cb, 3, "f32 again: back to 3 coils");
    let r = fwd_adj(&st, &mut t, &segt, &ptt, &xh, &yh);
    let (ef, ea) = (rel(&r[0], &r0[0]), rel(&r[1], &r0[1]));
    eprintln!("  f32 after re-budget vs f32: forward {ef:.1e}, adjoint {ea:.1e}");
    assert!(ef < 3e-5f64.max(6.0 * ff) && ea < 3e-5f64.max(6.0 * fa), "f32 after re-budget");
}
