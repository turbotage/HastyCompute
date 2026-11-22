//! research/sample_kernels.md §18: the segment-batched operator (`SegSplit`, segment groups inside a coil) and the
//! one-bin knot-major engine (frames 1, degree 0: the static operator through the knot kernels) against the existing
//! paths.
//! * `split_matches_plain`: SegSplit with groups of 1, 2 and 3 of L = 5 segments (two of them zero-padded) against the
//!   plain engine: forward, adjoint, the fused residual and adjoint accumulation; static orders 0 / 2 / 3 (order 3: the
//!   multi-image path) and 1-D cubic knot-major orders 0 / 3; several coil batches.
//! * `one_bin_knot_matches_static`: the one-bin knot engine against the static engine, orders 0–3, with the tile
//!   kernels and with the autotune's vt / ra / kl variants forced on.
//! Bounds: max(3e-5, 6 × the one-ulp conditioning floor of the path) for values (as tests/seg_ra_vt.rs), adjointness
//! ≤ max(2e-6, 2 × the reference's).
//!   cargo test --release -p hasty-nufft --test seg_split -- --nocapture --test-threads=1
use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, hi_exponents, set_thread_cb_cap, set_thread_static_knot, StaticKnot, SegNufft, SegNufftOpts, SegPoints, SegSplit, SegSplitSegments, NHI};
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

fn dot(a: &[C32], b: &[C32]) -> num_complex::Complex64 {
    a.iter().zip(b).map(|(x, y)| num_complex::Complex64::new(x.re as f64, x.im as f64).conj() * num_complex::Complex64::new(y.re as f64, y.im as f64)).sum()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Static,
    Knot1d,
    /// frames 1, degree 0 on a knot-major engine
    OneBin,
}

const DIMS: [usize; 3] = [24, 22, 24];
const RES: f64 = 0.004;
const MC: usize = 6;

/// the problem: samples, tables, segments, image, data (deterministic per seed)
struct Prob {
    m: usize,
    nc: usize,
    nseg: usize,
    order: u32,
    coords: [Vec<f32>; 3],
    tau: Vec<f32>,
    phi0: Vec<f32>,
    dc: Vec<f32>,
    t: Vec<f32>,
    u: Vec<f32>,
    hraw: Vec<f64>,
    smaps: Vec<C32>,
    sp: Vec<C32>,
    at: Vec<C32>,
    t0: f64,
    dt: f64,
    xh: Vec<C32>,
    yh: Vec<C32>,
}

fn prob(nc: usize, nseg: usize, order: u32, m: usize, nimg: usize) -> Prob {
    let nv: usize = DIMS.iter().product();
    let mut r = R(9191 + nseg as u64 * 31 + order as u64 * 7 + nc as u64 * 3 + m as u64 + nimg as u64 * 101);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    // strong ΔC (|d| ≲ 0.1 in kernel units) so that the e terms matter
    let dc: Vec<f32> = (0..6 * m).map(|_| (r.u() * 1.2e4) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * MC as f64) as f32).collect();
    let u: Vec<f32> = (0..4 * m).map(|_| r.u() as f32).collect();
    let hraw: Vec<f64> = (0..4 * NHI).map(|_| 0.2 * r.u()).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    let xh: Vec<C32> = (0..nimg * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yh: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    Prob { m, nc, nseg, order, coords, tau, phi0, dc, t, u, hraw, smaps, sp, at, t0, dt, xh, yh }
}

fn engine(dev: &Device, p: &Prob, kind: Kind, nseg: usize, cb_cap: usize) -> SegNufft {
    let frames = if kind == Kind::Knot1d { MC } else { 1 };
    let bspline = match kind {
        Kind::Static => 1,
        Kind::Knot1d => 3,
        Kind::OneBin => 0,
    };
    let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg, order: p.order, frames, bspline, max_grid_bytes: 1 << 30 };
    set_thread_cb_cap(Some(cb_cap));
    // the static reference is the old static engine (order 3: the multi-image kernels), not the one-bin default
    set_thread_static_knot(Some(StaticKnot::Never));
    let e = SegNufft::new_temporal(dev, DIMS, [RES; 3], p.nc, opts, kind != Kind::Static).unwrap();
    set_thread_static_knot(None);
    set_thread_cb_cap(None);
    e
}

/// as `engine`, the static-knot rule from the env (HASTY_SEG_STATIC_KNOT)
fn engine_env(dev: &Device, p: &Prob, kind: Kind, nseg: usize, cb_cap: usize) -> SegNufft {
    let frames = if kind == Kind::Knot1d { MC } else { 1 };
    let bspline = match kind {
        Kind::Static => 1,
        Kind::Knot1d => 3,
        Kind::OneBin => 0,
    };
    let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg, order: p.order, frames, bspline, max_grid_bytes: 1 << 30 };
    set_thread_cb_cap(Some(cb_cap));
    let e = SegNufft::new_temporal(dev, DIMS, [RES; 3], p.nc, opts, kind != Kind::Static).unwrap();
    set_thread_cb_cap(None);
    e
}

fn points(st: &Arc<CudaStream>, e: &mut SegNufft, p: &Prob, kind: Kind) -> SegPoints {
    let c = [&p.coords[0][..], &p.coords[1][..], &p.coords[2][..]];
    let t = (kind == Kind::Knot1d).then_some(&p.t[..]);
    if p.order >= 3 {
        // degree-3/4 coefficients ≈ 0.05 in kernel units
        let g: [f64; 3] = [0, 1, 2].map(|a| RES * e.plan.nf[a] as f64 / (std::f64::consts::PI * 8.0));
        let ex = hi_exponents();
        let h: Vec<f64> = (0..4 * NHI).map(|i| p.hraw[i] / (0..3).map(|a| g[a].powi(ex[i % NHI][a] as i32)).product::<f64>()).collect();
        e.points_hi(st, c, &p.tau, &p.phi0, &p.dc, &p.u, &h, t).unwrap()
    } else {
        e.points(st, c, &p.tau, &p.phi0, &p.dc, t).unwrap()
    }
}

fn fwd_plain(st: &Arc<CudaStream>, e: &mut SegNufft, seg: &hasty_nufft::SegSegments, pts: &SegPoints, x: &CudaSlice<C32>, n: usize) -> Vec<C32> {
    let mut y = st.alloc_zeros::<C32>(n).unwrap();
    e.forward(st, seg, pts, x, &mut y).unwrap();
    st.clone_dtoh(&y).unwrap()
}

fn adj_plain(st: &Arc<CudaStream>, e: &mut SegNufft, seg: &hasty_nufft::SegSegments, pts: &SegPoints, y: &CudaSlice<C32>, n: usize) -> Vec<C32> {
    let mut x = st.alloc_zeros::<C32>(n).unwrap();
    e.adjoint(st, seg, pts, y, &mut x).unwrap();
    st.clone_dtoh(&x).unwrap()
}

/// one-ulp perturbation of a vector (the conditioning floor of a path)
fn ulp(v: &[C32], seed: u64) -> Vec<C32> {
    let mut r = R(seed);
    v.iter().map(|z| { let f = 1.0 + if r.u() < 0.0 { -1.0 } else { 1.0 } * f32::EPSILON * 0.5; C32::new(z.re * f, z.im * f) }).collect()
}

#[test]
fn split_matches_plain() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    for (kind, order) in [(Kind::Static, 0u32), (Kind::Static, 2), (Kind::Static, 3), (Kind::Knot1d, 0), (Kind::Knot1d, 3), (Kind::OneBin, 3)] {
        let nimg = if kind == Kind::Knot1d { MC } else { 1 };
        let p = prob(3, 5, order, 6000, nimg);
        let (nx, ny) = (nimg * nv, p.nc * p.m);
        let (x, yd) = (st.clone_htod(&p.xh).unwrap(), st.clone_htod(&p.yh).unwrap());
        for cb_cap in [1usize, 3] {
            // reference: the plain engine with all L segments
            let mut e0 = engine(&dev, &p, kind, p.nseg, cb_cap);
            e0.set_maps(&st, &p.smaps).unwrap();
            let seg0 = e0.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
            let pts0 = points(&st, &mut e0, &p, kind);
            let y0 = fwd_plain(&st, &mut e0, &seg0, &pts0, &x, ny);
            let xa0 = adj_plain(&st, &mut e0, &seg0, &pts0, &yd, nx);
            // conditioning floors: the reference with one-ulp moved inputs
            let xp = st.clone_htod(&ulp(&p.xh, 7)).unwrap();
            let ydp = st.clone_htod(&ulp(&p.yh, 8)).unwrap();
            let (ff, fa) = (rel(&fwd_plain(&st, &mut e0, &seg0, &pts0, &xp, ny), &y0), rel(&adj_plain(&st, &mut e0, &seg0, &pts0, &ydp, nx), &xa0));
            let j0 = (dot(&p.yh, &y0) - dot(&xa0, &p.xh)).norm() / (dot(&p.yh, &p.yh).re * dot(&y0, &y0).re).sqrt();
            let (bf, ba, bj) = (3e-5f64.max(6.0 * ff), 3e-5f64.max(6.0 * fa), 2e-6f64.max(2.0 * j0));
            drop((pts0, seg0, e0));
            for lb in [1usize, 2, 3] {
                let groups = p.nseg.div_ceil(lb);
                let (lbb, padded) = (p.nseg.div_ceil(groups), p.nseg.div_ceil(groups) * groups > p.nseg);
                let opts_frames = if kind == Kind::Knot1d { MC } else { 1 };
                let bspline = match kind { Kind::Static => 1, Kind::Knot1d => 3, Kind::OneBin => 0 };
                set_thread_cb_cap(Some(cb_cap));
                let mut sp = SegSplit::new(&dev, DIMS, [RES; 3], p.nc, SegNufftOpts { sigma: 1.25, width: 8, nseg: p.nseg, order, frames: opts_frames, bspline, max_grid_bytes: 1 << 30 }, lb, kind != Kind::Static).unwrap();
                set_thread_cb_cap(None);
                assert_eq!((sp.groups, sp.lb), (groups, lbb));
                sp.eng.set_maps(&st, &p.smaps).unwrap();
                let seg: SegSplitSegments = sp.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
                let pts = points(&st, &mut sp.eng, &p, kind);
                let mut y = st.alloc_zeros::<C32>(ny).unwrap();
                sp.forward(&st, &seg, &pts, &x, &mut y).unwrap();
                let yh = st.clone_dtoh(&y).unwrap();
                let mut xa = st.alloc_zeros::<C32>(nx).unwrap();
                sp.adjoint(&st, &seg, &pts, &yd, &mut xa).unwrap();
                let xah = st.clone_dtoh(&xa).unwrap();
                // residual: A x − y_d
                let mut r = st.alloc_zeros::<C32>(ny).unwrap();
                sp.forward_residual(&st, &seg, &pts, &x, &yd, &mut r).unwrap();
                let rh = st.clone_dtoh(&r).unwrap();
                let rref: Vec<C32> = y0.iter().zip(&p.yh).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect();
                // adjoint accumulation: x ← x + Aᴴ y_d (x starts at the image)
                sp.eng.adjoint_accumulate = true;
                let mut xacc = st.clone_htod(&p.xh).unwrap();
                sp.adjoint(&st, &seg, &pts, &yd, &mut xacc).unwrap();
                sp.eng.adjoint_accumulate = false;
                let xacch = st.clone_dtoh(&xacc).unwrap();
                let accref: Vec<C32> = xa0.iter().zip(&p.xh).map(|(a, b)| C32::new(a.re + b.re, a.im + b.im)).collect();
                let (ef, ea, er, eacc) = (rel(&yh, &y0), rel(&xah, &xa0), rel(&rh, &rref), rel(&xacch, &accref));
                let j = (dot(&p.yh, &yh) - dot(&xah, &p.xh)).norm() / (dot(&p.yh, &p.yh).re * dot(&yh, &yh).re).sqrt();
                eprintln!("{kind:?} o{order} cb_cap {cb_cap} lb {lb} ({groups} groups of {lbb}{}, engine cb {}): forward {ef:.1e} (bound {bf:.1e}), adjoint {ea:.1e} (bound {ba:.1e}), residual {er:.1e}, accumulate {eacc:.1e}, adjointness {j:.1e} (plain {j0:.1e})",
                    if padded { ", padded" } else { "" }, sp.eng.cb);
                assert!(ef < bf && ea < ba && er < bf.max(1e-5) && eacc < ba && j < bj, "{kind:?} o{order} cb_cap {cb_cap} lb {lb}");
            }
        }
    }
}

#[test]
fn one_bin_knot_matches_static() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    for order in [0u32, 1, 2, 3] {
        for (nc, nseg, cb_cap) in [(3usize, 5usize, 3usize), (4, 3, 2)] {
            let p = prob(nc, nseg, order, 8000, 1);
            let (nx, ny) = (nv, p.nc * p.m);
            let (x, yd) = (st.clone_htod(&p.xh).unwrap(), st.clone_htod(&p.yh).unwrap());
            // reference: the static engine (order 3: the multi-image kernels; 0–2: the tile kernels)
            let mut es = engine(&dev, &p, Kind::Static, nseg, cb_cap);
            es.set_maps(&st, &p.smaps).unwrap();
            let segs = es.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
            let ptss = points(&st, &mut es, &p, Kind::Static);
            let y0 = fwd_plain(&st, &mut es, &segs, &ptss, &x, ny);
            let xa0 = adj_plain(&st, &mut es, &segs, &ptss, &yd, nx);
            let xp = st.clone_htod(&ulp(&p.xh, 17)).unwrap();
            let ydp = st.clone_htod(&ulp(&p.yh, 18)).unwrap();
            let (ff, fa) = (rel(&fwd_plain(&st, &mut es, &segs, &ptss, &xp, ny), &y0), rel(&adj_plain(&st, &mut es, &segs, &ptss, &ydp, nx), &xa0));
            let j0 = (dot(&p.yh, &y0) - dot(&xa0, &p.xh)).norm() / (dot(&p.yh, &p.yh).re * dot(&y0, &y0).re).sqrt();
            let (bf, ba, bj) = (3e-5f64.max(6.0 * ff), 3e-5f64.max(6.0 * fa), 2e-6f64.max(2.0 * j0));
            drop((ptss, segs, es));
            let mut ek = engine(&dev, &p, Kind::OneBin, nseg, cb_cap);
            assert_eq!(ek.ks, 1, "one-bin engine is knot-major with support 1");
            ek.set_maps(&st, &p.smaps).unwrap();
            let segk = ek.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
            let ptsk = points(&st, &mut ek, &p, Kind::OneBin);
            let mut check = |ek: &mut SegNufft, lab: &str| {
                let y = fwd_plain(&st, ek, &segk, &ptsk, &x, ny);
                let xa = adj_plain(&st, ek, &segk, &ptsk, &yd, nx);
                let (ef, ea) = (rel(&y, &y0), rel(&xa, &xa0));
                let j = (dot(&p.yh, &y) - dot(&xa, &p.xh)).norm() / (dot(&p.yh, &p.yh).re * dot(&y, &y).re).sqrt();
                eprintln!("one bin o{order} nc {nc} L {nseg} cb {} {lab}: forward {ef:.1e} (bound {bf:.1e}), adjoint {ea:.1e} (bound {ba:.1e}), adjointness {j:.1e} (static {j0:.1e})", ek.cb);
                assert!(ef < bf && ea < ba && j < bj, "one bin o{order} {lab}");
            };
            ek.set_vs(None).unwrap();
            ek.set_ra(None).unwrap();
            ek.set_kl(false);
            check(&mut ek, "tile kernels");
            // the autotune's choice, then each variant forced on at its best shape
            ek.autotune(&st, &segk, &ptsk).unwrap();
            let lab = format!("autotune (vs {:?}, ra {:?}, kl {} {:?})", ek.vs_active().map(|v| v.1), ek.ra_active(), ek.kl_active(), ek.kl_kernel());
            check(&mut ek, &lab);
            let (vc, rc, kc) = (ek.vs_tune_cfg(), ek.ra_tune_cfg(), ek.kl_tune_cfg());
            if vc.is_some() {
                ek.set_vs(vc).unwrap();
                check(&mut ek, &format!("vs/vt {vc:?}"));
            }
            if rc.is_some() {
                ek.set_ra(rc).unwrap();
                ek.set_kl(false);
                ek.set_kl_kernel(None).unwrap();
                check(&mut ek, &format!("ra {rc:?}"));
                ek.set_kl(true);
                check(&mut ek, "ra + skip empty tiles");
            }
            if kc.is_some() {
                ek.set_kl(true);
                ek.set_kl_kernel(kc).unwrap();
                check(&mut ek, &format!("kl kernel {kc:?}"));
            }
        }
    }
}

/// compute-sanitizer coverage of the one-bin knot kernels (KS = 1 specialisations) without the autotune's many
/// shapes: tile kernels, vt / vs, ra, ra + skip of empty tiles and the kl kernel at orders 0 and 3, and a segment
/// split on top (run under `compute-sanitizer --tool memcheck`); values checked as in `one_bin_knot_matches_static`.
#[test]
fn one_bin_sanitize() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    for order in [0u32, 3] {
        let p = prob(3, 4, order, 3000, 1);
        let (x, yd) = (st.clone_htod(&p.xh).unwrap(), st.clone_htod(&p.yh).unwrap());
        let mut es = engine(&dev, &p, Kind::Static, p.nseg, 3);
        es.set_maps(&st, &p.smaps).unwrap();
        let segs = es.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
        let ptss = points(&st, &mut es, &p, Kind::Static);
        let y0 = fwd_plain(&st, &mut es, &segs, &ptss, &x, p.nc * p.m);
        let xa0 = adj_plain(&st, &mut es, &segs, &ptss, &yd, nv);
        drop((ptss, segs, es));
        let mut ek = engine(&dev, &p, Kind::OneBin, p.nseg, 3);
        ek.set_maps(&st, &p.smaps).unwrap();
        let segk = ek.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
        let ptsk = points(&st, &mut ek, &p, Kind::OneBin);
        let rc = ek.ra_candidates().first().copied();
        let runs: Vec<(&str, Option<hasty_nufft::VsCfg>, Option<hasty_nufft::RaCfg>, bool, bool)> = vec![
            ("tile", None, None, false, false),
            ("vs", Some(hasty_nufft::VsCfg { nwc: 1, nws: 2, vxpad: 0 }), None, false, false),
            ("ra", None, rc, false, false),
            ("ra skip", None, rc, true, false),
            ("kl kernel", None, rc, true, true),
        ];
        for (lab, vc, rcfg, skip, klk) in runs {
            if ek.set_vs(vc).is_err() || ek.set_ra(rcfg).is_err() {
                eprintln!("o{order} {lab}: not available");
                continue;
            }
            ek.set_kl(skip);
            ek.set_kl_kernel(if klk { rcfg } else { None }).unwrap();
            let y = fwd_plain(&st, &mut ek, &segk, &ptsk, &x, p.nc * p.m);
            let xa = adj_plain(&st, &mut ek, &segk, &ptsk, &yd, nv);
            let (ef, ea) = (rel(&y, &y0), rel(&xa, &xa0));
            eprintln!("sanitize o{order} {lab}: forward {ef:.1e}, adjoint {ea:.1e}");
            assert!(ef < 1e-4 && ea < 2e-4, "o{order} {lab}");
        }
        drop((ptsk, segk, ek));
        // segment split of the one-bin engine (2 groups of 2)
        let mut sp = SegSplit::new(&dev, DIMS, [RES; 3], p.nc, SegNufftOpts { sigma: 1.25, width: 8, nseg: p.nseg, order, frames: 1, bspline: 0, max_grid_bytes: 1 << 30 }, 2, true).unwrap();
        sp.eng.set_maps(&st, &p.smaps).unwrap();
        let seg = sp.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
        let pts = points(&st, &mut sp.eng, &p, Kind::OneBin);
        let mut y = st.alloc_zeros::<C32>(p.nc * p.m).unwrap();
        sp.forward(&st, &seg, &pts, &x, &mut y).unwrap();
        let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
        sp.adjoint(&st, &seg, &pts, &yd, &mut xa).unwrap();
        let (ef, ea) = (rel(&st.clone_dtoh(&y).unwrap(), &y0), rel(&st.clone_dtoh(&xa).unwrap(), &xa0));
        eprintln!("sanitize o{order} split 2 x 2: forward {ef:.1e}, adjoint {ea:.1e}");
        assert!(ef < 1e-4 && ea < 2e-4);
    }
}

/// HASTY_SEG_STATIC_KNOT=1: a plain static engine (`SegNufft::new`, degree 1) is built knot-major with one bin and
/// equals the static path (orders 0 and 3); unset again afterwards
#[test]
fn static_knot_env_switch() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let nv: usize = DIMS.iter().product();
    for order in [0u32, 3] {
        let p = prob(3, 4, order, 4000, 1);
        let (x, yd) = (st.clone_htod(&p.xh).unwrap(), st.clone_htod(&p.yh).unwrap());
        let run = |env: bool| {
            // 1: every static engine one-bin; 0: none (the default since §20 makes order 3 one-bin, so the
            // reference sets 0 explicitly)
            std::env::set_var("HASTY_SEG_STATIC_KNOT", if env { "1" } else { "0" });
            let mut e = engine_env(&dev, &p, Kind::Static, p.nseg, 3);
            std::env::remove_var("HASTY_SEG_STATIC_KNOT");
            assert_eq!(e.ks, env as usize, "knot-major iff the switch is set");
            e.set_maps(&st, &p.smaps).unwrap();
            let seg = e.segments(&st, &p.sp, &p.at, p.t0, p.dt).unwrap();
            let pts = points(&st, &mut e, &p, Kind::Static);
            (fwd_plain(&st, &mut e, &seg, &pts, &x, p.nc * p.m), adj_plain(&st, &mut e, &seg, &pts, &yd, nv))
        };
        let (y0, xa0) = run(false);
        let (y1, xa1) = run(true);
        let (ef, ea) = (rel(&y1, &y0), rel(&xa1, &xa0));
        eprintln!("HASTY_SEG_STATIC_KNOT=1 o{order}: forward {ef:.1e}, adjoint {ea:.1e}");
        assert!(ef < 1e-4 && ea < 2e-4, "o{order}");
    }
}
