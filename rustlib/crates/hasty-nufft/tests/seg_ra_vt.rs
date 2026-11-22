//! The ra adjoint (segnufft_ra.cu, `SegNufft::set_ra`) and the vt forward (segnufft_vs.cu at orders 2 / 3,
//! `SegNufft::set_vs`; research/sample_kernels.md §16) against the tile kernels (variant 0, one coil per block):
//! * ra against the tile adjoint at orders 0–3 (static: 0–2; order 3 there is the multi-image path), static images and
//!   knot-major engines (1-D cubic, 2-D linear respiratory × cubic cardiac), several coil / segment / launch shapes
//!   (incl. several block rows, odd channel blocks, sets with and without split tile lists), under HASTY_SEG_ZERO=tiles
//!   and full (`zero_tiles`), with the fused pad / crop and L2 slab FFTs on and off;
//! * vt against the tile forward at orders 2 / 3 (static 2; knot 1-D / 2-D 2 and 3), several geometries and launch
//!   shapes;
//! * adjointness of each new kernel against the other side's tile kernel and of the new pair (vt forward, ra adjoint);
//! * the grouped and the source adjoint through the ra scatter; the autotune's choice.
//! Order-2 / 3 data are strong (ΔC up to ≈ 0.1 in kernel units, degree-3/4 coefficients ≈ 0.05) so that the e / hc
//! terms dominate the weight differences. Bounds as tests/seg_vs.rs: max(3e-5, 2 × the f32 floor of two tile shapes).
//!   cargo test --release -p hasty-nufft --test seg_ra_vt -- --nocapture --test-threads=1
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, hi_exponents, set_thread_cb_cap, RaCfg, SegNufft, SegNufftOpts, SegPoints, SegSegments, VsCfg, VsGeom, NHI};

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
    Knot2d,
}

struct Case {
    st: std::sync::Arc<cudarc::driver::CudaStream>,
    op: SegNufft,
    seg: SegSegments,
    pts: SegPoints,
    /// order 3: the same samples without the degree-3/4 data (the order-2 operator), to show the terms' size
    pts_lo: Option<SegPoints>,
    x: cudarc::driver::CudaSlice<C32>,
    yd: cudarc::driver::CudaSlice<C32>,
    xh: Vec<C32>,
    yh: Vec<C32>,
    nimg: usize,
    m: usize,
    nc: usize,
}

#[allow(clippy::too_many_arguments)]
fn build(kind: Kind, nc: usize, nseg: usize, order: u32, width: usize, m: usize, cb_cap: usize, geom: Option<VsGeom>) -> Case {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [24usize, 22, 24];
    let nv: usize = dims.iter().product();
    let res = 0.004;
    let (mc, kr) = (6, 3);
    let mut r = R(4242 + nseg as u64 * 31 + order as u64 * 7 + nc as u64 * 3 + m as u64);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    // ΔC: kernel-argument units d = dc · g², g = res nf / (π W) ≈ 4e-3 → |d| ≲ 0.1 at |dc| ≲ 1e4 (strong e terms)
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 1.2e4) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
    let nri = kr - 1;
    let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * nri as f64) as u32).min(nri as u32 - 1)).collect();
    let rv: Vec<f32> = (0..m).map(|_| (r.u() + 0.5) as f32).collect();
    let nimg = match kind {
        Kind::Static => 1,
        Kind::Knot1d => mc,
        Kind::Knot2d => mc * kr,
    };
    let xh: Vec<C32> = (0..nimg * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yh: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    set_thread_cb_cap(Some(cb_cap));
    let frames = if kind == Kind::Static { 1 } else { mc };
    let opts = SegNufftOpts { sigma: 1.25, width, nseg, order, frames, bspline: 3, max_grid_bytes: 1 << 30 };
    let mut op = match kind {
        Kind::Static => SegNufft::new(&dev, dims, [res; 3], nc, opts).unwrap(),
        Kind::Knot1d => SegNufft::new_temporal(&dev, dims, [res; 3], nc, opts, true).unwrap(),
        Kind::Knot2d => SegNufft::new_temporal2d(&dev, dims, [res; 3], nc, opts, kr, 1).unwrap(),
    };
    set_thread_cb_cap(None);
    op.set_maps(&st, &smaps).unwrap();
    op.set_vs_geom(geom).unwrap();
    let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
    let c = [&coords[0][..], &coords[1][..], &coords[2][..]];
    // order 3: u per sample (±0.5) and H' giving degree-3/4 coefficients of ≈ 0.05 in kernel units
    let hi = (order >= 3).then(|| {
        let u: Vec<f32> = (0..4 * m).map(|_| r.u() as f32).collect();
        let g: [f64; 3] = [0, 1, 2].map(|a| res * op.plan.nf[a] as f64 / (std::f64::consts::PI * width as f64));
        let ex = hi_exponents();
        let h: Vec<f64> = (0..4 * NHI).map(|i| 0.2 * r.u() / (0..3).map(|a| g[a].powi(ex[i % NHI][a] as i32)).product::<f64>()).collect();
        (u, h)
    });
    let pts = match (kind, &hi) {
        (Kind::Static, None) => op.points(&st, c, &tau, &phi0, &dcv, None).unwrap(),
        (Kind::Static, Some((u, h))) => op.points_hi(&st, c, &tau, &phi0, &dcv, u, h, None).unwrap(),
        (Kind::Knot1d, None) => op.points(&st, c, &tau, &phi0, &dcv, Some(&t)).unwrap(),
        (Kind::Knot1d, Some((u, h))) => op.points_hi(&st, c, &tau, &phi0, &dcv, u, h, Some(&t)).unwrap(),
        (Kind::Knot2d, None) => op.points2d(&st, c, &tau, &phi0, &dcv, &t, &ri, &rv).unwrap(),
        (Kind::Knot2d, Some((u, h))) => op.points2d_hi(&st, c, &tau, &phi0, &dcv, &t, &ri, &rv, u, h).unwrap(),
    };
    if geom.is_some() {
        assert!(pts.vs_table_bytes() > 0, "vs tables built");
    }
    let pts_lo = hi.as_ref().map(|_| match kind {
        Kind::Static => op.points(&st, c, &tau, &phi0, &dcv, None).unwrap(),
        Kind::Knot1d => op.points(&st, c, &tau, &phi0, &dcv, Some(&t)).unwrap(),
        Kind::Knot2d => op.points2d(&st, c, &tau, &phi0, &dcv, &t, &ri, &rv).unwrap(),
    });
    let (x, yd) = (st.clone_htod(&xh).unwrap(), st.clone_htod(&yh).unwrap());
    // base: tile kernels, variant 0, one coil per block
    let base = op.tile_shape(1, 1, 2, 1, 1, 0).unwrap();
    op.set_tiles(base, base).unwrap();
    op.set_vs(None).unwrap();
    op.set_ra(None).unwrap();
    Case { st, op, seg, pts, pts_lo, x, yd, xh, yh, nimg, m, nc }
}

impl Case {
    fn fwd(&mut self) -> Vec<C32> {
        let mut y = self.st.alloc_zeros::<C32>(self.nc * self.m).unwrap();
        self.op.forward(&self.st, &self.seg, &self.pts, &self.x, &mut y).unwrap();
        self.st.clone_dtoh(&y).unwrap()
    }
    fn adj(&mut self) -> Vec<C32> {
        let nv = self.op.nv();
        let mut xa = self.st.alloc_zeros::<C32>(self.nimg * nv).unwrap();
        self.op.adjoint(&self.st, &self.seg, &self.pts, &self.yd, &mut xa).unwrap();
        self.st.clone_dtoh(&xa).unwrap()
    }
    /// |⟨y_d, A x⟩ − ⟨Aᴴ y_d, x⟩| / (|y_d| |A x|) for a forward result y and an adjoint result xa
    fn adjness(&self, y: &[C32], xa: &[C32]) -> f64 {
        (dot(&self.yh, y) - dot(xa, &self.xh)).norm() / (dot(&self.yh, &self.yh).re * dot(y, y).re).sqrt()
    }
}

/// ra shapes the tests run (invalid ones skipped by `set_ra`)
fn ra_cfgs(c: &Case) -> Vec<RaCfg> {
    let mut v = c.op.ra_candidates();
    for (nxg, cb, chk) in [(2usize, 3usize, 16usize), (1, 1, 8), (4, 5, 32), (2, 7, 16), (1, 4, 16)] {
        let r = RaCfg { nxg, cb, chk };
        if !v.contains(&r) {
            v.push(r);
        }
    }
    v
}

/// ra at every shape against the tile adjoint (under ZERO=tiles and full); adjointness against the tile forward.
/// Returns the number of shapes run.
fn check_ra(c: &mut Case, tag: &str) -> usize {
    c.op.set_ra(None).unwrap();
    c.op.set_vs(None).unwrap();
    let y0 = c.fwd();
    let xa0 = c.adj();
    // f32 floor of the tile adjoint: (a) the conditioning, the base adjoint of the data moved by one ulp per element
    // (W = 10 with random data: the deapodisation amplifies grid-level rounding ≈ 1e-4 in the image); (b) knot engines:
    // the variant-1 tile adjoint (z = y a β folded first, as ra) against the base; variant-0 coil groupings are bitwise
    // equal and give no floor
    let (tb, ta) = (c.op.tiles_fwd, c.op.tiles_adj);
    let mut rp = R(99);
    let yp: Vec<C32> = c.yh.iter().map(|v| { let f = 1.0 + if rp.u() < 0.0 { -1.0 } else { 1.0 } * f32::EPSILON * 0.5; C32::new(v.re * f, v.im * f) }).collect();
    let ydp = c.st.clone_htod(&yp).unwrap();
    let ydk = std::mem::replace(&mut c.yd, ydp);
    let xap = c.adj();
    c.yd = ydk;
    let floor_p = rel(&xap, &xa0);
    let xk1 = match c.op.tile_shape(1, 1, 2, 1, 1, 1) {
        Some(k1) if c.op.ks > 0 => {
            c.op.set_tiles(tb, k1).unwrap();
            let x1 = c.adj();
            c.op.set_tiles(tb, ta).unwrap();
            Some(x1)
        }
        _ => None,
    };
    let floor_k = xk1.as_ref().map_or(0.0, |x1| rel(x1, &xa0));
    let floor_j = c.adjness(&y0, &xa0);
    eprintln!("{tag}: adjoint floors: one-ulp data {floor_p:.1e}, kv 1 vs kv 0 {floor_k:.1e}");
    // the kernels' own rounding is a few ulps per contribution: ≈ 4× the one-ulp conditioning (measured kv 1 vs kv 0)
    let (ba, bj) = (3e-5f64.max(6.0 * floor_p).max(3.0 * floor_k), 2e-6f64.max(2.0 * floor_j));
    let split = c.pts.has_split();
    let mut n = 0;
    for cfg in ra_cfgs(c) {
        if c.op.set_ra(Some(cfg)).is_err() {
            continue;
        }
        for zt in [true, false] {
            c.op.zero_tiles = zt;
            let xa = c.adj();
            let (e, j) = (rel(&xa, &xa0), c.adjness(&y0, &xa));
            // knot engines: against the variant-1 tile adjoint, the same z = y a β arithmetic
            let e1 = xk1.as_ref().map_or(f64::NAN, |x1| rel(&xa, x1));
            eprintln!("{tag} split {split} {cfg:?} zero {}: ra vs tile kv 0 {e:.1e} (bound {ba:.1e}), vs kv 1 {e1:.1e}, adjointness vs tile forward {j:.1e} (tile {floor_j:.1e})", if zt { "tiles" } else { "full" });
            assert!(std::env::var("RA_TEST_NOASSERT").is_ok() || (e < ba && j < bj), "{tag} {cfg:?} zero_tiles {zt}: adjoint {e:.1e} (bound {ba:.1e}), adjointness {j:.1e} (bound {bj:.1e})");
        }
        c.op.zero_tiles = true;
        n += 1;
    }
    c.op.set_ra(None).unwrap();
    n
}

fn vt_cfgs() -> Vec<VsCfg> {
    let mut v = Vec::new();
    for nwc in [1usize, 2, 3, 4, 5, 7] {
        for nws in [1usize, 2] {
            v.push(VsCfg { nwc, nws, vxpad: 0 });
        }
    }
    v
}

/// vt at every launch shape against the tile forward; adjointness against the tile adjoint and against the ra adjoint
/// (the new pair). Returns the number of shapes run.
fn check_vt(c: &mut Case, tag: &str) -> usize {
    c.op.set_ra(None).unwrap();
    c.op.set_vs(None).unwrap();
    let y0 = c.fwd();
    let xa = c.adj();
    let (tb, ta) = (c.op.tiles_fwd, c.op.tiles_adj);
    let alt = c.op.tile_shape(1, c.op.cb.min(4), 1, 1, 1, 0).unwrap();
    c.op.set_tiles(alt, ta).unwrap();
    let ya = c.fwd();
    c.op.set_tiles(tb, ta).unwrap();
    // conditioning: the base forward of the image moved by one ulp per element
    let mut rp = R(77);
    let xp: Vec<C32> = c.xh.iter().map(|v| { let f = 1.0 + if rp.u() < 0.0 { -1.0 } else { 1.0 } * f32::EPSILON * 0.5; C32::new(v.re * f, v.im * f) }).collect();
    let xdp = c.st.clone_htod(&xp).unwrap();
    let xk = std::mem::replace(&mut c.x, xdp);
    let yp = c.fwd();
    c.x = xk;
    let floor_p = rel(&yp, &y0);
    let (floor_f, floor_j) = (rel(&ya, &y0), c.adjness(&y0, &xa));
    eprintln!("{tag}: forward floors: one-ulp image {floor_p:.1e}, tile shapes {floor_f:.1e}");
    if let Some(lo) = c.pts_lo.take() {
        let hi = std::mem::replace(&mut c.pts, lo);
        let yl = c.fwd();
        c.pts_lo = Some(std::mem::replace(&mut c.pts, hi));
        let d = rel(&yl, &y0);
        eprintln!("{tag}: order-3 terms: tile forward with vs without the degree-3/4 data differs by {d:.1e}");
        assert!(d > 1e-3, "order-3 test data too weak ({d:.1e})");
    }
    let (bf, bj) = (3e-5f64.max(2.0 * floor_f).max(6.0 * floor_p), 2e-6f64.max(2.0 * floor_j));
    // the ra adjoint of this engine (default shape) for the pair
    let ra = c.op.ra_candidates().first().copied();
    let xr = ra.map(|r| {
        c.op.set_ra(Some(r)).unwrap();
        let v = c.adj();
        c.op.set_ra(None).unwrap();
        v
    });
    let mut n = 0;
    for cfg in vt_cfgs() {
        if c.op.set_vs(Some(cfg)).is_err() {
            continue;
        }
        let y = c.fwd();
        let (ef, j) = (rel(&y, &y0), c.adjness(&y, &xa));
        let jp = xr.as_ref().map_or(f64::NAN, |xr| c.adjness(&y, xr));
        eprintln!("{tag} {:?} {cfg:?}: vt vs tile forward {ef:.1e} (tile shapes {floor_f:.1e}), adjointness vs tile adjoint {j:.1e}, vs ra {ra:?} {jp:.1e} (tile {floor_j:.1e})", c.op.vs_geom().unwrap());
        assert!(std::env::var("RA_TEST_NOASSERT").is_ok() || (ef < bf && j < bj && (jp.is_nan() || jp < bj)), "{tag} {cfg:?}: forward {ef:.1e} (bound {bf:.1e}), adjointness {j:.1e} / pair {jp:.1e} (bound {bj:.1e})");
        n += 1;
    }
    c.op.set_vs(None).unwrap();
    n
}

#[test]
fn ra_static_orders() {
    for order in [0u32, 1, 2] {
        // 10 coils × 10 segments in one batch; dense (split tile lists) and sparse sets
        for m in [20_000usize, 2_500] {
            let mut c = build(Kind::Static, 10, 10, order, 10, m, 16, None);
            assert!(check_ra(&mut c, &format!("static o{order} 10x10 m{m}")) >= 4);
        }
        // 12 coils in batches of 4 (c0 > 0), 3 segments, W = 8
        let mut c = build(Kind::Static, 12, 3, order, 8, 12_000, 4, None);
        assert!(check_ra(&mut c, &format!("static o{order} 12(4)x3")) >= 4);
        // many coils, one segment
        let mut c = build(Kind::Static, 20, 1, order, 10, 15_000, 20, None);
        assert!(check_ra(&mut c, &format!("static o{order} 20x1")) >= 4);
    }
}

#[test]
fn ra_knot_orders() {
    for order in [0u32, 1, 2, 3] {
        let mut c = build(Kind::Knot1d, 8, 4, order, 8, 8_000, 8, None);
        assert!(check_ra(&mut c, &format!("knot1d o{order} 8x4")) >= 4);
        let mut c = build(Kind::Knot1d, 6, 5, order, 10, 12_000, 3, None);
        assert!(check_ra(&mut c, &format!("knot1d o{order} 6(3)x5")) >= 4);
        let mut c = build(Kind::Knot2d, 4, 3, order, 10, 10_000, 4, None);
        assert!(check_ra(&mut c, &format!("knot2d o{order} 4x3")) >= 4);
    }
}

const G16: VsGeom = VsGeom { tx: 4, ty: 4, tz: 16, smax: 48 };
const G8: VsGeom = VsGeom { tx: 2, ty: 4, tz: 8, smax: 32 };
const G48: VsGeom = VsGeom { tx: 4, ty: 2, tz: 8, smax: 48 };

#[test]
fn vt_static_order2() {
    let mut c = build(Kind::Static, 10, 10, 2, 10, 20_000, 16, Some(G16));
    // 7 channel groups: one block row (nwc 7) or two (nwc 5); nws 2 exceeds the 256-thread cap
    assert!(check_vt(&mut c, "static o2 10x10") >= 2);
    for g in [G8, G48] {
        let mut c = build(Kind::Static, 12, 3, 2, 8, 12_000, 4, Some(g));
        assert!(check_vt(&mut c, "static o2 12(4)x3") >= 2);   // one channel group: nws 1 / 2
    }
    let mut c = build(Kind::Static, 20, 1, 2, 10, 15_000, 20, Some(G16));
    assert!(check_vt(&mut c, "static o2 20x1") >= 2);
}

#[test]
fn vt_knot_orders() {
    for order in [2u32, 3] {
        let mut c = build(Kind::Knot1d, 8, 4, order, 8, 8_000, 8, Some(G16));
        assert!(check_vt(&mut c, &format!("knot1d o{order} 8x4")) >= 2);
        let mut c = build(Kind::Knot1d, 6, 5, order, 10, 8_000, 3, Some(G8));
        assert!(check_vt(&mut c, &format!("knot1d o{order} 6(3)x5")) >= 2);
        let mut c = build(Kind::Knot2d, 4, 3, order, 10, 10_000, 4, Some(G48));
        assert!(check_vt(&mut c, &format!("knot2d o{order} 4x3")) >= 2);
        let mut c = build(Kind::Knot2d, 10, 10, order, 10, 12_000, 5, Some(G16));
        assert!(check_vt(&mut c, &format!("knot2d o{order} 10(5)x10")) >= 1);
    }
}

/// ra and vt with the fused pad / crop and the L2 slab FFTs on and off (§23): the new pair equals the tile pair in
/// every combination
#[test]
fn ra_vt_fused_and_slabs() {
    for (kind, order) in [(Kind::Static, 2u32), (Kind::Knot2d, 3)] {
        let mut c = build(kind, 6, 4, order, 10, 9_000, 6, Some(G16));
        let (vc, rc) = (VsCfg { nwc: 2, nws: 1, vxpad: 0 }, c.op.ra_candidates()[0]);
        for (fuse, slab) in [(true, None), (true, Some(0)), (false, None), (false, Some(0)), (true, Some(2))] {
            c.op.fuse_padcrop = fuse;
            c.op.set_fft_slab(slab);
            c.op.set_vs(None).unwrap();
            c.op.set_ra(None).unwrap();
            let (y0, x0) = (c.fwd(), c.adj());
            c.op.set_vs(Some(vc)).unwrap();
            c.op.set_ra(Some(rc)).unwrap();
            let (y1, x1) = (c.fwd(), c.adj());
            let (ef, ea, j) = (rel(&y1, &y0), rel(&x1, &x0), c.adjness(&y1, &x1));
            eprintln!("{kind:?} o{order} fuse {fuse} slab {slab:?}: vt {ef:.1e}, ra {ea:.1e}, pair adjointness {j:.1e}");
            assert!(ef < 1e-4 && ea < 1e-4 && j < 2e-6, "{kind:?} fuse {fuse} slab {slab:?}: vt {ef:.1e}, ra {ea:.1e}, adjointness {j:.1e}");
        }
    }
}

/// the grouped adjoint (static) and the source adjoint (static and knot) through the ra scatter equal the tile path,
/// under ZERO=tiles and full
#[test]
fn ra_group_and_source() {
    for kind in [Kind::Static, Kind::Knot1d] {
        let mut c = build(kind, 6, 4, 2, 8, 9_000, 3, None);
        let (m, nc, nimg, nv) = (c.m, c.nc, c.nimg, c.op.nv());
        let run = |c: &mut Case| -> (Vec<C32>, Vec<C32>) {
            let g = if kind == Kind::Static {
                let mut x = c.st.alloc_zeros::<C32>(nimg * nv).unwrap();
                c.op.adjoint_group(&c.st, &c.seg, &[&c.pts, &c.pts], &[&c.yd, &c.yd], &mut x).unwrap();
                c.st.clone_dtoh(&x).unwrap()
            } else {
                vec![]
            };
            let mut buf = c.st.alloc_zeros::<C32>(nc * m).unwrap();
            let mut x = c.st.alloc_zeros::<C32>(nimg * nv).unwrap();
            let yh = c.yh.clone();
            {
                let mut src = |st: &std::sync::Arc<cudarc::driver::CudaStream>, _j: usize, c0: usize, ncb: usize, b: &mut cudarc::driver::CudaSlice<C32>| -> Result<(), hasty_nufft::NufftError> {
                    st.memcpy_htod(&yh[c0 * m..(c0 + ncb) * m], &mut b.slice_mut(c0 * m..(c0 + ncb) * m)).unwrap();
                    Ok(())
                };
                c.op.adjoint_source(&c.st, &c.seg, &[&c.pts, &c.pts], &mut buf, &mut src, &mut x).unwrap();
            }
            (g, c.st.clone_dtoh(&x).unwrap())
        };
        for zt in [true, false] {
            c.op.zero_tiles = zt;
            c.op.set_ra(None).unwrap();
            let (g0, s0) = run(&mut c);
            for cfg in c.op.ra_candidates().into_iter().take(3) {
                c.op.set_ra(Some(cfg)).unwrap();
                let (g1, s1) = run(&mut c);
                let eg = if g0.is_empty() { 0.0 } else { rel(&g1, &g0) };
                let es = rel(&s1, &s0);
                eprintln!("{kind:?} zero_tiles {zt} {cfg:?}: group {eg:.1e}, source {es:.1e}");
                assert!(eg < 3e-5 && es < 3e-5, "{kind:?} zero_tiles {zt} {cfg:?}: group {eg:.1e}, source {es:.1e}");
            }
        }
    }
}

/// the autotune keeps ra / vt only where it measured faster (unless forced by env); the chosen operator equals the base
#[test]
fn ra_vt_autotune_choice() {
    for (kind, order) in [(Kind::Static, 2u32), (Kind::Knot1d, 3)] {
        let mut c = build(kind, 10, 10, order, 10, 30_000, 16, Some(G16));
        let (y0, x0) = (c.fwd(), c.adj());
        c.op.autotune(&c.st, &c.seg, &c.pts).unwrap();
        let (ta, tr) = c.op.ra_tune_times().unwrap();
        let (tt, tv) = c.op.vs_tune_times().unwrap();
        eprintln!("{kind:?} o{order}: autotune adjoint tile {ta:.3} ms, ra {tr:.3} ms -> {:?}; forward tile {tt:.3} ms, vt {tv:.3} ms -> {:?}", c.op.ra_active(), c.op.vs_active());
        if std::env::var("HASTY_SEG_VSA").is_err() {
            assert_eq!(c.op.ra_active().is_some(), tr < ta, "ra kept iff faster");
        }
        if std::env::var("HASTY_SEG_VS").is_err() && std::env::var("HASTY_SEG_VT").is_err() {
            assert_eq!(c.op.vs_active().is_some(), tv < tt, "vt kept iff faster");
        }
        let (y, x) = (c.fwd(), c.adj());
        assert!(rel(&y, &y0) < 2e-4 && rel(&x, &x0) < 2e-4, "{kind:?}: forward {:.1e}, adjoint {:.1e}", rel(&y, &y0), rel(&x, &x0));
    }
}

/// one small case for compute-sanitizer (env RA_SAN = 1; ignored without it): every ra and vt shape once, static order
/// 0 / 2 and knot 2-D order 3, split lists
#[test]
fn ra_vt_sanitizer_case() {
    if std::env::var("RA_SAN").is_err() {
        return;
    }
    for (kind, order) in [(Kind::Static, 0u32), (Kind::Static, 2), (Kind::Knot2d, 3), (Kind::Knot1d, 1)] {
        let mut c = build(kind, 4, 3, order, 10, 3_000, 4, if order >= 2 { Some(G16) } else { None });
        for cfg in ra_cfgs(&c) {
            if c.op.set_ra(Some(cfg)).is_ok() {
                for zt in [true, false] {
                    c.op.zero_tiles = zt;
                    let _ = c.adj();
                }
            }
        }
        c.op.set_ra(None).unwrap();
        if order >= 2 {
            for cfg in vt_cfgs() {
                if c.op.set_vs(Some(cfg)).is_ok() {
                    let _ = c.fwd();
                }
            }
        }
        eprintln!("sanitizer case {kind:?} o{order} done (split {})", c.pts.has_split());
    }
}
