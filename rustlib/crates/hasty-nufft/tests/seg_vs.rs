//! The vs forward (segnufft_vs.cu, `SegNufft::set_vs`; research/sample_kernels.md §12, §14) against the tile kernels
//! (variant 0, one coil per block): same forward operator to f32 summation order at correction orders 0 and 1, static
//! images and knot-major engines (1-D cubic, 2-D linear respiratory × cubic cardiac), several home-tile geometries,
//! launch shapes (channel groups per block incl. two block rows, window splits, zero-padded x weights), coil batches
//! (several passes, c0 > 0) and segment counts; adjointness against the existing adjoint; the fused residual, the sink
//! and the group forward through the vs gather; the autotune's choice. Bounds as tests/seg_knot_kv.rs.
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, set_thread_cb_cap, SegNufft, SegNufftOpts, SegPoints, SegSegments, VsCfg, VsGeom};

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

#[derive(Clone, Copy, Debug)]
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
    x: cudarc::driver::CudaSlice<C32>,
    yd: cudarc::driver::CudaSlice<C32>,
    xh: Vec<C32>,
    yh: Vec<C32>,
    nimg: usize,
    m: usize,
    nc: usize,
}

#[allow(clippy::too_many_arguments)]
fn build(kind: Kind, nc: usize, nseg: usize, order: u32, width: usize, m: usize, cb_cap: usize, geom: VsGeom) -> Case {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [24usize, 22, 24];
    let nv: usize = dims.iter().product();
    let mc = 6;
    let kr = 3;
    let mut r = R(777 + nseg as u64 * 31 + order as u64 * 7 + nc as u64 * 3 + m as u64);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
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
    let frames = if matches!(kind, Kind::Static) { 1 } else { mc };
    let opts = SegNufftOpts { sigma: 1.25, width, nseg, order, frames, bspline: 3, max_grid_bytes: 1 << 30 };
    let mut op = match kind {
        Kind::Static => SegNufft::new(&dev, dims, [0.004; 3], nc, opts).unwrap(),
        Kind::Knot1d => SegNufft::new_temporal(&dev, dims, [0.004; 3], nc, opts, true).unwrap(),
        Kind::Knot2d => SegNufft::new_temporal2d(&dev, dims, [0.004; 3], nc, opts, kr, 1).unwrap(),
    };
    set_thread_cb_cap(None);
    op.set_maps(&st, &smaps).unwrap();
    op.set_vs_geom(Some(geom)).unwrap();
    let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
    let c = [&coords[0][..], &coords[1][..], &coords[2][..]];
    let pts = match kind {
        Kind::Static => op.points(&st, c, &tau, &phi0, &dcv, None).unwrap(),
        Kind::Knot1d => op.points(&st, c, &tau, &phi0, &dcv, Some(&t)).unwrap(),
        Kind::Knot2d => op.points2d(&st, c, &tau, &phi0, &dcv, &t, &ri, &rv).unwrap(),
    };
    assert!(pts.vs_table_bytes() > 0, "vs tables built");
    let (x, yd) = (st.clone_htod(&xh).unwrap(), st.clone_htod(&yh).unwrap());
    // base: tile kernels, variant 0, one coil per block
    let base = op.tile_shape(1, 1, 2, 1, 1, 0).unwrap();
    op.set_tiles(base, base).unwrap();
    Case { st, op, seg, pts, x, yd, xh, yh, nimg, m, nc }
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
}

/// vs forward at every launch shape in `cfgs` (invalid ones skipped) against the tile forward; adjointness against the
/// tile adjoint. Returns the number of shapes run.
fn check(c: &mut Case, cfgs: &[VsCfg], tag: &str) -> usize {
    c.op.set_vs(None).unwrap();
    let y0 = c.fwd();
    let xa = c.adj();
    let (yh, xh) = (c.yh.clone(), c.xh.clone());
    let adjness = |y: &[C32]| (dot(&yh, y) - dot(&xa, &xh)).norm() / (dot(&yh, &yh).re * dot(y, y).re).sqrt();
    // f32 noise floor of the tile kernels themselves: another launch shape (variant 0) against the base shape (W = 10
    // with random images: ≈ 5e-5, W = 8: ≈ 5e-6); bounds as tests/seg_knot_kv.rs or twice the floor, whichever is larger
    let (tb, ta) = (c.op.tiles_fwd, c.op.tiles_adj);
    let alt = c.op.tile_shape(1, c.op.cb.min(4), 1, 1, 1, 0).unwrap();
    c.op.set_tiles(alt, ta).unwrap();
    let ya = c.fwd();
    c.op.set_tiles(tb, ta).unwrap();
    let (floor_f, floor_a) = (rel(&ya, &y0), adjness(&y0));
    let (bf, ba) = (3e-5f64.max(2.0 * floor_f), 2e-6f64.max(2.0 * floor_a));
    let mut n = 0;
    for &cfg in cfgs {
        if c.op.set_vs(Some(cfg)).is_err() {
            continue;
        }
        let y = c.fwd();
        let ef = rel(&y, &y0);
        let e = adjness(&y);
        eprintln!("{tag} {:?} {cfg:?}: vs forward vs tile kv 0 {ef:.1e} (tile shapes {floor_f:.1e}), adjointness {e:.1e} (tile {floor_a:.1e})", c.op.vs_geom().unwrap());
        assert!(std::env::var("VS_TEST_NOASSERT").is_ok() || (ef < bf && e < ba), "{tag} {cfg:?}: forward {ef:.1e} (bound {bf:.1e}), adjointness {e:.1e} (bound {ba:.1e})");
        n += 1;
    }
    c.op.set_vs(None).unwrap();
    n
}

fn cfgs() -> Vec<VsCfg> {
    let mut v = Vec::new();
    for nwc in [1usize, 2, 3, 5, 7] {
        for (nws, vxpad) in [(1usize, 0usize), (2, 0), (1, 1), (4, 1)] {
            v.push(VsCfg { nwc, nws, vxpad });
        }
    }
    v
}

const G16: VsGeom = VsGeom { tx: 4, ty: 4, tz: 16, smax: 64 };
const G8: VsGeom = VsGeom { tx: 2, ty: 4, tz: 8, smax: 32 };
const G48: VsGeom = VsGeom { tx: 4, ty: 2, tz: 8, smax: 48 };

#[test]
fn vs_static_orders() {
    for order in [0u32, 1] {
        // 10 coils × 10 segments in one batch (7 channel groups: one block row, or 5 groups × 2 rows)
        let mut c = build(Kind::Static, 10, 10, order, 10, 20_000, 16, G16);
        assert!(check(&mut c, &cfgs(), &format!("static o{order} 10x10")) >= 2);
        // 12 coils in batches of 4 (c0 > 0), 3 segments, W = 8 (rows ≥ W zero), two geometries
        for g in [G8, G48] {
            let mut c = build(Kind::Static, 12, 3, order, 8, 12_000, 4, g);
            assert!(check(&mut c, &cfgs(), &format!("static o{order} 12(4)x3")) >= 3);
        }
        // many coils, one segment (3 channel groups; ragged last coil batch not possible: balanced)
        let mut c = build(Kind::Static, 20, 1, order, 10, 15_000, 20, G16);
        assert!(check(&mut c, &cfgs(), &format!("static o{order} 20x1")) >= 3);
    }
}

#[test]
fn vs_knot_orders() {
    for order in [0u32, 1] {
        let mut c = build(Kind::Knot1d, 8, 4, order, 8, 8_000, 8, G16);
        assert!(check(&mut c, &cfgs(), &format!("knot1d o{order} 8x4")) >= 3);
        let mut c = build(Kind::Knot1d, 6, 5, order, 10, 8_000, 3, G8);
        assert!(check(&mut c, &cfgs(), &format!("knot1d o{order} 6(3)x5")) >= 3);
        let mut c = build(Kind::Knot2d, 4, 3, order, 8, 10_000, 4, G48);
        assert!(check(&mut c, &cfgs(), &format!("knot2d o{order} 4x3")) >= 3);
    }
}

/// the fused residual, the sink (chunked use) and the group forward through the vs gather equal the tile path
#[test]
fn vs_residual_sink_group() {
    for kind in [Kind::Static, Kind::Knot1d] {
        let mut c = build(kind, 6, 4, 1, 8, 9_000, 3, G16);
        let cfg = VsCfg { nwc: 1, nws: 1, vxpad: 0 };
        let (m, nc) = (c.m, c.nc);
        let run = |c: &mut Case| {
            // residual A x − y_d
            let mut r = c.st.alloc_zeros::<C32>(nc * m).unwrap();
            c.op.forward_residual(&c.st, &c.seg, &c.pts, &c.x, &c.yd, &mut r).unwrap();
            // sink: parts summed on the host
            let mut buf = c.st.alloc_zeros::<C32>(nc * m).unwrap();
            let mut acc = vec![C32::new(0.0, 0.0); nc * m];
            {
                let mut sink = |st: &std::sync::Arc<cudarc::driver::CudaStream>, j: usize, c0: usize, ncb: usize, b: &cudarc::driver::CudaSlice<C32>| -> Result<(), hasty_nufft::NufftError> {
                    assert_eq!(j, 0);
                    let h = st.clone_dtoh(&b.slice(c0 * m..(c0 + ncb) * m)).unwrap();
                    for (a, v) in acc[c0 * m..(c0 + ncb) * m].iter_mut().zip(h) {
                        *a = C32::new(a.re + v.re, a.im + v.im);
                    }
                    Ok(())
                };
                c.op.forward_sink(&c.st, &c.seg, &[&c.pts], &c.x, &mut buf, &mut sink).unwrap();
            }
            // group (static engines only)
            let g = if matches!(kind, Kind::Static) {
                let mut y = c.st.alloc_zeros::<C32>(nc * m).unwrap();
                c.op.forward_group(&c.st, &c.seg, &[&c.pts], &c.x, &mut [&mut y]).unwrap();
                c.st.clone_dtoh(&y).unwrap()
            } else {
                vec![]
            };
            (c.st.clone_dtoh(&r).unwrap(), acc, g)
        };
        c.op.set_vs(None).unwrap();
        let (r0, s0, g0) = run(&mut c);
        c.op.set_vs(Some(cfg)).unwrap();
        let (r1, s1, g1) = run(&mut c);
        let (er, es) = (rel(&r1, &r0), rel(&s1, &s0));
        let eg = if g0.is_empty() { 0.0 } else { rel(&g1, &g0) };
        eprintln!("{kind:?}: residual {er:.1e}, sink {es:.1e}, group {eg:.1e}");
        assert!(er < 3e-5 && es < 3e-5 && eg < 3e-5, "{kind:?}: residual {er:.1e}, sink {es:.1e}, group {eg:.1e}");
    }
}

/// the autotune keeps vs only where it measured faster; the chosen operator equals the base forward
#[test]
fn vs_autotune_choice() {
    for kind in [Kind::Static, Kind::Knot1d] {
        let mut c = build(kind, 10, 10, 0, 10, 30_000, 16, G16);
        c.op.set_vs(None).unwrap();
        let y0 = c.fwd();
        c.op.autotune(&c.st, &c.seg, &c.pts).unwrap();
        let (tt, tv) = c.op.vs_tune_times().unwrap();
        let act = c.op.vs_active();
        eprintln!("{kind:?}: autotune tile {tt:.3} ms, vs {tv:.3} ms -> {act:?}");
        if std::env::var("HASTY_SEG_VS").is_err() {
            assert_eq!(act.is_some(), tv < tt, "vs kept iff faster");
        }
        // W = 10 with random images: the tile kernels' own shapes differ by ≈ 5e-5 (see `check`)
        let y = c.fwd();
        assert!(rel(&y, &y0) < 2e-4, "{kind:?}: {:.1e}", rel(&y, &y0));
    }
}

/// one case from env VS_DBG = kind(0 static, 1 knot1d, 2 knot2d),nc,nseg,order,w,m,cb_cap,tx,ty,tz,smax (ignored
/// without it; for debugging and shape studies)
#[test]
fn vs_env_case() {
    let Ok(v) = std::env::var("VS_DBG") else { return };
    let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect();
    let kind = [Kind::Static, Kind::Knot1d, Kind::Knot2d][p[0]];
    let mut c = build(kind, p[1], p[2], p[3] as u32, p[4], p[5], p[6], VsGeom { tx: p[7], ty: p[8], tz: p[9], smax: p[10] });
    check(&mut c, &cfgs(), &format!("env {v}"));
    // noise floor: another tile shape (variant 0) against the base shape
    c.op.set_vs(None).unwrap();
    let y0 = c.fwd();
    for (cpbf, ppf) in [(1usize, 1usize), (2, 4), (4, 1)] {
        let Some(t) = c.op.tile_shape(1, cpbf.min(c.op.cb), ppf, 1, 1, 0) else { continue };
        let ta = c.op.tiles_adj;
        c.op.set_tiles(t, ta).unwrap();
        let y = c.fwd();
        eprintln!("env {v}: tile cpbf {cpbf} ppf {ppf} vs base {:.1e}", rel(&y, &y0));
    }
}
