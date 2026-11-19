//! Launch shapes of research/coil_segment_compression.md §12 against the original shapes (GPU): forward coil register
//! blocking (`TileCfg::fcb`, segnufft.cu / segnufft_knot.cu variants 0 and 1), adjoint coil blocking with large coil
//! groups (acb, threads = tt³ · cpba / acb), ragged coil groups (cpb not dividing the batch: the last group masked),
//! the engine's default shapes (`TileCfg::new_pair`), balanced coil batches, and the multi-image kernels (static order
//! 3) with ragged groups. Every shape gives the same operator as the base shape (one coil per block, no blocking) to
//! f32 summation order: forward, adjoint, and the normalised adjointness of each shape. Bounds as tests/seg_knot_kv.rs.
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, set_thread_cb_cap, SegNufft, SegNufftOpts, SegPoints, SegSegments, TileCfg, NHI};

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

/// engine kind: static (frames 1), 1-D cubic knots, 2-D (3 linear respiratory × cubic cardiac) knots
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
    yh: Vec<C32>,
    nimg: usize,
    m: usize,
    nc: usize,
}

fn build(kind: Kind, nc: usize, nseg: usize, order: u32, m: usize, cb_cap: Option<usize>, max_grid_bytes: usize) -> Case {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [24usize, 20, 16];
    let nv: usize = dims.iter().product();
    let mc = 6;
    let mut r = R(4242 + nseg as u64 * 31 + order as u64 * 7 + nc as u64 * 3);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
    let kr = 3;
    let nri = kr - 1;
    let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * nri as f64) as u32).min(nri as u32 - 1)).collect();
    let rv: Vec<f32> = (0..m).map(|_| (r.u() + 0.5) as f32).collect();
    let uh: Vec<f32> = (0..4 * m).map(|_| (r.u() * 1.0) as f32).collect();
    let hh: Vec<f64> = (0..4 * NHI).map(|_| r.u() * 20.0).collect();
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
    set_thread_cb_cap(cb_cap);
    let mut op = match kind {
        Kind::Static => SegNufft::new(&dev, dims, [0.004; 3], nc, SegNufftOpts { sigma: 1.25, width: 8, nseg, order, frames: 1, bspline: 1, max_grid_bytes }).unwrap(),
        Kind::Knot1d => SegNufft::new_temporal(&dev, dims, [0.004; 3], nc, SegNufftOpts { sigma: 1.25, width: 8, nseg, order, frames: mc, bspline: 3, max_grid_bytes }, true).unwrap(),
        Kind::Knot2d => SegNufft::new_temporal2d(&dev, dims, [0.004; 3], nc, SegNufftOpts { sigma: 1.25, width: 8, nseg, order, frames: mc, bspline: 3, max_grid_bytes }, kr, 1).unwrap(),
    };
    set_thread_cb_cap(None);
    op.set_maps(&st, &smaps).unwrap();
    let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
    let c = [&coords[0][..], &coords[1][..], &coords[2][..]];
    let pts = match (kind, order >= 3) {
        (Kind::Static, false) => op.points(&st, c, &tau, &phi0, &dcv, None).unwrap(),
        (Kind::Static, true) => op.points_hi(&st, c, &tau, &phi0, &dcv, &uh, &hh, None).unwrap(),
        (Kind::Knot1d, false) => op.points(&st, c, &tau, &phi0, &dcv, Some(&t)).unwrap(),
        (Kind::Knot1d, true) => op.points_hi(&st, c, &tau, &phi0, &dcv, &uh, &hh, Some(&t)).unwrap(),
        (Kind::Knot2d, false) => op.points2d(&st, c, &tau, &phi0, &dcv, &t, &ri, &rv).unwrap(),
        (Kind::Knot2d, true) => op.points2d_hi(&st, c, &tau, &phi0, &dcv, &t, &ri, &rv, &uh, &hh).unwrap(),
    };
    let (x, yd) = (st.clone_htod(&xh).unwrap(), st.clone_htod(&yh).unwrap());
    Case { st, op, seg, pts, x, yd, yh, nimg, m, nc }
}

impl Case {
    fn run(&mut self) -> (Vec<C32>, Vec<C32>) {
        let st = self.st.clone();
        let nv = self.op.nv();
        let mut y = st.alloc_zeros::<C32>(self.nc * self.m).unwrap();
        let mut xa = st.alloc_zeros::<C32>(self.nimg * nv).unwrap();
        self.op.forward(&st, &self.seg, &self.pts, &self.x, &mut y).unwrap();
        self.op.adjoint(&st, &self.seg, &self.pts, &self.yd, &mut xa).unwrap();
        (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap())
    }

    fn xh(&self) -> Vec<C32> {
        self.st.clone_dtoh(&self.x).unwrap()
    }

    /// run `shapes` (forward, adjoint) against the base shape; returns how many ran
    fn compare(&mut self, label: &str, shapes: &[(TileCfg, TileCfg)]) -> usize {
        let base = self.op.tile_shape(1, 1, 2, 1, 1, 0).unwrap();
        self.op.set_tiles(base, base).unwrap();
        let (y0, x0) = self.run();
        let xh = self.xh();
        for &(tf, ta) in shapes {
            self.op.set_tiles(tf, ta).unwrap();
            let (y, xa) = self.run();
            let (ef, ea) = (rel(&y, &y0), rel(&xa, &x0));
            let e = (dot(&self.yh, &y) - dot(&xa, &xh)).norm() / (dot(&self.yh, &self.yh).re * dot(&y, &y).re).sqrt();
            eprintln!("{label}: fwd {tf:?}\n    adj {ta:?}: vs base forward {ef:.1e}, adjoint {ea:.1e}, adjointness {e:.1e}");
            assert!(ef < 3e-5 && ea < 6e-5 && e < 2e-6, "{label} {tf:?} / {ta:?}: forward {ef:.1e}, adjoint {ea:.1e}, adjointness {e:.1e}");
        }
        shapes.len()
    }
}

/// candidate shapes: (cpba, cpbf, ppf, acb, fsb, fcb, kv) pairs that fit the engine
fn shapes(op: &SegNufft, list: &[(usize, usize, usize, usize, usize, usize, usize)]) -> Vec<(TileCfg, TileCfg)> {
    let mut v = vec![];
    for &(cpba, cpbf, ppf, acb, fsb, fcb, kv) in list {
        // forward shape: one adjoint coil (its adjoint is not launched); adjoint shape: one forward coil
        let (Some(tf), Some(ta)) = (op.tile_shape_b(1, cpbf, ppf, 1, fsb, fcb, kv), op.tile_shape_b(cpba, 1, ppf, acb, 1, 1, if acb > 1 { 0 } else { kv })) else {
            eprintln!("  shape ({cpba}, {cpbf}, {ppf}, {acb}, {fsb}, {fcb}, {kv}) does not fit: skipped");
            continue;
        };
        v.push((tf, ta));
    }
    v
}

const STATIC_SHAPES: [(usize, usize, usize, usize, usize, usize, usize); 8] = [
    (8, 8, 2, 4, 1, 4, 0),
    (28, 12, 2, 4, 1, 4, 0),   // ragged on 40 coils: 2 adjoint groups of 28, 4 forward groups of 12
    (20, 40, 2, 4, 1, 8, 0),
    (12, 20, 1, 2, 2, 2, 0),
    (7, 6, 2, 1, 1, 2, 0),     // ragged, odd
    (40, 24, 2, 8, 1, 8, 0),
    (16, 9, 4, 4, 1, 3, 0),    // fcb 3: odd blocking
    (32, 16, 2, 4, 2, 4, 0),   // fcb × fsb
];

#[test]
fn static_blocked_and_ragged_shapes() {
    for (nseg, order) in [(1usize, 0u32), (1, 2), (2, 1), (3, 2)] {
        let mut c = build(Kind::Static, 40, nseg, order, 5000, Some(40), 1 << 30);
        assert_eq!(c.op.cb, 40);
        let mut list = shapes(&c.op, &STATIC_SHAPES);
        // the engine's own default (new_pair) shapes
        list.push((c.op.tiles_fwd, c.op.tiles_adj));
        let n = c.compare(&format!("static nseg {nseg} order {order}"), &list);
        assert!(n >= 7, "only {n} shapes ran");
    }
}

#[test]
fn knot_blocked_and_ragged_shapes() {
    for (kind, nseg, order) in [(Kind::Knot1d, 1usize, 0u32), (Kind::Knot1d, 2, 2), (Kind::Knot1d, 3, 1), (Kind::Knot2d, 1, 3), (Kind::Knot2d, 2, 3)] {
        let mut c = build(kind, 24, nseg, order, 4000, Some(24), 1 << 30);
        assert_eq!(c.op.cb, 24);
        let mut list = vec![];
        for kv in [0usize, 1] {
            list.extend(shapes(&c.op, &[(8, 8, 2, 4, 1, 4, kv), (24, 24, 2, 4, 1, 8, kv), (12, 9, 2, 2, 1, 3, kv), (5, 7, 2, 1, 2, 1, kv), (10, 16, 1, 1, 1, 2, kv), (20, 12, 4, 4, 1, 4, kv), (16, 16, 2, 4, 2, 2, kv)]));
        }
        list.push((c.op.tiles_fwd, c.op.tiles_adj));
        let n = c.compare(&format!("{kind:?} nseg {nseg} order {order}"), &list);
        assert!(n >= 10, "only {n} shapes ran");
    }
}

/// split tile lists (atomic items) with blocking and ragged groups
#[test]
fn blocked_shapes_split_lists() {
    let mut c = build(Kind::Static, 20, 1, 2, 40_000, Some(20), 1 << 30);
    let list = shapes(&c.op, &[(8, 8, 2, 4, 1, 4, 0), (12, 20, 2, 4, 1, 4, 0), (20, 6, 2, 4, 1, 2, 0)]);
    assert!(c.compare("static split lists", &list) >= 3);
    let mut c = build(Kind::Knot1d, 20, 2, 0, 40_000, Some(20), 1 << 30);
    let mut list = shapes(&c.op, &[(8, 8, 2, 4, 1, 4, 1), (12, 20, 2, 1, 1, 4, 1), (20, 6, 2, 4, 1, 2, 0)]);
    list.push((c.op.tiles_fwd, c.op.tiles_adj));
    assert!(c.compare("knot split lists", &list) >= 4);
}

/// balanced coil batches and the default shapes: an engine whose budget gives 7 coils per batch (30 coils: balanced
/// to 5 passes of 6) and a prime batch (13 coils, ragged default groups), against one coil per batch (the reference
/// sums the same products per coil: summation order only); static order 2 (single kernels), static order 3 (multi
/// kernels, ragged groups), 1-D knots
#[test]
fn balanced_batches_and_defaults_vs_one_coil_batches() {
    for (kind, nseg, order) in [(Kind::Static, 14usize, 2u32), (Kind::Static, 3, 3), (Kind::Knot1d, 14, 2)] {
        let mut refc = build(kind, 30, nseg, order, 3000, Some(1), 1 << 30);
        assert_eq!(refc.op.cb, 1);
        let (y0, x0) = refc.run();
        let per_coil = nseg * refc.op.plan.grid_elems() * 8;
        drop(refc);
        for (nc_budget, expect) in [(7usize, 6usize), (30, 15)] {
            let mut c = build(kind, 30, nseg, order, 3000, None, nc_budget * per_coil + 1);
            eprintln!("{kind:?} nseg {nseg} order {order}: budget {nc_budget} coils → batch {} (tiles fwd {:?}, adj {:?}, multi {:?})", c.op.cb, c.op.tiles_fwd, c.op.tiles_adj, c.op.multi_shapes(1));
            assert_eq!(c.op.cb, expect.min(16), "balanced batch");
            let (y, xa) = c.run();
            let (ef, ea) = (rel(&y, &y0), rel(&xa, &x0));
            let xh = c.xh();
            let e = (dot(&c.yh, &y) - dot(&xa, &xh)).norm() / (dot(&c.yh, &c.yh).re * dot(&y, &y).re).sqrt();
            eprintln!("    vs one-coil batches: forward {ef:.1e}, adjoint {ea:.1e}, adjointness {e:.1e}; multi {:?}", c.op.multi_shapes(1));
            assert!(ef < 3e-5 && ea < 6e-5 && e < 2e-6);
        }
        // prime batch: 13 coils in one batch (ragged default groups)
        let mut c = build(kind, 13, nseg, order, 3000, Some(13), 1 << 30);
        let mut r1 = build(kind, 13, nseg, order, 3000, Some(1), 1 << 30);
        let ((y, xa), (y0, x0)) = (c.run(), r1.run());
        eprintln!("{kind:?} 13 coils: tiles fwd {:?}, adj {:?}, multi {:?}: forward {:.1e}, adjoint {:.1e}", c.op.tiles_fwd, c.op.tiles_adj, c.op.multi_shapes(1), rel(&y, &y0), rel(&xa, &x0));
        assert!(rel(&y, &y0) < 3e-5 && rel(&xa, &x0) < 6e-5);
    }
}
