//! Knot-path layout (segnufft/kl.rs, research/sample_kernels.md §17; switch HASTY_SEG_KL / `SegNufft::set_kl`): the
//! knot-major adjoint launches no blocks for tiles without samples and zeroes, with `seg_zero_empty`, the ones among
//! them that the k-support pruned inverse FFT reads. Checked against the existing knot-major path (`set_kl(false)`):
//! * orders 0–3, 1-D (cubic) and 2-D (linear respiratory × cubic cardiac) knots, the tile adjoint and every ra launch
//!   shape (odd channel blocks included), coil batches with c0 > 0, sets with and without split tile lists
//!   (HASTY_SEG_MAXITEM = 64), under HASTY_SEG_ZERO = tiles and full (`zero_tiles`), with the fused pad / crop and
//!   the L2 slab FFTs on and off;
//! * a set whose empty tiles lie inside another set's k-support (the engine's support union): the zero kernel path;
//! * the whole-tile kl adjoint (segnufft_kl_adj.cu) at every launch shape against ra of the same shape;
//! * adjointness of the forward with the new adjoint; the autotune's choice; a control that the union test detects
//!   stale cells (KL_CONTROL = 1).
//! Samples lie in a ball of radius 0.62 π (the corners of the fine grid hold no samples, as the trajectory's k-space
//! ball in the real grid): about half the tiles are zero-count items.
//!   cargo test --release -p hasty-nufft --test seg_kl -- --nocapture --test-threads=1
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, hi_exponents, set_thread_cb_cap, SegNufft, SegNufftOpts, SegPoints, SegSegments, NHI};

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

/// coordinates (radians, ±π) of m samples in the ball |k| < r · π, or (half) in the half ball k_x > 0
fn ball(r: &mut R, m: usize, rad: f64, half: bool) -> [Vec<f32>; 3] {
    let mut c: [Vec<f32>; 3] = Default::default();
    while c[0].len() < m {
        let p = [r.u() * 2.0, r.u() * 2.0, r.u() * 2.0];
        if p.iter().map(|v| v * v).sum::<f64>() < 1.0 && (!half || p[0] > 0.0) {
            for a in 0..3 {
                c[a].push((p[a] * rad * std::f64::consts::PI) as f32);
            }
        }
    }
    c
}

/// (engine, segments, points) data of one case; `max_item` Some(n): HASTY_SEG_MAXITEM = n while the engine is built
#[allow(clippy::too_many_arguments)]
fn build(kind: Kind, nc: usize, nseg: usize, order: u32, width: usize, m: usize, cb_cap: usize, max_item: Option<usize>) -> Case {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [24usize, 22, 24];
    let nv: usize = dims.iter().product();
    let res = 0.004;
    let (mc, kr) = (6, 3);
    let mut r = R(9157 + nseg as u64 * 31 + order as u64 * 7 + nc as u64 * 3 + m as u64);
    let coords = ball(&mut r, m, 0.62, false);
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 1.2e4) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
    let nri = kr - 1;
    let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * nri as f64) as u32).min(nri as u32 - 1)).collect();
    let rv: Vec<f32> = (0..m).map(|_| (r.u() + 0.5) as f32).collect();
    let nimg = match kind {
        Kind::Knot1d => mc,
        Kind::Knot2d => mc * kr,
    };
    let xh: Vec<C32> = (0..nimg * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yh: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    if let Some(n) = max_item {
        std::env::set_var("HASTY_SEG_MAXITEM", n.to_string());
    }
    set_thread_cb_cap(Some(cb_cap));
    let opts = SegNufftOpts { sigma: 1.25, width, nseg, order, frames: mc, bspline: 3, max_grid_bytes: 1 << 30 };
    let mut op = match kind {
        Kind::Knot1d => SegNufft::new_temporal(&dev, dims, [res; 3], nc, opts, true).unwrap(),
        Kind::Knot2d => SegNufft::new_temporal2d(&dev, dims, [res; 3], nc, opts, kr, 1).unwrap(),
    };
    set_thread_cb_cap(None);
    op.set_maps(&st, &smaps).unwrap();
    let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
    let c = [&coords[0][..], &coords[1][..], &coords[2][..]];
    let hi = (order >= 3).then(|| {
        let u: Vec<f32> = (0..4 * m).map(|_| r.u() as f32).collect();
        let g: [f64; 3] = [0, 1, 2].map(|a| res * op.plan.nf[a] as f64 / (std::f64::consts::PI * width as f64));
        let ex = hi_exponents();
        let h: Vec<f64> = (0..4 * NHI).map(|i| 0.2 * r.u() / (0..3).map(|a| g[a].powi(ex[i % NHI][a] as i32)).product::<f64>()).collect();
        (u, h)
    });
    let pts = match (kind, &hi) {
        (Kind::Knot1d, None) => op.points(&st, c, &tau, &phi0, &dcv, Some(&t)).unwrap(),
        (Kind::Knot1d, Some((u, h))) => op.points_hi(&st, c, &tau, &phi0, &dcv, u, h, Some(&t)).unwrap(),
        (Kind::Knot2d, None) => op.points2d(&st, c, &tau, &phi0, &dcv, &t, &ri, &rv).unwrap(),
        (Kind::Knot2d, Some((u, h))) => op.points2d_hi(&st, c, &tau, &phi0, &dcv, &t, &ri, &rv, u, h).unwrap(),
    };
    if let Some(n) = max_item {
        std::env::remove_var("HASTY_SEG_MAXITEM");
        assert!(n >= 1024 || pts.has_split(), "split tile lists expected");
    }
    let (x, yd) = (st.clone_htod(&xh).unwrap(), st.clone_htod(&yh).unwrap());
    op.set_vs(None).unwrap();
    op.set_ra(None).unwrap();
    op.set_kl(false);
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
    fn adjness(&self, y: &[C32], xa: &[C32]) -> f64 {
        (dot(&self.yh, y) - dot(xa, &self.xh)).norm() / (dot(&self.yh, &self.yh).re * dot(y, y).re).sqrt()
    }
}

/// the knot adjoint with the layout on against off, for the tile adjoint and every ra shape, ZERO tiles / full;
/// adjointness of the forward with the new adjoint. Returns the number of (kernel, zero mode) pairs checked.
fn check(c: &mut Case, tag: &str) -> usize {
    let y0 = c.fwd();
    // f32 floor: the same path run twice (atomics order of the split tiles and of the fused crop)
    c.op.set_kl(false);
    c.op.set_ra(None).unwrap();
    let xa0 = c.adj();
    let floor = rel(&c.adj(), &xa0);
    let j0 = c.adjness(&y0, &xa0);
    let mut cfgs: Vec<Option<hasty_nufft::RaCfg>> = vec![None];
    cfgs.extend(c.op.ra_candidates().into_iter().take(4).map(Some));
    for (nxg, cb, chk) in [(2usize, 3usize, 16usize), (4, 5, 32), (1, 1, 8)] {
        let r = hasty_nufft::RaCfg { nxg, cb, chk };
        if !cfgs.contains(&Some(r)) {
            cfgs.push(Some(r));
        }
    }
    let bound = 2e-6f64.max(4.0 * floor);
    // the kl kernel sums every tile in one block (ra sums the parts of a split tile separately and adds them
    // atomically): against ra of the same shape, bound by the conditioning (the plain adjoint of the data moved by one
    // ulp per element) as tests/seg_ra_vt.rs
    let mut rp = R(99);
    let yp: Vec<C32> = c.yh.iter().map(|v| { let f = 1.0 + if rp.u() < 0.0 { -1.0 } else { 1.0 } * f32::EPSILON * 0.5; C32::new(v.re * f, v.im * f) }).collect();
    let ydp = c.st.clone_htod(&yp).unwrap();
    let ydk = std::mem::replace(&mut c.yd, ydp);
    let xap = c.adj();
    c.yd = ydk;
    let floor_p = rel(&xap, &xa0);
    // 10 one-ulp floors: split tiles are summed in one sequence instead of per part (the grouping alone; that the kl
    // sum is exactly the unsplit one is kl_split_equals_unsplit_ra)
    let bound_k = 3e-5f64.max(10.0 * floor_p).max(4.0 * floor);
    let mut n = 0;
    for cfg in cfgs {
        if c.op.set_ra(cfg).is_err() {
            continue;
        }
        for zt in [true, false] {
            c.op.zero_tiles = zt;
            c.op.set_kl(false);
            let xr = c.adj();
            c.op.set_kl(true);
            assert!(c.op.kl_active(), "{tag}: the knot-path layout applies");
            let xk = c.adj();
            let (e, j) = (rel(&xk, &xr), c.adjness(&y0, &xk));
            eprintln!("{tag} split {} {cfg:?} zero {}: layout on vs off {e:.1e} (bound {bound:.1e}, run-to-run {floor:.1e}), adjointness {j:.1e} (off {j0:.1e})", c.pts.has_split(), if zt { "tiles" } else { "full" });
            assert!(e < bound && j < 2e-6f64.max(2.0 * j0), "{tag} {cfg:?} zero_tiles {zt}: layout on vs off {e:.1e} (bound {bound:.1e}), adjointness {j:.1e}");
            n += 1;
        }
        c.op.zero_tiles = true;
    }
    // the whole-tile kl kernel at every candidate shape plus odd / extra ones, against ra of the same shape (layout off)
    let mut kcfgs = c.op.kl_candidates();
    for (nxg, cb, chk) in [(2usize, 3usize, 16usize), (4, 5, 32), (1, 1, 8), (2, 7, 16)] {
        let r = hasty_nufft::RaCfg { nxg, cb, chk };
        if !kcfgs.contains(&r) {
            kcfgs.push(r);
        }
    }
    for cfg in kcfgs {
        c.op.set_kl(false);
        if c.op.set_ra(Some(cfg)).is_err() {
            continue;
        }
        for zt in [true, false] {
            c.op.zero_tiles = zt;
            c.op.set_kl(false);
            let xr = c.adj();
            c.op.set_kl(true);
            if c.op.set_kl_kernel(Some(cfg)).is_err() {
                continue;
            }
            assert_eq!(c.op.kl_kernel(), Some(cfg));
            let xk = c.adj();
            let (e, e0, j) = (rel(&xk, &xr), rel(&xk, &xa0), c.adjness(&y0, &xk));
            eprintln!("{tag} split {} kl kernel {cfg:?} zero {}: vs ra (layout off) {e:.1e} (bound {bound_k:.1e}, one-ulp data {floor_p:.1e}), vs tile adjoint {e0:.1e}, adjointness {j:.1e} (off {j0:.1e})", c.pts.has_split(), if zt { "tiles" } else { "full" });
            assert!(e < bound_k && j < 5e-6f64.max(2.0 * j0), "{tag} kl kernel {cfg:?} zero_tiles {zt}: vs ra {e:.1e} (bound {bound_k:.1e}), adjointness {j:.1e}");
            n += 1;
        }
        c.op.zero_tiles = true;
    }
    c.op.set_ra(None).unwrap();
    c.op.set_kl(false);
    n
}

#[test]
fn kl_knot_orders() {
    for order in [0u32, 1, 2, 3] {
        // 1-D: 8 coils × 4 segments in one batch, W 8
        let mut c = build(Kind::Knot1d, 8, 4, order, 8, 8_000, 8, None);
        assert!(check(&mut c, &format!("knot1d o{order} 8x4")) >= 6);
        // 1-D: 6 coils in batches of 3 (c0 > 0), 5 segments (odd channel blocks: 15 channels), W 10
        let mut c = build(Kind::Knot1d, 6, 5, order, 10, 12_000, 3, None);
        assert!(check(&mut c, &format!("knot1d o{order} 6(3)x5")) >= 6);
        // 2-D: 4 coils × 3 segments, W 10
        let mut c = build(Kind::Knot2d, 4, 3, order, 10, 10_000, 4, None);
        assert!(check(&mut c, &format!("knot2d o{order} 4x3")) >= 6);
    }
}

#[test]
fn kl_knot_split_lists() {
    // HASTY_SEG_MAXITEM = 64: every dense tile split (zeroed split tiles, atomics) next to zero-count items
    for order in [0u32, 2, 3] {
        let mut c = build(Kind::Knot1d, 6, 5, order, 10, 12_000, 3, Some(64));
        assert!(check(&mut c, &format!("knot1d o{order} 6(3)x5 split")) >= 6);
        let mut c = build(Kind::Knot2d, 5, 3, order, 8, 10_000, 5, Some(64));
        assert!(check(&mut c, &format!("knot2d o{order} 5x3 split")) >= 6);
    }
}

/// fused pad / crop and L2 slabs on / off: layout on equals off in every combination
#[test]
fn kl_fused_and_slabs() {
    for (kind, order) in [(Kind::Knot1d, 2u32), (Kind::Knot2d, 3)] {
        let mut c = build(kind, 6, 4, order, 10, 9_000, 6, None);
        let y0 = c.fwd();
        let rc = c.op.ra_candidates()[0];
        for (fuse, slab) in [(true, None), (true, Some(0)), (false, None), (false, Some(0)), (true, Some(2))] {
            c.op.fuse_padcrop = fuse;
            c.op.set_fft_slab(slab);
            for ra in [None, Some(rc)] {
                c.op.set_ra(ra).unwrap();
                c.op.set_kl(false);
                let x0 = c.adj();
                c.op.set_kl(true);
                let x1 = c.adj();
                let (e, j) = (rel(&x1, &x0), c.adjness(&y0, &x1));
                eprintln!("{kind:?} o{order} fuse {fuse} slab {slab:?} ra {ra:?}: layout on vs off {e:.1e}, adjointness {j:.1e}");
                assert!(e < 1e-5 && j < 2e-6, "{kind:?} fuse {fuse} slab {slab:?} ra {ra:?}: {e:.1e}, adjointness {j:.1e}");
                if let Some(r) = ra {
                    c.op.set_kl_kernel(Some(r)).unwrap();
                    let x2 = c.adj();
                    c.op.set_kl_kernel(None).unwrap();
                    let (e2, j2) = (rel(&x2, &x0), c.adjness(&y0, &x2));
                    eprintln!("{kind:?} o{order} fuse {fuse} slab {slab:?} kl kernel {r:?}: vs ra {e2:.1e}, adjointness {j2:.1e}");
                    assert!(e2 < 1e-4 && j2 < 2e-6, "{kind:?} fuse {fuse} slab {slab:?} kl kernel: {e2:.1e}, adjointness {j2:.1e}");
                }
            }
        }
    }
}

/// a set (half ball k_x > 0) whose empty tiles lie in the support of an earlier set (full ball) on the same engine: the
/// pruned FFT reads them, `seg_zero_empty` must zero them (the slot holds the previous transform's data there)
#[test]
fn kl_support_union() {
    for (kind, order) in [(Kind::Knot1d, 1u32), (Kind::Knot2d, 3)] {
        let mut c = build(kind, 4, 3, order, 10, 9_000, 2, None);
        let st = c.st.clone();
        let m = 6_000;
        let mut r = R(31337);
        let co = ball(&mut r, m, 0.62, true);
        let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
        let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
        let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 1.2e4) as f32).collect();
        let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * 6.0) as f32).collect();
        let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * 2.0) as u32).min(1)).collect();
        let rv: Vec<f32> = (0..m).map(|_| (r.u() + 0.5) as f32).collect();
        let cc = [&co[0][..], &co[1][..], &co[2][..]];
        let pb = match (kind, order >= 3) {
            (Kind::Knot1d, _) => c.op.points(&st, cc, &tau, &phi0, &dcv, Some(&t)).unwrap(),
            (Kind::Knot2d, false) => c.op.points2d(&st, cc, &tau, &phi0, &dcv, &t, &ri, &rv).unwrap(),
            (Kind::Knot2d, true) => {
                let u = vec![0.1f32; 4 * m];
                let h = vec![0f64; 4 * NHI];
                c.op.points2d_hi(&st, cc, &tau, &phi0, &dcv, &t, &ri, &rv, &u, &h).unwrap()
            }
        };
        let yb: Vec<C32> = (0..c.nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
        let ybd = st.clone_htod(&yb).unwrap();
        let nv = c.op.nv();
        let run = |c: &mut Case, kl: bool, ra: Option<hasty_nufft::RaCfg>, kk: bool| -> Vec<C32> {
            c.op.set_ra(ra).unwrap();
            c.op.set_kl(kl);
            if kk {
                c.op.set_kl_kernel(ra).unwrap();
            }
            // the full-ball set first: its transforms leave data in the cells of the half-ball set's empty tiles
            let _ = c.adj();
            let mut xb = st.alloc_zeros::<C32>(c.nimg * nv).unwrap();
            c.op.adjoint(&st, &c.seg, &pb, &ybd, &mut xb).unwrap();
            st.clone_dtoh(&xb).unwrap()
        };
        let rc = c.op.ra_candidates()[0];
        // control (env KL_CONTROL = 1): without seg_zero_empty the result must be wrong, i.e. the test sees stale data
        let control = std::env::var("KL_CONTROL").is_ok_and(|v| v == "1");
        for (ra, kk) in [(None, false), (Some(rc), false), (Some(rc), true)] {
            for zt in [true, false] {
                c.op.zero_tiles = zt;
                let x0 = run(&mut c, false, ra, false);
                if control {
                    std::env::set_var("HASTY_SEG_KL_NOZERO", "1");
                }
                let x1 = run(&mut c, true, ra, kk);
                std::env::remove_var("HASTY_SEG_KL_NOZERO");
                let e = rel(&x1, &x0);
                eprintln!("{kind:?} o{order} ra {ra:?} kl kernel {kk} zero_tiles {zt}{}: half-ball set inside the union support, layout on vs off {e:.1e}", if control { " CONTROL (no zeroing)" } else { "" });
                if control {
                    // ZERO=full zeroes the whole slot: no stale data, the control cannot fail there
                    assert!(!zt || !(e <= 1e-3), "control: stale cells not detected ({e:.1e})");   // NaN: garbage too
                } else {
                    // the kl kernel sums split tiles in one block (ra: parts + atomics): conditioning-level differences
                    // (W 10 random data, as kl_knot_orders' one-ulp floor); exactness on unsplit tiles: kl_unsplit_equals_ra
                    assert!(e < if kk { 1e-3 } else { 3e-5 }, "{kind:?} ra {ra:?} kl kernel {kk} zero_tiles {zt}: {e:.1e}");
                }
            }
        }
    }
}

/// Without split tile lists (HASTY_SEG_MAXITEM = 16384: every tile one item) the kl kernel sums each tile's window in
/// the same order as ra of the same shape: equal to the run-to-run floor (the fused crop's atomics). With split lists
/// the two differ only in how the parts are grouped; that is checked against the conditioning in kl_knot_orders.
#[test]
fn kl_unsplit_equals_ra() {
    for (kind, order) in [(Kind::Knot1d, 0u32), (Kind::Knot1d, 2), (Kind::Knot1d, 3), (Kind::Knot2d, 1), (Kind::Knot2d, 3)] {
        std::env::set_var("HASTY_SEG_MAXITEM", "16384");
        let mut c = build(kind, 6, 5, order, 10, 9_000, 3, None);
        std::env::remove_var("HASTY_SEG_MAXITEM");
        assert!(!c.pts.has_split(), "no split lists expected");
        let mut n = 0;
        for cfg in c.op.kl_candidates().into_iter().take(4).chain([hasty_nufft::RaCfg { nxg: 2, cb: 3, chk: 16 }]) {
            if c.op.set_ra(Some(cfg)).is_err() {
                continue;
            }
            c.op.set_kl(false);
            let xr = c.adj();
            let floor = rel(&c.adj(), &xr);
            c.op.set_kl(true);
            c.op.set_kl_kernel(Some(cfg)).unwrap();
            let xk = c.adj();
            let e = rel(&xk, &xr);
            eprintln!("{kind:?} o{order} unsplit {cfg:?}: kl kernel vs ra {e:.1e} (run-to-run {floor:.1e})");
            assert!(e <= 4.0 * floor + 1e-7, "{kind:?} o{order} {cfg:?}: kl kernel vs ra {e:.1e} (run-to-run {floor:.1e})");
            c.op.set_kl(false);
            n += 1;
        }
        assert!(n >= 2);
    }
}

/// The kl kernel on split tile lists (HASTY_SEG_MAXITEM = 64) equals ra on unsplit lists of the same samples
/// (MAXITEM = 16384): both sum each tile's whole window in list order, so the parts of a split tile are handled exactly.
/// Two engines built from the same data differ already at f32 level (host-side table / segment builds); the bound is
/// that build-to-build floor, measured with a second unsplit engine.
#[test]
fn kl_split_equals_unsplit_ra() {
    for (kind, order) in [(Kind::Knot1d, 0u32), (Kind::Knot1d, 3), (Kind::Knot2d, 2), (Kind::Knot2d, 3)] {
        let mut cs = build(kind, 6, 5, order, 10, 9_000, 3, Some(64));
        let mut cu = build(kind, 6, 5, order, 10, 9_000, 3, Some(16384));
        let mut cu2 = build(kind, 6, 5, order, 10, 9_000, 3, Some(16384));
        assert!(cs.pts.has_split() && !cu.pts.has_split());
        for cfg in [hasty_nufft::RaCfg { nxg: 2, cb: 3, chk: 16 }, hasty_nufft::RaCfg { nxg: 1, cb: 1, chk: 16 }, hasty_nufft::RaCfg { nxg: 4, cb: 2, chk: 32 }] {
            if cu.op.set_ra(Some(cfg)).is_err() || cs.op.set_kl_kernel(Some(cfg)).is_err() {
                continue;
            }
            cs.op.set_kl(true);
            cs.op.set_kl_kernel(Some(cfg)).unwrap();
            let xr = cu.adj();
            let floor = rel(&cu.adj(), &xr);
            cu2.op.set_ra(Some(cfg)).unwrap();
            let floor_b = rel(&cu2.adj(), &xr);
            for zt in [true, false] {
                cs.op.zero_tiles = zt;
                let xk = cs.adj();
                let e = rel(&xk, &xr);
                eprintln!("{kind:?} o{order} {cfg:?} zero_tiles {zt}: kl kernel on split lists vs ra on unsplit lists {e:.1e} (build-to-build {floor_b:.1e}, run-to-run {floor:.1e})");
                assert!(e <= 2.0 * floor_b + 4.0 * floor + 1e-7, "{kind:?} o{order} {cfg:?}: {e:.1e} (build-to-build {floor_b:.1e})");
            }
        }
    }
}

/// the autotune keeps the layout only where faster (unless forced by env); the operator equals the plain one
#[test]
fn kl_autotune_choice() {
    for (kind, order) in [(Kind::Knot1d, 3u32), (Kind::Knot2d, 3)] {
        let mut c = build(kind, 10, 10, order, 10, 30_000, 5, None);
        let x0 = c.adj();
        c.op.autotune(&c.st, &c.seg, &c.pts).unwrap();
        let (ta, tk, tw) = c.op.kl_tune_times().expect("knot-path layout timed");
        eprintln!("{kind:?} o{order}: autotune knot-0 adjoint all items {ta:.3} ms, skip {tk:.3} ms, kl kernel {tw:.3} ms ({:?}) -> skip {} kernel {:?} (ra {:?})", c.op.kl_tune_cfg(), c.op.kl_active(), c.op.kl_kernel(), c.op.ra_active());
        if std::env::var("HASTY_SEG_KL").is_err() {
            let best = ta.min(tk).min(if tw.is_nan() { f32::INFINITY } else { tw });
            assert_eq!(c.op.kl_kernel().is_some(), tw == best, "kl kernel kept iff fastest");
            assert_eq!(c.op.kl_active(), tk == best || tw == best, "skip kept iff a layout path is fastest");
        }
        let x = c.adj();
        assert!(rel(&x, &x0) < 2e-4, "{kind:?}: adjoint {:.1e}", rel(&x, &x0));
    }
}

/// one small case for compute-sanitizer (env KL_SAN = 1; ignored without it): the layout on, tile and ra adjoints,
/// ZERO tiles / full, without and with split lists, the support-union zeroing
#[test]
fn kl_sanitizer_case() {
    if std::env::var("KL_SAN").is_err() {
        return;
    }
    for (kind, order, mi) in [(Kind::Knot1d, 0u32, None), (Kind::Knot2d, 3, None), (Kind::Knot1d, 2, Some(64)), (Kind::Knot2d, 1, Some(64))] {
        let mut c = build(kind, 4, 3, order, 10, 3_000, 2, mi);
        c.op.set_kl(true);
        let mut cfgs = vec![None];
        cfgs.extend(c.op.ra_candidates().into_iter().map(Some));
        for cfg in cfgs {
            if c.op.set_ra(cfg).is_ok() {
                for zt in [true, false] {
                    c.op.zero_tiles = zt;
                    let _ = c.adj();
                }
            }
        }
        for cfg in c.op.kl_candidates().into_iter().chain([hasty_nufft::RaCfg { nxg: 2, cb: 3, chk: 16 }, hasty_nufft::RaCfg { nxg: 1, cb: 1, chk: 8 }]) {
            if c.op.set_kl_kernel(Some(cfg)).is_ok() {
                for zt in [true, false] {
                    c.op.zero_tiles = zt;
                    let _ = c.adj();
                }
            }
        }
        c.op.set_kl_kernel(None).unwrap();
        eprintln!("sanitizer case {kind:?} o{order} done (split {})", c.pts.has_split());
    }
}

