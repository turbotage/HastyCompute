//! Phase-2 kernel variants (research/sample_kernels.md §19, research/arch_targets.md §4 items 2 / 3) against the
//! kernels they extend:
//! * ra staging variants (`RaVar`: list read in place, chunks 8 / 12 / 13 / 32, cp.async double buffer, launch-bounds
//!   hint) against the base ra of the same (nxg, cb) and against the tile adjoint (variant 0);
//! * kl staging variants (cp.async, chunks, hint) against the base kl kernel of the same (nxg, cb);
//! * vs / vt sub-chunk staging (`sch` < smax) against the one-pass vs / vt of the same launch shape and the tile forward;
//! * adjointness of every new forward / adjoint pair; orders 0–3; static, 1-D and 2-D knots; sets with and without split
//!   tile lists (HASTY_SEG_MAXITEM = 64 while the engine is built); HASTY_SEG_ZERO = tiles and full; odd channel blocks;
//! * spill screening (`ra_resources` on the known spilling 48 × 1 shape) and the autotune's choice.
//! Without split lists (HASTY_SEG_MAXITEM = 16384: no atomics) and for kl (no atomics) the variants must be bitwise
//! equal to their base: they change only staging, never the per-point summation order. With split lists (atomics in
//! a run-dependent order) the bound is the conditioning floor of the base (its adjoint of the data moved by one ulp,
//! × 6, as tests/seg_ra_vt.rs) and its run-to-run spread (× 4). Adjointness: no worse than the base kernel's.
//!   cargo test --release -p hasty-nufft --test seg_phase2 -- --nocapture --test-threads=1
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, hi_exponents, set_thread_cb_cap, RaCfg, RaVar, SegNufft, SegNufftOpts, SegPoints, SegSegments, VsCfg, VsGeom, VwCfg, NHI};

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

fn bitwise(a: &[C32], b: &[C32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits())
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
    x: cudarc::driver::CudaSlice<C32>,
    yd: cudarc::driver::CudaSlice<C32>,
    xh: Vec<C32>,
    yh: Vec<C32>,
    nimg: usize,
    m: usize,
    nc: usize,
}

/// one engine + set; samples in a ball of radius 0.62 π (about half the tiles zero-count, as the real grid); `max_item`
/// Some(n): HASTY_SEG_MAXITEM = n while the engine is built (64: split lists); `geom`: vs tables of that geometry
#[allow(clippy::too_many_arguments)]
fn build(kind: Kind, nc: usize, nseg: usize, order: u32, width: usize, m: usize, cb_cap: usize, max_item: Option<usize>, geom: Option<VsGeom>) -> Case {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [24usize, 22, 24];
    let nv: usize = dims.iter().product();
    let res = 0.004;
    let (mc, kr) = (6, 3);
    let mut r = R(77_001 + nseg as u64 * 31 + order as u64 * 7 + nc as u64 * 3 + m as u64);
    let mut coords: [Vec<f32>; 3] = Default::default();
    while coords[0].len() < m {
        let p = [r.u() * 2.0, r.u() * 2.0, r.u() * 2.0];
        if p.iter().map(|v| v * v).sum::<f64>() < 1.0 {
            for a in 0..3 {
                coords[a].push((p[a] * 0.62 * std::f64::consts::PI) as f32);
            }
        }
    }
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    // ΔC up to ≈ 0.1 in kernel units: strong e terms at orders ≥ 2
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
    if let Some(n) = max_item {
        std::env::set_var("HASTY_SEG_MAXITEM", n.to_string());
    }
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
    if let Some(n) = max_item {
        std::env::remove_var("HASTY_SEG_MAXITEM");
        assert!(n >= 1024 || pts.has_split(), "split tile lists expected");
    }
    if geom.is_some() {
        assert!(pts.vs_table_bytes() > 0, "vs tables built");
    }
    let (x, yd) = (st.clone_htod(&xh).unwrap(), st.clone_htod(&yh).unwrap());
    let base = op.tile_shape(1, 1, 2, 1, 1, 0).unwrap();
    op.set_tiles(base, base).unwrap();
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

/// the staging variants every ra / kl test runs: chunks 8, 12, 13 (odd), 32 with the register prefetch and the cp.async
/// double buffer, plus the launch-bounds hint 4 at chunk 12 / cp.async and on the base kernel
/// ra also runs the dynamic-staging variants (stages 3 / 4, research/sample_kernels.md §20) at the same chunks and with
/// the hint 4 at chunk 12 / 14, and the warp-specialised variants (stages 5 / 6: 4 / 2 producer warps, research/
/// sample_kernels.md §22) at the same chunks; env P2_RA_VARS = new: only stages 3 / 4 (ra), ws: only stages 5 / 6 (ra),
/// old: only the phase-2 ones.
fn variants(kl: bool) -> Vec<(usize, RaVar)> {
    let sel = std::env::var("P2_RA_VARS").unwrap_or_default();
    let mut v = Vec::new();
    if !kl && (sel.is_empty() || sel == "ws") {
        for chk in [8usize, 12, 13, 32] {
            for stages in [5usize, 6] {
                v.push((chk, RaVar { inplace: true, stages, lbmin: 0 }));
            }
        }
        v.push((14, RaVar { inplace: true, stages: 5, lbmin: 0 }));
        if sel == "ws" {
            return v;
        }
    }
    if kl || sel != "new" {
        for chk in [8usize, 12, 13, 32] {
            for stages in [1usize, 2] {
                v.push((chk, RaVar { inplace: !kl, stages, lbmin: 0 }));
            }
        }
        v.push((12, RaVar { inplace: !kl, stages: 2, lbmin: 4 }));
        v.push((16, RaVar { inplace: false, stages: 1, lbmin: 4 }));
    }
    if !kl && sel != "old" {
        for chk in [8usize, 12, 13, 32] {
            for stages in [3usize, 4] {
                v.push((chk, RaVar { inplace: true, stages, lbmin: 0 }));
            }
        }
        v.push((12, RaVar { inplace: true, stages: 4, lbmin: 4 }));
        v.push((14, RaVar { inplace: true, stages: 3, lbmin: 4 }));
    }
    v
}

/// (nxg, cb) pairs: the first two candidates plus odd channel blocks (cb 3 at nxg 2, cb 5 at nxg 4, cb 1 at nxg 1)
fn pairs(cands: &[RaCfg]) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = Vec::new();
    for c in cands.iter().take(2) {
        if !v.contains(&(c.nxg, c.cb)) {
            v.push((c.nxg, c.cb));
        }
    }
    for p in [(2usize, 3usize), (4, 5), (1, 1)] {
        if !v.contains(&p) {
            v.push(p);
        }
    }
    v
}

/// the adjoint of the data moved by one ulp per element against the adjoint (the f32 conditioning floor of the image)
fn ulp_floor(c: &mut Case, x0: &[C32]) -> f64 {
    let mut rp = R(99);
    let yp: Vec<C32> = c.yh.iter().map(|v| { let f = 1.0 + if rp.u() < 0.0 { -1.0 } else { 1.0 } * f32::EPSILON * 0.5; C32::new(v.re * f, v.im * f) }).collect();
    let ydp = c.st.clone_htod(&yp).unwrap();
    let ydk = std::mem::replace(&mut c.yd, ydp);
    let xp = c.adj();
    c.yd = ydk;
    rel(&xp, x0)
}

/// ra: every variant against the base ra of its (nxg, cb) (bitwise without split lists, else within the base's
/// conditioning floor / run-to-run spread) and the tile adjoint; adjointness against the tile forward. Returns the
/// variants run.
fn check_ra(c: &mut Case, tag: &str) -> usize {
    // the fused crop adds into the image atomically (static engines): unfused, an unsplit adjoint is deterministic
    let fuse = c.op.fuse_padcrop;
    c.op.fuse_padcrop = false;
    c.op.set_ra(None).unwrap();
    c.op.set_vs(None).unwrap();
    let y0 = c.fwd();
    let xt = c.adj();
    let jt = c.adjness(&y0, &xt);
    let split = c.pts.has_split();
    let mut n = 0;
    for (nxg, cb) in pairs(&c.op.ra_candidates()) {
        let base = RaCfg { nxg, cb, chk: 16 };
        if c.op.set_ra(Some(base)).is_err() {
            continue;
        }
        for zt in [true, false] {
            c.op.zero_tiles = zt;
            let (xb, xb2) = (c.adj(), c.adj());
            let spread = rel(&xb2, &xb);
            let et = rel(&xb, &xt);
            let jb = c.adjness(&y0, &xb);
            assert!(split || spread == 0.0, "{tag}: base ra not deterministic without split lists ({spread:.1e})");
            let floor = if split { ulp_floor(c, &xb) } else { 0.0 };
            for (chk, v) in variants(false) {
                let cfg = RaCfg { nxg, cb, chk };
                if c.op.set_ra_v(Some(cfg), v).is_err() {
                    continue;
                }
                let res = c.op.ra_active_v().unwrap().2;
                let xa = c.adj();
                let (e, j, same) = (rel(&xa, &xb), c.adjness(&y0, &xa), bitwise(&xa, &xb));
                let b = if split { (4.0 * spread).max(6.0 * floor).max(3e-5) } else { 0.0 };
                eprintln!(
                    "{tag} split {split} zero {} {cfg:?} {v:?} (regs {}, local {} B, spill {} B, {} warps): vs base ra {e:.1e}{} (bound {b:.1e}), base vs tile {et:.1e}, adjointness {j:.1e} (base ra {jb:.1e}, tile {jt:.1e})",
                    if zt { "tiles" } else { "full" },
                    res.regs,
                    res.local,
                    res.spill,
                    res.warps,
                    if same { " bitwise" } else { "" }
                );
                assert!(same || e <= b, "{tag} {cfg:?} {v:?} zero_tiles {zt}: variant vs base ra {e:.1e} (bound {b:.1e})");
                assert!(j <= 2e-6f64.max(2.0 * jt).max(1.5 * jb), "{tag} {cfg:?} {v:?}: adjointness {j:.1e} (base ra {jb:.1e})");
                n += 1;
            }
            c.op.set_ra(Some(base)).unwrap();
        }
        c.op.zero_tiles = true;
    }
    c.op.set_ra(None).unwrap();
    c.op.fuse_padcrop = fuse;
    n
}

/// kl: every variant against the base kl kernel of its (nxg, cb), bitwise (no atomics); adjointness
fn check_kl(c: &mut Case, tag: &str) -> usize {
    let fuse = c.op.fuse_padcrop;
    c.op.fuse_padcrop = false;
    c.op.set_ra(None).unwrap();
    c.op.set_kl(false);
    let y0 = c.fwd();
    let xt = c.adj();
    let jt = c.adjness(&y0, &xt);
    let mut n = 0;
    c.op.set_kl(true);
    for (nxg, cb) in pairs(&c.op.kl_candidates()) {
        let base = RaCfg { nxg, cb, chk: 16 };
        if c.op.set_kl_kernel(Some(base)).is_err() {
            continue;
        }
        for zt in [true, false] {
            c.op.zero_tiles = zt;
            c.op.set_kl_kernel(Some(base)).unwrap();
            let xb = c.adj();
            let et = rel(&xb, &xt);
            let jb = c.adjness(&y0, &xb);
            for (chk, v) in variants(true) {
                let cfg = RaCfg { nxg, cb, chk };
                if c.op.set_kl_kernel_v(Some(cfg), v).is_err() {
                    continue;
                }
                let res = c.op.kl_kernel_v().unwrap().2;
                let xa = c.adj();
                let (e, j, same) = (rel(&xa, &xb), c.adjness(&y0, &xa), bitwise(&xa, &xb));
                eprintln!(
                    "{tag} zero {} kl {cfg:?} {v:?} (regs {}, local {} B, spill {} B, {} warps): vs base kl {e:.1e}{}, base vs tile {et:.1e}, adjointness {j:.1e} (base kl {jb:.1e}, tile {jt:.1e})",
                    if zt { "tiles" } else { "full" },
                    res.regs,
                    res.local,
                    res.spill,
                    res.warps,
                    if same { " bitwise" } else { "" }
                );
                assert!(same, "{tag} kl {cfg:?} {v:?} zero_tiles {zt}: not bitwise equal to the base kl kernel ({e:.1e})");
                assert!(j <= 5e-6f64.max(2.0 * jt).max(1.5 * jb), "{tag} kl {cfg:?} {v:?}: adjointness {j:.1e} (base kl {jb:.1e})");
                n += 1;
            }
        }
        c.op.zero_tiles = true;
    }
    c.op.set_kl_kernel(None).unwrap();
    c.op.set_kl(false);
    c.op.fuse_padcrop = fuse;
    n
}

/// vs / vt: sub-chunks against the one-pass kernel of the same launch shape (bitwise) and the tile forward; adjointness
/// against the tile adjoint and against a ra cp.async variant (the new pair)
fn check_vs(c: &mut Case, tag: &str, cfgs: &[VsCfg]) -> usize {
    c.op.set_ra(None).unwrap();
    c.op.set_vs(None).unwrap();
    let yt = c.fwd();
    let xt = c.adj();
    let jt = c.adjness(&yt, &xt);
    // the new adjoint of the pair: ra in place + cp.async at chunk 12 of the first candidate (when ra runs here)
    let xr = c.op.ra_candidates().first().copied().and_then(|r| {
        c.op.set_ra_v(Some(RaCfg { chk: 12, ..r }), RaVar { inplace: true, stages: 2, lbmin: 0 }).ok()?;
        let v = c.adj();
        c.op.set_ra(None).unwrap();
        Some(v)
    });
    let smax = c.op.vs_geom().unwrap().smax;
    let mut n = 0;
    for &cfg in cfgs {
        if c.op.set_vs(Some(cfg)).is_err() {
            continue;
        }
        let y1 = c.fwd();
        let ef = rel(&y1, &yt);
        let j1 = c.adjness(&y1, &xt);
        let jp1 = xr.as_ref().map_or(f64::NAN, |xr| c.adjness(&y1, xr));
        for sch in [8usize, 13, smax / 2, smax - 1] {
            if sch >= smax || c.op.set_vs_v(Some(cfg), sch).is_err() {
                continue;
            }
            let res = c.op.vs_active_v().unwrap().3;
            let y = c.fwd();
            let (e, same, j) = (rel(&y, &y1), bitwise(&y, &y1), c.adjness(&y, &xt));
            let jp = xr.as_ref().map_or(f64::NAN, |xr| c.adjness(&y, xr));
            eprintln!(
                "{tag} {cfg:?} sch {sch} (regs {}, local {} B, spill {} B, {} warps): vs one-pass {e:.1e}{}, one-pass vs tile {ef:.1e}, adjointness vs tile adjoint {j:.1e}, vs ra cp.async {jp:.1e} (one-pass {j1:.1e} / {jp1:.1e}, tile pair {jt:.1e})",
                res.regs,
                res.local,
                res.spill,
                res.warps,
                if same { " bitwise" } else { "" }
            );
            assert!(same, "{tag} {cfg:?} sch {sch}: not bitwise equal to the one-pass kernel ({e:.1e})");
            let bj = 2e-6f64.max(2.0 * jt);
            assert!(j <= bj.max(1.5 * j1) && (jp.is_nan() || jp <= bj.max(1.5 * jp1)), "{tag} {cfg:?} sch {sch}: adjointness {j:.1e} / pair {jp:.1e} (one-pass {j1:.1e} / {jp1:.1e})");
            n += 1;
        }
    }
    c.op.set_vs(None).unwrap();
    n
}

#[test]
fn p2_ra_static() {
    ra_static(&[0, 1, 2]);
}

/// p2_ra_static / p2_ra_knot split by order (each part well under 15 min; run with --exact)
#[test]
fn p2_ra_static_o0() {
    ra_static(&[0]);
}
#[test]
fn p2_ra_static_o1() {
    ra_static(&[1]);
}
#[test]
fn p2_ra_static_o2() {
    ra_static(&[2]);
}
#[test]
fn p2_ra_knot_o01() {
    ra_knot(&[0, 1]);
}
#[test]
fn p2_ra_knot_o23() {
    ra_knot(&[2, 3]);
}

fn ra_static(ords: &[u32]) {
    for order in orders(ords) {
        for mi in mis() {
            // 10 coils × 10 segments in one batch
            let mut c = build(Kind::Static, 10, 10, order, 10, 9_000, 16, mi, None);
            assert!(check_ra(&mut c, &format!("static o{order} 10x10 mi{mi:?}")) >= 20);
            // 12 coils in batches of 4 (c0 > 0), 3 segments, W = 8
            let mut c = build(Kind::Static, 12, 3, order, 8, 6_000, 4, mi, None);
            assert!(check_ra(&mut c, &format!("static o{order} 12(4)x3 mi{mi:?}")) >= 20);
        }
    }
}

#[test]
fn p2_ra_knot() {
    ra_knot(&[0, 1, 2, 3]);
}

fn ra_knot(ords: &[u32]) {
    for order in orders(ords) {
        for mi in mis() {
            let mut c = build(Kind::Knot1d, 6, 5, order, 10, 8_000, 3, mi, None);
            assert!(check_ra(&mut c, &format!("knot1d o{order} 6(3)x5 mi{mi:?}")) >= 20);
            let mut c = build(Kind::Knot2d, 4, 3, order, 10, 8_000, 4, mi, None);
            assert!(check_ra(&mut c, &format!("knot2d o{order} 4x3 mi{mi:?}")) >= 20);
        }
    }
}

#[test]
fn p2_kl_knot() {
    for order in orders(&[0, 1, 2, 3]) {
        for mi in mis() {
            let mut c = build(Kind::Knot1d, 6, 5, order, 10, 8_000, 3, mi, None);
            assert!(check_kl(&mut c, &format!("knot1d o{order} 6(3)x5 mi{mi:?}")) >= 20);
            let mut c = build(Kind::Knot2d, 4, 3, order, 10, 8_000, 4, mi, None);
            assert!(check_kl(&mut c, &format!("knot2d o{order} 4x3 mi{mi:?}")) >= 20);
        }
    }
}

/// env P2_ORDERS = "0,2" / P2_MI = 16384 | 64: restrict a test to those orders / list splits (one process per part,
/// so each stays under the run timeout)
fn orders(all: &[u32]) -> Vec<u32> {
    match std::env::var("P2_ORDERS") {
        Ok(v) => all.iter().copied().filter(|o| v.split(',').any(|x| x.trim().parse::<u32>().ok() == Some(*o))).collect(),
        Err(_) => all.to_vec(),
    }
}
fn mis() -> Vec<Option<usize>> {
    match std::env::var("P2_MI").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(n) => vec![Some(n)],
        None => vec![Some(16384), Some(64)],
    }
}

const G16: VsGeom = VsGeom { tx: 4, ty: 4, tz: 16, smax: 48 };
const G8: VsGeom = VsGeom { tx: 2, ty: 4, tz: 8, smax: 32 };
const G64: VsGeom = VsGeom { tx: 4, ty: 4, tz: 16, smax: 64 };

#[test]
fn p2_vs_subchunks() {
    let shapes = |nw: &[usize]| -> Vec<VsCfg> { nw.iter().flat_map(|&nwc| [1usize, 2].map(|nws| VsCfg { nwc, nws, vxpad: 0 })).collect() };
    // the vs tables do not depend on the item length; it matters only for the ra adjoint of the pair. "Unsplit"
    // (16384) runs at the default (1024): the 10-coil tile kernels do not fit 16384-entry lists
    for mi in mis().into_iter().map(|m| m.filter(|&n| n < 16384)) {
        // vt: static order 2; knot 1-D / 2-D orders 2 / 3
        if !orders(&[2]).is_empty() {
            let mut c = build(Kind::Static, 10, 10, 2, 10, 20_000, 16, mi, Some(G16));
            assert!(check_vs(&mut c, &format!("vt static o2 10x10 mi{mi:?}"), &shapes(&[4, 7])) >= 4);
        }
        for order in orders(&[2, 3]) {
            let mut c = build(Kind::Knot1d, 6, 5, order, 10, 12_000, 3, mi, Some(G16));
            assert!(check_vs(&mut c, &format!("vt knot1d o{order} 6(3)x5 mi{mi:?}"), &shapes(&[1])) >= 4);
            let mut c = build(Kind::Knot2d, 10, 10, order, 10, 12_000, 5, mi, Some(G16));
            assert!(check_vs(&mut c, &format!("vt knot2d o{order} 10(5)x10 mi{mi:?}"), &shapes(&[2, 4])) >= 4);
        }
        // vs: orders 0 / 1, static and knot, with x padding too
        for order in orders(&[0, 1]) {
            let g = if order == 0 { G64 } else { G8 };
            let mut c = build(Kind::Static, 10, 10, order, 10, 20_000, 16, mi, Some(g));
            // 100 channels: nwc 7 (one block row) or 5 (two rows of 8 coils); x padding at both
            let mut cf = shapes(&[5, 7]);
            cf.push(VsCfg { nwc: 5, nws: 1, vxpad: 1 });
            cf.push(VsCfg { nwc: 7, nws: 1, vxpad: 1 });
            assert!(check_vs(&mut c, &format!("vs static o{order} 10x10 mi{mi:?}"), &cf) >= 4);
            let mut c = build(Kind::Knot2d, 4, 3, order, 10, 10_000, 4, mi, Some(g));
            assert!(check_vs(&mut c, &format!("vs knot2d o{order} 4x3 mi{mi:?}"), &shapes(&[1])) >= 2);
        }
    }
}

/// spill screening: the 48 coils × 1 segment static order-0 ra shape nxg 2, cb 3 spills on sm_89 / sm_90 (research/
/// arch_targets.md §5.2) and nxg 4, cb 3 does not; the screen reads ptxas, not the bare local size (every ra kernel has
/// the 32-byte sincosf frame)
#[test]
fn p2_spill_screen() {
    let mut c = build(Kind::Static, 48, 1, 0, 10, 4_000, 48, None, None);
    let arch = Device::new(0).unwrap().arch;
    let r23 = c.op.ra_resources(RaCfg { nxg: 2, cb: 3, chk: 16 }, RaVar::BASE).unwrap().unwrap();
    let r43 = c.op.ra_resources(RaCfg { nxg: 4, cb: 3, chk: 16 }, RaVar::BASE).unwrap().unwrap();
    eprintln!("{arch}: ra 48x1 nxg 2 cb 3: {r23:?}; nxg 4 cb 3: {r43:?}");
    assert!(r43.spill == 0 && r43.local > 0, "nxg 4 cb 3: no spill expected, a local frame (sincosf) present");
    if arch == "sm_89" || arch == "sm_90" {
        assert!(r23.spill > 0, "nxg 2 cb 3 spills on {arch}");
    }
    // the launch-bounds hint removes it (audit: 126 registers, no spill on sm_89)
    let r4 = c.op.ra_resources(RaCfg { nxg: 2, cb: 3, chk: 16 }, RaVar { lbmin: 4, ..RaVar::BASE }).unwrap().unwrap();
    eprintln!("{arch}: same with __launch_bounds__(128, 4): {r4:?}");
    // second query: memoised (no compile)
    let t = std::time::Instant::now();
    let again = c.op.ra_resources(RaCfg { nxg: 2, cb: 3, chk: 16 }, RaVar::BASE).unwrap().unwrap();
    eprintln!("repeat query {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    assert_eq!(again, r23);
    c.op.set_ra(None).unwrap();
}

/// the autotune (with the phase-2 candidates) keeps ra / kl / vs only where faster, and the chosen operator equals the
/// base within the tile pair's floor
#[test]
fn p2_autotune() {
    for (kind, order, geom) in [(Kind::Static, 2u32, Some(G16)), (Kind::Knot2d, 3, Some(G16)), (Kind::Knot1d, 3, Some(G16))] {
        let mut c = build(kind, 10, 10, order, 10, 30_000, 16, None, geom);
        let (y0, x0) = (c.fwd(), c.adj());
        c.op.autotune(&c.st, &c.seg, &c.pts).unwrap();
        eprintln!(
            "{kind:?} o{order}: ra {:?} (times {:?}), kl {:?} (times {:?}), vs {:?} (times {:?})",
            c.op.ra_active_v(),
            c.op.ra_tune_times(),
            c.op.kl_kernel_v(),
            c.op.kl_tune_times(),
            c.op.vs_active_v(),
            c.op.vs_tune_times()
        );
        let (y, x) = (c.fwd(), c.adj());
        assert!(rel(&y, &y0) < 2e-4 && rel(&x, &x0) < 2e-4, "{kind:?}: forward {:.1e}, adjoint {:.1e}", rel(&y, &y0), rel(&x, &x0));
    }
}

/// one small case per kernel family for compute-sanitizer (env P2_SAN = 1; returns at once without it): every new
/// variant once, ZERO tiles and full, split lists (P2_SAN_SPLIT = 1) or not
#[test]
fn p2_sanitizer_case() {
    if std::env::var("P2_SAN").is_err() {
        return;
    }
    let mi = Some(if std::env::var("P2_SAN_SPLIT").is_ok() { 64 } else { 16384 });
    // P2_SAN_KIND = 0..3: one of the four cases only (one sanitizer run each)
    let only: Option<usize> = std::env::var("P2_SAN_KIND").ok().and_then(|v| v.parse().ok());
    // P2_SAN_PAIRS = n: the first n (nxg, cb) shapes only (racecheck of the 2-D order-3 case at 3 shapes runs > 900 s)
    let npairs: usize = std::env::var("P2_SAN_PAIRS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    // P2_SAN_FAM = ra | kl | vs: one kernel family per process (racecheck's host memory grows with every loaded
    // module: all three families of the 1-D knot case exceeded a 24 GB cap)
    let fam = std::env::var("P2_SAN_FAM").ok();
    let want = |f: &str| fam.as_deref().is_none_or(|x| x == f);
    for (ci, (kind, order)) in [(Kind::Static, 0u32), (Kind::Static, 2), (Kind::Knot1d, 1), (Kind::Knot2d, 3)].into_iter().enumerate() {
        if only.is_some_and(|k| k != ci) {
            continue;
        }
        let mut c = build(kind, 4, 3, order, 10, 3_000, 4, mi, if order >= 2 || kind == Kind::Static { Some(G16) } else { None });
        let ps = pairs(&c.op.ra_candidates());
        for &(nxg, cb) in ps.iter().take(if want("ra") { npairs } else { 0 }) {
            for (chk, v) in variants(false) {
                if c.op.set_ra_v(Some(RaCfg { nxg, cb, chk }), v).is_ok() {
                    for zt in [true, false] {
                        c.op.zero_tiles = zt;
                        let _ = c.adj();
                    }
                }
            }
        }
        c.op.set_ra(None).unwrap();
        c.op.zero_tiles = true;
        if kind != Kind::Static && want("kl") {
            c.op.set_kl(true);
            for &(nxg, cb) in ps.iter().take(npairs) {
                for (chk, v) in variants(true) {
                    if c.op.set_kl_kernel_v(Some(RaCfg { nxg, cb, chk }), v).is_ok() {
                        for zt in [true, false] {
                            c.op.zero_tiles = zt;
                            let _ = c.adj();
                        }
                    }
                }
            }
            c.op.set_kl_kernel(None).unwrap();
            c.op.set_kl(false);
            c.op.zero_tiles = true;
        }
        if c.op.vs_geom().is_some() && want("vs") {
            for nwc in [1usize, 2] {
                for sch in [8usize, 13, 24, 47] {
                    if c.op.set_vs_v(Some(VsCfg { nwc, nws: 1, vxpad: 0 }), sch).is_ok() {
                        let _ = c.fwd();
                    }
                }
            }
            c.op.set_vs(None).unwrap();
        }
        eprintln!("sanitizer case {kind:?} o{order} done (split {})", c.pts.has_split());
    }
}

/// Tensor-core vt (VTC, research/sample_kernels.md §21: bf16 hi / lo split, hi·hi + hi·lo + lo·hi, f32 accumulation)
/// against the FMA vt of the same channel groups on the same tables: the error it adds must stay ≤ 1e-5 relative (at
/// or below the W 10 / σ 1.25 kernel's own truncation error); printed next to the FMA vt's one-ulp floor (x moved by one
/// ulp) and the vt-vs-tile difference. Adjointness against the tile adjoint no worse than 1e-5. Static order 2, knot
/// 1-D / 2-D orders 2 / 3, odd channel counts (the last 16-channel group partly filled).
///   cargo test --release -p hasty-nufft --test seg_phase2 p2_vtc -- --nocapture
#[test]
fn p2_vtc() {
    let cases: Vec<(Kind, usize, usize, u32, usize, usize)> = vec![
        (Kind::Static, 10, 10, 2, 20_000, 16),
        (Kind::Knot1d, 6, 5, 3, 12_000, 3),
        (Kind::Knot2d, 10, 10, 2, 12_000, 5),
        (Kind::Knot2d, 10, 10, 3, 12_000, 5),
    ];
    let mut n = 0;
    for (kind, nc, nseg, order, m, cb) in cases {
        if orders(&[order]).is_empty() {
            continue;
        }
        let mut c = build(kind, nc, nseg, order, 10, m, cb, None, Some(G16));
        let yt = c.fwd();
        let xt = c.adj();
        // one block row: all 16-channel groups of the batch (≤ 7 here)
        let nwc = (cb.min(nc) * nseg).div_ceil(16);
        assert!(nwc <= 8);
        let base = VsCfg { nwc, nws: 1, vxpad: 0 };
        c.op.set_vt_mma(0);
        c.op.set_vs(Some(base)).unwrap();
        let y0 = c.fwd();
        // one-ulp floor of the FMA vt: the image moved by one ulp per element (random signs)
        let mut rp = R(4242);
        let xp: Vec<C32> = c.xh.iter().map(|v| { let s = |r: &mut R| 1.0 + r.u().signum() as f32 * f32::EPSILON * 0.5; C32::new(v.re * s(&mut rp), v.im * s(&mut rp)) }).collect();
        let xd = std::mem::replace(&mut c.x, c.st.clone_htod(&xp).unwrap());
        let yu = c.fwd();
        c.x = xd;
        let (efl, evt) = (rel(&yu, &y0), rel(&y0, &yt));
        for (mk, nws, sch) in [(2usize, 1usize, 8usize), (2, 1, 16), (2, 1, 24), (2, 2, 8), (2, 2, 24), (1, 1, 16), (1, 2, 24)] {
            c.op.set_vt_mma(mk);
            let cfg = VsCfg { nwc, nws, vxpad: 0 };
            if let Err(e) = c.op.set_vs_v(Some(cfg), sch) {
                eprintln!("vtc {kind:?} o{order} mma{mk} {cfg:?} sch {sch}: not valid ({e})");
                continue;
            }
            let res = c.op.vs_active_v().unwrap().3;
            let y = c.fwd();
            let (e, j) = (rel(&y, &y0), c.adjness(&y, &xt));
            if std::env::var("P2_VTC_DEBUG").is_ok() {
                let a = dot(&y0, &y) / dot(&y0, &y0).re;
                let nm = |v: &C32| ((v.re as f64).powi(2) + (v.im as f64).powi(2)).sqrt();
                let df = |u: &C32, v: &C32| (((u.re - v.re) as f64).powi(2) + ((u.im - v.im) as f64).powi(2)).sqrt();
                let nz = y.iter().filter(|v| nm(v) > 0.0).count();
                let bad = y.iter().zip(&y0).filter(|(u, v)| df(u, v) > 1e-3 * nm(v).max(1e-6)).count();
                let i = y.iter().zip(&y0).position(|(u, v)| df(u, v) > 1e-3 * nm(v).max(1e-6)).unwrap_or(0);
                eprintln!("  debug: fit {a:.4}, nonzero {nz} / {}, bad {bad}, first bad {i}: {:?} vs {:?}", y.len(), y[i], y0[i]);
            }
            eprintln!(
                "vtc {kind:?} o{order} {nc}x{nseg} (batch {cb}) mma{mk} {cfg:?} sch {sch} (regs {}, spill {} B, {} warps): added error vs FMA vt {e:.2e}; FMA vt one-ulp floor {efl:.1e}, vt vs tile {evt:.1e}; adjointness vs tile adjoint {j:.1e}",
                res.regs, res.spill, res.warps
            );
            // bf16 (mma1, 16 bits per operand) is reported, not asserted: it misses the budget where the segments cancel
            // fp16 (mma2): ≤ 1e-5, or ≤ 2.5 × the f32 engine's own spread where these synthetic problems are worse
            // conditioned than that (strongly cancelling segments: one-ulp floor ≈ 1e-5)
            if mk == 2 {
                let bound = 1e-5f64.max(2.5 * efl.max(evt));
                assert!(e <= bound, "vtc {kind:?} o{order} {cfg:?} sch {sch}: added error {e:.2e} > {bound:.1e}");
                assert!(j <= 1e-5, "vtc {kind:?} o{order} {cfg:?} sch {sch}: adjointness {j:.1e}");
                n += 1;
            }
        }
        c.op.set_vt_mma(0);
        c.op.set_vs(None).unwrap();
    }
    assert!(n >= 8, "vtc cases run: {n}");
}

/// Warp-specialised vt (VW, research/sample_kernels.md §22: build warps write each (sample, plane) weight table once per
/// block, a copy warp stages the plane window by cp.async, consumers contract from shared memory; named-barrier hand-over)
/// against the FMA vt of the same channel groups on the same tables. Per sample the planes, rows and points are summed in
/// vt's order, so the result is expected bitwise equal; asserted ≤ 1e-6 relative (pure reordering budget) plus
/// adjointness against the tile adjoint ≤ 1e-5. Roles (consumer, build warps), window stages 1–3, sub-chunks 8 / 16 / 24
/// / smax (ring = sub-chunk, and one ring of sub-chunk + 1), odd channel counts (last slice partly filled).
///   cargo test --release -p hasty-nufft --test seg_phase2 p2_vw -- --nocapture
#[test]
fn p2_vw() {
    let cases: Vec<(Kind, usize, usize, u32, usize, usize)> = vec![
        (Kind::Static, 10, 10, 2, 20_000, 16),
        (Kind::Knot1d, 6, 5, 3, 12_000, 3),
        (Kind::Knot2d, 10, 10, 2, 12_000, 5),
        (Kind::Knot2d, 10, 10, 3, 12_000, 5),
    ];
    let mut n = 0;
    for (kind, nc, nseg, order, m, cb) in cases {
        if orders(&[order]).is_empty() {
            continue;
        }
        let mut c = build(kind, nc, nseg, order, 10, m, cb, None, Some(G16));
        let xt = c.adj();
        let nwc = (cb.min(nc) * nseg).div_ceil(16);
        let base = VsCfg { nwc, nws: 1, vxpad: 0 };
        c.op.set_vs(Some(base)).unwrap();
        let y0 = c.fwd();
        let shapes: Vec<(usize, usize, usize, usize, usize, usize)> = vec![
            // (consumers, builders, stages, sub-chunk, ring, copy warps)
            (10, 3, 2, 16, 16, 3),
            (8, 2, 2, 24, 24, 2),
            (6, 1, 2, 48, 48, 1),
            (6, 1, 1, 16, 16, 1),
            (4, 2, 3, 8, 8, 2),
            (5, 1, 2, 16, 17, 2),
            (1, 1, 2, 24, 24, 1),
            // VW 2 (shared build, register window): consumers 0 = the channel-group warps
            (0, 1, 0, 24, 24, 0),
            (0, 2, 0, 16, 16, 0),
            (0, 1, 0, 8, 8, 0),
            (0, 2, 0, 16, 21, 0),
            (0, 1, 0, 48, 48, 0),
        ];
        for (ncw, nbw, nst, sch, nwt, ncp) in shapes {
            let v = VwCfg { ncw, nbw, nst, nwt, ncp };
            if let Err(e) = c.op.set_vs_w(base, sch, v) {
                eprintln!("vw {kind:?} o{order} {v:?} sch {sch}: not valid ({e})");
                continue;
            }
            let res = c.op.vs_active_v().unwrap().3;
            let y = c.fwd();
            let y2 = c.fwd();
            let (e, j) = (rel(&y, &y0), c.adjness(&y, &xt));
            eprintln!(
                "vw {kind:?} o{order} {nc}x{nseg} (batch {cb}) {v:?} sch {sch} (regs {}, spill {} B, {} warps): vs FMA vt {e:.2e} (bitwise {}), rerun bitwise {}, adjointness vs tile adjoint {j:.1e}",
                res.regs, res.spill, res.warps, bitwise(&y, &y0), bitwise(&y, &y2)
            );
            assert!(e <= 1e-6, "vw {kind:?} o{order} {v:?} sch {sch}: vs FMA vt {e:.2e}");
            assert!(j <= 1e-5, "vw {kind:?} o{order} {v:?} sch {sch}: adjointness {j:.1e}");
            n += 1;
        }
        c.op.set_vs(None).unwrap();
    }
    assert!(n >= 24, "vw cases run: {n}");
}
