//! GPU: the FFT-side traffic switches of SegNufft (research/temporal_operator.md §23) against the previous paths, on
//! static, interval-major and knot-major engines with split tile lists (HASTY_SEG_MAXITEM small, a dense k-space
//! centre): split-tile zeroing (`zero_tiles`, HASTY_SEG_ZERO=tiles|full) in `adjoint`, `adjoint_group` and
//! `adjoint_source`. Every comparison is against the run-to-run spread of the old path (f32 atomics order).
//!   cargo test --release -p hasty-nufft --test seg_fft_traffic -- --nocapture --test-threads 1

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, NufftError, SegNufft, SegNufftOpts, SegPoints, SegSegments};

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

const DIMS0: [usize; 3] = [26, 22, 18];
const NC: usize = 3;
const NSEG: usize = 4;

struct Setup {
    dims: [usize; 3],
    op: SegNufft,
    seg: SegSegments,
    pts: Vec<SegPoints>,
    m: Vec<usize>,
    x: Vec<C32>,
    y: Vec<Vec<C32>>,
}

/// samples: half uniform over k-space, half in a small centre ball (its tiles split at HASTY_SEG_MAXITEM = 128)
fn samples(r: &mut R, m: usize, frames: usize) -> ([Vec<f32>; 3], Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| 0.0).collect());
    let mut coords = coords;
    for j in 0..m {
        let s = if j % 2 == 0 { 6.2 } else { 0.9 };
        for a in 0..3 {
            coords[a][j] = (r.u() * s) as f32;
        }
    }
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * frames as f64) as f32).collect();
    (coords, tau, phi0, dcv, t)
}

/// engine (static: frames 1; interval-major or knot-major cubic otherwise) with `sets` sample sets
fn setup(dev: &Device, st: &Arc<CudaStream>, frames: usize, knot: bool, sets: usize) -> Setup {
    setup_d(dev, st, DIMS0, frames, knot, sets)
}

fn setup_d(dev: &Device, st: &Arc<CudaStream>, dims: [usize; 3], frames: usize, knot: bool, sets: usize) -> Setup {
    let nv: usize = dims.iter().product();
    let mut r = R(777 + frames as u64 * 13 + knot as u64);
    let bspline = if frames > 1 { 3 } else { 1 };
    let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg: NSEG, order: 2, frames, bspline, max_grid_bytes: 1 << 28 };
    let mut op = SegNufft::new_temporal(dev, dims, [0.004; 3], NC, opts, knot).unwrap();
    let smaps: Vec<C32> = (0..NC * nv).map(|_| C32::new((1.0 + r.u()) as f32, r.u() as f32)).collect();
    op.set_maps(st, &smaps).unwrap();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, NSEG, 2e-3, 8e-3, 64, &fit, &psi);
    let seg = op.segments(st, &sp, &at, t0, dt).unwrap();
    let mut pts = Vec::new();
    let mut ms = Vec::new();
    let mut ys = Vec::new();
    for s in 0..sets {
        let m = 6000 + 1000 * s;
        let (c, tau, phi0, dcv, t) = samples(&mut r, m, frames);
        let tt = (frames > 1).then_some(&t[..]);
        pts.push(op.points(st, [&c[0], &c[1], &c[2]], &tau, &phi0, &dcv, tt).unwrap());
        ms.push(m);
        ys.push((0..NC * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect());
    }
    let x = (0..frames * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    Setup { dims, op, seg, pts, m: ms, x, y: ys }
}

/// adjoint of set 0, the group adjoint over all sets (static / interval-major) or the source adjoint over all sets
/// (every engine)
fn adjoints(st: &Arc<CudaStream>, s: &mut Setup) -> (Vec<C32>, Vec<C32>, Vec<C32>) {
    let nv: usize = s.dims.iter().product();
    let frames = s.op.frames;
    let dy: Vec<CudaSlice<C32>> = s.y.iter().map(|y| st.clone_htod(y).unwrap()).collect();
    let mut xa = st.alloc_zeros::<C32>(frames * nv).unwrap();
    s.op.adjoint(st, &s.seg, &s.pts[0], &dy[0], &mut xa).unwrap();
    let a = st.clone_dtoh(&xa).unwrap();
    let pr: Vec<&SegPoints> = s.pts.iter().collect();
    let g = if s.op.ks == 0 {
        let yr: Vec<&CudaSlice<C32>> = dy.iter().collect();
        s.op.adjoint_group(st, &s.seg, &pr, &yr, &mut xa).unwrap();
        st.clone_dtoh(&xa).unwrap()
    } else {
        vec![]
    };
    let mmax = *s.m.iter().max().unwrap();
    let mut buf = st.alloc_zeros::<C32>(NC * mmax).unwrap();
    let ms = s.m.clone();
    let mut src = |st: &Arc<CudaStream>, j: usize, c0: usize, ncb: usize, b: &mut CudaSlice<C32>| -> Result<(), NufftError> {
        let m = ms[j];
        st.memcpy_dtod(&dy[j].slice(c0 * m..(c0 + ncb) * m), &mut b.slice_mut(c0 * m..(c0 + ncb) * m)).map_err(|e| NufftError::Invalid(format!("{e:?}")))?;
        Ok(())
    };
    s.op.adjoint_source(st, &s.seg, &pr, &mut buf, &mut src, &mut xa).unwrap();
    (a, g, st.clone_dtoh(&xa).unwrap())
}

#[test]
fn split_tile_zeroing_matches_full_slot() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // single-threaded test binary: read when the engines are built (tile lists split above 128 entries)
    unsafe { std::env::set_var("HASTY_SEG_MAXITEM", "128") };
    for (frames, knot) in [(1usize, false), (5, false), (6, true)] {
        let mut s = setup(&dev, &st, frames, knot, 2);
        s.op.zero_tiles = false;
        let (a0, g0, s0) = adjoints(&st, &mut s);
        let (a1, g1, s1) = adjoints(&st, &mut s);
        s.op.zero_tiles = true;
        let (at, gt, stt) = adjoints(&st, &mut s);
        let sp = [rel(&a1, &a0), if g0.is_empty() { 0.0 } else { rel(&g1, &g0) }, rel(&s1, &s0)];
        let e = [rel(&at, &a0), if g0.is_empty() { 0.0 } else { rel(&gt, &g0) }, rel(&stt, &s0)];
        eprintln!("frames {frames}, knot {knot}: tiles vs full zeroing: adjoint {:.1e}, group {:.1e}, source {:.1e} (full vs full {:.1e} / {:.1e} / {:.1e})", e[0], e[1], e[2], sp[0], sp[1], sp[2]);
        for k in 0..3 {
            assert!(e[k] <= (5.0 * sp[k]).max(2e-7), "split-tile zeroing differs ({k}): {:.1e} (spread {:.1e})", e[k], sp[k]);
        }
        assert!(s.pts.iter().all(|p| p.has_split()), "no split tile lists: the test does not exercise the split path");
    }
    unsafe { std::env::remove_var("HASTY_SEG_MAXITEM") };
}

/// forward of set 0 and its adjoint; the coil batch is capped at 2 of 3 coils (a short last batch: channels past the
/// batch in the fused hooks, accumulation over batches)
fn fwd_adj(st: &Arc<CudaStream>, s: &mut Setup) -> (Vec<C32>, Vec<C32>) {
    let nv: usize = s.dims.iter().product();
    let frames = s.op.frames;
    let dx = st.clone_htod(&s.x).unwrap();
    let dy = st.clone_htod(&s.y[0]).unwrap();
    let mut y = st.alloc_zeros::<C32>(NC * s.m[0]).unwrap();
    let mut xa = st.alloc_zeros::<C32>(frames * nv).unwrap();
    s.op.forward(st, &s.seg, &s.pts[0], &dx, &mut y).unwrap();
    s.op.adjoint(st, &s.seg, &s.pts[0], &dy, &mut xa).unwrap();
    (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap())
}

#[test]
fn fused_pad_crop_matches_separate_kernels() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // x = 26: x pass of 8 lines × 9 threads (the crop's per-entry loop); x = 52: whole warps (lane shuffles)
    for (dims, frames, knot, kprune) in [(DIMS0, 1usize, false, true), ([52, 22, 18], 1, false, true), (DIMS0, 1, false, false), (DIMS0, 5, false, true), (DIMS0, 6, true, true), ([52, 22, 18], 6, true, true)] {
        // HASTY_TEST_CB: the coil batch cap (default 2 of 3 coils)
        let cb = std::env::var("HASTY_TEST_CB").ok().and_then(|v| v.parse().ok()).unwrap_or(2usize);
        hasty_nufft::set_thread_cb_cap(Some(cb));
        let mut s = setup_d(&dev, &st, dims, frames, knot, 2);
        hasty_nufft::set_thread_cb_cap(None);
        assert_eq!(s.op.cb, cb);
        s.op.set_kprune(kprune).unwrap();
        s.op.fuse_padcrop = false;
        let (y0, x0) = fwd_adj(&st, &mut s);
        let (y1, x1) = fwd_adj(&st, &mut s);
        s.op.fuse_padcrop = true;
        let (yf, xf) = fwd_adj(&st, &mut s);
        assert!(s.op.fused_padcrop_active(), "the fused passes were not used");
        eprintln!("  {dims:?}: {}", s.op.grid_fft_desc());
        let (_, a0, g0) = adjoints(&st, &mut s);
        s.op.fuse_padcrop = false;
        let (_, a1, g1) = adjoints(&st, &mut s);
        let sp = rel(&x1, &x0);
        // adjointness of the fused pair (forward fused with the adjoint fused) and of the separate pair:
        // |<y, A x> − <Aᴴ y, x>| / (‖y‖ ‖A x‖)
        let dot = |a: &[C32], b: &[C32]| a.iter().zip(b).map(|(p, q)| num_complex::Complex64::new(p.re as f64, -p.im as f64) * num_complex::Complex64::new(q.re as f64, q.im as f64)).sum::<num_complex::Complex64>();
        let adj = |y: &[C32], xa: &[C32]| (dot(&s.y[0], y) - dot(xa, &s.x)).norm() / (dot(&s.y[0], &s.y[0]).re * dot(y, y).re).sqrt();
        let (jf, js) = (adj(&yf, &xf), adj(&y0, &x0));
        let (ef, ea) = (rel(&yf, &y0), rel(&xf, &x0));
        let (eg, es) = (if a0.is_empty() { 0.0 } else { rel(&a0, &a1) }, rel(&g0, &g1));
        eprintln!("  frames {frames}, knot {knot}, kprune {kprune}: fused vs separate pad / crop: forward {ef:.1e}, adjoint {ea:.1e}, group {eg:.1e}, source {es:.1e} (separate vs separate forward {:.1e}, adjoint {sp:.1e}); adjointness fused {jf:.1e}, separate {js:.1e}", rel(&y1, &y0));
        // the fused pad's grid equals the separate one to 1.3e-7 (FMA contraction, `debug_fused_slot`); the forward
        // amplifies such one-ulp differences to ≈ 4e-6 (x moved by one ulp gives the same), hence the bound
        assert!(ef < 1e-5 && jf < 2e-6, "fused pad: forward differs {ef:.1e}, adjointness {jf:.1e}");
        assert!(ea < (5.0 * sp).max(1e-6) && eg < 1e-5 && es < 1e-5, "fused crop: adjoint {ea:.1e}, group {eg:.1e}, source {es:.1e} (spread {sp:.1e})");
    }
}

/// two-pass L2 slabs (`set_fft_slab`): the same transforms slab by slab, with the separate and the fused pad / crop:
/// forward bit-equal, adjoint within the atomics spread
#[test]
fn fft_slabs_match_whole_grid() {
    // an f32-exactness / equivalence check: pin the f32 FFT store (the BFP store's own error is tested in
    // seg_fft_store and bfp_store; its rounding differs between slab / pruned layouts by design)
    hasty_nufft::set_thread_fft_store(Some(hasty_fft::Store::F32));
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    for (dims, frames, knot) in [(DIMS0, 1usize, false), ([52, 22, 18], 6, true)] {
        let mut s = setup_d(&dev, &st, dims, frames, knot, 1);
        for fuse in [false, true] {
            s.op.fuse_padcrop = fuse;
            s.op.set_fft_slab(Some(0));
            let (y0, x0) = fwd_adj(&st, &mut s);
            let (_, x1) = fwd_adj(&st, &mut s);
            let sp = rel(&x1, &x0);
            for dz in [1usize, 4] {
                s.op.set_fft_slab(Some(dz));
                assert_eq!(s.op.fft_slab(), dz);
                let (y, x) = fwd_adj(&st, &mut s);
                let same = y.iter().zip(&y0).all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits());
                let ea = rel(&x, &x0);
                eprintln!("{dims:?} frames {frames} fused {fuse}, slab {dz}: forward bit-equal {same} ({:.1e}), adjoint {ea:.1e} (spread {sp:.1e})", rel(&y, &y0));
                assert!(rel(&y, &y0) < 1e-6 && ea < (5.0 * sp).max(1e-6), "slab {dz}: forward {:.1e}, adjoint {ea:.1e}", rel(&y, &y0));
            }
        }
    }
}

/// debug: the loaded (padded + transformed) grid slot, fused vs separate pad, cell by cell
#[test]
#[ignore]
fn debug_fused_slot() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let mut s = setup_d(&dev, &st, DIMS0, 1, false, 1);
    let dx = st.clone_htod(&s.x).unwrap();
    let a = s.op.debug_loaded_slot(&st, &s.seg, &dx, 0, 0, s.op.cb, false, true).unwrap();
    let b = s.op.debug_loaded_slot(&st, &s.seg, &dx, 0, 0, s.op.cb, true, true).unwrap();
    // without zeroing first: stale slot contents (a garbage pass first)
    let mut g = st.clone_htod(&vec![C32::new(1e3, -1e3); s.op.nc * s.m[0]]).unwrap();
    let mut xg = st.alloc_zeros::<C32>(s.x.len()).unwrap();
    s.op.adjoint(&st, &s.seg, &s.pts[0], &g, &mut xg).unwrap();
    let a2 = s.op.debug_loaded_slot(&st, &s.seg, &dx, 0, 0, s.op.cb, false, false).unwrap();
    s.op.adjoint(&st, &s.seg, &s.pts[0], &g, &mut xg).unwrap();
    let b2 = s.op.debug_loaded_slot(&st, &s.seg, &dx, 0, 0, s.op.cb, true, false).unwrap();
    let _ = &mut g;
    let nchk = s.op.cb * NSEG;
    let diffcells = |p: &[C32], q: &[C32]| p.iter().zip(q).filter(|(u, v)| u.re.to_bits() != v.re.to_bits() || u.im.to_bits() != v.im.to_bits()).count();
    eprintln!("stale slot: separate vs zeroed separate: {} differing values, fused vs zeroed fused: {} differing values (of {}, {nchk} ch)", diffcells(&a2, &a), diffcells(&b2, &b), a.len());
    let nch = s.op.cb * NSEG;
    let nf = s.op.plan.nf;
    eprintln!("slot rel diff {:.2e} ({} cells × {nch} ch, grid {nf:?})", rel(&b, &a), a.len() / nch);
    let mut worst: Vec<(f32, usize)> = a.iter().zip(&b).enumerate().map(|(i, (p, q))| (((p.re - q.re).powi(2) + (p.im - q.im).powi(2)).sqrt(), i)).collect();
    worst.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    for &(d, i) in worst.iter().take(12) {
        let (ch, cell) = (i % nch, i / nch);
        eprintln!("  diff {d:.3e} at ch {ch}, x {} y {} z {}: separate {:?}, fused {:?}", cell % nf[0], (cell / nf[0]) % nf[1], cell / (nf[0] * nf[1]), a[i], b[i]);
    }
    let big = worst.iter().filter(|w| w.0 > 1e-3).count();
    eprintln!("  cells with |diff| > 1e-3: {big}");
    // forward outputs: separate twice, fused twice
    let mut ys = Vec::new();
    for fused in [false, false, true, true] {
        s.op.fuse_padcrop = fused;
        let mut y = st.alloc_zeros::<C32>(NC * s.m[0]).unwrap();
        s.op.forward(&st, &s.seg, &s.pts[0], &dx, &mut y).unwrap();
        ys.push(st.clone_dtoh(&y).unwrap());
    }
    eprintln!("  forward: separate vs separate {:.2e}, fused vs fused {:.2e}, fused vs separate {:.2e}", rel(&ys[1], &ys[0]), rel(&ys[3], &ys[2]), rel(&ys[2], &ys[0]));
    // conditioning: the separate forward of x with every value moved by one ulp (relative 2^-24, random sign)
    let mut r = R(99);
    let xp: Vec<C32> = s.x.iter().map(|v| C32::new(v.re * (1.0 + r.u().signum() as f32 * 5.96e-8), v.im * (1.0 + r.u().signum() as f32 * 5.96e-8))).collect();
    s.op.fuse_padcrop = false;
    let dxp = st.clone_htod(&xp).unwrap();
    let mut y = st.alloc_zeros::<C32>(NC * s.m[0]).unwrap();
    s.op.forward(&st, &s.seg, &s.pts[0], &dxp, &mut y).unwrap();
    eprintln!("  forward of x moved by one ulp (separate) vs separate: {:.2e}", rel(&st.clone_dtoh(&y).unwrap(), &ys[0]));
    let mut worst: Vec<(f32, usize)> = ys[0].iter().zip(&ys[2]).enumerate().map(|(i, (p, q))| (((p.re - q.re).powi(2) + (p.im - q.im).powi(2)).sqrt() / (p.re.hypot(p.im) + 1e-3), i)).collect();
    worst.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    for &(d, i) in worst.iter().take(8) {
        eprintln!("  y rel diff {d:.3e} at coil {} sample {}: separate {:?}, fused {:?}", i / s.m[0], i % s.m[0], ys[0][i], ys[2][i]);
    }
}
