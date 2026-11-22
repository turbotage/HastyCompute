//! GPU: the compressed-DRAM switches of research/bfp_storage.md §10 against the f32 engine on the same problem:
//! coil maps and segment diagonals stored as BFP and decoded in the fused pad / crop staging (HASTY_SEG_MAPS_STORE,
//! HASTY_SEG_DIAG_STORE, `set_thread_map_stores`). Checks: the device copies equal the host codec bit for bit; forward,
//! adjoint and gradient within the predicted error (each map buffer ≈ 1–1.6 · 2^−b, through the problem's own
//! one-ulp sensitivity); adjointness of the BFP pair; the switch actually changes the result (bfp16 above the f32
//! floor); static and 2-D knot engines, orders 0 and 3, slabs on and off, k-support pruning on and off.
//! Run also with HASTY_SEG_MAXITEM=64 (split tile lists) and HASTY_SEG_ZERO=full.
//!   cargo test --release -p hasty-nufft --test seg_bfp_dram -- --nocapture --test-threads 1

use std::sync::Arc;

use cudarc::driver::CudaStream;
use hasty_cuda::{Device, C32};
use hasty_fft::Store;
use hasty_nufft::{build_segments, hi_exponents, set_thread_fft_store, set_thread_grid_store, set_thread_map_stores, SegNufft, SegNufftOpts, SegPoints, SegSegments, NHI};

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

fn bit_equal(a: &[C32], b: &[C32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits())
}

const DIMS: [usize; 3] = [26, 22, 18];
const NC: usize = 3;
const NSEG: usize = 4;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Static,
    /// 2-D knots: M cardiac (cubic), kr respiratory bins
    Knot2d { mc: usize, kr: usize },
}

struct Prob {
    m: usize,
    k: [Vec<f32>; 3],
    tau: Vec<f32>,
    phi0: Vec<f32>,
    dc: Vec<f32>,
    t: Vec<f32>,
    ri: Vec<u32>,
    rv: Vec<f32>,
    u: Vec<f32>,
    h: Vec<f64>,
    smaps: Vec<C32>,
    x: Vec<C32>,
    y: Vec<C32>,
}

fn problem(kind: Kind, m: usize) -> Prob {
    let nv: usize = DIMS.iter().product();
    let mut r = R(4242 + m as u64);
    let (mc, kr) = match kind {
        Kind::Static => (1, 1),
        Kind::Knot2d { mc, kr } => (mc, kr),
    };
    // half the samples uniform over k-space, half in a small centre ball (split tile lists)
    let k: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|j| (r.u() * if j % 2 == 0 { 6.2 } else { 0.9 }) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dc: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
    let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * kr as f64) as u32).min(kr as u32 - 1)).collect();
    let rv: Vec<f32> = vec![0.0; m];
    let u: Vec<f32> = (0..4 * m).map(|_| (2.0 * r.u()) as f32).collect();
    let rmax = DIMS.iter().map(|&d| d as f64 * 4e-3 / 2.0).fold(0.0, f64::max);
    let ex = hi_exponents();
    let h: Vec<f64> = (0..4 * NHI).map(|i| 0.04 * r.u() / rmax.powi(ex[i % NHI].iter().sum::<u32>() as i32)).collect();
    let smaps: Vec<C32> = (0..NC * nv).map(|_| C32::new((1.0 + r.u()) as f32, r.u() as f32)).collect();
    let x: Vec<C32> = (0..mc * kr * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let y: Vec<C32> = (0..NC * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    Prob { m, k, tau, phi0, dc, t, ri, rv, u, h, smaps, x, y }
}

struct Eng {
    op: SegNufft,
    seg: SegSegments,
    pts: SegPoints,
    /// the segment diagonals uploaded (host copy)
    sp: Vec<C32>,
}

/// engine of `kind` at `order` with the grid-FFT store `store` (thread default while it is built and planned)
fn engine(dev: &Device, st: &Arc<CudaStream>, p: &Prob, kind: Kind, order: u32, store: Option<Store>, kprune: bool, maps: (Store, Store)) -> Eng {
    engine_g(dev, st, p, kind, order, store, kprune, maps, Store::F32)
}

#[allow(clippy::too_many_arguments)]
fn engine_g(dev: &Device, st: &Arc<CudaStream>, p: &Prob, kind: Kind, order: u32, store: Option<Store>, kprune: bool, maps: (Store, Store), grid: Store) -> Eng {
    set_thread_fft_store(store);
    set_thread_map_stores(Some(maps));
    set_thread_grid_store(Some(grid));
    let nv: usize = DIMS.iter().product();
    let (frames, bspline) = match kind {
        Kind::Static => (1, 1),
        Kind::Knot2d { mc, .. } => (mc, 3),
    };
    let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg: NSEG, order, frames, bspline, max_grid_bytes: 1 << 28 };
    let mut op = match kind {
        Kind::Static => SegNufft::new(dev, DIMS, [4e-3; 3], NC, opts).unwrap(),
        Kind::Knot2d { kr, .. } => SegNufft::new_temporal2d(dev, DIMS, [4e-3; 3], NC, opts, kr, 0).unwrap(),
    };
    op.set_maps(st, &p.smaps).unwrap();
    op.set_kprune(kprune).unwrap();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, NSEG, 2e-3, 8e-3, 64, &fit, &psi);
    let seg = op.segments(st, &sp, &at, t0, dt).unwrap();
    let kk = [&p.k[0][..], &p.k[1][..], &p.k[2][..]];
    let dc: &[f32] = if order > 0 { &p.dc } else { &[] };
    let pts = match (kind, order >= 3) {
        (Kind::Static, false) => op.points(st, kk, &p.tau, &p.phi0, dc, None).unwrap(),
        (Kind::Static, true) => op.points_hi(st, kk, &p.tau, &p.phi0, dc, &p.u, &p.h, None).unwrap(),
        (Kind::Knot2d { .. }, false) => op.points2d(st, kk, &p.tau, &p.phi0, dc, &p.t, &p.ri, &p.rv).unwrap(),
        (Kind::Knot2d { .. }, true) => op.points2d_hi(st, kk, &p.tau, &p.phi0, dc, &p.t, &p.ri, &p.rv, &p.u, &p.h).unwrap(),
    };
    set_thread_fft_store(None);
    set_thread_map_stores(None);
    set_thread_grid_store(None);
    Eng { op, seg, pts, sp }
}

/// (A x, Aᴴ y, Aᴴ(A x − y))
fn apply(st: &Arc<CudaStream>, e: &mut Eng, p: &Prob) -> (Vec<C32>, Vec<C32>, Vec<C32>) {
    let dx = st.clone_htod(&p.x).unwrap();
    let dy = st.clone_htod(&p.y).unwrap();
    let mut yo = st.alloc_zeros::<C32>(NC * p.m).unwrap();
    let mut xo = st.alloc_zeros::<C32>(p.x.len()).unwrap();
    let mut go = st.alloc_zeros::<C32>(p.x.len()).unwrap();
    let mut rr = st.alloc_zeros::<C32>(NC * p.m).unwrap();
    e.op.forward(st, &e.seg, &e.pts, &dx, &mut yo).unwrap();
    e.op.adjoint(st, &e.seg, &e.pts, &dy, &mut xo).unwrap();
    e.op.forward_residual(st, &e.seg, &e.pts, &dx, &dy, &mut rr).unwrap();
    e.op.adjoint(st, &e.seg, &e.pts, &rr, &mut go).unwrap();
    (st.clone_dtoh(&yo).unwrap(), st.clone_dtoh(&xo).unwrap(), st.clone_dtoh(&go).unwrap())
}

fn adjointness(p: &Prob, y: &[C32], xa: &[C32]) -> f64 {
    let dot = |a: &[C32], b: &[C32]| a.iter().zip(b).map(|(p, q)| num_complex::Complex64::new(p.re as f64, -p.im as f64) * num_complex::Complex64::new(q.re as f64, q.im as f64)).sum::<num_complex::Complex64>();
    (dot(&p.y, y) - dot(xa, &p.x)).norm() / (dot(&p.y, &p.y).re * dot(y, y).re).sqrt()
}


fn host_codec(x: &[C32], bits: u32) -> (Vec<u32>, Vec<u8>) {
    let f = hasty_bfp::Bfp::new(bits);
    let n = 2 * x.len();
    let mut m = vec![0u32; f.mant_words(n)];
    let mut e = vec![0u8; hasty_bfp::Bfp::blocks(n)];
    f.encode_c(x, &mut m, &mut e);
    (m, e)
}

#[test]
fn bfp_maps_match_f32() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // (kind, order, kprune, slab planes: None = default)
    let cases: Vec<(Kind, u32, bool, Option<usize>)> = vec![
        (Kind::Static, 0, true, None),
        (Kind::Static, 3, false, Some(0)),
        (Kind::Knot2d { mc: 6, kr: 3 }, 0, true, Some(0)),
        (Kind::Knot2d { mc: 6, kr: 3 }, 3, true, None),
    ];
    let sel: Option<Vec<usize>> = std::env::var("HASTY_BFP_TEST_CASES").ok().map(|v| v.split(',').map(|c| c.parse().unwrap()).collect());
    let f32s = (Store::F32, Store::F32);
    for (ci, (kind, order, kprune, slab)) in cases.into_iter().enumerate() {
        if sel.as_ref().is_some_and(|s| !s.contains(&ci)) {
            continue;
        }
        let p = problem(kind, 9000);
        let mut a = engine(&dev, &st, &p, kind, order, None, kprune, f32s);
        a.op.set_fft_slab(slab);
        assert_eq!(a.op.map_stores(), f32s);
        let (ya, xa, ga) = apply(&st, &mut a, &p);
        let (_, xa2, ga2) = apply(&st, &mut a, &p);
        let (sx, sg) = (rel(&xa2, &xa), rel(&ga2, &ga));
        // engine-to-engine spread of a second f32 engine (the adjoint's atomics order differs between engines)
        let mut a2 = engine(&dev, &st, &p, kind, order, None, kprune, f32s);
        a2.op.set_fft_slab(slab);
        let (_, xe, ge) = apply(&st, &mut a2, &p);
        let (sx, sg) = (sx.max(rel(&xe, &xa)), sg.max(rel(&ge, &ga)));
        drop(a2);
        // one-ulp sensitivity of the f32 operator (x, y moved by one ulp with random signs)
        let mut rr = R(99);
        let ulp = |v: &[C32], r: &mut R| -> Vec<C32> { v.iter().map(|z| C32::new(z.re * (1.0 + r.u().signum() as f32 * 5.96e-8), z.im * (1.0 + r.u().signum() as f32 * 5.96e-8))).collect() };
        let pu = Prob { x: ulp(&p.x, &mut rr), y: ulp(&p.y, &mut rr), ..problem(kind, 9000) };
        let (yu, xu, gu) = apply(&st, &mut a, &pu);
        let (efu, eau, egu) = (rel(&yu, &ya), rel(&xu, &xa), rel(&gu, &ga));
        let ja = adjointness(&p, &ya, &xa);
        for (bits, ms) in [(16u32, (Store::Bfp(16), Store::F32)), (16, (Store::F32, Store::Bfp(16))), (16, (Store::Bfp(16), Store::Bfp(16))), (24, (Store::Bfp(24), Store::Bfp(24)))] {
            let mut b = engine(&dev, &st, &p, kind, order, None, kprune, ms);
            b.op.set_fft_slab(slab);
            assert_eq!(b.op.map_stores(), ms);
            // the device copies against the host codec, bit for bit
            if let Store::Bfp(mb) = ms.0 {
                let d = b.op.debug_maps_bfp(&st).unwrap().expect("maps copy");
                assert!(d == host_codec(&p.smaps, mb), "{kind:?}: coil-map BFP copy differs from the host codec");
            }
            if let Store::Bfp(db) = ms.1 {
                let d = b.seg.debug_diag_bfp(&st).unwrap().expect("diagonal copy");
                assert!(d == host_codec(&b.sp, db), "{kind:?}: diagonal BFP copy differs from the host codec");
            }
            let (yb, xb, gb) = apply(&st, &mut b, &p);
            assert!(b.op.fused_padcrop_active(), "fused pad / crop not active");
            let q = (2.0f64).powi(-(bits as i32));
            let (ef, ea, eg) = (rel(&yb, &ya), rel(&xb, &xa), rel(&gb, &ga));
            let jb = adjointness(&p, &yb, &xb);
            let nbuf = (ms.0 != Store::F32) as u32 + (ms.1 != Store::F32) as u32;
            eprintln!(
                "{kind:?} order {order} kprune {kprune} slab {}: maps {} diag {}: forward {ef:.2e} ({:.2} · 2^-b), adjoint {ea:.2e} ({:.2}), gradient {eg:.2e} ({:.2}); one ulp {efu:.1e} / {eau:.1e} / {egu:.1e}; spread {sx:.1e} / {sg:.1e}; adjointness f32 {ja:.1e}, bfp {jb:.1e}",
                b.op.fft_slab(), ms.0, ms.1, ef / q, ea / q, eg / q
            );
            // a map buffer's own error is ≈ 1–1.6 · 2^-b per buffer (research/bfp_storage.md §3.2); the operator moves
            // the outputs by its one-ulp sensitivity per 2^-24 of input change (b = 16: 2^8 ulp-equivalents)
            let amp = (2.0f64).powi(24 - bits as i32) * 2.0 * nbuf as f64;
            assert!(ef < amp * efu.max(q / 4.0) + 1e-7, "forward {ef:.2e} over its bound ({:.2e})", amp * efu.max(q / 4.0));
            assert!(ea < amp * eau.max(q / 4.0) + 3.0 * sx && eg < amp * egu.max(q / 4.0) + 3.0 * sg, "adjoint {ea:.2e} / gradient {eg:.2e} over their bounds");
            // the switch is live: bfp16 moves the forward well above the f32 floor
            if bits == 16 {
                assert!(ef > 0.05 * q, "{kind:?}: bfp16 maps leave the forward unchanged ({ef:.1e}): BFP copies not read");
            }
            // pad and crop decode the same values: adjointness as f32
            assert!(jb < 3.0 * ja.max(1e-7) + 1e-7, "adjointness {jb:.1e} (f32 {ja:.1e})");
        }
    }
}

/// the BFP grid slot (in place: block k's mantissa words are the first words of cells 16 k .. 16 k + 16, exponent
/// byte k after the cells) against the host codec applied to the f32 engine's slot, block by block: (blocks written,
/// mismatching blocks), and the decoded slot
fn check_grid_slot(bits: u32, cells: usize, b: &[C32], a: &[C32]) -> (usize, usize, Vec<C32>) {
    let f = hasty_bfp::Bfp::new(bits);
    let words: &[u32] = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u32, 2 * b.len()) };
    let ex: &[u8] = unsafe { std::slice::from_raw_parts(b.as_ptr().add(cells) as *const u8, cells / 16) };
    let (mut nw, mut bad) = (0, 0);
    let mut dec = vec![C32::new(0.0, 0.0); cells];
    for k in 0..cells / 16 {
        if ex[k] == 0 {
            continue;
        }
        let mant = &words[32 * k..32 * k + bits as usize];
        let mut out = [0f32; 32];
        f.decode_block(mant, ex[k], &mut out);
        // blocks with all 16 lines active (no exact zero decoded: inactive lines of a k-support block store zeros,
        // where the f32 slot holds whatever the cells held) are compared bit for bit with the host codec
        if out.iter().all(|&v| v != 0.0) {
            nw += 1;
            let (hm, he) = host_codec(&a[16 * k..16 * k + 16], bits);
            if hm[..] != mant[..] || he[0] != ex[k] {
                bad += 1;
            }
        }
        for i in 0..16 {
            dec[16 * k + i] = C32::new(out[2 * i], out[2 * i + 1]);
        }
    }
    (nw, bad, dec)
}

#[test]
fn bfp_grid_matches_f32() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // (kind, order, kprune, slab planes, fused pad / crop)
    let cases: Vec<(Kind, u32, bool, Option<usize>, bool)> = vec![
        (Kind::Static, 0, true, None, true),
        (Kind::Static, 0, false, Some(0), false),
        (Kind::Static, 3, true, Some(0), true),
        (Kind::Knot2d { mc: 6, kr: 3 }, 0, true, None, false),
        (Kind::Knot2d { mc: 6, kr: 3 }, 3, true, None, true),
        (Kind::Knot2d { mc: 6, kr: 3 }, 3, false, Some(0), true),
    ];
    let sel: Option<Vec<usize>> = std::env::var("HASTY_BFP_TEST_CASES").ok().map(|v| v.split(',').map(|c| c.parse().unwrap()).collect());
    let f32s = (Store::F32, Store::F32);
    for (ci, (kind, order, kprune, slab, fuse)) in cases.into_iter().enumerate() {
        if sel.as_ref().is_some_and(|s| !s.contains(&ci)) {
            continue;
        }
        let p = problem(kind, 9000);
        let mut a = engine_g(&dev, &st, &p, kind, order, None, kprune, f32s, Store::F32);
        a.op.set_fft_slab(slab);
        a.op.fuse_padcrop = fuse;
        let (ya, xa, ga) = apply(&st, &mut a, &p);
        let (_, xa2, ga2) = apply(&st, &mut a, &p);
        let (sx, sg) = (rel(&xa2, &xa), rel(&ga2, &ga));
        let mut rr = R(99);
        let ulp = |v: &[C32], r: &mut R| -> Vec<C32> { v.iter().map(|z| C32::new(z.re * (1.0 + r.u().signum() as f32 * 5.96e-8), z.im * (1.0 + r.u().signum() as f32 * 5.96e-8))).collect() };
        let pu = Prob { x: ulp(&p.x, &mut rr), y: ulp(&p.y, &mut rr), ..problem(kind, 9000) };
        let (yu, _xu, gu) = apply(&st, &mut a, &pu);
        let (efu, egu) = (rel(&yu, &ya), rel(&gu, &ga));
        let dx = st.clone_htod(&p.x).unwrap();
        let cb = a.op.cb;
        let slot_a = a.op.debug_loaded_slot(&st, &a.seg, &dx, 0, 0, cb, fuse, true).unwrap();
        // κ: the forward's relative change per relative change of the loaded slot (one-ulp x), as seg_fft_store
        let dxu = st.clone_htod(&pu.x).unwrap();
        let slot_u = a.op.debug_loaded_slot(&st, &a.seg, &dxu, 0, 0, cb, fuse, true).unwrap();
        let kappa = efu / rel(&slot_u, &slot_a).max(1e-12);
        // engine-to-engine spread of a second f32 engine (the adjoint's atomics order differs between engines)
        let mut a2 = engine_g(&dev, &st, &p, kind, order, None, kprune, f32s, Store::F32);
        a2.op.set_fft_slab(slab);
        a2.op.fuse_padcrop = fuse;
        let (_, xe, ge) = apply(&st, &mut a2, &p);
        let (sx, sg) = (sx.max(rel(&xe, &xa)), sg.max(rel(&ge, &ga)));
        drop(a2);
        for bits in [16u32, 24] {
            let mut b = engine_g(&dev, &st, &p, kind, order, None, kprune, f32s, Store::Bfp(bits));
            b.op.set_fft_slab(slab);
            b.op.fuse_padcrop = fuse;
            assert_eq!(b.op.grid_store(), Store::Bfp(bits), "{kind:?}: grid store not active ({})", b.op.grid_fft_desc());
            let (yb, xb, gb) = apply(&st, &mut b, &p);
            let slot_b = b.op.debug_loaded_slot(&st, &b.seg, &dx, 0, 0, cb, fuse, true).unwrap();
            let cells = slot_a.len();
            assert!(slot_b.len() >= cells + cells / 128, "BFP slot without its exponent bytes");
            let (nw, bad, dec) = check_grid_slot(bits, cells, &slot_b, &slot_a);
            // decoded slot vs f32, over the written blocks
            let sel_w: Vec<usize> = (0..cells).filter(|&i| dec[i] != C32::new(0.0, 0.0) || slot_a[i] == C32::new(0.0, 0.0)).collect();
            let (da, db): (Vec<C32>, Vec<C32>) = sel_w.iter().map(|&i| (slot_a[i], dec[i])).unzip();
            let q = (2.0f64).powi(-(bits as i32));
            let es = rel(&db, &da);
            let (ef, ea, eg) = (rel(&yb, &ya), rel(&xb, &xa), rel(&gb, &ga));
            let jb = adjointness(&p, &yb, &xb);
            eprintln!(
                "{kind:?} order {order} kprune {kprune} slab {} fused {fuse} forward kernel {}: grid bfp{bits}: {nw} whole blocks compared, {bad} differ from the host codec; slot {es:.2e} ({:.2} · 2^-b); forward {ef:.2e} ({:.2}), adjoint {ea:.1e} (spread {sx:.1e}), gradient {eg:.2e} ({:.2}; one ulp {egu:.1e}, spread {sg:.1e}); forward one ulp {efu:.1e}, κ {kappa:.1}; adjointness {jb:.1e}",
                b.op.fft_slab(), if b.op.vs_active().is_some() { "vs / vt" } else { "tile" }, es / q, ef / q, eg / q
            );
            assert!(nw > 0 && bad == 0, "{kind:?}: BFP grid blocks differ from the host codec ({bad} of {nw})");
            assert!(es < 2.6 * q, "grid slot error {es:.2e}");
            // the forward sees the grid error through the gather (≈ the slot error); the adjoint does not use it
            // (measured 2.0 κ · slot for grid-level BFP on this random problem, where the error is relative to each block's
            // local magnitude, against 0.65 κ for the y → z intermediate, whose error the z FFT spreads)
            assert!(ef < 4.0 * kappa * es + 4.0 * efu, "forward {ef:.2e} (κ {kappa:.1} · slot {es:.2e}, one ulp {efu:.1e})");
            assert!(ea < 3.0 * sx + 1e-7, "adjoint changed by the forward grid store ({ea:.1e}, spread {sx:.1e})");
            assert!(eg < 4.0 * kappa * es * (egu / efu).max(1.0) + 3.0 * sg + 4.0 * egu, "gradient {eg:.2e}");
            if bits == 16 {
                assert!(ef > 0.05 * q, "bfp16 grid leaves the forward unchanged ({ef:.1e})");
            }
        }
    }
}

#[test]
#[ignore]
fn debug_grid_adjoint() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let f32s = (Store::F32, Store::F32);
    let kind = Kind::Static;
    let p = problem(kind, 9000);
    let mut a = engine_g(&dev, &st, &p, kind, 0, None, false, f32s, Store::F32);
    let gb: u32 = std::env::var("DBG_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let mut b = engine_g(&dev, &st, &p, kind, 0, None, false, f32s, if gb == 0 { Store::F32 } else { Store::Bfp(gb) });
    let dx = st.clone_htod(&p.x).unwrap();
    let dy = st.clone_htod(&p.y).unwrap();
    let run_adj = |e: &mut Eng| { let mut xo = st.alloc_zeros::<C32>(p.x.len()).unwrap(); e.op.adjoint(&st, &e.seg, &e.pts, &dy, &mut xo).unwrap(); st.clone_dtoh(&xo).unwrap() };
    let run_fwd = |e: &mut Eng| { let mut yo = st.alloc_zeros::<C32>(NC * p.m).unwrap(); e.op.forward(&st, &e.seg, &e.pts, &dx, &mut yo).unwrap(); st.clone_dtoh(&yo).unwrap() };
    let (xa, xb) = (run_adj(&mut a), run_adj(&mut b));
    eprintln!("adjoint before any forward: {:.2e}; cb a {} b {}; desc a {} | b {}", rel(&xb, &xa), a.op.cb, b.op.cb, a.op.grid_fft_desc(), b.op.grid_fft_desc());
    let (ya, yb) = (run_fwd(&mut a), run_fwd(&mut b));
    eprintln!("forward: {:.2e}", rel(&yb, &ya));
    let (xa, xb) = (run_adj(&mut a), run_adj(&mut b));
    eprintln!("adjoint after forward: {:.2e}", rel(&xb, &xa));
}
