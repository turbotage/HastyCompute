//! GPU: block-floating-point y ↔ z intermediates of the grid FFT in the segmented NUFFT (HASTY_SEG_FFT_STORE,
//! `set_thread_fft_store`; research/bfp_storage.md §7 step 1) against the f32 engine on the same problem: forward,
//! adjoint and gradient Aᴴ(A x − y) at bfp22 (one stored buffer per direction: predicted ≈ 2 · 2^−22 = 4.8e-7),
//! adjointness of the BFP pair, at correction orders 0 and 3, on a static engine and a 2-D (respiratory × cardiac)
//! knot engine, with the fused pad / crop on and off, slabs on and off, k-support pruning on and off. With the switch
//! off the engine is bit-identical to an engine built with f32 explicitly.
//!   cargo test --release -p hasty-nufft --test seg_fft_store -- --nocapture --test-threads 1

use std::sync::Arc;

use cudarc::driver::CudaStream;
use hasty_cuda::{Device, C32};
use hasty_fft::Store;
use hasty_nufft::{build_segments, hi_exponents, set_thread_fft_store, SegNufft, SegNufftOpts, SegPoints, SegSegments, NHI};

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
}

/// engine of `kind` at `order` with the grid-FFT store `store` (thread default while it is built and planned)
fn engine(dev: &Device, st: &Arc<CudaStream>, p: &Prob, kind: Kind, order: u32, store: Option<Store>, kprune: bool) -> Eng {
    set_thread_fft_store(store);
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
    Eng { op, seg, pts }
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

#[test]
fn bfp22_engine_matches_f32() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let q = (2.0f64).powi(-22);
    // (kind, order, kprune, fused pad / crop, slab planes: None = default)
    let cases: Vec<(Kind, u32, bool, bool, Option<usize>)> = vec![
        (Kind::Static, 0, true, true, None),
        (Kind::Static, 0, true, false, Some(0)),
        (Kind::Static, 0, false, true, Some(1)),
        (Kind::Static, 3, true, true, None),
        (Kind::Knot2d { mc: 6, kr: 3 }, 0, true, true, None),
        (Kind::Knot2d { mc: 6, kr: 3 }, 0, true, false, Some(2)),
        (Kind::Knot2d { mc: 6, kr: 3 }, 3, true, true, None),
        (Kind::Knot2d { mc: 6, kr: 3 }, 3, false, true, Some(0)),
    ];
    // HASTY_BFP_TEST_CASES (e.g. "0,6" for the compute-sanitizer runs): the case indices to run
    let sel: Option<Vec<usize>> = std::env::var("HASTY_BFP_TEST_CASES").ok().map(|v| v.split(',').map(|c| c.parse().unwrap()).collect());
    let mut rows = Vec::new();
    for (ci, (kind, order, kprune, fuse, slab)) in cases.into_iter().enumerate() {
        if sel.as_ref().is_some_and(|s| !s.contains(&ci)) {
            continue;
        }
        let p = problem(kind, 9000);
        // one engine (its tuned kernel variants fixed), the store switched on it: f32, bfp22, f32 again
        let mut a = engine(&dev, &st, &p, kind, order, Some(Store::F32), kprune); // f32 engine (the default is bfp24 since §20)
        a.op.fuse_padcrop = fuse;
        a.op.set_fft_slab(slab);
        assert_eq!((a.op.fft_store(), a.op.fft_store_bytes()), (Store::F32, 0));
        let (ya, xa, ga) = apply(&st, &mut a, &p);
        let (_, xa2, ga2) = apply(&st, &mut a, &p);
        a.op.set_fft_store(Store::Bfp(22)).unwrap();
        assert_eq!(a.op.fft_store(), Store::Bfp(22), "{kind:?}: BFP store not active ({})", a.op.grid_fft_desc());
        let (yb, xb, gb) = apply(&st, &mut a, &p);
        let desc_b = a.op.grid_fft_desc();
        // the stored buffer's own error: the loaded grid slot (pad + forward grid FFT of coil batch 0), bfp22 vs f32
        let dx = st.clone_htod(&p.x).unwrap();
        let cb = a.op.cb;
        let slot_b = a.op.debug_loaded_slot(&st, &a.seg, &dx, 0, 0, cb, fuse, true).unwrap();
        assert_eq!(a.op.fused_padcrop_active(), fuse, "fused pad / crop state");
        a.op.set_fft_store(Store::F32).unwrap();
        let (yc, _, _) = apply(&st, &mut a, &p);
        let slot_a = a.op.debug_loaded_slot(&st, &a.seg, &dx, 0, 0, cb, fuse, true).unwrap();
        let es = rel(&slot_b, &slot_a);
        // the problem's sensitivity: x and y moved by one ulp (relative 2^-24, random signs) in f32; the forward maps a
        // relative grid change e_slot to a relative sample change κ e_slot (random x, random samples: κ ≫ 1)
        let mut rr = R(99);
        let ulp = |v: &[C32], r: &mut R| -> Vec<C32> { v.iter().map(|z| C32::new(z.re * (1.0 + r.u().signum() as f32 * 5.96e-8), z.im * (1.0 + r.u().signum() as f32 * 5.96e-8))).collect() };
        let pu = Prob { x: ulp(&p.x, &mut rr), y: ulp(&p.y, &mut rr), ..problem(kind, 9000) };
        let (yu, xu, gu) = apply(&st, &mut a, &pu);
        let dxu = st.clone_htod(&pu.x).unwrap();
        let slot_u = a.op.debug_loaded_slot(&st, &a.seg, &dxu, 0, 0, cb, fuse, true).unwrap();
        let kappa = rel(&yu, &ya) / rel(&slot_u, &slot_a).max(1e-12);
        let (eau, egu) = (rel(&xu, &xa), rel(&gu, &ga));
        // (the engine's f32 forward itself varies run to run at ≈ 2e-7: tuned kernels with atomics)
        assert!(rel(&yc, &ya) < 1e-6 && a.op.fft_store_bytes() == 0, "{kind:?} order {order}: forward after switching back to f32 differs from the first f32 forward ({:.1e})", rel(&yc, &ya));
        let b = &a;
        let (sx, sg) = (rel(&xa2, &xa), rel(&ga2, &ga));
        let (ef, ea, eg) = (rel(&yb, &ya), rel(&xb, &xa), rel(&gb, &ga));
        let (ja, jb) = (adjointness(&p, &ya, &xa), adjointness(&p, &yb, &xb));
        let desc = format!("{kind:?} order {order}, kprune {kprune}, fused {fuse}, slab {}", b.op.fft_slab());
        let _ = b;
        eprintln!(
            "{desc}: bfp22 vs f32 forward {ef:.2e} ({:.2} · 2^-22), adjoint {ea:.2e} ({:.2}), gradient {eg:.2e} ({:.2}); f32 run-to-run adjoint {sx:.1e}, gradient {sg:.1e}; adjointness f32 {ja:.1e}, bfp22 {jb:.1e}\n    {}",
            ef / q,
            ea / q,
            eg / q,
            desc_b
        );
        rows.push((desc.clone(), ef / q, ea / q, eg / q));
        eprintln!(
            "    stored buffer: loaded slot bfp22 vs f32 {es:.2e} ({:.2} · 2^-22); sensitivity: one-ulp x, y move the forward {:.1e} (κ = {kappa:.1} × the slot's change), adjoint {eau:.1e}, gradient {egu:.1e}; forward predicted κ · slot error = {:.2e}",
            es / q,
            rel(&yu, &ya),
            kappa * es
        );
        rows.push((format!("{desc} [slot]"), es / q, kappa, 0.0));
        // the buffer itself: ≈ 2 · 2^-22 (research/bfp_storage.md §3.4)
        assert!(es < 2.6 * q, "{kind:?} order {order}: loaded slot error {es:.2e}");
        // the operator outputs: the buffer error through the problem's own conditioning (κ forward; for the adjoint
        // and gradient the one-ulp sensitivity: a bfp22 buffer error of ≈ 2 · 2^-22 = 8 ulp-equivalents), plus the
        // run-to-run spread of the atomics
        assert!(ef < 2.0 * kappa * es + 1e-7, "{kind:?} order {order}: forward {ef:.2e} (κ · slot error {:.2e})", kappa * es);
        assert!(ea < 16.0 * eau + 3.0 * sx && eg < 16.0 * egu + 3.0 * sg, "{kind:?} order {order}: adjoint {ea:.2e} (one ulp {eau:.1e}, spread {sx:.1e}), gradient {eg:.2e} (one ulp {egu:.1e}, spread {sg:.1e})");
        assert!(jb < ja.max(1e-7) + 3.0 * q, "{kind:?} order {order}: adjointness {jb:.1e} (f32 {ja:.1e})");
    }
    eprintln!("summary (· 2^-22): forward / adjoint / gradient; [slot] rows: slot error · 2^-22 / κ");
    for (d, f, a, g) in rows {
        eprintln!("  {d:<55} {f:.2} / {a:.2} / {g:.2}");
    }
}

/// switch off: an engine built with f32 storage (HASTY_SEG_FFT_STORE=f32; the default is now bfp24) has f32 intermediates and no BFP buffer; its loaded grid
/// slot (pad + forward grid FFT of one coil batch, deterministic, unlike the whole forward, whose tuned kernels vary at
/// ≈ 2e-7 run to run) is bit-identical before and after a bfp22 round trip of the store; an engine built with a
/// thread default of bfp22 keeps it over its planning
#[test]
fn switch_off_bit_identical() {
    assert!(std::env::var("HASTY_SEG_FFT_STORE").is_err(), "run without HASTY_SEG_FFT_STORE");
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    for (kind, order) in [(Kind::Static, 0u32), (Kind::Knot2d { mc: 6, kr: 3 }, 3)] {
        let p = problem(kind, 6000);
        // the switch off is now `HASTY_SEG_FFT_STORE=f32` (thread form: Some(F32)); the default is bfp24
        // (research/sample_kernels.md §20, tests/seg_defaults.rs)
        let mut a = engine(&dev, &st, &p, kind, order, Some(Store::F32), true);
        assert_eq!((a.op.fft_store(), a.op.fft_store_bytes()), (Store::F32, 0));
        let dx = st.clone_htod(&p.x).unwrap();
        let cb = a.op.cb;
        let mut slots = Vec::new();
        for (fused, s) in [(true, Store::F32), (false, Store::F32), (true, Store::Bfp(22)), (true, Store::F32), (false, Store::F32)] {
            a.op.set_fft_store(s).unwrap();
            slots.push(a.op.debug_loaded_slot(&st, &a.seg, &dx, 0, 0, cb, fused, true).unwrap());
        }
        let (e1, e2) = (bit_equal(&slots[0], &slots[3]), bit_equal(&slots[1], &slots[4]));
        eprintln!("{kind:?} order {order}: loaded slot f32 before / after a bfp22 round trip: fused pad bit-identical {e1}, separate pad {e2}; bfp22 vs f32 {:.2e}", rel(&slots[2], &slots[0]));
        assert!(e1 && e2, "{kind:?}: f32 loaded slot not reproduced after the bfp22 round trip");
        let b = engine(&dev, &st, &p, kind, order, Some(Store::Bfp(22)), true);
        assert_eq!(b.op.fft_store(), Store::Bfp(22), "{kind:?}: thread default bfp22 not kept ({})", b.op.grid_fft_desc());
    }
}
