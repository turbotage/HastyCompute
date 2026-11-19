//! GPU: correction order 3 (per-sample degree-3/4 concomitant phase to first order, `points_hi` / `points2d_hi`) in the
//! knot-major tile kernels (segnufft_knot.cu, `#if CORR >= 3`).
//!   cargo test --release -p hasty-nufft --test seg_knot_hi -- --nocapture --test-threads=1
//! * 1-D knots (`new_temporal(.., true)`): order 3 vs the interval-major order-3 operator (multi kernels, frames as
//!   B-spline slots): same operator up to f32 summation order; adjointness; vs an f64 CPU emulation of the order-3
//!   approximation (`DirectSum` with the B-spline frame weights); split tile lists.
//! * 2-D knots (`new_temporal2d`): order 3 with u = 0 and without order-3 data equals order 2; with u ≠ 0 vs the f64
//!   emulation (respiratory × cardiac weights from the same 16-bit fractions as the tables); adjointness; kernel
//!   variants (KV 0, 1, 2 and several coil groups / lanes / sample blocking) against each other.
//! Small problems (16 × 14 × 10 voxels, 3 coils, a few thousand samples): well under 100 MB of device memory.

use cudarc::driver::CudaStream;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, hi_exponents, SegNufft, SegNufftOpts, SegPoints, SegSegments, NHI};
use hasty_testkit::physops::{bspline_weight, interp_table, rel_on, taylor_expm, DirectSum};
use hasty_testkit::{Rng, C64};
use std::sync::Arc;

const PI: f64 = std::f64::consts::PI;
const DIMS: [usize; 3] = [16, 14, 10];
const RES: f64 = 3e-3;
const NC: usize = 3;

struct Prob {
    nv: usize,
    m: usize,
    idx: Vec<[f64; 3]>,
    smaps: Vec<C64>,
    k: [Vec<f32>; 3],
    tau: Vec<f32>,
    phi0: Vec<f32>,
    dc: Vec<f32>,
    /// cardiac time in knot units (periodic over mc)
    t: Vec<f32>,
    /// respiratory interval and fraction (2-D)
    ri: Vec<u32>,
    rv: Vec<f32>,
    om: Vec<f64>,
}

fn problem(m: usize, mc: usize, nri: usize, seed: u64) -> Prob {
    let nv: usize = DIMS.iter().product();
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % DIMS[0], (q / DIMS[0]) % DIMS[1], q / (DIMS[0] * DIMS[1]));
        [i as f64 - (DIMS[0] / 2) as f64, j as f64 - (DIMS[1] / 2) as f64, k as f64 - (DIMS[2] / 2) as f64]
    }).collect();
    let smaps: Vec<C64> = (0..NC).flat_map(|c| idx.iter().map(move |p| C64::from_polar(1.0 + 0.3 * (p[0] * (c as f64 + 1.0) / 10.0).cos(), 0.1 * c as f64 * p[1] - 0.05 * p[2]))).collect();
    let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * 150.0 * ((p[0] / 8.0) * 0.8 + (p[1] / 7.0).powi(2) * 0.4 - 0.2 * (p[2] / 5.0))).collect();
    let mut r = Rng::new(seed);
    let k: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (0.97 * PI * (2.0 * r.uniform() - 1.0)) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (1.6e-3 + 8e-3 * r.uniform()) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect();
    // ΔC of up to ~0.15 rad over the FOV (rad/m²)
    let dc: Vec<f32> = (0..6 * m).map(|_| (300.0 * (2.0 * r.uniform() - 1.0)) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| (mc as f64 * r.uniform()) as f32).collect();
    let ri: Vec<u32> = (0..m).map(|_| ((nri as f64 * r.uniform()) as u32).min(nri as u32 - 1)).collect();
    let rv: Vec<f32> = (0..m).map(|_| r.uniform() as f32).collect();
    Prob { nv, m, idx, smaps, k, tau, phi0, dc, t, ri, rv, om }
}

/// order-3 data: u per sample (±1) and a table giving a degree-3/4 phase of ~0.05 rad rms over the FOV (as seg_multi.rs)
fn hi_data(p: &Prob, scale: f64, seed: u64) -> (Vec<f32>, Vec<f64>) {
    let mut r = Rng::new(seed);
    let u: Vec<f32> = (0..4 * p.m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect();
    let rmax = DIMS.iter().map(|&d| d as f64 * RES / 2.0).fold(0.0, f64::max);
    let ex = hi_exponents();
    let h: Vec<f64> = (0..4 * NHI).map(|i| {
        let deg: u32 = ex[i % NHI].iter().sum();
        scale * 0.02 * (2.0 * r.uniform() - 1.0) / rmax.powi(deg as i32)
    }).collect();
    (u, h)
}

fn c32(v: &[C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}
fn c64(v: &[C32]) -> Vec<C64> {
    v.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect()
}

/// temporal layout of an engine: 1-D cardiac knots or 2-D (kr respiratory knots of degree rdeg) × cardiac
#[derive(Clone, Copy)]
enum Knots {
    Interval { mc: usize, cdeg: u32 },
    Knot1 { mc: usize, cdeg: u32 },
    Knot2 { mc: usize, cdeg: u32, kr: usize, rdeg: u32 },
}
impl Knots {
    fn images(self) -> usize {
        match self {
            Knots::Interval { mc, .. } | Knots::Knot1 { mc, .. } => mc,
            Knots::Knot2 { mc, kr, .. } => mc * kr,
        }
    }
}

struct Op {
    eng: SegNufft,
    seg: SegSegments,
    pts: SegPoints,
    segph: Vec<C32>,
    atab: Vec<C32>,
    t0: f64,
    dt: f64,
}

/// engine of the layout at correction `order`, its segments (L segments of the B0 phase) and points; `hi`: order-3
/// tables (None: plain `points` / `points2d`)
fn op(dev: &Device, p: &Prob, l: usize, order: u32, kn: Knots, hi: Option<(&[f32], &[f64])>) -> Op {
    let st = dev.default_stream();
    let (mc, cdeg) = match kn {
        Knots::Interval { mc, cdeg } | Knots::Knot1 { mc, cdeg } | Knots::Knot2 { mc, cdeg, .. } => (mc, cdeg),
    };
    let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg: l, order, frames: mc, bspline: cdeg, max_grid_bytes: 1 << 28 };
    let mut eng = match kn {
        Knots::Interval { .. } => SegNufft::new_temporal(dev, DIMS, [RES; 3], NC, opts, false).unwrap(),
        Knots::Knot1 { .. } => SegNufft::new_temporal(dev, DIMS, [RES; 3], NC, opts, true).unwrap(),
        Knots::Knot2 { kr, rdeg, .. } => SegNufft::new_temporal2d(dev, DIMS, [RES; 3], NC, opts, kr, rdeg).unwrap(),
    };
    eng.set_maps(&st, &c32(&p.smaps)).unwrap();
    let fitv: Vec<usize> = (0..p.nv).step_by(2).collect();
    let psi = |v: usize, t: f64| p.om[v] * t;
    let (segph, atab, t0, dt) = build_segments(p.nv, l, 1.6e-3, 9.6e-3, 300, &fitv, &psi);
    let seg = eng.segments(&st, &segph, &atab, t0, dt).unwrap();
    let dc: &[f32] = if order > 0 { &p.dc } else { &[] };
    let kk = [&p.k[0][..], &p.k[1][..], &p.k[2][..]];
    let pts = match (kn, hi) {
        (Knots::Knot2 { .. }, Some((u, h))) => eng.points2d_hi(&st, kk, &p.tau, &p.phi0, dc, &p.t, &p.ri, &p.rv, u, h).unwrap(),
        (Knots::Knot2 { .. }, None) => eng.points2d(&st, kk, &p.tau, &p.phi0, dc, &p.t, &p.ri, &p.rv).unwrap(),
        (_, Some((u, h))) => eng.points_hi(&st, kk, &p.tau, &p.phi0, dc, u, h, Some(&p.t)).unwrap(),
        (_, None) => eng.points(&st, kk, &p.tau, &p.phi0, dc, Some(&p.t)).unwrap(),
    };
    Op { eng, seg, pts, segph, atab, t0, dt }
}

/// (A x, Aᴴ y, normalised adjointness |⟨y, Ax⟩ − ⟨Aᴴy, x⟩| / (‖y‖ ‖Ax‖))
fn apply(o: &mut Op, st: &Arc<CudaStream>, nimg: usize, x: &[C64], y: &[C64]) -> (Vec<C64>, Vec<C64>, f64) {
    let (nv, m) = (x.len() / nimg, o.pts.m);
    let dx = st.clone_htod(&c32(x)).unwrap();
    let dy = st.clone_htod(&c32(y)).unwrap();
    let mut yo = st.alloc_zeros::<C32>(NC * m).unwrap();
    let mut xo = st.alloc_zeros::<C32>(nimg * nv).unwrap();
    o.eng.forward(st, &o.seg, &o.pts, &dx, &mut yo).unwrap();
    o.eng.adjoint(st, &o.seg, &o.pts, &dy, &mut xo).unwrap();
    // the fused residual equals forward − y
    let mut rr = st.alloc_zeros::<C32>(NC * m).unwrap();
    o.eng.forward_residual(st, &o.seg, &o.pts, &dx, &dy, &mut rr).unwrap();
    let (yo, xo, rr) = (c64(&st.clone_dtoh(&yo).unwrap()), c64(&st.clone_dtoh(&xo).unwrap()), c64(&st.clone_dtoh(&rr).unwrap()));
    let res: Vec<C64> = yo.iter().zip(y).map(|(a, b)| a - b).collect();
    assert!(rel_on(&rr, &res, None) < 1e-5, "residual differs from forward − y: {:.1e}", rel_on(&rr, &res, None));
    let dot = |a: &[C64], b: &[C64]| a.iter().zip(b).map(|(p, q)| p.conj() * q).sum::<C64>();
    let e = (dot(y, &yo) - dot(&xo, x)).norm() / (dot(y, y).re * dot(&yo, &yo).re).sqrt();
    (yo, xo, e)
}

/// per-sample temporal weights (m × images) exactly as the tables hold them (1-D: f32 fraction; 2-D: 16-bit fractions)
fn weights(p: &Prob, kn: Knots) -> Vec<f64> {
    let n = kn.images();
    let mut bw = vec![0f64; p.m * n];
    for j in 0..p.m {
        let row = &mut bw[j * n..(j + 1) * n];
        match kn {
            Knots::Interval { mc, cdeg } | Knots::Knot1 { mc, cdeg } => {
                let tt = (p.t[j] as f64).rem_euclid(mc as f64);
                let i = (tt.floor() as usize).min(mc - 1);
                let tq = i as f64 + (tt - i as f64) as f32 as f64;
                for (f, w) in row.iter_mut().enumerate() {
                    *w = bspline_weight(mc, cdeg, f, tq);
                }
            }
            Knots::Knot2 { mc, cdeg, kr: _, rdeg } => {
                let tt = (p.t[j] as f64).rem_euclid(mc as f64);
                let ic = (tt.floor() as usize).min(mc - 1);
                let uq = ((tt - ic as f64) * 65535.0).round().clamp(0.0, 65535.0) / 65535.0;
                let vq = (p.rv[j].clamp(0.0, 1.0) as f64 * 65535.0).round() / 65535.0;
                let ri = p.ri[j] as usize;
                let resp: Vec<(usize, f64)> = if rdeg == 0 { vec![(ri, 1.0)] } else { vec![(ri, 1.0 - vq), (ri + 1, vq)] };
                for (k, g) in resp {
                    for b in 0..mc {
                        row[k * mc + b] += g * bspline_weight(mc, cdeg, b, ic as f64 + uq);
                    }
                }
            }
        }
    }
    bw
}

/// the order-3 approximation in f64 (as seg_multi.rs): Σ_f b_f(t_j) e^{−i(k·r + φ0)} Σ_l a_l(τ) P_l(r) [T2(ρᵀΔCρ) − i c(ρ)]
/// x_f (order 2: c = 0); returns (A x, Aᴴ y, rms of the degree-3/4 phase)
fn emulate(p: &Prob, o: &Op, l: usize, kn: Knots, hi: Option<(&[f32], &[f64])>, x: &[C64], y: &[C64]) -> (Vec<C64>, Vec<C64>, f64) {
    let (nv, m) = (p.nv, p.m);
    let p64 = c64(&o.segph);
    let a64 = c64(&o.atab);
    let at: Vec<C64> = p.tau.iter().flat_map(|&t| interp_table(&a64, l, o.t0, o.dt, t as f64)).collect();
    let ex = hi_exponents();
    let cpoly = |j: usize, r: &[f64; 3]| -> f64 {
        let Some((u, h)) = hi else { return 0.0 };
        (0..NHI).map(|i| (0..4).map(|q| u[q * m + j] as f64 * h[q * NHI + i]).sum::<f64>() * (0..3).map(|a| r[a].powi(ex[i][a] as i32)).product::<f64>()).sum()
    };
    let f = |j: usize, q: usize| {
        let ix = &p.idx[q];
        let r = [ix[0] * RES, ix[1] * RES, ix[2] * RES];
        let lin = (0..3).map(|a| p.k[a][j] as f64 * ix[a]).sum::<f64>() + p.phi0[j] as f64;
        let dc = |i: usize| p.dc[i * m + j] as f64;
        let psi = dc(0) * r[0] * r[0] + dc(1) * r[1] * r[1] + dc(2) * r[2] * r[2] + 2.0 * (dc(3) * r[0] * r[1] + dc(4) * r[0] * r[2] + dc(5) * r[1] * r[2]);
        let s: C64 = (0..l).map(|s| at[j * l + s] * p64[s * nv + q]).sum();
        C64::from_polar(1.0, -lin) * s * (taylor_expm(psi, 2) - C64::new(0.0, cpoly(j, &r)))
    };
    let bw = weights(p, kn);
    let vox: Vec<usize> = (0..nv).collect();
    let ds = DirectSum { nv, nc: NC, m, frames: kn.images(), smaps: &p.smaps, voxels: &vox, bw: Some(&bw) };
    let (ye, xe) = ds.apply(&f, x, y);
    let (mut s2, mut n) = (0.0, 0usize);
    for j in (0..m).step_by(7) {
        for q in (0..nv).step_by(5) {
            let ix = &p.idx[q];
            s2 += cpoly(j, &[ix[0] * RES, ix[1] * RES, ix[2] * RES]).powi(2);
            n += 1;
        }
    }
    (ye, xe, (s2 / n as f64).sqrt())
}

/// GPU vs emulation at order 2 (reference level of the implementation error) and order 3, as seg_multi.rs: the order-3
/// implementation error stays at the order-2 level plus the first-order derivative-kernel error of the degree-3/4
/// terms, while the terms themselves are far above that bound (a sign or kernel-unit scale error would show)
fn check_emulation(dev: &Device, p: &Prob, l: usize, kn: Knots, label: &str) {
    let st = dev.default_stream();
    let nimg = kn.images();
    let mut r = Rng::new(5);
    let x = r.complex_vec(nimg * p.nv);
    let y = r.complex_vec(NC * p.m);
    let (u, h) = hi_data(p, 1.0, 77);
    let mut o2 = op(dev, p, l, 2, kn, None);
    let (y2, x2, a2) = apply(&mut o2, &st, nimg, &x, &y);
    let (ye2, xe2, _) = emulate(p, &o2, l, kn, None, &x, &y);
    let (i2f, i2a) = (rel_on(&y2, &ye2, None), rel_on(&x2, &xe2, None));
    let mut o3 = op(dev, p, l, 3, kn, Some((&u, &h)));
    let (y3, x3, a3) = apply(&mut o3, &st, nimg, &x, &y);
    let (ye3, xe3, crms) = emulate(p, &o3, l, kn, Some((&u, &h)), &x, &y);
    let (i3f, i3a) = (rel_on(&y3, &ye3, None), rel_on(&x3, &xe3, None));
    let (d34f, d34a) = (rel_on(&ye3, &ye2, None), rel_on(&xe3, &xe2, None));
    eprintln!("{label}: degree-3/4 phase rms {crms:.3} rad (effect {d34f:.2e} / {d34a:.2e}); GPU vs emulation order 3 forward {i3f:.2e}, adjoint {i3a:.2e} (order 2: {i2f:.2e} / {i2a:.2e}); adjointness order 3 {a3:.1e} (order 2 {a2:.1e})");
    let tol = |i2: f64| 2.0 * i2 + 2e-3 * crms + 1e-5;
    assert!(d34f > 10.0 * tol(i2f) && d34a > 10.0 * tol(i2a), "{label}: the degree-3/4 phase is too small to check the terms");
    assert!(i3f < tol(i2f) && i3a < tol(i2a), "{label}: order 3 vs emulation {i3f:.2e} / {i3a:.2e}");
    assert!(a3 < 2e-6 && a2 < 2e-6, "{label}: adjointness order 3 {a3:.1e}, order 2 {a2:.1e}");
}

/// 1-D knots at order 3 vs the interval-major order-3 operator (the multi kernels with B-spline frame slots) and adjointness
fn knot1_vs_interval(dev: &Device, m: usize, mc: usize, cdeg: u32, l: usize) {
    let st = dev.default_stream();
    let p = problem(m, mc, 1, 41 + m as u64);
    let (u, h) = hi_data(&p, 1.0, 78);
    let mut r = Rng::new(6);
    let x = r.complex_vec(mc * p.nv);
    let y = r.complex_vec(NC * p.m);
    let mut oi = op(dev, &p, l, 3, Knots::Interval { mc, cdeg }, Some((&u, &h)));
    let (yi, xi, ai) = apply(&mut oi, &st, mc, &x, &y);
    drop(oi);
    let mut ok = op(dev, &p, l, 3, Knots::Knot1 { mc, cdeg }, Some((&u, &h)));
    let (yk, xk, ak) = apply(&mut ok, &st, mc, &x, &y);
    // the order-3 terms' size in this operator (knot order 3 without them)
    let mut o0 = op(dev, &p, l, 3, Knots::Knot1 { mc, cdeg }, None);
    let (y0, x0, _) = apply(&mut o0, &st, mc, &x, &y);
    let (ef, ea) = (rel_on(&yk, &yi, None), rel_on(&xk, &xi, None));
    let (df, da) = (rel_on(&y0, &yi, None), rel_on(&x0, &xi, None));
    eprintln!("1-D knots M {mc} deg {cdeg} nseg {l} m {m}: order 3 knot- vs interval-major forward {ef:.1e}, adjoint {ea:.1e} (degree-3/4 terms {df:.1e} / {da:.1e}); adjointness knot {ak:.1e}, interval {ai:.1e}");
    assert!(df > 1e-3 && da > 1e-3, "degree-3/4 terms too small to check");
    // f32 summation order only (bounds as tests/seg_knot.rs)
    assert!(ef < 3e-5 && ea < 6e-5, "knot-major order 3 differs from interval-major: forward {ef:.1e}, adjoint {ea:.1e}");
    assert!(ak < 2e-6, "knot-major order 3 adjointness {ak:.1e}");
}

#[test]
fn knot1d_order3_matches_interval_major() {
    let dev = Device::new(0).expect("CUDA device");
    knot1_vs_interval(&dev, 3000, 6, 3, 6);
    knot1_vs_interval(&dev, 3000, 5, 1, 5);   // odd segment count: scalar loads in the variant-1 kernels
}

#[test]
fn knot1d_order3_split_lists() {
    // ~30 k samples on a 20 × 18 × 13 fine grid: tile lists above the 1024-sample item limit (atomic items)
    let dev = Device::new(0).expect("CUDA device");
    knot1_vs_interval(&dev, 30_000, 4, 3, 4);
}

#[test]
fn knot1d_order3_matches_emulation() {
    let dev = Device::new(0).expect("CUDA device");
    let p = problem(2500, 5, 1, 43);
    check_emulation(&dev, &p, 6, Knots::Knot1 { mc: 5, cdeg: 3 }, "1-D knots M 5 cubic");
}

/// 2-D: order 3 with a zero u (and without order-3 data) equals order 2
fn knot2_zero(dev: &Device, mc: usize, cdeg: u32, kr: usize, rdeg: u32) {
    let st = dev.default_stream();
    let nri = if rdeg == 0 { kr } else { kr - 1 };
    let kn = Knots::Knot2 { mc, cdeg, kr, rdeg };
    let p = problem(3000, mc, nri, 44 + kr as u64);
    let mut r = Rng::new(7);
    let x = r.complex_vec(kn.images() * p.nv);
    let y = r.complex_vec(NC * p.m);
    let zero = (vec![0f32; 4 * p.m], hi_data(&p, 1.0, 79).1);
    let (y2, x2, _) = apply(&mut op(dev, &p, 6, 2, kn, None), &st, kn.images(), &x, &y);
    let (yz, xz, az) = apply(&mut op(dev, &p, 6, 3, kn, Some((&zero.0, &zero.1))), &st, kn.images(), &x, &y);
    let (yn, xn, _) = apply(&mut op(dev, &p, 6, 3, kn, None), &st, kn.images(), &x, &y);
    let (ezf, eza, enf, ena) = (rel_on(&yz, &y2, None), rel_on(&xz, &x2, None), rel_on(&yn, &y2, None), rel_on(&xn, &x2, None));
    eprintln!("2-D knots M {mc} deg {cdeg}, kr {kr} rdeg {rdeg}: order 3 (u = 0) vs order 2 forward {ezf:.1e}, adjoint {eza:.1e}; order 3 without order-3 data {enf:.1e} / {ena:.1e}; adjointness {az:.1e}");
    // f32 only (the order-3 engine's chunk differs: other sample grouping of the atomics / staging)
    assert!(ezf < 1e-5 && eza < 1e-5 && enf < 1e-5 && ena < 1e-5, "order 3 with zero degree-3/4 terms differs from order 2");
}

#[test]
fn knot2d_order3_zero_equals_order2() {
    let dev = Device::new(0).expect("CUDA device");
    knot2_zero(&dev, 4, 3, 3, 1);
    knot2_zero(&dev, 5, 1, 2, 0);
}

#[test]
fn knot2d_order3_matches_emulation() {
    let dev = Device::new(0).expect("CUDA device");
    let p = problem(2500, 4, 2, 45);
    check_emulation(&dev, &p, 6, Knots::Knot2 { mc: 4, cdeg: 3, kr: 3, rdeg: 1 }, "2-D knots M 4 cubic × kr 3 linear");
    let p = problem(2500, 4, 2, 46);
    check_emulation(&dev, &p, 5, Knots::Knot2 { mc: 4, cdeg: 1, kr: 2, rdeg: 0 }, "2-D knots M 4 linear × kr 2 bins");
}

/// kernel variants at order 3 (KV 0 / 1 / 2, coil groups, lanes, sample blocking) against KV 0, with adjointness
#[test]
fn knot2d_order3_variants() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kn = Knots::Knot2 { mc: 4, cdeg: 3, kr: 3, rdeg: 1 };
    let p = problem(4000, 4, 2, 47);
    let (u, h) = hi_data(&p, 1.0, 80);
    let mut r = Rng::new(8);
    let x = r.complex_vec(kn.images() * p.nv);
    let y = r.complex_vec(NC * p.m);
    let mut o = op(&dev, &p, 6, 3, kn, Some((&u, &h)));
    let base = o.eng.tile_shape(1, 1, 2, 1, 1, 0).unwrap();
    o.eng.set_tiles(base, base).unwrap();
    let (y0, x0, a0) = apply(&mut o, &st, kn.images(), &x, &y);
    assert!(a0 < 2e-6, "kv 0 adjointness {a0:.1e}");
    let mut n = 0;
    for kv in [0usize, 1, 2] {
        for (cpba, cpbf, ppf, fsb) in [(1, 1, 2, 1), (3, 3, 1, 1), (1, 3, 4, 2), (3, 1, 2, 2)] {
            let (Some(tf), Some(ta)) = (o.eng.tile_shape(cpba, cpbf, ppf, 1, fsb, kv), o.eng.tile_shape(cpba, cpbf, ppf, 1, fsb, kv)) else { continue };
            o.eng.set_tiles(tf, ta).unwrap();
            let (yv, xv, a) = apply(&mut o, &st, kn.images(), &x, &y);
            let (ef, ea) = (rel_on(&yv, &y0, None), rel_on(&xv, &x0, None));
            eprintln!("order 3 kv {kv} cpba {cpba} cpbf {cpbf} ppf {ppf} fsb {fsb} (chunk {}): vs kv 0 forward {ef:.1e}, adjoint {ea:.1e}, adjointness {a:.1e}", tf.chunk);
            assert!(ef < 3e-5 && ea < 6e-5 && a < 2e-6, "kv {kv} {tf:?}: forward {ef:.1e}, adjoint {ea:.1e}, adjointness {a:.1e}");
            n += 1;
        }
    }
    assert!(n >= 9, "only {n} variant shapes ran");
}

/// Tiny timing of the order-3 knot kernels vs order 2 (whole forward / adjoint calls, same engine shape; the FFTs and
/// pad / crop are common). Ignored by default: cargo test --release -p hasty-nufft --test seg_knot_hi -- --ignored --nocapture
#[test]
#[ignore]
fn knot_order3_timing() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let dims = [32usize, 32, 32];
    let nv: usize = dims.iter().product();
    let (m, mc, nc, l) = (100_000usize, 6usize, 4usize, 6usize);
    let mut r = Rng::new(99);
    let k: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (0.97 * PI * (2.0 * r.uniform() - 1.0)) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (1.6e-3 + 8e-3 * r.uniform()) as f32).collect();
    let phi0 = vec![0f32; m];
    let dc: Vec<f32> = (0..6 * m).map(|_| (300.0 * (2.0 * r.uniform() - 1.0)) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| (mc as f64 * r.uniform()) as f32).collect();
    let u: Vec<f32> = (0..4 * m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect();
    let h: Vec<f64> = (0..4 * NHI).map(|_| 1e3 * (2.0 * r.uniform() - 1.0)).collect();
    let smaps = vec![C32::new(1.0, 0.0); nc * nv];
    let fitv: Vec<usize> = (0..nv).step_by(5).collect();
    let psi = |v: usize, tt: f64| 2.0 * PI * 100.0 * ((v % 97) as f64 / 97.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, l, 1.6e-3, 9.6e-3, 128, &fitv, &psi);
    let x: Vec<C32> = (0..mc * nv).map(|_| C32::new(r.uniform() as f32, r.uniform() as f32)).collect();
    let yv: Vec<C32> = (0..nc * m).map(|_| C32::new(r.uniform() as f32, r.uniform() as f32)).collect();
    let (dx, dy) = (st.clone_htod(&x).unwrap(), st.clone_htod(&yv).unwrap());
    for order in [2u32, 3] {
        let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg: l, order, frames: mc, bspline: 3, max_grid_bytes: 1 << 28 };
        let mut e = SegNufft::new_temporal(&dev, dims, [RES; 3], nc, opts, true).unwrap();
        e.set_maps(&st, &smaps).unwrap();
        let seg = e.segments(&st, &sp, &at, t0, dt).unwrap();
        let kk = [&k[0][..], &k[1][..], &k[2][..]];
        let pts = if order == 3 { e.points_hi(&st, kk, &tau, &phi0, &dc, &u, &h, Some(&t)).unwrap() } else { e.points(&st, kk, &tau, &phi0, &dc, Some(&t)).unwrap() };
        let mut yo = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xo = st.alloc_zeros::<C32>(mc * nv).unwrap();
        let time = |f: &mut dyn FnMut()| {
            f();
            st.synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                f();
            }
            st.synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e3 / 5.0
        };
        let tf = time(&mut || e.forward(&st, &seg, &pts, &dx, &mut yo).unwrap());
        let ta = time(&mut || e.adjoint(&st, &seg, &pts, &dy, &mut xo).unwrap());
        eprintln!("order {order}: knot-major forward {tf:.2} ms, adjoint {ta:.2} ms (dims {dims:?}, m {m}, M {mc} cubic, {nc} coils, nseg {l}; default tiles fwd {:?})", e.tiles_fwd);
        // the same shape with the device's opt-in shared memory (chunk 32 at both orders)
        let tc = e.tile_shape(4, 4, 2, 1, 1, 1).unwrap();
        e.set_tiles(tc, tc).unwrap();
        let tf = time(&mut || e.forward(&st, &seg, &pts, &dx, &mut yo).unwrap());
        let ta = time(&mut || e.adjoint(&st, &seg, &pts, &dy, &mut xo).unwrap());
        eprintln!("order {order}: knot-major forward {tf:.2} ms, adjoint {ta:.2} ms with chunk {}", tc.chunk);
    }
}
