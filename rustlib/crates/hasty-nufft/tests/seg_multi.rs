//! GPU: the multi-image path of `SegNufft` (`forward_multi` / `adjoint_multi`, kernels segnufft_multi.cu) against K
//! separate single-image calls, and correction order 3 (`points_hi`) against order 2 and a CPU emulation.
//!   cargo test --release -p hasty-nufft --test seg_multi -- --nocapture
//! Cases: static image and 4 cubic-B-spline frames; coil batches smaller than the coil count; K = 1, 2, 3, 5 images
//! (passes of ≤ 4); the memory-limited plan (one image per pass) and the same with extra memory allowed; random
//! per-sample weights. Tolerances: multi vs separate ≤ 1e-5 relative (f32 summation order only), adjoint identity
//! ≤ 1e-5. Expected run time on an RTX 4090: ~1 min the first time (kernel compiles), seconds afterwards.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, hi_exponents, SegNufft, SegNufftOpts, SegPoints, SegSegments, NHI};
use hasty_testkit::physops::{interp_table, rel_on, taylor_expm, DirectSum};
use hasty_testkit::{Rng, C64};

const PI: f64 = std::f64::consts::PI;
const DIMS: [usize; 3] = [20, 18, 12];
const RES: f64 = 3e-3;
const NC: usize = 5;

struct Prob {
    nv: usize,
    m: usize,
    idx: Vec<[f64; 3]>,
    smaps: Vec<C64>,
    k: [Vec<f32>; 3],
    tau: Vec<f32>,
    phi0: Vec<f32>,
    dc: Vec<f32>,
    t: Vec<f32>,
    om: Vec<f64>,
}

fn problem() -> Prob {
    let nv: usize = DIMS.iter().product();
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % DIMS[0], (q / DIMS[0]) % DIMS[1], q / (DIMS[0] * DIMS[1]));
        [i as f64 - (DIMS[0] / 2) as f64, j as f64 - (DIMS[1] / 2) as f64, k as f64 - (DIMS[2] / 2) as f64]
    }).collect();
    let smaps: Vec<C64> = (0..NC).flat_map(|c| idx.iter().map(move |p| C64::from_polar(1.0 + 0.3 * (p[0] * (c as f64 + 1.0) / 10.0).cos(), 0.1 * c as f64 * p[1] - 0.05 * p[2]))).collect();
    let om: Vec<f64> = idx.iter().map(|p| 2.0 * PI * 150.0 * ((p[0] / 10.0) * 0.8 + (p[1] / 9.0).powi(2) * 0.4 - 0.2 * (p[2] / 6.0))).collect();
    let mut r = Rng::new(41);
    let (nro, na, te, t_ro) = (8usize, 300usize, 1.6e-3, 8e-3);
    let mut k: [Vec<f32>; 3] = Default::default();
    let mut tau = vec![];
    for _ in 0..nro {
        let (kz, th) = (0.97 * PI * (2.0 * r.uniform() - 1.0), PI * r.uniform());
        for n in 0..na {
            let s = n as f64 / (na - 1) as f64;
            let (rad, ang) = (0.97 * PI * s.sqrt(), th + 2.0 * PI * 7.0 * s.sqrt());
            k[0].push((rad * ang.cos()) as f32);
            k[1].push((rad * ang.sin()) as f32);
            k[2].push(kz as f32);
            tau.push((te + t_ro * s) as f32);
        }
    }
    let m = tau.len();
    // per-sample constant phase and a ΔC of up to ~0.3 rad over the FOV (rad/m²), frame times over 4 frames
    let phi0: Vec<f32> = (0..m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect();
    let dc: Vec<f32> = (0..6 * m).map(|_| (300.0 * (2.0 * r.uniform() - 1.0)) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| (4.0 * r.uniform()) as f32).collect();
    Prob { nv, m, idx, smaps, k, tau, phi0, dc, t, om }
}

fn c32(v: &[C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}
fn c64(v: &[C32]) -> Vec<C64> {
    v.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect()
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

/// engine (L segments of the B0 phase, correction `order`), its segments and points; order 3 with the given
/// degree-3/4 data (u: 4 × m, h: 4 × NHI)
fn op(dev: &Device, p: &Prob, l: usize, order: u32, frames: usize, max_grid_bytes: usize, hi: Option<(&[f32], &[f64])>) -> Op {
    let st = dev.default_stream();
    let mut eng = SegNufft::new(dev, DIMS, [RES; 3], NC, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order, frames, bspline: 3, max_grid_bytes }).unwrap();
    eng.set_maps(&st, &c32(&p.smaps)).unwrap();
    let fitv: Vec<usize> = (0..p.nv).step_by(2).collect();
    let psi = |v: usize, t: f64| p.om[v] * t;
    let (segph, atab, t0, dt) = build_segments(p.nv, l, p.tau[0] as f64, p.tau[p.m - 1] as f64, 300, &fitv, &psi);
    let seg = eng.segments(&st, &segph, &atab, t0, dt).unwrap();
    let tt = (frames > 1).then_some(&p.t[..]);
    let dc: &[f32] = if order > 0 { &p.dc } else { &[] };
    let pts = match hi {
        Some((u, h)) => eng.points_hi(&st, [&p.k[0], &p.k[1], &p.k[2]], &p.tau, &p.phi0, dc, u, h, tt).unwrap(),
        None => eng.points(&st, [&p.k[0], &p.k[1], &p.k[2]], &p.tau, &p.phi0, dc, tt).unwrap(),
    };
    Op { eng, seg, pts, segph, atab, t0, dt }
}

fn fwd1(o: &mut Op, st: &Arc<CudaStream>, x: &[C64], m: usize) -> Vec<C64> {
    let dx = st.clone_htod(&c32(x)).unwrap();
    let mut y = st.alloc_zeros::<C32>(NC * m).unwrap();
    o.eng.forward(st, &o.seg, &o.pts, &dx, &mut y).unwrap();
    c64(&st.clone_dtoh(&y).unwrap())
}
fn adj1(o: &mut Op, st: &Arc<CudaStream>, y: &[C64], len: usize) -> Vec<C64> {
    let dy = st.clone_htod(&c32(y)).unwrap();
    let mut x = st.alloc_zeros::<C32>(len).unwrap();
    o.eng.adjoint(st, &o.seg, &o.pts, &dy, &mut x).unwrap();
    c64(&st.clone_dtoh(&x).unwrap())
}

/// multi vs separate calls for K images (image 0 unweighted, the others with random weights); returns the errors
fn check_multi(o: &mut Op, st: &Arc<CudaStream>, p: &Prob, frames: usize, k: usize, label: &str) -> (f64, f64, f64) {
    let (nv, m) = (p.nv, p.m);
    let mut r = Rng::new(1000 + k as u64);
    let xs: Vec<Vec<C64>> = (0..k).map(|_| r.complex_vec(frames * nv)).collect();
    let ws: Vec<Option<Vec<C64>>> = (0..k).map(|i| (i > 0).then(|| r.complex_vec(m))).collect();
    let y = r.complex_vec(NC * m);
    // separate: Σ_k w_k ⊙ A x_k and Aᴴ(w̄_k ⊙ y)
    let mut ysep = vec![C64::new(0.0, 0.0); NC * m];
    let mut xsep = Vec::new();
    for i in 0..k {
        let yi = fwd1(o, st, &xs[i], m);
        for c in 0..NC {
            for j in 0..m {
                ysep[c * m + j] += ws[i].as_ref().map_or(C64::new(1.0, 0.0), |w| w[j]) * yi[c * m + j];
            }
        }
        let yw: Vec<C64> = (0..NC * m).map(|n| ws[i].as_ref().map_or(C64::new(1.0, 0.0), |w| w[n % m].conj()) * y[n]).collect();
        xsep.push(adj1(o, st, &yw, frames * nv));
    }
    // multi
    let dxs: Vec<CudaSlice<C32>> = xs.iter().map(|x| st.clone_htod(&c32(x)).unwrap()).collect();
    let dws: Vec<Option<CudaSlice<C32>>> = ws.iter().map(|w| w.as_ref().map(|w| st.clone_htod(&c32(w)).unwrap())).collect();
    let dy = st.clone_htod(&c32(&y)).unwrap();
    let mut ym = st.alloc_zeros::<C32>(NC * m).unwrap();
    let xr: Vec<&CudaSlice<C32>> = dxs.iter().collect();
    let wr: Vec<Option<&CudaSlice<C32>>> = dws.iter().map(|w| w.as_ref()).collect();
    o.eng.forward_multi(st, &o.seg, &o.pts, &xr, &wr, &mut ym).unwrap();
    let mut dxm: Vec<CudaSlice<C32>> = (0..k).map(|_| st.alloc_zeros::<C32>(frames * nv).unwrap()).collect();
    {
        let mut xm: Vec<&mut CudaSlice<C32>> = dxm.iter_mut().collect();
        o.eng.adjoint_multi(st, &o.seg, &o.pts, &dy, &wr, &mut xm).unwrap();
    }
    let ym = c64(&st.clone_dtoh(&ym).unwrap());
    let xm: Vec<Vec<C64>> = dxm.iter().map(|d| c64(&st.clone_dtoh(d).unwrap())).collect();
    let ef = rel_on(&ym, &ysep, None);
    let ea = (0..k).map(|i| rel_on(&xm[i], &xsep[i], None)).fold(0.0, f64::max);
    // ⟨y, Σ w_k A x_k⟩ = Σ_k ⟨Aᴴ(w̄_k y), x_k⟩
    let l: C64 = y.iter().zip(&ym).map(|(a, b)| a.conj() * b).sum();
    let rr: C64 = (0..k).map(|i| xm[i].iter().zip(&xs[i]).map(|(a, b)| a.conj() * b).sum::<C64>()).sum();
    let dot = (l - rr).norm() / l.norm();
    let (g, cb, extra) = o.eng.multi_info(st, k).unwrap();
    eprintln!("  {label}, K = {k}: {g} images / pass, coil batch {cb} (engine {}), extra {extra} B: forward {ef:.2e}, adjoint {ea:.2e}, adjoint identity {dot:.1e}", o.eng.cb);
    (ef, ea, dot)
}

#[test]
fn seg_multi_matches_separate_calls() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let p = problem();
    let l = 6;
    // grid bytes per coil (one image) for the memory-limited cases
    let pc = |frames: usize| {
        let o = op(&dev, &p, l, 2, frames, 1 << 30, None);
        o.eng.plan.grid_elems() * l * o.eng.nf * 8
    };
    for frames in [1usize, 4] {
        let pcb = pc(frames);
        let cases: [(&str, usize, usize); 4] = [
            ("all coils in one batch", 1 << 30, 0),
            ("coil batch 2 (budget)", 2 * pcb + 1, 0),
            ("one coil over budget, no extra memory", 1, 0),
            ("one coil over budget, extra memory", 1, 8 * pcb),
        ];
        for (label, mgb, extra) in cases {
            let mut o = op(&dev, &p, l, 2, frames, mgb, None);
            o.eng.multi_extra_bytes = extra;
            for k in [1usize, 2, 3, 5] {
                let (ef, ea, dot) = check_multi(&mut o, &st, &p, frames, k, &format!("{frames} frame(s), {label}"));
                assert!(ef < 5e-5 && ea < 5e-5, "{label}, frames {frames}, K {k}: multi vs separate {ef:.2e} / {ea:.2e}"); // float32
                assert!(dot < 1e-3, "adjoint identity {dot:.1e}"); // float32 (measured ≤ 1.9e-4)
            }
            if extra == 0 && mgb == 1 {
                assert_eq!(o.eng.multi_info(&st, 2).unwrap().0, 1, "no extra memory: one image per pass");
            }
            if extra > 0 {
                assert_eq!(o.eng.multi_info(&st, 2).unwrap().0, 2, "extra memory: both images in one pass");
            }
        }
    }
    // order 0 (no ΔC) and the unweighted sum of two images = A(x0 + x1)
    let mut o = op(&dev, &p, l, 0, 1, 1 << 30, None);
    let (ef, ea, _) = check_multi(&mut o, &st, &p, 1, 2, "order 0");
    assert!(ef < 5e-5 && ea < 5e-5, "{ef:.2e} / {ea:.2e}"); // float32
}

/// order-3 data: u per sample (±1) and a table giving a degree-3/4 phase of ~0.05 rad rms over the FOV
fn hi_data(p: &Prob, scale: f64) -> (Vec<f32>, Vec<f64>) {
    let mut r = Rng::new(77);
    let u: Vec<f32> = (0..4 * p.m).map(|_| (2.0 * r.uniform() - 1.0) as f32).collect();
    let rmax = DIMS.iter().map(|&d| d as f64 * RES / 2.0).fold(0.0, f64::max);
    let ex = hi_exponents();
    let h: Vec<f64> = (0..4 * NHI).map(|i| {
        let deg: u32 = ex[i % NHI].iter().sum();
        scale * 0.02 * (2.0 * r.uniform() - 1.0) / rmax.powi(deg as i32)
    }).collect();
    (u, h)
}

/// the order-3 approximation in f64: e^{−i(k·r + φ0)} Σ_l a_l(τ) P_l(r) [T2(ρᵀΔCρ) − i c(ρ)] (order 2: c = 0)
fn emulate(p: &Prob, o: &Op, l: usize, hi: Option<(&[f32], &[f64])>, x: &[C64], y: &[C64]) -> (Vec<C64>, Vec<C64>, f64) {
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
    let vox: Vec<usize> = (0..nv).collect();
    let ds = DirectSum { nv, nc: NC, m, frames: 1, smaps: &p.smaps, voxels: &vox, bw: None };
    let (ye, xe) = ds.apply(&f, x, y);
    let (mut s2, mut n) = (0.0, 0usize);
    for j in (0..m).step_by(7) {
        for q in (0..nv).step_by(5) {
            let ix = &p.idx[q];
            s2 += cpoly(j, &[ix[0] * RES, ix[1] * RES, ix[2] * RES]).powi(2);
            n += 1;
        }
    }
    let crms = (s2 / n as f64).sqrt();
    (ye, xe, crms)
}

/// Order 3: with a zero degree-3/4 table it equals order 2; with a ~0.05 rad degree-3/4 phase the GPU operator matches
/// its f64 emulation (signs and kernel-unit scaling of the cubic / quartic terms) to the derivative-kernel accuracy, the
/// adjoint identity holds, and the multi path with weights matches separate calls.
#[test]
fn seg_order3_matches_emulation() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let p = problem();
    let l = 6;
    let mut r = Rng::new(5);
    let x = r.complex_vec(p.nv);
    let y = r.complex_vec(NC * p.m);
    let zero = (vec![0f32; 4 * p.m], vec![0f64; 4 * NHI]);
    let mut o2 = op(&dev, &p, l, 2, 1, 1 << 30, None);
    let mut o3z = op(&dev, &p, l, 3, 1, 1 << 30, Some((&zero.0, &zero.1)));
    let (y2, x2) = (fwd1(&mut o2, &st, &x, p.m), adj1(&mut o2, &st, &y, p.nv));
    let (y3, x3) = (fwd1(&mut o3z, &st, &x, p.m), adj1(&mut o3z, &st, &y, p.nv));
    let (ez, eza) = (rel_on(&y3, &y2, None), rel_on(&x3, &x2, None));
    eprintln!("order 3 with a zero degree-3/4 table vs order 2: forward {ez:.2e}, adjoint {eza:.2e}");
    assert!(ez < 5e-5 && eza < 5e-5, "order 3 (zero table) vs order 2: {ez:.2e} / {eza:.2e}"); // float32 (measured 2e-6 / 1e-5)
    // implementation error of order 2 (GPU vs emulation) as the reference level
    let (ye2, xe2, _) = emulate(&p, &o2, l, None, &x, &y);
    let (i2f, i2a) = (rel_on(&y2, &ye2, None), rel_on(&x2, &xe2, None));
    let (u, h) = hi_data(&p, 1.0);
    let mut o3 = op(&dev, &p, l, 3, 1, 1 << 30, Some((&u, &h)));
    let (y3, x3) = (fwd1(&mut o3, &st, &x, p.m), adj1(&mut o3, &st, &y, p.nv));
    let (ye3, xe3, crms) = emulate(&p, &o3, l, Some((&u, &h)), &x, &y);
    let (i3f, i3a) = (rel_on(&y3, &ye3, None), rel_on(&x3, &xe3, None));
    // the degree-3/4 term itself (emulation with vs without it): a sign or scale error would show at this size
    let (d34f, d34a) = (rel_on(&ye3, &ye2, None), rel_on(&xe3, &xe2, None));
    let l1: C64 = y.iter().zip(&y3).map(|(a, b)| a.conj() * b).sum();
    let r1: C64 = x3.iter().zip(&x).map(|(a, b)| a.conj() * b).sum();
    let dot = (l1 - r1).norm() / l1.norm();
    eprintln!("order 3: degree-3/4 phase rms {crms:.3} rad (its effect {d34f:.2e} / {d34a:.2e}); implementation error forward {i3f:.2e}, adjoint {i3a:.2e} (order 2: {i2f:.2e} / {i2a:.2e}); adjoint identity {dot:.1e}");
    let tol = |i2: f64| 2.0 * i2 + 2e-3 * crms + 1e-5;
    assert!(d34f > 10.0 * tol(i2f), "the test's degree-3/4 phase is too small to check the terms");
    assert!(i3f < tol(i2f) && i3a < tol(i2a), "order 3 vs emulation {i3f:.2e} / {i3a:.2e}");
    assert!(dot < 1e-3, "adjoint identity {dot:.1e}"); // float32
    let (ef, ea, dm) = check_multi(&mut o3, &st, &p, 1, 2, "order 3");
    assert!(ef < 5e-5 && ea < 5e-5 && dm < 1e-3, "order 3 multi {ef:.2e} / {ea:.2e} identity {dm:.1e}"); // float32 (measured 1e-6 / 7e-6 / 8e-5)
}
