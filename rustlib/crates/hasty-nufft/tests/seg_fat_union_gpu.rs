//! GPU test of the per-frame fat path with the union SVD basis (`--seg-basis svdn-union`,
//! research/recon_physics_wiring.md item 3): `build_segments_svd_fat_union[_g]` → `SegNufft::segments_fat` +
//! `fat_maps` (one map per cardiac frame) → `forward_fat` / `adjoint_fat` with the frame's map, against the f64 direct
//! sum of the exact signal
//!   y_c(j) = Σ_q S_c(q) x(q) e^{−ik_j·(q − n/2)} g_q(τ_j) [(1 − f_q) + f_q c(τ_j)],   g_q(τ) = e^{−z_q (τ − τ0)} e^{−iω_q τ}
//! (six-peak fat c(τ), per-frame fat fraction f; z = 0 without, or a per-voxel complex rate R2* + iΔω: item 4).
//! Each case also evaluates the CPU emulation (f64) of the operator's own approximation, Σ_l a_l(τ) P_l (1 + f (c_l − 1)),
//! so the error splits into "model" (segments; emulation vs exact) and "impl" (GPU vs emulation: NUFFT, f32). Plus the
//! adjoint identity of the GPU operator.
//! Centre-out 9.5 ms readouts (the v2 window), ±200 Hz smooth field, a fat shell with mixed voxels whose fraction moves
//! between three frames.

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments_svd_fat_union, build_segments_svd_fat_union_g, seg_time_grid, SegNufft, SegNufftOpts};
use hasty_testkit::physops::{interp_table, rel_on, DirectSum};
use hasty_testkit::{Rng, C64};

const PI: f64 = std::f64::consts::PI;
const TE: f64 = 1.6e-3;
const T_RO: f64 = 9.5e-3;

fn fat_c(t: f64) -> C64 {
    let a = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048];
    let s: f64 = a.iter().sum();
    let ppm = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60];
    (0..6).map(|p| C64::from_polar(a[p] / s, -2.0 * PI * 42.577478e6 * 3.0 * ppm[p] * 1e-6 * t)).sum()
}

fn c32(v: &[C64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}
fn c64(v: &[C32]) -> Vec<C64> {
    v.iter().map(|z| C64::new(z.re as f64, z.im as f64)).collect()
}

struct Case {
    dims: [usize; 3],
    nv: usize,
    nc: usize,
    idx: Vec<[f64; 3]>,
    smaps: Vec<C64>,
    x: Vec<C64>,
    om: Vec<f64>,
    /// complex rate per voxel (s⁻¹; re = R2*, im = Δω)
    z: Vec<C64>,
    /// fat fraction per frame (frames × nv)
    ff: Vec<Vec<f64>>,
    k: [Vec<f64>; 3],
    tau: Vec<f64>,
}

fn case(with_rate: bool) -> Case {
    let dims = [24usize, 22, 16];
    let res = 3e-3;
    let nv = dims.iter().product::<usize>();
    let nc = 4;
    let idx: Vec<[f64; 3]> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [i as f64 - (dims[0] / 2) as f64, j as f64 - (dims[1] / 2) as f64, k as f64 - (dims[2] / 2) as f64]
    }).collect();
    let pos: Vec<[f64; 3]> = idx.iter().map(|p| p.map(|v| v * res)).collect();
    let ext = [0, 1, 2].map(|a| dims[a] as f64 * res);
    let mut r = Rng::new(41);
    let x = r.complex_vec(nv);
    let mut smaps = Vec::with_capacity(nc * nv);
    for c in 0..nc {
        let ang = 2.0 * PI * c as f64 / nc as f64;
        let ctr = [0.04 * ang.cos(), 0.04 * ang.sin(), 0.0];
        for p in &pos {
            let d2 = (p[0] - ctr[0]).powi(2) + (p[1] - ctr[1]).powi(2) + 0.5 * p[2].powi(2);
            smaps.push(C64::from_polar((-d2 / 0.0025).exp() + 0.2, 40.0 * (c as f64 + 1.0) * p[0] - 25.0 * p[1] + 10.0 * c as f64 * p[2]));
        }
    }
    let om: Vec<f64> = pos.iter().map(|p| 2.0 * PI * (150.0 * (2.0 * PI * p[0] / ext[0]).sin() * (2.0 * PI * p[1] / ext[1]).cos() + 50.0 * 2.0 * p[2] / ext[2])).collect();
    // complex rate: R2* 20–60 s⁻¹, Δω ±2π·40 Hz (the signal-weighted frequency offset of research/readout_physics_budget.md T4)
    let z: Vec<C64> = pos.iter().map(|p| if with_rate { C64::new(40.0 + 20.0 * (p[0] / ext[0] * 9.0).sin(), 2.0 * PI * 40.0 * (p[1] / ext[1] * 7.0).cos()) } else { C64::new(0.0, 0.0) }).collect();
    // fat shell (ellipsoid radius² 0.45–0.8) whose fraction moves between frames (heart-like), mixed voxels at the edges
    let ff: Vec<Vec<f64>> = (0..3).map(|fr| pos.iter().map(|p| {
        let sh = 0.02 * fr as f64;
        let r2 = (p[0] / (0.5 * ext[0]) - sh).powi(2) + (p[1] / (0.5 * ext[1])).powi(2) + (p[2] / (0.5 * ext[2])).powi(2);
        let a = ((r2 - 0.45) / 0.08).clamp(0.0, 1.0) * ((0.8 - r2) / 0.08).clamp(0.0, 1.0);
        a * (0.7 + 0.3 * fr as f64 / 2.0)
    }).collect()).collect();
    // centre-out 3D radial-like readouts over 9.5 ms (k rounded to f32 so both sides see the same samples)
    let (nro, na) = (12usize, 400usize);
    let mut k: [Vec<f64>; 3] = Default::default();
    let mut tau = Vec::with_capacity(nro * na);
    for _ in 0..nro {
        // Rng::uniform is in [−1, 1)
        let (th, ph) = (r.uniform().acos(), PI * r.uniform());
        let d = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
        for n in 0..na {
            let s = n as f64 / (na - 1) as f64;
            let rad = 0.97 * PI * s;
            let tw = 0.3 * (6.0 * PI * s).sin() * s;
            k[0].push((rad * d[0] + tw * d[1]) as f32 as f64);
            k[1].push((rad * d[1] - tw * d[0]) as f32 as f64);
            k[2].push((rad * d[2]) as f32 as f64);
            tau.push((TE + T_RO * s) as f32 as f64);
        }
    }
    Case { dims, nv, nc, idx, smaps, x, om, z, ff, k, tau }
}

fn run(with_rate: bool) {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let g = case(with_rate);
    let (nv, nc, m) = (g.nv, g.nc, g.tau.len());
    let na = 400;
    let vox: Vec<usize> = (0..nv).collect();
    let ds = DirectSum { nv, nc, m, frames: 1, smaps: &g.smaps, voxels: &vox, bw: None };
    let gq = |q: usize, t: f64| (-g.z[q] * (t - TE)).exp() * C64::from_polar(1.0, -g.om[q] * t);
    let yr = Rng::new(9).complex_vec(nc * m);
    // union basis: every voxel fitted, water weight (1 − f_min)², fat weight f_max² (recon_p2np union_weights)
    let tau_a: Vec<f64> = g.tau[..na].to_vec();
    let (ts, tw) = seg_time_grid(&tau_a, &vec![1.0; na], 96);
    let fit: Vec<usize> = (0..nv).collect();
    let fmin: Vec<f64> = (0..nv).map(|q| g.ff.iter().map(|f| f[q]).fold(1.0, f64::min)).collect();
    let fmax: Vec<f64> = (0..nv).map(|q| g.ff.iter().map(|f| f[q]).fold(0.0, f64::max)).collect();
    let ww: Vec<f64> = fmin.iter().map(|f| (1.0 - f).powi(2) + 1e-6).collect();
    let wf: Vec<f64> = fmax.iter().map(|f| f * f + 1e-6).collect();
    let ffm: Vec<f32> = g.ff.iter().flat_map(|f| f.iter().map(|&v| v as f32)).collect();
    let mut prev = f64::MAX;
    for nseg in [8usize, 12, 16] {
        let (sp, at, cl, t0, dt, _nodes, cond) = if with_rate {
            build_segments_svd_fat_union_g(nv, nseg, tau_a[0], tau_a[na - 1], na, &fit, &ww, &wf, &ts, &tw, &gq, &fat_c)
        } else {
            build_segments_svd_fat_union(nv, nseg, tau_a[0], tau_a[na - 1], na, &fit, &ww, &wf, &ts, &tw, &|q, t| g.om[q] * t, &fat_c)
        };
        let mut op = SegNufft::new(&dev, g.dims, [3e-3; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order: 0, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
        op.set_maps(&st, &c32(&g.smaps)).unwrap();
        let seg = op.segments_fat(&st, &sp, &at, &cl, t0, dt).unwrap();
        let maps = op.fat_maps(&st, &ffm).unwrap();
        let co: [Vec<f32>; 3] = [0, 1, 2].map(|a| g.k[a].iter().map(|&v| v as f32).collect());
        let tau32: Vec<f32> = g.tau.iter().map(|&v| v as f32).collect();
        let pts = op.points(&st, [&co[0], &co[1], &co[2]], &tau32, &vec![0f32; m], &[], None).unwrap();
        let atab = c64(&at);
        let spd = c64(&sp);
        let cld = c64(&cl);
        let (dx, dy) = (st.clone_htod(&c32(&g.x)).unwrap(), st.clone_htod(&c32(&yr)).unwrap());
        for fr in 0..g.ff.len() {
            let ff = &g.ff[fr];
            let exact = |j: usize, q: usize| {
                let p = &g.idx[q];
                let t = g.tau[j];
                C64::from_polar(1.0, -(g.k[0][j] * p[0] + g.k[1][j] * p[1] + g.k[2][j] * p[2])) * gq(q, t) * (C64::new(1.0 - ff[q], 0.0) + ff[q] * fat_c(t))
            };
            // the operator's own approximation (f64): Σ_l a_l(τ) P_l(q) (1 + f (c_l − 1)); the GPU applies the maps in f32
            let emul = |j: usize, q: usize| {
                let p = &g.idx[q];
                let a = interp_table(&atab, nseg, t0, dt, g.tau[j]);
                let f = ff[q] as f32 as f64;
                let s: C64 = (0..nseg).map(|l| a[l] * spd[l * nv + q] * (1.0 + f * (cld[l] - 1.0))).sum();
                C64::from_polar(1.0, -(g.k[0][j] * p[0] + g.k[1][j] * p[1] + g.k[2][j] * p[2])) * s
            };
            let (ye, xe) = ds.apply(&exact, &g.x, &yr);
            let (ym, xm) = ds.apply(&emul, &g.x, &yr);
            let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
            let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
            op.forward_fat(&st, &seg, &pts, &maps, Some(fr), &dx, &mut y).unwrap();
            op.adjoint_fat(&st, &seg, &pts, &maps, Some(fr), &dy, &mut xa).unwrap();
            let (y, xa) = (c64(&st.clone_dtoh(&y).unwrap()), c64(&st.clone_dtoh(&xa).unwrap()));
            let (ef, ea) = (rel_on(&y, &ye, None), rel_on(&xa, &xe, None));
            let (mf, ma) = (rel_on(&ym, &ye, None), rel_on(&xm, &xe, None));
            let (gf, ga) = (rel_on(&y, &ym, None), rel_on(&xa, &xm, None));
            let l: C64 = yr.iter().zip(&y).map(|(a, b)| a.conj() * b).sum();
            let r: C64 = xa.iter().zip(&g.x).map(|(a, b)| a.conj() * b).sum();
            // normalised by ‖y‖ ‖Ax‖ (random y: ⟨y, Ax⟩ itself is a sum of random phases); the raw ratio printed too
            let ny = yr.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
            let nax = y.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
            let eadj = (l - r).norm() / (ny * nax);
            let eadj_raw = (l - r).norm() / l.norm();
            eprintln!("{} L {nseg:>2} frame {fr} (cond {cond:.1}): GPU vs exact fwd {ef:.2e} adj {ea:.2e} | model {mf:.2e} / {ma:.2e} | impl {gf:.2e} / {ga:.2e} | adjoint identity {eadj:.1e} (of |<y, Ax>|: {eadj_raw:.1e})",
                      if with_rate { "complex rate" } else { "water + fat " });
            // implementation (NUFFT interpolation, f32): as the plain operator (seg_b0_direct: 8.6e-5 / 2.8e-4)
            assert!(gf < 5e-4 && ga < 3e-3, "impl error {gf:.2e} / {ga:.2e}");
            assert!(eadj < 1e-5, "adjoint identity {eadj:.1e}");
            if fr == 0 {
                assert!(mf < prev * 1.05, "model error not decreasing with L: {mf:.2e} after {prev:.2e}");
                prev = mf;
            }
            if nseg == 16 {
                assert!(mf < 2e-3 && ef < 3e-3, "L 16 model error {mf:.2e}, total {ef:.2e}");
            }
        }
    }
}

#[test]
fn fat_union_per_frame_maps_match_direct_sum() {
    run(false);
}

#[test]
fn fat_union_with_complex_rates_matches_direct_sum() {
    run(true);
}
