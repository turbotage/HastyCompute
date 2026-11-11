//! GPU tests of the readout-motion warp in SegNufft (`set_warp`, hasty_nufft::segwarp).
//!
//! * adjoint consistency with the warp on (static and temporal frames);
//! * the GPU correction (forward with warp − forward without) equals the CPU direct sum of the correction images;
//! * `set_warp(None)` restores the plain operator bit for bit;
//! * physics: a moving vessel in an off-resonant object, centre-out readouts: the warped operator against the f64 direct
//!   sum of moving point voxels.
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments_svd_grid, cardinal_form, seg_time_grid, SegNufft, SegNufftOpts, SegPoints, SegSegments, SegWarp, WarpStencil};
use num_complex::Complex64;

fn c64(z: C32) -> Complex64 {
    Complex64::new(z.re as f64, z.im as f64)
}
fn c32(z: Complex64) -> C32 {
    C32::new(z.re as f32, z.im as f32)
}

struct Case {
    dims: [usize; 3],
    vox: f64,
    nc: usize,
    om: Vec<f64>,
    x: Vec<Complex64>,
    vel: Vec<[f64; 3]>,
    smaps: Vec<Complex64>,
    // samples: k in rad/voxel, tau (s), t1 (s after the echo)
    k: [Vec<f64>; 3],
    tau: Vec<f64>,
    t1: Vec<f64>,
    te: f64,
    na: usize,
}

fn case() -> Case {
    let dims = [24usize, 24, 20];
    let vox = 2e-3;
    let nv = dims.iter().product::<usize>();
    let nc = 3;
    let (te, t_ro, na, nr) = (1.6e-3, 9.5e-3, 190usize, 40usize);
    let axis = {
        let a = [0.6f64, 0.3, 0.74];
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let mut c = Case { dims, vox, nc, om: vec![0.0; nv], x: vec![Complex64::new(0.0, 0.0); nv], vel: vec![[0.0; 3]; nv], smaps: vec![], k: [vec![], vec![], vec![]], tau: vec![], t1: vec![], te, na };
    let pos = |q: usize| [((q % dims[0]) as f64 - 12.0) * vox, (((q / dims[0]) % dims[1]) as f64 - 12.0) * vox, ((q / (dims[0] * dims[1])) as f64 - 10.0) * vox];
    for q in 0..nv {
        let p = pos(q);
        let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        if r2 > 0.02f64.powi(2) {
            continue;
        }
        c.om[q] = 2.0 * std::f64::consts::PI * (120.0 * (p[0] / 0.02) + 60.0 * (p[1] * p[2] / 4e-4));
        c.x[q] = Complex64::new(1.0 + 0.3 * (37.0 * p[0]).sin(), 0.2 * (23.0 * p[1]).cos());
        let s = p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2];
        let d2 = r2 - s * s;
        if d2 < 0.004f64.powi(2) {
            let sp = 1.0 - d2 / 0.004f64.powi(2);
            c.vel[q] = [sp * axis[0], sp * axis[1], sp * axis[2]];
            c.x[q] = Complex64::new(2.5, 0.0);
        }
    }
    for ci in 0..nc {
        for q in 0..nv {
            let p = pos(q);
            c.smaps.push(Complex64::from_polar((-(p[0] - 0.02 * ci as f64).powi(2) / 2e-3).exp(), 30.0 * (ci as f64 + 1.0) * p[0] + 20.0 * p[1]));
        }
    }
    for r in 0..nr {
        let phi = std::f64::consts::PI * (3.0 - 5f64.sqrt()) * r as f64;
        let z = 1.0 - 2.0 * (r as f64 + 0.5) / nr as f64;
        let rho = (1.0 - z * z).sqrt();
        let d0 = [rho * phi.cos(), rho * phi.sin(), z];
        let a = if d0[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
        let e = [d0[1] * a[2] - d0[2] * a[1], d0[2] * a[0] - d0[0] * a[2], d0[0] * a[1] - d0[1] * a[0]];
        let en = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
        for n in 0..na {
            let t = t_ro * n as f64 / (na - 1) as f64;
            let kr = std::f64::consts::PI * 0.98 * (t / t_ro).powf(0.4);
            let ang = 0.6 * t / t_ro;
            for ax in 0..3 {
                c.k[ax].push(kr * (d0[ax] * ang.cos() + e[ax] / en * ang.sin()));
            }
            c.tau.push(te + t);
            c.t1.push(t);
        }
    }
    c
}

fn rho(c: &Case, q: usize) -> [f64; 3] {
    let d = c.dims;
    [(q % d[0]) as f64 - (d[0] / 2) as f64, ((q / d[0]) % d[1]) as f64 - (d[1] / 2) as f64, (q / (d[0] * d[1])) as f64 - (d[2] / 2) as f64]
}

/// segments (svd, cardinal form, one table entry per ADC time), engine, points, warp of the vessel at the nodes
fn build(dev: &Device, c: &Case, nseg: usize, frames: usize, st: WarpStencil) -> (SegNufft, SegSegments, SegPoints, Vec<C32>, Vec<C32>, f64, f64, SegWarp) {
    let s = dev.default_stream();
    let nv = c.om.len();
    let tau_a: Vec<f64> = (0..c.na).map(|n| c.tau[n]).collect();
    let (ts, q) = seg_time_grid(&tau_a, &vec![1.0; c.na], 96);
    let fit: Vec<usize> = (0..nv).filter(|&v| c.x[v].norm() > 0.0).collect();
    let g = |v: usize, t: f64| Complex64::from_polar(1.0, -c.om[v] * t);
    let (segph, atab, t0, dt) = build_segments_svd_grid(nv, nseg, tau_a[0], tau_a[c.na - 1], c.na, &fit, None, &ts, &q, &g);
    let (segph, atab, nodes, _cond) = cardinal_form(&segph, &atab, nseg, nv);
    let mut op = SegNufft::new(dev, c.dims, [c.vox; 3], c.nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order: 2, frames, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
    op.set_maps(&s, &c.smaps.iter().map(|&z| c32(z)).collect::<Vec<_>>()).unwrap();
    let seg = op.segments(&s, &segph, &atab, t0, dt).unwrap();
    let m = c.tau.len();
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|a| c.k[a].iter().map(|&v| v as f32).collect());
    let tau: Vec<f32> = c.tau.iter().map(|&v| v as f32).collect();
    let tt: Vec<f32> = (0..m).map(|j| ((j / c.na) % frames) as f32 + 0.5).collect();
    let pts = op.points(&s, [&coords[0], &coords[1], &coords[2]], &tau, &vec![0f32; m], &vec![0f32; 6 * m], (frames > 1).then_some(&tt[..])).unwrap();
    let moving: Vec<u32> = (0..nv).filter(|&q| c.vel[q] != [0.0; 3]).map(|q| q as u32).collect();
    let vh: Vec<[f64; 3]> = moving.iter().map(|&q| c.vel[q as usize]).collect();
    let t1: Vec<f64> = nodes.iter().map(|&e| t0 + e as f64 * dt - c.te).collect();
    let w = SegWarp::new(c.dims, [c.vox; 3], moving, &vh, None, &t1, &vec![0.0; nseg], st);
    (op, seg, pts, segph, atab, t0, dt, w)
}

fn dot(a: &[C32], b: &[C32]) -> Complex64 {
    a.iter().zip(b).map(|(p, q)| c64(*p).conj() * c64(*q)).sum()
}

#[test]
fn warp_adjoint_gpu() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let c = case();
    let nv = c.om.len();
    // the plain operator's own adjoint mismatch on this case varies run to run (f32 atomics of the tile kernels):
    // envelope over 4 runs; the warped operator must stay inside 3x that envelope
    for frames in [1usize, 3] {
        let (mut op, seg, pts, _, _, _, _, w) = build(&dev, &c, 8, frames, WarpStencil::Cubic);
        let m = c.tau.len();
        let xr: Vec<C32> = (0..frames * nv).map(|i| C32::new(((i * 7919) % 101) as f32 / 50.0 - 1.0, ((i * 104729) % 97) as f32 / 48.0 - 1.0)).collect();
        let yr: Vec<C32> = (0..c.nc * m).map(|i| C32::new(((i * 31) % 89) as f32 / 44.0 - 1.0, ((i * 17) % 83) as f32 / 41.0 - 1.0)).collect();
        let (dx, dy) = (s.clone_htod(&xr).unwrap(), s.clone_htod(&yr).unwrap());
        let run = |op: &mut SegNufft| -> f64 {
            let mut ax = s.alloc_zeros::<C32>(c.nc * m).unwrap();
            let mut ahy = s.alloc_zeros::<C32>(frames * nv).unwrap();
            op.forward(&s, &seg, &pts, &dx, &mut ax).unwrap();
            op.adjoint(&s, &seg, &pts, &dy, &mut ahy).unwrap();
            let (ax, ahy) = (s.clone_dtoh(&ax).unwrap(), s.clone_dtoh(&ahy).unwrap());
            let (l, r) = (dot(&yr, &ax), dot(&ahy, &xr));
            (l - r).norm() / l.norm()
        };
        let plain = (0..4).map(|_| run(&mut op)).fold(0.0f64, f64::max);
        op.set_warp(&s, Some(&vec![w.clone(); frames])).unwrap();
        let warped = (0..4).map(|_| run(&mut op)).fold(0.0f64, f64::max);
        eprintln!("frames {frames}: adjoint mismatch plain {plain:.1e}, warped {warped:.1e} (max of 4 runs)");
        assert!(warped < 3.0 * plain + 1e-5, "warp adjoint mismatch {warped} vs plain {plain}");
    }
}

#[test]
fn warp_adjoint_correction_matches_cpu() {
    // (A_w^H - A^H) y on the GPU against the CPU: x̄ = sum_l (W_l - I)^T conj(D'_l) r_l, r_l(q) = sum_j conj(a_l(tau_j))
    // sum_c conj(S_c(q)) e^{+i k_j . rho_q} y_cj (type-1 direct sum at the footprint points)
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let c = case();
    let nv = c.om.len();
    let nseg = 8;
    let (mut op, seg, pts, segph, atab, _, _, w) = build(&dev, &c, nseg, 1, WarpStencil::Lanczos3);
    let gw: Vec<[f64; 3]> = (0..w.moving.len()).map(|i| [3e4 * ((i % 7) as f64 - 3.0), 2e4, -1e4 * ((i % 5) as f64)]).collect();
    let vh: Vec<[f64; 3]> = w.moving.iter().map(|&q| c.vel[q as usize]).collect();
    let tn: Vec<f64> = (0..nseg).map(|l| 9.5e-3 * l as f64 / (nseg - 1) as f64).collect();
    let w = w.path_field(&gw, &vh, &tn);
    let m = c.tau.len();
    let yr: Vec<C32> = (0..c.nc * m).map(|i| C32::new(((i * 31) % 89) as f32 / 44.0 - 1.0, ((i * 17) % 83) as f32 / 41.0 - 1.0)).collect();
    let dy = s.clone_htod(&yr).unwrap();
    let mut a0 = s.alloc_zeros::<C32>(nv).unwrap();
    op.adjoint(&s, &seg, &pts, &dy, &mut a0).unwrap();
    op.set_warp(&s, Some(std::slice::from_ref(&w))).unwrap();
    let mut a1 = s.alloc_zeros::<C32>(nv).unwrap();
    op.adjoint(&s, &seg, &pts, &dy, &mut a1).unwrap();
    let (a0, a1) = (s.clone_dtoh(&a0).unwrap(), s.clone_dtoh(&a1).unwrap());
    let nf = w.footprint.len();
    let mut r = vec![C32::new(0.0, 0.0); nseg * nf];
    for (fi, &q) in w.footprint.iter().enumerate() {
        let rq = rho(&c, q as usize);
        let mut acc = vec![Complex64::new(0.0, 0.0); nseg];
        for j in 0..m {
            let n = j % c.na;
            let ph = Complex64::from_polar(1.0, c.k[0][j] * rq[0] + c.k[1][j] * rq[1] + c.k[2][j] * rq[2]);
            let mut sy = Complex64::new(0.0, 0.0);
            for ci in 0..c.nc {
                sy += c.smaps[ci * nv + q as usize].conj() * c64(yr[ci * m + j]);
            }
            let z = sy * ph;
            for l in 0..nseg {
                acc[l] += c64(atab[n * nseg + l]).conj() * z;
            }
        }
        for l in 0..nseg {
            r[l * nf + fi] = c32(acc[l]);
        }
    }
    let mut xb = vec![C32::new(0.0, 0.0); nv];
    w.adjoint_cpu(&segph, &r, &mut xb);
    let (mut num, mut den) = (0.0, 0.0);
    for v in 0..nv {
        let got = c64(a1[v]) - c64(a0[v]);
        num += (got - c64(xb[v])).norm_sqr();
        den += c64(xb[v]).norm_sqr();
    }
    let e = (num / den).sqrt();
    eprintln!("GPU adjoint warp correction vs CPU: relative error {e:.2e}");
    assert!(e < 2e-3, "error {e}");
}

#[test]
fn warp_correction_matches_cpu() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let c = case();
    let nv = c.om.len();
    let nseg = 8;
    let (mut op, seg, pts, segph, atab, _t0, _dt, w) = build(&dev, &c, nseg, 1, WarpStencil::Cubic);
    // with the path-field phase (motion through a B0 gradient) of the pushed values
    let gw: Vec<[f64; 3]> = (0..w.moving.len()).map(|i| [3e4 * ((i % 7) as f64 - 3.0), 2e4, -1e4 * ((i % 5) as f64)]).collect();
    let vh: Vec<[f64; 3]> = w.moving.iter().map(|&q| c.vel[q as usize]).collect();
    let tn: Vec<f64> = (0..nseg).map(|l| 9.5e-3 * l as f64 / (nseg - 1) as f64).collect();
    let w = w.path_field(&gw, &vh, &tn);
    let x32: Vec<C32> = c.x.iter().map(|&z| c32(z)).collect();
    let dx = s.clone_htod(&x32).unwrap();
    let m = c.tau.len();
    let mut y0 = s.alloc_zeros::<C32>(c.nc * m).unwrap();
    op.forward(&s, &seg, &pts, &dx, &mut y0).unwrap();
    let y0 = s.clone_dtoh(&y0).unwrap();
    op.set_warp(&s, Some(std::slice::from_ref(&w))).unwrap();
    let mut y1 = s.alloc_zeros::<C32>(c.nc * m).unwrap();
    op.forward(&s, &seg, &pts, &dx, &mut y1).unwrap();
    let y1 = s.clone_dtoh(&y1).unwrap();
    // set_warp(None): bit-identical to the plain operator
    op.set_warp(&s, None).unwrap();
    let mut y2 = s.alloc_zeros::<C32>(c.nc * m).unwrap();
    op.forward(&s, &seg, &pts, &dx, &mut y2).unwrap();
    let y2 = s.clone_dtoh(&y2).unwrap();
    let mut y3 = s.alloc_zeros::<C32>(c.nc * m).unwrap();
    op.forward(&s, &seg, &pts, &dx, &mut y3).unwrap();
    let y3 = s.clone_dtoh(&y3).unwrap();
    let diff = |a: &[C32], b: &[C32]| (a.iter().zip(b).map(|(p, q)| (c64(*p) - c64(*q)).norm_sqr()).sum::<f64>() / a.iter().map(|p| c64(*p).norm_sqr()).sum::<f64>()).sqrt();
    eprintln!("plain operator run-to-run {:.1e} (atomic summation order), after set_warp(None) {:.1e}", diff(&y2, &y3), diff(&y0, &y2));
    assert!(diff(&y0, &y2) <= 10.0 * diff(&y2, &y3).max(1e-7), "set_warp(None) changed the operator");
    // CPU: sum_l a_l(tau_j) sum_q S_c(q) du_l(q) e^{-i k_j . rho_q} (sample times on the table entries)
    let du = w.forward_cpu(&segph, &x32);
    let nf = w.footprint.len();
    let (mut num, mut den) = (0.0, 0.0);
    for j in (0..m).step_by(5) {
        let n = j % c.na;
        let a: Vec<Complex64> = (0..nseg).map(|l| c64(atab[n * nseg + l])).collect();
        for ci in 0..c.nc {
            let mut acc = Complex64::new(0.0, 0.0);
            for (fi, &q) in w.footprint.iter().enumerate() {
                let r = rho(&c, q as usize);
                let ph = Complex64::from_polar(1.0, -(c.k[0][j] * r[0] + c.k[1][j] * r[1] + c.k[2][j] * r[2]));
                let mut z = Complex64::new(0.0, 0.0);
                for l in 0..nseg {
                    z += a[l] * c64(du[l * nf + fi]);
                }
                acc += c.smaps[ci * nv + q as usize] * z * ph;
            }
            let got = c64(y1[ci * m + j]) - c64(y0[ci * m + j]);
            num += (got - acc).norm_sqr();
            den += acc.norm_sqr();
        }
    }
    let e = (num / den).sqrt();
    eprintln!("GPU warp correction vs CPU direct sum: relative error {e:.2e} (|V| {}, footprint {nf})", w.moving.len());
    assert!(e < 2e-3, "error {e}");
}

#[test]
fn warp_moving_vessel_gpu() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let c = case();
    let nv = c.om.len();
    let m = c.tau.len();
    // f64 truth: moving point voxels, exact B0, coil maps
    let mut yt = vec![Complex64::new(0.0, 0.0); c.nc * m];
    let mut ys = vec![Complex64::new(0.0, 0.0); c.nc * m];
    for j in 0..m {
        for q in 0..nv {
            if c.x[q].norm() == 0.0 {
                continue;
            }
            let r = rho(&c, q);
            let d = c.vel[q];
            let lin = c.k[0][j] * r[0] + c.k[1][j] * r[1] + c.k[2][j] * r[2];
            let sh = (c.k[0][j] * d[0] + c.k[1][j] * d[1] + c.k[2][j] * d[2]) * c.t1[j] / c.vox;
            let e = c.x[q] * Complex64::from_polar(1.0, -(lin + c.om[q] * c.tau[j]));
            for ci in 0..c.nc {
                yt[ci * m + j] += c.smaps[ci * nv + q] * e * Complex64::from_polar(1.0, -sh);
                ys[ci * m + j] += c.smaps[ci * nv + q] * e;
            }
        }
    }
    let x32: Vec<C32> = c.x.iter().map(|&z| c32(z)).collect();
    let err = |y: &[C32], want: &[Complex64]| -> f64 { (y.iter().zip(want).map(|(a, b)| (c64(*a) - b).norm_sqr()).sum::<f64>() / want.iter().map(|b| b.norm_sqr()).sum::<f64>()).sqrt() };
    let mut out = vec![];
    for st in [WarpStencil::Trilinear, WarpStencil::Cubic, WarpStencil::Lanczos3] {
        let (mut op, seg, pts, _, _, _, _, w) = build(&dev, &c, 10, 1, st);
        let dx = s.clone_htod(&x32).unwrap();
        let mut y = s.alloc_zeros::<C32>(c.nc * m).unwrap();
        op.forward(&s, &seg, &pts, &dx, &mut y).unwrap();
        let yp = s.clone_dtoh(&y).unwrap();
        op.set_warp(&s, Some(std::slice::from_ref(&w))).unwrap();
        op.forward(&s, &seg, &pts, &dx, &mut y).unwrap();
        let yw = s.clone_dtoh(&y).unwrap();
        let (e_floor, e_static, e_warp) = (err(&yp, &ys), err(&yp, &yt), err(&yw, &yt));
        eprintln!("{st:?}: static operator vs static truth {e_floor:.2e}; vs moving truth: static {e_static:.4}, warped {e_warp:.4} ({:.2} of static)", e_warp / e_static);
        out.push((st, e_static, e_warp, e_floor));
    }
    for (st, es, ew, ef) in &out {
        assert!(*ef < 0.05 * es, "{st:?}: floor {ef} vs {es}");
        let lim = match st {
            WarpStencil::Trilinear => 0.8,
            WarpStencil::Cubic => 0.5,
            WarpStencil::Lanczos3 => 0.4,
        };
        assert!(ew / es < lim, "{st:?}: warped/static {}", ew / es);
    }
}

/// cost of the warp at the production grid (ph0001 2 mm: 254 × 144 × 114, 31 coils, L = 10, σ 1.25): a moving set of
/// `HASTY_WARP_NV` (default 150 000) voxels with displacements up to ~7 voxels; forward + adjoint with and without the warp
/// (cubic and Lanczos-3). Run: cargo test --release -p hasty-nufft --test seg_warp_gpu warp_cost -- --ignored --nocapture
#[test]
#[ignore]
fn warp_cost_realistic() {
    let dev = Device::new(0).unwrap();
    let s = dev.default_stream();
    let dims = [254usize, 144, 114];
    let nv = dims.iter().product::<usize>();
    let (nc, nseg) = (31usize, 10usize);
    let mut seed = 12345u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let smaps: Vec<C32> = (0..nc * nv).map(|i| { let a = ((i % 977) as f32) * 0.01; C32::new(a.cos(), a.sin()) }).collect();
    let segph: Vec<C32> = (0..nseg * nv).map(|i| { let a = ((i % 631) as f32) * 0.02; C32::new(a.cos(), a.sin()) }).collect();
    let ntab = 512;
    let atab: Vec<C32> = (0..ntab * nseg).map(|i| C32::new(((i % 7) as f32) * 0.1, 0.05)).collect();
    let mut op = SegNufft::new(&dev, dims, [2e-3; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order: 2, frames: 1, bspline: 1, max_grid_bytes: 6 << 30 }).unwrap();
    op.set_maps(&s, &smaps).unwrap();
    let seg = op.segments(&s, &segph, &atab, 1.6e-3, 9.5e-3 / (ntab - 1) as f64).unwrap();
    let nr = 1100usize;
    let na = 1200usize;
    let m = nr * na;
    let mut coords: [Vec<f32>; 3] = [vec![], vec![], vec![]];
    let mut tau = vec![];
    for _ in 0..nr {
        let z = 2.0 * rnd() - 1.0;
        let ph = 2.0 * std::f64::consts::PI * rnd();
        let r = (1.0 - z * z).sqrt();
        let d = [r * ph.cos(), r * ph.sin(), z];
        for n in 0..na {
            let t = n as f64 / (na - 1) as f64;
            let kr = std::f64::consts::PI * 0.98 * t.powf(0.4);
            for a in 0..3 {
                coords[a].push((kr * d[a]) as f32);
            }
            tau.push((1.6e-3 + 9.5e-3 * t) as f32);
        }
    }
    let pts = op.points(&s, [&coords[0], &coords[1], &coords[2]], &tau, &vec![0f32; m], &vec![0f32; 6 * m], None).unwrap();
    let x: Vec<C32> = (0..nv).map(|i| C32::new(((i % 13) as f32) * 0.1, 0.2)).collect();
    let dx = s.clone_htod(&x).unwrap();
    let mut y = s.alloc_zeros::<C32>(nc * m).unwrap();
    let mut xa = s.alloc_zeros::<C32>(nv).unwrap();
    op.autotune(&s, &seg, &pts).unwrap();
    let mut time = |op: &mut SegNufft| -> f64 {
        op.forward(&s, &seg, &pts, &dx, &mut y).unwrap();
        op.adjoint(&s, &seg, &pts, &y, &mut xa).unwrap();
        s.synchronize().unwrap();
        // the minimum of 5 single runs (robust to other GPU load)
        (0..5)
            .map(|_| {
                let t0 = std::time::Instant::now();
                op.forward(&s, &seg, &pts, &dx, &mut y).unwrap();
                op.adjoint(&s, &seg, &pts, &y, &mut xa).unwrap();
                s.synchronize().unwrap();
                t0.elapsed().as_secs_f64()
            })
            .fold(f64::MAX, f64::min)
    };
    let t_plain = time(&mut op);
    let nmov: usize = std::env::var("HASTY_WARP_NV").ok().and_then(|v| v.parse().ok()).unwrap_or(150_000);
    // a ball of moving voxels around the centre
    let c = [127.0, 72.0, 57.0];
    let mut idx: Vec<(f64, u32)> = (0..nv)
        .map(|q| {
            let p = [(q % dims[0]) as f64 - c[0], ((q / dims[0]) % dims[1]) as f64 - c[1], (q / (dims[0] * dims[1])) as f64 - c[2]];
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2], q as u32)
        })
        .collect();
    idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let moving: Vec<u32> = {
        let mut v: Vec<u32> = idx[..nmov].iter().map(|p| p.1).collect();
        v.sort_unstable();
        v
    };
    let vh: Vec<[f64; 3]> = moving.iter().map(|_| [1.5 * (rnd() - 0.5), 1.5 * (rnd() - 0.5), 1.5 * (rnd() - 0.5)]).collect();
    let t1: Vec<f64> = (0..nseg).map(|l| 9.4e-3 * l as f64 / (nseg - 1) as f64).collect();
    for st in [WarpStencil::Cubic, WarpStencil::Lanczos3] {
        let w = SegWarp::new(dims, [2e-3; 3], moving.clone(), &vh, None, &t1, &vec![0.0; nseg], st);
        op.set_warp(&s, Some(std::slice::from_ref(&w))).unwrap();
        let t_w = time(&mut op);
        op.set_warp(&s, None).unwrap();
        let t_plain = t_plain.min(time(&mut op));
        eprintln!("{st:?}: |V| {} footprint {}: forward + adjoint {:.1} ms with the warp vs {:.1} ms plain (+{:.1} ms, {:.1} %; M = {m}, L = {nseg}, {nc} coils, coil batch {})",
            w.moving.len(), w.footprint.len(), t_w * 1e3, t_plain * 1e3, (t_w - t_plain) * 1e3, 100.0 * (t_w - t_plain) / t_plain, op.cb);
        op.set_warp(&s, None).unwrap();
    }
}
