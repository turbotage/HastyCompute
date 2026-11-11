//! CPU tests of the in-readout motion warp inside the time segments (hasty_nufft::segwarp).
//!
//! * cardinal form: the QDEIM-rotated basis is the same operator (a'D' = aD) and interpolatory (a'(τ_m) = δ);
//! * stencil k-response: exact at the integer shifts, unit DC gain, the measured fall-off towards the Nyquist edge;
//! * adjoint of the correction: <ΔU x, r> = <x, ΔUᵀ r>;
//! * a moving vessel in an off-resonant object read out centre-out: the segmented operator with the warp reproduces
//!   the f64 direct sum (moving point voxels) far better than the static operator.
use hasty_nufft::{build_segments_svd_grid, cardinal_form, seg_time_grid, WarpStencil, SegWarp};
use hasty_cuda::C32;
use num_complex::Complex64;

fn c64(z: C32) -> Complex64 {
    Complex64::new(z.re as f64, z.im as f64)
}

fn interp(atab: &[C32], nseg: usize, t0: f64, dt: f64, t: f64) -> Vec<Complex64> {
    let ntab = atab.len() / nseg;
    let u = ((t - t0) / dt).clamp(0.0, (ntab - 1) as f64 - 1e-9);
    let i = u.floor() as usize;
    let f = u - i as f64;
    (0..nseg)
        .map(|l| {
            let a = atab[i * nseg + l];
            let b = atab[(i + 1).min(ntab - 1) * nseg + l];
            Complex64::new((a.re + (b.re - a.re) * f as f32) as f64, (a.im + (b.im - a.im) * f as f32) as f64)
        })
        .collect()
}

struct Toy {
    dims: [usize; 3],
    vox: f64,
    om: Vec<f64>,
    x: Vec<C32>,
    vel: Vec<[f64; 3]>,
}

fn toy() -> Toy {
    let dims = [24usize, 24, 20];
    let vox = 2e-3;
    let nv = dims[0] * dims[1] * dims[2];
    let mut om = vec![0.0; nv];
    let mut x = vec![C32::new(0.0, 0.0); nv];
    let mut vel = vec![[0.0; 3]; nv];
    let axis = {
        let a = [0.6f64, 0.3, 0.74];
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / n, a[1] / n, a[2] / n]
    };
    for q in 0..nv {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        let p = [(i as f64 - 12.0) * vox, (j as f64 - 12.0) * vox, (k as f64 - 10.0) * vox];
        let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
        if r2 > (0.02f64).powi(2) {
            continue;
        }
        // smooth off-resonance (±120 Hz) and a textured static object
        om[q] = 2.0 * std::f64::consts::PI * (120.0 * (p[0] / 0.02) + 60.0 * (p[1] * p[2] / 4e-4));
        x[q] = C32::new((1.0 + 0.3 * (37.0 * p[0]).sin()) as f32, (0.2 * (23.0 * p[1]).cos()) as f32);
        // a vessel of radius 4 mm through the centre along `axis`, parabolic profile, 1 m/s peak
        let s = p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2];
        let d2 = r2 - s * s;
        if d2 < (0.004f64).powi(2) {
            let sp = 1.0 * (1.0 - d2 / (0.004f64).powi(2));
            vel[q] = [sp * axis[0], sp * axis[1], sp * axis[2]];
            x[q] = C32::new(2.5, 0.0);
        }
    }
    Toy { dims, vox, om, x, vel }
}

/// centre-out readouts: |k| = kmax (t/T)^0.4, direction slowly turning; t1(τ) ≈ t (displacement time after the echo)
fn trajectory(nr: usize, na: usize, t_ro: f64, kmax: f64) -> Vec<Vec<([f64; 3], f64, f64)>> {
    (0..nr)
        .map(|r| {
            let phi = std::f64::consts::PI * (3.0 - 5f64.sqrt()) * r as f64;
            let z = 1.0 - 2.0 * (r as f64 + 0.5) / nr as f64;
            let rho = (1.0 - z * z).sqrt();
            let d0 = [rho * phi.cos(), rho * phi.sin(), z];
            let e1 = {
                let a = if d0[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
                let c = [d0[1] * a[2] - d0[2] * a[1], d0[2] * a[0] - d0[0] * a[2], d0[0] * a[1] - d0[1] * a[0]];
                let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                [c[0] / n, c[1] / n, c[2] / n]
            };
            (0..na)
                .map(|n| {
                    let t = t_ro * n as f64 / (na - 1) as f64;
                    let kr = kmax * (t / t_ro).powf(0.4);
                    let ang = 0.6 * t / t_ro;
                    let dir = [d0[0] * ang.cos() + e1[0] * ang.sin(), d0[1] * ang.cos() + e1[1] * ang.sin(), d0[2] * ang.cos() + e1[2] * ang.sin()];
                    ([kr * dir[0], kr * dir[1], kr * dir[2]], t, t)
                })
                .collect()
        })
        .collect()
}

#[test]
fn cardinal_form_is_the_same_operator() {
    let tw = toy();
    let nv = tw.om.len();
    let na = 400;
    let t_ro = 9.5e-3;
    let tau: Vec<f64> = (0..na).map(|n| t_ro * n as f64 / (na - 1) as f64).collect();
    let w = vec![1.0; na];
    let (ts, q) = seg_time_grid(&tau, &w, 96);
    let fit: Vec<usize> = (0..nv).filter(|&v| c64(tw.x[v]).norm() > 0.0).collect();
    let g = |v: usize, t: f64| Complex64::from_polar(1.0, -tw.om[v] * t);
    let nseg = 7;
    let (segph, atab, t0, dt) = build_segments_svd_grid(nv, nseg, 0.0, t_ro, na, &fit, None, &ts, &q, &g);
    let (segph2, atab2, nodes, cond) = cardinal_form(&segph, &atab, nseg, nv);
    assert!(cond < 1e4, "condition {cond}");
    let mut worst: f64 = 0.0;
    for (n, &t) in tau.iter().enumerate().step_by(7) {
        let a = interp(&atab, nseg, t0, dt, t);
        let b = interp(&atab2, nseg, t0, dt, t);
        for &v in fit.iter().step_by(13) {
            let za: Complex64 = (0..nseg).map(|l| a[l] * c64(segph[l * nv + v])).sum();
            let zb: Complex64 = (0..nseg).map(|l| b[l] * c64(segph2[l * nv + v])).sum();
            worst = worst.max((za - zb).norm());
        }
        let _ = n;
    }
    assert!(worst < 2e-5 * cond.max(1.0), "a'D' vs aD differ by {worst}");
    for (m, &e) in nodes.iter().enumerate() {
        for l in 0..nseg {
            let z = atab2[e * nseg + l];
            let want = if l == m { 1.0 } else { 0.0 };
            assert!((z.re as f64 - want).abs() < 1e-4 && (z.im as f64).abs() < 1e-4, "a'_{l}(node {m}) = {z:?}");
        }
    }
    println!("cardinal form: L {nseg}, nodes (ms) {:?}, cond {cond:.2}, max |a'D' − aD| {worst:.2e}", nodes.iter().map(|&e| (t0 + e as f64 * dt) * 1e3).collect::<Vec<_>>());
}

#[test]
fn stencil_response() {
    for s in [WarpStencil::Trilinear, WarpStencil::Cubic, WarpStencil::Lanczos3] {
        // integer shifts are exact, constants are pushed exactly
        for th in [0.3, 1.0, 2.0, 3.0] {
            assert!((s.response(th, 0.0) - 1.0).norm() < 1e-12, "{s:?} at an integer shift");
        }
        for fr in [0.1, 0.37, 0.5, 0.8] {
            assert!((s.response(0.0, fr) - 1.0).norm() < 1e-12, "{s:?} DC gain");
        }
        let worst = |th: f64| (0..=20).map(|i| (s.response(th, i as f64 / 20.0) - 1.0).norm()).fold(0.0f64, f64::max);
        println!("{s:?}: max_frac |response − 1| at k_a Δ = π/4, π/2, 3π/4, 0.9π: {:.3} {:.3} {:.3} {:.3}", worst(0.25 * std::f64::consts::PI), worst(0.5 * std::f64::consts::PI), worst(0.75 * std::f64::consts::PI), worst(0.9 * std::f64::consts::PI));
    }
    let w = |s: WarpStencil, th: f64| (0..=20).map(|i| (s.response(th, i as f64 / 20.0) - 1.0).norm()).fold(0.0f64, f64::max);
    let h = 0.5 * std::f64::consts::PI;
    assert!(w(WarpStencil::Cubic, h) < 0.5 * w(WarpStencil::Trilinear, h));
    assert!(w(WarpStencil::Lanczos3, h) < w(WarpStencil::Cubic, h));
}

#[test]
fn warp_adjoint() {
    let tw = toy();
    let nv = tw.om.len();
    let moving: Vec<u32> = (0..nv).filter(|&q| tw.vel[q] != [0.0; 3]).map(|q| q as u32).collect();
    let vh: Vec<[f64; 3]> = moving.iter().map(|&q| tw.vel[q as usize]).collect();
    let t1 = [0.0, 1.3e-3, 4.0e-3, 9.0e-3];
    let t2 = [0.0, 1e-6, 8e-6, 4e-5];
    let ah: Vec<[f64; 3]> = vh.iter().map(|v| [20.0 * v[1], -15.0 * v[0], 5.0]).collect();
    let nseg = t1.len();
    let segph: Vec<C32> = (0..nseg * nv).map(|i| { let z = Complex64::from_polar(1.0, 0.37 * (i % 97) as f64); C32::new(z.re as f32, z.im as f32) }).collect();
    let gw: Vec<[f64; 3]> = vh.iter().map(|v| [2e4 * v[2], -1.5e4, 3e4 * v[0]]).collect();
    for (s, path) in [(WarpStencil::Trilinear, false), (WarpStencil::Cubic, false), (WarpStencil::Lanczos3, false), (WarpStencil::Cubic, true)] {
        let w = SegWarp::new(tw.dims, [tw.vox; 3], moving.clone(), &vh, Some(&ah), &t1, &t2, s);
        let w = if path { w.path_field(&gw, &vh, &t1) } else { w };
        let nf = w.footprint.len();
        let x: Vec<C32> = (0..nv).map(|i| C32::new((i as f32 * 0.013).sin(), (i as f32 * 0.007).cos())).collect();
        let r: Vec<C32> = (0..nseg * nf).map(|i| C32::new((i as f32 * 0.011).cos(), (i as f32 * 0.017).sin())).collect();
        let fx = w.forward_cpu(&segph, &x);
        let mut xb = vec![C32::new(0.0, 0.0); nv];
        w.adjoint_cpu(&segph, &r, &mut xb);
        let lhs: Complex64 = fx.iter().zip(&r).map(|(a, b)| c64(*a).conj() * c64(*b)).sum();
        let rhs: Complex64 = x.iter().zip(&xb).map(|(a, b)| c64(*a).conj() * c64(*b)).sum();
        let rel = (lhs - rhs).norm() / lhs.norm();
        assert!(rel < 1e-5, "{s:?}: adjoint mismatch {rel}");
        let (m1, m2) = w.macs(31);
        println!("{s:?} (path field {path}): |V| {} footprint {nf}, <Ax,r> vs <x,Aᵀr> rel {rel:.1e}, MACs correction {m1} + grid {m2} (31 coils)", moving.len());
    }
}

#[test]
fn moving_vessel_against_direct_sum() {
    let tw = toy();
    let nv = tw.om.len();
    let (nr, na, t_ro, kmax) = (48usize, 240usize, 9.5e-3, std::f64::consts::PI / tw.vox);
    let traj = trajectory(nr, na, t_ro, kmax);
    let tau: Vec<f64> = (0..na).map(|n| t_ro * n as f64 / (na - 1) as f64).collect();
    let (ts, q) = seg_time_grid(&tau, &vec![1.0; na], 96);
    let fit: Vec<usize> = (0..nv).filter(|&v| c64(tw.x[v]).norm() > 0.0).collect();
    let g = |v: usize, t: f64| Complex64::from_polar(1.0, -tw.om[v] * t);
    let nseg = 9;
    let (segph, atab, t0, dt) = build_segments_svd_grid(nv, nseg, 0.0, t_ro, na, &fit, None, &ts, &q, &g);
    let (segph, atab, nodes, cond) = cardinal_form(&segph, &atab, nseg, nv);
    let t1: Vec<f64> = nodes.iter().map(|&e| t0 + e as f64 * dt).collect();
    let t2 = vec![0.0; nseg];
    let moving: Vec<u32> = (0..nv).filter(|&q| tw.vel[q] != [0.0; 3]).map(|q| q as u32).collect();
    let vh: Vec<[f64; 3]> = moving.iter().map(|&q| tw.vel[q as usize]).collect();
    let pos = |q: usize| [((q % tw.dims[0]) as f64 - 12.0) * tw.vox, (((q / tw.dims[0]) % tw.dims[1]) as f64 - 12.0) * tw.vox, ((q / (tw.dims[0] * tw.dims[1])) as f64 - 10.0) * tw.vox];
    let obj: Vec<usize> = fit.clone();
    let mut e_static = 0.0;
    let mut e_floor = 0.0;
    let mut e_signal = 0.0;
    let mut e_warp = std::collections::BTreeMap::new();
    let warps: Vec<(WarpStencil, SegWarp)> = [WarpStencil::Trilinear, WarpStencil::Cubic, WarpStencil::Lanczos3].iter().map(|&s| (s, SegWarp::new(tw.dims, [tw.vox; 3], moving.clone(), &vh, None, &t1, &t2, s))).collect();
    let corr: Vec<Vec<C32>> = warps.iter().map(|(_, w)| w.forward_cpu(&segph, &tw.x)).collect();
    for ro in traj.iter() {
        for &(k, t, tt1) in ro.iter().step_by(3) {
            let a = interp(&atab, nseg, t0, dt, t);
            let ph = |p: [f64; 3]| Complex64::from_polar(1.0, -(k[0] * p[0] + k[1] * p[1] + k[2] * p[2]));
            // truth: moving point voxels, exact B0
            let mut yt = Complex64::new(0.0, 0.0);
            let mut y0 = Complex64::new(0.0, 0.0);
            let mut ys = Complex64::new(0.0, 0.0);
            for &v in &obj {
                let p = pos(v);
                let d = tw.vel[v];
                let pm = [p[0] + d[0] * tt1, p[1] + d[1] * tt1, p[2] + d[2] * tt1];
                yt += c64(tw.x[v]) * Complex64::from_polar(1.0, -tw.om[v] * t) * ph(pm);
                y0 += c64(tw.x[v]) * Complex64::from_polar(1.0, -tw.om[v] * t) * ph(p);
                let seg: Complex64 = (0..nseg).map(|l| a[l] * c64(segph[l * nv + v])).sum();
                ys += c64(tw.x[v]) * seg * ph(p);
            }
            e_static += (yt - ys).norm_sqr();
            e_floor += (y0 - ys).norm_sqr();
            e_signal += yt.norm_sqr();
            for ((s, w), du) in warps.iter().zip(&corr) {
                let nf = w.footprint.len();
                let mut yc = Complex64::new(0.0, 0.0);
                for l in 0..nseg {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for (fi, &qf) in w.footprint.iter().enumerate() {
                        acc += c64(du[l * nf + fi]) * ph(pos(qf as usize));
                    }
                    yc += a[l] * acc;
                }
                *e_warp.entry(format!("{s:?}")).or_insert(0.0) += (yt - ys - yc).norm_sqr();
            }
        }
    }
    let rs = (e_static / e_signal).sqrt();
    println!("moving vessel (1 m/s peak, 9.5 ms centre-out, L = {nseg}, cond {cond:.1}): relative error static {rs:.4}, B0-segment floor (no motion) {:.2e}", (e_floor / e_signal).sqrt());
    for (s, e) in &e_warp {
        println!("  warp {s}: {:.4} ({:.2} of static)", (e / e_signal).sqrt(), (e / e_static).sqrt());
    }
    // measured on this toy (sharp-edged vessel, 1 m/s, 9.5 ms): trilinear 0.66, cubic 0.40, Lanczos-3 0.22 of the
    // static error; the floor is the stencil's k-response near the grid Nyquist edge, not the segments
    let r = |k: &str| (e_warp[k] / e_static).sqrt();
    assert!(e_floor < 0.01 * e_static, "the B0 segments should be exact here");
    assert!(r("Lanczos3") < 0.3 && r("Cubic") < 0.5 && r("Trilinear") < 0.8, "warp residuals {} {} {}", r("Trilinear"), r("Cubic"), r("Lanczos3"));
    assert!(r("Lanczos3") < r("Cubic") && r("Cubic") < r("Trilinear"));
}
