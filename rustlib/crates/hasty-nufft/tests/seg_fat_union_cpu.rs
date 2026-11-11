//! CPU test of the union SVD basis for per-frame fat maps (hasty_nufft::build_segments_svd_fat_union): its table is
//! interpolatory at the nodes, and with per-frame fat-fraction maps applied per node (P_l (1 + f (c_l − 1)), the
//! operator of `SegNufft::segments_fat`) the error of the joint water + fat time functions follows the union Gram tail
//! (within 2.5×) and falls with L. Uniform nodes (the per-frame path today) printed for comparison; the noise-referenced
//! comparison on the phantom is research/experiments/readout_physics/fat_union.py.
use hasty_cuda::C32;
use hasty_nufft::{build_segments_fat_nodes, build_segments_svd_fat_union, fat_union_spectrum, seg_time_grid};
use num_complex::Complex64;

fn c64(z: C32) -> Complex64 {
    Complex64::new(z.re as f64, z.im as f64)
}

fn fat_c(t: f64) -> Complex64 {
    let a = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048];
    let s: f64 = a.iter().sum();
    let ppm = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60];
    (0..6).map(|p| Complex64::from_polar(a[p] / s, -2.0 * std::f64::consts::PI * 42.577478e6 * 3.0 * ppm[p] * 1e-6 * t)).sum()
}

fn interp(atab: &[C32], nseg: usize, t0: f64, dt: f64, t: f64) -> Vec<Complex64> {
    let ntab = atab.len() / nseg;
    let u = ((t - t0) / dt).clamp(0.0, (ntab - 1) as f64 - 1e-9);
    let i = u.floor() as usize;
    let f = u - i as f64;
    (0..nseg).map(|l| c64(atab[i * nseg + l]) * (1.0 - f) + c64(atab[(i + 1).min(ntab - 1) * nseg + l]) * f).collect()
}

#[test]
fn fat_union_tracks_gram_tail() {
    // 4000 voxels: field ±250 Hz (smooth + a few outliers), fat fraction maps that change per frame
    let nv = 4000;
    let om: Vec<f64> = (0..nv).map(|v| 2.0 * std::f64::consts::PI * (250.0 * ((v as f64 * 0.0137).sin()) + if v % 97 == 0 { 300.0 } else { 0.0 })).collect();
    let frames: Vec<Vec<f64>> = (0..4).map(|fr| (0..nv).map(|v| (0.5 + 0.5 * ((v as f64 * 0.031) + fr as f64).sin()).powi(3)).collect()).collect();
    let (te, t_ro, na) = (1.6e-3, 9.5e-3, 400usize);
    let tau: Vec<f64> = (0..na).map(|n| te + t_ro * n as f64 / (na - 1) as f64).collect();
    let (ts, tw) = seg_time_grid(&tau, &vec![1.0; na], 96);
    let fit: Vec<usize> = (0..nv).step_by(3).collect();
    let fmin: Vec<f64> = fit.iter().map(|&v| frames.iter().map(|f| f[v]).fold(1.0, f64::min)).collect();
    let fmax: Vec<f64> = fit.iter().map(|&v| frames.iter().map(|f| f[v]).fold(0.0, f64::max)).collect();
    let ww: Vec<f64> = fmin.iter().map(|f| (1.0 - f).powi(2) + 1e-12).collect();
    let wf: Vec<f64> = fmax.iter().map(|f| f * f + 1e-12).collect();
    let psi = |v: usize, t: f64| om[v] * t;
    let c = |t: f64| fat_c(t);
    let err = |segph: &[C32], atab: &[C32], cl: &[C32], nseg: usize, t0: f64, dt: f64| -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for (n, &t) in tau.iter().enumerate().step_by(3) {
            let a = interp(atab, nseg, t0, dt, t);
            let _ = n;
            for f in &frames {
                for v in (0..nv).step_by(7) {
                    let exact = Complex64::from_polar(1.0, -psi(v, t)) * ((1.0 - f[v]) + f[v] * c(t));
                    let model: Complex64 = (0..nseg).map(|l| a[l] * c64(segph[l * nv + v]) * (1.0 + f[v] * (c64(cl[l]) - 1.0))).sum();
                    num += (exact - model).norm_sqr();
                    den += exact.norm_sqr();
                }
            }
        }
        (num / den).sqrt()
    };
    let ev = fat_union_spectrum(&fit, &ww, &wf, &ts, &tw, &psi, &c);
    let mut prev = f64::MAX;
    for nseg in [8usize, 10, 12, 14] {
        let (sp, at, cl, t0, dt, nodes, cond) = build_segments_svd_fat_union(nv, nseg, tau[0], tau[na - 1], na, &fit, &ww, &wf, &ts, &tw, &psi, &c);
        // interpolatory at the nodes
        for (m, &e) in nodes.iter().enumerate() {
            for l in 0..nseg {
                let z = c64(at[e * nseg + l]);
                assert!((z - if l == m { 1.0 } else { 0.0 }).norm() < 1e-3, "a'_{l}(node {m}) = {z}");
            }
        }
        let eu = err(&sp, &at, &cl, nseg, t0, dt);
        let tl: Vec<f64> = (0..nseg).map(|l| tau[0] + (tau[na - 1] - tau[0]) * l as f64 / (nseg - 1) as f64).collect();
        let wnod: Vec<f64> = vec![1.0; fit.len()];
        let (sp2, at2, cl2, t02, dt2) = build_segments_fat_nodes(nv, &tl, tau[0], tau[na - 1], na, &fit, Some(&wnod), &psi, &c);
        let ef = err(&sp2, &at2, &cl2, nseg, t02, dt2);
        let tail = (ev[nseg..].iter().sum::<f64>() / ev.iter().sum::<f64>()).sqrt();
        println!("L {nseg:>2}: union svdn (cardinal, cond {cond:.1}) {eu:.2e} (Gram tail {tail:.2e}), uniform nodes {ef:.2e}");
        assert!(eu < 2.5 * tail + 1e-3, "union error {eu} vs Gram tail {tail} at L {nseg}");
        assert!(eu < prev, "not decreasing with L");
        prev = eu;
        let _ = ef;
    }
}
