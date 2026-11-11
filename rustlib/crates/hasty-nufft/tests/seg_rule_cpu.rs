//! CPU: noise-referenced segment bases and the segment-count rule (`hasty_nufft::segbasis`, research/segment_basis.md)
//! on synthetic fields — no GPU, no data files.
//!   cargo test --release -p hasty-nufft --test seg_rule_cpu -- --nocapture

use std::f64::consts::PI;

use hasty_nufft::{build_segments_nodes, build_segments_svd_grid, build_segments_svd_weighted, cheb_nodes, nodes_weighted_error, seg_count_for, seg_time_grid, segment_model_error, snr_time_weights, tail_errors, weighted_spectrum};
use num_complex::Complex64;

/// ADC times of an 8 ms readout at 20 µs (400 samples) and an out-and-back data-energy profile: the k-space centre
/// (start and end, the end weaker by dephasing) carries almost all of the energy, the middle sits near the noise
const NA: usize = 400;
fn adc() -> (Vec<f64>, Vec<f64>) {
    let tau: Vec<f64> = (0..NA).map(|n| 1.6e-3 + 8e-3 * n as f64 / (NA - 1) as f64).collect();
    let e: Vec<f64> = (0..NA).map(|n| {
        let (a, b) = (n as f64, (NA - 1 - n) as f64);
        1e5 * (-a / 4.0).exp() + 1e4 * (-b / 4.0).exp() + 1.0
    }).collect();
    (tau, e)
}

/// a smooth field with a heavy tail: most voxels within ±`b`/2 Hz, 1 % out to ±2b (interfaces)
fn field(nv: usize, b: f64) -> Vec<f64> {
    let mut s = 0x1234_5678u64;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..nv).map(|v| {
        let x = v as f64 / nv as f64;
        let base = 0.5 * b * (2.0 * PI * 3.0 * x).sin() * (0.6 + 0.4 * u());
        2.0 * PI * if v % 100 == 0 { (u() * 4.0 - 2.0) * b } else { base }
    }).collect()
}

#[test]
fn time_grid_resolves_peaked_weights() {
    let (tau, e) = adc();
    let w = snr_time_weights(&e, 1.0);
    let (ts, tw) = seg_time_grid(&tau, &w, 64);
    assert!(ts.len() <= 64 && ts.len() >= 20, "{} samples", ts.len());
    assert_eq!(ts[0], tau[0]);
    assert_eq!(*ts.last().unwrap(), tau[NA - 1]);
    // quadrature of smooth functions against the per-sample weights
    for f in [|t: f64| 1.0 + 0.0 * t, |t: f64| (2.0 * PI * 150.0 * t).cos() + 2.0, |t: f64| (t * 1e3).powi(2)] {
        let exact: f64 = tau.iter().zip(&w).map(|(&t, &wi)| wi * f(t)).sum();
        let quad: f64 = ts.iter().zip(&tw).map(|(&t, &wi)| wi * f(t)).sum();
        assert!(((quad - exact) / exact).abs() < 1e-2, "quadrature {quad} vs {exact}");
    }
    // half of the samples within the first / last 5 % of the window (where the weight is)
    let ends = ts.iter().filter(|&&t| t < tau[NA / 20] || t > tau[NA - 1 - NA / 20]).count();
    assert!(ends * 3 >= ts.len(), "{ends} of {} samples at the ends", ts.len());
    // the uniform-only limit: equal weights give (nearly) uniform samples
    let (tu, twu) = seg_time_grid(&tau, &vec![1.0; NA], 40);
    let dmax = tu.windows(2).map(|p| p[1] - p[0]).fold(0.0f64, f64::max);
    assert!(dmax < 2.5 * 8e-3 / 39.0, "max spacing {dmax}");
    assert!((twu.iter().sum::<f64>() - NA as f64).abs() < 1e-9);
}

#[test]
fn spectrum_tail_predicts_the_built_basis() {
    let (tau, e) = adc();
    let w = snr_time_weights(&e, 1.0);
    let nv = 3000;
    let om = field(nv, 400.0);
    let fit: Vec<usize> = (0..nv).collect();
    let g = |v: usize, t: f64| Complex64::from_polar(1.0, -om[v] * t);
    let (ts, tw) = seg_time_grid(&tau, &w, 96);
    let ev = weighted_spectrum(&fit, None, &ts, &tw, &g);
    let tails = tail_errors(&ev);
    assert!((tails[0] - 1.0).abs() < 1e-12);
    assert!(tails.windows(2).all(|p| p[1] <= p[0] + 1e-15), "tails decrease");
    for l in [4usize, 6, 8] {
        let (sp, at, t0, dt) = build_segments_svd_grid(nv, l, tau[0], tau[NA - 1], NA, &fit, None, &ts, &tw, &g);
        let err = segment_model_error(&sp, &at, l, nv, t0, dt, &fit, None, &tau, &w, &g);
        eprintln!("L = {l}: weighted model error {err:.3e}, spectrum tail {:.3e}", tails[l]);
        // the refitted interpolators are least squares at every ADC time: at most the projection error, up to the
        // quadrature of the grid (and f32 storage)
        assert!(err <= 1.5 * tails[l] + 1e-6, "L = {l}: {err:.3e} vs tail {:.3e}", tails[l]);
        assert!(err >= 0.3 * tails[l], "L = {l}: {err:.3e} far below the tail {:.3e}", tails[l]);
    }
}

#[test]
fn noise_weighting_lowers_the_weighted_error() {
    // at equal L the SNR-weighted basis beats the unweighted rank-L basis and the fixed uniform nodes in the
    // SNR-weighted error (the noise-referenced data error), and the rule needs no more segments
    let (tau, e) = adc();
    let w = snr_time_weights(&e, 1.0);
    let nv = 3000;
    let om = field(nv, 500.0);
    let fit: Vec<usize> = (0..nv).collect();
    let g = |v: usize, t: f64| Complex64::from_polar(1.0, -om[v] * t);
    let (ts, tw) = seg_time_grid(&tau, &w, 96);
    let (t0, t1) = (tau[0], tau[NA - 1]);
    for l in [4usize, 6, 8] {
        let (sp, at, a0, dt) = build_segments_svd_grid(nv, l, t0, t1, NA, &fit, None, &ts, &tw, &g);
        let en = segment_model_error(&sp, &at, l, nv, a0, dt, &fit, None, &tau, &w, &g);
        let (sp, at, a0, dt) = build_segments_svd_weighted(nv, l, t0, t1, NA, &fit, None, None, 96, &g);
        let eu = segment_model_error(&sp, &at, l, nv, a0, dt, &fit, None, &tau, &w, &g);
        let tl: Vec<f64> = (0..l).map(|i| t0 + (t1 - t0) * i as f64 / (l - 1) as f64).collect();
        let (sp, at, a0, dt) = build_segments_nodes(nv, &tl, t0, t1, NA, &fit, None, &g);
        let ef = segment_model_error(&sp, &at, l, nv, a0, dt, &fit, None, &tau, &w, &g);
        eprintln!("L = {l}: SNR-weighted error: svdn {en:.2e}, svd {eu:.2e}, fixed {ef:.2e}");
        assert!(en <= eu * 1.02 && en <= ef * 1.02, "L = {l}: svdn {en:.2e} vs svd {eu:.2e} / fixed {ef:.2e}");
    }
}

#[test]
fn rule_is_monotone_in_the_field_spread() {
    let (tau, e) = adc();
    let w = snr_time_weights(&e, 1.0);
    let (ts, tw) = seg_time_grid(&tau, &w, 96);
    let nv = 2000;
    let fit: Vec<usize> = (0..nv).collect();
    let mut prev = 0usize;
    for b in [50.0, 200.0, 400.0, 800.0, 1600.0] {
        let om = field(nv, b);
        let g = |v: usize, t: f64| Complex64::from_polar(1.0, -om[v] * t);
        let ev = weighted_spectrum(&fit, None, &ts, &tw, &g);
        let l = seg_count_for(&ev, 1e-3, 1, 64);
        eprintln!("spread {b:.0} Hz (time-bandwidth {:.1}): L = {l}", b * 8e-3);
        assert!(l >= prev, "L must not drop with a wider field: {l} after {prev}");
        prev = l;
    }
    // a single frequency is rank 1: L = 1 at any tolerance
    let g1 = |_v: usize, t: f64| Complex64::from_polar(1.0, -2.0 * PI * 333.0 * t);
    assert_eq!(seg_count_for(&weighted_spectrum(&fit, None, &ts, &tw, &g1), 1e-6, 1, 64), 1);
}

#[test]
fn nodes_error_matches_the_tabulated_basis() {
    let (tau, e) = adc();
    let w = snr_time_weights(&e, 1.0);
    let (ts, tw) = seg_time_grid(&tau, &w, 96);
    let nv = 2000;
    let om = field(nv, 400.0);
    let fit: Vec<usize> = (0..nv).collect();
    let vw: Vec<f64> = (0..nv).map(|v| 0.5 + (v % 7) as f64).collect();
    let g = |v: usize, t: f64| Complex64::from_polar(1.0, -om[v] * t);
    let (t0, t1) = (tau[0], tau[NA - 1]);
    for l in [5usize, 8] {
        for tl in [cheb_nodes(l, t0, t1), (0..l).map(|i| t0 + (t1 - t0) * i as f64 / (l - 1) as f64).collect::<Vec<_>>()] {
            let quick = nodes_weighted_error(&fit, Some(&vw), Some(&vw), &ts, &tw, &tl, &g);
            let (sp, at, a0, dt) = build_segments_nodes(nv, &tl, t0, t1, NA, &fit, Some(&vw), &g);
            let full = segment_model_error(&sp, &at, l, nv, a0, dt, &fit, Some(&vw), &tau, &w, &g);
            eprintln!("L = {l}: nodes error on the grid {quick:.3e}, tabulated over all ADC times {full:.3e}");
            assert!((quick / full - 1.0).abs() < 0.3, "{quick:.3e} vs {full:.3e}");
        }
    }
    let c = cheb_nodes(6, 1.0, 3.0);
    assert!((c[0] - 1.0).abs() < 1e-15 && (c[5] - 3.0).abs() < 1e-15 && (c[1] - 1.0) < (c[3] - c[2]));
}

#[test]
fn white_floor_weights() {
    use hasty_nufft::{noise_time_weights, vol_energy};
    // an out-and-back |k| profile: 0 → k_max → 0
    let kap: Vec<f64> = (0..NA).map(|n| 1570.0 * (PI * n as f64 / (NA - 1) as f64).sin()).collect();
    // a flat energy profile: the volume average is that energy
    assert!((vol_energy(&vec![3.0; NA], &kap) - 3.0).abs() < 1e-12);
    // energy concentrated at the centre: the volume average is far below the time average (the shells there are small)
    let (_, e) = adc();
    let ev = vol_energy(&e, &kap);
    let et = e.iter().sum::<f64>() / NA as f64;
    assert!(ev < 0.1 * et && ev > 0.0, "volume average {ev:.3e} vs time average {et:.3e}");
    let w = noise_time_weights(&e, &kap, 2.0);
    let s = snr_time_weights(&e, 2.0);
    for n in 0..NA {
        assert!((w[n] - s[n] - ev / 2.0).abs() < 1e-9 * w[n], "the white floor E_vol / noise is added to every sample");
    }
}
