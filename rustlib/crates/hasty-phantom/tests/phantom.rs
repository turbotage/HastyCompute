use hasty_phantom::b0::dipole_field;
use hasty_phantom::flow::{womersley_shape, AorticWaveform};
use hasty_phantom::volume::Volume;

#[test]
fn womersley_profile_carries_unit_mean_flow() {
    // ∫_0^1 2 r T(r) dr = 1 for every α, including the large-α (asymptotic Bessel) regime
    for &alpha in &[0.0, 2.0, 8.0, 17.0, 30.0, 55.0, 80.0] {
        let n = 4000;
        let mut s = num_complex::Complex64::new(0.0, 0.0);
        for i in 0..n {
            let r = (i as f64 + 0.5) / n as f64;
            s += womersley_shape(alpha, r) * (2.0 * r / n as f64);
        }
        assert!((s - 1.0).norm() < 2e-3, "alpha {alpha}: mean {s}");
        // no-slip at the wall
        assert!(womersley_shape(alpha, 1.0).norm() < 1e-6, "alpha {alpha}: wall {}", womersley_shape(alpha, 1.0));
    }
}

#[test]
fn waveform_harmonics_reproduce_flow_and_stroke_volume() {
    let w = AorticWaveform::healthy(0.9);
    let c = w.harmonics(40);
    let om = 2.0 * std::f64::consts::PI / w.period;
    let mut err: f64 = 0.0;
    let mut qmax: f64 = 0.0;
    for j in 0..200 {
        let t = j as f64 * w.period / 200.0;
        let q: f64 = c.iter().enumerate().map(|(n, cn)| (cn * num_complex::Complex64::from_polar(1.0, n as f64 * om * t)).re).sum();
        err = err.max((q - w.q(t)).abs());
        qmax = qmax.max(w.q(t));
    }
    assert!(err < 0.03 * qmax, "harmonic reconstruction error {err} vs peak {qmax}");
    assert!((c[0].re * w.period - w.sv).abs() < 1e-3 * w.sv, "stroke volume {}", c[0].re * w.period);
}

#[test]
fn dipole_field_of_a_sphere() {
    // Δχ = 1 ppm sphere of radius 12 mm in a 1 mm grid: outside, ΔB/B0 = Δχ/3 (R/r)^3 (3cos²θ − 1); inside ≈ 0
    let n = 96;
    let aff = [[1e-3, 0.0, 0.0, 0.0], [0.0, 1e-3, 0.0, 0.0], [0.0, 0.0, 1e-3, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let mut chi = Volume::<f32>::new([n, n, n], aff);
    let c = (n / 2) as f64;
    let r0 = 12.0;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let r = ((i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2)).sqrt();
                let q = chi.idx(i, j, k);
                chi.data[q] = if r < r0 { 1.0 } else { 0.0 };
            }
        }
    }
    let f = dipole_field(&chi, 0.0, 48, 48);
    let at = |i: usize, j: usize, k: usize| f.get(i, j, k) as f64;
    let ci = c as usize;
    let on_axis = at(ci, ci, ci + 24); // θ = 0, r = 24: (1/3)(1/8)(2) = 1/12
    let equator = at(ci + 24, ci, ci); // θ = 90°: (1/3)(1/8)(−1) = −1/24
    let inside = at(ci, ci, ci);
    eprintln!("sphere: axis {on_axis:.4} (1/12 = 0.0833), equator {equator:.4} (−0.0417), centre {inside:.4}");
    assert!((on_axis - 1.0 / 12.0).abs() < 0.006);
    assert!((equator + 1.0 / 24.0).abs() < 0.004);
    assert!(inside.abs() < 0.01);
}
