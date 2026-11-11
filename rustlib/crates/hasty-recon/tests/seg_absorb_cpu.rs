//! CPU: the smooth field part moved from the time segments into the per-sample tables (`segphys::FieldPoly`,
//! `--b0-absorb`): the fit recovers a polynomial field, and the phase every (sample, voxel) pair sees is unchanged —
//! table phase (φ0 + k·ρ + ρᵀΔCρ) plus the segments' residual ψ_res equals the original table phase plus ψ.
//!   cargo test --release -p hasty-recon --test seg_absorb_cpu

use hasty_recon::segphys::{absorb_field_poly_at, fit_field_poly_at, FieldPoly, SegTables};

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed.max(1);
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// [1, x, y, z, x², y², z², 2xy, 2xz, 2yz]
fn basis(p: &[f64; 3]) -> [f64; 10] {
    [1.0, p[0], p[1], p[2], p[0] * p[0], p[1] * p[1], p[2] * p[2], 2.0 * p[0] * p[1], 2.0 * p[0] * p[2], 2.0 * p[1] * p[2]]
}

const TP: f64 = std::f64::consts::TAU;

#[test]
fn fit_recovers_polynomial_fields() {
    let mut u = rng(7);
    let pos: Vec<[f64; 3]> = (0..4000).map(|_| [0.4 * u(), 0.3 * u(), 0.2 * u()]).collect();
    let c_true = [TP * 40.0, TP * 300.0, TP * -150.0, TP * 80.0, TP * 900.0, TP * -400.0, TP * 1200.0, TP * 200.0, TP * -300.0, TP * 100.0];
    let om: Vec<f64> = pos.iter().map(|p| (0..10).map(|i| c_true[i] * basis(p)[i]).sum()).collect();
    let vox: Vec<usize> = (0..pos.len()).collect();
    let f2 = fit_field_poly_at(&|v| pos[v], &om, &vox, None, 2, 5e-3);
    for i in 0..10 {
        assert!((f2.c[i] - c_true[i]).abs() < 1e-6 * c_true[i].abs().max(1.0), "coefficient {i}: {} vs {}", f2.c[i], c_true[i]);
    }
    // degree 1 of a linear field: exact; weights do not matter then
    let om1: Vec<f64> = pos.iter().map(|p| c_true[0] + c_true[1] * p[0] + c_true[2] * p[1] + c_true[3] * p[2]).collect();
    let w: Vec<f64> = (0..pos.len()).map(|i| 0.1 + (i % 5) as f64).collect();
    let f1 = fit_field_poly_at(&|v| pos[v], &om1, &vox, Some(&w), 1, 5e-3);
    for i in 0..4 {
        assert!((f1.c[i] - c_true[i]).abs() < 1e-6 * c_true[i].abs(), "degree 1, coefficient {i}");
    }
    assert!(f1.c[4..].iter().all(|&c| c == 0.0));
    // the residual of a field that is exactly polynomial: zero (degree 1), the static voxel phase Q(ρ) τ_ref of the
    // quadratic part (degree 2), the same at every readout time (rank-neutral in the segments)
    for (v, p) in pos.iter().enumerate().take(50) {
        let stat = (f2.eval(p) - f2.eval_lin(p)) * f2.tau_ref;
        for &t in &[1.6e-3, 5e-3, 9.6e-3] {
            assert!(f1.residual_at(p, om1[v] * t, t).abs() < 1e-6);
            assert!((f2.residual_at(p, om[v] * t, t) - stat).abs() < 1e-5);
        }
    }
}

#[test]
fn absorbed_tables_keep_every_sample_phase() {
    let mut u = rng(11);
    let vox = [2e-3, 2e-3, 2e-3];
    let m = 500;
    let tau: Vec<f32> = (0..m).map(|j| (1.6e-3 + 8e-3 * (j % 100) as f64 / 99.0) as f32).collect();
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (2.0 * std::f64::consts::PI * u()) as f32).collect());
    let phi0: Vec<f32> = (0..m).map(|_| (0.1 * u()) as f32).collect();
    let dc: Vec<f32> = (0..6 * m).map(|_| (2.0 * u()) as f32).collect();
    let tab0 = SegTables { coords: coords.clone(), tau: tau.clone(), phi0: phi0.clone(), dc: dc.clone(), t: vec![0.0; m], wbar: vec![], tau_adc: vec![], max_resid_phase: 0.0 };
    let poly = FieldPoly { deg: 2, c: [TP * 30.0, TP * 250.0, TP * -120.0, TP * 60.0, TP * 4.0, TP * -2.0, TP * 3.0, TP * 1.0, TP * -1.0, TP * 0.5], tau_ref: 5.59e-3 };
    let pos: Vec<[f64; 3]> = (0..200).map(|_| [0.4 * u(), 0.25 * u(), 0.2 * u()]).collect();
    // table phase of sample j at position p: φ0 + Σ_a coords_a (p_a / Δ_a) + ρᵀΔCρ (dc against [x², y², z², 2xy, 2xz, 2yz])
    let phase = |t: &SegTables, j: usize, p: &[f64; 3]| -> f64 {
        let b = basis(p);
        t.phi0[j] as f64 + (0..3).map(|a| t.coords[a][j] as f64 * p[a] / vox[a]).sum::<f64>() + (0..6).map(|i| t.dc[i * m + j] as f64 * b[4 + i]).sum::<f64>()
    };
    for deg in [1u32, 2] {
        let mut p2 = poly.clone();
        p2.deg = deg;
        if deg == 1 {
            p2.c[4..].iter_mut().for_each(|c| *c = 0.0);
        }
        let mut tab = SegTables { coords: coords.clone(), tau: tau.clone(), phi0: phi0.clone(), dc: dc.clone(), t: vec![0.0; m], wbar: vec![], tau_adc: vec![], max_resid_phase: 0.0 };
        let q = absorb_field_poly_at(&mut tab, vox, &p2, &mut pos.iter().copied());
        // field = the polynomial + a rough residual; ψ = ω τ
        let om = |p: &[f64; 3]| p2.eval(p) + TP * 37.0 * (40.0 * p[0]).sin();
        let mut worst = 0.0f64;
        for j in 0..m {
            let t = tau[j] as f64;
            for p in &pos {
                let before = phase(&tab0, j, p) + om(p) * t;
                // (the residual keeps the static part Q τ_ref of the quadratic term, the table its Q (τ − τ_ref))
                let after = phase(&tab, j, p) + p2.residual_at(p, om(p) * t, t);
                worst = worst.max((before - after).abs());
            }
        }
        // f32 tables: k·Δ ≈ π rad/voxel times |p/Δ| ≤ 100 voxels
        assert!(worst < 2e-3, "degree {deg}: phase changed by {worst:.2e} rad");
        if deg == 2 {
            assert!(q > 0.0, "the Taylor phase of the quadratic part is reported");
        } else {
            assert_eq!(q, 0.0);
        }
        eprintln!("degree {deg}: largest phase change {worst:.1e} rad, quadratic Taylor phase {q:.3} rad");
    }
}
