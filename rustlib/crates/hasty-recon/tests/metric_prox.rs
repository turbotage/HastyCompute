//! CPU: the metric-consistent prox (`hasty_recon::p2np::metric_prox_root`, Becker–Fadili–Ochs rank-one case) equals the
//! prox of h in the metric V = dI + κ w wᵀ, computed by brute force (proximal gradient on h + ½‖x − y‖²_V, many
//! iterations), for complex ℓ1 (h = t‖x‖₁, prox = complex soft threshold) and both signs of κ (V positive definite).
//!   cargo test --release -p hasty-recon --test metric_prox

use hasty_cuda::C32;
use hasty_recon::p2np::metric_prox_root;

fn soft(v: &mut [C32], thr: f64) {
    for z in v.iter_mut() {
        let m = ((z.re as f64).powi(2) + (z.im as f64).powi(2)).sqrt();
        let s = if m > thr { (m - thr) / m } else { 0.0 };
        *z = C32::new((z.re as f64 * s) as f32, (z.im as f64 * s) as f32);
    }
}

fn dot(a: &[C32], b: &[C32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x.re as f64 * y.re as f64 + x.im as f64 * y.im as f64).sum()
}

/// argmin_x t‖x‖₁ + ½ (x − y)ᵀ V (x − y), V = dI + κ w wᵀ (real inner product on ℂⁿ = ℝ²ⁿ), by proximal gradient
fn brute(y: &[C32], w: &[C32], d: f64, kappa: f64, t: f64) -> Vec<C32> {
    let l = d + kappa.max(0.0) * dot(w, w);
    let mut x: Vec<C32> = y.to_vec();
    for _ in 0..200_000 {
        let r: Vec<C32> = x.iter().zip(y).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect();
        let wr = dot(w, &r);
        // V r = d r + κ w ⟨w, r⟩
        let mut z: Vec<C32> = x
            .iter()
            .zip(r.iter().zip(w))
            .map(|(xi, (ri, wi))| {
                let g_re = d * ri.re as f64 + kappa * wi.re as f64 * wr;
                let g_im = d * ri.im as f64 + kappa * wi.im as f64 * wr;
                C32::new((xi.re as f64 - g_re / l) as f32, (xi.im as f64 - g_im / l) as f32)
            })
            .collect();
        soft(&mut z, t / l);
        x = z;
    }
    x
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2)).sum();
    let d: f64 = b.iter().map(|y| (y.re as f64).powi(2) + (y.im as f64).powi(2)).sum();
    (n / d.max(1e-30)).sqrt()
}

#[test]
fn metric_prox_matches_brute_force() {
    let n = 40;
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let y: Vec<C32> = (0..n).map(|_| C32::new(u() as f32 * 2.0, u() as f32 * 2.0)).collect();
    let w: Vec<C32> = (0..n).map(|_| C32::new(u() as f32, u() as f32)).collect();
    let ww = dot(&w, &w);
    let (d, t) = (1.3, 0.25);
    // κ > 0, and κ < 0 with V = dI + κ w wᵀ still positive definite (d + κ‖w‖² = 0.4 d)
    for kappa in [4.0 * d / ww, -0.6 * d / ww] {
        let mut prox = |v: &mut Vec<C32>| soft(v, t / d);
        let (_, _, res6, calls6) = metric_prox_root(&y, &w, d, kappa, 30, 1e-6, &mut prox);
        eprintln!("κ {kappa:+.4e}: tolerance 1e-6 reached in {calls6} prox calls (|L| {res6:.1e})");
        let (x, a, res, calls) = metric_prox_root(&y, &w, d, kappa, 30, 1e-10, &mut prox);
        let xb = brute(&y, &w, d, kappa, t);
        let e = rel(&x, &xb);
        eprintln!("κ {kappa:+.4e}: metric prox vs brute force {e:.2e}, a* {a:.4e}, |L| {res:.1e}, {calls} prox calls");
        assert!(e < 1e-4, "metric prox differs from the brute-force prox in V: {e:.2e} (κ {kappa:.3e})");
        assert!(calls <= 20, "too many prox calls: {calls}");
        // and it differs from the Euclidean prox at step 1/d (the test can discriminate)
        let mut xe = y.clone();
        soft(&mut xe, t / d);
        let ed = rel(&xe, &xb);
        eprintln!("κ {kappa:+.4e}: Euclidean prox vs the prox in V {ed:.2e} (must be ≫ the error above)");
        assert!(ed > 100.0 * e.max(1e-9), "the rank-one term does not change the prox here: {ed:.2e} vs error {e:.2e}");
    }
}
