//! CPU: near-lossless checkpoints (`RunCtl::save`, every big vector quantised on its own scale at 1e-6) keep a resumed
//! quasi-Newton run on the uninterrupted one. Barzilai–Borwein steps α = ⟨s, s⟩ / ⟨s, y⟩ use the secant pair
//! (s = x − x_prev, y = g − g_prev) directly, so they expose any loss of precision in s late in the run; the state
//! stores s itself (`secant_diff`), not x_prev.
//!   cargo test --release -p hasty-recon --test checkpoint_lossy -- --nocapture

use hasty_recon::checkpoint::{get, secant_diff, RunCtl};
use hasty_value::{Tensor, Value};
use num_complex::Complex32 as C;

const N: usize = 1 << 18;

fn problem() -> (Vec<f32>, Vec<C>) {
    let mut s = 12345u64;
    let mut r = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32
    };
    let d: Vec<f32> = (0..N).map(|_| 0.05 + 10.0 * r() * r()).collect();
    let b: Vec<C> = (0..N).map(|_| C::new(100.0 * (r() - 0.5), 100.0 * (r() - 0.5))).collect();
    (d, b)
}

fn grad(d: &[f32], b: &[C], x: &[C]) -> Vec<C> {
    x.iter().zip(b).zip(d).map(|((x, b), d)| (x - b) * *d).collect()
}

fn dot(a: &[C], b: &[C]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p.re as f64 * q.re as f64 + p.im as f64 * q.im as f64).sum()
}

/// BB iterations k0..k1 from (x, x_prev, g_prev)
fn run(d: &[f32], b: &[C], x: &mut Vec<C>, xp: &mut Vec<C>, gp: &mut Vec<C>, k0: usize, k1: usize) {
    for k in k0..k1 {
        let g = grad(d, b, x);
        let a = if k == 0 {
            0.05
        } else {
            let s: Vec<C> = x.iter().zip(xp.iter()).map(|(p, q)| p - q).collect();
            let y: Vec<C> = g.iter().zip(gp.iter()).map(|(p, q)| p - q).collect();
            dot(&s, &s) / dot(&s, &y)
        };
        *xp = x.clone();
        *gp = g.clone();
        x.iter_mut().zip(&g).for_each(|(x, g)| *x -= g * a as f32);
    }
}

fn t(v: &[C]) -> Value {
    Value::Tensor(Tensor::from_slice(&[v.len()], v).unwrap())
}

fn back(v: &Value) -> Vec<C> {
    v.as_tensor().unwrap().to_vec().unwrap()
}

fn rel(a: &[C], b: &[C]) -> f64 {
    let d: Vec<C> = a.iter().zip(b).map(|(p, q)| p - q).collect();
    (dot(&d, &d) / dot(b, b)).sqrt()
}

#[test]
fn resumed_quasi_newton_matches_uninterrupted() {
    let (d, b) = problem();
    let (k_stop, k_end) = (25usize, 45usize);
    let zero = vec![C::new(0.0, 0.0); N];
    let (mut x, mut xp, mut gp) = (zero.clone(), zero.clone(), zero.clone());
    run(&d, &b, &mut x, &mut xp, &mut gp, 0, k_stop);
    let (xs, xps, gps) = (x.clone(), xp.clone(), gp.clone());
    let step = (dot(&xs.iter().zip(&xps).map(|(p, q)| p - q).collect::<Vec<_>>(), &xs.iter().zip(&xps).map(|(p, q)| p - q).collect::<Vec<_>>()) / dot(&xs, &xs)).sqrt();
    run(&d, &b, &mut x, &mut xp, &mut gp, k_stop, k_end);
    let full = x.clone();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp/checkpoint_lossy");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let ctl = RunCtl::new(&dir, 0);
    let mut out = vec![];
    for (name, as_diff) in [("s", true), ("x_prev", false)] {
        let xv = t(&xs);
        let second = if as_diff { secant_diff(&xv, &t(&xps)).unwrap() } else { t(&xps) };
        ctl.save(name, &Value::Dict(vec![("x".into(), xv.clone()), (name.into(), second), ("g_prev".into(), t(&gps))])).unwrap();
        let bytes = std::fs::metadata(dir.join(format!("state_{name}.hv"))).unwrap().len();
        let v = ctl.load(name).unwrap().unwrap();
        let mut x2 = back(get(&v, "x").unwrap());
        let mut xp2 = if as_diff { back(&secant_diff(get(&v, "x").unwrap(), get(&v, "s").unwrap()).unwrap()) } else { back(get(&v, "x_prev").unwrap()) };
        let mut gp2 = back(get(&v, "g_prev").unwrap());
        let e_state = rel(&x2, &xs);
        run(&d, &b, &mut x2, &mut xp2, &mut gp2, k_stop, k_end);
        let e = rel(&x2, &full);
        println!("stored {name}: {:.3} of raw, state err {e_state:.2e}, |s|/|x| {step:.2e}; resumed vs uninterrupted after {} iterations: {e:.2e}", bytes as f64 / (3 * N * 8) as f64, k_end - k_stop);
        out.push((e, bytes));
    }
    // the stored-secant state resumes within ~1e-6 of the uninterrupted run
    assert!(out[0].0 < 5e-6, "resumed run differs by {:e}", out[0].0);
    assert!(out[0].1 < (3 * N * 8) as u64 * 9 / 10, "checkpoint not compressed");
}
