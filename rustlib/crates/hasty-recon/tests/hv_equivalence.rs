//! CPU, on real data: a sim directory converted to `.hv` (research/experiments/io_policy/convert.py) loads like the
//! original through the Rust loaders (`SimData::load_lazy` + `load_set`, and the eager truth / map arrays).
//!   HASTY_TEST_SIM_A=<original scan> HASTY_TEST_SIM_B=<converted scan> \
//!     cargo test --release -p hasty-recon --test hv_equivalence -- --ignored --nocapture

use std::path::PathBuf;

use hasty_recon::data::SimData;
use hasty_recon::kload::LoadOpts;

fn maxdiff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let m = a.iter().zip(b).filter(|(x, _)| x.is_finite()).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max);
    let r = (a.iter().filter(|x| x.is_finite()).map(|&x| (x as f64).powi(2)).sum::<f64>() / a.len().max(1) as f64).sqrt();
    (m, r)
}

#[test]
#[ignore]
fn converted_sim_loads_like_original() {
    let (Ok(a), Ok(b)) = (std::env::var("HASTY_TEST_SIM_A"), std::env::var("HASTY_TEST_SIM_B")) else {
        eprintln!("HASTY_TEST_SIM_A / HASTY_TEST_SIM_B not set: skipped");
        return;
    };
    let t = std::time::Instant::now();
    let sa = SimData::load_lazy(&PathBuf::from(&a)).unwrap();
    let sb = SimData::load_lazy(&PathBuf::from(&b)).unwrap();
    println!("loaded both (lazy) in {:.1} s; B streams k-space from .hv: {}", t.elapsed().as_secs_f64(), sb.lazy.as_ref().unwrap().data.is_hv());
    assert!(sb.lazy.as_ref().unwrap().data.is_hv());
    for (n, x, y) in [("smaps_true", sa.smaps_true.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<f32>>(), sb.smaps_true.iter().flat_map(|c| [c.re, c.im]).collect::<Vec<f32>>()), ("truth_vel", sa.truth_vel.clone(), sb.truth_vel.clone()), ("truth_frac", sa.truth_frac.clone(), sb.truth_frac.clone()), ("field_ppm", sa.field_ppm.clone(), sb.field_ppm.clone())] {
        let (m, r) = maxdiff(&x, &y);
        println!("{n}: max |Δ| {m:.3e} = {:.2e} of rms", m / r.max(1e-30));
        assert!(m <= 2e-4 * r + 1e-6, "{n}");
    }
    let n_ro = sa.ro.enc.len();
    let ros: Vec<usize> = (0..256).map(|i| i * (n_ro - 1) / 255).collect();
    let nc = sa.nc;
    let w: Vec<Vec<f64>> = (0..nc).map(|i| (0..nc).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let o = LoadOpts { w: &w, cc: None, keep: None, data: true };
    let t = std::time::Instant::now();
    let (la, lb) = (sa.load_set(&ros, &o).unwrap(), sb.load_set(&ros, &o).unwrap());
    println!("256 readouts each in {:.2} s", t.elapsed().as_secs_f64());
    let mut kmax = 0f64;
    for ax in 0..3 {
        kmax = kmax.max(la.coords[ax].iter().zip(&lb.coords[ax]).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max));
    }
    let tmax = la.tau.iter().zip(&lb.tau).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let dy: Vec<f64> = la.y.iter().zip(&lb.y).map(|(x, y)| ((x.re - y.re) as f64).hypot((x.im - y.im) as f64)).collect();
    let ymax = dy.iter().cloned().fold(0.0, f64::max);
    let yrms = (dy.iter().map(|d| d * d).sum::<f64>() / dy.len() as f64).sqrt();
    let sig = (la.y.iter().map(|c| (c.re as f64).powi(2) + (c.im as f64).powi(2)).sum::<f64>() / la.y.len() as f64).sqrt();
    let step = sb.meta["storage"]["kspace_step"].as_f64();
    println!("coords (rad/voxel) max |Δ| {kmax:.2e}, τ max |Δ| {tmax:.2e} s, data max |Δ| {ymax:.3e} rms {yrms:.3e} (data rms incl. noise {sig:.3e}; rms Δ / rms {:.2e})", yrms / sig);
    let _ = step;
    assert!(kmax < 1e-5 && tmax < 1e-8);
    assert!(yrms / sig < 2e-4, "k-space differs: {:e}", yrms / sig);
}
