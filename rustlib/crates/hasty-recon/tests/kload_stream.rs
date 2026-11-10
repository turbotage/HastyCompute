//! Streaming loader (`kload`) vs the legacy in-memory loader on a stored spiral sim (CPU only):
//! - noise-free: identical whitened data (and coil-compressed data) per set, also with a sample filter;
//! - noisy: the added noise is white after whitening with σ²/2 per component (Ψ handled exactly);
//! - k-space store round trip of real data (ratios / speed printed).
//! Skipped when the sim is not present.

use std::path::{Path, PathBuf};
use std::time::Instant;

use hasty_cuda::C32;
use hasty_recon::coils::{CoilCompression, CoilGram, CoilRule};
use hasty_recon::data::SimData;
use hasty_recon::kload::LoadOpts;
use hasty_recon::kstore::{KStore, StoreSpec};
use num_complex::Complex64;

fn sim(name: &str) -> Option<PathBuf> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/phantom/factory/ph0001").join(name).join("scan");
    p.join("meta.json").exists().then_some(p)
}

fn apply_cc(y: &[C32], m: usize, nc: usize, rows: &[Vec<Complex64>]) -> Vec<C32> {
    let mut o = vec![C32::new(0.0, 0.0); rows.len() * m];
    for (v, r) in rows.iter().enumerate() {
        for j in 0..m {
            let mut a = Complex64::new(0.0, 0.0);
            for c in 0..nc {
                let z = y[c * m + j];
                a += r[c] * Complex64::new(z.re as f64, z.im as f64);
            }
            o[v * m + j] = C32::new(a.re as f32, a.im as f32);
        }
    }
    o
}

fn dn(x: &C32, y: &C32) -> f32 {
    ((x.re - y.re).powi(2) + (x.im - y.im).powi(2)).sqrt()
}

fn maxdiff(a: &[C32], b: &[C32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let d = a.iter().zip(b).map(|(x, y)| dn(x, y)).fold(0.0f32, f32::max);
    let s = b.iter().map(|y| (y.re * y.re + y.im * y.im).sqrt()).fold(0.0f32, f32::max);
    (d, s)
}

#[test]
fn lazy_matches_legacy_and_noise_is_white() {
    let Some(dir) = sim("sim_point") else {
        eprintln!("skipped: sim_point not present");
        return;
    };
    // noise-free comparison (both loaders read data_clean.npy without noise)
    unsafe { std::env::set_var("HASTY_NOISE_SIGMA", "0") };
    let legacy = SimData::load(&dir).unwrap();
    let lazy = SimData::load_lazy(&dir).unwrap();
    assert!(lazy.is_lazy() && lazy.data.is_empty());
    let w = legacy.whitener();
    let nc = legacy.nc;
    let sets: Vec<Vec<usize>> = (0..4).map(|e| legacy.readouts_of(e, None)).filter(|r| !r.is_empty()).collect();
    let mut g = CoilGram::new(nc);
    let s0 = legacy.sample_set(&sets[0], &w);
    g.add(&s0.y, s0.tau.len(), 7);
    let cc = CoilCompression::from_gram(&g, CoilRule::Energy(0.999));
    println!("coil compression: {}", cc.report());
    let t = Instant::now();
    for ros in &sets {
        let a = legacy.sample_set(ros, &w);
        let b = lazy.load_set(ros, &LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        assert_eq!(a.tau, b.tau);
        assert_eq!(a.coords, b.coords);
        let (d, s) = maxdiff(&b.y, &a.y);
        assert!(d <= 1e-6 * s, "whitened data differ: {d} of {s}");
        let m = a.tau.len();
        let bc = lazy.load_set(ros, &LoadOpts { w: &w, cc: Some(&cc.rows), keep: None, data: true }).unwrap();
        let (d, s) = maxdiff(&bc.y, &apply_cc(&a.y, m, nc, &cc.rows));
        assert!(d <= 1e-5 * s, "compressed data differ: {d} of {s}");
        // a sample filter (calibration box)
        let keep = |k: &[f32; 3]| k.iter().all(|v| v.abs() < 0.4);
        let bf = lazy.load_set(ros, &LoadOpts { w: &w, cc: None, keep: Some(&keep), data: true }).unwrap();
        let sel: Vec<usize> = (0..m).filter(|&j| keep(&[a.coords[0][j], a.coords[1][j], a.coords[2][j]])).collect();
        assert_eq!(bf.tau.len(), sel.len());
        let mf = sel.len();
        for c in 0..nc {
            for (i, &j) in sel.iter().enumerate() {
                assert!(dn(&bf.y[c * mf + i], &a.y[c * m + j]) <= 1e-6 * s);
            }
        }
    }
    println!("sets compared in {:.1}s", t.elapsed().as_secs_f64());
    // noise: lazy noisy − lazy clean is white with σ²/2 per component
    let sigma = legacy.meta["noise_sigma"].as_f64().unwrap();
    unsafe { std::env::remove_var("HASTY_NOISE_SIGMA") };
    let noisy = SimData::load_lazy(&dir).unwrap();
    let ros = &sets[0];
    let a = lazy.load_set(ros, &LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
    let b = noisy.load_set(ros, &LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
    let m = a.tau.len();
    let n: Vec<Complex64> = a.y.iter().zip(&b.y).map(|(x, y)| Complex64::new((y.re - x.re) as f64, (y.im - x.im) as f64)).collect();
    let var = n.iter().map(|z| z.norm_sqr()).sum::<f64>() / (2 * n.len()) as f64;
    let mut off = 0.0f64;
    for c in 0..nc.min(8) {
        for c2 in 0..c {
            let ip: Complex64 = (0..m).map(|j| n[c * m + j] * n[c2 * m + j].conj()).sum();
            off = off.max(ip.norm() / (m as f64 * sigma * sigma));
        }
    }
    println!("noise per component: var {var:.4e} (σ²/2 = {:.4e}), largest |coil correlation| {off:.4}", sigma * sigma / 2.0);
    assert!((var / (sigma * sigma / 2.0) - 1.0).abs() < 0.01);
    assert!(off < 5.0 / (m as f64).sqrt());
    // same noise however the set is split (counter-based)
    let half = noisy.load_set(&ros[ros.len() / 2..], &LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
    let mh = half.tau.len();
    for c in [0, nc - 1] {
        assert_eq!(&half.y[c * mh..(c + 1) * mh], &b.y[c * m + (m - mh)..(c + 1) * m]);
    }
    // a store of real data: ratios and speeds per representation (lossy one measured against the noise)
    let sc = noisy.load_set(ros, &LoadOpts { w: &w, cc: Some(&cc.rows), keep: None, data: true }).unwrap();
    let nvc = cc.nvc();
    let raw_mb = (sc.y.len() * 8) as f64 / 1e6;
    for spec in ["ram", "lz4", "zstd1", "zstd3", "zstd1+q0.25", "zstd1+q0.5", "lz4+q0.25"] {
        let ks = KStore::create(StoreSpec::parse(spec, Path::new("/unused")).unwrap(), nvc, vec![m], Some(sigma / 2f64.sqrt()), "").unwrap();
        let t = Instant::now();
        ks.put(0, &sc.y).unwrap();
        let te = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let back = ks.read_vec(0).unwrap();
        let td = t.elapsed().as_secs_f64();
        let err = back.iter().zip(&sc.y).map(|(x, y)| (dn(x, y) as f64).powi(2)).sum::<f64>() / (2 * back.len()) as f64;
        println!("{spec:>12}: ×{:.2}, encode {:.0} MB/s, decode {:.0} MB/s, added variance / noise variance {:.2e}", ks.stats().ratio(), raw_mb / te, raw_mb / td, err / (sigma * sigma / 2.0));
    }
}
