//! CPU, real pipeline data (ignored by default; needs the store): the BFP k-space tier on a k-space store the
//! reconstruction built (whitened, coil-compressed data with noise; lossless zstd), against the σ rule of
//! research/bfp_storage.md §3.6. σ is the store's noise level as the pipeline recorded it in the store key
//! (`sigma=Some(..)`: the whitened per-component noise std the loader works with); a data-only estimate from the
//! weakest virtual coil's median absolute value is printed next to it as a cross-check (signal inflates it).
//!   HASTY_KSPACE_STORE=<path.kst> cargo test --release -p hasty-recon --test kspace_bfp_real -- --ignored --nocapture
use std::time::Instant;

use hasty_bfp::{Bfp, BfpBuf};
use hasty_cuda::C32;
use hasty_recon::kstore::{BfpRule, KStore, StoreSpec};
use rayon::prelude::*;

#[test]
#[ignore]
fn sigma_rule_on_a_real_store() {
    let path = std::env::var("HASTY_KSPACE_STORE").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/kstore/69d8d33581799b5c.kst").to_string());
    let ks = KStore::open(std::path::Path::new(&path)).expect("k-space store");
    let tag = ks.tag().to_string();
    let sigma: f64 = tag.split("sigma=Some(").nth(1).and_then(|r| r.split(')').next()).and_then(|v| v.parse().ok()).expect("sigma in the store key");
    let every: usize = std::env::var("EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let (nvc, nsets) = (ks.nvc(), ks.n_sets());
    println!("store {path}: {nsets} sets × {nvc} virtual coils, {} samples; σ {sigma:.4} per component (store key); every {every}th set", ks.lens().iter().sum::<usize>());
    let rule = BfpRule::Sigma { q: 0.25 };
    let mut tot = std::collections::BTreeMap::<String, (f64, f64, usize, f64)>::new(); // (Σ err², max err, n reals, bytes)
    let mut hist = std::collections::BTreeMap::<u32, usize>::new();
    let (mut peak_max, mut mad_last) = (0f32, Vec::new());
    let (mut t_enc, mut t_dec, mut raw_b) = (0.0f64, 0.0f64, 0usize);
    for t in (0..nsets).step_by(every) {
        let y = ks.read_vec(t).unwrap();
        let m = ks.len(t);
        let peak = y.par_iter().map(|z| z.re.abs().max(z.im.abs())).reduce(|| 0.0f32, f32::max);
        peak_max = peak_max.max(peak);
        // data-only σ: median |component| of the weakest virtual coil / 0.6745
        let mut last: Vec<f32> = y[(nvc - 1) * m..].iter().flat_map(|z| [z.re.abs(), z.im.abs()]).collect();
        last.sort_by(|a, b| a.partial_cmp(b).unwrap());
        mad_last.push(last[last.len() / 2] as f64 / 0.6745);
        let fr = rule.bits(peak, Some(sigma)).unwrap();
        *hist.entry(fr.bits()).or_default() += 1;
        for (name, fmt) in [("rule", fr), ("bfp16", Bfp::new(16)), ("bfp14", Bfp::new(14)), ("bfp12", Bfp::new(12))] {
            let t0 = Instant::now();
            let rows: Vec<BfpBuf> = y.par_chunks(m).map(|r| BfpBuf::encode_c(fmt, r)).collect();
            let te = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let dec: Vec<Vec<C32>> = rows.par_iter().map(|b| b.decode_c()).collect();
            let td = t1.elapsed().as_secs_f64();
            if name == "rule" {
                t_enc += te;
                t_dec += td;
                raw_b += y.len() * 8;
            }
            let (s2, mx) = dec.par_iter().zip(y.par_chunks(m)).map(|(d, r)| {
                d.iter().zip(r).fold((0.0f64, 0.0f64), |(s, mx), (a, b)| {
                    let (er, ei) = ((a.re - b.re) as f64, (a.im - b.im) as f64);
                    (s + er * er + ei * ei, mx.max(er.abs()).max(ei.abs()))
                })
            }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1.max(b.1)));
            let e = tot.entry(name.to_string()).or_default();
            e.0 += s2;
            e.1 = e.1.max(mx);
            e.2 += 2 * y.len();
            e.3 += rows.iter().map(|b| b.bytes()).sum::<usize>() as f64;
        }
    }
    mad_last.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("peak / σ {:.0}; data-only σ estimate (weakest coil, median over sets) {:.4} = {:.3} × the key's", peak_max as f64 / sigma, mad_last[mad_last.len() / 2], mad_last[mad_last.len() / 2] / sigma);
    println!("σ rule widths: {}", hist.iter().map(|(b, n)| format!("{b} bits × {n}")).collect::<Vec<_>>().join(", "));
    for (name, (s2, mx, n, bytes)) in &tot {
        println!("{name:>6}: rms error {:.2e} σ, max {:.3} σ, {:.3} bytes per complex ({:.3} of f32)", (s2 / *n as f64).sqrt() / sigma, mx / sigma, bytes / (*n as f64 / 2.0), bytes / (*n as f64 * 4.0));
    }
    println!("host codec on this data (σ-rule width, {} rayon threads): encode {:.2} GB/s, decode {:.2} GB/s of f32 bytes", rayon::current_num_threads(), raw_b as f64 / t_enc / 1e9, raw_b as f64 / t_dec / 1e9);
    let (s2, mx, n, _) = tot["rule"];
    assert!(mx <= 0.125 * sigma * 1.0000001, "σ rule: worst error {:.3} σ above half the step bound", mx / sigma);
    assert!((s2 / n as f64).sqrt() / sigma < 5e-3);
    // a store built with the rule reports the same widths
    let sp = StoreSpec::parse("bfp", std::path::Path::new("/unused")).unwrap();
    let small = KStore::create(sp, nvc, vec![ks.len(0)], Some(sigma), "t").unwrap();
    small.put(0, &ks.read_vec(0).unwrap()).unwrap();
    println!("store built from set 0: {}", small.report());
}
