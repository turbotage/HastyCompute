//! Wavelet throughput: the db4 complex shrinkage (shared engine, f32 SIMD, threads within each image) against the
//! previous f64 per-image implementation, and the divergence-free wavelet shrinkage of a velocity field.
//!   cargo test --release -p hasty-recon --test wavelet_bench -- --ignored --nocapture

use hasty_cuda::C32;
use hasty_recon::dfwavelet::{DfRule, DfWavelet};
use hasty_recon::wavelet::Wavelet;
use rayon::prelude::*;
use std::time::Instant;

fn rnd(n: usize, mut s: u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

// ── previous implementation (f64, per-line Vec, one image per thread) for reference ──
const DB4: [f64; 8] = [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523];
fn fwd1(x: &[f64], out: &mut [f64]) {
    let n = x.len();
    let h = n / 2;
    for k in 0..h {
        let (mut a, mut dd) = (0.0, 0.0);
        for (t, &c) in DB4.iter().enumerate() {
            let v = x[(2 * k + t) % n];
            a += c * v;
            let g = if t % 2 == 0 { DB4[7 - t] } else { -DB4[7 - t] };
            dd += g * v;
        }
        out[k] = a;
        out[h + k] = dd;
    }
}
fn inv1(y: &[f64], out: &mut [f64]) {
    let n = y.len();
    let h = n / 2;
    out.iter_mut().for_each(|v| *v = 0.0);
    for k in 0..h {
        for (t, &c) in DB4.iter().enumerate() {
            let g = if t % 2 == 0 { DB4[7 - t] } else { -DB4[7 - t] };
            out[(2 * k + t) % n] += c * y[k] + g * y[h + k];
        }
    }
}
fn lines(buf: &mut [f64], p: [usize; 3], sz: [usize; 3], axis: usize, f: &dyn Fn(&[f64], &mut [f64])) {
    let n = sz[axis];
    let stride = [1, p[0], p[0] * p[1]][axis];
    let (o1, o2) = match axis { 0 => (1, 2), 1 => (0, 2), _ => (0, 1) };
    let s1 = [1, p[0], p[0] * p[1]][o1];
    let s2 = [1, p[0], p[0] * p[1]][o2];
    let mut line = vec![0f64; n];
    let mut out = vec![0f64; n];
    for j2 in 0..sz[o2] {
        for j1 in 0..sz[o1] {
            let base = j1 * s1 + j2 * s2;
            for i in 0..n {
                line[i] = buf[base + i * stride];
            }
            f(&line, &mut out);
            for i in 0..n {
                buf[base + i * stride] = out[i];
            }
        }
    }
}
fn old_shrink(img: &mut [C32], d: [usize; 3], levels: usize) {
    let m = 1 << levels;
    let p = [0, 1, 2].map(|a| (d[a] + m - 1).div_ceil(m) * m);
    let np = p[0] * p[1] * p[2];
    let mut re = vec![0f64; np];
    let mut im = vec![0f64; np];
    for z in 0..d[2] {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let v = img[x + d[0] * (y + d[1] * z)];
                re[x + p[0] * (y + p[1] * z)] = v.re as f64;
                im[x + p[0] * (y + p[1] * z)] = v.im as f64;
            }
        }
    }
    for buf in [&mut re, &mut im] {
        let mut sz = p;
        for _ in 0..levels {
            for axis in 0..3 {
                lines(buf, p, sz, axis, &fwd1);
            }
            sz = [sz[0] / 2, sz[1] / 2, sz[2] / 2];
        }
    }
    let tau = 0.01;
    for q in 0..np {
        let mag = (re[q] * re[q] + im[q] * im[q]).sqrt();
        let s = if mag > tau { 1.0 - tau / mag } else { 0.0 };
        re[q] *= s;
        im[q] *= s;
    }
    for buf in [&mut re, &mut im] {
        for l in (0..levels).rev() {
            let sz = [p[0] >> l, p[1] >> l, p[2] >> l];
            for axis in (0..3).rev() {
                lines(buf, p, sz, axis, &inv1);
            }
        }
    }
    for z in 0..d[2] {
        for y in 0..d[1] {
            for x in 0..d[0] {
                img[x + d[0] * (y + d[1] * z)] = C32::new(re[x + p[0] * (y + p[1] * z)] as f32, im[x + p[0] * (y + p[1] * z)] as f32);
            }
        }
    }
}

fn best<F: FnMut()>(rounds: usize, mut f: F) -> f64 {
    (0..rounds).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min)
}

#[test]
#[ignore]
fn wavelet_throughput() {
    let threads = rayon::current_num_threads();
    for (d, n) in [([128usize, 128, 96], 8usize), ([240, 200, 160], 2)] {
        let nv: usize = d.iter().product();
        let base: Vec<C32> = rnd(2 * nv * n, 5).chunks(2).map(|c| C32::new(c[0], c[1])).collect();
        let w = Wavelet::new(d, 3, 1.0);
        let mut a = base.clone();
        let t_new = best(3, || w.apply(&mut a, n));
        let mut b = base.clone();
        // the previous code: one image per thread
        let t_old = best(1, || b.par_chunks_mut(nv).for_each(|img| old_shrink(img, d, 3)));
        let gvox = (nv * n) as f64 / 1e9;
        println!("db4 3 levels {d:?} × {n} images ({threads} threads): new {:.3} s ({:.2} Gvox/s), previous {:.3} s ({:.2} Gvox/s), speed-up {:.1}×", t_new, gvox / t_new, t_old, gvox / t_old, t_old / t_new);
        // one image alone (the previous code used one thread for it)
        let mut a1 = base[..nv].to_vec();
        let t1 = best(3, || w.apply(&mut a1, 1));
        let mut b1 = base[..nv].to_vec();
        let t1o = best(1, || old_shrink(&mut b1, d, 3));
        println!("  single image: new {:.3} s, previous {:.3} s ({:.1}×)", t1, t1o, t1o / t1);
        // divergence-free wavelets on a velocity field, magnitude-weighted SURE
        let mut df = DfWavelet::new(d, 3, [1.0; 3], DfRule::Sure { tdf: 1.0, tn: 1.0 });
        df.set_iteration(1);
        let v0 = rnd(3 * nv, 9);
        let wgt: Vec<f32> = rnd(nv, 13).iter().map(|v| v.abs() + 0.1).collect();
        let mut v = v0.clone();
        df.apply(&mut v, Some(&wgt)); // noise amplitudes (once per size)
        let t_df = best(3, || { v.copy_from_slice(&v0); df.apply(&mut v, Some(&wgt)); });
        println!("  DFW 3 levels, SURE, weighted: {:.3} s per field ({:.2} Gvox/s)", t_df, nv as f64 / 1e9 / t_df);
    }
}
