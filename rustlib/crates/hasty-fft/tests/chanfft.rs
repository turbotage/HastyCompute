//! GPU: pruned channel-interleaved 3-D FFT (`PrunedChanFft3`) against the full partial-axis transform (`FftPlan::
//! new_partial`): forward from wrapped-block data written into a garbage-filled buffer (no zeroing), inverse restricted
//! to the block; mixed-radix and Bluestein lengths; timings at a NUFFT-like size.
//!   cargo test --release -p hasty-fft --test chanfft -- --nocapture

use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_fft::{Direction, FftPlan, KSupport, PrunedChanFft3};
use hasty_fft::Norm;

fn in_block(i: usize, n: usize, (h, l): (usize, usize)) -> bool {
    i < h || i >= n - l
}

fn check(dev: &Device, n: [usize; 3], nch: usize, keep: [(usize, usize); 3]) {
    let st = dev.default_stream();
    let len = n.iter().product::<usize>() * nch;
    let dims = [n[0], n[1], n[2], nch];
    let tr = [true, true, true, false];
    let full_f = FftPlan::<f32>::new_partial(dev, &dims, &tr, Direction::Forward, Norm::None).unwrap();
    let full_i = FftPlan::<f32>::new_partial(dev, &dims, &tr, Direction::Inverse, Norm::None).unwrap();
    let pr = PrunedChanFft3::<f32>::new(dev, n, nch, keep, 1.0).unwrap();
    let idx = |q: usize| -> (usize, usize, usize) { let p = q / nch; (p / (n[1] * n[2]), (p / n[2]) % n[1], p % n[2]) };
    let blk = |q: usize| { let (z, y, x) = idx(q); in_block(z, n[0], keep[0]) && in_block(y, n[1], keep[1]) && in_block(x, n[2], keep[2]) };
    let val = |q: usize| C32::new(((q as f32) * 0.37).sin(), ((q as f32) * 0.11 + 0.3).cos());
    // forward: reference from a zero-padded buffer; pruned from a buffer with garbage outside the block
    let zero: Vec<C32> = (0..len).map(|q| if blk(q) { val(q) } else { C32::new(0.0, 0.0) }).collect();
    let garb: Vec<C32> = (0..len).map(|q| if blk(q) { val(q) } else { C32::new(1e3, -7e2) }).collect();
    let mut a = st.clone_htod(&zero).unwrap();
    let mut b = st.clone_htod(&garb).unwrap();
    full_f.execute_inplace(&st, &mut a).unwrap();
    pr.forward(&st, &mut b).unwrap();
    let (ha, hb) = (st.clone_dtoh(&a).unwrap(), st.clone_dtoh(&b).unwrap());
    let err = |x: &[C32], y: &[C32], sel: &dyn Fn(usize) -> bool| {
        let (mut d, mut s) = (0.0f64, 0.0f64);
        for q in (0..x.len()).filter(|&q| sel(q)) {
            d += ((x[q].re - y[q].re) as f64).powi(2) + ((x[q].im - y[q].im) as f64).powi(2);
            s += (y[q].re as f64).powi(2) + (y[q].im as f64).powi(2);
        }
        (d / s).sqrt()
    };
    let ef = err(&hb, &ha, &|_| true);
    // inverse: full spectrum in, compare on the block
    let spec: Vec<C32> = (0..len).map(val).collect();
    let mut a = st.clone_htod(&spec).unwrap();
    let mut b = st.clone_htod(&spec).unwrap();
    full_i.execute_inplace(&st, &mut a).unwrap();
    pr.inverse(&st, &mut b).unwrap();
    let ei = err(&st.clone_dtoh(&b).unwrap(), &st.clone_dtoh(&a).unwrap(), &blk);
    eprintln!("n {n:?} × {nch} channels, block {keep:?}: forward {ef:.2e}, inverse (block) {ei:.2e}");
    assert!(ef < 2e-6 && ei < 2e-6, "pruned transform differs: forward {ef:.2e}, inverse {ei:.2e}");
}

#[test]
fn pruned_matches_full() {
    let dev = Device::new(0).expect("CUDA device");
    check(&dev, [20, 24, 30], 3, [(8, 8), (10, 9), (12, 12)]);
    check(&dev, [16, 20, 24], 8, [(16, 0), (10, 10), (1, 0)]);
    // Bluestein lengths (prime factor 17)
    check(&dev, [34, 20, 34], 4, [(13, 13), (8, 7), (14, 13)]);
}

#[test]
fn pruned_timing() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // a NUFFT grid at σ = 1.25 (image 144 × 144 × 40 → 180 × 180 × 50), 16 channels; and σ = 2
    for (n, img, nch) in [([50usize, 180, 180], [40usize, 144, 144], 16usize), ([80, 288, 288], [40, 144, 144], 4)] {
        let keep = [0, 1, 2].map(|a| (img[a] - img[a] / 2, img[a] / 2));
        let dims = [n[0], n[1], n[2], nch];
        let tr = [true, true, true, false];
        let full = FftPlan::<f32>::new_partial(&dev, &dims, &tr, Direction::Forward, Norm::None).unwrap();
        let fulli = FftPlan::<f32>::new_partial(&dev, &dims, &tr, Direction::Inverse, Norm::None).unwrap();
        let pr = PrunedChanFft3::<f32>::new(&dev, n, nch, keep, 1.0).unwrap();
        let mut buf = st.alloc_zeros::<C32>(pr.len()).unwrap();
        // the GPU may be shared: interleave the variants over rounds and keep each one's best round
        let mut best = [f64::INFINITY; 5];
        for _ in 0..6 {
            let mut fs: [&mut dyn FnMut(&mut cudarc::driver::CudaSlice<C32>); 5] = [
                &mut |b| st.memset_zeros(b).unwrap(),
                &mut |b| full.execute_inplace(&st, b).unwrap(),
                &mut |b| pr.forward(&st, b).unwrap(),
                &mut |b| fulli.execute_inplace(&st, b).unwrap(),
                &mut |b| pr.inverse(&st, b).unwrap(),
            ];
            for (k, f) in fs.iter_mut().enumerate() {
                f(&mut buf);
                st.synchronize().unwrap();
                let t = Instant::now();
                for _ in 0..10 {
                    f(&mut buf);
                }
                st.synchronize().unwrap();
                best[k] = best[k].min(t.elapsed().as_secs_f64() / 10.0 * 1e3);
            }
        }
        let [tz, tf, tp, tfi, tpi] = best;
        eprintln!(
            "grid {n:?} × {nch} (image {img:?}): forward memset + full {:.2} ms → pruned {tp:.2} ms ({:.2}×); inverse full {tfi:.2} → pruned {tpi:.2} ms ({:.2}×)",
            tz + tf,
            (tz + tf) / tp,
            tfi / tpi
        );
    }
}

/// k-support of a ball of radius `r` (index units) around k = 0 on a wrapped grid `n = [n0, n1, n2]` (x fastest), or
/// of the cube |k_a| ≤ r_a when `ball` is false
fn ksup(n: [usize; 3], r: [f64; 3], ball: bool) -> KSupport {
    let c = |i: usize, m: usize| { let i = i as f64; if i > m as f64 / 2.0 { i - m as f64 } else { i } };
    if ball {
        // per-column kz chord of the ellipsoid
        let q = |x: usize, y: usize| 1.0 - (c(x, n[0]) / r[0]).powi(2) - (c(y, n[1]) / r[1]).powi(2);
        KSupport::from_columns(n, |x, y| q(x, y) >= 0.0, |x, y| KSupport::hull(n[2], |z| c(z, n[2]).abs() <= r[2] * q(x, y).max(0.0).sqrt()))
    } else {
        KSupport::from_masks(n, |x, y| c(x, n[0]).abs() <= r[0] && c(y, n[1]).abs() <= r[1], |z| c(z, n[2]).abs() <= r[2])
    }
}

/// support-pruned transform against the full one: forward on the support cells (garbage outside the image block on
/// input), inverse on the block from a spectrum that is zero outside the support (garbage in the unused cells is not
/// allowed there: the inverse reads only the support)
fn check_ks(dev: &Device, n: [usize; 3], nch: usize, keep: [(usize, usize); 3], ks: &KSupport) {
    let st = dev.default_stream();
    let len = n.iter().product::<usize>() * nch;
    let dims = [n[0], n[1], n[2], nch];
    let tr = [true, true, true, false];
    let full_f = FftPlan::<f32>::new_partial(dev, &dims, &tr, Direction::Forward, Norm::None).unwrap();
    let full_i = FftPlan::<f32>::new_partial(dev, &dims, &tr, Direction::Inverse, Norm::None).unwrap();
    let pr = PrunedChanFft3::<f32>::new_ksupport(dev, n, nch, keep, 1.0, Some(ks)).unwrap();
    let idx = |q: usize| -> (usize, usize, usize) { let p = q / nch; (p / (n[1] * n[2]), (p / n[2]) % n[1], p % n[2]) };
    let blk = |q: usize| { let (z, y, x) = idx(q); in_block(z, n[0], keep[0]) && in_block(y, n[1], keep[1]) && in_block(x, n[2], keep[2]) };
    let sup = |q: usize| { let (z, y, x) = idx(q); let zr = ks.zcols[y * n[2] + x]; zr != (0, 0) && in_block(z, n[0], zr) };
    let val = |q: usize| C32::new(((q as f32) * 0.37).sin(), ((q as f32) * 0.11 + 0.3).cos());
    let zero: Vec<C32> = (0..len).map(|q| if blk(q) { val(q) } else { C32::new(0.0, 0.0) }).collect();
    let garb: Vec<C32> = (0..len).map(|q| if blk(q) { val(q) } else { C32::new(1e3, -7e2) }).collect();
    let mut a = st.clone_htod(&zero).unwrap();
    let mut b = st.clone_htod(&garb).unwrap();
    full_f.execute_inplace(&st, &mut a).unwrap();
    pr.forward(&st, &mut b).unwrap();
    let err = |x: &[C32], y: &[C32], sel: &dyn Fn(usize) -> bool| {
        let (mut d, mut s) = (0.0f64, 0.0f64);
        for q in (0..x.len()).filter(|&q| sel(q)) {
            d += ((x[q].re - y[q].re) as f64).powi(2) + ((x[q].im - y[q].im) as f64).powi(2);
            s += (y[q].re as f64).powi(2) + (y[q].im as f64).powi(2);
        }
        (d / s).sqrt()
    };
    let ef = err(&st.clone_dtoh(&b).unwrap(), &st.clone_dtoh(&a).unwrap(), &sup);
    let spec: Vec<C32> = (0..len).map(|q| if sup(q) { val(q) } else { C32::new(0.0, 0.0) }).collect();
    let mut a = st.clone_htod(&spec).unwrap();
    let mut b = st.clone_htod(&spec).unwrap();
    full_i.execute_inplace(&st, &mut a).unwrap();
    pr.inverse(&st, &mut b).unwrap();
    let ei = err(&st.clone_dtoh(&b).unwrap(), &st.clone_dtoh(&a).unwrap(), &blk);
    let (cols, pj) = ks.fractions([n[2], n[1], n[0]]);
    let cells = ks.cell_fraction([n[2], n[1], n[0]]);
    eprintln!("n {n:?} × {nch}, block {keep:?}, support columns {cols:.2}, proj {pj:.2?}, cells {cells:.2}: forward (support) {ef:.2e}, inverse (block) {ei:.2e}");
    assert!(ef < 2e-6 && ei < 2e-6, "k-support-pruned transform differs: forward {ef:.2e}, inverse {ei:.2e}");
}

#[test]
fn ksupport_matches_full() {
    let dev = Device::new(0).expect("CUDA device");
    // n = [n2, n1, n0]; supports in [x, y, z] order
    check_ks(&dev, [20, 24, 30], 3, [(8, 8), (10, 9), (12, 12)], &ksup([30, 24, 20], [11.0, 9.5, 8.0], true));
    check_ks(&dev, [20, 24, 30], 3, [(8, 8), (10, 9), (12, 12)], &ksup([30, 24, 20], [12.0, 10.0, 9.0], false));
    check_ks(&dev, [16, 20, 24], 8, [(16, 0), (10, 10), (1, 0)], &ksup([24, 20, 16], [4.0, 3.0, 2.5], true));
    // the full support equals the block-pruned transform
    check_ks(&dev, [20, 24, 30], 2, [(8, 8), (10, 9), (12, 12)], &KSupport::full([30, 24, 20]));
}

#[test]
fn ksupport_timing() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // a 1 mm NUFFT-like grid at σ = 1.25 reduced to 16 channels: image 198 × 189 × 228 → 250 × 240 × 288, half-width W/2
    let (img, nf, nch, hw) = ([228usize, 189, 198], [288usize, 240, 250], 16usize, 5.0);
    let keep = [0, 1, 2].map(|a| (img[a] - img[a] / 2, img[a] / 2));
    // the samples' k extent |k_a| ≤ 0.98 π spans ±0.49 nf_a fine-grid cells (σ refines the k grid, it does not extend
    // it), plus the kernel half-width: a ball (3-D radial, yarn ball) prunes the z-pass columns outside its disk, a
    // cylinder (stack of spirals) the same columns; the half-extent cube (an image-sized k box) is for comparison only
    let nx = [nf[2], nf[1], nf[0]];
    let rb = [0, 1, 2].map(|a| 0.49 * nx[a] as f64 + hw);
    let rh = [2, 1, 0].map(|a| img[a] as f64 / 2.0 + hw);
    let cases = [("block only", None), ("ball k-support", Some(ksup(nx, rb, true))), ("half-extent cube", Some(ksup(nx, rh, false)))];
    let plans: Vec<_> = cases.iter().map(|(_, k)| PrunedChanFft3::<f32>::new_ksupport(&dev, nf, nch, keep, 1.0, k.as_ref()).unwrap()).collect();
    let mut buf = st.alloc_zeros::<C32>(plans[0].len()).unwrap();
    let mut best = vec![[f64::INFINITY; 2]; plans.len()];
    for _ in 0..5 {
        for (k, p) in plans.iter().enumerate() {
            for d in 0..2 {
                let mut f = || if d == 0 { p.forward(&st, &mut buf).unwrap() } else { p.inverse(&st, &mut buf).unwrap() };
                f();
                st.synchronize().unwrap();
                let t = Instant::now();
                for _ in 0..5 {
                    f();
                }
                st.synchronize().unwrap();
                best[k][d] = best[k][d].min(t.elapsed().as_secs_f64() / 5.0 * 1e3);
            }
        }
    }
    for (k, (name, ks)) in cases.iter().enumerate() {
        let fr = ks.as_ref().map(|k| (k.fractions(nx), k.cell_fraction(nx)));
        eprintln!("{nf:?} × {nch}: {name:<16} forward {:.2} ms, inverse {:.2} ms ({:.2}× / {:.2}× of block-only){}", best[k][0], best[k][1], best[0][0] / best[k][0], best[0][1] / best[k][1], fr.map_or(String::new(), |((c, p), v)| format!(", columns {c:.2}, proj {p:.2?}, cells {v:.2}")));
    }
}

/// two-pass L2 slabs (`set_slab`): the same kernels on the same lines slab by slab, so forward and inverse equal the
/// whole-grid passes bit for bit, block-pruned and k-support pruned, slab sizes that do and do not divide the block's
/// head / tail z ranges
#[test]
fn slab_passes_equal_whole_grid() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let (n, nch, keep) = ([20usize, 24, 30], 3usize, [(8usize, 8usize), (10, 9), (12, 12)]);
    for ks in [None, Some(ksup([30, 24, 20], [11.0, 9.5, 8.0], true))] {
        let mut pr = PrunedChanFft3::<f32>::new_ksupport(&dev, n, nch, keep, 1.0, ks.as_ref()).unwrap();
        let len = pr.len();
        let val: Vec<C32> = (0..len).map(|q| C32::new(((q as f32) * 0.37).sin(), ((q as f32) * 0.11 + 0.3).cos())).collect();
        let run = |pr: &PrunedChanFft3<f32>| {
            let mut a = st.clone_htod(&val).unwrap();
            pr.forward(&st, &mut a).unwrap();
            let f = st.clone_dtoh(&a).unwrap();
            let mut b = st.clone_htod(&val).unwrap();
            pr.inverse(&st, &mut b).unwrap();
            (f, st.clone_dtoh(&b).unwrap())
        };
        let (f0, i0) = run(&pr);
        for dz in [1usize, 3, 8, 100] {
            pr.set_slab(dz);
            let (f, i) = run(&pr);
            let same = |x: &[C32], y: &[C32]| x.iter().zip(y).all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits());
            eprintln!("slab {dz} planes, k-support {}: forward identical {}, inverse identical {}", ks.is_some(), same(&f, &f0), same(&i, &i0));
            assert!(same(&f, &f0) && same(&i, &i0), "slab {dz}: results differ from the whole-grid passes");
        }
    }
}
