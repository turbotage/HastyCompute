//! GPU: pruned channel-interleaved 3-D FFT (`PrunedChanFft3`) against the full partial-axis transform (`FftPlan::
//! new_partial`): forward from wrapped-block data written into a garbage-filled buffer (no zeroing), inverse restricted
//! to the block; mixed-radix and Bluestein lengths; timings at a NUFFT-like size.
//!   cargo test --release -p hasty-fft --test chanfft -- --nocapture

use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_fft::{Direction, FftPlan, Norm, PrunedChanFft3};

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
