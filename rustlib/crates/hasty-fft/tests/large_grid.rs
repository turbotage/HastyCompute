//! Line counts beyond the 65535 grid y limit (1 mm full-torso fine grids: 800 × 480 × 480 at σ = 1.25 has 480 · 480 x
//! lines per channel). Before `launch_grid` folded the line groups into grid (y, z), such passes failed to launch with
//! CUDA_ERROR_INVALID_VALUE (`segnufft_bench 640 384 384 …`), and `PrunedChanFft3` refused blocks of > 65535 x lines
//! (the segmented NUFFT then fell back to the unpruned transform, which hit the same launch error).
//!   cargo test --release -p hasty-fft --test large_grid -- --nocapture
//!   (the full-size case is ignored by default: 3 GB of GPU memory) … -- --ignored --nocapture

use hasty_cuda::{Device, C32};
use hasty_fft::{launch_grid, AxisPlan, Direction, FftPlan, Norm, PrunedChanFft3, GRID_YZ_MAX};
use hasty_testkit::{fft_nd, rel_l2, Rng, C64};

#[test]
fn launch_grid_folds_line_groups() {
    // 1 mm full torso: x pass of the channel-interleaved transform, 40 channels, 480 · 480 line groups
    for (la, lb, lpb) in [(40usize, 480 * 480, 16usize), (1, 384 * 384, 1), (2, 65535, 2), (2, 65536, 2), (3, 69169, 4), (8, 1, 16), (8, 65535 * 3 + 1, 8)] {
        let (gx, gy, gz) = launch_grid(false, la, lb, lpb).unwrap();
        assert!(gy as usize <= GRID_YZ_MAX && gz as usize <= GRID_YZ_MAX, "{la} × {lb}: grid {gx} × {gy} × {gz}");
        let cover = gy as usize * gz as usize;
        assert!(cover >= lb && cover - lb < gz as usize, "{la} × {lb}: {gy} × {gz} line-group blocks");
        assert_eq!(gx as usize, la.div_ceil(lpb));
        if lb <= GRID_YZ_MAX {
            assert_eq!((gy as usize, gz), (lb, 1), "unfolded below the limit");
        }
    }
    // contiguous passes: all lines over grid x
    assert_eq!(launch_grid(true, 480 * 480 * 40, 1, 2).unwrap(), (480 * 480 * 20, 1, 1));
    // beyond y · z
    assert!(launch_grid(false, 1, GRID_YZ_MAX * GRID_YZ_MAX + 1, 1).is_err());
    assert!(launch_grid(true, 1 << 31, 1, 1).is_err());
}

fn c64(v: &[C32]) -> Vec<C64> {
    v.iter().map(|c| C64::new(c.re as f64, c.im as f64)).collect()
}

/// strided 1-D pass (AxisPlan::new with inner > 1) with more than 65535 line groups against rustfft
#[test]
fn strided_pass_many_groups() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // (n, inner, outer): mixed radix and Bluestein (n = 17), each with a padding block of the folded grid
    for (n, inner, outer) in [(8usize, 2usize, 65535 * 2 + 2), (17, 3, 70001)] {
        let total = n * inner * outer;
        let x = Rng::new(n as u64).complex_vec(total);
        let plan = AxisPlan::<f32>::new(&dev, n, inner, outer, Direction::Forward, 1.0).unwrap();
        assert!(plan.grid().2 > 1, "case must fold the line groups (grid {:?})", plan.grid());
        let mut d = st.clone_htod(&x.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
        plan.execute_inplace(&st, &mut d).unwrap();
        let got = c64(&st.clone_dtoh(&d).unwrap());
        // reference: lines (o, i) of length n at stride inner, transformed one by one
        let mut want = x.clone();
        for o in 0..outer {
            for i in 0..inner {
                let line: Vec<C64> = (0..n).map(|k| x[(o * n + k) * inner + i]).collect();
                let f = fft_nd(&line, &[n], -1);
                for k in 0..n {
                    want[(o * n + k) * inner + i] = f[k];
                }
            }
        }
        let e = rel_l2(&got, &want);
        eprintln!("strided n {n} inner {inner} outer {outer} (grid {:?}): rel err {e:.2e}", plan.grid());
        assert!(e < 2e-6, "strided pass with {outer} line groups: rel err {e:.2e}");
    }
}

/// channel-interleaved 3-D transform pruned to a block of > 65535 x lines (and the full partial-axis plan, whose x
/// pass has n[0] · n[1] line groups) against rustfft per channel
#[test]
fn pruned_many_x_lines() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    // n = [n2, n1, n0] (z, y, x; x fastest), 263 prime (Bluestein); block 260 × 260 x lines (67600 > 65535)
    let (n, nch) = ([263usize, 263, 4], 2usize);
    let keep = [(140usize, 120usize), (140, 120), (2, 1)];
    let inb = |i: usize, len: usize, (h, l): (usize, usize)| i < h || i >= len - l;
    let len = n.iter().product::<usize>() * nch;
    let x = Rng::new(7).complex_vec(len);
    let blk = |q: usize| {
        let p = q / nch;
        let (z, y, xx) = (p / (n[1] * n[2]), (p / n[2]) % n[1], p % n[2]);
        inb(z, n[0], keep[0]) && inb(y, n[1], keep[1]) && inb(xx, n[2], keep[2])
    };
    let up = |v: &[C64]| st.clone_htod(&v.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
    // reference per channel (channels interleaved fastest)
    let per_channel = |v: &[C64], sign: i32| -> Vec<C64> {
        let mut out = vec![C64::new(0.0, 0.0); len];
        for c in 0..nch {
            let ch: Vec<C64> = (0..len / nch).map(|p| v[p * nch + c]).collect();
            let f = fft_nd(&ch, &n, sign);
            for p in 0..len / nch {
                out[p * nch + c] = f[p];
            }
        }
        out
    };
    let zeroed: Vec<C64> = (0..len).map(|q| if blk(q) { x[q] } else { C64::new(0.0, 0.0) }).collect();
    let garbage: Vec<C64> = (0..len).map(|q| if blk(q) { x[q] } else { C64::new(1e3, -7e2) }).collect();
    let want_f = per_channel(&zeroed, -1);
    let pr = PrunedChanFft3::<f32>::new(&dev, n, nch, keep, 1.0).expect("pruned plan with > 65535 x lines");
    let mut b = up(&garbage);
    pr.forward(&st, &mut b).unwrap();
    let ef = rel_l2(&c64(&st.clone_dtoh(&b).unwrap()), &want_f);
    // full partial-axis plan (the segmented NUFFT's fallback): x pass over n[0] · n[1] = 69169 line groups
    let fp = FftPlan::<f32>::new_partial(&dev, &[n[0], n[1], n[2], nch], &[true, true, true, false], Direction::Forward, Norm::None).unwrap();
    let mut a = up(&zeroed);
    fp.execute_inplace(&st, &mut a).unwrap();
    let efull = rel_l2(&c64(&st.clone_dtoh(&a).unwrap()), &want_f);
    // inverse: full spectrum in, the block out
    let want_i = per_channel(&x, 1);
    let mut b = up(&x);
    pr.inverse(&st, &mut b).unwrap();
    let got_i = c64(&st.clone_dtoh(&b).unwrap());
    let sel: Vec<usize> = (0..len).filter(|&q| blk(q)).collect();
    let ei = rel_l2(&sel.iter().map(|&q| got_i[q]).collect::<Vec<_>>(), &sel.iter().map(|&q| want_i[q]).collect::<Vec<_>>());
    eprintln!("pruned n {n:?} × {nch}, block {keep:?}: forward {ef:.2e}, inverse (block) {ei:.2e}; full plan forward {efull:.2e}");
    assert!(ef < 5e-6 && ei < 5e-6 && efull < 5e-6, "forward {ef:.2e}, inverse {ei:.2e}, full {efull:.2e}");
}

/// The 1 mm full-torso fine grid itself (800 × 480 × 480, σ = 1.25 of 640 × 384 × 384), 2 channels, pruned to the
/// image block: a point source per channel must give the plane wave e^{−2πi k·r/n} (checked on sampled entries).
#[test]
#[ignore = "3 GB of GPU memory: run with --ignored"]
fn torso_1mm_grid_point_source() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let (n, img, nch) = ([480usize, 480, 800], [384usize, 384, 640], 2usize);
    let keep = [0, 1, 2].map(|a| (img[a] - img[a] / 2, img[a] / 2));
    let pr = PrunedChanFft3::<f32>::new(&dev, n, nch, keep, 1.0).expect("pruned 1 mm torso plan");
    let len = pr.len();
    let mut buf = st.alloc_zeros::<C32>(len).unwrap();
    // sources at r_c (inside the block: index i < keep.0 or ≥ n − keep.1), amplitude 1 and 2
    let src = [[3usize, 470, 700], [190, 5, 1]];
    let idx = |r: [usize; 3], c: usize| ((r[0] * n[1] + r[1]) * n[2] + r[2]) * nch + c;
    for (c, r) in src.iter().enumerate() {
        let mut v = st.alloc_zeros::<C32>(1).unwrap();
        st.memcpy_htod(&[C32::new((c + 1) as f32, 0.0)], &mut v).unwrap();
        let q = idx(*r, c);
        st.memcpy_dtod(&v, &mut buf.slice_mut(q..q + 1)).unwrap();
    }
    pr.forward(&st, &mut buf).unwrap();
    let mut rng = Rng::new(3);
    let mut worst = 0.0f64;
    for _ in 0..64 {
        let k = [0, 1, 2].map(|a| (rng.uniform() * n[a] as f64) as usize % n[a]);
        for (c, r) in src.iter().enumerate() {
            let q = idx(k, c);
            let got = st.clone_dtoh(&buf.slice(q..q + 1)).unwrap()[0];
            let ph = -2.0 * std::f64::consts::PI * (0..3).map(|a| ((k[a] * r[a]) % n[a]) as f64 / n[a] as f64).sum::<f64>();
            let want = C64::new(ph.cos(), ph.sin()) * (c + 1) as f64;
            worst = worst.max((C64::new(got.re as f64, got.im as f64) - want).norm() / (c + 1) as f64);
        }
    }
    eprintln!("1 mm torso grid {n:?} × {nch}: point-source max abs err {worst:.2e}");
    assert!(worst < 1e-3, "point source spectrum err {worst:.2e}");
}
