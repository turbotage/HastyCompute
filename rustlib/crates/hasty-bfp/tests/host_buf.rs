//! CPU: the host container (`BfpBuf`) — parallel encode / decode equal to the serial loops bit for bit, chunked
//! views equal to whole-buffer coding, zeros, tails, the NaN policy inside views, and the k-space step rule.
//!   cargo test --release -p hasty-bfp --test host_buf
use hasty_bfp::{Bfp, BfpBuf, CHUNK_BLOCKS, ENAN, VALS};
use hasty_cuda::C32;

fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// values with a per-block scale over 2^-30 .. 2^30 and a within-block spread of 2^-10 .. 1, a few exact zeros
fn mixed(n: usize, seed: u64) -> Vec<f32> {
    let mut r = xorshift(seed);
    let mut scale = 1.0;
    (0..n)
        .map(|i| {
            if i % VALS == 0 {
                scale = (2.0f64).powf(60.0 * r() - 30.0);
            }
            if r() < 0.01 {
                return 0.0;
            }
            ((r() - 0.5) * 2.0 * (2.0f64).powf(-10.0 * r()) * scale) as f32
        })
        .collect()
}

fn same_bits(a: &BfpBuf, b: &BfpBuf) -> bool {
    a.len == b.len && a.mant == b.mant && a.ex == b.ex
}

/// sizes around the parallel chunk boundary (CHUNK_BLOCKS blocks) and the block boundary, with tails
fn sizes() -> Vec<usize> {
    let c = CHUNK_BLOCKS * VALS;
    vec![0, 1, 31, 32, 33, 1000, c - 1, c, c + 1, c + 17, 3 * c + 5, 5 * c + 32 * 7 + 3]
}

#[test]
fn parallel_equals_serial_bit_for_bit() {
    for bits in [8, 12, 16, 20, 22, 24] {
        let fmt = Bfp::new(bits);
        for (j, &n) in sizes().iter().enumerate() {
            let x = mixed(n, 11 + j as u64);
            let p = BfpBuf::encode(fmt, &x);
            let s = BfpBuf::encode_serial(fmt, &x);
            assert!(same_bits(&p, &s), "bits {bits} n {n}: parallel encode differs from serial");
            let mut dp = vec![0f32; n];
            let mut ds = vec![0f32; n];
            p.decode_into(&mut dp);
            fmt.decode_serial(&s.mant, &s.ex, &mut ds);
            assert!(dp.iter().zip(&ds).all(|(a, b)| a.to_bits() == b.to_bits()), "bits {bits} n {n}: parallel decode differs");
            assert_eq!(p.bytes(), BfpBuf::bytes_for(fmt, n));
        }
    }
}

#[test]
fn chunked_views_equal_whole_buffer() {
    for bits in [10, 16, 21, 24] {
        let fmt = Bfp::new(bits);
        for (j, &n) in sizes().iter().enumerate() {
            let x = mixed(n, 101 + j as u64);
            let whole = BfpBuf::encode(fmt, &x);
            let full = whole.decode();
            for chunk in [32usize, 96, 4096, CHUNK_BLOCKS * VALS, 3 * CHUNK_BLOCKS * VALS + 64] {
                // encode through write views (each view codes its own range) = whole-buffer encode
                let mut b = BfpBuf::zeros(fmt, n);
                use rayon::prelude::*;
                b.par_views_mut(chunk).for_each(|mut v| {
                    let r0 = v.r0;
                    let l = v.len;
                    v.encode(&x[r0..r0 + l]);
                });
                assert!(same_bits(&b, &whole), "bits {bits} n {n} chunk {chunk}: view encode differs");
                // decode through read views = whole decode
                let mut out = vec![f32::NAN; n];
                let parts: Vec<(usize, Vec<f32>)> = whole
                    .par_views(chunk)
                    .map(|v| {
                        let mut o = vec![0f32; v.len];
                        v.decode(&mut o);
                        (v.r0, o)
                    })
                    .collect();
                for (r0, o) in parts {
                    out[r0..r0 + o.len()].copy_from_slice(&o);
                }
                assert!(out.iter().zip(&full).all(|(a, b)| a.to_bits() == b.to_bits()), "bits {bits} n {n} chunk {chunk}: view decode differs");
            }
            // single views and random access agree with the full decode
            if n > 0 {
                let r0 = (n / 3) / VALS * VALS;
                let v = whole.view(r0, n - r0);
                let mut o = vec![0f32; n - r0];
                v.decode(&mut o);
                assert!(o.iter().zip(&full[r0..]).all(|(a, b)| a.to_bits() == b.to_bits()));
                for i in [0, (n - r0) / 2, n - r0 - 1] {
                    assert_eq!(v.get(i).to_bits(), full[r0 + i].to_bits());
                }
            }
        }
    }
}

#[test]
fn zeros_tails_and_complex() {
    for bits in [8, 16, 24] {
        let fmt = Bfp::new(bits);
        for n in [0usize, 1, 2, 31, 33, 4095, 70000] {
            let z = BfpBuf::zeros(fmt, n);
            assert!(same_bits(&z, &BfpBuf::encode(fmt, &vec![0.0; n])), "zeros = encode(0), bits {bits} n {n}");
            assert!(z.decode().iter().all(|&v| v == 0.0));
        }
        // a tail write view: the last block is short and zero-padded, its neighbours untouched
        let n = 32 * 5 + 7;
        let x = mixed(n, 5);
        let mut b = BfpBuf::encode(fmt, &x);
        let y: Vec<f32> = x[128..].iter().map(|v| -2.0 * v).collect();
        b.view_mut(128, n - 128).encode(&y);
        let d = b.decode();
        let ref_head = BfpBuf::encode(fmt, &x[..128]).decode();
        assert!(d[..128].iter().zip(&ref_head).all(|(a, b)| a.to_bits() == b.to_bits()));
        let ref_tail = BfpBuf::encode(fmt, &y).decode();
        assert!(d[128..].iter().zip(&ref_tail).all(|(a, b)| a.to_bits() == b.to_bits()));
        // complex views
        let c: Vec<C32> = (0..1000).map(|i| C32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos())).collect();
        let cb = BfpBuf::encode_c(fmt, &c);
        let mut o = vec![C32::new(0.0, 0.0); 1000 - 16 * 10];
        cb.view(32 * 10, 2 * (1000 - 16 * 10)).decode_c(&mut o);
        let full = cb.decode_c();
        assert!(o.iter().zip(&full[160..]).all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()));
    }
}

#[test]
#[should_panic(expected = "ends inside a block")]
fn write_views_must_cover_whole_blocks() {
    let mut b = BfpBuf::zeros(Bfp::new(16), 100);
    let _ = b.view_mut(0, 40);
}

#[test]
fn nan_policy_inside_views() {
    let fmt = Bfp::new(16);
    let mut x = mixed(32 * 8, 9);
    x[32 * 3 + 5] = f32::NAN;
    x[32 * 6] = f32::INFINITY;
    let b = BfpBuf::encode(fmt, &x);
    assert_eq!(b.ex[3], ENAN);
    assert_eq!(b.ex[6], ENAN);
    let mut o = vec![0f32; 32 * 4];
    b.view(32 * 2, 32 * 4).decode(&mut o);
    // block 3 (view block 1) and block 6 are all-NaN, the others finite
    assert!(o[..32].iter().all(|v| v.is_finite()));
    assert!(o[32..64].iter().all(|v| v.is_nan()));
    assert!(o[64..].iter().all(|v| v.is_finite()));
    assert!(b.view(32 * 6, 32).get(31).is_nan());
}

#[test]
fn step_rule_for_kspace() {
    // b(peak, σ) = smallest width with step ≤ q σ; never coarser than the bound 4 · peak · 2^-b ≤ q σ predicts
    let mut r = xorshift(3);
    for _ in 0..2000 {
        let peak = (2.0f64).powf(40.0 * r() - 10.0) as f32;
        let sigma = peak as f64 / (2.0f64).powf(14.0 * r());
        let q = 0.25;
        let Some(b) = Bfp::bits_for_step(peak, q * sigma) else {
            assert!(peak as f64 / sigma > (2.0f64).powi(19), "peak/σ {} needs > 24 bits?", peak as f64 / sigma);
            continue;
        };
        let step = Bfp::new(b).block_step(peak);
        assert!(step <= q * sigma, "b {b}: step {step} > {}", q * sigma);
        if b > 8 {
            assert!(Bfp::new(b - 1).block_step(peak) > q * sigma, "b {b} not minimal");
        }
        let formula = ((peak as f64 / sigma).log2() + 4.0).ceil().clamp(8.0, 24.0) as u32;
        assert!(b <= formula, "rule {b} above the closed form {formula} (peak/σ {})", peak as f64 / sigma);
        // every block whose max is ≤ peak gets a step ≤ the peak block's
        let m = peak * r() as f32;
        assert!(Bfp::new(b).block_step(m) <= step);
    }
}
