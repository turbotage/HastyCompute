//! CPU codec: bit layout, round-trip bounds per width, edge cases.
use hasty_bfp::{Bfp, BfpBuf, ENAN, VALS};

pub fn xorshift(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// values with a per-block scale spread over 2^-40 .. 2^40 and a within-block spread of 2^-12 .. 1
fn mixed(n: usize, seed: u64) -> Vec<f32> {
    let mut r = xorshift(seed);
    let mut out = Vec::with_capacity(n);
    let mut scale = 1.0;
    for i in 0..n {
        if i % VALS == 0 {
            scale = (2.0f64).powf(80.0 * r() - 40.0);
        }
        let m = (2.0f64).powf(-12.0 * r());
        out.push(((r() - 0.5) * 2.0 * m * scale) as f32);
    }
    out
}

fn check_bound(fmt: Bfp, x: &[f32]) {
    let buf = BfpBuf::encode(fmt, x);
    let y = buf.decode();
    for (k, blk) in x.chunks(VALS).enumerate() {
        let es = buf.ex[k];
        assert_ne!(es, ENAN);
        let bound = fmt.error_bound(es);
        let maxabs = blk.iter().fold(0f64, |m, v| m.max(v.abs() as f64));
        // no value of the block reaches 2^e, and the exponent is the smallest admissible one (or the lower limit)
        let e = Bfp::exponent(es);
        assert!(maxabs < (2.0f64).powi(e), "block {k}: max {maxabs} >= 2^{e}");
        if e > fmt.bits() as i32 - 127 && maxabs > 0.0 {
            assert!(maxabs * (2.0f64).powi(fmt.bits() as i32 - e) >= (2.0f64).powi(fmt.bits() as i32 - 1) - 0.5, "block {k}: exponent {e} not minimal for max {maxabs}");
        }
        for (i, &v) in blk.iter().enumerate() {
            let err = (v as f64 - y[k * VALS + i] as f64).abs();
            assert!(err <= bound, "bits {} block {k} value {i}: |{v} - {}| = {err} > {bound}", fmt.bits(), y[k * VALS + i]);
        }
    }
}

#[test]
fn bit_layout_matches_spec() {
    // bits 12: value i at bits [12 i, 12 i + 12) of the block stream
    let fmt = Bfp::new(12);
    let lim = fmt.lim() as f32; // 2047
    // block max 2047 / 2048 · 2^e with e = 0: max < 1 and round(max · 2^11) = 2047 ≤ lim
    let mut x = [0f32; 32];
    x[0] = lim / 2048.0; // q = 2047 = 0x7ff
    x[1] = -1.0 / 2048.0; // q = -1 = 0xfff
    x[2] = 5.0 / 2048.0; // q = 5
    x[31] = -lim / 2048.0; // q = -2047 = 0x801
    let mut m = vec![0u32; 12];
    let es = fmt.encode_block(&x, &mut m);
    assert_eq!(Bfp::exponent(es), 0);
    // word 0 = q0 | q1 << 12 | (q2 & 0xff) << 24
    assert_eq!(m[0], 0x7ff | (0xfff << 12) | (5 << 24));
    assert_eq!(m[1] & 0xf, 0); // upper 4 bits of q2 = 0
    // value 31 at bits 372 .. 383: word 11 bits 20 .. 31
    assert_eq!(m[11] >> 20, 0x801);
    for i in 0..32 {
        assert_eq!(fmt.dequant(fmt.mantissa(&m, i), es), x[i]);
    }
}

#[test]
fn round_trip_bound_every_width() {
    for b in 8..=24 {
        let fmt = Bfp::new(b);
        check_bound(fmt, &mixed(32 * 4000 + 17, b as u64));
        // a block is exactly `bits` words
        assert_eq!(fmt.mant_words(32), b as usize);
        assert_eq!(fmt.run_values() * b as usize % 32, 0);
    }
}

#[test]
fn zeros_and_tails() {
    for b in [8, 13, 16, 24] {
        let fmt = Bfp::new(b);
        let x = vec![0f32; 70];
        let buf = BfpBuf::encode(fmt, &x);
        assert!(buf.mant.iter().all(|&w| w == 0));
        assert!(buf.decode().iter().all(|&v| v == 0.0 && v.is_sign_positive()));
        // tail: 70 reals = 2 full blocks + 6 values, zero padded
        let y: Vec<f32> = (0..70).map(|i| i as f32 - 35.0).collect();
        let buf = BfpBuf::encode(fmt, &y);
        assert_eq!(buf.mant.len(), 3 * b as usize);
        check_bound(fmt, &y);
        // -0.0 decodes as +0.0
        assert_eq!(fmt.round_trip(&[-0.0])[0].to_bits(), 0);
    }
}

#[test]
fn subnormals() {
    for b in [8, 16, 24] {
        let fmt = Bfp::new(b);
        let tiny = f32::from_bits(1); // 1.4e-45
        let x: Vec<f32> = (0..32).map(|i| tiny * (i as f32) * if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        check_bound(fmt, &x);
        let buf = BfpBuf::encode(fmt, &x);
        assert_eq!(Bfp::exponent(buf.ex[0]), b as i32 - 127, "lower exponent limit");
        // a normal block containing subnormals
        let mut y = vec![f32::MIN_POSITIVE * 3.0; 32];
        y[5] = tiny * 77.0;
        check_bound(fmt, &y);
        // a block at the lower exponent limit with values just above 2^-126
        let z: Vec<f32> = (0..32).map(|i| f32::MIN_POSITIVE * (1.0 + i as f32 / 64.0)).collect();
        check_bound(fmt, &z);
    }
}

#[test]
fn all_equal_blocks() {
    for b in 8..=24 {
        let fmt = Bfp::new(b);
        for v in [1.0f32, -3.0, 0.1, 1e-30, 7.77e20, -f32::MIN_POSITIVE] {
            let x = vec![v; 32];
            let y = fmt.round_trip(&x);
            assert!(y.iter().all(|&w| w == y[0]));
            check_bound(fmt, &x);
        }
        // powers of two are exact
        let y = fmt.round_trip(&[0.5f32; 32]);
        assert!(y.iter().all(|&w| w == 0.5));
    }
}

#[test]
fn rounding_bump_and_ties() {
    for b in 8..=24 {
        let fmt = Bfp::new(b);
        // max just below 2^e: rounds up to 2^(b-1) at exponent e, so the encoder takes e + 1
        let below = 1.0f32 - (2.0f32).powi(-(b as i32) - 1);
        let x = vec![below; 32];
        let buf = BfpBuf::encode(fmt, &x);
        assert_eq!(Bfp::exponent(buf.ex[0]), 1, "bits {b}: bump to e = 1");
        check_bound(fmt, &x);
        // ties to even: half-step values at e = 1 (step 2^(2-b))
        let step = (2.0f32).powi(2 - b as i32);
        let mut t = vec![1.0f32; 32];
        t[1] = 2.5 * step; // q 2.5 -> 2
        t[2] = 3.5 * step; // q 3.5 -> 4
        t[3] = -2.5 * step; // -> -2
        let y = fmt.round_trip(&t);
        assert_eq!(y[1], 2.0 * step);
        assert_eq!(y[2], 4.0 * step);
        assert_eq!(y[3], -2.0 * step);
    }
}

#[test]
fn max_exponent_blocks_stay_finite() {
    for b in 8..=24 {
        let fmt = Bfp::new(b);
        let x: Vec<f32> = (0..32).map(|i| if i % 3 == 0 { f32::MAX } else { -f32::MAX / (1 + i) as f32 }).collect();
        let buf = BfpBuf::encode(fmt, &x);
        assert_eq!(Bfp::exponent(buf.ex[0]), 128);
        let y = buf.decode();
        let step = fmt.step(buf.ex[0]) as f64;
        for (v, w) in x.iter().zip(&y) {
            assert!(w.is_finite(), "bits {b}: {v} decoded as {w}");
            assert!((*v as f64 - *w as f64).abs() <= step, "bits {b}: error above one step at the saturation limit");
        }
        // 2^127 is the largest power of two: exact, half-step bound holds
        check_bound(fmt, &vec![(2.0f32).powi(127); 32]);
    }
}

#[test]
fn non_finite_blocks_decode_as_nan() {
    for b in [8, 20, 24] {
        let fmt = Bfp::new(b);
        let mut x: Vec<f32> = (0..96).map(|i| i as f32).collect();
        x[40] = f32::INFINITY;
        x[70] = f32::NAN;
        x[95] = f32::NEG_INFINITY;
        let buf = BfpBuf::encode(fmt, &x);
        assert_ne!(buf.ex[0], ENAN);
        assert_eq!(buf.ex[1], ENAN);
        assert_eq!(buf.ex[2], ENAN);
        let y = buf.decode();
        assert!(y[..32].iter().zip(&x).all(|(a, b)| (a - b).abs() <= 1.0));
        assert!(y[32..].iter().all(|v| v.is_nan()));
        // NaN must not be dropped by the block max (fmaxf would)
        let mut z = vec![1.0f32; 32];
        z[0] = f32::NAN;
        assert_eq!(BfpBuf::encode(fmt, &z).ex[0], ENAN);
    }
}

#[test]
fn complex_round_trip() {
    use hasty_cuda::C32;
    let fmt = Bfp::new(20);
    let x: Vec<C32> = (0..1000).map(|i| C32::new((i as f32 * 0.37).sin(), (i as f32 * 0.11).cos() * 1e-3)).collect();
    let buf = BfpBuf::encode_c(fmt, &x);
    assert_eq!(buf.bytes(), 63 * 20 * 4 + 64); // 2000 reals = 62.5 -> 63 blocks; ex padded to 64
    let y = buf.decode_c();
    for (a, b) in x.iter().zip(&y) {
        assert!((a.re - b.re).abs() <= 2.0f32.powi(-19) && (a.im - b.im).abs() <= 2.0f32.powi(-19));
    }
}
