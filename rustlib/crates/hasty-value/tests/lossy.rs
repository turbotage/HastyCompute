use hasty_value::{decode, encode_lossy, Codec, DType, Decoder, LazyFile, Loss, Options, Quant, Scope, Tensor, Transfer, Value};
use num_complex::Complex32;

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
}

/// smooth complex field with a large dynamic range plus noise
fn field(n: usize, noise: f64) -> Vec<Complex32> {
    let mut s = 7u64;
    (0..n * n * n)
        .map(|i| {
            let (x, y, z) = ((i % n) as f64, ((i / n) % n) as f64, (i / (n * n)) as f64);
            let r = ((x - 9.0).powi(2) + (y - 7.0).powi(2) + (z - 5.0).powi(2)).sqrt();
            let a = 1000.0 * (-r / 6.0).exp();
            let ph = 0.1 * x + 0.05 * y;
            Complex32::new((a * ph.cos() + noise * lcg(&mut s)) as f32, (a * ph.sin() + noise * lcg(&mut s)) as f32)
        })
        .collect()
}

fn roundtrip(v: &Value, o: &Options, chunk: Option<Vec<usize>>, q: Quant) -> (Vec<u8>, Value) {
    let plan = |_: &str, _: &Tensor| chunk.clone();
    let loss = |_: &str, _: &Tensor| Some(q);
    let mut bytes = Vec::new();
    encode_lossy(v, o, Some(&plan), &loss, &mut |c| {
        bytes.extend_from_slice(&c);
        Ok(())
    })
    .unwrap();
    let d = decode(&bytes).unwrap();
    (bytes, d)
}

fn scalars(t: &Value) -> Vec<f64> {
    let Value::Tensor(t) = t else { panic!() };
    t.as_slice::<Complex32>().unwrap().iter().flat_map(|c| [c.re as f64, c.im as f64]).collect()
}

fn lossless_len(v: &Value, o: &Options) -> usize {
    hasty_value::encode_to_vec(v, o).unwrap().len()
}

#[test]
fn abs_error_bound_and_size() {
    let n = 24;
    let t = Tensor::from_slice(&[n, n, n], &field(n, 0.5)).unwrap();
    let v = Value::Tensor(t);
    let o = Options { codec: Codec::Zstd(3), ..Options::default() };
    let d = 0.05;
    let (bytes, out) = roundtrip(&v, &o, Some(vec![8, 8, 8]), Quant::new(Loss::Abs(d)));
    let (a, b) = (scalars(&v), scalars(&out));
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    assert!(worst <= d / 2.0 + 1e-4, "worst {worst}");
    assert!(bytes.len() * 10 < lossless_len(&v, &o) * 6, "{} vs {}", bytes.len(), lossless_len(&v, &o));
}

#[test]
fn relmax_bits_and_scope() {
    let n = 24;
    let v = Value::Tensor(Tensor::from_slice(&[n, n, n], &field(n, 0.0)).unwrap());
    let o = Options::default();
    let umax = scalars(&v).iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for scope in [Scope::Chunk, Scope::Tensor] {
        for bits in [8u32, 12, 16, 20] {
            let (_, out) = roundtrip(&v, &o, Some(vec![12, 12, 12]), Quant::new(Loss::RelMax(bits)).scope(scope));
            let worst = scalars(&v).iter().zip(scalars(&out)).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            let bound = umax / (((1u64 << (bits - 1)) - 1) as f64) / 2.0;
            // chunk scope has a tighter local scale, so the global bound holds for both; allow f32 rounding
            assert!(worst <= bound * 1.001 + 1e-3, "{scope:?} {bits} bits: {worst} > {bound}");
        }
    }
}

#[test]
fn relrms_rms_error() {
    let n = 24;
    let v = Value::Tensor(Tensor::from_slice(&[n, n, n], &field(n, 2.0)).unwrap());
    let eps = 0.01;
    let (_, out) = roundtrip(&v, &Options::default(), Some(vec![24, 24, 24]), Quant::new(Loss::RelRms(eps)));
    let (a, b) = (scalars(&v), scalars(&out));
    let rms = (a.iter().map(|x| x * x).sum::<f64>() / a.len() as f64).sqrt();
    let err = (a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
    assert!(err < 1.2 * eps * rms && err > 0.5 * eps * rms, "{err} vs {}", eps * rms);
}

#[test]
fn asinh_transfer_is_relative_above_knee() {
    let n = 24;
    let v = Value::Tensor(Tensor::from_slice(&[n, n, n], &field(n, 0.0)).unwrap());
    let (k, step) = (1.0, 0.01);
    let (_, out) = roundtrip(&v, &Options::default(), Some(vec![12, 12, 12]), Quant::new(Loss::Abs(step)).transfer(Transfer::Asinh(k)));
    for (x, y) in scalars(&v).iter().zip(scalars(&out)) {
        let bound = (1.0 + (x / k).powi(2)).sqrt() * step / 2.0;
        assert!((x - y).abs() <= bound * 1.001 + 1e-4 * x.abs().max(1.0), "{x} {y}");
    }
}

#[test]
fn mantissa_pointwise_and_plain_format() {
    let n = 16;
    let v = Value::Tensor(Tensor::from_slice(&[n, n, n], &field(n, 3.0)).unwrap());
    let o = Options { codec: Codec::Zstd(3), ..Options::default() };
    let m = 10;
    let plan = |_: &str, _: &Tensor| None;
    let loss = |_: &str, _: &Tensor| Some(Quant::new(Loss::Mantissa(m)));
    let mut bytes = Vec::new();
    encode_lossy(&v, &o, Some(&plan), &loss, &mut |c| {
        bytes.extend_from_slice(&c);
        Ok(())
    })
    .unwrap();
    assert_eq!(bytes[8], 8, "mantissa output is an ordinary tag-8 tensor");
    let out = decode(&bytes).unwrap();
    for (x, y) in scalars(&v).iter().zip(scalars(&out)) {
        assert!((x - y).abs() <= x.abs() * 2f64.powi(-(m as i32 + 1)) * 1.0001, "{x} {y}");
    }
    assert!(bytes.len() < lossless_len(&v, &o));
}

#[test]
fn nonfinite_chunk_falls_back_lossless_and_others_stay_lossy() {
    let mut x: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
    x[5] = f32::NAN; // chunk 0
    x[4000] = f32::INFINITY; // chunk 3 (chunk = 1024)
    let v = Value::Tensor(Tensor::from_slice(&[4096], &x).unwrap());
    let (_, out) = roundtrip(&v, &Options::default(), Some(vec![1024]), Quant::new(Loss::Abs(0.01)));
    let Value::Tensor(o) = &out else { panic!() };
    let y = o.as_slice::<f32>().unwrap();
    assert!(y[5].is_nan() && y[4000].is_infinite());
    assert_eq!(&x[6..1024], &y[6..1024]); // lossless chunk exact
    assert!((x[2000] - y[2000]).abs() <= 0.005 + 1e-6 && x[2000] != y[2000]);
}

#[test]
fn lazy_box_read_and_streaming_agree() {
    let n = 20;
    let v = Value::Tensor(Tensor::from_slice(&[n, n, n], &field(n, 0.3)).unwrap());
    let q = Quant::new(Loss::RelMax(14));
    let (bytes, full) = roundtrip(&v, &Options { block_size: 4096, ..Options::default() }, Some(vec![7, 7, 7]), q);
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("lossy_test.hv");
    std::fs::write(&path, &bytes).unwrap();
    let f = LazyFile::open(&path).unwrap();
    let t = f.root.tensor().unwrap();
    let b = f.read_box(t, &[3, 5, 2], &[15, 9, 20]).unwrap();
    let Value::Tensor(full) = full else { panic!() };
    let (fs, bs) = (full.as_slice::<Complex32>().unwrap(), b.as_slice::<Complex32>().unwrap());
    let mut k = 0;
    for z in 3..15 {
        for y in 5..9 {
            for x in 2..20 {
                assert_eq!(fs[(z * n + y) * n + x], bs[k]);
                k += 1;
            }
        }
    }
    let mut d = Decoder::new();
    for c in bytes.chunks(1013) {
        d.push(c).unwrap();
    }
    assert_eq!(d.finish().unwrap(), Value::Tensor(full));
}

#[test]
fn integer_dtype_rejected() {
    let v = Value::Tensor(Tensor::zeros(DType::I16, vec![4]));
    let loss = |_: &str, _: &Tensor| Some(Quant::new(Loss::Abs(1.0)));
    assert!(encode_lossy(&v, &Options::default(), None, &loss, &mut |_| Ok(())).is_err());
}
