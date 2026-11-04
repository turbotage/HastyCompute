use hasty_value::slice;
use hasty_value::{decode, encode, encode_to_vec, Codec, DType, Decoder, Options, Tensor, Value};
use num_complex::Complex32;

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

fn smooth_volume(nz: usize, ny: usize, nx: usize) -> Vec<Complex32> {
    let mut v = Vec::with_capacity(nz * ny * nx);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let r = ((x as f32 - 20.0).powi(2) + (y as f32 - 15.0).powi(2) + (z as f32 * 2.0 - 10.0).powi(2)).sqrt();
                v.push(Complex32::from_polar((-r / 12.0).exp(), 0.05 * x as f32 + 0.02 * z as f32));
            }
        }
    }
    v
}

fn sample() -> Value {
    let vol = Tensor::from_slice(&[11, 30, 41], &smooth_volume(11, 30, 41)).unwrap();
    let ints: Vec<u16> = (0..5000).map(|i| (1000.0 + 300.0 * ((i as f64) / 40.0).sin()) as u16).collect();
    Value::Dict(vec![
        ("vol".into(), Value::Tensor(vol)),
        ("ints".into(), Value::Tensor(Tensor::from_slice(&[50, 100], &ints).unwrap())),
        ("scalar".into(), Value::Tensor(Tensor::scalar(3.5f64))),
        ("empty".into(), Value::Tensor(Tensor::zeros(DType::F32, vec![0, 7]))),
        ("meta".into(), Value::Dict(vec![("name".into(), "test ✓".into()), ("n".into(), Value::Int(-7)), ("x".into(), Value::Float(1.25)), ("ok".into(), Value::Bool(true))])),
        ("list".into(), Value::List(vec![Value::None, Value::Tuple(vec![Value::Int(1), "a".into()]), Value::List(vec![]), Value::Dict(vec![])])),
    ])
}

#[test]
fn round_trip_all_codecs_and_filters() {
    let v = sample();
    for codec in [Codec::None, Codec::Lz4, Codec::Zstd(3)] {
        for (shuffle, delta, checksum) in [(false, false, false), (true, false, true), (true, true, false)] {
            for block_size in [64, 1000, 1 << 20] {
                let o = Options { codec, block_size, shuffle, delta, checksum, parallel: block_size < 1000 };
                let bytes = encode_to_vec(&v, &o).unwrap();
                assert_eq!(decode(&bytes).unwrap(), v, "{codec:?} shuffle {shuffle} delta {delta} block {block_size}");
            }
        }
    }
}

#[test]
fn incremental_decode_any_chunking() {
    let v = sample();
    let o = Options { block_size: 4096, ..Options::default() };
    let bytes = encode_to_vec(&v, &o).unwrap();
    let mut s = 17u64;
    for trial in 0..20 {
        let mut d = if trial % 2 == 0 { Decoder::new() } else { Decoder::new().sequential() };
        let mut i = 0;
        while i < bytes.len() {
            let n = if trial == 0 { 1 } else { 1 + (lcg(&mut s) as usize % 9000) };
            let e = (i + n).min(bytes.len());
            d.push(&bytes[i..e]).unwrap();
            i = e;
        }
        assert!(d.is_done());
        assert_eq!(d.finish().unwrap(), v);
    }
    // streaming encoder chunks decode chunk by chunk
    let mut d = Decoder::new();
    encode(&v, &Options { block_size: 4096, ..Options::default() }, &mut |c| d.push(&c)).unwrap();
    assert_eq!(d.finish().unwrap(), v);
}

#[test]
fn compression_pays_and_incompressible_is_stored() {
    let vol = Value::Tensor(Tensor::from_slice(&[22, 60, 82], &smooth_volume(22, 60, 82)).unwrap());
    let raw = 22 * 60 * 82 * 8;
    let plain = encode_to_vec(&vol, &Options { shuffle: false, ..Options::default() }).unwrap().len();
    let shuf = encode_to_vec(&vol, &Options::default()).unwrap().len();
    let zs = encode_to_vec(&vol, &Options::zstd(3)).unwrap().len();
    eprintln!("smooth complex64 volume {raw} B: lz4 {plain}, shuffle+lz4 {shuf}, shuffle+zstd {zs}");
    assert!(shuf < plain && zs < shuf);
    let mut s = 1u64;
    let noise: Vec<u8> = (0..300_000).map(|_| lcg(&mut s) as u8).collect();
    let nv = Value::Tensor(Tensor::from_slice(&[noise.len()], &noise).unwrap());
    let enc = encode_to_vec(&nv, &Options { block_size: 65536, ..Options::default() }).unwrap();
    assert!(enc.len() <= noise.len() + 8 + 16 + 5 * 12, "random bytes must be stored, got {}", enc.len());
    assert_eq!(decode(&enc).unwrap(), nv);
}

#[test]
fn detects_corruption_and_truncation() {
    let v = sample();
    let mut b = encode_to_vec(&v, &Options { checksum: true, block_size: 512, ..Options::default() }).unwrap();
    assert!(decode(&b[..b.len() - 3]).is_err(), "truncated stream must fail");
    let k = b.len() / 2;
    b[k] ^= 0x55;
    assert!(decode(&b).is_err() || decode(&b).unwrap() != v);
    assert!(decode(b"NOPE\x01\0\0\0\0").is_err());
}

#[test]
fn slicing_like_numpy() {
    let data: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32).collect();
    let t = Tensor::from_slice(&[2, 3, 4], &data).unwrap();
    let v = Value::Dict(vec![("a".into(), Value::List(vec![Value::Int(5), Value::Tensor(t.clone())]))]);
    let s = |p: &str| -> (Vec<usize>, Vec<f32>) {
        let r = slice::get(&v, p).unwrap();
        let t = r.as_tensor().unwrap().clone();
        (t.shape().to_vec(), t.to_vec::<f32>().unwrap())
    };
    assert_eq!(s(r#"["a"][1][1]"#), (vec![3, 4], (12..24).map(|i| i as f32).collect()));
    assert_eq!(s(r#"["a"][-1][:, 1, ::2]"#), (vec![2, 2], vec![4.0, 6.0, 16.0, 18.0]));
    assert_eq!(s(r#"["a"][1][..., -1]"#), (vec![2, 3], vec![3.0, 7.0, 11.0, 15.0, 19.0, 23.0]));
    assert_eq!(s(r#"["a"][1][0, ::-1, 3]"#), (vec![3], vec![11.0, 7.0, 3.0]));
    assert_eq!(s(r#"["a"][1][1:, 5:, :]"#).0, vec![1, 0, 4]);
    assert_eq!(slice::get(&v, r#"["a"][0]"#).unwrap(), Value::Int(5));
    assert!(slice::get(&v, r#"["b"]"#).is_err());
    assert!(slice::get(&v, r#"["a"][1][0, 0, 9]"#).is_err());
    // assignment into a selection, and broadcast of a scalar
    let mut w = v.clone();
    let patch = Tensor::from_slice(&[2, 2], &[-1.0f32, -2.0, -3.0, -4.0]).unwrap();
    slice::set(&mut w, r#"["a"][1][:, 2, 1:3]"#, Value::Tensor(patch)).unwrap();
    slice::set(&mut w, r#"["a"][1][0, 0, :]"#, Value::Tensor(Tensor::scalar(9.0f32))).unwrap();
    let got = slice::get(&w, r#"["a"][1]"#).unwrap().as_tensor().unwrap().to_vec::<f32>().unwrap();
    let mut want = data.clone();
    want[9] = -1.0;
    want[10] = -2.0;
    want[21] = -3.0;
    want[22] = -4.0;
    want[..4].fill(9.0);
    assert_eq!(got, want);
    slice::set(&mut w, r#"["new"]"#, "x".into()).unwrap();
    assert_eq!(slice::get(&w, r#"["new"]"#).unwrap(), Value::Str("x".into()));
}

#[test]
fn comprep_round_trip_and_compression() {
    use hasty_value::quant::{auto_config, decode_u16, encode_u16, Comprep};
    let t = Tensor::from_slice(&[22, 60, 82], &smooth_volume(22, 60, 82)).unwrap();
    let vals: Vec<f64> = t.as_slice::<Complex32>().unwrap().iter().map(|z| z.norm() as f64).collect();
    // C++ default: a, b = p1 / p99, clamp = [min, max], focus 1 (tails saturate)
    let c = auto_config(&t, &serde_json::json!({})).unwrap();
    assert!(c.clampa <= c.a && c.a < c.b && c.b <= c.clampb);
    let back = decode_u16(&encode_u16(&t, &c).unwrap(), &c).unwrap().to_vec::<f32>().unwrap();
    let step = (c.b - c.a) / 65535.0;
    for (x, y) in vals.iter().zip(&back) {
        let want = x.clamp(c.a, c.b);
        assert!((*y as f64 - want).abs() <= 0.51 * step + 1e-6 * want.abs(), "{x} -> {y}");
    }
    // focus 0.8 with log / gamma tails: window keeps 80 % of the codes, tails survive at reduced precision
    let c2 = auto_config(&t, &serde_json::json!({ "focus": 0.8, "left_mode": "gamma", "left_gamma": 2.0, "right_mode": "log", "right_logc": 10.0 })).unwrap();
    let back2 = decode_u16(&encode_u16(&t, &c2).unwrap(), &c2).unwrap().to_vec::<f32>().unwrap();
    let (mut win, mut tail) = (0f64, 0f64);
    for (x, y) in vals.iter().zip(&back2) {
        let e = (*y as f64 - x).abs();
        if *x >= c2.a && *x <= c2.b { win = win.max(e) } else { tail = tail.max(e) }
    }
    assert!(win <= 0.51 * (c2.b - c2.a) / (0.8 * 65535.0) + 1e-6, "window error {win}");
    assert!(tail < (c2.clampb - c2.clampa) * 1e-2, "tail error {tail}");
    // forward / inverse are inverses on every region and shape
    let c3 = Comprep::from_json(&serde_json::json!({"a": 1.0, "b": 2.0, "clampa": 0.0, "clampb": 5.0, "focus": 0.5, "left_mode": "log", "left_logc": 3.0, "right_mode": "gamma", "right_gamma": 0.5})).unwrap();
    for k in 0..=500 {
        let x = k as f64 / 100.0;
        assert!((c3.inverse(c3.forward(x)) - x).abs() < 1e-9, "x {x}");
    }
    // integers map directly (C++: u8 · 257)
    let u = Tensor::from_slice(&[4], &[0u8, 1, 128, 255]).unwrap();
    let cu = auto_config(&u, &serde_json::json!({})).unwrap();
    assert_eq!(encode_u16(&u, &cu).unwrap().to_vec::<u16>().unwrap(), vec![0, 257, 128 * 257, 65535]);
    // delta + shuffle + LZ4 on the display copy
    let q = encode_u16(&t, &c).unwrap();
    let n_d = encode_to_vec(&Value::Tensor(q.clone()), &Options { delta: true, ..Options::default() }).unwrap().len();
    let n_s = encode_to_vec(&Value::Tensor(q), &Options::default()).unwrap().len();
    eprintln!("u16 display volume {} B: shuffle+lz4 {n_s}, delta+shuffle+lz4 {n_d}", 22 * 60 * 82 * 2);
    assert!(n_d < n_s);
}

#[test]
fn npy_round_trip_and_fortran() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/hv-test");
    std::fs::create_dir_all(&dir).unwrap();
    let t = Tensor::from_slice(&[3, 4, 5], &(0..60).map(|i| i as f32).collect::<Vec<_>>()).unwrap();
    let p = dir.join("t.npy");
    hasty_value::npy::write(&p, &t).unwrap();
    let (r, f) = hasty_value::npy::read(&p).unwrap();
    assert_eq!((r, f), (t.clone(), false));
    // a Fortran-ordered (X=5, Y=4, Z=3) file holds the same bytes as C-order (3, 4, 5)
    let mut b = std::fs::read(&p).unwrap();
    let h = String::from_utf8_lossy(&b[10..128]).replace("'fortran_order': False, 'shape': (3, 4, 5)", "'fortran_order': True, 'shape': (5, 4, 3) ");
    b[10..128].copy_from_slice(h.as_bytes());
    let (r, f) = hasty_value::npy::parse(&b).unwrap();
    assert!(f);
    assert_eq!(r, t);
}

#[test]
fn chunked_tensors_stream_and_random_access() {
    use hasty_value::{encode_with, LazyFile};
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/hv-test");
    std::fs::create_dir_all(&dir).unwrap();
    // [X, Y, Z, F, E, 2] like a RACLETTE case, sizes not multiples of the chunk
    let (x, y, z, f, e) = (37, 29, 21, 5, 7);
    let n = x * y * z * f * e * 2;
    let data: Vec<u16> = (0..n).map(|i| ((i * 2654435761usize) >> 7) as u16 ^ (i as u16 / 3)).collect();
    let big = Tensor::from_slice(&[x, y, z, f, e, 2], &data).unwrap();
    let mask = Tensor::from_slice(&[x, y, z, f], &(0..x * y * z * f).map(|i| (i % 7 == 0) as u8).collect::<Vec<_>>()).unwrap();
    let v = Value::Dict(vec![("x".into(), Value::Tensor(big.clone())), ("mask".into(), Value::Tensor(mask.clone())), ("meta".into(), "case".into()), ("plain".into(), Value::Tensor(Tensor::from_slice(&[3, 4], &[1.5f32; 12]).unwrap()))]);
    let plan = |path: &str, t: &Tensor| -> Option<Vec<usize>> {
        match path {
            r#"["x"]"# => Some(vec![16, 16, 16, 1, t.shape()[4], 2]),
            r#"["mask"]"# => Some(vec![16, 16, 16, 1]),
            _ => None,
        }
    };
    for (o, checksum) in [(Options::zstd(9), false), (Options { block_size: 4096, checksum: true, ..Options::default() }, true)] {
        let mut bytes = Vec::new();
        encode_with(&v, &o, &plan, &mut |c| {
            bytes.extend_from_slice(&c);
            Ok(())
        })
        .unwrap();
        // stream decoding assembles the chunks (any chunking of the input)
        assert_eq!(decode(&bytes).unwrap(), v, "checksum {checksum}");
        let mut d = Decoder::new();
        for c in bytes.chunks(777) {
            d.push(c).unwrap();
        }
        assert_eq!(d.finish().unwrap(), v);
        // random access: boxes straddling chunk borders, single frames, the whole tensor
        let p = dir.join(format!("chunked_{checksum}.hv"));
        std::fs::write(&p, &bytes).unwrap();
        let lf = LazyFile::open(&p).unwrap();
        let tx = lf.root.get("x").unwrap().tensor().unwrap();
        assert_eq!(tx.chunk_shape(), Some(&[16, 16, 16, 1, 7, 2][..]));
        for (lo, hi) in [([0, 0, 0, 0, 0, 0], [x, y, z, f, e, 2]), ([5, 15, 3, 2, 0, 0], [33, 18, 21, 4, 7, 2]), ([36, 28, 20, 4, 6, 1], [37, 29, 21, 5, 7, 2]), ([10, 0, 0, 1, 2, 0], [11, 29, 21, 2, 3, 2])] {
            let got = lf.read_box(tx, &lo, &hi).unwrap();
            let path = format!("[{}]", (0..6).map(|d| format!("{}:{}", lo[d], hi[d])).collect::<Vec<_>>().join(", "));
            assert_eq!(got, slice::tensor_slice(&big, &slice::parse(&path).unwrap().into_iter().next().map(|s| match s { slice::Segment::Items(i) => i, _ => unreachable!() }).unwrap()).unwrap(), "box {lo:?}..{hi:?}");
        }
        assert_eq!(lf.read(lf.root.get("mask").unwrap().tensor().unwrap()).unwrap(), mask);
        assert_eq!(lf.value_of(&lf.root).unwrap(), v);
    }
}
