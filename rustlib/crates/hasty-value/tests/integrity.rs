use hasty_value::{decode, encode_lossy, encode_to_vec, Codec, Decoder, Integrity, LazyFile, Loss, Options, Quant, Tensor, Value};

fn sample() -> Value {
    let x: Vec<f32> = (0..20000).map(|i| ((i as f32) * 0.013).sin() * 100.0).collect();
    Value::Dict(vec![("x".into(), Value::Tensor(Tensor::from_slice(&[200, 100], &x).unwrap())), ("n".into(), Value::Int(3)), ("s".into(), "hi".into())])
}

fn opts(i: Integrity) -> Options {
    Options { block_size: 4096, integrity: i, ..Options::default() }
}

const ALGOS: [Integrity; 2] = [Integrity::Xxh3, Integrity::Blake3];

#[test]
fn off_is_the_old_format_and_on_adds_only_the_digest() {
    let v = sample();
    let off = encode_to_vec(&v, &opts(Integrity::None)).unwrap();
    assert_eq!(off[5], 0);
    for a in ALGOS {
        let on = encode_to_vec(&v, &opts(a)).unwrap();
        assert_eq!(on.len(), off.len() + a.digest_len());
        assert_eq!(&on[6..off.len()], &off[6..], "body is identical");
        assert_eq!(decode(&on).unwrap(), v);
    }
}

#[test]
fn streaming_any_chunking_and_completion_only_after_digest() {
    let v = sample();
    for a in ALGOS {
        let bytes = encode_to_vec(&v, &opts(a)).unwrap();
        for step in [1usize, 7, 1000] {
            let mut d = Decoder::new();
            let mut i = 0;
            while i < bytes.len() {
                let e = (i + step).min(bytes.len());
                d.push(&bytes[i..e]).unwrap();
                i = e;
                assert_eq!(d.is_done(), i == bytes.len(), "{a:?} step {step} at {i}");
            }
            assert_eq!(d.finish().unwrap(), v);
        }
    }
}

#[test]
fn any_corruption_truncation_or_extension_is_detected() {
    let v = sample();
    for a in ALGOS {
        // stored blocks, no per-block crc: only the stream digest can notice
        let o = Options { codec: Codec::None, ..opts(a) };
        let bytes = encode_to_vec(&v, &o).unwrap();
        let mut s = 99u64;
        for _ in 0..200 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pos = 6 + (s >> 33) as usize % (bytes.len() - 6); // flags byte 5 is checked as format
            let mut b = bytes.clone();
            b[pos] ^= 1 << ((s >> 20) % 8);
            // a flipped header-structure byte may also fail as a format error; a decoded value must never equal the original silently
            match decode(&b) {
                Err(_) => {}
                Ok(x) => panic!("{a:?}: corruption at {pos} undetected ({})", x == v),
            }
        }
        for cut in [1, a.digest_len() / 2, a.digest_len(), a.digest_len() + 5] {
            assert!(decode(&bytes[..bytes.len() - cut]).is_err(), "{a:?} cut {cut}");
        }
        let mut ext = bytes.clone();
        ext.push(0);
        assert!(decode(&ext).is_err());
    }
}

#[test]
fn lazy_file_verify() {
    let v = sample();
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    for a in [Integrity::None, Integrity::Xxh3, Integrity::Blake3] {
        let bytes = encode_to_vec(&v, &opts(a)).unwrap();
        let p = dir.join(format!("integrity_{a:?}.hv"));
        std::fs::write(&p, &bytes).unwrap();
        let f = LazyFile::open(&p).unwrap();
        assert_eq!(f.integrity(), a);
        assert_eq!(f.verify().unwrap(), a != Integrity::None);
        if a != Integrity::None {
            let mut b = bytes.clone();
            let m = b.len() / 2;
            b[m] ^= 0x10;
            std::fs::write(&p, &b).unwrap();
            assert!(LazyFile::open(&p).unwrap().verify().is_err());
        }
    }
}

#[test]
fn works_with_lossy_chunked_tensors() {
    let v = sample();
    let plan = |_: &str, _: &Tensor| Some(vec![50, 100]);
    let loss = |_: &str, _: &Tensor| Some(Quant::new(Loss::Abs(0.1)));
    let mut bytes = Vec::new();
    encode_lossy(&v, &opts(Integrity::Blake3), Some(&plan), &loss, &mut |c| {
        bytes.extend_from_slice(&c);
        Ok(())
    })
    .unwrap();
    assert!(decode(&bytes).is_ok());
    let m = bytes.len() / 2;
    bytes[m] ^= 1;
    assert!(decode(&bytes).is_err());
}
