//! store: policies' error bounds, streaming writer, row reads, .npy / .hv interchangeability.

use std::path::PathBuf;

use hasty_value::store::{self, Chunking, Order, Policy, RowFile, Save, TensorWriter};
use hasty_value::{npy, DType, LazyFile, Tensor, Value};
use num_complex::Complex32;

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("hasty_store_test_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn lcg(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / (1u64 << 53) as f64 - 0.5
}

/// image-like complex volume (x fastest): smooth blob + noise, shape (nx, ny, nz, ne)
fn volume(nx: usize, ny: usize, nz: usize, ne: usize) -> Vec<Complex32> {
    let mut s = 3u64;
    (0..nx * ny * nz * ne)
        .map(|i| {
            let (x, y, z) = ((i % nx) as f64, ((i / nx) % ny) as f64, ((i / (nx * ny)) % nz) as f64);
            let r2 = (x - nx as f64 / 2.0).powi(2) + (y - ny as f64 / 3.0).powi(2) + (z - nz as f64 / 2.0).powi(2);
            let a = 5.0 * (-r2 / 80.0).exp() + 0.02 * lcg(&mut s);
            let ph = 0.07 * x - 0.03 * z + (i / (nx * ny * nz)) as f64;
            Complex32::new((a * ph.cos()) as f32, (a * ph.sin()) as f32)
        })
        .collect()
}

fn bytes<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn sc(t: &Tensor) -> Vec<f64> {
    match t.dtype() {
        DType::C64 | DType::F32 => t.bytes().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        _ => t.bytes().chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
    }
}

fn err(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let rms = (a.iter().map(|x| x * x).sum::<f64>() / a.len() as f64).sqrt();
    let emax = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let erms = (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
    (rms, emax, erms)
}

#[test]
fn image_policy_bound_and_fortran_roundtrip() {
    let d = dir("image");
    let (nx, ny, nz, ne) = (37, 29, 23, 3);
    let v = volume(nx, ny, nz, ne);
    let p = store::save_slice(&d.join("img.npy"), &[nx, ny, nz, ne], Order::F, &v, &Save::image()).unwrap();
    assert_eq!(p.extension().unwrap(), "hv");
    let (t, o) = store::load_array(&d.join("img.npy")).unwrap(); // resolves the .hv sibling
    assert_eq!(o, Order::F);
    assert_eq!(t.shape(), &[ne, nz, ny, nx]);
    let a: Vec<f64> = v.iter().flat_map(|c| [c.re as f64, c.im as f64]).collect();
    // complex: pointwise relative ≤ IMAGE_REL (mantissa rounding, phase-safe)
    let b = sc(&t);
    let worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs() / x.abs().max(1e-30)).fold(0.0, f64::max);
    assert!(worst <= store::IMAGE_REL, "complex image: pointwise relative error {worst}");
    // real: uniform step, rms err = IMAGE_REL · rms
    let re: Vec<f32> = v.iter().map(|c| c.re).collect();
    store::save_slice(&d.join("re.npy"), &[nx, ny, nz, ne], Order::F, &re, &Save::image()).unwrap();
    let (tr, _) = store::load_array(&d.join("re.hv")).unwrap();
    let ar: Vec<f64> = re.iter().map(|&x| x as f64).collect();
    let (rms, emax, erms) = err(&ar, &tr.as_slice::<f32>().unwrap().iter().map(|&x| x as f64).collect::<Vec<_>>());
    let step = store::IMAGE_REL * 12f64.sqrt() * rms;
    assert!(emax <= step / 2.0 * (1.0 + 1e-6) + 1e-7, "max err {emax} > step/2 {}", step / 2.0);
    assert!((erms / rms - store::IMAGE_REL).abs() < 0.1 * store::IMAGE_REL, "rms err ratio {}", erms / rms);
    // same tensor as the npy reader gives for the Fortran npy
    npy::write_raw(&d.join("ref.npy"), DType::C64, &[nx, ny, nz, ne], true, bytes(&v)).unwrap();
    let (r, f) = npy::read(&d.join("ref.npy")).unwrap();
    assert!(f);
    assert_eq!(r.shape(), t.shape());
    // the file is an ordinary value stream too (stream decoder, lazy reader)
    let whole = hasty_value::read(&mut std::fs::File::open(&p).unwrap()).unwrap();
    assert_eq!(whole.get("order").and_then(|v| v.as_str()), Some("F"));
    assert!(whole.get("loss").and_then(|v| v.as_str()).unwrap().contains("Image"));
    assert_eq!(whole.get("array").unwrap().as_tensor().unwrap(), &t);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn lossless_ints_masks_nan() {
    let d = dir("lossless");
    // integer-valued float mask: stored losslessly whatever the policy
    let m: Vec<f32> = (0..50_000).map(|i| ((i / 97) % 2) as f32).collect();
    store::save_slice(&d.join("mask.npy"), &[50_000], Order::C, &m, &Save::image()).unwrap();
    let (t, _) = store::load_array(&d.join("mask.hv")).unwrap();
    assert_eq!(t.as_slice::<f32>().unwrap(), &m[..]);
    // integers
    let q: Vec<i32> = (0..10_000).map(|i| i * 7 - 3000).collect();
    store::save_slice(&d.join("ints.npy"), &[100, 100], Order::C, &q, &Save::image()).unwrap();
    assert_eq!(store::load_array(&d.join("ints.hv")).unwrap().0.as_slice::<i32>().unwrap(), &q[..]);
    // NaN background: finite values within the bound, NaN kept
    let mut s = 9u64;
    let x: Vec<f32> = (0..400_000).map(|i| if i < 200_000 { f32::NAN } else { (100.0 + lcg(&mut s)) as f32 }).collect();
    store::save_slice(&d.join("nan.npy"), &[400_000], Order::C, &x, &Save::image().chunk(Chunking::Bytes(1 << 16))).unwrap();
    let y = store::load_array(&d.join("nan.hv")).unwrap().0.to_vec::<f32>().unwrap();
    let fin: Vec<f64> = x.iter().filter(|v| v.is_finite()).map(|&v| v as f64).collect();
    let rms = (fin.iter().map(|v| v * v).sum::<f64>() / fin.len() as f64).sqrt();
    let step = store::IMAGE_REL * 12f64.sqrt() * rms;
    for (a, b) in x.iter().zip(&y) {
        if a.is_nan() {
            assert!(b.is_nan());
        } else {
            assert!(((a - b) as f64).abs() <= step / 2.0 + 1e-5, "{a} {b}");
        }
    }
    // lossless policy: bit-exact
    let v = volume(16, 16, 8, 1);
    store::save_slice(&d.join("ll.npy"), &[16, 16, 8], Order::F, &v, &Save::lossless()).unwrap();
    assert_eq!(store::load_array(&d.join("ll.hv")).unwrap().0.as_slice::<Complex32>().unwrap(), &v[..]);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn kspace_rows_stream_and_row_reads() {
    let d = dir("kspace");
    let (nro, na, nc) = (53, 301, 7);
    let mut s = 5u64;
    let k: Vec<Complex32> = (0..nro * na * nc)
        .map(|i| {
            let n = (i / nc) % na;
            let a = 2000.0 * (-(n as f64) / 40.0).exp();
            Complex32::new((a * lcg(&mut s) + 30.0 * lcg(&mut s)) as f32, (a * lcg(&mut s)) as f32)
        })
        .collect();
    let a: Vec<f64> = k.iter().flat_map(|c| [c.re as f64, c.im as f64]).collect();
    let rms_c = (a.iter().map(|x| x * x).sum::<f64>() / (a.len() / 2) as f64).sqrt();
    let sigma = 40.0;
    let step = store::kspace_step(sigma, rms_c);
    assert!(step <= sigma / 2f64.sqrt() / 8.0 + 1e-12);
    let save = Save::default().policy(Policy::Abs(step)).chunk(Chunking::Rows(1));
    // streamed in odd pieces
    let path = d.join("data_clean.hv");
    let mut w = TensorWriter::create(&path, DType::C64, &[nro, na, nc], Order::C, &save, None).unwrap();
    let b = bytes(&k);
    let mut o = 0;
    for piece in [1usize, 7, 4096, 100_003, 13] {
        let e = (o + piece).min(b.len());
        w.push(&b[o..e]).unwrap();
        o = e;
    }
    w.push(&b[o..]).unwrap();
    w.finish().unwrap();
    let (t, _) = store::load_array(&path).unwrap();
    let (_, emax, _) = err(&a, &sc(&t));
    assert!(emax <= step / 2.0 * (1.0 + 1e-6) + 1e-4, "{emax} vs {}", step / 2.0);
    // one readout per chunk
    let lf = LazyFile::open(&path).unwrap();
    assert_eq!(lf.root.get("array").unwrap().tensor().unwrap().chunk_shape(), Some(&[1, na, nc][..]));
    // row reads equal the whole-array decode, for aligned and unaligned ranges
    let rf = RowFile::open(&d.join("data_clean.npy")).unwrap();
    assert!(rf.is_hv());
    let tb = t.bytes();
    for (st, n) in [(0usize, na * nc), (5 * na * nc, na * nc), (3 * na * nc + 17, 2 * na * nc + 5), (nro * na * nc - 3, 3), (na * nc - 1, 2)] {
        let mut out = vec![0u8; n * 8];
        rf.read_elems(st, &mut out).unwrap();
        assert_eq!(&out[..], &tb[st * 8..(st + n) * 8], "range {st}+{n}");
    }
    let mut past = vec![0u8; 16];
    assert!(rf.read_elems(nro * na * nc - 1, &mut past).is_err());
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn per_field_bits_slab_chunks() {
    let d = dir("fields");
    let (nro, na, nf) = (19, 257, 5);
    // fields of very different scales (k in rad/m, tau in s, …), smooth along the ADC
    let scale = [3000.0, 3000.0, 11.0, 0.01, 0.0];
    let r: Vec<f32> = (0..nro * na * nf).map(|i| (scale[i % nf] * (((i / nf) % na) as f64 / 37.0 + (i / (na * nf)) as f64).sin()) as f32).collect();
    let save = Save::default().policy(Policy::Bits(24)).chunk(Chunking::Shape(vec![1, na, 1])).delta(true);
    let mut w = TensorWriter::create(&d.join("sched.hv"), DType::F32, &[nro, na, nf], Order::C, &save, None).unwrap();
    for c in bytes(&r).chunks(1000) {
        w.push(c).unwrap();
    }
    w.finish().unwrap();
    let back = store::load_array(&d.join("sched.npy")).unwrap().0.to_vec::<f32>().unwrap();
    for f in 0..nf {
        let mx = (f..r.len()).step_by(nf).map(|i| r[i].abs() as f64).fold(0.0, f64::max);
        let e = (f..r.len()).step_by(nf).map(|i| (r[i] - back[i]).abs() as f64).fold(0.0, f64::max);
        // per-field (per-chunk) scale: |err| ≤ max / (2^24 − 2) (+ f32 rounding of the decoded value)
        assert!(e <= mx / ((1u64 << 24) - 2) as f64 + mx * 6e-8 + 1e-30, "field {f}: {e} vs max {mx}");
    }
    // row reads of a few fields of one readout
    let rf = RowFile::open(&d.join("sched.hv")).unwrap();
    let mut b = vec![0u8; 3 * 4];
    rf.read_elems(4 * na * nf + 1, &mut b).unwrap();
    let got: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(got, back[4 * na * nf + 1..4 * na * nf + 4].to_vec());
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn mantissa_relative_and_value_trees() {
    let d = dir("tree");
    let v = volume(20, 10, 6, 2);
    store::save_slice(&d.join("m.npy"), &[20, 10, 6, 2], Order::F, &v, &Save::default().policy(Policy::Mantissa(12))).unwrap();
    let b = store::load_array(&d.join("m.hv")).unwrap().0;
    for (x, y) in v.iter().zip(b.as_slice::<Complex32>().unwrap()) {
        for (p, q) in [(x.re, y.re), (x.im, y.im)] {
            assert!(((p - q) as f64).abs() <= (p.abs() as f64) * 2f64.powi(-13) * 1.0001, "{p} {q}");
        }
    }
    // a value tree with mixed tensors
    let tree = Value::Dict(vec![
        ("img".into(), Value::Tensor(Tensor::from_slice(&[2, 6, 10, 20], &v).unwrap())),
        ("idx".into(), Value::Tensor(Tensor::from_slice(&[5], &[1i64, 2, 3, 4, 5]).unwrap())),
        ("name".into(), "x".into()),
    ]);
    store::save_value(&d.join("t.hv"), &tree, &Save::image()).unwrap();
    let back = store::load_value(&d.join("t.hv")).unwrap();
    assert_eq!(back.get("idx"), tree.get("idx"));
    let (a, bb) = (sc(tree.get("img").unwrap().as_tensor().unwrap()), sc(back.get("img").unwrap().as_tensor().unwrap()));
    let (rms, emax, _) = err(&a, &bb);
    assert!(emax <= store::IMAGE_REL * 12f64.sqrt() * rms / 2.0 * 1.001);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn npy_passthrough_and_sibling_resolution() {
    let d = dir("npy");
    let x: Vec<f64> = (0..1000).map(|i| i as f64 * 0.5).collect();
    npy::write_raw(&d.join("a.npy"), DType::F64, &[10, 100], false, bytes(&x)).unwrap();
    let (t, o) = store::load_array(&d.join("a.hv")).unwrap(); // only the .npy exists
    assert_eq!(o, Order::C);
    assert_eq!(t.to_vec::<f64>().unwrap(), x);
    assert!(store::exists_any(&d.join("a.hv")) && !store::exists_any(&d.join("b.npy")));
    let rf = RowFile::open(&d.join("a.npy")).unwrap();
    assert!(!rf.is_hv());
    let mut b = vec![0u8; 16];
    rf.read_elems(500, &mut b).unwrap();
    assert_eq!(f64::from_le_bytes(b[..8].try_into().unwrap()), 250.0);
    // an unfinished writer leaves nothing under the real name
    {
        let mut w = TensorWriter::create(&d.join("c.hv"), DType::F64, &[10, 100], Order::C, &Save::lossless(), None).unwrap();
        w.push(&bytes(&x)[..800]).unwrap();
        assert!(w.finish().is_err());
    }
    assert!(!d.join("c.hv").exists() && !d.join("c.hv.tmp").exists());
    std::fs::remove_dir_all(&d).unwrap();
}

/// several streamed tensors and small values in one file (host-stack checkpoints): every entry readable, quantised
/// tensors within their own bound
#[test]
fn value_writer_several_tensors() {
    use hasty_value::store::ValueWriter;
    let d = dir("vwriter");
    let p = d.join("multi.hv");
    let a: Vec<f32> = (0..300_000).map(|i| (i as f32 * 0.001).sin() * 50.0).collect();
    let b: Vec<f32> = (0..70_001).map(|i| (i % 13) as f32 * 1e-3 + 1e-6 * i as f32).collect();
    let mut w = ValueWriter::create(&p, 4).unwrap();
    w.put("meta", &Value::Str("{\"k\": 3}".into())).unwrap();
    for (k, v, q) in [("a", &a, Some(hasty_value::Quant::new(hasty_value::Loss::Abs(1e-3)))), ("b", &b, None)] {
        w.begin_tensor(k, DType::F32, &[v.len()], &Save::default().chunk(Chunking::Bytes(1 << 16)), q).unwrap();
        for part in bytes(v).chunks(12_345) {
            w.push(part).unwrap();
        }
        w.end_tensor().unwrap();
    }
    w.put("n", &Value::Int(7)).unwrap();
    w.finish().unwrap();
    let v = hasty_value::read(&mut std::fs::File::open(&p).unwrap()).unwrap();
    assert_eq!(v.get("meta").and_then(|x| x.as_str()), Some("{\"k\": 3}"));
    assert!(matches!(v.get("n"), Some(Value::Int(7))));
    let ra: Vec<f32> = v.get("a").unwrap().as_tensor().unwrap().to_vec().unwrap();
    assert!(a.iter().zip(&ra).all(|(x, y)| (x - y).abs() <= 0.5e-3 + 1e-5));
    assert_eq!(v.get("b").unwrap().as_tensor().unwrap().as_slice::<f32>().unwrap(), &b[..]);
    let lf = hasty_value::LazyFile::open(&p).unwrap();
    let t = lf.root.get("a").and_then(|l| l.tensor()).unwrap();
    let part = lf.read_box(t, &[1000], &[1010]).unwrap();
    assert_eq!(part.as_slice::<f32>().unwrap(), &ra[1000..1010]);
    std::fs::remove_dir_all(&d).unwrap();
}
