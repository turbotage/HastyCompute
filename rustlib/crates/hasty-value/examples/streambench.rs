//! In-memory encode / decode throughput of the value format (no files), against memcpy and raw-lz4 ceilings.
//!   RAYON_NUM_THREADS=1 cargo run --release -p hasty-value --example streambench [MB=256]
use std::time::Instant;

use hasty_value::{decode, encode, encode_lossy, Codec, Decoder, Loss, Options, Quant, Tensor, Value};

fn best<F: FnMut()>(mut f: F) -> f64 {
    (0..4).map(|_| {
        let t = Instant::now();
        f();
        t.elapsed().as_secs_f64()
    }).fold(f64::MAX, f64::min)
}

fn main() {
    let mb: usize = std::env::args().nth(1).map(|x| x.parse().unwrap()).unwrap_or(256);
    let n = mb * 1_000_000 / 4;
    let mut s = 1u64;
    let mut g = || {
        let mut u = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()
    };
    let noisy: Vec<f32> = (0..n).map(|i| (3000.0 * (i as f64 * 0.001).sin() + 94.0 * g()) as f32).collect();
    let smooth: Vec<f32> = (0..n).map(|i| (3000.0 * (i as f64 * 0.001).sin()) as f32).collect();
    let raw = (n * 4) as f64 / 1e6;
    println!("{} MB per case, {} rayon threads", raw as usize, rayon::current_num_threads());

    let src: Vec<u8> = noisy.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut dst = vec![0u8; src.len()];
    let t = best(|| dst.copy_from_slice(&src));
    println!("ceiling memcpy (warm dst)           {:>7.0} MB/s", raw / t);
    let t = best(|| {
        let mut v = Vec::new();
        v.extend_from_slice(&src);
        std::hint::black_box(&v);
    });
    println!("cold Vec::extend_from_slice          {:>7.0} MB/s", raw / t);
    let t = best(|| {
        let b = hasty_value::Buffer::zeroed(src.len());
        std::hint::black_box(&b);
    });
    println!("Buffer::zeroed alloc+free (untouched) {:>6.0} MB/s", raw / t);
    let t = best(|| {
        let mut b = hasty_value::Buffer::zeroed(src.len());
        b.as_bytes_mut().copy_from_slice(&src);
        std::hint::black_box(&b);
    });
    println!("Buffer::zeroed + copy (cold)         {:>7.0} MB/s", raw / t);
    let mut sh = vec![0u8; src.len()];
    let t = best(|| hasty_value::codec::shuffle(&src[..1 << 20], 4, &mut sh[..1 << 20]));
    let t_sh = best(|| {
        for c in 0..src.len() >> 20 {
            hasty_value::codec::shuffle(&src[c << 20..(c + 1) << 20], 4, &mut sh[c << 20..(c + 1) << 20]);
        }
    });
    let t_un = best(|| {
        for c in 0..src.len() >> 20 {
            hasty_value::codec::unshuffle(&src[c << 20..(c + 1) << 20], 4, &mut dst[c << 20..(c + 1) << 20]);
        }
    });
    let _ = t;
    println!("shuffle w=4, 1 MiB blocks           {:>7.0} MB/s   unshuffle {:>7.0} MB/s", (src.len() >> 20 << 20) as f64 / 1e6 / t_sh, (src.len() >> 20 << 20) as f64 / 1e6 / t_un);
    println!("\n{:<34}{:>7}{:>11}{:>11}", "case", "ratio", "enc MB/s", "dec MB/s");
    let cases: Vec<(&str, &Vec<f32>, Option<Quant>)> = vec![("noisy lossless", &noisy, None), ("smooth lossless", &smooth, None), ("noisy Abs(σ/4)", &noisy, Some(Quant::new(Loss::Abs(94.0 / 4.0))))];
    let only = std::env::var("ONLY").unwrap_or_default();
    for (name, data, q) in cases {
        if !name.contains(&only) {
            continue;
        }
        let v = Value::Tensor(Tensor::from_slice(&[n / 1024, 1024], data).unwrap());
        for codec in [Codec::Lz4, Codec::Zstd(1)] {
            let o = Options { codec, ..Options::default() };
            let mut bytes = Vec::new();
            let plan = |_: &str, t: &Tensor| Some(vec![64.min(t.shape()[0]), 1024]);
            let loss = |_: &str, _: &Tensor| q;
            let mut enc = || {
                bytes.clear();
                if q.is_some() {
                    encode_lossy(&v, &o, Some(&plan), &loss, &mut |c| Ok(bytes.extend_from_slice(&c))).unwrap();
                } else {
                    encode(&v, &o, &mut |c| Ok(bytes.extend_from_slice(&c))).unwrap();
                }
            };
            let te = best(&mut enc);
            let b = bytes.clone();
            let td = best(|| {
                std::hint::black_box(decode(&b).unwrap());
            });
            let tds = best(|| {
                let mut d = Decoder::new().sequential();
                d.push(&b).unwrap();
                std::hint::black_box(d.finish().unwrap());
            });
            println!("{:<34}{:>7.2}{:>11.0}{:>11.0}   (sequential decoder {:.0})", format!("{name} {codec:?}"), raw * 1e6 / b.len() as f64, raw / te, raw / td, raw / tds);
        }
    }
}
