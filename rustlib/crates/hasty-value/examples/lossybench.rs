//! Ratio vs error of the lossy storage modes on a slice of a complex64 k-space `.npy` (readouts × samples × coils).
//!   RAYON_NUM_THREADS=2 cargo run --release -p hasty-value --example lossybench -- <data.npy> <sigma> [r0 nr]
//! `sigma` = per-scalar noise std of the scan; the slice is benchmarked noisy (clean + N(0, sigma²)) and clean.
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use hasty_value::{decode, encode_lossy, Codec, Loss, Options, Quant, Scope, Tensor, Transfer, Value};
use num_complex::Complex32;

fn gauss(s: &mut u64) -> f64 {
    let mut u = || {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()
}

fn scalars(t: &Tensor) -> &[f32] {
    let c = t.as_slice::<Complex32>().unwrap();
    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const f32, c.len() * 2) }
}

struct Row {
    name: String,
    quant: Option<Quant>,
}

fn run(label: &str, t: &Tensor, sigma: f64, chunk: &[usize], rows: &[Row]) {
    let raw = t.bytes().len();
    let a = scalars(t);
    let rms = (a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / a.len() as f64).sqrt();
    println!("\n== {label}: shape {:?}, chunk {chunk:?}, {:.0} MB, signal rms {rms:.1}, sigma {sigma:.1}", t.shape(), raw as f64 / 1e6);
    println!("{:<28}{:>8}{:>8}{:>10}{:>10}{:>11}{:>11}{:>9}{:>9}", "mode", "lz4", "zstd3", "rms/σ", "max/σ", "rmsdB", "bits/val", "enc MB/s", "dec MB/s");
    let v = Value::Tensor(t.clone());
    for r in rows {
        let mut out = Vec::new();
        let mut res: Vec<(usize, f64, f64)> = vec![];
        let (mut enc_s, mut dec_s) = (0.0, 0.0);
        let mut dec: Option<Value> = None;
        for (ci, codec) in [Codec::Lz4, Codec::Zstd(3)].into_iter().enumerate() {
            let o = Options { codec, ..Options::default() };
            let plan = |_: &str, _: &Tensor| Some(chunk.to_vec());
            let loss = |_: &str, _: &Tensor| r.quant;
            let mut bytes = Vec::new();
            let t0 = Instant::now();
            encode_lossy(&v, &o, Some(&plan), &loss, &mut |c| {
                bytes.extend_from_slice(&c);
                Ok(())
            })
            .unwrap();
            let e = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let d = decode(&bytes).unwrap();
            let dt = t0.elapsed().as_secs_f64();
            if ci == 0 {
                (enc_s, dec_s) = (e, dt);
                dec = Some(d);
            }
            res.push((bytes.len(), e, dt));
            out = bytes;
        }
        drop(out);
        let Some(Value::Tensor(d)) = dec else { panic!() };
        let b = scalars(&d);
        let (mut se, mut mx) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            let e = (*x as f64 - *y as f64).abs();
            se += e * e;
            mx = mx.max(e);
        }
        let rmse = (se / a.len() as f64).sqrt();
        let db = if rmse > 0.0 { 20.0 * (rmse / rms).log10() } else { f64::NEG_INFINITY };
        println!(
            "{:<28}{:>8.2}{:>8.2}{:>10.4}{:>10.3}{:>11.1}{:>11.2}{:>9.0}{:>9.0}",
            r.name,
            raw as f64 / res[0].0 as f64,
            raw as f64 / res[1].0 as f64,
            rmse / sigma,
            mx / sigma,
            db,
            8.0 * res[1].0 as f64 / (a.len() as f64),
            raw as f64 / 1e6 / enc_s,
            raw as f64 / 1e6 / dec_s
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (path, sigma) = (&args[1], args[2].parse::<f64>().unwrap());
    let r0: usize = args.get(3).map(|x| x.parse().unwrap()).unwrap_or(2000);
    let nr: usize = args.get(4).map(|x| x.parse().unwrap()).unwrap_or(96);
    let mut f = std::fs::File::open(path).unwrap();
    let mut h = [0u8; 10];
    f.read_exact(&mut h).unwrap();
    let off = 10 + u16::from_le_bytes([h[8], h[9]]) as u64;
    let (ns, nc) = (8641usize, 32usize);
    f.seek(SeekFrom::Start(off + (r0 * ns * nc * 8) as u64)).unwrap();
    let mut bytes = vec![0u8; nr * ns * nc * 8];
    f.read_exact(&mut bytes).unwrap();
    let clean = Tensor::from_bytes(hasty_value::DType::C64, vec![nr, ns, nc], &bytes).unwrap();
    let mut s = 12345u64;
    let noisy: Vec<Complex32> = clean.as_slice::<Complex32>().unwrap().iter().map(|c| Complex32::new(c.re + (sigma * gauss(&mut s)) as f32, c.im + (sigma * gauss(&mut s)) as f32)).collect();
    let noisy = Tensor::from_slice(&[nr, ns, nc], &noisy).unwrap();

    let q = |l: Loss| Some(Quant::new(l));
    let mut rows = vec![Row { name: "lossless".into(), quant: None }];
    for m in [16u32, 12, 10, 8] {
        rows.push(Row { name: format!("Mantissa({m})"), quant: q(Loss::Mantissa(m)) });
    }
    for k in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
        rows.push(Row { name: format!("Abs(σ/{k})"), quant: q(Loss::Abs(sigma / k)) });
    }
    // asinh: step is in transfer space, so above the knee k the relative step is step/k
    for (div, kmul) in [(100.0, 1.0), (100.0, 10.0)] {
        rows.push(Row { name: format!("Abs(σ/{div}) Asinh({kmul}σ)"), quant: Some(Quant::new(Loss::Abs(sigma / div)).transfer(Transfer::Asinh(kmul * sigma))) });
    }
    for b in [12u32, 16, 20, 24] {
        rows.push(Row { name: format!("RelMax({b}) chunk"), quant: q(Loss::RelMax(b)) });
    }
    rows.push(Row { name: "RelMax(16) tensor".into(), quant: Some(Quant::new(Loss::RelMax(16)).scope(Scope::Tensor)) });
    rows.push(Row { name: "RelRms(0.01)".into(), quant: q(Loss::RelRms(0.01)) });
    if std::env::var("INTEG").is_ok() {
        // integrity overhead: lossless-lz4 stream of the noisy slice (≈ 200 MB, incompressible → hashing is the extra work)
        let v = Value::Tensor(noisy.clone());
        for a in [hasty_value::Integrity::None, hasty_value::Integrity::Xxh3, hasty_value::Integrity::Blake3] {
            let o = Options { integrity: a, ..Options::default() };
            let (mut e, mut d) = (f64::MAX, f64::MAX);
            let mut bytes = Vec::new();
            for _ in 0..3 {
                bytes.clear();
                let t0 = Instant::now();
                hasty_value::encode(&v, &o, &mut |c| {
                    bytes.extend_from_slice(&c);
                    Ok(())
                })
                .unwrap();
                e = e.min(t0.elapsed().as_secs_f64());
                let t0 = Instant::now();
                decode(&bytes).unwrap();
                d = d.min(t0.elapsed().as_secs_f64());
            }
            println!("integrity {a:?}: encode {:.0} MB/s, decode {:.0} MB/s", noisy.bytes().len() as f64 / 1e6 / e, noisy.bytes().len() as f64 / 1e6 / d);
        }
        return;
    }
    let chunks: Vec<Vec<usize>> = vec![vec![4, 512, 32], vec![1, ns, nc]];
    for ch in &chunks {
        run("noisy", &noisy, sigma, ch, &rows);
    }
    let crow: Vec<Row> = rows.into_iter().filter(|r| matches!(r.name.as_str(), "lossless" | "Abs(σ/4)" | "Abs(σ/100) Asinh(1σ)") || r.name.starts_with("RelMax(16) chunk") || r.name.starts_with("RelMax(24)")).collect();
    run("clean", &clean, sigma, &chunks[0], &crow);
}
