//! Precision-policy measurements for `research/io_policy.md`: size, error and throughput of `store::save_bytes` /
//! `store::load_array` per policy on (a row subset of) a real array file.
//!
//!   cargo run --release -p hasty-value --example iopolicy -- <in.npy|hv> --out <dir> [--rows start:stop:step]
//!       [--policy SPEC]... [--codec zstd1|zstd3|lz4] [--chunk rows:N|bytes:N] [--sigma σ] [--keep]
//!   SPEC: lossless | image:REL | rel:REL:FLOOR | abs:STEP | kspace (needs --sigma; step = store::kspace_step) |
//!         kspacerel (noise-free term only) | bits:B | mant:M, optional suffix +delta
//!   threads: HASTY_IO_THREADS (encode / decode pool)

use std::path::{Path, PathBuf};
use std::time::Instant;

use hasty_value::store::{self, Chunking, Order, Policy, RowFile, Save};
use hasty_value::{Codec, DType};

fn opt(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}

fn scalars(d: DType, b: &[u8]) -> Vec<f64> {
    match d {
        DType::F32 | DType::C64 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        DType::F64 | DType::C128 => b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        _ => b.iter().map(|&x| x as f64).collect(),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let inp = PathBuf::from(&a[0]);
    let out = PathBuf::from(opt(&a, "--out").expect("--out dir"));
    std::fs::create_dir_all(&out).unwrap();
    let sigma: f64 = opt(&a, "--sigma").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let keep = a.iter().any(|x| x == "--keep");
    // --fields: per last-axis-index (record field) max |x| and max |err| (heterogeneous records, e.g. sched_*.npy)
    let fields = a.iter().any(|x| x == "--fields");
    let codec = match opt(&a, "--codec").as_deref().unwrap_or("zstd1") {
        "lz4" => Codec::Lz4,
        z if z.starts_with("zstd") => Codec::Zstd(z[4..].parse().unwrap_or(3)),
        c => panic!("codec {c}"),
    };
    let chunk = match opt(&a, "--chunk") {
        None => Chunking::Auto,
        Some(c) => match c.split_once(':') {
            Some(("rows", n)) => Chunking::Rows(n.parse().unwrap()),
            Some(("bytes", n)) => Chunking::Bytes(n.parse().unwrap()),
            Some(("shape", n)) => Chunking::Shape(n.split('x').map(|v| v.parse().unwrap()).collect()),
            _ => panic!("--chunk rows:N|bytes:N"),
        },
    };
    // the input (optionally a strided subset of its leading-axis rows)
    let t0 = Instant::now();
    let rf = RowFile::open(&inp).unwrap();
    let (dtype, mut shape, order) = (rf.dtype, rf.shape.clone(), rf.order);
    let row: usize = shape[1..].iter().product::<usize>().max(1);
    let es = dtype.size();
    let rows: Vec<usize> = match opt(&a, "--rows") {
        Some(r) => {
            let v: Vec<usize> = r.split(':').map(|x| x.parse().unwrap()).collect();
            (v[0]..v[1].min(shape[0])).step_by(v[2]).collect()
        }
        None => (0..shape[0]).collect(),
    };
    let mut data = vec![0u8; rows.len() * row * es];
    for (k, &r) in rows.iter().enumerate() {
        rf.read_elems(r * row, &mut data[k * row * es..(k + 1) * row * es]).unwrap();
    }
    shape[0] = rows.len();
    // shape as the caller sees it
    let ushape: Vec<usize> = if order == Order::F { shape.iter().rev().copied().collect() } else { shape.clone() };
    let raw = data.len() as f64;
    println!("input {} {:?} {} ({} order): {} rows, {:.1} MB, read in {:.1} s", inp.display(), ushape, dtype.name(), if order == Order::F { "F" } else { "C" }, rows.len(), raw / 1e6, t0.elapsed().as_secs_f64());
    let x = scalars(dtype, &data);
    let fin: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let rms = (fin.iter().map(|v| v * v).sum::<f64>() / fin.len().max(1) as f64).sqrt();
    let mx = fin.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let comps: f64 = if dtype.is_complex() { 2.0 } else { 1.0 };
    println!("  rms {rms:.4e} per scalar (per complex {:.4e}), max |x| {mx:.4e}, non-finite {}", rms * comps.sqrt(), x.len() - fin.len());
    if sigma > 0.0 {
        println!("  noise σ {sigma:.4e} per complex sample = {:.4e} per component; data rms / σ_comp = {:.1}", sigma / 2f64.sqrt(), rms / (sigma / 2f64.sqrt()));
    }
    let rms_c = rms * comps.sqrt();
    let specs: Vec<String> = a.iter().enumerate().filter(|(_, x)| *x == "--policy").map(|(i, _)| a[i + 1].clone()).collect();
    println!("  {:<22} {:>10} {:>8} {:>8} {:>11} {:>11} {:>10} {:>9} {:>9}", "policy", "file MB", "ratio", "bits/sc", "max|err|", "rms err/rms", "err/σcomp", "enc MB/s", "dec MB/s");
    for spec in &specs {
        let (base, delta) = match spec.strip_suffix("+delta") {
            Some(b) => (b, true),
            None => (spec.as_str(), false),
        };
        let mut it = base.split(':');
        let policy = match it.next().unwrap() {
            "lossless" => Policy::Lossless,
            "image" => Policy::Image { rel: it.next().unwrap().parse().unwrap() },
            "rel" => Policy::Relative { rel: it.next().unwrap().parse().unwrap(), floor: it.next().unwrap().parse().unwrap() },
            "abs" => Policy::Abs(it.next().unwrap().parse().unwrap()),
            "kspace" => Policy::Abs(store::kspace_step(sigma, rms_c)),
            "kspacerel" => Policy::Abs(store::kspace_step(0.0, rms_c)),
            "bits" => Policy::Bits(it.next().unwrap().parse().unwrap()),
            "mant" => Policy::Mantissa(it.next().unwrap().parse().unwrap()),
            p => panic!("policy {p}"),
        };
        let save = Save { policy, codec, chunk: chunk.clone(), delta };
        let name = out.join(format!("bench_{}.hv", spec.replace([':', '+'], "_")));
        let t = Instant::now();
        let path = store::save_bytes(Path::new(&name), dtype, &ushape, order, &data, &save).unwrap();
        let te = t.elapsed().as_secs_f64();
        let fsz = std::fs::metadata(&path).unwrap().len() as f64;
        let t = Instant::now();
        let (back, o2) = store::load_array(&path).unwrap();
        let td = t.elapsed().as_secs_f64();
        assert_eq!(o2, order);
        assert_eq!(back.shape(), &shape[..]);
        let y = scalars(dtype, back.bytes());
        let (mut emax, mut ess, mut nf) = (0.0f64, 0.0f64, 0usize);
        for (p, q) in x.iter().zip(&y) {
            if p.is_finite() {
                let e = (p - q).abs();
                emax = emax.max(e);
                ess += e * e;
                nf += 1;
            } else {
                assert!(p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan()), "non-finite value changed");
            }
        }
        let erms = (ess / nf.max(1) as f64).sqrt();
        if fields {
            let nfld = *ushape.last().unwrap();
            let line: Vec<String> = (0..nfld)
                .map(|f| {
                    let (mut m, mut e) = (0.0f64, 0.0f64);
                    for i in (f..x.len()).step_by(nfld) {
                        m = m.max(x[i].abs());
                        e = e.max((x[i] - y[i]).abs());
                    }
                    format!("{f}:{e:.1e}/{m:.1e}")
                })
                .collect();
            println!("    per field max|err|/max|x|: {}", line.join(" "));
        }
        println!(
            "  {:<22} {:>10.2} {:>8.3} {:>8.2} {:>11.3e} {:>11.3e} {:>10.3e} {:>9.0} {:>9.0}",
            spec,
            fsz / 1e6,
            fsz / raw,
            fsz * 8.0 / x.len() as f64,
            emax,
            erms / rms,
            if sigma > 0.0 { erms / (sigma / 2f64.sqrt()) } else { f64::NAN },
            raw / 1e6 / te,
            raw / 1e6 / td
        );
        if !keep {
            std::fs::remove_file(&path).unwrap();
        }
    }
}
