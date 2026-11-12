//! hasty value files (`.hv`) from the command line.
//!
//!   hvtool info <file>                         tree summary (JSON)
//!   hvtool digest <file>                       canonical digest: summary with a CRC32 per tensor (cross-language tests)
//!   hvtool convert <in> <out> [--codec lz4|zstd|none] [--level n] [--delta] [--checksum] [--block-size bytes]
//!       lossy float storage: --loss abs:<step>|relmax:<bits>|relrms:<eps>|mantissa:<bits> [--asinh <knee>] [--scope tensor]
//!       [--chunk <n>] (float tensors only; chunk edge n per dimension, default: one chunk per tensor)
//!       integrity digest: --integrity xxh3|blake3 (verified by a read-back pass after writing)
//!   hvtool verify <file>                       check a file against its integrity digest
//!   hvtool gen <dir>                           the deterministic test sample, once per codec / filter combination
//!   hvtool pack-tree <root> [--level 19] [--keep-npy]
//!       every case directory (case.json + *.npy) under <root> → one case.hv: zstd + shuffle, x chunked
//!       [16,16,16,1,E,2] and mask [16,16,16,1] for random patch reads; verified byte-exact through the lazy
//!       reader before the npy files are deleted. Threads: RAYON_NUM_THREADS.

use std::path::Path;

use hasty_value::{encode_lossy, encode_with, npy, read, write, Codec, Integrity, LazyFile, Loss, Options, Quant, Scope, Tensor, Transfer, Value};
use num_complex::Complex32;

fn digest(v: &Value) -> serde_json::Value {
    use serde_json::json;
    match v {
        Value::Tensor(t) => json!({ "tensor": t.dtype().name(), "shape": t.shape(), "crc32": crc32fast::hash(t.bytes()) }),
        Value::List(x) => json!({ "list": x.iter().map(digest).collect::<Vec<_>>() }),
        Value::Tuple(x) => json!({ "tuple": x.iter().map(digest).collect::<Vec<_>>() }),
        Value::Dict(d) => json!({ "dict": d.iter().map(|(k, v)| json!([k, digest(v)])).collect::<Vec<_>>() }),
        Value::None => json!({ "none": null }),
        Value::Bool(b) => json!({ "bool": b }),
        Value::Int(i) => json!({ "int": i }),
        Value::Float(f) => json!({ "float": f }),
        Value::Str(s) => json!({ "str": s }),
    }
}

/// the cross-language test sample (rebuilt identically in Python and JS)
fn sample() -> Value {
    let a: Vec<Complex32> = (0..5 * 6 * 7).map(|i| Complex32::new(i as f32 * 0.5, -(i as f32) * 0.25)).collect();
    let b: Vec<u16> = (0..33 * 31u32).map(|i| ((i * 7919) % 65536) as u16).collect();
    let smooth: Vec<f32> = (0..64 * 64).map(|i| ((i % 64) as f32 / 9.0).sin() * ((i / 64) as f32 / 13.0).cos()).collect();
    Value::Dict(vec![
        ("a".into(), Value::Tensor(Tensor::from_slice(&[5, 6, 7], &a).unwrap())),
        ("b".into(), Value::Tensor(Tensor::from_slice(&[33, 31], &b).unwrap())),
        ("smooth".into(), Value::Tensor(Tensor::from_slice(&[64, 64], &smooth).unwrap())),
        ("scalar".into(), Value::Tensor(Tensor::scalar(3.5f64))),
        ("empty".into(), Value::Tensor(Tensor::zeros(hasty_value::DType::F32, vec![0, 3]))),
        ("mask".into(), Value::Tensor(Tensor::from_bytes(hasty_value::DType::Bool, vec![4], &[1, 0, 1, 1]).unwrap())),
        ("name".into(), "hasty ✓".into()),
        ("n".into(), Value::Int(-42)),
        ("x".into(), Value::Float(0.125)),
        ("flag".into(), Value::Bool(true)),
        ("nested".into(), Value::List(vec![Value::None, Value::Tuple(vec![Value::Int(1), "two".into()]), Value::Dict(vec![]), Value::List(vec![])])),
    ])
}

fn opt(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn load(p: &str) -> Value {
    read(&mut std::io::BufReader::new(std::fs::File::open(p).unwrap_or_else(|e| panic!("{p}: {e}")))).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// pack one case directory; returns (raw bytes, file bytes)
fn pack_case(dir: &Path, level: i32, keep: bool) -> Result<(u64, u64), String> {
    let mut names: Vec<String> = std::fs::read_dir(dir).map_err(|e| e.to_string())?.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).filter(|n| n.ends_with(".npy")).collect();
    names.sort_by_key(|n| (n != "x.npy", n != "mask.npy", n.clone()));
    if names.is_empty() {
        return Err("no npy files".into());
    }
    let mut items = Vec::new();
    let mut raw = 0u64;
    for n in &names {
        let (t, fortran) = npy::read(&dir.join(n)).map_err(|e| format!("{n}: {e}"))?;
        if fortran {
            return Err(format!("{n}: Fortran-ordered npy (expected C order)"));
        }
        raw += t.nbytes() as u64;
        items.push((n.trim_end_matches(".npy").to_string(), Value::Tensor(t)));
    }
    let v = Value::Dict(items);
    let plan = |path: &str, t: &Tensor| -> Option<Vec<usize>> {
        let s = t.shape();
        match (path, s.len()) {
            (r#"["x"]"#, 6) => Some(vec![16, 16, 16, 1, s[4], s[5]]),
            (r#"["mask"]"#, 4) => Some(vec![16, 16, 16, 1]),
            _ => None,
        }
    };
    let tmp = dir.join("case.hv.tmp");
    {
        let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(|e| e.to_string())?);
        encode_with(&v, &Options::zstd(level), &plan, &mut |c| Ok(std::io::Write::write_all(&mut w, &c)?)).map_err(|e| e.to_string())?;
        std::io::Write::flush(&mut w).map_err(|e| e.to_string())?;
        w.get_ref().sync_all().map_err(|e| e.to_string())?;
    }
    // verify every array through the random-access reader before anything is deleted
    let lf = LazyFile::open(&tmp).map_err(|e| e.to_string())?;
    if let Value::Dict(items) = &v {
        for (k, val) in items {
            let t = lf.root.get(k).and_then(|l| l.tensor()).ok_or_else(|| format!("{k} missing in case.hv"))?;
            let back = lf.read(t).map_err(|e| format!("{k}: {e}"))?;
            if Some(&back) != val.as_tensor() {
                return Err(format!("{k}: verification failed"));
            }
        }
    }
    let file = std::fs::metadata(&tmp).map_err(|e| e.to_string())?.len();
    std::fs::rename(&tmp, dir.join("case.hv")).map_err(|e| e.to_string())?;
    let cj = dir.join("case.json");
    if let Ok(txt) = std::fs::read_to_string(&cj) {
        let mut j: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
        j["storage"] = serde_json::json!({ "file": "case.hv", "codec": format!("zstd-{level} + byte shuffle"), "chunks": { "x": "[16,16,16,1,E,2]", "mask": "[16,16,16,1]" }, "arrays": names.iter().map(|n| n.trim_end_matches(".npy")).collect::<Vec<_>>(), "raw_bytes": raw, "file_bytes": file });
        if let Some(l) = j.get_mut("layout") {
            *l = serde_json::json!({ "x": "case.hv[\"x\"] float16 C [X,Y,Z,F,E,2] / scale (chunked)", "mask": "case.hv[\"mask\"] uint8 C [X,Y,Z,F] (chunked)", "smaps": "case.hv[\"smaps\"] float16 C [X,Y,Z,C,2]" });
        }
        std::fs::write(&cj, serde_json::to_string_pretty(&j).unwrap()).map_err(|e| e.to_string())?;
    }
    if !keep {
        for n in &names {
            std::fs::remove_file(dir.join(n)).map_err(|e| e.to_string())?;
        }
    }
    Ok((raw, file))
}

fn find_cases(root: &Path, out: &mut Vec<std::path::PathBuf>) {
    if root.join("case.json").exists() && root.join("x.npy").exists() {
        out.push(root.to_path_buf());
        return;
    }
    if let Ok(rd) = std::fs::read_dir(root) {
        let mut subs: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect();
        subs.sort();
        for s in subs {
            find_cases(&s, out);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => println!("{}", serde_json::to_string_pretty(&load(&args[1]).summary()).unwrap()),
        Some("digest") => println!("{}", serde_json::to_string(&digest(&load(&args[1]))).unwrap()),
        Some("convert") => {
            let v = load(&args[1]);
            let level = opt(&args, "--level").map(|s| s.parse().unwrap()).unwrap_or(0);
            let mut o = Options { codec: Codec::parse(&opt(&args, "--codec").unwrap_or_default(), level).unwrap(), ..Options::default() };
            o.delta = args.iter().any(|a| a == "--delta");
            o.checksum = args.iter().any(|a| a == "--checksum");
            if let Some(b) = opt(&args, "--block-size") {
                o.block_size = b.parse().unwrap();
            }
            o.integrity = Integrity::parse(&opt(&args, "--integrity").unwrap_or_default()).unwrap();
            let mut w = std::io::BufWriter::new(std::fs::File::create(&args[2]).unwrap());
            if let Some(spec) = opt(&args, "--loss") {
                let (kind, x) = spec.split_once(':').expect("--loss kind:value");
                let x: f64 = x.parse().expect("--loss value");
                let mut q = Quant::new(match kind {
                    "abs" => Loss::Abs(x),
                    "relmax" => Loss::RelMax(x as u32),
                    "relrms" => Loss::RelRms(x),
                    "mantissa" => Loss::Mantissa(x as u32),
                    k => panic!("unknown loss {k}"),
                });
                if let Some(k) = opt(&args, "--asinh") {
                    q = q.transfer(Transfer::Asinh(k.parse().unwrap()));
                }
                if opt(&args, "--scope").as_deref() == Some("tensor") {
                    q = q.scope(Scope::Tensor);
                }
                let edge: Option<usize> = opt(&args, "--chunk").map(|c| c.parse().unwrap());
                let plan = |_: &str, t: &Tensor| edge.map(|e| t.shape().iter().map(|&d| d.min(e).max(1)).collect::<Vec<_>>());
                let loss = |_: &str, t: &Tensor| t.dtype().name().starts_with("float").then_some(q).filter(|_| !matches!(t.dtype().name(), "float16" | "bfloat16")).or_else(|| t.dtype().is_complex().then_some(q));
                encode_lossy(&v, &o, Some(&plan), &loss, &mut |c| Ok(std::io::Write::write_all(&mut w, &c)?)).unwrap();
            } else {
                write(&v, &o, &mut w).unwrap();
            }
            drop(w);
            if o.integrity != Integrity::None {
                LazyFile::open(Path::new(&args[2])).unwrap().verify().expect("read-back verification");
                eprintln!("verified {:?} digest of {}", o.integrity, args[2]);
            }
        }
        Some("verify") => match LazyFile::open(Path::new(&args[1])).and_then(|f| f.verify()) {
            Ok(true) => println!("ok"),
            Ok(false) => println!("no digest in file"),
            Err(e) => {
                eprintln!("FAILED: {e}");
                std::process::exit(1);
            }
        },
        Some("gen") => {
            let dir = Path::new(&args[1]);
            std::fs::create_dir_all(dir).unwrap();
            let v = sample();
            for (name, o) in [
                ("none", Options::none()),
                ("lz4", Options { block_size: 4096, ..Options::default() }),
                ("lz4_delta_crc", Options { block_size: 1000, delta: true, checksum: true, ..Options::default() }),
                ("lz4_noshuffle", Options { shuffle: false, ..Options::default() }),
                ("zstd", Options { block_size: 2048, ..Options::zstd(5) }),
            ] {
                write(&v, &o, &mut std::fs::File::create(dir.join(format!("{name}.hv"))).unwrap()).unwrap();
            }
            std::fs::write(dir.join("digest.json"), serde_json::to_string(&digest(&v)).unwrap()).unwrap();
            eprintln!("wrote sample files to {}", dir.display());
        }
        Some("pack-tree") => {
            let level = opt(&args, "--level").map(|s| s.parse().unwrap()).unwrap_or(19);
            let keep = args.iter().any(|a| a == "--keep-npy");
            let mut cases = vec![];
            find_cases(Path::new(&args[1]), &mut cases);
            eprintln!("{} cases to pack (zstd-{level}, {} threads)", cases.len(), rayon::current_num_threads());
            let (mut tr, mut tf) = (0u64, 0u64);
            let t0 = std::time::Instant::now();
            for (k, c) in cases.iter().enumerate() {
                let t = std::time::Instant::now();
                match pack_case(c, level, keep) {
                    Ok((r, f)) => {
                        tr += r;
                        tf += f;
                        eprintln!("[{}/{}] {}: {:.0} MB -> {:.0} MB ({:.3}), {:.0} MB/s; total {:.3} ({:.1} GB -> {:.1} GB, {:.0} s)", k + 1, cases.len(), c.display(), r as f64 / 1e6, f as f64 / 1e6, f as f64 / r as f64, r as f64 / 1e6 / t.elapsed().as_secs_f64(), tf as f64 / tr as f64, tr as f64 / 1e9, tf as f64 / 1e9, t0.elapsed().as_secs_f64());
                    }
                    Err(e) => eprintln!("[{}/{}] {}: FAILED {e} (npy files kept)", k + 1, cases.len(), c.display()),
                }
            }
        }
        _ => eprintln!("usage: hvtool info|digest|convert|verify|gen|pack-tree ..."),
    }
}
