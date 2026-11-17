//! Storage audit: size and error of a file under the I/O precision policy (`research/io_policy.md`), measured by
//! encoding chunks sampled evenly across the file with the real codec, then extrapolated to the whole file. Read-only.
//!
//!   ioaudit <class> <file> [--sample-mb 384] [--codec zstd1]
//!
//! classes: kspace (complex [readout][adc][coil] .npy; noise σ from meta.json next to it), traj (float32
//! [readout][adc][fields] per-sample records), image (float / complex volumes: Policy::Image 1e-4), lossless (any
//! .npy, bit-exact), hvtree (an existing .hv tree: floats → image policy, the rest lossless), checkpoint (.hv solver
//! state: lossless), beats (LBM beat library .bin), raw (any file, bytes as float32 words, lossless).
//! One JSON line per file on stdout.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::Instant;

use hasty_value::lazy::{Lazy, LazyFile, TensorRef};
use hasty_value::lossy::{Loss, Quant};
use hasty_value::store::{self, Chunking, Policy};
use hasty_value::{npy, Codec, DType, Options};
use rayon::prelude::*;
use serde_json::json;

/// accumulated error statistics (scalars: complex = re, im)
#[derive(Default, Clone, Copy)]
struct Err {
    max: f64,
    se: f64,
    ss: f64,
    n: usize,
    xmax: f64,
}

impl Err {
    fn merge(a: Err, b: Err) -> Err {
        Err { max: a.max.max(b.max), se: a.se + b.se, ss: a.ss + b.ss, n: a.n + b.n, xmax: a.xmax.max(b.xmax) }
    }
}

fn scalars(dtype: DType, b: &[u8]) -> Vec<f64> {
    match dtype {
        DType::F32 | DType::C64 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        DType::F64 | DType::C128 => b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        _ => vec![],
    }
}

fn err_of(dtype: DType, a: &[u8], b: &[u8]) -> Err {
    let (x, y) = (scalars(dtype, a), scalars(dtype, b));
    let mut e = Err::default();
    for (p, q) in x.iter().zip(&y) {
        if !p.is_finite() {
            continue;
        }
        let d = (p - q).abs();
        e.max = e.max.max(d);
        e.se += d * d;
        e.ss += p * p;
        e.xmax = e.xmax.max(p.abs());
        e.n += 1;
    }
    if x.is_empty() {
        assert_eq!(a, b, "lossless chunk differs");
    }
    e
}

struct Run {
    raw: u64,
    enc: u64,
    err: Err,
    t_enc: f64,
    t_dec: f64,
}

/// encode (and decode + compare) `chunks` (raw C-order bytes) in parallel
fn run(chunks: &[Vec<u8>], dtype: DType, q: Option<&Quant>, o: &Options, check: bool) -> Run {
    let t = Instant::now();
    let enc: Vec<Vec<u8>> = store::io_pool().install(|| chunks.par_iter().map(|c| store::encode_chunk(c, dtype, q, o)).collect());
    let t_enc = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let dec: Vec<Vec<u8>> = store::io_pool().install(|| enc.par_iter().zip(chunks).map(|(e, c)| store::decode_chunk(e, dtype, c.len() / dtype.size(), q, o).unwrap()).collect());
    let t_dec = t.elapsed().as_secs_f64();
    let err = if check { store::io_pool().install(|| dec.par_iter().zip(chunks).map(|(d, c)| err_of(dtype, c, d)).reduce(Err::default, Err::merge)) } else { Err::default() };
    Run { raw: chunks.iter().map(|c| c.len() as u64).sum(), enc: enc.iter().map(|c| c.len() as u64).sum(), err, t_enc, t_dec }
}

fn rjson(r: &Run) -> serde_json::Value {
    let rms = (r.err.ss / r.err.n.max(1) as f64).sqrt();
    json!({"raw_sampled": r.raw, "enc_sampled": r.enc, "ratio": r.enc as f64 / r.raw.max(1) as f64,
        "max_abs_err": r.err.max, "rms_err": (r.err.se / r.err.n.max(1) as f64).sqrt(), "rms": rms, "max_abs": r.err.xmax,
        "rms_err_rel": (r.err.se / r.err.ss.max(1e-300)).sqrt(), "max_err_rel_rms": r.err.max / rms.max(1e-300),
        "enc_MBps": r.raw as f64 / 1e6 / r.t_enc.max(1e-9), "dec_MBps": r.raw as f64 / 1e6 / r.t_dec.max(1e-9)})
}

/// `k` evenly spread pieces of `len` bytes, aligned to `align`, from [off, off + total)
fn sample_pieces(f: &File, off: u64, total: u64, len: u64, align: u64, k: usize) -> Vec<Vec<u8>> {
    let n_units = (total / align).max(1);
    let per = (len / align).max(1);
    let slots = (n_units / per).max(1);
    let k = (k as u64).min(slots);
    (0..k)
        .into_par_iter()
        .map(|i| {
            let slot = if k <= 1 { 0 } else { i * (slots - 1) / (k - 1) };
            let start = slot * per * align;
            let l = (per * align).min(total - start);
            let mut b = vec![0u8; l as usize];
            f.read_exact_at(&mut b, off + start).unwrap();
            b
        })
        .collect()
}

fn meta_sigma(path: &Path) -> f64 {
    let m = path.parent().unwrap().join("meta.json");
    std::fs::read_to_string(m).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()).and_then(|j| j["noise_sigma"].as_f64()).unwrap_or(0.0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let (class, path) = (args[0].as_str(), Path::new(&args[1]));
    let budget = opt("--sample-mb").map(|s| s.parse::<u64>().unwrap()).unwrap_or(384) << 20;
    let codec = match opt("--codec").as_deref().unwrap_or("zstd1") {
        "lz4" => Codec::Lz4,
        z => Codec::Zstd(z.trim_start_matches("zstd").parse().unwrap_or(1)),
    };
    let o = Options { codec, ..Options::default() };
    let size = std::fs::metadata(path).unwrap().len();
    let f = File::open(path).unwrap();
    let mut out = json!({"file": path.display().to_string(), "class": class, "size": size});
    match class {
        "kspace" | "traj" | "image" | "lossless" => {
            let ((dtype, shape, fortran), off) = npy::header(path).unwrap();
            let cshape: Vec<usize> = if fortran { shape.iter().rev().copied().collect() } else { shape.clone() };
            let es = dtype.size();
            let total = cshape.iter().product::<usize>() as u64 * es as u64;
            let chunk = match class {
                "kspace" | "traj" => store::chunk_shape(&cshape, es, &Chunking::Rows(1)),
                _ => store::chunk_shape(&cshape, es, &Chunking::Auto),
            };
            let clen = chunk.iter().product::<usize>() as u64 * es as u64;
            let k = (budget / clen).max(1) as usize;
            let chunks = sample_pieces(&f, off, total, clen, clen, k);
            out["dtype"] = json!(dtype.name());
            out["shape"] = json!(shape);
            out["chunk_bytes"] = json!(clen);
            out["n_chunks_sampled"] = json!(chunks.len());
            let all: Vec<u8> = chunks.concat();
            match class {
                "kspace" => {
                    let sigma = meta_sigma(path);
                    let sc = scalars(dtype, &all);
                    let rms_c = (sc.iter().map(|x| x * x).sum::<f64>() / sc.len() as f64 * 2.0).sqrt();
                    let step = store::kspace_step(sigma, rms_c);
                    let q = Quant::new(Loss::Abs(step));
                    let r = run(&chunks, dtype, Some(&q), &o, true);
                    let mut j = rjson(&r);
                    let sc_noise = sigma / 2f64.sqrt();
                    j["step"] = json!(step);
                    j["sigma_complex"] = json!(sigma);
                    j["rms_complex"] = json!(rms_c);
                    j["err_rms_over_noise"] = json!(if sigma > 0.0 { (r.err.se / r.err.n as f64).sqrt() / sc_noise } else { f64::NAN });
                    j["adc_bits"] = json!((2.0 * r.err.xmax / step).log2().ceil());
                    j["noise_bits_below_step"] = json!(if sigma > 0.0 { (sc_noise / step).log2() } else { f64::NAN });
                    // the two terms of the step rule
                    j["step_noise_term"] = json!(store::KSPACE_NOISE_FRAC * sc_noise);
                    j["step_rel_term"] = json!(store::KSPACE_REL * 12f64.sqrt() * rms_c / 2f64.sqrt());
                    out["policy"] = json!(format!("Abs(kspace_step) = {step:.4e}"));
                    out["result"] = j;
                    let ll = run(&chunks, dtype, None, &o, false);
                    out["lossless_ratio"] = json!(ll.enc as f64 / ll.raw as f64);
                }
                "traj" => {
                    // per-field chunks [1, na, 1] (one step per field and readout), signed 22-bit range of the chunk
                    let nf = *cshape.last().unwrap();
                    let fields: Vec<Vec<u8>> = chunks.iter().flat_map(|c| (0..nf).map(move |fi| c.chunks_exact(4 * nf).flat_map(|r| r[4 * fi..4 * fi + 4].to_vec()).collect::<Vec<u8>>())).collect();
                    let od = Options { delta: true, ..o };
                    let mut variants = serde_json::Map::new();
                    for (name, q, oo, ch) in [
                        ("lossless", None, o, &chunks),
                        ("bits22_field_delta", Some(Quant::new(Loss::RelMax(22))), od, &fields),
                        ("bits20_field_delta", Some(Quant::new(Loss::RelMax(20))), od, &fields),
                        ("mantissa20", Some(Quant::new(Loss::Mantissa(20))), o, &chunks),
                        ("mantissa16", Some(Quant::new(Loss::Mantissa(16))), o, &chunks),
                    ] {
                        let r = run(ch, dtype, q.as_ref(), &oo, true);
                        let mut j = rjson(&r);
                        // worst error in the k fields (0..3, rad/m) and τ (13, s)
                        if q.is_some() {
                            let per_field = |fi: usize| -> f64 {
                                let sub: Vec<Vec<u8>> = if name.contains("field") { fields.iter().enumerate().filter(|(i, _)| i % nf == fi).map(|(_, v)| v.clone()).collect() } else { chunks.iter().map(|c| c.chunks_exact(4 * nf).flat_map(|r| r[4 * fi..4 * fi + 4].to_vec()).collect()).collect() };
                                // re-encode the field alone with the same spec (identical bytes for per-field chunks; Mantissa is pointwise)
                                run(&sub, dtype, q.as_ref(), &oo, true).err.max
                            };
                            j["max_err_k_radpm"] = json!((0..3).map(per_field).fold(0.0, f64::max));
                            j["max_err_tau_s"] = json!(per_field(match nf { 16 => 13, 8 => 6, 4 => 3, n => n - 1 }));
                        }
                        variants.insert(name.into(), j);
                    }
                    out["policy"] = json!("Bits(22) per field chunk [1, na, 1] + delta");
                    out["result"] = variants["bits22_field_delta"].clone();
                    out["variants"] = serde_json::Value::Object(variants);
                }
                "image" => {
                    let (q, rec) = store::resolve_policy(Policy::image(), dtype, Some(&[&all[..]])).unwrap();
                    let r = run(&chunks, dtype, q.as_ref(), &o, true);
                    out["policy"] = json!(rec);
                    out["result"] = rjson(&r);
                    let mut cod = serde_json::Map::new();
                    for (n, c) in [("lz4", Codec::Lz4), ("zstd1", Codec::Zstd(1)), ("zstd3", Codec::Zstd(3))] {
                        let rr = run(&chunks, dtype, q.as_ref(), &Options { codec: c, ..o }, false);
                        cod.insert(n.into(), json!({"ratio": rr.enc as f64 / rr.raw as f64, "enc_MBps": rr.raw as f64 / 1e6 / rr.t_enc, "dec_MBps": rr.raw as f64 / 1e6 / rr.t_dec}));
                    }
                    out["codecs"] = serde_json::Value::Object(cod);
                    let ll = run(&chunks, dtype, None, &o, false);
                    out["lossless_ratio"] = json!(ll.enc as f64 / ll.raw as f64);
                }
                _ => {
                    let r = run(&chunks, dtype, None, &o, true);
                    out["policy"] = json!("lossless");
                    out["result"] = rjson(&r);
                }
            }
        }
        "hvtree" | "checkpoint" => {
            let lf = LazyFile::open(path).unwrap();
            let mut ts: Vec<(String, TensorRef)> = vec![];
            fn walk(l: &Lazy, p: String, ts: &mut Vec<(String, TensorRef)>) {
                match l {
                    Lazy::Tensor(t) => ts.push((p, t.clone())),
                    Lazy::List(v) | Lazy::Tuple(v) => v.iter().enumerate().for_each(|(i, x)| walk(x, format!("{p}[{i}]"), ts)),
                    Lazy::Dict(d) => d.iter().for_each(|(k, x)| walk(x, format!("{p}[\"{k}\"]"), ts)),
                    Lazy::Value(_) => {}
                }
            }
            walk(&lf.root, String::new(), &mut ts);
            let nb = |t: &TensorRef| t.shape.iter().product::<usize>() as u64 * t.dtype.size() as u64;
            let raw_total: u64 = ts.iter().map(|(_, t)| nb(t)).sum();
            // groups by (dtype, shape): a spread subset of each group within the budget share of the group
            let mut groups: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
            for (i, (_, t)) in ts.iter().enumerate() {
                groups.entry(format!("{} {:?}", t.dtype.name(), t.shape)).or_default().push(i);
            }
            let (mut proj, mut sraw, mut senc, mut err) = (0f64, 0u64, 0u64, Err::default());
            let mut gj = vec![];
            for (g, ids) in &groups {
                let graw: u64 = ids.iter().map(|&i| nb(&ts[i].1)).sum();
                let share = ((budget as f64 * graw as f64 / raw_total.max(1) as f64) as u64).max(64 << 20);
                let per = nb(&ts[ids[0]].1).max(1);
                let k = ((share / per) as usize).clamp(1, ids.len());
                let pick: Vec<usize> = (0..k).map(|j| ids[if k == 1 { 0 } else { j * (ids.len() - 1) / (k - 1) }]).collect();
                let (mut graw_s, mut genc_s) = (0u64, 0u64);
                for &i in &pick {
                    let t = &ts[i].1;
                    // big tensors: a leading-axis sub-box of about the share (chunked tensors decode only that)
                    let tt = if nb(t) > share.max(256 << 20) && !t.shape.is_empty() {
                        let rows = ((share.max(256 << 20) / (nb(t) / t.shape[0] as u64).max(1)) as usize).clamp(1, t.shape[0]);
                        let lo0 = (t.shape[0] - rows) / 2;
                        let mut lo = vec![0; t.shape.len()];
                        let mut hi = t.shape.clone();
                        lo[0] = lo0;
                        hi[0] = lo0 + rows;
                        store::io_pool().install(|| lf.read_box(t, &lo, &hi)).unwrap()
                    } else {
                        store::io_pool().install(|| lf.read(t)).unwrap()
                    };
                    let pol = if class == "checkpoint" { Policy::Lossless } else { Policy::image() };
                    let (q, _) = store::resolve_policy(pol, tt.dtype(), Some(&[tt.bytes()])).unwrap();
                    let cs = store::chunk_shape(tt.shape(), tt.dtype().size(), &Chunking::Auto);
                    let clen = cs.iter().product::<usize>() * tt.dtype().size();
                    let chunks: Vec<Vec<u8>> = tt.bytes().chunks(clen.max(1)).map(|c| c.to_vec()).collect();
                    let r = run(&chunks, tt.dtype(), q.as_ref(), &o, q.is_some());
                    graw_s += r.raw;
                    genc_s += r.enc;
                    err = Err::merge(err, r.err);
                }
                let ratio = genc_s as f64 / graw_s.max(1) as f64;
                proj += graw as f64 * ratio;
                sraw += graw_s;
                senc += genc_s;
                gj.push(json!({"group": g, "n": ids.len(), "raw": graw, "sampled": graw_s, "ratio": ratio}));
            }
            let rms = (err.ss / err.n.max(1) as f64).sqrt();
            out["policy"] = json!(if class == "checkpoint" { "lossless (zstd1 + shuffle)" } else { "floats Image(1e-4), others lossless" });
            out["raw_total"] = json!(raw_total);
            out["projected"] = json!(proj);
            out["result"] = json!({"raw_sampled": sraw, "enc_sampled": senc, "ratio": proj / size as f64, "ratio_vs_raw": proj / raw_total.max(1) as f64, "max_abs_err": err.max, "rms_err_rel": (err.se / err.ss.max(1e-300)).sqrt(), "max_err_rel_rms": err.max / rms.max(1e-300)});
            out["groups"] = json!(gj);
        }
        "beats" => {
            // blocks [bin u32][n u32][samples f32][cell u32 × n][f16 × 28 n]
            let mut blocks = vec![];
            let mut p = 0u64;
            while p + 12 <= size {
                let mut h = [0u8; 12];
                f.read_exact_at(&mut h, p).unwrap();
                let n = u32::from_le_bytes(h[4..8].try_into().unwrap()) as u64;
                blocks.push((p, n));
                p += 12 + 4 * n + 56 * n;
            }
            assert_eq!(p, size, "beat file layout");
            let per = blocks.iter().map(|b| 60 * b.1).max().unwrap_or(1);
            let k = ((budget / per.max(1)) as usize).clamp(1, blocks.len());
            let pick: Vec<(u64, u64)> = (0..k).map(|j| blocks[if k == 1 { 0 } else { j * (blocks.len() - 1) / (k - 1) }]).collect();
            let od = Options { delta: true, ..o };
            let (mut raw, mut enc_a, mut enc_b) = (0u64, 0u64, 0u64);
            let t = Instant::now();
            for &(p, n) in &pick {
                let mut cells = vec![0u8; 4 * n as usize];
                f.read_exact_at(&mut cells, p + 12).unwrap();
                let mut rec = vec![0u8; 56 * n as usize];
                f.read_exact_at(&mut rec, p + 12 + 4 * n).unwrap();
                let pieces = |b: &[u8], dt: DType, oo: &Options| -> u64 { store::io_pool().install(|| b.par_chunks(4 << 20).map(|c| store::encode_chunk(c, dt, None, oo).len() as u64).sum()) };
                let c_enc = pieces(&cells, DType::U32, &od);
                // (a) records as written ([n][28] f16), (b) field-major ([28][n])
                let a = pieces(&rec, DType::F16, &o);
                let mut fm = vec![0u8; rec.len()];
                for i in 0..n as usize {
                    for fi in 0..28 {
                        fm[2 * (fi * n as usize + i)..2 * (fi * n as usize + i) + 2].copy_from_slice(&rec[2 * (i * 28 + fi)..2 * (i * 28 + fi) + 2]);
                    }
                }
                let b = pieces(&fm, DType::F16, &o);
                raw += 12 + 60 * n;
                enc_a += 12 + c_enc + a;
                enc_b += 12 + c_enc + b;
            }
            out["policy"] = json!("lossless (f16 already rel 4.9e-4): cells delta+shuffle, records shuffle; zstd1");
            out["result"] = json!({"raw_sampled": raw, "enc_sampled": enc_a, "ratio": enc_a as f64 / raw as f64, "ratio_field_major": enc_b as f64 / raw as f64, "max_abs_err": 0.0, "enc_MBps": raw as f64 / 1e6 / t.elapsed().as_secs_f64()});
            out["n_blocks"] = json!(blocks.len());
        }
        "raw" => {
            let chunks = sample_pieces(&f, 0, size - size % 4, 4 << 20, 4, (budget >> 22) as usize);
            let r = run(&chunks, DType::F32, None, &o, false);
            out["policy"] = json!("lossless (raw words, shuffle 4)");
            out["result"] = rjson(&r);
        }
        c => panic!("unknown class {c}"),
    }
    println!("{}", out);
}
