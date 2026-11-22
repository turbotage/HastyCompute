//! BFP codec microbenchmark and round-trip tool (research/bfp_storage.md §5).
//!
//!   bfp_bench bench [log2 n reals, default 27] [reps 20] [bits,bits,.. default 8,12,16,18,20,22,24] [peak GB/s 1008]
//!     GPU pack (f32 -> BFP) and unpack (BFP -> f32) of n reals per layout (warp block, half-warp complex, per-thread
//!     run) against plain f32 / u32 copies: median kernel time, achieved DRAM bandwidth (model bytes: f32 side 4 n,
//!     BFP side bits/8 n + n/32) and the time over the time a copy of the same bytes takes.
//!   bfp_bench roundtrip <bits> <in.f32> <out.f32>
//!     CPU codec: decode(encode(x)) of a raw little-endian f32 file (cross-check of the Python emulation).
//!   bfp_bench host [log2 n reals, default 26] [reps 5] [bits,.. default 12,16,20,24]
//!     CPU codec throughput: encode / decode of n reals on one thread and on the rayon pool (RAYON_NUM_THREADS), GB/s
//!     of f32 bytes (4 n per pass), median of reps; plus a memcpy of the same f32 bytes for reference.
use cudarc::driver::sys::CUevent_flags;
use hasty_bfp::gpu::{BfpKernels, Layout};
use hasty_bfp::Bfp;
use hasty_cuda::Device;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(|s| s.as_str()) {
        Some("roundtrip") => roundtrip(&a[1..]),
        Some("host") => host(&a[1..]),
        Some("bench") | None => bench(if a.is_empty() { &a[..] } else { &a[1..] }),
        Some(o) => panic!("unknown mode {o} (bench | roundtrip)"),
    }
}

fn roundtrip(a: &[String]) {
    let bits: u32 = a[0].parse().unwrap();
    let raw = std::fs::read(&a[1]).unwrap();
    let x: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let y = Bfp::new(bits).round_trip(&x);
    let out: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&a[2], out).unwrap();
    println!("bfp{bits}: {} reals round-tripped", x.len());
}

fn host(a: &[String]) {
    use std::time::Instant;
    let lg: u32 = a.first().map_or(26, |v| v.parse().unwrap());
    let reps: usize = a.get(1).map_or(5, |v| v.parse().unwrap());
    let bits: Vec<u32> = a.get(2).map_or(vec![12, 16, 20, 24], |v| v.split(',').map(|b| b.parse().unwrap()).collect());
    let n = 1usize << lg;
    let x = lcg_data(n);
    let th = rayon::current_num_threads();
    let med = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let gbs = |t: f64| 4.0 * n as f64 / t / 1e9;
    let mut y = vec![0f32; n];
    let tc = med((0..reps).map(|_| {
        let t = Instant::now();
        y.copy_from_slice(&x);
        t.elapsed().as_secs_f64()
    }).collect());
    println!("host codec, {n} reals ({:.0} MiB f32), {th} rayon threads; memcpy {:.2} GB/s (1 thread)", 4.0 * n as f64 / (1 << 20) as f64, gbs(tc));
    println!("{:>5} {:>14} {:>14} {:>14} {:>14}", "bits", "enc 1 thr", "dec 1 thr", format!("enc {th} thr"), format!("dec {th} thr"));
    for b in bits {
        let f = Bfp::new(b);
        let mut buf = hasty_bfp::BfpBuf::zeros(f, n);
        let es = med((0..reps).map(|_| {
            let t = Instant::now();
            f.encode_serial(&x, &mut buf.mant, &mut buf.ex);
            t.elapsed().as_secs_f64()
        }).collect());
        let ds = med((0..reps).map(|_| {
            let t = Instant::now();
            f.decode_serial(&buf.mant, &buf.ex, &mut y);
            t.elapsed().as_secs_f64()
        }).collect());
        let ep = med((0..reps).map(|_| {
            let t = Instant::now();
            buf.encode_from(&x);
            t.elapsed().as_secs_f64()
        }).collect());
        let dp = med((0..reps).map(|_| {
            let t = Instant::now();
            buf.decode_into(&mut y);
            t.elapsed().as_secs_f64()
        }).collect());
        println!("{b:>5} {:>11.2} GB/s {:>9.2} GB/s {:>9.2} GB/s {:>9.2} GB/s", gbs(es), gbs(ds), gbs(ep), gbs(dp));
    }
}

fn lcg_data(n: usize) -> Vec<f32> {
    // complex-like data with a smooth magnitude envelope over blocks (realistic dynamic range inside a block)
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n).map(|i| ((r() - 0.5) * (1.0 + ((i >> 9) % 97) as f64)) as f32).collect()
}

fn bench(a: &[String]) {
    let lg: u32 = a.first().and_then(|v| v.parse().ok()).unwrap_or(27);
    let reps: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
    let bits: Vec<u32> = a.get(2).map(|v| v.split(',').map(|b| b.parse().unwrap()).collect()).unwrap_or(vec![8, 12, 16, 18, 20, 22, 24]);
    let peak = a.get(3).and_then(|v| v.parse::<f64>().ok()).unwrap_or(1008.0);
    let n = 1usize << lg;
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let x = st.clone_htod(&lcg_data(n)).unwrap();
    let mut y = st.alloc_zeros::<f32>(n).unwrap();
    let mut mant = st.alloc_zeros::<u32>(Bfp::new(24).mant_words(n)).unwrap();
    let mut ex = st.alloc_zeros::<u8>(Bfp::blocks(n).div_ceil(4) * 4).unwrap();
    let mut mant2 = st.alloc_zeros::<u32>(Bfp::new(24).mant_words(n)).unwrap();
    let f = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let ev = (dev.ctx.new_event(f).unwrap(), dev.ctx.new_event(f).unwrap());
    let time = |run: &mut dyn FnMut()| -> f64 {
        run();
        let mut t = Vec::new();
        for _ in 0..reps {
            ev.0.record(&st).unwrap();
            run();
            ev.1.record(&st).unwrap();
            ev.1.synchronize().unwrap();
            t.push(ev.0.elapsed_ms(&ev.1).unwrap() as f64);
        }
        t.sort_by(|p, q| p.partial_cmp(q).unwrap());
        t[t.len() / 2]
    };
    let k0 = BfpKernels::new(&dev, 24).unwrap();
    let t_copy = time(&mut || k0.copy_f32(&st, &x, &mut y, n).unwrap());
    let bw_copy = 8.0 * n as f64 / (t_copy * 1e-3) / 1e9;
    println!("n = 2^{lg} reals ({:.0} MB f32), median of {reps}; peak {peak:.0} GB/s", 4.0 * n as f64 / 1e6);
    println!("copy_f32: {t_copy:.3} ms, {bw_copy:.0} GB/s ({:.0} % of peak)", 100.0 * bw_copy / peak);
    println!("{:>4} {:>10} {:>8} {:>9} {:>7} {:>9} | {:>8} {:>9} {:>7} {:>9} | {:>10}", "bits", "layout", "pack ms", "GB/s", "%peak", "t/t_copy", "unpk ms", "GB/s", "%peak", "t/t_copy", "u32copy ms");
    for &b in &bits {
        let k = BfpKernels::new(&dev, b).unwrap();
        let fmt = Bfp::new(b);
        let words = fmt.mant_words(n);
        let t_u32 = time(&mut || k.copy_u32(&st, &mant, &mut mant2, words).unwrap());
        // bytes moved by pack / unpack and the time a plain copy of those bytes takes at the f32-copy bandwidth
        let bytes = 4.0 * n as f64 + fmt.bytes(n) as f64;
        let t_ideal = bytes / (bw_copy * 1e9) * 1e3;
        for l in Layout::ALL {
            let tp = time(&mut || k.pack(&st, l, &x, n, &mut mant, &mut ex).unwrap());
            let tu = time(&mut || k.unpack(&st, l, &mant, &ex, n, &mut y).unwrap());
            let (bp, bu) = (bytes / (tp * 1e-3) / 1e9, bytes / (tu * 1e-3) / 1e9);
            println!(
                "{b:>4} {:>10} {tp:>8.3} {bp:>9.0} {:>7.0} {:>9.2} | {tu:>8.3} {bu:>9.0} {:>7.0} {:>9.2} | {t_u32:>10.3}",
                l.name(),
                100.0 * bp / peak,
                tp / t_ideal,
                100.0 * bu / peak,
                tu / t_ideal
            );
        }
        // random-access decode (one complex value per thread): the gather-side decoder
        let tr = time(&mut || k.unpack_random(&st, true, &mant, &ex, n, &mut y).unwrap());
        let br = bytes / (tr * 1e-3) / 1e9;
        println!("{b:>4} {:>10} {:>8} {:>9} {:>7} {:>9} | {tr:>8.3} {br:>9.0} {:>7.0} {:>9.2} |", "random-c", "-", "-", "-", "-", 100.0 * br / peak, tr / t_ideal);
    }
}
