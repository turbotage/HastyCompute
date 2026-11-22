//! BFP y ↔ z intermediates of the grid FFT (research/bfp_storage.md §7 step 1, `PrunedChanFft3::set_store`): the
//! SegNufft engines' pruned channel-interleaved transform (image block N inside grid n, nch channels, optionally
//! k-support pruned to a ball), forward and inverse, f32 against bfp<b>, with or without the two-pass L2 slabs;
//! median time (CUDA events) and the transform's error against f32 on random data.
//! `--once`: after one warm-up call of each direction, one forward and one inverse (for ncu: with whole-grid passes
//! the kernels are x, y, z, z, y, x per pair, plus one exponent memset per direction with a k support).
//!   cargo run --release -p hasty-fft --example bfp_fft_bench -- n0 n1 n2 N0 N1 N2 nch [--store f32|bfpB] [--dense] [--slab planes] [--reps 9] [--ball rxy,rz] [--once]
//! `--noerr`: timing only (no error against f32). `--dense`: the BFP intermediates in a separate buffer (default: in the grid buffer when n0 · nch is a multiple of 16).
use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_fft::{KSupport, PrunedChanFft3, Store};

fn main() {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |k: &str| -> Option<String> { a.iter().position(|x| x == k).map(|i| { a.remove(i); a.remove(i) }) };
    let store = take("--store").map(|s| Store::parse(&s).expect("--store f32|bfp16..24")).unwrap_or(Store::F32);
    let slab: usize = take("--slab").map(|s| s.parse().unwrap()).unwrap_or(0);
    let reps: usize = take("--reps").map(|s| s.parse().unwrap()).unwrap_or(9);
    let ball: Option<(f64, f64)> = take("--ball").map(|s| { let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect(); (v[0], v[1]) });
    let once = a.iter().position(|x| x == "--once").map(|i| a.remove(i)).is_some();
    let dense = a.iter().position(|x| x == "--dense").map(|i| a.remove(i)).is_some();
    // --noerr: skip the host-side error against f32 (timing only)
    let noerr = a.iter().position(|x| x == "--noerr").map(|i| a.remove(i)).is_some();
    let u = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let n = [u(0, 288), u(1, 288), u(2, 140)];
    let nn = [u(3, 225), u(4, 225), u(5, 112)];
    let nch = u(6, 50);
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let keep = [2, 1, 0].map(|k| (nn[k] - nn[k] / 2, nn[k] / 2));
    // a ball k support: columns |k_xy| ≤ rxy (index units), kz chord rz · sqrt(1 − (k_xy / rxy)²)
    let ks = ball.map(|(rxy, rz)| {
        let c = |i: usize, m: usize| { let i = i as f64; if i > m as f64 / 2.0 { i - m as f64 } else { i } };
        let q = move |x: usize, y: usize| 1.0 - (c(x, n[0]) / rxy).powi(2) - (c(y, n[1]) / rxy).powi(2);
        KSupport::from_columns(n, |x, y| q(x, y) >= 0.0, |x, y| KSupport::hull(n[2], |z| c(z, n[2]).abs() <= rz * q(x, y).max(0.0).sqrt()))
    });
    let mut p = PrunedChanFft3::<f32>::new_ksupport(&dev, [n[2], n[1], n[0]], nch, keep, 1.0, ks.as_ref()).unwrap();
    p.set_slab(slab);
    p.set_store_opts(&dev, store, dense).unwrap();
    let mut buf = st.alloc_zeros::<C32>(p.len()).unwrap();
    let desc = format!(
        "grid {n:?} (x, y, z) block {nn:?} nch {nch} store {store}{} slab {slab}{}",
        if store == Store::F32 { "" } else if p.store_inplace() { " in place" } else { " separate buffer" },
        ks.as_ref().map_or(String::new(), |k| { let (c, _) = k.fractions(n); format!(", ball support: z-pass columns {c:.3}, cells {:.3}", k.cell_fraction(n)) })
    );
    if once {
        p.forward(&st, &mut buf).unwrap();
        p.inverse(&st, &mut buf).unwrap();
        p.forward(&st, &mut buf).unwrap();
        p.inverse(&st, &mut buf).unwrap();
        st.synchronize().unwrap();
        println!("{desc}: one warm-up and one profiled pair done");
        return;
    }
    let f = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let ev = (dev.ctx.new_event(f).unwrap(), dev.ctx.new_event(f).unwrap());
    // random data (forward: image block; inverse: a full spectrum, zero outside the support for pruned plans)
    let len = p.len();
    let mut s = 0x2545f4914f6cdd1du64;
    let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 };
    let host: Vec<C32> = (0..len).map(|_| C32::new(rnd(), rnd())).collect();
    let mut time = |fw: bool| -> f64 {
        let mut t = Vec::new();
        for i in 0..=reps {
            st.memcpy_htod(&host, &mut buf).unwrap();
            ev.0.record(&st).unwrap();
            if fw { p.forward(&st, &mut buf).unwrap() } else { p.inverse(&st, &mut buf).unwrap() }
            ev.1.record(&st).unwrap();
            ev.1.synchronize().unwrap();
            if i > 0 {
                t.push(ev.0.elapsed_ms(&ev.1).unwrap() as f64);
            }
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t[t.len() / 2]
    };
    let (tf, ti) = (time(true), time(false));
    // error against an f32 plan of the same pruning (forward on the support, inverse on the image block)
    let err = if store != Store::F32 && !noerr {
        let mut r = PrunedChanFft3::<f32>::new_ksupport(&dev, [n[2], n[1], n[0]], nch, keep, 1.0, ks.as_ref()).unwrap();
        r.set_slab(slab);
        let inw = |i: usize, m: usize, (h, t): (usize, usize)| i < h || i >= m - t;
        let idx = |q: usize| { let c = q / nch; (c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1])) };
        let sup = |q: usize| { let (x, y, z) = idx(q); ks.as_ref().is_none_or(|k| { let zr = k.zcols[y * n[0] + x]; zr != (0, 0) && inw(z, n[2], zr) }) };
        let blk = |q: usize| { let (x, y, z) = idx(q); inw(x, n[0], keep[2]) && inw(y, n[1], keep[1]) && inw(z, n[2], keep[0]) };
        let spec: Vec<C32> = host.iter().enumerate().map(|(q, v)| if sup(q) { *v } else { C32::new(0.0, 0.0) }).collect();
        let run = |pl: &PrunedChanFft3<f32>, x: &[C32], fw: bool| { let mut d = st.clone_htod(x).unwrap(); if fw { pl.forward(&st, &mut d).unwrap() } else { pl.inverse(&st, &mut d).unwrap() } st.clone_dtoh(&d).unwrap() };
        let rel = |x: &[C32], y: &[C32], sel: &dyn Fn(usize) -> bool| {
            let (mut d, mut s) = (0.0f64, 0.0f64);
            for q in (0..x.len()).filter(|&q| sel(q)) {
                d += ((x[q].re - y[q].re) as f64).powi(2) + ((x[q].im - y[q].im) as f64).powi(2);
                s += (y[q].re as f64).powi(2) + (y[q].im as f64).powi(2);
            }
            (d / s).sqrt()
        };
        let ef = rel(&run(&p, &host, true), &run(&r, &host, true), &sup);
        let ei = rel(&run(&p, &spec, false), &run(&r, &spec, false), &blk);
        let b = match store { Store::Bfp(b) => b, _ => 32 };
        let q = (2.0f64).powi(-(b as i32));
        format!(", error vs f32: forward {ef:.2e} ({:.2} · 2^-b), inverse {ei:.2e} ({:.2} · 2^-b)", ef / q, ei / q)
    } else {
        String::new()
    };
    println!(
        "{desc}: forward {tf:.3} ms, inverse {ti:.3} ms, pair {:.3} ms; BFP buffer {:.3} GB (grid {:.3} GB){err}",
        tf + ti,
        p.store_bytes() as f64 / 1e9,
        (p.len() * 8) as f64 / 1e9
    );
}
