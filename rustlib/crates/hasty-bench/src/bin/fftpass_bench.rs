//! Grid-FFT pass bandwidth (research/sample_kernels.md §18): the block-pruned channel-interleaved transform of the
//! SegNufft engines (`hasty_fft::PrunedChanFft3`, image block N inside grid n, nch channels, no k-support pruning),
//! forward and inverse, with or without the two-pass L2 slabs; median time and the DRAM bytes of the model
//! (x pass: R N0·N1·N2, W n0·N1·N2; y: R n0·N1·N2, W n0·n1·N2; z: R n0·n1·N2, W n0·n1·n2; slabs: the x → y
//! intermediate stays in L2), as a fraction of `peak` GB/s. Run under nsys for per-pass times.
//!   fftpass_bench n0 n1 n2 N0 N1 N2 nch [slab planes, default 0] [reps 5] [peak 1008]
use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_fft::PrunedChanFft3;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let u = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let n = [u(0, 240), u(1, 240), u(2, 240)];
    let nn = [u(3, 192), u(4, 192), u(5, 192)];
    let nch = u(6, 16);
    let slab = u(7, 0);
    let reps = u(8, 5);
    let peak = a.get(9).and_then(|v| v.parse::<f64>().ok()).unwrap_or(1008.0) * 1e9;
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let keep = [2, 1, 0].map(|k| (nn[k] - nn[k] / 2, nn[k] / 2));
    let mut p = PrunedChanFft3::<f32>::new(&dev, [n[2], n[1], n[0]], nch, keep, 1.0).unwrap();
    p.set_slab(slab);
    let mut buf = st.alloc_zeros::<C32>(p.len()).unwrap();
    let f = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let ev = (dev.ctx.new_event(f).unwrap(), dev.ctx.new_event(f).unwrap());
    let mut time = |fw: bool| -> f64 {
        let mut t = Vec::new();
        for i in 0..=reps {
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
    let blk = (nn[0] * nn[1] * nn[2]) as f64;
    let (xo, yo, full) = ((n[0] * nn[1] * nn[2]) as f64, (n[0] * n[1] * nn[2]) as f64, (n[0] * n[1] * n[2]) as f64);
    let el = blk + if slab > 0 { 0.0 } else { 2.0 * xo } + yo + yo + full;
    let bytes = el * nch as f64 * 8.0;
    println!(
        "grid {n:?} block {nn:?} nch {nch} slab {slab} (plane {:.1} MB): forward {tf:.3} ms ({:.0} % of {:.0} GB/s), inverse {ti:.3} ms ({:.0} %), model {:.2} GB per transform",
        p.plane_bytes() as f64 / 1e6,
        100.0 * bytes / (tf * 1e-3) / peak,
        peak / 1e9,
        100.0 * bytes / (ti * 1e-3) / peak,
        bytes / 1e9
    );
}
