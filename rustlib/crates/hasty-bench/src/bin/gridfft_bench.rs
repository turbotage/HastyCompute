//! Fine-grid FFT time per grid size (research/temporal_operator.md §20): the block-pruned channel-interleaved
//! transform of the SegNufft engines, forward + inverse, for the smooth lengths from the σ rule's size up to `grow` ×,
//! one axis at a time, and the separable choice of `hasty_nufft::choose_grid`.
//!   gridfft_bench nx ny nz nch [sigma 1.25] [w 10] [grow 1.3] [kmax 8]
use hasty_cuda::Device;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let u = |i: usize, d: usize| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let f = |i: usize, d: f64| a.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let dims = [u(0, 198), u(1, 189), u(2, 228)];
    let nch = u(3, 44);
    let (sigma, w, grow, kmax) = (f(4, 1.25), u(5, 10), f(6, 1.3), u(7, 8));
    let dev = Device::new(0).unwrap();
    let c = hasty_nufft::choose_grid(&dev, dims, sigma, w, nch, grow, kmax).unwrap();
    println!("image {dims:?}, {nch} channels, σ ≥ {sigma}, W {w}: default grid {:?} (σ {:.3} / {:.3} / {:.3}) forward + inverse {:.2} ms", c.default, c.default[0] as f64 / dims[0] as f64, c.default[1] as f64 / dims[1] as f64, c.default[2] as f64 / dims[2] as f64, c.t_default);
    for ax in 0..3 {
        let row: Vec<String> = c.per_axis[ax].iter().map(|(n, t)| format!("{n}: {t:.2} ms ({:.3}×, σ {:.3})", t / c.t_default, *n as f64 / dims[ax] as f64)).collect();
        println!("  axis {ax}: {}", row.join(", "));
    }
    println!("  chosen {:?}: {:.2} ms ({:.3}× of default)", c.nf, c.t_chosen, c.t_chosen / c.t_default);
}
