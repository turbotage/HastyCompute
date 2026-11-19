//! Kernel variants of the knot-major tile kernels (`TileCfg::kv`, segnufft_knot.cu, research/temporal_operator.md §17)
//! against the original kernels (variant 0): same operator up to f32 summation order, for every warp layout of the
//! adjoint's sub-box warps (coil groups 1, 2, 4, 8: 32-, 16- and 8-point warps), forward lanes (1, 2, 4 per sample),
//! sample blocking, orders 0–2 (real and complex weights), even and odd segment counts (float4 and scalar loads),
//! 1-D cubic knots and 2-D linear respiratory × cubic cardiac knots, split tile lists; plus adjointness per variant.
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};

struct R(u64);
impl R {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2)).sum();
    let d: f64 = b.iter().map(|y| (y.re as f64).powi(2) + (y.im as f64).powi(2)).sum();
    (n / d.max(1e-30)).sqrt()
}

fn dot(a: &[C32], b: &[C32]) -> num_complex::Complex64 {
    a.iter().zip(b).map(|(x, y)| num_complex::Complex64::new(x.re as f64, x.im as f64).conj() * num_complex::Complex64::new(y.re as f64, y.im as f64)).sum()
}

/// `kr` = 0: 1-D knots (`new_temporal`), else 2-D with kr linear respiratory knots
// bounds as tests/seg_knot.rs (knot- vs interval-major: f32 summation order, here also the sample order of box_sort)
fn case(nseg: usize, order: u32, kr: usize, m: usize) {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [24usize, 20, 16];
    let nv: usize = dims.iter().product();
    let (nc, mc) = (8, 6);
    let mut r = R(9001 + nseg as u64 * 31 + order as u64 * 7 + kr as u64);
    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (r.u() * 6.2) as f32).collect());
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 20.0) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
    let nri = kr.saturating_sub(1).max(1);
    let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * nri as f64) as u32).min(nri as u32 - 1)).collect();
    let rv: Vec<f32> = (0..m).map(|_| (r.u() + 0.5) as f32).collect();
    let nimg = mc * kr.max(1);
    let x: Vec<C32> = (0..nimg * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yd: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    let opts = SegNufftOpts { sigma: 1.25, width: 8, nseg, order, frames: mc, bspline: 3, max_grid_bytes: 1 << 30 };
    let mut op = if kr > 0 { SegNufft::new_temporal2d(&dev, dims, [0.004; 3], nc, opts, kr, 1).unwrap() } else { SegNufft::new_temporal(&dev, dims, [0.004; 3], nc, opts, true).unwrap() };
    assert_eq!(op.cb, nc, "one coil batch of all coils (coil groups up to 8)");
    op.set_maps(&st, &smaps).unwrap();
    let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
    let pts = if kr > 0 { op.points2d(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, &t, &ri, &rv).unwrap() } else { op.points(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, Some(&t)).unwrap() };
    let (dx, dy) = (st.clone_htod(&x).unwrap(), st.clone_htod(&yd).unwrap());
    let run = |op: &mut SegNufft| {
        let mut y = st.alloc_zeros::<C32>(nc * m).unwrap();
        let mut xa = st.alloc_zeros::<C32>(nimg * nv).unwrap();
        op.forward(&st, &seg, &pts, &dx, &mut y).unwrap();
        op.adjoint(&st, &seg, &pts, &dy, &mut xa).unwrap();
        (st.clone_dtoh(&y).unwrap(), st.clone_dtoh(&xa).unwrap())
    };
    let base = op.tile_shape(4, 4, 2, 1, 1, 0).unwrap();
    op.set_tiles(base, base).unwrap();
    let (y0, x0) = run(&mut op);
    let mut n = 0;
    for kv in [1usize, 2] {
        for (cpba, cpbf, ppf, fsb) in [(1, 1, 2, 1), (2, 2, 1, 1), (4, 4, 2, 1), (8, 8, 2, 2), (4, 2, 4, 2), (8, 4, 1, 1), (2, 8, 4, 1)] {
            let (Some(tf), Some(ta)) = (op.tile_shape(cpba, cpbf, ppf, 1, fsb, kv), op.tile_shape(cpba, cpbf, ppf, 1, fsb, kv)) else { continue };
            op.set_tiles(tf, ta).unwrap();
            let (y, xa) = run(&mut op);
            let (ef, ea) = (rel(&y, &y0), rel(&xa, &x0));
            let e = (dot(&yd, &y) - dot(&xa, &x)).norm() / (dot(&yd, &yd).re * dot(&y, &y).re).sqrt();
            eprintln!("nseg {nseg} order {order} kr {kr} m {m}: kv {kv} cpba {cpba} cpbf {cpbf} ppf {ppf} fsb {fsb} (chunk {}): vs kv 0 forward {ef:.1e}, adjoint {ea:.1e}, adjointness {e:.1e}", tf.chunk);
            assert!(ef < 3e-5 && ea < 6e-5 && e < 2e-6, "kv {kv} {tf:?}: forward {ef:.1e}, adjoint {ea:.1e}, adjointness {e:.1e}");
            n += 1;
        }
    }
    // variant 1 with adjoint coil register blocking (acb coils per thread; research/sample_kernels.md)
    for (cpba, acb) in [(2usize, 2usize), (4, 2), (4, 4), (8, 2), (8, 4), (8, 8)] {
        let Some(ta) = op.tile_shape(cpba, 4, 2, acb, 1, 1) else { continue };
        assert_eq!(ta.acb, acb);
        let tf = op.tile_shape(4, 4, 2, 1, 1, 1).unwrap();
        op.set_tiles(tf, ta).unwrap();
        let (y, xa) = run(&mut op);
        let (ef, ea) = (rel(&y, &y0), rel(&xa, &x0));
        let e = (dot(&yd, &y) - dot(&xa, &x)).norm() / (dot(&yd, &yd).re * dot(&y, &y).re).sqrt();
        eprintln!("nseg {nseg} order {order} kr {kr} m {m}: kv 1 cpba {cpba} acb {acb}: vs kv 0 forward {ef:.1e}, adjoint {ea:.1e}, adjointness {e:.1e}");
        assert!(ef < 3e-5 && ea < 6e-5 && e < 2e-6, "kv 1 acb {acb} {ta:?}: forward {ef:.1e}, adjoint {ea:.1e}, adjointness {e:.1e}");
        n += 1;
    }
    assert!(n >= 11, "only {n} variant shapes ran");
}

#[test]
fn kv_1d_cubic_orders() {
    case(4, 0, 0, 3000);
    case(5, 1, 0, 3000);   // odd segment count: scalar loads
    case(4, 2, 0, 3000);
}

#[test]
fn kv_2d_linear_cubic() {
    case(4, 0, 3, 4000);
    case(3, 2, 3, 4000);
}

#[test]
fn kv_split_lists() {
    // ~40 k samples on a 30 × 25 × 20 fine grid: tile lists above the 1024-sample item limit (atomic items)
    case(4, 0, 0, 40_000);
    case(4, 2, 3, 40_000);
}
