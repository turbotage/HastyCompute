//! Per-frame image SNR of the synthetic data as a function of the noise level: one (frame, encode) set reconstructed
//! from the noise-free data and from pure whitened noise (PCGLS + circulant preconditioner, no prior), SNR = mean
//! |x_clean| / std(x_noise) over the lumen (blood) and over the static tissue.
//!   snrcheck <sim_dir> [--frame f] [--iters n] [--vcoils n]
//! Whitening (Ψ^{-1/2}) and the unitary coil compression leave the noise white, σ² per complex sample, so the SNR at
//! any σ follows from one noise realisation at σ = 1.

use std::path::PathBuf;

use hasty_cuda::{Device, C32};
use hasty_recon::cg::pcgls;
use hasty_recon::data::SimData;
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_recon::precond::CirculantPrecond;
use nalgebra::DMatrix;
use num_complex::Complex64;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let frame: usize = take("--frame").map(|s| s.parse().unwrap()).unwrap_or(3);
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(30);
    let nvc: usize = take("--vcoils").map(|s| s.parse().unwrap()).unwrap_or(12);
    // noise-free data
    std::env::set_var("HASTY_NOISE_SIGMA", "0");
    let sd = SimData::load(&PathBuf::from(&args[0])).unwrap();
    let (nv, nc) = (sd.nv(), sd.nc);
    let w = sd.whitener();
    let set = sd.sample_set(&sd.readouts_of(0, Some(frame)), &w);
    let m = set.tau.len();
    // coil compression (unitary rows) from the data
    let a = DMatrix::<Complex64>::from_fn(nc, m.min(20000), |c, j| {
        let v = set.y[c * m + j * (m / m.min(20000))];
        Complex64::new(v.re as f64, v.im as f64)
    });
    let u = (&a * a.adjoint()).svd(true, false).u.unwrap();
    let nvc = nvc.min(nc);
    let cc = |x: &[C32], len: usize| -> Vec<C32> {
        let mut out = vec![C32::new(0.0, 0.0); nvc * len];
        for v in 0..nvc {
            for i in 0..len {
                let z: Complex64 = (0..nc).map(|c| u[(c, v)].conj() * Complex64::new(x[c * len + i].re as f64, x[c * len + i].im as f64)).sum();
                out[v * len + i] = C32::new(z.re as f32, z.im as f32);
            }
        }
        out
    };
    let grid = sd.physics_grid(false, &w);
    let smaps = cc(&grid.smaps, nv);
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let mut op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nvc, vec![], 1e-4).unwrap();
    let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), sd.dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nvc, 1e-3).unwrap();
    let mut recon = |y: &[C32]| -> Vec<C32> {
        let dy = st.clone_htod(y).unwrap();
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        pcgls(&mut op, Some(&mut pre), &st, &dy, &mut x, 0.0, iters).unwrap();
        st.clone_dtoh(&x).unwrap()
    };
    let xc = recon(&cc(&set.y, m));
    let mut seed = 0x1234_5678u64;
    let mut gauss = || {
        let mut u = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
        (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
    };
    let noise: Vec<C32> = (0..nvc * m).map(|_| C32::new((gauss() / 2f64.sqrt()) as f32, (gauss() / 2f64.sqrt()) as f32)).collect();
    let xn = recon(&noise);
    let stats = |sel: &dyn Fn(usize) -> bool| -> (f64, f64, usize) {
        let (mut sig, mut n2, mut n) = (0.0, 0.0, 0usize);
        for q in (0..nv).filter(|&q| sel(q)) {
            sig += (xc[q].re as f64).hypot(xc[q].im as f64);
            n2 += (xn[q].re as f64).powi(2) + (xn[q].im as f64).powi(2);
            n += 1;
        }
        (sig / n as f64, (n2 / n as f64).sqrt(), n)
    };
    let (sb, nb, kb) = stats(&|q| sd.truth_frac[q] > 0.9);
    let mag: Vec<f64> = xc.iter().map(|c| (c.re as f64).hypot(c.im as f64)).collect();
    let mut sorted = mag.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = 0.3 * sorted[sorted.len() * 99 / 100];
    let (st_, nt, kt) = stats(&|q| sd.truth_frac[q] == 0.0 && mag[q] > thr);
    let sim = sd.meta["noise_sigma"].as_f64().unwrap();
    println!("frame {frame}, encode 0: {} readouts, {} PCGLS iterations, {nvc} virtual coils", set.readouts.len(), iters);
    println!("  lumen ({kb} voxels): mean |x| {sb:.4e}, noise std per unit σ {nb:.4e}  → SNR = {:.1} / σ", sb / nb);
    println!("  tissue ({kt} voxels): mean |x| {st_:.4e}, noise std per unit σ {nt:.4e}  → SNR = {:.1} / σ", st_ / nt);
    println!("  simulated σ = {sim:.3}: blood SNR {:.1}, tissue SNR {:.1}", sb / nb / sim, st_ / nt / sim);
    for target in [10.0, 20.0, 30.0] {
        println!("  blood SNR {target:>4}: σ = {:.3} ({:.2} × simulated)", sb / nb / target, sb / nb / target / sim);
    }
}
