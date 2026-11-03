//! CG on self-consistent data y = E x_true (x_true = truth lumen fraction) with the operator of encode 0.
//!   cgcheck <sim_dir> [iters]
use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_recon::cg::{cg, pcg, pcgls, power_norm};
use hasty_recon::precond::CirculantPrecond;
use hasty_recon::data::SimData;
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use num_complex::Complex64;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let iters: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(30);
    let sd = SimData::load(&dir).unwrap();
    let w = sd.whitener();
    let nv = sd.nv();
    let set = sd.sample_set(&sd.readouts_of(0, None), &w);
    let mut grid = sd.physics_grid(false, &w);
    let single = std::env::var("SINGLE").is_ok();
    let nc_used = if single { 1 } else { sd.nc };
    if single {
        grid.smaps = vec![C32::new(1.0, 0.0); nv];
    }
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let mut op = SenseOp::new(&dev, &st, kern.clone(), sd.dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &grid.smaps, nc_used, vec![], 1e-5).unwrap();
    let xt: Vec<C32> = sd.truth_frac.iter().map(|&f| C32::new(f, 0.0)).collect();
    let d_xt = st.clone_htod(&xt).unwrap();
    let mut y = st.alloc_zeros::<C32>(nc_used * set.tau.len()).unwrap();
    op.forward(&st, &d_xt, &mut y).unwrap();
    let mut rhs = st.alloc_zeros::<C32>(nv).unwrap();
    op.adjoint(&st, &y, &mut rhs).unwrap();
    {
        // adjointness at full size
        let m = set.tau.len();
        let mut r = 12345u64;
        let mut u = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; ((r >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 };
        let xr: Vec<C32> = (0..nv).map(|_| C32::new(u(), u())).collect();
        let yr: Vec<C32> = (0..nc_used * m).map(|_| C32::new(u(), u())).collect();
        let dx = st.clone_htod(&xr).unwrap();
        let dy = st.clone_htod(&yr).unwrap();
        let mut ex = st.alloc_zeros::<C32>(nc_used * m).unwrap();
        let mut ehy = st.alloc_zeros::<C32>(nv).unwrap();
        op.forward(&st, &dx, &mut ex).unwrap();
        op.adjoint(&st, &dy, &mut ehy).unwrap();
        let a = kern.dot(&st, &dy, &ex).unwrap();
        let b = kern.dot(&st, &ehy, &dx).unwrap();
        println!("full-size adjointness: <y,Ex> = {a:.6e}, <E^H y,x> = {b:.6e}, rel {:.2e}", (a - b).norm() / a.norm());
        let mut nx = st.alloc_zeros::<C32>(nv).unwrap();
        op.normal(&st, &dx, &mut nx).unwrap();
        println!("<x,Nx> = {:.6e}", kern.dot(&st, &dx, &nx).unwrap());
    }
    let lmax = power_norm(&mut op, &st, 10).unwrap();
    let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), sd.dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &grid.smaps, nc_used, 1e-3).unwrap();
    println!("preconditioner dynamic range {:.2e}", pre.dyn_range);
    {
        let mut r = 4242u64;
        let mut u = || { r ^= r << 13; r ^= r >> 7; r ^= r << 17; ((r >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 };
        let a: Vec<C32> = (0..nv).map(|_| C32::new(u(), u())).collect();
        let b: Vec<C32> = (0..nv).map(|_| C32::new(u(), u())).collect();
        let (da, db) = (st.clone_htod(&a).unwrap(), st.clone_htod(&b).unwrap());
        let (mut ma, mut mb) = (st.alloc_zeros::<C32>(nv).unwrap(), st.alloc_zeros::<C32>(nv).unwrap());
        pre.apply(&st, &da, &mut ma).unwrap();
        pre.apply(&st, &db, &mut mb).unwrap();
        let x1 = kern.dot(&st, &db, &ma).unwrap(); // <b, M a>
        let x2 = kern.dot(&st, &mb, &da).unwrap(); // <M b, a>
        let q = kern.dot(&st, &da, &ma).unwrap();
        println!("M Hermitian check: <b,Ma> = {x1:.5e}, <Mb,a> = {x2:.5e}; <a,Ma> = {q:.5e}");
    }
    for (lam, mode) in [(0.0, 0), (0.0, 2), (0.0, 3)] {
        let use_pre = mode == 1;
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        let hist = match mode {
            0 => cg(&mut op, &st, &rhs, &mut x, lam * lmax, iters).unwrap(),
            1 => pcg(&mut op, &mut pre, &st, &rhs, &mut x, lam * lmax, iters).unwrap(),
            2 => pcgls(&mut op, None, &st, &y, &mut x, lam * lmax, iters).unwrap(),
            _ => pcgls(&mut op, Some(&mut pre), &st, &y, &mut x, lam * lmax, iters).unwrap(),
        };
        print!("{} ", ["CG   ", "PCG  ", "CGLS ", "PCGLS"][mode]);
        let h = st.clone_dtoh(&x).unwrap();
        let (mut e, mut n) = (0.0, 0.0);
        for q in 0..nv {
            let d = Complex64::new((h[q].re - xt[q].re) as f64, (h[q].im - xt[q].im) as f64);
            e += d.norm_sqr();
            n += (xt[q].re as f64).powi(2);
        }
        // error restricted to well-sampled Fourier modes (λ > 1e-3 λ_max) of the N grid
        let fwd = hasty_fft::FftPlan::<f32>::new(&dev, &[sd.dims[2], sd.dims[1], sd.dims[0]], hasty_fft::Direction::Forward, hasty_fft::Norm::None).unwrap();
        let diff: Vec<C32> = (0..nv).map(|q| C32::new(h[q].re - xt[q].re, h[q].im - xt[q].im)).collect();
        if !use_pre {
            let bytes: Vec<u8> = h.iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat()).collect();
            hasty_phantom::volume::write_npy(&dir.join("cgcheck_x.npy"), "<c8", &sd.dims, true, &bytes).unwrap();
        }
        let mut dd = st.clone_htod(&diff).unwrap();
        fwd.execute_inplace(&st, &mut dd).unwrap();
        let mut dt = st.clone_htod(&xt).unwrap();
        fwd.execute_inplace(&st, &mut dt).unwrap();
        let (fd, ft) = (st.clone_dtoh(&dd).unwrap(), st.clone_dtoh(&dt).unwrap());
        let lmx = pre.lambda.iter().cloned().fold(0f32, f32::max);
        let (mut es, mut ns, mut ntot) = (0.0, 0.0, 0.0);
        for q in 0..nv {
            let t = (ft[q].re as f64).powi(2) + (ft[q].im as f64).powi(2);
            ntot += t;
            if pre.lambda[q] > 1e-3 * lmx {
                es += (fd[q].re as f64).powi(2) + (fd[q].im as f64).powi(2);
                ns += t;
            }
        }
        {
            let de = st.clone_htod(&diff).unwrap();
            let mut ne = st.alloc_zeros::<C32>(nv).unwrap();
            op.normal(&st, &de, &mut ne).unwrap();
            let ee = kern.dot(&st, &de, &de).unwrap().re;
            let ene = kern.dot(&st, &de, &ne).unwrap().re;
            let mut nxt = st.alloc_zeros::<C32>(nv).unwrap();
            op.normal(&st, &d_xt, &mut nxt).unwrap();
            let tt = kern.dot(&st, &d_xt, &d_xt).unwrap().re;
            let tnt = kern.dot(&st, &d_xt, &nxt).unwrap().re;
            print!("[Rayleigh: error {:.3e}, truth {:.3e}; |rhs - N xt|/|rhs| = ", ene / ee, tnt / tt);
            let mut chk = st.clone_dtod(&rhs).unwrap();
            kern.axpby(&st, Complex64::new(-1.0, 0.0), &nxt, Complex64::new(1.0, 0.0), &mut chk).unwrap();
            print!("{:.2e}] ", (kern.dot(&st, &chk, &chk).unwrap().re / kern.dot(&st, &rhs, &rhs).unwrap().re).sqrt());
        }
        print!("[sampled modes hold {:.3} of the truth energy; error there {:.3e}] ", ns / ntot, (es / ns).sqrt());
        println!("lambda {lam:e}·‖N‖ ({:.3e}): residual {:?} ... {:.2e}; image rel err {:.3e}", lmax, &hist[..4].iter().map(|v| format!("{v:.2e}")).collect::<Vec<_>>(), hist.last().unwrap(), (e / n).sqrt());
    }
}
