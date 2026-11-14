//! GPU (kernels only; the toy objective is host arithmetic): RED (`hasty_recon::red::Red`) reaches the stationary point
//! of f + λθ·envelope, with the same answer for every preconditioner (identity, plain and per-set ZMSHR1, the metric
//! step, ZMSHR1 in the metric) and on the device and host loops.
//!
//! Toy: f(x) = ½ Σ |a_i x_i − b_i|² (a_i real), D = complex soft threshold at θ (the prox of θ|·|). Stationary point
//! per element, c = a b: x = c / (a² + λ) where |c| ≤ θ (a² + λ), else x = (|c| − λθ)/a² · c/|c|.
//!   cargo test --release -p hasty-recon --test red -- --nocapture

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_recon::gpu::Kernels;
use hasty_recon::hoststack::{run_objective_host, HostStack};
use hasty_recon::p2np::{run_objective, Objective, Precond, Stack};
use hasty_recon::red::{metric_gain, Red};
use hasty_recon::Error;
use hasty_testkit::Rng;

struct Toy {
    kern: Arc<Kernels>,
    lens: Vec<usize>,
    a2: Vec<f32>,
    ab: Vec<C32>,
    bb: f64,
    /// diagonal metric ≈ inverse Hessian of f
    m: Vec<f32>,
}

impl Toy {
    fn grad(&self, x: &[C32], g: &mut [C32]) -> f64 {
        let mut f = self.bb;
        for i in 0..x.len() {
            let (a2, c) = (self.a2[i], self.ab[i]);
            g[i] = C32::new(a2 * x[i].re - c.re, a2 * x[i].im - c.im);
            f += (a2 * (x[i].re * x[i].re + x[i].im * x[i].im) - 2.0 * (c.re * x[i].re + c.im * x[i].im)) as f64;
        }
        0.5 * f
    }
    fn off(&self, t: usize) -> usize {
        self.lens[..t].iter().sum()
    }
}

impl Objective for Toy {
    fn kernels(&self) -> &Kernels {
        &self.kern
    }
    fn lens(&self) -> Vec<usize> {
        self.lens.clone()
    }
    fn gradient(&mut self, st: &Arc<CudaStream>, x: &Stack, g: &mut Stack) -> Result<f64, Error> {
        let xh: Vec<C32> = x.v.iter().flat_map(|v| st.clone_dtoh(v).unwrap()).collect();
        let mut gh = vec![C32::new(0.0, 0.0); xh.len()];
        let f = self.grad(&xh, &mut gh);
        for t in 0..self.lens.len() {
            let o = self.off(t);
            st.memcpy_htod(&gh[o..o + self.lens[t]], &mut g.v[t])?;
        }
        Ok(f)
    }
    fn gradient_host(&mut self, _st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
        Ok(self.grad(&x.data, &mut g.data))
    }
    fn metric_host(&mut self, _st: &Arc<CudaStream>, t: usize, r: &[C32], z: &mut [C32], inverse: bool) -> Result<(), Error> {
        let o = self.off(t);
        for i in 0..r.len() {
            let m = if inverse { 1.0 / self.m[o + i] } else { self.m[o + i] };
            z[i] = C32::new(m * r[i].re, m * r[i].im);
        }
        Ok(())
    }
    fn metric(&mut self, st: &Arc<CudaStream>, t: usize, r: &CudaSlice<C32>, z: &mut CudaSlice<C32>, inverse: bool) -> Result<(), Error> {
        let rh = st.clone_dtoh(r)?;
        let mut zh = vec![C32::new(0.0, 0.0); rh.len()];
        self.metric_host(st, t, &rh, &mut zh, inverse)?;
        st.memcpy_htod(&zh, z)?;
        Ok(())
    }
}

fn soft(x: &mut [C32], th: f32) {
    for v in x.iter_mut() {
        let m = (v.re * v.re + v.im * v.im).sqrt();
        let w = if m > th { 1.0 - th / m } else { 0.0 };
        *v = C32::new(w * v.re, w * v.im);
    }
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n).sqrt()
}

#[test]
fn red_reaches_the_envelope_minimiser_for_every_preconditioner() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let lens = vec![700usize, 500, 600];
    let n: usize = lens.iter().sum();
    let mut rng = Rng::new(7);
    let a: Vec<f32> = (0..n).map(|_| 1.25 + 0.75 * rng.uniform() as f32).collect(); // [0.5, 2)
    let b: Vec<C32> = rng.complex_vec(n).iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect();
    let (lam, th) = (0.8f32, 0.25f32);
    let a2: Vec<f32> = a.iter().map(|v| v * v).collect();
    let ab: Vec<C32> = a.iter().zip(&b).map(|(&ai, bi)| C32::new(ai * bi.re, ai * bi.im)).collect();
    let bb: f64 = b.iter().map(|v| (v.re * v.re + v.im * v.im) as f64).sum();
    let xs: Vec<C32> = (0..n)
        .map(|i| {
            let c = ab[i];
            let mc = (c.re * c.re + c.im * c.im).sqrt();
            if mc <= th * (a2[i] + lam) {
                C32::new(c.re / (a2[i] + lam), c.im / (a2[i] + lam))
            } else {
                let s = (mc - lam * th) / (a2[i] * mc);
                C32::new(s * c.re, s * c.im)
            }
        })
        .collect();
    let in_band = (0..n).filter(|&i| (xs[i].re * xs[i].re + xs[i].im * xs[i].im).sqrt() <= th).count();
    println!("{n} elements, {in_band} at the envelope's quadratic part (|x| ≤ θ)");
    let lmax = a2.iter().fold(0f32, |m, &v| m.max(v)) as f64;
    let m: Vec<f32> = a2.iter().map(|v| 1.0 / v).collect();
    let mut toy = Toy { kern, lens: lens.clone(), a2, ab, bb, m };
    // the metric's mean gain: exact for a diagonal metric on ±1 vectors (mean of 1/a²)
    let mu = metric_gain(&mut toy, &st, &[0, 1, 2]).unwrap();
    let mu_exact = toy.m.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    assert!((mu - mu_exact).abs() < 1e-6 * mu_exact, "metric gain {mu} vs {mu_exact}");
    let cases: [(Precond, f64, usize); 5] = [
        (Precond::None, 1.0 / (lmax + lam as f64), 400),
        (Precond::Zmshr1, 1.0 / (lmax + lam as f64), 200),
        (Precond::Zmshr1Set, 1.0 / (lmax + lam as f64), 200),
        // metric M = 1/a² ≈ H⁻¹ of f: α = 1 / λ_max(M H) with H = a² + λ
        (Precond::Circulant, 1.0 / (1.0 + lam as f64 * 4.0), 200),
        (Precond::Zmshr1MSet, 1.0 / (1.0 + lam as f64 * 4.0), 100),
    ];
    for (pc, alpha, iters) in cases {
        let mut x = HostStack::zeros(&lens);
        let mut den = |_k: usize, h: &mut Vec<C32>| soft(h, th);
        let mut red = Red::new(&mut toy, &mut den, lam as f64);
        red.log_every = 0;
        let mut errs = Vec::new();
        let mut rep = |k: usize, xh: &HostStack, _f: f64| {
            if k != usize::MAX && (k + 1) % 25 == 0 {
                errs.push(rel(&xh.data, &xs));
            }
        };
        run_objective_host(&mut red, &st, &mut x, pc, alpha, iters, &mut |_, _| {}, &mut rep, None).unwrap();
        let e = rel(&x.data, &xs);
        println!("host {pc:?}: {iters} it, rel error {e:.2e} (every 25: {})", errs.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" "));
        assert!(e < 2e-5, "{pc:?}: rel error {e:.3e}");
    }
    // the device loop: same objective, same answer
    let mut xd = Stack::zeros_lens(&st, &lens).unwrap();
    let mut den = |_k: usize, h: &mut Vec<C32>| soft(h, th);
    let mut red = Red::new(&mut toy, &mut den, lam as f64);
    red.log_every = 0;
    run_objective(&mut red, &st, &mut xd, Precond::Zmshr1, 1.0 / (lmax + lam as f64), 200, &mut |_, _| {}, &mut |_, _, _| {}, None).unwrap();
    let xh: Vec<C32> = xd.v.iter().flat_map(|v| st.clone_dtoh(v).unwrap()).collect();
    let e = rel(&xh, &xs);
    println!("device Zmshr1: rel error {e:.2e}");
    assert!(e < 2e-5, "device: rel error {e:.3e}");
    // P²nP with the same denoiser and the identity step stops elsewhere: its fixed point depends on α (the contrast
    // the RED form removes): x = soft(x − α∇f(x), θ) ⇔ a²x − c + (1/α)(x − soft) ∋ 0
    let mut x = HostStack::zeros(&lens);
    let alpha = 1.0 / (lmax + lam as f64);
    let mut pnp = |_k: usize, h: &mut Vec<C32>| soft(h, th);
    run_objective_host(&mut toy, &st, &mut x, Precond::None, alpha, 400, &mut pnp, &mut |_, _, _| {}, None).unwrap();
    println!("P²nP (identity, α {alpha:.3}): distance to the RED point {:.2e}", rel(&x.data, &xs));
}
