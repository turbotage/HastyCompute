//! GPU: the knot-space solvers of `knotncg` on a small problem (A = a diagonal, spatially and per-knot varying sample
//! weight, so the metric matters):
//! - `PriorMetric` with σ = 0 and a flat spectrum equals `KnotPrecond` (G⁻¹ ⊗ I); P⁻¹ P = I with prior terms;
//! - NCG without priors reaches the PCGLS solution;
//! - NCG, FISTA and ZMSHR1 on the smoothed wavelet + LLR objective reach the same minimiser; ADMM decreases the
//!   exact objective and its iterate lies near the smoothed minimiser (γ small).
//!   cargo test --release -p hasty-recon --test knotncg -- --nocapture
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_recon::gpu::Kernels;
use hasty_recon::knotncg::{self, hdot, Info, LlrBackend, LlrHuber, PriorMetric, Priors, WavHuber};
use hasty_recon::llr::Llr;
use hasty_recon::setop::LinOp;
use hasty_recon::Error;

struct DiagOp {
    w: CudaSlice<C32>,
    kern: Arc<Kernels>,
    n: usize,
}

impl LinOp for DiagOp {
    fn len_x(&self) -> usize {
        self.n
    }
    fn len_y(&self) -> usize {
        self.n
    }
    fn kernels(&self) -> &Kernels {
        &self.kern
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.kern.mul_real(st, x, &self.w, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.kern.mul_real(st, y, &self.w, x)
    }
}

fn rnd(n: usize, seed: u64, amp: f32) -> Vec<C32> {
    let mut s = seed;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    (0..n).map(|_| C32::new(amp * u(), amp * u())).collect()
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let d: Vec<C32> = a.iter().zip(b).map(|(x, y)| C32::new(x.re - y.re, x.im - y.im)).collect();
    (hdot(&d, &d) / hdot(b, b)).sqrt()
}

const D: [usize; 3] = [16, 16, 8];
const K: usize = 4;

struct Setup {
    dev: Device,
    st: Arc<CudaStream>,
    kern: Arc<Kernels>,
    w: Vec<f32>,
    y: Vec<C32>,
    gram: Vec<f64>,
}

fn setup() -> Setup {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let nv: usize = D.iter().product();
    let n = nv * K;
    // sample weight: smooth in space × per-knot scale (as diag G), range ≈ 30
    let w: Vec<f32> = (0..n).map(|i| { let (q, t) = (i % nv, i / nv); let x = (q % 16) as f32; (0.3 + 1.5 * (1.0 + (x * 0.4).sin())) * (0.5 + 0.5 * t as f32) }).collect();
    // truth: smooth, slowly varying over the knots; data = w x + noise
    let xt: Vec<C32> = (0..n).map(|i| { let (q, t) = (i % nv, i / nv); let (x, y) = ((q % 16) as f32, ((q / 16) % 16) as f32); let v = (x * 0.35).cos() * (y * 0.25).sin() * (1.0 + 0.2 * t as f32); C32::new(v, 0.4 * v) }).collect();
    let nz = rnd(n, 99, 0.3);
    let y: Vec<C32> = (0..n).map(|i| C32::new(w[i] * xt[i].re + nz[i].re, w[i] * xt[i].im + nz[i].im)).collect();
    // a Gram with off-diagonal coupling (as linear / cubic bases): G = tridiag
    let mut gram = vec![0.0; K * K];
    for a in 0..K {
        gram[a * K + a] = 0.6 + 0.1 * a as f64;
        if a + 1 < K {
            gram[a * K + a + 1] = 0.15;
            gram[(a + 1) * K + a] = 0.15;
        }
    }
    Setup { dev, st, kern, w, y, gram }
}

fn metric(s: &Setup, c: f32) -> PriorMetric {
    let nv: usize = D.iter().product();
    PriorMetric::new(&s.dev, &s.st, K, D, &s.gram, 1e-3, vec![c; nv], vec![1.0; nv]).unwrap()
}

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn metric_matches_kronecker_and_inverts() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let s = setup();
    let nv: usize = D.iter().product();
    let n = nv * K;
    let mut pm = metric(&s, 1.0);
    let gi = hasty_recon::knotsolve::gram_inverse(&s.gram, K, 1e-3);
    let mut kp = hasty_recon::knotsolve::KnotPrecond::new(&s.dev, &s.st, K, nv, &gi, None).unwrap();
    let r = s.st.clone_htod(&rnd(n, 5, 1.0)).unwrap();
    let mut z1 = s.st.alloc_zeros::<C32>(n).unwrap();
    let mut z2 = s.st.alloc_zeros::<C32>(n).unwrap();
    use hasty_recon::cg::Precond;
    kp.apply(&s.st, &r, &mut z1).unwrap();
    pm.apply_metric(&s.st, &r, &mut z2).unwrap();
    let (a, b) = (s.st.clone_dtoh(&z1).unwrap(), s.st.clone_dtoh(&z2).unwrap());
    let e = rel(&b, &a);
    println!("PriorMetric(σ = 0) vs KnotPrecond: rel {e:.2e}");
    assert!(e < 1e-5, "{e}");
    // with prior terms and a non-flat spectrum / coil scale: P⁻¹ P = I
    let lam: Vec<f32> = (0..nv).map(|q| 0.05 + (q % 7) as f32 * 0.3).collect();
    let dinv: Vec<f32> = (0..nv).map(|q| 0.5 + (q % 5) as f32 * 0.2).collect();
    let mut pm = PriorMetric::new(&s.dev, &s.st, K, D, &s.gram, 1e-3, lam, dinv).unwrap();
    let sw: Vec<f32> = (0..nv).map(|q| (q % 3) as f32 * 0.7).collect();
    pm.set_prior(&s.st, &[0.1, 0.5, 1.0, 2.0], Some(&sw)).unwrap();
    pm.apply_metric(&s.st, &r, &mut z1).unwrap();
    pm.apply_inv(&s.st, &z1, &mut z2).unwrap();
    let e = rel(&s.st.clone_dtoh(&z2).unwrap(), &s.st.clone_dtoh(&r).unwrap());
    println!("P⁻¹ P r vs r: rel {e:.2e}");
    assert!(e < 1e-5, "{e}");
}

#[test]
fn ncg_reaches_cg_solution() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let s = setup();
    let nv: usize = D.iter().product();
    let n = nv * K;
    let wd = s.st.clone_htod(&s.w.iter().map(|&v| C32::new(v, 0.0)).collect::<Vec<_>>()).unwrap();
    let mut op = DiagOp { w: wd, kern: s.kern.clone(), n };
    let mut pm = metric(&s, 1.0);
    let y = s.st.clone_htod(&s.y).unwrap();
    // reference: PCGLS in the same metric, 60 iterations from 0
    let mut xr = s.st.alloc_zeros::<C32>(n).unwrap();
    hasty_recon::cg::pcgls_dyn(&mut op, Some(&mut pm as &mut dyn hasty_recon::cg::Precond), &s.st, hasty_recon::cg::Data::Borrowed(&y), &mut xr, 0.0, 60, &mut |_, _| {}, None).unwrap();
    let xr = s.st.clone_dtoh(&xr).unwrap();
    // exact: y / w
    let ex: Vec<C32> = s.y.iter().zip(&s.w).map(|(a, &w)| C32::new(a.re / w, a.im / w)).collect();
    println!("PCGLS vs exact: {:.2e}", rel(&xr, &ex));
    let mut x = s.st.alloc_zeros::<C32>(n).unwrap();
    let r = s.st.clone_dtod(&y).unwrap();
    let mut pri = Priors::new(K, nv, None, None);
    let mut its = 0;
    knotncg::ncg(&mut op, &s.kern, &s.st, &pm, &mut pri, &mut x, r, &mut |i: &Info, _| { its = i.iter; i.iter < 60 }).unwrap();
    let xn = s.st.clone_dtoh(&x).unwrap();
    let e = rel(&xn, &xr);
    println!("NCG ({its} iterations) vs exact: {:.2e}, vs PCGLS (same metric, same iterations): {e:.2e}", rel(&xn, &ex));
    assert!(e < 1e-3, "{e}");
}

fn priors(nv: usize, theta_w: f64, theta_l: f64, gamma: f64) -> Priors {
    let w = WavHuber::new(D, 2, theta_w, gamma);
    let l = Llr::new(D, 4, 1.0, vec![true; nv]);
    Priors::new(K, nv, Some(w), Some(LlrHuber { be: LlrBackend::Cpu(l), theta: theta_l, gamma, shift: 7 }))
}

#[test]
fn smoothed_solvers_agree() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let s = setup();
    let nv: usize = D.iter().product();
    let n = nv * K;
    let wd = s.st.clone_htod(&s.w.iter().map(|&v| C32::new(v, 0.0)).collect::<Vec<_>>()).unwrap();
    let mut op = DiagOp { w: wd, kern: s.kern.clone(), n };
    let y = s.st.clone_htod(&s.y).unwrap();
    // bulk ≈ mean w² · mean diag G; γ = 0.3 / bulk; thresholds of the order of the noise
    let bulk = s.w.iter().map(|&v| (v * v) as f64).sum::<f64>() / n as f64;
    let gamma = 0.3 / bulk;
    let (lw, ll) = (0.5 * bulk * 0.3, 0.5 * bulk * 1.0);
    let mk = || priors(nv, lw * gamma, ll * gamma, gamma);
    let pm = metric(&s, 1.0);
    let fobj = |i: &Info| i.f + i.pw + i.pl;
    // NCG
    let mut x = s.st.alloc_zeros::<C32>(n).unwrap();
    let mut pri = mk();
    let mut hist = vec![];
    knotncg::ncg(&mut op, &s.kern, &s.st, &pm, &mut pri, &mut x, s.st.clone_dtod(&y).unwrap(), &mut |i: &Info, _| { hist.push(fobj(i)); i.iter < 150 }).unwrap();
    let xn = s.st.clone_dtoh(&x).unwrap();
    let fn_ = *hist.last().unwrap();
    println!("NCG: F {:.6e} → {:.6e} (its 1, 5, 10, 20: {:.6e} {:.6e} {:.6e} {:.6e}); wavelet transforms {}, LLR sweeps {}", hist[0], fn_, hist[1], hist[5], hist[10], hist[20], pri.n_wt, pri.n_llr);
    assert!(hist.windows(2).all(|w| w[1] <= w[0] * (1.0 + 1e-6)), "NCG not monotone: {hist:?}");
    // FISTA in the same metric, L from the power iteration
    let mut pri = mk();
    let x0 = s.st.alloc_zeros::<C32>(n).unwrap();
    let (lip, lh) = knotncg::power_lmax(&mut op, &s.kern, &s.st, &pm, &mut pri, &x0, n, 15).unwrap();
    println!("power iteration λ_max(P∇²F): {:?}", lh.iter().map(|v| (v * 1e3).round() / 1e3).collect::<Vec<_>>());
    let mut x = s.st.alloc_zeros::<C32>(n).unwrap();
    let mut ff = vec![];
    knotncg::fista(&mut op, &s.kern, &s.st, &pm, &mut pri, &mut x, s.st.clone_dtod(&y).unwrap(), 1.05 * lip, &mut |i: &Info, _| { ff.push(fobj(i)); i.iter < 1500 }).unwrap();
    let xf = s.st.clone_dtoh(&x).unwrap();
    println!("FISTA: F → {:.6e} (it 20: {:.6e}, 100: {:.6e}); NCG vs FISTA iterate: {:.2e}", ff.last().unwrap(), ff[20], ff[100], rel(&xf, &xn));
    assert!(rel(&xf, &xn) < 2e-3, "NCG and FISTA disagree: {}", rel(&xf, &xn));
    assert!((ff.last().unwrap() - fn_).abs() < 1e-5 * fn_.abs());
    // ZMSHR1-M
    let mut pri = mk();
    let mut x = s.st.alloc_zeros::<C32>(n).unwrap();
    let mut fz = vec![];
    knotncg::zmshr1(&mut op, &s.kern, &s.st, &pm, &mut pri, &mut x, s.st.clone_dtod(&y).unwrap(), 1.0 / lip, &mut |i: &Info, _| { fz.push(fobj(i)); i.iter < 600 }).unwrap();
    let xz = s.st.clone_dtoh(&x).unwrap();
    println!("ZMSHR1-M: F → {:.6e} (it 20: {:.6e}); vs NCG iterate {:.2e}", fz.last().unwrap(), fz[20], rel(&xz, &xn));
    assert!(rel(&xz, &xn) < 5e-3, "ZMSHR1 and NCG disagree: {}", rel(&xz, &xn));
    // ADMM (exact ℓ1 + nuclear): ridge 2ρ in the metric
    let rho = 0.5 * bulk;
    let mut pma = metric(&s, 1.0);
    pma.set_prior(&s.st, &vec![2.0 * rho; K], None).unwrap();
    let mut pri = mk();
    let mut x = s.st.alloc_zeros::<C32>(n).unwrap();
    knotncg::admm(&mut op, &s.kern, &s.st, &pma, &mut pri, &mut x, s.st.clone_dtod(&y).unwrap(), rho, 3, &mut |i: &Info, _| i.iter < 900).unwrap();
    let xa = s.st.clone_dtoh(&x).unwrap();
    println!("ADMM (exact objective) vs NCG (smoothed, γ = 0.3/bulk): {:.2e}", rel(&xa, &xn));
    assert!(rel(&xa, &xn) < 0.05, "ADMM far from the smoothed minimiser: {}", rel(&xa, &xn));
}

#[test]
fn llr_envelope_gpu_matches_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let nv: usize = D.iter().product();
    let n = nv * K;
    let x: Vec<C32> = rnd(n, 3, 2.0).iter().enumerate().map(|(i, z)| { let q = i % nv; C32::new(z.re + ((q % 16) as f32 * 0.3).sin(), z.im) }).collect();
    let basis: Vec<f64> = (0..K * K).map(|q| if q / K == q % K { 1.0 } else { 0.0 }).collect();
    let mut cpu = Llr::new(D, 4, 1.0, vec![true; nv]);
    cpu.zeta_basis = Some(basis.clone());
    let mut gpu = Llr::new(D, 4, 1.0, vec![true; nv]);
    gpu.zeta_basis = Some(basis);
    let mut hc = LlrHuber { be: LlrBackend::Cpu(cpu), theta: 1.2, gamma: 0.5, shift: 2 };
    let mut hg = LlrHuber { be: LlrBackend::Gpu(hasty_recon::llr_gpu::LlrGpu::new(gpu, &[0]).unwrap()), theta: 1.2, gamma: 0.5, shift: 2 };
    let (mut g1, mut g2) = (vec![C32::new(0.0, 0.0); n], vec![C32::new(0.0, 0.0); n]);
    let v1 = hc.eval(&x, &mut g1, K).unwrap();
    let v2 = hg.eval(&x, &mut g2, K).unwrap();
    let (z1, z2) = (hc.be.llr().last.zeta.clone(), hg.be.llr().last.zeta.clone());
    println!("LLR envelope CPU {v1:.6e} GPU {v2:.6e}; gradient rel {:.2e}; ζ {:?} vs {:?}", rel(&g2, &g1), z1, z2);
    assert!((v1 - v2).abs() < 1e-4 * v1.abs());
    assert!(rel(&g2, &g1) < 1e-4);
    assert!(z1.iter().zip(&z2).all(|(a, b)| (a - b).abs() < 1e-4));
}
