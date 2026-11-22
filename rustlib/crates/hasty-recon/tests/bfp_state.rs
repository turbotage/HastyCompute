//! The BFP host solver state (`hasty_recon::bfpstate`) against the f32 host loop on a 2-D multi-coil Cartesian toy
//! (host arithmetic; a CUDA context only for the stream handle the API takes, no GPU work):
//! f(x) = ½ Σ_t ‖M_t F S x_t − y_t‖² + ½ μ ‖x_t‖², 3 sets (frames) of 50 × 38 (not a multiple of 16 values: BFP tail
//! blocks), 4 coils, variable-density masks of different seeds, noise at image SNR ≈ 30.
//! - no prior: every preconditioner, f32 vs bfp16 / bfp12 g + pair; the fixed point is x★ (f64 CG), predicted exact
//!   for BFP g and pair (research/bfp_storage.md §3.7);
//! - RED (soft threshold): the stationary point does not depend on the step; bfp16 against a converged f32 run;
//! - P²nP (soft threshold after the step): the secant record is x_prev (the denoiser changes x), pair width matters;
//! - checkpoint / resume of the BFP state (STOP after iteration 14, resumed) against an uninterrupted run;
//! - a diagonal toy with sets of 150 001 values (several 64 Ki chunks per set: the chunked paths), no prior and P²nP
//!   with a support projection (the denoiser changes x every iteration: x_prev record; the fixed point is known
//!   exactly), a projection switched on late (difference record → invalid pair → x_prev record), and a P²nP resume.
//!   cargo test --release -p hasty-recon --test bfp_state -- --nocapture --test-threads=1
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use hasty_cuda::C32;
use hasty_recon::bfpstate::{run_objective_host_bfp, GradSink, StateStore};
use hasty_recon::checkpoint::{Outcome, RunCtl};
use hasty_recon::gpu::Kernels;
use hasty_recon::hoststack::{run_objective_host, HostStack};
use hasty_recon::p2np::{Objective, Precond, Stack};
use hasty_recon::red::Red;
use hasty_recon::Error;
use hasty_testkit::Rng;
use num_complex::Complex64 as Z;
use rustfft::{Fft, FftPlanner};

const NY: usize = 50;
const NX: usize = 38;
const NC: usize = 4;
const NT: usize = 3;
const N: usize = NY * NX;

struct Fft2 {
    fy: Arc<dyn Fft<f64>>,
    fx: Arc<dyn Fft<f64>>,
    iy: Arc<dyn Fft<f64>>,
    ix: Arc<dyn Fft<f64>>,
}

impl Fft2 {
    fn new() -> Self {
        let mut p = FftPlanner::new();
        Self { fy: p.plan_fft_forward(NY), fx: p.plan_fft_forward(NX), iy: p.plan_fft_inverse(NY), ix: p.plan_fft_inverse(NX) }
    }
    /// unitary 2-D DFT in place (row-major NY × NX)
    fn apply(&self, v: &mut [Z], inverse: bool) {
        let (fx, fy) = if inverse { (&self.ix, &self.iy) } else { (&self.fx, &self.fy) };
        for r in v.chunks_mut(NX) {
            fx.process(r);
        }
        let mut col = vec![Z::new(0.0, 0.0); NY];
        for c in 0..NX {
            for y in 0..NY {
                col[y] = v[y * NX + c];
            }
            fy.process(&mut col);
            for y in 0..NY {
                v[y * NX + c] = col[y];
            }
        }
        let s = 1.0 / (N as f64).sqrt();
        v.iter_mut().for_each(|z| *z *= s);
    }
}

struct Toy {
    fft: Fft2,
    smaps: Vec<Vec<Z>>,
    masks: Vec<Vec<bool>>,
    /// A^H y per set
    ahy: Vec<Vec<Z>>,
    yy: Vec<f64>,
    mu: f64,
    /// diagonal k-space metric w_t(k) ≈ 1 / (curvature at k)
    w: Vec<Vec<f64>>,
}

impl Toy {
    fn normal(&self, t: usize, x: &[Z]) -> Vec<Z> {
        let mut out = vec![Z::new(0.0, 0.0); N];
        for c in 0..NC {
            let mut v: Vec<Z> = x.iter().zip(&self.smaps[c]).map(|(a, s)| a * s).collect();
            self.fft.apply(&mut v, false);
            for (z, &m) in v.iter_mut().zip(&self.masks[t]) {
                if !m {
                    *z = Z::new(0.0, 0.0);
                }
            }
            self.fft.apply(&mut v, true);
            for ((o, a), s) in out.iter_mut().zip(&v).zip(&self.smaps[c]) {
                *o += a * s.conj();
            }
        }
        for (o, a) in out.iter_mut().zip(x) {
            *o += self.mu * a;
        }
        out
    }
    fn grad_set(&self, t: usize, x: &[C32]) -> (Vec<C32>, f64) {
        let xz: Vec<Z> = x.iter().map(|v| Z::new(v.re as f64, v.im as f64)).collect();
        let nx = self.normal(t, &xz);
        let g: Vec<Z> = nx.iter().zip(&self.ahy[t]).map(|(a, b)| a - b).collect();
        // f = ½ xᴴ N x − Re xᴴ Aᴴy + ½ yᴴy
        let f = 0.5 * xz.iter().zip(&nx).map(|(a, b)| (a.conj() * b).re).sum::<f64>() - xz.iter().zip(&self.ahy[t]).map(|(a, b)| (a.conj() * b).re).sum::<f64>() + 0.5 * self.yy[t];
        (g.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect(), f)
    }
    fn exact(&self) -> Vec<C32> {
        // CG in f64 per set to 1e-14
        let mut out = Vec::with_capacity(NT * N);
        for t in 0..NT {
            let b = &self.ahy[t];
            let mut x = vec![Z::new(0.0, 0.0); N];
            let mut r = b.clone();
            let mut p = r.clone();
            let dot = |a: &[Z], b: &[Z]| a.iter().zip(b).map(|(p, q)| (p.conj() * q).re).sum::<f64>();
            let mut rr = dot(&r, &r);
            let b0 = rr;
            for _ in 0..2000 {
                let ap = self.normal(t, &p);
                let a = rr / dot(&p, &ap);
                for i in 0..N {
                    x[i] += a * p[i];
                    r[i] -= a * ap[i];
                }
                let rn = dot(&r, &r);
                if rn < 1e-28 * b0 {
                    break;
                }
                for i in 0..N {
                    p[i] = r[i] + (rn / rr) * p[i];
                }
                rr = rn;
            }
            out.extend(x.iter().map(|z| C32::new(z.re as f32, z.im as f32)));
        }
        out
    }
}

impl Objective for Toy {
    fn kernels(&self) -> &Kernels {
        unimplemented!("host toy: no device kernels")
    }
    fn lens(&self) -> Vec<usize> {
        vec![N; NT]
    }
    fn gradient(&mut self, _st: &Arc<CudaStream>, _x: &Stack, _g: &mut Stack) -> Result<f64, Error> {
        unimplemented!("host toy")
    }
    fn metric(&mut self, _st: &Arc<CudaStream>, _t: usize, _r: &CudaSlice<C32>, _z: &mut CudaSlice<C32>, _inverse: bool) -> Result<(), Error> {
        unimplemented!("host toy")
    }
    fn gradient_host(&mut self, _st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
        let mut f = 0.0;
        for t in 0..NT {
            let (gt, ft) = self.grad_set(t, x.chunk(t));
            g.chunk_mut(t).copy_from_slice(&gt);
            f += ft;
        }
        Ok(f)
    }
    fn has_gradient_sink(&self) -> bool {
        true
    }
    fn gradient_host_sink(&mut self, _st: &Arc<CudaStream>, x: &HostStack, sink: &mut dyn GradSink) -> Result<Option<f64>, Error> {
        let mut f = 0.0;
        for t in 0..NT {
            let (gt, ft) = self.grad_set(t, x.chunk(t));
            sink.put(t, &gt)?;
            f += ft;
        }
        Ok(Some(f))
    }
    fn metric_host(&mut self, _st: &Arc<CudaStream>, t: usize, r: &[C32], z: &mut [C32], inverse: bool) -> Result<(), Error> {
        let mut v: Vec<Z> = r.iter().map(|c| Z::new(c.re as f64, c.im as f64)).collect();
        self.fft.apply(&mut v, false);
        for (a, &w) in v.iter_mut().zip(&self.w[t]) {
            *a *= if inverse { 1.0 / w } else { w };
        }
        self.fft.apply(&mut v, true);
        for (o, a) in z.iter_mut().zip(&v) {
            *o = C32::new(a.re as f32, a.im as f32);
        }
        Ok(())
    }
}

fn toy() -> Toy {
    let mut rng = Rng::new(0xB0F5);
    let fft = Fft2::new();
    // smooth coil maps around the image, magnitude ~1
    let smaps: Vec<Vec<Z>> = (0..NC)
        .map(|c| {
            let ang = 2.0 * std::f64::consts::PI * c as f64 / NC as f64;
            let (cy, cx) = (NY as f64 / 2.0 + 30.0 * ang.sin(), NX as f64 / 2.0 + 30.0 * ang.cos());
            (0..N).map(|i| {
                let (y, x) = ((i / NX) as f64, (i % NX) as f64);
                let d2 = ((y - cy).powi(2) + (x - cx).powi(2)) / 900.0;
                Z::from_polar((-0.5 * d2).exp() * 1.5, 0.3 * c as f64 + 0.02 * (x - y))
            }).collect()
        })
        .collect();
    // phantom per frame: ellipse with an inner disc whose intensity changes over frames, smooth phase; dark background
    let truth: Vec<Vec<Z>> = (0..NT)
        .map(|t| {
            (0..N).map(|i| {
                let (y, x) = ((i / NX) as f64 - NY as f64 / 2.0, (i % NX) as f64 - NX as f64 / 2.0);
                let e = (y / 21.0).powi(2) + (x / 15.0).powi(2);
                let d = ((y - 4.0).powi(2) + (x + 3.0).powi(2)).sqrt();
                let m = if e < 1.0 { 1.0 + if d < 5.0 { 0.5 + 0.4 * t as f64 } else { 0.0 } } else { 0.0 };
                Z::from_polar(m, 0.4 + 0.03 * x + 0.1 * t as f64)
            }).collect()
        })
        .collect();
    // variable-density masks (R ≈ 2.5), the 6 × 6 centre always
    let masks: Vec<Vec<bool>> = (0..NT)
        .map(|_| {
            (0..N).map(|i| {
                let (ky, kx) = (((i / NX) as f64 - NY as f64 / 2.0) / (NY as f64 / 2.0), ((i % NX) as f64 - NX as f64 / 2.0) / (NX as f64 / 2.0));
                let r = (ky * ky + kx * kx).sqrt();
                // Rng::uniform is in [−1, 1)
                r < 0.13 || 0.5 * (rng.uniform() + 1.0) < 0.55 * (1.0 - r).max(0.08)
            }).collect()
        })
        .collect();
    // fftshift-free masks: the mask above is in centred coordinates; rotate to the DFT's index order
    let masks: Vec<Vec<bool>> = masks.iter().map(|m| (0..N).map(|i| { let (y, x) = (i / NX, i % NX); m[((y + NY / 2) % NY) * NX + (x + NX / 2) % NX] }).collect()).collect();
    let sigma = 1.0 / 30.0 / (N as f64).sqrt() * (N as f64).sqrt(); // image-domain noise std ≈ 1/30 per unitary sample
    let mut ahy = Vec::new();
    let mut yy = Vec::new();
    for t in 0..NT {
        let mut a = vec![Z::new(0.0, 0.0); N];
        let mut ys = 0.0;
        for c in 0..NC {
            let mut v: Vec<Z> = truth[t].iter().zip(&smaps[c]).map(|(a, s)| a * s).collect();
            fft.apply(&mut v, false);
            for (z, &m) in v.iter_mut().zip(&masks[t]) {
                if m {
                    let nz = rng.complex_vec(1)[0];
                    *z += Z::new(nz.re, nz.im) * (sigma / 2f64.sqrt());
                    ys += z.norm_sqr();
                } else {
                    *z = Z::new(0.0, 0.0);
                }
            }
            fft.apply(&mut v, true);
            for ((o, b), s) in a.iter_mut().zip(&v).zip(&smaps[c]) {
                *o += b * s.conj();
            }
        }
        ahy.push(a);
        yy.push(ys);
    }
    let emean = (0..N).map(|i| smaps.iter().map(|s| s[i].norm_sqr()).sum::<f64>()).sum::<f64>() / N as f64;
    let mu = 0.02;
    let w: Vec<Vec<f64>> = masks.iter().map(|m| m.iter().map(|&s| 1.0 / (if s { emean } else { 0.15 * emean } + mu)).collect()).collect();
    Toy { fft, smaps, masks, ahy, yy, mu, w }
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n).sqrt()
}

fn soft(h: &mut [C32], th: f32) {
    for v in h.iter_mut() {
        let m = (v.re * v.re + v.im * v.im).sqrt();
        let w = if m > th { 1.0 - th / m } else { 0.0 };
        *v = C32::new(w * v.re, w * v.im);
    }
}

fn stream() -> Arc<CudaStream> {
    CudaContext::new(0).expect("CUDA context (stream handle only)").default_stream()
}

/// λ_max of the normal operator (power iteration on set 0), and of M N for the metric steps
fn lmax(toy: &mut Toy, st: &Arc<CudaStream>, metric: bool) -> f64 {
    let mut v: Vec<C32> = (0..N).map(|i| C32::new(((i * 7919) % 101) as f32 / 101.0 - 0.5, ((i * 104729) % 97) as f32 / 97.0 - 0.5)).collect();
    let mut l = 0.0;
    for _ in 0..60 {
        let vz: Vec<Z> = v.iter().map(|c| Z::new(c.re as f64, c.im as f64)).collect();
        let nv = toy.normal(0, &vz);
        let mut out: Vec<C32> = nv.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect();
        if metric {
            let inp = out.clone();
            toy.metric_host(st, 0, &inp, &mut out, false).unwrap();
        }
        let n = out.iter().map(|c| (c.re as f64).powi(2) + (c.im as f64).powi(2)).sum::<f64>().sqrt();
        let n0 = v.iter().map(|c| (c.re as f64).powi(2) + (c.im as f64).powi(2)).sum::<f64>().sqrt();
        l = n / n0;
        v = out.iter().map(|c| C32::new((c.re as f64 / n) as f32, (c.im as f64 / n) as f32)).collect();
    }
    l
}

type Den<'a> = &'a mut dyn FnMut(usize, &mut Vec<C32>);

/// one run: f32 (`store` None → run_objective_host, HASTY_STATE_STORE unset) or BFP; returns (x, history of rel(x, refx))
fn run(obj: &mut dyn Objective, st: &Arc<CudaStream>, pc: Precond, alpha: f64, iters: usize, den: Den, store: Option<StateStore>, refx: &[C32]) -> (Vec<C32>, Vec<f64>) {
    let mut x = HostStack::zeros(&vec![N; NT]);
    let mut hist = Vec::new();
    let mut rep = |k: usize, xh: &HostStack, _f: f64| {
        if k != usize::MAX {
            hist.push(rel(&xh.data, refx));
        }
    };
    match store {
        None => run_objective_host(obj, st, &mut x, pc, alpha, iters, den, &mut rep, None).unwrap(),
        Some(s) => run_objective_host_bfp(obj, st, &mut x, pc, alpha, iters, den, &mut rep, None, s).unwrap(),
    };
    (x.data, hist)
}

/// every 20th iteration's error
fn fmt_hist(h: &[f64]) -> String {
    h.iter().skip(19).step_by(20).map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" ")
}

/// first iteration (1-based) whose error is ≤ tol (None: never)
fn reach(h: &[f64], tol: f64) -> Option<usize> {
    h.iter().position(|&e| e <= tol).map(|i| i + 1)
}

/// a linear smoothing denoiser: x ← (1 − κ) x + κ · (3 × 3 box mean) per set image (non-expansive, smooth)
fn blur_mix(h: &mut [C32], kappa: f32) {
    for img in h.chunks_mut(N) {
        let src = img.to_vec();
        for y in 0..NY {
            for x in 0..NX {
                let (mut a, mut n) = (C32::new(0.0, 0.0), 0.0f32);
                for dy in [-1i32, 0, 1] {
                    for dx in [-1i32, 0, 1] {
                        let (yy, xx) = (y as i32 + dy, x as i32 + dx);
                        if yy >= 0 && yy < NY as i32 && xx >= 0 && xx < NX as i32 {
                            let v = src[yy as usize * NX + xx as usize];
                            a = C32::new(a.re + v.re, a.im + v.im);
                            n += 1.0;
                        }
                    }
                }
                let o = &mut img[y * NX + x];
                *o = C32::new((1.0 - kappa) * o.re + kappa * a.re / n, (1.0 - kappa) * o.im + kappa * a.im / n);
            }
        }
    }
}

#[test]
fn store_parse_and_bytes() {
    assert!(StateStore::parse("f32").unwrap().is_f32());
    let s = StateStore::parse("bfp16").unwrap();
    assert_eq!((s.grad.unwrap().bits(), s.pair.unwrap().bits()), (16, 16));
    let s = StateStore::parse("bfp12:bfp24").unwrap();
    assert_eq!((s.grad.unwrap().bits(), s.pair.unwrap().bits()), (12, 24));
    let s = StateStore::parse("bfp16:f32").unwrap();
    assert!(s.pair.is_none() && s.grad.is_some());
    assert!(StateStore::parse("bfp30").is_err() && StateStore::parse("fp16").is_err());
    // 1 mm per-frame state of research/bfp_storage.md §7 step 5: 4 stacks of 17.8 GiB → x f32 + 3 × bfp16
    let lens = vec![(17.8 * (1u64 << 30) as f64 / 8.0 / 280.0) as usize; 280];
    let gib = (1u64 << 30) as f64;
    let f = StateStore::F32.state_bytes(&lens, true, false) as f64 / gib;
    let b = StateStore::parse("bfp16").unwrap().state_bytes(&lens, true, false) as f64 / gib;
    println!("1 mm per-frame host state: f32 {f:.1} GiB, bfp16 {b:.1} GiB");
    assert!((f - 71.2).abs() < 0.2 && (b - 44.9).abs() < 0.3, "f32 {f} bfp16 {b}");
}

#[test]
fn no_prior_fixed_point_is_exact_for_every_preconditioner() {
    assert!(std::env::var("HASTY_STATE_STORE").map_or(true, |v| v.is_empty()), "run without HASTY_STATE_STORE");
    let st = stream();
    let mut t = toy();
    let xs = t.exact();
    let (l, lm) = (lmax(&mut t, &st, false), lmax(&mut t, &st, true));
    println!("λ_max {l:.3}, λ_max(M N) {lm:.3}; x★ by f64 CG");
    let cases: [(Precond, f64, usize); 6] = [
        (Precond::None, 1.0 / l, 300),
        (Precond::Circulant, 1.0 / lm, 150),
        (Precond::Zmshr1, 1.0 / l, 150),
        (Precond::Zmshr1Set, 1.0 / l, 150),
        (Precond::Zmshr1M, 1.0 / lm, 100),
        (Precond::Zmshr1MSet, 1.0 / lm, 100),
    ];
    for (pc, alpha, iters) in cases {
        let (x32, h32) = run(&mut t, &st, pc, alpha, iters, &mut |_, _| {}, None, &xs);
        let e32 = rel(&x32, &xs);
        println!("{pc:?} f32: rel error {e32:.2e}  [{}]", fmt_hist(&h32));
        for spec in ["f32:f32", "bfp16", "bfp12", "bfp24"] {
            let s = StateStore::parse(spec).unwrap();
            let (xb, hb) = run(&mut t, &st, pc, alpha, iters, &mut |_, _| {}, Some(s), &xs);
            let eb = rel(&xb, &xs);
            let d = rel(&xb, &x32);
            println!("{pc:?} {spec:>7}: rel error {eb:.2e}, vs f32 run {d:.2e}  [{}]", fmt_hist(&hb));
            // fixed point exact at the f32 floor; the BFP loop with f32 stores equals the f32 loop to rounding
            assert!(eb < (3.0 * e32).max(2e-6), "{pc:?} {spec}: rel error {eb:.3e} (f32 {e32:.3e})");
            // history close: iterations to reach 1e-2, 1e-3, 1e-4, 1e-5 within 25 % (+3) of the f32 run's
            let mut line = String::new();
            for tol in [1e-2, 1e-3, 1e-4, 1e-5] {
                let (a, b) = (reach(&hb, tol), reach(&h32, tol));
                line += &format!(" {tol:.0e}: {} / {}", a.map_or("-".into(), |v| v.to_string()), b.map_or("-".into(), |v| v.to_string()));
                // bfp16 / bfp24 within 25 % (+3 iterations); bfp12 (below the recommended width) within 50 %
                let slack = if spec == "bfp12" { 1.5 } else { 1.25 };
                if let Some(b) = b {
                    let a = a.unwrap_or_else(|| panic!("{pc:?} {spec}: never reaches {tol:e} (f32 at iteration {b})"));
                    assert!(a as f64 <= slack * b as f64 + 3.0, "{pc:?} {spec}: reaches {tol:e} at iteration {a}, f32 at {b}");
                }
            }
            println!("        iterations to reach (bfp / f32):{line}");
        }
    }
}

#[test]
fn red_stationary_point_with_bfp_state() {
    let st = stream();
    let mut t = toy();
    let xs = t.exact();
    let l = lmax(&mut t, &st, false);
    let (lam, th) = (0.5f64, 0.08f32);
    let alpha = 1.0 / (l + lam);
    // reference: a long f32 RED run
    let mut den = |_k: usize, h: &mut Vec<C32>| soft(h, th);
    let mut red = Red::new(&mut t, &mut den, lam);
    red.log_every = 0;
    let (xr, _) = run(&mut red, &st, Precond::Zmshr1, alpha, 1500, &mut |_, _| {}, None, &xs);
    for (pc, iters) in [(Precond::Zmshr1, 300), (Precond::Zmshr1Set, 300)] {
        let (x32, h32) = run(&mut red, &st, pc, alpha, iters, &mut |_, _| {}, None, &xr);
        let e32 = rel(&x32, &xr);
        println!("RED {pc:?} f32: vs converged {e32:.2e}  [{}]", fmt_hist(&h32));
        for spec in ["bfp16", "bfp12"] {
            let (xb, hb) = run(&mut red, &st, pc, alpha, iters, &mut |_, _| {}, Some(StateStore::parse(spec).unwrap()), &xr);
            let eb = rel(&xb, &xr);
            println!("RED {pc:?} {spec}: vs converged {eb:.2e}, vs f32 run {:.2e}  [{}]", rel(&xb, &x32), fmt_hist(&hb));
            assert!(eb < (3.0 * e32).max(2e-6), "RED {pc:?} {spec}: {eb:.3e} vs f32 {e32:.3e}");
        }
    }
}

#[test]
fn pnp_fixed_point_with_bfp_state() {
    // P²nP x ← D(x − α P ∇f): the denoiser changes x after the step, so the secant record is x_prev at the pair
    // width (s error 2^−b ‖x‖). The fixed point depends on the step (P), so it is compared with the f32 run's own
    // fixed point, only where that run converges (drift between iterations n/2 and n below 1e-6); otherwise reported.
    let st = stream();
    let mut t = toy();
    let xs = t.exact();
    let (l, lm) = (lmax(&mut t, &st, false), lmax(&mut t, &st, true));
    let th = 0.02f32;
    let dens: [(&str, &(dyn Fn(&mut [C32]) + Sync)); 4] = [("soft 0.02", &move |h: &mut [C32]| soft(h, th)), ("blur κ 0.3", &|h: &mut [C32]| blur_mix(h, 0.3)), ("soft 0.002", &|h: &mut [C32]| soft(h, 0.002)), ("blur κ 0.03", &|h: &mut [C32]| blur_mix(h, 0.03))];
    for (dname, dfn) in dens {
        for (pc, alpha, iters) in [(Precond::None, 1.0 / l, 400), (Precond::Circulant, 1.0 / lm, 400), (Precond::Zmshr1, 1.0 / l, 400), (Precond::Zmshr1M, 1.0 / lm, 400)] {
            let (x32, h32) = run(&mut t, &st, pc, alpha, iters, &mut |_, h: &mut Vec<C32>| dfn(h), None, &xs);
            let (xh, _) = run(&mut t, &st, pc, alpha, iters / 2, &mut |_, h: &mut Vec<C32>| dfn(h), None, &xs);
            let drift = rel(&xh, &x32);
            let conv = drift < 1e-6;
            println!("P²nP {dname} {pc:?} f32: distance to x★ (no prior) {:.2e}, drift it {} → {iters}: {drift:.2e}{}  [{}]", rel(&x32, &xs), iters / 2, if conv { "" } else { " (not converged: reported only)" }, fmt_hist(&h32));
            for spec in ["bfp16", "bfp16:bfp20", "bfp16:bfp24", "bfp24", "bfp12"] {
                let (xb, hb) = run(&mut t, &st, pc, alpha, iters, &mut |_, h: &mut Vec<C32>| dfn(h), Some(StateStore::parse(spec).unwrap()), &x32);
                let d = rel(&xb, &x32);
                println!("P²nP {dname} {pc:?} {spec:>11}: vs the f32 fixed point {d:.2e}  [{}]", fmt_hist(&hb));
                if conv {
                    assert!(d < 1e-4, "P²nP {dname} {pc:?} {spec}: {d:.3e}");
                }
            }
        }
    }
}

#[test]
fn checkpoint_resume_of_the_bfp_state() {
    let st = stream();
    let mut t = toy();
    let xs = t.exact();
    let l = lmax(&mut t, &st, false);
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp/bfp_state");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let s = StateStore::parse("bfp16").unwrap();
    // x bit-exact in the checkpoint (the BFP stacks always are), so the resumed run repeats the uninterrupted one
    std::env::set_var("HASTY_IO_LOSSLESS", "1");
    for (pc, name) in [(Precond::Zmshr1, "ck_z"), (Precond::Zmshr1M, "ck_zm")] {
        let alpha = if pc.metric() { 1.0 / lmax(&mut t, &st, true) } else { 1.0 / l };
        let (xa, _) = run(&mut t, &st, pc, alpha, 40, &mut |_, _| {}, Some(s), &xs);
        let ctl = RunCtl::new(&dir, 0);
        let mut x = HostStack::zeros(&vec![N; NT]);
        let stop = dir.join("STOP");
        let mut rep = |k: usize, _x: &HostStack, _f: f64| {
            if k == 14 {
                std::fs::write(&stop, b"").unwrap();
            }
        };
        let o = run_objective_host_bfp(&mut t, &st, &mut x, pc, alpha, 40, &mut |_, _| {}, &mut rep, Some((&ctl, name)), s).unwrap();
        assert!(matches!(o, Outcome::Stopped(15)), "{o:?}");
        assert!(dir.join(format!("state_{name}.hsb")).exists());
        // the f32 loop refuses the BFP checkpoint
        let ctl2 = RunCtl::new(&dir, 0);
        let mut xf = HostStack::zeros(&vec![N; NT]);
        assert!(run_objective_host(&mut t, &st, &mut xf, pc, alpha, 40, &mut |_, _| {}, &mut |_, _, _| {}, Some((&ctl2, name))).is_err());
        // resume (RunCtl::new removes STOP)
        let ctl3 = RunCtl::new(&dir, 0);
        let mut x2 = HostStack::zeros(&vec![N; NT]);
        let o = run_objective_host_bfp(&mut t, &st, &mut x2, pc, alpha, 40, &mut |_, _| {}, &mut |_, _, _| {}, Some((&ctl3, name)), s).unwrap();
        assert!(matches!(o, Outcome::Finished));
        let d = rel(&x2.data, &xa);
        println!("{pc:?}: resumed at 15 vs uninterrupted (40 its): {d:.2e} (lossless checkpoint)");
        assert!(d < 1e-6, "{pc:?}: resumed run differs by {d:.3e}");
        assert!(!dir.join(format!("state_{name}.hsb")).exists(), "checkpoint removed at the end");
    }
    std::env::remove_var("HASTY_IO_LOSSLESS");
}

/// f(x) = ½ Σ |a_i x_i − b_i|² over sets of unequal, multi-chunk length; metric M = 1 / a² (≈ H⁻¹)
struct Diag {
    lens: Vec<usize>,
    a2: Vec<f32>,
    ab: Vec<C32>,
}

impl Diag {
    fn off(&self, t: usize) -> usize {
        self.lens[..t].iter().sum()
    }
    fn grad_set(&self, t: usize, x: &[C32]) -> (Vec<C32>, f64) {
        let o = self.off(t);
        let mut f = 0.0;
        let g = x.iter().enumerate().map(|(i, v)| {
            let (a2, c) = (self.a2[o + i], self.ab[o + i]);
            f += 0.5 * (a2 * (v.re * v.re + v.im * v.im) - 2.0 * (c.re * v.re + c.im * v.im)) as f64;
            C32::new(a2 * v.re - c.re, a2 * v.im - c.im)
        }).collect();
        (g, f)
    }
}

impl Objective for Diag {
    fn kernels(&self) -> &Kernels {
        unimplemented!("host toy")
    }
    fn lens(&self) -> Vec<usize> {
        self.lens.clone()
    }
    fn gradient(&mut self, _st: &Arc<CudaStream>, _x: &Stack, _g: &mut Stack) -> Result<f64, Error> {
        unimplemented!("host toy")
    }
    fn metric(&mut self, _st: &Arc<CudaStream>, _t: usize, _r: &CudaSlice<C32>, _z: &mut CudaSlice<C32>, _inverse: bool) -> Result<(), Error> {
        unimplemented!("host toy")
    }
    fn gradient_host(&mut self, _st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
        let mut f = 0.0;
        for t in 0..self.lens.len() {
            let (gt, ft) = self.grad_set(t, x.chunk(t));
            g.chunk_mut(t).copy_from_slice(&gt);
            f += ft;
        }
        Ok(f)
    }
    fn has_gradient_sink(&self) -> bool {
        true
    }
    fn gradient_host_sink(&mut self, _st: &Arc<CudaStream>, x: &HostStack, sink: &mut dyn GradSink) -> Result<Option<f64>, Error> {
        let mut f = 0.0;
        for t in 0..self.lens.len() {
            let (gt, ft) = self.grad_set(t, x.chunk(t));
            sink.put(t, &gt)?;
            f += ft;
        }
        Ok(Some(f))
    }
    fn metric_host(&mut self, _st: &Arc<CudaStream>, t: usize, r: &[C32], z: &mut [C32], inverse: bool) -> Result<(), Error> {
        let o = self.off(t);
        for i in 0..r.len() {
            let m = if inverse { self.a2[o + i] } else { 1.0 / self.a2[o + i] };
            z[i] = C32::new(m * r[i].re, m * r[i].im);
        }
        Ok(())
    }
}

fn diag() -> (Diag, Vec<C32>, Vec<C32>) {
    let lens = vec![150_001usize, 70_000, 150_001];
    let n: usize = lens.iter().sum();
    let mut rng = Rng::new(0xD1A6);
    // a over a 4:1 range, b with a few large and many small entries (BFP blocks with dynamic range)
    // Rng::uniform is in [−1, 1): a in [0.8, 1.6)
    let a: Vec<f32> = (0..n).map(|_| 1.2 + 0.4 * rng.uniform() as f32).collect();
    let b: Vec<C32> = rng.complex_vec(n).iter().enumerate().map(|(i, z)| { let s = if i % 97 == 0 { 30.0 } else { 1.0 }; C32::new(s * z.re as f32, s * z.im as f32) }).collect();
    let a2: Vec<f32> = a.iter().map(|v| v * v).collect();
    let ab: Vec<C32> = a.iter().zip(&b).map(|(&ai, bi)| C32::new(ai * bi.re, ai * bi.im)).collect();
    // exact minimiser, and the P²nP fixed point with the support projection (zero where i % 7 == 0)
    let xs: Vec<C32> = (0..n).map(|i| C32::new(ab[i].re / a2[i], ab[i].im / a2[i])).collect();
    let xp: Vec<C32> = (0..n).map(|i| if i % 7 == 0 { C32::new(0.0, 0.0) } else { xs[i] }).collect();
    (Diag { lens, a2, ab }, xs, xp)
}

fn proj(h: &mut [C32]) {
    for (i, v) in h.iter_mut().enumerate() {
        if i % 7 == 0 {
            *v = C32::new(0.0, 0.0);
        }
    }
}

fn run_d(obj: &mut Diag, st: &Arc<CudaStream>, pc: Precond, alpha: f64, iters: usize, den: Den, store: Option<StateStore>, refx: &[C32]) -> (Vec<C32>, Vec<f64>) {
    let mut x = HostStack::zeros(&obj.lens.clone());
    let mut hist = Vec::new();
    let mut rep = |k: usize, xh: &HostStack, _f: f64| {
        if k != usize::MAX {
            hist.push(rel(&xh.data, refx));
        }
    };
    match store {
        None => run_objective_host(obj, st, &mut x, pc, alpha, iters, den, &mut rep, None).unwrap(),
        Some(s) => run_objective_host_bfp(obj, st, &mut x, pc, alpha, iters, den, &mut rep, None, s).unwrap(),
    };
    (x.data, hist)
}

#[test]
fn multi_chunk_sets_projection_pnp_and_pair_invalidation() {
    let st = stream();
    let (mut d, xs, xproj) = diag();
    let lmax = 2.56;
    let cases = [(Precond::Zmshr1, 1.0 / lmax, 80), (Precond::Zmshr1Set, 1.0 / lmax, 80), (Precond::Zmshr1M, 0.9, 40), (Precond::Zmshr1MSet, 0.9, 40)];
    for (pc, alpha, iters) in cases {
        // no prior: the difference record from the second iteration on
        let (x32, h32) = run_d(&mut d, &st, pc, alpha, iters, &mut |_, _| {}, None, &xs);
        let e32 = rel(&x32, &xs);
        for spec in ["f32:f32", "bfp16", "bfp24"] {
            let (xb, hb) = run_d(&mut d, &st, pc, alpha, iters, &mut |_, _| {}, Some(StateStore::parse(spec).unwrap()), &xs);
            let eb = rel(&xb, &xs);
            println!("diag {pc:?} {spec:>7} no prior: rel error {eb:.2e} (f32 {e32:.2e}), vs f32 run {:.2e}; reach 1e-5 at {:?} / f32 {:?}", rel(&xb, &x32), reach(&hb, 1e-5), reach(&h32, 1e-5));
            assert!(eb < (3.0 * e32).max(2e-6), "diag {pc:?} {spec}: {eb:.3e} vs f32 {e32:.3e}");
            if spec == "f32:f32" {
                assert!(rel(&xb, &x32) < 1e-6, "the BFP loop with f32 stores differs from the f32 loop");
            }
        }
        // P²nP with the support projection: the fixed point is the projected minimiser, for every preconditioner
        let (x32, _) = run_d(&mut d, &st, pc, alpha, iters, &mut |_, h: &mut Vec<C32>| proj(h), None, &xproj);
        let e32 = rel(&x32, &xproj);
        for spec in ["bfp16", "bfp16:f32", "f32:bfp16", "f32:bfp24", "bfp24:f32", "bfp24", "bfp12:f32"] {
            let (xb, hb) = run_d(&mut d, &st, pc, alpha, iters, &mut |_, h: &mut Vec<C32>| proj(h), Some(StateStore::parse(spec).unwrap()), &xproj);
            let eb = rel(&xb, &xproj);
            println!("diag {pc:?} {spec:>11} P²nP projection (x_prev record): rel error {eb:.2e} (f32 {e32:.2e}); reach 1e-6 at {:?}  [{}]", reach(&hb, 1e-6), hb.iter().step_by(5).map(|v| format!("{v:.0e}")).collect::<Vec<_>>().join(" "));
            // the width recommended for P²nP: within 5e-7 of the projected minimiser (f32: 3–5e-8); lower widths
            // are reported (measured bias ≈ 2 · 2^-b)
            if spec == "bfp24" {
                assert!(eb < 5e-7, "diag P²nP {pc:?} bfp24: {eb:.3e} (f32 {e32:.3e})");
            }
        }
        // the projection switched on at iteration 15: difference record, then an invalid pair, then x_prev
        let late = |k: usize, h: &mut Vec<C32>| {
            if k >= 15 {
                proj(h)
            }
        };
        let (x32, _) = run_d(&mut d, &st, pc, alpha, iters + 15, &mut { late }, None, &xproj);
        for spec in ["bfp24", "bfp16"] {
            let (xb, _) = run_d(&mut d, &st, pc, alpha, iters + 15, &mut { late }, Some(StateStore::parse(spec).unwrap()), &xproj);
            let (e32, eb) = (rel(&x32, &xproj), rel(&xb, &xproj));
            println!("diag {pc:?} {spec} projection from it 15 (record switch): rel error {eb:.2e} (f32 {e32:.2e})");
            if spec == "bfp24" {
                assert!(eb < 5e-7, "diag late projection {pc:?} bfp24: {eb:.3e} vs f32 {e32:.3e}");
            }
        }
    }
    // resume of a P²nP run (x_prev record, x stored bit-exact for it; default lossy policy otherwise)
    assert!(std::env::var("HASTY_IO_LOSSLESS").is_err());
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp/bfp_state_diag");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let s = StateStore::parse("bfp16").unwrap();
    let (xa, _) = run_d(&mut d, &st, Precond::Zmshr1, 0.25, 40, &mut |_, h: &mut Vec<C32>| proj(h), Some(s), &xproj);
    let ctl = RunCtl::new(&dir, 0);
    let mut x = HostStack::zeros(&d.lens.clone());
    let stop = dir.join("STOP");
    let o = run_objective_host_bfp(&mut d, &st, &mut x, Precond::Zmshr1, 0.25, 40, &mut |_, h: &mut Vec<C32>| proj(h), &mut |k, _, _| if k == 14 { std::fs::write(&stop, b"").unwrap() }, Some((&ctl, "pnp")), s).unwrap();
    assert!(matches!(o, Outcome::Stopped(15)));
    let ctl = RunCtl::new(&dir, 0);
    let mut x2 = HostStack::zeros(&d.lens.clone());
    run_objective_host_bfp(&mut d, &st, &mut x2, Precond::Zmshr1, 0.25, 40, &mut |_, h: &mut Vec<C32>| proj(h), &mut |_, _, _| {}, Some((&ctl, "pnp")), s).unwrap();
    let dd = rel(&x2.data, &xa);
    println!("diag P²nP resume at 15 (x_prev record): resumed vs uninterrupted {dd:.2e}");
    assert!(dd < 1e-6, "P²nP resume differs by {dd:.3e}");
    // a checkpoint of another width is refused before anything is read
    let o = run_objective_host_bfp(&mut d, &st, &mut x, Precond::Zmshr1, 0.25, 40, &mut |_, h: &mut Vec<C32>| proj(h), &mut |k, _, _| if k == 3 { std::fs::write(&stop, b"").unwrap() }, Some((&RunCtl::new(&dir, 0), "w")), s).unwrap();
    assert!(matches!(o, Outcome::Stopped(4)));
    let e = run_objective_host_bfp(&mut d, &st, &mut x, Precond::Zmshr1, 0.25, 40, &mut |_, _| {}, &mut |_, _, _| {}, Some((&RunCtl::new(&dir, 0), "w")), StateStore::parse("bfp20").unwrap());
    assert!(e.is_err_and(|e| e.to_string().contains("this run uses")));
}

#[test]
#[ignore]
fn step_cost_at_scale() {
    // wall time of the host update (everything but the gradient) per iteration, f32 loop vs BFP loop, on 3 sets of
    // 20 M complex values (one f32 stack 0.45 GiB); the toy gradient's own time is measured and subtracted
    let st = stream();
    let n1 = std::env::var("STEP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000_000usize);
    let lens = vec![n1; 3];
    let n: usize = lens.iter().sum();
    let mut rng = Rng::new(5);
    let a2: Vec<f32> = (0..n).map(|_| { let a = 1.2 + 0.4 * rng.uniform() as f32; a * a }).collect();
    let ab: Vec<C32> = (0..n).map(|i| C32::new(((i % 1013) as f32 - 500.0) / 300.0, ((i % 769) as f32 - 380.0) / 250.0)).collect();
    struct Timed(Diag, f64);
    impl Objective for Timed {
        fn kernels(&self) -> &Kernels { unimplemented!() }
        fn lens(&self) -> Vec<usize> { self.0.lens.clone() }
        fn gradient(&mut self, _: &Arc<CudaStream>, _: &Stack, _: &mut Stack) -> Result<f64, Error> { unimplemented!() }
        fn metric(&mut self, _: &Arc<CudaStream>, _: usize, _: &CudaSlice<C32>, _: &mut CudaSlice<C32>, _: bool) -> Result<(), Error> { unimplemented!() }
        fn gradient_host(&mut self, st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
            let t = std::time::Instant::now();
            let r = self.0.gradient_host(st, x, g);
            self.1 += t.elapsed().as_secs_f64();
            r
        }
        fn has_gradient_sink(&self) -> bool { true }
        fn gradient_host_sink(&mut self, st: &Arc<CudaStream>, x: &HostStack, sink: &mut dyn GradSink) -> Result<Option<f64>, Error> {
            // the encode into the sink is part of the BFP state's cost: only the gradient arithmetic is subtracted
            let mut f = 0.0;
            for t in 0..self.0.lens.len() {
                let t0 = std::time::Instant::now();
                let (gt, ft) = self.0.grad_set(t, x.chunk(t));
                self.1 += t0.elapsed().as_secs_f64();
                sink.put(t, &gt)?;
                f += ft;
            }
            let _ = st;
            Ok(Some(f))
        }
        fn metric_host(&mut self, st: &Arc<CudaStream>, t: usize, r: &[C32], z: &mut [C32], inv: bool) -> Result<(), Error> { self.0.metric_host(st, t, r, z, inv) }
    }
    let mut obj = Timed(Diag { lens: lens.clone(), a2, ab }, 0.0);
    let iters = 6;
    println!("{} rayon threads, 3 sets × {n1} complex ({:.2} GiB per f32 stack)", rayon::current_num_threads(), (8 * n) as f64 / (1u64 << 30) as f64);
    for pc in [Precond::Zmshr1, Precond::Zmshr1M] {
        for spec in [None, Some("bfp16")] {
            let mut x = HostStack::zeros(&lens);
            obj.1 = 0.0;
            let t = std::time::Instant::now();
            let alpha = if pc.metric() { 0.9 } else { 0.25 };
            match spec {
                None => run_objective_host(&mut obj, &st, &mut x, pc, alpha, iters, &mut |_, _| {}, &mut |_, _, _| {}, None).unwrap(),
                Some(s) => run_objective_host_bfp(&mut obj, &st, &mut x, pc, alpha, iters, &mut |_, _| {}, &mut |_, _, _| {}, None, StateStore::parse(s).unwrap()).unwrap(),
            };
            let tot = t.elapsed().as_secs_f64();
            let per = (tot - obj.1) / iters as f64;
            println!("{pc:?} {:>5}: host update {per:.3} s per iteration ({:.2} GB/s of f32-stack bytes per pass-equivalent: {:.1} ns per complex)", spec.unwrap_or("f32"), (8 * n) as f64 / per / 1e9, per / n as f64 * 1e9);
        }
    }
}

