//! GPU (kernels only; host toy objective): the temporal subspace wrapper (`hasty_recon::tsub::Subspace`) solves
//! min_c f(T c) — per-frame diagonal "sampling" a_f with frame-dependent holes, data consistent with x★ = T c★ — for an
//! orthonormal Fourier and a non-orthogonal B-spline basis, with plain and per-set ZMSHR1 steps, and the per-frame
//! problem alone cannot (the holes are its null space). Also times T and Tᵀ at a 2 mm-like size.
//!   cargo test --release -p hasty-recon --test tsub -- --nocapture

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_recon::gpu::Kernels;
use hasty_recon::hoststack::{run_objective_host, HostStack};
use hasty_recon::p2np::{Objective, Precond, Stack};
use hasty_recon::tsub::{apply_sets, Subspace, TemporalBasis};
use hasty_recon::Error;
use hasty_testkit::Rng;

/// f(x) = ½ Σ_t ‖a_t ∘ x_t − y_t‖² over frame sets (real a, zero where frame t does not sample voxel q)
struct Frames {
    kern: Arc<Kernels>,
    lens: Vec<usize>,
    a2: Vec<f32>,
    ay: Vec<C32>,
    yy: f64,
    /// ‖y_t‖² per set, and whether the subset form (`gradient_host_sets`) is offered
    yy_t: Vec<f64>,
    subsets: bool,
}

impl Objective for Frames {
    fn kernels(&self) -> &Kernels {
        &self.kern
    }
    fn lens(&self) -> Vec<usize> {
        self.lens.clone()
    }
    fn gradient(&mut self, _st: &Arc<CudaStream>, _x: &Stack, _g: &mut Stack) -> Result<f64, Error> {
        unimplemented!("host only")
    }
    fn gradient_host(&mut self, _st: &Arc<CudaStream>, x: &HostStack, g: &mut HostStack) -> Result<f64, Error> {
        let mut f = self.yy;
        for i in 0..x.data.len() {
            let (a2, c, v) = (self.a2[i], self.ay[i], x.data[i]);
            g.data[i] = C32::new(a2 * v.re - c.re, a2 * v.im - c.im);
            f += (a2 * (v.re * v.re + v.im * v.im) - 2.0 * (c.re * v.re + c.im * v.im)) as f64;
        }
        Ok(0.5 * f)
    }
    fn gradient_host_sets(&mut self, _st: &Arc<CudaStream>, sets: &[usize], x: &HostStack, g: &mut HostStack) -> Result<Option<f64>, Error> {
        if !self.subsets {
            return Ok(None);
        }
        let nv = self.lens[0];
        let mut f = 0.0;
        for &t in sets {
            f += self.yy_t[t];
            let xc = x.chunk(t).to_vec();
            let gc = g.chunk_mut(t);
            for i in 0..nv {
                let (a2, c, v) = (self.a2[t * nv + i], self.ay[t * nv + i], xc[i]);
                gc[i] = C32::new(a2 * v.re - c.re, a2 * v.im - c.im);
                f += (a2 * (v.re * v.re + v.im * v.im) - 2.0 * (c.re * v.re + c.im * v.im)) as f64;
            }
        }
        Ok(Some(0.5 * f))
    }
    fn metric(&mut self, _st: &Arc<CudaStream>, _t: usize, _r: &CudaSlice<C32>, _z: &mut CudaSlice<C32>, _inverse: bool) -> Result<(), Error> {
        unimplemented!("no metric")
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
fn subspace_recovers_what_single_frames_cannot() {
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let (n, nr, ne, nv) = (20usize, 2usize, 3usize, 4000usize);
    let nf = n * nr;
    let mut rng = Rng::new(11);
    for basis in [TemporalBasis::cardiac_fourier(n, nr, 3), TemporalBasis::periodic_bspline(n, nr, 6, 3)] {
        let nk = basis.nk;
        let cs: Vec<C32> = rng.complex_vec(nk * ne * nv).iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect();
        let mut xs = vec![C32::new(0.0, 0.0); nf * ne * nv];
        apply_sets(&basis.b, nf, nk, ne, nv, &cs, &mut xs);
        // every frame misses 25 % of the voxels, a different quarter per frame (voxel q is not sampled in the frames
        // f ≡ −7q mod 4: spread in time, so TᵀAᵀAT stays well conditioned; random holes cluster in time and leave some
        // voxels nearly singular); sampled voxels have weights in [0.75, 1.25)
        let a: Vec<f32> = (0..nf * ne * nv).map(|i| { let (f, q) = (i / (ne * nv), i % nv); if (7 * q + f) % 4 == 0 { 0.0 } else { 1.0 + 0.25 * rng.uniform() as f32 } }).collect();
        let a2: Vec<f32> = a.iter().map(|v| v * v).collect();
        let y: Vec<C32> = a.iter().zip(&xs).map(|(&w, v)| C32::new(w * v.re, w * v.im)).collect();
        let ay: Vec<C32> = a.iter().zip(&y).map(|(&w, v)| C32::new(w * v.re, w * v.im)).collect();
        let yy: f64 = y.iter().map(|v| (v.re * v.re + v.im * v.im) as f64).sum();
        let lmax = a2.iter().fold(0f32, |m, &v| m.max(v)) as f64;
        let mut fr = Frames { kern: kern.clone(), lens: vec![nv; nf * ne], a2, ay, yy, yy_t: vec![], subsets: false };
        // per frame: the holes stay at the start value 0
        let mut x = HostStack::zeros(&vec![nv; nf * ne]);
        run_objective_host(&mut fr, &st, &mut x, Precond::Zmshr1Set, 1.0 / lmax, 100, &mut |_, _| {}, &mut |_, _, _| {}, None).unwrap();
        let e_frame = rel(&x.data, &xs);
        // subspace: α = 1 / λ_max(Tᵀ N T) ≤ 1 / (λ_max(N) λ_max(TᵀT))
        let gmax = {
            let g = basis.gram();
            (0..nk).map(|i| (0..nk).map(|j| g[i * nk + j].abs()).sum::<f64>()).fold(0.0, f64::max)
        };
        for pc in [Precond::None, Precond::Zmshr1Set] {
            let mut sub = Subspace::new(&mut fr, basis.clone(), ne);
            let mut c = HostStack::zeros(&sub.coeff_lens());
            let iters = if pc == Precond::None { 300 } else { 150 };
            let dbg = std::env::var("TSUB_DBG").is_ok();
            run_objective_host(&mut sub, &st, &mut c, pc, 1.0 / (lmax * gmax), iters, &mut |_, _| {}, &mut |k, cc: &HostStack, f| if dbg && k < 40 { println!("  it {k}: f {f:.4e}, coefficient error {:.3e}", rel(&cc.data, &cs)) }, None).unwrap();
            let mut xc = vec![C32::new(0.0, 0.0); nf * ne * nv];
            sub.synth(&c.data, &mut xc);
            let (ec, ex) = (rel(&c.data, &cs), rel(&xc, &xs));
            println!("{} ({nk} of {nf} frames): per-frame ZMSHR1 rel error {e_frame:.3}; subspace {pc:?}: coefficients {ec:.2e}, frames {ex:.2e}", basis.name);
            let tol = if pc == Precond::None { 1e-3 } else { 1e-4 };
            assert!(ex < tol, "{} {pc:?}: frame error {ex:.3e}", basis.name);
        }
        assert!(e_frame > 0.3, "the per-frame problem should not recover the holes: {e_frame}");
    }
}

#[test]
fn blocked_gradient_matches_full_stacks() {
    // Subspace::gradient_host through frame blocks (the inner objective's subset form, g accumulated block by block)
    // against the full frame stacks x = T c, ∇f(x): 40 frames in blocks of 3 (the last block shorter)
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let (n, nr, ne, nv) = (20usize, 2usize, 3usize, 1000usize);
    let nf = n * nr;
    let mut rng = Rng::new(5);
    for basis in [TemporalBasis::cardiac_fourier(n, nr, 3), TemporalBasis::periodic_bspline(n, nr, 6, 3)] {
        let a2: Vec<f32> = (0..nf * ne * nv).map(|_| (0.5 + 0.5 * rng.uniform()) as f32).collect();
        let y: Vec<C32> = rng.complex_vec(nf * ne * nv).iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect();
        let ay: Vec<C32> = a2.iter().zip(&y).map(|(&w, v)| C32::new(w.sqrt() * v.re, w.sqrt() * v.im)).collect();
        let yy_t: Vec<f64> = (0..nf * ne).map(|t| y[t * nv..(t + 1) * nv].iter().map(|v| (v.re * v.re + v.im * v.im) as f64).sum()).collect();
        let yy = yy_t.iter().sum();
        let mut fr = Frames { kern: kern.clone(), lens: vec![nv; nf * ne], a2, ay, yy, yy_t, subsets: false };
        let c = HostStack { data: rng.complex_vec(basis.nk * ne * nv).iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect(), ..HostStack::zeros(&vec![nv; basis.nk * ne]) };
        let lc = vec![nv; basis.nk * ne];
        let (mut g_full, mut g_blk) = (HostStack::zeros(&lc), HostStack::zeros(&lc));
        let f_full = {
            let mut sub = Subspace::new(&mut fr, basis.clone(), ne);
            sub.block_bytes = 3 * ne * nv * 8;
            sub.gradient_host(&st, &c, &mut g_full).unwrap()
        };
        fr.subsets = true;
        let f_blk = {
            let mut sub = Subspace::new(&mut fr, basis.clone(), ne);
            sub.block_bytes = 3 * ne * nv * 8;
            // twice: the block buffers are reused, g overwritten (not accumulated across calls)
            sub.gradient_host(&st, &c, &mut g_blk).unwrap();
            sub.gradient_host(&st, &c, &mut g_blk).unwrap()
        };
        let (eg, ef) = (rel(&g_blk.data, &g_full.data), ((f_blk - f_full) / f_full).abs());
        let nrm = |v: &[C32]| v.iter().map(|z| (z.re * z.re + z.im * z.im) as f64).sum::<f64>().sqrt();
        println!("{}: blocked vs full gradient rel {eg:.2e}, f rel {ef:.2e} (f {f_full:.4e} / {f_blk:.4e}, |g| {:.3e} / {:.3e}, |c| {:.3e})", basis.name, nrm(&g_full.data), nrm(&g_blk.data), nrm(&c.data));
        assert!(eg < 1e-5 && ef < 1e-9, "{}: gradient {eg:.2e}, f {ef:.2e}", basis.name);
    }
}

#[test]
fn transform_throughput() {
    // 2 mm-like stack scaled to a quarter of the voxels (nv 1.04 M, 40 frames × 7 encodes, 18 coefficients)
    let (n, nr, ne, nv) = (20usize, 2usize, 7usize, 1usize << 20);
    let b = TemporalBasis::cardiac_fourier(n, nr, 4);
    let nf = n * nr;
    let c: Vec<C32> = (0..b.nk * ne * nv).map(|i| C32::new((i % 977) as f32, (i % 131) as f32)).collect();
    let mut x = vec![C32::new(0.0, 0.0); nf * ne * nv];
    apply_sets(&b.b, nf, b.nk, ne, nv, &c, &mut x);
    let t = Instant::now();
    apply_sets(&b.b, nf, b.nk, ne, nv, &c, &mut x);
    let ts = t.elapsed().as_secs_f64();
    let bt = b.transpose();
    let mut c2 = vec![C32::new(0.0, 0.0); c.len()];
    let t = Instant::now();
    apply_sets(&bt, b.nk, nf, ne, nv, &x, &mut c2);
    let ta = t.elapsed().as_secs_f64();
    let gb = ((nf + b.nk) * ne * nv * 8) as f64 / 1e9;
    println!("T: {ts:.3}s, Tᵀ: {ta:.3}s for {:.2} GB moved each ({:.1} / {:.1} GB/s); at 2 mm (×4 voxels) ≈ {:.2}s / {:.2}s", gb, gb / ts, gb / ta, 4.0 * ts, 4.0 * ta);
}
