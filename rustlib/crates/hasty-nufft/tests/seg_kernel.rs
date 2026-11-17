//! GPU: gridding window of the segmented NUFFT. Kaiser–Bessel evaluated by Horner polynomials (`HASTY_SEG_KERNEL=kb`,
//! `set_thread_seg_kernel`) against ES (FINUFFT form, derivative clamp at |z| > 0.98), both against an f64 direct sum
//! (research/temporal_operator.md §20):
//! - plain operator (L = 1, order 0), forward and adjoint, W 6 / 8 / 10, σ 1.25;
//! - the ΔC-corrected operator (orders 1 and 2: the window's derivatives up to order 2 / 4) at a small and a moderate
//!   per-sample chirp, where the derivative terms (not the Taylor truncation) decide the error at small W.
//! On these operator-level (rms over samples) errors KB must be within 1.1 × ES + 5e-6 (the f32 floor) in every case; it
//! is 0.63× at W 8. The window-level max-norm derivative errors, where KB is worse, are checked in kernel.rs.
//! Fine-grid size selection (`GridRule`): a larger grid (every σ_a ≥ σ) with β set for σ_eff = min_a σ_a is at least
//! as accurate as the default grid, for both windows; the measured (`Auto`) choice never goes below the default size.
//!   cargo test --release -p hasty-nufft --test seg_kernel -- --nocapture --test-threads 1

use hasty_cuda::{Device, C32};
use hasty_nufft::{grid_default_size, set_thread_grid_rule, set_thread_seg_kernel, GridRule, SegKernel, SegNufft, SegNufftOpts};
use num_complex::Complex64;

struct R(u64);
impl R {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

struct Prob {
    dims: [usize; 3],
    res: f64,
    nc: usize,
    x: Vec<Complex64>,
    yd: Vec<Complex64>,
    smaps: Vec<Complex64>,
    k: [Vec<f64>; 3],
    /// per-sample residual quadratic phase coefficients (rad/m²), as SegNufft's `dc`
    dc: Vec<[f64; 6]>,
}

fn prob(m: usize, chi: f64, seed: u64) -> Prob {
    let dims = [20usize, 18, 14];
    let res = 3e-3;
    let nv: usize = dims.iter().product();
    let nc = 2;
    let mut r = R(seed);
    let x = (0..nv).map(|_| Complex64::new(r.u(), r.u())).collect();
    let yd = (0..nc * m).map(|_| Complex64::new(r.u(), r.u())).collect();
    let mut smaps = vec![];
    for c in 0..nc {
        for v in 0..nv {
            let p = rho(dims, res, v);
            smaps.push(Complex64::from_polar(1.0 - 4.0 * (p[0] - 0.01 * c as f64).powi(2) / 0.03f64.powi(2) * 0.1, 30.0 * (c as f64 + 1.0) * p[0] + 20.0 * p[1]));
        }
    }
    let k = [0, 1, 2].map(|_| (0..m).map(|_| r.u() * 2.0 * std::f64::consts::PI * 0.98).collect());
    // |ρᵀ ΔC ρ| ≲ chi over the FOV (half extent ≈ 0.03 m)
    let r2 = 0.03f64 * 0.03;
    let dc = (0..m).map(|_| [0, 1, 2, 3, 4, 5].map(|_| r.u() * 2.0 * chi / (3.0 * r2))).collect();
    Prob { dims, res, nc, x, yd, smaps, k, dc }
}

fn rho(d: [usize; 3], res: f64, v: usize) -> [f64; 3] {
    let (i, j, k) = (v % d[0], (v / d[0]) % d[1], v / (d[0] * d[1]));
    [(i as f64 - (d[0] / 2) as f64) * res, (j as f64 - (d[1] / 2) as f64) * res, (k as f64 - (d[2] / 2) as f64) * res]
}

/// f64 direct sums: (A x, Aᴴ yd), phase k·idx + ρᵀ ΔC_j ρ (`chirp` = false: plain)
fn direct(p: &Prob, chirp: bool) -> (Vec<Complex64>, Vec<Complex64>) {
    let nv = p.x.len();
    let m = p.k[0].len();
    let d = p.dims;
    let mut y = vec![Complex64::new(0.0, 0.0); p.nc * m];
    let mut xa = vec![Complex64::new(0.0, 0.0); nv];
    for v in 0..nv {
        let (i, jj, kk) = (v % d[0], (v / d[0]) % d[1], v / (d[0] * d[1]));
        let idx = [i as f64 - (d[0] / 2) as f64, jj as f64 - (d[1] / 2) as f64, kk as f64 - (d[2] / 2) as f64];
        let q = rho(d, p.res, v);
        let sx: Vec<Complex64> = (0..p.nc).map(|c| p.smaps[c * nv + v] * p.x[v]).collect();
        let mut acc = Complex64::new(0.0, 0.0);
        for j in 0..m {
            let mut ph = p.k[0][j] * idx[0] + p.k[1][j] * idx[1] + p.k[2][j] * idx[2];
            if chirp {
                let c = &p.dc[j];
                ph += c[0] * q[0] * q[0] + c[1] * q[1] * q[1] + c[2] * q[2] * q[2] + 2.0 * (c[3] * q[0] * q[1] + c[4] * q[0] * q[2] + c[5] * q[1] * q[2]);
            }
            let e = Complex64::from_polar(1.0, -ph);
            for c in 0..p.nc {
                y[c * m + j] += sx[c] * e;
                acc += p.smaps[c * nv + v].conj() * e.conj() * p.yd[c * m + j];
            }
        }
        xa[v] = acc;
    }
    (y, xa)
}

fn rel(a: &[C32], b: &[Complex64]) -> f64 {
    let (mut n, mut d) = (0.0, 0.0);
    for (p, q) in a.iter().zip(b) {
        n += (Complex64::new(p.re as f64, p.im as f64) - q).norm_sqr();
        d += q.norm_sqr();
    }
    (n / d).sqrt()
}

fn c32(v: &[Complex64]) -> Vec<C32> {
    v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
}

/// (forward, adjoint) relative errors of one engine (window `k`, width `w`, correction `order`) against the direct sums
fn run(dev: &Device, p: &Prob, want: &(Vec<Complex64>, Vec<Complex64>), k: SegKernel, w: usize, order: u32) -> (f64, f64) {
    run_g(dev, p, want, k, w, order, GridRule::Fixed).0
}

/// as [`run`] with a fine-grid rule; also returns the engine's grid and ES β
#[allow(clippy::too_many_arguments)]
fn run_g(dev: &Device, p: &Prob, want: &(Vec<Complex64>, Vec<Complex64>), k: SegKernel, w: usize, order: u32, g: GridRule) -> ((f64, f64), [usize; 3], f64) {
    let st = dev.default_stream();
    let nv = p.x.len();
    let m = p.k[0].len();
    set_thread_seg_kernel(Some(k));
    set_thread_grid_rule(Some(g));
    let mut op = SegNufft::new(dev, p.dims, [p.res; 3], p.nc, SegNufftOpts { sigma: 1.25, width: w, nseg: 1, order, frames: 1, bspline: 1, max_grid_bytes: 1 << 30 }).unwrap();
    set_thread_seg_kernel(None);
    set_thread_grid_rule(None);
    assert_eq!(op.kernel_kind(), k);
    let (nf, beta) = ([op.plan.nf[0], op.plan.nf[1], op.plan.nf[2]], op.plan.ker.beta);
    op.set_maps(&st, &c32(&p.smaps)).unwrap();
    let seg = op.segments(&st, &vec![C32::new(1.0, 0.0); nv], &[C32::new(1.0, 0.0); 2], 0.0, 1.0).unwrap();
    let co: [Vec<f32>; 3] = [0, 1, 2].map(|a| p.k[a].iter().map(|&v| v as f32).collect());
    let mut dc = vec![0f32; 6 * m];
    if order > 0 {
        for j in 0..m {
            for q in 0..6 {
                dc[q * m + j] = p.dc[j][q] as f32;
            }
        }
    }
    let dcs: &[f32] = if order > 0 { &dc } else { &[] };
    let pts = op.points(&st, [&co[0], &co[1], &co[2]], &vec![0f32; m], &vec![0f32; m], dcs, None).unwrap();
    let (dx, dy) = (st.clone_htod(&c32(&p.x)).unwrap(), st.clone_htod(&c32(&p.yd)).unwrap());
    let mut y = st.alloc_zeros::<C32>(p.nc * m).unwrap();
    let mut xa = st.alloc_zeros::<C32>(nv).unwrap();
    op.forward(&st, &seg, &pts, &dx, &mut y).unwrap();
    op.adjoint(&st, &seg, &pts, &dy, &mut xa).unwrap();
    ((rel(&st.clone_dtoh(&y).unwrap(), &want.0), rel(&st.clone_dtoh(&xa).unwrap(), &want.1)), nf, beta)
}

#[test]
fn kb_window_at_least_as_accurate_as_es() {
    let dev = Device::new(0).expect("CUDA device");
    let m = 3000;
    // (order, chirp amplitude in rad): plain; ΔC small (derivative terms dominate) and moderate (Taylor truncation shows)
    let cases = [(0u32, 0.0f64), (1, 0.05), (2, 0.05), (2, 0.4)];
    let mut worst = 0.0f64;
    for (order, chi) in cases {
        let p = prob(m, chi.max(1e-9), 7 + order as u64);
        let want = direct(&p, order > 0);
        for w in [6usize, 8, 10] {
            let es = run(&dev, &p, &want, SegKernel::Es, w, order);
            let kb = run(&dev, &p, &want, SegKernel::Kb, w, order);
            eprintln!("order {order}, χ {chi:.2} rad, W {w:>2}: ES forward {:.2e} adjoint {:.2e} | KB forward {:.2e} adjoint {:.2e} (KB / ES {:.2} / {:.2})", es.0, es.1, kb.0, kb.1, kb.0 / es.0, kb.1 / es.1);
            // f32 floor of these sums ≈ 1e-5 (forward) / 2.5e-5 (adjoint): both windows reach it at W = 10
            for (e, k) in [(es.0, kb.0), (es.1, kb.1)] {
                worst = worst.max(k / e.max(5e-6));
                assert!(k <= 1.1 * e + 5e-6, "KB less accurate than ES: order {order}, χ {chi}, W {w}: {k:.2e} vs {e:.2e}");
            }
        }
    }
    eprintln!("worst KB / ES error ratio (floor 5e-6): {worst:.2}");
}

#[test]
fn larger_grid_with_effective_sigma_is_not_less_accurate() {
    let dev = Device::new(0).expect("CUDA device");
    let m = 3000;
    let dims = [20usize, 18, 14];
    for w in [6usize, 8] {
        let def = dims.map(|n| grid_default_size(n, 1.25, w));
        // larger grids: a power-of-two-like pick and a bigger one
        let bigger = [def.map(|n| hasty_nufft::next_smooth_even(n + 1)), def.map(|n| hasty_nufft::next_smooth_even(n * 5 / 4))];
        for (order, chi) in [(0u32, 0.0f64), (2, 0.05)] {
            let p = prob(m, chi.max(1e-9), 31 + order as u64);
            let want = direct(&p, order > 0);
            for k in [SegKernel::Es, SegKernel::Kb] {
                let (e0, nf0, b0) = run_g(&dev, &p, &want, k, w, order, GridRule::Fixed);
                assert_eq!(nf0, def);
                for g in bigger {
                    let (e1, nf1, b1) = run_g(&dev, &p, &want, k, w, order, GridRule::Explicit(g));
                    assert_eq!(nf1, g);
                    eprintln!("W {w}, order {order}, {k:?}: default {def:?} (ES β {b0:.3}) forward {:.2e} adjoint {:.2e} | grid {g:?} (σ_eff {:.3}, ES β {b1:.3}) forward {:.2e} adjoint {:.2e}", e0.0, e0.1, hasty_nufft::sigma_eff(g, dims), e1.0, e1.1);
                    assert!(b1 > b0, "β not raised for the larger σ_eff");
                    assert!(e1.0 <= 1.05 * e0.0 + 5e-6 && e1.1 <= 1.05 * e0.1 + 5e-6, "larger grid less accurate: {e1:?} vs {e0:?}");
                }
            }
        }
    }
    // the measured choice keeps every axis at or above the default size and is consistent within the process
    let p = prob(500, 1e-9, 5);
    let want = direct(&p, false);
    let (_, a1, _) = run_g(&dev, &p, &want, SegKernel::Es, 8, 0, GridRule::Auto);
    let (_, a2, _) = run_g(&dev, &p, &want, SegKernel::Es, 8, 0, GridRule::Auto);
    let def = dims.map(|n| grid_default_size(n, 1.25, 8));
    eprintln!("auto grid for {dims:?}, W 8: {a1:?} (default {def:?})");
    assert!((0..3).all(|a| a1[a] >= def[a]) && a1 == a2);
}
