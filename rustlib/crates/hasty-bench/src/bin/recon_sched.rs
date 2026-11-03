//! Fidelity schedules for the data-consistency gradient in a P2nP-style preconditioned plug-and-play loop (Hong, Xu, Hu, Fessler 2024).
//!
//! * Model: multi-coil, **no density compensation** (the preconditioner replaces it), coil noise covariance Psi handled by whitening the data
//!   and the sensitivity maps (S~ = Psi^{-1/2} S, y~ = Psi^{-1/2} y) so that the normal operator is  N = sum_c conj(S~_c) A^H A S~_c.
//! * Preconditioners: none (PnP-ISTA), fixed F-1 (P = 2 - aN), fixed F-Cheb (P = 4 - 10/3 aN), dynamic ZMSHR1 (zero-memory SHR1 from gradient
//!   differences, 1 operator application per iteration). Operator scaled so that ||N|| = 1, step a = 1 (the paper uses a = 1/||A^H A||).
//! * Denoiser: Haar soft-threshold with threshold proportional to the mean detail magnitude => normalisation-equivariant D(mu x + c) = mu D(x) + c.
//! * Gradient strategies: exact; cheap surrogate only; lagged-correction (SVRG-like) with adaptive anchor refresh.
//! Exact model = dense A (K x V) including the concomitant phase; surrogates = GPU `hasty-toeplitz` operators.

use cudarc::driver::CudaSlice;
use hasty_cuda::{Device, C32};
use hasty_toeplitz::*;
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;

const N: usize = 16;
const V: usize = N * N * N;
const DIMS: [usize; 3] = [N, N, N];
const NC: usize = 4;
const THREADS: usize = 8;

type Vc = Vec<Complex64>;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        let mut s = self.0;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.0 = s;
        (s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn norm(v: &Vc) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}
fn sub(a: &Vc, b: &Vc) -> Vc {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}
/// real inner product of complex vectors viewed as R^{2n}
fn rdot(a: &Vc, b: &Vc) -> f64 {
    a.iter().zip(b).map(|(x, y)| x.re * y.re + x.im * y.im).sum()
}

// ----------------------------------------------------------------------- normalisation-equivariant Haar denoiser
fn haar_1d(v: &mut [Complex64], n: usize, inverse: bool) {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let mut tmp = vec![Complex64::new(0.0, 0.0); n];
    if !inverse {
        for i in 0..n / 2 {
            tmp[i] = (v[2 * i] + v[2 * i + 1]) * h;
            tmp[n / 2 + i] = (v[2 * i] - v[2 * i + 1]) * h;
        }
    } else {
        for i in 0..n / 2 {
            tmp[2 * i] = (v[i] + v[n / 2 + i]) * h;
            tmp[2 * i + 1] = (v[i] - v[n / 2 + i]) * h;
        }
    }
    v[..n].copy_from_slice(&tmp);
}
fn haar_3d(x: &mut Vc, levels: usize, inverse: bool) {
    let lv: Vec<usize> = if inverse { (0..levels).rev().collect() } else { (0..levels).collect() };
    for l in lv {
        let n = N >> l;
        for axis in 0..3 {
            let stride = [1usize, N, N * N][axis];
            let (o1, o2) = match axis {
                0 => (N, N * N),
                1 => (1, N * N),
                _ => (1, N),
            };
            for a in 0..n {
                for b in 0..n {
                    let base = a * o1 + b * o2;
                    let mut line: Vec<Complex64> = (0..n).map(|i| x[base + i * stride]).collect();
                    haar_1d(&mut line, n, inverse);
                    for i in 0..n {
                        x[base + i * stride] = line[i];
                    }
                }
            }
        }
    }
}
/// threshold = kappa * mean |detail coefficient|: D(mu x) = mu D(x) and the coarse band (constants) is untouched => normalisation-equivariant
fn denoise(x: &Vc, kappa: f64) -> Vc {
    let mut c = x.clone();
    haar_3d(&mut c, 3, false);
    let coarse = N >> 3;
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for z in 0..N {
        for y in 0..N {
            for xx in 0..N {
                if !(z < coarse && y < coarse && xx < coarse) {
                    sum += c[(z * N + y) * N + xx].norm();
                    cnt += 1;
                }
            }
        }
    }
    let thr = kappa * sum / cnt as f64;
    for z in 0..N {
        for y in 0..N {
            for xx in 0..N {
                if z < coarse && y < coarse && xx < coarse {
                    continue;
                }
                let i = (z * N + y) * N + xx;
                let m = c[i].norm();
                c[i] = if m > thr { c[i] * ((m - thr) / m) } else { Complex64::new(0.0, 0.0) };
            }
        }
    }
    haar_3d(&mut c, 3, true);
    c
}

// ----------------------------------------------------------------------- problem
struct Problem {
    a: Vec<Complex64>, // K x V, no density weighting
    k: usize,
    s: SampleSetHost,
    smaps: Vec<Vc>, // whitened coil maps S~_c
}

impl Problem {
    fn av(&self, x: &Vc) -> Vc {
        let mut out = vec![Complex64::new(0.0, 0.0); self.k];
        let chunk = self.k.div_ceil(THREADS);
        std::thread::scope(|sc| {
            for (t, o) in out.chunks_mut(chunk).enumerate() {
                let a = &self.a;
                sc.spawn(move || {
                    for (i, oi) in o.iter_mut().enumerate() {
                        let j = t * chunk + i;
                        *oi = (0..V).map(|r| a[j * V + r] * x[r]).sum();
                    }
                });
            }
        });
        out
    }
    fn ahv(&self, y: &Vc) -> Vc {
        let mut out = vec![Complex64::new(0.0, 0.0); V];
        let chunk = V.div_ceil(THREADS);
        std::thread::scope(|sc| {
            for (t, o) in out.chunks_mut(chunk).enumerate() {
                let a = &self.a;
                let k = self.k;
                sc.spawn(move || {
                    for (i, oi) in o.iter_mut().enumerate() {
                        let r = t * chunk + i;
                        *oi = (0..k).map(|j| a[j * V + r].conj() * y[j]).sum();
                    }
                });
            }
        });
        out
    }
    fn normal_single(&self, x: &Vc) -> Vc {
        self.ahv(&self.av(x))
    }
    /// sum_c conj(S_c) A^H A (S_c x)
    fn normal_exact(&self, x: &Vc) -> Vc {
        let mut out = vec![Complex64::new(0.0, 0.0); V];
        for s in &self.smaps {
            let t: Vc = x.iter().zip(s).map(|(a, b)| a * b).collect();
            let n = self.normal_single(&t);
            for i in 0..V {
                out[i] += s[i].conj() * n[i];
            }
        }
        out
    }
}

fn make_problem(phi_max: f64) -> (Problem, Vc, Vec<Vc>, DMatrix<f64>) {
    let (spokes, t) = (110usize, 30usize);
    let m = spokes * t;
    let mut r = Rng(7);
    let mut k = [vec![0f64; m], vec![0f64; m], vec![0f64; m]];
    let mut w: [Vec<f64>; 4] = [vec![0f64; m], vec![0f64; m], vec![0f64; m], vec![0f64; m]];
    for s in 0..spokes {
        let zc = r.u();
        let ph = (r.u() + 1.0) * std::f64::consts::PI;
        let st = (1.0 - zc * zc).sqrt();
        let n = [st * ph.cos(), st * ph.sin(), zc];
        for tt in 0..t {
            let j = s * t + tt;
            let u = (tt as f64 + 0.5) / t as f64;
            for a in 0..3 {
                k[a][j] = 2.8 * u * n[a];
            }
            let g = u * u * u; // accumulated g^2 grows during the readout
            w[0][j] = g * (n[0] * n[0] + n[1] * n[1]);
            w[1][j] = g * n[2] * n[2] / 4.0;
            w[2][j] = -g * n[0] * n[2];
            w[3][j] = -g * n[1] * n[2];
        }
    }
    let mut pmax: f64 = 0.0;
    for j in 0..m {
        for &(x, y, z) in &[(1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 1.0)] {
            let p: f64 = (0..4).map(|q| w[q][j] * m_eval(q, x, y, z)).sum();
            pmax = pmax.max(p.abs());
        }
    }
    let sc = phi_max / pmax;
    for q in 0..4 {
        for j in 0..m {
            w[q][j] *= sc;
        }
    }
    let s = SampleSetHost { k, wt: vec![1.0; m], w };
    let mut a = vec![Complex64::new(0.0, 0.0); m * V];
    for j in 0..m {
        for rr in 0..V {
            let (ix, iy, iz) = (rr % N, (rr / N) % N, rr / (N * N));
            let ph = s.k[0][j] * ix as f64 + s.k[1][j] * iy as f64 + s.k[2][j] * iz as f64 + s.phi(j, xi(ix, N), xi(iy, N), xi(iz, N));
            a[j * V + rr] = Complex64::from_polar(1.0, -ph);
        }
    }
    // coil maps: Gaussian lobes at the corners of the FOV with smooth phase
    let centers = [[1.0, 1.0, 0.0], [-1.0, 1.0, 0.5], [-1.0, -1.0, -0.5], [1.0, -1.0, 0.0]];
    let raw: Vec<Vc> = (0..NC)
        .map(|c| {
            (0..V)
                .map(|rr| {
                    let p = [xi(rr % N, N), xi((rr / N) % N, N), xi(rr / (N * N), N)];
                    let d2: f64 = (0..3).map(|a| (p[a] - centers[c][a]).powi(2)).sum();
                    Complex64::from_polar((-d2 / 1.6).exp(), 0.5 * (c as f64 + 1.0) * (p[0] - p[1]) + 0.3 * p[2])
                })
                .collect()
        })
        .collect();
    // noise covariance (correlated coils) and its inverse square root
    let mut psi = DMatrix::<f64>::identity(NC, NC) * 1.0;
    for i in 0..NC {
        for j in 0..NC {
            if i != j {
                psi[(i, j)] = 0.35 / (1.0 + (i as f64 - j as f64).abs());
            }
        }
    }
    let eig = SymmetricEigen::new(psi.clone());
    let mut wmat = DMatrix::<f64>::zeros(NC, NC);
    for q in 0..NC {
        let vq = eig.eigenvectors.column(q);
        wmat += (vq * vq.transpose()) * (1.0 / eig.eigenvalues[q].sqrt());
    }
    let smaps: Vec<Vc> = (0..NC).map(|c| (0..V).map(|rr| (0..NC).map(|cc| raw[cc][rr] * wmat[(c, cc)]).sum()).collect()).collect();
    (Problem { a, k: m, s, smaps }, vec![], raw, psi)
}

fn phantom() -> Vc {
    let mut x = vec![Complex64::new(0.0, 0.0); V];
    let blobs = [([0.0, 0.0, 0.0], 0.45, 1.0), ([0.6, 0.5, -0.3], 0.3, 0.8), ([-0.55, -0.6, 0.5], 0.28, -0.6), ([0.1, -0.7, -0.6], 0.25, 0.7), ([-0.6, 0.55, -0.5], 0.25, 0.5)];
    for z in 0..N {
        for y in 0..N {
            for xx in 0..N {
                let p = [xi(xx, N), xi(y, N), xi(z, N)];
                let mut val = 0.0;
                for (c, rad, amp) in &blobs {
                    let d2: f64 = (0..3).map(|a| (p[a] - c[a]).powi(2)).sum();
                    if d2 < rad * rad {
                        val += amp * (1.0 - d2 / (rad * rad));
                    }
                }
                x[(z * N + y) * N + xx] = Complex64::from_polar(val, 0.3 * (p[0] + p[2]));
            }
        }
    }
    x
}

// ----------------------------------------------------------------------- GPU surrogate operators (single-coil A^H A, multi-coil through the maps)
struct GpuOp {
    name: String,
    cost: f64, // plain-Toeplitz application units (per coil)
    op: MixOp<f32>,
}

fn build_ops(dev: &Device, p: &Problem) -> Vec<GpuOp> {
    let stream = dev.default_stream();
    let up = |v: &Vec<f64>| stream.clone_htod(&v.iter().map(|&x| x as f32).collect::<Vec<f32>>()).unwrap();
    let (kx, ky, kz) = (up(&p.s.k[0]), up(&p.s.k[1]), up(&p.s.k[2]));
    let wt = up(&p.s.wt);
    let w = [up(&p.s.w[0]), up(&p.s.w[1]), up(&p.s.w[2]), up(&p.s.w[3])];
    let m = p.k;
    let mut out = Vec::new();
    let mut push = |spec: MixSpec, cost: f64| {
        let extra: Vec<CudaSlice<C32>> = vec![];
        let samples = Samples::<f32> { m, coords: [&kx, &ky, &kz], wt: &wt, w: [&w[0], &w[1], &w[2], &w[3]], extra: &extra };
        let kernels = build_kernels(dev, &stream, DIMS, &samples, &spec.kernels, &KernelBuildOpts { tol: 1e-7, sigma: 2.0 }).unwrap();
        let name = spec.label.clone();
        out.push(GpuOp { name, cost, op: MixOp::<f32>::new(dev, spec, DIMS, kernels, vec![]).unwrap() });
    };
    push(plain(), 1.0); // measured N=128 ratios (toeplitz_bench): fused plain 1, commutator n=1 6.7, n=2 32
    push(weyl(1), 6.7);
    push(weyl(2), 32.0);
    out
}

fn apply_single(dev: &Device, op: &mut GpuOp, x: &Vc) -> Vc {
    let stream = dev.default_stream();
    let dx = stream.clone_htod(&x.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
    let mut dout = stream.alloc_zeros::<C32>(x.len()).unwrap();
    op.op.apply(&stream, &dx, &mut dout, false).unwrap();
    stream.clone_dtoh(&dout).unwrap().iter().map(|c| Complex64::new(c.re as f64, c.im as f64)).collect()
}

fn apply_multi(dev: &Device, op: &mut GpuOp, smaps: &[Vc], x: &Vc) -> Vc {
    let mut out = vec![Complex64::new(0.0, 0.0); V];
    for s in smaps {
        let t: Vc = x.iter().zip(s).map(|(a, b)| a * b).collect();
        let n = apply_single(dev, op, &t);
        for i in 0..V {
            out[i] += s[i].conj() * n[i];
        }
    }
    out
}

// ----------------------------------------------------------------------- algorithm
#[derive(Clone, Copy, Debug, PartialEq)]
enum Precond {
    None,
    F1,
    FCheb,
    Dynamic,
}

#[derive(Clone, Copy, Debug)]
enum Grad {
    Exact,
    Cheap(usize),
    /// lagged correction with adaptive anchor: (surrogate index, tau)
    Lagged(usize, f64),
}

#[derive(Clone, Copy, Debug)]
enum PrecOp {
    Exact,
    Cheap(usize),
}

struct Ctx<'a> {
    dev: &'a Device,
    p: &'a Problem,
    ops: &'a mut Vec<GpuOp>,
    lam: f64, // 1/||N|| scaling applied to the operator and the data term
    exact_cost: f64,
}

impl Ctx<'_> {
    fn nexact(&self, x: &Vc) -> Vc {
        self.p.normal_exact(x).iter().map(|c| c * self.lam).collect()
    }
    fn ncheap(&mut self, l: usize, x: &Vc) -> Vc {
        apply_multi(self.dev, &mut self.ops[l], &self.p.smaps, x).iter().map(|c| c * self.lam).collect()
    }
}

/// ZMSHR1 (Algorithm 2 of the paper): returns a function applying P_k
fn zmshr1(s: &Vc, m: &Vc, delta: f64, th1: f64, th2: f64) -> (f64, Option<(Vc, f64)>) {
    let ss = rdot(s, s);
    let sm = rdot(s, m);
    let mm = rdot(m, m);
    let mut best = 1.0;
    for i in 0..=2000 {
        let a = i as f64 / 2000.0;
        let sv = a * ss + (1.0 - a) * sm;
        let vv = a * a * ss + 2.0 * a * (1.0 - a) * sm + (1.0 - a) * (1.0 - a) * mm;
        if sv > 0.0 && sv / ss >= th1 && vv / sv <= th2 {
            best = a;
            break;
        }
    }
    let a = best;
    let v: Vc = s.iter().zip(m).map(|(s, m)| s * a + m * (1.0 - a)).collect();
    let sv = rdot(s, &v);
    let vv = rdot(&v, &v);
    let tau = ss / sv - ((ss / sv).powi(2) - ss / vv).max(0.0).sqrt();
    let u: Vc = s.iter().zip(&v).map(|(s, v)| s - v * tau).collect();
    let uv = rdot(&u, &v);
    if uv <= delta * norm(&u) * norm(&v) {
        (tau, None)
    } else {
        (tau, Some((u, uv)))
    }
}

struct Trace {
    nrmse: Vec<f64>,
    cost: Vec<f64>,
    n_exact: usize,
}

#[allow(clippy::too_many_arguments)]
fn run(ctx: &mut Ctx, ahy: &Vc, x_true: &Vc, grad: Grad, prec: Precond, prec_op: PrecOp, iters: usize, kappa: f64) -> Trace {
    let mut x = vec![Complex64::new(0.0, 0.0); V];
    let mut tr = Trace { nrmse: vec![], cost: vec![], n_exact: 0 };
    let mut cost = 0.0;
    let ncoil = NC as f64;
    let mut anchor: Option<(Vc, Vc, Vc)> = None; // (x_a, N_exact x_a, N_l x_a)
    let (mut x_prev, mut g_prev): (Option<Vc>, Option<Vc>) = (None, None);
    let ahy_s: Vc = ahy.iter().map(|c| c * ctx.lam).collect();
    for _k in 0..iters {
        // gradient of the (scaled) data term:  N x - A^H y
        let nx = match grad {
            Grad::Exact => {
                tr.n_exact += 1;
                cost += ctx.exact_cost * ncoil;
                ctx.nexact(&x)
            }
            Grad::Cheap(l) => {
                cost += ctx.ops[l].cost * ncoil;
                ctx.ncheap(l, &x)
            }
            Grad::Lagged(l, tau) => {
                let need = match &anchor {
                    None => true,
                    Some((xa, _, _)) => norm(&sub(&x, xa)) > tau * norm(&x),
                };
                if need {
                    tr.n_exact += 1;
                    cost += (ctx.exact_cost + ctx.ops[l].cost) * ncoil;
                    let ne = ctx.nexact(&x);
                    let ns = ctx.ncheap(l, &x);
                    anchor = Some((x.clone(), ne, ns));
                }
                cost += ctx.ops[l].cost * ncoil;
                let cur = ctx.ncheap(l, &x);
                let (_, ne, ns) = anchor.as_ref().unwrap();
                (0..V).map(|i| ne[i] + cur[i] - ns[i]).collect()
            }
        };
        let g = sub(&nx, &ahy_s);
        let pg: Vc = match prec {
            Precond::None => g.clone(),
            Precond::F1 | Precond::FCheb => {
                // P = p1 - p2 N  with (p1,p2) = (2,1) or (4,10/3); one extra operator application
                let (p1, p2) = if prec == Precond::F1 { (2.0, 1.0) } else { (4.0, 10.0 / 3.0) };
                let ng = match prec_op {
                    PrecOp::Exact => {
                        cost += ctx.exact_cost * ncoil;
                        ctx.nexact(&g)
                    }
                    PrecOp::Cheap(l) => {
                        cost += ctx.ops[l].cost * ncoil;
                        ctx.ncheap(l, &g)
                    }
                };
                (0..V).map(|i| g[i] * p1 - ng[i] * p2).collect()
            }
            Precond::Dynamic => match (&x_prev, &g_prev) {
                (Some(xp), Some(gp)) => {
                    let s = sub(&x, xp);
                    let m = sub(&g, gp);
                    let (tau, u) = zmshr1(&s, &m, 1e-8, 2e-6, 200.0);
                    match u {
                        None => g.iter().map(|c| c * tau).collect(),
                        Some((u, uv)) => {
                            let c = rdot(&u, &g) / uv;
                            (0..V).map(|i| g[i] * tau + u[i] * c).collect()
                        }
                    }
                }
                _ => g.clone(),
            },
        };
        x_prev = Some(x.clone());
        g_prev = Some(g.clone());
        let step: Vc = (0..V).map(|i| x[i] - pg[i]).collect();
        x = denoise(&step, kappa);
        tr.nrmse.push(norm(&sub(&x, x_true)) / norm(x_true));
        tr.cost.push(cost);
    }
    tr
}

fn power(p: &Problem, its: usize) -> f64 {
    let mut v: Vc = (0..V).map(|i| Complex64::new(1.0 + 0.1 * (i % 7) as f64, 0.2 * (i % 3) as f64)).collect();
    let mut l = 0.0;
    for _ in 0..its {
        let nv = norm(&v);
        v = v.iter().map(|c| c / nv).collect();
        let w = p.normal_exact(&v);
        l = rdot(&v, &w);
        v = w;
    }
    l
}

fn main() {
    let phi_max: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.5);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let kappa: f64 = std::env::var("KAPPA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6);
    let dev = Device::new(0).unwrap();
    println!("N={N}^3, {NC} coils with correlated noise (whitened), K=3300 samples, NO density compensation, max|Phi|={phi_max} rad, kappa={kappa}");
    let (p, _, raw, psi) = make_problem(phi_max);
    let mut ops = build_ops(&dev, &p);
    let x_true = phantom();
    // data: y_c = A (S_c x) + noise with covariance Psi  -> whiten
    let mut rng = Rng(3);
    let clean: Vec<Vc> = raw.iter().map(|s| p.av(&x_true.iter().zip(s).map(|(a, b)| a * b).collect())).collect();
    let sig = clean.iter().map(|c| norm(c) / (c.len() as f64).sqrt()).fold(0.0, f64::max) * 0.03;
    let chol = psi.clone().cholesky().unwrap().l();
    let noise: Vec<Vc> = {
        let white: Vec<Vc> = (0..NC).map(|_| (0..p.k).map(|_| Complex64::new(rng.u(), rng.u()) * (sig * 1.7)).collect()).collect();
        (0..NC).map(|c| (0..p.k).map(|j| (0..NC).map(|cc| white[cc][j] * chol[(c, cc)]).sum()).collect()).collect()
    };
    let y: Vec<Vc> = (0..NC).map(|c| clean[c].iter().zip(&noise[c]).map(|(a, b)| a + b).collect()).collect();
    // whitening of the data with the same Psi^{-1/2}
    let eig = SymmetricEigen::new(psi.clone());
    let mut wmat = DMatrix::<f64>::zeros(NC, NC);
    for q in 0..NC {
        let vq = eig.eigenvectors.column(q);
        wmat += (vq * vq.transpose()) * (1.0 / eig.eigenvalues[q].sqrt());
    }
    let yw: Vec<Vc> = (0..NC).map(|c| (0..p.k).map(|j| (0..NC).map(|cc| y[cc][j] * wmat[(c, cc)]).sum()).collect()).collect();
    // A^H y = sum_c conj(S~_c) A^H y~_c
    let mut ahy = vec![Complex64::new(0.0, 0.0); V];
    for c in 0..NC {
        let t = p.ahv(&yw[c]);
        for i in 0..V {
            ahy[i] += p.smaps[c][i].conj() * t[i];
        }
    }
    let lmax = power(&p, 20);
    let lam = 1.0 / lmax;
    println!("||N|| ~ {lmax:.1}; operators scaled by 1/||N||");
    {
        let ne = p.normal_exact(&x_true);
        for o in ops.iter_mut() {
            let n = apply_multi(&dev, o, &p.smaps, &x_true);
            println!("  surrogate {:<18} rel.err vs exact normal op on the phantom: {:.2e}   cost {:.1}", o.name, norm(&sub(&n, &ne)) / norm(&ne), o.cost);
        }
    }
    let exact_cost: f64 = std::env::var("EXACT_COST").ok().and_then(|v| v.parse().ok()).unwrap_or(30.0);
    let mut ctx = Ctx { dev: &dev, p: &p, ops: &mut ops, lam, exact_cost };
    let cases: Vec<(String, Grad, Precond, PrecOp)> = vec![
        ("PnP-ISTA, exact".into(), Grad::Exact, Precond::None, PrecOp::Exact),
        ("P2nP-F-1, exact".into(), Grad::Exact, Precond::F1, PrecOp::Exact),
        ("P2nP-F-Cheb, exact".into(), Grad::Exact, Precond::FCheb, PrecOp::Exact),
        ("P2nP-D, exact".into(), Grad::Exact, Precond::Dynamic, PrecOp::Exact),
        ("P2nP-F-Cheb, exact grad + plain prec".into(), Grad::Exact, Precond::FCheb, PrecOp::Cheap(0)),
        ("P2nP-D, plain only (no correction)".into(), Grad::Cheap(0), Precond::Dynamic, PrecOp::Exact),
        ("P2nP-D, commutator-1 only".into(), Grad::Cheap(1), Precond::Dynamic, PrecOp::Exact),
        ("P2nP-D, commutator-2 only".into(), Grad::Cheap(2), Precond::Dynamic, PrecOp::Exact),
        ("P2nP-D, lagged plain tau=0.1".into(), Grad::Lagged(0, 0.1), Precond::Dynamic, PrecOp::Exact),
        ("P2nP-D, lagged plain tau=0.05".into(), Grad::Lagged(0, 0.05), Precond::Dynamic, PrecOp::Exact),
        ("P2nP-D, lagged commut-1 tau=0.1".into(), Grad::Lagged(1, 0.1), Precond::Dynamic, PrecOp::Exact),
        ("P2nP-F-Cheb, lagged plain tau=.05 + plain prec".into(), Grad::Lagged(0, 0.05), Precond::FCheb, PrecOp::Cheap(0)),
    ];
    println!("\n{:<48} {:>9} {:>9} {:>9} {:>10} {:>11} {:>12}", format!("method (exact cost = {exact_cost} plain units/coil)"), "NRMSE k=5", "k=10", "k=20", format!("k={iters}"), "exact evals", "cost (units)");
    let only = std::env::var("ONLY").ok();
    for (name, g, pr, po) in cases {
        if only.as_ref().is_some_and(|o| !o.split(';').any(|t| name.contains(t))) {
            continue;
        }
        let tr = run(&mut ctx, &ahy, &x_true, g, pr, po, iters, kappa);
        let at = |k: usize| tr.nrmse[(k - 1).min(tr.nrmse.len() - 1)];
        println!("{:<48} {:>9.4} {:>9.4} {:>9.4} {:>10.4} {:>11} {:>12.0}", name, at(5), at(10), at(20), at(iters), tr.n_exact, tr.cost.last().unwrap());
    }
}
