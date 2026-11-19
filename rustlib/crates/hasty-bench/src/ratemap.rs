//! Ratemap estimation from multi-echo images (research/ratemap_estimation.md): per voxel the complex rate
//! z = R + iω (decay rate, off-resonance) and the water / fat amplitudes (six-peak fat), from echo images of one or
//! more respiratory bins.
//!
//! Two methods:
//! - `Voxel` (the original recon_b0 fit): every voxel on its own, the profile likelihood over |ω| ≤ ω_max picks the
//!   LM starts. With 50 undersampled echo images the per-voxel optimum often sits in the wrong water–fat basin.
//! - `Smooth` (default): global field first. A full-range voxel fit gives ω_v and a confidence c_v (signal energy ×
//!   model consistency); a robust (Cauchy IRLS) cubic B-spline fit of ω_v with a second-difference penalty gives a
//!   smooth field ω_s, coarse to fine (`knots_mm`); after every level each voxel is re-fitted on its demodulated
//!   signal y_d e^{+iω_s TE_d} with |ω − ω_s| ≤ `window_hz` (< the 434 Hz fat shift at 3 T: the swapped basin is
//!   outside the bounds) and the next level refits the field from those estimates. Uses no truth and no masks other
//!   than the signal threshold of the caller.

use hasty_cuda::Device;
use hasty_qfit::waterfat::gpu::WaterFatGpu;
use hasty_qfit::waterfat::{FatSpectrum, Species, WaterFat, WfConfig};
use num_complex::Complex64 as C;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Voxel,
    Smooth,
}

impl Method {
    pub fn parse(s: &str) -> Method {
        match s {
            "voxel" => Method::Voxel,
            "smooth" => Method::Smooth,
            _ => panic!("--fit voxel|smooth, got {s}"),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Method::Voxel => "voxel",
            Method::Smooth => "smooth",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cfg {
    pub method: Method,
    /// full-range search |ω| ≤ om_max_hz (Voxel method and the first pass of Smooth)
    pub om_max_hz: f64,
    /// demodulated window |ω − ω_s| ≤ window_hz (Smooth)
    pub window_hz: f64,
    /// knot spacings of the field spline, coarse → fine, mm
    pub knots_mm: Vec<f64>,
    /// second-difference penalty relative to the mean data curvature per coefficient
    pub lam: f64,
    /// Cauchy scale of the robust field fit, Hz
    pub huber_hz: f64,
    pub irls: usize,
    pub r2_max: f64,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg { method: Method::Smooth, om_max_hz: 880.0, window_hz: 200.0, knots_mm: vec![48.0, 32.0, 20.0], lam: 0.05, huber_hz: 40.0, irls: 6, r2_max: 1000.0 }
    }
}

impl Cfg {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"method": self.method.name(), "om_max_hz": self.om_max_hz, "window_hz": self.window_hz, "knots_mm": self.knots_mm,
                           "lam": self.lam, "cauchy_hz": self.huber_hz, "irls": self.irls, "r2_max": self.r2_max, "fat": "six_peak", "species": "WaterFat"})
    }
}

/// maps of one bin set, [bin][voxel] (x fastest), NaN where not fitted
pub struct Maps {
    pub om_hz: Vec<f32>,
    pub r2: Vec<f32>,
    pub ff: Vec<f32>,
    pub wm: Vec<f32>,
    pub fm: Vec<f32>,
    /// the smooth field ω_s (Smooth method; NaN for Voxel), defined on the whole grid
    pub om_smooth_hz: Vec<f32>,
    /// relative model residual ‖y − ŷ‖² / ‖y‖² of the final fit
    pub rel_res: Vec<f32>,
    pub log: Vec<String>,
}

/// separable cubic B-spline on the voxel grid (knot spacing h voxels)
struct Spline {
    dims: [usize; 3],
    nb: [usize; 3],
    /// per axis and grid index: first coefficient, 4 weights
    idx: [Vec<(usize, [f64; 4])>; 3],
}

fn bsp(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [(1.0 - t).powi(3) / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0]
}

impl Spline {
    fn new(dims: [usize; 3], h: f64) -> Spline {
        let nb = dims.map(|n| ((n as f64 - 1.0) / h).floor() as usize + 4);
        let idx = [0, 1, 2].map(|a| {
            (0..dims[a])
                .map(|i| {
                    let u = i as f64 / h;
                    let j = (u.floor() as usize).min(nb[a] - 4);
                    (j, bsp(u - j as f64))
                })
                .collect()
        });
        Spline { dims, nb, idx }
    }
    fn n(&self) -> usize {
        self.nb[0] * self.nb[1] * self.nb[2]
    }
    fn coords(&self, q: usize) -> [usize; 3] {
        [q % self.dims[0], (q / self.dims[0]) % self.dims[1], q / (self.dims[0] * self.dims[1])]
    }
    fn eval_at(&self, c: &[f64], q: usize) -> f64 {
        let [i, j, k] = self.coords(q);
        let (a, b, d) = (&self.idx[0][i], &self.idx[1][j], &self.idx[2][k]);
        let mut s = 0.0;
        for z in 0..4 {
            for y in 0..4 {
                let wzy = d.1[z] * b.1[y];
                let base = ((d.0 + z) * self.nb[1] + b.0 + y) * self.nb[0] + a.0;
                for x in 0..4 {
                    s += wzy * a.1[x] * c[base + x];
                }
            }
        }
        s
    }
    fn scatter(&self, acc: &mut [f64], q: usize, v: f64) {
        let [i, j, k] = self.coords(q);
        let (a, b, d) = (&self.idx[0][i], &self.idx[1][j], &self.idx[2][k]);
        for z in 0..4 {
            for y in 0..4 {
                let wzy = d.1[z] * b.1[y] * v;
                let base = ((d.0 + z) * self.nb[1] + b.0 + y) * self.nb[0] + a.0;
                for x in 0..4 {
                    acc[base + x] += wzy * a.1[x];
                }
            }
        }
    }
    /// Σ_v w_v B_v f_v
    fn bt(&self, vox: &[usize], w: &[f64], f: &[f64]) -> Vec<f64> {
        let n = self.n();
        vox.par_iter()
            .enumerate()
            .fold(|| vec![0.0; n], |mut acc, (i, &q)| {
                if w[i] != 0.0 {
                    self.scatter(&mut acc, q, w[i] * f[i]);
                }
                acc
            })
            .reduce(|| vec![0.0; n], |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            })
    }
    /// RᵀR c with R the second differences along each axis of the coefficient grid
    fn reg(&self, c: &[f64]) -> Vec<f64> {
        let nb = self.nb;
        let st = [1, nb[0], nb[0] * nb[1]];
        let mut out = vec![0.0; c.len()];
        for a in 0..3 {
            for q in 0..c.len() {
                let i = (q / st[a]) % nb[a];
                if i == 0 || i + 1 >= nb[a] {
                    continue;
                }
                let d2 = c[q - st[a]] - 2.0 * c[q] + c[q + st[a]];
                out[q - st[a]] += d2;
                out[q] -= 2.0 * d2;
                out[q + st[a]] += d2;
            }
        }
        out
    }
    /// argmin Σ w_v (B_v c − f_v)² + λ ‖R c‖² by CG (warm start c0)
    fn fit(&self, vox: &[usize], w: &[f64], f: &[f64], lam_rel: f64, c0: Vec<f64>, iters: usize) -> Vec<f64> {
        let n = self.n();
        // mean data curvature per coefficient: Σ w_v ‖B_v‖² ≈ Σ w (cubic B-spline ‖B‖² ≈ (0.4790)³) / n
        let wsum: f64 = w.iter().sum();
        let lam = lam_rel * wsum * 0.4790f64.powi(3) / n as f64;
        let apply = |c: &[f64]| -> Vec<f64> {
            let s: Vec<f64> = vox.par_iter().enumerate().map(|(i, &q)| if w[i] != 0.0 { self.eval_at(c, q) } else { 0.0 }).collect();
            let mut y = self.bt(vox, w, &s);
            let r = self.reg(c);
            y.iter_mut().zip(r).for_each(|(a, b)| *a += lam * b);
            y
        };
        let b = self.bt(vox, w, f);
        let mut x = c0;
        let ax = apply(&x);
        let mut r: Vec<f64> = b.iter().zip(&ax).map(|(a, c)| a - c).collect();
        let mut p = r.clone();
        let mut rr: f64 = r.iter().map(|v| v * v).sum();
        let r0 = rr.max(1e-300);
        for _ in 0..iters {
            if rr < 1e-12 * r0 {
                break;
            }
            let ap = apply(&p);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            let al = rr / pap.max(1e-300);
            x.iter_mut().zip(&p).for_each(|(a, b)| *a += al * b);
            r.iter_mut().zip(&ap).for_each(|(a, b)| *a -= al * b);
            let rn: f64 = r.iter().map(|v| v * v).sum();
            let be = rn / rr;
            rr = rn;
            p.iter_mut().zip(&r).for_each(|(a, b)| *a = b + be * *a);
        }
        x
    }
    /// coefficients of another spline's field (least squares at the voxels, unweighted)
    fn eval_all(&self, c: &[f64]) -> Vec<f64> {
        let nv = self.dims[0] * self.dims[1] * self.dims[2];
        (0..nv).into_par_iter().map(|q| self.eval_at(c, q)).collect()
    }
}

/// one voxel-fit pass. `y`: [nd re | nd im][n] (problem fastest), `demod` ω_s per problem (rad/s) or None.
/// Returns the parameters [6][n] with ω already re-modulated (absolute), and the relative residual per problem.
fn voxel_pass(gpu: &WaterFatGpu<f32>, wf: &WaterFat, y: &[f64], n: usize, demod: Option<&[f64]>) -> (Vec<f64>, Vec<f64>) {
    let nd = wf.te.len();
    let yd: Vec<f64> = match demod {
        None => y.to_vec(),
        Some(om) => {
            let mut o = vec![0.0; y.len()];
            for d in 0..nd {
                let t = wf.te[d];
                let (re, im) = o.split_at_mut(nd * n);
                re[d * n..(d + 1) * n].par_iter_mut().zip(im[d * n..(d + 1) * n].par_iter_mut()).enumerate().for_each(|(p, (a, b))| {
                    let z = C::new(y[d * n + p], y[(nd + d) * n + p]) * C::from_polar(1.0, om[p] * t);
                    *a = z.re;
                    *b = z.im;
                });
            }
            o
        }
    };
    let fit = gpu.fit(&yd, n).unwrap();
    let mut par = fit.params;
    let basis: Vec<C> = wf.te.iter().map(|&t| wf.fat.basis(wf.b0, t)).collect();
    let rel: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|p| {
            let (w, f, r2, om) = (C::new(par[p], par[n + p]), C::new(par[2 * n + p], par[3 * n + p]), par[4 * n + p], par[5 * n + p]);
            let (mut e, mut s) = (0.0, 0.0);
            for d in 0..nd {
                let t = wf.te[d];
                let m = (w + f * basis[d]) * (-(C::new(r2, om)) * t).exp();
                let z = C::new(yd[d * n + p], yd[(nd + d) * n + p]);
                e += (z - m).norm_sqr();
                s += z.norm_sqr();
            }
            if s > 0.0 { e / s } else { 1.0 }
        })
        .collect();
    if let Some(om) = demod {
        for p in 0..n {
            par[5 * n + p] += om[p];
        }
    }
    (par, rel)
}

/// fit the maps. `y`: [nd re | nd im][nb · nvx] (problem = bin · nvx + voxel index), `vox` the fitted voxels.
pub fn fit(dev: &Device, b0: f64, te: &[f64], dims: [usize; 3], res: f64, vox: &[usize], nb: usize, y: &[f64], cfg: &Cfg) -> Maps {
    let nv = dims[0] * dims[1] * dims[2];
    let nvx = vox.len();
    let nprob = nb * nvx;
    let nd = te.len();
    let tw = 2.0 * std::f64::consts::PI;
    let wf = WaterFat::new(Species::WaterFat, te.to_vec(), b0, FatSpectrum::six_peak());
    let grid_r2 = vec![0.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0];
    let cfg_full = WfConfig { om_max: tw * cfg.om_max_hz, r2_max: cfg.r2_max, grid_r2: grid_r2.clone(), ..Default::default() };
    let gpu_full = WaterFatGpu::<f32>::new(dev, &wf, &cfg_full).unwrap();
    let mut log = Vec::new();
    let mut say = |s: String| {
        println!("  ratemap: {s}");
        log.push(s);
    };
    let (mut par, mut rel) = voxel_pass(&gpu_full, &wf, y, nprob, None);
    say(format!("full-range voxel fit (|ω| ≤ {:.0} Hz): {nprob} problems", cfg.om_max_hz));
    let mut om_s = vec![f32::NAN; nb * nv];
    if cfg.method == Method::Smooth {
        let cfg_win = WfConfig { om_max: tw * cfg.window_hz, r2_max: cfg.r2_max, grid_r2, ..Default::default() };
        let gpu_win = WaterFatGpu::<f32>::new(dev, &wf, &cfg_win).unwrap();
        // signal energy per problem (compressed: square root, so the coil-proximal hot spots do not own the fit)
        let energy: Vec<f64> = (0..nprob).map(|p| (0..2 * nd).map(|d| y[d * nprob + p].powi(2)).sum::<f64>().sqrt()).collect();
        let mut demod = vec![0.0f64; nprob];
        for b in 0..nb {
            let rng = b * nvx..(b + 1) * nvx;
            let mut c_prev: Option<(Spline, Vec<f64>)> = None;
            for (lvl, &kmm) in cfg.knots_mm.iter().enumerate() {
                let sp = Spline::new(dims, kmm * 1e-3 / res);
                let f: Vec<f64> = rng.clone().map(|p| par[5 * nprob + p] / tw).collect();
                // confidence: energy × model consistency; estimates at the window edge carry no basin information
                let conf: Vec<f64> = rng
                    .clone()
                    .map(|p| {
                        let edge = lvl > 0 && (par[5 * nprob + p] - demod[p]).abs() > 0.97 * tw * cfg.window_hz;
                        if edge { 0.0 } else { energy[p] * (1.0 - rel[p].min(1.0)).powi(2) }
                    })
                    .collect();
                // robust weights from the previous level's field (the window refit already sits in its basin)
                let mut w: Vec<f64> = match &c_prev {
                    Some((sp0, c0)) => vox
                        .par_iter()
                        .enumerate()
                        .map(|(i, &q)| {
                            let r = (sp0.eval_at(c0, q) - f[i]) / cfg.huber_hz;
                            conf[i] / (1.0 + r * r)
                        })
                        .collect(),
                    None => conf.clone(),
                };
                let mut c = vec![0.0; sp.n()];
                for it in 0..cfg.irls {
                    c = sp.fit(vox, &w, &f, cfg.lam, c, 80);
                    if it + 1 < cfg.irls {
                        let sc = cfg.huber_hz * if c_prev.is_none() && it < 2 { 4.0 } else { 1.0 };
                        w = vox
                            .par_iter()
                            .enumerate()
                            .map(|(i, &q)| {
                                let r = (sp.eval_at(&c, q) - f[i]) / sc;
                                conf[i] / (1.0 + r * r)
                            })
                            .collect();
                    }
                }
                let field = sp.eval_all(&c);
                let inl = vox.iter().enumerate().filter(|(i, &q)| conf[*i] > 0.0 && (field[q] - f[*i]).abs() < 100.0).count();
                let nconf = conf.iter().filter(|&&x| x > 0.0).count();
                for (i, &q) in vox.iter().enumerate() {
                    demod[b * nvx + i] = tw * field[q];
                }
                om_s[b * nv..(b + 1) * nv].iter_mut().zip(&field).for_each(|(o, &v)| *o = v as f32);
                // re-fit this bin's voxels inside the window around the smooth field
                let yb: Vec<f64> = (0..2 * nd).flat_map(|d| y[d * nprob + b * nvx..d * nprob + (b + 1) * nvx].iter().copied()).collect();
                let (pb, rb) = voxel_pass(&gpu_win, &wf, &yb, nvx, Some(&demod[b * nvx..(b + 1) * nvx]));
                for j in 0..6 {
                    par[j * nprob + b * nvx..j * nprob + (b + 1) * nvx].copy_from_slice(&pb[j * nvx..(j + 1) * nvx]);
                }
                rel[b * nvx..(b + 1) * nvx].copy_from_slice(&rb);
                say(format!("bin {b} level {lvl} ({kmm} mm knots, {} coefficients): {inl} of {nconf} confident voxels within 100 Hz of the field; window refit |ω − ω_s| ≤ {:.0} Hz", sp.n(), cfg.window_hz));
                c_prev = Some((sp, c));
            }
        }
    }
    let mut m = Maps {
        om_hz: vec![f32::NAN; nb * nv],
        r2: vec![f32::NAN; nb * nv],
        ff: vec![f32::NAN; nb * nv],
        wm: vec![0.0; nb * nv],
        fm: vec![0.0; nb * nv],
        om_smooth_hz: om_s,
        rel_res: vec![f32::NAN; nb * nv],
        log,
    };
    for b in 0..nb {
        for (vi, &q) in vox.iter().enumerate() {
            let p = b * nvx + vi;
            let pr = |j: usize| par[j * nprob + p];
            let (w, f) = ((pr(0).powi(2) + pr(1).powi(2)).sqrt(), (pr(2).powi(2) + pr(3).powi(2)).sqrt());
            let o = b * nv + q;
            m.om_hz[o] = (pr(5) / tw) as f32;
            m.r2[o] = pr(4) as f32;
            m.ff[o] = (f / (w + f).max(1e-30)) as f32;
            m.wm[o] = w as f32;
            m.fm[o] = f as f32;
            m.rel_res[o] = rel[p] as f32;
        }
    }
    m
}
