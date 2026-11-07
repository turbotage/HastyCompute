//! B0 / R2* / fat estimation from multi-echo complex images (voxel-wise), in the simulator's sign convention:
//!
//!   s(t) = (ρ_w + ρ_f c(t)) e^{−(R2* + iω) t},   c(t) = Σ_p α_p e^{−i ω_p t},   ω_p = 2π γ̄ B0 δ_p,
//!
//! the same as `hasty-mrsim` (a source contributes a e^{−R2* τ} e^{−iωτ}) and `hasty-phantom`
//! (ω = 2π γ̄ B0 (field + shift) ppm · 1e-6), so ω and R2* are the complex rate z = R2* + iω of the 4D-flow segmentor.
//! t is the time after the RF centre (where water and fat have a common phase), ω in rad/s, R2* in 1/s.
//!
//! The off-resonance makes the likelihood multi-modal: with echo spacing ΔTE it is exactly periodic in ω with period
//! 2π/ΔTE when the echoes are uniform, and pure-water / pure-fat voxels have a nearly equivalent swapped solution
//! (exactly equivalent for a single-peak fat model). Estimation is therefore:
//! 1. profile likelihood on a (ω, R2*) grid: at fixed (ω, R2*) the model is linear, s = A ρ, so the amplitudes are
//!    eliminated exactly, R(ω, R2*) = ‖y‖²_W − ‖Qᴴ W^{½} y‖² (Q an orthonormal basis of W^{½}A, one per node, shared
//!    by all voxels). The profile likelihood has the same maximiser as the full likelihood, so its grid minima are
//!    starts in the basins of the MLE and its competitors; the best `n_start` local minima over ω are kept;
//! 2. full Levenberg–Marquardt from each start, the lowest objective wins (the per-voxel MLE among those basins).
//!
//! Spatial regularisation of the remaining ambiguity (graph cuts, smoothness priors, model-based CS) is a
//! reconstruction-level step and uses the candidates' residuals; it is not done here.
//!
//! Data are normalised per voxel by c = RMS(y) before fitting (amplitudes ~ 1 for conditioning, especially in f32)
//! and the amplitudes are scaled back afterwards. This multiplies the voxel's objective by 1/c² and leaves its MLE
//! unchanged; weights (if any) must be inverse noise variances.

use num_complex::Complex64 as C;

use crate::fit::{fit_cpu, Data};
use crate::lm::{LmOptions, Status};
use crate::model::Model;

/// γ/2π in Hz/T (same value as `hasty-mrsim`)
pub const GAMMA_BAR: f64 = 2.675_221_874_4e8 / (2.0 * std::f64::consts::PI);

#[derive(Clone, Debug)]
pub struct FatSpectrum {
    /// chemical shift of each peak relative to water, ppm
    pub ppm: Vec<f64>,
    /// relative amplitudes, Σ = 1
    pub amp: Vec<f64>,
}

impl FatSpectrum {
    /// six-peak fat spectrum relative to water (Hamilton et al., NMR Biomed 2011; the ISMRM fat–water toolbox values)
    pub fn six_peak() -> Self {
        let a = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048];
        let s: f64 = a.iter().sum();
        Self { ppm: vec![-3.80, -3.40, -2.60, -1.94, -0.39, 0.60], amp: a.iter().map(|x| x / s).collect() }
    }
    /// one peak (the phantom currently uses −3.4 ppm)
    pub fn single(ppm: f64) -> Self {
        Self { ppm: vec![ppm], amp: vec![1.0] }
    }
    /// angular frequency of each peak at field `b0` (T), rad/s
    pub fn omega(&self, b0: f64) -> Vec<f64> {
        self.ppm.iter().map(|d| 2.0 * std::f64::consts::PI * GAMMA_BAR * b0 * d * 1e-6).collect()
    }
    /// c(t) = Σ_p α_p e^{−i ω_p t}
    pub fn basis(&self, b0: f64, t: f64) -> C {
        self.omega(b0).iter().zip(&self.amp).map(|(w, a)| C::from_polar(*a, -w * t)).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    /// [wr, wi, r2, om]
    Water,
    /// [wr, wi, fr, fi, r2, om]: independent complex amplitudes, one R2*
    WaterFat,
    /// [w, f, ph, r2, om]: real water/fat amplitudes with a common phase at t = 0 (needs phase-accurate echoes)
    WaterFatCommonPhase,
    /// [wr, wi, fr, fi, r2w, r2f, om]: separate R2* for water and fat
    WaterFatDualR2,
}

impl Species {
    pub fn names(&self) -> &'static [&'static str] {
        match self {
            Species::Water => &["wr", "wi", "r2", "om"],
            Species::WaterFat => &["wr", "wi", "fr", "fi", "r2", "om"],
            Species::WaterFatCommonPhase => &["w", "f", "ph", "r2", "om"],
            Species::WaterFatDualR2 => &["wr", "wi", "fr", "fi", "r2w", "r2f", "om"],
        }
    }
    /// indices of the amplitude parameters (scaled by the data normalisation)
    pub fn amp_idx(&self) -> &'static [usize] {
        match self {
            Species::Water => &[0, 1],
            Species::WaterFatCommonPhase => &[0, 1],
            Species::WaterFat | Species::WaterFatDualR2 => &[0, 1, 2, 3],
        }
    }
    /// number of complex linear amplitudes in the profile (1 water, 2 water + fat)
    pub fn n_lin(&self) -> usize {
        if *self == Species::Water { 1 } else { 2 }
    }
}

/// configuration of the estimation (grid, bounds, starts, solver)
#[derive(Clone, Debug)]
pub struct WfConfig {
    /// |ω| range of the grid and the bounds, rad/s; with uniform echoes keep it ≤ π/ΔTE (the period)
    pub om_max: f64,
    pub r2_min: f64,
    pub r2_max: f64,
    /// R2* nodes of the profile grid (the ω nodes follow from the echo span, see [`Grid::for_echoes`])
    pub grid_r2: Vec<f64>,
    /// number of profile minima used as LM starts
    pub n_start: usize,
    /// solver options; `scale` is set by [`WaterFat::options`]
    pub lm: LmOptions,
}

impl Default for WfConfig {
    fn default() -> Self {
        Self {
            om_max: 2.0 * std::f64::consts::PI * 500.0,
            r2_min: 0.0,
            r2_max: 500.0,
            grid_r2: vec![0.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0],
            n_start: 3,
            lm: LmOptions::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct WaterFat {
    pub species: Species,
    /// echo times after the RF centre, s
    pub te: Vec<f64>,
    /// field, T
    pub b0: f64,
    pub fat: FatSpectrum,
    pub model: Model,
}

impl WaterFat {
    /// the (real, imaginary) output expressions over `species.names()` and the vars t, cr, ci (c(t) = cr + i ci)
    pub fn output_exprs(species: Species) -> [String; 2] {
        let e = "exp(-r2 * t)";
        let (c, s) = ("cos(om * t)", "sin(om * t)");
        // complex A + iB times e^{−iθ}: re = A cos θ + B sin θ, im = B cos θ − A sin θ
        let rot = |a: &str, b: &str, e: &str| -> [String; 2] {
            [format!("{e} * (({a}) * {c} + ({b}) * {s})"), format!("{e} * (({b}) * {c} - ({a}) * {s})")]
        };
        match species {
            Species::Water => rot("wr", "wi", e),
            Species::WaterFat => rot("wr + fr * cr - fi * ci", "wi + fr * ci + fi * cr", e),
            Species::WaterFatCommonPhase => {
                rot("(w + f * cr) * cos(ph) - f * ci * sin(ph)", "(w + f * cr) * sin(ph) + f * ci * cos(ph)", e)
            }
            Species::WaterFatDualR2 => {
                let w = rot("wr", "wi", "exp(-r2w * t)");
                let f = rot("fr * cr - fi * ci", "fr * ci + fi * cr", "exp(-r2f * t)");
                [format!("{} + {}", w[0], f[0]), format!("{} + {}", w[1], f[1])]
            }
        }
    }

    pub fn new(species: Species, te: Vec<f64>, b0: f64, fat: FatSpectrum) -> WaterFat {
        let outs = Self::output_exprs(species);
        let model = Model::new(species.names(), &["t", "cr", "ci"], &[], &[&outs[0], &outs[1]]).expect("water–fat model");
        WaterFat { species, te, b0, fat, model }
    }

    pub fn n_echo(&self) -> usize {
        self.te.len()
    }
    pub fn n_param(&self) -> usize {
        self.model.n_param()
    }

    /// model vars [t | Re c(t) | Im c(t)], each [n_echo]
    pub fn vars(&self) -> Vec<f64> {
        let c: Vec<C> = self.te.iter().map(|&t| self.fat.basis(self.b0, t)).collect();
        self.te.iter().copied().chain(c.iter().map(|x| x.re)).chain(c.iter().map(|x| x.im)).collect()
    }

    /// reference signal at the echo times (direct complex formula, independent of the symbolic model)
    pub fn signal(&self, p: &[f64]) -> Vec<C> {
        let i = C::new(0.0, 1.0);
        self.te
            .iter()
            .map(|&t| {
                let c = self.fat.basis(self.b0, t);
                match self.species {
                    Species::Water => C::new(p[0], p[1]) * (-(p[2] + i * p[3]) * t).exp(),
                    Species::WaterFat => (C::new(p[0], p[1]) + C::new(p[2], p[3]) * c) * (-(p[4] + i * p[5]) * t).exp(),
                    Species::WaterFatCommonPhase => C::from_polar(1.0, p[2]) * (p[0] + p[1] * c) * (-(p[3] + i * p[4]) * t).exp(),
                    Species::WaterFatDualR2 => {
                        let e = (-i * p[6] * t).exp();
                        (C::new(p[0], p[1]) * (-p[4] * t).exp() + C::new(p[2], p[3]) * c * (-p[5] * t).exp()) * e
                    }
                }
            })
            .collect()
    }

    /// box bounds for normalised data
    pub fn bounds(&self, cfg: &WfConfig) -> (Vec<f64>, Vec<f64>) {
        let inf = f64::INFINITY;
        let (mut lo, mut hi) = (vec![-inf; self.n_param()], vec![inf; self.n_param()]);
        for (j, name) in self.species.names().iter().enumerate() {
            match *name {
                "r2" | "r2w" | "r2f" => {
                    lo[j] = cfg.r2_min;
                    hi[j] = cfg.r2_max;
                }
                "om" => {
                    lo[j] = -cfg.om_max;
                    hi[j] = cfg.om_max;
                }
                _ => {}
            }
        }
        (lo, hi)
    }

    /// solver options with the parameter scaling for normalised data: amplitudes 1, phase π, R2* r2_max, ω om_max
    /// (a reparametrisation only — see `lm` module docs: same MLE, damping in the scaled variables)
    pub fn options(&self, cfg: &WfConfig) -> LmOptions {
        let scale = self
            .species
            .names()
            .iter()
            .map(|n| match *n {
                "r2" | "r2w" | "r2f" => cfg.r2_max.abs().max(cfg.r2_min.abs()).max(1.0),
                "om" => cfg.om_max,
                "ph" => std::f64::consts::PI,
                _ => 1.0,
            })
            .collect();
        LmOptions { scale: Some(scale), ..cfg.lm.clone() }
    }

    /// LM starts [n_param][n_prob] (normalised amplitudes) from profile candidate `k`
    pub fn starts(&self, prof: &Profile, cand: &Candidates, k: usize) -> Vec<f64> {
        let (np, n) = (self.n_param(), cand.n_prob);
        let mut p = vec![0.0; np * n];
        for v in 0..n {
            // a missing candidate (fewer local minima than starts) repeats the best one
            let kk = if cand.node[k * n + v] >= 0 { k } else { 0 };
            let node = cand.node[kk * n + v].max(0) as usize;
            let (om, r2) = (prof.om[node], prof.r2[node]);
            let rho = |l: usize| cand.rho[(kk * cand.l + l) * n + v];
            let vals: Vec<f64> = match self.species {
                Species::Water => vec![rho(0).re, rho(0).im, r2, om],
                Species::WaterFat => vec![rho(0).re, rho(0).im, rho(1).re, rho(1).im, r2, om],
                Species::WaterFatCommonPhase => {
                    let ph = (rho(0) + rho(1)).arg();
                    let r = C::from_polar(1.0, -ph);
                    vec![(rho(0) * r).re, (rho(1) * r).re, ph, r2, om]
                }
                Species::WaterFatDualR2 => vec![rho(0).re, rho(0).im, rho(1).re, rho(1).im, r2, r2, om],
            };
            for j in 0..np {
                p[j * n + v] = vals[j];
            }
        }
        p
    }

    /// CPU reference of the whole estimation; `y` [2][n_echo][n_prob] (real, imaginary), unweighted
    pub fn fit_cpu(&self, y: &[f64], n_prob: usize, cfg: &WfConfig) -> WfFit {
        let nd = self.n_echo();
        let (yn, c) = normalize(y, nd, n_prob);
        let prof = Profile::new(self, &Grid::for_echoes(&self.te, cfg.om_max, cfg.grid_r2.clone()), None);
        let cand = prof.candidates_cpu(&yn, n_prob, cfg.n_start);
        let vars = self.vars();
        let d = Data { n_prob, n_data: nd, vars: &vars[..], fixed: &[][..], y: &yn[..], w: None };
        let (lo, hi) = self.bounds(cfg);
        let o = self.options(cfg);
        let mut best = WfFit::empty(self.n_param(), n_prob, cand);
        for k in 0..cfg.n_start {
            let mut p = self.starts(&prof, &best.cand, k);
            let out = fit_cpu(&self.model, &d, &mut p, &lo, &hi, &o);
            best.take_better(k, &p, &out.iter().map(|x| (x.f, x.iters, x.status)).collect::<Vec<_>>());
        }
        best.unnormalize(self.species, &c);
        best
    }
}

/// result of a multi-start estimation (original data units)
#[derive(Clone, Debug)]
pub struct WfFit {
    /// [n_param][n_prob]
    pub params: Vec<f64>,
    /// objective ½‖y − s‖² at the estimate
    pub f: Vec<f64>,
    pub iters: Vec<u32>,
    pub status: Vec<Status>,
    /// which start won
    pub start: Vec<u8>,
    pub cand: Candidates,
}

impl WfFit {
    fn empty(np: usize, n: usize, cand: Candidates) -> WfFit {
        WfFit { params: vec![0.0; np * n], f: vec![f64::INFINITY; n], iters: vec![0; n], status: vec![Status::MaxIter; n], start: vec![0; n], cand }
    }
    fn take_better(&mut self, k: usize, p: &[f64], out: &[(f64, u32, Status)]) {
        let n = self.f.len();
        let np = self.params.len() / n;
        for v in 0..n {
            let (f, it, st) = out[v];
            if f < self.f[v] || (k == 0 && !self.f[v].is_finite()) {
                self.f[v] = f;
                self.iters[v] = it;
                self.status[v] = st;
                self.start[v] = k as u8;
                for j in 0..np {
                    self.params[j * n + v] = p[j * n + v];
                }
            }
        }
    }
    fn unnormalize(&mut self, sp: Species, c: &[f64]) {
        let n = c.len();
        for &j in sp.amp_idx() {
            for v in 0..n {
                self.params[j * n + v] *= c[v];
            }
        }
        for v in 0..n {
            self.f[v] *= c[v] * c[v];
        }
        canonicalize(sp, &mut self.params, n);
    }
}

/// common-phase model: (w, f, φ) and (−w, −f, φ + π) are the same signal and φ is 2π-periodic; report w + f ≥ 0 and
/// φ ∈ (−π, π] (the same convention as the GPU `wf_unnorm` kernel)
pub fn canonicalize(sp: Species, p: &mut [f64], n: usize) {
    if sp != Species::WaterFatCommonPhase {
        return;
    }
    for v in 0..n {
        let (w, f, mut ph) = (p[v], p[n + v], p[2 * n + v]);
        if w + f < 0.0 {
            p[v] = -w;
            p[n + v] = -f;
            ph += std::f64::consts::PI;
        }
        p[2 * n + v] = ph.sin().atan2(ph.cos());
    }
}

/// per-voxel RMS normalisation of `y` [2][nd][n]: returns (y / c, c); c = 1 for an all-zero voxel
pub fn normalize(y: &[f64], nd: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(y.len(), 2 * nd * n);
    let mut c = vec![0.0; n];
    for i in 0..2 * nd {
        for v in 0..n {
            c[v] += y[i * n + v] * y[i * n + v];
        }
    }
    for x in c.iter_mut() {
        *x = (*x / nd as f64).sqrt();
        if !(*x > 0.0) {
            *x = 1.0;
        }
    }
    let mut yn = y.to_vec();
    for i in 0..2 * nd {
        for v in 0..n {
            yn[i * n + v] /= c[v];
        }
    }
    (yn, c)
}

/// (ω, R2*) grid of the profile
#[derive(Clone, Debug)]
pub struct Grid {
    pub om: Vec<f64>,
    pub r2: Vec<f64>,
}

impl Grid {
    /// ω nodes over [−om_max, om_max] spaced 2π / (6 T), T the echo span (≈ 1/6 of the profile's main-lobe width)
    pub fn for_echoes(te: &[f64], om_max: f64, r2: Vec<f64>) -> Grid {
        let span = te.iter().cloned().fold(f64::MIN, f64::max) - te.iter().cloned().fold(f64::MAX, f64::min);
        let dom = 2.0 * std::f64::consts::PI / (6.0 * span.max(1e-4));
        let n = ((2.0 * om_max / dom).ceil() as usize + 1).max(2);
        Grid { om: (0..n).map(|i| -om_max + 2.0 * om_max * i as f64 / (n - 1) as f64).collect(), r2 }
    }
}

/// the linear basis of every grid node, orthonormalised (shared by all voxels)
#[derive(Clone, Debug)]
pub struct Profile {
    pub l: usize,
    pub nd: usize,
    pub n_om: usize,
    pub n_r2: usize,
    /// per node (ω-major): ω and R2*
    pub om: Vec<f64>,
    pub r2: Vec<f64>,
    /// W^{½} Q, Q orthonormal basis of W^{½} A: [node][l][nd]
    pub q: Vec<C>,
    /// ρ = M b with b = (W^{½}Q)ᴴ y, M = R⁻¹ upper triangular: [node][l][l]
    pub m: Vec<C>,
    pub valid: Vec<bool>,
    /// per-echo weights (inverse noise variances, shared by all voxels)
    pub w: Vec<f64>,
}

impl Profile {
    pub fn new(wf: &WaterFat, grid: &Grid, w: Option<&[f64]>) -> Profile {
        let (l, nd) = (wf.species.n_lin(), wf.n_echo());
        let w: Vec<f64> = w.map_or(vec![1.0; nd], |w| w.to_vec());
        let sw: Vec<f64> = w.iter().map(|x| x.sqrt()).collect();
        let c: Vec<C> = wf.te.iter().map(|&t| wf.fat.basis(wf.b0, t)).collect();
        let nn = grid.om.len() * grid.r2.len();
        let mut p = Profile {
            l,
            nd,
            n_om: grid.om.len(),
            n_r2: grid.r2.len(),
            om: Vec::with_capacity(nn),
            r2: Vec::with_capacity(nn),
            q: vec![C::new(0.0, 0.0); nn * l * nd],
            m: vec![C::new(0.0, 0.0); nn * l * l],
            valid: vec![false; nn],
            w,
        };
        let i = C::new(0.0, 1.0);
        for &om in &grid.om {
            for &r2 in &grid.r2 {
                let node = p.om.len();
                p.om.push(om);
                p.r2.push(r2);
                // columns of W^{½}A: water e^{−zt}, fat c(t) e^{−zt}
                let e: Vec<C> = (0..nd).map(|n| sw[n] * (-(r2 + i * om) * wf.te[n]).exp()).collect();
                let a0 = e.clone();
                let dot = |a: &[C], b: &[C]| a.iter().zip(b).map(|(x, y)| x.conj() * y).sum::<C>();
                let r00 = dot(&a0, &a0).re.sqrt();
                if !(r00 > 0.0) {
                    continue;
                }
                let q0: Vec<C> = a0.iter().map(|x| x / r00).collect();
                let mut ok = true;
                if l == 2 {
                    let a1: Vec<C> = (0..nd).map(|n| e[n] * c[n]).collect();
                    let r01 = dot(&q0, &a1);
                    let g11 = dot(&a1, &a1).re;
                    let r11sq = g11 - r01.norm_sqr();
                    if !(r11sq > 1e-10 * g11) {
                        ok = false;
                    } else {
                        let r11 = r11sq.sqrt();
                        for n in 0..nd {
                            p.q[(node * l + 1) * nd + n] = sw[n] * (a1[n] - r01 * q0[n]) / r11;
                        }
                        p.m[node * 4] = C::new(1.0 / r00, 0.0);
                        p.m[node * 4 + 1] = -r01 / (r00 * r11);
                        p.m[node * 4 + 3] = C::new(1.0 / r11, 0.0);
                    }
                } else {
                    p.m[node] = C::new(1.0 / r00, 0.0);
                }
                if ok {
                    for n in 0..nd {
                        p.q[(node * l) * nd + n] = sw[n] * q0[n];
                    }
                    p.valid[node] = true;
                }
            }
        }
        p
    }

    /// profile residual minima per voxel (CPU reference of the `wf_profile` kernel, same streaming logic);
    /// `y` [2][nd][n]
    pub fn candidates_cpu(&self, y: &[f64], n: usize, k: usize) -> Candidates {
        let (l, nd) = (self.l, self.nd);
        assert_eq!(y.len(), 2 * nd * n);
        let nt = std::thread::available_parallelism().map_or(1, |x| x.get()).min(n.max(1));
        let chunk = n.div_ceil(nt);
        let parts: Vec<Vec<(i32, f64, Vec<C>)>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nt)
                .map(|t| {
                    sc.spawn(move || {
                        let mut out = vec![];
                        for v in t * chunk..((t + 1) * chunk).min(n) {
                            let yv: Vec<C> = (0..nd).map(|i| C::new(y[i * n + v], y[(nd + i) * n + v])).collect();
                            out.extend(self.voxel(&yv, k));
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut cand = Candidates { k, n_prob: n, l, node: vec![-1; k * n], resid: vec![f64::INFINITY; k * n], rho: vec![C::new(0.0, 0.0); k * l * n] };
        let flat: Vec<(i32, f64, Vec<C>)> = parts.into_iter().flatten().collect();
        for v in 0..n {
            for kk in 0..k {
                let (node, r, b) = &flat[v * k + kk];
                cand.set(self, kk, v, *node, *r, b);
            }
        }
        cand
    }

    /// top-k local minima over ω of min_{R2*} R for one voxel: (node, residual, b)
    fn voxel(&self, y: &[C], k: usize) -> Vec<(i32, f64, Vec<C>)> {
        let (l, nd) = (self.l, self.nd);
        let yy: f64 = (0..nd).map(|i| self.w[i] * y[i].norm_sqr()).sum();
        let mut top: Vec<(i32, f64, Vec<C>)> = vec![(-1, f64::INFINITY, vec![C::new(0.0, 0.0); l]); k];
        let insert = |top: &mut Vec<(i32, f64, Vec<C>)>, e: (i32, f64, Vec<C>)| {
            if let Some(pos) = top.iter().position(|t| e.1 < t.1) {
                top.insert(pos, e);
                top.pop();
            }
        };
        let (mut r1, mut r2) = (f64::INFINITY, f64::INFINITY);
        let mut n1: i32 = -1;
        let mut b1 = vec![C::new(0.0, 0.0); l];
        for i in 0..self.n_om {
            let (mut rc, mut nc) = (f64::INFINITY, -1i32);
            let mut bc = vec![C::new(0.0, 0.0); l];
            for j in 0..self.n_r2 {
                let node = i * self.n_r2 + j;
                if !self.valid[node] {
                    continue;
                }
                let mut s = 0.0;
                let mut bb = vec![C::new(0.0, 0.0); l];
                for (ll, b) in bb.iter_mut().enumerate() {
                    let q = &self.q[(node * l + ll) * nd..(node * l + ll + 1) * nd];
                    *b = q.iter().zip(y).map(|(q, y)| q.conj() * y).sum();
                    s += b.norm_sqr();
                }
                let r = yy - s;
                if r < rc {
                    rc = r;
                    nc = node as i32;
                    bc = bb;
                }
            }
            if i > 0 && (i - 1 == 0 || r1 <= r2) && r1 < rc && n1 >= 0 {
                insert(&mut top, (n1, r1, b1.clone()));
            }
            r2 = r1;
            r1 = rc;
            n1 = nc;
            b1 = bc;
        }
        if n1 >= 0 && (self.n_om == 1 || r1 <= r2) {
            insert(&mut top, (n1, r1, b1));
        }
        top
    }
}

/// best profile minima per voxel
#[derive(Clone, Debug)]
pub struct Candidates {
    pub k: usize,
    pub n_prob: usize,
    pub l: usize,
    /// grid node per candidate, −1 if fewer minima were found: [k][n]
    pub node: Vec<i32>,
    /// profile residual (normalised data): [k][n]
    pub resid: Vec<f64>,
    /// eliminated amplitudes (normalised data): [k][l][n]
    pub rho: Vec<C>,
}

impl Candidates {
    fn set(&mut self, prof: &Profile, kk: usize, v: usize, node: i32, r: f64, b: &[C]) {
        let (n, l) = (self.n_prob, self.l);
        self.node[kk * n + v] = node;
        self.resid[kk * n + v] = r;
        if node >= 0 {
            let m = &prof.m[node as usize * l * l..(node as usize + 1) * l * l];
            for a in 0..l {
                let mut s = C::new(0.0, 0.0);
                for c in a..l {
                    s += m[a * l + c] * b[c];
                }
                self.rho[(kk * l + a) * n + v] = s;
            }
        }
    }
}

pub mod gpu {
    //! GPU path, device-resident end to end: per-voxel normalisation, the profile kernel (one voxel per thread, grid
    //! nodes broadcast from global memory; writes the eliminated amplitudes and the LM starts), the generic `qfit_lm`
    //! kernel per start, per-voxel selection of the best start and the un-normalisation. Only the input data and the
    //! results cross the bus.

    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use hasty_cuda::{CompileSettings, Device};
    use num_complex::Complex64 as C;

    use super::{Candidates, Grid, Profile, Species, WaterFat, WfConfig, WfFit};
    use crate::fit::Data;
    use crate::gpu::{Error, GpuData, GpuFit, GpuScalar};
    use crate::lm::Status;

    const KERNELS: &str = r#"
// per-voxel RMS normalisation in place (y [2][ND][nprob]); c = 1 for an all-zero voxel
extern "C" __global__ void wf_norm(T* y, int nprob, T* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nprob) return;
    T s = 0;
    for (int i = 0; i < 2 * ND; ++i) { T v = y[i * nprob + tid]; s += v * v; }
    T cc = sqrt_t(s / (T)ND);
    if (!(cc > (T)0) || !isfinite(cc)) cc = (T)1;
    for (int i = 0; i < 2 * ND; ++i) y[i * nprob + tid] /= cc;
    c[tid] = cc;
}

__device__ void top_insert(T* kr, int* kn, T* kb, T r, int node, const T* b) {
    int pos = K;
    for (int k = K - 1; k >= 0; --k) if (r < kr[k]) pos = k;
    if (pos >= K) return;
    for (int k = K - 1; k > pos; --k) { kr[k] = kr[k - 1]; kn[k] = kn[k - 1]; for (int c = 0; c < 2 * L; ++c) kb[k * 2 * L + c] = kb[(k - 1) * 2 * L + c]; }
    kr[pos] = r; kn[pos] = node; for (int c = 0; c < 2 * L; ++c) kb[pos * 2 * L + c] = b[c];
}

// profile residual R(ω, R2*) = ‖y‖²_W − ‖b‖², b = (W^½Q)ᴴ y per node; keeps the K best local minima over ω of
// min_{R2*} R; writes node, residual, ρ = M b and the LM start (normalised units) per candidate
extern "C" __global__ void wf_profile(const T* q, const T* w, const int* valid, const T* m, const T* om_n, const T* r2_n,
    const T* y, int nprob, int* cnode, T* cres, T* crho, T* pstart)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nprob) return;
    T yr[ND], yi[ND]; T yy = 0;
    for (int n = 0; n < ND; ++n) { yr[n] = y[n * nprob + tid]; yi[n] = y[(ND + n) * nprob + tid]; yy += w[n] * (yr[n] * yr[n] + yi[n] * yi[n]); }
    T kr[K]; int kn[K]; T kb[K * 2 * L];
    for (int k = 0; k < K; ++k) { kr[k] = (T)INFINITY; kn[k] = -1; for (int c = 0; c < 2 * L; ++c) kb[k * 2 * L + c] = 0; }
    T r1 = (T)INFINITY, r2 = (T)INFINITY; int n1 = -1; T b1[2 * L];
    for (int c = 0; c < 2 * L; ++c) b1[c] = 0;
    for (int i = 0; i < NOM; ++i) {
        T rc = (T)INFINITY; int nc = -1; T bc[2 * L];
        for (int c = 0; c < 2 * L; ++c) bc[c] = 0;
        for (int j = 0; j < NR2; ++j) {
            int node = i * NR2 + j;
            if (!valid[node]) continue;
            T bb[2 * L]; T s = 0;
            for (int l = 0; l < L; ++l) {
                const T* qq = q + ((size_t)(node * L + l) * ND) * 2;
                T br = 0, bi = 0;
                for (int n = 0; n < ND; ++n) { T qr = qq[2 * n], qi = qq[2 * n + 1]; br += qr * yr[n] + qi * yi[n]; bi += qr * yi[n] - qi * yr[n]; }
                bb[2 * l] = br; bb[2 * l + 1] = bi; s += br * br + bi * bi;
            }
            T r = yy - s;
            if (r < rc) { rc = r; nc = node; for (int c = 0; c < 2 * L; ++c) bc[c] = bb[c]; }
        }
        if (i > 0 && (i - 1 == 0 || r1 <= r2) && r1 < rc && n1 >= 0) top_insert(kr, kn, kb, r1, n1, b1);
        r2 = r1; r1 = rc; n1 = nc; for (int c = 0; c < 2 * L; ++c) b1[c] = bc[c];
    }
    if (n1 >= 0 && (NOM == 1 || r1 <= r2)) top_insert(kr, kn, kb, r1, n1, b1);
    for (int k = 0; k < K; ++k) {
        cnode[k * nprob + tid] = kn[k]; cres[k * nprob + tid] = kr[k];
        // a missing candidate repeats the best one
        int kk = kn[k] >= 0 ? k : 0;
        int node = kn[kk];
        T rho[2 * L];
        for (int a = 0; a < L; ++a) {
            T sr = 0, si = 0;
            if (node >= 0) for (int c = a; c < L; ++c) {
                T mr = m[((node * L + a) * L + c) * 2], mi = m[((node * L + a) * L + c) * 2 + 1];
                T br = kb[kk * 2 * L + 2 * c], bi = kb[kk * 2 * L + 2 * c + 1];
                sr += mr * br - mi * bi; si += mr * bi + mi * br;
            }
            rho[2 * a] = sr; rho[2 * a + 1] = si;
            crho[((k * L + a) * 2) * nprob + tid] = sr; crho[((k * L + a) * 2 + 1) * nprob + tid] = si;
        }
        T om = node >= 0 ? om_n[node] : (T)0, r2v = node >= 0 ? r2_n[node] : (T)0;
        T p[NP];
#if SPECIES == 0
        p[0] = rho[0]; p[1] = rho[1]; p[2] = r2v; p[3] = om;
#elif SPECIES == 1
        p[0] = rho[0]; p[1] = rho[1]; p[2] = rho[2]; p[3] = rho[3]; p[4] = r2v; p[5] = om;
#elif SPECIES == 2
        { T ph = atan2_t(rho[1] + rho[3], rho[0] + rho[2]); T cp = cos_t(ph), sp = sin_t(ph);
          p[0] = rho[0] * cp + rho[1] * sp; p[1] = rho[2] * cp + rho[3] * sp; p[2] = ph; p[3] = r2v; p[4] = om; }
#else
        p[0] = rho[0]; p[1] = rho[1]; p[2] = rho[2]; p[3] = rho[3]; p[4] = r2v; p[5] = r2v; p[6] = om;
#endif
        for (int j = 0; j < NP; ++j) pstart[(k * NP + j) * nprob + tid] = p[j];
    }
}

// keep the better start per voxel (start 0 always taken)
extern "C" __global__ void wf_select(const T* p, const T* f, const int* it, const int* st, int k, int nprob,
    T* bp, T* bf, int* bit, int* bst, int* bk)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nprob) return;
    if (k == 0 || f[tid] < bf[tid]) {
        for (int j = 0; j < NP; ++j) bp[j * nprob + tid] = p[j * nprob + tid];
        bf[tid] = f[tid]; bit[tid] = it[tid]; bst[tid] = st[tid]; bk[tid] = k;
    }
}

// back to data units: amplitudes × c, objective × c²
extern "C" __global__ void wf_unnorm(T* bp, T* bf, const T* c, int nprob) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nprob) return;
    T cc = c[tid];
    for (int a = 0; a < NAMP; ++a) bp[amp_idx(a) * nprob + tid] *= cc;
    bf[tid] *= cc * cc;
#if SPECIES == 2
    { T w = bp[tid], f = bp[nprob + tid], ph = bp[2 * nprob + tid];
      if (w + f < (T)0) { bp[tid] = -w; bp[nprob + tid] = -f; ph += (T)3.14159265358979323846; }
      bp[2 * nprob + tid] = atan2_t(sin_t(ph), cos_t(ph)); }
#endif
}
"#;

    fn species_code(s: Species) -> i32 {
        match s {
            Species::Water => 0,
            Species::WaterFat => 1,
            Species::WaterFatCommonPhase => 2,
            Species::WaterFatDualR2 => 3,
        }
    }

    /// compiled kernels + LM solver for one echo set, species and configuration
    pub struct WaterFatGpu<T: GpuScalar> {
        pub wf: WaterFat,
        pub cfg: WfConfig,
        pub prof: Profile,
        stream: Arc<CudaStream>,
        k_norm: CudaFunction,
        k_profile: CudaFunction,
        k_select: CudaFunction,
        k_unnorm: CudaFunction,
        fit: GpuFit,
        q_d: CudaSlice<T>,
        w_d: CudaSlice<T>,
        valid_d: CudaSlice<i32>,
        m_d: CudaSlice<T>,
        om_d: CudaSlice<T>,
        r2_d: CudaSlice<T>,
        vars: Vec<T>,
        lo: Vec<T>,
        hi: Vec<T>,
    }

    fn launch_cfg(n: usize) -> LaunchConfig {
        let block = 128u32;
        LaunchConfig { grid_dim: ((n as u32).div_ceil(block).max(1), 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 }
    }

    impl<T: GpuScalar> WaterFatGpu<T> {
        pub fn new(dev: &Device, wf: &WaterFat, cfg: &WfConfig) -> Result<Self, Error> {
            let prof = Profile::new(wf, &Grid::for_echoes(&wf.te, cfg.om_max, cfg.grid_r2.clone()), None);
            let f32 = T::C_TYPE == "float";
            let fun = |b: &str| if f32 { format!("{b}f") } else { b.to_string() };
            let amp = wf.species.amp_idx();
            let amp_fn = format!(
                "__device__ __forceinline__ int amp_idx(int a) {{ const int t[{}] = {{{}}}; return t[a]; }}\n",
                amp.len(),
                amp.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
            );
            let hdr = format!(
                "typedef {} T;\n#define INFINITY __int_as_float(0x7f800000)\n#define ND {}\n#define L {}\n#define K {}\n#define NOM {}\n#define NR2 {}\n#define NP {}\n#define SPECIES {}\n#define NAMP {}\n\
                 #define sqrt_t {}\n#define atan2_t {}\n#define cos_t {}\n#define sin_t {}\n{amp_fn}",
                T::C_TYPE,
                prof.nd,
                prof.l,
                cfg.n_start,
                prof.n_om,
                prof.n_r2,
                wf.n_param(),
                species_code(wf.species),
                amp.len(),
                fun("sqrt"),
                fun("atan2"),
                fun("cos"),
                fun("sin"),
            );
            let module = dev.load_module("wf_kernels", &format!("{hdr}{KERNELS}"), &CompileSettings::default())?;
            let stream = dev.default_stream();
            let q: Vec<T> = prof.q.iter().flat_map(|c| [T::from_f64(c.re), T::from_f64(c.im)]).collect();
            let m: Vec<T> = prof.m.iter().flat_map(|c| [T::from_f64(c.re), T::from_f64(c.im)]).collect();
            let w: Vec<T> = prof.w.iter().map(|&x| T::from_f64(x)).collect();
            let valid: Vec<i32> = prof.valid.iter().map(|&b| b as i32).collect();
            let om: Vec<T> = prof.om.iter().map(|&x| T::from_f64(x)).collect();
            let r2: Vec<T> = prof.r2.iter().map(|&x| T::from_f64(x)).collect();
            let (lo, hi) = wf.bounds(cfg);
            Ok(WaterFatGpu {
                k_norm: module.load_function("wf_norm").map_err(hasty_cuda::Error::from)?,
                k_profile: module.load_function("wf_profile").map_err(hasty_cuda::Error::from)?,
                k_select: module.load_function("wf_select").map_err(hasty_cuda::Error::from)?,
                k_unnorm: module.load_function("wf_unnorm").map_err(hasty_cuda::Error::from)?,
                fit: GpuFit::new::<T>(dev, &wf.model, &wf.options(cfg))?,
                q_d: stream.clone_htod(&q)?,
                m_d: stream.clone_htod(&m)?,
                w_d: stream.clone_htod(&w)?,
                valid_d: stream.clone_htod(&valid)?,
                om_d: stream.clone_htod(&om)?,
                r2_d: stream.clone_htod(&r2)?,
                vars: wf.vars().into_iter().map(T::from_f64).collect(),
                lo: lo.into_iter().map(T::from_f64).collect(),
                hi: hi.into_iter().map(T::from_f64).collect(),
                wf: wf.clone(),
                cfg: cfg.clone(),
                prof,
                stream,
            })
        }

        /// the whole estimation; `y` [2][n_echo][n_prob] in data units (already in T); results in data units
        pub fn fit_t(&self, y: &[T], n_prob: usize) -> Result<WfFit, Error> {
            let timing = std::env::var("HASTY_QFIT_TIMING").is_ok();
            let t0 = std::time::Instant::now();
            let st = &self.stream;
            let lap = |what: &str| {
                if timing {
                    let _ = st.synchronize();
                    eprintln!("  wf gpu: {what} at {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
                }
            };
            let (nd, n, k, l, np) = (self.wf.n_echo(), n_prob, self.cfg.n_start, self.prof.l, self.wf.n_param());
            let d = Data { n_prob: n, n_data: nd, vars: &self.vars[..], fixed: &[][..], y, w: None };
            let mut gd = GpuData::upload(st, &d)?;
            lap("uploaded");
            let nprob = n as i32;
            let cfg = launch_cfg(n);
            let mut c_d = st.alloc_zeros::<T>(n)?;
            unsafe { st.launch_builder(&self.k_norm).arg(gd.y_mut()).arg(&nprob).arg(&mut c_d).launch(cfg)? };
            let mut node_d = st.alloc_zeros::<i32>(k * n)?;
            let mut res_d = st.alloc_zeros::<T>(k * n)?;
            let mut rho_d = st.alloc_zeros::<T>(k * l * 2 * n)?;
            let mut start_d = st.alloc_zeros::<T>(k * np * n)?;
            unsafe {
                st.launch_builder(&self.k_profile)
                    .arg(&self.q_d)
                    .arg(&self.w_d)
                    .arg(&self.valid_d)
                    .arg(&self.m_d)
                    .arg(&self.om_d)
                    .arg(&self.r2_d)
                    .arg(gd.y())
                    .arg(&nprob)
                    .arg(&mut node_d)
                    .arg(&mut res_d)
                    .arg(&mut rho_d)
                    .arg(&mut start_d)
                    .launch(cfg)?;
            }
            lap("normalised + profile");
            let o = self.wf.options(&self.cfg);
            let mut bp = st.alloc_zeros::<T>(np * n)?;
            let mut bf = st.alloc_zeros::<T>(n)?;
            let mut bit = st.alloc_zeros::<i32>(n)?;
            let mut bst = st.alloc_zeros::<i32>(n)?;
            let mut bk = st.alloc_zeros::<i32>(n)?;
            for kk in 0..k {
                let mut p_d = st.alloc_zeros::<T>(np * n)?;
                st.memcpy_dtod(&start_d.slice(kk * np * n..(kk + 1) * np * n), &mut p_d)?;
                let r = self.fit.fit_dev(&gd, &mut p_d, &self.lo, &self.hi, &o)?;
                let kk32 = kk as i32;
                unsafe {
                    st.launch_builder(&self.k_select)
                        .arg(&p_d)
                        .arg(&r.f)
                        .arg(&r.iters)
                        .arg(&r.status)
                        .arg(&kk32)
                        .arg(&nprob)
                        .arg(&mut bp)
                        .arg(&mut bf)
                        .arg(&mut bit)
                        .arg(&mut bst)
                        .arg(&mut bk)
                        .launch(cfg)?;
                }
                lap(&format!("start {kk}"));
            }
            unsafe { st.launch_builder(&self.k_unnorm).arg(&mut bp).arg(&mut bf).arg(&c_d).arg(&nprob).launch(cfg)? };
            let params: Vec<f64> = st.clone_dtoh(&bp)?.into_iter().map(|x| x.to_f64()).collect();
            let f: Vec<f64> = st.clone_dtoh(&bf)?.into_iter().map(|x| x.to_f64()).collect();
            let iters: Vec<u32> = st.clone_dtoh(&bit)?.into_iter().map(|x| x as u32).collect();
            let status: Vec<Status> = st.clone_dtoh(&bst)?.into_iter().map(Status::from_i32).collect();
            let start: Vec<u8> = st.clone_dtoh(&bk)?.into_iter().map(|x| x as u8).collect();
            let node = st.clone_dtoh(&node_d)?;
            let resid: Vec<f64> = st.clone_dtoh(&res_d)?.into_iter().map(|x| x.to_f64()).collect();
            let rr = st.clone_dtoh(&rho_d)?;
            let mut rho = vec![C::new(0.0, 0.0); k * l * n];
            for (i, r) in rho.iter_mut().enumerate() {
                let (kl, v) = (i / n, i % n);
                *r = C::new(rr[(kl * 2) * n + v].to_f64(), rr[(kl * 2 + 1) * n + v].to_f64());
            }
            lap("downloaded");
            Ok(WfFit { params, f, iters, status, start, cand: Candidates { k, n_prob: n, l, node, resid, rho } })
        }

        /// as [`Self::fit_t`] from f64 data
        pub fn fit(&self, y: &[f64], n_prob: usize) -> Result<WfFit, Error> {
            let yt: Vec<T> = y.iter().map(|&x| T::from_f64(x)).collect();
            self.fit_t(&yt, n_prob)
        }
    }
}
