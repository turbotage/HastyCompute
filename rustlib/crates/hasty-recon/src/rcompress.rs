//! Exact per-readout sample compression of the data term (research/sample_compression.md).
//!
//! Within one readout every signal the operator can produce for an image inside a region of interest (ROI) is band
//! limited along the readout: a source at ρ with complex rate z gives v(t_j) = e^{−i k_j·ρ − z (τ_j − τ₀)} times slowly
//! varying factors (cardiac knot weights, concomitant phase). Its instantaneous frequency lies in a band of width
//! B(t) = 2 h(k̇) + (ω_hi − ω_lo), h the ROI's support width along k̇ (ball 2 r |k̇|, box 2 Σ_a h_a |k̇_a|). Where the
//! readout samples that band faster than Nyquist, a subset S of its own samples determines all of them:
//!   v_j ≈ Σ_{s ∈ S_j} E_js v_s   (E: q-tap local least-squares interpolator, kriging with the ROI's covariance)
//! and the data term is exact up to an image-independent constant:
//!   ‖A x − y‖² = ‖E A_S x − y‖² = (A_S x)ᴴ W (A_S x) − 2 Re (A_S x)ᴴ Eᴴ y + ‖y‖²,  W = Eᴴ E (banded).
//! This module uses the Cholesky factor of the banded Gram, W = Rᴴ R (R upper banded, q − 1 superdiagonals):
//!   ‖A x − y‖² = ‖R A_S x − ỹ‖² + (‖y‖² − ‖ỹ‖²),   ỹ = R⁻ᴴ Eᴴ y.
//! ỹ = Qᴴ y for the QR factorisation E = Q R, i.e. the coordinates of y's projection onto range(E): ‖ỹ‖ ≤ ‖y‖ and
//! computing it is backward stable (cond(R) = cond(W)^{1/2} ≈ 300 at the measured cond(W) ≈ 1e5; f64 forward
//! substitution). W⁻¹ is never formed, and there are no "whitened virtual samples" W^{−1/2} b. With this form the
//! solvers see an ordinary least-squares problem: operator R A_S (one banded multiply after the SegNufft forward, Rᴴ
//! before the adjoint, [`RcSets`] / [`RcLin`]) and data ỹ, with the per-readout, per-coil constant ‖y‖² − ‖ỹ‖² kept
//! for objectives and held-out residuals ([`SetComp::cnorm`]).
//!
//! Design ([`Design::build`]): subset S uniform in warped time s(t) = ∫ B dt / 2π at `os` samples per Nyquist unit,
//! ≤ `max_gap` original samples between kept ones, the first / last `keep_ends` samples all kept (gradient ramps at
//! k ≈ 0, where the instantaneous-band estimate fails); E row j from the q subset samples nearest to j, fitted against
//! the covariance K(i, j) = E[v_i conj(v_j)] of a source measure that puts half its mass uniformly in the ROI × band
//! and half on the band edge (ROI boundary, band ends) — the P → ∞ limit of the training sketch of the study (uniform
//! training leaves the band edge 10× worse). Kept samples are reproduced exactly.
//!
//! One design serves every readout that is a rotation of the design's readout (same τ grid, k_r = R_r k₀) when the
//! ROI is a ball: the ball's covariance and band depend on |Δk| only. A box ROI needs one design per readout direction
//! (cached per distinct readout: the encodes of one rotation share it). ROIs are centred on the operator's image
//! origin (voxel d/2 of every axis), so no demodulation is needed. The ROI comes from the prescription (the
//! reconstruction grid) and the rate band from the operator's own ratemap, never from phantom truth.
//!
//! Not DCW: the estimator is the unweighted LS estimator (the objective changes by a constant only); noise in ỹ is
//! white (cov ỹ = σ² R⁻ᴴ W R⁻¹ = σ² I). Per-sample weights that vary within a readout (masks, IRLS) must be applied
//! before compressing; per-readout scalars, held-out masks (whole readouts), coil whitening / compression / subsets
//! commute. Preconditioners (circulant metric) must be built from the original sample coordinates.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};
use num_complex::Complex64 as C64;
use rayon::prelude::*;

use crate::data::SampleSet;
use crate::gpu::Kernels;
use crate::setop::{LinOp, SetOp};
use crate::Error;

/// Region of interest, centred on the operator's image origin (logical frame, metres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Roi {
    /// ball of radius r
    Ball { r: f64 },
    /// box of half extents h
    Box { h: [f64; 3] },
    /// axis-aligned ellipsoid of semi-axes h (e.g. the one inscribed in the grid: a body-like support; per readout,
    /// unknowns outside masked)
    Ellipsoid { h: [f64; 3] },
}

impl Roi {
    /// the box spanned by a reconstruction grid (voxels at (i − d/2) Δ, plus half a voxel)
    pub fn grid_box(dims: [usize; 3], vox: [f64; 3]) -> Self {
        Roi::Box { h: [0, 1, 2].map(|a| (dims[a] / 2) as f64 * vox[a] + 0.5 * vox[a]) }
    }
    /// the ball enclosing [`Roi::grid_box`]
    pub fn grid_ball(dims: [usize; 3], vox: [f64; 3]) -> Self {
        match Self::grid_box(dims, vox) {
            Roi::Box { h } => Roi::Ball { r: (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt() },
            _ => unreachable!(),
        }
    }
    pub fn describe(&self) -> String {
        match self {
            Roi::Ball { r } => format!("ball r {:.1} mm", r * 1e3),
            Roi::Box { h } => format!("box {:.1} × {:.1} × {:.1} mm", 2e3 * h[0], 2e3 * h[1], 2e3 * h[2]),
            Roi::Ellipsoid { h } => format!("ellipsoid {:.1} × {:.1} × {:.1} mm", 2e3 * h[0], 2e3 * h[1], 2e3 * h[2]),
        }
    }
    /// support width along a direction (2 × the support function of the centred ROI) times |d|
    #[inline]
    fn width(&self, d: &[f64; 3]) -> f64 {
        match self {
            Roi::Ball { r } => 2.0 * r * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt(),
            Roi::Box { h } => 2.0 * (h[0] * d[0].abs() + h[1] * d[1].abs() + h[2] * d[2].abs()),
            Roi::Ellipsoid { h } => 2.0 * ((h[0] * d[0]).powi(2) + (h[1] * d[1]).powi(2) + (h[2] * d[2]).powi(2)).sqrt(),
        }
    }
}

/// Rate band of the signal (from the operator's ratemap): frequencies ω ∈ [w_lo, w_hi] rad/s (signal e^{−iωτ}),
/// decay R ∈ [0, r_max] 1/s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    pub w_lo: f64,
    pub w_hi: f64,
    pub r_max: f64,
}

impl Band {
    /// ±1.5 kHz, decay ≤ 300 1/s: when no ratemap is at hand
    pub fn default_wide() -> Self {
        let w = 2.0 * std::f64::consts::PI * 1500.0;
        Band { w_lo: -w, w_hi: w, r_max: 300.0 }
    }
    /// band of a ratemap: weighted [q, 1 − q] quantiles of ω (rad/s; `w` per voxel, e.g. coil energy) widened by
    /// `margin` × its half width on each side (and to at least ±`floor_hz`), decay up to the weighted 1 − q quantile
    /// of `r` (None: 0)
    pub fn from_ratemap(omega: &[f64], r: Option<&[f64]>, w: &[f64], q: f64, margin: f64, floor_hz: f64) -> Self {
        let wq = |v: &[f64], qq: f64| -> f64 {
            let mut o: Vec<usize> = (0..v.len()).filter(|&i| w[i] > 0.0 && v[i].is_finite()).collect();
            if o.is_empty() {
                return 0.0;
            }
            o.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
            let tot: f64 = o.iter().map(|&i| w[i]).sum();
            let mut c = 0.0;
            for &i in &o {
                c += w[i];
                if c >= qq * tot {
                    return v[i];
                }
            }
            v[*o.last().unwrap()]
        };
        let (lo, hi) = (wq(omega, q), wq(omega, 1.0 - q));
        let (mid, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
        let half = (half * (1.0 + margin)).max(2.0 * std::f64::consts::PI * floor_hz);
        let r_max = r.map_or(0.0, |r| wq(r, 1.0 - q).max(0.0));
        Band { w_lo: mid - half, w_hi: mid + half, r_max }
    }
    /// add a frequency interval (e.g. the fat peaks: ω + 2π γ̄ B0 ppm)
    pub fn union(self, w_lo: f64, w_hi: f64) -> Self {
        Band { w_lo: self.w_lo.min(w_lo), w_hi: self.w_hi.max(w_hi), r_max: self.r_max }
    }
    pub fn describe(&self) -> String {
        let tw = 2.0 * std::f64::consts::PI;
        format!("[{:.0}, {:.0}] Hz, decay ≤ {:.0} 1/s", self.w_lo / tw, self.w_hi / tw, self.r_max)
    }
}

/// Design options. Defaults (`RcOpts::new`): os 1.35, q 32, 64 end samples, ≤ 3 samples apart, half the source
/// measure on the band edge, ridge 1e-10 (relative to the window's mean diagonal).
#[derive(Clone, Debug)]
pub struct RcOpts {
    pub roi: Roi,
    pub band: Band,
    pub os: f64,
    pub q: usize,
    pub keep_ends: usize,
    pub max_gap: usize,
    pub edge: f64,
    pub ridge: f64,
    /// samples kept on each side of a gap in τ (a readout cut into pieces, e.g. by a k-space crop): the pieces' ends
    /// are interpolated one-sided otherwise
    pub gap_keep: usize,
}

impl RcOpts {
    pub fn new(roi: Roi, band: Band) -> Self {
        let gap_keep = std::env::var("HASTY_RC_GAPKEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        RcOpts { roi, band, os: 1.35, q: 32, keep_ends: 64, max_gap: 3, edge: 0.5, ridge: 1e-10, gap_keep }
    }
    /// shared designs (one per τ grid, valid for all rotations): ball ROIs
    pub fn shared(&self) -> bool {
        matches!(self.roi, Roi::Ball { .. })
    }
}

#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        1.0 - x * x / 6.0
    } else {
        x.sin() / x
    }
}

/// characteristic function of the uniform ball: 3 (sin x − x cos x) / x³
#[inline]
fn jball(x: f64) -> f64 {
    if x.abs() < 1e-2 {
        let x2 = x * x;
        1.0 - x2 / 10.0 + x2 * x2 / 280.0
    } else {
        3.0 * (x.sin() - x * x.cos()) / (x * x * x)
    }
}

/// E[e^{−R s}], R uniform in [0, rm]
#[inline]
fn decay_mean(rm: f64, s: f64) -> f64 {
    let a = rm * s;
    if a.abs() < 1e-6 {
        1.0 - 0.5 * a
    } else {
        (1.0 - (-a).exp()) / a
    }
}

/// The covariance K(i, j) = E[v_i conj(v_j)] of the design's source measure on one readout.
struct Cov<'a> {
    o: &'a RcOpts,
    k: &'a [[f64; 3]],
    tau: &'a [f64],
    t0: f64,
}

impl Cov<'_> {
    #[inline]
    fn eval(&self, i: usize, j: usize) -> C64 {
        let (ki, kj) = (&self.k[i], &self.k[j]);
        let d = [ki[0] - kj[0], ki[1] - kj[1], ki[2] - kj[2]];
        let (rho_u, rho_e) = match self.o.roi {
            Roi::Ball { r } => {
                let x = r * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let u = jball(x);
                // edge sketch: 70 % on the sphere, 30 % uniform
                (u, 0.7 * sinc(x) + 0.3 * u)
            }
            Roi::Box { h } => {
                let (mut u, mut e) = (1.0, 1.0);
                for a in 0..3 {
                    let x = h[a] * d[a];
                    let s = sinc(x);
                    u *= s;
                    // edge sketch: each coordinate at ±h with probability 0.67
                    e *= 0.67 * x.cos() + 0.33 * s;
                }
                (u, e)
            }
            Roi::Ellipsoid { h } => {
                let x = ((h[0] * d[0]).powi(2) + (h[1] * d[1]).powi(2) + (h[2] * d[2]).powi(2)).sqrt();
                let u = jball(x);
                (u, 0.7 * sinc(x) + 0.3 * u)
            }
        };
        let b = &self.o.band;
        let dt = self.tau[i] - self.tau[j];
        let wm = 0.5 * (b.w_lo + b.w_hi);
        let wu = C64::from_polar(sinc(0.5 * (b.w_hi - b.w_lo) * dt), -wm * dt);
        let we = 0.5 * wu + 0.25 * (C64::from_polar(1.0, -b.w_lo * dt) + C64::from_polar(1.0, -b.w_hi * dt));
        let dec = decay_mean(b.r_max, (self.tau[i] - self.t0) + (self.tau[j] - self.t0));
        ((1.0 - self.o.edge) * rho_u * wu + self.o.edge * rho_e * we) * dec
    }
}

/// instantaneous two-sided band (rad/s) per sample and the median dwell
pub fn band_of(k: &[[f64; 3]], tau: &[f64], o: &RcOpts) -> (Vec<f64>, f64) {
    let n = k.len();
    let mut dts: Vec<f64> = tau.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0.0).collect();
    dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dwell = dts.get(dts.len() / 2).copied().unwrap_or(1e-6);
    if n < 2 {
        return (vec![0.0; n], dwell);
    }
    let kd: Vec<[f64; 3]> = (0..n - 1).map(|j| {
        let dt = (tau[j + 1] - tau[j]).max(1e-12);
        [0, 1, 2].map(|a| (k[j + 1][a] - k[j][a]) / dt)
    }).collect();
    let wb = o.band.w_hi - o.band.w_lo;
    let b = (0..n).map(|j| {
        let d = if j == 0 { kd[0] } else if j == n - 1 { kd[n - 2] } else { [0, 1, 2].map(|a| 0.5 * (kd[j - 1][a] + kd[j][a])) };
        o.roi.width(&d) + wb
    }).collect();
    (b, dwell)
}

/// subset uniform in warped time at `os` per Nyquist unit (see the module docs); ascending sample indices
pub fn select_warped(b: &[f64], tau: &[f64], dwell: f64, o: &RcOpts) -> Vec<u32> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }
    let bmin = 2.0 * std::f64::consts::PI / (dwell * o.max_gap as f64 * o.os);
    let mut s = vec![0.0f64; n];
    for j in 1..n {
        // a gap in τ (cropped samples) counts at most one dwell's worth of the floor: never skip across it
        let dt = (tau[j] - tau[j - 1]).max(0.0);
        let gap = dt > 1.5 * dwell;
        s[j] = s[j - 1] + if gap { 1.0 / o.os } else { b[j].max(bmin) * dt / (2.0 * std::f64::consts::PI) };
    }
    let mut keep = vec![false; n];
    let step = 1.0 / o.os;
    let mut tgt = 0.0;
    let mut j = 0;
    while tgt < s[n - 1] && j < n {
        while j < n && s[j] < tgt {
            j += 1;
        }
        if j < n {
            keep[j] = true;
        }
        tgt += step;
    }
    keep[n - 1] = true;
    for j in 0..o.keep_ends.min(n) {
        keep[j] = true;
        keep[n - 1 - j] = true;
    }
    // gaps: `gap_keep` samples on each side of a τ jump (at least the two neighbours)
    for j in 1..n {
        if tau[j] - tau[j - 1] > 1.5 * dwell {
            let g = o.gap_keep.max(1);
            for i in j.saturating_sub(g)..(j + g).min(n) {
                keep[i] = true;
            }
        }
    }
    (0..n).filter(|&j| keep[j]).map(|j| j as u32).collect()
}

/// Reduction n / |S| of the selection rule alone (no interpolator: cheap) over up to `max_readouts` evenly spread
/// readouts of a set (`lens` samples per readout, coordinates k·Δ with voxel size `vox`): the `auto` decision.
pub fn estimate_reduction(set: &SampleSet, lens: &[usize], vox: [f64; 3], o: &RcOpts, max_readouts: usize) -> f64 {
    let offs: Vec<usize> = lens.iter().scan(0, |a, &l| { let r = *a; *a += l; Some(r) }).collect();
    let step = (lens.len() / max_readouts.max(1)).max(1);
    let (n, s) = (0..lens.len())
        .step_by(step)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&r| {
            let (a, l) = (offs[r], lens[r]);
            let k: Vec<[f64; 3]> = (a..a + l).map(|j| [0, 1, 2].map(|x| set.coords[x][j] as f64 / vox[x])).collect();
            let (b, dwell) = band_of(&k, &set.tau[a..a + l], o);
            (l, select_warped(&b, &set.tau[a..a + l], dwell, o).len())
        })
        .reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1));
    n as f64 / s.max(1) as f64
}

/// In-place Cholesky of a Hermitian positive-definite q × q matrix (row-major, lower factor L, G = L Lᴴ).
fn chol(g: &mut [C64], q: usize) -> bool {
    for i in 0..q {
        for j in 0..=i {
            let mut s = g[i * q + j];
            for k in 0..j {
                s -= g[i * q + k] * g[j * q + k].conj();
            }
            if i == j {
                if !(s.re > 0.0) {
                    return false;
                }
                g[i * q + i] = C64::new(s.re.sqrt(), 0.0);
            } else {
                g[i * q + j] = s / g[j * q + j].re;
            }
        }
    }
    true
}

/// solve L Lᴴ x = b in place (L from [`chol`])
fn chol_solve(l: &[C64], q: usize, b: &mut [C64]) {
    for i in 0..q {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * q + k] * b[k];
        }
        b[i] = s / l[i * q + i].re;
    }
    for i in (0..q).rev() {
        let mut s = b[i];
        for k in i + 1..q {
            s -= l[k * q + i].conj() * b[k];
        }
        b[i] = s / l[i * q + i].re;
    }
}

/// One readout's compression: subset, interpolator and the Cholesky factor of its banded Gram.
#[derive(Clone)]
pub struct Design {
    /// original samples
    pub n: usize,
    /// window width (taps)
    pub q: usize,
    /// kept samples (ascending indices into the readout)
    pub sel: Vec<u32>,
    /// per original sample: the first subset position of its window (q consecutive subset samples)
    pub start: Vec<u32>,
    /// per original sample: the q interpolation weights (n × q); empty after [`Design::strip`]
    pub coef: Vec<C64>,
    /// R (upper banded): row a holds R[a, a + d], d < q (|S| × q; entries past |S| are 0); empty after
    /// [`Design::strip`], which keeps the f32 copy `r32`
    pub r: Vec<C64>,
    /// R as f32 (|S| × q), filled by [`Design::strip`]
    pub r32: Vec<C32>,
}

impl Design {
    /// Design for one readout: k in rad/m (logical frame, ROI frame), τ in s (ascending).
    pub fn build(k: &[[f64; 3]], tau: &[f64], o: &RcOpts) -> Design {
        let n = k.len();
        assert_eq!(tau.len(), n);
        if n == 0 {
            return Design { n, q: 1, sel: vec![], start: vec![], coef: vec![], r: vec![], r32: vec![] };
        }
        let (b, dwell) = band_of(k, tau, o);
        let sel = select_warped(&b, tau, dwell, o);
        let ns = sel.len();
        let q = o.q.min(ns).max(1);
        let cov = Cov { o, k, tau, t0: tau.first().copied().unwrap_or(0.0) };
        // K over the subset, banded: ks[a * q + d] = K(S[a], S[a + d])
        let mut ks = vec![C64::new(0.0, 0.0); ns * q];
        ks.par_chunks_mut(q).enumerate().for_each(|(a, row)| {
            for d in 0..q.min(ns - a) {
                row[d] = cov.eval(sel[a] as usize, sel[a + d] as usize);
            }
        });
        let kss = |a: usize, b: usize| -> C64 { if b >= a { ks[a * q + (b - a)] } else { ks[b * q + (a - b)].conj() } };
        // window start per original sample
        let mut start = vec![0u32; n];
        let mut pos = 0usize;
        for j in 0..n {
            while pos < ns && (sel[pos] as usize) < j {
                pos += 1;
            }
            start[j] = (pos as i64 - (q / 2) as i64).clamp(0, (ns - q) as i64) as u32;
        }
        // rows grouped by window start (start is non-decreasing in j)
        let mut groups: Vec<(usize, usize)> = vec![];
        let mut j0 = 0;
        for j in 1..=n {
            if j == n || start[j] != start[j0] {
                groups.push((j0, j));
                j0 = j;
            }
        }
        let where_: HashMap<u32, usize> = sel.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let mut coef = vec![C64::new(0.0, 0.0); n * q];
        let parts: Vec<(usize, Vec<C64>)> = groups
            .par_iter()
            .map(|&(a, bnd)| {
                let st = start[a] as usize;
                // G[q1][q2] = E[conj(v_s1) v_s2] = K(s2, s1)
                let mut g = vec![C64::new(0.0, 0.0); q * q];
                for q1 in 0..q {
                    for q2 in 0..q {
                        g[q1 * q + q2] = kss(st + q2, st + q1);
                    }
                }
                let tr: f64 = (0..q).map(|i| g[i * q + i].re).sum::<f64>() / q as f64;
                let mut ridge = o.ridge;
                let mut l = g.clone();
                loop {
                    for i in 0..q {
                        l[i * q + i] = g[i * q + i] + ridge * tr;
                    }
                    if chol(&mut l, q) {
                        break;
                    }
                    l.copy_from_slice(&g);
                    ridge *= 10.0;
                }
                let mut out = vec![C64::new(0.0, 0.0); (bnd - a) * q];
                for j in a..bnd {
                    let row = &mut out[(j - a) * q..(j - a + 1) * q];
                    if let Some(&p) = where_.get(&(j as u32)) {
                        row[p - st] = C64::new(1.0, 0.0);
                        continue;
                    }
                    for q1 in 0..q {
                        row[q1] = cov.eval(j, sel[st + q1] as usize);
                    }
                    chol_solve(&l, q, row);
                }
                (a, out)
            })
            .collect();
        for (a, out) in parts {
            coef[a * q..a * q + out.len()].copy_from_slice(&out);
        }
        // W = Eᴴ E (upper band), then W = Rᴴ R
        let mut w = vec![C64::new(0.0, 0.0); ns * q];
        for j in 0..n {
            let st = start[j] as usize;
            let c = &coef[j * q..(j + 1) * q];
            for q1 in 0..q {
                let a = c[q1].conj();
                if a == C64::new(0.0, 0.0) {
                    continue;
                }
                for q2 in q1..q {
                    w[(st + q1) * q + (q2 - q1)] += a * c[q2];
                }
            }
        }
        let r = band_cholesky(&w, ns, q);
        Design { n, q, sel, start, coef, r, r32: vec![] }
    }

    pub fn ns(&self) -> usize {
        self.sel.len()
    }

    /// v ≈ E v_S (for tests and gates): `vs` per subset sample, returns n values
    pub fn interpolate(&self, vs: &[C64]) -> Vec<C64> {
        (0..self.n).map(|j| {
            let st = self.start[j] as usize;
            (0..self.q).map(|q1| self.coef[j * self.q + q1] * vs[st + q1]).sum()
        }).collect()
    }

    /// ỹ = R⁻ᴴ Eᴴ y of one coil's readout (n values) into `out` (|S|); returns (‖y‖², ‖ỹ‖²)
    pub fn compress(&self, y: &[C64], out: &mut [C64]) -> (f64, f64) {
        assert!(self.n == 0 || !self.coef.is_empty(), "rcompress: design stripped of its interpolator");
        let (q, ns) = (self.q, self.ns());
        out.iter_mut().for_each(|v| *v = C64::new(0.0, 0.0));
        let mut yy = 0.0;
        for j in 0..self.n {
            yy += y[j].norm_sqr();
            let st = self.start[j] as usize;
            for q1 in 0..q {
                out[st + q1] += self.coef[j * q + q1].conj() * y[j];
            }
        }
        // Rᴴ ỹ = b (forward substitution; (Rᴴ)[a][k] = conj(R[k][a]))
        for a in 0..ns {
            let mut s = out[a];
            for d in 1..q.min(a + 1) {
                s -= self.r[(a - d) * q + d].conj() * out[a - d];
            }
            out[a] = s / self.r[a * q].re;
        }
        (yy, out.iter().map(|v| v.norm_sqr()).sum())
    }

    /// R v (forward virtual data) of subset values v (|S|), f64 (tests)
    pub fn apply_r(&self, v: &[C64]) -> Vec<C64> {
        let (q, ns) = (self.q, self.ns());
        (0..ns).map(|a| (0..q.min(ns - a)).map(|d| self.r[a * q + d] * v[a + d]).sum()).collect()
    }

    /// drop the interpolator and keep R as f32 only (what the operator needs)
    pub fn strip(&mut self) {
        if self.r32.is_empty() && !self.r.is_empty() {
            self.r32 = self.r.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect();
        }
        self.coef = Vec::new();
        self.start = Vec::new();
        self.r = Vec::new();
    }

    pub fn is_stripped(&self) -> bool {
        self.r.is_empty() && self.n > 0
    }

    /// R as f32 (|S| × q_out, rows zero-padded when the design has fewer taps: readouts shorter than q), the
    /// operator's table
    pub fn r_f32(&self, q_out: usize) -> Vec<C32> {
        assert!(q_out >= self.q);
        let mut out = vec![C32::new(0.0, 0.0); self.ns() * q_out];
        for a in 0..self.ns() {
            for d in 0..self.q {
                out[a * q_out + d] = if self.r.is_empty() { self.r32[a * self.q + d] } else { let z = self.r[a * self.q + d]; C32::new(z.re as f32, z.im as f32) };
            }
        }
        out
    }
}

/// Cholesky of a Hermitian banded matrix given by its upper band (w[a * q + d] = W[a, a + d]): R upper banded with
/// W = Rᴴ R. A tiny diagonal shift (1e-13 of the largest diagonal, then growing) keeps it definite.
fn band_cholesky(w: &[C64], ns: usize, q: usize) -> Vec<C64> {
    let dmax = (0..ns).map(|a| w[a * q].re).fold(0.0, f64::max).max(1e-300);
    let mut shift = 1e-13 * dmax;
    loop {
        let mut r = vec![C64::new(0.0, 0.0); ns * q];
        let mut ok = true;
        for a in 0..ns {
            // diagonal
            let mut s = w[a * q].re + shift;
            for d in 1..q.min(a + 1) {
                s -= r[(a - d) * q + d].norm_sqr();
            }
            if !(s > 0.0) {
                ok = false;
                break;
            }
            let rd = s.sqrt();
            r[a * q] = C64::new(rd, 0.0);
            // row a: R[a][b], b = a + e
            for e in 1..q.min(ns - a) {
                let mut s = w[a * q + e];
                // Σ_k conj(R[k][a]) R[k][b], k < a, a − k < q and b − k < q
                for d in (e + 1)..q {
                    // k = a − (d − e) so that b − k = d
                    let back = d - e;
                    if back > a {
                        break;
                    }
                    let kk = a - back;
                    s -= r[kk * q + back].conj() * r[kk * q + d];
                }
                r[a * q + e] = s / rd;
            }
        }
        if ok {
            return r;
        }
        shift = (shift * 100.0).max(1e-12 * dmax);
    }
}

/// Rotation R (row-major) with k_r ≈ R k₀ over the samples (Kabsch), and the relative residual max |R k₀ − k_r| /
/// max |k_r|.
pub fn fit_rotation(k0: &[[f64; 3]], kr: &[[f64; 3]]) -> ([[f64; 3]; 3], f64) {
    let mut h = nalgebra::Matrix3::<f64>::zeros();
    for (a, b) in k0.iter().zip(kr) {
        for i in 0..3 {
            for j in 0..3 {
                h[(i, j)] += a[i] * b[j];
            }
        }
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut d = nalgebra::Matrix3::<f64>::identity();
    if (vt.transpose() * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    let rm = vt.transpose() * d * u.transpose();
    let mut err = 0.0f64;
    let mut kmax = 0.0f64;
    for (a, b) in k0.iter().zip(kr) {
        let p = rm * nalgebra::Vector3::new(a[0], a[1], a[2]);
        err = err.max(((p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2) + (p[2] - b[2]).powi(2)).sqrt());
        kmax = kmax.max((b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt());
    }
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = rm[(i, j)];
        }
    }
    (out, err / kmax.max(1e-300))
}

/// The compression layout of one sample set (coil-major data, readouts in order).
#[derive(Clone, Debug, Default)]
pub struct SetComp {
    /// samples per readout before compression (the set's layout)
    pub lens: Vec<usize>,
    /// per readout: its design
    pub design: Vec<u32>,
    /// kept samples (ascending indices into the set's samples)
    pub sel: Vec<u32>,
    /// virtual-sample offsets per readout (nro + 1)
    pub ro_off: Vec<u32>,
    /// per readout and coil (nro × nc, readout-major): ‖y‖² − ‖ỹ‖², the objective's constant (filled by
    /// [`Compressor::compress_data`])
    pub cnorm: Vec<f64>,
    /// per readout and coil: ‖y‖² (the original data energy, e.g. for relative held-out residuals)
    pub ynorm: Vec<f64>,
}

impl SetComp {
    pub fn m(&self) -> usize {
        self.sel.len()
    }
    pub fn n_orig(&self) -> usize {
        self.lens.iter().sum()
    }
}

/// Designs for many readouts plus the per-set layouts.
pub struct Compressor {
    pub opts: RcOpts,
    pub designs: Vec<Design>,
    /// cache: readout signature → design
    cache: HashMap<u64, u32>,
    /// shared mode: per τ-grid signature the base design's k (for the rotation check)
    base_k: HashMap<u64, (u32, Vec<[f64; 3]>)>,
    /// readouts whose rotation check failed (own design)
    pub rotation_misses: usize,
    pub max_rotation_resid: f64,
}

fn hash_f64(h: &mut u64, v: f64) {
    for b in v.to_bits().to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

impl Compressor {
    pub fn new(opts: RcOpts) -> Self {
        Compressor { opts, designs: vec![], cache: HashMap::new(), base_k: HashMap::new(), rotation_misses: 0, max_rotation_resid: 0.0 }
    }

    /// the design of one readout (k rad/m, τ s), created when no cached one fits; returns its index
    pub fn design_for(&mut self, k: &[[f64; 3]], tau: &[f64]) -> u32 {
        let mut ht = 0xcbf2_9ce4_8422_2325u64;
        for &t in tau {
            hash_f64(&mut ht, t);
        }
        if self.opts.shared() {
            if let Some((id, k0)) = self.base_k.get(&ht) {
                let (_, res) = fit_rotation(k0, k);
                if res < 1e-5 {
                    self.max_rotation_resid = self.max_rotation_resid.max(res);
                    return *id;
                }
                self.rotation_misses += 1;
            } else {
                let d = Design::build(k, tau, &self.opts);
                let id = self.designs.len() as u32;
                self.designs.push(d);
                self.base_k.insert(ht, (id, k.to_vec()));
                return id;
            }
        }
        let mut h = ht;
        for kk in k {
            for &v in kk {
                hash_f64(&mut h, v);
            }
        }
        if let Some(&id) = self.cache.get(&h) {
            return id;
        }
        let d = Design::build(k, tau, &self.opts);
        let id = self.designs.len() as u32;
        self.designs.push(d);
        self.cache.insert(h, id);
        id
    }

    /// Plan a set: `lens` samples per readout (in order), k from the set's coordinates (k·Δ → rad/m by `vox`).
    /// Designs are built in parallel over the distinct readouts that need one.
    pub fn plan_set(&mut self, set: &SampleSet, lens: &[usize], vox: [f64; 3]) -> SetComp {
        let m = set.coords[0].len();
        assert_eq!(lens.iter().sum::<usize>(), m, "rcompress: readout lengths vs the set's samples");
        let offs: Vec<usize> = lens.iter().scan(0, |o, &l| { let r = *o; *o += l; Some(r) }).collect();
        let kof = |r: usize| -> (Vec<[f64; 3]>, Vec<f64>) {
            let (o, l) = (offs[r], lens[r]);
            ((o..o + l).map(|j| [0, 1, 2].map(|a| set.coords[a][j] as f64 / vox[a])).collect(), set.tau[o..o + l].to_vec())
        };
        // box mode: build the missing designs in parallel first
        if !self.opts.shared() {
            let mut todo: Vec<(u64, usize)> = vec![];
            let mut seen = std::collections::HashSet::new();
            for r in 0..lens.len() {
                let (k, t) = kof(r);
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for &v in &t {
                    hash_f64(&mut h, v);
                }
                for kk in &k {
                    for &v in kk {
                        hash_f64(&mut h, v);
                    }
                }
                if !self.cache.contains_key(&h) && seen.insert(h) {
                    todo.push((h, r));
                }
            }
            let built: Vec<(u64, Design)> = todo.par_iter().map(|&(h, r)| {
                let (k, t) = kof(r);
                let mut d = Design::build(&k, &t, &self.opts);
                // box designs keep S and an f32 R only (n × q interpolators of every readout would not fit);
                // compress_set rebuilds the interpolator per readout for the data
                d.strip();
                (h, d)
            }).collect();
            for (h, d) in built {
                let id = self.designs.len() as u32;
                self.designs.push(d);
                self.cache.insert(h, id);
            }
        }
        let mut sc = SetComp { lens: lens.to_vec(), ..Default::default() };
        sc.ro_off.push(0);
        for r in 0..lens.len() {
            let (k, t) = kof(r);
            let id = self.design_for(&k, &t);
            let d = &self.designs[id as usize];
            assert_eq!(d.n, lens[r], "rcompress: design length");
            sc.design.push(id);
            sc.sel.extend(d.sel.iter().map(|&s| (offs[r] + s as usize) as u32));
            sc.ro_off.push(sc.sel.len() as u32);
        }
        sc
    }

    /// Compress coil-major data (nc × m, the set's layout) into nc × |S| virtual data; fills `sc.cnorm` / `sc.ynorm`.
    /// Needs the designs' interpolators (shared designs; box designs: use [`compress_set`](Self::compress_set)).
    pub fn compress_data(&self, sc: &mut SetComp, y: &[C32], nc: usize) -> Vec<C32> {
        let (m, ms, nro) = (sc.n_orig(), sc.m(), sc.lens.len());
        assert_eq!(y.len(), nc * m, "rcompress: data length");
        let offs: Vec<usize> = sc.lens.iter().scan(0, |o, &l| { let r = *o; *o += l; Some(r) }).collect();
        let parts: Vec<(Vec<C32>, Vec<(f64, f64)>)> = (0..nro)
            .into_par_iter()
            .map(|r| {
                let d = &self.designs[sc.design[r] as usize];
                let (o, l, ns) = (offs[r], sc.lens[r], d.ns());
                let mut out = vec![C32::new(0.0, 0.0); nc * ns];
                let mut nrm = Vec::with_capacity(nc);
                let mut yb = vec![C64::new(0.0, 0.0); l];
                let mut ob = vec![C64::new(0.0, 0.0); ns];
                for c in 0..nc {
                    for j in 0..l {
                        let v = y[c * m + o + j];
                        yb[j] = C64::new(v.re as f64, v.im as f64);
                    }
                    let (a, b) = d.compress(&yb, &mut ob);
                    for i in 0..ns {
                        out[c * ns + i] = C32::new(ob[i].re as f32, ob[i].im as f32);
                    }
                    nrm.push((a, b));
                }
                (out, nrm)
            })
            .collect();
        let mut yc = vec![C32::new(0.0, 0.0); nc * ms];
        sc.cnorm = vec![0.0; nro * nc];
        sc.ynorm = vec![0.0; nro * nc];
        for (r, (out, nrm)) in parts.into_iter().enumerate() {
            let (o, ns) = (sc.ro_off[r] as usize, (sc.ro_off[r + 1] - sc.ro_off[r]) as usize);
            for c in 0..nc {
                yc[c * ms + o..c * ms + o + ns].copy_from_slice(&out[c * ns..(c + 1) * ns]);
                sc.cnorm[r * nc + c] = (nrm[c].0 - nrm[c].1).max(0.0);
                sc.ynorm[r * nc + c] = nrm[c].0;
            }
        }
        yc
    }

    /// achieved reduction n / |S| over sets
    pub fn reduction(sets: &[SetComp]) -> f64 {
        let (n, s): (usize, usize) = sets.iter().map(|c| (c.n_orig(), c.m())).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        n as f64 / s.max(1) as f64
    }

    /// device bytes of the R tables (f32)
    pub fn table_bytes(&self) -> usize {
        self.designs.iter().map(|d| d.ns() * d.q * 8).sum()
    }

    /// Plan and compress one set in one pass ([`plan_set`](Self::plan_set) + [`compress_data`](Self::compress_data)),
    /// for any number of sets: box designs are built per readout in chunks, used for the readout's data and kept as S
    /// and an f32 R only (a design of an earlier set is reused for its R; its interpolator is rebuilt for the data);
    /// the shared ball design keeps its interpolator.
    pub fn compress_set(&mut self, set: &SampleSet, lens: &[usize], vox: [f64; 3], y: &[C32], nc: usize) -> (SetComp, Vec<C32>) {
        if self.opts.shared() {
            let mut sc = self.plan_set(set, lens, vox);
            let yc = self.compress_data(&mut sc, y, nc);
            return (sc, yc);
        }
        let m = set.coords[0].len();
        assert_eq!(lens.iter().sum::<usize>(), m, "rcompress: readout lengths vs the set's samples");
        assert_eq!(y.len(), nc * m, "rcompress: data length");
        let offs: Vec<usize> = lens.iter().scan(0, |o, &l| { let r = *o; *o += l; Some(r) }).collect();
        let nro = lens.len();
        let mut sc = SetComp { lens: lens.to_vec(), ..Default::default() };
        sc.ro_off.push(0);
        sc.cnorm = vec![0.0; nro * nc];
        sc.ynorm = vec![0.0; nro * nc];
        let mut parts: Vec<Vec<C32>> = Vec::with_capacity(nro);
        const CHUNK: usize = 512;
        for c0 in (0..nro).step_by(CHUNK) {
            let c1 = (c0 + CHUNK).min(nro);
            let opts = &self.opts;
            let out: Vec<(u64, Design, Vec<C32>, Vec<(f64, f64)>)> = (c0..c1)
                .into_par_iter()
                .map(|r| {
                    let (o, l) = (offs[r], lens[r]);
                    let k: Vec<[f64; 3]> = (o..o + l).map(|j| [0, 1, 2].map(|a| set.coords[a][j] as f64 / vox[a])).collect();
                    let tau = &set.tau[o..o + l];
                    let mut h = 0xcbf2_9ce4_8422_2325u64;
                    for &t in tau {
                        hash_f64(&mut h, t);
                    }
                    for kk in &k {
                        for &v in kk {
                            hash_f64(&mut h, v);
                        }
                    }
                    let mut d = Design::build(&k, tau, opts);
                    let ns = d.ns();
                    let mut yc = vec![C32::new(0.0, 0.0); nc * ns];
                    let mut nrm = Vec::with_capacity(nc);
                    let (mut yb, mut ob) = (vec![C64::new(0.0, 0.0); l], vec![C64::new(0.0, 0.0); ns]);
                    for c in 0..nc {
                        for j in 0..l {
                            let v = y[c * m + o + j];
                            yb[j] = C64::new(v.re as f64, v.im as f64);
                        }
                        nrm.push(d.compress(&yb, &mut ob));
                        for i in 0..ns {
                            yc[c * ns + i] = C32::new(ob[i].re as f32, ob[i].im as f32);
                        }
                    }
                    d.strip();
                    (h, d, yc, nrm)
                })
                .collect();
            for (i, (h, d, yc, nrm)) in out.into_iter().enumerate() {
                let r = c0 + i;
                let id = match self.cache.get(&h) {
                    Some(&id) => id,
                    None => {
                        let id = self.designs.len() as u32;
                        self.designs.push(d);
                        self.cache.insert(h, id);
                        id
                    }
                };
                let dd = &self.designs[id as usize];
                sc.design.push(id);
                sc.sel.extend(dd.sel.iter().map(|&s| (offs[r] + s as usize) as u32));
                sc.ro_off.push(sc.sel.len() as u32);
                for c in 0..nc {
                    sc.cnorm[r * nc + c] = (nrm[c].0 - nrm[c].1).max(0.0);
                    sc.ynorm[r * nc + c] = nrm[c].0;
                }
                parts.push(yc);
            }
        }
        let ms = sc.m();
        let mut yc = vec![C32::new(0.0, 0.0); nc * ms];
        for (r, p) in parts.into_iter().enumerate() {
            let (o, ns) = (sc.ro_off[r] as usize, (sc.ro_off[r + 1] - sc.ro_off[r]) as usize);
            for c in 0..nc {
                yc[c * ms + o..c * ms + o + ns].copy_from_slice(&p[c * ns..(c + 1) * ns]);
            }
        }
        (sc, yc)
    }

    pub fn describe(&self) -> String {
        format!("{}, band {}, os {}, q {}, {} design(s){}", self.opts.roi.describe(), self.opts.band.describe(), self.opts.os, self.opts.q, self.designs.len(), if self.opts.shared() { format!(" (rotation check: {} misses, max residual {:.1e})", self.rotation_misses, self.max_rotation_resid) } else { String::new() })
    }
}

/// Gate statistics of a design on test signatures (see [`gate`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct GateStats {
    /// ‖V − E V_S‖_F / ‖V‖_F
    pub frob: f64,
    /// worst source: max_p ‖v_p − E v_p,S‖ / ‖v_p‖
    pub worst: f64,
    /// ‖Vᴴ V − (R V_S)ᴴ (R V_S)‖₂ / ‖Vᴴ V‖₂ (power iteration): the normal-operator error on the span of the sources
    pub gram: f64,
}

impl GateStats {
    pub fn max(self, o: GateStats) -> GateStats {
        GateStats { frob: self.frob.max(o.frob), worst: self.worst.max(o.worst), gram: self.gram.max(o.gram) }
    }
}

/// Direct-sum gate of a design on one readout: `p` point sources (half uniform in the ROI × band, half on its edge;
/// independent of the design's measure: a fresh deterministic draw from `seed`), signatures
///   v_j = β(t_j) e^{−i k_j·ρ − z (τ_j − τ₀) − i w_j·mono(ρ)}
/// with the readout's concomitant coefficients `wq` (rad/m², [z², x²+y², xz, yz]; None: no concomitant phase) and a
/// cubic B-spline knot weight β of the per-sample cardiac knot coordinate `knot_t` (a random active knot per source;
/// None: β = 1). Errors on the readout's own samples (k rad/m).
pub fn gate(d: &Design, k: &[[f64; 3]], tau: &[f64], wq: Option<&[[f64; 4]]>, knot_t: Option<&[f64]>, o: &RcOpts, p: usize, seed: u64) -> GateStats {
    let n = k.len();
    let mut s = seed.max(1) ^ 0x9E37_79B9_7F4A_7C15;
    let mut u = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let t0 = tau[0];
    let b = o.band;
    let mut cols: Vec<Vec<C64>> = Vec::with_capacity(p);
    for pi in 0..p {
        let edge = pi % 2 == 1;
        let rho: [f64; 3] = match o.roi {
            Roi::Ball { r } => {
                // direction from a Gaussian, radius r u^{1/3} (edge: 70 % on the sphere)
                let g: [f64; 3] = [0; 3].map(|_| { let (a, bb) = (u().max(1e-300), u()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * bb).cos() });
                let ng = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-300);
                let rad = if edge && u() < 0.7 { 1.0 } else { u().cbrt() };
                g.map(|v| r * rad * v / ng)
            }
            Roi::Box { h } => [0, 1, 2].map(|a| {
                let v = 2.0 * u() - 1.0;
                h[a] * if edge && u() < 0.67 { v.signum() } else { v }
            }),
            Roi::Ellipsoid { h } => {
                let g: [f64; 3] = [0; 3].map(|_| { let (a, bb) = (u().max(1e-300), u()); (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * bb).cos() });
                let ng = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-300);
                let rad = if edge && u() < 0.7 { 1.0 } else { u().cbrt() };
                [0, 1, 2].map(|a| h[a] * rad * g[a] / ng)
            }
        };
        let mut om = b.w_lo + (b.w_hi - b.w_lo) * u();
        if edge && u() < 0.5 {
            om = if u() < 0.5 { b.w_lo } else { b.w_hi };
        }
        let rr = b.r_max * u();
        let mono = [rho[2] * rho[2], rho[0] * rho[0] + rho[1] * rho[1], rho[0] * rho[2], rho[1] * rho[2]];
        let knot = knot_t.map(|t| {
            // an active knot of this readout
            let lo = t.iter().cloned().fold(f64::INFINITY, f64::min).floor() - 1.0;
            let hi = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max).floor() + 2.0;
            (lo + ((hi - lo + 1.0) * u()).floor()).min(hi)
        });
        let col: Vec<C64> = (0..n).map(|j| {
            let mut ph = -(k[j][0] * rho[0] + k[j][1] * rho[1] + k[j][2] * rho[2]) - om * (tau[j] - t0);
            if let Some(w) = wq {
                ph -= w[j][0] * mono[0] + w[j][1] * mono[1] + w[j][2] * mono[2] + w[j][3] * mono[3];
            }
            let mut a = (-rr * (tau[j] - t0)).exp();
            if let (Some(kb), Some(t)) = (knot, knot_t) {
                a *= bspline3(t[j] - kb);
            }
            C64::from_polar(a, ph)
        }).collect();
        if col.iter().map(|v| v.norm_sqr()).sum::<f64>() > 0.0 {
            cols.push(col);
        }
    }
    let np = cols.len();
    let (mut num, mut den, mut worst) = (0.0, 0.0, 0.0f64);
    let mut rv: Vec<Vec<C64>> = Vec::with_capacity(np);
    for c in &cols {
        let vs: Vec<C64> = d.sel.iter().map(|&j| c[j as usize]).collect();
        let e = d.interpolate(&vs);
        let (a, bb): (f64, f64) = c.iter().zip(&e).map(|(x, y)| ((x - y).norm_sqr(), x.norm_sqr())).fold((0.0, 0.0), |s, v| (s.0 + v.0, s.1 + v.1));
        num += a;
        den += bb;
        worst = worst.max((a / bb.max(1e-300)).sqrt());
        rv.push(d.apply_r(&vs));
    }
    // Gram difference D = Vᴴ V − (R V_S)ᴴ (R V_S) (np × np), spectral norms by power iteration
    let gram = |i: usize, j: usize, r: bool| -> C64 {
        if r {
            rv[i].iter().zip(&rv[j]).map(|(a, b)| a.conj() * b).sum()
        } else {
            cols[i].iter().zip(&cols[j]).map(|(a, b)| a.conj() * b).sum()
        }
    };
    let g0: Vec<C64> = (0..np * np).into_par_iter().map(|ij| gram(ij / np, ij % np, false)).collect();
    let g1: Vec<C64> = (0..np * np).into_par_iter().map(|ij| gram(ij / np, ij % np, true)).collect();
    let diff: Vec<C64> = g0.iter().zip(&g1).map(|(a, b)| a - b).collect();
    let pw = |m: &[C64]| -> f64 {
        let mut x: Vec<C64> = (0..np).map(|i| C64::new(1.0 + (i % 7) as f64 * 0.1, (i % 3) as f64 * 0.05)).collect();
        let mut lam = 0.0;
        for _ in 0..60 {
            let nx = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt().max(1e-300);
            x.iter_mut().for_each(|v| *v /= nx);
            let y: Vec<C64> = (0..np).map(|i| (0..np).map(|j| m[i * np + j] * x[j]).sum()).collect();
            lam = x.iter().zip(&y).map(|(a, b)| (a.conj() * b).re).sum::<f64>().abs();
            x = y;
        }
        lam
    };
    GateStats { frob: (num / den.max(1e-300)).sqrt(), worst, gram: pw(&diff) / pw(&g0).max(1e-300) }
}

/// centred cubic B-spline (support [−2, 2])
fn bspline3(x: f64) -> f64 {
    let a = x.abs();
    if a < 1.0 {
        (4.0 - 6.0 * a * a + 3.0 * a * a * a) / 6.0
    } else if a < 2.0 {
        (2.0 - a).powi(3) / 6.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------------------------------------------
// GPU: the banded R step and operator wrappers

/// The R / Rᴴ banded multiply (kernels: `rcompress.cu`).
pub struct RcKernel {
    f: CudaFunction,
    f_mask: CudaFunction,
}

/// threads per block of the banded multiply (one block per readout)
const RC_T: u32 = 128;

impl RcKernel {
    pub fn new(dev: &Device) -> Result<Self, Error> {
        let m = dev.load_module("recon_rcompress", include_str!("rcompress.cu"), &CompileSettings::default())?;
        Ok(Self { f: m.load_function("rc_apply")?, f_mask: m.load_function("rc_mask")? })
    }
}

/// The voxels inside a ball ROI smaller than the grid (`ball:OS:R_MM`): the unknowns outside are excluded from the
/// problem (x ↦ M x before the operator, M after the adjoint), so the compressed data term is exact for every x.
/// None when the ROI holds the whole grid.
pub fn roi_mask(roi: &Roi, dims: [usize; 3], vox: [f64; 3]) -> Option<Vec<f32>> {
    let pos = |q: usize| -> [f64; 3] {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [(i as f64 - (dims[0] / 2) as f64) * vox[0], (j as f64 - (dims[1] / 2) as f64) * vox[1], (k as f64 - (dims[2] / 2) as f64) * vox[2]]
    };
    let nv = dims[0] * dims[1] * dims[2];
    let r = match roi {
        Roi::Ball { r } => *r,
        Roi::Box { .. } => return None,
        Roi::Ellipsoid { h } => {
            return Some((0..nv).map(|q| { let p = pos(q); ((0..3).map(|a| (p[a] / h[a]).powi(2)).sum::<f64>() <= 1.0) as u8 as f32 }).collect());
        }
    };
    let full = match Roi::grid_ball(dims, vox) {
        Roi::Ball { r } => r,
        _ => unreachable!(),
    };
    if r >= full {
        return None;
    }
    Some((0..nv).map(|q| {
        let p = pos(q);
        ((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() <= r) as u8 as f32
    }).collect())
}

/// One set's layout on the device.
pub struct RcDevSet {
    ro_off: CudaSlice<u32>,
    tab_off: CudaSlice<u64>,
    nro: usize,
    m: usize,
}

/// The R tables of all designs on the device plus the per-set layouts.
pub struct RcDev {
    pub k: RcKernel,
    pub rtab: CudaSlice<C32>,
    pub sets: Vec<RcDevSet>,
    pub q: usize,
    /// voxel mask of a ball ROI smaller than the grid ([`roi_mask`]); None: all voxels
    pub mask: Option<CudaSlice<f32>>,
}

impl RcDev {
    pub fn new(dev: &Device, st: &Arc<CudaStream>, comp: &Compressor, sets: &[SetComp]) -> Result<Self, Error> {
        let q = comp.designs.iter().map(|d| d.q).max().unwrap_or(1).max(comp.opts.q.min(1));
        let mut offs = vec![0u64; comp.designs.len()];
        let mut all: Vec<C32> = Vec::with_capacity(comp.table_bytes() / 8);
        for (i, d) in comp.designs.iter().enumerate() {
            offs[i] = all.len() as u64;
            all.extend(d.r_f32(q));
        }
        if all.is_empty() {
            all.push(C32::new(0.0, 0.0));
        }
        let rtab = st.clone_htod(&all)?;
        let sets = sets.iter().map(|s| -> Result<RcDevSet, Error> {
            // (an empty set keeps one-element tables: no zero-size allocations)
            let mut to: Vec<u64> = s.design.iter().map(|&d| offs[d as usize]).collect();
            let ro = if s.ro_off.is_empty() { vec![0u32] } else { s.ro_off.clone() };
            if to.is_empty() {
                to.push(0);
            }
            Ok(RcDevSet { ro_off: st.clone_htod(&ro)?, tab_off: st.clone_htod(&to)?, nro: s.lens.len(), m: s.m() })
        }).collect::<Result<_, _>>()?;
        Ok(RcDev { k: RcKernel::new(dev)?, rtab, sets, q, mask: None })
    }

    /// attach the ROI's voxel mask (nv values; applied to every image of a stack)
    pub fn with_mask(mut self, st: &Arc<CudaStream>, mask: Option<&[f32]>) -> Result<Self, Error> {
        self.mask = mask.map(|m| st.clone_htod(m)).transpose()?;
        Ok(self)
    }

    /// out = M x (no-op copy semantics are the caller's: only called with a mask)
    fn mask_into(&self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, out: &mut CudaSlice<C32>) -> Result<(), Error> {
        let m = self.mask.as_ref().unwrap();
        let (n, nv) = (x.len() as i64, m.len() as i64);
        let cfg = LaunchConfig { grid_dim: ((n as u64).div_ceil(256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe { st.launch_builder(&self.k.f_mask).arg(x).arg(m).arg(out).arg(&n).arg(&nv).launch(cfg)? };
        Ok(())
    }

    /// x ← M x in place
    fn mask_inplace(&self, st: &Arc<CudaStream>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let m = self.mask.as_ref().unwrap();
        let (n, nv) = (x.len() as i64, m.len() as i64);
        let cfg = LaunchConfig { grid_dim: ((n as u64).div_ceil(256).max(1) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        // the kernel reads x[i] before writing out[i] (same index): in place is safe; one pointer passed twice
        let p = { use cudarc::driver::DevicePtrMut; x.device_ptr_mut(st).0 };
        unsafe { st.launch_builder(&self.k.f_mask).arg(&p).arg(m).arg(&p).arg(&n).arg(&nv).launch(cfg)? };
        Ok(())
    }

    /// the masked copy of x (or None without a mask)
    fn masked(&self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, scratch: &mut Option<CudaSlice<C32>>) -> Result<bool, Error> {
        if self.mask.is_none() {
            return Ok(false);
        }
        let mut s = match scratch.take() {
            Some(s) if s.len() == x.len() => s,
            _ => st.alloc_zeros::<C32>(x.len())?,
        };
        self.mask_into(st, x, &mut s)?;
        *scratch = Some(s);
        Ok(true)
    }

    /// y ← R y (adjoint: Rᴴ y) for every readout and coil of set t, in place (y: nc × m_S coil-major)
    pub fn apply(&self, st: &Arc<CudaStream>, t: usize, y: &mut CudaSlice<C32>, adjoint: bool) -> Result<(), Error> {
        let s = &self.sets[t];
        if s.nro == 0 || y.len() == 0 {
            return Ok(());
        }
        let nc = y.len() / s.m.max(1);
        assert_eq!(nc * s.m, y.len(), "rcompress: data length of set {t}");
        let (m_i, q_i, nc_i, adj) = (s.m as i64, self.q as i32, nc as i32, adjoint as i32);
        let cfg = LaunchConfig { grid_dim: (s.nro as u32, 1, 1), block_dim: (RC_T, 1, 1), shared_mem_bytes: (RC_T as usize + self.q) as u32 * 8 };
        unsafe { st.launch_builder(&self.k.f).arg(y).arg(&self.rtab).arg(&s.ro_off).arg(&s.tab_off).arg(&m_i).arg(&q_i).arg(&nc_i).arg(&adj).launch(cfg)? };
        Ok(())
    }

    pub fn len_y(&self, t: usize, nc: usize) -> usize {
        self.sets[t].m * nc
    }
}

/// A [`SetOp`] on the compressed samples: set t maps x to R A_S x (A_S = `inner` on the kept samples), the adjoint is
/// A_Sᴴ Rᴴ. Everything else (coil subsets, table placement, surrogates) passes through.
pub struct RcSets<'a> {
    pub inner: &'a mut dyn SetOp,
    pub rc: RcDev,
    scratch: Option<CudaSlice<C32>>,
    xs: Option<CudaSlice<C32>>,
}

impl<'a> RcSets<'a> {
    pub fn new(inner: &'a mut dyn SetOp, rc: RcDev) -> Self {
        RcSets { inner, rc, scratch: None, xs: None }
    }

    /// a copy of y with Rᴴ applied (scratch reused)
    fn rh(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>) -> Result<CudaSlice<C32>, Error> {
        let mut s = match self.scratch.take() {
            Some(s) if s.len() == y.len() => s,
            _ => st.alloc_zeros::<C32>(y.len())?,
        };
        st.memcpy_dtod(y, &mut s)?;
        self.rc.apply(st, t, &mut s, true)?;
        Ok(s)
    }
}

impl SetOp for RcSets<'_> {
    fn nsets(&self) -> usize {
        self.inner.nsets()
    }
    fn len_x(&self, t: usize) -> usize {
        self.inner.len_x(t)
    }
    fn len_y(&self, t: usize) -> usize {
        self.inner.len_y(t)
    }
    fn kernels(&self) -> &Kernels {
        self.inner.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        if self.rc.masked(st, x, &mut self.xs)? {
            self.inner.forward(st, t, self.xs.as_ref().unwrap(), y)?;
        } else {
            self.inner.forward(st, t, x, y)?;
        }
        self.rc.apply(st, t, y, false)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let s = self.rh(st, t, y)?;
        let r = self.inner.adjoint(st, t, &s, x);
        self.scratch = Some(s);
        if self.rc.mask.is_some() {
            self.rc.mask_inplace(st, x)?;
        }
        r
    }
    fn forward_multi(&mut self, st: &Arc<CudaStream>, t: usize, xs: &[&CudaSlice<C32>], ws: &[Option<&CudaSlice<C32>>], y: &mut CudaSlice<C32>) -> Result<(), Error> {
        assert!(self.rc.mask.is_none(), "rcompress: ROI mask with multi-image forwards not supported");
        self.inner.forward_multi(st, t, xs, ws, y)?;
        self.rc.apply(st, t, y, false)
    }
    fn adjoint_multi(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, ws: &[Option<&CudaSlice<C32>>], xs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert!(self.rc.mask.is_none(), "rcompress: ROI mask with multi-image adjoints not supported");
        let s = self.rh(st, t, y)?;
        let r = self.inner.adjoint_multi(st, t, &s, ws, xs);
        self.scratch = Some(s);
        r
    }
    fn forward_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        if self.rc.masked(st, x, &mut self.xs)? {
            self.inner.forward_group(st, members, self.xs.as_ref().unwrap(), ys)?;
        } else {
            self.inner.forward_group(st, members, x, ys)?;
        }
        for (&t, y) in members.iter().zip(ys.iter_mut()) {
            self.rc.apply(st, t, y, false)?;
        }
        Ok(())
    }
    fn forward_residual_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &[&CudaSlice<C32>], rs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        self.forward_group(st, members, x, rs)?;
        let k = self.inner.kernels();
        for (y, r) in ys.iter().zip(rs.iter_mut()) {
            k.axpby(st, C64::new(-1.0, 0.0), y, C64::new(1.0, 0.0), r)?;
        }
        Ok(())
    }
    fn adjoint_group(&mut self, st: &Arc<CudaStream>, members: &[usize], ys: &[&CudaSlice<C32>], x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let mut tmp: Vec<CudaSlice<C32>> = Vec::with_capacity(members.len());
        for (&t, y) in members.iter().zip(ys) {
            let mut s = st.alloc_zeros::<C32>(y.len())?;
            st.memcpy_dtod(*y, &mut s)?;
            self.rc.apply(st, t, &mut s, true)?;
            tmp.push(s);
        }
        let refs: Vec<&CudaSlice<C32>> = tmp.iter().collect();
        self.inner.adjoint_group(st, members, &refs, x)?;
        if self.rc.mask.is_some() {
            self.rc.mask_inplace(st, x)?;
        }
        Ok(())
    }
    fn table_bytes(&self) -> (usize, usize) {
        self.inner.table_bytes()
    }
    fn tables_on_host(&self) -> bool {
        self.inner.tables_on_host()
    }
    fn place_tables(&mut self, ctx: &Arc<cudarc::driver::CudaContext>, st: &Arc<CudaStream>, on_host: bool) -> Result<i64, Error> {
        self.inner.place_tables(ctx, st, on_host)
    }
    fn select_coils(&mut self, st: &Arc<CudaStream>, idx: Option<&[usize]>) -> Result<bool, Error> {
        self.inner.select_coils(st, idx)
    }
    fn use_surrogate(&mut self, on: bool) -> bool {
        self.inner.use_surrogate(on)
    }
    fn n_coils(&self) -> usize {
        self.inner.n_coils()
    }
}

/// A [`LinOp`] on the compressed samples of one set (`t` in `rc`): R A_S and A_Sᴴ Rᴴ.
pub struct RcLin<'a, L: LinOp> {
    pub inner: L,
    pub rc: &'a RcDev,
    pub t: usize,
    scratch: Option<CudaSlice<C32>>,
    xs: Option<CudaSlice<C32>>,
}

impl<'a, L: LinOp> RcLin<'a, L> {
    pub fn new(inner: L, rc: &'a RcDev, t: usize) -> Self {
        RcLin { inner, rc, t, scratch: None, xs: None }
    }
}

impl<L: LinOp> LinOp for RcLin<'_, L> {
    fn len_x(&self) -> usize {
        self.inner.len_x()
    }
    fn len_y(&self) -> usize {
        self.inner.len_y()
    }
    fn kernels(&self) -> &Kernels {
        self.inner.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        if self.rc.masked(st, x, &mut self.xs)? {
            self.inner.forward(st, self.xs.as_ref().unwrap(), y)?;
        } else {
            self.inner.forward(st, x, y)?;
        }
        self.rc.apply(st, self.t, y, false)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let mut s = match self.scratch.take() {
            Some(s) if s.len() == y.len() => s,
            _ => st.alloc_zeros::<C32>(y.len())?,
        };
        st.memcpy_dtod(y, &mut s)?;
        self.rc.apply(st, self.t, &mut s, true)?;
        let r = self.inner.adjoint(st, &s, x);
        self.scratch = Some(s);
        if self.rc.mask.is_some() {
            self.rc.mask_inplace(st, x)?;
        }
        r
    }
}

/// A `&mut dyn LinOp` as a [`LinOp`] value (for the generic solvers).
pub struct DynLin<'a>(pub &'a mut dyn LinOp);

impl LinOp for DynLin<'_> {
    fn len_x(&self) -> usize {
        self.0.len_x()
    }
    fn len_y(&self) -> usize {
        self.0.len_y()
    }
    fn kernels(&self) -> &Kernels {
        self.0.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.0.forward(st, x, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.0.adjoint(st, y, x)
    }
}

/// Rows of length m (SoA tables, coil-major data) restricted to the kept samples (empty stays empty).
pub fn select_rows<T: Copy + Send + Sync>(v: &[T], m: usize, sel: &[u32]) -> Vec<T> {
    if v.is_empty() {
        vec![]
    } else {
        crate::pyramid::gather_rows(v, m, sel)
    }
}

/// [`gate`] on `n` evenly spread readouts of a set (concomitant coefficients when the set has them), the worst of
/// each statistic.
pub fn gate_set(set: &SampleSet, lens: &[usize], vox: [f64; 3], o: &RcOpts, n: usize) -> GateStats {
    let offs: Vec<usize> = lens.iter().scan(0, |a, &l| { let r = *a; *a += l; Some(r) }).collect();
    let mut g = GateStats::default();
    for i in 0..n.min(lens.len()) {
        let r = i * lens.len() / n.max(1);
        let (a, l) = (offs[r], lens[r]);
        if l < 2 {
            continue;
        }
        let k: Vec<[f64; 3]> = (a..a + l).map(|j| [0, 1, 2].map(|x| set.coords[x][j] as f64 / vox[x])).collect();
        let d = Design::build(&k, &set.tau[a..a + l], o);
        let wq = (set.wq.len() == set.tau.len()).then(|| &set.wq[a..a + l]);
        g = g.max(gate(&d, &k, &set.tau[a..a + l], wq, None, o, 160, 17 + r as u64));
    }
    g
}

/// The gate's acceptance: normal-operator error on the sources ≤ 1e-4 (the data-term target)
pub const GATE_TOL: f64 = 1e-4;

/// The `auto` decision and the design options for a grid: ROI from the prescription (grid box per readout, or the
/// enclosing ball with one shared design), reductions estimated from the selection rule on `set` (whole readouts of
/// `lens` samples). Returns the options to use (None = off) and a report line.
pub fn decide(mode: &RcMode, dims: [usize; 3], vox: [f64; 3], band: Band, set: &SampleSet, lens: &[usize]) -> (Option<RcOpts>, String) {
    match mode {
        RcMode::Off => (None, "off".into()),
        RcMode::Auto { os, min_gain } => {
            let cands: Vec<RcOpts> = [RcMode::Ball { os: *os, r: None }, RcMode::Box { os: *os }].iter().filter_map(|m| m.opts(dims, vox, band)).collect();
            let est: Vec<f64> = cands.iter().map(|o| estimate_reduction(set, lens, vox, o, 256)).collect();
            let (i, best) = est.iter().cloned().enumerate().fold((0, 0.0), |a, (i, v)| if v > a.1 { (i, v) } else { a });
            let line = format!("auto: estimated reduction {:.3}× ({}) / {:.3}× ({}); threshold {min_gain}×", est[0], cands[0].roi.describe(), est[1], cands[1].roi.describe());
            if best < *min_gain {
                return (None, format!("{line} → off"));
            }
            // the gate decides the oversampling: os raised until the normal-operator error on the sources ≤ 1e-4
            let mut o = cands[i].clone();
            loop {
                let g = gate_set(set, lens, vox, &o, 3);
                let red = estimate_reduction(set, lens, vox, &o, 256);
                if g.gram <= GATE_TOL {
                    if red < *min_gain {
                        return (None, format!("{line}; gate at os {} ({:.1e}) leaves {red:.3}× → off", o.os, g.gram));
                    }
                    return (Some(o.clone()), format!("{line} → {}, os {} ({red:.3}×, gate {:.1e})", o.roi.describe(), o.os, g.gram));
                }
                if o.os >= 2.0 || red < *min_gain {
                    return (None, format!("{line}; gate {:.1e} > {GATE_TOL:.0e} at os {} → off", g.gram, o.os));
                }
                o.os = (o.os + 0.15).min(2.0);
            }
        }
        m => {
            let o = m.opts(dims, vox, band).unwrap();
            let est = estimate_reduction(set, lens, vox, &o, 256);
            let line = format!("{}: estimated reduction {est:.3}×", o.roi.describe());
            (Some(o), line)
        }
    }
}

/// `--rcompress` modes.
#[derive(Clone, Debug, PartialEq)]
pub enum RcMode {
    Off,
    /// ball (shared design) when its measured reduction reaches `min_gain`, else off
    Auto { os: Option<f64>, min_gain: f64 },
    Ball { os: Option<f64>, r: Option<f64> },
    Box { os: Option<f64> },
    /// the ellipsoid inscribed in the grid box (per readout; unknowns outside masked)
    Ellipsoid { os: Option<f64> },
}

impl RcMode {
    /// `off | auto[:OS] | ball[:OS[:R_MM]] | box[:OS] | ellipsoid[:OS]`
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut it = s.split(':');
        let kind = it.next().unwrap_or("");
        let num = |v: Option<&str>, what: &str| -> Result<Option<f64>, String> { v.filter(|x| !x.is_empty()).map(|x| x.parse::<f64>().map_err(|_| format!("--rcompress {s}: {what} {x}"))).transpose() };
        let os = num(it.next(), "OS")?;
        match kind {
            "off" => Ok(RcMode::Off),
            "auto" => Ok(RcMode::Auto { os, min_gain: 1.2 }),
            "ball" => Ok(RcMode::Ball { os, r: num(it.next(), "R_MM")?.map(|v| v * 1e-3) }),
            "box" => Ok(RcMode::Box { os }),
            "ellipsoid" => Ok(RcMode::Ellipsoid { os }),
            o => Err(format!("--rcompress {o}: off | auto[:OS] | ball[:OS[:R_MM]] | box[:OS] | ellipsoid[:OS]")),
        }
    }
    pub fn is_off(&self) -> bool {
        *self == RcMode::Off
    }
    /// the design options of the mode on a grid (ROI from the prescription)
    pub fn opts(&self, dims: [usize; 3], vox: [f64; 3], band: Band) -> Option<RcOpts> {
        let (roi, os) = match self {
            RcMode::Off => return None,
            RcMode::Auto { os, .. } => (Roi::grid_ball(dims, vox), *os),
            RcMode::Ball { os, r } => (r.map_or(Roi::grid_ball(dims, vox), |r| Roi::Ball { r }), *os),
            RcMode::Box { os } => (Roi::grid_box(dims, vox), *os),
            RcMode::Ellipsoid { os } => (match Roi::grid_box(dims, vox) { Roi::Box { h } => Roi::Ellipsoid { h }, _ => unreachable!() }, *os),
        };
        let mut o = RcOpts::new(roi, band);
        if let Some(v) = os {
            o.os = v;
        }
        Some(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a synthetic yarn-like readout: constant |k̇| on a winding curve out to k_max and back, dwell 2 µs
    fn readout(n: usize, kmax: f64) -> (Vec<[f64; 3]>, Vec<f64>) {
        let tau: Vec<f64> = (0..n).map(|j| 1.6e-3 + 2e-6 * j as f64).collect();
        let k = (0..n).map(|j| {
            let s = j as f64 / (n - 1) as f64;
            let r = kmax * (std::f64::consts::PI * s).sin();
            let (th, ph) = (2.0 * std::f64::consts::PI * 3.0 * s, 1.0 + 2.5 * s);
            [r * ph.sin() * th.cos(), r * ph.sin() * th.sin(), r * ph.cos()]
        }).collect();
        (k, tau)
    }

    #[test]
    fn kept_rows_exact_and_band_cholesky() {
        let (k, tau) = readout(1200, 900.0);
        let o = RcOpts::new(Roi::Ball { r: 0.06 }, Band::default_wide());
        let d = Design::build(&k, &tau, &o);
        for (p, &j) in d.sel.iter().enumerate() {
            let st = d.start[j as usize] as usize;
            for q1 in 0..d.q {
                let c = d.coef[j as usize * d.q + q1];
                let want = if st + q1 == p { 1.0 } else { 0.0 };
                assert!((c - C64::new(want, 0.0)).norm() < 1e-14);
            }
        }
        // Rᴴ R = Eᴴ E on random vectors
        let ns = d.ns();
        let v: Vec<C64> = (0..ns).map(|i| C64::new(((i * 37) % 11) as f64 - 5.0, ((i * 13) % 7) as f64 - 3.0)).collect();
        let ev = d.interpolate(&v);
        let rv = d.apply_r(&v);
        let (a, b): (f64, f64) = (ev.iter().map(|z| z.norm_sqr()).sum(), rv.iter().map(|z| z.norm_sqr()).sum());
        assert!((a - b).abs() / a < 1e-9, "‖E v‖² {a} vs ‖R v‖² {b}");
        // objective identity ‖E v − y‖² = ‖R v − ỹ‖² + ‖y‖² − ‖ỹ‖²
        let y: Vec<C64> = (0..d.n).map(|j| C64::new((j as f64 * 0.37).sin(), (j as f64 * 0.11).cos())).collect();
        let mut yt = vec![C64::new(0.0, 0.0); ns];
        let (yy, tt) = d.compress(&y, &mut yt);
        let f0: f64 = ev.iter().zip(&y).map(|(a, b)| (a - b).norm_sqr()).sum();
        let f1: f64 = rv.iter().zip(&yt).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() + yy - tt;
        assert!((f0 - f1).abs() / f0 < 1e-9, "objective {f0} vs {f1}");
    }

    /// The direct-sum gate of the study (dense_check.py form) on a synthetic readout: normal operator, Aᴴy and the
    /// objective of a voxel-grid operator inside the ROI with ratemap rates, concomitant phase and cardiac knots,
    /// compressed against uncompressed, ≤ 1e-4.
    #[test]
    fn direct_sum_gate() {
        // a readout oversampled ≈ 2.6× for a ball of r 40 mm: n · dwell · |k̇| ≈ 2π n / FOV_readout
        let n = 1600;
        let (k, tau) = readout(n, 1100.0);
        let roi = Roi::Ball { r: 0.04 };
        let band = Band { w_lo: -2.0 * std::f64::consts::PI * 600.0, w_hi: 2.0 * std::f64::consts::PI * 400.0, r_max: 400.0 };
        let o = RcOpts { os: 1.5, ..RcOpts::new(roi, band) };
        let d = Design::build(&k, &tau, &o);
        let red = n as f64 / d.ns() as f64;
        assert!(red > 1.3, "reduction {red}");
        let wq: Vec<[f64; 4]> = (0..n).map(|j| { let s = j as f64 / n as f64; [20.0 + 60.0 * s, 2.0 + 5.0 * s, -10.0 * s, -30.0 * s] }).collect();
        let kt: Vec<f64> = (0..n).map(|j| 3.2 + 0.15 * j as f64 / n as f64).collect();
        let g = gate(&d, &k, &tau, Some(&wq), Some(&kt), &o, 200, 7);
        assert!(g.gram < 1e-4 && g.worst < 3e-4, "gate {g:?} (reduction {red:.2})");
        // dense operator on a 7³ grid inside the ball, 4 knots, voxel rates
        let gpts = 7;
        let mut pos = vec![];
        for i in 0..gpts {
            for j in 0..gpts {
                for l in 0..gpts {
                    let p = [i, j, l].map(|v| (v as f64 / (gpts - 1) as f64 - 0.5) * 2.0 * 0.04 / 3f64.sqrt());
                    pos.push(p);
                }
            }
        }
        let nvx = pos.len();
        let z: Vec<C64> = (0..nvx).map(|v| C64::new(30.0 + 10.0 * (v % 5) as f64, 2.0 * std::f64::consts::PI * (-500.0 + 900.0 * ((v * 7) % 13) as f64 / 13.0))).collect();
        let nk = 4;
        let a = |j: usize, b: usize, v: usize| -> C64 {
            let p = pos[v];
            let mono = [p[2] * p[2], p[0] * p[0] + p[1] * p[1], p[0] * p[2], p[1] * p[2]];
            let ph = -(k[j][0] * p[0] + k[j][1] * p[1] + k[j][2] * p[2]) - (0..4).map(|c| wq[j][c] * mono[c]).sum::<f64>();
            bspline3(kt[j] - (2.0 + b as f64)) * C64::from_polar(1.0, ph) * (-z[v] * (tau[j] - tau[0])).exp()
        };
        let nx = nk * nvx;
        let amat: Vec<C64> = (0..n * nx).into_par_iter().map(|ij| { let (j, x) = (ij / nx, ij % nx); a(j, x / nvx, x % nvx) }).collect();
        let fwd = |x: &[C64]| -> Vec<C64> { (0..n).map(|j| (0..nx).map(|c| amat[j * nx + c] * x[c]).sum()).collect() };
        let adj = |y: &[C64]| -> Vec<C64> { (0..nx).map(|c| (0..n).map(|j| amat[j * nx + c].conj() * y[j]).sum()).collect() };
        let sel: Vec<usize> = d.sel.iter().map(|&s| s as usize).collect();
        let fwd_c = |x: &[C64]| -> Vec<C64> { let f = fwd(x); let vs: Vec<C64> = sel.iter().map(|&j| f[j]).collect(); d.apply_r(&vs) };
        let adj_c = |r: &[C64]| -> Vec<C64> {
            // Rᴴ r scattered onto the kept samples, then Aᴴ
            let (q, ns) = (d.q, d.ns());
            let mut full = vec![C64::new(0.0, 0.0); n];
            for i in 0..ns {
                let mut s = C64::new(0.0, 0.0);
                for dd in 0..q.min(i + 1) {
                    s += d.r[(i - dd) * q + dd].conj() * r[i - dd];
                }
                full[sel[i]] = s;
            }
            adj(&full)
        };
        let x: Vec<C64> = (0..nx).map(|i| C64::new(((i * 31) % 17) as f64 / 17.0 - 0.5, ((i * 7) % 5) as f64 / 5.0 - 0.4)).collect();
        let n0 = adj(&fwd(&x));
        let n1 = adj_c(&fwd_c(&x));
        let rel = |a: &[C64], b: &[C64]| (a.iter().zip(b).map(|(u, v)| (u - v).norm_sqr()).sum::<f64>() / a.iter().map(|u| u.norm_sqr()).sum::<f64>()).sqrt();
        let en = rel(&n0, &n1);
        // data: y = A x_true + noise, Aᴴ y and the objective
        let xt: Vec<C64> = x.iter().map(|v| v * 1.3 + C64::new(0.1, 0.0)).collect();
        let mut yv = fwd(&xt);
        let ns_ = 0.02 * (yv.iter().map(|v| v.norm_sqr()).sum::<f64>() / n as f64).sqrt();
        for (j, v) in yv.iter_mut().enumerate() {
            *v += ns_ * C64::new((j as f64 * 12.9898).sin() * 43758.5453 % 1.0, (j as f64 * 78.233).sin() * 12345.678 % 1.0);
        }
        let mut yt = vec![C64::new(0.0, 0.0); d.ns()];
        let (yy, tt) = d.compress(&yv, &mut yt);
        let g0: Vec<C64> = { let f = fwd(&x); adj(&f.iter().zip(&yv).map(|(a, b)| a - b).collect::<Vec<_>>()) };
        let g1: Vec<C64> = { let f = fwd_c(&x); adj_c(&f.iter().zip(&yt).map(|(a, b)| a - b).collect::<Vec<_>>()) };
        let eg = rel(&g0, &g1);
        let f0: f64 = fwd(&x).iter().zip(&yv).map(|(a, b)| (a - b).norm_sqr()).sum();
        let f1: f64 = fwd_c(&x).iter().zip(&yt).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>() + yy - tt;
        let ef = (f0 - f1).abs() / f0;
        eprintln!("direct-sum gate: reduction {red:.2}, normal {en:.2e}, gradient {eg:.2e}, objective {ef:.2e}, gate {g:?}");
        assert!(en < 1e-4 && eg < 1e-4 && ef < 1e-4, "normal {en:.2e}, gradient {eg:.2e}, objective {ef:.2e} (gate {g:?}, reduction {red:.2})");
    }

    #[test]
    fn rotation_shares_ball_design() {
        let (k, tau) = readout(800, 700.0);
        let th = 0.7f64;
        let rot = |p: &[f64; 3]| [th.cos() * p[0] - th.sin() * p[1], th.sin() * p[0] + th.cos() * p[1], p[2]];
        let kr: Vec<[f64; 3]> = k.iter().map(rot).collect();
        let mut c = Compressor::new(RcOpts::new(Roi::Ball { r: 0.05 }, Band::default_wide()));
        let a = c.design_for(&k, &tau);
        let b = c.design_for(&kr, &tau);
        assert_eq!(a, b);
        assert_eq!(c.designs.len(), 1);
        let mut cb = Compressor::new(RcOpts::new(Roi::Box { h: [0.05; 3] }, Band::default_wide()));
        assert_ne!(cb.design_for(&k, &tau), cb.design_for(&kr, &tau));
    }

    #[test]
    fn parse_modes() {
        assert_eq!(RcMode::parse("off").unwrap(), RcMode::Off);
        assert_eq!(RcMode::parse("box:1.5").unwrap(), RcMode::Box { os: Some(1.5) });
        assert_eq!(RcMode::parse("ball::80").unwrap(), RcMode::Ball { os: None, r: Some(0.08) });
        assert!(RcMode::parse("cube").is_err());
    }
}
