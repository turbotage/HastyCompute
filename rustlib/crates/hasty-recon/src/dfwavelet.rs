//! Divergence-free wavelet shrinkage of velocity fields (Deriaz & Perrier, "Divergence-free and curl-free wavelets in
//! 2D and 3D, application to turbulent flows", J. Turbulence 2006; Ong et al., "Robust 4D flow denoising using divergence-free wavelet
//! transform", MRM 2015). The filters and the linear combinations follow BART's dfwavelet (v0.4.04, src/dfwavelet,
//! Frank Ong, BSD licence); the transform here is periodic (exact reconstruction for every even length, no
//! redundancy) on the shared separable engine ([`crate::sepwt`]).
//!
//! - Component v_i is transformed with the CDF 3.1 bank along axis i and the CDF 2.2 bank along the other two axes
//!   (the 1D pairs satisfy φ1' = Δφ0, ψ1' = −4ψ0), each coefficient divided by the voxel size of axis i.
//! - Per level and detail subband ε, the three component coefficients (c_x, c_y, c_z) are recombined into two
//!   divergence-free coefficients (df1, df2) and one non-divergence-free coefficient n ∝ the subband's divergence
//!   d = Σ_{a∈H} (−4) c_a + Σ_{a∉H} Δ_a c_a, with H = {a : ε_a = 1} and Δ_a the backward difference along a inside the
//!   subband:
//!     |H| = 1 (pivot p, others q1 < q2): df = (c_q1, c_q2),               n = c_p − ¼ (Δ_q1 c_q1 + Δ_q2 c_q2)
//!     |H| = 2 (p1 < p2, other q):        df = ((c_p1 − c_p2)/2, c_q),     n = (c_p1 + c_p2)/2 − ⅛ Δ_q c_q
//!     |H| = 3:                           df = ((−2c_x + c_y + c_z)/3, (−c_x + 2c_y − c_z)/3), n = (c_x + c_y + c_z)/3
//!   The approximation band keeps the component coefficients. (BART's tables have +¼ / +⅛; for these filters the
//!   wavelet derivative relation is ψ1' = −4ψ0, and the test `divergence_free_field_has_no_n_coefficients` confirms
//!   the minus sign: a discrete curl field gives n / df ≈ 1e-7 at every level, the plus sign ≈ 0.5.)
//! - Shrinkage: soft thresholds per level × subband × output, in units of that output's noise amplitude (the standard
//!   deviation of the transform of unit white noise per component, estimated once per padded size). Velocity noise is
//!   not uniform (σ_v ∝ 1 / |m|), so with a magnitude weight w the shrinkage acts on u = c · w̄ / (s₀ · amp), w̄ the
//!   block mean of w over the coefficient's support, s₀ the MAD noise level of u in the finest diagonal subband over
//!   the signal-bearing coefficients: unit noise everywhere. Threshold rule: fixed (λ = t) or SureShrink per subband
//!   (Donoho & Johnstone 1995, hybrid with the universal threshold for sparse subbands), separately for df and n.
//!   Scaling v leaves u's distribution unchanged up to the same factor: the map is scale-equivariant.
//! - Cycle spinning: a random circular shift per iteration (deterministic in the iteration).

use rayon::prelude::*;

use crate::sepwt::{forward, inverse, pad_in, pad_out, padded, Bank};
use crate::wavelet::cycle_shift;

const R2: f64 = std::f64::consts::SQRT_2;

/// (filters 0: CDF 2.2, used along the axes other than the component's; filters 1: CDF 3.1, the component's axis)
pub fn cdf_banks() -> (Bank, Bank) {
    let s = |v: &[f64]| v.iter().map(|c| c * R2).collect::<Vec<f64>>();
    // BART's tables (6 taps, ×√2 normalisation): lod/hid analysis, lor/hir synthesis
    let lod0 = s(&[0.0, -0.125, 0.25, 0.75, 0.25, -0.125]);
    let hid0 = s(&[0.0, 0.25, -0.5, 0.25, 0.0, 0.0]);
    let lor0 = s(&[0.0, 0.25, 0.5, 0.25, 0.0, 0.0]);
    let hir0 = s(&[0.0, 0.125, 0.25, -0.75, 0.25, 0.125]);
    let lod1 = s(&[0.0, -0.25, 0.75, 0.75, -0.25, 0.0]);
    let hid1 = s(&[0.0, -0.125, 0.375, -0.375, 0.125, 0.0]);
    let lor1 = s(&[0.0, 0.125, 0.375, 0.375, 0.125, 0.0]);
    let hir1 = s(&[0.0, -0.25, -0.75, 0.75, 0.25, 0.0]);
    (Bank::bart(&lod0, &hid0, &lor0, &hir0), Bank::bart(&lod1, &hid1, &lor1, &hir1))
}

#[derive(Clone, Copy, Debug)]
pub enum DfRule {
    /// λ = t (in units of the coefficient noise) for the df and the n coefficients
    Fixed { tdf: f64, tn: f64 },
    /// λ = t · λ_SURE per subband
    Sure { tdf: f64, tn: f64 },
}

#[derive(Clone, Debug, Default)]
pub struct DfStats {
    /// noise level s₀ of the weighted coefficients
    pub s0: f64,
    /// mean threshold of the df and the n coefficients (units of their noise)
    pub lam_df: f64,
    pub lam_n: f64,
    /// fraction of detail coefficients kept (≠ 0 after shrinkage), df and n
    pub kept_df: f64,
    pub kept_n: f64,
}

pub struct DfWavelet {
    pub dims: [usize; 3],
    pub levels: usize,
    /// voxel size per axis (any unit; only ratios matter)
    pub vox: [f64; 3],
    pub rule: DfRule,
    shift: [usize; 3],
    b0: Bank,
    b1: Bank,
    /// noise amplitude per (level − 1) · 7 + (ε − 1), per output (df1, df2, n)
    amp: std::sync::OnceLock<Vec<[f32; 3]>>,
}

#[derive(Clone, Copy)]
struct Shared(*mut f32);
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// subband geometry: ε as bits (x = 1, y = 2, z = 4), offset and size in the padded box at `level` (1 = finest)
fn band(p: [usize; 3], level: usize, e: usize) -> ([usize; 3], [usize; 3], [usize; 3]) {
    let s = p.map(|d| d >> level);
    let eps = [e & 1, (e >> 1) & 1, (e >> 2) & 1];
    (eps, [0, 1, 2].map(|a| eps[a] * s[a]), s)
}

impl DfWavelet {
    pub fn new(dims: [usize; 3], levels: usize, vox: [f64; 3], rule: DfRule) -> Self {
        let (b0, b1) = cdf_banks();
        Self { dims, levels: levels.max(1), vox, rule, shift: [0; 3], b0, b1, amp: std::sync::OnceLock::new() }
    }

    pub fn set_iteration(&mut self, k: u64) {
        self.shift = cycle_shift(k, 0x4446_5741_5645, self.levels);
    }

    pub fn padded(&self) -> [usize; 3] {
        padded(self.dims, self.levels)
    }

    fn banks(&self, i: usize) -> [&Bank; 3] {
        [0, 1, 2].map(|a| if a == i { &self.b1 } else { &self.b0 })
    }

    /// forward DFW of three padded component buffers (in place): component transforms, 1/vox scaling, recombination
    pub fn forward(&self, bufs: &mut [Vec<f32>; 3]) {
        let p = self.padded();
        for (i, b) in bufs.iter_mut().enumerate() {
            forward(b, p, 1, self.levels, self.banks(i));
            let s = (1.0 / self.vox[i]) as f32;
            b.par_iter_mut().for_each(|v| *v *= s);
        }
        for l in 1..=self.levels {
            self.lincomb(bufs, p, l, false);
        }
    }

    /// inverse of [`DfWavelet::forward`]
    pub fn inverse(&self, bufs: &mut [Vec<f32>; 3]) {
        let p = self.padded();
        for l in 1..=self.levels {
            self.lincomb(bufs, p, l, true);
        }
        for (i, b) in bufs.iter_mut().enumerate() {
            let s = self.vox[i] as f32;
            b.par_iter_mut().for_each(|v| *v *= s);
            inverse(b, p, 1, self.levels, self.banks(i));
        }
    }

    /// (c_x, c_y, c_z) ↔ (df1, df2, n) in every detail subband of `level`
    fn lincomb(&self, bufs: &mut [Vec<f32>; 3], p: [usize; 3], level: usize, inv: bool) {
        let mut t: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        for e in 1..8usize {
            let (eps, off, s) = band(p, level, e);
            let n = s[0] * s[1] * s[2];
            // copy of the subband (the differences read neighbours of the unmodified values)
            for i in 0..3 {
                t[i].resize(n, 0.0);
                let (src, dst) = (&bufs[i], &mut t[i]);
                dst.par_chunks_mut(s[0]).enumerate().for_each(|(row, d)| {
                    let (y, z) = (row % s[1], row / s[1]);
                    let q = off[0] + p[0] * ((off[1] + y) + p[1] * (off[2] + z));
                    d.copy_from_slice(&src[q..q + s[0]]);
                });
            }
            let h: Vec<usize> = (0..3).filter(|&a| eps[a] == 1).collect();
            let l: Vec<usize> = (0..3).filter(|&a| eps[a] == 0).collect();
            let ptr = [Shared(bufs[0].as_mut_ptr()), Shared(bufs[1].as_mut_ptr()), Shared(bufs[2].as_mut_ptr())];
            let t = &t;
            (0..s[1] * s[2]).into_par_iter().for_each(|row| {
                let ptr = ptr;
                let (y, z) = (row % s[1], row / s[1]);
                let q0 = off[0] + p[0] * ((off[1] + y) + p[1] * (off[2] + z));
                let r0 = s[0] * row;
                // backward neighbour index (inside the subband, periodic) of local voxel (x, y, z) along axis a
                let nb = |x: usize, a: usize| -> usize {
                    match a {
                        0 => r0 + (x + s[0] - 1) % s[0],
                        1 => s[0] * ((y + s[1] - 1) % s[1] + s[1] * z) + x,
                        _ => s[0] * (y + s[1] * ((z + s[2] - 1) % s[2])) + x,
                    }
                };
                for x in 0..s[0] {
                    let k = r0 + x;
                    let d = |i: usize, a: usize| t[i][k] - t[i][nb(x, a)];
                    let out: [f32; 3] = match (h.len(), inv) {
                        (1, false) => {
                            let (pv, q1, q2) = (h[0], l[0], l[1]);
                            [t[q1][k], t[q2][k], t[pv][k] - 0.25 * (d(q1, q1) + d(q2, q2))]
                        }
                        (1, true) => {
                            // t = (df1, df2, n): c_q1 = df1, c_q2 = df2, c_p = n + ¼ (Δ df1 + Δ df2)
                            let (pv, q1, q2) = (h[0], l[0], l[1]);
                            let mut c = [0f32; 3];
                            c[q1] = t[0][k];
                            c[q2] = t[1][k];
                            c[pv] = t[2][k] + 0.25 * (d(0, q1) + d(1, q2));
                            c
                        }
                        (2, false) => {
                            let (p1, p2, q) = (h[0], h[1], l[0]);
                            [0.5 * (t[p1][k] - t[p2][k]), t[q][k], 0.5 * (t[p1][k] + t[p2][k]) - 0.125 * d(q, q)]
                        }
                        (2, true) => {
                            let (p1, p2, q) = (h[0], h[1], l[0]);
                            let m = t[2][k] + 0.125 * d(1, q);
                            let mut c = [0f32; 3];
                            c[q] = t[1][k];
                            c[p1] = t[0][k] + m;
                            c[p2] = m - t[0][k];
                            c
                        }
                        (_, false) => {
                            let (cx, cy, cz) = (t[0][k], t[1][k], t[2][k]);
                            [(-2.0 * cx + cy + cz) / 3.0, (-cx + 2.0 * cy - cz) / 3.0, (cx + cy + cz) / 3.0]
                        }
                        (_, true) => {
                            let (d1, d2, nn) = (t[0][k], t[1][k], t[2][k]);
                            [nn - d1, nn + d2, nn + d1 - d2]
                        }
                    };
                    for i in 0..3 {
                        // SAFETY: each task writes its own row of the subband in each buffer
                        unsafe { *ptr[i].0.add(q0 + x) = out[i] };
                    }
                }
            });
        }
    }

    /// noise amplitude per level × subband × output: std of the transform of unit white noise per component
    fn amps(&self) -> &Vec<[f32; 3]> {
        self.amp.get_or_init(|| {
            let p = self.padded();
            let np = p[0] * p[1] * p[2];
            let mut s = 0x9e37_79b9_7f4a_7c15u64;
            let mut g = || {
                // Box–Muller from xorshift
                let mut u = || {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
                };
                ((-2.0 * u().ln()).sqrt() * (std::f64::consts::TAU * u()).cos()) as f32
            };
            let mut bufs: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..np).map(|_| g()).collect());
            self.forward(&mut bufs);
            let mut out = vec![[0f32; 3]; self.levels * 7];
            for l in 1..=self.levels {
                for e in 1..8 {
                    let (_, off, sz) = band(p, l, e);
                    for o in 0..3 {
                        let (mut s2, mut n) = (0f64, 0usize);
                        for z in 0..sz[2] {
                            for y in 0..sz[1] {
                                let q = off[0] + p[0] * ((off[1] + y) + p[1] * (off[2] + z));
                                for v in &bufs[o][q..q + sz[0]] {
                                    s2 += (*v as f64).powi(2);
                                    n += 1;
                                }
                            }
                        }
                        out[(l - 1) * 7 + e - 1][o] = (s2 / n.max(1) as f64).sqrt() as f32;
                    }
                }
            }
            out
        })
    }

    /// shrink a velocity field v (3 components × nv, component-major) in place; `weight` (nv): the magnitude the
    /// velocity noise scales inversely with (None: uniform noise)
    pub fn apply(&self, v: &mut [f32], weight: Option<&[f32]>) -> DfStats {
        let d = self.dims;
        let nv = d[0] * d[1] * d[2];
        assert_eq!(v.len(), 3 * nv);
        let p = self.padded();
        let amp = self.amps().clone();
        let mut bufs: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        for i in 0..3 {
            pad_in(&v[i * nv..(i + 1) * nv], d, p, 1, self.shift, &mut bufs[i]);
        }
        self.forward(&mut bufs);
        // weight pyramid: level-l block means of the padded, shifted weight
        let mut wp: Vec<Vec<f32>> = vec![];
        if let Some(w) = weight {
            assert_eq!(w.len(), nv);
            let mut w0 = vec![];
            pad_in(w, d, p, 1, self.shift, &mut w0);
            let mut cur = w0;
            let mut cs = p;
            for _ in 1..=self.levels {
                let ns = cs.map(|v| v / 2);
                let mut nx = vec![0f32; ns[0] * ns[1] * ns[2]];
                nx.par_chunks_mut(ns[0]).enumerate().for_each(|(row, o)| {
                    let (y, z) = (row % ns[1], row / ns[1]);
                    for (x, ov) in o.iter_mut().enumerate() {
                        let mut a = 0f32;
                        for dz in 0..2 {
                            for dy in 0..2 {
                                let b = 2 * x + cs[0] * ((2 * y + dy) + cs[1] * (2 * z + dz));
                                a += cur[b] + cur[b + 1];
                            }
                        }
                        *ov = 0.125 * a;
                    }
                });
                wp.push(nx);
                cur = wp.last().unwrap().clone();
                cs = ns;
            }
        }
        let wat = |l: usize, pos: [usize; 3]| -> f32 {
            if wp.is_empty() {
                1.0
            } else {
                let s = p.map(|v| v >> l);
                wp[l - 1][pos[0] + s[0] * (pos[1] + s[1] * pos[2])]
            }
        };
        // signal-bearing coefficients: weight ≥ 0.2 × the 95th percentile of the finest weight level
        let wmin = if wp.is_empty() {
            0.0
        } else {
            let mut s: Vec<f32> = wp[0].iter().step_by(wp[0].len().div_ceil(1 << 18).max(1)).cloned().collect();
            let k = ((s.len() as f64 * 0.95) as usize).min(s.len() - 1);
            0.2 * *s.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap()).1
        };
        // normalised samples u = c w̄ / amp of one (level, ε, output), strided to at most 2^15 (signal-bearing only)
        let samples = |bufs: &[Vec<f32>; 3], l: usize, e: usize, o: usize, scale: f32| -> Vec<f32> {
            let (_, off, sz) = band(p, l, e);
            let n = sz[0] * sz[1] * sz[2];
            let step = n.div_ceil(1 << 15).max(1);
            let a = amp[(l - 1) * 7 + e - 1][o];
            (0..n)
                .step_by(step)
                .filter_map(|i| {
                    let pos = [i % sz[0], (i / sz[0]) % sz[1], i / (sz[0] * sz[1])];
                    let w = wat(l, pos);
                    (w >= wmin).then(|| {
                        let q = (off[0] + pos[0]) + p[0] * ((off[1] + pos[1]) + p[1] * (off[2] + pos[2]));
                        (bufs[o][q] * w / (a * scale)).abs()
                    })
                })
                .collect()
        };
        // s₀: MAD of the finest diagonal subband (ε = 111), all three outputs
        let s0 = {
            let mut u: Vec<f32> = (0..3).flat_map(|o| samples(&bufs, 1, 7, o, 1.0)).collect();
            if u.is_empty() {
                0.0
            } else {
                let mid = u.len() / 2;
                *u.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1 as f64 / 0.6745
            }
        };
        let mut st = DfStats { s0, ..Default::default() };
        if s0 <= 0.0 || !s0.is_finite() {
            self.inverse(&mut bufs);
            for i in 0..3 {
                pad_out(&bufs[i], d, p, 1, self.shift, &mut v[i * nv..(i + 1) * nv]);
            }
            return st;
        }
        // thresholds of every (level, ε, output), searched in parallel
        let jobs: Vec<(usize, usize, usize)> = (1..=self.levels).flat_map(|l| (1..8usize).flat_map(move |e| (0..3).map(move |o| (l, e, o)))).collect();
        let lams: Vec<f64> = jobs
            .par_iter()
            .map(|&(l, e, o)| {
                let (t, sure) = match self.rule {
                    DfRule::Fixed { tdf, tn } => (if o < 2 { tdf } else { tn }, false),
                    DfRule::Sure { tdf, tn } => (if o < 2 { tdf } else { tn }, true),
                };
                let (_, _, sz) = band(p, l, e);
                if sure { t * sure_threshold(&mut samples(&bufs, l, e, o, s0 as f32), (sz[0] * sz[1] * sz[2]) as f64) } else { t }
            })
            .collect();
        let (mut nl, mut kept) = ([0usize; 2], [0usize; 2]);
        let (mut lsum, mut ntot) = ([0f64; 2], [0usize; 2]);
        for (&(l, e, o), &lam) in jobs.iter().zip(&lams) {
            let kind = (o == 2) as usize;
            let (_, off, sz) = band(p, l, e);
            lsum[kind] += lam;
            nl[kind] += 1;
            let a = amp[(l - 1) * 7 + e - 1][o];
            // c ← c max(0, 1 − λ s₀ amp / (|c| w̄))
            let cut = (lam * s0) as f32 * a;
            let base = Shared(bufs[o].as_mut_ptr());
            let k: usize = (0..sz[1] * sz[2])
                .into_par_iter()
                .map(|row| {
                    let base = base;
                    let (y, z) = (row % sz[1], row / sz[1]);
                    let q0 = off[0] + p[0] * ((off[1] + y) + p[1] * (off[2] + z));
                    let mut kept = 0usize;
                    for x in 0..sz[0] {
                        let w = wat(l, [x, y, z]);
                        // SAFETY: rows are disjoint
                        let c = unsafe { &mut *base.0.add(q0 + x) };
                        let m = c.abs() * w;
                        let f = if m > cut { 1.0 - cut / m } else { 0.0 };
                        *c *= f;
                        kept += (f > 0.0) as usize;
                    }
                    kept
                })
                .sum();
            kept[kind] += k;
            ntot[kind] += sz[0] * sz[1] * sz[2];
        }
        st.lam_df = lsum[0] / nl[0].max(1) as f64;
        st.lam_n = lsum[1] / nl[1].max(1) as f64;
        st.kept_df = kept[0] as f64 / ntot[0].max(1) as f64;
        st.kept_n = kept[1] as f64 / ntot[1].max(1) as f64;
        self.inverse(&mut bufs);
        for i in 0..3 {
            pad_out(&bufs[i], d, p, 1, self.shift, &mut v[i * nv..(i + 1) * nv]);
        }
        st
    }
}

/// SureShrink threshold for unit-noise samples |u| (Donoho & Johnstone 1995): minimiser of
/// SURE(λ) = N − 2 #{|u| ≤ λ} + Σ min(u², λ²) over λ ∈ [0, √(2 ln n)], or the universal threshold √(2 ln n) when the
/// subband is sparse ((Σu² − N) / N ≤ log₂(N)^{3/2} / √N). `n_full`: the subband size (the samples may be strided).
pub fn sure_threshold(u: &mut [f32], n_full: f64) -> f64 {
    let n = u.len();
    let lu = (2.0 * n_full.max(2.0).ln()).sqrt();
    if n < 8 {
        return lu;
    }
    u.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let nf = n as f64;
    let e2: f64 = u.iter().map(|&v| (v as f64).powi(2)).sum();
    if (e2 - nf) / nf <= nf.log2().powf(1.5) / nf.sqrt() {
        return lu;
    }
    let (mut best, mut lam) = (nf, 0.0);
    let mut cum = 0.0;
    for (k, &a) in u.iter().enumerate() {
        let a = a as f64;
        if a > lu {
            break;
        }
        cum += a * a;
        let s = nf - 2.0 * (k + 1) as f64 + cum + (nf - (k + 1) as f64) * a * a;
        if s < best {
            best = s;
            lam = a;
        }
    }
    lam
}

/// DFW shrinkage of the velocity encoded in one frame's complex encode images (ne × nv, encode-major): decode v per
/// voxel, shrink with the magnitude weight w = mean_e |x_e|, re-encode as x_e ← x_e exp(i (φ_e(v') − φ_e(v))) (the
/// model-consistent change of v with magnitude and background phase kept). Velocity in the image axes (logical frame).
pub fn shrink_encodes(dfw: &DfWavelet, x: &mut [hasty_cuda::C32], ne: usize, dec: &crate::flow::FlowDecoder) -> DfStats {
    use num_complex::Complex64;
    let nv = dfw.dims.iter().product::<usize>();
    assert_eq!(x.len(), ne * nv);
    let mut v = vec![0f32; 3 * nv];
    let mut w = vec![0f32; nv];
    {
        let (vx, rest) = v.split_at_mut(nv);
        let (vy, vz) = rest.split_at_mut(nv);
        (vx, vy, vz, &mut w[..]).into_par_iter().enumerate().for_each(|(q, (a, b, c, wq))| {
            let mut zb = [Complex64::new(0.0, 0.0); crate::flow::MAX_ENC + 1];
            for e in 0..ne {
                let s = x[e * nv + q];
                zb[e] = Complex64::new(s.re as f64, s.im as f64);
            }
            let z = &zb[..ne];
            let vel = dec.velocity(z);
            *a = vel[0] as f32;
            *b = vel[1] as f32;
            *c = vel[2] as f32;
            *wq = (z.iter().map(|c| c.norm()).sum::<f64>() / ne as f64) as f32;
        });
    }
    let v0 = v.clone();
    let st = dfw.apply(&mut v, Some(&w));
    // x_e ← x_e exp(i (φ_e(v') − φ_e(v)))
    let (v, v0) = (&v, &v0);
    x.par_chunks_mut(nv).enumerate().for_each(|(e, img)| {
        for (q, s) in img.iter_mut().enumerate() {
            let dv = [0, 1, 2].map(|a| (v[a * nv + q] - v0[a * nv + q]) as f64);
            let ph = dec.phase(e, &dv);
            let (sn, cs) = ph.sin_cos();
            let (re, im) = (s.re as f64, s.im as f64);
            *s = hasty_cuda::C32::new((re * cs - im * sn) as f32, (re * sn + im * cs) as f32);
        }
    });
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, mut s: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// smooth periodic potential A and v = curl A with backward differences: the backward divergence is exactly 0
    fn curl_field(p: [usize; 3]) -> Vec<f32> {
        curl_field_dir(p, false)
    }

    fn curl_field_dir(p: [usize; 3], fwd: bool) -> Vec<f32> {
        let np = p[0] * p[1] * p[2];
        let tau = std::f64::consts::TAU;
        let a: Vec<Vec<f64>> = (0..3)
            .map(|c| {
                (0..np)
                    .map(|q| {
                        let (x, y, z) = ((q % p[0]) as f64 / p[0] as f64, ((q / p[0]) % p[1]) as f64 / p[1] as f64, (q / (p[0] * p[1])) as f64 / p[2] as f64);
                        (tau * (x + 2.0 * y + c as f64 * 0.3)).sin() * (tau * (z - 0.2 * c as f64)).cos() + 0.5 * (tau * (2.0 * x - y + 3.0 * z + c as f64)).cos()
                    })
                    .collect()
            })
            .collect();
        let idx = |x: usize, y: usize, z: usize| x % p[0] + p[0] * (y % p[1] + p[1] * (z % p[2]));
        let bd = |f: &Vec<f64>, x: usize, y: usize, z: usize, ax: usize| -> f64 {
            if fwd {
                let (xm, ym, zm) = match ax { 0 => (x + 1, y, z), 1 => (x, y + 1, z), _ => (x, y, z + 1) };
                return f[idx(xm, ym, zm)] - f[idx(x, y, z)];
            }
            let (xm, ym, zm) = match ax { 0 => (x + p[0] - 1, y, z), 1 => (x, y + p[1] - 1, z), _ => (x, y, z + p[2] - 1) };
            f[idx(x, y, z)] - f[idx(xm, ym, zm)]
        };
        let mut v = vec![0f32; 3 * np];
        for z in 0..p[2] {
            for y in 0..p[1] {
                for x in 0..p[0] {
                    let q = idx(x, y, z);
                    v[q] = (bd(&a[2], x, y, z, 1) - bd(&a[1], x, y, z, 2)) as f32;
                    v[np + q] = (bd(&a[0], x, y, z, 2) - bd(&a[2], x, y, z, 0)) as f32;
                    v[2 * np + q] = (bd(&a[1], x, y, z, 0) - bd(&a[0], x, y, z, 1)) as f32;
                }
            }
        }
        v
    }

    #[test]
    fn perfect_reconstruction() {
        let d = [20usize, 14, 9];
        let nv = d.iter().product::<usize>();
        let mut w = DfWavelet::new(d, 2, [1.0, 1.2, 0.8], DfRule::Fixed { tdf: 0.0, tn: 0.0 });
        w.set_iteration(5);
        let v0 = rnd(3 * nv, 3);
        let mut v = v0.clone();
        w.apply(&mut v, None);
        let err = v.iter().zip(&v0).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-5, "DFW perfect reconstruction failed: {err}");
    }

    #[test]
    fn divergence_free_field_has_no_n_coefficients() {
        // periodic box (no padding, no shift): the discrete curl lives in the df subspace at every level
        let w = DfWavelet::new([25, 25, 9], 3, [1.0; 3], DfRule::Fixed { tdf: 0.0, tn: 0.0 });
        let p = w.padded();
        assert_eq!(p, [32, 32, 16]);
        let np = p[0] * p[1] * p[2];
        let v = curl_field(p);
        let mut bufs: [Vec<f32>; 3] = [v[..np].to_vec(), v[np..2 * np].to_vec(), v[2 * np..].to_vec()];
        w.forward(&mut bufs);
        let (mut en, mut edf) = (0f64, 0f64);
        for l in 1..=3 {
            for e in 1..8 {
                let (_, off, sz) = band(p, l, e);
                for z in 0..sz[2] {
                    for y in 0..sz[1] {
                        for x in 0..sz[0] {
                            let q = (off[0] + x) + p[0] * ((off[1] + y) + p[1] * (off[2] + z));
                            edf += (bufs[0][q] as f64).powi(2) + (bufs[1][q] as f64).powi(2);
                            en += (bufs[2][q] as f64).powi(2);
                        }
                    }
                }
            }
        }
        let rel = (en / edf).sqrt();
        assert!(rel < 1e-4, "n / df coefficient energy of a divergence-free field: {rel:e}");
        // a gradient field (curl-free) is the opposite case: its detail energy is not divergence-free
        let g: Vec<f32> = {
            let mut g = vec![0f32; 3 * np];
            let f: Vec<f64> = (0..np).map(|q| { let (x, y, z) = ((q % p[0]) as f64, ((q / p[0]) % p[1]) as f64, (q / (p[0] * p[1])) as f64); (0.4 * x).sin() * (0.3 * y).cos() + (0.7 * z).sin() }).collect();
            let idx = |x: usize, y: usize, z: usize| x % p[0] + p[0] * (y % p[1] + p[1] * (z % p[2]));
            for z in 0..p[2] { for y in 0..p[1] { for x in 0..p[0] {
                let q = idx(x, y, z);
                g[q] = (f[idx(x + 1, y, z)] - f[q]) as f32;
                g[np + q] = (f[idx(x, y + 1, z)] - f[q]) as f32;
                g[2 * np + q] = (f[idx(x, y, z + 1)] - f[q]) as f32;
            }}}
            g
        };
        let mut bufs: [Vec<f32>; 3] = [g[..np].to_vec(), g[np..2 * np].to_vec(), g[2 * np..].to_vec()];
        w.forward(&mut bufs);
        let (_, off, sz) = band(p, 1, 1);
        let n1: f64 = (0..sz[2]).flat_map(|z| (0..sz[1]).map(move |y| (y, z))).map(|(y, z)| (0..sz[0]).map(|x| (bufs[2][(off[0] + x) + p[0] * ((off[1] + y) + p[1] * (off[2] + z))] as f64).powi(2)).sum::<f64>()).sum();
        assert!(n1 > 1e-3, "gradient field left no n energy: {n1}");
    }

    #[test]
    fn shrinkage_denoises_a_divergence_free_field_and_is_scale_equivariant() {
        let p = [32usize, 32, 32];
        let np = p[0] * p[1] * p[2];
        let clean = curl_field(p);
        let sc = clean.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let noise = rnd(3 * np, 11);
        let noisy: Vec<f32> = clean.iter().zip(&noise).map(|(c, n)| c + 0.6 * sc * n).collect();
        let e0: f64 = noisy.iter().zip(&clean).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let mut w = DfWavelet::new(p, 3, [1.0; 3], DfRule::Sure { tdf: 1.0, tn: 1.0 });
        w.set_iteration(2);
        let mut a = noisy.clone();
        let st = w.apply(&mut a, None);
        let e1: f64 = a.iter().zip(&clean).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        assert!(e1 < 0.5 * e0, "DFW SURE did not denoise: {e1:.3e} vs {e0:.3e} ({st:?})");
        // n coefficients of noise around a divergence-free field are thresholded at least as hard as df
        assert!(st.kept_n <= st.kept_df + 0.05, "{st:?}");
        let mut b: Vec<f32> = noisy.iter().map(|v| v * 5.0).collect();
        w.apply(&mut b, None);
        let rel = (a.iter().zip(&b).map(|(x, y)| ((5.0 * x - y) as f64).powi(2)).sum::<f64>() / b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>()).sqrt();
        assert!(rel < 1e-4, "not scale-equivariant: {rel:e}");
    }
}
